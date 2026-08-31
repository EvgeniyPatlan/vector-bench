"use strict";

const TABLE_COLUMNS = [
  "engine", "dataset", "phase", "resource_pass", "m", "ef_construction",
  "ef_search", "build_mode", "storage_engine", "clients", "selectivity",
  "churn_fraction", "recall_at_k", "qps", "latency_p50_ms", "latency_p95_ms",
  "latency_p99_ms", "build_wall_s", "peak_rss_bytes", "index_bytes",
  "ingest_rows_per_s",
];

const E = { y: null, x: null, groupBy: "engine", logY: null, sort: { column: null, desc: false } };

function currentView() { return viewById(S.viewId); }

/** A view's axes and filters, with the raw view falling back to the run's own fields. */
function resolveView() {
  const view = currentView();
  if (view.id !== "raw") {
    return {
      view,
      x: view.x,
      y: E.y && view.ys.includes(E.y) ? E.y : view.ys.find((f) => S.measures.includes(f)) || view.ys[0],
      ys: view.ys.filter((f) => S.measures.includes(f)),
      logY: E.logY === null ? view.logY : E.logY,
      filters: view.filters.filter((f) => (S.facets[f] || []).length > 1),
      phaseFilter: { phase: [view.phase] },
    };
  }
  const numericFacets = ["m", "ef_search", "ef_construction", "clients",
                         "selectivity", "churn_fraction"].filter((f) => S.facets[f]);
  const xs = [...numericFacets, ...S.measures];
  return {
    view,
    x: E.x && xs.includes(E.x) ? E.x : xs[0],
    xs,
    y: E.y && S.measures.includes(E.y) ? E.y : S.measures[0],
    ys: S.measures,
    logY: E.logY === null ? false : E.logY,
    filters: Object.keys(S.facets).filter((f) => (S.facets[f] || []).length > 1),
    phaseFilter: {},
  };
}

function queryFor(resolved) {
  const params = new URLSearchParams();
  for (const [field, values] of Object.entries(resolved.phaseFilter)) {
    for (const value of values) params.append(field, value);
  }
  for (const [field, values] of Object.entries(S.filters)) {
    if (field === "phase" && resolved.view.id !== "raw") continue;
    for (const value of values) params.append(field, value);
  }
  return params;
}

// -- controls ----------------------------------------------------------

function renderViewPicker() {
  const pick = document.getElementById("view-pick");
  clear(pick);
  for (const view of VIEWS) {
    pick.append(el("option", { value: view.id, selected: view.id === S.viewId }, view.label));
  }
  pick.onchange = (ev) => {
    S.viewId = ev.target.value;
    E.y = null; E.x = null; E.logY = null; S.filters = {};
    load();
  };
  document.getElementById("view-hint").textContent = currentView().hint;
}

function renderControls(resolved) {
  const host = document.getElementById("view-controls");
  clear(host);

  if (resolved.view.id === "raw") {
    host.append(el("label", {}, "X ",
      select(resolved.xs, resolved.x, (v) => { E.x = v; load(); })));
  } else {
    host.append(el("span", { class: "axis-fixed" },
      `X: ${fieldLabel(resolved.x)}`));
  }

  if (resolved.ys.length > 1) {
    host.append(el("label", {}, "Y ",
      select(resolved.ys, resolved.y, (v) => { E.y = v; load(); })));
  } else {
    host.append(el("span", { class: "axis-fixed" }, `Y: ${fieldLabel(resolved.y)}`));
  }

  host.append(el("label", {}, "Group ",
    select(["engine", "engine,dataset", "engine,m", "engine,resource_pass",
            "engine,build_mode", "engine,storage_engine"],
           E.groupBy, (v) => { E.groupBy = v; load(); },
           (v) => v.split(",").map(fieldLabel).join(" + "))));

  host.append(el("label", { class: "check" },
    el("input", {
      type: "checkbox", checked: resolved.logY,
      onchange: (ev) => { E.logY = ev.target.checked; load(); },
    }), " log Y"));

  host.append(el("span", { id: "match-count", class: "muted" }));
}

function select(options, current, onChange, labelFn) {
  const node = el("select", { onchange: (ev) => onChange(ev.target.value) });
  for (const option of options) {
    node.append(el("option", { value: option, selected: option === current },
      (labelFn || fieldLabel)(option)));
  }
  return node;
}

function renderFilters(resolved) {
  const host = document.getElementById("filters");
  clear(host);
  if (!resolved.filters.length) return;

  host.append(el("div", { class: "filters-head muted" }, "Narrow to"));
  for (const field of resolved.filters) {
    const values = S.facets[field] || [];
    const box = el("div", { class: "facet" }, el("strong", {}, fieldLabel(field)));
    for (const value of values) {
      box.append(el("label", {},
        el("input", {
          type: "checkbox",
          checked: (S.filters[field] || []).includes(String(value)),
          onchange: (ev) => toggleFilter(field, String(value), ev.target.checked),
        }), " ", String(value)));
    }
    host.append(box);
  }
}

function toggleFilter(field, value, on) {
  const current = S.filters[field] || [];
  const next = on ? [...new Set([...current, value])] : current.filter((v) => v !== value);
  const filters = { ...S.filters, [field]: next };
  if (!next.length) delete filters[field];
  S.filters = filters;
  load();
}

// -- chart -------------------------------------------------------------

function alignSeries(series) {
  const xs = [...new Set(series.flatMap((s) => s.x))].sort((a, b) => a - b);
  const index = new Map(xs.map((x, i) => [x, i]));
  const columns = series.map((s) => {
    const column = new Array(xs.length).fill(null);
    s.x.forEach((x, i) => { column[index.get(x)] = s.y[i]; });
    return column;
  });
  return [xs, ...columns];
}

function drawChart(payload, resolved) {
  const host = document.getElementById("chart");
  const note = document.getElementById("chart-note");
  clear(host); clear(note);
  if (S.chart) { S.chart.destroy(); S.chart = null; }

  const series = payload.series.filter((s) => s.x.length);
  if (!series.length) {
    host.append(el("p", { class: "empty" }, resolved.view.empty));
    return;
  }

  const bytesAxis = resolved.y.endsWith("_bytes");
  S.chart = new uPlot({
    width: Math.max(host.clientWidth || 800, 420),
    height: 340,
    scales: { y: resolved.logY ? { distr: 3 } : {} },
    axes: [
      { label: fieldLabel(resolved.x), stroke: "#8a94a3",
        grid: { stroke: "#8a94a333" }, ticks: { stroke: "#8a94a333" } },
      { label: fieldLabel(resolved.y), stroke: "#8a94a3",
        grid: { stroke: "#8a94a333" }, ticks: { stroke: "#8a94a333" },
        values: bytesAxis ? (_u, ticks) => ticks.map(fmtBytes) : undefined },
    ],
    series: [
      { label: fieldLabel(resolved.x) },
      ...series.map((s, i) => ({
        label: s.key || "series",
        stroke: engineColor(s.group.engine, i),
        width: 2,
        points: { show: true, size: 6 },
        spanGaps: true,
        value: (_u, v) => (v === null ? "—" : fmtValue(resolved.y, v)),
      })),
    ],
  }, alignSeries(series), host);

  const singles = series.filter((s) => s.x.length === 1).map((s) => s.key);
  if (singles.length) {
    note.textContent = `${singles.length} series has a single point (${singles.join(", ")}) — `
      + `a line needs at least two values of ${fieldLabel(resolved.x)}.`;
  }
}

// -- table -------------------------------------------------------------

function renderTable(records) {
  const wrap = document.getElementById("table-wrap");
  clear(wrap);
  if (!records.length) return;

  const present = TABLE_COLUMNS.filter((c) =>
    records.some((r) => r[c] !== null && r[c] !== undefined));
  const sorted = [...records];
  if (E.sort.column) {
    const c = E.sort.column;
    sorted.sort((a, b) => {
      const av = a[c], bv = b[c];
      if (av === bv) return 0;
      if (av === null || av === undefined) return 1;
      if (bv === null || bv === undefined) return -1;
      const cmp = typeof av === "number" && typeof bv === "number"
        ? av - bv : String(av).localeCompare(String(bv));
      return E.sort.desc ? -cmp : cmp;
    });
  }

  const head = el("tr", {}, ...present.map((c) => el("th", {
    class: typeof records[0][c] === "number" ? "num" : "",
    title: c,
    onclick: () => {
      E.sort = { column: c, desc: E.sort.column === c && !E.sort.desc };
      renderTable(records);
    },
  }, fieldLabel(c) + (E.sort.column === c ? (E.sort.desc ? " ↓" : " ↑") : ""))));

  const body = sorted.slice(0, 500).map((r) => el("tr", {}, ...present.map((c) =>
    el("td", { class: typeof r[c] === "number" ? "num" : "" }, fmtValue(c, r[c])))));

  wrap.append(el("table", {}, el("thead", {}, head), el("tbody", {}, ...body)));
  if (sorted.length > 500) {
    wrap.append(el("p", { class: "muted pad" }, `showing first 500 of ${sorted.length}`));
  }
}

// -- load --------------------------------------------------------------

async function load() {
  const runId = encodeURIComponent(S.runId);
  const resolved = resolveView();
  renderViewPicker();
  renderControls(resolved);
  renderFilters(resolved);

  if (!resolved.x || !resolved.y) {
    clear(document.getElementById("chart"));
    document.getElementById("chart").append(el("p", { class: "empty" }, resolved.view.empty));
    clear(document.getElementById("table-wrap"));
    return;
  }

  const params = queryFor(resolved);
  const chartParams = new URLSearchParams(params);
  chartParams.set("x", resolved.x);
  chartParams.set("y", resolved.y);
  chartParams.set("group_by", E.groupBy);

  const [chartData, recordData] = await Promise.all([
    api(`/api/runs/${runId}/series?${chartParams}`),
    api(`/api/runs/${runId}/records?${params}`),
  ]);

  const count = document.getElementById("match-count");
  if (count) {
    count.textContent = `${recordData.matched} of ${recordData.total} records · ${recordData.source}`;
  }
  drawChart(chartData, resolved);
  renderTable(recordData.records);
}

let wired = false;

window.renderExplore = function renderExplore() {
  if (!S.runId) return;
  if (!wired) {
    window.addEventListener("resize", () => {
      if (!S.chart) return;
      const host = document.getElementById("chart");
      S.chart.setSize({ width: Math.max(host.clientWidth, 420), height: 340 });
    });
    wired = true;
  }
  return load().catch((err) => {
    document.getElementById("chart").append(el("p", { class: "err" }, String(err)));
  });
};
