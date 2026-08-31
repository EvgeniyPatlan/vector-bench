"use strict";

const TABLE_COLUMNS = [
  "engine", "dataset", "phase", "resource_pass", "m", "ef_construction",
  "ef_search", "build_mode", "storage_engine", "clients", "selectivity",
  "churn_fraction", "recall_at_k", "qps", "latency_p50_ms", "latency_p95_ms",
  "latency_p99_ms", "build_wall_s", "peak_rss_bytes", "index_bytes",
  "ingest_rows_per_s",
];

let tableSort = { column: null, descending: false };

function queryString() {
  const params = new URLSearchParams();
  for (const [field, values] of Object.entries(S.filters)) {
    for (const value of values) params.append(field, value);
  }
  return params;
}

function renderFacets() {
  const host = document.getElementById("facets");
  clear(host);

  const entries = Object.entries(S.facets);
  if (!entries.length) {
    host.append(el("p", { class: "muted" }, "No records for this run yet."));
    return;
  }

  for (const [field, values] of entries) {
    if (values.length < 2) continue;
    const box = el("div", { class: "facet" }, el("strong", {}, field.replace(/_/g, " ")));
    for (const value of values) {
      const selected = (S.filters[field] || []).includes(String(value));
      box.append(el("label", {},
        el("input", {
          type: "checkbox", checked: selected,
          onchange: (ev) => toggleFilter(field, String(value), ev.target.checked),
        }),
        " ", String(value)));
    }
    host.append(box);
  }
}

function toggleFilter(field, value, on) {
  const current = S.filters[field] || [];
  const next = on ? [...current, value] : current.filter((v) => v !== value);
  S.filters = { ...S.filters, [field]: next };
  if (!next.length) delete S.filters[field];
  loadExplore();
}

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

function drawChart(payload) {
  const host = document.getElementById("chart");
  clear(host);
  if (S.chart) { S.chart.destroy(); S.chart = null; }

  const series = payload.series.filter((s) => s.x.length);
  if (!series.length) {
    host.append(el("p", { class: "muted" }, "Nothing to plot for this X/Y pair."));
    return;
  }

  const logY = document.getElementById("log-y").checked;
  const data = alignSeries(series);
  const width = Math.max(host.clientWidth || 800, 420);

  S.chart = new uPlot({
    width, height: 340,
    scales: { y: logY ? { distr: 3 } : {} },
    axes: [
      { label: payload.x, stroke: "#8a94a3", grid: { stroke: "#8a94a333" }, ticks: { stroke: "#8a94a333" } },
      { label: payload.y, stroke: "#8a94a3", grid: { stroke: "#8a94a333" }, ticks: { stroke: "#8a94a333" } },
    ],
    series: [
      { label: payload.x },
      ...series.map((s, i) => ({
        label: s.key || "series",
        stroke: engineColor(s.group.engine, i),
        width: 2,
        points: { show: true, size: 6 },
        spanGaps: true,
        value: (_u, v) => (v === null ? "—" : fmtNum(v)),
      })),
    ],
  }, data, host);
}

function renderTable(records) {
  const wrap = document.getElementById("table-wrap");
  clear(wrap);
  if (!records.length) {
    wrap.append(el("p", { class: "muted", style: "padding:12px" }, "No matching records."));
    return;
  }

  const present = TABLE_COLUMNS.filter((c) => records.some((r) => r[c] !== null && r[c] !== undefined));
  const sorted = [...records];
  if (tableSort.column) {
    const c = tableSort.column;
    sorted.sort((a, b) => {
      const av = a[c], bv = b[c];
      if (av === bv) return 0;
      if (av === null || av === undefined) return 1;
      if (bv === null || bv === undefined) return -1;
      const cmp = typeof av === "number" && typeof bv === "number"
        ? av - bv : String(av).localeCompare(String(bv));
      return tableSort.descending ? -cmp : cmp;
    });
  }

  const head = el("tr", {}, ...present.map((c) => el("th", {
    class: typeof records[0][c] === "number" ? "num" : "",
    onclick: () => {
      tableSort = { column: c, descending: tableSort.column === c && !tableSort.descending };
      renderTable(records);
    },
  }, c + (tableSort.column === c ? (tableSort.descending ? " ↓" : " ↑") : ""))));

  const body = sorted.slice(0, 500).map((r) => el("tr", {}, ...present.map((c) => {
    const v = r[c];
    const numeric = typeof v === "number";
    const text = c.endsWith("_bytes") && numeric ? fmtBytes(v) : fmtNum(v);
    return el("td", { class: numeric ? "num" : "" }, text);
  })));

  wrap.append(el("table", {}, el("thead", {}, head), el("tbody", {}, ...body)));
  if (sorted.length > 500) {
    wrap.append(el("p", { class: "muted", style: "padding:8px" },
      `showing first 500 of ${sorted.length}`));
  }
}

function populateAxes() {
  const xSelect = document.getElementById("axis-x");
  const ySelect = document.getElementById("axis-y");
  const numericFacets = ["m", "ef_search", "ef_construction", "clients",
                         "selectivity", "churn_fraction"]
    .filter((f) => S.facets[f]);
  const xChoices = [...numericFacets, ...S.measures];
  const yChoices = S.measures;

  const fill = (select, choices, preferred) => {
    const previous = select.value;
    clear(select);
    for (const choice of choices) select.append(el("option", { value: choice }, choice));
    const wanted = choices.includes(previous) ? previous
      : preferred.find((p) => choices.includes(p)) || choices[0];
    if (wanted) select.value = wanted;
  };

  fill(xSelect, xChoices, ["recall_at_k", "clients", "m"]);
  fill(ySelect, yChoices, ["qps", "build_wall_s", "recall_at_k"]);
}

async function loadExplore() {
  const runId = encodeURIComponent(S.runId);
  const x = document.getElementById("axis-x").value;
  const y = document.getElementById("axis-y").value;
  const groupBy = document.getElementById("group-by").value;
  if (!x || !y) return;

  const params = queryString();
  params.set("x", x); params.set("y", y); params.set("group_by", groupBy);

  const [chartData, recordData] = await Promise.all([
    api(`/api/runs/${runId}/series?${params}`),
    api(`/api/runs/${runId}/records?${queryString()}`),
  ]);

  document.getElementById("match-count").textContent =
    `${recordData.matched} of ${recordData.total} records · source ${recordData.source}`;
  renderFacets();
  drawChart(chartData);
  renderTable(recordData.records);
}

let exploreWired = false;

window.renderExplore = function renderExplore() {
  if (!S.runId) return;
  if (!exploreWired) {
    for (const id of ["axis-x", "axis-y", "group-by", "log-y"]) {
      document.getElementById(id).addEventListener("change", () => loadExplore());
    }
    window.addEventListener("resize", () => {
      if (S.chart) S.chart.setSize({ width: Math.max(document.getElementById("chart").clientWidth, 420), height: 340 });
    });
    exploreWired = true;
  }
  populateAxes();
  loadExplore().catch((err) => {
    document.getElementById("chart").append(el("p", { class: "err" }, String(err)));
  });
};
