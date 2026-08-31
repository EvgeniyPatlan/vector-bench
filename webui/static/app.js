"use strict";

const ENGINE_COLOR = {
  mariadb: "#1f77b4", mariadb123: "#5fa8d3", alisql: "#d62728",
  pgvector: "#2ca02c", mongodb: "#9467bd", valkey: "#e377c2",
};
const ENGINE_LABEL = {
  mariadb: "MariaDB 11.8 (MHNSW)", mariadb123: "MariaDB 12.3 (MHNSW)",
  alisql: "AliSQL (VIDX)", pgvector: "PostgreSQL (pgvector)",
  mongodb: "Percona Search (mongot)", valkey: "Valkey (valkey-search)",
};
const PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#e377c2",
                 "#ff7f0e", "#8c564b", "#17becf"];

const S = {
  runs: [], runId: null, run: null, tab: "overview",
  control: false, facets: {}, measures: [], filters: {},
  records: [], source: "", chart: null,
};

// -- utilities ---------------------------------------------------------

async function api(path, opts) {
  const res = await fetch(path, opts);
  const body = await res.json().catch(() => ({ error: res.statusText }));
  if (!res.ok) throw new Error(body.error || `HTTP ${res.status}`);
  return body;
}

function el(tag, attrs, ...kids) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "html") node.innerHTML = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const kid of kids.flat()) {
    if (kid === null || kid === undefined || kid === false) continue;
    node.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return node;
}

function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

function engineColor(name, index) {
  return ENGINE_COLOR[name] || PALETTE[index % PALETTE.length];
}

function fmtBytes(n) {
  if (n === null || n === undefined) return "—";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let v = Number(n), i = 0;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i += 1; }
  return `${v.toFixed(v < 10 && i > 0 ? 1 : 0)} ${units[i]}`;
}

function fmtDuration(s) {
  if (!s) return "—";
  if (s < 90) return `${s.toFixed(1)}s`;
  if (s < 5400) return `${(s / 60).toFixed(1)}m`;
  return `${(s / 3600).toFixed(1)}h`;
}

function fmtNum(v) {
  if (v === null || v === undefined) return "";
  if (typeof v !== "number") return String(v);
  if (Number.isInteger(v)) return String(v);
  return Math.abs(v) >= 100 ? v.toFixed(1) : v.toPrecision(4);
}

// -- run list ----------------------------------------------------------

function renderRunList() {
  const list = document.getElementById("run-list");
  const needle = document.getElementById("run-filter").value.toLowerCase();
  clear(list);

  const shown = S.runs.filter((r) => {
    if (!needle) return true;
    const hay = [r.dir_name, r.profile, r.resource_pass,
                 (r.engines || []).join(" "), (r.datasets || []).join(" ")]
      .join(" ").toLowerCase();
    return hay.includes(needle);
  });

  if (!shown.length) {
    list.append(el("li", { class: "muted" }, "no runs"));
    return;
  }

  for (const run of shown) {
    const started = (run.started_at || "").replace("T", " ").replace("Z", "");
    list.append(el("li", {
      class: run.dir_name === S.runId ? "active" : "",
      onclick: () => selectRun(run.dir_name),
    },
      el("div", { class: "rid" }, run.dir_name),
      el("div", { class: "meta" },
        `${run.profile || "?"} · ${run.resource_pass || "?"} · ${(run.engines || []).length} engines`),
      el("div", { class: "meta" },
        `${started} · ${run.status}${run.record_count ? ` · ${run.record_count} records` : ""}`)));
  }
}

// -- overview ----------------------------------------------------------

function renderOverview() {
  const panel = document.getElementById("panel-overview");
  clear(panel);
  if (!S.run) {
    panel.append(el("p", { class: "empty" }, "Select a run from the left."));
    return;
  }

  const { summary, manifest } = S.run;
  const host = manifest.host || {};
  const cpu = host.cpu || {};

  panel.append(el("h2", {}, summary.run_id),
    summary.description ? el("p", { class: "muted" }, summary.description) : null);

  const dl = el("dl", { class: "kv" });
  const pairs = [
    ["status", summary.status],
    ["profile", summary.profile],
    ["resource pass", summary.resource_pass],
    ["started", summary.started_at],
    ["finished", summary.finished_at],
    ["measured time", fmtDuration(summary.duration_s)],
    ["datasets", (summary.datasets || []).join(", ") || "—"],
    ["records", `${summary.record_count} (${summary.has_records ? "merged" : "ops only"})`],
    ["CPU", cpu.model],
    ["cores", cpu.physical_cores ? `${cpu.physical_cores} physical / ${cpu.logical_cpus} logical${cpu.hybrid ? " (hybrid)" : ""}` : null],
    ["SIMD", (cpu.simd_flags || []).join(" ") || null],
    ["AVX-512", cpu.has_avx512 === undefined ? null : (cpu.has_avx512 ? "yes" : "no")],
    ["RAM", host.total_ram_bytes ? fmtBytes(host.total_ram_bytes) : null],
    ["kernel", host.kernel],
    ["docker", host.docker_version],
  ];
  for (const [k, v] of pairs) {
    if (v === null || v === undefined) continue;
    dl.append(el("dt", {}, k), el("dd", {}, String(v)));
  }
  panel.append(dl);

  const engines = manifest.engines || {};
  if (Object.keys(engines).length) {
    panel.append(el("h3", {}, "Engines"));
    const rows = Object.entries(engines).map(([name, info]) => {
      const build = info.build || {};
      return el("tr", {},
        el("td", {}, el("span", { class: "pill", style: `border-color:${engineColor(name, 0)};color:${engineColor(name, 0)}` }, name)),
        el("td", {}, ENGINE_LABEL[name] || name),
        el("td", {}, build.tag || info.tag || "—"),
        el("td", {}, build.march || "—"),
        el("td", {}, build.build_type || "—"));
    });
    panel.append(el("table", {},
      el("thead", {}, el("tr", {}, ...["engine", "implementation", "tag", "-march", "build"].map((h) => el("th", {}, h)))),
      el("tbody", {}, ...rows)));
  }

  const resolved = (manifest.config || {}).resolved_resources;
  if (resolved) {
    panel.append(el("h3", {}, "Resolved resources"));
    const rdl = el("dl", { class: "kv" });
    for (const [k, v] of Object.entries(resolved)) {
      if (k.startsWith("_") || v === null || v === "") continue;
      const shown = k.endsWith("_bytes") ? fmtBytes(v) : String(v);
      rdl.append(el("dt", {}, k.replace(/_/g, " ")), el("dd", {}, shown));
    }
    panel.append(rdl);
  }

  const warnings = manifest.warnings || [];
  if (warnings.length) {
    panel.append(el("h3", {}, `Validity — ${warnings.length} warning${warnings.length > 1 ? "s" : ""}`));
    for (const w of warnings) panel.append(el("div", { class: "warn" }, w));
  }

  const phases = manifest.phases || [];
  if (phases.length) {
    panel.append(el("h3", {}, "Phases"));
    const rows = phases.map((p) => el("tr", {},
      el("td", {}, p.phase), el("td", {}, p.engine), el("td", {}, p.dataset),
      el("td", {}, p.resource_pass),
      el("td", {}, el("span", { class: `pill ${p.status === "completed" ? "ok" : "bad"}` }, p.status)),
      el("td", { class: "num" }, fmtDuration(p.duration_s))));
    panel.append(el("table", {},
      el("thead", {}, el("tr", {}, ...["phase", "engine", "dataset", "pass", "status", "duration"].map((h) => el("th", {}, h)))),
      el("tbody", {}, ...rows)));
  }
}

// -- report tab --------------------------------------------------------

function renderReport() {
  const panel = document.getElementById("panel-report");
  clear(panel);
  if (!S.run) { panel.append(el("p", { class: "empty" }, "Select a run.")); return; }
  if (!S.run.summary.has_report) {
    panel.append(el("p", { class: "empty" },
      "No generated report for this run. Run: ./run-benchmark.sh report --run-dir results/" + S.runId));
    return;
  }
  panel.append(el("iframe", {
    class: "report", src: `/runs/${encodeURIComponent(S.runId)}/report/report.html`,
  }));
}

// -- navigation --------------------------------------------------------

const RENDERERS = {
  overview: renderOverview,
  explore: () => window.renderExplore(),
  report: renderReport,
  configure: () => window.renderConfigure(),
  jobs: () => window.renderJobs(),
};

function showTab(name) {
  S.tab = name;
  writeHash();
  for (const button of document.querySelectorAll("#tabs button")) {
    button.classList.toggle("active", button.dataset.tab === name);
  }
  for (const panel of document.querySelectorAll(".panel")) {
    panel.hidden = panel.id !== `panel-${name}`;
  }
  const render = RENDERERS[name];
  if (render) render();
}

async function selectRun(runId, tab) {
  S.runId = runId;
  S.filters = {};
  S.records = [];
  document.getElementById("tabs").hidden = false;
  renderRunList();
  try {
    S.run = await api(`/api/runs/${encodeURIComponent(runId)}`);
    const meta = await api(`/api/runs/${encodeURIComponent(runId)}/facets`);
    S.facets = meta.facets; S.measures = meta.measures; S.source = meta.source;
  } catch (err) {
    S.run = null;
    document.getElementById("panel-overview").append(el("p", { class: "err" }, String(err)));
    return;
  }
  showTab(tab || "overview");
}

async function boot() {
  document.getElementById("run-filter").addEventListener("input", renderRunList);
  for (const button of document.querySelectorAll("#tabs button")) {
    button.addEventListener("click", () => showTab(button.dataset.tab));
  }

  const health = await api("/api/health");
  S.control = !!health.control_enabled;
  document.getElementById("control-badge").textContent =
    S.control ? "control enabled" : "read-only";
  for (const button of document.querySelectorAll("#tabs button[data-control]")) {
    button.hidden = !S.control;
  }

  const { runs } = await api("/api/runs");
  S.runs = runs;
  renderRunList();
  renderOverview();

  window.addEventListener("hashchange", onHashChange);
  const wanted = parseHash();
  const runId = wanted.runId && runs.some((r) => r.dir_name === wanted.runId)
    ? wanted.runId : (runs.length ? runs[0].dir_name : null);
  if (runId) await selectRun(runId, wanted.tab);
}

// -- hash routing ------------------------------------------------------

function parseHash() {
  const match = /^#\/run\/([^/]+)(?:\/([a-z]+))?/.exec(window.location.hash || "");
  if (!match) return {};
  return { runId: decodeURIComponent(match[1]), tab: match[2] };
}

function writeHash() {
  if (!S.runId) return;
  const next = `#/run/${encodeURIComponent(S.runId)}/${S.tab}`;
  if (window.location.hash !== next) {
    history.replaceState(null, "", next);
  }
}

function onHashChange() {
  const wanted = parseHash();
  if (!wanted.runId) return;
  if (wanted.runId !== S.runId) selectRun(wanted.runId, wanted.tab);
  else if (wanted.tab && wanted.tab !== S.tab) showTab(wanted.tab);
}

window.S = S;
window.api = api;
window.el = el;
window.clear = clear;
window.engineColor = engineColor;
window.fmtNum = fmtNum;
window.fmtBytes = fmtBytes;
window.fmtDuration = fmtDuration;
window.showTab = showTab;

boot().catch((err) => {
  document.getElementById("main").append(el("p", { class: "err" }, String(err)));
});
