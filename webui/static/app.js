"use strict";

const S = {
  runs: [], runId: null, run: null,
  route: { kind: "run", tab: "overview", section: "profiles" },
  control: false, facets: {}, measures: [], source: "",
  filters: {}, viewId: "recall", chart: null,
};

const RUN_TABS = [
  { id: "overview", label: "Overview" },
  { id: "explore", label: "Explore" },
  { id: "report", label: "Report" },
];

// Flat, in the order you need them: is the machine ready, what can it measure,
// what shall it measure, what is it doing. "Control" as a grouping label told
// you nothing about what was inside it.
const SECTIONS = [
  { id: "status", label: "Status" },
  { id: "engines", label: "Engines" },
  { id: "datasets", label: "Datasets" },
  { id: "profiles", label: "Profiles & launch" },
  { id: "jobs", label: "Jobs" },
];

// -- utilities ---------------------------------------------------------

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (res.status === 401) { window.location.href = "/login.html"; throw new Error("not signed in"); }
  const body = await res.json().catch(() => ({ error: res.statusText }));
  if (!res.ok) throw new Error(body.error || (body.errors || []).join("; ") || `HTTP ${res.status}`);
  return body;
}

function post(path, payload) {
  return api(path, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload || {}),
  });
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
  return ENGINE_COLOR[name] || PALETTE[(index || 0) % PALETTE.length];
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

function fmtValue(field, v) {
  if (typeof v === "number" && field.endsWith("_bytes")) return fmtBytes(v);
  return fmtNum(v);
}

// -- sidebar -----------------------------------------------------------

const RUNS_COLLAPSED_KEY = "vb.runs.collapsed";

function runsCollapsed() {
  try {
    return window.localStorage.getItem(RUNS_COLLAPSED_KEY) === "1";
  } catch (err) {
    // Private windows and blocked site data throw on access, not on read.
    return false;
  }
}

function setRunsCollapsed(collapsed) {
  const group = document.getElementById("runs-group");
  const toggle = document.getElementById("runs-toggle");
  group.classList.toggle("collapsed", collapsed);
  toggle.setAttribute("aria-expanded", String(!collapsed));
  document.getElementById("runs-body").hidden = collapsed;
  try {
    window.localStorage.setItem(RUNS_COLLAPSED_KEY, collapsed ? "1" : "0");
  } catch (err) {
    // Remembering the choice is a convenience, not a requirement.
  }
}

function renderRunList() {
  const list = document.getElementById("run-list");
  const needle = document.getElementById("run-filter").value.toLowerCase();
  clear(list);

  const shown = S.runs.filter((r) => {
    if (!needle) return true;
    return [r.dir_name, r.profile, r.resource_pass, (r.engines || []).join(" "),
            (r.datasets || []).join(" ")].join(" ").toLowerCase().includes(needle);
  });

  // The count belongs in the header so a collapsed group still tells you
  // whether there is anything in it.
  const count = document.getElementById("runs-count");
  if (count) {
    count.textContent = needle && shown.length !== S.runs.length
      ? `${shown.length}/${S.runs.length}` : String(S.runs.length);
  }

  if (!shown.length) { list.append(el("li", { class: "muted pad" }, "no runs")); return; }

  for (const run of shown) {
    const active = S.route.kind === "run" && run.dir_name === S.runId;
    const started = (run.started_at || "").replace("T", " ").replace("Z", "").slice(0, 16);
    list.append(el("li", {
      class: active ? "active" : "",
      onclick: () => go(`#/run/${encodeURIComponent(run.dir_name)}/${S.route.tab}`),
    },
      el("div", { class: "rid" }, run.dir_name),
      el("div", { class: "meta" },
        `${run.profile || "?"} · ${(run.engines || []).length} engines · ${run.status}`),
      el("div", { class: "meta" }, started)));
  }
}

function renderSectionList() {
  const list = document.getElementById("section-list");
  clear(list);
  for (const section of SECTIONS) {
    const active = S.route.kind === "control" && S.route.section === section.id;
    list.append(el("li", {
      class: active ? "active" : "",
      onclick: () => go(`#/control/${section.id}`),
    }, el("div", { class: "rid" }, section.label)));
  }
}

function renderTabs() {
  const bar = document.getElementById("tabs");
  const context = document.getElementById("context");
  clear(bar);

  if (S.route.kind !== "run" || !S.runId) {
    bar.hidden = true;
    context.textContent = S.route.kind === "control"
      ? (SECTIONS.find((s) => s.id === S.route.section) || {}).label || ""
      : "";
    return;
  }

  bar.hidden = false;
  context.textContent = S.runId;
  for (const tab of RUN_TABS) {
    bar.append(el("button", {
      class: tab.id === S.route.tab ? "active" : "",
      onclick: () => go(`#/run/${encodeURIComponent(S.runId)}/${tab.id}`),
    }, tab.label));
  }
}

// -- routing -----------------------------------------------------------

const PANELS = ["status", "datasets", "overview", "explore", "report",
                "profiles", "engines", "jobs"];

function showPanel(name) {
  for (const id of PANELS) {
    document.getElementById(`panel-${id}`).hidden = id !== name;
  }
}

function parseHash() {
  const hash = window.location.hash || "";
  let m = /^#\/run\/([^/]+)(?:\/([a-z]+))?/.exec(hash);
  if (m) {
    const tab = RUN_TABS.some((t) => t.id === m[2]) ? m[2] : "overview";
    return { kind: "run", runId: decodeURIComponent(m[1]), tab };
  }
  m = /^#\/control\/([a-z]+)/.exec(hash);
  if (m && SECTIONS.some((s) => s.id === m[1])) {
    return { kind: "control", section: m[1] };
  }
  return null;
}

function go(hash) {
  if (window.location.hash === hash) applyRoute();
  else window.location.hash = hash;
}

const RENDERERS = {
  status: () => window.renderStatus(),
  datasets: () => window.renderDatasets(),
  overview: () => window.renderOverview(),
  explore: () => window.renderExplore(),
  report: () => renderReport(),
  profiles: () => window.renderProfiles(),
  engines: () => window.renderEngines(),
  jobs: () => window.renderJobs(),
};

async function applyRoute() {
  const wanted = parseHash()
    || (S.runs.length ? { kind: "run", tab: "overview" }
                      : { kind: "control", section: "status" });

  if (wanted.kind === "control") {
    S.route = { ...S.route, kind: "control", section: wanted.section };
    renderRunList(); renderSectionList(); renderTabs();
    showPanel(wanted.section);
    await RENDERERS[wanted.section]();
    return;
  }

  const runId = wanted.runId && S.runs.some((r) => r.dir_name === wanted.runId)
    ? wanted.runId
    : (S.runs.length ? S.runs[0].dir_name : null);

  S.route = { ...S.route, kind: "run", tab: wanted.tab };
  if (!runId) {
    renderRunList(); renderSectionList(); renderTabs();
    showPanel("overview");
    const panel = document.getElementById("panel-overview");
    clear(panel);
    panel.append(el("p", { class: "empty" }, "No runs in results/ yet."));
    return;
  }

  if (runId !== S.runId) await loadRun(runId);
  renderRunList(); renderSectionList(); renderTabs();
  showPanel(wanted.tab);
  await RENDERERS[wanted.tab]();
}

async function loadRun(runId) {
  S.runId = runId;
  S.filters = {};
  S.run = await api(`/api/runs/${encodeURIComponent(runId)}`);
  const meta = await api(`/api/runs/${encodeURIComponent(runId)}/facets`);
  S.facets = meta.facets;
  S.measures = meta.measures;
  S.source = meta.source;
}

// -- report tab --------------------------------------------------------

function renderReport() {
  const panel = document.getElementById("panel-report");
  clear(panel);
  if (!S.run) { panel.append(el("p", { class: "empty" }, "Select a run.")); return; }
  const inputs = S.run.report_inputs || {};
  if (inputs.regenerate_note) {
    panel.append(el("div", { class: "warn" },
      el("strong", {}, inputs.regenerate_risk === "loses_recall"
        ? "Regenerating would drop the recall section. "
        : "Regenerating would not use only this run's data. "),
      inputs.regenerate_note));
  }
  if (inputs.measured_elsewhere) {
    panel.append(el("p", { class: "muted" },
      `Measured on ${inputs.measured_on}. The report below is self-contained — `
      + "its charts are inlined — so it displays exactly as it did there."));
  }

  const regenerate = el("div", { class: "row" },
    el("button", {
      class: "action" + (S.run.summary.has_report ? " secondary" : ""),
      disabled: !S.control,
      onclick: () => startJob({ kind: "report", run_id: S.runId },
                              document.getElementById("report-status")),
    }, S.run.summary.has_report
      ? (inputs.regenerate_risk && inputs.regenerate_risk !== "none"
          ? "Regenerate anyway" : "Regenerate")
      : "Generate report"),
    el("span", { id: "report-status" }),
    S.control ? null : el("span", { class: "muted" },
      "read-only; start with --allow-control to generate from here"));

  if (!S.run.summary.has_report) {
    panel.append(el("p", { class: "empty" }, "No report generated for this run yet."),
      el("pre", { class: "log cmd" },
        `./run-benchmark.sh report --run-dir results/${S.runId}`),
      regenerate);
    return;
  }
  panel.append(shareRow(), regenerate, el("iframe", {
    class: "report", src: `/runs/${encodeURIComponent(S.runId)}/report/report.html`,
  }));
}

function shareRow() {
  const id = encodeURIComponent(S.runId);
  return el("div", { class: "row" },
    el("a", {
      class: "action secondary", download: "",
      href: `/runs/${id}/report/report.html`,
    }, "Download report.html"),
    el("a", {
      class: "action secondary", href: `/runs/${id}/bundle`,
    }, "Download run bundle (.tar.gz)"),
    el("span", { class: "muted" },
      "report.html is self-contained — charts inlined, nothing fetched — so it "
      + "opens anywhere offline. The bundle adds the raw records and a README, "
      + "and drops into someone else's results/ directory."));
}

// -- boot --------------------------------------------------------------

async function boot() {
  document.getElementById("run-filter").addEventListener("input", renderRunList);
  const toggle = document.getElementById("runs-toggle");
  toggle.addEventListener("click", () => setRunsCollapsed(!runsCollapsed()));
  setRunsCollapsed(runsCollapsed());
  window.addEventListener("hashchange", () => applyRoute().catch(showError));

  const health = await api("/api/health");
  S.control = !!health.control_enabled;
  const badge = document.getElementById("control-badge");
  clear(badge);
  badge.append(S.control ? "control enabled" : "read-only");
  if (health.auth_enabled) {
    badge.append(" · ", el("a", { href: "#", onclick: signOut }, "sign out"));
  }

  S.runs = (await api("/api/runs")).runs;
  await applyRoute();
}

async function signOut(ev) {
  ev.preventDefault();
  await post("/api/logout");
  window.location.href = "/login.html";
}

function showError(err) {
  const main = document.getElementById("main");
  main.append(el("p", { class: "err" }, String(err)));
}

Object.assign(window, {
  S, api, post, el, clear, engineColor, fmtBytes, fmtDuration, fmtNum, fmtValue, go,
});

boot().catch(showError);
