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

const CONTROL_SECTIONS = [
  { id: "profiles", label: "Profiles" },
  { id: "engines", label: "Engines" },
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

function renderRunList() {
  const list = document.getElementById("run-list");
  const needle = document.getElementById("run-filter").value.toLowerCase();
  clear(list);

  const shown = S.runs.filter((r) => {
    if (!needle) return true;
    return [r.dir_name, r.profile, r.resource_pass, (r.engines || []).join(" "),
            (r.datasets || []).join(" ")].join(" ").toLowerCase().includes(needle);
  });

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

function renderControlList() {
  const group = document.getElementById("control-group");
  group.hidden = !S.control;
  if (!S.control) return;
  const list = document.getElementById("control-list");
  clear(list);
  for (const section of CONTROL_SECTIONS) {
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
      ? (CONTROL_SECTIONS.find((s) => s.id === S.route.section) || {}).label || ""
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

const PANELS = ["overview", "explore", "report", "profiles", "engines", "jobs"];

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
  if (m && CONTROL_SECTIONS.some((s) => s.id === m[1])) {
    return { kind: "control", section: m[1] };
  }
  return null;
}

function go(hash) {
  if (window.location.hash === hash) applyRoute();
  else window.location.hash = hash;
}

const RENDERERS = {
  overview: () => window.renderOverview(),
  explore: () => window.renderExplore(),
  report: () => renderReport(),
  profiles: () => window.renderProfiles(),
  engines: () => window.renderEngines(),
  jobs: () => window.renderJobs(),
};

async function applyRoute() {
  const wanted = parseHash() || { kind: "run", tab: "overview" };

  if (wanted.kind === "control") {
    S.route = { ...S.route, kind: "control", section: wanted.section };
    renderRunList(); renderControlList(); renderTabs();
    showPanel(wanted.section);
    await RENDERERS[wanted.section]();
    return;
  }

  const runId = wanted.runId && S.runs.some((r) => r.dir_name === wanted.runId)
    ? wanted.runId
    : (S.runs.length ? S.runs[0].dir_name : null);

  S.route = { ...S.route, kind: "run", tab: wanted.tab };
  if (!runId) {
    renderRunList(); renderControlList(); renderTabs();
    showPanel("overview");
    const panel = document.getElementById("panel-overview");
    clear(panel);
    panel.append(el("p", { class: "empty" }, "No runs in results/ yet."));
    return;
  }

  if (runId !== S.runId) await loadRun(runId);
  renderRunList(); renderControlList(); renderTabs();
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
  if (!S.run.summary.has_report) {
    panel.append(el("p", { class: "empty" },
      "No generated report for this run yet."),
      el("pre", { class: "log" },
        `./run-benchmark.sh report --run-dir results/${S.runId}`));
    return;
  }
  panel.append(el("iframe", {
    class: "report", src: `/runs/${encodeURIComponent(S.runId)}/report/report.html`,
  }));
}

// -- boot --------------------------------------------------------------

async function boot() {
  document.getElementById("run-filter").addEventListener("input", renderRunList);
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
