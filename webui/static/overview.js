"use strict";

function kvList(pairs) {
  const dl = el("dl", { class: "kv" });
  for (const [k, v] of pairs) {
    if (v === null || v === undefined || v === "") continue;
    dl.append(el("dt", {}, k), el("dd", {}, String(v)));
  }
  return dl;
}

function enginesTable(engines) {
  const rows = Object.entries(engines).map(([name, info]) => {
    const build = info.build || {};
    const colour = engineColor(name, 0);
    return el("tr", {},
      el("td", {}, el("span", { class: "pill", style: `border-color:${colour};color:${colour}` }, name)),
      el("td", {}, ENGINE_LABEL[name] || name),
      el("td", {}, build.tag || info.tag || "—"),
      el("td", {}, build.march || "—"),
      el("td", {}, build.build_type || "—"));
  });
  return el("table", {},
    el("thead", {}, el("tr", {}, ...["engine", "implementation", "tag", "-march", "build"]
      .map((h) => el("th", {}, h)))),
    el("tbody", {}, ...rows));
}

function phasesTable(phases) {
  const rows = phases.map((p) => el("tr", {},
    el("td", {}, p.phase), el("td", {}, p.engine), el("td", {}, p.dataset),
    el("td", {}, p.resource_pass),
    el("td", {}, el("span", { class: `pill ${p.status === "completed" ? "ok" : "bad"}` }, p.status)),
    el("td", { class: "num" }, fmtDuration(p.duration_s))));
  return el("table", {},
    el("thead", {}, el("tr", {}, ...["measurement", "engine", "dataset", "pass", "status", "duration"]
      .map((h) => el("th", {}, h)))),
    el("tbody", {}, ...rows));
}

window.renderOverview = function renderOverview() {
  const panel = document.getElementById("panel-overview");
  clear(panel);
  if (!S.run) { panel.append(el("p", { class: "empty" }, "Select a run.")); return; }

  const { summary, manifest } = S.run;
  const host = manifest.host || {};
  const cpu = host.cpu || {};

  panel.append(el("h2", {}, summary.run_id),
    summary.description ? el("p", { class: "muted" }, summary.description) : null);

  // Validity first: the report puts caveats above the charts for the same
  // reason, and a warning read after a conclusion is read too late.
  const warnings = manifest.warnings || [];
  if (warnings.length) {
    panel.append(el("h3", {}, `Validity — read before the numbers`));
    for (const w of warnings) panel.append(el("div", { class: "warn" }, w));
  }

  panel.append(el("h3", {}, "This run"));
  panel.append(kvList([
    ["status", summary.status],
    ["profile", summary.profile],
    ["resource pass", summary.resource_pass],
    ["datasets", (summary.datasets || []).join(", ")],
    ["started", summary.started_at],
    ["measured time", fmtDuration(summary.duration_s)],
    ["records", `${summary.record_count} (${summary.has_records ? "merged" : "ops only"})`],
  ]));

  panel.append(el("h3", {}, "Hardware the numbers are scoped to"));
  panel.append(kvList([
    ["CPU", cpu.model],
    ["cores", cpu.physical_cores
      ? `${cpu.physical_cores} physical / ${cpu.logical_cpus} logical${cpu.hybrid ? " (hybrid)" : ""}`
      : null],
    ["SIMD", (cpu.simd_flags || []).join(" ")],
    ["AVX-512", cpu.has_avx512 === undefined ? null : (cpu.has_avx512 ? "yes" : "no")],
    ["RAM", host.total_ram_bytes ? fmtBytes(host.total_ram_bytes) : null],
    ["kernel", host.kernel],
    ["docker", host.docker_version],
  ]));

  const engines = manifest.engines || {};
  if (Object.keys(engines).length) {
    panel.append(el("h3", {}, "Engines"), enginesTable(engines));
  }

  const resolved = (manifest.config || {}).resolved_resources;
  if (resolved) {
    panel.append(el("h3", {}, "Resolved limits — identical per engine by construction"));
    panel.append(kvList(Object.entries(resolved)
      .filter(([k, v]) => !k.startsWith("_") && v !== null && v !== "")
      .map(([k, v]) => [k.replace(/_/g, " "),
                        k.endsWith("_bytes") ? fmtBytes(v) : v])));
  }

  const phases = manifest.phases || [];
  if (phases.length) {
    panel.append(el("h3", {}, "What ran"), phasesTable(phases));
  }
};
