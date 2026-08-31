"use strict";

window.renderStatus = async function renderStatus() {
  const panel = document.getElementById("panel-status");
  clear(panel);
  const st = await api("/api/status");

  panel.append(el("h2", {}, "Status"));

  const missingImages = st.engines.filter((e) => !e.bench_built);
  const noDatasets = st.datasets_present.length === 0;
  const freeGb = st.disk.free_bytes ? st.disk.free_bytes / 1024 ** 3 : null;

  // What to do next, before the inventory. An engine list is only useful once
  // you know whether you can measure anything with it.
  const todo = [];
  if (missingImages.length) {
    todo.push([`${missingImages.length} engine image(s) not built`,
               "Build them on the Engines page", "#/control/engines"]);
  }
  if (noDatasets) {
    todo.push(["no datasets downloaded",
               "Get one on the Datasets page", "#/control/datasets"]);
  }
  if (freeGb !== null && freeGb < 80) {
    todo.push([`${freeGb.toFixed(0)} GB free`,
               "A full run wants about 80 GB", null]);
  }

  if (st.active_job) {
    panel.append(el("div", { class: "warn" },
      `${st.active_job.kind || "job"} in progress: `,
      el("a", { href: "#/control/jobs" }, st.active_job.id),
      ` — ${st.active_job.command_display}`));
  }

  if (todo.length) {
    panel.append(el("h3", {}, "Before you can measure"));
    for (const [what, hint, href] of todo) {
      panel.append(el("div", { class: "warn" },
        what + " — ",
        href ? el("a", { href }, hint) : hint));
    }
  } else {
    panel.append(el("div", { class: "ready" },
      "Ready to measure: every engine image is built and at least one dataset "
      + "is here."));
  }

  panel.append(el("h3", {}, "Engines"));
  panel.append(el("table", {},
    el("thead", {}, el("tr", {}, ...["engine", "tag", "runtime image", "bench image"]
      .map((h) => el("th", {}, h)))),
    el("tbody", {}, ...st.engines.map((e) => el("tr", {},
      el("td", {}, el("span", {
        class: "pill", style: `border-color:${e.color};color:${e.color}`,
      }, e.name)),
      el("td", {}, e.tag || "—"),
      el("td", {}, el("span", { class: `pill ${e.runtime_built ? "ok" : "bad"}` },
        e.runtime_built ? "built" : "missing")),
      el("td", {}, el("span", { class: `pill ${e.bench_built ? "ok" : "bad"}` },
        e.bench_built ? "built" : "missing")))))));

  panel.append(el("h3", {}, "This machine"));
  const dl = el("dl", { class: "kv" });
  const pairs = [
    ["engines with a bench image", `${st.engines_ready} of ${st.engines.length}`],
    ["datasets present", `${st.datasets_present.length} of ${st.datasets_known} known`],
    ["runs recorded", st.runs],
    ["disk free", st.disk.free_bytes ? fmtBytes(st.disk.free_bytes) : "—"],
    ["control", st.control_enabled ? "enabled" : "read-only"],
  ];
  for (const [k, v] of pairs) dl.append(el("dt", {}, k), el("dd", {}, String(v)));
  panel.append(dl);

  panel.append(el("h3", {}, "What this can do"),
    el("p", { class: "muted" },
      "Every action here runs the same command you would type, and shows it "
      + "before it does."),
    el("table", {},
      el("thead", {}, el("tr", {}, ...["page", "does", "command"]
        .map((h) => el("th", {}, h)))),
      el("tbody", {}, ...[
        ["Engines", "build images, add an engine variant", "build"],
        ["Datasets", "download the HDF5 corpora", "fetch"],
        ["Profiles & launch", "edit a profile, estimate cost, start a run", "run"],
        ["Jobs", "watch a running command, stop it", "—"],
        ["Runs → Report", "regenerate a report", "report"],
      ].map(([page, does, cmd]) => el("tr", {},
        el("td", {}, page), el("td", {}, does),
        el("td", {}, el("code", {}, cmd === "—" ? "—" : `./run-benchmark.sh ${cmd}`)))))));
};
