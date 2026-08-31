"use strict";

window.renderStatus = async function renderStatus() {
  const panel = document.getElementById("panel-status");
  clear(panel);
  const st = await api("/api/status");

  panel.append(el("h2", {}, "Status"));

  // What to do about any of this lives on Setup. Two pages answering the same
  // question is what made the old navigation hard to follow.
  if (st.active_job) {
    panel.append(el("div", { class: "warn" },
      `${st.active_job.kind || "job"} in progress: `,
      el("a", { href: "#/control/jobs" }, st.active_job.id),
      ` — ${st.active_job.command_display}`));
  }

  const setup = await api("/api/setup");
  if (setup.ready) {
    panel.append(el("div", { class: "ready" },
      "Ready to measure: every engine image is built, a corpus is here, and the "
      + "smoke profile has passed."));
  } else {
    const pending = setup.steps.filter((s) => !s.done && s.id !== "measure");
    panel.append(el("div", { class: "warn" },
      `Not ready: ${pending.map((s) => s.title.toLowerCase()).join(", ")}. `,
      el("a", { href: "#/control/setup" }, "Setup"),
      " walks through it."));
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
        ["Setup", "get this machine ready, in order", "build, fetch, run"],
        ["Engines", "build images, add an engine variant", "build"],
        ["Datasets", "download a corpus, or generate one", "fetch, generate"],
        ["Profiles & launch", "edit a profile, estimate cost, start a run", "run"],
        ["Jobs", "watch a running command, stop it", "—"],
        ["Runs → Report", "regenerate, download, export a bundle", "report, export"],
      ].map(([page, does, cmd]) => el("tr", {},
        el("td", {}, page), el("td", {}, does),
        el("td", {}, el("code", {}, cmd === "—" ? "—" : `./run-benchmark.sh ${cmd}`)))))));
};
