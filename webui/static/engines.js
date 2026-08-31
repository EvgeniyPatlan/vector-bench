"use strict";

const NG = { engines: [], drivers: [], name: null, text: "", dirty: false,
             chosen: [], target: "all", march: "" };

async function loadEngines() {
  const data = await api("/api/engines");
  NG.engines = data.engines;
  NG.drivers = data.drivers;
}

function engineTable() {
  const rows = NG.engines.map((e) => el("tr", {
    style: e.name === NG.name ? "font-weight:600" : "",
  },
    el("td", {},
      el("input", {
        type: "checkbox", checked: NG.chosen.includes(e.name),
        onchange: (ev) => {
          NG.chosen = ev.target.checked
            ? [...new Set([...NG.chosen, e.name])]
            : NG.chosen.filter((n) => n !== e.name);
          render();
        },
      })),
    el("td", { style: "cursor:pointer", onclick: () => openEngine(e.name) },
      el("span", {
        class: "pill", style: `border-color:${e.color};color:${e.color}`,
      }, e.name)),
    el("td", {}, e.label),
    el("td", {}, e.tag || "—"),
    el("td", { class: "num" }, e.port),
    el("td", {}, e.driver),
    el("td", {}, e.group),
    el("td", {}, ...["runtime", "bench"].map((kind) => el("span", {
      class: `pill ${e.built && e.built[kind] ? "ok" : "bad"}`,
    }, kind)))));

  return el("table", {},
    el("thead", {}, el("tr", {}, ...["", "engine", "implementation", "tag", "port",
                                     "driver", "group", "images"]
      .map((h) => el("th", {}, h)))),
    el("tbody", {}, ...rows));
}

function buildSection() {
  const box = el("div", {});
  box.append(el("h3", {}, "Build images"),
    el("p", { class: "muted" },
      "Each engine produces a runtime image (the server alone) and a bench "
      + "image (runtime plus the Python stack). AliSQL takes 1.5–3 hours; "
      + "pgvector about ten minutes."),
    el("div", { class: "warn" },
      "Never mix -march between engines. All of them compile SIMD distance "
      + "kernels, so rebuilding one with a different value turns the benchmark "
      + "into a comparison of compiler flags. Use native only when this machine "
      + "is also the benchmark host."));

  const argv = [];
  if (NG.chosen.length) argv.push("--engines", NG.chosen.join(","));
  if (NG.target !== "all") argv.push("--target", NG.target);
  if (NG.march) argv.push("--march", NG.march);

  box.append(el("div", { class: "row" },
    el("label", { class: "muted" }, "target ",
      el("select", { onchange: (ev) => { NG.target = ev.target.value; render(); } },
        ...["all", "runtime", "bench"].map((t) =>
          el("option", { value: t, selected: t === NG.target }, t)))),
    el("label", { class: "muted" }, "-march ",
      el("input", {
        value: NG.march, size: 14, placeholder: "x86-64-v3",
        oninput: (ev) => { NG.march = ev.target.value.trim(); },
      }))),
    commandPreview("build", argv.length ? argv : ["--engines", "…"]),
    el("div", { class: "row" },
      el("button", {
        class: "action", disabled: !NG.chosen.length,
        onclick: () => startJob(
          { kind: "build", engines: NG.chosen, target: NG.target,
            march: NG.march || undefined },
          document.getElementById("build-status")),
      }, NG.chosen.length ? `Build ${NG.chosen.join(", ")}` : "Tick an engine above"),
      el("span", { id: "build-status" })));
  return box;
}

async function openEngine(name) {
  const found = await api(`/api/engines/${encodeURIComponent(name)}`);
  NG.name = name;
  NG.text = found.text;
  NG.dirty = false;
  render();
}

async function addVariant() {
  const base = document.getElementById("clone-base").value;
  const name = (document.getElementById("clone-name").value || "").trim();
  const status = document.getElementById("engine-status");
  clear(status);
  if (!name) { status.append(el("span", { class: "err" }, "give the new engine a name")); return; }
  try {
    const res = await post("/api/engines/clone", { base, name });
    NG.name = res.name;
    NG.text = res.text;
    NG.dirty = true;
    render();
  } catch (err) {
    status.append(el("span", { class: "err" }, String(err)));
  }
}

async function validateEngine() {
  const status = document.getElementById("engine-status");
  clear(status);
  const text = document.getElementById("engine-text").value;
  const res = await post(`/api/engines/${encodeURIComponent(NG.name)}/validate`, { text });
  status.append(el("span", { class: res.ok ? "muted" : "err" },
    res.ok ? "valid" : "invalid"));
  for (const e of res.errors) status.append(el("div", { class: "err" }, e));
  for (const w of res.warnings) status.append(el("div", { class: "warn" }, w));
}

async function saveEngine() {
  const status = document.getElementById("engine-status");
  clear(status);
  const text = document.getElementById("engine-text").value;
  try {
    const res = await api(`/api/engines/${encodeURIComponent(NG.name)}`, {
      method: "PUT", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    NG.dirty = false;
    status.append(el("span", { class: "muted" }, `saved config/engines/${NG.name}.yml`));
    for (const w of res.warnings || []) status.append(el("div", { class: "warn" }, w));
    status.append(el("div", { class: "muted" }, "Now build it — tick it above, "
                                                + "or:"),
                  el("pre", { class: "log cmd" }, res.next));
    await loadEngines();
    render();
  } catch (err) {
    status.append(el("div", { class: "err" }, String(err)));
  }
}

function render() {
  const panel = document.getElementById("panel-engines");
  const open = document.getElementById("engine-text");
  if (open) NG.text = open.value;
  clear(panel);

  panel.append(el("h2", {}, "Engines"), engineTable());

  if (requiresControl(panel)) return;

  panel.append(buildSection());

  panel.append(el("h3", {}, "Add a variant"),
    el("p", { class: "muted" },
      "Copies an existing engine so you can point it at a different source tag "
      + "or build. The driver stays the same, which is what makes this "
      + "configuration rather than code."),
    el("div", { class: "warn" },
      "A new architecture — something none of these drivers speak — needs a "
      + "driver in harness/drivers/, an ann-benchmarks module and a Dockerfile. "
      + "This form cannot add one, and will say so if you name a driver that "
      + "does not exist. Drivers available: " + NG.drivers.join(", ")),
    el("div", { class: "row" },
      el("label", { class: "muted" }, "copy ",
        el("select", { id: "clone-base" },
          ...NG.engines.map((e) => el("option", { value: e.name }, e.name)))),
      el("label", { class: "muted" }, "as ",
        el("input", { id: "clone-name", placeholder: "perconaserver", size: 20 })),
      el("button", { class: "action secondary", onclick: addVariant }, "Create")));

  if (!NG.name) return;

  panel.append(el("h3", {}, `config/engines/${NG.name}.yml`),
    el("textarea", {
      class: "yaml", id: "engine-text", spellcheck: "false",
      oninput: () => { NG.dirty = true; },
    }, NG.text),
    el("div", { class: "row" },
      el("button", { class: "action secondary", onclick: validateEngine }, "Validate"),
      el("button", { class: "action", onclick: saveEngine }, "Save"),
      NG.dirty ? el("span", { class: "muted" }, "unsaved") : null,
      el("span", { id: "engine-status" })));
}

window.renderEngines = async function renderEngines() {
  if (!NG.engines.length) await loadEngines();
  render();
};
