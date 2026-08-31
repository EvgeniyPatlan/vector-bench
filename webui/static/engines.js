"use strict";

const NG = { engines: [], drivers: [], name: null, text: "", dirty: false };

async function loadEngines() {
  const data = await api("/api/engines");
  NG.engines = data.engines;
  NG.drivers = data.drivers;
}

function engineTable() {
  const rows = NG.engines.map((e) => el("tr", {
    onclick: () => openEngine(e.name),
    style: "cursor:pointer" + (e.name === NG.name ? ";font-weight:600" : ""),
  },
    el("td", {}, el("span", {
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
    el("thead", {}, el("tr", {}, ...["engine", "implementation", "tag", "port",
                                     "driver", "group", "images"]
      .map((h) => el("th", {}, h)))),
    el("tbody", {}, ...rows));
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
    status.append(el("div", { class: "muted" }, "Now build it:"),
                  el("pre", { class: "log" }, res.next));
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

  if (!S.control) {
    panel.append(el("p", { class: "muted" },
      "Read-only. Restart with --allow-control to add or edit an engine."));
    return;
  }

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
