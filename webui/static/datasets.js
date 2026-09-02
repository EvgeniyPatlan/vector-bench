"use strict";

const DS = { datasets: [], localOnly: [], dir: "", chosen: [] };

async function loadDatasets() {
  const data = await api("/api/datasets");
  DS.datasets = data.datasets;
  DS.localOnly = data.local_only;
  DS.dir = data.datasets_dir;
}

function shape(d) {
  if (!d.train || !d.dim) return "—";
  const rows = d.train >= 1e6 ? `${(d.train / 1e6).toFixed(2)}M` : `${(d.train / 1e3).toFixed(0)}k`;
  return `${rows} × ${d.dim}`;
}

window.renderDatasets = async function renderDatasets() {
  if (!DS.datasets.length) await loadDatasets();
  const panel = document.getElementById("panel-datasets");
  clear(panel);

  const missing = DS.datasets.filter((d) => !d.downloaded && !d.generated);
  panel.append(el("h2", {}, "Datasets"),
    el("p", { class: "muted" },
      `The same HDF5 files feed both measurement paths, so ann-benchmarks and `
      + `the ops harness score against identical ground truth. In ${DS.dir}`));

  const rows = DS.datasets.map((d) => el("tr", {},
    el("td", {},
      el("input", {
        type: "checkbox",
        disabled: d.downloaded || d.generated,
        checked: DS.chosen.includes(d.name),
        onchange: (ev) => {
          DS.chosen = ev.target.checked
            ? [...new Set([...DS.chosen, d.name])]
            : DS.chosen.filter((n) => n !== d.name);
          window.renderDatasets();
        },
      })),
    el("td", {}, d.name),
    el("td", {}, shape(d)),
    el("td", {}, d.metric || "—"),
    el("td", {}, d.role || "—"),
    el("td", { class: "num" },
      d.bytes_on_disk ? fmtBytes(d.bytes_on_disk)
                      : (d.approx_bytes ? "~" + fmtBytes(d.approx_bytes) : "—")),
    el("td", {}, d.downloaded
      ? el("span", { class: "pill ok" }, "present")
      : (d.generated ? el("span", { class: "pill" }, "must be generated")
                     : el("span", { class: "pill bad" }, "missing")))));

  panel.append(el("table", {},
    el("thead", {}, el("tr", {}, ...["", "dataset", "shape", "metric", "role",
                                     "size", "status"].map((h) => el("th", {}, h)))),
    el("tbody", {}, ...rows)));

  if (DS.localOnly.length) {
    panel.append(el("h3", {}, "Also here"),
      el("p", { class: "muted" },
        DS.localOnly.map((d) => `${d.name} (${fmtBytes(d.bytes_on_disk)})`).join(", ")));
  }

  panel.append(el("h3", {}, "Download"));
  if (requiresControl(panel)) return;

  if (!missing.length) {
    panel.append(el("p", { class: "muted" }, "Every downloadable dataset is here."));
  } else {
    panel.append(el("p", { class: "muted" },
      "Downloads are resumable and size-verified, so an interrupted transfer is "
      + "never mistaken for a complete dataset."));
  }

  panel.append(commandPreview("fetch", DS.chosen.length
    ? ["--datasets", DS.chosen.join(",")] : ["--datasets", "…"]));
  panel.append(el("div", { class: "row" },
    el("button", {
      class: "action", disabled: !DS.chosen.length,
      onclick: (ev) => startJob({ kind: "fetch", datasets: DS.chosen },
                                document.getElementById("fetch-status")),
      // Name them while the list is short; "1 dataset(s)" is neither.
    }, DS.chosen.length === 0 ? "Select a dataset"
      : DS.chosen.length <= 2 ? `Download ${DS.chosen.join(", ")}`
      : `Download ${DS.chosen.length} datasets`),
    el("span", { id: "fetch-status" })));

  const generated = DS.datasets.filter((d) => d.generated && !d.downloaded);
  if (!generated.length) return;

  panel.append(el("h3", {}, "Generate"),
    el("p", { class: "muted" },
      "These are not published as prebuilt files, so fetch cannot retrieve "
      + "them. ann-benchmarks assembles them from a source corpus and then "
      + "computes exact ground truth by brute force."),
    el("div", { class: "warn" },
      "This is not a download. For the dbpedia family the full 1M-row source "
      + "is fetched whichever size you pick (~6–10 GB) — choosing a smaller "
      + "variant only shrinks the ground-truth computation and every later "
      + "engine load. Budget hours and about 20 GB of working space for the "
      + "1000k variant, and a bench image must already be built."),
    el("div", { class: "row" },
      el("label", { class: "muted" }, "dataset ",
        el("select", { id: "gen-pick" },
          ...generated.map((d) => el("option", { value: d.name }, d.name))))),
    commandPreview("generate", [generated[0].name]),
    el("div", { class: "row" },
      el("button", {
        class: "action secondary",
        onclick: () => {
          const name = document.getElementById("gen-pick").value;
          const rows = (DS.datasets.find((d) => d.name === name) || {}).train;
          const warning = rows && rows >= 500000
            ? `Generate ${name}? Ground truth for ${(rows / 1e6).toFixed(2)}M `
              + "vectors at 1536 dimensions is hours of brute force."
            : `Generate ${name}?`;
          if (!confirm(warning)) return;
          startJob({ kind: "generate", datasets: [name] },
                   document.getElementById("gen-status"));
        },
      }, "Generate"),
      el("span", { id: "gen-status" })));

  // The preview should follow the picker, not stay on the first entry.
  const picker = document.getElementById("gen-pick");
  if (picker) {
    picker.addEventListener("change", () => {
      const preview = panel.querySelector("pre.cmd:last-of-type");
      if (preview) {
        preview.textContent =
          `./run-benchmark.sh generate ${picker.value}`;
      }
    });
  }
};
