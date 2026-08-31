"use strict";

const IM = { file: null, busy: false };

function importForm() {
  const panel = document.getElementById("panel-import");
  clear(panel);

  panel.append(el("h2", {}, "Import a run"),
    el("p", { class: "muted" },
      "A run measured on another machine, as the .tar.gz that "
      + "./run-benchmark.sh export produces. Its manifest travels with it, so "
      + "it keeps that machine's CPU, SIMD flags and limits — the numbers stay "
      + "scoped to where they were taken."),
    el("p", { class: "muted" },
      "If you have a shell on this box, copying the directory into results/ "
      + "does the same thing and needs no upload."));

  if (requiresControl(panel)) return;

  panel.append(
    el("div", { class: "row" },
      el("input", {
        type: "file", id: "import-file", accept: ".gz,.tgz,.tar.gz,application/gzip",
        onchange: (ev) => { IM.file = ev.target.files[0] || null; importForm(); },
      })),
    IM.file
      ? el("p", { class: "muted" },
          `${IM.file.name} — ${fmtBytes(IM.file.size)}`)
      : null,
    el("div", { class: "row" },
      el("label", { class: "muted" }, "call it ",
        el("input", { id: "import-name", size: 22,
                      placeholder: "keep the name in the archive" })),
      el("label", { class: "muted" }, "label ",
        el("input", { id: "import-label", size: 26,
                      placeholder: "EPYC rig, main profile" })),
      el("label", { class: "muted" }, "measured on ",
        el("input", { id: "import-source", size: 16, placeholder: "bench-rig-2" }))),
    el("p", { class: "muted" },
      "A name is only needed when one is already taken — two machines running "
      + "the same profile on the same day produce the same run id. The label is "
      + "a nickname shown in the run list; it is kept beside the manifest, never "
      + "written into it."),
    el("div", { class: "row" },
      el("button", {
        class: "action", disabled: !IM.file || IM.busy,
        onclick: uploadRun,
      }, IM.busy ? "Importing…" : "Import"),
      el("span", { id: "import-status" })));
}

async function uploadRun() {
  const status = document.getElementById("import-status");
  clear(status);
  if (!IM.file) return;

  const params = new URLSearchParams();
  for (const [id, key] of [["import-name", "run_id"], ["import-label", "label"],
                           ["import-source", "source"]]) {
    const value = (document.getElementById(id).value || "").trim();
    if (value) params.set(key, value);
  }

  IM.busy = true;
  importForm();
  document.getElementById("import-status").textContent = "uploading…";
  try {
    const res = await fetch(`/api/import?${params}`, {
      method: "POST",
      headers: { "Content-Type": "application/gzip" },
      body: IM.file,
    });
    const body = await res.json().catch(() => ({}));
    IM.busy = false;
    if (res.status === 401) {
      window.location.href = "/login.html";
      return;
    }
    if (!res.ok) {
      importForm();
      const node = document.getElementById("import-status");
      for (const e of body.errors || [body.error || `HTTP ${res.status}`]) {
        node.append(el("div", { class: "err" }, e));
      }
      return;
    }
    IM.file = null;
    S.runs = (await api("/api/runs")).runs;
    go(`#/run/${encodeURIComponent(body.run_id)}/overview`);
  } catch (err) {
    IM.busy = false;
    importForm();
    const node = document.getElementById("import-status");
    node.append(el("div", { class: "err" }, String(err)));
    // fetch() reports a connection that ends mid-upload as "Failed to fetch"
    // and keeps the status to itself, so say what it usually means rather than
    // leaving a browser-level string as the whole explanation.
    if (String(err).includes("Failed to fetch") || err instanceof TypeError) {
      node.append(el("div", { class: "muted" },
        "The connection ended before the upload finished. Usually the session "
        + "expired mid-upload — reload the page, sign in, and try again. If it "
        + "persists, copy the archive into results/ on the server instead; "
        + "an unpacked run directory needs no upload."));
    }
  }
}

window.renderImport = async function renderImport() {
  importForm();
};
