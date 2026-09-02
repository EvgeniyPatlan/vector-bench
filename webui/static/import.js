"use strict";

const IM = { file: null, busy: false, sent: 0 };

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

function uploadRun() {
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
  const node = document.getElementById("import-status");
  const progress = el("span", { class: "muted" }, "starting…");
  node.append(progress);

  // XMLHttpRequest rather than fetch: it reports how much of the body has
  // actually gone, and it still hands over the status when the response is an
  // error. fetch() does neither -- a large upload that is refused or
  // interrupted arrives as "Failed to fetch" with the reason discarded, which
  // is indistinguishable from the server being unreachable.
  const xhr = new XMLHttpRequest();
  xhr.open("POST", `/api/import?${params}`);
  xhr.setRequestHeader("Content-Type", "application/gzip");

  xhr.upload.onprogress = (ev) => {
    if (!ev.lengthComputable) { progress.textContent = "uploading…"; return; }
    const pct = Math.round((ev.loaded / ev.total) * 100);
    progress.textContent = `${pct}% — ${fmtBytes(ev.loaded)} of ${fmtBytes(ev.total)}`;
  };

  const failed = (headline, detail) => {
    IM.busy = false;
    importForm();
    const where = document.getElementById("import-status");
    where.append(el("div", { class: "err" }, headline));
    if (detail) where.append(el("div", { class: "muted" }, detail));
  };

  xhr.onload = () => {
    IM.busy = false;
    if (xhr.status === 401) { window.location.href = "/login.html"; return; }
    let body = {};
    try { body = JSON.parse(xhr.responseText); } catch (err) { /* not JSON */ }
    if (xhr.status >= 200 && xhr.status < 300) {
      IM.file = null;
      api("/api/runs").then((data) => { S.runs = data.runs; })
        .finally(() => go(`#/run/${encodeURIComponent(body.run_id)}/overview`));
      return;
    }
    importForm();
    const where = document.getElementById("import-status");
    for (const e of body.errors || [body.error || `HTTP ${xhr.status}`]) {
      where.append(el("div", { class: "err" }, e));
    }
  };

  xhr.onerror = () => failed(
    `The connection dropped after ${fmtBytes(IM.sent)} of ${fmtBytes(IM.file.size)}.`,
    "Nothing was left behind on the server — an import is only unpacked once the "
    + "whole archive has arrived. If it keeps happening, copy the archive onto "
    + "the server and extract it into results/ instead; that needs no upload at "
    + "all. Check the server's own view with: "
    + "journalctl -u vector-bench-web | grep import");

  xhr.ontimeout = () => failed("The upload timed out.", null);
  xhr.onabort = () => failed("The upload was cancelled.", null);
  xhr.upload.onprogress = ((original) => (ev) => {
    IM.sent = ev.loaded;
    original(ev);
  })(xhr.upload.onprogress);

  xhr.send(IM.file);
}

window.renderImport = async function renderImport() {
  importForm();
};
