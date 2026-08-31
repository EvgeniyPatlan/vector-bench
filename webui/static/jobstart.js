"use strict";

// One place that starts a long command and shows what it is about to run.
// Every button that kicks off work goes through here, so the command is always
// visible — the UI teaches the CLI rather than replacing it with a mystery.

function commandPreview(kind, argv) {
  return el("pre", { class: "log cmd" },
    ["./run-benchmark.sh", kind, ...argv].join(" "));
}

async function startJob(spec, statusNode) {
  clear(statusNode);
  try {
    const res = await post("/api/jobs", spec);
    statusNode.append(
      el("span", { class: "muted" }, "started "),
      el("a", { href: "#/control/jobs" }, res.job.id));
    go("#/control/jobs");
    return res.job;
  } catch (err) {
    statusNode.append(el("div", { class: "err" }, String(err)));
    return null;
  }
}

function requiresControl(node) {
  if (S.control) return false;
  node.append(el("p", { class: "muted" },
    "Read-only. Restart with --allow-control to do this from here."));
  return true;
}

Object.assign(window, { commandPreview, startJob, requiresControl });
