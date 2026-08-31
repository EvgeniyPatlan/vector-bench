"use strict";

const SU = { plan: null, march: "", engines: [] };

function stepHead(index, step) {
  return el("div", { class: "step-head" },
    el("span", { class: `step-num ${step.done ? "done" : ""}` },
      step.done ? "✓" : String(index + 1)),
    el("div", {},
      el("div", { class: "step-title" }, step.title),
      el("div", { class: "muted" }, step.summary)));
}

// -- step 1: images ----------------------------------------------------

function imagesStep(step) {
  const { engines, march } = step.detail;
  const missing = engines.filter((e) => !e.bench_built);
  const box = el("div", {});

  box.append(el("table", {},
    el("thead", {}, el("tr", {}, ...["", "engine", "tag", "image", "-march"]
      .map((h) => el("th", {}, h)))),
    el("tbody", {}, ...engines.map((e) => el("tr", {},
      el("td", {}, el("input", {
        type: "checkbox", disabled: e.bench_built,
        checked: SU.engines.includes(e.name),
        onchange: (ev) => {
          SU.engines = ev.target.checked
            ? [...new Set([...SU.engines, e.name])]
            : SU.engines.filter((n) => n !== e.name);
          renderSetupBody();
        },
      })),
      el("td", {}, el("span", {
        class: "pill", style: `border-color:${e.color};color:${e.color}`,
      }, e.name)),
      el("td", {}, e.tag || "—"),
      el("td", {}, el("span", { class: `pill ${e.bench_built ? "ok" : "bad"}` },
        e.bench_built ? "built" : "missing")),
      el("td", {}, e.march || "—"))))));

  if (march.mixed) {
    box.append(el("div", { class: "warn" },
      el("strong", {}, "These images do not agree on -march: "),
      march.values.join(", ")
      + ". Every engine compiles SIMD distance kernels, so a comparison across "
      + "these measures compiler flags as much as implementations. Rebuild them "
      + "on one value before measuring anything."));
  }

  if (!missing.length) {
    return box;
  }

  const pinned = march.agreed;
  const value = SU.march || pinned || "x86-64-v3";
  box.append(el("div", { class: "row" },
    el("label", { class: "muted" }, "-march ",
      el("input", {
        id: "setup-march", value, size: 14,
        oninput: (ev) => { SU.march = ev.target.value.trim(); renderSetupBody(); },
      })),
    el("button", {
      class: "action secondary",
      onclick: () => { SU.march = "native"; renderSetupBody(); },
    }, "native"),
    el("span", { class: "muted" },
      "native only when this machine is also the benchmark host")));

  if (pinned && value !== pinned) {
    box.append(el("div", { class: "warn" },
      `Images already here were built with -march=${pinned}. Building with `
      + `${value} would make the comparison a comparison of compiler flags. `
      + "Match it, or rebuild everything on the new value."));
  }

  // The extras are opt-in for the same reason `build` leaves them out of its
  // own default: each is another hour of compiling or another process to stand
  // up, and nothing needs them to produce a result.
  const missingOriginal = missing.filter((e) => e.group === "original");
  const chosen = SU.engines.length ? SU.engines : missingOriginal.map((e) => e.name);
  const missingExtra = missing.filter((e) => e.group === "extra");

  box.append(
    el("p", { class: "muted" },
      "pgvector takes about ten minutes; MariaDB about forty; AliSQL 1.5–3 hours, "
      + "and looks stalled while it compiles a bundled DuckDB it does not use."),
    missingExtra.length
      ? el("p", { class: "muted" },
          `${missingExtra.map((e) => e.name).join(", ")} are not in the original `
          + "comparison and are left out unless you tick them — each is another "
          + "image to compile or another process to stand up, and a result does "
          + "not need them.")
      : null,
    commandPreview("build", chosen.length
      ? ["--engines", chosen.join(","), "--march", value]
      : ["--march", value]),
    el("div", { class: "row" },
      el("button", {
        class: "action", disabled: !chosen.length,
        onclick: () => startJob({ kind: "build", engines: chosen, march: value },
                                document.getElementById("setup-build-status")),
      }, SU.engines.length
        ? `Build ${SU.engines.length} engine(s)`
        : `Build the ${missingOriginal.length} missing`),
      el("span", { id: "setup-build-status" })));
  return box;
}

// -- step 2: datasets --------------------------------------------------

function datasetsStep(step) {
  const d = step.detail;
  const box = el("div", {});
  if (d.present.length) {
    box.append(el("p", { class: "muted" }, "Here: " + d.present.join(", ")));
  }
  if (!d.smoke_dataset_present) {
    box.append(
      el("p", { class: "muted" },
        `${d.smoke_dataset} is what the smoke profile uses — 217 MB, the `
        + "smallest thing that proves the pipeline on real data."),
      commandPreview("fetch", ["--datasets", d.smoke_dataset]),
      el("div", { class: "row" },
        el("button", {
          class: "action",
          onclick: () => startJob({ kind: "fetch", datasets: [d.smoke_dataset] },
                                  document.getElementById("setup-fetch-status")),
        }, "Download it"),
        el("span", { id: "setup-fetch-status" })));
  }
  box.append(el("p", { class: "muted" },
    el("a", { href: "#/control/datasets" }, "Datasets"),
    " has the rest, and the ones that must be generated rather than downloaded."));
  return box;
}

// -- step 3: smoke -----------------------------------------------------

function smokeStep(step) {
  const box = el("div", {});
  if (step.detail.run) {
    box.append(el("p", { class: "muted" },
      "Already proved by ",
      el("a", { href: `#/run/${step.detail.run.dir_name}/overview` },
        step.detail.run.dir_name),
      ". Run it again after rebuilding images."));
  }
  box.append(
    el("p", { class: "muted" },
      "Exercises every stage for every engine on a small corpus: images start, "
      + "vector DDL is accepted, the index is actually used, records are written "
      + "and the report renders. About 45 minutes per resource pass."),
    el("div", { class: "warn" },
      "Do not skip it. It is far cheaper than discovering a broken image eight "
      + "hours into a real run."),
    commandPreview("run", ["--profile", "smoke"]),
    el("div", { class: "row" },
      el("button", {
        class: "action",
        onclick: () => startJob({ kind: "run", profile: "smoke" },
                                document.getElementById("setup-smoke-status")),
      }, "Run the smoke profile"),
      el("span", { id: "setup-smoke-status" })));
  return box;
}

// -- step 4: measure ---------------------------------------------------

function measureStep() {
  return el("div", {},
    el("p", { class: "muted" },
      "The smoke profile proves the pipeline; it measures nothing — its grids "
      + "are far too sparse to draw a curve from. "),
    el("p", { class: "muted" },
      "Pick a profile on ",
      el("a", { href: "#/control/profiles" }, "Profiles & launch"),
      ", where the ingest estimate updates as you narrow the engines, datasets "
      + "and resource pass. Read it before starting: main is about two days."));
}

const BODIES = {
  images: imagesStep, datasets: datasetsStep,
  smoke: smokeStep, measure: measureStep,
};

function renderSetupBody() {
  const panel = document.getElementById("panel-setup");
  const open = document.getElementById("setup-march");
  if (open) SU.march = open.value.trim();
  clear(panel);

  const plan = SU.plan;
  panel.append(el("h2", {}, "Setup"));

  if (plan.ready) {
    panel.append(el("div", { class: "ready" },
      "This machine can measure: every engine image is built, a corpus is here, "
      + "and the smoke profile has passed."));
  } else {
    panel.append(el("p", { class: "muted" },
      "Four things stand between a fresh checkout and a number you can trust. "
      + "Each one below says what it needs and does it."));
  }

  if (plan.disk && plan.disk.enough === false) {
    panel.append(el("div", { class: "warn" },
      `${fmtBytes(plan.disk.free_bytes)} free. A full run wants about `
      + `${fmtBytes(plan.disk.wanted_bytes)}; a long one that fills the disk `
      + "fails at the end, having spent the time."));
  }

  if (!S.control) {
    panel.append(el("p", { class: "muted" },
      "Read-only: the steps below show what is missing but cannot act. "
      + "Restart with --allow-control."));
  }

  plan.steps.forEach((step, i) => {
    const section = el("section", {
      class: "step" + (step.done ? " done" : "")
        + (step.id === plan.next ? " next" : ""),
    }, stepHead(i, step));
    // Only the step you are on, and anything unfinished after it, needs its
    // controls open; a finished step is a tick, not a form.
    if (!step.done || step.id === plan.next) {
      if (S.control || step.id === "measure") section.append(BODIES[step.id](step));
    }
    panel.append(section);
  });
}

window.renderSetup = async function renderSetup() {
  SU.plan = await api("/api/setup");
  renderSetupBody();
};
