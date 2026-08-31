"use strict";

const C = {
  engines: [], datasets: [], profiles: [],
  profile: null, profileText: "", dirty: false,
  plan: { engines: [], datasets: [], resource_pass: "", phases: "both",
          run_id: "", resume: false, force: false, fail_fast: false,
          no_report: false },
  jobs: [], jobId: null, logOffset: 0, poll: null,
};

async function loadReference() {
  if (C.engines.length) return;
  const [engines, datasets, profiles] = await Promise.all([
    api("/api/engines"), api("/api/datasets"), api("/api/profiles"),
  ]);
  C.engines = engines.engines;
  C.datasets = datasets.datasets;
  C.profiles = profiles.profiles;
}

// -- configure ---------------------------------------------------------

function planFromForm() {
  const spec = { ...C.plan };
  if (C.profile) spec.profile = C.profile;
  return spec;
}

async function refreshEstimate() {
  const box = document.getElementById("estimate");
  if (!box || !C.profile) return;
  try {
    const est = await api("/api/estimate", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(planFromForm()),
    });
    clear(box);
    const per = Object.entries(est.per_engine_hours || {})
      .map(([e, h]) => `${e} ~${h.toFixed(1)}h`).join("  |  ");
    box.append(el("strong", {}, `estimated ingest ~${est.total_hours.toFixed(1)} h`),
      el("div", { class: "muted" },
        `${est.passes} pass(es) · ${est.m_values} ann M value(s) · ${est.ops_m_values} ops M value(s)`),
      per ? el("div", { class: "muted" }, per) : null);
    if (est.datasets_without_estimate.length) {
      box.append(el("div", { class: "warn" },
        `no measured ingest rate for: ${est.datasets_without_estimate.join(", ")}`));
    }
    if (est.long_run) {
      box.append(el("div", { class: "warn" },
        "Long run. Every M value reloads the whole corpus; cut ann.m_values or the dataset list to reduce it roughly proportionally."));
    }
  } catch (err) {
    clear(box);
    box.append(el("span", { class: "err" }, String(err)));
  }
}

async function openProfile(name) {
  C.profile = name;
  const profile = await api(`/api/profiles/${encodeURIComponent(name)}`);
  C.profileText = profile.text;
  C.dirty = false;
  const area = document.getElementById("profile-text");
  if (area) area.value = profile.text;
  const datasets = (profile.parsed || {}).datasets || [];
  C.plan = { ...C.plan, datasets: [...datasets] };
  renderProfilesBody();
  refreshEstimate();
}

async function saveProfile() {
  const name = document.getElementById("profile-name").value.trim();
  const text = document.getElementById("profile-text").value;
  const status = document.getElementById("profile-status");
  clear(status);
  try {
    const res = await api(`/api/profiles/${encodeURIComponent(name)}`, {
      method: "PUT", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    C.dirty = false;
    status.append(el("span", { class: "muted" }, `saved config/profiles/${name}.yml`));
    for (const w of res.warnings || []) status.append(el("div", { class: "warn" }, w));
    C.profiles = (await api("/api/profiles")).profiles;
    C.profile = name;
    renderProfilesBody();
  } catch (err) {
    status.append(el("div", { class: "err" }, String(err)));
  }
}

async function validateProfile() {
  const name = document.getElementById("profile-name").value.trim();
  const text = document.getElementById("profile-text").value;
  const status = document.getElementById("profile-status");
  clear(status);
  const res = await api(`/api/profiles/${encodeURIComponent(name)}/validate`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  status.append(el("span", { class: res.ok ? "muted" : "err" },
    res.ok ? "valid" : "invalid"));
  for (const e of res.errors) status.append(el("div", { class: "err" }, e));
  for (const w of res.warnings) status.append(el("div", { class: "warn" }, w));
}

function toggleIn(list, value, on) {
  return on ? [...new Set([...list, value])] : list.filter((v) => v !== value);
}

function checkboxRow(title, items, selected, onChange) {
  const box = el("div", { class: "facet" }, el("strong", {}, title));
  for (const item of items) {
    box.append(el("label", {},
      el("input", {
        type: "checkbox", checked: selected.includes(item.value),
        onchange: (ev) => onChange(item.value, ev.target.checked),
      }), " ", item.label));
  }
  return box;
}

function renderProfilesBody() {
  // Toggling anything in the run plan re-renders the whole panel, which would
  // otherwise reset the editor to the last loaded text and lose unsaved edits.
  const open = document.getElementById("profile-text");
  if (open) C.profileText = open.value;

  const panel = document.getElementById("panel-profiles");
  clear(panel);

  panel.append(el("h2", {}, "Configure a run"));

  const picker = el("select", {
    onchange: (ev) => openProfile(ev.target.value),
  }, el("option", { value: "" }, "— choose a profile —"),
    ...C.profiles.map((p) => el("option", { value: p.name, selected: p.name === C.profile },
      `${p.name}${p.description ? " — " + p.description : ""}`)));

  panel.append(el("div", { class: "row" },
    el("label", { class: "muted" }, "profile ", picker),
    el("button", {
      class: "action secondary",
      onclick: () => {
        const name = prompt("new profile name (a-z0-9_-)");
        if (!name) return;
        C.profile = name;
        C.profileText = `name: ${name}\ndescription: \ndatasets:\n  - fashion-mnist-784-euclidean\nk: 10\nruns: 1\n\nann:\n  enabled: true\n  m_values: [16]\n  ef_search: [10, 40, 160]\n\nops:\n  enabled: true\n  workloads: [build, concurrency, filtered, churn]\n  m_values: [16]\n`;
        renderProfilesBody();
      },
    }, "New")));

  if (!C.profile) {
    panel.append(el("p", { class: "muted" }, "Pick a profile to edit it and plan a run."));
    return;
  }

  panel.append(el("h3", {}, "Profile YAML"),
    el("div", { class: "row" },
      el("label", { class: "muted" }, "name ",
        el("input", { id: "profile-name", value: C.profile, size: 26 }))),
    el("textarea", {
      class: "yaml", id: "profile-text", spellcheck: "false",
      oninput: () => { C.dirty = true; },
    }, C.profileText),
    el("div", { class: "row" },
      el("button", { class: "action secondary", onclick: validateProfile }, "Validate"),
      el("button", { class: "action", onclick: saveProfile }, "Save"),
      el("span", { id: "profile-status" })));

  panel.append(el("h3", {}, "Run plan"));
  const grid = el("div", { class: "facets" });

  grid.append(checkboxRow("engines",
    C.engines.map((e) => ({ value: e.name, label: `${e.name} (${e.tag || "?"})` })),
    C.plan.engines,
    (value, on) => {
      C.plan = { ...C.plan, engines: toggleIn(C.plan.engines, value, on) };
      renderProfilesBody(); refreshEstimate();
    }));

  const datasetItems = C.datasets.map((d) => ({
    value: d.name,
    label: `${d.name}${d.downloaded ? "" : " (not downloaded)"}`,
  }));
  grid.append(checkboxRow("datasets", datasetItems, C.plan.datasets,
    (value, on) => {
      C.plan = { ...C.plan, datasets: toggleIn(C.plan.datasets, value, on) };
      renderProfilesBody(); refreshEstimate();
    }));

  const options = el("div", { class: "facet" }, el("strong", {}, "options"));
  options.append(el("label", {}, "resource pass ",
    el("select", {
      onchange: (ev) => {
        C.plan = { ...C.plan, resource_pass: ev.target.value }; refreshEstimate();
      },
    }, ...["", "normalized", "tuned", "both"].map((v) =>
      el("option", { value: v, selected: v === C.plan.resource_pass }, v || "profile default")))));
  options.append(el("label", {}, "phases ",
    el("select", {
      onchange: (ev) => { C.plan = { ...C.plan, phases: ev.target.value }; refreshEstimate(); },
    }, ...["both", "ann", "ops"].map((v) =>
      el("option", { value: v, selected: v === C.plan.phases }, v)))));
  options.append(el("label", {}, "run id ",
    el("input", {
      value: C.plan.run_id, size: 22, placeholder: "auto",
      oninput: (ev) => { C.plan = { ...C.plan, run_id: ev.target.value }; },
    })));
  for (const flag of ["resume", "force", "fail_fast", "no_report"]) {
    options.append(el("label", {},
      el("input", {
        type: "checkbox", checked: C.plan[flag],
        onchange: (ev) => { C.plan = { ...C.plan, [flag]: ev.target.checked }; },
      }), " ", flag.replace("_", " ")));
  }
  grid.append(options);
  panel.append(grid);

  panel.append(el("div", { class: "facet", id: "estimate" }, "…"));
  panel.append(el("div", { class: "row" },
    el("button", { class: "action", onclick: launchRun }, "Launch run"),
    el("span", { id: "launch-status" })));
}

async function launchRun() {
  const status = document.getElementById("launch-status");
  clear(status);
  if (C.dirty && !confirm("The profile has unsaved edits. Launch with the saved version?")) {
    return;
  }
  try {
    const res = await api("/api/jobs", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(planFromForm()),
    });
    C.jobId = res.job.id;
    C.logOffset = 0;
    go("#/control/jobs");
  } catch (err) {
    status.append(el("span", { class: "err" }, String(err)));
  }
}

window.renderProfiles = async function renderProfiles() {
  await loadReference();
  if (!C.profile && C.profiles.length) {
    await openProfile(C.profiles[0].name);
    return;
  }
  renderProfilesBody();
  refreshEstimate();
};

// -- jobs --------------------------------------------------------------

function stopPolling() {
  if (C.poll) { clearInterval(C.poll); C.poll = null; }
}

async function pollLog() {
  if (!C.jobId) return;
  const chunk = await api(`/api/jobs/${encodeURIComponent(C.jobId)}/log?offset=${C.logOffset}`);
  const pre = document.getElementById("job-log");
  if (!pre) { stopPolling(); return; }
  if (chunk.data) {
    const atBottom = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 40;
    pre.append(document.createTextNode(chunk.data));
    C.logOffset = chunk.offset;
    if (atBottom) pre.scrollTop = pre.scrollHeight;
  }
  const badge = document.getElementById("job-status");
  if (badge) {
    badge.textContent = chunk.status + (chunk.exit_code === null || chunk.exit_code === undefined
      ? "" : ` (exit ${chunk.exit_code})`);
    badge.className = `pill ${chunk.status === "completed" ? "ok" : chunk.status === "running" ? "" : "bad"}`;
  }
  if (chunk.status !== "running" && chunk.status !== "stopping") {
    stopPolling();
    C.jobs = (await api("/api/jobs")).jobs;
  }
}

async function selectJob(jobId) {
  C.jobId = jobId;
  C.logOffset = 0;
  stopPolling();
  await renderJobsBody();
}

async function stopJob() {
  if (!C.jobId) return;
  if (!confirm("Terminate the running benchmark? Completed units stay checkpointed.")) return;
  await api(`/api/jobs/${encodeURIComponent(C.jobId)}/stop`, { method: "POST" });
}

async function renderJobsBody() {
  const panel = document.getElementById("panel-jobs");
  clear(panel);
  panel.append(el("h2", {}, "Runs"));

  if (!C.jobs.length) {
    panel.append(el("p", { class: "muted" }, "No runs launched from here yet."));
    return;
  }

  const rows = C.jobs.map((job) => el("tr", {
    onclick: () => selectJob(job.id),
    style: "cursor:pointer" + (job.id === C.jobId ? ";font-weight:600" : ""),
  },
    el("td", {}, job.id),
    el("td", {}, el("span", {
      class: `pill ${job.status === "completed" ? "ok" : job.status === "running" ? "" : "bad"}`,
    }, job.status)),
    el("td", {}, job.started_at || ""),
    el("td", {}, job.finished_at || ""),
    el("td", {}, job.command_display || "")));

  panel.append(el("table", {},
    el("thead", {}, el("tr", {}, ...["job", "status", "started", "finished", "command"]
      .map((h) => el("th", {}, h)))),
    el("tbody", {}, ...rows)));

  if (!C.jobId) return;
  const job = C.jobs.find((j) => j.id === C.jobId);
  if (!job) return;

  panel.append(el("h3", {}, "Output"),
    el("div", { class: "row" },
      el("span", { id: "job-status", class: "pill" }, job.status),
      el("code", { class: "muted" }, job.command_display),
      job.status === "running"
        ? el("button", { class: "action secondary", onclick: stopJob }, "Stop")
        : null),
    el("pre", { class: "log", id: "job-log" }));

  C.logOffset = 0;
  await pollLog();
  if (job.status === "running" || job.status === "stopping") {
    stopPolling();
    C.poll = setInterval(() => pollLog().catch(stopPolling), 1500);
  }
}

window.renderJobs = async function renderJobs() {
  const data = await api("/api/jobs");
  C.jobs = data.jobs;
  if (!C.jobId && data.active) C.jobId = data.active;
  if (!C.jobId && C.jobs.length) C.jobId = C.jobs[0].id;
  await renderJobsBody();
};

// Polling must not outlive the panel that shows the log.
window.addEventListener("hashchange", () => {
  if (S.route.kind !== "control" || S.route.section !== "jobs") stopPolling();
});

window.renderEngines = async function renderEngines() {
  const panel = document.getElementById("panel-engines");
  clear(panel);
  panel.append(el("h2", {}, "Engines"),
    el("p", { class: "muted" }, "Coming in the next change."));
};
