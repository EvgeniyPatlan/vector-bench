"use strict";

const ENGINE_COLOR = {
  mariadb: "#1f77b4", mariadb123: "#5fa8d3", alisql: "#d62728",
  pgvector: "#2ca02c", mongodb: "#9467bd", valkey: "#e377c2",
};
const ENGINE_LABEL = {
  mariadb: "MariaDB 11.8 (MHNSW)", mariadb123: "MariaDB 12.3 (MHNSW)",
  alisql: "AliSQL (VIDX)", pgvector: "PostgreSQL (pgvector)",
  mongodb: "Percona Search (mongot)", valkey: "Valkey (valkey-search)",
};
const PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#e377c2",
                 "#ff7f0e", "#8c564b", "#17becf"];

// Raw record fields are what the schema calls them; these are what a reader
// calls them. Anything missing falls back to the field name.
const FIELD_LABEL = {
  recall_at_k: "Recall@k",
  qps: "Queries / sec",
  latency_p50_ms: "p50 latency (ms)",
  latency_p95_ms: "p95 latency (ms)",
  latency_p99_ms: "p99 latency (ms)",
  build_wall_s: "Build time (s)",
  build_cpu_s: "Build CPU time (s)",
  peak_rss_bytes: "Peak server memory",
  index_bytes: "Index size on disk",
  ingest_rows_per_s: "Ingest rate (rows/s)",
  ingest_wall_s: "Ingest time (s)",
  m: "M (graph degree)",
  ef_search: "ef_search (query effort)",
  ef_construction: "ef_construction (build effort)",
  clients: "Concurrent clients",
  selectivity: "Selectivity",
  churn_fraction: "Churn fraction",
  engine: "Engine",
  dataset: "Dataset",
  phase: "Measurement",
  resource_pass: "Resource pass",
  storage_engine: "Storage engine",
  build_mode: "Build mode",
  metric_space: "Metric space",
  k: "k",
};

function fieldLabel(name) {
  return FIELD_LABEL[name] || name;
}

// One view per phase `harness/metrics/records.py` can write, plus an escape
// hatch. Each view fixes the phase and the axes, because reconstructing them
// by hand from raw facets is the work the report already does for you.
//
// Covering every phase is the point and not tidiness: `explore.js` narrows a
// non-raw view's records to `view.phase`, so a phase with no entry here is a
// phase the Explore tab cannot reach at all. Two were in that position.
// `ef_calibration` because T2b added the phase, the records and the report
// section and stopped there; `ingest` since this tab was written. An operator
// wanting either had to fall back to the Raw view and rebuild the axes by
// hand, which is the work this list exists to have already done. A new
// `PHASE_*` needs an entry here in the same change.
//
// Each `ys` is a CLOSED allowlist -- `explore.js` builds the Y picker by
// filtering it, never by union with the data -- and each one below is derived
// from the keys records of that phase actually carry. That derivation has to
// be per phase, because `available_measures` (`webui/records.py`) is computed
// over the WHOLE run: a field populated only by some other phase passes that
// filter, reaches the dropdown, and draws nothing. An axis that cannot be
// honoured is worse than a missing one, because an empty chart reads as "this
// run measured nothing" rather than as "this is not measured here".
//
// ONE corpus backs every count quoted below, named here rather than per
// comment: the archive of completed runs kept beside this repo, which is 239
// `ops-*.jsonl` files plus 41 merged `report/records.jsonl` files and 7,284
// records in all -- 2,986 recall_qps, 1,862 concurrency, 818 churn, 650
// filtered, 484 ingest and 484 index_build. Three different corpora got
// mixed into one set of figures while this view list was being written, which
// moved denominators without moving any decision; if you add a number here,
// measure it on that corpus and say so.
const VIEWS = [
  {
    id: "recall",
    label: "Recall vs throughput",
    phase: "recall_qps",
    x: "recall_at_k",
    ys: ["qps", "latency_p50_ms", "latency_p95_ms", "latency_p99_ms"],
    logY: true,
    filters: ["dataset", "engine", "m", "ef_construction", "resource_pass",
              "build_mode", "storage_engine"],
    hint: "The quality/speed frontier. Up and to the right is better.",
    empty: "No recall records. They exist only after `run-benchmark.sh report` merges the ann tree.",
  },
  {
    id: "ingest",
    label: "Ingest cost",
    phase: "ingest",
    // `m` and not `rows`: every M value reloads the whole corpus, so this is
    // the axis the cost actually moves along, and it is the same x as the
    // build view below -- the two records are written side by side by
    // `harness/workloads/build.py` out of one load.
    x: "m",
    // Two entries because an ingest record carries two measurable numbers.
    // It also carries `rows` and `ingest_threads`, which are absent from
    // `MEASURE_FIELDS` in `webui/records.py`: `available_measures` never
    // reports them, so `explore.js` would strip either straight back out of
    // the picker. Build time, index size and peak RSS are not omissions
    // either -- they are on the `index_build` record, not this one.
    ys: ["ingest_rows_per_s", "ingest_wall_s"],
    // Log by default, which no other ops view needs. Ingest rate is the one
    // measure here that separates the engines by orders of magnitude rather
    // than by a factor: across the 484 archived ingest records it runs from
    // 38.9 rows/s (AliSQL VIDX, maintaining the graph per INSERT) to 71,372.8
    // (Percona Search), a factor of 1,837, and `ingest_wall_s` on the same
    // records spans 0.407 s to 25,479.3 s. On a linear axis the three slowest
    // engines lie on the x axis as a flat line and the view says only that
    // one engine is fast. The `recall` view sets this for the same reason.
    // Toggleable, as everywhere, so this is a default and not a decision.
    logY: true,
    filters: ["dataset", "engine", "resource_pass", "build_mode", "storage_engine"],
    hint: "What loading the corpus costs. MHNSW and VIDX maintain the graph on every INSERT, so for them this is nearly all of the build.",
    empty: "No ingest records in this run.",
  },
  {
    id: "build",
    label: "Index build cost",
    phase: "index_build",
    x: "m",
    // `build_cpu_s` used to sit at the end of this list and has been removed.
    // Nothing in `harness/`, `report/` or `orchestrator/` ever assigned it, so
    // it offered an axis no record could fill: 0 of the archive's 484
    // index_build records carry it, against 484 of 484 for both
    // `build_wall_s` and `index_bytes`. It is gone from `MEASURE_FIELDS` too,
    // which is where the reasoning is written down; the `Record` column
    // stays, because a schema field nothing writes is a separate question
    // from a dropdown offering it.
    //
    // `peak_rss_bytes` is the opposite case and is kept: the harness never
    // writes it either, but `report/generate.py` merges it in from the
    // `mem-*.jsonl` timeseries, which is why it is on 238 of the 245 merged
    // index_build records and on none of the 239 raw ones. Partial, not dead.
    //
    // Not offered, and deliberately: `ingest_wall_s`, which all 484 of those
    // index_build records carry. The ingest view above owns the load cost,
    // and showing the same number under both headings invites reading one as
    // the build cost on top of the load rather than as the same seconds
    // twice.
    ys: ["build_wall_s", "index_bytes", "peak_rss_bytes"],
    logY: false,
    filters: ["dataset", "engine", "resource_pass", "build_mode", "storage_engine"],
    hint: "What the index costs to make. pgvector bulk-builds; MHNSW and VIDX build on every INSERT.",
    empty: "No index-build records in this run.",
  },
  {
    id: "calibration",
    label: "ef_search calibration",
    phase: "ef_calibration",
    x: "ef_search",
    // Recall leads because `explore.js` opens a view on `ys[0]` and recall is
    // the question the sweep exists to answer: which ef_search puts this
    // engine at the target. Drawing the same points as a recall/QPS frontier
    // would be the mistake `records.py` gave them their own phase to prevent
    // -- they are single-client probes against the index the ops pass just
    // built, not Pareto points.
    //
    // QPS and the three percentiles are the same `LatencyStats` every other
    // query workload records, so they are offered on the same footing. The
    // record also carries `latency_mean_ms`, which `MEASURE_FIELDS` does not
    // list, so naming it here would add a dropdown entry `explore.js`
    // immediately discards.
    ys: ["recall_at_k", "qps", "latency_p95_ms", "latency_p50_ms",
         "latency_p99_ms"],
    logY: false,
    // No `ef_construction`, although the recall view above has it and the
    // calibration record carries the field. These are ops-pass records: the
    // whole sweep for one engine runs under one `IndexSpec`, so every ops
    // record an engine writes carries the same value. Measured on the
    // archive's 484 ingest records, written by this same path from the same
    // spec: the one engine that reports it at all (pgvector) reports the
    // single value 200 on all 114 of its own, and the other five engines
    // report nothing on any of the remaining 370.
    //
    // And a one-value facet is NOT quietly dropped, which is the trap. The
    // filter list is narrowed by `(S.facets[f] || []).length > 1`, and
    // `S.facets` is run-global by the same chain as `S.measures`: the run's
    // recall records sweep ef_construction, so the facet arrives holding 64,
    // 200 and 400, the checkbox renders, and a reader who picks 64 narrows
    // THIS view to zero records. Measured: that is the shape of 33 of the 39
    // archived merged runs that record the field at all. So the reason to
    // leave it out is the same one the ingest view gives for `selectivity` --
    // a facet this phase does not vary is a checkbox that narrows to nothing
    // the moment the run also holds a phase that does vary it.
    //
    // The recall view keeps the filter because the ann pass genuinely sweeps
    // it: those same records hold 64, 200 and 400 for both engines that
    // expose the knob. `build_mode` is listed here for that reason too --
    // the ops pass calibrates once per (M, build_mode), and pgvector's
    // archived ingest records split 84 `post` to 30 `incremental`.
    filters: ["dataset", "engine", "m", "resource_pass", "build_mode",
              "storage_engine"],
    hint: "The sweep the measured ef_search was chosen from, so the choice is auditable. Which point was chosen is in the report.",
    empty: "No calibration records — this run measured at a fixed ef_search.",
  },
  {
    id: "concurrency",
    label: "Concurrency scaling",
    phase: "concurrency",
    x: "clients",
    // recall_at_k belongs here as much as it does in the filtered view below:
    // every concurrency record carries it, and a throughput curve with no
    // quality axis available presents two engines at one ef_search as
    // comparable when they can sit 0.18 apart.
    ys: ["qps", "recall_at_k", "latency_p99_ms", "latency_p95_ms",
         "latency_p50_ms"],
    logY: false,
    filters: ["dataset", "engine", "m", "resource_pass", "storage_engine"],
    hint: "Does throughput grow with clients, or flatten?",
    empty: "No concurrency records — this run's profile did not include that workload.",
  },
  {
    id: "filtered",
    label: "Filtered search",
    phase: "filtered",
    x: "selectivity",
    ys: ["qps", "recall_at_k", "latency_p95_ms"],
    logY: false,
    filters: ["dataset", "engine", "m", "resource_pass", "storage_engine"],
    hint: "Vector search with a WHERE clause, scored against recomputed ground truth.",
    empty: "No filtered-search records — this run's profile did not include that workload.",
  },
  {
    id: "churn",
    label: "Churn",
    phase: "churn",
    x: "churn_fraction",
    ys: ["recall_at_k", "qps"],
    logY: false,
    filters: ["dataset", "engine", "m", "resource_pass", "storage_engine"],
    hint: "Recall drift after deleting and re-inserting part of the corpus.",
    empty: "No churn records — this run's profile did not include that workload.",
  },
  {
    id: "raw",
    label: "Raw — pick any two fields",
    phase: null,
    x: null,
    ys: null,
    logY: false,
    filters: null,
    hint: "Every record in the run. Filter on phase first, or the axes mix measurements.",
    empty: "This run has no records.",
  },
];

function viewById(id) {
  return VIEWS.find((v) => v.id === id) || VIEWS[0];
}
