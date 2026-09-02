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

// The five things this framework measures, plus an escape hatch. Each view
// fixes the phase and the axes, because reconstructing them by hand from raw
// facets is the work the report already does for you.
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
    id: "build",
    label: "Index build cost",
    phase: "index_build",
    x: "m",
    ys: ["build_wall_s", "index_bytes", "peak_rss_bytes", "build_cpu_s"],
    logY: false,
    filters: ["dataset", "engine", "resource_pass", "build_mode", "storage_engine"],
    hint: "What the index costs to make. pgvector bulk-builds; MHNSW and VIDX build on every INSERT.",
    empty: "No index-build records in this run.",
  },
  {
    id: "concurrency",
    label: "Concurrency scaling",
    phase: "concurrency",
    x: "clients",
    ys: ["qps", "latency_p99_ms", "latency_p95_ms", "latency_p50_ms"],
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
