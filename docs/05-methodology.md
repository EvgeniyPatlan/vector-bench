# Methodology

This document states exactly what is measured, how, and what the numbers do *not*
mean. Read it before quoting any result.

## 1. What is being compared

Three relational databases with **built-in** vector search, all using an
HNSW-family index:

| Engine | Version | Index implementation | Source |
| --- | --- | --- | --- |
| MariaDB | 11.8.8 (LTS) | MHNSW (`sql/vector_mhnsw.cc`) | `mariadb-11.8.8` tag |
| AliSQL | 8.0.44-2 | VIDX (`sql/vidx/vidx_hnsw.cc`) | `AliSQL-8.0.44-2` tag |
| PostgreSQL | 17 + pgvector 0.8.x | pgvector HNSW | `pgvector` release tag |

All three run HNSW. The comparison therefore isolates **implementation quality**,
not algorithm choice. That is deliberate: comparing HNSW against IVFFlat or
DiskANN would measure a different thing.

This is **not** a comparison against dedicated vector databases (Milvus, Qdrant,
Weaviate). Those live in the same ann-benchmarks harness and can be added later,
but the question here is "MariaDB vs AliSQL vs PostgreSQL".

## 2. Measurement dimensions

### 2.1 Recall vs QPS (ann-benchmarks)

The standard approximate-nearest-neighbour quality/speed tradeoff.

- **Recall@k** — for each of the dataset's held-out test vectors, the fraction of
  the true top-`k` neighbours that appear in the engine's returned top-`k`.
  Ground truth ships with the HDF5 dataset; it is exact, computed by brute force.
- **QPS** — queries per second, single client, best of `--runs` repetitions.
- Sweep: for each `M` (graph degree), query at several `ef_search` values. Each
  `(M, ef_search)` pair is one point. The upper-left envelope of those points is
  the **Pareto frontier** — that is what gets plotted and compared.

Higher and further right is better. An engine is faster than another *only* if
its frontier is above the other's at the same recall.

`k = 10` by default (`--count`). Recall at k=100 is also recorded.

### 2.2 Index build cost (ops harness)

Measured per `(engine, dataset, M)`:

- **Ingest throughput** — rows/sec loading the training vectors, excluding index
  build where the engine allows separating them.
- **Index build wall time** and **CPU time**.
- **Peak RSS during build** — read from the container's cgroup v2
  `memory.peak` (falls back to sampling `memory.current` on cgroup v1).
- **Index size on disk** — the on-disk size of index structures only, measured
  per engine:
  - MariaDB: the `#i#` auxiliary index files / tablespace for the vector index.
  - AliSQL: the InnoDB auxiliary table holding the HNSW graph.
  - pgvector: `pg_relation_size()` of the HNSW index relation.

These are not directly comparable to the byte, because each engine stores graph
nodes differently. They are comparable in *order of magnitude* and in how they
scale with `M`, which is the useful signal.

### 2.3 Concurrency scaling (ops harness)

QPS and latency percentiles (p50/p95/p99) at 1, 2, 4, 8, 16, 32 concurrent
clients, at a fixed `ef_search` chosen to put all engines at comparable recall.

This is the dimension that separates a database from an ANN library: it exercises
the engines' graph caches, buffer pools and locking. MariaDB caches the MHNSW
graph per `TABLE_SHARE`; AliSQL keeps a shared cache plus a per-transaction
cache; pgvector reads graph pages through `shared_buffers`. Those designs behave
differently under concurrency and that difference is the point.

Reported as: absolute QPS curve, and **scaling efficiency** = `QPS(n) / (n × QPS(1))`.

**Only the queries that finished inside the window are counted.** The window is
a closed interval: it opens at the instant the clients' barrier releases and
closes at the instant recorded immediately *before* the stop flag is set — not
when a client later notices the flag, which is the whole point. QPS is the
queries completing inside that interval divided by its length. Every client
finishes the query it is holding when the window closes, so counting those put
one query per client in the numerator that the denominator never covered.

The relative error is that one query over the ones a client did finish:
`1 ÷ queries-completed-per-client`, which in record fields is
`extra.inflight_discarded ÷ queries_executed` and is about **per-query latency ÷
window length**. The client count cancels — each client contributes one spurious
query and roughly `window ÷ latency` real ones — so it is the same at 1 client as
at 32, measured flat at 2.564% across 1, 2, 4, 8, 16 and 32 clients over a
two-second window of 50 ms queries. What it depends on is a **slow engine over a
short window**: 0.005% for a 1.4 ms query in a 30-second window, 2.5% for a
50 ms query in two seconds, whatever the concurrency. The uncorrected figure also
reported more queries in flight on average than there were clients to hold them,
which no client count can do. `extra.inflight_discarded` counts the queries
dropped at the close; `extra.prewindow_discarded` counts those dropped at the
open, and is zero by construction because the window opens at the barrier release
and no measured query can precede it, so **a non-zero value there means the
window's two ends disagree about when it began**. Recall is scored over the same
trimmed set, so `extra.recall_queries` and `queries_executed` always describe the
same queries. A window in which nothing finished at all — a configuration whose
queries outlast `ops.concurrency_duration_s` — records a QPS of 0, no latency
figures, and says so on stderr while it runs.

**Recall is recorded beside the throughput**, as `recall_at_k` on every
concurrency record. The sentence above — "a fixed `ef_search` chosen to put all
engines at comparable recall" — is an intention, and until the ids were kept
nothing in the run checked it. Two engines at `ef_search=100` can sit at recall
0.98 and 0.80, which is the ordinary spread of these implementations at a shared
ef rather than a pathology, and a throughput curve with no quality figure beside
it presents them as comparable while one is doing materially less work per
query. Each client records the ids from the first pass over its own shard — one
result list per query however long the window runs — and those are scored against
the same ground truth every other phase uses, recomputed over the loaded row
count when a profile loads a subset. `extra.recall_queries` says how many
distinct queries the figure covers; a window too short for one pass reports the
coverage it managed rather than padding the rest into a recall collapse. With
`concurrency_repeats > 1` the recall is the median window's, and
`extra.recall_spread` carries the difference between windows: recall is a
property of the index and the ef, so a spread above 0.005 means the mapping or
the index moved and the run says so on stderr.

**The query pool is its own profile key.** `ops.concurrency_queries` defaults to
`0`, meaning the whole test set — 10,000 queries for glove-100 and sift-128.
It used to be `ops.max_queries`, which exists to bound the *filtered* and *churn*
workloads: those answer each query once to completion, and `main.yml` holds the
key at 500 for exactly that reason. The concurrency workload instead cycles its
shard for a fixed duration, so at 32 clients each client was repeating about
fifteen queries for thirty seconds and every graph cache in the comparison was
measured against a hot set small enough to hold entirely — erasing the
difference the chart exists to show. `extra.query_pool` and
`extra.query_reuse` (executed ÷ distinct) record what the caches were actually
shown. Setting the key is the way back to a small pool for reproducing an
earlier run.

Each client's shard is `queries[c::clients]`, a strided slice rather than a
contiguous block, so no two clients hold adjacent queries and part of the window
is not one client warming a cache for another. The shards are disjoint and
together cover the pool exactly.

### 2.4 Filtered / hybrid search (ops harness)

Vector search with a scalar `WHERE` predicate, at three selectivities: ~1%, ~10%,
~50% of rows passing.

Recall here is computed against ground truth **recomputed for the filtered
subset** — filtering changes the correct answer set, so reusing the unfiltered
ground truth would be wrong. The harness computes filtered ground truth by brute
force over the qualifying rows.

Ground truth is also recomputed whenever a profile loads only **part** of the
training set (the smoke profile loads 20,000 of fashion-MNIST's 60,000 rows).
The neighbours shipped in the HDF5 file describe the full corpus, so scoring a
partially-loaded engine against them would point at rows the engine never
received — understating recall for every engine simultaneously, which looks like
a legitimate result rather than a measurement fault. The row count is part of
the ground-truth cache key so a subset run cannot poison a full run's cache.

This is where integrated vector search is supposed to beat standalone vector
stores, and where the engines' query planners diverge most: the SQL engines
post-filter, Percona Search pre-filters, pgvector can iterate its index scan,
and Valkey's planner chooses per query.

### 2.5 Churn (ops harness)

Recall and QPS re-measured after deleting and re-inserting 10% and 25% of rows.
HNSW graphs degrade under deletion; how much, and whether the engine offers a
repair path, differs.

## 3. Fairness

Two passes are run. Both are reported; neither alone is the answer.

### Normalized pass

Every engine gets identical constraints:

- Same `--cpuset-cpus`, restricted to a homogeneous core set. On hybrid CPUs
  (Intel P-core/E-core) the harness detects core types and uses performance cores
  only; hyperthread siblings are excluded unless explicitly enabled. Mixing core
  types produces run-to-run variance that swamps the effect being measured.
- Same `--memory` container limit.
- Equalized graph-cache budget: `mhnsw_max_cache_size` ≈ `vidx_hnsw_cache_size` ≈
  the pgvector `shared_buffers` allocation.
- Identical `M` grid and identical `ef_search` grid.
- Single client (for the recall/QPS pass).

### Tuned pass

Each engine configured per its own vendor documentation, full machine available.
This answers "what can this engine actually do", at the cost of confounding
implementation quality with documentation quality.

## 4. Known asymmetries

These are real differences between the engines that no amount of configuration
removes. They must accompany any published number.

| Asymmetry | Consequence |
| --- | --- |
| **`ef_construction` is not exposed** by MariaDB MHNSW or AliSQL VIDX; pgvector exposes it. | The build-quality sweep can only vary `M` for the MySQL-family engines. pgvector is benchmarked at its default `ef_construction=200` in the normalized pass so it is not given a tuning axis the others lack. |
| **AliSQL requires `READ-COMMITTED`** and `vidx_disabled=OFF`; vector indexes are InnoDB-only. | AliSQL runs at RC. MariaDB and PostgreSQL are run at their defaults in the tuned pass and at RC in the normalized pass where the engine allows it. |
| **MariaDB supports MyISAM** for vector tables (and its own published benchmark uses MyISAM); AliSQL cannot. | Headline comparison is **InnoDB for both**. MariaDB MyISAM is reported as an additional MariaDB-only curve in the tuned pass, clearly labelled. |
| **`M` is set differently** — MariaDB via the `mhnsw_default_m` server variable at startup, AliSQL via a per-index DDL option, pgvector via `WITH (m=…)`. | Handled per-driver; the effective `M` is recorded in the manifest, not assumed. |
| **SIMD paths differ.** Both AliSQL and MariaDB document AVX-512 distance kernels. | The harness records CPU flags. On a machine without AVX-512 both fall back to narrower paths, and the result is only valid for that class of hardware. Any report states the CPU and its flags. |
| **Default distance metric differs** — AliSQL defaults to `EUCLIDEAN`, and metric is fixed at index creation for all three. | Metric is always set explicitly per dataset; never left to a default. |

## 5. Reproducibility

Every run writes `results/<run-id>/run-manifest.json` containing:

- Docker image digests for each engine image actually used.
- Engine source commit SHAs and tags.
- Full resolved configuration (profile + engine + resource pass), plus its hash.
- CPU model, flags (`avx512f`, `avx2`, `avx_vnni`, …), core topology, hybrid-core
  detection result, total RAM, kernel version, Docker version.
- Start/end timestamps per phase.
- The ops measurement version (`config.ops_measurement_version`), which is also
  written into every ops record as `extra.ops_measurement_version`.

A result without its manifest is not a result. The report generator refuses to
emit a report for a run directory with no manifest.

The ops measurement version is bumped whenever a change to `harness/` alters what
an ops number means, as distinct from `SCHEMA_VERSION`, which describes only the
record's shape; records carrying no version at all were measured under version 1.
**Version 2 measures exactly what version 1 measured** — it is simply the first
value the scheme writes down, so a version 1 number and a version 2 number are
directly comparable and nothing needs re-measuring to get one. That pair is
listed in `OPS_MEASUREMENT_EQUIVALENT_VERSIONS` (`harness/metrics/records.py`),
which is what the refusal below reads so that it never tells an operator to
re-measure numbers this page calls comparable.

**Version 3 is not comparable with either**, and is deliberately absent from that
set. It separated the concurrency query pool from `ops.max_queries` (§2.3), so a
version 2 concurrency figure is throughput over a 500-query working set and a
version 3 figure is throughput over the whole 10,000, measured against caches
that behave differently at the two sizes. That is the same workload over two
different amounts of the corpus, not two measurements of one quantity, and the
engines whose caches differ most are exactly the ones the difference moves.
Concurrency recall also arrives in version 3, but a field that was absent
becoming populated is new information rather than a changed number and would not
on its own have earned a bump. Version 3 also counts only the queries that
finish inside each concurrency window (§2.3), and that correction is part of what
version 3 means rather than a fourth version — no run has ever been measured at
version 3, because the value was introduced and then extended inside the same
change set, so there is no version 3 figure anywhere for a trimmed one to be
incomparable with. The magnitude says the same thing from the other side: the
correction moves a figure by roughly `per-query latency ÷ window length`, which
at `main`'s 30-second windows and the p50s these engines actually record is
about 0.005% — well inside the window-to-window spread
`ops.concurrency_repeats` exists to expose.

The bump invalidates comparison across the other
ops phases too — build cost, filtered, churn — none of which version 3 changes;
the version names the harness that wrote the record rather than the phase, which
is the cost of having one constant instead of four that can each be forgotten.

The report refuses
to draw records from two versions on one chart, naming both versions and the
records involved, exactly as `ann_fingerprint` keeps a 16 GB recall curve out of a
report whose manifest says 64 GB. A version already recorded in a manifest is
carried forward untouched rather than re-stamped, so resuming an old run does not
relabel measurements it did not take.

**A resume across a bump says so before it measures anything.** The refusal
above is correct but it arrives at report time, which is after the expensive
half: resuming a `main` run under newer code meant up to days of ingest before
anything mentioned that the directory could not be reported as one. `set_config`
now prints a warning naming both versions and the remedy — one report per
version — at the start of the resume, once per run. It is not fatal: finishing
the run and splitting the `ops-*.jsonl` files by
`extra.ops_measurement_version` afterwards is legitimate, and the older records
are not wrong, only incomparable.

## 6. What would change the conclusions

Stated up front so readers can judge relevance:

- **Different CPU.** AVX-512 availability plausibly reorders the engines.
- **Dataset scale.** These datasets are 60k–1M vectors. Behaviour at 100M is not
  extrapolable from them — cache-resident graphs become disk-resident.
- **Version drift.** All three projects are actively developing vector search.
  Results are pinned to the tags in the manifest and go stale quickly.
- **Client-side overhead.** ann-benchmarks drives queries from Python. At high
  QPS the client can become the bottleneck; the harness records client CPU
  utilisation so this is detectable rather than silent.
