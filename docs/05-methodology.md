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

Vector search with a scalar `WHERE` predicate, at three selectivities (~1%, ~10%,
~50% of rows passing) on **two different predicates**. Six points per engine,
dataset and index configuration.

#### The two filter modes

The two columns exist because one predicate measures half the problem. Both are
loaded in the same build, so both are measured against the same graph.

| Column | Definition | Relationship to the vector space |
| --- | --- | --- |
| `tag` | `row_id % 100` | **Independent.** The qualifying rows are a uniform sample, so a query's true neighbours pass the filter about as often as anything else does. |
| `ctag` | nearest-pivot cluster id (100 pivots, fixed seed) | **Correlated.** `ctag < t` selects `t` whole clusters, so for most queries the real nearest neighbours sit in a cluster that fails. |

`tag` is the control case and `ctag` is the treatment. An engine that
pre-filters sees little difference between them: it compares only the
qualifying vectors either way. An engine that post-filters — walk the graph,
then discard what fails — sees a large one, because under `ctag` the
candidates it walks are exactly the rows the predicate rejects. This is the
case a single uncorrelated column cannot distinguish, and it is the realistic
one: a filter in production usually correlates with the embedding (a category,
a tenant, a language), it rarely correlates with a row id.

**The selectivity recorded is the one that was delivered, not the one that was
asked for.** Cluster sizes are unequal, so a nominal 1% on `ctag` is never 1%:
measured, it delivers 1.90% on the 2,000-row test fixture and between 1.00%
and 4.96% on a 100,000-row corpus depending only on the clustering seed. The
`selectivity` field is a counted share of rows and `extra.nominal_selectivity`
is the request beside it. Comparing the two columns means comparing them at
their measured shares; comparing at the nominal share compares two different
predicates and calls the difference an engine property.

#### The near_pass / near_fail split

Within one column and one selectivity, queries divide into two groups by
whether **the query's true unfiltered nearest neighbour passes the filter**:

* `near_pass` — it passes. The best answer is inside the candidate list the
  engine was going to walk anyway, so the filter costs little.
* `near_fail` — it fails. The engine has to keep looking, and this is where
  post-filtering collapses: it exhausts `ef_search` candidates without finding
  `k` that qualify, and returns a short result set or a poor one.

Recall is reported for each group (`extra.recall_near_pass`,
`extra.recall_near_fail`) with the query counts beside them. For `tag` the two
groups track the selectivity and their recalls stay close. For `ctag` they
separate: on a clustered 2,000-row corpus at a 10% predicate the two groups
measured 0.983 and 0.241 recall for `ctag`, against 0.625 and 0.641 for `tag`
— the same split applied to both columns, separating them by a factor of four
on one and not at all on the other.

A dash in the report means the group was empty, which is a real state rather
than a missing measurement: `tag` at 1% puts zero of fifty queries in
`near_pass`, because a uniform 1% of rows contains a given query's nearest
neighbour about 1% of the time. The group's recall is recorded as absent, not
as zero.

#### Short and truncated results

The distribution of returned result counts is recorded in
`extra.returned_count_hist`, not only a count of the short ones: an engine
returning three rows for most queries and an engine returning nine are both
"short" and are not the same behaviour.

**Six buckets, and every query lands in exactly one at every `k`:**

| Bucket | Meaning |
| --- | --- |
| `0` | nothing qualified in the candidate list at all |
| `1-4` | a token result |
| `5-9` | most of the way |
| `short_other` | 10 to `k-1`. Only reachable for `k > 10` |
| `k` | the complete answer |
| `over_k` | more than `k` rows, which no correct driver returns |

The last three are why there are six rather than the four an earlier draft of
this section listed. `k` is tested **before** the ranges, so at `k = 5` a
complete answer of five rows is reported as complete rather than filed under
`5-9`. `short_other` exists because at `k = 20` a result of twelve rows
belongs to no other bucket — measured on the test fixture at `k = 20`,
`ef_search = 32` and a 50% predicate, 43 of 50 queries returned between ten
and nineteen rows, and the four-bucket reading would have dropped all 43 and
left a histogram summing to 7 of 50. `over_k` is counted precisely because it
should always be zero: every driver slices its result to `k`, so a non-empty
`over_k` means one stopped and the recall denominator is no longer what the
report says it is.

All six keys are always present, including the zeros. An absent key cannot be
told from a zero by a reader or a chart.

Each point is also **time-bounded**: `ops.filtered_budget_s` caps it (300 s in
`main`), and the query loop stops when the budget is spent. Recall and latency
are then computed over the queries that finished, and the record says so —
`extra.time_bounded`, `queries_executed` against `extra.queries_planned`, and
`extra.insufficient_sample` when fewer than 50 queries completed. The report
greys those points out. The budget exists because this dimension is the one
whose cost cannot be predicted: MariaDB measured 0.9 QPS at 1% selectivity
against 326 unfiltered, a 360x spread, which made 2,000 queries at one point
cost 37 minutes. A truncated point is a measurement of a slow configuration,
not a harness fault, so it is recorded and flagged rather than dropped.

#### Ground truth

Recall here is computed against ground truth **recomputed for the filtered
subset** — filtering changes the correct answer set, so reusing the unfiltered
ground truth would be wrong. The harness computes filtered ground truth by
brute force over the qualifying rows and caches it on nine components —
dataset, metric, `k`, column, threshold, row count, `tag_buckets`,
`ctag_clusters` and `ctag_seed`. Every one of them changes the correct answer
set: `tag` and `ctag` at the same threshold have entirely different qualifying
sets, so a key missing the column would score one filter against the other's
answers, and the last three are present for either column though each applies
to one, because a key that varies with something irrelevant only wastes work
while one that misses something relevant publishes a wrong number.

That brute force is **not** inside `ops.filtered_budget_s`, and it runs over
the whole test set rather than `ops.max_queries`: it is harness CPU spent on
the yardstick, and charging the engine's budget for it would make a cold cache
look like a slow engine.

Measured at **~150 s per million qualifying rows** over 10,000 test queries,
with a 5–8% spread within a run and ~10% drift between sessions on a shared
box. The figure is close to **dimension-independent** at these widths, which
is worth stating because it looks like it should not be: the dominant
`queries @ subset.T` term is linear in the vector width, so 128 dimensions
against 100 would be 1.28× and 960 would be 9.6× if the dot products ruled.
The measured medians are 158.7 s/M at 100 dimensions, 155.0 at 128 (0.98×,
marginally *faster*, inside the spread), 164.7 at 256 and 240.2 at 960 — so
what dominates is the dimension-independent work instead: materialising the
`n_test × n_qualifying` score matrix and partially sorting it. One rate
therefore covers both of `main`'s datasets.

For `main` that is ~3.6 minutes for glove's two-column grid and ~3.1 for
sift's — **~6.7 minutes for the profile**, against the 4.0 h filtered ceiling
it prints, so ~2.8%. Paid once per (dataset, metric, k, column, threshold,
rows) and then shared across every engine, ops unit and resource pass. The run
estimate names it under the costs neither of its figures includes.

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
and Valkey's planner chooses per query. The report does not take any of those
descriptions on trust — §5's strategy table derives what each engine did from
how its own median latency moves as the predicate narrows, per column, because
the report once asserted for weeks that Valkey chose its strategy per query on
the strength of its documentation while its own numbers said otherwise.

pgvector's `hnsw.iterative_scan` is set explicitly rather than left at its
default and is recorded on every filtered record. The workload can measure the
grid once per declared mode against one loaded index; `ops.pgvector_iterative_scan`
declares the modes and the orchestrator currently pins the first of them, so a
run measures one mode and the record names which.

### 2.5 Churn (ops harness)

Recall and QPS re-measured after deleting and re-inserting 10% and 25% of rows.
HNSW graphs degrade under deletion; how much, and whether the engine offers a
repair path, differs.

Three things about the post-churn state are decided here rather than left to
the engine's own timing, because each of them was previously deciding what got
measured.

**Autovacuum is off for the whole churn phase on pgvector.** `prepare_churn()`
runs `ALTER TABLE ... SET (autovacuum_enabled = false)` once, before the
baseline, and `maintain()` runs `VACUUM (ANALYZE)` and then `RESET`s the
option. Without this, whether a churned index still held its tombstones when
recall was re-measured depended on when the autovacuum daemon last looked at
the table, and nothing in the record said which it had been — two runs of the
same profile could differ by the whole size of the effect being measured, with
the difference read as a property of the engine.

**An asynchronous index is confirmed caught up before re-measuring.**
`wait_index_synced(rows)` runs after the re-insert, and its cost is recorded as
`extra.index_sync_s`. For Valkey it waits on FT.INFO's mutation queue; for
Percona Search it polls `$vectorSearch` for the newest written document until
the index returns it. The other four engines maintain their index on the write
path and record `0.0`. "The engine is slow and inaccurate" and "we asked before
the index was ready" are the same numbers in a record otherwise.

**A small `index_sync_s` for Valkey is weaker evidence than it looks**, and a
reader of the published number needs to know which way. The shared wait it
delegates to tests two conditions — the mutation queue, and the indexed
document count against the row count expected — but the count comparison was
written for the delete path, where the count falls *to* the target. A
re-insert raises it towards the target instead, so that half is already
satisfied when the wait begins and stops binding unless the index is
*over*counting. The queue is then the only signal in force, and the queue has
been observed drained while the index still disagreed with the keyspace by 481
documents. So a near-zero `index_sync_s` on Valkey means "the mutation queue
was empty", not "the index had absorbed the writes"; every record carries
`extra.index_sync_note` saying which of the two it rested on.

**The engines with a repair path are measured twice per churn step, and the
second fraction starts from the post-maintenance state.** Each step writes a
record stamped `extra.maintenance_state = "before"`, and then, if the engine
has a repair path, runs it, times it, and writes a second record stamped
`"after"` carrying `extra.maintenance` and `extra.maintenance_s`. Engines with
no repair path (MariaDB, AliSQL, Valkey, Percona Search) write one record per
step and carry `extra.repair_path = "none"`; pgvector carries
`"vacuum_analyze"`. `OPTIMIZE TABLE` is not used for MHNSW or VIDX: at 1M rows
it rebuilds the whole vector index, which is a separate experiment and not a
churn step.

That the 25% step therefore starts from a *repaired* index on pgvector and from
a degraded one everywhere else is a choice, and the alternative is defensible:
never maintaining between steps would measure cumulative degradation under a
realistic write pattern, which is arguably the more interesting question. It is
written down here because it is not derivable from the numbers. The reason for
this choice is comparability in the other direction — the sequence is "churn,
measure, repair, measure" per step, so every record states which of the two
states it describes, and a reader who wants the un-repaired cumulative curve
reads the `"before"` records across fractions and gets exactly it. Under the
alternative, the repaired state would exist only at the end and pgvector's
repair cost could not be attributed to a fraction at all.

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
directly comparable and nothing needs re-measuring to get one. That pair is what
`OPS_MEASUREMENT_EQUIVALENT_VERSIONS` (`harness/metrics/records.py`) holds — not
as a literal set but *derived* from the two constants that name the versions,
`IMPLICIT_OPS_MEASUREMENT_VERSION` and `FIRST_RECORDED_OPS_MEASUREMENT_VERSION`,
so a third member cannot be added without first naming a third concept and
saying why that version changed nothing measured. That set is what the refusal
below reads, so that it never tells an operator to re-measure numbers this page
calls comparable.

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
