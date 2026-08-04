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
stores, and where the three engines' query planners diverge most (pre-filter vs
post-filter vs iterative scan).

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

A result without its manifest is not a result. The report generator refuses to
emit a report for a run directory with no manifest.

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
