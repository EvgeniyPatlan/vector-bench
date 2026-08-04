# vector-bench

Benchmark framework comparing **built-in vector search** across three
relational databases, all using an HNSW index:

| Engine | Version | Implementation |
| --- | --- | --- |
| MariaDB | 11.8.8 LTS | MHNSW (`sql/vector_mhnsw.cc`) |
| AliSQL | 8.0.44-2 | VIDX (`sql/vidx/vidx_hnsw.cc`) |
| PostgreSQL | 17 + pgvector 0.8.x | pgvector HNSW |

All three run HNSW, so the comparison isolates **implementation quality**
rather than algorithm choice.

```bash
./run-benchmark.sh build                 # build all six images from pinned tags
./run-benchmark.sh fetch                 # download datasets
./run-benchmark.sh run --profile smoke   # validate the pipeline (~15 min)
./run-benchmark.sh run --profile full --resource-pass both
```

Requirements on the host: **docker**, **python3 + PyYAML**, **git**. Nothing
else — the database clients and the scientific Python stack live inside the
images, so running a benchmark does not modify the machine being benchmarked.

## What it measures

| Dimension | Tool | Why it is here |
| --- | --- | --- |
| Recall vs QPS | ann-benchmarks | The standard ANN quality/speed frontier |
| Index build cost | ops harness | Build time, peak server RSS, index size on disk |
| Concurrency scaling | ops harness | QPS and p50/p95/p99 at 1→32 clients |
| Filtered (hybrid) search | ops harness | Vector search + `WHERE`, at 1% / 10% / 50% selectivity |
| Churn | ops harness | Recall drift after 10% / 25% delete+reinsert |

Only the first is what ann-benchmarks provides. The rest are what separate a
database from an ANN library, and are where these three diverge most.

## Documentation

| Document | Contents |
| --- | --- |
| [docs/01-architecture.md](docs/01-architecture.md) | System architecture, container model, per-engine internals, data flow — with diagrams |
| [docs/02-running-with-framework.md](docs/02-running-with-framework.md) | Every command and flag, resumption, customising profiles, troubleshooting |
| [docs/03-running-manually.md](docs/03-running-manually.md) | **Standalone.** Run each engine by hand with `docker run` and raw SQL — no framework involved |
| [docs/04-engine-notes.md](docs/04-engine-notes.md) | What each implementation does, and the traps in each |
| [docs/05-methodology.md](docs/05-methodology.md) | What is measured, how, fairness policy, and known asymmetries |

If you only want to try the three engines yourself, read
[03-running-manually.md](docs/03-running-manually.md) — it needs nothing from
the rest of this project.

## Design

Two measurement paths write into one flat record schema:

- **ann-benchmarks**, run unmodified inside each engine image (`run.py --local`),
  produces recall/QPS. The orchestrator launches the container so it controls
  cpuset, memory and environment; ann-benchmarks keeps ownership of definitions,
  ground truth and recall computation. No patching of the upstream project.
- **The ops harness**, a separate client container talking to a server container
  over a private network, produces everything else. Two containers rather than
  one so the server's cgroup accounting measures the server and not the several
  hundred megabytes of NumPy holding the dataset.

Each engine has two images: `<engine>-runtime` (server only, what you use by
hand) and `<engine>-bench` (runtime + the Python stack). Sources are exported
from pinned tags; the vendor repositories are **read-only inputs and are never
modified**.

## Fairness

Two passes, both reported:

- **normalized** — identical CPU set (pinned to one core class on hybrid CPUs),
  identical memory limit, equalized graph-cache budget, identical parameter
  grid, identical `-march`. Differences belong to the implementations.
- **tuned** — each engine configured per its own documentation on the full
  machine. Reflects deployment, but confounds implementation quality with
  documentation quality.

Neither alone is the answer, which is why both are run.

Three asymmetries no configuration removes, all recorded in every report:
`ef_construction` is exposed only by pgvector; pgvector bulk-builds while MHNSW
and VIDX build incrementally; AliSQL is InnoDB-only and requires READ COMMITTED.
See [05-methodology.md](docs/05-methodology.md) §4.

## Validity

Every one of these engines can silently fall back to a full table scan, which
returns exact results slowly — indistinguishable in the output from "very
accurate but slow". Every driver therefore runs `EXPLAIN` for each configuration
and records whether the vector index was used. Any measurement where it was not
appears in the report's **Validity** section, above the charts.

Every run writes `run-manifest.json`: CPU model and SIMD flags, core topology,
RAM, kernel, Docker version, engine source commits and tags, image ids, and the
fully resolved resource limits. The report generator refuses to run without it.

## Layout

```
vector-bench/
├── run-benchmark.sh          single entrypoint
├── config/
│   ├── profiles/             smoke · quick · full — what to measure
│   ├── resources/            normalized · tuned — how much machine
│   └── engines/              per-engine build, server flags, SQL dialect
├── docker/                   multi-stage Dockerfiles + entrypoints
├── overlay/ann-benchmarks/   our algorithm modules (alisql is new)
├── harness/                  ops harness: drivers, workloads, metrics
├── orchestrator/             host-side: containers, limits, manifest
├── report/                   charts, Markdown and self-contained HTML
├── docs/                     the five documents above
├── tests/                    unit tests + synthetic-data report check
└── sources/ work/ datasets/ results/     generated, gitignored
```

## Relationship to the vendor repositories

`AliSQL/`, `server/` (MariaDB) and `ann-benchmarks/` in the parent directory are
**read-only inputs**. This project clones or exports from them and writes
everything under `vector-bench/`. Point it elsewhere with `VB_REPO_ALISQL`,
`VB_REPO_MARIADB`, `VB_REPO_ANNB`.
