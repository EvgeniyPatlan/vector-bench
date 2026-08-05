# Running the benchmark with the framework

Everything goes through one entrypoint: `./run-benchmark.sh`.

## 0. Requirements

On the host: **docker**, **python3 with PyYAML**, **git**. Nothing else. The
database clients and the scientific Python stack live inside the images, so
running a benchmark does not install anything onto the machine being measured.

```bash
# Debian / Ubuntu
sudo apt-get install -y docker.io python3-yaml git
sudo usermod -aG docker "$USER"   # then log out and back in
```

Disk: budget roughly **40 GB** for images and sources, plus 2–4× your dataset
size for index data. The `gist-960-euclidean` dataset alone is 3.6 GB.

## 1. First run, end to end

```bash
cd vector-bench

# 1. Export engine sources at their pinned tags and build all six images.
#    pgvector ~10 min, MariaDB ~40 min, AliSQL 1.5-3 h. See the note below.
./run-benchmark.sh build

# 2. Download the datasets.
./run-benchmark.sh fetch --datasets fashion-mnist-784-euclidean

# 3. Prove the whole pipeline works before spending real time on it.
./run-benchmark.sh run --profile smoke
```

> **On build times.** AliSQL dominates. Its CMake calls `MYSQL_CHECK_DUCKDB()`
> unconditionally on Linux, so the bundled DuckDB in `extra/duckdb` is compiled
> on every build — `WITH_DUCKDB_STORAGE_ENGINE=OFF` only makes the plugin a
> separate `.so` instead of linking it into `mysqld`; it does not skip the
> compile. DuckDB's ExternalProject also redirects its output to `/dev/null`, so
> the build looks stalled for a long stretch. It is not. Build engines
> separately (`--engines alisql`) if you want to start measuring the other two
> sooner.

The smoke profile takes about 15 minutes and exercises every stage for all
three engines. **Do not skip it.** It is far cheaper to discover a broken image
or a missing dataset here than eight hours into a full run.

When it finishes:

```
results/
├── annb/                                ann-benchmarks recall/QPS results
│   ├── normalized/<dataset>/<k>/<engine>/*.hdf5
│   └── tuned/<dataset>/<k>/<engine>/*.hdf5
└── smoke-20260803-141500/
    ├── run-manifest.json                provenance: CPU, versions, limits
    ├── checkpoints.json                 what completed, for --resume
    ├── ops-*.jsonl                      ops-harness records
    ├── mem-*.jsonl                      server memory timeseries
    └── report/
        ├── report.html                  self-contained, shareable
        ├── report.md
        ├── records.jsonl                every measurement, one flat schema
        └── charts/*.svg, *.png
```

The `annb/` tree is split by resource pass on purpose. ann-benchmarks names its
result files from the algorithm and its parameters only, with no notion of a
resource pass — a shared tree would make the tuned pass skip every point the
normalized pass had already computed, and the report would then present
normalized numbers as tuned ones. It also sits outside the per-run directory so
an interrupted run can resume without recomputing points it already has.

## 2. A real run

```bash
# Download the rest of the datasets first (~5 GB).
./run-benchmark.sh fetch

# The profile sized to actually be run: two datasets at 1M scale, ~2 days.
./run-benchmark.sh run --profile main

# One dataset, ~1 day — a complete three-way comparison at 1M scale.
./run-benchmark.sh run --profile main --datasets glove-100-angular
```

> `--profile full --resource-pass both` is **~272 h of ingest — about 11 days**
> — before a single query runs. It describes the complete measurement space; it
> is not a recommendation. Every run prints its own estimate before starting.
> [07-planning-a-run.md](07-planning-a-run.md) has the measured rates and what
> each way of narrowing the scope costs you.

## 3. Commands

| Command | Purpose |
| --- | --- |
| `build` | Export sources at pinned tags, build runtime + bench images |
| `fetch` | Download datasets into `datasets/` |
| `run` | Execute a benchmark run |
| `report` | Regenerate the report from an existing run directory |
| `render` | Regenerate the ann-benchmarks configs for a profile |
| `sources` | Export sources only, without building |
| `clean` | Remove containers, networks and volumes left by a run |

### `run` options

| Option | Default | Meaning |
| --- | --- | --- |
| `--profile` | `quick` | `smoke`, `quick`, `full`, or your own in `config/profiles/` |
| `--engines` | all three | e.g. `mariadb,alisql` |
| `--datasets` | from profile | Override the profile's dataset list |
| `--resource-pass` | `both` | `normalized`, `tuned`, or `both` |
| `--phases` | `both` | `ann` (recall/QPS), `ops` (build/concurrency/filtered/churn) |
| `--resume` | off | Skip units already recorded complete |
| `--force` | off | Re-run ann-benchmarks points that already have results |
| `--fail-fast` | off | Stop at the first failure instead of continuing |
| `--no-report` | off | Skip report generation |
| `--run-id` | timestamped | Reuse an id to add to an existing run directory |

### `build` options

| Option | Meaning |
| --- | --- |
| `--engines` | Build one engine only |
| `--target` | `runtime` (server only), `bench` (server + Python), or `all` |
| `--march` | **SIMD baseline for every engine.** See below. |
| `--jobs` | Compile parallelism (default: all cores) |
| `--no-cache` | Force a clean rebuild |

## 4. `-march` — the one build flag that changes results

All three engines compile SIMD distance kernels. `-march` decides which
instructions those kernels get, and it is applied identically to all three:

```bash
./run-benchmark.sh build --march x86-64-v3   # AVX2 + FMA. Portable. Default.
./run-benchmark.sh build --march native      # Whatever this CPU has, incl. AVX-512.
```

Use `native` when the build host and the benchmark host are the same machine and
you want each engine's best case. Use `x86-64-v3` when the images must move
between machines. **Never mix**: rebuilding one engine with a different `-march`
turns the benchmark into a comparison of compiler flags.

The value used is baked into each image and reported in the manifest.

## 5. Resumption and failures

Work is checkpointed on success at two levels: per
`(resource pass, engine, dataset, phase)`, and within an ops phase per
`(M, build_mode)`. The second matters — an ops phase can run a dozen hours
across several M values, and checkpointing only the whole phase would discard
every M that had already finished. After an interruption:

```bash
./run-benchmark.sh run --profile main --resume --run-id main-20260804-185624
```

Completed units are skipped, so an interruption costs one unit rather than the
run. Failed units are not checkpointed, so `--resume` retries exactly those.

If a run leaves containers behind (SIGKILL, host reboot):

```bash
./run-benchmark.sh clean --run-id full-20260803-120000
```

## 6. Choosing a resource pass

`normalized` gives every engine identical CPU, memory and cache budgets, so any
difference belongs to the implementations. `tuned` gives each engine its
vendor-recommended configuration on the full machine, so the numbers reflect
what each can actually do.

Run both. They answer different questions, and either one alone is easy to
argue with. See [05-methodology.md](05-methodology.md) §3.

## 7. Customising

Profiles and resource passes are plain YAML; copy and edit.

```bash
cp config/profiles/quick.yml config/profiles/mine.yml
$EDITOR config/profiles/mine.yml
./run-benchmark.sh run --profile mine
```

The knobs worth changing first:

- `ann.m_values` and `ann.ef_search` — sweep density. More points make a
  smoother Pareto frontier and cost linearly more time.
- `ops.client_counts` — how far the concurrency sweep goes. Push past your core
  count to find the saturation point.
- `ops.subset_rows` — load only part of the training set, for faster iteration.
- `memory.server_limit_gb` in `config/resources/normalized.yml` — the single
  most consequential value, because an engine whose graph does not fit in cache
  behaves completely differently from one whose graph does.

## 8. Reading the output

Start with `report/report.html`. Its order is deliberate:

1. **Environment** — the hardware and versions the numbers are scoped to.
2. **Validity** — anything that invalidates or narrows a result: measurements
   that ran without the vector index, filtered queries that returned short,
   environment warnings. Read this before the charts.
3. **Known asymmetries** — structural differences no configuration removes.
4. **Recall vs throughput**, then build cost, concurrency, filtered, churn.

`report/records.jsonl` holds every measurement in one flat schema, so you can
query it directly:

```bash
jq -r 'select(.phase=="recall_qps" and .recall_at_k>0.95)
       | [.engine, .m, .ef_search, .recall_at_k, .qps] | @tsv' \
  results/<run-id>/report/records.jsonl | sort -k5 -rn | head
```

## 9. Troubleshooting

**"image not found"** — build it: `./run-benchmark.sh build --engines <name>`.

**"dataset not found"** — fetch it: `./run-benchmark.sh fetch --datasets <name>`.

**A server container exits during startup.** The orchestrator prints the
container's log tail with the error. The usual causes are a memory limit too
small for the configured buffer pool, or a `shm_size` too small for PostgreSQL.

**"the vector index is NOT in the query plan".** Not a framework failure — the
engine's optimizer chose a full scan. The run continues and the affected
measurements are listed in the report's Validity section. For AliSQL this is
expected at large `LIMIT` values; see [04-engine-notes.md](04-engine-notes.md).

**Warnings about CPU topology.** On a machine with fewer homogeneous cores than
the profile requests, the server and client containers are forced to share
cores. That adds latency noise. Lower `cpu.server_cpus` in the resource config,
or accept the noise — it is recorded in the manifest either way.

**Results owned by root.** The engines run as root inside their containers. The
orchestrator chowns the results back at the end of a run; if it was killed
first, `sudo chown -R "$USER" results/` fixes it.
