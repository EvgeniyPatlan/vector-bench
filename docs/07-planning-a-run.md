# Planning a run

Benchmarking these engines is expensive in a way that is not obvious until you
are eight hours into it. This document exists so you can choose the size of a
run before starting it rather than discovering the size afterwards.

Every number here was measured, not estimated. Where a figure is extrapolated
it says so.

---

## 1. Why runs are expensive

Three multiplying factors, none of them avoidable:

### Incremental graph building

MariaDB MHNSW and AliSQL VIDX maintain the HNSW graph on **every INSERT**.
There is no bulk build. Measured on a Xeon Gold 6230, glove-100-angular
(1,183,514 × 100), single-threaded, batched:

| Engine | Ingest rate | Time to load glove-100 once |
| --- | ---: | ---: |
| pgvector (bulk build after load) | ~3,000 rows/s | ~7 min |
| MariaDB (incremental) | ~150 rows/s | ~2.2 h |
| AliSQL (incremental) | ~55 rows/s | ~6 h |

That gap is itself one of the benchmark's findings. It is also why a run that
is trivial for pgvector takes days for the other two.

### The cost rises steeply with M

More neighbours per node means more distance evaluations and more link updates
per insert. Measured on glove-100 with MariaDB:

| M | Ingest rate | Load time |
| ---: | ---: | ---: |
| 8 | ~320 rows/s | ~1.0 h |
| 16 | ~200 rows/s | ~1.6 h |
| 32 | ~70 rows/s | ~4.7 h |

**M=32 alone is about 65% of the cost of the `[8, 16, 32]` grid.** Sweeping M
is not a linear expense.

### Every M value reloads the entire dataset

ann-benchmarks constructs a fresh index per configuration, and the ops harness
does the same. Three M values means three full loads — per engine, per dataset,
per phase, per resource pass. The multiplication is quiet and it dominates
everything else.

---

## 2. What each profile actually costs

Ingest time only, before any query runs. Three engines, from the measured rates
above.

| Profile | Datasets | Passes | Ingest |
| --- | --- | --- | ---: |
| `dev` | tiny synthetic | 1 | ~1 min |
| `smoke` | fashion-mnist | 1 | ~30 min |
| `quick` | fashion-mnist, glove-100 | 1 | ~14 h |
| `main` | glove-100, sift-128 | 1 | **~46 h (~2 days)** |
| `full` | all four | 2 | **~272 h (~11 days)** |
| `mariadb-blog` | dbpedia-openai-1000k | 1 | **~72 h (~3 days)** |

`full` describes the complete measurement space. It is not a recommendation.
Running it end to end is a deliberate decision to spend a fortnight, and
`gist-960-euclidean` accounts for a large share of that on its own — at 960
dimensions it is ~19 h of loading per phase.

**`main` is the profile to run.** It was sized from these measurements to
answer the questions the smoke run raised without spending a fortnight.

A run prints its own estimate before starting:

```
estimated ingest time (loading only, before any queries):
  mariadb ~12.1 h  |  alisql ~33.1 h  |  pgvector ~0.6 h
  total ~45.8 h across 1 pass(es), 2 M value(s)
  ! This is a long run...
```

Read it. It is the cheapest part of the whole exercise.

---

## 3. Choosing a scope

Measured options, all three engines:

| Scope | Ingest |
| --- | ---: |
| `--profile full --resource-pass both` | 272 h |
| `--profile main` | 46 h |
| `--profile main --datasets glove-100-angular` | **25 h** |
| `--profile main --datasets glove-100-angular --phases ops` | 8 h |

If you run one thing, run this:

```bash
./run-benchmark.sh run --profile main \
  --datasets glove-100-angular --resource-pass normalized
```

Roughly a day, and it produces a complete three-way comparison at 1M scale
across every measured dimension.

### What each cut costs you

Cuts are not equivalent. What you lose matters as much as what you save.

| Cut | Saves | Costs you |
| --- | --- | --- |
| Drop the `tuned` pass | ~50% | Whether tuning changes the ranking. On the smoke run it changed little except AliSQL's concurrency (0.27 → 0.65 efficiency) — measurable separately at one M in about an hour. **Usually the best first cut.** |
| Drop `gist-960` | ~40% of `full` | The high-dimensionality data point. 960 dims is where SIMD and cache behaviour diverge most, so this is a real loss if that is your workload. |
| Drop M=32 | ~65% of the M grid | The high-recall end of the Pareto frontier — the part that matters for production accuracy targets. **Cut the middle value instead.** |
| Drop 1% selectivity | ~37 min per configuration | The hardest filtered case. MariaDB measured 0.9 QPS there against 326 unfiltered — a 360× penalty that is itself a finding. Keep it for a targeted run, drop it from broad sweeps. |
| Fewer `ops.m_values` | proportional | Little. Build cost, concurrency and churn describe engine behaviour, not index-parameter sensitivity. One M value is usually enough. |
| Fewer `ann.ef_search` points | small | Little — ef_search is swept without reloading, so extra points are cheap. This is the wrong place to economise. |

The last two lines are the important ones: **sweeping `ef_search` is nearly
free, sweeping `M` is not.** Extra ef_search points reuse the built index;
every extra M rebuilds it.

---

## 4. Watching a run

Every long phase reports rate and ETA every 20 seconds
(`VB_PROGRESS_INTERVAL` overrides it):

```
[mariadb]   412,500/1,183,514 rows, 148 rows/s, ETA 86.8 min
[mariadb]   filtered 10% queries 340/500, 34.1/s, ETA 0.1 min
[pgvector]  bulk index build — still running, 120s elapsed
```

Operations that cannot report a fraction — a bulk `CREATE INDEX`, a NumPy
brute-force ground-truth pass — emit a heartbeat instead, so silence always
means something is wrong.

If you are running an older checkout without this, poll the server directly:

```bash
C=$(docker ps --format '{{.Names}}' | grep srv)
M="/opt/mariadb/bin/mariadb -ubench -pbench --socket=/var/run/vbench/mariadb.sock -N -e"
prev=$(docker exec $C $M "SELECT COUNT(*) FROM ann.t1")
while sleep 60; do
  n=$(docker exec $C $M "SELECT COUNT(*) FROM ann.t1") || break
  r=$(( (n - prev) / 60 )); prev=$n
  echo "$(date +%T)  $n rows  ${r} rows/s"
done
```

A count that stops advancing for several minutes is stuck. A count that keeps
moving is not, however slow it looks.

---

## 5. Interruption and resumption

Work is checkpointed at two levels:

- per `(resource pass, engine, dataset, phase)`
- and within an ops phase, per `(M, build_mode)`

The second matters: an ops phase can run a dozen hours across several M values,
and checkpointing only the whole phase meant an interruption discarded every M
that had already finished.

```bash
./run-benchmark.sh run --profile main --resume --run-id main-<timestamp>
```

Completed units are skipped. Failed units are not checkpointed, so `--resume`
retries exactly those.

ann-benchmarks additionally skips result files it has already produced, so an
interrupted recall sweep resumes at configuration granularity without any help
from the checkpoint file.

---

## 6. Reproducing MariaDB's published benchmark

MariaDB's [big vector search benchmark](https://mariadb.org/big-vector-search-benchmark-10-databases-comparison/)
used **dbpedia-openai-1000k** — one million DBpedia texts as 1536-dimensional
OpenAI embeddings, angular distance — run through their fork of ann-benchmarks,
which is what this framework is built on.

That dataset is **not** published as a prebuilt HDF5, so it has to be built
locally from HuggingFace:

```bash
./scripts/generate-dataset.sh dbpedia-openai-1000k-angular   # multi-GB download,
                                                             # hours, ~20 GB working space
./run-benchmark.sh run --profile mariadb-blog
```

Ground truth is computed by brute force inside the generator, which is the slow
part.

### What will not match, and why

| Difference | Effect |
| --- | --- |
| Their CPU (Xeon E5-2660 v4) has **no AVX-512** | If yours does, MHNSW and VIDX take wider SIMD paths and will look faster than published. The largest single source of divergence, and not configurable away — building `--march x86-64-v3` gets closer at the cost of not measuring your own hardware. |
| They compared ten systems | This framework covers MariaDB, AliSQL and pgvector. The other seven are out of scope. |
| Versions | MariaDB 11.8.8 here vs their 11.8 **and** 12.3; pgvector 0.8.6 vs their master of 2026-02-08. |
| Index parameters were not published | The M and ef_search grids in the profile are ann-benchmarks' conventional ranges, not a claim to match theirs. |

At 1536 dimensions this is the heaviest run the framework supports — roughly 5×
the per-row cost of glove-100. Estimated ingest:

| Engine | Ingest |
| --- | ---: |
| pgvector | ~1 h |
| MariaDB | ~19 h |
| AliSQL | ~52 h |
| **all three** | **~72 h** |

AliSQL is 72% of that, which makes it the obvious thing to drop. Resist it:
MariaDB versus AliSQL is the comparison this framework was built for, and the
article's framing — MariaDB against everything else — is a different question.
If three days is too long, **cut the corpus rather than the engine list**:
`dbpedia-openai-500k-angular` and `-100k-angular` are the same embeddings at
smaller scale and keep all three engines in the picture.

---

## 7. Sizing a new dataset

Before adding a dataset, estimate its load:

```
hours ≈ rows / rate / 3600 × (M values) × (phases) × (passes) × (engines)
```

with `rate` from §1 — and remember it falls as the graph grows and falls again
with M. For anything near 10M vectors (`deep-image-96`), assume days per engine
for the incremental builders and plan a dedicated run rather than folding it
into a sweep.

---

## 8. A note on the numbers in this document

The ingest rates here come from real runs on a dual-socket Xeon Gold 6230 with
AVX-512, engines built `--march=native`. They will differ on your hardware,
possibly a lot: a machine without AVX-512 runs the distance kernels on narrower
paths, and MHNSW and VIDX both document AVX-512 support.

An earlier version of the estimator used rates guessed from a 60,000-row smoke
run and was optimistic by 2.7× against the first real 1.18M-row load. That is
worse than having no estimate at all, because it invites you to start something
you would not have started. If your measured rates diverge from these, update
`_INGEST_ROWS_PER_S` in `orchestrator/cli.py` — the estimate is only as honest
as the numbers behind it.
