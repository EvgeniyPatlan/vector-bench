# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

`vector-bench` benchmarks **built-in vector search across seven database engines**, all
running HNSW, so the comparison isolates implementation quality rather than algorithm
choice. It is a measurement instrument: the most damaging bugs here do not crash — they
quietly hand one engine a different budget, grid or dataset and produce a credible-looking
report that is wrong.

The repo root is `vector-bench/` (git repo, branch `master`). All commands below run from there.

| Engine key | What it is | Source pin |
| --- | --- | --- |
| `mariadb` | MariaDB MHNSW | tag `mariadb-11.8.8` |
| `mariadb123` | MariaDB MHNSW, second version | tag `mariadb-12.3.2` |
| `alisql` | AliSQL VIDX | tag `AliSQL-8.0.44-2` |
| `pgvector` | PostgreSQL + pgvector | tag `v0.8.6` |
| `mongodb` | Percona Search for MongoDB (mongot) | Percona package |
| `valkey` | Valkey + valkey-search | Percona repo `valkey-91` |
| `percona` | Percona Server, InnoDB HNSW | branch `vector-mvp`, commit `02a8b80c` |

`KNOWN_ENGINES` is whatever `config/engines/*.yml` holds — there is no list in Python to keep
in step with it. `orchestrator/engines.py` is the registry everything derives from.

## Commands

```bash
python3 -m pytest tests/ -q                 # ~1,700 tests, ~70 s. The fast inner loop.
./tests/make-tiny-dataset.sh                # synthetic corpus, no download
./run-benchmark.sh run --profile dev        # ~1 min/engine end-to-end cycle
./run-benchmark.sh run --profile smoke      # ~45 min/pass gate before any long run
./run-benchmark.sh report --run-dir results/<run-id> [--datasets glove-100-angular]
./run-benchmark.sh render --profile <p>     # regenerate ann-benchmarks configs only
./run-benchmark.sh build --engines alisql --march native
./run-benchmark.sh clean --run-id <run-id>
./run-benchmark.sh web [--allow-control]    # web UI; --no-container for host mode
./run-benchmark.sh generate <dataset>       # datasets fetch cannot download
./run-benchmark.sh lab --engine valkey probe-valkey-filter.py   # diagnose, don't measure
./run-benchmark.sh export --run-dir <id>    # bundle a run to send to someone
```

Never pipe pytest through `tail`/`head` in a command whose exit status matters — a broken
commit has already gone out that way, because the pipeline swallowed the non-zero exit.

Host requirements are deliberately only `docker`, `python3`, `pyyaml`, `git`, `curl`. Do not
add a host-side Python dependency: a benchmark framework must not modify the machine it
measures. Everything heavy (numpy, h5py, matplotlib, DB clients) lives inside the images.

**That constraint binds the test suite too.** `report/charts.py` guards its matplotlib import
and `_new_axes()` refuses loudly if it is missing, so `report.generate`'s pure-logic helpers
stay importable on a bare host; the dozen tests that actually render are marked
`@needs_matplotlib` and skip. A test that fails on a machine set up from the documented
prerequisites makes a correct install look broken — which is exactly what happened.

## Architecture

Two measurement paths converge on **one flat record schema** (`harness/metrics/records.py`,
`SCHEMA_VERSION`, `PHASE_*`). Every chart routes off `phase`; unused fields stay `None`.

- **Path A — recall/QPS.** ann-benchmarks, used *unmodified*, run as `run.py --local` inside
  each engine's `*-bench` image (`orchestrator/ann_pass.py`). The orchestrator launches the
  container so it owns cpuset/memory/env; ann-benchmarks keeps ownership of definitions,
  ground truth and recall computation. Our algorithm modules live in
  `overlay/ann-benchmarks/` and are copied over a disposable clone at `work/ann-benchmarks/`.
- **Path B — ops harness.** `harness/` runs in a *second* container talking to the server
  container over a private network (`orchestrator/ops_pass.py`). Two containers, not one, so
  the server's cgroup accounting measures the server and not the several hundred MB of NumPy
  holding the dataset. Produces build cost, concurrency, filtered search, churn.

The harness never starts or configures a server — the orchestrator does, so every engine gets
identical treatment and that treatment lands in the manifest.

**Web UI** (`webui/`, `docs/08-web-ui.md`). Covers every long command — `run`, `fetch`,
`build`, `generate`, `report`, `render` — through one job supervisor (`webui/jobs.py`), with
live log streaming. Nothing may run alongside a benchmark; setup jobs may overlap each other. Stdlib-only HTTP server + vanilla-JS front end,
served from `vector-bench/webui` (ubuntu + python3/pyyaml/numpy/git + the Docker CLI). Read-only
by default; `--allow-control` adds profile editing and run launching and mounts `docker.sock`.
It reads `report/records.jsonl` when a report exists and falls back to `load_ops_records()`.
The repo is bind-mounted **at its own absolute host path** — the orchestrator hands host paths
to the Docker daemon, so a container-only path would mount nothing on the host.

**Three config layers** resolve into one plan (`orchestrator/config.py`):
`config/profiles/<p>.yml` (what to measure) × `config/resources/{normalized,tuned}.yml`
(how much machine) × `config/engines/<e>.yml` (how to build, start and speak to it).
Resolution turns fractions into concrete byte counts and cpusets before they are recorded.

**`run-manifest.json` is mandatory.** The report generator refuses a run directory without
one — a result without its environment is not a result.

## Editing without rebuilding

`orchestrator/` and `config/` run on the host; `harness/` and `report/` are **bind-mounted
read-only into the containers at run time** (`/opt/harness`, `/vb/report`). Images carry the
third-party Python stack but none of this repo's own code, so edits take effect on the next
command with no rebuild:

| Changed | What is needed |
| --- | --- |
| `harness/`, `report/`, `orchestrator/`, `config/` | nothing — just re-run |
| `overlay/ann-benchmarks/` | nothing — `run` and `render` re-apply the overlay onto `work/` every time |
| `docker/*/Dockerfile`, entrypoints, `docker/_shared/requirements-*.txt` | `./run-benchmark.sh build --engines <e>` |
| engine `source.tag` | `./run-benchmark.sh build` (AliSQL is 1.5–3 h) |
| `webui/` (server or static) | nothing in `--no-container` mode; the repo is mounted, so container mode also just needs a restart |
| `docker/webui/Dockerfile` | `./scripts/build-images.sh --engine webui` (~1 min) |

`sources/`, `work/`, `datasets/`, `results/`, `state/` are generated and gitignored. The
vendor repositories (`$VB_REPO_MARIADB`, `$VB_REPO_ALISQL`, `$VB_REPO_ANNB`) are **read-only
inputs** — never check out, fetch into or otherwise modify them.

## Adding or changing an engine

An engine is not one file. Touch all of these, or a run gets as far as starting a server and
then dies on `argument --engine: invalid choice`:

**A variant of a family the harness already drives is config only** — a second MariaDB, a
Percona Server, another Postgres build. Write `config/engines/<e>.yml` (the `runtime:` block
carries driver, ann_constructor, port, mounts, credentials, probe, chart colour and label),
build the image, done. The Engines tab in the web UI does exactly this.

**A new architecture needs code:**

```
config/engines/<e>.yml                                  runtime: block + build + server flags
harness/drivers/<e>.py                                  + register in _driver_classes()
                                                        (harness/drivers/postgres.py)
overlay/ann-benchmarks/ann_benchmarks/algorithms/<e>/   module.py + config.yml
docker/<e>/Dockerfile + entrypoint-<e>.sh               runtime + bench targets
scripts/build-images.sh, scripts/prepare-sources.sh
tests/test_engine_registry.py                           add to GOLDEN
```

`orchestrator/engines.py` is the registry; `KNOWN_ENGINES`, chart styles and the driver
lookup all derive from it. `_driver_classes()` maps driver *class name* to class — one entry
per architecture, not per engine, which is what makes a variant configuration rather than code.

**Two collisions the validator refuses, both of which produce plausible-but-wrong configs:**
a duplicate `ann_constructor` (ann-benchmarks keys result files on it, so the second engine
reports the first's recall — this is why `mariadb123` exists) and a duplicate image tag.

## Invariants that fail silently

These are the reasons the code looks over-defensive. Do not relax them.

- **Never mix `-march` between engines.** The five built from source compile SIMD distance
  kernels — `mariadb`, `mariadb123`, `alisql`, `pgvector`, `percona`, all at `x86-64-v3`;
  `mongodb` and `valkey` are installed from packages and declare `march: none`. A different
  flag on one turns the benchmark into a comparison of compiler flags.
- **`harness/` and `report/` must not import `orchestrator`.** Each runs in a container that
  mounts only itself. Importing the orchestrator from `report/` once crashed generation at the
  end of a 20-hour run; the same mistake in `harness/` kills a run right after the server comes
  up. Anything they need is decided on the host and passed in — `--driver` for the harness, the
  manifest's `engines.<name>.presentation` for the report. Both are enforced by tests.
- **Every engine can fall back to a full table scan** — exact results, slowly, indistinguishable
  in output from "very accurate but slow". Every driver implements
  `explain_uses_vector_index()` and it is called per configuration; anything that missed the
  index goes into the report's Validity section. The marker lives in the driver
  (`Dialect.index_name` for the MySQL family) and nowhere else. `config/engines/*.yml` once
  carried an `sql:` block naming it too, alongside a `create_table` and a `query_filtered`,
  and this bullet used to describe that key as "documentation only and read by nothing".
  Read by nothing it was — and **also wrong**: `mariadb.yml` and `mariadb123.yml` declared
  `index_plan_marker: "v"` where the driver names the index `vi`, and the real check is a
  substring test against the joined EXPLAIN text, so `"v"` would have accepted almost any
  plan. The rest of the block had drifted the same way, for the same reason: a `create_table`
  with no `tag` column beside its own `tag` filter, and then no mention of `ctag` while every
  live driver created it. The blocks are gone and `tests/test_c1_engine_sql.py` fails if one
  returns.
- **The ann results tree at `results/annb/<pass>/<fingerprint>/` is shared across runs.**
  ann-benchmarks caches by algorithm and index parameters only — it has no idea how much
  memory or how many cores the engine had. `ann_fingerprint()` keys the tree by the resource
  pass's *declared* knobs so a 16 GB curve is never silently reused under a 64 GB manifest.
  The fingerprint is deliberately **engine-invariant**; making it engine-specific fragments the
  tree and produces a seven-engine recall chart containing one engine.
- **Filtered ground truth is recomputed** (`harness/datasets.py`) — a `WHERE` predicate changes
  the correct answer set, and reusing unfiltered neighbours scores every engine against the
  wrong targets.
- **Recall is ann-benchmarks' distance-threshold definition**, not an id intersection
  (`report/loaders.py:knn_recall`), so numbers stay comparable to published results.
- **MariaDB and AliSQL ship a 16 MiB graph cache by default** (`mhnsw_max_cache_size`,
  `vidx_hnsw_cache_size`). The framework sets both from the resource pass; leaving the default
  judges a vendor on a value it plainly intended you to change.
- **Percona's `innodb_hnsw_max_memory` is NOT a third member of that family** and must never be
  fed from `graph_cache_bytes`. Those two are caches — a graph larger than the cache still
  works and performance degrades. Percona's is a hard allocation ceiling with no per-node
  eviction: exceeding it fails the statement with `ER_CAPACITY_EXCEEDED`, and not only during
  ingest, because the limit is re-checked as a cold search faults the graph back in after a
  restart. Equal byte counts would therefore compare a cache budget against a wall, and one
  engine would be unable to finish the load where two merely slowed down. It is set to `0`
  (unlimited) so the container's memory limit bounds it like every other engine, and
  `Innodb_hnsw_memory_used` is recorded per phase — a figure that **omits the node map
  (~30–50 B/node) and so is a floor on graph memory, not a measurement of it**. See
  `docs/09-adding-percona-server.md` §3.
- **A profile's `resources:` block deep-merges dict-into-dict.** To *clear* an inherited key
  write `null`, not `{}` — `{}` recurses into the inherited map and changes nothing. This
  silently halved the `m-sweep` profile once.
- Structural asymmetries no config removes, and which the report always states:
  `ef_construction` is exposed by pgvector, Valkey and Percona Server, and by no other
  MySQL-family engine here — Percona is the only engine in the study with both a build-time
  and a query-time effort knob; pgvector bulk-builds while MHNSW and VIDX build incrementally
  on every INSERT, and Percona Server does either, so its `build_mode` is a comparison within
  one engine (ops path only — the ann overlay module has no `build_mode`, and
  `render_config` refuses the key rather than emitting two identical curves); AliSQL VIDX
  is InnoDB-only and requires READ COMMITTED.

## Cost awareness

Ingest dominates everything. MHNSW and VIDX build their graph on every INSERT (~150 and ~55
rows/s on 1M rows), and **every M value reloads the whole corpus**. `--profile full
--resource-pass both` is ~726 h of ingest before a single query runs, and ~1,042 h of run time
in the worst case once querying is counted — all seven engines, which is what that command
measures when `--engines` is omitted. **The ingest figure covers six of those seven**: nothing
has measured Percona's ingest rate, so the estimator names it in `engines_without_ingest_rate`
rather than modelling it, and 726 h is a floor by an amount nothing here knows. Before proposing a run,
check `docs/07-planning-a-run.md` and the estimate the run prints. Prefer `dev` → `smoke` when
validating framework changes; long profiles are the user's call, not a default.

Runs are resumable, checkpointed per `(pass, engine, dataset, phase)` and, inside an ops
phase, per `(M, build_mode)`:

```bash
./run-benchmark.sh run --profile main --resume --run-id main-<timestamp>
```

## Conventions

- Commit subjects are lowercase conventional-commit types (`fix:`, `feat:`, `docs:`, `test:`,
  `config:`, `perf:`, `chore:`) and name **the problem in plain language**, not the file
  touched — "fix: the tuned M pin silently halved the m-sweep profile". Bodies explain why,
  with the concrete evidence that produced the change (a run date, a measured number, the
  symptom it was mistaken for). Attribution is disabled globally.
- Comments in this codebase carry the *rationale and the incident* behind a decision, often at
  length. That density is intentional — match it when the code encodes a fairness or
  correctness decision, and keep it terse where the code is ordinary.
- Documentation that asserts engine behaviour should be executable: see
  `tests/verify-alisql-traps.sh`, which checks `docs/04-engine-notes.md` against a live server.
- Prefer adding a test in `tests/test_config.py` (fairness invariants), `tests/test_harness.py`
  (recall math, percentiles, Pareto, topology) or `tests/test_webui*.py` (API, guards, job
  validation) over a manual check — a crash gets noticed, a subtly wrong recall number gets
  published. Web UI tests build synthetic run dirs rather than reading `results/`, which is
  gitignored.

## Documentation map

`docs/01-architecture.md` (diagrams, container model) · `02-running-with-framework.md` (every
flag) · `03-running-manually.md` (standalone: each engine by hand with `docker run` and raw
SQL) · `04-engine-notes.md` (per-implementation traps) · `05-methodology.md` (what is measured,
fairness policy, known asymmetries) · `06-new-machine.md` · `07-planning-a-run.md` (read before
any long run) · `08-web-ui.md` (the browser front end, its access model and its limits) ·
`11-methodology-roadmap.md` (what the measurement methodology cannot yet observe, what it
already handles, and in what order to address the rest — a design document, nothing in it is
implemented; it cites code **by symbol and never by line**, and
`tests/test_e15_methodology_roadmap_doc.py` enforces that).
