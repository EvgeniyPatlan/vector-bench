# Moving vector-bench to another machine

The repository is ~450 KB. Everything large is regenerated on the target:
engine sources, Docker images, datasets and results are all gitignored.

**The framework is standalone.** It does not need the `AliSQL/`, `server/` or
`ann-benchmarks/` checkouts that sit beside it on the original machine. When
they are absent, `prepare-sources.sh` and `prepare-harness.sh` clone from
upstream instead. Copying them across is an optimisation, not a requirement.

---

## 1. Prerequisites on the target

```bash
# Debian / Ubuntu
sudo apt-get update
sudo apt-get install -y docker.io python3-yaml git curl
sudo usermod -aG docker "$USER"     # then log out and back in
```

Check before starting:

| Resource | Needed | Why |
| --- | ---: | --- |
| Disk | **~80 GB** | ~10 GB images, ~25 GB engine sources, ~5 GB datasets, the rest index data |
| RAM | 32 GB+ | the normalized pass defaults to a 16 GB server limit |
| Cores | 8+ physical | server and client containers are pinned to disjoint CPU sets |
| Network | first run only | cloning sources and downloading datasets |

```bash
df -h /          # want 80 GB+ free
nproc; free -g
docker info >/dev/null && echo "docker ok"
python3 -c 'import yaml' && echo "pyyaml ok"
```

---

## 2. Get the code across

> **Check what the remote actually has first.** Work done on a branch, or on a
> machine that has not pushed, is not on the remote — and a clone will silently
> give you an older tree that is missing it. Two commands settle it:
>
> ```bash
> git ls-remote --heads origin          # which branches exist there at all
> git fetch origin && git status -sb    # how far this checkout is from them
> ```

### Option A — via a git remote

```bash
# on the source machine: push the branch you actually want
git push -u origin HEAD

# on the target
git clone <your-repo-url> vector-bench
cd vector-bench
git checkout <branch>                 # if it is not the default branch
```

### Option B — direct copy, no remote needed

```bash
# from the source machine
rsync -avz --exclude-from=<(printf 'sources/\nwork/\ndatasets/\nresults/\n.git/\n') \
  ~/AI_WORK/VECTOR_RESEARCH/vector-bench/ user@target:~/vector-bench/
```

Either way, confirm the scripts survived the trip with their permissions:

```bash
chmod +x run-benchmark.sh scripts/*.sh tests/*.sh
```

Option B copies the working tree, branch and all, and needs nothing pushed
anywhere — which makes it the shorter path when the work you want is not on the
remote yet.

---

## 3. Build

```bash
cd ~/vector-bench

# If the build host IS the benchmark host, use native so each engine gets the
# best SIMD path its CPU offers (including AVX-512, which this framework's
# original host did not have). Otherwise keep x86-64-v3 for portability.
./run-benchmark.sh build --march native
```

Expect roughly:

| Engine | Cold build |
| --- | ---: |
| pgvector | ~10 min |
| MariaDB | ~40 min |
| AliSQL | **1.5–3 h** |

AliSQL dominates because its CMake compiles the bundled DuckDB unconditionally
on Linux and redirects that output to `/dev/null` — the build looks stalled for
a long stretch. It is not. Build one engine at a time if you want to start
measuring the other two sooner:

```bash
./run-benchmark.sh build --engines pgvector --march native
./run-benchmark.sh build --engines mariadb  --march native
./run-benchmark.sh build --engines alisql   --march native   # start this early
```

> **Never mix `-march` between engines.** Rebuilding one with a different value
> turns the benchmark into a comparison of compiler flags. The value is baked
> into each image and reported in the manifest, so a mismatch is visible after
> the fact — but it wastes the run.

---

## 4. Datasets

```bash
./run-benchmark.sh fetch --list                 # sizes and roles
./run-benchmark.sh fetch                        # the four profile datasets, ~5 GB
```

Downloads are resumable and size-verified, so an interrupted transfer will not
be mistaken for a complete dataset.

---

## 5. The web interface (optional)

One more image, about a minute, and the rest of this document becomes something
you can drive from a browser instead of a terminal.

```bash
./scripts/build-images.sh --engine webui
./run-benchmark.sh web --allow-control
```

It binds `127.0.0.1`, so reach a headless machine through SSH rather than
exposing it:

```bash
ssh -N -L 8080:127.0.0.1:8080 you@target      # from your laptop
```

Then open `http://127.0.0.1:8080`. Its **Setup** page is this document's
sections 3, 4, 6 and 7 as a checklist that does each step — build the images, fetch a
corpus, run the smoke gate, then measure — and checks that every image agrees on
`-march`.

To expose it properly rather than tunnelling, see
[08-web-ui.md](08-web-ui.md): binding a non-loopback address turns password auth
on by itself, and `docker/webui/compose.yml` puts Caddy in front for TLS.

---

## 6. Verify before measuring

```bash
python3 -m pytest tests/ -q                     # 777 unit tests, ~30 s
./tests/verify-alisql-traps.sh                  # 8 engine-behaviour checks
./run-benchmark.sh run --profile smoke          # ~45 min/pass, every engine
```

A dozen of those tests draw a chart and skip unless matplotlib is installed.
That is expected: matplotlib lives in the bench images, not on the host, and
this framework does not put a scientific Python stack on the machine it
measures. Skipped is the right outcome; failed would not be.

**Do not skip the smoke profile.** It exercises every stage end to end and is
far cheaper than discovering a broken image eight hours into a full run.

There is also a one-minute synthetic cycle that needs no dataset download:

```bash
./tests/make-tiny-dataset.sh
./run-benchmark.sh run --profile dev
```

---

## 7. Measure

```bash
./run-benchmark.sh run --profile main
```

`main` is the profile sized to actually be run — two datasets at 1M scale,
roughly two days of ingest. `full --resource-pass both` describes the complete
measurement space and is about **eleven days** on this hardware; see
[07-planning-a-run.md](07-planning-a-run.md) before choosing it.

Give the machine to it. Nothing else running — no CI, no other containers, no
builds. Concurrent load distorts results by around 2× and the harness cannot
detect it: the Validity section reports CPU, SIMD and cpuset problems, but a
competing workload is invisible to it.

Resumable after an interruption:

```bash
./run-benchmark.sh run --profile main --resume --run-id main-<timestamp>
```

---

## 8. Check the environment the report recorded

After the first run, confirm the target is what you think it is:

```bash
python3 -c "
import json; m=json.load(open('results/<run-id>/run-manifest.json'))
c=m['host']['cpu']
print('cpu       ', c['model'])
print('avx512    ', c['has_avx512'])
print('hybrid    ', c['hybrid'])
print('cpuset    ', m['config']['resolved_resources']['server_cpuset'])
print('warnings  '); [print('  -', w) for w in m['warnings']]
"
```

`has_avx512: true` is the main thing worth confirming — both MariaDB MHNSW and
AliSQL VIDX document AVX-512 distance kernels, and results from a machine
without it do not transfer to one with it.

---

## 9. Bringing results back

```bash
rsync -avz user@target:~/vector-bench/results/<run-id>/ ./results/<run-id>/
```

The run directory is self-contained: manifest, raw records, charts and both
report formats. `report.html` inlines its charts and needs no network.

Copied runs appear in the web UI with no further step: run discovery is any
directory under `results/` holding a manifest. With no shell on the receiving
box, the UI's **Import a run** page takes the `.tar.gz` that
`./run-benchmark.sh export` produces, and can rename it on the way in — two
machines running the same profile on the same day produce the same run id.

To regenerate the report locally from copied results:

```bash
./run-benchmark.sh report --run-dir results/<run-id>
```

> **Regenerating a copied run is not the same as viewing it.** The recall
> measurements live in `results/annb/`, a sibling of the run directory rather
> than part of it, and scoring them needs the dataset file. Neither is included
> by the rsync above. Without them the regenerated report loses its recall
> section; with unrelated ann results present it reads those instead, since
> nothing ties them to this run. The web UI's Report tab checks both and warns
> before the button does anything. Viewing the copied `report.html` is
> unaffected — it is self-contained.

---

## Updating an existing checkout

Most changes do **not** require rebuilding images. The images contain only the
compiled servers plus a pinned Python stack; the orchestrator, ops harness,
report generator, profiles and resource configs are all read from the working
tree at run time.

```bash
cd ~/vector-bench
git pull
python3 -m pytest tests/ -q        # confirm the pull is sane
```

That is usually the whole procedure. Decide whether more is needed by what the
pull touched:

| Changed path | What to do |
| --- | --- |
| `orchestrator/`, `harness/`, `report/`, `config/profiles/`, `config/resources/`, `docs/` | Nothing. Picked up on the next run. |
| `overlay/` | Nothing — the working copy is refreshed on every run. |
| `docker/`, `config/engines/` (build flags or source tag) | Rebuild that engine: `./run-benchmark.sh build --engines <name> --march native` |

To check whether a rebuild is required after pulling:

```bash
git diff --name-only HEAD@{1} HEAD -- docker/ config/engines/
```

Empty output means your images are still valid.

Re-running the report over results you already have needs no new measurement at
all — useful after a report-generator fix:

```bash
./run-benchmark.sh report --run-dir results/<run-id>
```

---

## Optional: reuse local vendor repos instead of cloning

If the target already has MariaDB, AliSQL or ann-benchmarks checkouts, point at
them and skip the upstream clones:

```bash
export VB_REPO_MARIADB=/path/to/server
export VB_REPO_ALISQL=/path/to/AliSQL
export VB_REPO_ANNB=/path/to/ann-benchmarks
./run-benchmark.sh build
```

They are read only — the framework clones or `git archive`s out of them and
never writes to them.
