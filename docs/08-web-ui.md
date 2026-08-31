# 08 — The web UI

A browser front end for two jobs the CLI does not do well: finding your way
around a tree of past runs, and slicing their measurements without writing `jq`.
It can also author profiles and launch runs.

```bash
./run-benchmark.sh build --engine webui     # once, ~1 min
./run-benchmark.sh web                      # read-only, http://127.0.0.1:8080
./run-benchmark.sh web --allow-control      # plus profile editing and launching
```

Stop it with Ctrl-C.

---

## Contents

- [What it is for](#what-it-is-for)
- [Getting a machine ready](#getting-a-machine-ready)
- [Access model](#access-model)
- [Reading results](#reading-results)
- [Configuring and launching](#configuring-and-launching)
- [Sharing a result](#sharing-a-result)
- [Viewing results measured somewhere else](#viewing-results-measured-somewhere-else)
- [Deploying it with TLS](#deploying-it-with-tls)
- [How it runs](#how-it-runs)
- [What it deliberately does not do](#what-it-deliberately-does-not-do)
- [Troubleshooting](#troubleshooting)

---

## What it is for

`report.html` is self-contained and shareable, and stays that way. What it
cannot do is answer questions across a run — "show me every point above recall
0.95, grouped by engine and M" — or tell you what is in the other fifteen
directories under `results/`.

The UI adds:

| Page | What it does | Command it runs |
| --- | --- | --- |
| **Setup** | The ordered path from a fresh checkout to a number you can trust, doing each step from the page | `build`, `fetch`, `run` |
| **Status** | The inventory: which images are built and with what `-march`, which datasets are here, disk free, what every page does | — |
| **Engines** | Build images; add a variant of an engine family | `build` |
| **Datasets** | Every corpus, its shape and whether it is here; download the published ones, generate the ones that are not published | `fetch`, `generate` |
| **Profiles & launch** | Edit a profile with validation, see the ingest estimate, start a run | `run` |
| **Jobs** | Every long command started here, with live output and a stop button | — |
| **Runs → Overview** | The manifest as a page: CPU and SIMD, engine tags and `-march`, resolved limits, validity warnings, per-phase outcomes | — |
| **Runs → Explore** | The five measurements, each with its axes and filters already set; raw mode for anything else | — |
| **Runs → Report** | The generated `report.html` in a frame, and a button to regenerate it | `report` |

Every button shows the command it is about to run, so the UI teaches the CLI
rather than replacing it with something you cannot reproduce in a terminal.

---

## Getting a machine ready

**Setup** is four steps in the order you do them, each showing whether it is
done and doing it from the page:

1. **Build the engine images.** Lists every engine with its tag, whether its
   bench image exists, and the `-march` it was built with. The original three
   are selected by default and the rest are opt-in, for the same reason `build`
   leaves them out of its own default: each is another image to compile or
   another process to stand up, and a result does not need them.
2. **Get a corpus.** Offers the smoke dataset — 217 MB, the smallest thing that
   proves the pipeline on real data — and points at Datasets for the rest.
3. **Prove the pipeline.** Runs the smoke profile, which exercises every stage
   for every engine: images start, vector DDL is accepted, the index is actually
   used, records are written and the report renders. It says so, but: do not
   skip it.
4. **Measure.** The smoke profile is a gate, not a measurement — its grids are
   far too sparse to draw a curve from.

### The `-march` guard

Every engine compiles SIMD distance kernels, so building one with a different
`-march` turns the benchmark into a comparison of compiler flags — and nothing
about the resulting numbers looks wrong. Setup reads the value each image was
actually built with from `sources/<engine>.image.json`, defaults the field to
whatever the existing images agree on, and says so plainly if you change it or
if the images already here disagree with each other.

`native` is offered but is only correct when the machine building the images is
also the machine running the benchmark.

## Access model

Two independent switches: **who can reach it** (`--host`) and **what it can do**
(`--allow-control`).

`--allow-control` is off by default. Without it every mutating endpoint returns
403 and the Profiles, Engines and Jobs sections are hidden, so read-only mode is
safe to leave running.

### Local — the default

Binds 127.0.0.1, no password. Reach a remote rig over SSH:

```bash
ssh -N -L 8080:127.0.0.1:8080 you@bench-host
```

SSH is the authentication. A `Host` header allowlist also rejects anything that
is not localhost, so a hostile page in your browser cannot reach it by DNS
rebinding.

### Remote — password plus TLS

**Binding a non-loopback address turns auth on by itself.** `--host 0.0.0.0`
without a password refuses to run silently; pass `--no-auth` to override, which
you should only do on a network that is already private. In a container the bind
address is always `0.0.0.0`, so `--published-host` carries the address you
actually reach it on and the decision follows that.

```bash
export VB_WEB_PASSWORD='…'          # or let one be generated and printed once
./run-benchmark.sh web --allow-control --host 0.0.0.0 --behind-proxy
```

| | |
| --- | --- |
| Password | `VB_WEB_PASSWORD`, else generated on first run and printed once |
| Stored as | `scrypt` hash + per-install salt in `state/webui/credentials.json`, mode 0600. A password given by environment is never written down. |
| Session | random 256-bit id in a cookie: `HttpOnly`, `SameSite=Strict`, `Secure` under `--behind-proxy`, idle and absolute expiry |
| Revocation | `POST /api/logout`, and a server restart ends every session |
| Brute force | per-client exponential backoff after 3 failures; no lockout, because a lockout lets anyone lock you out |
| CSRF | `SameSite=Strict` plus an `Origin` check on every mutation |

**Why not JWT.** JWT buys stateless verification across many servers. This is one
process on one machine, so the only thing it would add is a token that cannot be
revoked without a server-side blocklist — which is the state back again, minus
the simplicity. A random session id in a dict this process owns is smaller and
can actually be revoked.

> **Auth is not enough on its own.** Over plain HTTP the password and the session
> cookie cross the network in cleartext and can be replayed, and this endpoint
> holds the Docker socket. The server says so at startup if you bind a public
> address without `--behind-proxy`. Put TLS in front — see below.

---

## Reading results

**Explore** loads `report/records.jsonl` when the run has a generated report,
and falls back to the raw `ops-*.jsonl` files when it does not. The fallback has
no recall/QPS records — those exist only after `run-benchmark.sh report` merges
the ann tree — and the tab says which source it used.

Filters are conjunctive: selecting `engine=mariadb` and `phase=recall_qps` shows
records matching both. Chart axes accept any numeric field present in the run,
so recall-vs-QPS, build time vs M, and QPS vs client count are all the same
control.

Every view is deep-linkable: `#/run/<run-id>/explore` restores the run and tab.

> **One dataset filter is worth knowing about.** The ann results tree is keyed
> by resource configuration, not by corpus, so a machine that measured two
> corpora under one configuration has both in one tree and both appear in the
> run's records. Filter on `dataset` before drawing conclusions, exactly as
> `report --datasets` does for the static report.

---

## Configuring and launching

**Configure** edits `config/profiles/*.yml` and nothing else. Resource passes
(`config/resources/`) and engine definitions (`config/engines/`) encode the
fairness invariants of the comparison and stay in git, where a change to them is
reviewable.

Validation runs before any write and catches the mistakes that do not crash:

- YAML that does not parse, or is not a mapping
- a missing `name` or `datasets`, or a `name` that disagrees with the filename
- `resources:` keys set to `{}`, which **does not** clear the inherited value —
  a profile's `resources` block merges dict-into-dict, so an empty map recurses
  into the inherited one and changes nothing. Use `null`. This silently halved
  the `m-sweep` profile once.
- more than four `ann.m_values`, because every M value reloads the whole corpus

The run plan below the editor produces the same ingest estimate the CLI prints
before a run — it calls `estimate_load_hours()`, the function the CLI's own
estimate is built from, so the two cannot disagree.

**Jobs** shows the launched run's output live, polling from a byte offset so a
long log does not re-transfer. Stop terminates the process group; units already
checkpointed stay done, so `--resume` picks up where it left off.

### Nothing runs alongside a benchmark

A download or a compile during an ingest measurement perturbs exactly what is
being measured — a competing build distorted MariaDB's numbers by 2× once,
which is why the README says to give the machine to the benchmark. So a run
blocks `fetch` and `build`, and they block a run. Setup jobs may overlap each
other, because nothing is being measured then.

The refusal names the job in the way and why, and an identical command that is
already running is refused rather than started twice.

### Command construction

Every plan becomes an argv list, never a shell string, and each token is checked
against an allowlist before it is used. An engine name that is not a known
engine, a dataset name with a space in it, or a run id carrying a terminal
escape sequence is rejected with a message rather than executed.

---

## Sharing a result

Three ways out, in order of how little the recipient needs.

**The report file.** `results/<run-id>/report/report.html` is self-contained —
charts inlined, nothing fetched — so it opens in any browser, offline, and looks
the same everywhere. The Report tab has a Download button for it.

**The run bundle.** `Download run bundle (.tar.gz)` on the same tab, or:

```bash
./run-benchmark.sh export --run-dir results/<run-id>
```

The archive is the run directory plus a README naming the machine the numbers
belong to, its SIMD flags, what was measured, and how to view it. Someone with
vector-bench extracts it into their `results/` and it appears in their UI;
someone without opens the HTML.

**The raw records.** `report/records.jsonl` is one flat JSON object per
measurement, for a recipient who would rather run their own analysis than read
your charts.

## Viewing results measured somewhere else

A run directory is self-contained to *view*. Copy one in and it appears:

```bash
rsync -avz you@other-rig:~/vector-bench/results/<run-id>/ ./results/<run-id>/
```

Run discovery is "any directory under `results/` holding a `run-manifest.json`",
so nothing has to be imported or registered. Overview shows *that machine's*
CPU, SIMD flags, engine tags and resource limits, because the manifest travels
with the run — which is the point, since a number without the environment that
produced it is not a result. `report.html` inlines its charts, so it renders
exactly as it did there.

**Regenerating a copied run is a different matter**, and the Report tab says so
before you click. The recall measurements live in `results/annb/`, a sibling of
the run directory, and scoring them needs the dataset file. Neither travels.
Two things can then go wrong, and they are opposites:

| What the tab says | What would happen |
| --- | --- |
| *Regenerating would drop the recall section* | There are no ann results here at all, so the new report would have the ops measurements and no recall curves |
| *Regenerating would not use only this run's data* | There are ann results here, but nothing ties them to this run — so it would read every ann result on this machine. If the run was measured elsewhere, the tab names the host and says plainly that those are somebody else's measurements |

A run that recorded its own `ann_fingerprint` and whose tree is present gets no
warning, because the generator can narrow to exactly its own results.

## Deploying it with TLS

`docker/webui/compose.yml` runs the UI behind Caddy, which obtains and renews a
real certificate. The UI gets no public port of its own.

```bash
cp docker/webui/.env.example docker/webui/.env   # fill in the five values
docker compose -f docker/webui/compose.yml --env-file docker/webui/.env up -d
docker compose -f docker/webui/compose.yml logs webui   # the generated password
```

On a machine with no public DNS name, prefer WireGuard or Tailscale and drop
Caddy: the transport is already encrypted, and the UI still wants `--auth`.

## How it runs

```
host                                    container (vector-bench/webui)
  browser ──▶ 127.0.0.1:8080 ──────────▶ webui.server
                                            │
  /var/run/docker.sock ◀────────────────────┤  (only with --allow-control)
  <repo> ◀── bind-mounted at <repo> ────────┘
                                            │
                                            └─▶ ./run-benchmark.sh run …
                                                  └─▶ docker run (engine containers)
```

The image carries python3, pyyaml, numpy, git and the Docker **CLI** — the daemon
stays on the host. It is not built from an engine bench image: the UI does not
need matplotlib or h5py, and building separately means it does not go stale when
an engine is rebuilt.

The CLI is fetched in a build stage and checked against a **pinned sha256** for
each architecture, then exactly one binary is copied forward; the published
tarball also carries dockerd, containerd and runc, none of which belong here.
There is no `.sha256` sidecar published for these artifacts, so the checksums are
constants in the Dockerfile and must be updated with `DOCKER_CLI_VERSION`. A
truncated download is a real failure mode — one occurred while this was written.

**The repo is mounted at its own absolute host path**, not at some tidy `/app`.
The orchestrator hands *host* paths to the Docker daemon when it launches engine
containers, so a container-only path would be resolved by the daemon against the
host filesystem, where it does not exist, and the engine would come up with an
empty mount. Mounting at the same path makes every path line up.

**The container runs as the invoking user**, not root. Two things break when it
does not: git refuses the ann-benchmarks working copy as "dubious ownership",
which stops every run before it starts, and everything the server writes under
`state/` and `config/profiles/` lands root-owned on the host, where its owner
cannot delete it. Reaching the Docker socket as a non-root user needs its group,
which is read from the socket itself rather than assumed to be called `docker`.

If you started the UI as root at some point, `state/webui/` may still be
root-owned and launching will fail with a permission error. Remove it with a
root container rather than sudo:

```bash
docker run --rm --user 0 -v "$PWD/state:/s" --entrypoint sh \
  vector-bench/webui:latest -c 'rm -rf /s/webui'
```

`./run-benchmark.sh web --no-container` runs the server directly on the host
instead. It is standard-library-only apart from PyYAML, so it works with the
host requirements this framework already assumes, and it is the easier mode to
develop against.

---

## What it deliberately does not do

- **Replace `report.html`.** The static report stays self-contained and
  shareable; the UI frames it rather than reimplementing it.
- **Edit resource passes.** They carry the fairness invariants of the
  comparison, and a change to one should be reviewable in git.
- **Add a new database architecture.** The Engines tab adds a *variant* of a
  family the harness already drives — another MySQL fork, another Postgres
  build — because that is genuinely just `config/engines/<name>.yml`: a source
  tag, an image name, server flags and a SQL dialect. Something none of the
  existing drivers speak needs a driver in `harness/drivers/`, an
  ann-benchmarks module and a Dockerfile, and the form refuses a config that
  names a driver which does not exist rather than accepting one nothing can
  serve.

  Two collisions it also refuses, both of which would otherwise produce a
  plausible config that measures the wrong thing: reusing another engine's
  `ann_constructor`, because ann-benchmarks keys its result files on that name
  and the second engine would report the first's recall; and reusing another
  engine's image tag, because the two builds would overwrite each other.
- **Compare runs measured under different budgets.** The ann fingerprint exists
  precisely to keep those apart. Overlaying them is a feature that would need to
  refuse more often than it accepted.
- **Queue runs.** One at a time, refused rather than queued.
- **Survive a restart mid-run.** A run launched here keeps going if the server
  dies — it is a detached process group — but the UI cannot re-attach to its
  output. The job is listed as `orphaned` and the run's own log in
  `results/<run-id>/` remains authoritative.

---

## Troubleshooting

**`vector-bench/webui:latest not found`** — build it:
`./scripts/build-images.sh --engine webui`.

**Configure and Jobs tabs missing** — the server is read-only. Restart with
`--allow-control`.

**403 "control is disabled"** — same cause.

**403 "host not allowed"** — you reached the server by a name that is not
localhost. Use the SSH port-forward.

**A launch fails immediately** — read the Jobs log. The usual causes are a
missing engine image and a dataset that was never fetched; both name themselves
in the first few lines.

**The engine container starts and finds nothing to load** — the repo was not
mounted at its own path. Use `./run-benchmark.sh web`, which does this
correctly, rather than a hand-written `docker run`.
