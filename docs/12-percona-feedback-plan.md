# Implementing the Percona vector team's feedback

Four requests came back from the server team on the benchmark we published, and a
fifth message pointed at a new commit. This document is the plan for all five. It
is a plan and nothing in it is implemented.

The short version: the commit we were pointed at is not a commit on the branch we
measure. It is the last commit of a pull request that **merged the whole vector
search MVP into Percona Server's `9.7` branch**, on a base roughly four months
newer than our pin, and that merge contains named fixes for two of the four
requests. So the re-pin is not one item among five. It gates the other four, and
until it is done, at least one of the requests cannot be answered honestly because
we would be reporting a bug that already has a fix.

---

## 0. How this document cites code, and why it cites no line numbers

Symbols only — a file and a function, class or key name, never `file:line`.
`docs/11-methodology-roadmap.md` §0 is where that policy is written down and
`tests/test_e15_methodology_roadmap_doc.py` enforces it there. The reason applies
here with extra force: this document's whole subject is a vendor tree that moved
721 commits under a pin, and §3 below is partly about the line citations that move
cost us. A line number dies to any edit above it; a symbol name dies only to a
rename.

---

## 1. Provenance of every claim here

**The four requests.** Relayed to this session on 2026-10-27, from the server
team's review of the benchmark report and from a standup comment. Paraphrased
below with their substance kept:

1. Get stack traces for the crashes, to get an idea of the cause.
2. `ef_construction` having so little effect is surprising, and worth digging into.
3. Filtering going that badly against pgvector is interesting, since the two
   implementations are more or less the same thing for iterative scan.
4. Benchmark the scenario where loading data and building the index are
   **separated**, as was done for pgvector in an earlier document. The reasons
   given: a bulk build parallelises the distance calculations across cores, and it
   avoids rewriting neighbour lists at every insert, so the writes become fewer and
   sequential instead of random. The ask includes a recommendation to users to
   bulk-load with a separate index build where possible.
5. A new commit, `56124b03`.

The report's own counterpart to request 2 is that `ef_construction` 50 → 800 bought
0.9 points of recall for 16× the build cost, and to request 3 that filtered
throughput falls to 0.13-0.22 QPS on sift at 3.3% selectivity while pgvector holds
flat. Both are in the published findings.

**The upstream state.** Measured on 2026-10-27 through the GitHub API against
`percona/percona-server`, not read from a page or inferred:

| What | Value |
| --- | --- |
| Our pin (`config/engines/percona.yml`, `source.tag`) | `02a8b80c`, 2026-10-13, PS-11626 |
| `vector-mvp` head now | `c0f76a5e`, 2026-10-20, PS-11751 — 7 commits past our pin |
| The commit we were sent | `56124b03`, 2026-10-26 |
| What it is the head of | PR #6278, `vector-mvp-97-squashed` → base `9.7`, **merged 2026-10-26 12:41 UTC** |
| That PR | 30 commits, 331 files, +37,587 / −322 |
| `9.7` head | `22e64c5b`, 2026-10-26 19:26 — 2 commits past the merge, neither one vector |
| Our pin against the merged series | **diverged**: 721 ahead, 43 behind |
| Server version | `9.7.1-1` at our pin, `9.7.2-2` at `9.7` |

**What is unconfirmed.** Nothing in §4 onward has been measured at the new pin,
because the image has not been built from it. Every estimate of what a phase costs
comes from the 2026-10-15 run or from the estimator, both of which describe the old
pin. The vendor's design document `Docs/percona-hnsw.md`, which now ships in the
tree, is read as a statement of intent by its authors: where this plan quotes it,
the quote is evidence about what the server is *meant* to do and the measurement is
what settles whether it does.

---

## 2. The merge, and what it changes about the four requests

The series was not fast-forwarded onto a newer base. It was re-landed: 721 commits
ahead and 43 behind means our 43 vector commits were replaced by a re-sequenced set
on top of roughly four months of upstream MySQL and Percona work. Pulling the new
pin therefore changes the InnoDB underneath the graph as well as the graph code.

Commits in the merge that bear directly on a request, each with the request it
touches:

| Ticket | What it is | Bears on |
| --- | --- | --- |
| PS-11748 | SIGSEGV in `HNSW::Node::state()` on ANN SELECT or INSERT/REPLACE after an unclean restart, release and debug | request 1 |
| PS-11591 | Assertion `inserted.second` in `HNSW::insert()` — crash on INSERT into a vector-indexed table after an unclean restart | request 1 |
| PS-11623 | Assertion in `ddl0loader` when a vector index build aborts on `innodb_hnsw_max_memory` exhaustion | request 1, request 4 |
| PS-11615 | Assertion in `vec_index_of()` when a vector search runs during an in-place vector index replace | request 1, our churn phase |
| PS-11617 | A same-name DROP + ADD of a vector index with a different `M` is a silent no-op | our churn repair path, §5.8 |
| PS-11612 | Resolved HNSW parameters stored in the data dictionary | request 2 |
| PS-11588 | HNSW index options silently accept duplicate keys | request 2 |
| PS-11618 | COSINE distance support for vector indexes | puts the angular corpora on the index path |
| PS-11212, PS-11616 | Vector index optimizer changes, and the remaining optimizer improvements | request 3 |
| PS-11750 | Status variable for total vector index memory usage | the gauge `docs/11` §4.1 says we collect nowhere |
| PS-11614 | `SELECT … FOR UPDATE/FOR SHARE` locked rows outside the result set via a stale candidate | correctness of the read path |

Two of those, PS-11748 and PS-11591, are **crashes on INSERT into a vector-indexed
table**, which is the shape of the failure the team wants traced. Both are in the
merge and neither is in our pin — and PS-11748 is also in `vector-mvp` itself, three
commits past the SHA we build.

Beneath them, the newer base carries InnoDB buffer-pool fixes that a crash
investigation would otherwise have to rule out by hand, including PS-11850
(`buf_page_init_for_read()` latching a stale page-hash cell during a buffer pool
resize) and PS-11495.

**The consequence for request 1.** Do not open a crash ticket from the
2026-10-15 run. The failures there were measured on a tree that is now missing two
named INSERT-crash fixes, and a backtrace taken against it would send the server
team after bugs they have already closed. Re-measure at the new pin first. That is
cheap: the fastest known reproducer failed in 43 seconds.

---

## 3. Phase 0 — re-pin, which gates everything else

1. **`source.tag` → `56124b03`.** Not `9.7`'s head: that branch moves daily and the
   two commits past the merge are an OIDC install-script option and a `buf_block_t`
   padding computation, neither of which is vector work. `56124b03` is an ancestor
   of `9.7`, so the tree is exactly what PR #6278 landed and it does not move under
   us. The key stays spelled `tag` for the reason that file already records:
   `orchestrator/engines.py`'s source-tag helper reads `tag` or `version` and
   nothing else, and a renamed key would publish a run whose engine version is
   blank. The surrounding comment names branch `vector-mvp` and must be rewritten —
   the engine is no longer built from a feature branch.

2. **Re-verify the cmake flags against the new base.** The flag list in
   `config/engines/percona.yml` and its deliberate duplicate in
   `docker/percona/Dockerfile` were established by five failed configures, and the
   lesson recorded there is that on this tree switching a dependency off breaks a
   Percona component that wanted the target. A four-month base move is exactly the
   event that invalidates that work. `march: x86-64-v3` does not change, for the
   reason CLAUDE.md gives.

3. **Version strings.** `9.7.1-1` → `9.7.2-2` wherever the figure is asserted
   rather than read, including the probe banner the harness prints on connect.

4. **`docs/09-adding-percona-server.md` citations will be wrong, and the suite will
   say so.** That document cites `file:line` into the pinned tree and into recorded
   MTR `.result` output, and `tests/test_e1_percona_design.py` checks them. The base
   moved and the vector commits were re-sequenced, so most of those citations now
   point at unrelated lines. Two ways forward: re-derive every one at the new pin,
   or convert the document to symbol-only citations as `docs/11` §0 prescribes.
   **Recommend the conversion.** Line citations into a moving vendor tree have now
   cost a red `master` twice and a full re-derivation once; this is the third event
   of the same kind, and the conversion is the only one of the two that does not
   have a fourth.

5. **`Docs/percona-hnsw.md` now ships in the vendor tree.** It is the authoritative
   design document for the feature, written by its authors, and several of our
   §2.x subsections exist because no such document was available. Cite it by
   section and delete what it supersedes rather than keeping a second account that
   can drift from it.

6. **Build, then `dev`, then `smoke`.** No long profile runs until smoke is green at
   the new pin. The smoke run is also what finally supplies Percona's ingest rate,
   which the estimator still names in `engines_without_ingest_rate` rather than
   modelling.

7. **A guard for this document.** `docs/09`, `docs/10` and `docs/11` are each
   guarded by a test that fails when the document drifts from the tree. This one
   should be too, symbol-only, in the shape of
   `tests/test_e15_methodology_roadmap_doc.py`.

---

## 4. Phase 1 — stack traces for the crashes (request 1)

### 4.1 Re-measure before building any diagnostics

What the 2026-10-15 run recorded, as the evidence a re-run is compared against:
13 of 24 ops units died inside the first INSERT with `Lost connection to server
during query` within 0.5-5.5 s, server RSS flat at the buffer-pool baseline and CPU
at 0.16 cores; across the million-row load configurations, 12 of 18 killed the
server. The failure was non-monotonic in `M`, which rules out a simple memory
ceiling, and peak RSS was 48.5 GiB against a 112.5 GiB cap. One case went further:
gist at tuned `M=32` loaded successfully in 20,760 s and then died during churn,
which puts the fault on the write path generally rather than on bulk load alone.

Re-run those configurations at the new pin. If they survive, the answer to the
request is the ticket list in §2 and no backtrace is needed. If any still dies, the
capture chain below is what turns it into something the server team can act on.

### 4.2 Container exit facts — the one artefact that cannot be reconstructed

Nothing in the orchestrator records the server container's exit state. `docker
inspect` carries `ExitCode`, `OOMKilled` and `FinishedAt`, and `OOMKilled` settles
cgroup-OOM against an in-process crash without any cooperation from the server.
Record all three per failed phase. This is the cheapest item in the whole plan and
the only one whose data is gone forever once teardown runs.

### 4.3 The server's own backtrace

mysqld's fatal-signal handler prints an `Attempting backtrace` block to
`--log-error`, and the build is `RelWithDebInfo`, so those frames carry names. The
error log is already copied out before teardown — `OpsRun._save_server_error_log`
exists for exactly this, and `config/engines/percona.yml`'s `error_log` key records
what its absence cost in the 2026-10-15 run. What is missing is the check that the
block actually arrives, plus `--core-file` in the engine's server flags.

### 4.4 Core dumps

Two gaps. The server container runs with no `--ulimit`, and `orchestrator/docker_ctl`
has no ulimit plumbing at all, so the core size limit is whatever the daemon
defaults to; add `--ulimit core=-1` for this engine. And the kernel's `core_pattern`
is host-wide: a benchmark framework must not modify the machine it measures, so
**read it, print it, and say plainly that no core will be produced when it is a pipe
to apport or systemd-coredump.** Document the one-line opt-in for the build host
instead of performing it. A core that lands at an absolute path lands inside the
container's namespace, so pointing the pattern into the data mount puts it where
the error log already goes — and under the same teardown trap, since the volume is
removed by `OpsRun.__exit__`.

### 4.5 Turning a core into text

gdb is in the builder stage's apt list and not in the runtime image. Add a `lab`
route — `./run-benchmark.sh lab --engine percona percona-backtrace.py` — that runs
`gdb --batch -ex 'thread apply all bt full'` against the saved core and the matching
mysqld. `lab`'s documented job is "diagnose, don't measure", which is exactly this.
The deliverable to the server team is the backtrace text and the statement in
flight, not our reading of them.

### 4.6 Reproducer context

With the backtrace, send the statement in flight, the row offset it died at, the
resolved server flags, and `SHOW ENGINE INNODB STATUS` where the server still
answers. The team asked for an idea of the cause; a reproducer they can run is worth
more than a trace they cannot.

---

## 5. Phase 2 — load and index build separated (request 4)

This is the highest-value request, and the framework already has the machinery: it
is `build_mode: post`, implemented, tested, and switched off because
`ALTER TABLE … ADD VECTOR KEY … ALGORITHM=INPLACE` killed the server at our pin.
`config/engines/percona.yml`'s `capabilities.bulk_vector_index_build` is the switch,
and its comment records the reproduction and says to flip it when the server is
fixed.

### 5.1 What the vendor document now says the bulk path does

`Docs/percona-hnsw.md` §15 states, as the authors' design: the first
`ADD VECTOR KEY` is an in-place **rebuild**, because the hidden label column has to
reach every row; the rebuild takes the sort rather than skipping it, so the
clustered scan and the graph inserts run on `innodb_parallel_read_threads` threads;
and the aux table is written **once at the end** by walking the finished graph,
instead of rewriting a node's row every time a later insert rewires it, which the
document costs as O(N·M·log N) row updates avoided. Those are the two reasons the
server team gave, in the server team's own words. §18 adds the third variant:
`ALGORITHM=COPY` writes the aux row by row with undo and redo, and is both slower to
build and larger.

### 5.2 The work

1. `capabilities.bulk_vector_index_build` → `true`, once a `post` unit survives at
   the new pin. Flipping it restores the ops loop's default and the churn repair
   path in one move; both read that one key.
2. `config/profiles/percona.yml`: `percona_build_modes: [incremental]` →
   `[post, incremental]`. The profile's comment already says the one-value list is
   the measurable half of a two-value axis.
3. **`innodb_parallel_read_threads` is set in none of our configs, and pgvector's
   counterpart is set in ours.** pgvector's tuned pass sets
   `max_parallel_maintenance_workers` from `build_threads`, with `max_parallel_workers`
   and `max_worker_processes` beside it, and its normalized pass pins all three to
   zero. Enabling Percona's bulk build without the same treatment would compare
   pgvector at `build_threads` workers against Percona at the server default: a
   comparison of defaults wearing the label of a comparison of engines, which is the
   failure mode CLAUDE.md opens with. Set it from `build_threads` in tuned and to 1
   in normalized, and record it in the manifest.
4. `PerconaDriver.create_index` should spell `LOCK=SHARED`. `LOCK=NONE` is refused
   while a vector index is on the table, and an unqualified ALTER leaves the choice
   to the server.
5. **Report time-to-queryable as its two components, and name the asymmetry.**
   Percona's first `ADD VECTOR KEY` rebuilds the table; pgvector's `CREATE INDEX`
   does not. A single "bulk build is faster" number hides a table rebuild inside one
   engine's figure. This belongs in the known-asymmetries list that the report always
   prints, beside the existing `ef_construction` and bulk-versus-incremental entries.
6. **Add the load-parallelism axis**, so the shape matches the pgvector document the
   request points at: one thread with the graph maintained per INSERT, against N
   threads of pure load followed by one bulk build. `load_threads: 1` is what every
   profile but two sets today, and `load_threads: 8` is already live in
   `config/profiles/tuned-complete.yml`, so this is a profile change and not a code
   change.
7. Optionally a third point, `ALGORITHM=COPY`, to put a number on what §18 claims
   about it.

### 5.3 The churn repair path is our measurement bug, not theirs

`PerconaDriver.maintain()` issues a same-name
`ALTER TABLE … DROP KEY vi, ADD VECTOR KEY vi …, ALGORITHM=INPLACE`. The vendor
document states that a same-name DROP + ADD with an identical definition is
cancelled by the server before InnoDB sees it, and PS-11617 fixes only the case
where `M` differs. That is a complete explanation for what we published 20 times:
1-4 ms, `index_bytes` identical to the byte, recall identical to five decimals. We
read it as a repair that does nothing; it is a statement the server discards.

The repair that does rebuild is `ALTER TABLE … FORCE` or `OPTIMIZE TABLE`, which the
vendor document names as the rebuild that repairs a corrupt index. Switching to it
changes what a churn number means, so it takes an `OPS_MEASUREMENT_VERSION` bump —
the charter in `harness/metrics/records.py` is explicit that a bump belongs to any
change that alters what an ops number means.

---

## 6. Phase 3 — ef_construction (request 2)

### 6.1 Why our current answer is weak evidence

The only sweep we have ran on fashion-mnist at 20,000 rows, where recall at the
smallest grid `ef_search` is already 0.989. The 0.9647 → 0.9734 figure for 16× the
build cost was measured against a ceiling, and a ceiling is not an answer about the
knob. The team's surprise is the right instinct and our measurement is not strong
enough to justify it either way.

### 6.2 What to measure instead

1. **Read the resolved parameters back.** PS-11612 stores them in the data
   dictionary, so the value the graph was actually built with is observable rather
   than inferred. Record it per build. The same read catches PS-11588's silently
   accepted duplicate option keys and PS-11617's silent no-op, both of which would
   otherwise look like a knob that does nothing.
2. **Sweep where there is headroom.** gist-960 euclidean, and glove-100 on cosine
   now that PS-11618 puts the angular corpora on the index path at all. At `M=8` as
   well as `M=16`: a smaller out-degree is where edge quality should matter most, and
   `M=16` alone cannot separate "the knob does nothing" from "the graph is already
   good enough at this width".
3. **Report the `ef_search` needed to reach recall 0.95 per `ef_construction`, not
   peak recall.** If build effort buys nothing, that number does not move. Peak
   recall at a saturated operating point cannot show the difference, which is how we
   got here.
4. **Keep it inside the `percona` profile.** One engine, nothing added to the shared
   seven-way grid, which is the reason that profile exists.
5. **State the two bounds the vendor document gives.** `ef_construction` is the
   candidate-search width inside `insert()` and graph quality is fixed at insert
   time, so no query-time setting can recover edges that were never created; and dead
   nodes accumulate per mutation, so a sweep measured after a churn phase is not
   measuring the same graph.

Cost: one full corpus reload per `(ef_construction, M)` pair. Three values at two
`M` on gist is six reloads, and at the 48 rows/s we measured on gist that is the
figure to put in front of the estimator before anyone starts it. It may be the item
that argues for a smaller corpus than gist.

---

## 7. Phase 4 — filtered search against pgvector (request 3)

### 7.1 Where the server team is right

The mechanism is the same shape. `Docs/percona-hnsw.md` §10 describes one
`nn_search_start` followed by `nn_search_next` per candidate, with the traversal's
visited set and unexplored frontier kept in the scan, so continuing costs only the
nodes not yet seen, a candidate is never offered twice, and the scan ends when the
frontier empties rather than at a guessed bound. `LIMIT` sizes the first batch and
nothing more. That is iterative scan, and the colleague's reading of it is correct.

### 7.2 Two differences the same document states, and how to measure each

**No cost model.** Vector index access carries `prohibitive_cost = 1e100` so that it
always loses against any other access method, and activation is therefore a *shape*
match rather than a cost decision; §18 says so directly — the read path is the old
optimizer's, with no cost model, and nothing refuses a `LIMIT` that asks an
approximate search to scan most of the index. pgvector's planner costs the
alternatives. Measure it: run every filtered point twice, once as published and once
with the index suppressed for the `ORDER BY`, and publish the pair. PS-11626 — the
last commit in our own pin — is what makes `IGNORE INDEX … FOR ORDER BY` work for a
vector index at all, so the experiment is available from the current pin forward. If
the suppressed plan wins at 1% selectivity, the finding is "the index is chosen where
a scan is cheaper", which is a planner result and not a graph-quality one. That is a
different conversation with the team than the one the current report starts.

**One clustered primary-key read per candidate.** §10 is explicit that the only disk
access per candidate is a primary-key read, performed under the session's own read
view, before the filter above can reject the row. At 1% selectivity that is on the
order of a hundred row reads per returned row. Instrument it rather than inferring
it: the `Handler_read_key` and `Handler_read_next` session counters, differenced per
query, give candidates consumed per result row directly. pgvector's symmetric
instrument is `EXPLAIN (ANALYZE, BUFFERS)` with rows removed by filter and heap
fetches. Then "filtering is bad" becomes a number per selectivity, on both engines,
in the same units.

### 7.3 What that needs from the harness

Per-phase engine status variables, which the harness collects for no engine today.
`docs/11` §4.1 already names the same gap from the other end:
`Innodb_hnsw_memory_used` is documented as recorded per phase in CLAUDE.md, the
README and `docs/09`, and is collected nowhere. Build the capture once and both the
memory figure and the candidate counters land; PS-11750 adds the vector-memory
variable on the new pin, so the gauge and its collector arrive in the same change.

Whether that bumps `OPS_MEASUREMENT_VERSION` is a question for the charter in
`harness/metrics/records.py` and not an assumption to make here: new fields beside
unchanged numbers are not the same as a changed meaning, and the charter is the
thing that decides.

### 7.4 One asymmetry to state rather than fix

pgvector has two iterative-scan modes and we sweep both. Percona has no mode to
sweep, so its comparable axis is `innodb_hnsw_ef_search` per selectivity. Say that
in the report instead of leaving a column that looks empty for one engine.

---

## 8. Order, gates and what each phase produces

| # | Phase | Gated by | Produces |
| --- | --- | --- | --- |
| 0 | Re-pin, rebuild, `dev`, `smoke` | nothing | a buildable engine at `56124b03`, and Percona's first measured ingest rate |
| 1 | Re-run the known crash configurations | phase 0 green | either "fixed upstream, here are the tickets" or a list of what still dies |
| 2 | Crash capture chain | a surviving failure from phase 1 | exit facts, error-log backtrace, core, `thread apply all bt full` |
| 3 | Bulk build restored, parallel load axis | phase 0; `post` surviving once | the separated load-and-build comparison the team asked for, with time-to-queryable split in two |
| 4 | `ef_construction` on a corpus with headroom | phase 0, estimator check | the `ef_search`-to-reach-0.95 curve per `ef_construction`, at two `M` |
| 5 | Status-variable capture, filtered instrumentation | phase 0 | candidates consumed per result row, both engines; index-suppressed comparison |
| 6 | Churn repair switched to a real rebuild | phase 3, version bump | a churn before/after pair that measures something |
| 7 | Report and docs | all of the above | `docs/05` asymmetries, `docs/09` re-pinned and re-cited, the team-facing write-up |

Phases 3, 4 and 5 are independent of each other and all three depend only on phase
0, so they can run in any order or in parallel on separate machines. Phase 2 is the
only one that might never run, and that is the good outcome.

---

## 9. What this plan deliberately does not do

- **It does not re-run `full`.** The `percona` profile answers requests 2 and 4 and
  most of 3, one engine at a time, at corpus sizes measured in minutes to hours
  rather than the 726 h of ingest a full seven-engine pass costs before a single
  query runs.
- **It does not change the `innodb_hnsw_max_memory = 0` policy.** The argument in
  `docs/09` §3 is about limit semantics, not about a value, and nothing in the merge
  changes that the knob is a hard allocation ceiling rather than a cache.
- **It does not touch `-march`.** Five engines compile SIMD distance kernels at
  `x86-64-v3` and a different flag on one turns the benchmark into a comparison of
  compiler flags.
- **It does not pin to `9.7`'s head.** A moving pin makes every result
  unreproducible, and the two commits past the merge are not vector work.
- **It does not add a host dependency.** gdb runs inside a container, like
  everything heavy here.
- **It does not take the team's statements as measurements, or ours as settled.**
  The vendor design document says what the server is meant to do; §5.3 is an example
  of our own published number having a server-side explanation we did not have, and
  §6.1 is an example of our evidence being weaker than the conclusion drawn from it.
