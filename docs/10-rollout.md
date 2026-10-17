# Rolling out the ops-harness work: what to run, in what order, what to check

Every step in this document that builds an image, starts a server or runs a
profile is the operator's. Every figure in the document itself came from a
module in this repository, from a measurement taken against
`tests/fake_driver.py`, or from a document — none of it from a running server.
**The first live run is the first measurement of these engines**, and several
numbers below exist only to say what that run will *not* settle. That no live
run has happened yet is a claim about history rather than about the tree, and
§1 says what this repository can and cannot confirm of it.

Read `docs/07-planning-a-run.md` for cost and `docs/05-methodology.md` for what
each figure means. This page is about sequence and about the two checks that
cannot be performed as the specification writes them.

## 0. Provenance of every claim here

The project specification's rollout section is the authority for what the six
steps and the seven checklist items *say*. It is not in this repository, so
every quotation from it is marked **supplied** and reproduced verbatim rather
than paraphrased.

| Kind of claim | How it was established | Marked |
|---|---|---|
| A constant, a profile key, a driver attribute | imported or loaded and read back, at the commit this document ships on | **measured** |
| A sentence or literal in this repository | cited `file:line`, and `tests/test_e3_rollout_doc.py` opens the cited line | **read** |
| The six steps, the seven checklist items, the cost envelope | from the project spec | **supplied** |
| Anything about MariaDB, AliSQL, pgvector, Valkey, MongoDB or Percona behaviour | from a vendor source, an MTR result or a document | **supplied** |

Two consequences, and they are the reason this section is first:

* **A figure whose inputs are supplied is not a measurement of this system**,
  however arithmetically sound it is. §5 item 1 is the worked example: its
  formula is asserted by a vendor test, and every number fed into it comes from
  this repo's profiles, so none of its outputs is a measurement.
* **A figure measured against `tests/fake_driver.py` is a measurement of this
  harness**, not of an engine. §4 item 1 depends on exactly that distinction.

## 1. What the spec's rollout assumed, and what happened instead

The spec interleaves merges with live runs: *"The next PR's tasks may start in
parallel, but nothing is merged to the main branch until the human confirms the
step for the previous PR."* (`SPEC.md:527`, supplied.) That did not happen.
Every PR in this series merged without a run between them.

**This repo carries no evidence either way**, and that is worth stating
plainly: `/results/` is gitignored, so an absence of run artefacts here is not
an absence of runs. The claim that no profile was run is **supplied** by the
orchestrator that drove this work, not established from the tree.

The repository's PR numbers are not the spec's. The spec has PR 0 to PR 5; the
repository numbers its own independently, and #4 does not appear in this
history at all. The mapping, from the merge commits:

This paragraph deliberately does **not** name the highest pull request merged
so far. It used to, and that figure went stale within the hour of being
written — the next merge was this document's own. A monotonically increasing
fact about the repository cannot be stated in prose and stay true; the gap at
#4 can, because history does not acquire new gaps. The table below carries the
numbers that matter, and each row is checked against the merge commits.

| Spec PR | Tasks | Landed as | State |
|---|---|---|---|
| PR 0 | T0, and T5a if it finished first | #2 `pr-0-fake-driver` | merged |
| PR 1 | T1a–T1d | #3 `pr-1-concurrency` | merged |
| PR 2 | T2a–T2b | merged into PR 3's branch, landed via #5 and #6 | merged |
| PR 3 | T3a–T3c | #5 `pr-3-filter`, then #7 `t3b-drivers`, #8 `t3c-filtered` | merged |
| PR 4 | T4a–T4b | #9 `t4a-churn`, #12 `t4b-readers` | merged |
| PR 5 | T5a, T5b, T5c | T5a via PR 0's branch; #16 `t5b-stats` | merged; **T5c optional and absent** |

PR 5 is **not** a single task, and the draft order this document replaces was
wrong to treat it as one. T5a landed in PR 0's branch, which the spec explicitly
permits — `harness/metrics/stats.py:49` is `bootstrap_ci`, with
`tests/test_stats.py` beside it. **T5b landed as #16**, bringing
`report/crosscheck.py`, the report's section 10 and the uncertainty intervals.
T5c (open-loop latency, `harness/workloads/rate.py`) is optional and the file
does not exist.

**So every spec PR has merged, and nothing in §2 is waiting on a task.** What
is left is the run.

The remaining merges are not spec PRs: #10, #11, #13 and #14 are corrections to
work already merged, #15 is the Percona design document, and #1 predates this
series.

An earlier revision of this page said T5b was unmerged and built §2's order and
§3's table on its absence. It was written one pull request earlier; both are
corrected below, and `tests/test_e3_rollout_doc.py` now fails if either claim
drifts back. It failed twice in one sitting on exactly that — once when #16 was
written and again when #16 merged — which is the guard working, not the guard
being wrong.

### The one step that cannot be performed as written

> *"3. Merge PR 3 (largest); run `dev` per engine, then
> `verify-alisql-traps.sh`, then `smoke`. Compare tag results with the previous
> smoke run: they should match within noise, since only the measurement version
> changed."* (`SPEC.md:530`, supplied.)

There is no previous smoke run to compare against **through the report**, and
the obstacle is a deliberate one.

`OPS_MEASUREMENT_VERSION` is 7 (`harness/metrics/records.py:413`). Records
written before the constant existed carry no version key at all and are
version 1 by definition, not "unknown"
(`harness/metrics/records.py:420`). The archived corpus is entirely version 1:
`harness/metrics/records.py:240-246` records a count of it, and that count was
re-taken independently for this document — 239 ops record files, 2,134 ops
records, `extra.ops_measurement_version` absent from every one, every record in
an ops phase. That tree is **not part of this repository** and is resolved
relative to the checkout's parent directory, so whether a test can reach it
depends on where the checkout sits. The guards count it where it is reachable
and otherwise check that the comment still says what this document quotes.

**The refusal is unconditional on any two distinct version values.** It is not
"two versions that mean different things":
`report/measurement_versions.py:148` returns early only when fewer than two
versions are present, so a version 1 record and a version 5 record in one
report raise `MixedMeasurementVersions`, and so do a version 1 and a version 2
— which `report/measurement_versions.py:187-191` says **denote the same
measurement**. The equivalence set
(`harness/metrics/records.py:451`) changes the *wording* of the refusal,
dropping the advice to re-measure; it does not change the decision. And the
refusal is not a caveat on a chart: `report/generate.py:1213-1214` prints the
message and returns 2, so there is no report.

So step 3's comparison has no in-report form. It can be done by hand, by
reading the two JSONL sets side by side, with the version difference held in
mind.

**Its expectation was wrong anyway**, and this matters more than the mechanism.
The step says the results "should match within noise, since only the
measurement version changed". Version 5 deliberately changed what one recorded
quantity *is*: `harness/metrics/records.py:229-234` says
`extra.reinsert_rows_per_s` **"CHANGES WHAT IT MEASURES"** — the re-insert now
runs against four concurrent readers by default, which the comment calls *"the
same workload against a different amount of load, not two samples of one
quantity."* Values are supposed to move. The check to run is not "did it match"
but "is each move explainable", and the version stamp is what licenses the
move.

This document does not state how many defects in this series moved a measured
value. The orchestrator's draft said twelve; no source in this repository
supports a count, so it is **not established** and has been removed rather than
repeated.

Step 6's wording holds exactly as written: *"Old `main` results stay valid as
`ops_measurement_version` 1 and are not mixed into new charts."*
(`SPEC.md:533`, supplied.) The refusal above is what enforces it.

## 2. The revised order

**Nothing below is blocked.** Every merge the spec's six steps gate has
happened, including #16, which brought the consistency section spec step 5 asks
you to review. What is left of each step is its run.

All six of the spec's steps are accounted for below:

| Spec step, verbatim (`SPEC.md:528-533`, supplied) | Where it lands here |
|---|---|
| 1. Merge PR 0 and PR 1; run `dev` for all six engines, then `smoke` for pgvector and MariaDB 11.8. | steps A and C, widened to all engines |
| 2. Merge PR 2; run `smoke` for all engines. Check the operating-point table: MHNSW and VIDX should show `floor_above_target` or a low chosen `ef`. | step C |
| 3. Merge PR 3 (largest); run `dev` per engine, then `verify-alisql-traps.sh`, then `smoke`. Compare tag results with the previous `smoke` run: they should match within noise, since only the measurement version changed. | steps A, B and C — **the comparison cannot be done through the report**, see §1 |
| 4. Merge PR 4; run churn-only `smoke` (`--workloads build,churn`) on pgvector and MongoDB first, where the new hooks do something. | step D |
| 5. Merge PR 5; regenerate reports for the PR 1–4 `smoke` runs (no rerun needed) and review the consistency section. | step E — the section now exists, see §4 item 5 |
| 6. Only then start `main`. Old `main` results stay valid as `ops_measurement_version` 1 and are not mixed into new charts. | step F |

Steps 1 and 2 collapse into one `smoke` across all engines rather than two,
because every merge they gate has already happened and running pgvector and
MariaDB alone first would buy nothing.

**Step A — `dev`, all engines.** The cheapest end-to-end exercise: images build,
every driver connects, every phase writes records. `dev` pins
`ops.churn_readers: 0` (`config/profiles/dev.yml:77`), which is the write-only
re-insert pass and is deliberate, not an omission.

**What step A cannot confirm, and the draft of this page had it wrong.** With
readers off there is no reader pool, so the `during_writes` churn record is
never written: `harness/workloads/churn_readers.py:156-157` returns a count of
zero for `readers <= 0`, and `harness/workloads/churn.py:279-280` says
`reader_baseline` being None is *"the one state in which the whole
during-writes record is absent"*. Step A therefore exercises two of the three
new record shapes — the `ef_calibration` phase and the second churn record
stamped `extra.maintenance_state` — and not the third. The third first appears
at step C.

**Step B — `verify-alisql-traps.sh`.** Spec step 3's middle item, and
independent of the comparison above. `tests/verify-alisql-traps.sh` checks
`docs/04-engine-notes.md` against a live server, so it is a documentation check
that needs an engine.

**Step C — `smoke`, all engines.** The real gate. `config/profiles/smoke.yml`
does not pin `ops.churn_readers`, so it inherits the default of 4 — this is
where the reader pool and the `during_writes` record are first exercised. It is
also where the operating-point table gets its first look, which is spec step 2's
check: *"Check the operating-point table: MHNSW and VIDX should show
`floor_above_target` or a low chosen `ef`."* (`SPEC.md:529`, supplied.) The
status names are real — `harness/workloads/calibrate.py:102-109` declares `ok`,
`target_unreached` and `floor_above_target` — but **what MHNSW and VIDX will
show is supplied**, from the tuned-profile notes the spec cites. Anything else
is a finding, not a config slip, because calibration makes `ef_search` measured
rather than declared.

**Step D — churn-only smoke on pgvector and MongoDB**, `--workloads
build,churn`. Spec step 4: *"run churn-only smoke (`--workloads build,churn`)
on pgvector and MongoDB first, where the new hooks do something."*
(`SPEC.md:531`, supplied.) Which hook is different for each, and §4 item 4
below says why that matters.

**Step E — regenerate the reports for steps A, C and D and read the consistency
section.** No rerun needed. Spec step 5. The section is **section 10 of the
report** (`report/render.py:3201`), computed by
`report/crosscheck.py:327`'s `consistency_checks`. §4 item 5 says what to look
at and what the coverage table means.

**Step F — only then `main`.** Spec step 6. On `main` the cross-check is no
longer empty by construction — see §4 item 5 — so step F is also the first run
that exercises it against published numbers.

### Do not pool one run's churn records with another's

`main.yml` pins `ops.churn_readers: 4` (`config/profiles/main.yml:200`) and
`dev.yml` pins `0` (`config/profiles/dev.yml:77`). Both stamp version 5, so the
cross-version refusal in §1 will **not** catch a reader-contended re-insert rate
pooled with a write-only one. The code names the gap itself:
`harness/metrics/records.py:253-260` says *"`extra.readers` is a diagnostic,
not a guard. Two `reinsert_rows_per_s` figures both stamped version 5, one at 4
readers and one at 0, load together with no refusal."*

**This is reachable inside this runbook, not only between `dev` and `main`.**
Step A runs at 0 readers and step C runs at 4. A write-only re-insert rate and
a reader-contended one are different quantities; keep the two steps' reports
separate.

No shipped profile sweeps the value within one run, and nothing shipped can: the
orchestrator resolves `ops.churn_readers` to a single integer per run, so a list
would not survive resolution. The hazard is pooling across runs by hand, which
is also what the code's own closing sentence anticipates.

## 3. One checklist item is not satisfiable as written

The spec's pre-publication checklist has seven items. Six of them are work. One
of them cannot be discharged as the sentence asks, and the reason is mechanical
rather than a matter of effort. It is listed separately here because it is the
claim a later reader is most likely to tidy into something false.

| Item | What cannot be done | Why, mechanically |
|---|---|---|
| 1 | Establish "no point is `client_bound`" by reading that flag | `CLIENT_BOUND_UTIL` is a CPU-utilisation threshold and is sufficient but not necessary; a client ceiling that burns no CPU leaves the flag false |

Everything else on the list is checkable at the point the spec puts it.

**Item 5 was in this table and has left it.** It was unsatisfiable for a reason
that has since been removed rather than argued away: the consistency section did
not exist, and now `report/crosscheck.py` does and the report renders section 10
from it. The row is gone because the condition is gone, and the guard that held
it now asserts the opposite — if `report/crosscheck.py` were ever removed, this
page would have to say so again. Item 1's row has no such condition attached to
it, which is the difference between the two and the reason only one survives:
nothing short of replacing the quantity the flag is computed from would retire
it.

## 4. The seven-item checklist, annotated

Quoted verbatim from the spec (`SPEC.md:535-541`, supplied).

### 1. "Every concurrency record has recall, and no point is `client_bound` (or the report marks it)"

The recall half is straightforward and enforced. **The `client_bound` half
cannot be satisfied by reading that flag**, and this is the flag's own code
rather than an opinion about where a threshold sits.

`harness/workloads/pool_window.py:58` sets `CLIENT_BOUND_UTIL = 0.80`, and
`harness/workloads/pool_window.py:184` is the whole decision:
`self.client_cpu_util > CLIENT_BOUND_UTIL`. So a client that is not burning CPU
is not flagged, whatever else is limiting it. The module states why moving the
threshold does not help: *"a CPU-utilisation threshold cannot see a GIL
ceiling, because the lock prevents the CPU consumption the threshold looks
for"* (`harness/workloads/clients.py:34-45`).

**The shape that demonstrates it was measured against the fake driver, and the
module says it may not reproduce.** `harness/workloads/clients.py:34-45`
reports the concurrency workload's own CPU flat at 2.06 cores across 4, 8, 16
and 32 clients — 0.515 of a four-CPU container, comfortably under the 0.80
threshold — while aggregate throughput peaks at two clients and falls by better
than half by thirty-two, with the flag firing on none of those points. That is
a measurement of `tests/fake_driver.py` on a twenty-CPU box, and it was taken
on the **retired thread pool**: the clients are OS processes from
`OPS_MEASUREMENT_VERSION` 6, so no interpreter-lock ceiling is reachable any
more and what the flag is now about is the cpuset alone. The shape is still
the one to recognise, which is why both the module and this page keep it.
The next paragraph,
`harness/workloads/clients.py:47-55`, says the fake's queries are numpy-bound
in-process whereas a real driver spends most of a query blocked on a socket with
the lock released, so **"the synthetic SHAPE may not reproduce"**. The earlier
draft of this page presented the figure as a measurement of this system; it is
not, and the direction of the error is known: the regime it was taken in
under-reports what a real client costs.

**So read the condition as a shape, not a flag.** A flat Client CPU column
beside a falling QPS curve is the signature. If QPS falls as clients rise and
client CPU does not move, treat the point as client-bound whatever the flag
says. The flag's `None` is a third state and means nobody looked, not that the
clients were fine.

**That shape is now measured rather than left to the reader, and a live `smoke`
run is why.** MariaDB on fashion-mnist-784 at M=16 measured 2,920.7 q/s at one
client and 2,927.5 at four with every latency percentile unchanged — about one
query in service, and a `scaling_efficiency` of 0.25 that describes the client.
`harness/workloads/clients.py` now records, per window, how many queries the
pool had outstanding and how many it had in service, and
`extra.scaling_status` is `concurrency_not_achieved` where the second fell
below half the client count; `report/render.py` then renders no efficiency
figure for that point and names it in the Validity section.
**It does not retire this item's row.** The quantity is computed from the
spans, not from the CPU counter, so it is a different flag rather than a
reading of this one — and on the four-client point that delivered 0.26 of
linear, this one's utilisation read 0.177 against its own 0.80 limit and
stayed false. (That 0.177 is a figure from a containerised run. The counter
behind it is a cgroup's, and outside a container it reads the whole machine,
so it is not reproducible on a developer box; §7 of the methodology says so
beside it.) Both are kept, because each sees a case the other cannot:
`client_bound` names a saturated cpuset, and the new status cannot. See
`docs/05-methodology.md` §7 for the five-driver calibration and for why the
exact span-sum is the blind one.

### 2. "Operating-point table reviewed; every non-`ok` status explained in the text"

See step C. The three statuses are real and named in
`harness/workloads/calibrate.py:102-109`; which engine shows which is supplied
and is what the run settles.

### 3. "Filtered: `near_fail` recall shown for `ctag`; time-bounded points greyed"

Note what changed underneath this item: the selectivity label used to be the
share *requested*, and is now the share *delivered*
(`docs/05-methodology.md:490-497`). If you compare against an older chart, the
axis means a different thing.

**The overshoot is not a single number**, and the draft of this page quoted one
as if it were. `harness/datasets.py:456-468` measures a nominal 1% delivering
between 1.04% and 3.47% of the rows with a median of 1.43% over sixteen
clustering seeds, and an independent re-derivation from a different draw came
out *wider*, so those are a lower bound on the gap rather than its extent. The
overshoot is always upwards but its size depends on which cluster the
renumbering put first, so it is not a bias a reader could correct for. The
single figure the methodology quotes is for a 2,000-row test fixture, not for a
corpus a run would use.

### 4. "Churn: before/after maintenance shown where a repair path exists; `repair_path: none` stated elsewhere"

**Two engines have a repair path: pgvector and Percona Server.**
`harness/drivers/base.py:258` sets `repair_path: str = "none"` as the class
default and `harness/drivers/base.py:254-257` names the inheritors; the two
overrides in the tree are `harness/drivers/postgres.py:172`,
`repair_path = "vacuum_analyze"`, and
`harness/drivers/mysql_family.py:751`, `repair_path = "index-replace"`.
MongoDB inherits `"none"`. The draft of this
page said pgvector and MongoDB were both non-trivial here; they are not, and a
before/after maintenance panel for MongoDB would be an empty comparison
presented as a measurement.

**This page said "only pgvector" and that was true when it was written.**
Percona Server's path is an `ALTER TABLE … DROP KEY vi, ADD VECTOR KEY vi (…)`
which rebuilds the graph from the live rows, so the tombstones a DELETE leaves
behind are collected — and **by design** it is the expensive one of the two: a
whole graph rebuild per churn fraction, on the order of a bulk build at 1M
rows, against pgvector's VACUUM. `orchestrator/estimate.py` discloses both by
name rather than charging for either. `docs/09-adding-percona-server.md` §2.7
has the mechanism and the lock level (SHARED — readers continue, every writer
stops for the duration).

> **CORRECTION: at the pinned `source.tag` that repair does not run, so there
> is nothing to budget for and it is not the expensive one of the two — it is
> the one that costs nothing.** Three sentences above are now wrong and are
> left standing because the design claim they make is still right if the
> server is fixed. `maintain()`'s `ALTER … ADD VECTOR KEY … ALGORITHM=INPLACE`
> is the same statement `capabilities.bulk_vector_index_build: false` in
> `config/engines/percona.yml` records as killing the server. Measured on
> `full-20261015-215006`: **1–4 ms**, `index_bytes` identical to the byte and
> recall identical to five decimals in all 20 before/after pairs. So:
> "Neither has a measured rate at any corpus size" — pgvector's still has
> none, Percona's is now measured at effectively zero *because nothing runs*;
> and "budget for it before scheduling a churn phase on this engine" —
> **do not**, at this pin the churn phase pays no repair at all.
> `PerconaDriver.declare_bulk_vector_index_build` withdraws the path per
> instance, the class keeps declaring the mechanism, and §4's heading above
> ("before/after maintenance shown where a repair path exists") resolves for
> Percona to the `repair_path: none` branch until the capability is flipped.

What MongoDB has instead is the index-sync wait, a different hook:
`harness/drivers/base.py:340-353` returns `0.0` by default and MongoDB and
Valkey override it. So spec step 4's "where the new hooks do something" is
correct about which two engines to run first, and this checklist item is about a
different hook than that step is.

**Valkey is no longer a no-op either.** The draft said MariaDB, AliSQL and
Valkey all record exactly what they recorded before. Since #14, Valkey's churn
records carry `extra.index_sync_s` and a note saying what the wait observed —
`docs/05-methodology.md` documents the jq to read them. MariaDB and AliSQL
override no churn hook at all and are the only engines for which that sentence
is still true.

### 5. "Consistency checks pass, or each failure is explained"

**This section now exists.** `report/crosscheck.py:327`'s `consistency_checks`
matches ops calibration points to ann recall points, and the report renders it
as section 10 (`report/render.py:3201`). It is the check that cross-references
the two measurement paths against each other, which is the one check that could
have caught a published-result defect that neither path contradicted on its own.

What to read, in the order it matters:

* **The coverage table, before the matches.** An empty match list has two
  causes and they are not the same news. "No `M` measured by both paths" is a
  profile choice; "`M` overlaps but no point matched" is a **fault**. The
  coverage table says which case each engine and dataset is in, so an empty
  section is only reassuring once you have read it.
* **The tolerance, when a row fails.** It is
  `max(0.01, 2 × the ops 95% CI half-width)` — one recall point is allowed
  unconditionally because the two paths score different query subsets, and a
  record with no usable interval gets exactly the floor
  (`report/crosscheck.py:105`). A failure at a wide interval and a failure at
  the floor are different findings.
* **The QPS inversions, which are flagged and never failed.** The ops path pays
  a TCP round trip per query that the in-process path does not, so two engines
  close together in-process can legitimately swap places.

**On `main` an empty section is a fault, not a design.** That profile sweeps
`ann.m_values: [8, 16, 32]` against `ops.m_values: [16]` — see §5 item 3 — so
the grids meet, and every engine with a calibration record should produce
matched rows. `report/render.py:2826` and `docs/05-methodology.md:953` both say
so.

Until #16 was corrected they said the opposite: each described the `[8, 32]`
grid `main` carried before the pull request that added this section, and
called an empty section there "empty by design". The profile and those two
sentences moved in the same change and only the profile was updated, so an
operator reading either one would have filed a real fault as expected
behaviour. This page flagged them until they were fixed; they are fixed, and
the guard that held the flag now asserts the correction instead.

### 6. "`05-methodology.md` updated for all five PRs"

**Satisfied at this commit**, and it was the item that took longest to get
there — it started as four gaps and closed one PR at a time. Scanning
`docs/05-methodology.md` for the names the new work introduced:

* **PR 2 is now covered**, in a §2.0 placed before §2.1: the sweep and the
  `ef_calibration` records it writes, the selection rule, `floor_above_target`
  and `target_unreached` with the direction each one qualifies the sections
  below in, what the operating-point table's marks mean, and which later
  sections inherit the choice. The figures it quotes about real engine
  behaviour are marked supplied, because no run this repository holds contains
  an `ef_calibration` record.
* **PR 1's client-bound flag is now covered**, at the end of §2.3: the three
  numbers the judgement rests on, the 0.80 threshold and what it is and is not,
  and the blindness §4 item 1 above turns on — including the signature of the
  case the flag misses, which is flat QPS with flat latency.
* **PR 4's third record shape is now covered**, in §2.5: `during_writes`, why
  it carries no recall, and the one function every surface excludes it through.
* PR 3 and the version scheme are well covered.
* **PR 5 is now covered**, in §2.6 (uncertainty intervals, and when two engines
  are not ordered) and §2.7 (cross-checking the two measurement paths).
  §2.7's account of the empty case named a profile configuration that no longer
  held when it was written; that is corrected, and §4 item 5 above records what
  it said.

So the item has gone from four gaps to none. The last one to close was the
oldest and the largest, and it closed as a section rather than a paragraph for
the reason this entry gave while it was still open: calibration is not a
seventh measurement dimension but the premise of three of them, which is why
the report puts the operating-point table before concurrency, filtered search
and churn rather than among them. §2.0 is numbered 2.0 rather than inserted as
a new 2.1 so that nothing below it was renumbered, because the §-references in
this document and in `tests/test_e3_rollout_doc.py` point into that numbering.

The guards for this item are a restated list and not a derivation of the
sentences above, whatever an earlier revision of this paragraph claimed: the
names live in `TestTheMethodologyGapsAreDerived.ABSENT` and `PRESENT`, and a
gap closes when a name moves from the first tuple to the second. `ABSENT` is
now empty, which that class asserts is the terminal state rather than a
disabled guard — so this entry cannot quietly go back to calling a covered
name absent, and a name that regresses out of docs/05 fails the `PRESENT` half.

### 7. "Run manifest records `ops_measurement_version`, seeds and the resolved profile"

Satisfied for the version: `orchestrator/manifest.py:219` writes
`config.ops_measurement_version` from the constant, `report/render.py:326-327`
prints it above the numbers, and `docs/05-methodology.md:1021-1022`
documents it.
This is what keeps a version 1 curve out of a version 5 report, and what makes
everything above auditable a year from now.

## 5. Three things a green smoke run does not tell you

### 1. The HNSW memory policy, once Percona lands

`innodb_hnsw_max_memory` is a hard bound and not a cache, which makes it a
fairness problem rather than a tuning knob
(`docs/09-adding-percona-server.md:700`). That document holds the floor
formula, the per-dataset table and the policy decision, and
**the numbers are not repeated here on purpose** — `docs/09` carries its own
guards, including a test that recomputes every row of that table from the
formula it cites, and a second copy in this file would be a second thing to go
stale. This repository's most frequent defect is a figure restated in two
places and corrected in one.

Where to look: `docs/09-adding-percona-server.md:754-793` for the table and its
provenance — which says in its own words that the figures are projections and
not measurements, because row counts, dimensionality and `M` are all supplied —
and `docs/09-adding-percona-server.md:859-862` for the conclusion that matters
to a rollout: the smoke profile fits under the shipped default, so **a green
smoke run is not evidence the memory policy is sound**. The `main` profile does
not fit. The policy chosen is unlimited, which removes the hazard; the point is
that the gate would not have surfaced it either way.

### 2. Percona's ingest rate, which the estimator now names instead of guessing

**This was a live defect when this page was written and has since been fixed;
the description is kept because the shape recurs and the fix is what you now
have to read the printed estimate against.**

`orchestrator/estimate.py` ended `_effective_rate` with
`_INGEST_ROWS_PER_S.get(engine, 300)`, so an engine with no entry in that table
was modelled at 300 rows/s. The table's own measured MySQL-family rates are
mariadb 150, mariadb123 110 and alisql 55 (`orchestrator/estimate.py:50`), so
the fallback was **5.45× optimistic against the slowest engine already
measured** — for an engine that builds its HNSW graph inside every INSERT,
which is the slowest shape in the study.

That contradicted the rule the comment above the table sets for itself. It is
headed "Measured ingest rates, rows/s, on a Xeon Gold 6230"
(`orchestrator/estimate.py:43`) and explains that the first guesses were
replaced because an estimate optimistic *by 3x* is worse than none
(`orchestrator/estimate.py:43-51`). The fallback was a guess, was not measured,
and was optimistic by more than the margin that argument rejects. Measured on
`main` before the fix: 273.45 rows/s for an unmeasured engine on glove-100
against MariaDB's measured 147, putting **9.10 h** of invented ingest into a
published total (102.16 h against the honest 93.06 h) with nothing printed to
say so.

**What it does now.** `_effective_rate` returns `None` where nothing has been
measured — `orchestrator/estimate.py:161` is `_INGEST_ROWS_PER_S.get(engine)`
with no default — and `estimate_load_hours` reports
`engines_without_ingest_rate` on the same terms the query half has always used
for `engines_without_query_rate`: the engine is **named, contributes nothing to
the total, and the printed block says which engines the total covers**:

```
  total ~93.1 h across 1 pass(es), 3 M value(s)
  ! No measured ingest rate for percona, so the total above is
    6 of 7 engine(s) and excludes it entirely rather than modelling
    it at a guessed rate. A smoke run supplies the figure; until then the ingest
    cost of this plan is higher than the number above, by an amount nothing here knows.
```

**So the rollout reading is the opposite of what it was.** A printed ingest
figure is now trustworthy as far as it goes, and it tells you how far that is.
What you must not do is read `~726 h` for `--profile full --resource-pass both`
as the whole plan: it is six of the seven engines, and the smoke run is what
supplies the seventh. Percona has no `_INGEST_ROWS_PER_S` entry and must not be
given one until a run measures it.

### 3. The cost envelope, and what D1 cost

The spec's own figure is roughly 2–3 hours typical and up to about 13 hours
worst case on top of the current ~2 days, for six engines on two datasets, and
requires the estimate to print the worst case (`SPEC.md:525`, supplied).

**The spec's decision point D1 is decided, and `main` got more expensive.**
`config/profiles/main.yml:110` is now `m_values: [8, 16, 32]` against an ops
`m_values` of `[16]`, so the two grids meet and §4 item 5's cross-check runs on
the profile the report is published from. There is still no crosscheck profile
in `config/profiles/`, and that is the point: the other option on the table
would have put the check on a profile nobody publishes from.

What it cost, from `orchestrator/estimate.py` and not from any prose:

| Engines | `[8, 32]` | `[8, 16, 32]` | Added |
|---|---:|---:|---:|
| the six with a measured ingest rate | 69.77 h | **93.06 h** | +23.29 h |
| the three headline engines | 51.07 h | **68.11 h** | +17.04 h |

**That first row is computed over all SEVEN engines and is nevertheless a
six-engine quantity, which is why it is labelled that way rather than "all".**
Percona Server has no measured ingest rate, so `orchestrator/estimate.py`
names it in `engines_without_ingest_rate` and it contributes nothing to either
column — the figures are unchanged by its arrival and the label is what moved.
A run of `main` loads seven engines; this table prices six of them.

An earlier revision of this page carried the two "added" figures and removed
them as unsourced, because at that commit nothing in the repo supported them.
They are now in the repo — `config/profiles/main.yml:84-87` states them with
the reasoning — and the table above is recomputed from the estimator at test
time rather than copied from that comment.

**Those are ingest hours, and they are not the number in `README.md`.** The
README profile table reports ~61 h for `main` by the hand method of
`docs/07-planning-a-run.md` §1 — flat per-engine rates over the three original
engines, no dimension or size penalty — while the figures above come from the
estimator, which carries both penalties. Each basis is named where it is used
and both are pinned by `tests/test_estimate.py`; an operator who compares the
two without reading the bases will think one of them is broken.

`main.yml`'s own comment is worth reading before you plan against either: it
records that the printed estimate is about a third **high** on this marginal
`M`, because `estimate.py` charges every `M` the same mid-grid rate while
`M=16` is the cheap middle of the grid, and about 9% **low** on the total for
the same reason. Plan against the printed figure anyway — for this decision it
is the conservative direction.

## 6. Executable guards

`tests/test_e3_rollout_doc.py` holds the checkable half of this document. It
asserts, against the repository rather than against prose:

* that every `file:line` citation above points at the thing this document says
  it points at, in both directions — a number edited here without its row, and
  a row whose anchor has moved, each fail;
* that every figure stated above about this repository is recomputed from the
  repository at test time: the measurement version, the equivalence set, the
  `client_bound` threshold and its arithmetic, both profiles' `churn_readers`,
  every driver's `repair_path` and churn-hook overrides, the estimator's
  fallback and the ratio §5 item 2 quotes, and the list of names §4 item 6 calls
  absent from `docs/05-methodology.md`;
* that the refusal in §1 is *called* and raises for versions 1 and 5 and for
  versions 1 and 2, rather than being asserted in prose;
* that all six rollout steps and all seven checklist items are present, quoted
  as the spec writes them — a step silently dropped from a runbook is the
  failure mode this guards against;
* that §3's table of items that are **not** satisfiable still has exactly the
  one row it claims, with its mechanical reason, and that the reason item 5 left
  that table still holds — `report/crosscheck.py` exists and the report renders
  section 10 — so if either were removed this page would have to say so again;
* that §5 item 3's cost table is recomputed from the estimator for both grids
  and both engine sets, and that `main`'s two M grids really do meet;
* that the two statements §4 item 5 used to call stale now say that `main`'s
  grids **do** meet, pinned by polarity rather than by sentence so that a
  differently wrong rewrite fails too, and that neither still carries the
  `[8, 32]` claim;
* that §1's PR mapping names merges that really happened, including the gap in
  the numbering, read from `git log --merges` rather than from this table;
* that none of `docs/09` §3's figures has been copied into this page, with the
  parser that looks for them proved against `docs/09` itself;
* that this document states no test count and no figure that the next merge
  would falsify.

**The prose claims are patterns, not sentences.** A guard that demands one exact
wording fires on a legitimate rewrite, and a guard that fires on correct prose
teaches the next author to delete it. So each claim this page must keep making —
that `client_bound` cannot be read off the flag, that only pgvector has a repair
path, that a green smoke is not evidence about the memory policy — is pinned by
a vocabulary that fixes the subject, the mechanism and the polarity and leaves
the wording free. `CLAIMS` in the test file carries, for each one, the
rewordings it was checked to accept and the tidied-up forms it must still
refuse, both asserted. If you reword a sentence here and the guard fires,
widen the pattern; do not restore the old sentence.

What those guards cannot do is establish anything about an engine. No server is
reachable from a test run. The archived corpus count in §1 is the one figure
they cannot always recompute: that tree is resolved from the checkout's parent,
so the guard counts it where it is reachable and otherwise checks the comment
the figure is quoted from.
