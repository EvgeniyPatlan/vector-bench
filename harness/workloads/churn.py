"""Churn: how recall and speed hold up after updates and deletes.

HNSW graphs degrade under deletion. A deleted node's edges either stay as
tombstones (wasting traversal) or are repaired (costing write time), and the
three engines make different choices. A vector index that is excellent on a
static corpus and mediocre after a week of writes is a different product from
one that holds steady, and only this workload shows the difference.

Method: delete a fraction of rows chosen deterministically, re-insert the same
vectors under fresh ids, then re-measure recall and QPS against ground truth
remapped to the new ids. Re-inserting the same vectors keeps the data
distribution — and therefore the correct answers — unchanged, so any recall drop
is graph degradation rather than a different dataset.

PR 4 adds three driver hooks to that method, because three things were being
measured by accident rather than on purpose:

* `prepare_churn()` runs once before the baseline. pgvector's autovacuum
  decided on its own schedule whether the tombstones were still in the heap
  when recall was re-measured, so the same profile could differ by the whole
  size of the effect and the record said nothing either way.
* `wait_index_synced(rows)` runs after the re-insert. An index maintained by
  another process (mongot) or behind a mutation queue (valkey) can still be
  catching up when the queries start, and "the engine is slow and inaccurate"
  is indistinguishable in a record from "we asked too early".
* `maintain()` runs after the post-churn measurement, so an engine with a
  repair path is measured twice — degraded and repaired — and the pair is
  stamped `extra.maintenance_state`. Without it, pgvector's churn number was
  whichever of the two autovacuum happened to produce.

Every hook has a no-op default, so MariaDB, AliSQL and Valkey's `maintain`
need no code: they record `extra.repair_path = "none"` and one record per
fraction, which is exactly what they recorded before.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import numpy

from ..datasets import (FILTER_CTAG, FILTER_TAG, Dataset,
                        cached_ground_truth, filter_values, recall_at_k)
from ..drivers.base import EngineDriver, IndexSpec, RowAttrs
from ..metrics.latency import LatencyCollector, Timer
from ..metrics.records import PHASE_CHURN, Record
from ..progress import Heartbeat, Progress
from .context import RunContext

# Wall clock allowed for one re-insertion. An engine that cannot write its rows
# back in half an hour is not going to, and holding the machine to find out
# costs a night. Overridable per profile.
DEFAULT_CHURN_BUDGET_S = 1800.0

DEFAULT_FRACTIONS = (0.10, 0.25)


def _measure(driver: EngineDriver, queries: numpy.ndarray, k: int,
             id_map: Dict[int, int], truth: numpy.ndarray,
             warmup: int = 20) -> Dict[str, float]:
    """Query and score against ground truth translated through `id_map`."""
    for i in range(min(warmup, len(queries))):
        driver.query(queries[i], k)

    collector = LatencyCollector(warmup=0)
    returned: List[List[int]] = []
    progress = Progress(len(queries), "churn queries", prefix=driver.name)
    collector.start()
    for q in queries:
        started = time.perf_counter()
        ids = driver.query(q, k)
        collector.add(time.perf_counter() - started)
        returned.append(ids)
        progress.step()
    collector.stop()
    progress.finish()

    # Ground truth refers to original ids; rows that were re-inserted now carry
    # new ids, so the expected set is translated rather than the results.
    translated = numpy.array(
        [[id_map.get(int(t), int(t)) for t in row[:k]] for row in truth],
        dtype=numpy.int64,
    )
    stats = collector.stats()
    out = stats.as_dict()
    out["recall_at_k"] = round(recall_at_k(returned, translated, k), 6)
    out["qps"] = round(collector.qps(clients=1), 2)
    return out


def run(ctx: RunContext, driver: EngineDriver, dataset: Dataset,
        index: IndexSpec, ef_search: int,
        fractions: Sequence[float] = DEFAULT_FRACTIONS,
        max_queries: int = 1_000, seed: int = 20260803,
        indexed_rows: Optional[int] = None,
        budget_s: float = DEFAULT_CHURN_BUDGET_S) -> None:
    """Measure recall/QPS before churn and after each churn step."""
    queries = dataset.test[:max_queries]
    # Recomputed when only part of the training set was loaded; the shipped
    # neighbours would otherwise point at rows the engine never received.
    truth = cached_ground_truth(ctx.cache_dir, dataset, ctx.k,
                               indexed_rows=indexed_rows)[:max_queries]
    driver.set_ef_search(ef_search)

    common = ctx.record_defaults(driver, dataset, index)
    n_rows = driver.count_rows()
    rng = numpy.random.default_rng(seed)

    # A non-positive or absent budget means "no cap", the same conversion
    # `filtered.run` makes and for the same reason: `ops.churn_budget_s: null`
    # reached `parse_args` as the string "None" and killed the whole ops unit
    # with SystemExit(2), taking concurrency and filtered down with churn
    # because `harness/main.py` wraps all three in one `except`. The
    # orchestrator now emits "0" for that spelling, so the value that arrives
    # here is a number and this is where it stops being a deadline.
    #
    # The DECLARED value is still what the record states. A record saying
    # `churn_budget_s: inf` would be invalid JSON and a claim no profile made.
    #
    # Note what the cap is NOT: a deadline tested before each slice cannot
    # abandon the slice already running, so a budget of B seconds can spend B
    # plus one slice. `_reinsert` has always had that overrun and the
    # filtered loop has the same shape.
    effective_budget = budget_s if budget_s > 0 else float("inf")

    # §4.2 item 1. Once, and before the baseline rather than before the first
    # delete: whatever this puts the engine into has to be in force for the
    # number the later ones are compared against, or the comparison is partly
    # a comparison of engine settings.
    driver.prepare_churn()

    baseline = _measure(driver, queries, ctx.k, {}, truth)
    print(
        f"[churn] {driver.name}: baseline recall@{ctx.k}={baseline['recall_at_k']:.4f} "
        f"qps={baseline['qps']:,.1f}"
    )
    ctx.recorder.write(Record(
        **common, phase=PHASE_CHURN, ef_search=ef_search, clients=1,
        churn_fraction=0.0, rows=n_rows,
        index_bytes=driver.index_bytes(), **baseline,
        notes="baseline before any churn",
    ))

    id_map: Dict[int, int] = {}
    next_id = n_rows
    churned_so_far = 0.0

    for fraction in fractions:
        # Each step churns the *additional* fraction, so 0.10 then 0.25 means a
        # cumulative 25%, not 35%.
        step = max(0.0, fraction - churned_so_far)
        if step <= 0:
            continue
        count = int(round(step * n_rows))
        if count == 0:
            continue

        # Only rows still present under their current id are eligible.
        live_original = [i for i in range(n_rows) if i not in id_map]
        victims = rng.choice(len(live_original), size=min(count, len(live_original)),
                             replace=False)
        original_ids = [live_original[int(v)] for v in victims]
        current_ids = [id_map.get(o, o) for o in original_ids]
        vectors = dataset.train[original_ids]
        # Both filter columns, each row keeping its OWN values, which is the
        # fix for a defect that was latent only because of statement order.
        # This used to be `assign_tags(n_rows)[original_ids]` -- one array --
        # handed to `RowAttrs.coerce`, which mirrors a bare array into `ctag`.
        # Every re-inserted row therefore carried `ctag = tag` instead of the
        # cluster id its vector belongs to, so a `ctag`-filtered measurement
        # taken after a churn would have been scored against a ground truth
        # computed from the real clustering. Nothing did that, because
        # `main.py` runs filtered (`:305`) before churn (`:317`) in fixed
        # source order and `build.run` reloads the schema per (M, build_mode);
        # the failure signature would have been the two filter modes agreeing,
        # which reads as a finding rather than as a bug.
        #
        # `filter_values` for both columns, not `assign_tags` for one and
        # `cached_ctags` for the other, for the reason `build.run` gives at
        # the call site that loads them: the filtered workload computes its
        # exact ground truth from this same function keyed on the same
        # (dataset, rows), so the values in the engine and the values the
        # yardstick was built from cannot be two arrays that merely agree
        # today. `n_rows` is `driver.count_rows()`, which is the number
        # `build.run` clustered over (`len(vectors)`) and the number
        # `main.py` passes as `indexed_rows` -- clustering a different row
        # count relabels the rows, measured at 497 of 500.
        attrs = RowAttrs(
            tag=filter_values(ctx.cache_dir, dataset, FILTER_TAG,
                              indexed_rows=n_rows),
            ctag=filter_values(ctx.cache_dir, dataset, FILTER_CTAG,
                               indexed_rows=n_rows),
        )[original_ids]

        print(f"[churn] {driver.name}: deleting and re-inserting {count:,} rows "
              f"(cumulative {fraction:.0%})")

        with Heartbeat(f"deleting {len(current_ids):,} rows", prefix=driver.name):
            with Timer() as delete_timer:
                driver.delete_ids(current_ids)

        new_ids = list(range(next_id, next_id + len(original_ids)))
        next_id += len(original_ids)
        inserted, insert_seconds = _reinsert(
            driver, new_ids, vectors, attrs, effective_budget)

        if inserted < len(new_ids):
            # Bounded rather than open-ended, because an engine can take the
            # write and never finish it. Valkey loaded this corpus at 38,845
            # rows/s into an unindexed keyspace and then managed 364 of 99,000
            # back into the populated index before writes stopped entirely,
            # with the server responsive and its mutation queue empty. Two runs
            # were lost waiting on it.
            #
            # What it achieved is recorded as the measurement. Recall and
            # throughput are not: the corpus is missing rows, so any figure
            # taken now describes a state that is neither the baseline nor a
            # churned corpus, and it would be read as though it were.
            rate = inserted / insert_seconds if insert_seconds else 0.0
            print(f"[churn] {driver.name}: re-insert did not complete — "
                  f"{inserted:,} of {len(new_ids):,} rows in "
                  f"{insert_seconds:.0f}s ({rate:,.1f} rows/s); "
                  f"budget was {budget_s:.0f}s")
            ctx.recorder.write(Record(
                **common, phase=PHASE_CHURN, ef_search=ef_search, clients=1,
                churn_fraction=fraction, rows=driver.count_rows(),
                index_bytes=driver.index_bytes(),
                notes="re-insert did not complete within the churn budget",
                extra={
                    "rows_churned": count,
                    "delete_seconds": round(delete_timer.elapsed, 3),
                    "insert_seconds": round(insert_seconds, 3),
                    "reinsert_completed": False,
                    "rows_reinserted": inserted,
                    "rows_expected": len(new_ids),
                    "reinsert_rows_per_s": round(rate, 3),
                    # The DECLARED budget, not the deadline `_reinsert` was
                    # given: a record saying `inf` would be invalid JSON and
                    # a claim no profile made. This is also the only record
                    # that states it, because it is the only one the budget
                    # changed.
                    "churn_budget_s": budget_s,
                    "repair_path": driver.repair_path,
                    # No `index_sync_s`, and the absence is the honest answer
                    # rather than an oversight: the corpus is missing rows, so
                    # `wait_index_synced(n_rows)` would wait for a count that
                    # is not coming and spend its whole timeout -- 900 s for
                    # Valkey, an hour for mongot -- to learn nothing about an
                    # index nobody is going to measure. A key that appeared
                    # here with a 0.0 in it would read as "the index was
                    # ready".
                    **_write_shape(driver),
                },
            ))
            # Every later fraction builds on this one, so there is nothing
            # meaningful left to measure.
            break

        for original, new in zip(original_ids, new_ids):
            id_map[original] = new
        churned_so_far = fraction

        # §4.2 item 4. Before the measurement, because the point of waiting is
        # that the numbers after it describe an index that has the writes.
        with Heartbeat(f"waiting for the index to reflect {n_rows:,} rows",
                       prefix=driver.name):
            synced_s = driver.wait_index_synced(n_rows)

        step_extra = {
            "rows_churned": count,
            "delete_seconds": round(delete_timer.elapsed, 3),
            "insert_seconds": round(insert_seconds, 3),
            "reinsert_completed": True,
            "reinsert_rows_per_s": round(
                len(new_ids) / insert_seconds, 3) if insert_seconds else 0.0,
            "index_sync_s": round(synced_s, 3),
            "repair_path": driver.repair_path,
            **_write_shape(driver),
            **_sync_shape(driver),
        }

        after = _measure(driver, queries, ctx.k, id_map, truth)
        drop = baseline["recall_at_k"] - after["recall_at_k"]
        print(
            f"[churn] {driver.name}: after {fraction:.0%} churn "
            f"recall@{ctx.k}={after['recall_at_k']:.4f} "
            f"(Δ{-drop:+.4f}) qps={after['qps']:,.1f}"
        )

        # §4.2 item 5. Written BEFORE `maintain()` runs, not held back until
        # the pair is complete: this is a finished measurement, and a repair
        # path that fails -- pgvector's VACUUM hitting a lock timeout, mongot
        # going away -- must not take it with it. The engines with no repair
        # path write this record and nothing else, which is what they wrote
        # before PR 4 plus the three keys above.
        ctx.recorder.write(Record(
            **common, phase=PHASE_CHURN, ef_search=ef_search, clients=1,
            churn_fraction=fraction, rows=driver.count_rows(),
            index_bytes=driver.index_bytes(), **after,
            extra={**step_extra,
                   "maintenance_state": "before",
                   "recall_drop_vs_baseline": round(drop, 6)},
        ))

        # §4.2 item 6. The call is timed whether or not it turns out to have
        # done anything, because "did it run a repair path?" is answerable
        # only from the return value and the clock has to have been started
        # before the call to time it at all.
        with Timer() as maintain_timer:
            what = driver.maintain()
        _record_maintenance(ctx, driver, common, queries, truth, ef_search,
                            fraction, id_map, baseline, after, what,
                            maintain_timer.elapsed, step_extra)



def _record_maintenance(ctx: RunContext, driver: EngineDriver,
                        common: Dict[str, object], queries: numpy.ndarray,
                        truth: numpy.ndarray, ef_search: int,
                        fraction: float, id_map: Dict[int, int],
                        baseline: Dict[str, float],
                        degraded: Dict[str, float],
                        what: object, maintenance_s: float,
                        step_extra: Dict[str, object]) -> None:
    """Measure and record the repaired state, when there was a repair.

    `what` is `maintain()`'s return value: `Optional[str]` per PR 4 §4.1, a
    short description of what ran. **The gate is that type and not
    truthiness**, and the difference is a published number rather than a
    style: `tests/fake_driver.py`'s `maintain()` predates this hook and
    returns the COUNT of rows it reclaimed -- pinned at
    `tests/test_fake_driver.py:274` and `tests/test_t3b_drivers.py:1999` --
    so a truthy check would write `extra.maintenance: 200` under a key a
    reader reads as prose, and would write an "after" record for a driver
    that never claimed to have an "after".

    A second record rather than two columns on the first, because the two
    states are two measurements of the same corpus taken the same way, and
    this repo's one flat record schema is what makes two measurements
    comparable: `maintenance_state` names which state each describes, and the
    raw JSONL answers "what did the repair buy?" by subtracting one record
    from the other.

    **The report cannot yet tell the pair apart, and that is published-wrong
    rather than merely incomplete.** `maintenance`, `maintenance_state`,
    `repair_path` and `index_sync` have **zero** occurrences anywhere in
    `report/`, so nothing there names the state, and all three churn artifacts
    consume both records as peers. Each figure below says what it reproduces
    over, because two of these quantities are deterministic and one is a ratio
    of wall clocks; the structural claims held in 9 of 9 runs of the identical
    configuration on `tests/fake_driver.py`, two fractions:

      * `render._churn_table` emits two rows per fraction whose four identity
        columns (engine, dataset, storage, churn fraction) are byte-identical
        -- the fraction renders `'10%'` twice -- carrying two different
        recalls (0.8960, 0.9980) and two different index sizes (275.0 KiB,
        250.0 KiB). Both of those pairs are exact in 9 of 9: the fake's recall
        comes off a seeded generator and its `index_bytes` is arithmetic on
        the row count. `identities distinguishable: False`, 9 of 9. The only
        two readings a reader has are "a noisy measurement" and "a duplicate
        row", and both are false.
      * `charts.churn` plots x = [0.0, 0.1, 0.1, 0.25, 0.25] as one series --
        9 of 9 -- so each fraction is a vertical segment.
      * `charts.churn_retention`, which feeds `churn_impact`, divides each
        post-churn QPS by the series' own pre-churn QPS and returns
        `sorted(points)`, so the pair is duplicated at every x AND before and
        after cannot be told apart by position: x = [0.1, 0.1, 0.25, 0.25] in
        9 of 9. The ratio can exceed 1.0 on an axis labelled "QPS as a
        fraction of pre-churn (higher is better ↑)" -- 10 of 36 points over 9
        runs, at least one point above 1.0 in 5 of those 9.

    **No mechanism is claimed for the ratios above 1.0, and an earlier
    revision of this docstring claimed the wrong one.** It said a vacuumed
    index outruns its own baseline. The measurement refutes it: in 2 of the 9
    runs BOTH points at a fraction exceeded 1.0, so the un-maintained,
    tombstone-laden "before" measurement outran the baseline too -- and this
    fake runs no vacuum at all, because `maintain()` here only drops rows from
    an array. It is jitter in a **50**-query QPS figure, and the figure's
    spread says so: 6,028 to 16,811 QPS across those runs, a factor of 2.8,
    against recall that did not move in the last digit. Fifty and not the 70
    an earlier revision said: `tiny_dataset().test` holds 50 rows, so
    `queries` is 50 whether `max_queries` arrives as 50 or as its 1,000
    default; `_measure` runs its 20 warm-up queries BEFORE
    `collector.start()`, so they are never collected; and `qps(clients=1)` is
    `len(self._samples) / measured_seconds`. 70 is the per-measurement
    `driver.query` CALL count -- `N_QUERIES + min(20, N_QUERIES)`, which is
    what the sequence test asserts on -- and there is no configuration in
    which it is the QPS sample size. The smaller n is the stronger reading:
    jitter at 50 samples explains a factor of 2.8 more comfortably than at
    70. Whether a real pgvector's
    post-`VACUUM (ANALYZE)` throughput can legitimately beat its own pre-churn
    baseline is plausible and **unmeasured** -- no live engine has been run
    for PR 4 -- so it is not asserted here.

    So this record is written for T4b, which owns `report/*`: the table needs
    an identity column for `maintenance_state`, and the two charts need the
    pair split into separate series (or the `"after"` records excluded from
    the retention ratio). Everything needed is in the record --
    `maintenance_state`, `maintenance`, `maintenance_s`, `repair_path` -- and
    none of it is read yet. `docs/05-methodology.md` is accurate as written
    because it tells a reader to read the `"before"` RECORDS, which is true of
    the JSONL and not of the rendered report.
    """
    if not isinstance(what, str) or not what:
        return
    repaired = _measure(driver, queries, ctx.k, id_map, truth)
    print(
        f"[churn] {driver.name}: {what} took {maintenance_s:.1f}s; "
        f"recall@{ctx.k}={repaired['recall_at_k']:.4f} "
        f"(was {degraded['recall_at_k']:.4f} before it, "
        f"{baseline['recall_at_k']:.4f} at the baseline) "
        f"qps={repaired['qps']:,.1f}"
    )
    ctx.recorder.write(Record(
        **common, phase=PHASE_CHURN, ef_search=ef_search, clients=1,
        churn_fraction=fraction, rows=driver.count_rows(),
        index_bytes=driver.index_bytes(), **repaired,
        extra={**step_extra,
               "maintenance_state": "after",
               "maintenance": what,
               "maintenance_s": round(maintenance_s, 3),
               "recall_drop_vs_baseline": round(
                   baseline["recall_at_k"] - repaired["recall_at_k"], 6)},
    ))


def _sync_shape(driver) -> Dict[str, object]:
    """Why `index_sync_s` came out the way it did, when the driver says.

    Optional, read through `getattr` exactly as `_write_shape` below reads
    Valkey's write batch: the engines that maintain their index on the write
    path have nothing to explain and carry no key.

    It exists because a seconds-valued hook returning 0.0 is ambiguous in the
    one direction that matters. `index_sync_s: 0.0` is what a synchronous
    engine records, what a hook nobody implemented records, and what a mongot
    that was already caught up records -- and H1's acceptance criterion is
    "non-zero or a reason documented", which a bare zero cannot satisfy.
    """
    note = getattr(driver, "index_sync_note", None)
    return {"index_sync_note": note} if note else {}


def _write_shape(driver) -> Dict[str, object]:
    """How the driver had to write, when that is not how it normally writes.

    Valkey re-inserted 99,000 rows at 86 rows/s against a load that ran at
    30,195, and almost none of that gap is the HNSW graph: pipelined writes
    into an indexed prefix do not return at all, so the re-insert had to fall
    back to one command per round trip while the load pipelined a thousand.
    Comparing those two rates as though they measured the same operation would
    attribute to the index a cost that belongs to the write shape, so the shape
    travels with the number.
    """
    used = getattr(driver, "write_batch_used", None)
    if used is None:
        return {}
    shape = {"write_batch_rows": used}
    if used < getattr(driver, "load_batch_rows", used):
        shape["write_batch_reduced"] = True
    return shape


def _reinsert(driver: EngineDriver, ids: Sequence[int], vectors, attrs,
              budget_s: float):
    """Re-insert in slices, stopping if the budget runs out.

    One blocking call cannot tell a slow write path from a stalled one, and
    cannot be abandoned. Slicing gives both: progress is visible, and a path
    that is not going to finish releases the machine instead of holding it
    until someone notices.

    `budget_s` is a deadline tested before each slice, so it cannot abandon
    the slice already running: B seconds of budget can spend B plus one
    slice. `run` converts a non-positive budget to `inf` before calling this,
    which is how "no cap" is spelled; nothing here special-cases it.

    `attrs` is coerced once on entry rather than per slice, so the write path
    below only ever sees both columns and every slice comes from one array
    pair. **A bare array still arrives here and is still mirrored** --
    `RowAttrs.coerce` gives such a caller `ctag = tag` -- and that path is
    reachable: `tests/test_config.py` and `tests/test_t3b_drivers.py` call
    this function directly with an array. `run` above does not use it; it
    builds both columns from `filter_values`, which is the fix for the churn
    mirror. What is left is the bare-array convenience for a diagnostic
    caller that has no cache directory to compute a cluster assignment from,
    and it is spelled at the call site rather than hidden inside five drivers
    so that a caller mirroring a column can see that it did.

    Returns (rows actually written, seconds spent).
    """
    attrs = RowAttrs.coerce(attrs)
    total = len(ids)
    # Fifty slices: enough that the budget is honoured to within 2% of itself,
    # few enough that the slicing costs nothing against the write.
    chunk = max(1, total // 50)
    started = time.time()
    written = 0
    with Heartbeat(f"re-inserting {total:,} rows", prefix=driver.name):
        for begin in range(0, total, chunk):
            if time.time() - started > budget_s:
                break
            end = min(begin + chunk, total)
            try:
                # Sliced through `RowAttrs.__getitem__`, which slices both
                # columns together: slicing them independently is how a row
                # ends up with another row's cluster id, and Valkey slices
                # these slices again.
                driver.insert_rows(ids[begin:end], vectors[begin:end],
                                   attrs[begin:end])
            except Exception as exc:
                # A write that times out or a connection that dies ends the
                # re-insertion; it does not end the run. What was written is
                # still a measurement, and the caller records it.
                print(f"[churn] {driver.name}: re-insert stopped after "
                      f"{written:,} rows: {type(exc).__name__}: {exc}")
                break
            written = end
    return written, time.time() - started
