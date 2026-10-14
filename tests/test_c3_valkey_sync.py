"""C3: the Valkey index-sync wait, once the caller's direction is passed in.

No engine is started. Every test drives the **real**
`_wait_for_mutations` / `wait_index_synced` against a scripted `FT.INFO` and a
clock that only `sleep` moves, so a poll count maps to an exact elapsed time
(2 s per poll) and both are reproducible rather than wall-clock.

**What was wrong.** The shared wait's document condition was
`docs <= expected_docs`, written for `delete_ids`, where the indexed count
falls TO the target. `wait_index_synced` delegated to it after a *re-insert*,
where the count rises to the target -- so the comparison was already true at
the first poll and the document half was dead. Measured against the pristine
file at 97b31c9 with the queue drained: **1 poll** with the index 200 documents
short of its target, and **1 poll** with the index holding nothing at all. The
mutation queue was the only signal left guarding a published post-churn recall
figure.

**Why not `>=`.** It repairs the direction and keeps the trap. The queue has
been observed drained while this index reported 481 documents MORE than the
keyspace held, and "at least the expected count" is satisfied by stale entries
while the new rows are still unindexed: measured, that form ends the wait at
**1 poll** holding **2,481** documents against an expected 2,000, where
equality waits the **3 polls** it takes to reach 2,000. So the insert path
waits for equality, with a stopped-moving escape (`DOCS_SETTLED_POLLS`) so an
index that converges two documents short ends at **4 polls** saying so instead
of at **450 polls** and the 900 s timeout.

**A fix can invalidate its own test's stated consequence, and two did here.**
Both were caught by re-measuring the docstrings rather than by any assertion,
because in both cases the assertion still worked and only the sentence
explaining it had gone stale:

* the escape floor (`DOCS_SETTLED_FLOOR_S`) made "the mutant ends at 3 polls
  instead of 4" untrue of `DOCS_SETTLED_POLLS` -- 16 polls either way, because
  the floor dominates the settle count;
* the `ended_by`-keyed note lead made "byte-identical to a stubbed wait"
  untrue of the `_observed_sync_phrase` mutant, by distinguishing that case a
  second way.

Anyone adding a third condition to this wait should expect to do it again: a
poll count is a property of the whole loop, so every new condition can change
what an old mutant produces. Quote values a guard reads directly -- the unit
return of `_docs_ready`, a field of `index_sync_detail` -- in preference to a
poll count, where the two would say the same thing.

**Mutation discipline.** Every guard below names the regression it catches.
Each was injected as a rewrite of the *source* in a fresh temporary copy of the
package -- never a rebind of a module attribute, because the expectation
reaches the wait as a function DEFAULT captured in `__defaults__` at def time,
and never an in-place rewrite, because CPython invalidates a `.pyc` on
(whole-second mtime, size). Each mutant was read back off disk before it was
measured, and the honest and mutant values in these docstrings came out of
`index_sync_detail` and the FT.INFO script's own poll counter.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple
from unittest import mock

import pytest

import numpy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.drivers import valkey as valkey_mod                      # noqa: E402
from harness.drivers.base import ConnectionSpec                       # noqa: E402

AT_MOST = valkey_mod.DOCS_AT_MOST
EQUAL = valkey_mod.DOCS_EQUAL
TIMEOUT = valkey_mod.INDEX_SYNC_TIMEOUT_S
#: An elapsed time past the escape's floor, for the cases that are about the
#: comparison and not about the floor. Derived from the constant rather than
#: written as 31.0, so raising the floor cannot leave these reading as
#: "past the floor" when they are not.
PAST_FLOOR = valkey_mod.DOCS_SETTLED_FLOOR_S + 1.0


class Clock:
    """A clock only `sleep` moves: elapsed is 2 s per poll, exactly."""

    def __init__(self, start: float = 1000.0):
        self.now = start

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class Wait:
    """One run of the real wait, with the readings it was driven by.

    `queue` and `docs` are the only fixture-supplied inputs: lists of FT.INFO
    readings, the shorter of which repeats its last value forever. `polls`,
    `elapsed`, `detail` and `note` are all read back out of the driver.
    """

    def __init__(self, queue: List[Any], docs: Optional[List[Any]]):
        self.driver = valkey_mod.ValkeyDriver(ConnectionSpec(host="h",
                                                             port=6379))
        self.polls = 0
        self.driver._ft_info = self._ft_info
        self._queue = queue
        self._docs = docs

    def _ft_info(self) -> Dict[str, Any]:
        i = self.polls
        self.polls += 1
        info: Dict[str, Any] = {}
        if self._queue is not None:
            info["mutation_queue_size"] = self._queue[min(i,
                                                          len(self._queue) - 1)]
        if self._docs is not None:
            info["num_docs"] = self._docs[min(i, len(self._docs) - 1)]
        return info

    def wait(self, expected=None, timeout_s=60.0, **kw) -> float:
        with mock.patch.object(valkey_mod, "time", Clock()):
            return self.driver._wait_for_mutations(expected_docs=expected,
                                                   timeout_s=timeout_s, **kw)

    def synced(self, expected_rows: int) -> float:
        with mock.patch.object(valkey_mod, "time", Clock()):
            return self.driver.wait_index_synced(expected_rows)

    @property
    def detail(self) -> Dict[str, Any]:
        return self.driver.index_sync_detail

    @property
    def note(self) -> str:
        return self.driver.index_sync_note


# The reading scripts, named once so a scenario cannot drift between tests.
# `RISING` is a re-insert catching up; `EMPTY` is the same from nothing;
# `OVER` is the 481-document overcount draining away; `PLATEAU` converges two
# documents short; `FALLING` is a delete reaching its target and `OVERSHOOT` a
# delete whose count lands below it (which happens whenever some of the ids
# handed to `delete_ids` were not in the keyspace).
RISING = [1800, 1860, 1930, 2000]
EMPTY = [0, 700, 1400, 2000]
OVER = [2481, 2481, 2000]
PLATEAU = [1995, 1998, 1998, 1998]
FALLING = [2000, 1900, 1800]
OVERSHOOT = [2000, 1900, 1750]


class _Conn:
    """Enough of a write connection for `delete_ids`."""

    def __init__(self):
        self.deleted = 0

    def delete(self, *keys):
        self.deleted += len(keys)

    def close(self):
        pass


# ---------------------------------------------------------------------------
# The shape: the direction is passed, not guessed
# ---------------------------------------------------------------------------

class TestTheDirectionIsPassedNotGuessed:
    """Both callers of the shared wait say what they expect the count to do.

    A flipped comparison would repair one path and break the other: `>=` on
    the delete path is satisfied at the first poll, because `delete_ids`
    passes `expected_docs = before - len(keys)` = 1,800 while the index still
    reports 2,000. Measured on the pristine file with that one character
    changed: the delete scenario ends at **1 poll** instead of **3**.
    """

    def test_the_insert_path_names_equality_in_the_call(self):
        """Honest: `docs_expectation="equal"` reaches the wait, beside the
            row count as a plain int and the explicit timeout.
        Mutant (the shipped call, which named no direction and so got the
            delete path's `at_most`): `wait_index_synced` against an index
            200 documents short returns after **1 poll** holding **1,800**
            documents, against **4 polls** holding **2,000** honestly.
        """
        calls: List[Dict[str, Any]] = []
        driver = valkey_mod.ValkeyDriver(ConnectionSpec(host="h", port=6379))
        driver._wait_for_mutations = lambda **kw: (calls.append(kw) or 7.5)

        assert driver.wait_index_synced(4321) == 7.5
        assert calls == [{"expected_docs": 4321, "timeout_s": TIMEOUT,
                          "docs_expectation": EQUAL}]
        # The value, not just the key: "equal" and "at_most" are both truthy
        # strings and a membership test would pass for either.
        assert calls[0]["docs_expectation"] == EQUAL != AT_MOST
        assert type(calls[0]["expected_docs"]) is int

    def test_the_delete_path_names_at_most_in_the_call(self):
        """The delete path's intent is at its own call site rather than left
        to the default, so the next reader cannot repair the comparison for
        the insert path and silently unguard this one.

        Honest: `[{"expected_docs": 1800, "docs_expectation": "at_most"}]`.
        Mutant (the keyword dropped from the call): `[{"expected_docs":
            1800}]`, measured -- behaviourally identical today, since
            `at_most` IS the default, which is exactly why this is asserted on
            the call and not on a poll count.
        """
        calls: List[Dict[str, Any]] = []
        driver = valkey_mod.ValkeyDriver(ConnectionSpec(host="h", port=6379))
        conn = _Conn()
        driver.count_rows = lambda: 2000
        driver._write_connection = lambda timeout_s=None: conn
        driver._wait_for_mutations = lambda **kw: (calls.append(kw) or 1.0)

        driver.delete_ids(list(range(200)))
        assert conn.deleted == 200
        assert calls == [{"expected_docs": 1800,
                          "docs_expectation": AT_MOST}]
        assert calls[0]["docs_expectation"] == AT_MOST != EQUAL

    def test_the_default_is_the_delete_paths_comparison(self):
        """`scripts/verify-valkey-churn.py` calls this wait with neither a
        direction nor an expected count, and two existing tests call it with
        an expected count and no direction. The default therefore has to stay
        the comparison they were written against.

        Honest: a bare `expected_docs=2000` against an index holding 1,800
            ends at **1 poll**, 0.000 s -- the delete path's semantics,
            unchanged, and `docs_expectation` reads "at_most".
        Mutant (the parameter's default flipped to `equal`, which is the shape
            that would need no third argument at the insert call site):
            **4 polls**, 6.000 s, and `__defaults__` reads
            `(None, 900.0, 'equal')` -- a comparison no existing caller asked
            for, in a wait a lab script also drives.
        """
        run = Wait([0], RISING)
        assert run.wait(expected=2000) == 0.0
        assert run.polls == 1
        assert run.detail["docs_expectation"] == AT_MOST
        # The default is what the lab script and the two existing callers get.
        assert valkey_mod.ValkeyDriver._wait_for_mutations.__defaults__ == \
            (None, 900.0, AT_MOST)


# ---------------------------------------------------------------------------
# The repair: the count binds on an insert
# ---------------------------------------------------------------------------

class TestTheDocumentCountBindsOnAnInsert:
    """What the whole task is for. Each honest value is a poll count under the
    stubbed clock, which is deterministic; the elapsed time is quoted only
    because that clock makes it 2 s per poll exactly."""

    def test_an_index_short_of_the_target_is_waited_out(self):
        """Honest: 4 polls, 6.000 s, final `num_docs` 2,000.
        Mutant (`docs_expectation=DOCS_AT_MOST`, i.e. the shipped delegation):
            1 poll, 0.000 s, final `num_docs` 1,800 -- the wait returns with
            the index 200 documents short, and `index_sync_s` publishes 0.0.
        """
        run = Wait([0], RISING)
        assert run.synced(2000) == 6.000
        assert run.polls == 4
        assert run.detail["observed_docs"] == 2000
        assert run.detail["ended_by"] == "num_docs reached the expected count"

    def test_an_index_holding_nothing_is_not_read_as_synced(self):
        """The sharpest form of the same defect: an index that has absorbed
        none of the re-insert at all.

        Honest: 4 polls, final `num_docs` 2,000.
        Mutant (the shipped delegation): 1 poll, final `num_docs` **0**.
        """
        run = Wait([0], EMPTY)
        assert run.synced(2000) == 6.000
        assert run.polls == 4
        assert run.detail["observed_docs"] == 2000

    def test_an_overcount_is_not_accepted_as_at_least_the_expected(self):
        """Why `>=` is not the repair.

        Honest (equality): 3 polls, final `num_docs` 2,000.
        Mutant (`if docs >= expected`): 1 poll, final `num_docs` **2,481** --
            the queue drained while the index held 481 documents the keyspace
            did not, which is the incident this file's own docstring is built
            around, and `>=` calls it synced.
        """
        run = Wait([0], OVER)
        assert run.synced(2000) == 4.000
        assert run.polls == 3
        assert run.detail["observed_docs"] == 2000


# ---------------------------------------------------------------------------
# The constraint: the delete path must not move
# ---------------------------------------------------------------------------

class TestTheDeletePathDidNotMove:
    """Both values measured against the pristine file at 97b31c9 and against
    this one, through `_wait_for_mutations` and end to end through
    `delete_ids`: **3 polls, 4.000 s, 200 keys deleted**, identical."""

    def test_the_count_falling_to_the_target_ends_where_it_always_did(self):
        """Honest, pristine and now: 3 polls, 4.000 s, final count 1,800.
        Mutant (`docs_expectation=DOCS_EQUAL` on the delete call): measured at
            3 polls and 4.000 s as well. This scenario -- the one the task's
            own table names -- is the case where `<=` and `==` agree, so it
            cannot detect a tightened delete comparison. That is what the
            overshoot test below is for.
        """
        run = Wait([0], FALLING)
        assert run.wait(expected=1800, docs_expectation=AT_MOST) == 4.000
        assert run.polls == 3
        assert run.detail["observed_docs"] == 1800

    def test_a_count_that_lands_below_the_target_still_ends_the_wait(self):
        """`delete_ids` expects `before - len(keys)`, so an id that was not in
        the keyspace leaves the real count BELOW the expectation. `<=` is what
        makes that a finished delete rather than a hung wait.

        Honest: 3 polls, 4.000 s, final `num_docs` 1,750 against 1,800.
        Mutant (the delete comparison tightened to `==`): **30 polls** and
            **60.000 s**, the whole timeout, reporting `timed_out` True.
        """
        run = Wait([0], OVERSHOOT)
        assert run.wait(expected=1800, docs_expectation=AT_MOST) == 4.000
        assert run.polls == 3
        assert run.detail["observed_docs"] == 1750
        assert run.detail["timed_out"] is False

    def test_the_delete_call_site_still_waits_before_returning(self):
        """End to end, with the connection faked and the clock stubbed."""
        run = Wait([0], FALLING)
        conn = _Conn()
        run.driver.count_rows = lambda: 2000
        run.driver._write_connection = lambda timeout_s=None: conn
        with mock.patch.object(valkey_mod, "time", Clock()):
            run.driver.delete_ids(list(range(200)))
        assert (conn.deleted, run.polls) == (200, 3)
        assert run.detail["expected_docs"] == 1800
        assert run.detail["docs_expectation"] == AT_MOST


# ---------------------------------------------------------------------------
# The escape: equality must not become a guaranteed timeout
# ---------------------------------------------------------------------------

class TestAnIndexThatNeverLandsExactlyDoesNotBecomeATimeout:
    """Equality is the only honest insert condition and it has a failure mode:
    an index that converges near the target and stays there would wait out
    INDEX_SYNC_TIMEOUT_S, and every Valkey churn would publish
    `index_sync_s: 900.0` with "timed out" beside it. A count that has stopped
    moving ends the wait instead -- off target, and saying so."""

    def test_a_count_that_stops_moving_ends_the_wait_off_target(self):
        """Honest: 16 polls, 30.000 s, final `num_docs` 1,998, `ended_by`
            naming the stop and the note reading "(2 short)". 16 polls
            because the escape is floored at `DOCS_SETTLED_FLOOR_S`.
        Mutant (the stopped-moving escape removed, i.e. `return False`):
            **450 polls**, **900.000 s** and `timed_out` True -- the
            measurement degrades to "the index never caught up" for an index
            that was two documents off.
        """
        run = Wait([0], PLATEAU)
        assert run.synced(2000) == 30.000
        assert run.polls == 16
        assert run.detail["observed_docs"] == 1998
        assert run.detail["timed_out"] is False
        assert run.detail["ended_by"] == \
            "num_docs stopped moving while still off target"

    def test_a_few_flat_readings_are_not_treated_as_a_finished_index(self):
        """The escape's floor, and the attack it answers.

        Three equal readings is four seconds at a two-second poll, and the
        only thing separating that from a finished index is `queued == 0` --
        the signal this file documents as unreliable, since the 481-document
        incident IS "the queue drained while the index disagreed". A GC or
        CPU-contention stall mid-re-insert produces exactly that shape.

        Honest: readings 1,800 / 1,800 / 1,800 / 1,900 / 2,000 end at **5
            polls, 8.000 s, observed 2,000, "num_docs reached the expected
            count"** -- the stall is waited out and the real answer found.
        Mutant (the `elapsed_s >= DOCS_SETTLED_FLOOR_S` conjunct removed):
            **3 polls, 4.000 s, observed 1,800**, `ended_by` "stopped moving
            while still off target", note "(200 short)" -- a converging index
            called finished 200 documents early, four seconds in.
        """
        run = Wait([0], [1800, 1800, 1800, 1900, 2000])
        assert run.synced(2000) == 8.000
        assert run.polls == 5
        assert run.detail["observed_docs"] == 2000
        assert run.detail["ended_by"] == "num_docs reached the expected count"

    def test_an_index_holding_nothing_still_ends_the_wait_but_not_early(self):
        """The floor must not become a hang: a count that is flat at zero is
        still a stopped count, and 900 s of it tells nobody anything.

        Honest: 16 polls, 30.000 s, observed 0, note "(2,000 short)".
        Mutant (floor removed): 3 polls, 4.000 s -- four seconds, and the
            record would carry `index_sync_s: 4.0` for an index that had
            absorbed nothing at all.
        """
        run = Wait([0], [0])
        assert run.synced(2000) == 30.000
        assert run.polls == 16
        assert run.detail["observed_docs"] == 0
        assert "(2,000 short)" in run.note, run.note

    def test_an_off_target_wait_does_not_lead_with_the_word_settled(self):
        """What a skim takes away. The full sentence is honest either way,
        but "settled in 30.0s" beside a gap is what a reader carries into a
        recall table.

        Honest: the note begins "stopped off target after 30.0s" and says
            what a premature exit looks like.
        Mutant (one unconditional lead, this round's first form): "settled in
            30.0s: num_docs=1,998 … (2 short)" -- accurate in full, and read
            as a pass.
        """
        run = Wait([0], PLATEAU)
        run.synced(2000)
        note = run.note
        assert note.startswith("stopped off target after 30.0s"), note
        assert "settled in" not in note, note
        assert "premature exit" in note, note
        # And the exact case still leads with "settled", or the word would
        # have become meaningless rather than reserved.
        exact = Wait([0], RISING)
        exact.synced(2000)
        assert exact.note.startswith("settled in 6.0s"), exact.note

    def test_the_timeout_branch_still_reports_a_timeout(self):
        """The shared wait signals its timeout only on stderr, so the hook
        compares the returned elapsed against the ceiling it passed in. A
        queue that never drains is the case that must still reach it.

        Honest: 450 polls, 900.000 s, `timed_out` True, and a note saying the
            wait gave up and that what follows measures an index still
            catching up.
        Mutant (the `waited >= INDEX_SYNC_TIMEOUT_S` comparison removed):
            the same 900.000 s comes back under "settled in 900.0s:
            num_docs=2,000 against an expected 2,000 (exactly on target) after
            450 poll(s)" -- a wait that gave up, published as one that
            finished.
        """
        run = Wait([5], [2000])
        assert run.synced(2000) == TIMEOUT
        assert run.polls == 450
        assert run.detail["timed_out"] is True
        assert "gave up" in run.note and "still catching up" in run.note
        assert "settled in" not in run.note


# ---------------------------------------------------------------------------
# The evidence: what the wait actually observed
# ---------------------------------------------------------------------------

class TestTheWaitPublishesWhatItObserved:
    """The part a live run has to settle. No fake can say whether a real index
    converges to the expected count, stops a few short, or comes back over, so
    the wait records the readings and `churn.run` carries the rendered note
    into `extra.index_sync_note`."""

    def test_the_detail_carries_the_facts_a_reader_needs(self):
        """Honest: the final count, the expected count, the queue reading, the
            poll count, the expectation, the condition that ended the wait and
            the binding signal.
        Mutant (the assignment on the success path deleted, which is the
            shipped state): `None`, and the note loses its whole observation
            clause -- "settled in 6.0s. The insert path waits for ..." with
            not one reading in it, measured.
        """
        run = Wait([0], RISING)
        run.synced(2000)
        assert run.detail == {"expected_docs": 2000, "observed_docs": 2000,
                              "queue_size": 0, "polls": 4,
                              "docs_expectation": EQUAL,
                              "ended_by": "num_docs reached the expected count",
                              "binding_signal": "the document count",
                              "timed_out": False}

    def test_the_note_quotes_the_observed_count_and_the_poll_count(self):
        """Anchored on delimiters: "2,000" is a substring of "12,000" and
        "4 poll(s)" of "14 poll(s)", and both shapes have bitten this repo.

        Honest: a note naming num_docs=2,000, the target, the poll count and
            the binding signal, and still explaining why the insert path does
            not use the delete path's comparison.
        Mutant (the old note, which asserted the count half "binds only while
            the index is OVERCOUNTING ... and otherwise the queue is the only
            signal left"): a sentence that this change makes false.
        """
        run = Wait([0], RISING)
        run.synced(2000)
        note = run.note
        assert "settled in 6.0s" in note, note
        assert "num_docs=2,000 against an expected 2,000" in note, note
        assert " 4 poll(s)" in note, note
        assert "the binding signal was the document count" in note, note
        # The explanation the measurement replaced, and the one it replaced
        # it with. Asserted as a pair: a negative on its own is satisfied by
        # a note that says nothing at all.
        assert "binds only while the index is OVERCOUNTING" not in note, note
        assert "`num_docs == 2,000`" in note, note

    def test_the_gap_is_measured_rather_than_described(self):
        """Honest: "(2 short)" for an index that stops at 1,998 and
            "(exactly on target)" for one that reaches 2,000 -- different
            phrases out of the same code path.
        Mutant (the gap's operands swapped, `expected - docs`): the index that
            stops two short reads "(2 over)" and the one holding 481 too many
            reads "(481 short)" -- the magnitude survives and the direction
            inverts, which is the error that would send someone looking for a
            leak when the index is behind.
        """
        short = Wait([0], PLATEAU)
        short.synced(2000)
        assert "(2 short)" in short.note, short.note

        exact = Wait([0], RISING)
        exact.synced(2000)
        assert "(exactly on target)" in exact.note, exact.note

        # And an overcount is reported as over, not as its absolute distance:
        # readings that stop above the target.
        over = Wait([0], [2481, 2481, 2481])
        over.synced(2000)
        assert "(481 over)" in over.note, over.note
        assert over.detail["observed_docs"] == 2481

    def test_the_binding_signal_distinguishes_the_queue_from_the_count(self):
        """Which half actually held the wait open is the question
        `index_sync_note` exists to answer, and the answer differs.

        Honest: "the document count" when the queue was drained throughout and
            the count was catching up; "the mutation queue" when the count was
            exact from the first reading and the queue was not.
        Mutant (`blocked_by` recomputed on the poll that ends the wait, where
            both conditions are true by construction): both read
            "neither: the first reading already agreed".
        """
        by_count = Wait([0], RISING)
        by_count.synced(2000)
        assert by_count.detail["binding_signal"] == "the document count"

        by_queue = Wait([5, 5, 0], [2000])
        by_queue.synced(2000)
        assert by_queue.detail["binding_signal"] == "the mutation queue"
        assert by_queue.polls == 3

        at_once = Wait([0], [2000])
        at_once.synced(2000)
        assert at_once.polls == 1
        assert at_once.detail["binding_signal"] == \
            "neither: the first reading already agreed"

    def test_a_timed_out_wait_still_carries_the_observation(self):
        """The timeout is the state this engine is known to reach, so it is
        the one where the readings matter most.

        Honest: "... the wait gave up: num_docs=2,000 against an expected
            2,000 (exactly on target) after 450 poll(s) ...".
        Mutant (the observation not recorded on the timeout path): "the index
            had not absorbed the re-insert after 900s and the wait gave up;
            what follows measures an index that was still catching up" --
            true, and silent about what the index was actually holding.
        """
        run = Wait([5], [2000])
        run.synced(2000)
        assert "num_docs=2,000" in run.note, run.note
        assert " 450 poll(s)" in run.note, run.note
        assert run.detail["queue_size"] == 5

    def test_a_caller_that_stubs_the_wait_still_gets_a_note(self):
        """`index_sync_detail` is None when nothing polled, and the note has
        to render anyway rather than raising inside a churn step.

        Honest: a note that still explains the condition, with no observation
            clause.
        Mutant (the `or {}` dropped, so the detail is read straight):
            `AttributeError: 'NoneType' object has no attribute 'get'`,
            mid-churn, after the re-insert has already been paid for.
        """
        driver = valkey_mod.ValkeyDriver(ConnectionSpec(host="h", port=6379))
        driver._wait_for_mutations = lambda **kw: 3.25
        assert driver.wait_index_synced(2000) == 3.25
        assert driver.index_sync_detail is None
        note = driver.index_sync_note
        assert "settled in 3.2s." in note, note
        assert "num_docs=" not in note, note

    def test_the_delete_waits_readings_are_not_published_as_the_inserts(self):
        """`delete_ids` runs earlier in the same churn step and leaves its own
        readings behind, so the insert's note must not be able to render them.

        Honest: nothing observed, so nothing quoted.
        Mutant (the detail not cleared before delegating), measured:

            settled in 3.2s: num_docs=1,800 against an expected 1,800
            (exactly on target) after 3 poll(s), ended because num_docs fell
            to the expected count; the binding signal was the document count

        for a re-insert of 2,000 rows that was never measured. Note which
        operand leaked: `_observed_sync_phrase` reads BOTH the observed and
        the expected count out of the stale detail, so the gap computes to
        zero and the sentence reads **(exactly on target)**. A fabricated gap
        would at least look wrong to a reader; this publishes a clean pass for
        a wait that did not happen, in the one field the live smoke run has to
        read. The tells are on the **record**, not in that sentence, and there
        are three: `docs_expectation: 'at_most'` on a re-insert record, which
        is the cleanest query for it; `expected_docs: 1800` where the
        re-insert wanted 2,000; and `ended_by: 'num_docs fell to the expected
        count'` for a count that rose. Measured, the note itself carries only
        the third -- `'at_most'` and `'docs_expectation'` appear nowhere in
        it -- so a reader of the prose has no way to see the substitution.
        Queryable afterwards is not the same as prevented, which is what the
        clear below is for.
        """
        run = Wait([0], FALLING)
        run.wait(expected=1800, docs_expectation=AT_MOST)
        assert run.detail["observed_docs"] == 1800          # the delete's

        run.driver._wait_for_mutations = lambda **kw: 3.25
        run.driver.wait_index_synced(2000)
        assert run.driver.index_sync_detail is None
        assert "num_docs=" not in run.driver.index_sync_note, \
            run.driver.index_sync_note


# ---------------------------------------------------------------------------
# The condition on its own
# ---------------------------------------------------------------------------

class TestTheConditionIsTotal:
    """`_docs_ready` is a module function so the comparison can be read
    without a clock in the way. Every branch returns a verdict AND the phrase
    that names it, because the phrase is what the record publishes."""

    def test_a_build_that_publishes_no_count_does_not_hang(self):
        """Reading a missing field as agreement is what made `percent_indexed`
        a no-op; refusing to proceed without it would hang every churn on a
        build that does not report it. The queue is then all there is.

        Honest: (True, "this build publishes no num_docs"), and an empty
            FT.INFO ends the wait at 1 poll, 0.000 s.
        Mutant (False): the same empty FT.INFO runs 15 polls to a 30 s
            ceiling with `timed_out` True -- so at the real ceiling, 900 s per
            churn step on a build that reports neither field.
        """
        assert valkey_mod._docs_ready(None, 2000, EQUAL, 0, 0.0) == \
            (True, "this build publishes no num_docs")

    def test_no_expected_count_still_settles_on_two_equal_readings(self):
        """The `expected_docs is None` branch, which the lab script uses.

        Honest: `ready_at == [False, False, True]` -- not ready at one
            repeat, ready at two.
        Mutant (`DOCS_SETTLED_POLLS = 1`): `[False, True, True]`, a wait
            decided by one pair of equal readings, which is the trap this
            branch was written against.

        **No poll count is quoted, and that is deliberate.** An earlier
        version of this docstring said the mutant "ends at 3 polls instead of
        4" on the plateau. Measured after the escape floor landed: **16 polls
        / 30.000 s either way** -- the floor dominates the settle count, so
        that consequence no longer discriminates anything. The unit values
        above do, and `DOCS_SETTLED_POLLS == 2` is pinned separately.
        """
        ready_at = [valkey_mod._docs_ready(1800, None, AT_MOST, n, 0.0)[0]
                    for n in (0, 1, 2)]
        assert ready_at == [False, False, True]

    def test_the_two_expectations_disagree_where_it_matters(self):
        """The table the whole change turns on, read directly off the
        condition. `at_most` is satisfied by an index that is short; `equal`
        is not, and neither accepts an overcount except `at_most`, which only
        ever sees one on the way down."""
        short = (1800, 2000)
        over = (2481, 2000)
        exact = (2000, 2000)
        assert valkey_mod._docs_ready(*short, AT_MOST, 0, PAST_FLOOR)[0] is True
        assert valkey_mod._docs_ready(*short, EQUAL, 0, PAST_FLOOR)[0] is False
        assert valkey_mod._docs_ready(*over, AT_MOST, 0, PAST_FLOOR)[0] is False
        assert valkey_mod._docs_ready(*over, EQUAL, 0, PAST_FLOOR)[0] is False
        assert valkey_mod._docs_ready(*exact, AT_MOST, 0, PAST_FLOOR)[0] is True
        assert valkey_mod._docs_ready(*exact, EQUAL, 0, PAST_FLOOR)[0] is True

    def test_the_observation_reaches_the_record_as_fields(self):
        """`index_sync_note` is prose, so reading it back means matching a
        sentence. The same facts also travel as `extra.index_sync_detail`, and
        the methodology page now publishes a `jq` recipe that reads them as
        fields -- so the carry is a claim, and this is it.

        Honest: `_sync_shape` returns both keys, and the detail is the dict the
            wait recorded, not a copy that could drift.
        Mutant (`_sync_shape` carrying only the note, the form it shipped in):
            `{'index_sync_note': ...}` alone, every field in the recipe reads
            `null`, and the one question the live smoke run exists to answer
            has to be grepped out of a sentence.
        """
        from harness.workloads.churn import _sync_shape

        run = Wait([0], RISING)
        run.synced(2000)
        shape = _sync_shape(run.driver)
        assert set(shape) == {"index_sync_note", "index_sync_detail"}
        assert shape["index_sync_detail"] is run.driver.index_sync_detail
        assert shape["index_sync_detail"]["observed_docs"] == 2000
        assert shape["index_sync_detail"]["expected_docs"] == 2000
        assert shape["index_sync_detail"]["polls"] == 4
        assert shape["index_sync_detail"]["ended_by"] == \
            "num_docs reached the expected count"

        # An engine with neither attribute carries neither key -- the property
        # `_sync_shape`'s own docstring rests on, and the one the hookless
        # record-shape assertion in `test_t4a_churn.py` would fail on.
        class Bare:
            name = "bare"

        assert _sync_shape(Bare()) == {}

        # And a driver with a note but no detail still carries just the note,
        # which is every other engine that implements the hook.
        class NoteOnly:
            index_sync_note = "mongot already had the row at the first poll"

        assert _sync_shape(NoteOnly()) == {
            "index_sync_note": "mongot already had the row at the first poll"}

    def test_the_methodology_paragraph_is_pinned_to_the_code(self):
        """`docs/05-methodology.md` asserts engine behaviour, and this repo
        keeps such assertions executable. Nothing pinned the Valkey
        index-sync paragraph, which is how it came to describe a wait on the
        queue alone with "the queue is then the only signal in force" -- a
        claim the code made false and no test caught.

        Honest: the paragraph names both signals, names equality and the
            off-target report, and does not say either signal is the only one
            in force.
        Mutant (the old paragraph restored): "only signal in force" is back in
            the file and "num_docs == rows" is absent, which this fails on.
        """
        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "docs", "05-methodology.md")
        # Collapsed, because this document is hard-wrapped and an assertion
        # against the raw text passes or fails on where the wrap falls.
        text = " ".join(open(path).read().split())

        # The claim the doc now makes, each half checked against the code.
        assert "mutation queue and the indexed document count" in text
        assert "`num_docs == rows`" in text
        assert "ends the wait when the count stops moving" in text
        assert "(n short)" in text and "(n over)" in text
        # And the claim it must no longer make. Paired with the positives
        # above: a negative assertion on its own is satisfied by a file that
        # says nothing about Valkey at all.
        assert "only signal in force" not in text
        # The page's `jq` recipe reads these as fields, so the field names it
        # spells have to be the ones the driver writes -- a recipe naming a
        # key nobody records returns `null` for every row and looks like an
        # engine with nothing to report.
        from harness.workloads.churn import _sync_shape
        run = Wait([0], RISING)
        run.synced(2000)
        recorded = _sync_shape(run.driver)["index_sync_detail"]
        for field in ("observed_docs", "expected_docs", "polls", "ended_by",
                      "binding_signal", "queue_size", "docs_expectation",
                      "timed_out"):
            assert field in recorded, field
            assert field in text, field
        assert "extra.index_sync_detail" in text

        # **Every key the recipe cites must be one of those.** The loop above
        # asserts the eight are documented and does NOT do this: it iterates
        # over the recorded fields looking for them in the page, so a recipe
        # naming something else entirely is invisible to it -- measured, a
        # `select` rewritten to `.observed_count != .expected_count`, keys no
        # driver writes, left this file at 31 passed and exit 0 while real
        # `jq` returns empty output and exit 0 for that recipe, which reads as
        # "no Valkey churn step had an off-target index". A publishable claim
        # out of a broken query is the worst failure this page can have.
        cited = set(re.findall(r"\.index_sync_detail\.(\w+)", text))
        assert cited, "the page cites no fields at all"
        assert cited <= set(recorded), cited - set(recorded)
        # And the path they hang off, because the names being real is not the
        # same as the path being right: a `select` rewritten to
        # `.index_sync_detail.observed_docs` without the `.extra.` cites only
        # real recorded keys, passes everything above, and returns empty
        # output and exit 0 under real `jq` -- measured, 74 passed either way
        # before this line.
        assert text.count(".index_sync_detail.") == \
            text.count(".extra.index_sync_detail.")
        # It must no longer say the fields are unreachable, which was true
        # until `_sync_shape` carried the dict.
        assert "carries only the string" not in text

        # The code the paragraph is about. Without this the doc could be
        # "fixed" to say equality while the driver kept comparing with `<=`.
        assert valkey_mod._docs_ready(1800, 2000, EQUAL, 0, PAST_FLOOR)[0] is False
        assert valkey_mod._docs_ready(2000, 2000, EQUAL, 0, PAST_FLOOR)[0] is True
        assert valkey_mod._docs_ready(1800, 2000, AT_MOST, 0, PAST_FLOOR)[0] is True
        source = open(valkey_mod.__file__).read()
        block = source.split("def wait_index_synced")[1].split("\n    def ")[0]
        assert "docs_expectation=DOCS_EQUAL" in block, block[:400]

    def test_an_overcount_cannot_inflate_the_churn_recall_figure(self):
        """The paragraph's claim about DIRECTION, checked against the scoring
        function instead of against its own vocabulary.

        The first version of that paragraph said the two directions were
        "opposite errors" and that an over-count could leave recall
        *overstated*. It cannot, and the assertions above would not have
        caught it: they pin which words appear, and a wrong interpretation
        uses the same words. The churn path scores with `recall_at_k` -- an id
        intersection against truth translated through the re-insert's id map
        (`churn.py`) -- so a stale id is absent from the translated set by
        construction and can only consume a top-k slot.

        Honest: 1.0000 correct, 0.6000 for stale ids returned in place of the
            re-inserted ones, 0.6000 for the two new rows missing, 0.0000 when
            nothing live comes back. No over-count case exceeds the correct
            one.
        Mutant (the claim restored -- "it can be overstated"): the doc says a
            figure can read high while this measurement says every over-count
            case reads at or below 0.6000. Fails on the assertion below, and
            on the two phrases, which is why both halves are here.
        """
        from harness.datasets import recall_at_k
        k = 5
        truth = numpy.array([[10, 11, 12, 13, 14]], dtype=numpy.int64)
        id_map = {10: 1010, 11: 1011}        # two rows deleted, re-inserted
        translated = numpy.array(
            [[id_map.get(int(t), int(t)) for t in row[:k]] for row in truth],
            dtype=numpy.int64)

        correct = recall_at_k([[1010, 1011, 12, 13, 14]], translated, k)
        over_in_place = recall_at_k([[10, 11, 12, 13, 14]], translated, k)
        over_crowding = recall_at_k([[10, 11, 1010, 1011, 12]], translated, k)
        short_two = recall_at_k([[12, 13, 14, 777, 888]], translated, k)
        over_total = recall_at_k([[10, 11, 7, 8, 9]], translated, k)

        assert correct == 1.0
        assert over_in_place == 0.6 and over_crowding == 0.6
        assert short_two == 0.6
        assert over_total == 0.0
        # The claim itself: no over-count result can beat a correct one.
        assert max(over_in_place, over_crowding, over_total) < correct

        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "docs", "05-methodology.md")
        text = " ".join(open(path).read().split())
        assert "understates recall either way and cannot inflate it" in text
        # The two phrasings this measurement refutes, paired with the positive
        # above so neither assertion can pass against a silent document.
        assert "opposite errors" not in text
        assert "overstated" not in text

    def test_an_unknown_expectation_raises_rather_than_falling_through(self):
        """The one axis this task is about must not fail open.

        Honest: every value that is neither constant raises `ValueError`
            naming the two that are accepted.
        Mutant (the final branch left as a bare fallthrough, which is the
            form this file shipped mid-round): `'eqaul'`, `'EQUAL'`,
            `'exact'`, `''` and `None` ALL returned
            `(True, 'num_docs fell to the expected count')` for an index 200
            documents short -- a typo silently reinstating the delete path's
            `<=` on the insert path, which is the defect being fixed, under a
            phrase saying the count fell when it had risen and not arrived.
        """
        for bad in ("eqaul", "EQUAL", "exact", "", None, 0, AT_MOST.upper()):
            with pytest.raises(ValueError) as caught:
                valkey_mod._docs_ready(1800, 2000, bad, 0, PAST_FLOOR)
            message = str(caught.value)
            assert "docs_expectation" in message, message
            assert repr(AT_MOST) in message and repr(EQUAL) in message, message
        # The check is the first thing the function does, so no earlier
        # branch can let a misspelling through. Measured with it hoisted back
        # down: a build publishing no `num_docs` returned
        # `(True, 'this build publishes no num_docs')` for expectation
        # `'eqaul'`, validating nothing.
        with pytest.raises(ValueError):
            valkey_mod._docs_ready(None, 2000, "eqaul", 0, 0.0)
        with pytest.raises(ValueError):
            valkey_mod._docs_ready(1800, None, "eqaul", 0, 0.0)
        # Unreachable from either call site today -- which is the reason it is
        # asserted here rather than left to a future caller to discover.
        assert valkey_mod._docs_ready(1800, 2000, AT_MOST, 0, 0.0)[0] is True
        assert valkey_mod._docs_ready(2000, 2000, EQUAL, 0, 0.0)[0] is True
        assert valkey_mod._docs_ready(None, 2000, EQUAL, 0, 0.0) == \
            (True, "this build publishes no num_docs")

    def test_the_escape_floor_is_a_named_constant_well_above_a_stall(self):
        """Pinned so the floor cannot be trimmed back towards the four
        seconds that `DOCS_SETTLED_POLLS` alone buys.

        Honest: 30 s, and `_docs_ready` refuses the off-target exit below it.
        Mutant (floor set to 0, or the conjunct dropped): the stall case ends
            at 3 polls / 4.000 s holding 1,800 instead of 5 polls / 8.000 s
            holding 2,000, measured.
        """
        floor = valkey_mod.DOCS_SETTLED_FLOOR_S
        assert floor >= 30.0
        # Two seconds per poll, so the floor has to outlast a handful of them.
        assert floor > 2.0 * (valkey_mod.DOCS_SETTLED_POLLS + 1)
        assert valkey_mod._docs_ready(1998, 2000, EQUAL, 9, floor - 0.001)[0] \
            is False
        assert valkey_mod._docs_ready(1998, 2000, EQUAL, 9, floor)[0] is True

    def test_a_build_with_no_count_says_so_instead_of_reading_as_a_stub(self):
        """The original defect's twin, and the record has to be able to name
        it: a build that publishes no `num_docs` leaves the document half
        satisfied unconditionally and the queue as the only guard.

        Honest: the note names the missing field, the poll count and that the
            document half was satisfied unconditionally.
        Mutant, two forms, both measured. With the early return widened back
            to `if docs is None or expected is None`, the note keeps its lead
            and loses every reading: "the document count could not be
            checked; the 0.0s this took means the queue was empty, not that
            the index had absorbed the re-insert" -- no missing field named,
            no poll count, which is what the assertions below catch.
            **That mutant was byte-identical to a stubbed wait until this
            same round made the lead `ended_by`-keyed**, and it is not any
            more (measured: identical False, "settled in 0.0s" False). The
            two guards are now independent -- the lead distinguishes the case
            without `_observed_sync_phrase`, and the clause supplies the
            numbers -- which is better than the single point of failure the
            docstring used to describe. With the clause simply deleted
            from the current form, `TypeError: unsupported operand type(s)
            for -: 'NoneType' and 'int'` mid-churn, because the gap
            arithmetic below has no count to subtract.
        """
        run = Wait([0], None)                  # queue drained, no num_docs
        assert run.synced(2000) == 0.0
        assert run.polls == 1
        assert run.detail["observed_docs"] is None
        assert run.detail["ended_by"] == "this build publishes no num_docs"
        note = run.note
        assert "published no num_docs" in note, note
        assert " 1 poll(s)" in note, note
        assert "satisfied unconditionally" in note, note
        # And it does not lead with "settled" -- this is the one case where
        # nothing about the count was checked at all, so "settled in 0.0s"
        # would be the most misleading sentence this driver can write.
        assert note.startswith("the document count could not be checked"), note
        assert "settled in" not in note, note

        # And the stubbed-out wait, which must NOT say any of that.
        stubbed = valkey_mod.ValkeyDriver(ConnectionSpec(host="h", port=6379))
        stubbed._wait_for_mutations = lambda **kw: 0.0
        stubbed.wait_index_synced(2000)
        assert "published no num_docs" not in stubbed.index_sync_note
        assert stubbed.index_sync_note != note

    def test_the_settled_poll_count_is_a_named_constant(self):
        """Two consecutive repeats, not one: a single sample cannot end a
        wait. Pinned so the escape cannot be loosened to one reading without
        this failing."""
        assert valkey_mod.DOCS_SETTLED_POLLS == 2
        assert valkey_mod._docs_ready(1998, 2000, EQUAL, 1, PAST_FLOOR)[0] is False
        assert valkey_mod._docs_ready(1998, 2000, EQUAL, 2, PAST_FLOOR)[0] is True
