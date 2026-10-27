"""E20 -- the vendor's memory gauge is documented as recorded and is collected
nowhere.

`CLAUDE.md`, `README.md`, `config/engines/percona.yml` -- which calls it "the
audit trail" for the decision to leave `innodb_hnsw_max_memory` unlimited --
and `docs/09-adding-percona-server.md` all state that
`Innodb_hnsw_memory_used` is recorded per phase. Nothing in `harness/`,
`report/` or `orchestrator/` read a status variable from ANY engine, so the
audit trail for a deliberate fairness decision did not exist.
`docs/11-methodology-roadmap.md` §4.1 says so in its own words and calls it
the most embarrassing item on that page.

**The gauge is not a missing server feature.** It is in the status-variable
array at the pin (`docs/09-adding-percona-server.md` §2.6, from
`innobase_system_variables[]` and the status array beside it, which together
contain exactly one HNSW status entry and no second), and the vendor suite has
a result file for it. Only the collector was ours to write.

WHAT THE FIX IS, by symbol:

* `EngineDriver.status_gauges` -- `()` on the base class, so the six engines
  that have no such variable record exactly what they recorded before.
* `EngineDriver.status_counters` -- `{}` on the base class. One optional hook
  beside `prepare_churn`, `wait_index_synced` and `maintain`, same
  no-op-default discipline.
* `MySQLFamilyDriver.status_counters` -- one `SHOW {GLOBAL|SESSION} STATUS
  LIKE` per name, coerced to `int`, **omitting** a name the server does not
  expose rather than zeroing it. Zero is a legitimate reading of this
  particular gauge -- §2.6 records it reading `0` after a restart, because the
  graph loads lazily -- so "absent" and "zero" are two different facts and
  have to stay two different facts.
* `PerconaDriver.status_gauges` -- `("Innodb_hnsw_memory_used",)`, the only
  engine in the set that declares one.
* `harness/workloads/build.py` -- `_server_status`, and one `extra` key,
  `extra.server_status`, a `{name: int}` dict on the `index_build` record,
  written only when a gauge was actually read.

WHAT THESE TESTS DO NOT ESTABLISH, stated here because a collector that has
never run against a live server is unproven however green this file is.
Nothing here starts a container, a server or a build: every statement goes to
a recording cursor, and the value that comes back is one this file made up.
So these tests establish the *shape* of the capture -- which statement is
issued, in which scope, what the record carries, and that no other engine's
record moves -- and they establish nothing about whether a real Percona Server
answers `SHOW GLOBAL STATUS LIKE 'Innodb_hnsw_memory_used'` with a row this
collector can parse. The first ops run on the build host is what confirms
that, and the thing to look at is `extra.server_status` on the `index_build`
record in `ops-percona-*.jsonl`.

THE CAVEAT THAT TRAVELS WITH EVERY FIGURE FROM THIS GAUGE, and the reason it
is repeated in the hook's docstring, in the driver's declaration and in
`build.py`: it omits the node map (*HNSW::m_nodes*, a supplied 30-50 bytes per
node) and search scratch, so it is a **floor** on graph memory and not a
measurement of it. At glove-100's 1,183,514 rows that is 35.5-59.2 MB
unreported per vector index.

PROBE STYLE, per this suite's convention (`tests/test_e9_percona_bulk_build_crash.py`,
`tests/test_e13_percona_churn_repair_is_a_noop.py`): a decision test reports
its honest and mutant value rather than a bare pass/fail, and an identical
pair is called out as a dead probe. The one mutant that can be injected at run
time -- `status_gauges` declared on the base class instead of on
`PerconaDriver` -- is injected with `mock.patch.object`, never by editing a
file on disk: CPython invalidates a `.pyc` on (whole-second mtime, size), so a
same-size rewrite inside one second can leave stale bytecode valid and measure
the previous version. The other two mutants the spec names are source edits
and were applied and reverted by hand.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence
from unittest import mock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import datasets as ds                                   # noqa: E402
from harness.drivers.base import (ConnectionSpec, EngineDriver,       # noqa: E402
                                  IndexSpec)
from harness.drivers.mysql_family import (AliSQLDriver, MariaDBDriver,  # noqa: E402
                                          PerconaDriver)
from harness.drivers.postgres import _driver_classes                 # noqa: E402
from harness.metrics import records                                  # noqa: E402
from harness.workloads import build as build_workload                # noqa: E402
from harness.workloads.context import RunContext                     # noqa: E402
from tests.fake_driver import (DIM, FakeDriver, K, N_ROWS,           # noqa: E402
                               index_spec, tiny_dataset)

#: The one HNSW status variable the pinned server registers. Spelled once.
GAUGE = "Innodb_hnsw_memory_used"

#: A value this file invented. It is deliberately not round and deliberately
#: not zero: a collector that returned a default would have to return one of
#: those, and an assertion against either could pass without a statement ever
#: having been issued. Nothing about it came off a server -- see the module
#: docstring on what these tests do not establish.
MEMORY_USED = 2_013_265_921

#: The `extra` key the `index_build` record carries. One key holding a
#: `{name: int}` dict, not one key per variable, so a second gauge needs no
#: schema decision and a report can look for a name.
KEY = "server_status"


# ---------------------------------------------------------------------------
# A recording cursor, so the SQL the driver sends -- or does not send -- is the
# thing under test
# ---------------------------------------------------------------------------

#: `SHOW <scope> STATUS LIKE '<name>'`, as the cursor below reads it back.
#: Matching the scope as a group rather than accepting either is what makes
#: assertion 1's "GLOBAL, not SESSION" an assertion and not a hope: the gauge
#: is GLOBAL status (`docs/09-adding-percona-server.md` §2.6), and a SESSION
#: read of a global-only variable is a different question with a different
#: answer.
STATUS_RE = re.compile(r"^\s*SHOW\s+(GLOBAL|SESSION)\s+STATUS\s+LIKE\s+'([^']*)'",
                       re.IGNORECASE)

#: What a live server answers `SHOW CREATE TABLE` with. Needed only because
#: `PerconaDriver.maintain()` reads the index options back off the server; no
#: test here calls it, and it is here so a cursor that is asked cannot answer
#: `None` and send the driver down a fall-back path by accident.
SHOW_CREATE = (
    "t1",
    "CREATE TABLE `t1` (\n"
    "  `id` bigint unsigned NOT NULL,\n"
    "  `v` vector(16) NOT NULL,\n"
    "  PRIMARY KEY (`id`),\n"
    "  VECTOR KEY `vi` (`v`) USING `hnsw` "
    "(`M`=16, `ef_construction`=200, `metric`=`euclidean`)\n"
    ") ENGINE=InnoDB",
)

TABLE_BYTES = 123_456_789
VERSION_STRING = "8.4.0-1.percona"


class StatusError(RuntimeError):
    """What a dead connection raises. Named so the assertion can say which."""


class RecordingCursor:
    """Records every statement and answers by STATEMENT, like a server.

    Deliberately self-contained rather than imported from
    `tests/test_e13_percona_churn_repair_is_a_noop.py` or
    `tests/test_e2_percona_wiring.py`: the central claim of this file is "no
    status statement reached the cursor", and it must not be able to break
    because another test file reshaped its fixture.

    Keyed on the statement rather than queued because `build.run` issues a
    dozen statements in an order this file should not have to encode, and a
    queue drained by one of them hands the next caller another caller's row.

    `status` is the server's status table as this fixture pretends it to be: a
    name absent from it is a name the server does not expose, which is the
    case assertion 3 is about.
    """

    def __init__(self, status: Optional[Dict[str, Any]] = None,
                 rows: int = N_ROWS, raises: bool = False):
        self.statements: List[str] = []
        self._status = dict(status or {})
        self._rows = rows
        self._raises = raises
        self._answer: List[tuple] = []

    def execute(self, sql, params=None):
        self.statements.append(str(sql))
        if self._raises:
            raise StatusError(f"server has gone away: {sql}")
        self._answer = self._answer_for(str(sql))

    def _answer_for(self, sql: str) -> List[tuple]:
        status = STATUS_RE.match(sql)
        if status is not None:
            # The server matches the LIKE pattern against its own names, and
            # returns the name in its own casing. Both are reproduced: the
            # collector is supposed to key the result by the name it ASKED
            # for, so a fixture that echoed the request verbatim could not
            # tell that apart from one that trusted the server's string.
            wanted = status.group(2)
            return [(name, value) for name, value in self._status.items()
                    if name.lower() == wanted.lower()]
        upper = sql.upper()
        if upper.startswith("SHOW CREATE TABLE"):
            return [SHOW_CREATE]
        if upper.startswith("SELECT COUNT(*)"):
            return [(self._rows,)]
        if upper.startswith("SELECT DATA_LENGTH"):
            return [(TABLE_BYTES,)]
        if upper.startswith("SELECT VERSION()"):
            return [(VERSION_STRING,)]
        return []

    def executemany(self, sql, rows):
        self.statements.append(str(sql))
        if self._raises:
            raise StatusError(f"server has gone away: {sql}")
        self._answer = []

    def fetchone(self):
        return self._answer[0] if self._answer else None

    def fetchall(self):
        return list(self._answer)

    def close(self):
        pass

    @property
    def status_statements(self) -> List[str]:
        """Every `SHOW ... STATUS` statement, which is where the proof lives."""
        return [s for s in self.statements if STATUS_RE.match(s)]


def wired(cls, *, status: Optional[Dict[str, Any]] = None,
          raises: bool = False, build_mode: str = "post"):
    """A real driver of `cls` wired to a recording cursor, with no server.

    `connect()` is not called: it needs the `mariadb` connector, which is a
    container-side dependency this suite does not have on the host, and the
    only thing it would do here is hand the driver a cursor -- which is what
    this function does instead.
    """
    driver = cls(ConnectionSpec(host="h", port=3306))
    cursor = RecordingCursor(status=status)
    cursor._raises = raises
    driver._cur = cursor
    driver._index = IndexSpec(dim=DIM, m=16, metric=ds.EUCLIDEAN,
                              build_mode=build_mode)
    return driver


def cursor_of(driver) -> RecordingCursor:
    return driver._cur


# ---------------------------------------------------------------------------
# 1. The statement, its scope, and the value that comes back
# ---------------------------------------------------------------------------

class TestTheStatementIssued:
    def test_a_global_status_read_naming_the_gauge_returns_its_value(self):
        driver = wired(PerconaDriver, status={GAUGE: MEMORY_USED})

        got = driver.status_counters((GAUGE,))

        assert got == {GAUGE: MEMORY_USED}
        issued = cursor_of(driver).status_statements
        assert len(issued) == 1, issued
        scope, named = STATUS_RE.match(issued[0]).groups()
        print(f"STATEMENT = {issued[0]!r}")
        # GLOBAL status, established in `docs/09-adding-percona-server.md`
        # §2.6 from the server's own status-variable array. A SESSION read
        # here would be a different question.
        assert scope.upper() == "GLOBAL"
        assert named == GAUGE

    def test_the_scope_argument_is_the_only_thing_that_changes_the_keyword(self):
        """A probe, because `scope` has a default and a default is easy to
        ignore. The session arm asserts the parameter is wired, not that a
        session read of this gauge is meaningful -- it is not; the variable is
        GLOBAL status and nothing in this repo asks for it per session."""
        honest = wired(PerconaDriver, status={GAUGE: MEMORY_USED})
        honest.status_counters((GAUGE,))
        mutant = wired(PerconaDriver, status={GAUGE: MEMORY_USED})
        mutant.status_counters((GAUGE,), scope="session")

        honest_scope = STATUS_RE.match(cursor_of(honest).status_statements[0]).group(1)
        mutant_scope = STATUS_RE.match(cursor_of(mutant).status_statements[0]).group(1)
        print(f"HONEST (default scope)   = {honest_scope}")
        print(f"MUTANT (scope='session') = {mutant_scope}")
        assert honest_scope.upper() != mutant_scope.upper(), (
            "DEAD PROBE: the scope argument changes nothing")
        assert honest_scope.upper() == "GLOBAL"
        assert mutant_scope.upper() == "SESSION"

    def test_a_scope_that_is_neither_is_refused_rather_than_interpolated(self):
        """`checked_filter_column`'s discipline, for the same reason: the
        scope reaches a statement string, so an unrecognised one is a
        `ValueError` and never an interpolation. A caller bug, not a server
        failure -- and `build.py` swallows it either way, so no run dies of
        it."""
        driver = wired(PerconaDriver, status={GAUGE: MEMORY_USED})
        with pytest.raises(ValueError):
            driver.status_counters((GAUGE,), scope="GLOBAL; DROP DATABASE ann")
        assert cursor_of(driver).statements == []

    def test_a_name_that_could_not_be_spelled_safely_is_refused(self):
        """Same reason. The names come from a code-declared tuple, so this can
        only fire on a typo in `status_gauges` -- and failing loudly is what
        makes that typo findable, where silently skipping the name would make
        the gauge absent forever and look exactly like a server that does not
        expose it."""
        driver = wired(PerconaDriver)
        for bad in ("Innodb_hnsw_memory_used'; SELECT 1 --", "Innodb%", "", "a b"):
            with pytest.raises(ValueError):
                driver.status_counters((bad,))
        assert cursor_of(driver).statements == []

    def test_the_result_is_keyed_by_the_name_asked_for(self):
        """Not by the server's string, which is a different object and whose
        casing is the server's business. `checked_filter_column` returns the
        whitelisted element rather than the caller's string for the same
        reason: the key set of `extra.server_status` has to be decidable from
        this repository, because that is what a report greps for."""
        driver = wired(PerconaDriver, status={"INNODB_HNSW_MEMORY_USED": MEMORY_USED})
        got = driver.status_counters((GAUGE,))
        print(f"server spelled it INNODB_HNSW_MEMORY_USED; record says "
              f"{list(got)}")
        assert got == {GAUGE: MEMORY_USED}


# ---------------------------------------------------------------------------
# 2. A cursor that raises loses no measurement
# ---------------------------------------------------------------------------

class TestAFailingServerIsNotAFailedRun:
    def test_a_cursor_that_raises_yields_nothing_and_propagates_nothing(self):
        driver = wired(PerconaDriver, status={GAUGE: MEMORY_USED}, raises=True)
        got = driver.status_counters((GAUGE,))
        print(f"cursor raised {StatusError.__name__}; collector returned {got!r}")
        assert got == {}

    def test_one_dead_name_does_not_cost_the_others(self):
        """Per-name, not per-call. The loop swallows per statement, so a gauge
        a future server drops does not take the gauges beside it with it."""
        class HalfDead(RecordingCursor):
            def execute(self, sql, params=None):
                if "doomed" in str(sql):
                    self.statements.append(str(sql))
                    raise StatusError(sql)
                super().execute(sql, params)

        driver = wired(PerconaDriver)
        driver._cur = HalfDead(status={GAUGE: MEMORY_USED})
        got = driver.status_counters((GAUGE, "Innodb_doomed_counter"))
        assert got == {GAUGE: MEMORY_USED}

    def test_build_run_swallows_a_driver_whose_hook_raises(self):
        """`_driver_capabilities`'s discipline, which is the pattern the hook
        is modelled on: a measurement already taken is not lost to a driver
        that misbehaves afterwards. The record is written, and it carries no
        status key."""
        class Exploding(FakeDriver):
            status_gauges = (GAUGE,)

            def status_counters(self, names, scope="global"):
                raise StatusError("the hook itself is broken")

        built = _one_build(Exploding)
        assert KEY not in built["extra"]
        assert built["phase"] == records.PHASE_BUILD
        assert built["index_bytes"] > 0


# ---------------------------------------------------------------------------
# 3. Absent is not zero
# ---------------------------------------------------------------------------

class TestAbsentIsNotZero:
    """`docs/09-adding-percona-server.md` §2.6 is why this is the sharpest
    assertion in the file: `hnsw_memory_used.result` shows the gauge reading
    `0` after a restart, because the graph is faulted in lazily and rises on
    the first search. So zero is a MEASUREMENT of this variable, and a
    collector that zeroed a variable it could not read would publish "the
    graph is empty" and "this server has no such variable" as the same
    number."""

    def test_a_variable_the_server_does_not_expose_is_absent(self):
        driver = wired(PerconaDriver, status={})
        got = driver.status_counters((GAUGE,))
        print(f"HONEST (omit)        = {got!r}")
        print(f"MUTANT (zero-fill)   = {{{GAUGE!r}: 0}}")
        assert got == {}
        assert GAUGE not in got

    def test_a_real_zero_is_carried_through(self):
        """The other half, and the half that makes the one above mean
        something. Dropping zeros would be an equally wrong collector that
        passes the test above."""
        driver = wired(PerconaDriver, status={GAUGE: 0})
        got = driver.status_counters((GAUGE,))
        assert got == {GAUGE: 0}

    def test_a_value_that_is_not_a_number_is_omitted_rather_than_guessed(self):
        driver = wired(PerconaDriver, status={GAUGE: "disabled"})
        assert driver.status_counters((GAUGE,)) == {}

    @pytest.mark.parametrize("raw,expected", [
        (MEMORY_USED, MEMORY_USED),
        (str(MEMORY_USED), MEMORY_USED),
        (str(MEMORY_USED).encode(), MEMORY_USED),
        ("0", 0),
        ("  17  ", 17),
    ])
    def test_the_spellings_a_connector_may_hand_back(self, raw, expected):
        """A status value arrives as `int`, `str` or `bytes` depending on the
        connector and the column type, and `int(b'17')` is a `TypeError`. The
        record field is an int either way."""
        driver = wired(PerconaDriver, status={GAUGE: raw})
        assert driver.status_counters((GAUGE,)) == {GAUGE: expected}


# ---------------------------------------------------------------------------
# 4. Which engines declare a gauge, and what that costs the ones that do not
# ---------------------------------------------------------------------------

class TestOnlyPerconaDeclaresAGauge:
    """The answer to "what happens for the engines that have no such
    variable", in code rather than by implication: `()` on the base class and
    an early return in `build.py`, so no statement is issued and no key is
    written."""

    def test_the_base_class_declares_none_and_collects_nothing(self):
        assert EngineDriver.status_gauges == ()
        # The no-op default, same discipline as `prepare_churn`,
        # `wait_index_synced` and `maintain`.
        assert EngineDriver.status_counters(
            mock.Mock(spec=EngineDriver), (GAUGE,)) == {}

    def test_percona_is_the_only_driver_in_the_registry_that_declares_one(self):
        """Derived from `_driver_classes()` rather than from a list written
        here, which would be the drift-by-omission hole the claim is meant to
        close: a driver added to the registry would otherwise be silently
        exempt from "nothing else moved". `_driver_classes` is one entry per
        ARCHITECTURE, which is the right granularity -- `status_gauges` is a
        class attribute, so mariadb and mariadb123 share one declaration."""
        declared = {name: tuple(cls.status_gauges)
                    for name, cls in sorted(_driver_classes().items())}
        print(f"DECLARED = {declared}")
        assert declared["PerconaDriver"] == (GAUGE,)
        assert all(gauges == () for name, gauges in declared.items()
                   if name != "PerconaDriver"), declared

    def test_mariadb_and_alisql_send_no_status_statement_through_build_run(
            self, tmp_path, dataset):
        """Asserted on the cursor's statement list, which is where the proof
        lives: "the record has no key" could also be produced by a driver that
        asked the server and got nothing, and that is a different claim --
        it would mean a statement per build per engine against five servers
        that have nothing to answer."""
        for cls in (MariaDBDriver, AliSQLDriver):
            driver = wired(cls, status={GAUGE: MEMORY_USED})
            built = _run_build(tmp_path / f"no-gauge-{cls.__name__}",
                               dataset, driver)[1]
            issued = cursor_of(driver).status_statements
            print(f"{cls.__name__}: status statements = {issued}")
            assert issued == []
            assert KEY not in built["extra"]

    def test_declaring_the_gauge_on_the_base_class_is_the_mutant(
            self, tmp_path, dataset):
        """The run-time mutant for assertion 4, injected rather than edited
        onto disk. A `status_gauges` on `EngineDriver` instead of on
        `PerconaDriver` would be invisible to every assertion above -- Percona
        would still collect, and the result dict would still be right -- and
        would send a `SHOW GLOBAL STATUS` to MariaDB and AliSQL on every
        build, for a variable neither server has, and write them an `extra`
        key no record of theirs has ever carried."""
        def mariadb_build(suffix):
            driver = wired(MariaDBDriver, status={GAUGE: MEMORY_USED})
            built = _run_build(tmp_path / f"mutant-{suffix}", dataset, driver)[1]
            return cursor_of(driver).status_statements, built["extra"]

        honest_issued, honest_extra = mariadb_build("honest")
        with mock.patch.object(EngineDriver, "status_gauges", (GAUGE,)):
            mutant_issued, mutant_extra = mariadb_build("base-class")

        print(f"HONEST (gauge on PerconaDriver) = {len(honest_issued)} "
              f"statements, {KEY} in extra: {KEY in honest_extra}")
        print(f"MUTANT (gauge on EngineDriver)  = {len(mutant_issued)} "
              f"statements, {KEY} in extra: {KEY in mutant_extra}")
        assert (honest_issued, KEY in honest_extra) != \
               (mutant_issued, KEY in mutant_extra), (
            "DEAD PROBE: moving the declaration to the base class changed "
            "nothing about a MariaDB build")
        assert honest_issued == [] and KEY not in honest_extra
        assert len(mutant_issued) == 1 and mutant_extra[KEY] == {GAUGE: MEMORY_USED}


# ---------------------------------------------------------------------------
# 5. End to end through `build.run`: one new key, and only when there is one
# ---------------------------------------------------------------------------

def _run_build(path, dataset, driver, build_mode: str = "post"):
    """One build phase against `driver`, and the two records it wrote."""
    recorder = records.Recorder(str(path.parent / f"ops-{path.name}.jsonl"))
    ctx = RunContext(run_id="e20", resource_pass="normalized",
                     recorder=recorder, engine=driver.name, engine_tag="pin",
                     march="x86-64-v3", k=K,
                     cache_dir=str(path.parent / f"cache-{path.name}"))
    try:
        build_workload.run(ctx, driver, dataset,
                           index_spec(dataset, build_mode=build_mode))
    finally:
        recorder.close()
    written = list(records.read_records(recorder.path))
    assert [r["phase"] for r in written] == [records.PHASE_INGEST,
                                             records.PHASE_BUILD]
    return written


def _one_build(driver_cls, **kwargs):
    """The `index_build` record a `FakeDriver` subclass produces."""
    import tempfile
    from pathlib import Path
    dataset = tiny_dataset()
    with tempfile.TemporaryDirectory() as tmp:
        driver = driver_cls(dim=dataset.dim, metric=dataset.metric, **kwargs)
        driver.connect()
        return _run_build(Path(tmp) / "one", dataset, driver)[1]


class NoGauges(FakeDriver):
    """Today's five unaffected engines, in the shape the suite already has for
    them: `status_gauges` inherited from the base class and untouched."""


class OneGauge(FakeDriver):
    """A driver that reports a gauge, without editing `tests/fake_driver.py`
    -- which belongs to no unit here and is shared by the whole suite."""

    status_gauges = (GAUGE,)

    def status_counters(self, names: Sequence[str],
                        scope: str = "global") -> Dict[str, int]:
        assert scope == "global"
        return {str(name): MEMORY_USED for name in names}


class SilentGauge(FakeDriver):
    """Declares a gauge the server does not answer for -- the shape a Percona
    build on a server without the variable would take. Assertion 3 at the
    record level: nothing to report is no key, not a key holding `{}` and not
    a key holding a zero."""

    status_gauges = (GAUGE,)

    def status_counters(self, names: Sequence[str],
                        scope: str = "global") -> Dict[str, int]:
        return {}


class TestTheRecordIsAdditive:
    def test_a_driver_that_reports_a_gauge_gets_the_key(self):
        built = _one_build(OneGauge)
        print(f"extra = {built['extra']}")
        assert built["extra"][KEY] == {GAUGE: MEMORY_USED}

    def test_a_driver_that_reports_none_gets_no_key_at_all(self):
        built = _one_build(NoGauges)
        print(f"extra keys = {sorted(built['extra'])}")
        assert KEY not in built["extra"]

    def test_a_gauge_the_server_does_not_answer_for_gets_no_key_either(self):
        """The mutant this one exists for is a `build.py` that writes
        `extra.server_status: {}` unconditionally. It passes every assertion
        above -- the dict is right whenever there is a dict -- and it changes
        the key set of every record of all seven engines, which is the one
        thing "additive" has to mean."""
        built = _one_build(SilentGauge)
        print(f"HONEST (no key)              = {sorted(built['extra'])}")
        print(f"MUTANT (unconditional empty) = "
              f"{sorted(list(built['extra']) + [KEY])}")
        assert KEY not in built["extra"]

    def test_the_key_set_of_an_unaffected_engine_record_is_unchanged(self):
        """The assertion that makes "additive" true rather than claimed. The
        expected set is written out literally rather than diffed against
        another run of the same code, which would hold under a mutant that
        added the key to both."""
        built = _one_build(NoGauges)
        # `FakeDriver.capabilities()` declares exactly one key on purpose
        # (see its `_capabilities`), so this set is the fake's and not the
        # seven engines'. What it pins is that `build.py` added nothing.
        assert set(built["extra"]) == {
            "incremental_index", "separable_build",
            records.OPS_MEASUREMENT_VERSION_KEY,
        }, built["extra"]

    def test_only_the_index_build_record_carries_it(self):
        """Scope, stated so nobody reads an omission as an oversight: this
        unit collects at the build phase only. The `ingest` record written
        beside it is not a second place to look."""
        import tempfile
        from pathlib import Path
        dataset = tiny_dataset()
        with tempfile.TemporaryDirectory() as tmp:
            driver = OneGauge(dim=dataset.dim, metric=dataset.metric)
            driver.connect()
            ingest, built = _run_build(Path(tmp) / "phases", dataset, driver)
        assert KEY not in (ingest.get("extra") or {})
        assert KEY in built["extra"]

    def test_a_percona_build_records_what_the_global_status_read_returned(
            self, tmp_path, dataset):
        """The whole unit in one assertion, through the real driver: a
        `PerconaDriver` in `post` mode, one bulk `ALTER TABLE`, one `SHOW
        GLOBAL STATUS`, and the value on the record. Against a recording
        cursor -- see the module docstring on what that does not
        establish."""
        driver = wired(PerconaDriver, status={GAUGE: MEMORY_USED})
        built = _run_build(tmp_path / "percona", dataset, driver)[1]

        print(f"statements = {cursor_of(driver).status_statements}")
        print(f"extra.{KEY} = {built['extra'][KEY]}")
        assert built["extra"][KEY] == {GAUGE: MEMORY_USED}
        assert built["engine"] == "percona"
        assert len(cursor_of(driver).status_statements) == 1
        # The gauge is read after the build and before the sizing queries, so
        # it describes the graph the build produced.
        statements = cursor_of(driver).statements
        alter = max(i for i, s in enumerate(statements)
                    if s.upper().startswith("ALTER TABLE"))
        status = statements.index(cursor_of(driver).status_statements[0])
        count = max(i for i, s in enumerate(statements)
                    if s.upper().startswith("SELECT COUNT(*)"))
        assert alter < status < count, statements

    def test_the_value_is_an_int_on_the_record_and_survives_jsonl(self):
        """The record is read back off disk, not out of the object, because
        `Recorder.write` is the only point every ops record passes through
        and JSON is what a report loads."""
        built = _one_build(OneGauge)
        value = built["extra"][KEY][GAUGE]
        assert isinstance(value, int) and not isinstance(value, bool)
        assert value == MEMORY_USED


# ---------------------------------------------------------------------------
# The measurement-version verdict
# ---------------------------------------------------------------------------

class TestNoMeasurementVersionBumpIsOwed:
    """Decided against the charter above `OPS_MEASUREMENT_VERSION` in
    `harness/metrics/records.py` -- "bump in any change that alters what an
    ops number MEANS", extended by version 8 to re-partitioning -- rather than
    copied from anywhere.

    No existing number moves and no identity component changes. The version 8
    block names the identity components by symbol: `OPS_UNIT_FIELDS`
    (`report/generate.py`), `OPS_UNIT_KEYS` (`workloads/churn_readers.py`),
    `build_signature` and `memory_stem` (`report/loaders.py`), and
    `charts.series_key` through the first. Every one of them reads NAMED
    COLUMNS -- `m`, `build_mode`, `storage_engine`, `resource_pass`, engine,
    dataset -- except `build_signature`, which reads one `extra` key by name,
    `extra.source_file`, and that key is written by the ann overlay modules
    and not by this harness. So a new `extra` key cannot re-partition
    anything, and the test below pins that rather than asserting it in a
    docstring.

    `ANN_MEASUREMENT_VERSION` is not in question: it is the first field of
    `ann_fingerprint`, which keys `results/annb/<pass>/<fingerprint>/`, and
    nothing in this unit touches the ann path at all. This unit writes to the
    ops records only.

    `docs/11-methodology-roadmap.md` §4.1 reaches the same verdict in its own
    words -- "stamped into `extra`, which is additive and owes no bump".
    """

    def test_the_ops_version_is_unchanged(self):
        assert records.OPS_MEASUREMENT_VERSION == 8

    def test_the_new_key_reaches_no_identity_component(self):
        from report import generate, loaders
        from harness.workloads import churn_readers
        assert KEY not in generate.OPS_UNIT_FIELDS
        assert KEY not in churn_readers.OPS_UNIT_KEYS
        with_status = {"engine": "percona", "dataset": "glove-100-angular",
                       "resource_pass": "normalized", "m": 16,
                       "build_mode": "post", "storage_engine": "InnoDB",
                       "extra": {KEY: {GAUGE: MEMORY_USED}}}
        without = dict(with_status, extra={})
        for fn in (loaders.build_signature, loaders.memory_stem,
                   generate.ops_unit):
            assert fn(with_status) == fn(without), fn.__name__


@pytest.fixture
def dataset():
    return tiny_dataset()
