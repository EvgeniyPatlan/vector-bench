"""Tests for the calibrated report: PR 2 §2.3 and §2.1 item 6's requirement.

What §2.3 adds is a *premise* rather than a result. Concurrency, filtered
search and churn are each measured at one `ef_search` per engine, and once that
ef is chosen per engine instead of declared in the profile, every number in
those three sections depends on a choice the report did not previously show.
The failure mode this file guards is therefore not a crash and not an empty
page: it is a report that renders perfectly while attributing one engine's ef,
status or recall to another, or while quietly publishing the calibration
sweep's own single-client probes as throughput.

Two fixture layers, deliberately, and the split is the reason the assertions
can be trusted:

1. **Real harness output.** `_sweep_records` drives `harness.main.main` with
   `tests/fake_driver.py`, patching only the dataset loader and the driver
   factory. The records that come back are written by the production
   `Recorder`, carry whatever `stamp_run_extra` and `calibrate.status_extra`
   actually put in `extra`, and have the real phase names. Every test of
   `loaders.operating_points` runs against those, so the join cannot be
   passing because a hand-written fixture invented the keys it needed. The
   three statuses are reached by moving `FakeDriver.ef_half`, not by writing a
   status down: `ef_half=0` makes the fake exact and its floor sits above any
   target; a huge `ef_half` puts every grid point below it.

2. **Synthetic records, for rendering only.** `render_markdown` and the chart
   functions are pure functions of a record list, so a record list is their
   honest input. The values in those fixtures are chosen to be mutually
   distinctive -- ef 24 against 240, recall 0.9611 against 0.8123 -- because
   the thing being checked is which record reached which cell, and two engines
   with similar numbers cannot tell a correct table from a transposed one.

**Numbers rendered into text are the highest-risk assertion shape here**, and
this file does not assert them with `in`. `"ef 24" in report` is satisfied by a
table that says `ef 240`; `"1 point(s)"` is satisfied by `"11 point(s)"`. So
`_table_rows` parses the rendered Markdown back into cells and the assertions
compare whole cells for equality. It parses the artifact under test -- it does
not compute the expected rows, which would make the test an echo of the
implementation.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import main as main_mod  # noqa: E402
from harness.metrics import records as records_mod  # noqa: E402
from report import charts, generate, loaders, render  # noqa: E402
from tests.fake_driver import N_QUERIES, loaded_driver, tiny_dataset  # noqa: E402

VB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _can_draw() -> bool:
    return charts.have_matplotlib()


needs_matplotlib = pytest.mark.skipif(
    not _can_draw(),
    reason="draws a chart; matplotlib lives in the bench images, not on the host")


# ---------------------------------------------------------------------------
# Layer 1: records written by the real harness against the fake driver
# ---------------------------------------------------------------------------

#: `--ef-search` is deliberately not a grid value, so any grid value appearing
#: as a chosen ef proves the substitution happened rather than the literal
#: surviving.
FIXED_EF = 100
GRID = "8,18,98,198"


def _sweep_records(tmp_path, ef_half: float, *, grid: str = GRID,
                   engine: str = "mariadb", m: int = 16,
                   resource_pass: str = "normalized",
                   extra_argv: Sequence[str] = ()) -> List[Dict[str, Any]]:
    """Drive the real harness with the fake driver; return the records.

    Only `load_dataset` and `get_driver` are replaced. Argument parsing, the
    calibration call, the substitution of the calibrated ef into the
    downstream workloads and `Recorder.declare_run_extra` are all the shipped
    code, so the `extra` keys these tests read are the ones a real run writes.
    """
    dataset = tiny_dataset()
    fake = loaded_driver(dataset, ef_half=ef_half)
    saved = (main_mod.load_dataset, main_mod.get_driver)
    out = str(tmp_path / f"ops-{engine}-{dataset.name}-{resource_pass}-m{m}.jsonl")
    try:
        main_mod.load_dataset = lambda directory, name: dataset
        main_mod.get_driver = lambda *a, **kw: fake.clone()
        argv = [
            "--engine", engine, "--host", "h", "--port", "0",
            "--dataset", dataset.name, "--datasets-dir", str(tmp_path),
            "--m", str(m), "--run-id", "run-0", "--output", out,
            "--resource-pass", resource_pass,
            "--cache-dir", str(tmp_path / f"cache-{engine}-{m}"),
            "--ef-search", str(FIXED_EF), "--max-queries", str(N_QUERIES),
            "--client-counts", "1", "--concurrency-duration", "0.2",
            "--concurrency-queries", "20",
            "--churn-fractions", "0.10", "--selectivities", "0.50",
            "--workloads", "concurrency",
            "--ef-mode", "calibrated", "--ef-grid", grid,
        ] + list(extra_argv)
        assert main_mod.main(argv) == 0, "the harness run failed"
    finally:
        main_mod.load_dataset, main_mod.get_driver = saved
    return [json.loads(line) for line in open(out)]


#: `ef_half` values that land the sweep on each of the three statuses. Chosen
#: from `FakeDriver`'s documented model p(ef) = ef / (ef + ef_half) rather than
#: by trying numbers: at ef_half=2 the grid above gives 0.800, 0.900, 0.980 and
#: 0.990, so 98 is the first confirmed point above a 0.95 target and sits ~4.8
#: standard errors clear of it over 500 Bernoulli draws. ef_half=0 makes the
#: fake exact, so its smallest grid point is already at 1.0 -- more than the
#: 0.02 margin above the target. A huge ef_half puts p(198) near zero.
EF_HALF_OK = 2.0
EF_HALF_FLOOR = 0.0
EF_HALF_UNREACHED = 100_000.0


# ---------------------------------------------------------------------------
# Layer 2: synthetic records, for the renderers
# ---------------------------------------------------------------------------

def _record(engine: str, phase: str, **kw) -> Dict[str, Any]:
    row = {"run_id": "r", "engine": engine, "phase": phase,
           "dataset": "glove-100-angular", "m": 16, "build_mode": "post",
           "storage_engine": "InnoDB", "resource_pass": "normalized",
           "k": 10, "timestamp": "2026-01-01T00:00:00Z"}
    row.update(kw)
    return row


def _curve(engine: str, points: Sequence[tuple], *, target: float = 0.95,
           grid_size: Optional[int] = None, **kw) -> List[Dict[str, Any]]:
    """Calibration records for one unit, from (ef, recall) pairs."""
    size = grid_size if grid_size is not None else len(points)
    return [
        _record(engine, "ef_calibration", ef_search=ef, recall_at_k=recall,
                qps=11.0 + i, clients=1, vector_index_used=True,
                extra={"effective_ef": ef, "target_recall": target,
                       "reached_target": recall >= target,
                       "grid_position": i, "grid_size": size,
                       "warmup_queries": 20}, **kw)
        for i, (ef, recall) in enumerate(points)
    ]


def _downstream(engine: str, ef: int, status: str, *, target: float = 0.95,
                clients: int = 8, qps: float = 1234.0,
                recall: float = 0.5, **kw) -> Dict[str, Any]:
    """One concurrency record stamped the way `stamp_run_extra` stamps it.

    `recall` defaults to a value no calibration fixture here uses, so a table
    that reads the Recall@k column off this record instead of off the curve
    prints a number that appears nowhere else and is recognisable on sight.
    """
    return _record(engine, "concurrency", ef_search=ef, clients=clients,
                   qps=qps, recall_at_k=recall, latency_p50_ms=1.0,
                   latency_p95_ms=2.0, latency_p99_ms=3.0,
                   extra={"ef_mode": "calibrated", "target_recall": target,
                          "calibration_status": status,
                          "scaling_efficiency": 0.9, "query_pool": 10000,
                          "query_reuse": 1.2}, **kw)


MANIFEST = {
    "run_id": "run-0", "started_at": "2026-01-01T00:00:00Z",
    "finished_at": "2026-01-01T01:00:00Z", "status": "completed",
    "host": {"cpu": {}}, "engines": {},
    "config": {"resource_pass": "normalized", "ops_measurement_version": 4,
               "profile": {"name": "quick", "description": "d",
                           "ops": {"workloads": ["concurrency"]}}},
}


def _render(records: List[Dict[str, Any]], *,
            manifest: Optional[Dict[str, Any]] = None,
            chart_paths: Optional[Dict[str, Dict[str, str]]] = None) -> str:
    manifest = manifest if manifest is not None else MANIFEST
    summary = generate.summarize(records, manifest)
    return render.render_markdown(manifest, summary, records,
                                  chart_paths or {}, "T2b")


# ---------------------------------------------------------------------------
# Parsing the rendered artifact
# ---------------------------------------------------------------------------

#: The operating-point table's header, in order. Matched in full rather than
#: on its first cell: §1's "Engines under test" table also starts with
#: "Engine", and a parser keyed on the first cell alone collected both tables
#: into one list -- after which `_row_for` found two MariaDB rows and every
#: cell index pointed at the wrong column. Matching the whole header also pins
#: the column ORDER, so the index-based assertions below cannot be satisfied
#: by a table whose columns were rearranged.
OPERATING_HEADER = ["Engine", "Dataset", "Pass", "M", "Build",
                    "Chosen ef_search", "Effective ef", "Recall@k", "Target",
                    "Status", "Grid points"]
ENVIRONMENT_HEADER = ["Property", "Value"]


def _table_rows(markdown: str, header: Sequence[str]) -> List[List[str]]:
    """Cells of the Markdown table whose header row is exactly `header`.

    Parses the rendered report rather than recomputing it. The point is that
    every assertion downstream compares a whole CELL for equality: `"24" in
    report` is true of a table that says `240`, `"1 point(s)"` is true of
    `"11 point(s)"`, and both shapes have shipped in this repo. A cell is the
    smallest unit that cannot be satisfied by a longer string.
    """
    want = list(header)
    rows: List[List[str]] = []
    collecting = False
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            collecting = False
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if not collecting:
            collecting = cells == want
            continue
        if all(c and set(c) <= {"-"} for c in cells):
            continue
        rows.append(cells)
    return rows


def _operating_rows(markdown: str) -> List[List[str]]:
    return _table_rows(markdown, OPERATING_HEADER)


def _row_for(rows: List[List[str]], label_fragment: str) -> List[str]:
    """The single row whose engine label contains `label_fragment`.

    Asserts uniqueness: a helper that silently returned the first of two
    matches would hide exactly the transposition these tests are looking for.
    """
    found = [r for r in rows if label_fragment in r[0]]
    assert len(found) == 1, f"{label_fragment}: {len(found)} rows, want 1"
    return found[0]


# ---------------------------------------------------------------------------
# §2.3.1 -- loading the sweep's records
# ---------------------------------------------------------------------------

class TestTheSweepsRecordsAreSelectedByPhase:
    """§2.3.1, first half, and finding F-1.

    Every consumer in `report/` filters by explicit phase equality, so
    `ef_calibration` records are read out of the JSONL and then dropped by
    everything downstream. The selector is what stops that, and the regression
    it guards is the selector widening: a `startswith`, an `in`, or a drift
    between the constant the writer uses and the one the reader uses.
    """

    def test_it_selects_the_sweeps_records_and_nothing_else(self, tmp_path):
        rows = _sweep_records(tmp_path, EF_HALF_OK)
        selected = loaders.calibration_records(rows)
        # Both halves asserted. "every calibration record is selected" is
        # satisfied by a selector that returns everything; "nothing else is
        # selected" is satisfied by one that returns nothing.
        assert [r["ef_search"] for r in selected] == [8, 18, 98, 198]
        assert {r["phase"] for r in selected} == {"ef_calibration"}
        assert len(rows) - len(selected) == len(
            [r for r in rows if r["phase"] == "concurrency"])

    def test_the_nothing_else_half_is_tested_against_every_other_phase(self):
        """The "nothing else" half had nothing to exclude.

        The harness fixture above only ever writes `ef_calibration` and
        `concurrency`, so a selector widened to admit `index_build` or
        `ingest` selected more records and the suite stayed green -- the
        appendix chart happens to re-filter on `ef_search`/`recall_at_k`,
        which contains the damage by luck rather than by a guard. Every phase
        in `ALL_PHASES` is offered here, so widening the selector to any of
        them changes this count.
        """
        others = [_record("mariadb", phase, ef_search=24, recall_at_k=0.9,
                          qps=1.0, clients=1)
                  for phase in records_mod.ALL_PHASES
                  if phase != records_mod.PHASE_CALIBRATION]
        sweep = _curve("mariadb", [(24, 0.9611), (48, 0.9702)])
        assert len(others) == len(records_mod.ALL_PHASES) - 1
        selected = loaders.calibration_records(sweep + others)
        assert len(selected) == 2
        assert {r["phase"] for r in selected} == {"ef_calibration"}

    def test_the_selector_follows_the_phase_constant_the_writer_uses(
            self, tmp_path, monkeypatch):
        """The mutant: the reader and the writer disagree about the name.

        `PHASE_CALIBRATION` is imported inside each function that needs it, so
        patching the constant on the records module reaches every reader --
        which is the point. If the selector were comparing against a literal
        string of its own instead, this patch would not move its answer and the
        two could drift apart silently.
        """
        rows = _sweep_records(tmp_path, EF_HALF_OK)
        honest = len(loaders.calibration_records(rows))
        monkeypatch.setattr(records_mod, "PHASE_CALIBRATION", "concurrency")
        mutant = loaders.calibration_records(rows)
        assert honest == 4
        assert {r["phase"] for r in mutant} == {"concurrency"}
        assert len(mutant) != honest

    def test_an_empty_record_set_selects_nothing_rather_than_everything(self):
        assert loaders.calibration_records([]) == []


class TestCalibrationRecordsStayOutOfTheThroughputNumbers:
    """Finding F-1's other half: the consumers that must keep ignoring them.

    A calibration point carries `clients=1` and a `qps`, so it satisfies every
    field test the concurrency aggregation applies. Admitting it would add up
    to twelve single-client probes per engine to the published throughput --
    measured over `--max-queries` fresh queries rather than the shared pool,
    which is the one comparison `extra.query_pool` exists to prevent.
    """

    def test_the_concurrency_bucket_holds_only_concurrency_records(
            self, tmp_path):
        rows = _sweep_records(tmp_path, EF_HALF_OK)
        summary = generate.summarize(rows, MANIFEST)
        assert {r["phase"] for r in summary["concurrency"]} == {"concurrency"}
        assert len(summary["concurrency"]) == 1

    def test_admitting_them_would_move_the_published_qps(self, tmp_path,
                                                         monkeypatch):
        """The mutant is the sixth entry the phase map does not have.

        Reported as two values rather than as a verdict, and the values that
        matter are the QPS figures the concurrency section publishes. Honestly
        the bucket holds one point, the concurrency window's. Under the mutant
        it holds the sweep's four single-client probes -- measured over
        `--max-queries` fresh queries rather than the shared pool, at roughly
        half the throughput -- and the real measurement is not in it at all.
        Both render as a concurrency table without complaint.
        """
        rows = _sweep_records(tmp_path, EF_HALF_OK)
        honest = generate.summarize(rows, MANIFEST)["concurrency"]
        honest_qps = sorted(r["qps"] for r in honest)
        monkeypatch.setattr(
            generate, "_PHASE_GROUPS",
            generate._PHASE_GROUPS + (("ef_calibration", "concurrency"),))
        mutant = generate.summarize(rows, MANIFEST)["concurrency"]
        mutant_qps = sorted(r["qps"] for r in mutant)
        assert len(honest) == 1, f"honest bucket: {honest_qps}"
        assert {r["phase"] for r in honest} == {"concurrency"}
        assert len(mutant) == 4, f"mutant bucket: {mutant_qps}"
        assert {r["phase"] for r in mutant} == {"ef_calibration"}
        # Disjoint: not one published QPS survives the mutation.
        assert not set(honest_qps) & set(mutant_qps)

    def test_the_version_guard_does_see_the_sweeps_records(self):
        """`_is_ops_record` is an INEQUALITY, so it includes the new phase.

        Pinned because `calibration_records`' docstring used to list it among
        the consumers that drop `ef_calibration` on the floor -- a
        counter-example inside its own enumeration. The sweep's records are
        ops records, written by the same harness under the same
        `OPS_MEASUREMENT_VERSION`, and they must be version-checked with the
        rest of the ops path rather than exempted from it.
        """
        from report import measurement_versions as mv

        row = _record("mariadb", "ef_calibration", ef_search=8,
                      recall_at_k=0.9, qps=1.0, clients=1,
                      extra={"ops_measurement_version": 4})
        assert mv._is_ops_record(row) is True
        assert {k: len(v) for k, v in
                mv.ops_measurement_versions([row]).items()} == {4: 1}

    def test_a_calibration_point_that_missed_the_index_is_still_reported(
            self, tmp_path):
        """The one generic consumer, and it is generic on purpose.

        `summarize`'s plan-failure loop has no phase filter, so a calibration
        probe that ran without the vector index in the plan is surfaced like
        any other full scan. That is the correct behaviour for this consumer
        and the opposite of the one above, which is why F-1 has to be decided
        per consumer rather than globally.
        """
        rows = _sweep_records(tmp_path, EF_HALF_OK)
        calib = loaders.calibration_records(rows)
        calib[0]["vector_index_used"] = False
        failures = generate.summarize(rows, MANIFEST)["plan_failures"]
        assert [(f["phase"], f["ef_search"]) for f in failures] == [
            ("ef_calibration", 8)]


# ---------------------------------------------------------------------------
# §2.3.1 -- the summary
# ---------------------------------------------------------------------------

class TestTheOperatingPointSummary:
    """§2.3.1, second half: chosen ef, recall and status per unit.

    Driven entirely off real harness output. The statuses are reached through
    `FakeDriver.ef_half`, so the status in the summary was produced by
    `calibrate._choose` running over a measured curve rather than written down
    by the test.
    """

    def test_the_chosen_ef_is_the_one_the_workloads_measured_at(self, tmp_path):
        rows = _sweep_records(tmp_path, EF_HALF_OK)
        [point] = loaders.operating_points(rows)
        downstream = [r for r in rows if r["phase"] == "concurrency"]
        # Equality against the records, and the two values differ from the
        # profile's literal: `--ef-search 100` is not a grid value, so `98`
        # here cannot be the flag surviving.
        assert point["ef_search"] == 98
        assert {r["ef_search"] for r in downstream} == {98}
        assert point["ef_search"] != FIXED_EF
        assert point["status"] == "ok"
        assert point["target_recall"] == 0.95

    def test_the_recall_comes_from_the_sweep_record_at_the_chosen_ef(
            self, tmp_path):
        """The sweep's recall AT the chosen ef, not at any other grid point.

        The docstring here used to promise that this test also distinguishes
        the sweep's recall from the concurrency window's. It does not: against
        the fake driver the two are close and nothing here asserts they
        differ. That guard is
        `TestTheOperatingPointTableCells::test_a_wrong_join_is_caught_rather_than_rendering_cleanly`,
        which runs a mutant reading the recall off the downstream record and
        watches the published cell move from 0.8456 to 0.5000. What this test
        pins is the other half: of the four measured grid points, the one the
        column reports is the chosen one.
        """
        rows = _sweep_records(tmp_path, EF_HALF_OK)
        [point] = loaders.operating_points(rows)
        curve = {r["ef_search"]: r["recall_at_k"] for r in rows
                 if r["phase"] == "ef_calibration"}
        assert sorted(curve) == [8, 18, 98, 198]
        assert point["ef_search"] == 98
        assert point["recall_at_k"] == curve[98]
        # And not any of the other three, which a `curve[0]` or `curve[-1]`
        # implementation would have reported. The fake's error model is
        # monotone, so all four differ.
        assert len(set(curve.values())) == 4
        assert point["recall_at_k"] not in (curve[8], curve[18], curve[198])

    def test_a_floor_above_the_target_chooses_the_smallest_grid_ef(
            self, tmp_path):
        rows = _sweep_records(tmp_path, EF_HALF_FLOOR)
        [point] = loaders.operating_points(rows)
        assert point["status"] == "floor_above_target"
        assert point["ef_search"] == 8
        assert point["recall_at_k"] == 1.0
        assert point["recall_at_k"] > point["target_recall"] + 0.02

    def test_an_unreached_target_chooses_the_largest_grid_ef(self, tmp_path):
        rows = _sweep_records(tmp_path, EF_HALF_UNREACHED)
        [point] = loaders.operating_points(rows)
        assert point["status"] == "target_unreached"
        assert point["ef_search"] == 198
        assert point["recall_at_k"] < point["target_recall"]
        # The grid ran to the end: early stop needs two consecutive points at
        # the target, so a sweep that found none measured every point.
        assert point["points_measured"] == point["grid_size"] == 4

    def test_the_grid_column_distinguishes_an_early_stop_from_a_short_grid(
            self, tmp_path):
        """"chose 8 out of 2 measured" and "out of 4 declared" differ."""
        rows = _sweep_records(tmp_path, EF_HALF_FLOOR)
        [point] = loaders.operating_points(rows)
        assert (point["points_measured"], point["grid_size"]) == (2, 4)

    def test_the_two_halves_of_the_join_agree_on_the_unit_key(self, tmp_path):
        """If they did not, every row would come out as two half-rows.

        A silent failure: the table would render, with the ef and status in one
        row and the recall in another, both naming the same engine.
        """
        rows = _sweep_records(tmp_path, EF_HALF_OK)
        keys = {loaders.operating_point_key(r) for r in rows}
        assert len(keys) == 1, f"the join would split: {keys}"
        assert len(loaders.operating_points(rows)) == 1

    def test_fixed_mode_claims_no_operating_point(self, tmp_path):
        """`stamp_run_extra` drops a None, so no record carries a status.

        This is the empty-input case the table must handle explicitly rather
        than by rendering an empty table -- see
        `TestTheOperatingPointTableIsExplicitWhenThereIsNothingToSay`.
        """
        rows = _sweep_records(tmp_path, EF_HALF_OK,
                              extra_argv=["--ef-mode", "fixed"])
        assert loaders.calibration_records(rows) == []
        assert loaders.operating_points(rows) == []
        assert loaders.calibration_targets(rows) == []
        assert {r["ef_search"] for r in rows} == {FIXED_EF}


class TestTheUnitKeyDoesNotCollapseASweptProfile:
    """The spec's literal key is not unique, and the collapse is silent.

    §2.3.1 says "keyed by (engine, dataset, pass)". `orchestrator/cli.py`
    invokes the harness once per (M, build_mode, storage_engine), each with its
    own sweep, so `config/profiles/m-sweep.yml` -- `m_values: [6, 8, 16, 24,
    32]` -- calibrates five times per engine, dataset and pass. Keyed by the
    literal triple, five independent operating points become one row holding
    whichever the iteration order reached last.
    """

    def test_two_m_values_are_two_operating_points(self, tmp_path):
        rows = (_sweep_records(tmp_path, EF_HALF_OK, m=16)
                + _sweep_records(tmp_path, EF_HALF_FLOOR, m=32))
        points = loaders.operating_points(rows)
        assert [(p["m"], p["ef_search"], p["status"]) for p in points] == [
            (16, 98, "ok"), (32, 8, "floor_above_target")]

    def test_the_literal_triple_would_lose_one_of_them(self, tmp_path,
                                                       monkeypatch):
        """The mutant is the spec's own key, injected at run time.

        Two values, not a verdict: the honest derivation reports two operating
        points, at two efs and two statuses. Under the mutant they merge into
        a single row — and what that row says is worth stating exactly,
        because it is not what I first assumed. The two halves disagree about
        both the ef and the status, so the conflict guard fires and the row
        reports NEITHER: status `None` with a `status_conflict`, ef `None`
        with an `ef_conflict`. So the collapse does not publish one engine's
        status as the other's; it destroys both, and §2.1.6's "cannot be
        slowed down to the target" warning goes with them, because the
        footnote keys on a status this row no longer has.

        The earlier assertion here was `len(sorted(...)) == 1`, which is a
        tautology given the `len(mutant) == 1` above it and would have passed
        whatever the row said.
        """
        rows = (_sweep_records(tmp_path, EF_HALF_OK, m=16)
                + _sweep_records(tmp_path, EF_HALF_FLOOR, m=32))
        honest = loaders.operating_points(rows)
        monkeypatch.setattr(
            loaders, "operating_point_key",
            lambda r: (r.get("engine"), r.get("dataset"),
                       r.get("resource_pass")))
        mutant = loaders.operating_points(rows)
        assert len(honest) == 2, [p["status"] for p in honest]
        assert sorted(p["status"] for p in honest) == [
            "floor_above_target", "ok"]
        assert sorted(p["ef_search"] for p in honest) == [8, 98]
        assert len(mutant) == 1, [p["status"] for p in mutant]
        [merged] = mutant
        assert merged["status"] is None
        assert merged["status_conflict"] == ["floor_above_target", "ok"]
        assert merged["ef_search"] is None
        assert merged["ef_conflict"] == [8, 98]
        # And the §2.1.6 warning is gone from the page, which is the damage.
        assert "cannot be slowed down to the target" in _render_points(honest)
        assert "cannot be slowed down to the target" not in _render_points(
            mutant)

    def test_the_m_column_sorts_numerically(self):
        """A swept-M table is read for a trend down the column.

        `_operating_point_sort_key` stringified every axis, so m-sweep's five
        values came out 16, 24, 32, 6, 8 -- in the profile this key's own
        docstring cites as its reason for existing. Asserted on the rendered
        CELLS, so the fix has to reach the page and not just the sort key.
        """
        rows = []
        for m in (6, 8, 16, 24, 32):
            rows += [dict(r, m=m) for r in _curve("mariadb", [(24, 0.9611)])]
            rows.append(dict(_downstream("mariadb", 24, "ok"), m=m))
        cells = _operating_rows(_render(rows))
        assert [c[3] for c in cells] == ["6", "8", "16", "24", "32"]

    def test_two_passes_are_two_operating_points(self, tmp_path):
        """`build_signature` omits `resource_pass`, so the key adds it.

        For the MySQL-family engines nothing else in the signature separates
        the two passes: `config/resources/tuned.yml` pins
        `mariadb_storage_engines: [InnoDB]` and MariaDB exposes no
        `ef_construction`, so its normalized and tuned sweeps agree on every
        field the signature covers. They are still two sweeps on two different
        memory and CPU budgets, and merging them would report one ef for both.
        """
        rows = (_sweep_records(tmp_path, EF_HALF_OK, resource_pass="normalized")
                + _sweep_records(tmp_path, EF_HALF_FLOOR, resource_pass="tuned"))
        points = loaders.operating_points(rows)
        assert [(p["resource_pass"], p["ef_search"]) for p in points] == [
            ("normalized", 98), ("tuned", 8)]


class TestConflictingRecordsAreNotResolvedBySilentChoice:
    """Two chosen efs under one key are two measurements shown as one.

    A run directory cannot produce this -- one JSONL file per unit, emptied per
    attempt -- but `--from-records` over a hand-merged file can, and the
    failure that matters is a table printing one of the two as though it were
    the answer.
    """

    def test_two_efs_under_one_key_report_a_conflict_rather_than_a_value(self):
        rows = (_curve("mariadb", [(24, 0.9611)])
                + [_downstream("mariadb", 24, "ok"),
                   _downstream("mariadb", 240, "ok", clients=16)])
        [point] = loaders.operating_points(rows)
        assert point["ef_search"] is None
        assert point["ef_conflict"] == [24, 240]
        # And the recall is withheld too: it is the recall AT the chosen ef,
        # and there is no chosen ef.
        assert point["recall_at_k"] is None

    def test_the_conflict_reaches_the_page(self):
        rows = (_curve("mariadb", [(24, 0.9611)])
                + [_downstream("mariadb", 24, "ok"),
                   _downstream("mariadb", 240, "ok", clients=16)])
        report = _render(rows)
        cells = _row_for(_operating_rows(report), "MariaDB")
        assert "conflict: 24, 240" in cells
        assert "24" not in cells and "240" not in cells


# ---------------------------------------------------------------------------
# §2.3.2 -- the table
# ---------------------------------------------------------------------------

class TestTheOperatingPointTableComesBeforeConcurrency:
    """§2.3.2: "New table 'Operating point' before the concurrency section".

    Position is the content here. The table is the premise of the three
    sections after it, and a reader who meets the throughput first has already
    read it as an equal-effort comparison by the time the caveat arrives.
    """

    def _records(self):
        return (_curve("mariadb", [(24, 0.9611), (48, 0.9702)])
                + [_downstream("mariadb", 24, "ok")])

    def test_the_markdown_puts_it_before_concurrency_scaling(self):
        report = _render(self._records())
        assert "## 6. Operating point" in report
        assert "## 7. Concurrency scaling" in report
        assert report.index("## 6. Operating point") < report.index(
            "## 7. Concurrency scaling")

    def test_the_html_puts_it_before_concurrency_scaling_too(self):
        """The HTML is built from its own section list.

        Fixing only the Markdown has already shipped in this file: the page a
        reader actually opens kept saying "No data." for workloads the Markdown
        explained.
        """
        summary = generate.summarize(self._records(), MANIFEST)
        page = render.render_html(MANIFEST, summary, self._records(), {}, "T2b")
        assert "6. Operating point" in page
        assert page.index("6. Operating point") < page.index(
            "7. Concurrency scaling")

    def test_every_section_number_is_used_once_and_in_order(self):
        """Inserting a section renumbers the four after it.

        A duplicated "## 6." or a jump from 6 to 8 is the visible half of
        having edited one renderer and not the other.
        """
        report = _render(self._records())
        numbers = [int(line.split(".")[0][3:]) for line in report.splitlines()
                   if line.startswith("## ") and line[3].isdigit()]
        assert numbers == list(range(1, 11))


class TestTheOperatingPointTableCells:
    """§2.3.2: engine, chosen ef, recall, status -- in the right cells.

    Asserted cell by cell against a parse of the rendered table, never with
    `in` against the whole report. Two engines whose numbers differ in a digit
    (`ef 24` against `ef 240`, recall `0.9611` against `0.8123`) are what make
    a transposition visible: a substring assertion for `"24"` passes on a table
    that says `240`, and this repo has shipped seven instances of that shape.
    """

    def _records(self):
        return (
            _curve("mariadb", [(24, 0.9611), (48, 0.9702)])
            + _curve("pgvector", [(240, 0.8123), (480, 0.8456)])
            + [_downstream("mariadb", 24, "ok", qps=1111.0),
               _downstream("pgvector", 480, "target_unreached", qps=2222.0)]
        )

    def test_each_engine_gets_its_own_ef_recall_and_status(self):
        rows = _operating_rows(_render(self._records()))
        maria = _row_for(rows, "MariaDB")
        pg = _row_for(rows, "pgvector")
        assert len(maria) == len(OPERATING_HEADER) == len(pg)
        # Whole-cell equality. `maria[5] == "24"` is false for a cell of "240".
        assert maria[5] == "24"
        assert maria[7] == "0.9611"
        assert maria[9] == "ok"
        assert pg[5] == "480"
        assert pg[7] == "0.8456"
        assert pg[9].startswith("target_unreached")

    def test_the_recall_is_the_one_measured_at_the_chosen_ef(self):
        """Not the curve's first point, not its last, not the other engine's.

        pgvector's curve holds 0.8123 at ef 240 and 0.8456 at ef 480, and the
        chosen ef is 480. A row showing 0.8123 would be the implementation
        taking `curve[0]`; one showing MariaDB's 0.9611 would be a transposed
        join. Both render perfectly.
        """
        rows = _operating_rows(_render(self._records()))
        pg = _row_for(rows, "pgvector")
        assert pg[7] == "0.8456"
        assert pg[7] != "0.8123"
        assert pg[7] != "0.9611"

    def test_the_column_is_not_the_concurrency_windows_recall(self):
        """`_downstream` carries recall 0.5, which no curve here contains."""
        rows = _operating_rows(_render(self._records()))
        assert all(cells[7] not in ("0.5000", "0.5") for cells in rows)

    def test_a_wrong_join_is_caught_rather_than_rendering_cleanly(self):
        """The mutant: read the recall off the record that used the ef.

        Two values: honestly pgvector's Recall@k cell is 0.8456, the sweep's
        figure at the ef it chose; under the mutant it is 0.5000, the
        concurrency window's. Both are four-decimal recalls in the right
        column of a table that renders identically.
        """
        records = self._records()
        honest = _row_for(_operating_rows(_render(records)), "pgvector")[7]

        real = loaders.operating_points

        def mutant_points(rows):
            out = real(rows)
            by_unit = {}
            for r in rows:
                if (r.get("extra") or {}).get("calibration_status"):
                    by_unit[loaders.operating_point_key(r)] = r
            for row in out:
                source = by_unit.get(row["key"])
                if source is not None:
                    row["recall_at_k"] = source.get("recall_at_k")
            return out

        from unittest import mock
        with mock.patch.object(loaders, "operating_points", mutant_points):
            mutant = _row_for(_operating_rows(_render(records)), "pgvector")[7]
        assert honest == "0.8456"
        assert mutant == "0.5000"
        assert honest != mutant

    def test_the_grid_cell_says_how_much_of_the_grid_ran(self):
        rows = (_curve("mariadb", [(24, 0.9611), (48, 0.9702)], grid_size=12)
                + [_downstream("mariadb", 24, "ok")])
        cells = _row_for(_operating_rows(_render(rows)), "MariaDB")
        # Whole cell: "2 of 12" is not satisfied by "12 of 12".
        assert cells[10] == "2 of 12"

    def test_the_effective_ef_is_shown_only_where_it_differs(self):
        """MongoDB cannot search fewer candidates than k, so its "ef 8" is 10.

        Two engines compared at one chosen ef are not comparable without this,
        and a column repeating the requested value on every row is one a reader
        learns to skip.
        """
        same = _curve("mariadb", [(24, 0.9611)])
        differs = _curve("mongodb", [(8, 0.9800)])
        for r in differs:
            r["extra"]["effective_ef"] = 10
        rows = (same + differs + [_downstream("mariadb", 24, "ok"),
                                  _downstream("mongodb", 8, "ok")])
        table = _operating_rows(_render(rows))
        assert _row_for(table, "MariaDB")[6] == "—"
        assert _row_for(table, "MongoDB")[6] == "10"


class TestTheOperatingPointTableIsExplicitWhenThereIsNothingToSay:
    """A table test that passes on empty input tests nothing.

    So this asserts the opposite deliberately: with no calibration records the
    section must carry a sentence, and must NOT carry a table. An empty table
    here would read as "the sweep ran and chose nothing", when the honest
    statement is that the ef came straight out of the profile.
    """

    def test_a_fixed_ef_run_says_so_instead_of_printing_a_table(self):
        rows = [_record("mariadb", "concurrency", ef_search=100, clients=8,
                        qps=1000.0, extra={"ef_mode": "fixed"})]
        report = _render(rows)
        assert "_No ef_search calibration in this run._" in report
        assert "ef_mode: fixed" in report
        # No table: the operating-point header must not appear at all.
        assert "Chosen ef_search" not in report
        assert _operating_rows(report) == []

    def test_it_does_not_fall_back_to_the_generic_no_data_text(self):
        """`_md_table` prints "_No data._" for an empty row list.

        That is the wording this section must not use: it says the measurement
        is missing, where the truth is that a different and defensible policy
        was in force.
        """
        report = _render([_record("mariadb", "concurrency", ef_search=100,
                                 clients=8, qps=1000.0)])
        head = report.split("## 7. Concurrency")[0].split(
            "## 6. Operating point")[1]
        assert "_No data._" not in head

    def test_the_section_still_exists_so_its_absence_is_not_the_signal(self):
        report = _render([_record("mariadb", "concurrency", ef_search=100,
                                  clients=8, qps=1000.0)])
        assert "## 6. Operating point" in report


class TestTheReportSaysTheEngineCannotBeSlowedToTheTarget:
    """§2.1 item 6's literal requirement, which §2.3 exists to satisfy.

    "`floor_above_target`: ... **the report must say this engine cannot be
    slowed down to the target**." A dagger with no sentence behind it reads as
    a minor caveat; the sentence is the deliverable.
    """

    def _records(self):
        return (_curve("mariadb", [(8, 0.9810), (16, 0.9900)], grid_size=12)
                + _curve("pgvector", [(24, 0.9500), (48, 0.9600)])
                + [_downstream("mariadb", 8, "floor_above_target"),
                   _downstream("pgvector", 24, "ok")])

    def test_the_sentence_is_in_the_report(self):
        report = _render(self._records())
        assert "cannot be slowed down to the target" in report

    def test_it_names_the_engine_it_is_true_of_and_not_the_other(self):
        """A footnote that names every engine says nothing about any of them."""
        report = _render(self._records())
        note = report.split("cannot be slowed down to the target")[1]
        note = note.split("\n\n")[0]
        assert "MariaDB" in note
        assert "pgvector" not in note

    def test_it_says_which_direction_the_comparison_is_wrong_in(self):
        """Above the target, so the throughput is understated, not flattered.

        A reader who takes a dagger as "this number is too good" draws the
        opposite conclusion from the right one.
        """
        report = _render(self._records())
        assert "**higher** recall" in report

    def test_the_sentence_is_absent_when_no_engine_earned_it(self):
        """Printed on every report, it trains a reader to skip the footnote.

        The same reasoning `_client_cpu_gap_note` is built on.
        """
        rows = (_curve("mariadb", [(24, 0.9611), (48, 0.9702)])
                + [_downstream("mariadb", 24, "ok")])
        report = _render(rows)
        assert "cannot be slowed down to the target" not in report

    def test_an_unreached_target_gets_the_opposite_warning(self):
        rows = (_curve("pgvector", [(240, 0.8123), (480, 0.8456)])
                + [_downstream("pgvector", 480, "target_unreached")])
        report = _render(rows)
        assert "reached the target at no grid point" in report
        assert "**lower** recall" in report
        # The recall actually achieved, as a whole cell-sized token: "0.8456"
        # and not "0.8123", which is the same engine's curve one point lower.
        note = report.split("reached the target at no grid point")[1]
        assert "0.8456" in note
        assert "0.8123" not in note

    def test_the_two_statuses_do_not_share_a_footnote_mark(self):
        assert render.FLOOR_MARK != render.UNREACHED_MARK
        rows = (_curve("mariadb", [(8, 0.9810)])
                + _curve("pgvector", [(480, 0.8456)])
                + [_downstream("mariadb", 8, "floor_above_target"),
                   _downstream("pgvector", 480, "target_unreached")])
        table = _operating_rows(_render(rows))
        assert render.FLOOR_MARK in _row_for(table, "MariaDB")[9]
        assert render.UNREACHED_MARK in _row_for(table, "pgvector")[9]
        assert render.UNREACHED_MARK not in _row_for(table, "MariaDB")[9]
        assert render.FLOOR_MARK not in _row_for(table, "pgvector")[9]

    def test_the_floor_mark_is_the_dagger_the_spec_names(self):
        """§2.3.3 says `floor_above_target` gets "a dagger".

        Nothing else pinned which glyph goes with which status, so a
        consistent swap of the two would have left the suite green while
        putting the spec's dagger on the opposite condition.
        """
        assert render.FLOOR_MARK == "†"
        assert render.UNREACHED_MARK != render.FLOOR_MARK

    def test_an_engine_measured_but_never_calibrated_gets_a_footnote(self):
        """Item 3's table half: throughput below, no row here.

        Measured before the fix: §6 printed one row for the calibrated engine,
        §7 plotted both, and nothing on the page said the second engine had
        not been held to the floor §6's own introduction tells the reader to
        check.
        """
        rows = (_curve("mariadb", [(24, 0.9611)])
                + [_downstream("mariadb", 24, "ok")]
                + _uncalibrated_concurrency("pgvector"))
        report = _render(rows)
        assert [r[0] for r in _operating_rows(report)] == \
            ["MariaDB 11.8 (MHNSW)"]
        assert "were measured but never calibrated" in report
        note = report.split("were measured but never calibrated")[1]
        note = note.split("\n\n")[0]
        assert "pgvector" in note
        assert render.UNCALIBRATED_MARK in note

    def test_being_calibrated_on_one_dataset_is_not_being_calibrated_on_another(
            self):
        """The table half of the dataset axis, which nothing pinned.

        `_engines_without_an_operating_point` pairs (engine, dataset)
        correctly; keyed on the engine alone, an engine calibrated anywhere
        counts as calibrated everywhere. Measured with that mutation: the
        function returned `[]` instead of
        `[('mariadb', 'sift-128-euclidean')]` and §6's footnote vanished —
        while the sift chart still marked MariaDB `*`, because
        `_calibration_marks` filters the dataset correctly. That is a chart
        asserting an exception the table does not explain, which is the exact
        drift the shared derivation exists to prevent, and the suite was green
        under it. `test_marks_do_not_cross_from_one_dataset_to_another` covers
        the chart half only.
        """
        sift = "sift-128-euclidean"
        rows = (_curve("mariadb", [(24, 0.9611)])
                + [_downstream("mariadb", 24, "ok")]
                # Same engine, other dataset, measured before calibration.
                + _uncalibrated_concurrency("mariadb", dataset=sift)
                + [dict(r, dataset=sift)
                   for r in _curve("pgvector", [(48, 0.9702)])]
                + [dict(_downstream("pgvector", 48, "ok"), dataset=sift)])
        summary = generate.summarize(rows, MANIFEST)
        assert render._engines_without_an_operating_point(summary) == [
            ("mariadb", sift)]
        report = _render(rows)
        assert "were measured but never calibrated" in report
        note = report.split("were measured but never calibrated")[1]
        note = note.split("\n\n")[0]
        # Names the pair, not just the engine: MariaDB IS calibrated on glove.
        assert f"MariaDB 11.8 (MHNSW) on {sift}" in note
        assert "glove-100-angular" not in note
        # And the chart agrees: marked on sift, unmarked on glove.
        assert charts._mark_suffix(
            charts._calibration_marks(rows, sift).get("mariadb")) == \
            " " + render.UNCALIBRATED_MARK
        assert charts._mark_suffix(
            charts._calibration_marks(rows, "glove-100-angular").get(
                "mariadb")) == ""

    def test_that_footnote_is_absent_when_every_engine_was_calibrated(self):
        rows = (_curve("mariadb", [(24, 0.9611)])
                + _curve("pgvector", [(48, 0.9702)])
                + [_downstream("mariadb", 24, "ok"),
                   _downstream("pgvector", 48, "ok")])
        assert "were measured but never calibrated" not in _render(rows)

    def test_a_fixed_ef_run_does_not_list_every_engine_as_uncalibrated(self):
        """On `ef_mode: fixed` the table's one sentence is the whole truth.

        Repeating it per engine would turn one true statement into six, which
        is the footnote-fatigue failure `_client_cpu_gap_note` guards against.
        """
        rows = (_uncalibrated_concurrency("mariadb")
                + _uncalibrated_concurrency("pgvector"))
        report = _render(rows)
        assert "_No ef_search calibration in this run._" in report
        assert "were measured but never calibrated" not in report

    def test_a_row_with_a_status_but_no_curve_is_footnoted_too(self):
        """The mirror of the no-status row, which had no footnote of its own.

        Reachable through `--datasets`, which narrows records after they are
        loaded. The ef and the status are trustworthy; the empty Recall@k has
        to be distinguishable from a measured zero.
        """
        rows = [_downstream("mariadb", 24, "ok")]
        [point] = loaders.operating_points(rows)
        assert (point["ef_search"], point["status"]) == (24, "ok")
        assert point["points_measured"] is None
        report = _render(rows)
        cells = _row_for(_operating_rows(report), "MariaDB")
        assert cells[5] == "24" and cells[9] == "ok"
        assert cells[7] == "—" and cells[10] == "—"
        assert "show an ef and a status but no curve" in report
        assert "not that the recall was zero" in report

    def test_the_intro_does_not_claim_a_confirmation_the_rule_may_skip(self):
        """`calibrate._choose` has a third branch the sentence used to deny.

        Driven through the real `select`: a target first met at the LARGEST
        grid ef returns status `ok` with `confirmed=False`, and `confirmed` is
        on no record, so that row is indistinguishable from a confirmed one.
        The old sentence said the chosen ef "reaches the target recall and is
        confirmed by the next grid point" over both. Reachable at low M --
        `config/resources/tuned.yml` records all three MySQL-family engines
        around 0.74-0.82 at M=6.
        """
        from harness.workloads.calibrate import CalibrationPoint, select

        def points(pairs):
            return [CalibrationPoint(ef_search=ef, effective_ef=ef,
                                     recall_at_k=recall, qps=1.0)
                    for ef, recall in pairs]

        unconfirmed = select(points([(8, 0.80), (16, 0.88), (32, 0.96)]), 0.95)
        confirmed = select(
            points([(8, 0.80), (16, 0.88), (32, 0.96), (64, 0.97)]), 0.95)
        # Same ef, same status, different `confirmed` -- and the report cannot
        # tell them apart, which is why this is fixed in prose.
        assert (unconfirmed.ef_search, unconfirmed.status) == (32, "ok")
        assert (confirmed.ef_search, confirmed.status) == (32, "ok")
        assert unconfirmed.confirmed is False and confirmed.confirmed is True

        report = _render(self._records())
        intro = report.split("## 6. Operating point")[1].split("| Engine")[0]
        assert "where the grid has a next point" in intro
        assert "used unconfirmed" in intro
        # The old, stronger claim must not survive anywhere in the section.
        assert "target recall and is confirmed by the next grid point" \
            not in intro

    def test_the_direction_words_reach_the_html_as_emphasis(self):
        """`_inline_md` converts `**` and has no italic rule at all.

        So `*higher*` arrived on the page a reader actually opens as literal
        asterisks around the one token a misreading inverts. Asserted on the
        HTML, because that was the half the Markdown assertions could not see:
        `"**higher** recall" in report` is true of Markdown that renders as
        plain asterisks.
        """
        summary = generate.summarize(self._records(), MANIFEST)
        page = render.render_html(MANIFEST, summary, self._records(), {},
                                  "T2b")
        assert "<strong>higher</strong>" in page
        assert "*higher*" not in page
        unreached = (_curve("pgvector", [(480, 0.8456)])
                     + [_downstream("pgvector", 480, "target_unreached")])
        page = render.render_html(
            MANIFEST, generate.summarize(unreached, MANIFEST), unreached, {},
            "T2b")
        assert "<strong>lower</strong>" in page
        assert "*lower*" not in page

    def test_the_recall_columns_are_distinguished_in_prose(self):
        """Two Recall@k columns, two sections apart, two measurements.

        Without the note a reader compares them and reads the difference as an
        inconsistency in the report.
        """
        report = _render(self._records())
        assert "Recall@k here is the calibration sweep's" in report


# ---------------------------------------------------------------------------
# §2.3.3 -- the concurrency chart
# ---------------------------------------------------------------------------

class TestTheConcurrencyChartTitle:
    """§2.3.3's title, with the configurable target handled honestly.

    Not marked `needs_matplotlib` at class level: the title is a string built
    from records and nothing here draws until the last test, which carries the
    marker itself. Marking the class would skip seven assertions about what
    the report claims on exactly the hosts this repo promises to support --
    python3 and PyYAML, no matplotlib.

    The spec writes the title with a literal `0.95`. That is
    `DEFAULT_TARGET_RECALL`, not a constant: `--target-recall` is a flag and
    `extra.target_recall` is on every record of a calibrated run. The title is
    therefore built from the records, which renders the spec's exact string
    whenever the run used the default and tells the truth when it did not.
    """

    DATASET = "glove-100-angular"

    def _records(self, target=0.95):
        return (_curve("mariadb", [(24, 0.9611)], target=target)
                + [_downstream("mariadb", 24, "ok", target=target),
                   _downstream("mariadb", 24, "ok", target=target, clients=2)])

    def _title(self, records, dataset=None):
        return charts._qps_panel_title(records, dataset or self.DATASET)

    def test_the_default_target_renders_the_specs_own_string(self):
        assert self._title(self._records()) == "QPS at recall ≥ 0.95"

    def test_a_different_target_is_named_rather_than_rounded_to_the_default(
            self):
        """The regression: 0.95 hard-coded in the title.

        Two values: the honest title for a `--target-recall 0.90` run says
        0.9; a hard-coded one says 0.95, a floor none of its engines was held
        to.
        """
        honest = self._title(self._records(target=0.90))
        assert honest == "QPS at recall ≥ 0.9"
        assert "0.95" not in honest

    def test_a_three_decimal_target_is_printed_exactly(self):
        """Two decimals moves the number in BOTH directions, measured.

        `{0.999:.2f}` is `1.00`, which captions the chart as a claim of perfect
        recall. `{0.995:.2f}` is `0.99`, which understates the floor the
        engines were held to -- 0.995 is slightly below 0.995 in binary, so it
        rounds down. Both targets are asserted, because a test using only
        0.995 passes a formatter that rounds up and only up.
        """
        assert self._title(self._records(target=0.995)) == \
            "QPS at recall ≥ 0.995"
        assert self._title(self._records(target=0.999)) == \
            "QPS at recall ≥ 0.999"
        # `_fmt`'s two decimals, which every other number in these tables
        # uses, is what would have produced those two values.
        assert render._fmt(0.999, 2) == "1.00"
        assert render._fmt(0.995, 2) == "0.99"

    def test_the_table_spells_the_target_the_same_way_the_chart_does(self):
        """One target, named twice, two sections apart.

        The Target column went through `_fmt(..., 2)` first, so a 0.999 run had
        a chart saying "recall ≥ 0.999" above a table saying "1.00" -- and the
        table's version was a claim of perfect recall.
        """
        for target, want in ((0.95, "0.95"), (0.9, "0.9"),
                             (0.995, "0.995"), (0.999, "0.999")):
            rows = (_curve("mariadb", [(24, 0.9611)], target=target)
                    + [_downstream("mariadb", 24, "ok", target=target)])
            cells = _row_for(_operating_rows(_render(rows)), "MariaDB")
            # Whole cell, and the whole title: `want in title` would be
            # satisfied for 0.9 by a title naming 0.95.
            assert cells[8] == want, (target, cells[8])
            assert self._title(rows) == f"QPS at recall ≥ {want}"

    def test_a_fixed_ef_run_claims_no_recall_floor(self):
        """The title is a claim. With no calibration there is nothing to claim.

        A chart captioned "QPS at recall >= 0.95" over a run where every engine
        sat at the profile's literal ef asserts a comparison the run did not
        make -- and on these implementations one ef_search buys recalls 0.18
        apart.
        """
        title = self._title(
            [_record("mariadb", "concurrency", ef_search=100, clients=8,
                     qps=1.0, extra={"ef_mode": "fixed"})])
        assert title == "QPS at a fixed ef_search — recall differs per engine"

    def test_two_targets_in_one_report_name_both(self):
        """`--from-records` over a merged file can hand the report two.

        Naming either one would be false about the other half of the chart.

        Compared as a WHOLE TITLE. This read `"0.9" in title and "0.95" in
        title`, and `"0.9"` is a prefix of `"0.95"`: a title naming only 0.95
        satisfied both halves, so a mutant dropping half the information from
        a title whose entire job is not to be false about either half left the
        suite green. That is the substring-satisfied-by-a-longer-string shape
        this file's own parser exists to avoid, three lines from an assertion
        that avoids it correctly.
        """
        title = self._title(
            self._records(target=0.90) + self._records(target=0.95))
        assert title == "QPS at mixed recall targets (0.9, 0.95)"

    def test_one_datasets_target_does_not_title_anothers_chart(self):
        """`_qps_panel_title` was computed over every record, not the panel's.

        `ops.target_recall` is one profile-level value, so a single run cannot
        vary it across datasets -- but a `--from-records` rebuild over two
        merged runs can, and it titled the glove panel "QPS at mixed recall
        targets (0.8, 0.95)" when every record plotted on it had been held to
        0.95.
        """
        other = [dict(r, dataset="sift-128-euclidean")
                 for r in self._records(target=0.80)]
        both = self._records(target=0.95) + other
        assert loaders.calibration_targets(both) == [0.8, 0.95]
        assert self._title(both) == "QPS at recall ≥ 0.95"
        assert self._title(both, "sift-128-euclidean") == \
            "QPS at recall ≥ 0.8"

    def test_an_engine_that_was_never_calibrated_degrades_the_title(self):
        """One calibrated engine plus one resumed from a fixed-ef pass.

        Measured before the fix: §6 printed one row, §7 plotted two engines,
        the panel said "QPS at recall ≥ 0.95", `_calibration_marks` was `{}`
        and no footnote appeared — so pgvector's curve sat on a panel titled
        with a floor it had never been held to. The title must say the floor
        does not cover every line.
        """
        title = self._title(self._records() + _uncalibrated_concurrency())
        assert title == (f"QPS at recall ≥ 0.95 — not for every engine "
                         f"({render.UNCALIBRATED_MARK})")

    def test_a_fully_calibrated_chart_is_not_degraded(self):
        """The qualifier must not appear on every chart.

        Printed unconditionally it would train a reader to ignore it, which is
        the reasoning `_client_cpu_gap_note` is already built on.
        """
        assert self._title(self._records()) == "QPS at recall ≥ 0.95"
        assert render.UNCALIBRATED_MARK not in self._title(self._records())

    @needs_matplotlib
    def test_the_title_is_on_the_qps_panel_and_not_the_whole_figure(
            self, tmp_path):
        """The other two panels are not throughput at a recall.

        Read back off the Axes rather than off the file, so the assertion is
        about which panel carries it.
        """
        records = self._records()
        paths = charts.concurrency(records, "glove-100-angular",
                                   str(tmp_path), "concurrency-x")
        assert paths
        titles = _axes_titles(records, "glove-100-angular")
        assert titles[0] == "QPS at recall ≥ 0.95"
        assert titles[1] == "" and titles[2] == ""


def _uncalibrated_concurrency(engine: str = "pgvector",
                              dataset: str = "glove-100-angular"
                              ) -> List[Dict[str, Any]]:
    """Concurrency records from a pass that ran before calibration existed.

    `stamp_run_extra` drops a None, so in `fixed` mode no `target_recall` and
    no `calibration_status` are written at all -- these records carry only
    `ef_mode`. Reached without hand-editing: checkpoints are per (pass,
    engine, dataset, phase), so flipping `ops.ef_mode` to `calibrated` and
    resuming keeps exactly these for every engine already measured.
    """
    return [_record(engine, "concurrency", dataset=dataset, ef_search=100,
                    clients=clients, qps=qps, recall_at_k=0.77,
                    latency_p50_ms=1.0, latency_p95_ms=2.0,
                    latency_p99_ms=3.0, extra={"ef_mode": "fixed"})
            for clients, qps in ((1, 900.0), (2, 1500.0))]


def _render_points(points: List[Dict[str, Any]]) -> str:
    """The operating-point section rendered from a given list of rows.

    For the cases where the rows are the subject and the records behind them
    are not -- comparing what the page says about an honest derivation against
    what it says about a mutated one.
    """
    return render._operating_point_table({"operating_points": points})


def _run_dir_with(tmp_path, records: List[Dict[str, Any]]):
    """A run directory the real `generate.main` will accept.

    One `ops-mariadb-*.jsonl` plus the manifest the generator refuses to work
    without. The filename matters: `load_ops_records` relabels every record
    with the engine the filename names, so a stem that does not start with a
    known engine would silently rewrite the fixture's engines to None.
    """
    run_dir = tmp_path / "run-t2b"
    run_dir.mkdir(exist_ok=True)
    stem = "ops-mariadb-glove-100-angular-normalized-m16-post.jsonl"
    manifest = json.loads(json.dumps(MANIFEST))
    manifest["phases"] = []
    manifest["config"]["resolved_resources"] = {}
    # The manifest's declared version has to match the records', or
    # `refuse_mixed_measurement_versions` refuses the directory and
    # `generate.main` exits 2 before drawing anything -- which is the guard
    # working, and it caught this fixture first. Taken from the constant on
    # both sides so a bump cannot make these two disagree.
    version = records_mod.OPS_MEASUREMENT_VERSION
    manifest["config"]["ops_measurement_version"] = version
    stamped = []
    for r in records:
        extra = dict(r.get("extra") or {})
        extra[records_mod.OPS_MEASUREMENT_VERSION_KEY] = version
        stamped.append({**r, "extra": extra})
    with open(run_dir / stem, "w") as fh:
        for r in stamped:
            fh.write(json.dumps(r) + "\n")
    with open(run_dir / "run-manifest.json", "w") as fh:
        json.dump(manifest, fh)
    return run_dir


def _axes_titles(records, dataset):
    """Panel titles of the concurrency figure, without saving it.

    Re-drawn through the same code path, with the figure closed afterwards.
    Reading the titles off the Axes rather than out of the SVG is what makes
    the assertion about which panel carries the title.
    """
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3)
    try:
        axes[0].set_title(charts._qps_panel_title(records, dataset))
        return [ax.get_title() for ax in axes]
    finally:
        plt.close(fig)


@needs_matplotlib
class TestTheConcurrencyChartMarksTheEnginesItCannotCompare:
    """§2.3.3's marks and footnote, read off the Axes rather than the file.

    An SVG assertion would be a substring assertion against a text blob -- the
    shape this file exists to avoid. The legend labels and the figure text are
    the actual objects.
    """

    def _records(self):
        return (
            _curve("mariadb", [(8, 0.9810)])
            + _curve("pgvector", [(480, 0.8456)])
            + _curve("alisql", [(24, 0.9611)])
            + [_downstream("mariadb", 8, "floor_above_target"),
               _downstream("mariadb", 8, "floor_above_target", clients=2),
               _downstream("pgvector", 480, "target_unreached"),
               _downstream("pgvector", 480, "target_unreached", clients=2),
               _downstream("alisql", 24, "ok"),
               _downstream("alisql", 24, "ok", clients=2)]
        )

    def _labels(self, records, dataset="glove-100-angular"):
        import matplotlib.pyplot as plt
        charts.concurrency(records, dataset, "/tmp", "unused")
        # `concurrency` closes its own figure in `_save`, so the labels are
        # rebuilt here from the same two helpers the chart uses. Asserting on
        # those directly is what pins WHICH engine got WHICH mark; the chart
        # test below checks the wiring is actually in the plot call.
        plt.close("all")
        marks = charts._calibration_marks(records, dataset)
        return {engine: charts._mark_suffix(marks.get(engine))
                for engine in ("mariadb", "pgvector", "alisql")}

    def test_only_the_floor_engine_gets_the_dagger(self):
        suffixes = self._labels(self._records())
        assert suffixes["mariadb"] == " " + charts.FLOOR_MARK
        assert suffixes["alisql"] == ""
        assert charts.FLOOR_MARK not in suffixes["pgvector"]

    def test_the_unreached_engine_gets_a_different_mark(self):
        suffixes = self._labels(self._records())
        assert suffixes["pgvector"] == " " + charts.UNREACHED_MARK
        assert charts.UNREACHED_MARK not in suffixes["mariadb"]

    def test_the_legend_label_actually_carries_the_mark(self, tmp_path):
        """The mark must reach the plot call, not only the helper.

        Drawn through `charts.concurrency` with the legend read back off the
        Axes before the figure is saved and closed, by patching `_save` to
        capture it.
        """
        captured: Dict[str, Any] = {}
        real_save = charts._save

        def capture(fig, out_dir, stem):
            captured["labels"] = list(
                fig.axes[0].get_legend_handles_labels()[1])
            captured["texts"] = [t.get_text() for t in fig.texts]
            return real_save(fig, out_dir, stem)

        from unittest import mock
        with mock.patch.object(charts, "_save", capture):
            charts.concurrency(self._records(), "glove-100-angular",
                               str(tmp_path), "concurrency-x")
        labels = captured["labels"]
        assert len(labels) == 3, labels
        daggered = [l for l in labels if charts.FLOOR_MARK in l]
        double = [l for l in labels if charts.UNREACHED_MARK in l]
        assert len(daggered) == 1 and "MariaDB" in daggered[0]
        assert len(double) == 1 and "pgvector" in double[0]
        assert any("AliSQL" in l for l in labels)
        assert not any(charts.FLOOR_MARK in l or charts.UNREACHED_MARK in l
                       for l in labels if "AliSQL" in l)
        # And the footnote is on the figure, not only in the helper.
        footnote = "\n".join(captured["texts"])
        assert "cannot be slowed down to the target" in footnote

    def test_the_footnote_names_the_engine_and_the_recall_it_achieved(self):
        note = charts._calibration_footnote(
            charts._calibration_marks(self._records(), "glove-100-angular"))
        assert "cannot be slowed down to the target" in note
        floor_line = [l for l in note.splitlines()
                      if l.startswith(charts.FLOOR_MARK)][0]
        unreached_line = [l for l in note.splitlines()
                          if l.startswith(charts.UNREACHED_MARK)][0]
        assert "MariaDB" in floor_line and "pgvector" not in floor_line
        assert "pgvector" in unreached_line and "MariaDB" not in unreached_line
        # The recall actually achieved, §2.3.3's requirement, and the other
        # engine's 0.9810 must not be the number shown.
        assert "0.8456" in unreached_line
        assert "0.9810" not in unreached_line

    def test_there_is_no_footnote_when_every_engine_reached_the_target(self):
        rows = (_curve("alisql", [(24, 0.9611)])
                + [_downstream("alisql", 24, "ok")])
        assert charts._calibration_footnote(
            charts._calibration_marks(rows, "glove-100-angular")) == ""

    def test_a_fixed_ef_run_marks_nothing(self):
        """No target was pursued, so no engine is an exception to one.

        Every engine lacks a `calibration_status` here, and the title already
        says the chart is at a fixed ef: marking all of them would be six
        exceptions to a claim nobody made.
        """
        rows = [_record("mariadb", "concurrency", ef_search=100, clients=8,
                        qps=1.0, extra={"ef_mode": "fixed"})]
        assert charts._calibration_marks(rows, "glove-100-angular") == {}

    def test_marks_do_not_cross_from_one_dataset_to_another(self):
        """The dataset filter, which every other fixture here leaves untested.

        Measured with the filter removed: the sift chart's legend suffix went
        from `''` to `' †'` and its footnote from empty to an assertion that
        MariaDB cannot be slowed down to the target — on a dataset where that
        was never measured. Suite under that mutant before this test existed:
        green, because every mark fixture used a single dataset.
        """
        glove = (_curve("mariadb", [(8, 0.9810)])
                 + [_downstream("mariadb", 8, "floor_above_target")])
        sift = ([dict(r, dataset="sift-128-euclidean")
                 for r in _curve("mariadb", [(24, 0.9611), (48, 0.9702)])]
                + [dict(_downstream("mariadb", 24, "ok"),
                        dataset="sift-128-euclidean")])
        both = glove + sift
        # Glove's MariaDB is daggered; sift's MariaDB is the same engine and
        # is not, because on sift it reached the target.
        assert charts._calibration_marks(both, "glove-100-angular") == {
            "mariadb": {"statuses": {"floor_above_target"},
                        "recalls": [("floor_above_target", "normalized", 16,
                                     0.9810)]}}
        assert charts._calibration_marks(both, "sift-128-euclidean") == {}
        assert charts._calibration_footnote(
            charts._calibration_marks(both, "sift-128-euclidean")) == ""
        assert "cannot be slowed down" in charts._calibration_footnote(
            charts._calibration_marks(both, "glove-100-angular"))

    def test_an_uncalibrated_engine_is_marked_and_footnoted(self):
        """Item 3: present on the chart, absent from §6, held to no floor.

        Measured before the fix: marks `{}`, legend suffix `''`, footnote
        `''`, while the panel title claimed `QPS at recall ≥ 0.95` over this
        engine's curve.
        """
        rows = (_curve("mariadb", [(24, 0.9611)])
                + [_downstream("mariadb", 24, "ok")]
                + _uncalibrated_concurrency("pgvector"))
        marks = charts._calibration_marks(rows, "glove-100-angular")
        assert set(marks) == {"pgvector"}
        assert marks["pgvector"]["statuses"] == {"uncalibrated"}
        assert charts._mark_suffix(marks["pgvector"]) == \
            " " + render.UNCALIBRATED_MARK
        assert charts._mark_suffix(marks.get("mariadb")) == ""
        note = charts._calibration_footnote(marks)
        assert note.startswith(render.UNCALIBRATED_MARK)
        assert "was NOT calibrated" in note
        assert "pgvector" in note and "MariaDB" not in note

    def test_the_uncalibrated_mark_differs_from_the_other_two(self):
        assert len({render.FLOOR_MARK, render.UNREACHED_MARK,
                    render.UNCALIBRATED_MARK}) == 3

    def test_the_footnotes_agree_in_number_with_the_engines_they_name(self):
        """Multi-engine is the NORMAL case for two of the three footnotes.

        An operator who flips `ops.ef_mode` mid-run has every engine already
        measured carrying no status, and `target_unreached` lands on whole
        families at low M. Written singular, the uncalibrated paragraph read,
        verbatim over two engines: "AliSQL (VIDX), PostgreSQL (pgvector) WAS
        not calibrated: IT carries no calibration_status ... does not apply to
        IT. ITS throughput ...". The `target_unreached` paragraph had the same
        defect in milder form: "<two labels> ... IS shown at ITS best
        achievable recall".
        """
        def note(uncalibrated, unreached):
            rows = (_curve("mariadb", [(24, 0.9611)])
                    + [_downstream("mariadb", 24, "ok")])
            for engine in uncalibrated:
                rows += _uncalibrated_concurrency(engine)
            for engine in unreached:
                rows += [dict(r, engine=engine)
                         for r in _curve(engine, [(480, 0.8456)])]
                rows.append(_downstream(engine, 480, "target_unreached"))
            return charts._calibration_footnote(
                charts._calibration_marks(rows, "glove-100-angular"))

        one = note(["pgvector"], ["mongodb"])
        two = note(["alisql", "pgvector"], ["mongodb", "valkey"])

        # Singular over one engine.
        assert "was NOT calibrated: carries no" in one
        assert "and is shown at its best achievable recall" in one
        # Plural over two, and none of the five singular forms survives.
        assert "were NOT calibrated: carry no" in two
        assert "and are shown at their best achievable recall" in two
        for singular in ("was NOT calibrated", "carries no calibration_status",
                         "is shown at its best"):
            assert singular not in two, singular
        # Both engines are still named in each paragraph.
        for label in ("AliSQL (VIDX)", "PostgreSQL (pgvector)"):
            assert label in two.splitlines()[0]
        for label in ("Percona Search (mongot)", "Valkey (valkey-search)"):
            assert label in two.splitlines()[1]

    def test_half_an_engine_calibrated_still_marks_it(self):
        """One M calibrated, the next not: the line through both is partly
        held to the target and partly not.

        The checkpoint grain is (pass, engine, dataset, phase) and, inside an
        ops phase, (M, build_mode) — so a resume can stop exactly here.
        """
        rows = (_curve("mariadb", [(24, 0.9611)])
                + [_downstream("mariadb", 24, "ok")]
                + [dict(r, m=32) for r in _uncalibrated_concurrency("mariadb")])
        marks = charts._calibration_marks(rows, "glove-100-angular")
        assert marks["mariadb"]["statuses"] == {"uncalibrated"}
        assert render.UNCALIBRATED_MARK in charts._mark_suffix(
            marks["mariadb"])

    def test_a_marks_derivation_that_ignores_the_status_is_caught(self):
        """The mutant: mark every engine that was calibrated at all.

        Two values: honestly one engine carries the dagger and one the double
        dagger out of three; under the mutant all three carry the dagger, and
        the footnote then says three engines cannot be slowed to the target
        when one of them sat exactly on it.
        """
        records = self._records()
        honest = self._labels(records)
        real = loaders.operating_points

        def mutant_points(rows):
            out = real(rows)
            for row in out:
                row["status"] = "floor_above_target"
            return out

        from unittest import mock
        with mock.patch.object(loaders, "operating_points", mutant_points):
            mutant = {
                engine: charts._mark_suffix(
                    charts._calibration_marks(
                        records, "glove-100-angular").get(engine))
                for engine in ("mariadb", "pgvector", "alisql")}
        assert honest == {"mariadb": " " + charts.FLOOR_MARK,
                          "pgvector": " " + charts.UNREACHED_MARK,
                          "alisql": ""}
        assert mutant == {"mariadb": " " + charts.FLOOR_MARK,
                          "pgvector": " " + charts.FLOOR_MARK,
                          "alisql": " " + charts.FLOOR_MARK}
        assert honest != mutant


# ---------------------------------------------------------------------------
# §2.3.4 -- the appendix chart
# ---------------------------------------------------------------------------

@needs_matplotlib
class TestTheCalibrationCurvesChart:
    """§2.3.4: recall against ef, one line per engine, in an appendix.

    A chart drawn from the wrong group of records renders perfectly, so these
    assertions read the plotted data back off the Axes and compare it to the
    records it should have come from -- not merely that a file was written.
    """

    def _records(self):
        return (
            _curve("mariadb", [(8, 0.9810), (16, 0.9900)])
            + _curve("pgvector", [(240, 0.8123), (480, 0.8456)])
            + [_downstream("mariadb", 8, "floor_above_target"),
               _downstream("pgvector", 480, "target_unreached")]
        )

    def _drawn(self, records, dataset="glove-100-angular", tmp="/tmp"):
        """The chart's lines and rings, captured before the figure closes."""
        captured: Dict[str, Any] = {}
        real_save = charts._save

        def capture(fig, out_dir, stem):
            ax = fig.axes[0]
            captured["lines"] = [
                (line.get_label(), list(line.get_xdata()),
                 [round(float(y), 6) for y in line.get_ydata()])
                for line in ax.get_lines() if not line.get_label().startswith("_")
            ]
            captured["rings"] = sorted(
                tuple(round(float(v), 6) for v in point)
                for coll in ax.collections for point in coll.get_offsets()
            )
            captured["hlines"] = [
                round(float(line.get_ydata()[0]), 6) for line in ax.get_lines()
                if line.get_label().startswith("_")
            ]
            captured["title"] = ax.get_title()
            return real_save(fig, out_dir, stem)

        from unittest import mock
        with mock.patch.object(charts, "_save", capture):
            paths = charts.calibration_curves(records, dataset, tmp,
                                              "calibration-x")
        captured["paths"] = paths
        return captured

    def test_one_line_per_engine_carrying_that_engines_curve(self, tmp_path):
        drawn = self._drawn(self._records(), tmp=str(tmp_path))
        lines = {label: (xs, ys) for label, xs, ys in drawn["lines"]}
        assert len(lines) == 2, list(lines)
        maria = next(v for k, v in lines.items() if "MariaDB" in k)
        pg = next(v for k, v in lines.items() if "pgvector" in k)
        # The exact points, in order: a chart fed the other engine's records
        # draws just as cleanly.
        assert maria == ([8, 16], [0.981, 0.99])
        assert pg == ([240, 480], [0.8123, 0.8456])

    def test_the_curve_is_the_sweeps_records_and_not_the_concurrency_points(
            self, tmp_path):
        """`_downstream` sits at ef 8 and 480 with recall 0.5.

        A chart selecting on `recall_at_k and ef_search` rather than on the
        phase would pick them up and add a 0.5 point to each curve.
        """
        drawn = self._drawn(self._records(), tmp=str(tmp_path))
        for _label, _xs, ys in drawn["lines"]:
            assert 0.5 not in ys

    def test_the_chosen_point_is_ringed_and_the_others_are_not(self, tmp_path):
        """One ring per engine, at the ef the downstream workloads used.

        MariaDB chose 8 (its floor) and pgvector chose 480 (its largest grid
        point), so the rings are at opposite ends of the two curves -- a
        marker placed by position rather than by the join would put both at the
        same end.
        """
        drawn = self._drawn(self._records(), tmp=str(tmp_path))
        assert drawn["rings"] == [(8.0, 0.981), (480.0, 0.8456)]

    def test_no_ring_is_drawn_where_no_workload_ran_at_the_ef(self, tmp_path):
        """Sweep records with no downstream half: a curve and no choice.

        The honest chart draws the curve and rings nothing, which is what
        "the ef was chosen and then nothing was measured at it" looks like.
        """
        drawn = self._drawn(_curve("mariadb", [(8, 0.9810), (16, 0.9900)]),
                            tmp=str(tmp_path))
        assert drawn["rings"] == []
        assert len(drawn["lines"]) == 1

    def test_the_target_line_is_drawn_at_the_runs_target(self, tmp_path):
        drawn = self._drawn(self._records(), tmp=str(tmp_path))
        assert 0.95 in drawn["hlines"]

    def test_it_returns_none_when_no_sweep_ran(self, tmp_path):
        rows = [_record("mariadb", "concurrency", ef_search=100, clients=8,
                        qps=1.0, extra={"ef_mode": "fixed"})]
        assert charts.calibration_curves(rows, "glove-100-angular",
                                         str(tmp_path), "calibration-x") is None

    def test_it_draws_only_the_dataset_it_was_asked_for(self, tmp_path):
        """A chart stem names a dataset; a chart holding two is mislabelled."""
        other = _curve("mariadb", [(8, 0.5000), (16, 0.6000)])
        for r in other:
            r["dataset"] = "sift-128-euclidean"
        drawn = self._drawn(self._records() + other, tmp=str(tmp_path))
        for _label, _xs, ys in drawn["lines"]:
            assert 0.5 not in ys and 0.6 not in ys

    def test_a_real_sweep_renders(self, tmp_path):
        """End to end on harness output, not only on synthetic records."""
        rows = _sweep_records(tmp_path, EF_HALF_OK)
        drawn = self._drawn(rows, dataset=rows[0]["dataset"],
                            tmp=str(tmp_path))
        assert drawn["paths"]
        [(label, xs, ys)] = drawn["lines"]
        assert xs == [8, 18, 98, 198]
        assert len(ys) == 4
        assert drawn["rings"] == [(98.0, round(
            next(r["recall_at_k"] for r in rows
                 if r["phase"] == "ef_calibration" and r["ef_search"] == 98),
            6))]


class TestTheAppendixSectionIsOnlyThereWhenTheChartIs:
    """§2.3.4 says "in an appendix", and an appendix over nothing is the
    empty-chart problem with a heading on it."""

    def _records(self):
        return (_curve("mariadb", [(24, 0.9611)])
                + [_downstream("mariadb", 24, "ok")])

    def test_the_markdown_appendix_appears_with_the_chart(self):
        paths = {"calibration-glove-100-angular":
                 {"svg": "/x/calibration-glove-100-angular.svg",
                  "png": "/x/calibration-glove-100-angular.png"}}
        report = _render(self._records(), chart_paths=paths)
        assert "## Appendix A. Calibration curves" in report
        assert "calibration-glove-100-angular.svg" in report
        # Last on the page, after the reproduction instructions.
        assert report.index("## 10. Reproducing this run") < report.index(
            "## Appendix A. Calibration curves")

    def test_there_is_no_appendix_without_a_chart(self):
        report = _render(self._records(), chart_paths={})
        assert "Appendix A" not in report

    def test_the_section_six_pointer_and_the_appendix_agree(self):
        """§6 tells the reader where the curves are; it must be right."""
        paths = {"calibration-glove-100-angular":
                 {"svg": "/x/c.svg", "png": "/x/c.png"}}
        report = _render(self._records(), chart_paths=paths)
        assert "in **Appendix A**" in report
        assert "## Appendix A." in report

    def test_the_html_appendix_follows_the_reproduction_section(self):
        summary = generate.summarize(self._records(), MANIFEST)
        page = render.render_html(
            MANIFEST, summary, self._records(),
            {"calibration-g": {"svg": "/nonexistent.svg", "png": "/x.png"}},
            "T2b")
        # The SVG cannot be inlined (no such file), so `figures()` yields
        # nothing and the appendix must be absent rather than empty -- which is
        # the same guard as the Markdown's, reached by a different route.
        assert "Appendix A" not in page

    def test_the_html_appendix_appears_when_the_svg_can_be_inlined(
            self, tmp_path):
        svg = tmp_path / "calibration-g.svg"
        svg.write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>")
        summary = generate.summarize(self._records(), MANIFEST)
        page = render.render_html(
            MANIFEST, summary, self._records(),
            {"calibration-g": {"svg": str(svg), "png": "/x.png"}}, "T2b")
        assert "Appendix A. Calibration curves" in page
        assert page.index("10. Reproducing this run") < page.index(
            "Appendix A. Calibration curves")


@needs_matplotlib
class TestTheChartIsSkippedRatherThanDrawnEmpty:
    """The generator's own guard, through `needs`.

    §2.3.4 calls the chart optional; this is what makes it optional, and it
    reuses the mechanism every other absent workload already uses rather than
    adding a second one.
    """

    def test_a_fixed_ef_run_draws_no_calibration_chart(self, tmp_path):
        """Behaviour, through `generate.main`, not a source-text search.

        This was an `inspect.getsource` substring check on `main`, which
        passes on a dispatch entry that is present and unreachable and fails
        on a correct one spelled differently. Driving the generator over a run
        directory with no sweep asserts the thing itself: no calibration chart
        is written, the other charts still are, and §6 explains the absence.
        """
        run_dir = _run_dir_with(tmp_path, _uncalibrated_concurrency("mariadb"))
        assert generate.main(["--run-dir", str(run_dir),
                              "--annb-results", str(tmp_path / "none"),
                              "--datasets-dir", str(tmp_path)]) == 0
        drawn = sorted(os.listdir(run_dir / "report" / "charts"))
        assert not [n for n in drawn if n.startswith("calibration-")]
        assert [n for n in drawn if n.startswith("concurrency-")]
        report = (run_dir / "report" / "report.md").read_text()
        assert "_No ef_search calibration in this run._" in report
        assert "Appendix A" not in report

    def test_a_calibrated_run_draws_it(self, tmp_path):
        """The other half: without this, "no chart" passes by never drawing."""
        records = (_curve("mariadb", [(24, 0.9611), (48, 0.9702)])
                   + [_downstream("mariadb", 24, "ok")])
        run_dir = _run_dir_with(tmp_path, records)
        assert generate.main(["--run-dir", str(run_dir),
                              "--annb-results", str(tmp_path / "none"),
                              "--datasets-dir", str(tmp_path)]) == 0
        drawn = sorted(os.listdir(run_dir / "report" / "charts"))
        assert "calibration-glove-100-angular.svg" in drawn, drawn
        report = (run_dir / "report" / "report.md").read_text()
        assert "## Appendix A. Calibration curves" in report


# ---------------------------------------------------------------------------
# F-2 -- the provenance row
# ---------------------------------------------------------------------------

class TestTheProvenanceTableNamesTheMeasurementVersion:
    """F-2, first half.

    `OPS_MEASUREMENT_VERSION` exists because a change to `harness/` can alter
    what an ops number means without altering the record's shape -- version 3
    moved the concurrency query pool off `ops.max_queries`, so a version 2
    throughput and a version 3 throughput are the same column measured over
    different fractions of the corpus. The report already refuses to put two
    versions on one chart; it printed neither of them above the numbers, so a
    reader of a published report could not tell which single version produced
    them.
    """

    def _rows(self, config):
        return render._environment_rows({"host": {"cpu": {}},
                                         "config": config})

    def test_the_version_is_in_the_environment_table(self):
        rows = self._rows({"resource_pass": "normalized",
                           "ops_measurement_version": 4})
        found = [r for r in rows if r[0] == "Ops measurement version"]
        assert found == [["Ops measurement version", "4"]]

    def test_the_old_form_of_this_table_had_no_such_row(self):
        """Rule 2's two values, measured rather than read.

        The pre-change `_environment_rows` is reconstructed here by dropping
        the row this change added, and the assertion above is replayed against
        it: the old form yields no row at all, the new form yields the
        version. A test asserting only "the table has some rows" passes both.
        """
        config = {"resource_pass": "normalized", "ops_measurement_version": 4}
        new = self._rows(config)
        old = [r for r in new if r[0] != "Ops measurement version"]
        assert len(new) - len(old) == 1
        assert [r for r in new if r[0] == "Ops measurement version"]
        assert [r for r in old if r[0] == "Ops measurement version"] == []

    def test_a_run_predating_the_field_says_so_rather_than_guessing_one(self):
        """An absent manifest key means a run older than the field.

        Printing `1` there would be a claim the manifest does not make -- it is
        the RECORDS that count as version 1 when they carry no version, and
        this cell is the manifest's statement.
        """
        rows = self._rows({"resource_pass": "normalized"})
        found = [r for r in rows if r[0] == "Ops measurement version"]
        assert found == [["Ops measurement version", "not recorded"]]
        assert found[0][1] != "1"

    def test_it_reaches_the_rendered_report_and_the_page(self):
        records = (_curve("mariadb", [(24, 0.9611)])
                   + [_downstream("mariadb", 24, "ok")])
        report = _render(records)
        version_row = [r for r in _table_rows(report, ENVIRONMENT_HEADER)
                       if r[0] == "Ops measurement version"]
        assert version_row == [["Ops measurement version", "4"]]
        summary = generate.summarize(records, MANIFEST)
        page = render.render_html(MANIFEST, summary, records, {}, "T2b")
        assert "Ops measurement version" in page

    def test_the_row_carries_the_manifests_value_and_not_the_constant(self):
        """A row hard-coded to `OPS_MEASUREMENT_VERSION` would be the report
        describing the code that rendered it, not the run that produced it.

        The second assertion is a RELATION, not a value. It read
        `OPS_MEASUREMENT_VERSION == 4`, which made this test a pin on the
        current version: the next bump turns a passing suite red here, and
        every later task in this stack bumps it when it changes what an ops
        number means. The constant's value is incidental to what this test is
        about -- all it needs is that the manifest's 2 is NOT what the code
        would have printed from its own constant, so the row demonstrably came
        from the manifest.

        The fixture-driven assertions elsewhere in this class deliberately do
        name `"4"`: `MANIFEST` declares `ops_measurement_version: 4`, so those
        assert that the manifest's own value is echoed and stay correct across
        a bump rather than going stale.
        """
        rows = self._rows({"ops_measurement_version": 2})
        assert [r for r in rows if r[0] == "Ops measurement version"] == [
            ["Ops measurement version", "2"]]
        assert str(records_mod.OPS_MEASUREMENT_VERSION) != "2"


# ---------------------------------------------------------------------------
# F-2 -- the refusal's count
# ---------------------------------------------------------------------------

class TestTheRefusalCountsTheVersionsItFound:
    """F-2, second half: "ops records from N different measurement versions".

    `N` was taken over every version involved, the manifest's declared one
    included. That version has no records -- the refusal's own listing says so
    four lines further down -- so over v1 and v3 records under a manifest
    declaring v2 the headline claimed three versions of records above a listing
    showing two, and the remedy it gave ("one report per version") sent an
    operator looking for a third pile that does not exist.
    """

    def _row(self, version=None, **kw):
        row = {"run_id": "r", "engine": "mariadb", "phase": "index_build",
               "dataset": "d", "m": 16, "resource_pass": "normalized",
               "timestamp": "2026-01-01T00:00:00Z"}
        row.update(kw)
        if version is not None:
            row["extra"] = {"ops_measurement_version": version}
        return row

    def _message(self, rows, declared=None):
        from report.measurement_versions import (
            MixedMeasurementVersions, refuse_mixed_measurement_versions)
        try:
            refuse_mixed_measurement_versions(rows, source="RUN",
                                              declared=declared)
        except MixedMeasurementVersions as exc:
            return str(exc)
        raise AssertionError("no refusal")

    def _three_way(self):
        """v1 and v3 records under a manifest declaring v2."""
        return [self._row(),
                self._row(version=3, engine="alisql",
                          timestamp="2026-09-09T00:00:00Z")]

    def test_the_count_is_of_the_versions_records_were_found_at(self):
        message = self._message(self._three_way(), declared=2)
        assert "ops records from 2 different measurement versions" in message
        # `from 3`, not `3`: a bare negative on "3 different..." would also
        # fire on a correct "13 different...", so the anchor word belongs on
        # the negative assertion for the same reason it belongs on the
        # positive one.
        assert "from 3 different measurement versions" not in message

    def test_the_headline_no_longer_contradicts_the_listing_below_it(self):
        """The two halves of one message, compared to each other.

        The listing is authoritative -- it is built from the groups -- so the
        count in the headline has to equal the number of listed versions that
        have records. Asserting the literal "2" alone would pass a message
        whose listing had changed instead.
        """
        message = self._message(self._three_way(), declared=2)
        head = message.splitlines()[0]
        listed = [l for l in message.splitlines()
                  if l.strip().startswith("version ")
                  and "no records measured under it" not in l]
        declared_only = [l for l in message.splitlines()
                         if "no records measured under it" in l]
        assert len(listed) == 2
        assert len(declared_only) == 1
        assert f"from {len(listed)} different measurement versions" in head

    def test_the_old_wording_survives_the_input_the_new_one_catches(self):
        """Rule 2, with both values.

        `_mixed_records_text` is called the old way -- one list serving as both
        the equivalence set and the count -- and the new way, over the same
        `sides`. Old headline: "3 different measurement versions". New: "2".
        The input is the three-way directory above, not a hypothetical.
        """
        from report import measurement_versions as mv
        sides = [(1, [self._row()]),
                 (3, [self._row(version=3)]),
                 (2, [])]
        all_versions = [v for v, _r in sides]
        measured = [v for v, r in sides if r]
        old_head, _ = mv._mixed_records_text("RUN", all_versions, 2)
        new_head, _ = mv._mixed_records_text("RUN", all_versions, 2,
                                             measured_versions=measured)
        assert "from 3 different measurement versions" in old_head[0]
        assert "from 2 different measurement versions" in new_head[0]
        assert old_head[0] != new_head[0]

    def test_a_pure_mix_with_no_manifest_still_counts_every_side(self):
        """The default path must not change: here every version has records."""
        message = self._message(
            [self._row(version=2), self._row(version=3, engine="alisql")])
        assert "ops records from 2 different measurement versions" in message

    def test_the_equivalence_claim_is_still_made_over_every_version(self):
        """The two lists are not interchangeable.

        Evaluated over the record versions alone, {1, 2, 3} is {1, 3} here and
        {1, 2} in the v1/v2-plus-declared-3 case -- which comes back
        "comparable" and asserts comparability across a real bump. The count
        moved; the equivalence set must not have.
        """
        message = self._message(
            [self._row(), self._row(version=2, engine="alisql")], declared=3)
        assert "ops records from 2 different measurement versions" in message
        assert "denote the same measurement" not in message
        assert "two different measurements" in message


# ---------------------------------------------------------------------------
# F-3 -- the methodology wording
# ---------------------------------------------------------------------------

class TestTheMethodologyDoesNotCallADerivedSetAList:
    """F-3. T2a made `OPS_MEASUREMENT_EQUIVALENT_VERSIONS` derived from two
    named constants, so "that pair is listed in" it describes a literal set
    that no longer exists -- and the derivation is the safety property the
    sentence should be pointing at."""

    PATH = os.path.join(VB_ROOT, "docs", "05-methodology.md")

    def test_the_set_really_is_derived_and_not_a_literal(self):
        """The claim the doc now makes, checked against the code it is about.

        Without this the doc could be "fixed" to say "derived" over a literal
        set and the test would still pass.
        """
        source = open(os.path.join(VB_ROOT, "harness", "metrics",
                                   "records.py")).read()
        block = source.split("OPS_MEASUREMENT_EQUIVALENT_VERSIONS = ")[1]
        block = block.split(")")[0]
        assert "IMPLICIT_OPS_MEASUREMENT_VERSION" in block
        assert "FIRST_RECORDED_OPS_MEASUREMENT_VERSION" in block
        assert "1" not in block.replace("VERSION", "")
        assert "2" not in block.replace("VERSION", "")

    def test_the_doc_no_longer_says_the_pair_is_listed_there(self):
        text = open(self.PATH).read()
        assert "listed in `OPS_MEASUREMENT_EQUIVALENT_VERSIONS`" not in text

    def test_the_doc_names_the_two_constants_it_is_derived_from(self):
        text = open(self.PATH).read()
        assert "IMPLICIT_OPS_MEASUREMENT_VERSION" in text
        assert "FIRST_RECORDED_OPS_MEASUREMENT_VERSION" in text
        assert "derived" in text

    def test_the_rest_of_the_paragraph_is_intact(self):
        """Only that sentence was in scope; the surrounding claims stay.

        Compared against the text with its whitespace collapsed: this
        document's prose is hard-wrapped and two of these three sentences span
        a newline, so an assertion against the raw text passes or fails on
        where the wrap happens to fall rather than on what the page says.
        """
        text = " ".join(open(self.PATH).read().split())
        assert "**Version 2 measures exactly what version 1 measured**" in text
        assert "**Version 3 is not comparable with either**" in text
        assert "deliberately absent from that set" in text


# ---------------------------------------------------------------------------
# House rules this change has to keep
# ---------------------------------------------------------------------------

class TestTheHouseRules:
    def test_the_loader_stays_within_the_line_ceiling(self):
        """`report/loaders.py` is pinned at 800 lines by `test_config.py`;
        asserted here too so the reason travels with this change."""
        for parts in (("report", "loaders.py"),
                      ("report", "measurement_versions.py")):
            path = os.path.join(VB_ROOT, *parts)
            assert len(open(path).read().splitlines()) <= 800, parts

    def test_nothing_added_here_imports_the_orchestrator(self):
        """report/ runs in a bench image mounting report/ and harness/ only.

        Importing the orchestrator from report/ once crashed generation at the
        end of a twenty-hour run.
        """
        for name in ("loaders.py", "generate.py", "render.py", "charts.py",
                     "measurement_versions.py"):
            src = open(os.path.join(VB_ROOT, "report", name)).read()
            for line in src.splitlines():
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                assert "import orchestrator" not in stripped, (name, stripped)

    def test_the_two_renderers_use_the_same_footnote_marks(self):
        """A dagger on the chart explained by a footnote under a different
        glyph in the table is worse than no mark at all."""
        assert charts.FLOOR_MARK is render.FLOOR_MARK
        assert charts.UNREACHED_MARK is render.UNREACHED_MARK
