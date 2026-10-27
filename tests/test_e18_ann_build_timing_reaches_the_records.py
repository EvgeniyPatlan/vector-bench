"""E18 -- the recall harness measures the build and then throws the number away.

THE DEFECT
----------
Every ann-benchmarks result file this framework writes carries TWO build
timings, and the report reads neither:

* `build_time`, written by upstream `ann_benchmarks/runner.py:build_index`
  around its `algo.fit` call;
* `ingest_seconds`, written by our own
  `overlay/ann-benchmarks/ann_benchmarks/vb_mysql.py:get_additional`, which
  rounds the loader's own wall clock to 3 decimals.

`report/loaders.py:load_annb_results` copies a dozen other attributes into the
record and, out of the whole result file, takes exactly one derived value off
the file itself -- `os.path.getmtime(path)` into `extra.source_mtime`. So the
cost half of "what does a higher `ef_construction` buy" is in the artifacts, in
no record, and therefore in no report.

What that cost. Read straight out of the archived `fashion-mnist-784-euclidean`
sweep at `annb/tuned/a9349676/.../percona/`, M=16, `ef_search` 10:

    ef_construction    build_time    ingest_seconds
                 50     130.790         130.372
                200     161.559         161.545
                800     283.275         283.260

50 -> 800 is **2.17x** the build time. The published report told the server
team it was "16 times the build effort" -- which is the ratio of the
`ef_construction` values, not of anything measured, and is the figure a reader
reaches for when the measured one is not in the report. The numbers were in the
files the whole time.

The two agree to within half a second at the low end and 15 ms at the high end,
which is the expected relationship and not a redundancy: `build_time` is
upstream's clock around `fit()` and `ingest_seconds` is our module's clock
around the load alone, so the difference is this engine's non-load build work.
Both are kept, because a future engine where they diverge is exactly the
engine worth noticing.

THE COLLATERAL HAZARD, which is why this unit also edits `report/charts.py`
----------------------------------------------------------------------------
`charts.build_cost` selected its records on dataset and `build_wall_s is not
None` and did **not** filter on phase. Nothing leaked while no `recall_qps`
record carried a build time -- which is precisely what the loader change
above ends. Populating the field without a phase filter would have put the ann
path's builds onto the ops path's build-cost chart, where the `peak_rss_bytes`
and `index_bytes` panels sit: four panels of ops-harness quantities with
ann-container figures drawn through them, each measured in a different
container under a different memory budget. The fix is a named predicate,
`charts._is_build_cost_point`, the sibling of `charts._is_concurrency_point`,
which exists for the same reason -- selecting a chart's records on "has the
field" rather than "is the measurement" has now produced this class of defect
twice.

The two neighbours were already safe and stay untouched: `render.py`'s §5
build table reads `summary["build"]`, which `generate._PHASE_GROUPS` filters to
`index_build`, and `charts.pass_comparison` names its phase in its dimension
table.

WHAT IS PINNED HERE
-------------------
* `TestTheTwoTimingsReachTheRecord` -- the headline. Both fields land, at the
  artifacts' own precision.
* `TestAnAbsentAttributeIsAbsentAndNotZero` -- a result file carrying neither
  attribute yields a record carrying neither key. A `0.0` would read as a
  build that cost nothing, and `build_cost`'s own selection is
  `is not None`, so a zero would also be a plotted point.
* `TestNoRateIsDerivedFromAFileThatCannotSupportOne` -- the result file holds
  no training-row count (its full attribute set is enumerated below), so
  `ingest_rows_per_s` stays `None` rather than being computed from a number
  the file does not have.
* `TestAnnBuildsStayOffTheOpsBuildCostChart` -- the phase filter, asserted
  through the predicate rather than off the PNG, so it is checked on a host
  with no matplotlib.

HOW THE SYNTHETIC RESULT FILES ARE BUILT, and why not with h5py
---------------------------------------------------------------
h5py lives in the bench images and not on a host built from the documented
prerequisites (`CLAUDE.md`, "Host requirements"), so these tests cannot write a
real HDF5 container -- and a test that skips on the measuring host's own
toolchain is not a test of this loader. `load_annb_results` imports `h5py`
inside its body, once there is something to read, so a stub in `sys.modules`
reaches it. The stub is an HDF5 *reader* only: the files are real files, on
disk, with the real `.hdf5` extension, in the real `<dataset>/<k>/<engine>/`
layout, reached by the real `os.walk`, gated by the real ground-truth
existence check and stamped with the real `os.path.getmtime`. Only the binary
container is JSON. Attribute values are handed back as the numpy scalars h5py
hands back, because that is what the coercers have to survive.

`tests/test_e14_per_pass_manifest.py` and `tests/test_e17_report_scope.py`
monkeypatch `load_annb_results` away wholesale for the same reason. That is
the right call where the loader is not the subject; here it is the subject.
"""

from __future__ import annotations

import json
import os
import sys
import types
from typing import Any, Dict, List, Optional, Sequence

import numpy
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.metrics import records as records_mod  # noqa: E402
from report import charts, loaders  # noqa: E402

#: Skips the one test that reads a drawn figure. The phase filter is asserted
#: through `charts._is_build_cost_point` in a test that never imports
#: matplotlib, so the invariant is covered on a bare host and this only adds
#: the "and `build_cost` actually calls it" half.
needs_matplotlib = pytest.mark.skipif(
    not charts.have_matplotlib(),
    reason="reads a drawn figure; matplotlib lives in the bench images")


# ---------------------------------------------------------------------------
# The measured figures this unit exists for
# ---------------------------------------------------------------------------

#: `(ef_construction, build_time, ingest_seconds)` from
#: `results/annb/tuned/a9349676/fashion-mnist-784-euclidean/10/percona/`,
#: M=16, `ef_search` 10, read out of the archived artifacts. Rounded to the
#: three decimals `ingest_seconds` itself carries.
ARCHIVED_EF_CONSTRUCTION_SWEEP = (
    (50, 130.790, 130.372),
    (200, 161.559, 161.545),
    (800, 283.275, 283.260),
)

#: `build_time` at `ef_construction` 800, at the full float64 precision the
#: artifact holds, so that a loader which rounds on the way in is caught.
#: `ingest_seconds` has no such value: the overlay rounds it before writing.
ARCHIVED_BUILD_TIME_FULL_PRECISION = 283.274920463562

#: Every attribute name in that artifact, which is the evidence for
#: `TestNoRateIsDerivedFromAFileThatCannotSupportOne`: there is no row count,
#: no training-set size and no `rows` among them, so no rate is derivable.
ARCHIVED_RESULT_ATTRIBUTES = (
    "M", "algo", "batch_mode", "best_search_time", "build_time", "count",
    "dataset", "distance", "ef_construction", "ef_search", "engine",
    "engine_version", "index_bytes", "index_size", "ingest_seconds",
    "insert_threads", "march", "metric", "name", "resource_pass",
    "run_count", "storage_engine", "vector_index_used",
)


# ---------------------------------------------------------------------------
# The HDF5 reader stub and the synthetic tree
# ---------------------------------------------------------------------------

def _as_h5py_value(value: Any) -> Any:
    """What h5py hands back for a scalar attribute of this Python type.

    Numbers come back as numpy scalars rather than Python ones, and that is
    the whole reason this function exists: `float(numpy.float64(...))` is the
    conversion the loader's coercer has to perform, and a test that handed it
    a Python float would not exercise it.
    """
    if isinstance(value, bool):
        return numpy.bool_(value)
    if isinstance(value, int):
        return numpy.int64(value)
    if isinstance(value, float):
        return numpy.float64(value)
    return value


class _StubH5File:
    """An HDF5 file whose container happens to be JSON. See the module docstring."""

    def __init__(self, path: str, mode: str = "r") -> None:
        if mode != "r":
            raise AssertionError("the loader only ever opens for reading")
        with open(path) as fh:
            blob = json.load(fh)
        self.attrs = {key: _as_h5py_value(value)
                      for key, value in blob.get("attrs", {}).items()}
        self._datasets = {key: numpy.asarray(value, dtype="float64")
                          for key, value in blob.get("datasets", {}).items()}

    def __enter__(self) -> "_StubH5File":
        return self

    def __exit__(self, *_exc: Any) -> bool:
        return False

    def __contains__(self, key: str) -> bool:
        return key in self._datasets

    def __getitem__(self, key: str) -> numpy.ndarray:
        return self._datasets[key]


@pytest.fixture
def stub_h5py(monkeypatch):
    """Put the reader where `load_annb_results`' own `import h5py` finds it."""
    monkeypatch.setitem(sys.modules, "h5py",
                        types.SimpleNamespace(File=_StubH5File))
    return None


#: k, and the width of every distance row below. Matches the `10` in the
#: archived tree's `<dataset>/10/<engine>/` path.
COUNT = 10

DATASET = "fashion-mnist-784-euclidean"


def _write_hdf5(path: str, attrs: Dict[str, Any],
                datasets: Dict[str, List[List[float]]]) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"attrs": attrs, "datasets": datasets}, fh)
    return path


def _truth_rows(rows: int = 4) -> List[List[float]]:
    """Ground-truth distances. Ascending within a row, as a kNN result is."""
    return [[0.1 * (i + 1) for i in range(COUNT)] for _ in range(rows)]


def _write_truth(datasets_dir: str, dataset: str = DATASET) -> str:
    return _write_hdf5(os.path.join(datasets_dir, f"{dataset}.hdf5"),
                       {"dataset": dataset}, {"distances": _truth_rows()})


def _result_attrs(ef_construction: int, **overrides: Any) -> Dict[str, Any]:
    """The attribute set the overlay and upstream actually write, trimmed to
    what this loader reads, with the archived values where there are any."""
    attrs: Dict[str, Any] = {
        "dataset": DATASET,
        "count": COUNT,
        "distance": "euclidean",
        "algo": "percona",
        "engine": "percona",
        "engine_version": "8.0.44-PerconaServer",
        "storage_engine": "InnoDB",
        "resource_pass": "tuned",
        "M": 16,
        "ef_search": 10,
        "ef_construction": ef_construction,
        "index_bytes": 249856 * 1024,
        "best_search_time": 0.0004973109960556031,
        "insert_threads": 8,
        "march": "x86-64-v3",
        "vector_index_used": True,
        "batch_mode": False,
        "run_count": 1,
        "name": f"PerconaServer(M=16, efConstruction={ef_construction})",
    }
    for key, value in overrides.items():
        if value is _ABSENT:
            attrs.pop(key, None)
        else:
            attrs[key] = value
    return attrs


class _Absent:
    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<absent>"


#: Sentinel for "this attribute is not in the file at all", which is a
#: different thing from "it is in the file holding zero".
_ABSENT = _Absent()


def _write_result(annb_dir: str, attrs: Dict[str, Any],
                  filename: Optional[str] = None,
                  rows: int = 4) -> str:
    """One result file, in the layout ann-benchmarks writes."""
    engine = str(attrs.get("engine", "percona"))
    stem = filename or ("euclidean_M_16_efConstruction_"
                        f"{attrs.get('ef_construction', 0)}_engine_InnoDB_10.hdf5")
    path = os.path.join(annb_dir, "tuned", str(attrs.get("dataset", DATASET)),
                        str(COUNT), engine, stem)
    # Run distances inside the ground-truth threshold, so recall is 1.0 and no
    # assertion below is reading past a NaN.
    return _write_hdf5(path, attrs, {
        "distances": [[0.01 * (i + 1) for i in range(COUNT)]
                      for _ in range(rows)],
        "times": [0.0005] * rows,
    })


def _load(tmp_path, results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    annb_dir = str(tmp_path / "annb")
    datasets_dir = str(tmp_path / "datasets")
    os.makedirs(annb_dir, exist_ok=True)
    os.makedirs(datasets_dir, exist_ok=True)
    _write_truth(datasets_dir)
    for attrs in results:
        _write_result(annb_dir, attrs)
    return loaders.load_annb_results(annb_dir, datasets_dir, "e18-run")


# ---------------------------------------------------------------------------
# 1. Both timings reach the record
# ---------------------------------------------------------------------------

class TestTheTwoTimingsReachTheRecord:

    def test_the_archived_sweep_lands_as_two_fields(self, tmp_path, stub_h5py):
        """The headline, over the three archived configurations.

        `build_wall_s` from `build_time` and `ingest_wall_s` from
        `ingest_seconds`, on the one record each file produces.
        """
        records = _load(tmp_path, [
            _result_attrs(ef, build_time=build, ingest_seconds=ingest)
            for ef, build, ingest in ARCHIVED_EF_CONSTRUCTION_SWEEP])
        assert len(records) == len(ARCHIVED_EF_CONSTRUCTION_SWEEP)
        by_ef = {r["ef_construction"]: r for r in records}
        for ef, build, ingest in ARCHIVED_EF_CONSTRUCTION_SWEEP:
            assert by_ef[ef]["build_wall_s"] == build, ef
            assert by_ef[ef]["ingest_wall_s"] == ingest, ef

    def test_the_single_configuration_the_spec_names(self, tmp_path, stub_h5py):
        """`ef_construction` 800 on its own, the slowest of the three."""
        records = _load(tmp_path, [_result_attrs(
            800, build_time=283.275, ingest_seconds=283.260)])
        assert len(records) == 1
        assert records[0]["build_wall_s"] == 283.275
        assert records[0]["ingest_wall_s"] == 283.26

    def test_the_build_cost_ratio_is_answerable_from_the_records_alone(
            self, tmp_path, stub_h5py):
        """The question the report could not answer, answered off the records.

        2.17x, not the 16x the ratio of the ef_construction values suggests
        and the published report claimed.
        """
        records = _load(tmp_path, [
            _result_attrs(ef, build_time=build, ingest_seconds=ingest)
            for ef, build, ingest in ARCHIVED_EF_CONSTRUCTION_SWEEP])
        by_ef = {r["ef_construction"]: r["build_wall_s"] for r in records}
        ratio = by_ef[800] / by_ef[50]
        assert round(ratio, 2) == 2.17
        # Where "16x" came from: it is the ratio of the knob, not of any
        # measurement. Both numbers are now in the records, so a reader does
        # not have to reach for the knob.
        assert 800 / 50 == 16.0
        assert round(ratio, 2) != 16.0

    def test_the_artifacts_precision_is_not_rounded_away(self, tmp_path,
                                                         stub_h5py):
        """`build_time` arrives as the file holds it.

        A reviewer checking a record against the artifact compares against the
        artifact's number, so the loader must not re-round it. `qps` and
        `recall_at_k` above it are rounded because they are DERIVED here; this
        is copied.
        """
        records = _load(tmp_path, [_result_attrs(
            800, build_time=ARCHIVED_BUILD_TIME_FULL_PRECISION,
            ingest_seconds=283.260)])
        assert records[0]["build_wall_s"] == ARCHIVED_BUILD_TIME_FULL_PRECISION

    def test_a_numpy_scalar_is_coerced_to_a_plain_float(self, tmp_path,
                                                        stub_h5py):
        """h5py hands back `numpy.float64`, which `json.dump` cannot write.

        A record field that cannot be serialised would fail at the end of a
        long run, writing `report/records.jsonl` -- the same shape of failure
        as importing the orchestrator from `report/`.
        """
        records = _load(tmp_path, [_result_attrs(
            800, build_time=283.275, ingest_seconds=283.260)])
        assert type(records[0]["build_wall_s"]) is float
        assert type(records[0]["ingest_wall_s"]) is float
        json.dumps(records[0])

    def test_the_record_is_still_a_recall_record(self, tmp_path, stub_h5py):
        """Nothing about the phase or the recall half moves.

        The timings are additive: a `recall_qps` record that now also says what
        the build cost. If this started arriving under another phase the ops
        charts would pick it up by phase and the collateral fix below would be
        pointless.
        """
        records = _load(tmp_path, [_result_attrs(
            800, build_time=283.275, ingest_seconds=283.260)])
        assert records[0]["phase"] == records_mod.PHASE_RECALL
        assert records[0]["recall_at_k"] == 1.0
        assert records[0]["engine"] == "percona"
        assert records[0]["extra"]["source_mtime"] > 0


# ---------------------------------------------------------------------------
# 2. Absent is absent
# ---------------------------------------------------------------------------

class TestAnAbsentAttributeIsAbsentAndNotZero:

    def test_neither_attribute_yields_neither_key(self, tmp_path, stub_h5py):
        """Not `0.0`, and not `None` under a key that was not there before.

        A zero reads as a build that cost nothing, and since
        `charts._is_build_cost_point` and the webui both select on `is not
        None`, a zero is a plotted point at the origin rather than an absent
        one. Every result file written before the overlay's `ingest_seconds`
        existed is in this case, and so is every engine whose module does not
        write it.
        """
        records = _load(tmp_path, [_result_attrs(
            800, build_time=_ABSENT, ingest_seconds=_ABSENT)])
        assert len(records) == 1
        assert "build_wall_s" not in records[0]
        assert "ingest_wall_s" not in records[0]

    def test_one_present_and_one_absent_is_one_key(self, tmp_path, stub_h5py):
        """Upstream's `build_time` with no overlay `ingest_seconds`: what a
        third-party algorithm module's result file looks like."""
        records = _load(tmp_path, [_result_attrs(
            800, build_time=283.275, ingest_seconds=_ABSENT)])
        assert records[0]["build_wall_s"] == 283.275
        assert "ingest_wall_s" not in records[0]

    def test_a_recorded_zero_is_kept(self, tmp_path, stub_h5py):
        """The converse, so that "absent" is not implemented as "falsy".

        A zero that the file actually holds is a measurement -- of something
        wrong, which is exactly when a reader needs to see it rather than have
        it silently dropped.
        """
        records = _load(tmp_path, [_result_attrs(
            800, build_time=0.0, ingest_seconds=0.0)])
        assert records[0]["build_wall_s"] == 0.0
        assert records[0]["ingest_wall_s"] == 0.0

    def test_an_unparseable_attribute_is_absent_rather_than_fatal(
            self, tmp_path, stub_h5py):
        """One unreadable attribute must not cost the whole record.

        The loader's `except Exception` is per FILE, so a raise here would
        drop a recall point as well as a build time.
        """
        records = _load(tmp_path, [_result_attrs(
            800, build_time="n/a", ingest_seconds=float("nan"))])
        assert len(records) == 1
        assert records[0]["recall_at_k"] == 1.0
        assert "build_wall_s" not in records[0]
        # NaN too: `json.dump` writes a bare `NaN`, which is not JSON, and the
        # line above it already drops a NaN `qps` for this reason.
        assert "ingest_wall_s" not in records[0]


# ---------------------------------------------------------------------------
# 3. No rate is invented
# ---------------------------------------------------------------------------

class TestNoRateIsDerivedFromAFileThatCannotSupportOne:

    def test_the_result_file_holds_no_row_count(self):
        """The evidence, enumerated: nothing in the artifact counts rows.

        `ingest_rows_per_s` needs rows / seconds. The seconds are there twice
        over; the rows are nowhere, and the dataset's own training-set size is
        not the loaded row count whenever a load was interrupted or a
        configuration was re-run against a populated table. Deriving the rate
        from the dataset's nominal size would produce a number that looks
        measured and is modelled.
        """
        for name in ARCHIVED_RESULT_ATTRIBUTES:
            assert "row" not in name, name
        assert "rows" not in ARCHIVED_RESULT_ATTRIBUTES
        assert "train_size" not in ARCHIVED_RESULT_ATTRIBUTES

    def test_no_rate_and_no_row_count_reach_the_record(self, tmp_path,
                                                       stub_h5py):
        records = _load(tmp_path, [_result_attrs(
            800, build_time=283.275, ingest_seconds=283.260)])
        assert records[0].get("ingest_rows_per_s") is None
        assert records[0].get("rows") is None


# ---------------------------------------------------------------------------
# 4. The collateral fix: the ops build-cost chart stays an ops chart
# ---------------------------------------------------------------------------

def _ops_build_record(engine: str = "pgvector", m: int = 16
                      ) -> Dict[str, Any]:
    return {"run_id": "e18-run", "phase": records_mod.PHASE_BUILD,
            "engine": engine, "dataset": DATASET, "m": m,
            "resource_pass": "tuned", "storage_engine": None,
            "build_wall_s": 37.14, "ingest_wall_s": 900.0,
            "peak_rss_bytes": 10 ** 9, "index_bytes": 10 ** 8,
            "table_bytes": 10 ** 8, "extra": {"separable_build": True}}


def _ann_recall_record(engine: str = "percona", m: int = 16
                       ) -> Dict[str, Any]:
    """What this unit's loader change now produces: a recall record with a
    build time on it."""
    return {"run_id": "e18-run", "phase": records_mod.PHASE_RECALL,
            "engine": engine, "dataset": DATASET, "m": m,
            "resource_pass": "tuned", "storage_engine": "InnoDB",
            "ef_construction": 800, "ef_search": 10,
            "recall_at_k": 0.98, "qps": 2010.8,
            "build_wall_s": 283.275, "ingest_wall_s": 283.26,
            "index_bytes": 249856 * 1024, "extra": {}}


class TestAnnBuildsStayOffTheOpsBuildCostChart:

    def test_the_predicate_takes_the_ops_build_phase_alone(self):
        """Assertion 3, through the predicate rather than the PNG.

        Deliberately not `@needs_matplotlib`: this is the invariant, and an
        invariant that can only be checked where matplotlib is installed is
        unchecked on the host the documented prerequisites produce.
        """
        ops = _ops_build_record()
        ann = _ann_recall_record()
        selected = [r for r in (ann, ops)
                    if charts._is_build_cost_point(r, DATASET)]
        assert selected == [ops]

    def test_the_phase_it_selects_is_the_schemas_build_phase(self):
        """The predicate spells the phase as a literal, because
        `report/charts.py` must not import `harness` at module scope -- the
        comment above `RECALL_CI_EXTRA_KEY` records the long run that died of
        it. This is what stops the literal drifting from the constant.
        """
        assert records_mod.PHASE_BUILD == "index_build"
        kept = _ops_build_record()
        moved = dict(kept, phase="index_build_v2")
        assert charts._is_build_cost_point(kept, DATASET)
        assert not charts._is_build_cost_point(moved, DATASET)

    def test_the_other_phases_that_carry_a_build_time_are_also_excluded(self):
        """Not just `recall_qps`.

        `ef_calibration` runs on the index the ops pass has just built and
        nothing stops a future record from carrying the build time forward.
        The predicate is an allow-list of one phase, not a deny-list of the
        phases that have gone wrong so far.
        """
        for phase in (records_mod.PHASE_RECALL, records_mod.PHASE_CALIBRATION,
                      records_mod.PHASE_INGEST, records_mod.PHASE_CONCURRENCY,
                      records_mod.PHASE_FILTERED, records_mod.PHASE_CHURN):
            record = dict(_ops_build_record(), phase=phase)
            assert not charts._is_build_cost_point(record, DATASET), phase

    def test_the_dataset_and_the_field_still_gate_it(self):
        """The two terms that were already there are still there."""
        ops = _ops_build_record()
        assert not charts._is_build_cost_point(ops, "glove-100-angular")
        assert not charts._is_build_cost_point(
            dict(ops, build_wall_s=None), DATASET)

    def test_build_cost_selects_through_the_predicate(self):
        """A correct predicate `build_cost` does not call is the other way to
        fail this, and the test that catches it off the drawn figure is
        `@needs_matplotlib` -- so on a host built from the documented
        prerequisites it would not be caught at all. Asserted off the source
        for the same reason `tests/test_config.py` reads
        `harness/drivers/valkey.py`'s `delete_ids` body: the invariant is
        which selection the function performs, and that is legible without a
        rendering stack.
        """
        import inspect
        body = inspect.getsource(charts.build_cost)
        assert "_is_build_cost_point(r, dataset)" in body, body
        # The clause the predicate replaced must not come back beside it.
        assert 'r.get("build_wall_s") is not None' not in body, body

    @needs_matplotlib
    def test_the_drawn_chart_plots_the_ops_record_alone(self, tmp_path,
                                                        monkeypatch):
        """And `build_cost` actually uses the predicate.

        Captured off `Axes.plot`, the way
        `tests/test_t5b_stats.py::TestTheBuildChartCannotPublishTheMisleadingComparison`
        does, because a correct predicate that the chart does not call is the
        other way to fail this.
        """
        captured: List[Dict[str, Any]] = []
        real_plot = charts.plt.Axes.plot

        def capture(self, *args, **kw):
            if len(args) >= 2:
                captured.append({"ylabel": self.get_ylabel(),
                                 "label": kw.get("label"),
                                 "points": list(zip(list(args[0]),
                                                    list(args[1])))})
            return real_plot(self, *args, **kw)

        monkeypatch.setattr(charts.plt.Axes, "plot", capture)
        ops = [_ops_build_record(m=m) for m in (8, 16, 32)]
        ann = [_ann_recall_record(m=m) for m in (8, 16, 32)]
        out = charts.build_cost(ann + ops, DATASET, str(tmp_path), "e18-build")
        assert out is not None
        labels = {s["label"] for s in captured}
        assert labels and all("pgvector" in str(label) for label in labels), \
            labels
        # 283.275 is the ann build time. It must be on no panel, and the
        # memory and index-size panels are the ones where an ann figure would
        # be a different container's number entirely.
        plotted = {y for s in captured for _x, y in s["points"]}
        assert 283.275 not in plotted
        assert 249856 * 1024 not in plotted
