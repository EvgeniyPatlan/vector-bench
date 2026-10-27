"""A chart's width was decided by its legend, not by its data.

`charts._save` writes every figure with `bbox_inches="tight"`, which grows the
SAVED image to contain whatever artists hang off the canvas. Four charts then
called `fig.legend(..., ncol=len(handles))`, which lays every series on ONE
row. So the legend row grew with the series count and took the PNG with it.
`storage_breakdown` had a second, unrelated version of the same mistake: it
asked for `max(8, 1.5 * len(labels))` inches of width with no ceiling.

Measured on `PS/new/full-20261015-215006`, the seven-engine `--resource-pass
both` run of 2026-10-15. fashion-mnist is the one corpus all seven engines
ran, so there each (engine, M, pass) combination is its own legend entry:

    churn-fashion-mnist-784-euclidean.png         64,730 x 792   81.7 : 1
    concurrency-fashion-mnist-784-euclidean.png   25,943 x 864   30.0 : 1
    filtered-tag-fashion-mnist-784-euclidean.png  25,326 x 817   31.0 : 1
    filtered-ctag-fashion-mnist-784-euclidean.png 25,326 x 817   31.0 : 1
    build-fashion-mnist-784-euclidean.png         25,326 x 792   32.0 : 1
    storage-fashion-mnist-784-euclidean.png        7,484 x 826    9.1 : 1
    churn-sift-128-euclidean.png                   4,514 x 792    5.7 : 1

Seven of that run's fifty charts, and six of the seven were the headline
corpus. The single-engine versions of the same charts measured 1.4:1 to 3.8:1
and are fine, which is the whole point: nothing is wrong with the data, the
panels or the series. The legend was a width.

These tests do not read that bundle -- `results/` is gitignored and a test
that needs a 126 MB archive is not a test. They rebuild its shape: seven
engines over three graph degrees, which is 21 plotted series, and for churn a
maintenance state on top of that, which is 42 legend entries.

**And then they sweep DOWNWARD, which is the part that was missing.** The
first version of this file swept 21 series and then 63, and every chart
passed. `latency_percentiles` was over the bound the whole time, at 4.07:1 --
but only at exactly THREE engines, where `columns = min(3, len(engines))`
makes it one row of three panels. At 21 and 63 series the same chart is a 3x3
grid at 1.41:1, comfortably inside. The violating shape was smaller than
anything being tested, so making the input bigger could never find it, and
the chart that broke was the project's headline MariaDB/AliSQL/pgvector
comparison. `SHAPES` now runs one engine through seven as well.

The same sweep, once it existed, found a second overhang the aspect ratio
cannot see at all: a `fig.text` CAPTION wider than a narrow figure.
`latency-gist-960-euclidean.png` in that run is 837 pixels on a 660-pixel
canvas at 1.73:1 -- inside the bound, 27% too wide for its own figure. That
is why `test_the_saved_image_is_no_wider_than_the_canvas` exists beside the
ratio test rather than instead of it.
"""
from __future__ import annotations

import os
import struct
import sys
import tempfile

import pytest

VB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if VB_ROOT not in sys.path:
    sys.path.insert(0, VB_ROOT)

from report import charts as charts_mod  # noqa: E402

needs_matplotlib = pytest.mark.skipif(
    not charts_mod.have_matplotlib(),
    reason="matplotlib is a container dependency, not a host one")

DATASET = "fashion-mnist-784-euclidean"

#: The seven engines of the 2026-10-15 run, in `config/engines/` order.
ENGINES = ("mariadb", "mariadb123", "alisql", "pgvector", "mongodb", "valkey",
           "percona")
#: Three graph degrees, which is what made fashion-mnist 21 series.
DEGREES = (16, 32, 64)

#: Seven engines x three degrees: the shape that produced the strips above.
MANY = tuple((engine, m) for engine in ENGINES for m in DEGREES)


def _record(engine, m, **fields):
    record = {"engine": engine, "dataset": DATASET, "m": m,
              "resource_pass": "tuned", "storage_engine": "InnoDB",
              "ef_construction": 200, "build_mode": "post", "extra": {}}
    record.update(fields)
    return record


def churn_records(units=MANY):
    """Two maintenance states per unit, as PR 4 §4.2 writes them.

    `charts.churn` widens the series key by that state, so this is the chart
    with twice as many legend entries as plotted units -- and it is the chart
    that reached 64,730 pixels.
    """
    return [_record(engine, m, phase="churn", churn_fraction=fraction,
                    recall_at_k=0.95 - fraction, qps=1000.0 - 100 * fraction,
                    extra={"maintenance_state": state})
            for engine, m in units
            for fraction in (0.0, 0.1, 0.25)
            for state in ("before", "after")]


def build_records(units=MANY):
    """`ingest_wall_s` is present because `time_to_queryable_index_s` drops a
    bulk record without it, and that panel is the one the legend is read off."""
    return [_record(engine, m, phase="index_build", build_wall_s=100.0 + m,
                    ingest_wall_s=900.0, peak_rss_bytes=10 ** 9,
                    index_bytes=10 ** 8, table_bytes=10 ** 8,
                    extra={"separable_build": True})
            for engine, m in units]


def storage_records(units=MANY):
    """Both resource passes, which is what made the run's chart 33 bars."""
    return [_record(engine, m, phase="index_build", resource_pass=resource_pass,
                    build_wall_s=100.0 + m, index_bytes=10 ** 8,
                    table_bytes=10 ** 8, extra={"separable_build": True})
            for engine, m in units
            for resource_pass in ("normalized", "tuned")]


def concurrency_records(units=MANY):
    return [_record(engine, m, phase="concurrency", clients=clients,
                    qps=500.0 * clients, latency_p99_ms=2.0 * clients,
                    recall_at_k=0.95)
            for engine, m in units
            for clients in (1, 4, 16)]


def filtered_records(units=MANY):
    return [_record(engine, m, phase="filtered", selectivity=selectivity,
                    recall_at_k=0.9, qps=1000.0,
                    extra={"filter_column": "tag"})
            for engine, m in units
            for selectivity in (0.01, 0.1, 0.5)]


def recall_qps_records(units=MANY):
    return [_record(engine, m, phase="recall_qps", ef_search=ef, clients=1,
                    recall_at_k=0.90 + 0.0003 * ef, qps=3000.0 / ef,
                    latency_p50_ms=0.4 * ef, latency_p99_ms=1.0 * ef)
            for engine, m in units
            for ef in (16, 64, 256)]


def memory_series(units=MANY):
    return {f"{engine}-m{m}-ops": [
        {"t": float(t), "rss_bytes": 10 ** 8 + t * 10 ** 6}
        for t in range(10)] for engine, m in units}


def pareto_by_engine(units=MANY):
    return {(engine, "tuned", m): [
        _record(engine, m, phase="recall_qps", recall_at_k=0.9 + 0.003 * i,
                qps=2000.0 - 40 * i) for i in range(12)]
        for engine, m in units}


def calibration_sweep_records(units=MANY):
    """`PHASE_CALIBRATION` is spelled `ef_calibration`, which is worth a
    helper: a fixture that wrote `phase="calibration"` makes
    `loaders.calibration_records` return nothing and `calibration_curves`
    return None, and a chart that drew nothing would have sat in the sweep
    below looking like a pass."""
    return [_record(engine, m, phase="ef_calibration", ef_search=ef,
                    clients=1, recall_at_k=0.90 + 0.0003 * ef,
                    qps=3000.0 / ef, extra={"target_recall": 0.95})
            for engine, m in units
            for ef in (16, 64, 256)]


def qps_summary(units=MANY):
    return {"per_dataset": {DATASET: {"tuned": {
        engine: {"qps_at_recall_90": 1500.0, "qps_at_recall_95": 900.0,
                 "qps_at_recall_99": 300.0} for engine, _m in units}}}}


#: `(name, how to draw it, how to build its records)` for every chart that
#: carries a figure legend or sizes itself from its series count -- which is
#: exactly the set of charts the 2026-10-15 run produced a strip for, and the
#: set the legend and bar-width fixes touch.
WIDENING_CHARTS = (
    ("churn", lambda records, out: charts_mod.churn(
        records, DATASET, out, "churn"), churn_records),
    ("build", lambda records, out: charts_mod.build_cost(
        records, DATASET, out, "build"), build_records),
    ("concurrency", lambda records, out: charts_mod.concurrency(
        records, DATASET, out, "concurrency"), concurrency_records),
    ("filtered", lambda records, out: charts_mod.filtered(
        records, DATASET, out, "filtered", column="tag"), filtered_records),
    ("storage", lambda records, out: charts_mod.storage_breakdown(
        records, DATASET, out, "storage"), storage_records),
)

#: `(name, how to draw it from a set of units)` for EVERY chart this module
#: draws, not only the five the legend broke.
#:
#: Restricting the bound to the five was how `latency_percentiles` reached
#: review at 4.07:1. It is sized `(4.4 * columns, 3.1 * rows)` with
#: `columns = min(3, len(engines))`, so it passes through neither the bar
#: helper nor a fixed panel count, and at exactly three engines -- the
#: project's headline MariaDB/AliSQL/pgvector comparison -- that is one row of
#: three panels, over the bound. The bound is a property of the module, so the
#: module is what gets swept.
ALL_CHARTS = WIDENING_CHARTS + (
    ("pareto", lambda records, out: charts_mod.pareto(
        records, DATASET, 10, out, "pareto"), pareto_by_engine),
    ("memory_timeline", lambda records, out: charts_mod.memory_timeline(
        records, out, "memory"), memory_series),
    ("qps_at_recall", lambda records, out: charts_mod.qps_at_recall(
        records, DATASET, out, "qps"), qps_summary),
    ("pass_comparison", lambda records, out: charts_mod.pass_comparison(
        records, DATASET, out, "passcompare"),
     lambda units=MANY: (concurrency_records(units) + build_records(units)
                         + filtered_records(units)
                         + recall_qps_records(units))),
    ("latency_percentiles", lambda records, out: charts_mod.latency_percentiles(
        records, DATASET, out, "latency"), recall_qps_records),
    ("churn_impact", lambda records, out: charts_mod.churn_impact(
        records, DATASET, out, "churnimpact"), churn_records),
    ("calibration_curves", lambda records, out: charts_mod.calibration_curves(
        records, DATASET, out, "calibration"), calibration_sweep_records),
)

#: Engine counts ONE THROUGH SEVEN, then the two high series counts.
#:
#: The downward half is the half that was missing. The first version of this
#: suite swept 21 series and then 63, and `latency_percentiles` is a 3x3 grid
#: at 1.41:1 at both -- comfortably inside the bound. Its violating shape is
#: three engines, which is SMALLER than anything that was measured. A bound
#: that is only ever tested by making the input bigger is not tested.
SHAPES = tuple(
    (f"{n}e", tuple((engine, 16) for engine in ENGINES[:n]))
    for n in range(1, len(ENGINES) + 1)
) + (
    ("21s", MANY),
    ("63s", tuple((engine, m) for engine in ENGINES
                  for m in (4, 8, 16, 24, 32, 48, 64, 96, 128))),
)


def png_size(path):
    """Width and height straight out of the IHDR chunk.

    Read from the file rather than from the figure, because the figure is not
    what went wrong: `bbox_inches="tight"` means the canvas and the artifact
    can disagree by a factor of sixty, and the artifact is what goes in the
    report.
    """
    with open(path, "rb") as handle:
        header = handle.read(33)
    assert header[:8] == b"\x89PNG\r\n\x1a\n", path
    return struct.unpack(">II", header[16:24])


def draw(draw_chart, records):
    """Render one chart and return `(width, height)` of its PNG.

    `TemporaryDirectory`, not `mkdtemp`: each call writes an SVG and a PNG,
    and `mkdtemp` would leave both behind on every suite run.
    """
    with tempfile.TemporaryDirectory() as out:
        paths = draw_chart(records, out)
        assert paths, "the chart drew nothing, so its size asserts nothing"
        return png_size(paths["png"])


# ---------------------------------------------------------------------------
# The arithmetic, which needs no matplotlib
# ---------------------------------------------------------------------------

class TestTheBarChartWidthIsCapped:
    """`storage_breakdown`'s own widening, separate from any legend."""

    def test_a_few_bars_keep_the_width_they_always_had(self):
        """Under the cap nothing changes, which is what keeps the charts that
        were already fine byte-identical."""
        assert charts_mod.bar_chart_figsize(7, 1.5, 8.0, 5.2) == (10.5, 5.2)

    def test_the_minimum_still_applies(self):
        assert charts_mod.bar_chart_figsize(2, 1.5, 8.0, 5.2) == (8.0, 5.2)

    def test_the_runs_thirty_three_bars_stop_at_the_bound(self):
        """33 bars asked for 49.5 inches and got 7,484 x 826 pixels."""
        width, height = charts_mod.bar_chart_figsize(33, 1.5, 8.0, 5.2)
        assert 1.5 * 33 == 49.5, "the fixture no longer asks for an over-wide figure"
        assert width == charts_mod.MAX_ASPECT_RATIO * height
        assert width / height <= charts_mod.MAX_ASPECT_RATIO

    def test_no_bar_count_can_exceed_the_bound(self):
        for count in (0, 1, 13, 14, 42, 500, 5000):
            width, height = charts_mod.bar_chart_figsize(count, 1.5, 8.0, 5.2)
            assert width / height <= charts_mod.MAX_ASPECT_RATIO, count


class TestEveryFigsizeGoesThroughTheBound:
    """The module-wide claim, made executable instead of asserted.

    `latency_percentiles` reached review at 4.07:1 because `MAX_ASPECT_RATIO`
    was referenced in exactly ONE place -- `bar_chart_figsize` -- while three
    other call sites did their own arithmetic inline. Inline arithmetic in a
    `figsize` is the shape of every instance of this defect:
    `(4.4 * columns, 3.1 * rows)`, `(max(8, 1.5 * len(labels)), 5.2)`,
    `(4.0 * len(panels), 4.8)`. So the rule is that a figsize is either a
    tuple of literal numbers or the return of one of the two helpers, and
    nothing in between.

    Source-level, so it needs no matplotlib and no rendering, and it fails on
    the NEXT chart that grows a dimension rather than on the next run that
    happens to have the shape that exposes it.
    """

    HELPERS = {"bar_chart_figsize", "grid_figsize"}

    @staticmethod
    def _tree():
        import ast
        path = os.path.join(VB_ROOT, "report", "charts.py")
        return ast.parse(open(path).read()), ast

    @classmethod
    def _chart_functions(cls):
        """Every top-level function in the module that writes a chart file.

        `_save` is the one way a chart reaches disk, so calling it is what
        makes a function a chart rather than a helper -- derived from the
        module instead of listed, because a list is the thing being checked.
        """
        tree, ast = cls._tree()
        drawn = set()
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            for inner in ast.walk(node):
                if (isinstance(inner, ast.Call)
                        and getattr(inner.func, "id", None) == "_save"):
                    drawn.add(node.name)
                    break
        return drawn

    def test_the_sweep_covers_every_chart_the_module_draws(self):
        """`ALL_CHARTS` is hand-maintained, and a chart missing from it is
        bounded by nothing.

        **This is the third time in this change's history that a
        forward-looking generalisation turned out to be false**, and the first
        two were both caught by measurement rather than by reading: "the first
        round number above every intrinsic panel layout" (it was not --
        `latency_percentiles` was at 4.07:1) and "every data-dependent figsize
        passes through a helper" (it did not -- `pass_comparison` did its own
        arithmetic). The sweep above made the first executable and the AST
        guard made the second; the completeness of `ALL_CHARTS` is the same
        shape of claim, so it gets the same treatment rather than a third
        assertion.

        It also closes the residual hole in the guard above. That guard
        catches inline arithmetic -- the historical defect shape -- but a
        figsize can still evade it, through a local under another name,
        `set_size_inches`, a positional `Figure((w, h))` or a `**kwargs`
        splat. None of those is exploitable while the chart is swept, because
        the rendering tests measure the artifact rather than the source. They
        become exploitable only for a chart that is BOTH written evasively and
        absent from this registry. This test removes the second condition.
        """
        drawn = self._chart_functions()
        covered = set()
        for _name, draw_chart, _make_records in ALL_CHARTS:
            # Each entry's lambda names exactly one `charts_mod.<fn>`, and the
            # attribute lands in the code object's names.
            covered |= set(draw_chart.__code__.co_names)
        missing = drawn - covered
        assert not missing, (
            f"{sorted(missing)} call `_save` but are not in ALL_CHARTS, so "
            f"nothing measures their aspect ratio or their canvas width at "
            f"any shape. Add an entry with a fixture builder.")
        assert len(drawn) >= 12, (
            f"only {len(drawn)} chart functions found; this guard has lost "
            f"its grip on the module")

    def test_every_figsize_argument_is_a_literal_or_a_helper(self):
        tree, ast = self._tree()
        seen = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for keyword in node.keywords:
                if keyword.arg != "figsize":
                    continue
                seen += 1
                value = keyword.value
                where = f"line {value.lineno}"
                if isinstance(value, ast.Tuple):
                    assert all(isinstance(e, ast.Constant) for e in value.elts), (
                        f"{where}: figsize does arithmetic inline. Route it "
                        f"through {sorted(self.HELPERS)} so the bound applies.")
                elif isinstance(value, ast.Call):
                    name = getattr(value.func, "id",
                                   getattr(value.func, "attr", None))
                    assert name in self.HELPERS, (
                        f"{where}: figsize comes from {name!r}, which does not "
                        f"enforce MAX_ASPECT_RATIO")
                else:
                    assert isinstance(value, ast.Name), (
                        f"{where}: unrecognised figsize expression")
        assert seen >= 10, (
            f"only found {seen} figsize arguments; this guard has lost its "
            f"grip on the module")

    def test_every_figsize_variable_is_assigned_from_a_helper(self):
        """The `ast.Name` escape above, closed. `_new_axes` takes a `figsize`
        parameter and `storage_breakdown` binds one, so a bare name is legal
        -- but only when what was bound to it came from a helper."""
        tree, ast = self._tree()
        assignments = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(t, ast.Name) and t.id == "figsize"
                       for t in node.targets):
                continue
            assignments += 1
            assert isinstance(node.value, ast.Call), node.lineno
            name = getattr(node.value.func, "id",
                           getattr(node.value.func, "attr", None))
            assert name in self.HELPERS, (
                f"line {node.lineno}: `figsize` is bound from {name!r}, which "
                f"does not enforce MAX_ASPECT_RATIO")
        assert assignments >= 1, "no `figsize = ...` left; update this guard"


class TestThePanelGridMeetsTheBound:
    """`latency_percentiles`' sizing, the one the first round missed.

    `grid_figsize` raises the HEIGHT where `bar_chart_figsize` caps the width,
    and the asymmetry is deliberate: a bar chart's width grows with the data
    and has no limit, so height cannot save it; a panel grid's width is
    already capped by its column count, so only the ratio is wrong.
    """

    def test_three_panels_in_a_row_are_raised_to_the_bound(self):
        """13.2 x 3.1 saves at 4.07:1. It needs 3.3 inches of height."""
        assert 4.4 * 3 / 3.1 > charts_mod.MAX_ASPECT_RATIO, (
            "the fixture no longer describes an over-wide row")
        assert charts_mod.grid_figsize(3, 1, 4.4, 3.1) == pytest.approx(
            (13.2, 3.3))

    def test_a_grid_that_already_fits_is_untouched(self):
        """Four engines upward are two or three rows and were always fine;
        raising those would move charts that had nothing wrong with them."""
        assert charts_mod.grid_figsize(3, 2, 4.4, 3.1) == pytest.approx(
            (13.2, 6.2))
        assert charts_mod.grid_figsize(3, 3, 4.4, 3.1) == pytest.approx(
            (13.2, 9.3))

    def test_one_panel_gets_the_modules_usual_width_floor(self):
        """A 4.4-inch figure is narrower than anything else here and cannot
        hold its own suptitle, which is why the single-engine latency charts
        in the run are 837 and 835 pixels on a 660-pixel canvas."""
        assert charts_mod.grid_figsize(1, 1, 4.4, 3.1) == (8.0, 3.1)

    def test_no_grid_shape_can_exceed_the_bound(self):
        for columns in range(1, 5):
            for rows in range(1, 6):
                width, height = charts_mod.grid_figsize(
                    columns, rows, 4.4, 3.1)
                assert width / height <= charts_mod.MAX_ASPECT_RATIO, (
                    columns, rows, width, height)


class TestTextBelowTheAxesClearsWhatIsAlreadyThere:
    """`_below` is what stops the wrapped legend and `concurrency`'s
    calibration footnote -- both anchored in figure fractions under the
    canvas -- from being drawn on top of each other."""

    def test_nothing_below_leaves_the_floor_alone(self):
        """The branch every already-fine chart takes."""
        assert charts_mod._below(-0.04, None) == -0.04

    def test_text_moves_under_a_reserved_artist(self):
        assert charts_mod._below(-0.04, -0.30, gap=0.02) == -0.32

    def test_a_shallow_reservation_does_not_raise_the_text(self):
        """A legend that ends above the floor must not pull the footnote up
        into the axes it is explaining."""
        assert charts_mod._below(-0.06, -0.01, gap=0.02) == -0.06


# ---------------------------------------------------------------------------
# The rendered artifact
# ---------------------------------------------------------------------------

@needs_matplotlib
class TestTheFixtureReallyDoesStretch:
    """Non-vacuity. Without it every bound below could be passing on a chart
    that never had enough series to stretch in the first place, which is the
    way this suite has been fooled before -- see
    `test_ops_chart_series_pooling.py::test_the_old_key_pooled_them`.
    """

    @staticmethod
    def _one_row_legend(fig, handles, labels, fontsize=10):
        """The pre-fix call, verbatim."""
        if handles:
            fig.legend(handles, labels, frameon=False, fontsize=fontsize,
                       loc="upper center", ncol=len(handles),
                       bbox_to_anchor=(0.5, 1.06))
        return None

    @pytest.mark.parametrize("name,draw_chart,make_records", WIDENING_CHARTS)
    def test_the_old_layout_stretches_this_fixture(
            self, name, draw_chart, make_records, monkeypatch):
        """Every chart in the table at the top, reproduced."""
        monkeypatch.setattr(charts_mod, "_series_legend", self._one_row_legend)
        if name == "storage":
            # `storage_breakdown` never had a figure legend; its width came
            # from the uncapped per-bar arithmetic, so that is what has to be
            # put back to reproduce it.
            monkeypatch.setattr(
                charts_mod, "bar_chart_figsize",
                lambda count, per_bar, minimum, height:
                    (max(minimum, per_bar * count), height))
        width, height = draw(draw_chart, make_records())
        assert width / height > charts_mod.MAX_ASPECT_RATIO, (
            f"{name} came out {width} x {height}, inside the bound even with "
            f"the pre-fix layout -- the fixture has drifted to a series count "
            f"that never reproduced the defect, and every bound below is "
            f"passing vacuously")


#: Memo for the sweep below, keyed by `(chart, shape)`.
#:
#: The two properties are asserted by two tests so that a failure names which
#: one broke, but they are two statements about ONE rendering, and the sweep
#: is 11 charts x 9 shapes. Rendering each cell twice doubled this file's
#: share of the suite for nothing. Sound because
#: `test_the_wrapped_layout_is_still_byte_deterministic` pins that a second
#: render of the same records gives the same file.
_MEASURED = {}


def draw_measuring_canvas(name, shape, draw_chart, records):
    """`(width, height, canvas_width_px)` for one rendered chart.

    The canvas is read off the figure at `savefig` time, because the whole
    mechanism is that the two can disagree: `bbox_inches="tight"` grows the
    SAVED image past the canvas to hold anything that overhangs it.
    """
    import matplotlib.pyplot as plt

    if (name, shape) in _MEASURED:
        return _MEASURED[(name, shape)]

    canvases = []
    real = plt.Figure.savefig

    def record_canvas(self, *args, **kwargs):
        canvases.append(self.get_size_inches()[0] * 150)
        return real(self, *args, **kwargs)

    plt.Figure.savefig = record_canvas
    try:
        width, height = draw(draw_chart, records)
    finally:
        plt.Figure.savefig = real
    _MEASURED[(name, shape)] = (width, height, max(canvases))
    return _MEASURED[(name, shape)]


#: matplotlib's Agg backend refuses a canvas of 2^16 pixels in either
#: direction. A legend row long enough takes a figure past it.
AGG_PIXEL_LIMIT = 2 ** 16


@needs_matplotlib
class TestTheWideShapeDoesNotDestroyTheReport:
    """At 63 series the pre-fix churn chart did not draw a strip. It RAISED.

    Measured against `acaf032`, this branch's base, by rendering that
    revision's own `report/charts.py` with seven engines over nine graph
    degrees -- 63 units, 126 legend entries:

        ValueError: Image size of 67288x792 pixels is too large.
        It must be less than 2^16 in each direction.

    from `_RendererAgg(int(width), int(height), dpi)` inside `savefig`. The
    same revision renders 21 series fine, at 22,494 x 792.

    **That is a different severity from everything else in this file, and it
    is the true one.** A chart too wide to use is a presentation problem a
    reader works around. This is `generate.py` raising at report time, after
    the measurement work is already spent -- the run has done its hours and
    has nothing to show for them. This module already refuses to draw rather
    than draw nothing, for exactly that reason; it should not then lose the
    report to a legend.

    **Why this is its own test when the sweep already renders this shape.**
    It would indeed go red: a raise turns those into errors. But it would
    report as "the aspect ratio test errored", which names the wrong defect,
    and the protection would be incidental -- it survives only as long as
    `63s` stays in `SHAPES`, and `SHAPES` is a runtime/coverage trade-off
    someone will reasonably revisit. The one failure mode that DESTROYS
    output rather than degrading it should not depend on a list that exists
    for another purpose. So it is pinned here, by name, with the shape
    spelled out.
    """

    def test_sixty_three_series_still_renders(self):
        units = tuple((engine, m) for engine in ENGINES
                      for m in (4, 8, 16, 24, 32, 48, 64, 96, 128))
        assert len(units) == 63, len(units)
        # `draw` asserts the chart produced files at all; reaching this line
        # already means `savefig` did not raise.
        width, height = draw(
            lambda records, out: charts_mod.churn(
                records, DATASET, out, "churn"),
            churn_records(units))
        assert max(width, height) < AGG_PIXEL_LIMIT, (
            f"churn came out {width} x {height}, which the Agg backend "
            f"refuses -- this is the shape that used to raise")


@needs_matplotlib
@pytest.mark.parametrize("shape,units", SHAPES)
@pytest.mark.parametrize("name,draw_chart,make_records", ALL_CHARTS)
class TestNoChartBecomesAStrip:
    """The bound itself, over EVERY chart and EVERY shape.

    One render per cell, two assertions on it. Swept from one engine upward
    rather than only from twenty-one series: see `SHAPES`.
    """

    def test_the_aspect_ratio_stays_within_the_bound(
            self, name, draw_chart, make_records, shape, units):
        width, height, _canvas = draw_measuring_canvas(
            name, shape, draw_chart, make_records(units))
        assert width / height <= charts_mod.MAX_ASPECT_RATIO, (
            f"{name} at {shape} came out {width} x {height} = "
            f"{width / height:.2f} : 1")

    def test_the_saved_image_is_no_wider_than_the_canvas(
            self, name, draw_chart, make_records, shape, units):
        """The sharper statement, and the one that names the mechanism:
        nothing may decide the width except the figure itself. A bound on the
        ratio alone could be met by a figure that grew in both directions, and
        this is also the assertion that catches an over-wide CAPTION, which
        the ratio does not -- `latency-gist-960-euclidean.png` in the
        2026-10-15 run is 837 pixels on a 660-pixel canvas at 1.73:1, inside
        the bound and still 27% too wide for its own figure."""
        width, _height, canvas = draw_measuring_canvas(
            name, shape, draw_chart, make_records(units))
        assert width <= canvas, (
            f"{name} at {shape} is {width}px wide on a {canvas:.0f}px canvas "
            f"(+{100 * (width / canvas - 1):.0f}%): something is hanging off "
            f"the side and `bbox_inches=\"tight\"` is paying for it in width")


#: The largest engine count at which each chart's legend still fits on ONE
#: row, measured. They differ because the canvases do -- `build_cost` has four
#: panels and twenty inches, `filtered` and `churn` have two panels and
#: eleven -- and because `charts.churn` carries two legend entries per engine.
#:
#: The small numbers are not a defect of the fix; they are a measurement of
#: how marginal the old layout already was. `churn-glove-100-angular.png` in
#: the 2026-10-15 run is 3,039 pixels wide on an 1,650-pixel canvas: it was
#: inside the aspect bound and still hanging nine inches off the side of its
#: own figure. One engine is genuinely where that chart stopped fitting.
ONE_ROW_FITS = {"churn": 1, "build": 7, "concurrency": 6, "filtered": 4}


@needs_matplotlib
class TestTheFixCostsNothingItDidNotHaveTo:
    """Three things the fix must not quietly buy the bound with."""

    @pytest.mark.parametrize("name,draw_chart,make_records", (
        c for c in WIDENING_CHARTS if c[0] in ONE_ROW_FITS))
    def test_a_legend_that_fits_renders_exactly_as_it_did_before(
            self, name, draw_chart, make_records, monkeypatch):
        """Where one row fitted, the output must not have moved by a pixel.

        The PNGs here are byte-deterministic and tests compare them, so
        "narrower" was never the goal -- "unchanged wherever nothing was
        wrong" is. This renders at `ONE_ROW_FITS[name]` engines twice, once
        through the fix and once through the pre-fix call, and compares bytes.
        """
        records = make_records(
            tuple((engine, 16) for engine in ENGINES[:ONE_ROW_FITS[name]]))
        with tempfile.TemporaryDirectory() as out:
            fixed = open(draw_chart(records, out)["png"], "rb").read()
        monkeypatch.setattr(
            charts_mod, "_series_legend",
            TestTheFixtureReallyDoesStretch._one_row_legend)
        with tempfile.TemporaryDirectory() as out:
            before = open(draw_chart(records, out)["png"], "rb").read()
        assert fixed == before, (
            f"{name} renders differently at {ONE_ROW_FITS[name]} engine(s), "
            f"where the legend fitted on one row and nothing needed to change")

    @pytest.mark.parametrize("name,draw_chart,make_records", (
        c for c in WIDENING_CHARTS if c[0] in ONE_ROW_FITS))
    def test_that_count_really_is_the_one_row_case(
            self, name, draw_chart, make_records):
        """Non-vacuity for the test above: if `ONE_ROW_FITS` ever drifts past
        where the legend wraps, the comparison would be between two wrapped
        renderings and would pass without saying anything about the branch it
        exists to pin."""
        import matplotlib.pyplot as plt

        calls = []
        real = plt.Figure.legend

        def spy(self, *args, **kwargs):
            calls.append((len(args[0]), kwargs.get("ncol")))
            return real(self, *args, **kwargs)

        plt.Figure.legend = spy
        try:
            with tempfile.TemporaryDirectory() as out:
                draw_chart(make_records(
                    tuple((engine, 16)
                          for engine in ENGINES[:ONE_ROW_FITS[name]])), out)
        finally:
            plt.Figure.legend = real
        assert calls, f"{name} drew no figure legend"
        handles, ncol = calls[0]
        assert ncol == handles, (
            f"{name} wrapped at {ONE_ROW_FITS[name]} engine(s) "
            f"({ncol} columns for {handles} entries)")

    @pytest.mark.parametrize("name,draw_chart,make_records", WIDENING_CHARTS)
    def test_the_wrapped_layout_is_still_byte_deterministic(
            self, name, draw_chart, make_records):
        """The wrapped layout measures text to decide its column count, and a
        layout that measures is a layout that could have gone
        non-deterministic. The report's PNGs are compared byte for byte
        elsewhere in this suite; rendering the same records twice has to give
        the same file."""
        records = make_records()
        with tempfile.TemporaryDirectory() as out:
            first = open(draw_chart(records, out)["png"], "rb").read()
        with tempfile.TemporaryDirectory() as out:
            second = open(draw_chart(records, out)["png"], "rb").read()
        assert first == second, f"{name} rendered differently twice"

    @pytest.mark.parametrize("name,draw_chart,make_records", (
        c for c in WIDENING_CHARTS if c[0] != "storage"))
    def test_every_series_still_reaches_the_legend(
            self, name, draw_chart, make_records):
        """Wrapping, not dropping. The cheapest way to narrow a legend is to
        stop drawing most of it, and that would take the chart's series with
        it -- a reader would have no way to attribute a line."""
        import matplotlib.pyplot as plt

        seen = []
        real = plt.Figure.legend

        def record_entries(self, *args, **kwargs):
            if len(args) > 1:
                seen.append(list(args[1]))
            return real(self, *args, **kwargs)

        plt.Figure.legend = record_entries
        try:
            records = make_records()
            draw(draw_chart, records)
        finally:
            plt.Figure.legend = real

        assert seen, f"{name} drew no figure legend at all"
        entries = seen[0]
        expected = len({charts_mod.series_key(r) for r in records})
        if name == "churn":
            # `charts.churn` widens the key by the maintenance state, so it
            # has two legend entries per plotted unit.
            expected *= 2
        assert len(entries) == expected, (
            f"{name} drew {len(entries)} legend entries for {expected} "
            f"series")
        assert len(set(entries)) == len(entries), "legend entries collided"


@needs_matplotlib
class TestTheBarChartOnlyCrowdsWhenItRanOutOfWidth:
    """`crowded` has to be a statement about having hit the cap.

    Nothing pinned that. `ONE_ROW_FITS` covers the four figure-legend charts
    and excludes `storage_breakdown`, which has no figure legend, so a mutant
    that turned `crowded` on for EVERY storage chart -- `figsize[0] <= 1.5 *
    len(labels)` instead of `<` -- rotated every caption on every run and the
    whole suite stayed green. The four arithmetic tests above cannot see it:
    `bar_chart_figsize` is correct in that mutant, and only its caller reads
    the answer wrongly.
    """

    @staticmethod
    def _rotations(units):
        """Rotation of the bar captions and of the per-bar totals."""
        import matplotlib.pyplot as plt

        seen = {"captions": [], "totals": []}
        real_ticks = plt.Axes.set_xticklabels
        real_text = plt.Axes.text

        def spy_ticks(self, labels, **kwargs):
            seen["captions"].append(kwargs.get("rotation", 0))
            return real_ticks(self, labels, **kwargs)

        def spy_text(self, x, y, s, **kwargs):
            seen["totals"].append(kwargs.get("rotation", 0))
            return real_text(self, x, y, s, **kwargs)

        plt.Axes.set_xticklabels = spy_ticks
        plt.Axes.text = spy_text
        try:
            with tempfile.TemporaryDirectory() as out:
                charts_mod.storage_breakdown(
                    storage_records(units), DATASET, out, "storage")
        finally:
            plt.Axes.set_xticklabels = real_ticks
            plt.Axes.text = real_text
        return seen

    def test_a_chart_under_the_cap_keeps_everything_horizontal(self):
        """Three engines is six bars, nine inches -- nowhere near the cap."""
        units = tuple((engine, 16) for engine in ENGINES[:3])
        assert 1.5 * 2 * len(units) < charts_mod.MAX_ASPECT_RATIO * 5.2, (
            "the fixture is no longer under the cap")
        seen = self._rotations(units)
        assert seen["captions"] == [0], seen["captions"]
        assert set(seen["totals"]) == {0}, set(seen["totals"])

    def test_a_chart_over_the_cap_turns_upright(self):
        """Non-vacuity for the test above: the rotation must actually be
        reachable, or `== [0]` would be passing on a branch that never runs."""
        seen = self._rotations(MANY)
        assert seen["captions"] == [90], seen["captions"]
        assert set(seen["totals"]) == {90}, set(seen["totals"])


@needs_matplotlib
class TestTheFootnoteClearsTheWrappedLegend:
    """`concurrency` places two things in figure fractions below the canvas.

    Unit-tested and integration-untested is how this one hid: `_below` keeps
    all three of its unit tests under a mutant that simply stops CALLING it
    with the legend's position -- `legend_bottom = _series_legend(...) and
    None` -- and the calibration footnote then lands on top of the wrapped
    legend at exactly the seven-engine run the code comment says it is there
    for. The helper being right is not the property anyone wants; the two
    artists not colliding is.
    """

    @staticmethod
    def _placed(units):
        import matplotlib.pyplot as plt

        records = concurrency_records(units)
        # `calibration_targets` non-empty plus points carrying no
        # `calibration_status` is what earns every engine the uncalibrated
        # mark, which is what makes `_calibration_footnote` return a sentence.
        records[0]["extra"] = dict(records[0]["extra"], target_recall=0.95)

        seen = {"texts": []}
        real_legend = plt.Figure.legend
        real_text = plt.Figure.text

        def spy_legend(self, *args, **kwargs):
            legend = real_legend(self, *args, **kwargs)
            seen["ncol"] = kwargs.get("ncol")
            seen["handles"] = len(args[0])
            extent = legend.get_window_extent(self.canvas.get_renderer())
            seen["legend_bottom"] = extent.y0 / (
                self.get_size_inches()[1] * self.dpi)
            return legend

        def spy_text(self, x, y, s, **kwargs):
            seen["texts"].append((y, s))
            return real_text(self, x, y, s, **kwargs)

        plt.Figure.legend = spy_legend
        plt.Figure.text = spy_text
        try:
            with tempfile.TemporaryDirectory() as out:
                charts_mod.concurrency(records, DATASET, out, "concurrency")
        finally:
            plt.Figure.legend = real_legend
            plt.Figure.text = real_text
        # Only what was placed BELOW the canvas: the suptitle sits at y=1.14
        # and the width probes `_legend_columns` makes sit at y=0.
        seen["below"] = [t for t in seen["texts"] if t[0] < 0]
        return seen

    def test_the_footnote_sits_under_the_legend_at_seven_engines(self):
        seen = self._placed(tuple((engine, 16) for engine in ENGINES))
        assert seen["ncol"] < seen["handles"], (
            "the legend did not wrap, so there is nothing for the footnote "
            "to collide with and this test asserts nothing")
        assert len(seen["below"]) == 1, seen["below"]
        footnote_y, footnote = seen["below"][0]
        assert footnote.strip(), "the fixture produced no footnote text"
        assert footnote_y <= seen["legend_bottom"], (
            f"the calibration footnote is at y={footnote_y:.3f} and the "
            f"wrapped legend reaches down to y={seen['legend_bottom']:.3f}: "
            f"they are drawn on top of each other")

    def test_a_footnote_with_no_wrapped_legend_keeps_its_old_place(self):
        """The other half of `_below`: where nothing came down here, the
        footnote must not drift from the -0.04 it always used."""
        seen = self._placed(tuple((engine, 16) for engine in ENGINES[:3]))
        assert seen["ncol"] == seen["handles"], "the legend wrapped after all"
        assert len(seen["below"]) == 1, seen["below"]
        assert seen["below"][0][0] == -0.04
