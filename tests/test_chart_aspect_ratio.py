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


#: `(name, how to draw it, how to build its records)` for every chart that
#: carries a figure legend or sizes itself from its series count -- which is
#: exactly the set of charts the run produced a strip for.
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


@needs_matplotlib
class TestNoChartBecomesAStrip:
    """The bound itself, which is the thing a reader cares about."""

    @pytest.mark.parametrize("name,draw_chart,make_records", WIDENING_CHARTS)
    def test_the_aspect_ratio_stays_within_the_bound(
            self, name, draw_chart, make_records):
        width, height = draw(draw_chart, make_records())
        assert width / height <= charts_mod.MAX_ASPECT_RATIO, (
            f"{name} came out {width} x {height} = "
            f"{width / height:.1f} : 1")

    @pytest.mark.parametrize("name,draw_chart,make_records", WIDENING_CHARTS)
    def test_the_saved_image_is_no_wider_than_the_canvas(
            self, name, draw_chart, make_records):
        """The sharper statement, and the one that names the mechanism: the
        legend must stop being able to decide the width at all. A bound on the
        ratio alone could be met by a figure that grew in both directions."""
        import matplotlib.pyplot as plt

        widths = []
        real = plt.Figure.savefig

        def record_canvas(self, *args, **kwargs):
            widths.append(self.get_size_inches()[0] * 150)
            return real(self, *args, **kwargs)

        plt.Figure.savefig = record_canvas
        try:
            width, _height = draw(draw_chart, make_records())
        finally:
            plt.Figure.savefig = real
        assert width <= max(widths), (
            f"{name}'s PNG is {width}px wide but its canvas is only "
            f"{max(widths):.0f}px: something is hanging off the side and "
            f"`bbox_inches=\"tight\"` is paying for it in width")

    @pytest.mark.parametrize("name,draw_chart,make_records", WIDENING_CHARTS)
    def test_a_far_larger_series_count_still_holds(
            self, name, draw_chart, make_records):
        """Seven engines over nine degrees -- 63 units, and 126 legend entries
        on the churn chart. The bound is a property of the layout, not a
        number that happens to work at the size the run happened to be."""
        units = tuple((engine, m) for engine in ENGINES
                      for m in (4, 8, 16, 24, 32, 48, 64, 96, 128))
        width, height = draw(draw_chart, make_records(units))
        assert width / height <= charts_mod.MAX_ASPECT_RATIO, (
            f"{name} came out {width} x {height} = {width / height:.1f} : 1")


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
