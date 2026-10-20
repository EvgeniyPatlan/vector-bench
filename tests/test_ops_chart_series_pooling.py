"""Every ops chart pooled the two resource passes into one series.

`charts.series_key` was `(engine, build_signature(record))` and both branches
of `build_signature` are blind to `resource_pass`: the column fallback does
not list it, and the file branch reads a path relative to a per-pass tree, so
one configuration measured in both passes produces the same relative path.
Six charts group through that key, so on a `--resource-pass both` run each of
their series carried two points at every x and the line between them was one
no machine ever produced.

Measured on `PS/new/full-20261015-215006`, the seven-engine run of 2026-10-15:
the fashion-mnist filtered chart drew 24 series and every single one of them
had x = [0.01, 0.01, 0.1, 0.1, 0.5, 0.5].

These tests do not read that bundle -- `results/` is gitignored and a test
that needs a 126 MB archive is not a test. They rebuild its shape: two passes
of one configuration, which is the smallest set that reproduces the pooling.
"""
import os
import sys

import pytest

VB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if VB_ROOT not in sys.path:
    sys.path.insert(0, VB_ROOT)

from report import charts as charts_mod  # noqa: E402
from report import loaders as loaders_mod  # noqa: E402

needs_matplotlib = pytest.mark.skipif(
    not charts_mod.have_matplotlib(),
    reason="matplotlib is a container dependency, not a host one")


def _filtered_record(resource_pass, selectivity, recall, qps, **kw):
    """One filtered record of the shape the 2026-10-15 run wrote."""
    extra = {"filter_column": "tag"}
    extra.update(kw.pop("extra", {}))
    record = {
        "phase": "filtered", "engine": "pgvector",
        "dataset": "fashion-mnist-784-euclidean",
        "resource_pass": resource_pass, "m": 16, "build_mode": "post",
        "storage_engine": "InnoDB", "ef_construction": 200,
        "selectivity": selectivity, "recall_at_k": recall, "qps": qps,
        "extra": extra,
    }
    record.update(kw)
    return record


def _two_passes():
    """One pgvector configuration, both passes, three selectivities."""
    rows = []
    for rp, bump in (("normalized", 0.0), ("tuned", 0.0012)):
        for sel, recall, qps in ((0.01, 1.0, 1850.0),
                                 (0.10, 0.0965, 1760.0),
                                 (0.50, 0.4989, 1240.0)):
            rows.append(_filtered_record(rp, sel, round(recall + bump, 6),
                                         qps + (40.0 if rp == "tuned" else 0.0)))
    return rows


class TestTheTwoResourcePassesAreNotOneSeries:
    """Defect 1, which predates PR #45 and moves all seven engines."""

    @staticmethod
    def _old_key(record):
        """The pre-fix key, verbatim."""
        return (record.get("engine"), loaders_mod.build_signature(record))

    def test_the_old_key_pooled_them(self):
        """Non-vacuity: without this fix the fixture really does collide.

        If this ever stops failing to separate, the fixture has drifted and
        every assertion below is passing on a set that was never pooled.
        """
        rows = _two_passes()
        assert len({self._old_key(r) for r in rows}) == 1
        assert len({r["resource_pass"] for r in rows}) == 2

    def test_the_pass_separates_the_series(self):
        rows = _two_passes()
        keys = {charts_mod.series_key(r) for r in rows}
        assert len(keys) == 2, keys

    def test_one_point_per_x_in_each_series(self):
        """The pooling's observable signature: two points at every x.

        Old form, on this fixture: one series with
        x = [0.01, 0.01, 0.1, 0.1, 0.5, 0.5] -- which is what all 24 of the
        2026-10-15 fashion-mnist series looked like.
        """
        rows = _two_passes()
        old = {}
        for r in rows:
            old.setdefault(self._old_key(r), []).append(r["selectivity"])
        assert [sorted(v) for v in old.values()] == [
            [0.01, 0.01, 0.10, 0.10, 0.50, 0.50]]

        groups = {}
        for r in rows:
            groups.setdefault(charts_mod.series_key(r), []).append(r["selectivity"])
        assert sorted(sorted(v) for v in groups.values()) == [
            [0.01, 0.10, 0.50], [0.01, 0.10, 0.50]]

    def test_the_engine_is_still_the_first_element(self):
        """Five chart bodies index `key[0]` for the engine to look up its
        storage engines, its graph degrees and its calibration mark. Widening
        the key in front of the engine would have fed `style_for` a build
        signature and coloured every series as an unknown engine."""
        key = charts_mod.series_key(_two_passes()[0])
        assert key[0] == "pgvector"

    def test_the_constant_moved_with_the_key(self):
        """`churn` and `churn_impact` slice this key back at
        `SERIES_KEY_WIDTH`. `tests/test_config.py` pins the equality too; it is
        restated here because it is the one thing that makes widening the key
        a one-line change rather than an eight-site one."""
        assert charts_mod.SERIES_KEY_WIDTH == len(
            charts_mod.series_key({"engine": "e"}))


class TestASinglePassReportDoesNotMove:
    """The fix must split a two-pass run and leave a one-pass run alone.

    Every chart in this repo has had a legend regression from an axis being
    named that the run never varied, which is why `series_labels`,
    `pareto_series_style`, `charts.churn` and `churn_impact` all carry the same
    "only when more than one" rule.
    """

    def test_a_one_pass_label_is_unchanged(self):
        rows = [_filtered_record("tuned", sel, 0.9, 100.0)
                for sel in (0.01, 0.10, 0.50)]
        assert set(charts_mod.series_labels(rows).values()) == {
            "PostgreSQL (pgvector)"}

    def test_a_two_pass_label_names_the_pass(self):
        labels = sorted(charts_mod.series_labels(_two_passes()).values())
        assert labels == ["PostgreSQL (pgvector) / normalized",
                          "PostgreSQL (pgvector) / tuned"]

    def test_a_record_with_no_pass_is_not_split_from_another(self):
        """A `--from-records` rebuild of an archive predating the field has
        `resource_pass` None on every record. None is one value, so the series
        stay exactly as they were rather than splitting on an absence."""
        rows = [_filtered_record(None, sel, 0.9, 100.0)
                for sel in (0.01, 0.10, 0.50)]
        assert len({charts_mod.series_key(r) for r in rows}) == 1
        assert set(charts_mod.series_labels(rows).values()) == {
            "PostgreSQL (pgvector)"}


class TestTheSignatureStaysPassBlind:
    """Where the pass went, and where it must not go.

    `loaders.build_signature` is the join key between a calibration sweep and
    the workloads that consumed it (`loaders.operating_points`), the key
    `generate.py`'s duplicate and inversion detectors are built on, and the
    key `render.py`'s unit tables are grouped by. `operating_point_key`
    composes the pass on top of it instead, and
    `tests/test_e14_per_pass_manifest.py` asserts the signature collides
    across passes. `series_key` follows that pattern; this pins that it did.
    """

    def test_the_signature_does_not_name_the_pass(self):
        rows = _two_passes()
        assert len({loaders_mod.build_signature(r) for r in rows}) == 1

    def test_the_composition_matches_operating_point_key(self):
        """Both keys are the signature plus the pass. If one ever stops being
        so, a chart series and an operating-point row stop meaning the same
        unit, which is the drift `calibration_curves`' ring marks would show
        first and as a measurement disagreement rather than a key bug."""
        record = _two_passes()[0]
        series = charts_mod.series_key(record)
        unit = loaders_mod.operating_point_key(record)
        assert set(unit) == {series[1], series[2]}


@needs_matplotlib
class TestTheFilteredChartRendersTheSplit:
    """The grouping is what `filtered` hands to `plot()`, asserted there."""

    def _plotted(self, records, column="tag"):
        import tempfile

        from matplotlib.axes import Axes

        calls = []
        real = Axes.plot

        def recording(self, *args, **kwargs):
            if len(args) >= 2 and isinstance(args[0], list):
                calls.append((list(args[0]), list(args[1]),
                              kwargs.get("label")))
            return real(self, *args, **kwargs)

        Axes.plot = recording
        try:
            charts_mod.filtered(records, "fashion-mnist-784-euclidean",
                                tempfile.mkdtemp(), "ev", column=column)
        finally:
            Axes.plot = real
        return [c for c in calls if c[2] is not None]

    def test_two_lines_with_one_point_per_x_and_distinct_legends(self):
        plotted = self._plotted(_two_passes())
        assert len(plotted) == 2, plotted
        assert sorted(c[2] for c in plotted) == [
            "PostgreSQL (pgvector) / normalized",
            "PostgreSQL (pgvector) / tuned"]
        for xs, _ys, _label in plotted:
            assert xs == [0.01, 0.10, 0.50], xs
