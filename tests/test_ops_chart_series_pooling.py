"""Every ops chart pooled the two resource passes into one series, and the
filtered chart pooled pgvector's two iterative_scan modes on top of that.

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

PR #45 then made pgvector's filtered phase measure two `hnsw.iterative_scan`
modes, which the same key also pooled -- and those differ by an order of
magnitude in recall at the same selectivity, 0.0963 against about 0.97, so a
pooled series plots as one line doubling back on itself at every x with
nothing in the legend to attribute either reading.

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


def _two_scan_modes():
    """PR #45's shape: one pgvector unit, one pass, two scan modes.

    The recalls are the measured order-of-magnitude split the mode decides --
    `off` collapses under a predicate, which is the collapse pgvector 0.8
    shipped iterative scan to fix.
    """
    rows = []
    for mode, recalls in (("off", (0.0963, 0.0971, 0.0990)),
                          ("relaxed_order", (0.9690, 0.9712, 0.9745))):
        for sel, recall in zip((0.01, 0.10, 0.50), recalls):
            rows.append(_filtered_record(
                "tuned", sel, recall,
                1760.0 if mode == "off" else 180.0,
                extra={"iterative_scan": mode}))
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


class TestTheTwoIterativeScanModesAreNotOneSeries:
    """Defect 2, which PR #45 made live for pgvector's filtered phase."""

    @staticmethod
    def _pass_only_key(record):
        """The key as of the resource-pass split: still no scan mode."""
        return (record.get("engine"), loaders_mod.build_signature(record),
                record.get("resource_pass"))

    def test_the_pass_split_alone_still_pooled_them(self):
        """Non-vacuity. One pass, one configuration, two modes: the pass in
        the key does nothing here, which is why this is a second change and
        not a consequence of the first."""
        rows = _two_scan_modes()
        assert len({self._pass_only_key(r) for r in rows}) == 1
        assert len({(r["extra"] or {})["iterative_scan"] for r in rows}) == 2

    def test_the_mode_separates_the_series(self):
        rows = _two_scan_modes()
        assert len({charts_mod.series_key(r) for r in rows}) == 2

    def test_the_pooled_line_doubled_back_on_itself(self):
        """The defect as a reader saw it: one series whose x repeats every
        selectivity and whose y alternates between a tenth and nearly all of
        the correct answers."""
        rows = _two_scan_modes()
        pooled = {}
        for r in rows:
            pooled.setdefault(self._pass_only_key(r), []).append(
                (r["selectivity"], r["recall_at_k"]))
        points = sorted(list(pooled.values())[0])
        assert [x for x, _ in points] == [0.01, 0.01, 0.10, 0.10, 0.50, 0.50]
        assert [y for _, y in points] == [0.0963, 0.9690, 0.0971, 0.9712,
                                          0.0990, 0.9745]

        split = {}
        for r in rows:
            split.setdefault(charts_mod.series_key(r), []).append(
                r["selectivity"])
        assert sorted(sorted(v) for v in split.values()) == [
            [0.01, 0.10, 0.50], [0.01, 0.10, 0.50]]

    def test_the_legend_attributes_each_reading(self):
        """Splitting the lines without naming them leaves the reader exactly
        where they were: two curves a decade apart and no way to say which
        strategy produced either."""
        labels = sorted(charts_mod.series_labels(_two_scan_modes()).values())
        assert labels == [
            "PostgreSQL (pgvector) / iterative_scan=off",
            "PostgreSQL (pgvector) / iterative_scan=relaxed_order"]

    def test_modes_and_passes_compose(self):
        """A `--resource-pass both` run of a two-mode profile is four lines,
        and the legend names both axes."""
        rows = []
        for rp in ("normalized", "tuned"):
            for r in _two_scan_modes():
                row = dict(r, resource_pass=rp)
                row["extra"] = dict(r["extra"])
                rows.append(row)
        assert len({charts_mod.series_key(r) for r in rows}) == 4
        base = "PostgreSQL (pgvector)"
        assert sorted(charts_mod.series_labels(rows).values()) == [
            f"{base} / normalized / iterative_scan=off",
            f"{base} / normalized / iterative_scan=relaxed_order",
            f"{base} / tuned / iterative_scan=off",
            f"{base} / tuned / iterative_scan=relaxed_order"]


class TestTheModeDidNotGoIntoTheSignature:
    """Where it must NOT go, and the twelve rows that decided it.

    `extra.iterative_scan` looks like a peer of `extra.quantization`, which
    `build_signature`'s column fallback does list. It is not one.
    `quantization` is None on every record of every phase in every shipped
    profile, so it cannot separate two records that belong together;
    `iterative_scan` is written by `harness/workloads/filtered.py` and by
    nothing else, so within ONE ops unit the filtered records carry a mode
    while that unit's calibration, concurrency and churn records carry None.

    `loaders.operating_points` joins a calibration sweep to the workloads that
    consumed it, across exactly those phases, through `build_signature`. On
    `full-20261015-215006` that join goes from 54 rows to 66 if the field is
    added to the fallback: twelve pgvector filtered half-rows, each with no
    recall, no grid size and no point count, stranded on a key their own
    calibration curve cannot reach. None of the 54 is lost, so nothing in the
    suite would have caught it.
    """

    def _unit(self):
        """One ops unit, two phases, as the harness writes it: a calibration
        record with no mode, and the filtered records that followed it."""
        rows = [{
            "phase": "ef_calibration", "engine": "pgvector",
            "dataset": "fashion-mnist-784-euclidean",
            "resource_pass": "tuned", "m": 16, "build_mode": "post",
            "storage_engine": "InnoDB", "ef_construction": 200,
            "ef_search": ef, "recall_at_k": recall, "extra": {},
        } for ef, recall in ((10, 0.91), (16, 0.95), (40, 0.99))]
        for mode in ("off", "relaxed_order"):
            rows.append(_filtered_record(
                "tuned", 0.10, 0.0963 if mode == "off" else 0.969, 900.0,
                ef_search=16,
                extra={"iterative_scan": mode, "calibration_status": "ok"}))
        return rows

    def test_the_signature_is_blind_to_the_mode(self):
        rows = self._unit()
        assert len({loaders_mod.build_signature(r) for r in rows}) == 1

    def test_the_sweep_and_its_workloads_stay_one_operating_point(self):
        """The join this would have split. One unit in, one row out, carrying
        both halves -- the chosen ef from the filtered records and the recall
        from the calibration curve at that ef."""
        rows = loaders_mod.operating_points(self._unit())
        assert len(rows) == 1, rows
        assert rows[0]["ef_search"] == 16
        assert rows[0]["recall_at_k"] == 0.95
        assert rows[0]["points_measured"] == 3

    def test_the_mode_is_in_the_chart_key_and_not_the_unit_key(self):
        """Both facts in one assertion, because together they are the design:
        the charts separate the modes, the unit join does not see them."""
        rows = [r for r in self._unit() if r["phase"] == "filtered"]
        assert len({charts_mod.series_key(r) for r in rows}) == 2
        assert len({loaders_mod.operating_point_key(r) for r in rows}) == 1


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

    def test_records_with_no_scan_mode_group_exactly_as_before(self):
        """`extra.iterative_scan` is absent on every pre-#45 ops record and on
        every record of the other six engines, so it reads None. This is the
        assertion that the key's growth is inert for all of them: the grouping
        and the labels are identical to what the key produced before the field
        was part of it, not merely similar.
        """
        rows = []
        for engine in ("mariadb", "alisql", "valkey", "mongodb", "percona"):
            for sel in (0.01, 0.10, 0.50):
                rows.append(dict(_filtered_record("tuned", sel, 0.97, 500.0),
                                 engine=engine))
        without = {}
        for r in rows:
            # The key with the mode sliced back off, which is what it was.
            without.setdefault(charts_mod.series_key(r)[:-1], []).append(r)
        with_mode = {}
        for r in rows:
            with_mode.setdefault(charts_mod.series_key(r), []).append(r)
        assert len(with_mode) == len(without) == 5
        assert ([sorted(id(r) for r in v) for v in sorted(
                    with_mode.values(), key=lambda v: v[0]["engine"])]
                == [sorted(id(r) for r in v) for v in sorted(
                    without.values(), key=lambda v: v[0]["engine"])])
        assert all(not label.endswith("unset")
                   for label in charts_mod.series_labels(rows).values())

    def test_a_series_with_no_mode_beside_one_with_a_mode_says_so(self):
        """`off` and "the engine's own default" are different measurements and
        the record distinguishes them -- `cli._ops_iterative_scan` returns None
        for a profile that declares nothing, and pinning `off` explicitly was
        worth doing on its own grounds. A bare label beside a named one would
        have published them as the same line."""
        rows = _two_scan_modes()[:3] + [
            _filtered_record("tuned", sel, 0.5, 700.0)
            for sel in (0.01, 0.10, 0.50)]
        assert sorted(charts_mod.series_labels(rows).values()) == [
            "PostgreSQL (pgvector) / iterative_scan=off",
            "PostgreSQL (pgvector) / iterative_scan=unset"]


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

    def test_the_two_scan_modes_reach_plot_as_two_attributed_lines(self):
        """The whole defect, at the level the chart hands to matplotlib:
        before, one line with x = [0.01, 0.01, 0.1, 0.1, 0.5, 0.5] and y
        alternating 0.0963 / 0.9690; after, two lines of three points each,
        each legend naming its mode."""
        plotted = self._plotted(_two_scan_modes())
        assert len(plotted) == 2, plotted
        by_label = {c[2]: (c[0], c[1]) for c in plotted}
        assert sorted(by_label) == [
            "PostgreSQL (pgvector) / iterative_scan=off",
            "PostgreSQL (pgvector) / iterative_scan=relaxed_order"]
        off = by_label["PostgreSQL (pgvector) / iterative_scan=off"]
        relaxed = by_label[
            "PostgreSQL (pgvector) / iterative_scan=relaxed_order"]
        assert off == ([0.01, 0.10, 0.50], [0.0963, 0.0971, 0.0990])
        assert relaxed == ([0.01, 0.10, 0.50], [0.9690, 0.9712, 0.9745])


# ---------------------------------------------------------------------------
# The FILE branch of build_signature
# ---------------------------------------------------------------------------

def _ann_record(resource_pass, ef_search, *, engine="mariadb123", m=16,
                stem=None, **kw):
    """One recall/QPS record of the shape `loaders.load_annb_results` writes.

    `extra.source_file` is a path RELATIVE to the ann results tree it was read
    from, and those trees are per resource pass
    (`results/annb/<pass>/<fingerprint>/`), so one build configuration
    measured in both passes produces the IDENTICAL relative path. That is why
    the file branch is every bit as pass-blind as the column fallback, and it
    is the claim this fixture exists to make executable. Shape copied from a
    real record of `full-20261015-215006`:

        fashion-mnist-784-euclidean/10/mariadb123/euclidean_M_8_engine_InnoDB_200.hdf5
    """
    stem = stem or f"euclidean_M_{m}_engine_InnoDB"
    record = {
        "phase": "recall_qps", "engine": engine,
        "dataset": "fashion-mnist-784-euclidean",
        "resource_pass": resource_pass, "m": m, "storage_engine": "InnoDB",
        "ef_search": ef_search, "recall_at_k": 0.97, "qps": 900.0,
        "latency_p50_ms": 1.2 + ef_search / 1000.0,
        "latency_p99_ms": 3.4 + ef_search / 500.0,
        "extra": {"source_file": f"fashion-mnist-784-euclidean/10/{engine}/"
                                 f"{stem}_{ef_search}.hdf5"},
    }
    record.update(kw)
    return record


def _two_passes_ann():
    """One ann configuration, both passes, a four-point ef sweep each."""
    return [_ann_record(rp, ef) for rp in ("normalized", "tuned")
            for ef in (10, 120, 160, 200)]


class TestTheFileBranchIsSplitByThePassToo:
    """**The widest claim this change makes, and it rested on nothing.**

    `series_key`'s docstring argues that BOTH branches of `build_signature`
    are blind to the resource pass, and that this is therefore not a
    column-fallback defect. Every other fixture in this file is an ops record
    with no `extra.source_file`, so every one of them exercises the column
    branch -- which meant a regression applying the pass on the `cols` branch
    only would have shipped with the whole suite green, and
    `latency_percentiles`, the one chart here that plots ann records, would
    have silently resumed pooling the two passes.

    Measured on `full-20261015-215006`: 590 ann records, 52 distinct
    file-branch signatures, and **21 of those 52 span both resource passes.**
    """

    def test_the_fixture_really_does_take_the_file_branch(self):
        """Non-vacuity, and the one assertion that keeps this class honest: if
        the fixture ever loses its `source_file` it silently becomes a fourth
        copy of the column-branch tests."""
        signatures = {loaders_mod.build_signature(r)
                      for r in _two_passes_ann()}
        assert all(s[0] == "file" for s in signatures), signatures

    def test_the_file_branch_collides_across_the_two_passes(self):
        """The mechanism: per-pass trees, identical relative path. One
        signature for both passes, which is what the charts then pooled."""
        rows = _two_passes_ann()
        assert len({loaders_mod.build_signature(r) for r in rows}) == 1
        assert len({r["resource_pass"] for r in rows}) == 2
        assert len({(r["extra"] or {})["source_file"] for r in rows
                    if r["ef_search"] == 200}) == 1

    def test_the_pass_separates_the_series_on_this_branch(self):
        rows = _two_passes_ann()
        assert len({charts_mod.series_key(r) for r in rows}) == 2

    def test_the_label_names_the_pass_on_this_branch(self):
        """`series_labels`' filename branch compares result stems within an
        engine and finds nothing differing, so it returns the bare engine
        label for both passes -- the pass can only come from
        `_key_suffixes`."""
        labels = sorted(charts_mod.series_labels(_two_passes_ann()).values())
        assert labels == ["MariaDB 12.3 (MHNSW) / normalized",
                          "MariaDB 12.3 (MHNSW) / tuned"]

    def test_an_ann_record_carries_no_scan_mode_and_is_not_split_by_one(self):
        """`extra.iterative_scan` is written by the ops harness only, so on
        this branch it is absent on every record. The mode must therefore do
        nothing here at all."""
        rows = _two_passes_ann()
        assert not any((r["extra"] or {}).get("iterative_scan") for r in rows)
        assert len({charts_mod.series_key(r) for r in rows}) == 2
        assert not any("iterative_scan" in label for label
                       in charts_mod.series_labels(rows).values())

    @needs_matplotlib
    def test_latency_percentiles_draws_one_line_per_pass(self):
        """At the level `latency_percentiles` hands to matplotlib.

        Before: one series per configuration, so both passes' p50 points
        landed in one sorted-by-ef list and the line ran 10, 10, 120, 120,
        160, 160, 200, 200 -- doubling back at every search width. After: two
        lines of four points, and because the engine is the panel this chart
        strips the engine off the label, so the legend reads bare
        "normalized" and "tuned".
        """
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
            charts_mod.latency_percentiles(
                _two_passes_ann(), "fashion-mnist-784-euclidean",
                tempfile.mkdtemp(), "ev")
        finally:
            Axes.plot = real

        labelled = [c for c in calls if c[2] is not None]
        assert len(labelled) == 2, labelled
        assert sorted(c[2] for c in labelled) == ["normalized", "tuned"]
        for xs, _ys, _label in labelled:
            assert xs == [10, 120, 160, 200], xs


class TestTheResourcePassIsNamedOnceAndOnlyOnce:
    """`churn_impact` named the pass itself while `series_key` lacked it.

    Removing that block was necessary -- keeping it alongside the shared label
    printed the pass twice, measured over a two-pass churn fixture as
    "MariaDB 11.8 (MHNSW) / tuned / tuned" -- and nothing then stopped it
    coming back. This is that guard.
    """

    def _rows(self):
        rows = []
        for resource_pass in ("normalized", "tuned"):
            for frac, qps in ((0.0, 100.0), (0.10, 40.0), (0.25, 30.0)):
                rows.append({
                    "phase": "churn", "engine": "mariadb", "dataset": "d",
                    "resource_pass": resource_pass, "m": 16,
                    "build_mode": "post", "storage_engine": "InnoDB",
                    "churn_fraction": frac, "qps": qps, "recall_at_k": 0.9,
                    "extra": {},
                })
        return rows

    def test_the_shared_label_carries_the_pass_exactly_once(self):
        """Pure logic, so this half holds on a host with no matplotlib."""
        rows = self._rows()
        retained = charts_mod.churn_retention(rows, "d")
        labels = charts_mod.series_labels(rows)
        assert len(retained) == 2, retained
        for key in retained:
            label = labels[key[:charts_mod.SERIES_KEY_WIDTH]]
            name = key[charts_mod.SERIES_KEY_WIDTH]
            assert label.count(name) == 1, (label, name)

    @needs_matplotlib
    def test_no_plotted_legend_entry_names_a_pass_twice(self):
        import tempfile

        from matplotlib.axes import Axes

        labels = []
        real = Axes.plot

        def recording(self, *args, **kwargs):
            if kwargs.get("label"):
                labels.append(kwargs["label"])
            return real(self, *args, **kwargs)

        Axes.plot = recording
        try:
            assert charts_mod.churn_impact(
                self._rows(), "d", tempfile.mkdtemp(), "ci") is not None
        finally:
            Axes.plot = real

        assert len(labels) == 2, labels
        assert sorted(labels) == ["MariaDB 11.8 (MHNSW) / normalized",
                                  "MariaDB 11.8 (MHNSW) / tuned"]
        for label in labels:
            for name in ("normalized", "tuned"):
                assert label.count(name) <= 1, label
