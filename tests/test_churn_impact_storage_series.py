"""`churn_impact` built its storage-engine list out of literal `None`s, so the
chart raised and was dropped from every report that measured churn.

The body was

    storages: Dict[str, List[Optional[str]]] = {}
    for key in {series_key(r) for r in churn_records}:
        storages.setdefault(key[0], []).append(None)

-- one entry per series, and the entry is `None` rather than that series'
`storage_engine`. `series_style` then asks `storages.index(record["storage_engine"])`
for the linestyle, but only `if storages and len(storages) > 1`, which is the
whole reason this survived: an engine with ONE series gets `[None]`, the guard
is False, the bad value is never looked up and the chart draws. The second
series for the same engine makes the list `[None, None]`, the guard opens, and
`.index("InnoDB")` raises `ValueError: 'InnoDB' is not in list`.

**Measured on `PS/new/full-20261015-215006`**, the seven-engine
`--resource-pass both` run of 2026-10-15, regenerated against this tree:

    [report] chart churnimpact-fashion-mnist-784-euclidean failed: 'InnoDB' is not in list
    [report] chart churnimpact-glove-100-angular failed: 'InnoDB' is not in list
    [report] chart churnimpact-sift-128-euclidean failed: 'InnoDB' is not in list

Every dataset that ran the churn workload, three of four. `generate.py` catches
per chart so one bad figure cannot kill a twenty-hour run's report, prints that
line and continues with `paths = None` -- so the stem never enters
`chart_paths` and neither renderer emits or links it. The bundle's own shipped
`report/` is the proof that this is not theoretical: 47 SVGs, `churn-*` among
them, and no `churnimpact-*` at all.

What was lost is the only figure in the report that puts throughput retention
under churn on one axis. On fashion-mnist it is 48 series. Once a tenth of the
corpus has been deleted and re-inserted, MariaDB 11.8 holds 0.508-0.586 of its
pre-churn QPS and MariaDB 12.3 holds 0.470-0.571, against AliSQL 0.837-0.892,
Percona Server 0.793-0.912, Percona Search 0.867-0.981, Valkey 0.893-0.971 and
pgvector 0.857-1.004; at a quarter replaced the two MHNSW builds are at
0.443-0.531 and 0.363-0.517 while pgvector is still at 0.877-1.009. "MHNSW
loses half its throughput where pgvector loses none" is a headline of this
benchmark and it was in no published report.

**Why the suite passed.** The one test that rendered this chart,
`test_config.py::TestTwoGraphDegreesAreTwoLines::test_churn_still_renders_with_the_wider_key`,
builds two records of ONE configuration of ONE engine -- one series, so
`storages == {"mariadb": [None]}`, `len(storages) > 1` is False and the literal
never reaches `.index`. `test_ops_chart_series_pooling.py` names `churn_impact`
in three docstrings and renders only `filtered`. So the gap is not that no
fixture had a storage engine; it is that no fixture gave one engine two
series, which is what every real run does and what the fixtures here do.

These tests do not read that bundle -- `results/` is gitignored and a test that
needs a 194 MB archive is not a test. They rebuild its shape at the smallest
size that reaches the guard: one engine, two storage engines, two graph
degrees, which is four series where the real run has six.
"""
import ast
import os
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


def _churn_record(storage_engine, m, fraction, qps):
    """One churn record of the shape the 2026-10-15 run wrote.

    `build_signature` takes its column branch for these -- the ops harness
    writes no `extra.source_file` and all 203 churn records in that bundle
    confirmed it -- so `storage_engine` and `m` are inside `series_key` and are
    constant within a series. That is what makes `_grouped`'s one-sample-per-
    series read of them correct rather than merely convenient.
    """
    return {
        "phase": "churn", "engine": "mariadb",
        "dataset": "fashion-mnist-784-euclidean",
        "resource_pass": "tuned", "build_mode": "post",
        "storage_engine": storage_engine, "m": m, "ef_construction": None,
        "churn_fraction": fraction, "qps": qps, "recall_at_k": 0.95,
    }


def _two_storage_engines():
    """MariaDB measured on InnoDB and on MyISAM, two graph degrees each.

    Four series for one engine. The storage pair is the one
    `churn_retention`'s docstring records a published defect against, so it is
    the shape this chart has already been wrong about once.
    """
    rows = []
    for storage in ("InnoDB", "MyISAM"):
        for m in (8, 16):
            for fraction, qps in ((0.0, 1000.0), (0.1, 500.0), (0.25, 400.0)):
                rows.append(_churn_record(storage, m, fraction, qps))
    return rows


def _one_series():
    """What the pre-existing fixture gives: one configuration, one engine."""
    return [_churn_record("InnoDB", 16, fraction, qps)
            for fraction, qps in ((0.0, 1000.0), (0.1, 400.0))]


def _plotted(records):
    """Every `(label, linestyle, linewidth, alpha)` `churn_impact` draws.

    Asserted at the `plot()` boundary rather than off a returned structure
    because the defect was in what the chart handed matplotlib, and because
    the storage engine and the graph degree reach the figure as a linestyle
    and a weight and in no other form -- there is nothing else to read them
    off.
    """
    from matplotlib.axes import Axes

    calls = []
    real = Axes.plot

    def recording(self, *args, **kwargs):
        if kwargs.get("label"):
            calls.append({"label": kwargs["label"],
                          "linestyle": kwargs.get("linestyle"),
                          "linewidth": kwargs.get("linewidth"),
                          "alpha": kwargs.get("alpha")})
        return real(self, *args, **kwargs)

    Axes.plot = recording
    try:
        with tempfile.TemporaryDirectory() as out:
            paths = charts_mod.churn_impact(
                records, "fashion-mnist-784-euclidean", out, "ev")
            assert paths is not None
    finally:
        Axes.plot = real
    return calls


@needs_matplotlib
class TestTheStorageEngineListHoldsStorageEngines:
    """The crash itself, and the linestyle it was supposed to be choosing."""

    def test_one_engine_with_two_series_renders(self):
        """The reproduction. Before: `ValueError: 'InnoDB' is not in list`,
        raised out of `series_style` and swallowed by `generate.py` into one
        printed line and a missing figure."""
        assert len(_plotted(_two_storage_engines())) == 4

    def test_each_storage_engine_gets_its_own_linestyle(self):
        """What the broken list was for: colour is the engine, so within one
        engine the linestyle is the only channel the storage engine has."""
        by_label = {c["label"]: c["linestyle"] for c in _plotted(
            _two_storage_engines())}
        innodb = {ls for label, ls in by_label.items() if "InnoDB" in label}
        myisam = {ls for label, ls in by_label.items() if "MyISAM" in label}
        assert innodb == {charts_mod.STORAGE_LINESTYLES[0]}, by_label
        assert myisam == {charts_mod.STORAGE_LINESTYLES[1]}, by_label

    def test_the_list_is_de_duplicated(self):
        """Two storage engines over two graph degrees is four series. A list
        that collects one entry per SERIES rather than per distinct value
        holds `[InnoDB, InnoDB, MyISAM, MyISAM]`, and `.index("MyISAM")` is 2
        there against 1 here -- so every series past the first distinct value
        silently shifts to a different linestyle as a run adds degrees. The
        assertion is on the second value for that reason: the first is index 0
        either way and proves nothing."""
        myisam = {c["linestyle"] for c in _plotted(_two_storage_engines())
                  if "MyISAM" in c["label"]}
        assert myisam == {"--"}
        assert myisam != {":"}

    def test_a_single_storage_engine_leaves_the_linestyle_alone(self):
        """`series_style`'s `len(storages) > 1` guard is load-bearing and is
        not being relaxed: on the run that produced the reproduction every
        churn record of all seven engines carries `storage_engine: InnoDB`, so
        after this fix each engine's list is one value long and the guard stays
        shut. The engine's own linestyle survives, which is what keeps
        `pgvector` dash-dotted and `alisql` dashed on a chart that never varied
        storage."""
        drawn = {c["linestyle"] for c in _plotted(
            [_churn_record("InnoDB", m, f, q)
             for m in (8, 16)
             for f, q in ((0.0, 1000.0), (0.1, 500.0))])}
        assert drawn == {charts_mod.style_for("mariadb")["linestyle"]}


@needs_matplotlib
class TestTheGraphDegreesAreToldApart:
    """`churn_impact` never built a `degrees` list at all, which is the same
    root cause one step further on: `_grouped` returns one and the hand-rolled
    block had no equivalent. Six series per engine on the 2026-10-15 run --
    three degrees over two passes -- arrived at the same colour, marker,
    linestyle and width, distinguishable only by reading the legend, which is
    verbatim the defect `series_style`'s own comment records for AliSQL."""

    def test_two_degrees_reach_plot_with_different_weights(self):
        weights = {(c["linewidth"], c["alpha"]) for c in _plotted(
            _two_storage_engines())}
        assert len(weights) == 2, weights

    def test_the_denser_graph_is_the_prominent_one(self):
        """`series_style`'s rule, asserted on this chart's output rather than
        restated: the largest M stays solid and heavy and the sparser degrees
        recede, so which is which is legible without the legend."""
        drawn = _plotted(_two_storage_engines())
        dense = {(c["linewidth"], c["alpha"]) for c in drawn
                 if "M=16" in c["label"]}
        sparse = {(c["linewidth"], c["alpha"]) for c in drawn
                  if "M=8" in c["label"]}
        assert len(dense) == 1 and len(sparse) == 1, drawn
        assert dense.pop() > sparse.pop()

    def test_a_single_degree_draws_exactly_as_before(self):
        """A run that swept one M must be byte-identical to what shipped. The
        weight falls back to the 1.9 this chart has always hardcoded and no
        alpha is passed, because `series_style` sets neither key unless more
        than one degree is present."""
        drawn = _plotted(_one_series())
        assert [(c["linewidth"], c["alpha"]) for c in drawn] == [(1.9, None)]


class TestTheSharedHelperIsReused:
    """A second hand-rolled copy of `_grouped`'s de-duplicating loop is how
    this diverged: the correct version sits forty lines up in the same file and
    `charts.churn`, the sibling chart on the same records, calls it."""

    def _tree(self):
        """`churn_impact` parsed -- its live statements, not its prose.

        Pinned on the host as well as behaviourally because the tests above
        need matplotlib, which this repo deliberately keeps inside the bench
        images, so on a host set up from the documented prerequisites they
        skip and only this class runs. A regression that none of the tests
        which actually execute can see is not caught.

        Two reasons it is the AST and not the source text, both of them
        mistakes this file made on its way in.

        The text idiom `test_config.py` uses on this same function,
        `source.split("def churn_impact")[1]`, returns everything to end of
        file. `calibration_curves` sits below and calls `_grouped(points)`, so
        `"_grouped(" in block` passed over that slice on the unfixed tree.
        A negative assertion survives an over-broad slice; a positive one is
        satisfied by the neighbours.

        And a text search for the defect cannot tell live code from the
        comment describing it. Comments in this codebase carry the incident
        behind a decision and quote the code they replaced, so the repaired
        `churn_impact` contains the string `append(None)` inside the paragraph
        that explains why it no longer runs one -- which failed this test on
        the fixed tree. Asking the AST whether a `.append(None)` call EXISTS
        answers the question the text search was approximating.
        """
        source = open(os.path.join(VB_ROOT, "report", "charts.py")).read()
        return next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "churn_impact")

    def test_churn_impact_calls_grouped(self):
        assert "_grouped" in {node.func.id
                              for node in ast.walk(self._tree())
                              if isinstance(node, ast.Call)
                              and isinstance(node.func, ast.Name)}

    def test_it_appends_no_literal_none_anywhere(self):
        """The defect in one assertion: a list built by appending `None` is a
        list `series_style` cannot look a storage engine up in."""
        appends = [node for node in ast.walk(self._tree())
                   if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Attribute)
                   and node.func.attr == "append"
                   and len(node.args) == 1
                   and isinstance(node.args[0], ast.Constant)
                   and node.args[0].value is None]
        assert appends == [], [node.lineno for node in appends]

    def test_the_helper_gives_the_same_labels_the_chart_had(self):
        """Non-vacuity for the reuse: `_grouped`'s second return value is
        `series_labels(records)` over the identical record set, so swapping the
        hand-rolled block for the helper changes the storage and degree lists
        and provably changes nothing about the legend."""
        rows = _two_storage_engines()
        from report.churn_state import churn_measurements
        churn_records = [r for r in churn_measurements(rows)
                         if r.get("dataset") == "fashion-mnist-784-euclidean"
                         and r.get("qps")]
        _groups, labels, _st, _deg = charts_mod._grouped(churn_records)
        assert labels == charts_mod.series_labels(churn_records)

    def test_the_old_fixture_could_not_reach_the_index(self):
        """Why the suite was green, asserted rather than asserted about. One
        series per engine is `[None]`, and `series_style` returns the engine's
        own linestyle without ever looking the record's storage engine up --
        so the literal was inert for exactly as long as no fixture gave one
        engine a second series."""
        record = _churn_record("InnoDB", 16, 0.1, 500.0)
        assert charts_mod.series_style(record, [None])["linestyle"] == \
            charts_mod.style_for("mariadb")["linestyle"]
        with pytest.raises(ValueError):
            charts_mod.series_style(record, [None, None])
