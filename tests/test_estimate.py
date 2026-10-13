"""Tests for the run-time estimate: `orchestrator/estimate.py`.

This function had no tests at all, which is how it came to be load-bearing for a
convention it could not satisfy. The convention says the printed estimate must
include every cost and that *a run longer than its printed estimate is a bug* —
but the function modelled ingest only, said so in its label, and every run had
therefore always exceeded it.

Two things are pinned here, in this order of importance:

1. **The ingest numbers did not move.** They are the published figures: the
   printed block, `docs/07-planning-a-run.md`, and two web-UI renderers all
   carry them. Every real profile's ingest total is pinned to the value it
   produced before query time existed, so a reader can see it did not change.
2. **`total_hours` still means ingest.** `webui/static/control.js` and
   `webui/static/setup.js` both render it as `estimated ingest ~N h`. Silently
   redefining it would have changed what the UI has been telling people without
   touching the UI.

The arithmetic tests state their own numbers. An estimate is arithmetic over a
config, so a test that recomputes the implementation's expression passes
whatever that expression is; the hand-computed cases here are written out from
the profile and the measured rate tables so a wrong formula cannot satisfy them.
"""

from __future__ import annotations

import io
import os
import re
import sys
import unittest.mock as mock
from contextlib import redirect_stdout

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator import cli as cli_mod  # noqa: E402
from orchestrator import estimate as est_mod  # noqa: E402
from orchestrator.cli import KNOWN_ENGINES  # noqa: E402
from orchestrator.config import load_profile  # noqa: E402
from orchestrator.estimate import (print_load_estimate,  # noqa: E402
                                   estimate_load_hours)

VB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ALL = list(KNOWN_ENGINES)
BOTH_PHASES = ["ann", "ops"]


def est(profile_name, engines=None, datasets=None, passes=("normalized",),
        phases=BOTH_PHASES):
    profile = load_profile(profile_name)
    return estimate_load_hours(
        profile, list(engines or ALL),
        list(datasets if datasets is not None else profile.get("datasets", [])),
        list(passes), list(phases))


def printed(profile_name, engines=None, datasets=None, passes=("normalized",),
            phases=BOTH_PHASES):
    profile = load_profile(profile_name)
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_load_estimate(
            profile, list(engines or ALL),
            list(datasets if datasets is not None else profile.get("datasets", [])),
            list(passes), list(phases))
    return buf.getvalue()


def term(estimate, name):
    found = [t for t in estimate["query_terms"] if t["name"] == name]
    assert len(found) == 1, (
        f"{name!r} not in {[t['name'] for t in estimate['query_terms']]}")
    return found[0]


# A budget is enforced by comparing elapsed time against it, which is what
# `churn.py:244` does: `if time.time() - started > budget_s`. Accepting a
# mention of the word instead passed two states in which nothing capped
# anything: the realistic one, where the flag is plumbed from ops_pass through
# the argument parser into the workload's signature and never looked at again,
# and the trivial one, where a comment in the workload says no budget is
# applied -- and in this codebase, prose about a missing bound is likely.
_ENFORCES_A_BUDGET = re.compile(
    r"(?:time\.(?:time|monotonic)|perf_counter)\s*\(\s*\)[^\n]*[<>][^\n]*budget"
    r"|budget[^\n]*[<>][^\n]*(?:time\.(?:time|monotonic)|perf_counter)\s*\(\s*\)")


def _filtered_budget_is_enforced() -> bool:
    """Whether the filtered workload stops when its budget runs out.

    Only `harness/workloads/filtered.py` is consulted. `harness/main.py` was in
    an earlier version of this check and is not enforcement: parsing a flag and
    handing it on is exactly the state that passed while the query loop ran
    unbounded.

    If T3c enforces the budget in a shape this pattern does not recognise -- a
    precomputed deadline, say -- widen the pattern deliberately and say in the
    commit how the loop stops. Do not relax it to a mention.
    """
    # Calibrated against the one budget this repo does enforce, so a pattern
    # that has stopped matching real enforcement fails here rather than
    # silently reporting that nothing is enforced anywhere.
    with open(os.path.join(VB_ROOT, "harness", "workloads", "churn.py")) as fh:
        assert _ENFORCES_A_BUDGET.search(fh.read()), (
            "the enforcement pattern no longer matches churn.py, which does "
            "enforce its budget; fix the pattern, not the conclusion")
    with open(os.path.join(VB_ROOT, "harness", "workloads", "filtered.py")) as fh:
        source = fh.read()
    return bool(re.search(r"\bbudget", source)
                and _ENFORCES_A_BUDGET.search(source))


def _iterative_scan_gate_name() -> str:
    """The driver attribute `harness/workloads/filtered.py` gates on.

    Read from the gate rather than written here, so that a rename has to happen
    on both sides of it: the workload records `iterative_scan: None` and says
    nothing when the driver lacks the method.
    """
    path = os.path.join(VB_ROOT, "harness", "workloads", "filtered.py")
    with open(path) as fh:
        gate = re.search(r'hasattr\(driver,\s*"([A-Za-z_][A-Za-z0-9_]*)"\)',
                         fh.read())
    assert gate, "filtered.py no longer gates on hasattr(driver, ...)"
    return gate.group(1)


def _one_hour_per_pass_term(plan):
    """A probe term worth one hour per (engine, dataset, pass).

    `main` with one pass is six engines x two datasets, so it adds exactly
    12 h — a number the extension tests can state rather than derive.
    """
    return est_mod._QueryTerm(
        name="test term", typical_s=3600.0, worst_s=3600.0,
        basis="one hour, flat", bounded=True, scope="pass")


# ---------------------------------------------------------------------------
# 1. The ingest estimate, pinned as it was before query time was added
# ---------------------------------------------------------------------------

class TestTheIngestEstimateDidNotMove:
    """These were the numbers on origin/master. The task that added query time
    was explicitly not allowed to change them, and the published docs quote
    them, so they are pinned rather than recomputed."""

    def test_main_one_pass(self):
        e = est("main")
        assert e["total_hours"] == 69.77
        assert e["per_engine_hours"] == {
            "mariadb": 13.15, "alisql": 37.24, "pgvector": 0.68,
            "mariadb123": 18.62, "mongodb": 0.03, "valkey": 0.05,
        }
        assert (e["m_values"], e["ops_m_values"], e["passes"]) == (2, 1, 1)
        assert e["long_run"] is True

    def test_main_both_passes_is_exactly_double(self):
        assert est("main", passes=("normalized", "tuned"))["total_hours"] == 139.57

    def test_main_ops_only(self):
        """Dropping the recall phase drops the ann M grid from the estimate."""
        assert est("main", phases=["ops"])["total_hours"] == 23.26

    def test_smoke(self):
        assert est("smoke")["total_hours"] == 0.49

    def test_full_two_passes(self):
        assert est("full", passes=("normalized", "tuned"))["total_hours"] == 726.45

    def test_a_dataset_with_no_measured_rate_is_still_named_not_guessed(self):
        e = est("dev")
        assert e["datasets_without_estimate"] == ["tiny-16-euclidean"]
        assert e["total_hours"] == 0.0

    def test_ingest_hours_is_the_same_number_under_its_own_name(self):
        """`total_hours` keeps its published meaning; `ingest_hours` is the
        unambiguous name for the same figure, for code written from now on."""
        e = est("main")
        assert e["ingest_hours"] == e["total_hours"] == 69.77


# ---------------------------------------------------------------------------
# 2. total_hours still means ingest
# ---------------------------------------------------------------------------

class TestTotalHoursStillMeansIngest:
    """webui/static/setup.js:237 and webui/static/control.js:41 both print this
    field as `estimated ingest ~N h`. setup.js is not owned by this change, so
    a redefinition here would have made a file nobody touched start lying."""

    def test_total_hours_excludes_query_time(self):
        e = est("main")
        assert e["query_hours"] > 0
        assert e["total_hours"] == e["ingest_hours"]
        assert e["worst_case_total_hours"] > e["total_hours"]

    def test_a_new_query_term_does_not_move_total_hours(self, monkeypatch):
        """The guard that matters for the four tasks that will add terms: an
        added term must reach the worst case and must not reach the ingest
        figure the UI labels 'ingest'."""
        before = est("main")
        monkeypatch.setattr(est_mod, "_QUERY_TERMS",
                            tuple(est_mod._QUERY_TERMS) + (_one_hour_per_pass_term,))
        after = est("main")
        assert after["total_hours"] == before["total_hours"]
        assert after["per_engine_hours"] == before["per_engine_hours"]
        assert after["worst_case_total_hours"] == round(
            before["worst_case_total_hours"] + 12.0, 2)

    def test_long_run_is_still_the_ingest_warning(self):
        """The warning text is about M values reloading the corpus, so the flag
        that triggers it stays keyed on ingest. pgvector alone is 0.68 h of
        loading and 4.88 h of worst-case run time, and the M-reload warning
        would be the wrong thing to print about it."""
        e = est("main", engines=["pgvector"])
        assert e["total_hours"] == 0.68
        assert e["worst_case_total_hours"] == 5.28
        assert e["long_run"] is False


# ---------------------------------------------------------------------------
# 3. Query terms, pinned against hand arithmetic
# ---------------------------------------------------------------------------

class TestTheConcurrencyTermIsTimeNotQueries:
    """`main` runs client_counts [1, 2, 4, 8, 16, 32] for 30 s each, three
    times per point. Each point runs for a fixed span whatever the engine
    manages, so this term is exact.

    By hand, for `main`, one pass, six engines, two datasets:
        6 points x 30 s x 3 repeats               =   540 s per ops unit
        ops units per (engine, dataset): 1 each,
            except pgvector, which runs both
            pgvector_build_modes               -> 7 units per dataset
        7 units x 2 datasets x 1 pass             =    14 unit-instances
        14 x 540 s = 7,560 s                      =  2.10 h
    """

    def test_main(self):
        t = term(est("main"), "concurrency windows")
        assert t["typical_hours"] == 2.10
        assert t["worst_hours"] == 2.10
        assert t["bounded"] is True

    def test_the_client_count_list_is_counted_not_summed(self):
        """[1, 2, 4, 8, 16, 32] is six windows, not 63 of anything."""
        profile = {"name": "t", "ops": {"enabled": True,
                                        "workloads": ["concurrency"],
                                        "m_values": [16],
                                        "client_counts": [1, 2, 4, 8, 16, 32],
                                        "concurrency_duration_s": 30,
                                        "max_queries": 500}}
        e = estimate_load_hours(profile, ["mariadb"], ["glove-100-angular"],
                                ["normalized"], ["ops"])
        # 6 x 30 s = 180 s = 0.05 h
        assert term(e, "concurrency windows")["typical_hours"] == 0.05

    def test_pgvector_runs_every_build_mode_so_it_costs_twice(self):
        base = {"enabled": True, "workloads": ["concurrency"], "m_values": [16],
                "client_counts": [1, 2, 4, 8, 16, 32],
                "concurrency_duration_s": 30, "max_queries": 500}
        two = {"name": "t", "ops": base,
               "ann": {"pgvector_build_modes": ["post", "incremental"]}}
        one = {"name": "t", "ops": base, "ann": {"pgvector_build_modes": ["post"]}}
        args = (["pgvector"], ["glove-100-angular"], ["normalized"], ["ops"])
        assert term(estimate_load_hours(two, *args),
                    "concurrency windows")["typical_hours"] == 0.10
        assert term(estimate_load_hours(one, *args),
                    "concurrency windows")["typical_hours"] == 0.05

    def test_more_ops_m_values_cost_proportionally(self):
        base = {"enabled": True, "workloads": ["concurrency"],
                "client_counts": [1, 2, 4, 8, 16, 32],
                "concurrency_duration_s": 30, "max_queries": 500}
        three = {"name": "t", "ops": {**base, "m_values": [8, 16, 32]}}
        e = estimate_load_hours(three, ["mariadb"], ["glove-100-angular"],
                                ["normalized"], ["ops"])
        assert term(e, "concurrency windows")["typical_hours"] == 0.15

    def test_repeats_multiply_because_the_runner_already_honours_them(self):
        """ops_pass.harness_args passes --concurrency-repeats today, defaulting
        to 1. A profile that sets it gets that many windows per point, so the
        estimate has to count them or it under-reports a cost the run pays."""
        base = {"enabled": True, "workloads": ["concurrency"], "m_values": [16],
                "client_counts": [1, 2, 4, 8, 16, 32],
                "concurrency_duration_s": 30, "max_queries": 500}
        args = (["mariadb"], ["glove-100-angular"], ["normalized"], ["ops"])
        plain = estimate_load_hours({"name": "t", "ops": base}, *args)
        thrice = estimate_load_hours(
            {"name": "t", "ops": {**base, "concurrency_repeats": 3}}, *args)
        assert term(plain, "concurrency windows")["typical_hours"] == 0.05
        assert term(thrice, "concurrency windows")["typical_hours"] == 0.15

    def test_a_point_with_more_clients_than_queries_is_skipped(self):
        """The workload skips it, so the estimate must not charge for it."""
        base = {"enabled": True, "workloads": ["concurrency"], "m_values": [16],
                "client_counts": [1, 2, 4, 8, 16, 32],
                "concurrency_duration_s": 30, "max_queries": 4}
        e = estimate_load_hours({"name": "t", "ops": base}, ["mariadb"],
                                ["glove-100-angular"], ["normalized"], ["ops"])
        # Only 1, 2 and 4 clients can run against four queries: 3 x 30 s.
        assert term(e, "concurrency windows")["typical_hours"] == 0.03


class TestTheChurnFractionsAreCumulative:
    """`churn_fractions: [0.10, 0.25]` re-inserts 10% and then a further 15% —
    a cumulative 25%, not 35%. The workload says so and the estimate must agree.

    By hand, mariadb on glove-100-angular (1,183,514 rows, measured 147 rows/s),
    one ops unit, one pass:
        step 1:  10% of 1,183,514 = 118,351 rows / 147 =   805.11 s
        step 2:  15% of 1,183,514 = 177,527 rows / 147 = 1,207.67 s
        total                                          = 2,012.78 s = 0.56 h
    and the worst case is the budget, twice: 2 x 1,800 s = 3,600 s = 1.00 h.
    """

    PROFILE = {"name": "t", "ops": {"enabled": True, "workloads": ["churn"],
                                    "m_values": [16],
                                    "churn_fractions": [0.10, 0.25]}}

    def test_typical_is_the_incremental_rows_at_the_measured_rate(self):
        e = estimate_load_hours(self.PROFILE, ["mariadb"],
                                ["glove-100-angular"], ["normalized"], ["ops"])
        assert term(e, "churn re-insert")["typical_hours"] == 0.56

    def test_worst_case_is_the_budget_per_step(self):
        e = estimate_load_hours(self.PROFILE, ["mariadb"],
                                ["glove-100-angular"], ["normalized"], ["ops"])
        assert term(e, "churn re-insert")["worst_hours"] == 1.00

    def test_a_single_fraction_is_one_step(self):
        """10% alone: 118,351 rows / 147 = 805.11 s = 0.22 h, budget once."""
        profile = {"name": "t", "ops": {**self.PROFILE["ops"],
                                        "churn_fractions": [0.10]}}
        t = term(estimate_load_hours(profile, ["mariadb"],
                                     ["glove-100-angular"], ["normalized"],
                                     ["ops"]), "churn re-insert")
        assert (t["typical_hours"], t["worst_hours"]) == (0.22, 0.50)

    def test_the_subset_a_profile_loads_is_what_gets_churned(self):
        """A profile that loads 20,000 rows of fashion-mnist churns 10% of
        those 2,000 rows at the measured 745 rows/s — 2.7 s, not 10% of
        60,000 rows."""
        profile = {"name": "t", "ops": {**self.PROFILE["ops"],
                                        "churn_fractions": [0.10],
                                        "subset_rows": 20000}}
        e = estimate_load_hours(profile, ["mariadb"],
                                ["fashion-mnist-784-euclidean"],
                                ["normalized"], ["ops"])
        t = term(e, "churn re-insert")
        assert t["typical_seconds"] == pytest.approx(2000 / 745.0, abs=0.01)
        assert t["worst_hours"] == 0.50


class TestTheFilteredTermChargesOneModeAndSaysWhatItCannotKnow:
    """Filtered search runs `selectivities x max_queries` queries per ops unit,
    over the one iterative-scan mode the run pins. The shape is exact; the
    duration is not, because no QPS has been measured and nothing caps the
    loop. Both facts are carried by the term so the output does not charge for
    the cost and disclaim it at the same time.

    `main`: 2 selectivities x 500 queries = 1,000 queries per ops unit, over
    the one iterative-scan mode the run pins.
    """

    PROFILE = {"name": "t", "ops": {"enabled": True, "workloads": ["filtered"],
                                    "m_values": [16], "max_queries": 500,
                                    "selectivities": [0.10, 0.50]}}

    def _term(self, profile=None, engine="mariadb"):
        return term(estimate_load_hours(profile or self.PROFILE, [engine],
                                        ["glove-100-angular"], ["normalized"],
                                        ["ops"]), "filtered search")

    def test_the_shape_is_stated_even_though_the_duration_is_not(self):
        t = self._term()
        assert "2 selectivities x 500 queries = 1,000 queries" in t["basis"]
        assert "one iterative-scan mode" in t["basis"]
        assert "1,000 queries per ops unit" in t["basis"]

    def test_an_unmeasurable_cost_is_flagged_not_scored_zero(self):
        t = self._term()
        assert t["quantified"] is False
        assert t["bounded"] is False
        assert (t["typical_hours"], t["worst_hours"]) == (0.0, 0.0)

    def test_it_makes_the_run_worst_case_stop_claiming_to_be_a_ceiling(self):
        e = estimate_load_hours(self.PROFILE, ["mariadb"],
                                ["glove-100-angular"], ["normalized"], ["ops"])
        assert e["query_worst_case_is_a_ceiling"] is False

    def test_a_budget_key_alone_buys_no_ceiling(self):
        """`ops.filtered_budget_s` exists nowhere but in a profile a reader
        might write: not in ops_pass.harness_args, not in harness/main.py, not
        in filtered.py. Honouring it here would print a concrete ceiling, flip
        query_worst_case_is_a_ceiling and drop the caveat while nothing capped
        anything — a false ceiling, which is the one thing the estimate must
        never produce. The cross-check below is what tells T3c to turn this on
        once the run enforces it."""
        profile = {"name": "t", "ops": {**self.PROFILE["ops"],
                                        "filtered_budget_s": 300}}
        t = self._term(profile)
        assert (t["typical_hours"], t["worst_hours"]) == (0.0, 0.0)
        assert t["bounded"] is False
        assert t["quantified"] is False
        assert "300" not in t["basis"]

    def test_the_estimate_and_the_runner_honour_the_budget_together(self):
        """Whichever part of T3c lands first, this fails until the rest does.

        The chain has three links, like `churn_budget_s` before it:
        `ops_pass.harness_args` emits the flag (ops_pass.py:432 for churn),
        `harness/main.py` passes it to the workload (main.py:242), and the
        workload stops when it runs out (churn.py:244). Only the first and the
        third are checked, and the third has to be an elapsed-time comparison
        rather than a reference to the word.

        Two states have already passed a weaker version of this check while
        nothing capped the loop. Emission alone, with the flag never plumbed.
        And then the realistic one: the flag emitted, parsed, and accepted as
        `budget_s: float = 1800.0` by the workload, which never looks at it --
        every link in place and no bound anywhere. The second is how a feature
        like this is actually built, one layer at a time, which is why the test
        has to care about the layer that stops rather than the layers that
        carry.

        Today none of the three exists, which is consistent, so this passes."""
        from orchestrator import ops_pass
        from orchestrator.config import load_resources
        profile = {"name": "t", "k": 10,
                   "ops": {**self.PROFILE["ops"], "filtered_budget_s": 300}}
        estimate_honours = self._term(profile)["bounded"]
        argv = ops_pass.harness_args(profile, 16, "pgvector", None,
                                     "normalized", load_resources("normalized"))
        runner_emits = any(str(a).startswith("--filtered-budget") for a in argv)

        # The workload has to be the thing that stops.
        enforced = _filtered_budget_is_enforced()
        wired = runner_emits and enforced
        assert estimate_honours == wired, (
            "ops.filtered_budget_s is honoured by some of the run and not all "
            f"of it (estimate={estimate_honours}, harness_args={runner_emits}, "
            f"workload={enforced}); the estimate must not claim a ceiling "
            "nothing enforces, nor ignore one that is enforced")

    def test_a_measured_rate_turns_it_into_a_duration(self, monkeypatch):
        """The other half: 1,000 queries at 0.9 QPS = 1,111.1 s = 0.31 h — the
        measured MariaDB filtered rate, which docs/07 records as 37 minutes for
        2,000 queries at 1% selectivity."""
        monkeypatch.setitem(est_mod._MEASURED_QUERY_QPS,
                            ("mariadb", "glove-100-angular"), 0.9)
        t = self._term()
        assert t["typical_hours"] == 0.31
        assert t["quantified"] is True
        assert t["bounded"] is False      # a rate is not a cap

    def test_more_selectivities_cost_proportionally(self, monkeypatch):
        """3 x 500 = 1,500 queries at 0.9 QPS = 1,666.7 s = 0.46 h."""
        monkeypatch.setitem(est_mod._MEASURED_QUERY_QPS,
                            ("mariadb", "glove-100-angular"), 0.9)
        profile = {"name": "t", "ops": {**self.PROFILE["ops"],
                                        "selectivities": [0.01, 0.10, 0.50]}}
        assert self._term(profile)["typical_hours"] == 0.46

    def test_it_scales_with_ops_units_like_everything_else(self, monkeypatch):
        """pgvector runs both build modes: 2 units x 1,000 queries at 2 QPS =
        1,000 s = 0.28 h."""
        monkeypatch.setitem(est_mod._MEASURED_QUERY_QPS,
                            ("pgvector", "glove-100-angular"), 2.0)
        two = {"name": "t",
               "ann": {"pgvector_build_modes": ["post", "incremental"]},
               "ops": self.PROFILE["ops"]}
        one = {"name": "t", "ann": {"pgvector_build_modes": ["post"]},
               "ops": self.PROFILE["ops"]}
        assert self._term(two, engine="pgvector")["typical_hours"] == 0.28
        assert self._term(one, engine="pgvector")["typical_hours"] == 0.14

    def test_a_profile_without_the_workload_pays_nothing(self):
        profile = {"name": "t", "ops": {**self.PROFILE["ops"],
                                        "workloads": ["build"]}}
        e = estimate_load_hours(profile, ["mariadb"], ["glove-100-angular"],
                                ["normalized"], ["ops"])
        assert [t["name"] for t in e["query_terms"]] == []

    def test_a_profile_without_selectivities_still_runs_one(self, monkeypatch):
        """ops_pass.harness_args defaults --selectivities to 0.1, so such a
        profile pays one point. Charging nothing dropped that cost out of the
        terms list and out of the disclosure list both — a third state, neither
        counted nor named."""
        monkeypatch.setitem(est_mod._MEASURED_QUERY_QPS,
                            ("mariadb", "glove-100-angular"), 0.9)
        ops = {k: v for k, v in self.PROFILE["ops"].items()
               if k != "selectivities"}
        t = self._term({"name": "t", "ops": ops})
        # 1 x 500 queries at 0.9 QPS = 555.6 s = 0.15 h
        assert t["typical_hours"] == 0.15
        assert "1 selectivity x 500 queries = 500 queries" in t["basis"]

    def test_the_default_matches_the_one_the_runner_uses(self):
        """Pinned against ops_pass rather than restated, because the defect was
        the two diverging."""
        from orchestrator import ops_pass
        from orchestrator.config import load_resources
        ops = {k: v for k, v in self.PROFILE["ops"].items()
               if k != "selectivities"}
        argv = ops_pass.harness_args({"name": "t", "k": 10, "ops": ops}, 16,
                                     "mariadb", None, "normalized",
                                     load_resources("normalized"))
        emitted = argv[argv.index("--selectivities") + 1].split(",")
        assert len(emitted) == 1
        assert f"{len(emitted)} selectivity" in self._term(
            {"name": "t", "ops": ops})["basis"]


class TestTheIterativeScanModeNowReachesTheRun:
    """`ops.pgvector_iterative_scan` was declared by `main`, `full` and the
    tuned extras, the plumbing to `SET hnsw.iterative_scan` was complete, and
    the report rendered a column for it — but `_run_unit` never passed it, so
    every archived pgvector filtered record carries `iterative_scan: null`.

    There was a second fault behind the first: YAML 1.1 reads a bare `off` as
    boolean False, so the declared list parses as `[False, "relaxed_order"]`.
    Passing the value through naively would have emitted nothing at all
    (`if iterative_scan:` is false for False) or `SET hnsw.iterative_scan =
    False`, which is not a mode pgvector has.
    """

    def _resources(self, name):
        from orchestrator.config import load_resources
        return load_resources(name)

    def test_the_profile_key_really_does_parse_as_a_boolean(self):
        """The premise, pinned: if this ever changes the normaliser is dead
        weight rather than load-bearing, and a reader should be told."""
        assert load_profile("main")["ops"]["pgvector_iterative_scan"][0] is False

    def test_the_first_declared_mode_is_resolved_to_its_yaml_name(self):
        mode = cli_mod._ops_iterative_scan(
            "pgvector", load_profile("main"),
            self._resources("normalized"), "normalized")
        assert mode == "off"

    def test_the_tuned_pass_reads_its_own_extras(self):
        """The two sources have to disagree for this to mean anything:
        `main.yml:64` and `tuned.yml:68` both declare `[off, relaxed_order]`,
        so a fixture built from the real files returns "off" whether the extras
        branch is read or falls through to the profile. This also pins the
        precedence, which nothing else did."""
        profile = {"ops": {"pgvector_iterative_scan": ["off"]}}
        extras = {"extras": {"pgvector_iterative_scan": ["relaxed_order"]}}
        assert cli_mod._ops_iterative_scan(
            "pgvector", profile, extras, "tuned") == "relaxed_order"
        # and the extras are a tuned-pass thing only
        assert cli_mod._ops_iterative_scan(
            "pgvector", profile, extras, "normalized") == "off"

    def test_the_real_tuned_pass_resolves(self):
        """Both real sources declare `off` first, so the pass makes no
        difference today — pinned so that stops being invisible if one of them
        changes."""
        for pass_name in ("normalized", "tuned"):
            assert cli_mod._ops_iterative_scan(
                "pgvector", load_profile("main"),
                self._resources(pass_name), pass_name) == "off"

    def test_nothing_is_pinned_for_an_engine_without_the_knob(self):
        assert cli_mod._ops_iterative_scan(
            "mariadb", load_profile("main"),
            self._resources("tuned"), "tuned") is None

    def test_a_profile_that_declares_nothing_still_passes_nothing(self):
        """So the engine keeps its own default and the record keeps saying so,
        rather than this inventing a mode on the profile's behalf."""
        assert cli_mod._ops_iterative_scan(
            "pgvector", {"ops": {}}, {}, "normalized") is None

    def test_an_unknown_mode_is_refused_rather_than_interpolated(self):
        """The driver builds `SET hnsw.iterative_scan = {mode}` by string
        interpolation, so an unrecognised value would reach the server."""
        profile = {"ops": {"pgvector_iterative_scan": ["sideways"]}}
        assert cli_mod._ops_iterative_scan(
            "pgvector", profile, {}, "normalized") is None

    def test_the_run_unit_passes_the_mode_to_the_harness(self, tmp_path):
        """The guard the rest of this class does not give: every other test
        here wires `iterative_scan=` itself, which proves the resolver works
        and that `harness_args` consumes it, and proves nothing about the one
        line in between. A call site that forgets the argument is the exact
        defect that produced 58 ops files and 69 null records, and `_run_unit`
        had no test anywhere in the repo.

        No container and no network: `harness_args` is captured and `OpsRun` is
        a fake whose `run_harness` returns 0."""
        import argparse
        from orchestrator import ops_pass
        from orchestrator.config import load_resources

        captured = []

        def capture(*a, **kw):
            captured.append(kw)
            return ["--fake"]

        class FakeRun:
            def __init__(self, *a, **kw):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def run_harness(self, *a, **kw):
                return 0

        paths = {"run_dir": str(tmp_path),
                 "checkpoints": str(tmp_path / "checkpoints")}
        with mock.patch.object(ops_pass, "harness_args", capture), \
             mock.patch.object(ops_pass, "OpsRun", FakeRun):
            for engine, expected in (("pgvector", "off"), ("mariadb", None),
                                     ("alisql", None)):
                captured.clear()
                rc = cli_mod._run_unit(
                    "ops", engine, "glove-100-angular", load_profile("main"),
                    {}, load_resources("normalized"), None, "normalized",
                    paths, "run-0", argparse.Namespace(force=False),
                    checkpoints=None)
                assert rc == 0
                assert captured, f"{engine}: harness_args was never called"
                for kwargs in captured:
                    assert kwargs.get("iterative_scan") == expected, engine

    def test_the_harness_passes_it_on_to_the_workload(self):
        """The other unguarded link in the same chain, asserted at source
        level: `harness/main.py` hands `args.iterative_scan` to
        `filtered.run`. Read rather than executed because running it needs a
        server; this suite already reads source this way to check that the web
        UI's field names exist."""
        path = os.path.join(VB_ROOT, "harness", "main.py")
        with open(path) as fh:
            source = fh.read()
        call = re.search(r"filtered_workload\.run\((?P<args>.*?)\n\s*\)",
                         source, re.S)
        assert call, "filtered_workload.run( is no longer where this expects"
        assert "iterative_scan=args.iterative_scan" in call.group("args")

    def test_the_harness_command_line_carries_it(self):
        """`harness_args` ignores its `resolved` argument, so None keeps this
        test off the hardware-detection path."""
        from orchestrator import ops_pass
        resources = self._resources("normalized")
        profile = load_profile("main")
        argv = ops_pass.harness_args(
            profile, 16, "pgvector", None, "normalized", resources,
            iterative_scan=cli_mod._ops_iterative_scan(
                "pgvector", profile, resources, "normalized"),
        )
        assert "--iterative-scan" in argv
        assert argv[argv.index("--iterative-scan") + 1] == "off"

    def test_the_old_call_site_emitted_nothing(self):
        """What the run has always done: no mode argument at all, which is how
        69 filtered records came to say null."""
        from orchestrator import ops_pass
        argv = ops_pass.harness_args(
            load_profile("main"), 16, "pgvector", None, "normalized",
            self._resources("normalized"))
        assert "--iterative-scan" not in argv

    def test_the_driver_implements_the_method_the_workload_gates_on(self):
        """The last link, and the one the test below cannot prove because it
        attaches the method itself. `filtered.py:56` gates on
        `hasattr(driver, ...)` and records None when it is absent, so renaming
        the method on `PostgresDriver` puts every pgvector filtered record back
        to null with the whole suite green. Nothing asserted it existed: the
        only references outside the driver are that gate and the fake below.

        The name is read out of the gate rather than written here, so a rename
        has to happen on both sides or this fails."""
        from harness.drivers.postgres import PostgresDriver
        name = _iterative_scan_gate_name()
        assert callable(getattr(PostgresDriver, name, None)), (
            f"filtered.py asks for driver.{name}() and PostgresDriver does not "
            f"implement it, so the mode is silently dropped")

    def test_a_filtered_record_now_carries_the_mode(self, tmp_path):
        """End of the chain, against the fake driver rather than a server: the
        record's `extra["iterative_scan"]` was None for every run ever made."""
        from harness.drivers.postgres import PostgresDriver
        from harness.metrics import records
        from harness.workloads import filtered as filtered_workload
        from harness.workloads.context import RunContext
        from tests.fake_driver import index_spec, loaded_driver, tiny_dataset

        dataset = tiny_dataset()
        driver = loaded_driver(dataset, ef_half=0.0, ef_search=64)
        # Attached under the name the workload actually gates on, and only if
        # the real driver implements it, so this fake cannot stand in for a
        # method PostgresDriver does not have.
        applied = []
        name = _iterative_scan_gate_name()
        assert hasattr(PostgresDriver, name)
        setattr(driver, name, applied.append)

        recorder = records.Recorder(str(tmp_path / "ops-fake.jsonl"))
        try:
            ctx = RunContext(
                run_id="t0-fake", resource_pass="normalized", recorder=recorder,
                engine="fake", engine_tag="none", march="x86-64-v3", k=5,
                cache_dir=str(tmp_path / "cache"))
            filtered_workload.run(
                ctx, driver, dataset, index_spec(dataset), ef_search=64,
                selectivities=(0.50,), max_queries=8, warmup=1,
                indexed_rows=len(dataset.train),
                iterative_scan=cli_mod._ops_iterative_scan(
                    "pgvector", load_profile("main"), {}, "normalized"),
            )
        finally:
            recorder.close()

        written = list(records.read_records(recorder.path))
        assert applied == ["off"]
        assert [r["extra"]["iterative_scan"] for r in written] == ["off"]


class TestACapIsPrintedAsACapNotADuration:
    """A budget is a ceiling, not a duration, and the gap is not academic:
    Valkey loaded this corpus at 30,195 rows/s and then wrote the same rows
    back at 86, because pipelined writes into a populated index do not return.
    Modelling churn from the load rate alone would have under-reported by the
    whole budget, so both figures are carried."""

    def test_alisql_on_glove_is_capped_at_the_budget(self):
        """50.1 rows/s modelled: step 1 is 118,351 rows = 2,361 s and step 2 is
        177,527 rows = 3,541 s. Both exceed the 1,800 s budget, so typical
        collapses onto the ceiling: 2 x 1,800 s = 1.00 h."""
        profile = {"name": "t", "ops": {"enabled": True, "workloads": ["churn"],
                                        "m_values": [16],
                                        "churn_fractions": [0.10, 0.25]}}
        t = term(estimate_load_hours(profile, ["alisql"], ["glove-100-angular"],
                                     ["normalized"], ["ops"]), "churn re-insert")
        assert t["typical_hours"] == 1.00
        assert t["worst_hours"] == 1.00

    def test_a_profile_may_lower_the_budget(self):
        """ops.churn_budget_s is honoured by ops_pass.harness_args today."""
        profile = {"name": "t", "ops": {"enabled": True, "workloads": ["churn"],
                                        "m_values": [16], "churn_budget_s": 300,
                                        "churn_fractions": [0.10, 0.25]}}
        t = term(estimate_load_hours(profile, ["alisql"], ["glove-100-angular"],
                                     ["normalized"], ["ops"]), "churn re-insert")
        # 2 x 300 s = 600 s = 0.17 h, and both steps still exceed it.
        assert (t["typical_hours"], t["worst_hours"]) == (0.17, 0.17)

    def test_typical_never_exceeds_the_worst_case(self):
        for name in ("main", "full", "smoke", "quick", "dev"):
            e = est(name)
            assert e["query_hours"] <= e["query_hours_worst"], name
            assert e["typical_total_hours"] <= e["worst_case_total_hours"], name
            for t in e["query_terms"]:
                assert t["typical_hours"] <= t["worst_hours"], (name, t["name"])


class TestTheWholeQueryEstimateForMain:
    """The aggregate, so a reader can see the number the convention now refers
    to and check it against the parts: 2.10 h of concurrency windows and 4.77 h
    of modelled churn re-insert, against a 14.00 h churn ceiling."""

    def test_the_parts_and_the_totals(self):
        e = est("main")
        assert [t["name"] for t in e["query_terms"]] == [
            "concurrency windows", "filtered search", "churn re-insert"]
        assert term(e, "churn re-insert")["typical_hours"] == 4.77
        assert term(e, "churn re-insert")["worst_hours"] == 14.00
        assert e["query_hours"] == 6.87
        assert e["query_hours_worst"] == 16.10
        assert e["typical_total_hours"] == 76.64
        assert e["worst_case_total_hours"] == 85.87

    def test_the_totals_are_the_sums_a_reader_can_do_on_the_printed_lines(self):
        e = est("main")
        assert e["typical_total_hours"] == round(
            e["ingest_hours"] + e["query_hours"], 2)
        assert e["worst_case_total_hours"] == round(
            e["ingest_hours"] + e["query_hours_worst"], 2)

    def test_the_per_engine_split_mirrors_the_ingest_one(self):
        e = est("main")
        assert e["query_per_engine_hours"] == {
            "mariadb": 1.40, "alisql": 2.30, "pgvector": 0.71,
            "mariadb123": 1.85, "mongodb": 0.30, "valkey": 0.30,
        }

    def test_ops_disabled_means_no_ops_terms(self):
        e = est("m-sweep")
        assert e["query_terms"] == []
        assert e["query_hours"] == 0.0


# ---------------------------------------------------------------------------
# 4. What it cannot know, reported rather than guessed
# ---------------------------------------------------------------------------

class TestAMissingQueryRateIsReportedNotInvented:
    """The ingest path names datasets it has no rate for instead of scoring
    them zero. Query time needs the same honesty about engines: the per-query
    workloads (filtered, the churn re-measurements, the warm-ups, the whole
    recall sweep) cost a number of queries, and nothing in this repo records a
    per-(engine, dataset) QPS to multiply them by."""

    def test_every_engine_is_named_while_the_table_is_empty(self):
        assert est("main")["engines_without_query_rate"] == ALL

    def test_a_measured_rate_closes_the_gap_for_that_engine(self, monkeypatch):
        """Proves the list is derived from the table rather than hardcoded."""
        monkeypatch.setitem(est_mod._MEASURED_QUERY_QPS,
                            ("mariadb", "glove-100-angular"), 326.0)
        monkeypatch.setitem(est_mod._MEASURED_QUERY_QPS,
                            ("mariadb", "sift-128-euclidean"), 326.0)
        assert "mariadb" not in est("main")["engines_without_query_rate"]
        assert "alisql" in est("main")["engines_without_query_rate"]

    def test_the_unmodelled_costs_are_named_one_by_one(self):
        names = est("main")["query_costs_not_modelled"]
        assert any("ef_search" in n for n in names)          # the recall sweep
        assert any("churn" in n for n in names)
        assert any("warm-up" in n for n in names)

    def test_filtered_is_charged_as_a_term_and_so_is_not_also_disclaimed(self):
        """It may appear in one list or the other, never both: output that
        charges for a cost and calls it unmodelled in the same breath tells a
        reader nothing."""
        e = est("main")
        assert "filtered search" in [t["name"] for t in e["query_terms"]]
        assert not any("filtered" in n for n in e["query_costs_not_modelled"])

    def test_an_ann_only_run_still_declares_its_query_cost(self):
        e = est("m-sweep")
        assert e["query_terms"] == []
        assert any("ef_search" in n for n in e["query_costs_not_modelled"])
        assert e["engines_without_query_rate"] == ALL

    def test_nothing_is_claimed_when_nothing_queries(self):
        profile = {"name": "t", "ann": {"enabled": False},
                   "ops": {"enabled": False}}
        e = estimate_load_hours(profile, ["mariadb"], ["glove-100-angular"],
                                ["normalized"], ["ann", "ops"])
        assert e["query_costs_not_modelled"] == []
        assert e["engines_without_query_rate"] == []


# ---------------------------------------------------------------------------
# 5. The extension point the four later tasks will use
# ---------------------------------------------------------------------------

class TestTheExtensionPoint:
    """T1d (concurrency repeats), T2a (the ef-calibration sweep), T3c (the
    filtered budget) and T4a (churn maintenance) each owe this function one
    named term. Adding one is: write a function taking a `_QueryPlan` and
    returning a `_QueryTerm` or None, and append it to `_QUERY_TERMS`. Nothing
    else may need to change, and this test is what says so."""

    def test_an_added_term_reaches_every_total_and_the_print(self, monkeypatch):
        before = est("main")
        monkeypatch.setattr(est_mod, "_QUERY_TERMS",
                            tuple(est_mod._QUERY_TERMS) + (_one_hour_per_pass_term,))
        after = est("main")
        assert [t["name"] for t in after["query_terms"]][-1] == "test term"
        assert after["query_hours"] == round(before["query_hours"] + 12.0, 2)
        assert after["query_hours_worst"] == round(
            before["query_hours_worst"] + 12.0, 2)
        assert after["worst_case_total_hours"] == round(
            before["worst_case_total_hours"] + 12.0, 2)
        out = printed("main")
        assert "test term" in out
        assert "one hour, flat" in out

    def test_a_term_may_decline_this_profile(self, monkeypatch):
        monkeypatch.setattr(est_mod, "_QUERY_TERMS",
                            tuple(est_mod._QUERY_TERMS) + (lambda plan: None,))
        assert est("main")["query_hours"] == 6.87

    def test_scope_unit_is_charged_per_ops_unit_and_pass(self, monkeypatch):
        """main has 7 ops units per dataset x 2 datasets = 14 per pass."""
        def one_second_per_unit(plan):
            return est_mod._QueryTerm(
                name="unit term", typical_s=1.0, worst_s=1.0,
                basis="one second", bounded=True, scope="unit")
        monkeypatch.setattr(est_mod, "_QUERY_TERMS", (one_second_per_unit,))
        one = est("main")
        two = est("main", passes=("normalized", "tuned"))
        assert term(one, "unit term")["typical_seconds"] == 14.0
        assert term(two, "unit term")["typical_seconds"] == 28.0

    def test_scope_pass_skips_the_ops_unit_multiplier(self, monkeypatch):
        """12, not 14: six engines x two datasets, where unit scope would have
        charged pgvector twice for its two build modes."""
        def one_second_per_pass(plan):
            return est_mod._QueryTerm(
                name="pass term", typical_s=1.0, worst_s=1.0,
                basis="one second", bounded=True, scope="pass")
        monkeypatch.setattr(est_mod, "_QUERY_TERMS", (one_second_per_pass,))
        assert term(est("main"), "pass term")["typical_seconds"] == 12.0
        assert term(est("main", passes=("normalized", "tuned")),
                    "pass term")["typical_seconds"] == 24.0

    def test_an_unknown_scope_is_refused_loudly(self, monkeypatch):
        """Silently charging it once per (engine, dataset) is how a term meant
        to be counted once became twelve."""
        def nonsense(plan):
            return est_mod._QueryTerm(
                name="bad", typical_s=1.0, worst_s=1.0, basis="-",
                bounded=True, scope="run")
        monkeypatch.setattr(est_mod, "_QUERY_TERMS", (nonsense,))
        with pytest.raises(ValueError, match="scope"):
            est("main")

    def test_a_term_declares_whether_its_worst_case_is_a_ceiling(self):
        """An unbounded term must not be read as a worst case. Both of today's
        terms are bounded by the configuration; a future one may not be."""
        def unbounded(plan):
            return est_mod._QueryTerm(
                name="open ended", typical_s=60.0, worst_s=60.0,
                basis="no cap exists", bounded=False, scope="pass")
        with mock.patch.object(est_mod, "_QUERY_TERMS", (unbounded,)):
            e = est("main")
            assert term(e, "open ended")["bounded"] is False
            assert e["query_worst_case_is_a_ceiling"] is False
            assert "open ended" in printed("main")
        # And with only the two bounded terms, it is a ceiling again.
        with mock.patch.object(est_mod, "_QUERY_TERMS",
                               (est_mod._concurrency_windows,
                                est_mod._churn_reinsert)):
            assert est("main")["query_worst_case_is_a_ceiling"] is True

    def test_the_plan_a_term_receives_carries_what_it_needs(self):
        """Documented so an implementer does not have to read the accumulator."""
        seen = []

        def spy(plan):
            seen.append(plan)
            return None

        with mock.patch.object(est_mod, "_QUERY_TERMS", (spy,)):
            est("main", engines=["mariadb"], datasets=["glove-100-angular"])
        assert len(seen) == 1
        plan = seen[0]
        assert plan.engine == "mariadb"
        assert plan.dataset == "glove-100-angular"
        assert plan.rows == 1_183_514
        assert plan.write_rows_per_s == 147.0
        assert plan.query_qps is None
        assert plan.units == 1
        assert plan.passes == 1
        assert plan.workloads == ["build", "concurrency", "filtered", "churn"]
        assert plan.phases == ["ann", "ops"]
        assert plan.ops["max_queries"] == 500
        assert plan.k == 10


# ---------------------------------------------------------------------------
# 6. The printed output
# ---------------------------------------------------------------------------

class TestThePrintedOutput:
    def test_the_ingest_block_is_unchanged_word_for_word(self):
        out = printed("main")
        assert ("\nestimated ingest time (loading only, before any queries):\n"
                "  mariadb ~13.2 h  |  alisql ~37.2 h  |  pgvector ~0.7 h  |  "
                "mariadb123 ~18.6 h  |  mongodb ~0.0 h  |  valkey ~0.1 h\n"
                "  total ~69.8 h across 1 pass(es), 2 M value(s)\n") in out
        assert "! This is a long run." in out

    def test_query_time_is_a_separate_block_with_both_figures(self):
        out = printed("main")
        assert "estimated query time" in out
        assert "concurrency windows" in out
        assert "churn re-insert" in out
        assert "~0.7 h" in out
        assert "~4.8 h typical" in out
        assert "~14.0 h worst case" in out

    def test_an_unquantified_term_says_so_instead_of_printing_zero(self):
        """`~0.0 h` for filtered search would read as free rather than as
        unmeasured, which is the opposite of what is known about it."""
        out = printed("main")
        assert "filtered search      cost not quantified" in out
        assert "1,000 queries per ops unit" in out

    def test_the_total_is_printed(self):
        out = printed("main")
        assert "~85.9 h worst case" in out
        assert "~76.6 h typical" in out

    def test_the_bug_criterion_is_only_claimed_when_it_holds(self, monkeypatch):
        """Two adjacent lines used to say the worst case was a figure a run
        cannot exceed and then that a run can exceed it. The caveat now comes
        first and the claim is conditional on it."""
        out = printed("main")
        run_line = out.index("estimated run time")
        caveat = out.index("Not yet a ceiling")
        assert caveat > run_line
        assert "nothing caps filtered search" in out
        assert "a run longer than it is a bug" not in out
        assert "largest figure this configuration justifies" in out

        # With only bounded terms the claim is made, and the caveat is not.
        monkeypatch.setattr(est_mod, "_QUERY_TERMS",
                            (est_mod._concurrency_windows,
                             est_mod._churn_reinsert))
        bounded_out = printed("main")
        assert "a run longer than it is a bug" in bounded_out
        assert "Not yet a ceiling" not in bounded_out

    def test_the_printed_total_is_the_dict_total(self):
        """The two cannot drift: the printed figure is the dict's own."""
        e = est("main")
        assert f"~{e['worst_case_total_hours']:.1f} h worst case" in printed("main")

    def test_what_it_cannot_know_is_printed_too(self):
        """Asserted against the whole engine list in the disclosure sentence,
        not engine names loose in the output: the ingest block above prints
        every engine, so `"mariadb" in out` passed even with the gap list
        emptied — and `"mariadb"` is a substring of `"mariadb123"`, which has
        now cost this series four tests."""
        out = printed("main")
        assert ("there is no measured query rate for "
                f"{', '.join(ALL)} to multiply them by") in out
        assert "ef_search" in out
        assert "churn delete" in out

    def test_a_trivial_run_still_prints_nothing(self):
        """`dev` is a one-minute framework exercise; an estimate for it is
        noise. The gate is the worst-case total, not ingest alone."""
        assert printed("dev") == ""

    def test_a_profile_whose_cost_is_query_time_is_no_longer_silent(self):
        """smoke is 0.49 h of ingest — under the old 1 h gate, so it printed
        nothing at all — against a 3.5 h churn ceiling across six engines."""
        out = printed("smoke")
        assert "estimated ingest time" in out
        assert "churn re-insert" in out
        assert "~4.0 h worst case" in out


# ---------------------------------------------------------------------------
# 7. The web UI reads the same dict
# ---------------------------------------------------------------------------

class TestTheDocumentedFiguresAreTheOnesTheCodeProduces:
    """docs/07-planning-a-run.md is what a reader consults before committing
    days of machine time, and its figures had already drifted: the sample
    output block said 45.8 h for `main` while the estimator said 51.1, because
    measured rates were added to the table afterwards and nobody re-ran it.

    The planning table is therefore executable, like `verify-alisql-traps.sh`
    is for the engine notes. It is written for the three original engines, one
    or two passes, which is what the surrounding prose says.
    """

    THREE = ["mariadb", "alisql", "pgvector"]
    ROW = re.compile(
        r"^\| `(?P<profile>[a-z-]+)` \| (?P<passes>\d) \| ~(?P<ingest>[\d.]+) h"
        r" \| ~(?P<typical>[\d.]+) h \| ~(?P<worst>[\d.]+) h"
        r" \| \*\*~(?P<run>[\d.]+) h\*\* \|$", re.M)

    def _rows(self):
        path = os.path.join(VB_ROOT, "docs", "07-planning-a-run.md")
        with open(path) as fh:
            found = list(self.ROW.finditer(fh.read()))
        assert len(found) >= 6, "the planning table is gone or was reshaped"
        return found

    def test_every_documented_profile_matches_the_estimator(self):
        for row in self._rows():
            name = row.group("profile")
            passes = ["normalized", "tuned"][:int(row.group("passes"))]
            e = est(name, engines=self.THREE, passes=passes)
            for column, key in (("ingest", "ingest_hours"),
                                ("typical", "query_hours"),
                                ("worst", "query_hours_worst"),
                                ("run", "worst_case_total_hours")):
                assert float(row.group(column)) == pytest.approx(e[key], abs=0.05), (
                    f"docs/07 {name} {column}: says {row.group(column)}, "
                    f"estimator says {e[key]:.2f}")

    def test_the_sample_output_block_is_what_the_tool_prints(self):
        """The whole fenced block, not a few lines from it. Pinning only the
        numeric lines covered the 45.8-vs-51.1 drift and left sixteen lines of
        basis and caveat prose free to rot: changing a basis string left the
        suite green and docs/07 misquoting the tool."""
        path = os.path.join(VB_ROOT, "docs", "07-planning-a-run.md")
        with open(path) as fh:
            doc = fh.read()
        block = re.search(
            r"A run prints all of it before starting:\n\n```\n(?P<body>.*?)```",
            doc, re.S)
        assert block, "the sample output block moved or lost its heading"
        assert block.group("body").strip() == printed(
            "main", engines=self.THREE).strip()

    def test_claude_mds_standing_figure_is_the_one_the_command_produces(self):
        """CLAUDE.md restates the cost convention as standing guidance, and the
        sentence names a command rather than an engine list: with `--engines`
        omitted, `cli.py` resolves to all six KNOWN_ENGINES and `full.yml`
        declares no engines key. Quoting docs/07's three-engine figures there
        left an operator planning ~270 h short of the worst case."""
        path = os.path.join(VB_ROOT, "CLAUDE.md")
        with open(path) as fh:
            claude = fh.read()
        sentence = re.search(
            r"`--profile full\n--resource-pass both` is ~(?P<ingest>[\d,]+) h of "
            r"ingest[^~]*~(?P<worst>[\d,]+) h of run time\s*\nin the worst case",
            claude)
        assert sentence, "the standing cost sentence moved or was reworded"
        e = est("full", engines=ALL, passes=("normalized", "tuned"))
        assert float(sentence.group("ingest").replace(",", "")) == pytest.approx(
            e["ingest_hours"], abs=0.5)
        assert float(sentence.group("worst").replace(",", "")) == pytest.approx(
            e["worst_case_total_hours"], abs=0.5)
        assert "all six engines" in claude

    def test_the_stale_rows_stay_marked(self):
        """The first table's `quick` and `mariadb-blog` rows match no method,
        and a later editor dropping the markers would leave two misleading
        figures looking endorsed."""
        path = os.path.join(VB_ROOT, "docs", "07-planning-a-run.md")
        with open(path) as fh:
            doc = fh.read()
        quick = re.search(r"^\| `quick` \|.*$", doc, re.M).group(0)
        blog = re.search(r"^\| `mariadb-blog` \|.*$", doc, re.M).group(0)
        assert quick.rstrip().endswith("† |"), quick
        assert blog.rstrip().endswith("‡ |"), blog
        assert "14 h" not in quick, (
            "the hand figure for an M grid the profile no longer has")
        # Scoped to the footnotes themselves: whole-document membership would
        # have tolerated the figures appearing anywhere else.
        dagger = re.search(r"^† \*\*Corrected\.\*\*(?P<body>.*?)\n\n", doc,
                           re.M | re.S)
        double = re.search(r"^‡ \*\*Measured, but(?P<body>.*?)\n\n", doc,
                           re.M | re.S)
        assert dagger and double, "a footnote heading was removed or reworded"
        assert "~36 h" in dagger.group("body")
        assert "~21 h" in double.group("body")


class TestTheWebUiAndTheCliCannotDisagree:
    """The docstring's reason for returning the breakdown rather than printing
    it. /api/estimate serves this function's return value verbatim, and two
    static front-ends read fields off it by name."""

    def _api(self, body):
        from webui import control as control_mod
        status, payload = control_mod.estimate(None, None, None, body)
        assert status == 200
        return payload

    def test_the_endpoint_returns_the_function_unchanged(self):
        payload = self._api({"profile": "main", "resource_pass": "normalized"})
        assert payload == est("main")

    def test_the_old_import_path_still_resolves(self):
        """`webui/control.py:268` and `tests/test_config.py:4171` import these
        from orchestrator.cli, and the move out to orchestrator/estimate.py
        must not be visible to either."""
        from orchestrator import cli
        assert cli.estimate_load_hours is est_mod.estimate_load_hours
        assert cli.print_load_estimate is est_mod.print_load_estimate
        assert cli._INGEST_ROWS_PER_S is est_mod._INGEST_ROWS_PER_S

    @pytest.mark.parametrize("script", ["control.js", "setup.js"])
    def test_every_field_the_front_end_reads_exists(self, script):
        """A renamed or dropped key shows up here rather than as `undefined` in
        a browser. setup.js is included although this change does not own it:
        it renders total_hours as 'estimated ingest', which is exactly the
        field a redefinition would have broken."""
        path = os.path.join(VB_ROOT, "webui", "static", script)
        with open(path) as fh:
            source = fh.read()
        read = set(re.findall(r"\best\.([A-Za-z_][A-Za-z0-9_]*)", source))
        assert read, f"no est.<field> reads found in {script}"
        payload = self._api({"profile": "main", "resource_pass": "normalized"})
        missing = sorted(read - set(payload))
        assert missing == [], f"{script} reads fields the estimate does not return"

    def test_the_front_end_still_labels_total_hours_as_ingest(self):
        """If total_hours ever stops being ingest, these two lines have to
        change in the same commit. The test is here so they cannot not."""
        for script in ("control.js", "setup.js"):
            with open(os.path.join(VB_ROOT, "webui", "static", script)) as fh:
                source = fh.read()
            assert "estimated ingest ~${est.total_hours.toFixed(1)} h" in source

    def test_every_per_term_field_the_front_end_reads_exists(self):
        """`est.<field>` was pinned and `t.<field>` was not, so a renamed term
        key would still have reached a browser as `undefined`."""
        path = os.path.join(VB_ROOT, "webui", "static", "control.js")
        with open(path) as fh:
            source = fh.read()
        read = set(re.findall(r"\bt\.([A-Za-z_][A-Za-z0-9_]*)", source))
        assert read, "no t.<field> reads found in control.js"
        row = est("main")["query_terms"][0]
        assert sorted(read - set(row)) == []

    def test_the_front_end_shows_the_worst_case_total_as_well(self):
        with open(os.path.join(VB_ROOT, "webui", "static", "control.js")) as fh:
            source = fh.read()
        assert "worst_case_total_hours" in source
        assert "query_terms" in source
