"""Tests for the fake engine itself: its error model, not the harness.

`tests/fake_driver.py` is the instrument every ops-workload test is measured
with, so it needs its own calibration. A FakeDriver whose recall did not fall
as `ef_search` dropped would be worse than no fake at all: every recall
assertion in every test that uses it would pass without measuring anything.

So these tests are deliberately hard on the model rather than on the harness.
They pin p(ef) to its formula, prove the draw is per neighbour rather than per
query, prove a missed neighbour cannot return as a filler, prove tombstones
cost recall until `maintain()` runs and prove the fake's distance ordering
agrees with `harness.datasets`' ground truth in both metric spaces. Several of
them exist because the cheaper version of the same test passed against a
deliberately broken model.

The workloads themselves are tested in tests/test_workloads.py.
"""

from __future__ import annotations

import os
import sys
import time

import numpy
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import datasets as ds  # noqa: E402
from harness.datasets import Dataset  # noqa: E402
from harness.metrics import latency  # noqa: E402

from .fake_driver import (K, N_QUERIES, N_ROWS, FakeDriver,  # noqa: E402
                          loaded_driver, measure_recall, rotate_ids,
                          tiny_dataset)


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    return tiny_dataset()


class TestFakeDriverContract:
    def test_implements_every_abstract_method(self):
        # abc would refuse to instantiate it otherwise; this names the contract
        # so that a new abstract method on EngineDriver fails here, with a
        # readable message, rather than inside a workload.
        assert FakeDriver.__abstractmethods__ == frozenset()

    def test_capabilities_default_to_an_incremental_index(self):
        assert FakeDriver().capabilities() == {"incremental_index": True}

    def test_capabilities_are_configurable(self):
        driver = FakeDriver(capabilities={"incremental_index": False,
                                          "ef_construction_tunable": True})
        assert driver.capabilities()["incremental_index"] is False
        assert driver.incremental_index is False

    def test_explain_flag_is_configurable(self, dataset):
        assert FakeDriver().explain_uses_vector_index(dataset.test[0], K) is True
        blind = FakeDriver(uses_vector_index=False)
        assert blind.explain_uses_vector_index(dataset.test[0], K) is False
        assert blind.explain_uses_vector_index(
            dataset.test[0], K, tag_threshold=10) is False

    def test_querying_without_connecting_raises(self, dataset):
        # A workload that forgot to connect must fail loudly here rather than
        # measure an empty table.
        driver = FakeDriver(dim=dataset.dim)
        with pytest.raises(RuntimeError):
            driver.query(dataset.test[0], K)

    @pytest.mark.parametrize("metric", [ds.EUCLIDEAN, ds.ANGULAR])
    def test_an_exact_fake_returns_the_true_neighbours(self, metric):
        # ef_half = 0 makes p(ef) = 1. If this is not exactly 1.0 the fake's
        # distance ordering disagrees with harness.datasets' ground truth, and
        # every recall number measured against it is off by that disagreement.
        # Both metric spaces, because the fake has a branch per metric and the
        # angular one is what most of this framework's datasets use.
        data = tiny_dataset(metric=metric)
        driver = loaded_driver(data, ef_half=0.0)
        assert measure_recall(driver, data) == 1.0

    def test_the_two_metric_spaces_disagree_on_this_corpus(self):
        """Otherwise the angular anchor above would pass against L2 ordering.

        The anchor test is only as strong as the difference between the two
        orderings: on a corpus where cosine and L2 rank the same rows the same
        way, a fake that ignored the metric entirely would still score 1.0
        against angular ground truth. This measures the gap, so that if the
        corpus ever changes into one where the metrics coincide, the test that
        loses its power says so rather than going quietly green.
        """
        l2 = tiny_dataset(metric=ds.EUCLIDEAN)
        cosine = tiny_dataset(metric=ds.ANGULAR)
        overlap = numpy.array([
            len(set(a[:K].tolist()) & set(b[:K].tolist()))
            for a, b in zip(l2.neighbors, cosine.neighbors)
        ])
        # Not one query's top-k agrees completely between the two metrics, so
        # an L2-ordered fake cannot score 1.0 against angular truth.
        assert overlap.max() < K
        assert overlap.mean() < 0.8 * K


class TestAnnErrorModel:
    """p(ef) must be applied per neighbour, independently, and actually bite."""

    def test_recall_is_monotone_in_ef(self, dataset):
        driver = loaded_driver(dataset, ef_half=64.0)
        measured = []
        for ef in (8, 32, 128, 512):
            driver.set_ef_search(ef)
            measured.append((ef, measure_recall(driver, dataset)))

        recalls = [r for _, r in measured]
        assert recalls == sorted(recalls), (
            f"recall is not monotone in ef: {measured}")
        assert len(set(recalls)) == len(recalls), (
            f"ef changed nothing between two points: {measured}")
        # Not just ordered — the curve has to span a useful range, or a later
        # test could assert "recall > 0.9" at every ef and learn nothing.
        assert recalls[0] < 0.2
        assert recalls[-1] > 0.8

    def test_recall_tracks_the_p_ef_formula(self, dataset):
        # Measured recall is the mean of 500 independent Bernoulli(p) draws
        # (50 queries x k=10), so it must sit within a few percent of p(ef).
        # A per-query draw instead of a per-neighbour one would still be
        # monotone in ef but would miss this by a mile.
        driver = loaded_driver(dataset, ef_half=64.0)
        for ef in (8, 32, 128, 512):
            driver.set_ef_search(ef)
            # Five passes of the query set: 2,500 draws, a standard error of
            # about 0.01, so the tolerance is three sigma rather than the five
            # a single pass would need. Pinning it this tightly is what makes
            # the test sensitive to a model that is close but not the formula.
            assert measure_recall(driver, dataset, repeats=5) == pytest.approx(
                driver.p_ef(ef), abs=0.03), f"at ef={ef}"

    def test_the_draw_is_per_neighbour_not_per_query(self, dataset):
        """One draw per query would pass both tests above and still be useless.

        Keeping or dropping a whole result set with probability p is monotone
        in ef and averages to p(ef) just the same, but per-query recall would
        only ever be 0 or 1 — so no downstream test could tell a half-found
        result set from a missed one, and the short-result and churn cases
        would have no gradient to measure. The discriminator is the spread:
        k independent draws per query give a binomial spread of
        sqrt(p(1-p)/k) ~= 0.16, against 0.5 for a per-query draw.
        """
        driver = loaded_driver(dataset, ef_half=64.0, ef_search=64)  # p = 0.5
        truth = dataset.neighbors[:, :K]
        per_query = numpy.array([
            len(set(driver.query(q, K)) & {int(t) for t in want}) / K
            for q, want in zip(dataset.test, truth)
        ])
        assert per_query.mean() == pytest.approx(0.5, abs=0.05)
        assert per_query.std() == pytest.approx((0.25 / K) ** 0.5, abs=0.06)
        assert len(set(per_query.tolist())) >= 4, (
            f"per-query recall took only {sorted(set(per_query.tolist()))}, "
            f"which is what a single draw per query looks like")

    def test_ef_half_dials_the_curve(self, dataset):
        # The constructor knob is what lets a test pick a recall regime instead
        # of discovering which one it got.
        sharp = loaded_driver(dataset, ef_half=4.0, ef_search=32)
        blunt = loaded_driver(dataset, ef_half=256.0, ef_search=32)
        assert measure_recall(sharp, dataset) > 0.8
        assert measure_recall(blunt, dataset) < 0.2

    def test_probabilities_stay_inside_the_unit_interval(self, dataset):
        """Hostile knob values must not produce a probability outside [0, 1].

        Neither failure announces itself. A negative `ef_search` gives a
        negative p(ef); a `tombstone_penalty` above 1.0 gives a negative
        tombstone factor; multiplied together the two sign errors cancel into
        a perfectly plausible keep rate. And `rng.random() < p` never
        complains about a p outside the interval — it simply always or never
        fires, which reads as a working model with a surprising recall.
        """
        driver = loaded_driver(dataset, ef_half=64.0, tombstone_penalty=5.0)
        driver.delete_ids(list(range(1_000)))   # fraction 0.5, penalty 2.5
        assert driver.keep_probability() == 0.0

        driver.set_ef_search(-10)
        assert driver.p_ef() == 0.0
        assert driver.keep_probability() == 0.0
        # Clamped to zero has to mean "finds nothing", not "finds everything".
        assert measure_recall(driver, dataset) == 0.0

    def test_results_hold_k_distinct_live_ids(self, dataset):
        """Exactly k distinct ids, every one of them a live row.

        Measured on a churned store, where the live id set is neither
        contiguous nor the whole range: ids 0-1599 and 2000-2399 are live and
        1600-1999 are tombstoned. Against a freshly loaded store the
        containment assertion cannot fail, because every id in existence is
        live — which makes it look like a guard without being one.
        """
        driver = loaded_driver(dataset, ef_half=64.0, ef_search=16)
        id_map = rotate_ids(driver, dataset, 400)
        live = set(range(N_ROWS - 400)) | set(id_map.values())
        for q in dataset.test:
            got = driver.query(q, K)
            assert len(got) == K
            assert len(set(got)) == K, f"duplicate id in {got}"
            assert set(got) <= live, f"not a live id: {set(got) - live}"

    def test_a_missed_neighbour_is_not_handed_back_as_a_filler(self, dataset):
        """No true neighbour may return as a filler, ever.

        The filler pool excludes the whole true top-k, not just the part of it
        still in the result. A filler that happened to be the neighbour just
        dropped would hand recall back and push measured recall above p(ef).

        Aggregated over ten passes — 5,000 filler draws — rather than one,
        because this is the only direct guard for the invariant and a single
        pass is probabilistic: a regression to "exclude only the survivors"
        leaks about one true id per twenty queries, so one pass of fifty misses
        it roughly one time in ten. Ten passes make a clean run essentially
        impossible under the regression, and nothing else would catch it:
        the leak moves recall by about half a percent, well inside the
        tolerance of test_recall_tracks_the_p_ef_formula.
        """
        driver = loaded_driver(dataset, ef_half=1e6, ef_search=1)
        truth = dataset.neighbors[:, :K]
        leaked = 0
        draws = 0
        for _ in range(10):
            for q, want in zip(dataset.test, truth):
                got = driver.query(q, K)
                # p(1) with ef_half = 1e6 is ~1e-6: nothing true survives, so
                # every id returned is a filler.
                leaked += len(set(got) & {int(t) for t in want})
                draws += len(got)
        assert draws == 10 * N_QUERIES * K, "the result sets were not full"
        assert leaked == 0, f"{leaked} of {draws} fillers were true neighbours"


class TestTombstones:
    def test_delete_marks_rather_than_reclaims(self, dataset):
        driver = loaded_driver(dataset, ef_half=0.0)
        before = driver.index_bytes()
        driver.delete_ids(list(range(200)))
        assert driver.count_rows() == N_ROWS - 200
        # The graph still carries the deleted nodes, which is why churn records
        # index_bytes beside recall.
        assert driver.index_bytes() == before
        assert driver.tombstone_fraction() == pytest.approx(0.1)

    def test_a_tombstoned_id_is_never_returned(self, dataset):
        driver = loaded_driver(dataset, ef_half=64.0, ef_search=32)
        deleted = set(range(0, N_ROWS, 2))
        driver.delete_ids(sorted(deleted))
        for q in dataset.test:
            assert not (set(driver.query(q, K)) & deleted)

    def test_tombstones_lower_recall_until_maintain(self, dataset):
        driver = loaded_driver(dataset, ef_half=0.0, tombstone_penalty=1.0)
        clean = measure_recall(driver, dataset)
        assert clean == 1.0

        id_map = rotate_ids(driver, dataset, 400)
        churned = measure_recall(driver, dataset, id_map=id_map, repeats=4)
        # 400 tombstones in a 2,400-row array, at a penalty of 1.0, is a
        # one-sixth cut, and the tolerance is three standard errors of the
        # 2,000 draws four passes give. The figure matters less than that it is
        # a real cut: the corpus is intact and the vectors are unchanged, so
        # the only thing that moved is the dead weight in the graph.
        assert churned == pytest.approx(1.0 - 400 / 2400, abs=0.03)

        assert driver.maintain() == 400
        assert driver.tombstone_fraction() == 0.0
        assert measure_recall(driver, dataset, id_map=id_map) == clean

    def test_penalty_of_zero_leaves_recall_alone(self, dataset):
        # The way back, for a test that wants deletes without a recall story.
        driver = loaded_driver(dataset, ef_half=0.0, tombstone_penalty=0.0)
        id_map = rotate_ids(driver, dataset, 400)
        assert driver.tombstone_fraction() > 0
        assert measure_recall(driver, dataset, id_map=id_map) == 1.0


class TestFilteredQueries:
    def test_every_returned_row_passes_the_predicate(self, dataset):
        driver = loaded_driver(dataset, ef_half=0.0, ef_search=256)
        tags = ds.assign_tags(N_ROWS)
        for q in dataset.test:
            for row in driver.query_filtered(q, K, 10):
                assert tags[row] < 10

    def test_post_filtering_returns_short_result_sets(self, dataset):
        # Only the top-ef candidates are filtered, so a selective predicate
        # runs out of qualifying rows. This is a real engine behaviour and the
        # filtered workload reports it; the fake has to produce it.
        driver = loaded_driver(dataset, ef_half=0.0, ef_search=64)
        short = sum(1 for q in dataset.test
                    if len(driver.query_filtered(q, K, 1)) < K)
        assert short == N_QUERIES
        # The same predicate at 50% selectivity fills every result set.
        full = sum(1 for q in dataset.test
                   if len(driver.query_filtered(q, K, 50)) == K)
        assert full == N_QUERIES

    def test_a_wider_candidate_list_recovers_qualifying_rows(self, dataset):
        narrow = loaded_driver(dataset, ef_half=0.0, ef_search=20)
        wide = loaded_driver(dataset, ef_half=0.0, ef_search=400)
        counts = [sum(len(d.query_filtered(q, K, 5)) for q in dataset.test)
                  for d in (narrow, wide)]
        assert counts[0] < counts[1]

    def test_an_unknown_column_is_refused(self, dataset):
        driver = loaded_driver(dataset, ef_half=0.0)
        with pytest.raises(ValueError):
            driver.query_filtered(dataset.test[0], K, 10, column="category")
        # And the configured one is accepted.
        driver.query_filtered(dataset.test[0], K, 10, column="tag")


class TestSimulatedLatency:
    def test_latency_s_is_paid_per_query(self, dataset):
        driver = loaded_driver(dataset, ef_half=0.0, latency_s=0.004)
        started = time.perf_counter()
        for q in dataset.test[:5]:
            driver.query(q, K)
        assert time.perf_counter() - started >= 5 * 0.004

    def test_jitter_does_not_disturb_the_error_model(self, dataset):
        """Turning latency on must not move a single recall draw.

        Both streams come from one seed, so two drivers that differ only in
        their latency settings must return bit-identical results. Sharing one
        Generator between the jitter and the error model would break this: a
        later task adding jitter to a percentile test would watch its recall
        numbers shift, with nothing in the model to explain it and no way to
        see why.
        """
        quiet = loaded_driver(dataset, ef_half=64.0, ef_search=32)
        noisy = loaded_driver(dataset, ef_half=64.0, ef_search=32,
                              latency_s=0.0002, latency_jitter=0.001)
        assert [noisy.query(q, K) for q in dataset.test] == \
               [quiet.query(q, K) for q in dataset.test]

    def test_jitter_spreads_the_percentiles(self, dataset):
        # Without jitter every sample is the same and a percentile assertion
        # anywhere downstream is untestable.
        driver = loaded_driver(dataset, ef_half=0.0, latency_s=0.002,
                               latency_jitter=0.006)
        collector = latency.LatencyCollector(warmup=0)
        collector.start()
        for q in dataset.test[:40]:
            began = time.perf_counter()
            driver.query(q, K)
            collector.add(time.perf_counter() - began)
        collector.stop()
        stats = collector.stats()
        assert stats.p50_ms >= 2.0
        assert stats.p99_ms > stats.p50_ms * 1.2
