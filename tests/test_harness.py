"""Unit tests for the parts of the harness that can be wrong silently.

Priority is on the code where a bug would corrupt results rather than crash:
recall math, percentiles, the Pareto frontier, filtered ground truth, CPU
topology detection, and resource resolution. A crash gets noticed; a subtly
wrong recall number gets published.

Run: python3 -m pytest tests/ -v      (from the vector-bench directory)
"""

from __future__ import annotations

import os
import sys

import numpy
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import datasets as ds  # noqa: E402
from harness.metrics import latency, records, sysinfo  # noqa: E402


# ---------------------------------------------------------------------------
# Recall
# ---------------------------------------------------------------------------

class TestRecall:
    def test_perfect_recall(self):
        truth = numpy.array([[1, 2, 3], [4, 5, 6]])
        got = [[1, 2, 3], [4, 5, 6]]
        assert ds.recall_at_k(got, truth, 3) == 1.0

    def test_zero_recall(self):
        truth = numpy.array([[1, 2, 3]])
        assert ds.recall_at_k([[7, 8, 9]], truth, 3) == 0.0

    def test_partial_recall_ignores_order(self):
        # Recall is set intersection: order within the result must not matter.
        truth = numpy.array([[1, 2, 3, 4]])
        assert ds.recall_at_k([[4, 3, 99, 98]], truth, 4) == pytest.approx(0.5)

    def test_truncates_to_k(self):
        # An engine returning more than k must not be credited for the extras.
        truth = numpy.array([[1, 2]])
        assert ds.recall_at_k([[1, 2, 3, 4]], truth, 2) == 1.0
        assert ds.recall_at_k([[9, 9, 1, 2]], truth, 2) == 0.0

    def test_short_result_set_scores_proportionally(self):
        # Filtered search can legitimately return fewer than k.
        truth = numpy.array([[1, 2, 3, 4]])
        assert ds.recall_at_k([[1, 2]], truth, 4) == pytest.approx(0.5)

    def test_empty_input_is_nan_not_zero(self):
        # Zero would silently look like a real, terrible result.
        assert numpy.isnan(ds.recall_at_k([], numpy.array([[1]]), 1))


# ---------------------------------------------------------------------------
# Filtered ground truth
# ---------------------------------------------------------------------------

class TestFilteredGroundTruth:
    def test_tags_are_deterministic_and_uniform(self):
        tags = ds.assign_tags(1000, buckets=100)
        assert len(tags) == 1000
        assert set(tags.tolist()) == set(range(100))
        assert numpy.array_equal(tags, ds.assign_tags(1000, buckets=100))

    def test_selectivity_conversion(self):
        assert ds.selectivity_to_threshold(0.01) == 1
        assert ds.selectivity_to_threshold(0.10) == 10
        assert ds.selectivity_to_threshold(0.50) == 50
        # Never zero: a 0.1% request must still select a non-empty set.
        assert ds.selectivity_to_threshold(0.001) == 1
        assert ds.selectivity_to_threshold(2.0) == 100

    def test_only_qualifying_rows_are_returned(self):
        rng = numpy.random.default_rng(0)
        train = rng.random((500, 8), dtype=numpy.float32)
        test = rng.random((5, 8), dtype=numpy.float32)
        truth = ds.filtered_ground_truth(train, test, ds.EUCLIDEAN, k=5,
                                         tag_threshold=10, buckets=100)
        tags = ds.assign_tags(len(train))
        assert truth.shape == (5, 5)
        assert (tags[truth] < 10).all(), "returned a row that fails the filter"

    def test_matches_brute_force_euclidean(self):
        rng = numpy.random.default_rng(1)
        train = rng.random((200, 4), dtype=numpy.float32)
        test = rng.random((3, 4), dtype=numpy.float32)
        truth = ds.filtered_ground_truth(train, test, ds.EUCLIDEAN, k=3,
                                         tag_threshold=50, buckets=100)
        tags = ds.assign_tags(len(train))
        qualifying = numpy.nonzero(tags < 50)[0]
        for i, q in enumerate(test):
            distances = numpy.linalg.norm(train[qualifying] - q, axis=1)
            expected = qualifying[numpy.argsort(distances)[:3]]
            assert set(truth[i].tolist()) == set(expected.tolist())

    def test_matches_brute_force_angular(self):
        rng = numpy.random.default_rng(2)
        train = rng.random((200, 4), dtype=numpy.float32)
        test = rng.random((3, 4), dtype=numpy.float32)
        truth = ds.filtered_ground_truth(train, test, ds.ANGULAR, k=3,
                                         tag_threshold=50, buckets=100)
        tags = ds.assign_tags(len(train))
        qualifying = numpy.nonzero(tags < 50)[0]
        norm = lambda a: a / numpy.linalg.norm(a, axis=-1, keepdims=True)
        subset = norm(train[qualifying])
        for i, q in enumerate(test):
            similarity = subset @ norm(q)
            expected = qualifying[numpy.argsort(-similarity)[:3]]
            assert set(truth[i].tolist()) == set(expected.tolist())

    def test_empty_filter_raises(self):
        train = numpy.zeros((10, 2), dtype=numpy.float32)
        test = numpy.zeros((1, 2), dtype=numpy.float32)
        with pytest.raises(ValueError):
            ds.filtered_ground_truth(train, test, ds.EUCLIDEAN, k=1,
                                     tag_threshold=0, buckets=100)


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------

class TestLatency:
    def test_nearest_rank_percentile_returns_a_real_sample(self):
        samples = [1.0, 2.0, 3.0, 4.0, 5.0]
        for q in (0.0, 0.25, 0.5, 0.75, 1.0):
            assert latency.percentile(samples, q) in samples

    def test_percentile_bounds(self):
        samples = list(range(1, 101))
        assert latency.percentile(samples, 0.50) == 50
        assert latency.percentile(samples, 0.95) == 95
        assert latency.percentile(samples, 0.99) == 99
        assert latency.percentile(samples, 1.0) == 100

    def test_empty_percentile_is_nan(self):
        assert numpy.isnan(latency.percentile([], 0.5))

    def test_summarize_converts_seconds_to_ms(self):
        stats = latency.summarize([0.001, 0.002, 0.003])
        assert stats.count == 3
        assert stats.mean_ms == pytest.approx(2.0)
        assert stats.min_ms == pytest.approx(1.0)
        assert stats.max_ms == pytest.approx(3.0)

    def test_warmup_samples_are_excluded(self):
        collector = latency.LatencyCollector(warmup=3)
        collector.start()
        for value in (10.0, 10.0, 10.0, 1.0, 1.0):
            collector.add(value)
        collector.stop()
        stats = collector.stats()
        assert stats.count == 2, "warm-up queries leaked into the measurement"
        assert stats.mean_ms == pytest.approx(1000.0)

    def test_multi_client_qps_uses_wall_clock(self):
        # With N clients, summing per-query latencies would count concurrent
        # work N times over and inflate QPS by roughly N.
        a = latency.LatencyCollector()
        b = latency.LatencyCollector()
        for c in (a, b):
            c.start()
            for _ in range(100):
                c.add(0.01)
            c.stop()
        merged = latency.merge_collectors([a, b], wall_seconds=1.0)
        assert merged["qps"] == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# Trimming a measured window to the queries that finished inside it
# ---------------------------------------------------------------------------

class TestWindowTrimming:
    """QPS is counted queries over wall time, and the two used to disagree
    about where the window ended.

    Every client finishes the query it holds when `stop` is set and records
    it, but the wall clock stops at `stop`. So the numerator included one query
    per client that the denominator did not cover, and the overstatement is
    that query over the ones the client did finish:
    `1 / queries-completed-per-client`, about `latency / window`. The client
    count cancels, so it is 2.5% over a two-second window of 50 ms queries at
    any client count from 1 to 32 -- what makes it bite is a slow engine over
    a short window, not a wide client sweep.

    These tests use synthetic spans with exact instants, so every number below
    is arithmetic rather than a measurement. The window is [0, 1] throughout.
    """

    #: A span wholly inside the window, 100 ms long.
    INSIDE = (0.10, 0.20)
    #: A span that began inside the window and finished after the cutoff:
    #: the in-flight query §1.3 is about. Deliberately a full second long, so
    #: a merge that keeps it cannot help but move the percentiles too.
    OVERRUNS = (0.90, 1.90)
    #: A span that began before the window opened, which is what a client that
    #: passed the barrier late leaves behind.
    STARTED_EARLY = (-0.30, 0.40)

    def _collector(self, *spans) -> latency.LatencyCollector:
        collector = latency.LatencyCollector()
        collector.start()
        for started, finished in spans:
            collector.add_span(started, finished)
        collector.stop()
        return collector

    def test_the_record_keys_are_the_names_the_report_will_read(self):
        # Pinned as literals because they are a published part of the record,
        # not an internal name: a rename here silently renames a field in
        # every concurrency record. The workload and this test both spell the
        # key through the constant, so only this assertion notices.
        assert latency.INFLIGHT_DISCARDED == "inflight_discarded"
        assert latency.PREWINDOW_DISCARDED == "prewindow_discarded"

    def test_a_query_finishing_after_the_cutoff_is_not_counted(self):
        collector = self._collector(self.INSIDE, (0.3, 0.4), self.OVERRUNS)
        merged = latency.merge_collectors(
            [collector], wall_seconds=1.0, window_start=0.0, cutoff=1.0)
        assert merged["queries_executed"] == 2
        assert merged["qps"] == pytest.approx(2.0)
        assert merged[latency.INFLIGHT_DISCARDED] == 1
        assert merged[latency.PREWINDOW_DISCARDED] == 0

    def test_a_query_finishing_exactly_at_the_cutoff_is_counted(self):
        # The boundary belongs to the window: `finished > cutoff` drops, not
        # `>=`. A query that completed at the instant the clock stopped was
        # paid for inside the window the denominator measures.
        merged = latency.merge_collectors(
            [self._collector((0.5, 1.0))],
            wall_seconds=1.0, window_start=0.0, cutoff=1.0)
        assert merged["queries_executed"] == 1
        assert merged[latency.INFLIGHT_DISCARDED] == 0

    def test_the_discarded_query_is_gone_from_the_percentiles_too(self):
        # Not only from the count. The in-flight query is the one that ran
        # longest -- it is still running -- so leaving its latency in while
        # dropping it from the count would move p99 by more than the count
        # error it was fixing.
        collector = self._collector(self.INSIDE, (0.3, 0.4), self.OVERRUNS)
        merged = latency.merge_collectors(
            [collector], wall_seconds=1.0, window_start=0.0, cutoff=1.0)
        # Both kept spans are exactly 100 ms, so every percentile is 100.0 and
        # the 1,000 ms overrun cannot hide inside a tolerance.
        assert merged["latency_p99_ms"] == pytest.approx(100.0)
        assert merged["latency_mean_ms"] == pytest.approx(100.0)

    def test_a_query_that_began_before_the_window_is_not_counted(self):
        # Should never happen: every client waits on the barrier and the
        # window opens after it releases. Recording it is what turns "should
        # never happen" into something the record can be checked against.
        collector = self._collector(self.STARTED_EARLY, self.INSIDE)
        merged = latency.merge_collectors(
            [collector], wall_seconds=1.0, window_start=0.0, cutoff=1.0)
        assert merged["queries_executed"] == 1
        assert merged[latency.PREWINDOW_DISCARDED] == 1
        assert merged[latency.INFLIGHT_DISCARDED] == 0
        # 700 ms long against the survivor's 100 ms, so a merge that counted
        # it would be visible here even if the count above were right.
        assert merged["latency_mean_ms"] == pytest.approx(100.0)

    def test_every_span_lands_in_exactly_one_of_the_three_buckets(self):
        # Kept + discarded == recorded, so neither counter can double-count a
        # span that both started early and overran, and no span can vanish
        # without being accounted for in one of the two fields the record
        # publishes.
        spans = (self.INSIDE, self.OVERRUNS, self.STARTED_EARLY,
                 (-0.5, 1.5), (0.0, 1.0))
        merged = latency.merge_collectors(
            [self._collector(*spans)],
            wall_seconds=1.0, window_start=0.0, cutoff=1.0)
        accounted = (merged["queries_executed"]
                     + merged[latency.INFLIGHT_DISCARDED]
                     + merged[latency.PREWINDOW_DISCARDED])
        assert accounted == len(spans)
        # And the two kept are the two that fit: [0.1, 0.2] and [0.0, 1.0].
        assert merged["queries_executed"] == 2

        # WHICH bucket, not just how many buckets. The accounting assertion
        # above is satisfied by either ordering of the two tests inside
        # `span_disposition`, and `(-0.5, 1.5)` offends at both ends: it
        # started before the window opened AND outlived the cutoff. The
        # cutoff is tested first, so it counts as in-flight — 2 in-flight
        # here, not 1.
        #
        # This is diagnostic rather than numeric: the kept set, the QPS and
        # the percentiles are identical either way. What changes is the one
        # field whose entire value is being zero. `prewindow_discarded` is
        # documented to mean "the window's two ends disagree about when it
        # began", and under the other ordering an ordinary in-flight query
        # that merely happened to start early raises that alarm. Swapping the
        # two tests passed every other test in this repository.
        assert merged[latency.INFLIGHT_DISCARDED] == 2
        assert merged[latency.PREWINDOW_DISCARDED] == 1

    def test_without_a_cutoff_nothing_is_trimmed(self):
        # The way back, and what the other three workloads get: they measure
        # to completion rather than to a deadline, so there is no cutoff to
        # pass and no span of theirs may be dropped.
        collector = self._collector(self.INSIDE, self.OVERRUNS,
                                    self.STARTED_EARLY)
        merged = latency.merge_collectors([collector], wall_seconds=1.0)
        assert merged["queries_executed"] == 3
        assert merged[latency.INFLIGHT_DISCARDED] == 0
        assert merged[latency.PREWINDOW_DISCARDED] == 0

    def test_a_window_with_nothing_to_trim_is_untouched(self):
        # The trimmed and untrimmed merges of spans that all fit must agree
        # exactly, or the cutoff is doing something besides trimming.
        spans = ((0.0, 0.1), (0.2, 0.35), (0.4, 0.9))
        trimmed = latency.merge_collectors(
            [self._collector(*spans)],
            wall_seconds=1.0, window_start=0.0, cutoff=1.0)
        untrimmed = latency.merge_collectors(
            [self._collector(*spans)], wall_seconds=1.0)
        for key in ("queries_executed", "qps", "latency_mean_ms",
                    "latency_p50_ms", "latency_p95_ms", "latency_p99_ms"):
            assert trimmed[key] == pytest.approx(untrimmed[key]), key

    def test_durations_recorded_without_a_span_survive_a_cutoff(self):
        """`add(seconds)` is what the filtered and churn workloads call.

        A duration carries no instants, so there is nothing to compare
        against a cutoff and the only honest thing to do is keep it. If a
        cutoff silently dropped spanless samples, passing one would empty
        those workloads' percentiles rather than trim them.
        """
        collector = latency.LatencyCollector()
        collector.start()
        for _ in range(4):
            collector.add(0.25)
        collector.stop()
        merged = latency.merge_collectors(
            [collector], wall_seconds=1.0, window_start=0.0, cutoff=1.0)
        assert merged["queries_executed"] == 4
        assert merged["latency_mean_ms"] == pytest.approx(250.0)
        assert merged[latency.INFLIGHT_DISCARDED] == 0

    def test_add_span_and_add_agree_about_the_duration(self):
        # One quantity, two spellings. `add_span` must not be an independent
        # path to a different latency.
        spanned = self._collector((10.5, 10.75))
        duration = latency.LatencyCollector()
        duration.start()
        duration.add(0.25)
        duration.stop()
        assert (spanned.stats().mean_ms
                == pytest.approx(duration.stats().mean_ms))
        assert spanned.measured_seconds == pytest.approx(0.25)

    def test_warmup_still_counts_spans(self):
        # The prefix discard is about which queries are measured, not about
        # how they were recorded, so `add_span` has to honour it exactly as
        # `add` does -- the concurrency workload passes warmup=0 today, and a
        # span path that ignored it would make that silently load-bearing.
        collector = latency.LatencyCollector(warmup=2)
        collector.start()
        for started in (0.0, 0.1, 0.2, 0.3):
            collector.add_span(started, started + 0.05)
        collector.stop()
        assert collector.stats().count == 2

    def test_a_window_that_kept_nothing_reports_no_latency_at_all(self):
        """NaN is not JSON, and this is the path that reaches it.

        `summarize([])` returns NaN for every percentile and `json.dumps`
        writes that as the bare literal `NaN`: Python's `json.loads` takes it
        as an extension, the browser's `JSON.parse` throws. Before the window
        was trimmed the sample set was never empty, because every client
        completed at least one query; a configuration whose queries outlast
        the window now keeps nothing.
        """
        collector = self._collector((0.5, 1.5), (1.5, 2.5))
        merged = latency.merge_collectors(
            [collector], wall_seconds=1.0, window_start=0.0, cutoff=1.0)
        assert merged["queries_executed"] == 0
        assert merged[latency.INFLIGHT_DISCARDED] == 2
        for field in ("latency_mean_ms", "latency_p50_ms", "latency_p95_ms",
                      "latency_p99_ms"):
            assert merged[field] is None, (field, merged[field])
        # Zero completions over the window is what was observed, and it both
        # serialises and plots.
        assert merged["qps"] == 0.0

    def test_the_null_is_scoped_to_the_merged_window(self):
        """`summarize` itself is deliberately untouched.

        It predates this window and the build, filtered and churn workloads
        all reach their percentiles through it. Those loops run to completion,
        so an empty sample set there means nothing was asked of the engine at
        all — a different fact from "nothing finished in time", and not one
        this change has any business relabelling. Pinned because the obvious
        tidy-up is to move the null down into `summarize` and make all four
        workloads share it.
        """
        assert numpy.isnan(latency.summarize([]).p99_ms)
        empty = latency.LatencyCollector()
        empty.start()
        empty.stop()
        assert numpy.isnan(empty.stats().p99_ms)
        assert numpy.isnan(empty.stats().mean_ms)

    def test_trimming_cannot_let_more_work_through_than_the_clients_could_do(
            self):
        """Little's law, which is the invariant §1.3 restores.

        Mean latency x QPS is the average number of queries in flight, and a
        client runs one query at a time, so it cannot exceed the client count.
        Counting a query that outlived the window breaks that: here two
        clients each hold a 0.6 s query across the cutoff of a 1 s window, so
        untrimmed the four queries average 0.55 s at 4 q/s = 2.2 queries in
        flight across two clients, which no two clients can have done.

        This is the property the workload test pins, where the spans are real
        and no absolute rate can be asserted. Stated here on exact numbers so
        that what it means is not left to a measurement.
        """
        clients = [self._collector((0.1, 0.6), (0.6, 1.2)),
                   self._collector((0.2, 0.7), (0.7, 1.3))]
        trimmed = latency.merge_collectors(
            clients, wall_seconds=1.0, window_start=0.0, cutoff=1.0)
        untrimmed = latency.merge_collectors(clients, wall_seconds=1.0)
        in_flight = trimmed["qps"] * trimmed["latency_mean_ms"] / 1000.0
        assert in_flight <= len(clients)
        assert in_flight == pytest.approx(1.0)
        over = untrimmed["qps"] * untrimmed["latency_mean_ms"] / 1000.0
        assert over > len(clients), over


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

class TestRecords:
    def test_roundtrip(self, tmp_path):
        path = str(tmp_path / "r.jsonl")
        with records.Recorder(path) as recorder:
            recorder.write(records.Record(
                run_id="r1", engine="mariadb", phase=records.PHASE_RECALL,
                dataset="d", timestamp="t", recall_at_k=0.9, qps=100.0,
            ))
        loaded = list(records.read_records(path))
        assert len(loaded) == 1
        assert loaded[0]["recall_at_k"] == 0.9

    def test_none_fields_are_omitted(self):
        record = records.Record(run_id="r", engine="e", phase=records.PHASE_INGEST,
                                dataset="d", timestamp="t")
        as_dict = record.to_dict()
        assert "recall_at_k" not in as_dict
        assert as_dict["engine"] == "e"

    def test_unknown_phase_rejected(self, tmp_path):
        with records.Recorder(str(tmp_path / "r.jsonl")) as recorder:
            with pytest.raises(ValueError):
                recorder.write(records.Record(
                    run_id="r", engine="e", phase="nonsense",
                    dataset="d", timestamp="t"))

    def test_truncated_tail_is_tolerated(self, tmp_path):
        # A killed run leaves one partial line; everything before it is valid.
        path = tmp_path / "r.jsonl"
        path.write_text('{"a": 1}\n{"b": 2}\n{"c": ')
        assert list(records.read_records(str(path))) == [{"a": 1}, {"b": 2}]

    def test_corruption_before_valid_data_raises(self, tmp_path):
        path = tmp_path / "r.jsonl"
        path.write_text('{"a": 1}\n{broken\n{"c": 3}\n')
        with pytest.raises(ValueError):
            list(records.read_records(str(path)))


# ---------------------------------------------------------------------------
# CPU topology
# ---------------------------------------------------------------------------

class TestSysinfo:
    def test_collect_returns_plausible_values(self):
        info = sysinfo.collect()
        assert info.cpu.logical_cpus >= 1
        assert info.total_ram_bytes > 0
        assert isinstance(info.cpu.simd_flags, list)

    def test_cpuset_formatting_collapses_runs(self):
        assert sysinfo.format_cpuset([0, 1, 2, 3]) == "0-3"
        assert sysinfo.format_cpuset([0, 2, 4]) == "0,2,4"
        assert sysinfo.format_cpuset([0, 1, 2, 5, 7, 8]) == "0-2,5,7-8"
        assert sysinfo.format_cpuset([3]) == "3"
        assert sysinfo.format_cpuset([]) == ""

    def test_cpulist_parsing(self):
        assert sysinfo._parse_cpulist("0-3") == [0, 1, 2, 3]
        assert sysinfo._parse_cpulist("0,2,4") == [0, 2, 4]
        assert sysinfo._parse_cpulist("0-2,8,10-11") == [0, 1, 2, 8, 10, 11]
        assert sysinfo._parse_cpulist("") == []
        assert sysinfo._parse_cpulist(None) == []

    def test_hybrid_classes_are_disjoint(self):
        topology = sysinfo.detect_hybrid_topology()
        overlap = set(topology["performance"]) & set(topology["efficiency"])
        assert not overlap, f"a CPU was classified as both P and E: {overlap}"

    def test_recommended_cpuset_excludes_smt_siblings(self):
        chosen = sysinfo.recommended_cpuset(1000, allow_smt=False)
        siblings = sysinfo.hyperthread_siblings()
        if not siblings:
            pytest.skip("no topology information available")
        seen = set()
        for cpu in chosen:
            group = frozenset(siblings.get(cpu, [cpu]))
            assert group not in seen, f"two SMT siblings selected: {group}"
            seen.add(group)

    def test_recommended_cpuset_respects_the_requested_size(self):
        assert len(sysinfo.recommended_cpuset(2, allow_smt=True)) <= 2
        assert len(sysinfo.recommended_cpuset(0)) == 0


# ---------------------------------------------------------------------------
# Pareto frontier
# ---------------------------------------------------------------------------

class TestPareto:
    def _frontier(self, points):
        from report.loaders import pareto_frontier
        return pareto_frontier(points)

    def test_dominated_points_are_dropped(self):
        points = [
            {"recall_at_k": 0.90, "qps": 1000},   # on the frontier
            {"recall_at_k": 0.90, "qps": 500},    # dominated: same recall, slower
            {"recall_at_k": 0.95, "qps": 800},    # on the frontier
            {"recall_at_k": 0.80, "qps": 900},    # dominated by the 0.90/1000 point
        ]
        frontier = self._frontier(points)
        assert {(p["recall_at_k"], p["qps"]) for p in frontier} == {(0.90, 1000), (0.95, 800)}

    def test_frontier_is_ordered_by_recall(self):
        points = [{"recall_at_k": r, "qps": 1000 - r * 500}
                  for r in (0.5, 0.7, 0.9, 0.99)]
        frontier = self._frontier(points)
        recalls = [p["recall_at_k"] for p in frontier]
        assert recalls == sorted(recalls)

    def test_incomplete_points_are_ignored(self):
        points = [{"recall_at_k": 0.9, "qps": None}, {"recall_at_k": None, "qps": 10},
                  {"recall_at_k": 0.8, "qps": 50}]
        assert len(self._frontier(points)) == 1

    def test_empty_input(self):
        assert self._frontier([]) == []


class TestSubsetGroundTruth:
    """Ground truth must describe the rows the engine actually received.

    Profiles can load part of the training set (smoke loads 20,000 of 60,000).
    Scoring against ground truth computed over the full set would point at rows
    the engine never got — understating recall for every engine at once, which
    looks like a real result rather than a harness fault.
    """

    def _dataset(self, n=500, dim=8, queries=4):
        rng = numpy.random.default_rng(3)
        train = rng.random((n, dim), dtype=numpy.float32)
        test = rng.random((queries, dim), dtype=numpy.float32)
        d2 = ((test ** 2).sum(1)[:, None] - 2 * test @ train.T
              + (train ** 2).sum(1)[None, :])
        neighbors = numpy.argsort(d2, axis=1)[:, :10]
        return ds.Dataset("t", train, test, neighbors, None, ds.EUCLIDEAN)

    def test_never_references_unloaded_rows(self, tmp_path):
        dataset = self._dataset()
        truth = ds.cached_ground_truth(str(tmp_path), dataset, 5, indexed_rows=100)
        assert truth.max() < 100

    def test_matches_brute_force_over_the_subset(self, tmp_path):
        dataset = self._dataset()
        truth = ds.cached_ground_truth(str(tmp_path), dataset, 5, indexed_rows=100)
        for i, q in enumerate(dataset.test):
            distances = numpy.linalg.norm(dataset.train[:100] - q, axis=1)
            expected = numpy.argsort(distances)[:5]
            assert set(truth[i].tolist()) == set(expected.tolist())

    def test_full_load_reuses_shipped_neighbours(self, tmp_path):
        # Recomputing a 1M-row ground truth that already ships in the HDF5 file
        # would add hours to every run for no benefit.
        dataset = self._dataset()
        truth = ds.cached_ground_truth(str(tmp_path), dataset, 5, indexed_rows=None)
        assert numpy.array_equal(truth, dataset.neighbors[:, :5])

    def test_filtered_truth_also_respects_the_subset(self, tmp_path):
        dataset = self._dataset()
        truth = ds.cached_filtered_ground_truth(
            str(tmp_path), dataset, 5, 0.10, indexed_rows=100)
        tags = ds.assign_tags(100)
        assert truth.max() < 100
        assert (tags[truth] < 10).all()

    def test_row_count_is_part_of_the_cache_key(self, tmp_path):
        # Otherwise a subset run would poison the cache for a full run.
        dataset = self._dataset()
        a = ds.cached_filtered_ground_truth(
            str(tmp_path), dataset, 5, 0.50, indexed_rows=100)
        b = ds.cached_filtered_ground_truth(
            str(tmp_path), dataset, 5, 0.50, indexed_rows=300)
        assert a.max() < 100
        assert b.max() >= 100, "the 300-row truth was served from the 100-row cache"


class TestProgress:
    """Progress reporting must be time-driven and must not spam.

    Every long phase in this harness independently shipped with no output, so a
    working run was indistinguishable from a hung one. These guard the shared
    helper that replaced four near-identical loops.
    """

    def _capture(self):
        import io
        return io.StringIO()

    def test_reports_rate_and_eta_on_a_time_interval(self):
        import time as _t
        from harness.progress import Progress
        buf = self._capture()
        p = Progress(100, "widgets", prefix="eng", interval_s=0.05, stream=buf)
        for _ in range(100):
            p.step()
            _t.sleep(0.002)
        p.finish()
        out = buf.getvalue()
        assert "widgets" in out and "ETA" in out and "/s" in out
        assert out.count("\n") >= 2

    def test_silent_when_work_finishes_inside_one_interval(self):
        # A count-based trigger would print here; a time-based one must not.
        from harness.progress import Progress
        buf = self._capture()
        p = Progress(10, "fast", interval_s=60, stream=buf)
        for _ in range(10):
            p.step()
        p.finish()
        assert buf.getvalue() == ""

    def test_heartbeat_reports_liveness_then_completion(self):
        import time as _t
        from harness.progress import Heartbeat
        buf = self._capture()
        with Heartbeat("long thing", prefix="eng", interval_s=0.05, stream=buf):
            _t.sleep(0.22)
        out = buf.getvalue()
        assert "still running" in out
        assert "done in" in out

    def test_heartbeat_silent_for_short_operations(self):
        from harness.progress import Heartbeat
        buf = self._capture()
        with Heartbeat("quick", interval_s=60, stream=buf):
            pass
        assert buf.getvalue() == ""

    def test_interval_is_configurable_from_the_environment(self):
        # Operators on very slow hardware need to widen it without a code change.
        import importlib, os
        import harness.progress as prog
        old = os.environ.get("VB_PROGRESS_INTERVAL")
        try:
            os.environ["VB_PROGRESS_INTERVAL"] = "7"
            importlib.reload(prog)
            assert prog.DEFAULT_INTERVAL_S == 7.0
        finally:
            if old is None:
                os.environ.pop("VB_PROGRESS_INTERVAL", None)
            else:
                os.environ["VB_PROGRESS_INTERVAL"] = old
            importlib.reload(prog)


# ---------------------------------------------------------------------------
# Ops measurement version
# ---------------------------------------------------------------------------

class TestOpsMeasurementVersion:
    """Every ops record must say which measurement path produced it.

    `ANN_MEASUREMENT_VERSION` already does this for the recall/QPS path, where
    it feeds the fingerprint that keeps a 16 GB curve out of a 64 GB report.
    The ops path had no equivalent, so a harness change that altered what a
    number means left no trace in the number: a concurrency figure measured
    before the change and one measured after were indistinguishable in the
    record, and the report drew them as one series.

    The stamp is applied by `Recorder.write` rather than by
    `RunContext.record_defaults` -- the choke point every workload's provenance
    already passes through -- because four of the workloads pass `extra=` to
    `Record()` explicitly and an `extra` key in the defaults dict collides with
    that keyword. `Recorder.write` is the only point every ops record provably
    passes through, which is why `ALL_PHASES` is enforced there too.
    """

    def _written(self, tmp_path, record):
        path = str(tmp_path / "ops.jsonl")
        with records.Recorder(path) as recorder:
            recorder.write(record)
        return list(records.read_records(path))[0]

    def test_the_constant_starts_at_two(self):
        # Today's behaviour is implicitly 1: a record written before the
        # constant existed carries no version at all, so 1 is the value every
        # reader must infer for it and the first explicit value is 2.
        assert records.OPS_MEASUREMENT_VERSION >= 2
        assert records.IMPLICIT_OPS_MEASUREMENT_VERSION == 1
        assert records.OPS_MEASUREMENT_VERSION_KEY == "ops_measurement_version"

    def test_every_phase_carries_the_version(self, tmp_path):
        """ALL_PHASES, not a sample: a new phase must not be able to opt out."""
        for phase in records.ALL_PHASES:
            got = self._written(tmp_path, records.Record(
                run_id="r", engine="e", phase=phase, dataset="d", timestamp="t"))
            assert got["extra"][records.OPS_MEASUREMENT_VERSION_KEY] == \
                records.OPS_MEASUREMENT_VERSION, phase

    def test_a_workload_supplied_extra_is_kept_beside_the_version(self, tmp_path):
        # build.py writes extra={**driver.capabilities(), 'separable_build': ...}.
        # The stamp must neither replace that dict nor be shadowed by it.
        got = self._written(tmp_path, records.Record(
            run_id="r", engine="e", phase=records.PHASE_BUILD, dataset="d",
            timestamp="t", extra={"separable_build": False,
                                  "incremental_index": True}))
        assert got["extra"]["separable_build"] is False
        assert got["extra"]["incremental_index"] is True
        assert got["extra"][records.OPS_MEASUREMENT_VERSION_KEY] == \
            records.OPS_MEASUREMENT_VERSION

    def test_the_version_is_not_a_default_a_record_can_preset(self, tmp_path):
        """The stamp is unconditional, not setdefault.

        The version says which code produced the number. A record that arrives
        already claiming one is making a claim about a measurement this
        Recorder did not take -- a merge or replay tool feeding old records
        through a new Recorder, say -- and honouring it would relabel this
        run's own output. `setdefault` here passes every other test in this
        class, so this is the only one that pins it.
        """
        got = self._written(tmp_path, records.Record(
            run_id="r", engine="e", phase=records.PHASE_BUILD, dataset="d",
            timestamp="t",
            extra={records.OPS_MEASUREMENT_VERSION_KEY: 999}))
        assert got["extra"][records.OPS_MEASUREMENT_VERSION_KEY] == \
            records.OPS_MEASUREMENT_VERSION
        assert got["extra"][records.OPS_MEASUREMENT_VERSION_KEY] != 999

    def test_the_stamp_leaves_the_payload_it_was_given_alone(self):
        """A direct unit test of the function, which is where the copy is.

        The previous version of this guard asserted that `Recorder.write` left
        the caller's `Record` untouched and could not fail: `to_dict()` goes
        through `dataclasses.asdict`, which deep-copies `extra`, so the stamp
        never sees the caller's dict however it is written. asdict was doing
        the work the test credited to this code -- a stamp written straight
        into `payload` passed the whole suite.

        What is genuinely this function's contract is that it copies rather
        than writes, which is what makes it safe for a caller that has not
        been through asdict. The sibling test below pins the deep copy itself.
        """
        payload = {"phase": records.PHASE_BUILD,
                   "extra": {"separable_build": False}}
        stamped = records.stamp_measurement_version(payload)
        assert payload == {"phase": records.PHASE_BUILD,
                           "extra": {"separable_build": False}}
        assert stamped["extra"] is not payload["extra"]
        assert stamped["extra"][records.OPS_MEASUREMENT_VERSION_KEY] == \
            records.OPS_MEASUREMENT_VERSION

    def test_to_dict_hands_back_a_private_extra(self):
        """The deep copy `Recorder.write` relies on for the caller's Record.

        Not decoration: `to_dict` is a hot path that builds a dict per record
        and `asdict`'s recursive copy is the obvious thing to replace with a
        comprehension over `__dataclass_fields__`. Doing so would make the
        stamp -- and `attach_peak_rss`, and anything else that edits a record
        dict -- reach back into the live `Record`.
        """
        shared = {"separable_build": False}
        record = records.Record(
            run_id="r", engine="e", phase=records.PHASE_BUILD, dataset="d",
            timestamp="t", extra=shared)
        as_dict = record.to_dict()
        assert as_dict["extra"] is not shared
        as_dict["extra"]["scribbled"] = True
        assert shared == {"separable_build": False}
        assert record.extra == {"separable_build": False}

    def test_an_empty_extra_still_produces_the_key(self, tmp_path):
        # Record.to_dict() drops `extra` when it is empty, so the stamp has to
        # survive that filter rather than be applied before it. Asserted on the
        # value, not just membership -- a key present with the wrong number is
        # the failure that matters.
        got = self._written(tmp_path, records.Record(
            run_id="r", engine="e", phase=records.PHASE_INGEST, dataset="d",
            timestamp="t"))
        assert got["extra"][records.OPS_MEASUREMENT_VERSION_KEY] == \
            records.OPS_MEASUREMENT_VERSION

    def test_write_many_stamps_every_record(self, tmp_path):
        path = str(tmp_path / "ops.jsonl")
        with records.Recorder(path) as recorder:
            recorder.write_many([
                records.Record(run_id="r", engine="e", phase=p, dataset="d",
                               timestamp="t")
                for p in (records.PHASE_INGEST, records.PHASE_CHURN)
            ])
        loaded = list(records.read_records(path))
        assert len(loaded) == 2
        assert all(r["extra"][records.OPS_MEASUREMENT_VERSION_KEY]
                   == records.OPS_MEASUREMENT_VERSION for r in loaded)

    def test_a_real_workloads_records_carry_it(self, tmp_path):
        """Through the actual path, including the files this change cannot edit.

        build.py writes two records: an ingest record with no `extra` at all
        and a build record with an explicit `extra`. Both must come out
        stamped, which is the whole claim -- the version reaches records
        written by workload modules that know nothing about it.
        """
        from harness.workloads import build as build_workload
        from harness.workloads.context import RunContext

        from .fake_driver import K, FakeDriver, index_spec, tiny_dataset

        dataset = tiny_dataset()
        recorder = records.Recorder(str(tmp_path / "ops-fake.jsonl"))
        try:
            ctx = RunContext(run_id="t1a", resource_pass="normalized",
                             recorder=recorder, engine="fake", engine_tag="none",
                             march="x86-64-v3", k=K,
                             cache_dir=str(tmp_path / "cache"))
            driver = FakeDriver(dim=dataset.dim, metric=dataset.metric)
            driver.connect()
            build_workload.run(ctx, driver, dataset, index_spec(dataset))
        finally:
            recorder.close()

        written = list(records.read_records(str(tmp_path / "ops-fake.jsonl")))
        assert [r["phase"] for r in written] == [records.PHASE_INGEST,
                                                 records.PHASE_BUILD]
        for r in written:
            assert r["extra"][records.OPS_MEASUREMENT_VERSION_KEY] == \
                records.OPS_MEASUREMENT_VERSION, r["phase"]
        # The build record's own extra is still there, so the stamp did not
        # displace what build.py wrote.
        assert written[1]["extra"]["separable_build"] is False
