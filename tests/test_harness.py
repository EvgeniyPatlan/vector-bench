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
