"""End-to-end tests for the four ops workloads, run against FakeDriver.

The ops harness is the half of this framework that no test could reach. Build,
concurrency, filtered and churn all need a populated server, so the cheapest
check that any of them still works was a benchmark run — which meant they were
in practice checked by reading the code and by noticing, hours later, that a
chart was empty. `tests/fake_driver.py` removes that: the four workloads below
run end to end, against a 2,000-row in-process corpus, in well under a second.

What these tests are for is the plumbing between a driver and a record: that
each workload writes the phases it claims to write, that the numbers land in
the fields the report reads, and that the quantities which are supposed to move
together actually do. They do not measure any engine and the numbers in them
mean nothing outside this file.

The fake's own error model — the thing that makes a recall assertion here mean
anything — is calibrated in tests/test_fake_driver.py.
"""

from __future__ import annotations

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import datasets as ds  # noqa: E402
from harness.datasets import Dataset  # noqa: E402
from harness.metrics import records  # noqa: E402
from harness.workloads import build as build_workload  # noqa: E402
from harness.workloads import churn as churn_workload  # noqa: E402
from harness.workloads import concurrency as concurrency_workload  # noqa: E402
from harness.workloads import filtered as filtered_workload  # noqa: E402
from harness.workloads.context import RunContext  # noqa: E402

from .fake_driver import (K, N_QUERIES, N_ROWS, FakeDriver,  # noqa: E402
                          index_spec, loaded_driver, tiny_dataset)


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    return tiny_dataset()


@pytest.fixture(scope="module")
def angular_dataset() -> Dataset:
    return tiny_dataset(metric=ds.ANGULAR)


@pytest.fixture
def ctx(tmp_path):
    """A RunContext whose records and ground-truth cache land in tmp_path."""
    recorder = records.Recorder(str(tmp_path / "ops-fake.jsonl"))
    try:
        yield RunContext(
            run_id="t0-fake", resource_pass="normalized", recorder=recorder,
            engine="fake", engine_tag="none", march="x86-64-v3", k=K,
            cache_dir=str(tmp_path / "cache"),
        )
    finally:
        recorder.close()


def written(ctx: RunContext):
    """Every record the workload under test wrote, in order."""
    return list(records.read_records(ctx.recorder.path))


class TestBuildWorkload:
    def test_records_ingest_and_build_for_an_incremental_engine(self, ctx,
                                                                dataset):
        # `load_latency_s` buys a measurable ingest cost, so the equality
        # below is between two real numbers. The fake loads 2,000 x 16 rows in
        # about 0.3 ms and build.py rounds to milliseconds, so without it both
        # figures are literally 0.0 — measured 20 times out of 20 — and
        # `build_wall_s == ingest_wall_s` is 0.0 == 0.0.
        #
        # It has to be a knob inside the driver rather than a wrapper around
        # `driver.load`, and the reason is an asymmetry in build.py worth
        # knowing before it costs someone a red test: build.py times
        # `create_index` with a timer of its own, but takes the load cost from
        # the driver's self-reported `LoadResult.wall_seconds`. So the sibling
        # test below can wrap `create_index` and have the number move, while a
        # wrapper around `load` sleeps outside the measurement entirely and
        # the record still reads 0.0.
        #
        # The direction being covered matters: the invariant is that an
        # incrementally built engine's build cost IS its load cost, and the
        # fault to catch is that cost being recorded as the bulk-build timer,
        # which for such an engine never runs and is zero. MHNSW and VIDX
        # reported as having no build cost at all is precisely the
        # plausible-looking wrong number this repo opens by warning about. The
        # opposite direction — a bulk build misfiled as load time — is covered
        # by the sibling test below.
        driver = FakeDriver(dim=dataset.dim, metric=dataset.metric,
                            load_latency_s=0.02)
        driver.connect()
        build_workload.run(ctx, driver, dataset, index_spec(dataset))

        ingest, built = written(ctx)
        assert ingest["phase"] == records.PHASE_INGEST
        assert built["phase"] == records.PHASE_BUILD
        assert ingest["rows"] == built["rows"] == N_ROWS
        assert ingest["ingest_rows_per_s"] > 0
        assert ingest["ingest_threads"] == 1
        # An engine that maintains its graph on INSERT has no separable build,
        # so the build cost recorded is the load cost.
        assert built["extra"]["separable_build"] is False
        assert built["extra"]["incremental_index"] is True
        assert ingest["ingest_wall_s"] >= 0.02, "the load cost was not recorded"
        assert built["build_wall_s"] == ingest["ingest_wall_s"]
        assert built["index_bytes"] > 0 and built["table_bytes"] > 0
        assert driver.create_index_calls == 0
        assert driver.count_rows() == N_ROWS
        # The provenance every record shares, so a new workload cannot drop it.
        assert ingest["engine"] == "fake"
        assert ingest["engine_version"] == "fake-engine 0.1"
        assert ingest["resource_pass"] == "normalized"
        assert ingest["schema_version"] == records.SCHEMA_VERSION

    def test_separates_a_bulk_build_when_the_engine_can(self, ctx, dataset):
        # A load and a bulk build that each cost measurable time, and
        # deliberately different amounts. The fake does both in microseconds
        # and build.py rounds to milliseconds, so without this "separable"
        # could not mean two different numbers rather than one number written
        # twice — and the load equality at the end would be 0.0 == 0.0. The
        # two costs differ so the figures stay apart after rounding.
        #
        # Note the two mechanisms, for the reason given in the test above:
        # the load cost is a driver knob because build.py believes the driver
        # about it, while the bulk build is wrapped because build.py times
        # that one itself.
        driver = FakeDriver(dim=dataset.dim, metric=dataset.metric,
                            capabilities={"incremental_index": False},
                            load_latency_s=0.03)
        driver.connect()

        instant_build = driver.create_index

        def deliberate_build(index):
            time.sleep(0.02)
            instant_build(index)

        driver.create_index = deliberate_build
        build_workload.run(ctx, driver, dataset, index_spec(dataset))

        ingest, built = written(ctx)
        assert driver.create_index_calls == 1
        assert built["extra"]["separable_build"] is True
        # The build cost is the bulk build, not the load; for the incremental
        # engine above it is the load cost and the two are equal by
        # construction. Getting these the wrong way round would report
        # pgvector's cheap bulk build as its load time and vice versa.
        assert built["build_wall_s"] >= 0.02
        assert ingest["ingest_wall_s"] >= 0.03
        assert built["build_wall_s"] != built["ingest_wall_s"]
        assert built["ingest_wall_s"] == ingest["ingest_wall_s"]

    def test_subset_rows_loads_only_part_of_the_corpus(self, ctx, dataset):
        driver = FakeDriver(dim=dataset.dim, metric=dataset.metric)
        driver.connect()
        build_workload.run(ctx, driver, dataset, index_spec(dataset),
                           subset_rows=500)
        assert driver.count_rows() == 500
        assert written(ctx)[0]["rows"] == 500


class TestConcurrencyWorkload:
    def test_records_one_point_per_client_count(self, ctx, dataset):
        driver = loaded_driver(dataset, ef_half=64.0, ef_search=32)
        concurrency_workload.run(
            ctx, driver.clone, dataset, index_spec(dataset), ef_search=32,
            client_counts=(1, 2), duration_s=0.1, warmup=2,
            max_queries=N_QUERIES,
        )

        rows = written(ctx)
        assert [r["clients"] for r in rows] == [1, 2]
        assert all(r["phase"] == records.PHASE_CONCURRENCY for r in rows)
        for row in rows:
            assert row["ef_search"] == 32
            assert row["qps"] > 0
            assert row["queries_executed"] > 0
            # The wall-clock basis of QPS, asserted rather than described.
            # With N busy clients the wall clock is shared, so throughput is
            # queries over elapsed time; summing per-query latencies instead
            # counts concurrent work N times and divides QPS by roughly N.
            #
            # `queries_executed / qps` reconstructs the window the workload
            # measured, because both figures come out of the same merge. On
            # the wall-clock basis that quotient IS the measured wall, which
            # is a `time.sleep(duration_s)` and therefore never shorter than
            # `duration_s`, only longer by whatever the scheduler adds. On a
            # summed-latency basis it is `clients` times longer.
            #
            # The bound is one-sided on purpose. All the noise lives above
            # `duration_s` and none below it, so a two-sided tolerance tests
            # the machine rather than the code: a 5% one failed at a 5.7%
            # sleep overshoot on a box running three pytest processes at once,
            # while the fault being guarded against is a whole factor of N.
            #
            # Deliberately not asserted here: scaling_efficiency. The honest
            # value for two clients on this fake is 0.56-0.72 and varies run
            # to run, because the query path is Python and holds the GIL — the
            # same range a summed-latency QPS would report. An assertion on it
            # would be flaky and would not discriminate.
            implied_window = row["queries_executed"] / row["qps"]
            # The floor is structural, not measured: the quotient IS the wall
            # handed to merge_collectors, and that wall is taken around the
            # sleep, so it cannot come in under `duration_s` while the basis
            # stays the wall clock. Over 80 loaded windows the closest it came
            # was 1.00048x. It is kept as a statement of the relation — if the
            # basis ever changes to something unrelated to the window, this is
            # the line that says so — but all the discrimination lives in the
            # upper bound below and in the cross-client-count ratio after the
            # loop. Making the floor live would mean recording the measured
            # wall in the record, which is a production change.
            assert implied_window >= row["extra"]["duration_s"] * 0.99
            assert implied_window < row["extra"]["duration_s"] * 1.5
            # Separately, and unrelated to the above: percentiles are ordered.
            assert row["latency_p99_ms"] >= row["latency_p50_ms"]
            assert row["extra"]["duration_s"] == 0.1
            assert row["extra"]["repeats"] == 1
        assert rows[0]["extra"]["scaling_efficiency"] == 1.0
        # The same invariant in the form that does not care how loaded the
        # machine is: both windows are the same requested length, so the
        # reconstructed window must not grow with the client count. A
        # summed-latency QPS would double it from one client to two.
        windows = [r["queries_executed"] / r["qps"] for r in rows]
        assert windows[1] < windows[0] * 1.5

    def test_repeats_record_every_sample(self, ctx, dataset):
        # The spread is the point: a single sample has no error bar, which is
        # how two engines differing by less than the noise came to be presented
        # as ordered.
        driver = loaded_driver(dataset, ef_half=64.0, ef_search=32)
        concurrency_workload.run(
            ctx, driver.clone, dataset, index_spec(dataset), ef_search=32,
            client_counts=(1,), duration_s=0.05, warmup=1,
            max_queries=N_QUERIES, repeats=3,
        )
        row = written(ctx)[0]
        assert row["extra"]["repeats"] == 3
        assert len(row["extra"]["qps_samples"]) == 3
        assert row["extra"]["qps_spread_pct"] >= 0

    def test_a_client_count_above_the_query_count_is_skipped(self, ctx,
                                                             dataset):
        driver = loaded_driver(dataset, ef_half=64.0)
        concurrency_workload.run(
            ctx, driver.clone, dataset, index_spec(dataset), ef_search=32,
            client_counts=(1, 8), duration_s=0.05, warmup=1, max_queries=4,
        )
        assert [r["clients"] for r in written(ctx)] == [1]


class TestFilteredWorkload:
    def test_records_recall_and_short_results_per_selectivity(self, ctx,
                                                              dataset):
        driver = loaded_driver(dataset, ef_half=0.0, ef_search=64)
        filtered_workload.run(
            ctx, driver, dataset, index_spec(dataset), ef_search=64,
            selectivities=(0.01, 0.50), max_queries=N_QUERIES, warmup=2,
            indexed_rows=N_ROWS,
        )

        rows = written(ctx)
        assert [r["selectivity"] for r in rows] == [0.01, 0.50]
        assert all(r["phase"] == records.PHASE_FILTERED for r in rows)
        assert all(r["vector_index_used"] is True for r in rows)
        assert all(r["clients"] == 1 and r["qps"] > 0 for r in rows)
        assert [r["extra"]["tag_threshold"] for r in rows] == [1, 50]

        selective, permissive = rows
        # Post-filtering a 64-candidate list when 1% of rows qualify cannot
        # find ten of them, so the result sets are short and recall collapses.
        assert selective["extra"]["returned_fewer_than_k"] is True
        assert selective["extra"]["short_result_queries"] == N_QUERIES
        assert permissive["extra"]["short_result_queries"] == 0
        assert permissive["extra"]["returned_fewer_than_k"] is False
        assert permissive["recall_at_k"] > selective["recall_at_k"] + 0.5
        # Scored against filtered ground truth, which is what makes the 50%
        # number near-perfect rather than near-zero.
        assert permissive["recall_at_k"] > 0.95

    def test_runs_against_an_angular_corpus(self, ctx, angular_dataset):
        """The angular distance branch, through a real workload.

        `filtered.run` scores against `filtered_ground_truth`, which has its
        own angular branch and normalizes before comparing. A near-perfect
        recall here means the fake's cosine ordering and the harness's agree;
        if either drifted, this is where it would show rather than in a
        six-hour run on glove-100-angular.
        """
        driver = loaded_driver(angular_dataset, ef_half=0.0, ef_search=64)
        filtered_workload.run(
            ctx, driver, angular_dataset, index_spec(angular_dataset),
            ef_search=64, selectivities=(0.50,), max_queries=N_QUERIES,
            warmup=2, indexed_rows=N_ROWS,
        )
        row = written(ctx)[0]
        assert row["metric_space"] == ds.ANGULAR
        assert row["recall_at_k"] > 0.95
        assert row["extra"]["short_result_queries"] == 0

    def test_a_plan_failure_is_carried_into_the_record(self, ctx, dataset):
        # "Exact results, slowly" is indistinguishable from "accurate and slow"
        # unless the flag survives all the way to the record.
        driver = loaded_driver(dataset, ef_half=0.0, ef_search=64,
                               uses_vector_index=False)
        filtered_workload.run(
            ctx, driver, dataset, index_spec(dataset), ef_search=64,
            selectivities=(0.50,), max_queries=N_QUERIES, warmup=2,
            indexed_rows=N_ROWS,
        )
        assert written(ctx)[0]["vector_index_used"] is False


class TestChurnWorkload:
    def test_records_a_baseline_and_each_churn_step(self, ctx, dataset):
        # ef high enough that the baseline is near-exact, so the drop measured
        # after churn is the tombstones and not the ANN error.
        driver = loaded_driver(dataset, ef_half=1.0, ef_search=512,
                               tombstone_penalty=1.0)
        churn_workload.run(
            ctx, driver, dataset, index_spec(dataset), ef_search=512,
            fractions=(0.10, 0.25), max_queries=N_QUERIES,
        )

        rows = written(ctx)
        assert [r["churn_fraction"] for r in rows] == [0.0, 0.10, 0.25]
        assert all(r["phase"] == records.PHASE_CHURN for r in rows)
        assert rows[0]["notes"] == "baseline before any churn"

        for row in rows[1:]:
            assert row["extra"]["reinsert_completed"] is True
            assert row["extra"]["rows_churned"] > 0
            assert row["extra"]["insert_seconds"] >= 0
        # Each step churns the additional fraction, so 10% then 25% is 200 then
        # 300 rows, not 200 then 500.
        assert [r["extra"]["rows_churned"] for r in rows[1:]] == [200, 300]

        # The whole reason this workload exists: recall after churn, scored
        # against ground truth remapped to the re-inserted ids.
        recalls = [r["recall_at_k"] for r in rows]
        assert recalls[0] > recalls[1] > recalls[2], (
            f"recall did not degrade under churn: {recalls}")
        assert rows[2]["extra"]["recall_drop_vs_baseline"] > 0
        # The corpus is whole again after each step, and the index is not.
        assert driver.count_rows() == N_ROWS
        assert rows[2]["index_bytes"] > rows[0]["index_bytes"]

    def test_an_unfinished_reinsert_is_recorded_without_a_recall_figure(
            self, ctx, dataset):
        # A corpus missing rows is neither the baseline nor a churned corpus,
        # so what was achieved is recorded and recall deliberately is not.
        driver = loaded_driver(dataset, ef_half=1.0, ef_search=512,
                               latency_s=0.0)

        slow = driver.insert_rows

        def crawl(ids, vectors, tags):
            time.sleep(0.02)
            slow(ids, vectors, tags)

        driver.insert_rows = crawl
        churn_workload.run(
            ctx, driver, dataset, index_spec(dataset), ef_search=512,
            fractions=(0.10,), max_queries=N_QUERIES, budget_s=0.05,
        )

        rows = written(ctx)
        assert len(rows) == 2
        stopped = rows[1]
        assert stopped["extra"]["reinsert_completed"] is False
        assert stopped["extra"]["rows_reinserted"] < stopped["extra"]["rows_expected"]
        assert "recall_at_k" not in stopped
        assert stopped["notes"] == (
            "re-insert did not complete within the churn budget")


class TestWorkloadsShareOneSchema:
    def test_all_four_write_accepted_phases_into_one_file(self, ctx, dataset):
        """Every workload in one run, as a real invocation orders them.

        Churn runs last because it mutates the table: everything before it sees
        a pristine index and nothing after it inherits a degraded one. The
        ordering is part of what this asserts.
        """
        driver = FakeDriver(dim=dataset.dim, metric=dataset.metric,
                            ef_half=16.0, ef_search=64)
        driver.connect()
        index = index_spec(dataset)

        build_workload.run(ctx, driver, dataset, index)
        indexed_rows = driver.count_rows()
        concurrency_workload.run(
            ctx, driver.clone, dataset, index, ef_search=64,
            client_counts=(1, 2), duration_s=0.05, warmup=1,
            max_queries=N_QUERIES)
        filtered_workload.run(
            ctx, driver, dataset, index, ef_search=64, selectivities=(0.50,),
            max_queries=N_QUERIES, warmup=2, indexed_rows=indexed_rows)
        churn_workload.run(
            ctx, driver, dataset, index, ef_search=64, fractions=(0.10,),
            max_queries=N_QUERIES, indexed_rows=indexed_rows)

        phases = [r["phase"] for r in written(ctx)]
        assert phases == [
            records.PHASE_INGEST, records.PHASE_BUILD,
            records.PHASE_CONCURRENCY, records.PHASE_CONCURRENCY,
            records.PHASE_FILTERED,
            records.PHASE_CHURN, records.PHASE_CHURN,
        ]
        # Recorder.write refuses an unknown phase, so getting this far proves
        # every phase these workloads emit is in ALL_PHASES.
        assert set(phases) <= set(records.ALL_PHASES)
