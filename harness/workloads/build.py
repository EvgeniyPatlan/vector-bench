"""Index build cost: ingest throughput, build time, index size on disk.

This is the dimension ann-benchmarks does not measure at all, and the one where
the three engines differ most structurally:

* MariaDB MHNSW and AliSQL VIDX maintain the graph on every INSERT. There is no
  separable build step — load cost and build cost are the same number.
* pgvector builds the graph as one bulk operation after the load, which is
  substantially cheaper per vector.

Reporting a single "build time" for all three would compare different
operations. This workload therefore records `ingest_wall_s` and `build_wall_s`
separately and marks which engine could separate them, so the report can say
"pgvector's bulk build is N× cheaper than its own incremental build" and
"pgvector's incremental build costs X versus MHNSW's Y" — two honest claims
instead of one misleading one.
"""

from __future__ import annotations

from typing import Optional

import numpy

from ..datasets import FILTER_CTAG, FILTER_TAG, Dataset, filter_values
from ..drivers.base import EngineDriver, IndexSpec, RowAttrs
from ..metrics.latency import Timer
from ..metrics.records import PHASE_BUILD, PHASE_INGEST, Record, Recorder
from ..progress import Heartbeat
from .context import RunContext


def run(ctx: RunContext, driver: EngineDriver, dataset: Dataset,
        index: IndexSpec, load_threads: int = 1,
        subset_rows: Optional[int] = None) -> None:
    """Load the dataset and build the index, recording the cost of each."""
    vectors = dataset.train
    if subset_rows is not None and subset_rows < len(vectors):
        vectors = vectors[:subset_rows]

    # Both filter columns, loaded in the one build -- PR 3's option A, argued
    # in `drivers/base.py:RowAttrs` and `datasets.py:FILTER_COLUMNS`. Three
    # things about these two lines are load-bearing.
    #
    # `filter_values` for BOTH columns, including `tag`, whose values it just
    # computes with `assign_tags`. The filtered workload asks the same
    # function for the values it builds exact ground truth over, keyed on the
    # same (dataset, rows) pair, so the column in the engine and the column
    # the yardstick was computed from cannot be two different arrays. Calling
    # `assign_tags` here and `filter_values` there would be two paths that
    # agree today.
    #
    # `len(vectors)`, not `subset_rows`: the pivots are drawn from the rows
    # given, so clustering 2,000 rows and clustering the first 500 of them
    # produce different labels for those 500 -- measured, 497 of 500 differ.
    # `main.py` takes `indexed_rows` from `driver.count_rows()` after this,
    # which is this number and not the profile's request.
    #
    # Before `drop_schema` and before either timer below, so the cluster
    # assignment is charged to neither ingest nor build. It is cached and
    # measured at 0.5-3.2 s for 1M rows against a ~1.9 h MHNSW ingest, but
    # "small" is not "outside the measurement".
    #
    # It is also a new SIDE EFFECT in a phase that previously had none:
    # `cached_ctags` does `makedirs(cache_dir, exist_ok=True)` and saves an
    # .npy, so a build-only profile -- `config/profiles/mariadb-blog-repro.yml`
    # declares `workloads: [build]` -- now writes a cache entry for labels
    # that run will never query. ~0.6 s and a few megabytes for glove, and
    # the alternative is a flag threading "will anything filter later?" from
    # the profile down to here, which is more coupling than the cost buys.
    #
    # One new failure mode, worth knowing before it costs someone an hour:
    # `assign_ctags` refuses a corpus smaller than its cluster count (100 by
    # default), so `build.run` now needs at least that many rows where it
    # previously took any. Clamping the count to the row count is the obvious
    # alternative and the dangerous one -- the count is part of the ctag cache
    # key, so a build that clamped and a ground truth that did not would score
    # the engine against a different clustering, which is the one failure the
    # shared `filter_values` call above exists to prevent.
    attrs = RowAttrs(
        tag=filter_values(ctx.cache_dir, dataset, FILTER_TAG,
                          indexed_rows=len(vectors)),
        ctag=filter_values(ctx.cache_dir, dataset, FILTER_CTAG,
                           indexed_rows=len(vectors)),
    )

    driver.drop_schema()

    with Timer() as schema_timer:
        driver.create_schema(index)
    print(f"[{driver.name}] schema created in {schema_timer.elapsed:.2f}s")

    incremental = driver.capabilities().get("incremental_index", True)
    if index.build_mode == "incremental":
        incremental = True

    print(
        f"[{driver.name}] loading {len(vectors):,} x {index.dim} vectors "
        f"with {load_threads} thread(s); "
        f"{'graph maintained during load' if incremental else 'bulk build after load'}"
    )
    load = driver.load(vectors, attrs, threads=load_threads)
    print(
        f"[{driver.name}] load: {load.wall_seconds:.1f}s "
        f"({load.rows_per_second:,.0f} rows/s)"
    )

    build_seconds = 0.0
    if not incremental:
        # A bulk CREATE INDEX is a single call that can run for many minutes
        # with no way to report fractional progress.
        with Heartbeat("bulk index build", prefix=driver.name):
            with Timer() as build_timer:
                driver.create_index(index)
        build_seconds = build_timer.elapsed
        print(f"[{driver.name}] bulk index build: {build_seconds:.1f}s")

    # For incrementally-built graphs the honest build cost IS the load cost:
    # the graph work happened inside those INSERTs and cannot be separated.
    effective_build = build_seconds if not incremental else load.wall_seconds

    index_bytes = driver.index_bytes()
    table_bytes = driver.table_bytes()
    rows = driver.count_rows()
    print(
        f"[{driver.name}] index {index_bytes:,} B, table {table_bytes:,} B, "
        f"{rows:,} rows"
    )

    common = ctx.record_defaults(driver, dataset, index)

    ctx.recorder.write(Record(
        **common,
        phase=PHASE_INGEST,
        rows=rows,
        ingest_wall_s=round(load.wall_seconds, 3),
        ingest_rows_per_s=round(load.rows_per_second, 2),
        ingest_threads=load.threads,
        notes=("graph maintained during INSERT" if incremental
               else "plain load, index built separately"),
    ))

    ctx.recorder.write(Record(
        **common,
        phase=PHASE_BUILD,
        rows=rows,
        build_wall_s=round(effective_build, 3),
        ingest_wall_s=round(load.wall_seconds, 3),
        ingest_threads=load.threads,
        index_bytes=index_bytes,
        table_bytes=table_bytes,
        notes=("incremental: build cost is the load cost" if incremental
               else "bulk build after load"),
        # The driver's own capabilities go in first. Without them an engine
        # whose index is built asynchronously by another process reported only
        # separable_build=True and was filed beside pgvector's bulk build,
        # which is a different operation, and its time-to-ready was dropped
        # entirely. separable_build is set last because it is a property of
        # how this workload ran, not of the driver.
        extra={**_driver_capabilities(driver),
               "separable_build": not incremental},
    ))


def _driver_capabilities(driver) -> dict:
    """Whatever the driver knows about itself that the record should carry.

    Defensive: a driver whose capabilities() raises should not lose the
    measurement that has already been taken.
    """
    try:
        return dict(driver.capabilities() or {})
    except Exception:
        return {}
