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

from ..datasets import Dataset, assign_tags
from ..drivers.base import EngineDriver, IndexSpec
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
    tags = assign_tags(len(vectors))

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
    load = driver.load(vectors, tags, threads=load_threads)
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
        extra={"separable_build": not incremental},
    ))
