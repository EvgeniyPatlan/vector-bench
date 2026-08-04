"""Filtered (hybrid) vector search: ANN combined with a scalar predicate.

This is the case integrated vector search is supposed to win — the reason to put
vectors in a relational database instead of a dedicated vector store — and it is
where the three implementations diverge most visibly, because each has to decide
between pre-filtering, post-filtering, or iterating the graph until enough rows
qualify.

Two things make this measurement easy to get wrong, and both are handled here:

1. **Ground truth changes with the filter.** The correct top-k over the rows
   passing `tag < t` is not the unfiltered top-k. Scoring filtered results
   against unfiltered ground truth would report near-zero recall for every
   engine and say nothing. Filtered ground truth is computed exactly, by brute
   force over the qualifying rows, and cached per (dataset, k, selectivity).

2. **Short result sets are not the same as wrong ones.** With post-filtering, an
   engine can exhaust its `ef_search` candidate list before finding k qualifying
   rows and return fewer than k. That is a real behaviour worth reporting, so
   the number of results returned is recorded alongside recall rather than being
   silently padded.
"""

from __future__ import annotations

import time
from typing import List, Optional, Sequence

import numpy

from ..datasets import (Dataset, cached_filtered_ground_truth, recall_at_k,
                        selectivity_to_threshold)
from ..drivers.base import EngineDriver, IndexSpec
from ..metrics.latency import LatencyCollector
from ..metrics.records import PHASE_FILTERED, Record
from .context import RunContext

DEFAULT_SELECTIVITIES = (0.01, 0.10, 0.50)


def run(ctx: RunContext, driver: EngineDriver, dataset: Dataset,
        index: IndexSpec, ef_search: int,
        selectivities: Sequence[float] = DEFAULT_SELECTIVITIES,
        max_queries: int = 1_000, warmup: int = 20,
        iterative_scan: Optional[str] = None,
        indexed_rows: Optional[int] = None) -> None:
    """Measure filtered recall and QPS at each selectivity."""
    queries = dataset.test[:max_queries]
    driver.set_ef_search(ef_search)

    # pgvector 0.8 can iterate the index when a filter rejects candidates.
    # Which mode is active changes the numbers substantially, so it is set
    # explicitly and recorded rather than left at whatever the default is.
    applied_mode = None
    if iterative_scan is not None and hasattr(driver, "set_iterative_scan"):
        driver.set_iterative_scan(iterative_scan)
        applied_mode = iterative_scan

    common = ctx.record_defaults(driver, dataset, index)

    for selectivity in selectivities:
        threshold = selectivity_to_threshold(selectivity)
        actual_selectivity = threshold / 100.0

        print(
            f"[filtered] {driver.name}: computing exact ground truth for "
            f"tag < {threshold} ({actual_selectivity:.0%} of rows)…"
        )
        # indexed_rows matters: a profile may have loaded only part of the
        # training set, and ground truth over the full set would score every
        # engine against rows it was never given.
        truth = cached_filtered_ground_truth(
            ctx.cache_dir, dataset, ctx.k, actual_selectivity,
            indexed_rows=indexed_rows,
        )
        truth = truth[:len(queries)]

        index_used = driver.explain_uses_vector_index(
            queries[0], ctx.k, tag_threshold=threshold
        )

        for i in range(min(warmup, len(queries))):
            driver.query_filtered(queries[i], ctx.k, threshold)

        collector = LatencyCollector(warmup=0)
        returned: List[List[int]] = []
        short_results = 0

        collector.start()
        for q in queries:
            started = time.perf_counter()
            ids = driver.query_filtered(q, ctx.k, threshold)
            collector.add(time.perf_counter() - started)
            returned.append(ids)
            if len(ids) < ctx.k:
                short_results += 1
        collector.stop()

        recall = recall_at_k(returned, truth, ctx.k)
        stats = collector.stats()
        qps = collector.qps(clients=1)

        print(
            f"[filtered] {driver.name}: selectivity={actual_selectivity:.0%} "
            f"recall@{ctx.k}={recall:.4f} qps={qps:,.1f} "
            f"p99={stats.p99_ms:.3f}ms "
            f"short_results={short_results}/{len(queries)} "
            f"index_used={index_used}"
        )

        ctx.recorder.write(Record(
            **common,
            phase=PHASE_FILTERED,
            ef_search=ef_search,
            selectivity=actual_selectivity,
            recall_at_k=round(recall, 6),
            qps=round(qps, 2),
            clients=1,
            vector_index_used=index_used,
            **stats.as_dict(),
            extra={
                "tag_threshold": threshold,
                "short_result_queries": short_results,
                "iterative_scan": applied_mode,
                # A filtered result set shorter than k is a legitimate engine
                # behaviour, not a harness bug — flagged so the report can say so.
                "returned_fewer_than_k": short_results > 0,
            },
        ))
