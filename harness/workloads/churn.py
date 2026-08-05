"""Churn: how recall and speed hold up after updates and deletes.

HNSW graphs degrade under deletion. A deleted node's edges either stay as
tombstones (wasting traversal) or are repaired (costing write time), and the
three engines make different choices. A vector index that is excellent on a
static corpus and mediocre after a week of writes is a different product from
one that holds steady, and only this workload shows the difference.

Method: delete a fraction of rows chosen deterministically, re-insert the same
vectors under fresh ids, then re-measure recall and QPS against ground truth
remapped to the new ids. Re-inserting the same vectors keeps the data
distribution — and therefore the correct answers — unchanged, so any recall drop
is graph degradation rather than a different dataset.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import numpy

from ..datasets import (Dataset, assign_tags, cached_ground_truth,
                        recall_at_k)
from ..drivers.base import EngineDriver, IndexSpec
from ..metrics.latency import LatencyCollector, Timer
from ..metrics.records import PHASE_CHURN, Record
from ..progress import Heartbeat, Progress
from .context import RunContext

DEFAULT_FRACTIONS = (0.10, 0.25)


def _measure(driver: EngineDriver, queries: numpy.ndarray, k: int,
             id_map: Dict[int, int], truth: numpy.ndarray,
             warmup: int = 20) -> Dict[str, float]:
    """Query and score against ground truth translated through `id_map`."""
    for i in range(min(warmup, len(queries))):
        driver.query(queries[i], k)

    collector = LatencyCollector(warmup=0)
    returned: List[List[int]] = []
    progress = Progress(len(queries), "churn queries", prefix=driver.name)
    collector.start()
    for q in queries:
        started = time.perf_counter()
        ids = driver.query(q, k)
        collector.add(time.perf_counter() - started)
        returned.append(ids)
        progress.step()
    collector.stop()
    progress.finish()

    # Ground truth refers to original ids; rows that were re-inserted now carry
    # new ids, so the expected set is translated rather than the results.
    translated = numpy.array(
        [[id_map.get(int(t), int(t)) for t in row[:k]] for row in truth],
        dtype=numpy.int64,
    )
    stats = collector.stats()
    out = stats.as_dict()
    out["recall_at_k"] = round(recall_at_k(returned, translated, k), 6)
    out["qps"] = round(collector.qps(clients=1), 2)
    return out


def run(ctx: RunContext, driver: EngineDriver, dataset: Dataset,
        index: IndexSpec, ef_search: int,
        fractions: Sequence[float] = DEFAULT_FRACTIONS,
        max_queries: int = 1_000, seed: int = 20260803,
        indexed_rows: Optional[int] = None) -> None:
    """Measure recall/QPS before churn and after each churn step."""
    queries = dataset.test[:max_queries]
    # Recomputed when only part of the training set was loaded; the shipped
    # neighbours would otherwise point at rows the engine never received.
    truth = cached_ground_truth(ctx.cache_dir, dataset, ctx.k,
                               indexed_rows=indexed_rows)[:max_queries]
    driver.set_ef_search(ef_search)

    common = ctx.record_defaults(driver, dataset, index)
    n_rows = driver.count_rows()
    rng = numpy.random.default_rng(seed)

    baseline = _measure(driver, queries, ctx.k, {}, truth)
    print(
        f"[churn] {driver.name}: baseline recall@{ctx.k}={baseline['recall_at_k']:.4f} "
        f"qps={baseline['qps']:,.1f}"
    )
    ctx.recorder.write(Record(
        **common, phase=PHASE_CHURN, ef_search=ef_search, clients=1,
        churn_fraction=0.0, rows=n_rows,
        index_bytes=driver.index_bytes(), **baseline,
        notes="baseline before any churn",
    ))

    id_map: Dict[int, int] = {}
    next_id = n_rows
    churned_so_far = 0.0

    for fraction in fractions:
        # Each step churns the *additional* fraction, so 0.10 then 0.25 means a
        # cumulative 25%, not 35%.
        step = max(0.0, fraction - churned_so_far)
        if step <= 0:
            continue
        count = int(round(step * n_rows))
        if count == 0:
            continue

        # Only rows still present under their current id are eligible.
        live_original = [i for i in range(n_rows) if i not in id_map]
        victims = rng.choice(len(live_original), size=min(count, len(live_original)),
                             replace=False)
        original_ids = [live_original[int(v)] for v in victims]
        current_ids = [id_map.get(o, o) for o in original_ids]
        vectors = dataset.train[original_ids]
        tags = assign_tags(n_rows)[original_ids]

        print(f"[churn] {driver.name}: deleting and re-inserting {count:,} rows "
              f"(cumulative {fraction:.0%})")

        with Heartbeat(f"deleting {len(current_ids):,} rows", prefix=driver.name):
            with Timer() as delete_timer:
                driver.delete_ids(current_ids)

        new_ids = list(range(next_id, next_id + len(original_ids)))
        next_id += len(original_ids)
        with Heartbeat(f"re-inserting {len(new_ids):,} rows", prefix=driver.name):
            with Timer() as insert_timer:
                driver.insert_rows(new_ids, vectors, tags)

        for original, new in zip(original_ids, new_ids):
            id_map[original] = new
        churned_so_far = fraction

        after = _measure(driver, queries, ctx.k, id_map, truth)
        drop = baseline["recall_at_k"] - after["recall_at_k"]
        print(
            f"[churn] {driver.name}: after {fraction:.0%} churn "
            f"recall@{ctx.k}={after['recall_at_k']:.4f} "
            f"(Δ{-drop:+.4f}) qps={after['qps']:,.1f}"
        )

        ctx.recorder.write(Record(
            **common, phase=PHASE_CHURN, ef_search=ef_search, clients=1,
            churn_fraction=fraction, rows=driver.count_rows(),
            index_bytes=driver.index_bytes(), **after,
            extra={
                "rows_churned": count,
                "delete_seconds": round(delete_timer.elapsed, 3),
                "insert_seconds": round(insert_timer.elapsed, 3),
                "recall_drop_vs_baseline": round(drop, 6),
            },
        ))
