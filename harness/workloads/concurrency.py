"""Concurrency scaling: QPS and latency percentiles as client count rises.

This is the dimension that separates a database from an ANN library, and where
the three engines' cache designs diverge most:

* MariaDB MHNSW keeps one graph cache per TABLE_SHARE, bounded by
  `mhnsw_max_cache_size`, shared across sessions.
* AliSQL VIDX keeps a shared cache for read-only transactions plus a
  per-transaction cache for read-write ones, bounded by `vidx_hnsw_cache_size`.
* pgvector has no vector-specific cache at all; graph pages come through
  PostgreSQL's `shared_buffers` like any other index pages.

Those are three different answers to "what happens when 32 clients traverse the
same graph at once", and single-client QPS cannot reveal any of them.

Each client gets its own connection and a disjoint slice of the query set, so no
two clients issue the same query at the same instant — otherwise part of what
was measured would be one client hitting a cache another client just warmed.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, List, Optional, Sequence

import numpy

from ..datasets import Dataset
from ..drivers.base import EngineDriver, IndexSpec
from ..metrics.latency import LatencyCollector, merge_collectors
from ..metrics.records import PHASE_CONCURRENCY, Record
from ..progress import Heartbeat
from .context import RunContext

DriverFactory = Callable[[], EngineDriver]


class _Client(threading.Thread):
    """One benchmark client: own connection, own query slice, own latencies."""

    def __init__(self, driver_factory: DriverFactory, queries: numpy.ndarray,
                 k: int, ef_search: int, index: IndexSpec, warmup: int,
                 ready: threading.Barrier, stop: threading.Event):
        super().__init__(daemon=True)
        self._factory = driver_factory
        self._queries = queries
        self._k = k
        self._ef_search = ef_search
        self._index = index
        self._warmup = warmup
        self._ready = ready
        self._stop_event = stop
        self.collector = LatencyCollector(warmup=0)
        self.error: Optional[BaseException] = None

    def run(self) -> None:
        driver = None
        try:
            driver = self._factory()
            driver.connect()
            # The driver needs the metric to build its SELECT; supplying the
            # spec directly avoids a catalog round-trip per client.
            driver._index = self._index
            driver.set_ef_search(self._ef_search)

            # Warm up before the barrier: cold-cache cost belongs to neither the
            # measured window nor to one unlucky client, and every client must
            # enter the measured window equally warm.
            for i in range(min(self._warmup, len(self._queries))):
                driver.query(self._queries[i], self._k)
        except BaseException as exc:  # noqa: BLE001 - reported, not swallowed
            self.error = exc
        finally:
            # Always release the barrier, or a failing client deadlocks the run.
            try:
                self._ready.wait(timeout=120)
            except threading.BrokenBarrierError:
                pass

        if self.error is not None or driver is None:
            return

        try:
            self.collector.start()
            i = 0
            n = len(self._queries)
            while not self._stop_event.is_set():
                q = self._queries[i % n]
                started = time.perf_counter()
                driver.query(q, self._k)
                self.collector.add(time.perf_counter() - started)
                i += 1
            self.collector.stop()
        except BaseException as exc:  # noqa: BLE001
            self.error = exc
        finally:
            try:
                driver.close()
            except Exception:
                pass


def run(ctx: RunContext, driver_factory: DriverFactory, dataset: Dataset,
        index: IndexSpec, ef_search: int,
        client_counts: Sequence[int] = (1, 2, 4, 8, 16, 32),
        duration_s: float = 20.0, warmup: int = 50,
        max_queries: int = 10_000, repeats: int = 1) -> None:
    """Measure QPS and latency at each client count.

    Each point runs for a fixed duration rather than a fixed query count, so a
    slow configuration cannot stretch the run indefinitely and every point is
    measured over an identical window.

    `repeats` measures each point more than once and records the median window
    with the spread beside it. Every figure in this framework was a single
    sample until a Valkey point was re-measured by accident, five hours later
    on the same build and configuration, and came back 21% higher at sixteen
    and thirty-two clients. Nothing else had ever been repeated, so nothing
    else had an error bar, and two engines whose results differ by less than
    that were being presented as ordered.

    Concurrency is where this is worth paying for: it is the headline finding,
    it is the noisiest measurement here, and a point costs a minute. Repeating
    every point three times adds about a quarter of an hour per engine to a
    run that takes days.
    """
    queries = dataset.test[:max_queries]

    # One driver held open for metadata, so version/size lookups do not open and
    # discard a connection per data point.
    meta_driver = driver_factory()
    meta_driver.connect()
    meta_driver._index = index
    try:
        common = ctx.record_defaults(meta_driver, dataset, index)
    finally:
        meta_driver.close()

    baseline_qps: Optional[float] = None

    for clients in client_counts:
        if clients > len(queries):
            print(f"[concurrency] skipping {clients} clients: only "
                  f"{len(queries)} queries available")
            continue

        samples = []
        for attempt in range(max(1, repeats)):
            label = (f" (run {attempt + 1} of {repeats})" if repeats > 1 else "")
            print(f"[concurrency] {clients} client(s){label}: warming up "
                  f"({warmup} queries each), then measuring for "
                  f"{duration_s:.0f}s", flush=True)
            samples.append(_one_window(
                driver_factory, queries, clients, ctx.k, ef_search, index,
                warmup, duration_s))

        # The median window, not an average of several: latency percentiles
        # from one measured window mean something, and the mean of several
        # engines' percentiles means nothing.
        samples.sort(key=lambda s: s[0])
        qps, merged = samples[len(samples) // 2]
        spread = ((samples[-1][0] - samples[0][0]) / qps * 100.0) if qps else 0.0

        if clients == 1:
            baseline_qps = qps
        efficiency = (qps / (clients * baseline_qps)) if baseline_qps else None

        eff_text = f"{efficiency:.2f}" if efficiency is not None else "n/a"
        spread_text = f" spread={spread:.1f}%" if len(samples) > 1 else ""
        print(
            f"[concurrency] {common['engine']:>8} clients={clients:>3} "
            f"qps={qps:11,.1f} "
            f"p50={merged['latency_p50_ms']:7.3f}ms "
            f"p95={merged['latency_p95_ms']:7.3f}ms "
            f"p99={merged['latency_p99_ms']:7.3f}ms "
            f"efficiency={eff_text}{spread_text}"
        )

        ctx.recorder.write(Record(
            **common,
            phase=PHASE_CONCURRENCY,
            ef_search=ef_search,
            clients=clients,
            qps=round(qps, 2),
            **merged,
            extra={
                "scaling_efficiency": round(efficiency, 4) if efficiency is not None else None,
                "duration_s": duration_s,
                "warmup_queries_per_client": warmup,
                "repeats": len(samples),
                # Every sample, so a reader can see the spread rather than
                # trust a summary of it.
                "qps_samples": [round(q, 1) for q, _ in samples],
                "qps_spread_pct": round(spread, 1),
            },
        ))


def _one_window(driver_factory: DriverFactory, queries, clients: int, k: int,
                ef_search: int, index: IndexSpec, warmup: int,
                duration_s: float):
    """One warm-up and one measured window. Returns (qps, latency stats)."""
    shards = [queries[c::clients] for c in range(clients)]
    ready = threading.Barrier(clients + 1)
    stop = threading.Event()

    workers = [
        _Client(driver_factory, shards[c], k, ef_search, index,
                warmup, ready, stop)
        for c in range(clients)
    ]
    for w in workers:
        w.start()

    # Warm-up is unbounded from the operator's point of view: every client
    # runs `warmup` queries before the barrier releases, and on a slow
    # configuration that is minutes of silence before the measured window
    # even starts.
    try:
        with Heartbeat(f"{clients}-client warm-up", prefix="concurrency"):
            ready.wait(timeout=300)
    except threading.BrokenBarrierError:
        ready.abort()
        raise RuntimeError("a client failed during warm-up")

    wall_start = time.perf_counter()
    time.sleep(duration_s)
    stop.set()
    wall = time.perf_counter() - wall_start

    for w in workers:
        w.join(timeout=120)

    failures = [w.error for w in workers if w.error is not None]
    if failures:
        raise RuntimeError(
            f"{len(failures)} of {clients} clients failed; "
            f"first error: {failures[0]}")

    merged = merge_collectors([w.collector for w in workers], wall)
    return merged.pop("qps"), merged
