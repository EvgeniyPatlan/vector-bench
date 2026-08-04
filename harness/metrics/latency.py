"""Latency accumulation and percentile reporting.

Percentiles are computed from every recorded sample rather than from a running
approximation. At the query counts used here (10k test vectors per run) keeping
all samples costs a few megabytes and removes a whole class of "is the p99
real?" doubt. `nearest-rank` is used rather than interpolation so a reported p99
is always a latency that actually occurred.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence


@dataclass
class LatencyStats:
    count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "queries_executed": self.count,
            "latency_mean_ms": round(self.mean_ms, 4),
            "latency_p50_ms": round(self.p50_ms, 4),
            "latency_p95_ms": round(self.p95_ms, 4),
            "latency_p99_ms": round(self.p99_ms, 4),
        }


def percentile(sorted_samples: Sequence[float], q: float) -> float:
    """Nearest-rank percentile. `q` in [0, 1]. Input must already be sorted."""
    if not sorted_samples:
        return float("nan")
    if q <= 0:
        return sorted_samples[0]
    if q >= 1:
        return sorted_samples[-1]
    rank = math.ceil(q * len(sorted_samples))
    return sorted_samples[min(max(rank - 1, 0), len(sorted_samples) - 1)]


def summarize(samples_s: Sequence[float]) -> LatencyStats:
    """Summarize latencies given in seconds; results are in milliseconds."""
    if not samples_s:
        return LatencyStats(0, float("nan"), float("nan"), float("nan"),
                            float("nan"), float("nan"), float("nan"))
    ms = sorted(s * 1000.0 for s in samples_s)
    return LatencyStats(
        count=len(ms),
        mean_ms=sum(ms) / len(ms),
        p50_ms=percentile(ms, 0.50),
        p95_ms=percentile(ms, 0.95),
        p99_ms=percentile(ms, 0.99),
        min_ms=ms[0],
        max_ms=ms[-1],
    )


class Timer:
    """Context manager returning elapsed wall time in seconds."""

    def __init__(self) -> None:
        self.elapsed: float = 0.0
        self._start: Optional[float] = None

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed = time.perf_counter() - (self._start or time.perf_counter())


class LatencyCollector:
    """Collects per-query latencies, optionally discarding a warm-up prefix.

    Warm-up matters here: the first queries against a freshly started engine pay
    for cold graph caches (MHNSW's per-TABLE_SHARE cache, VIDX's share cache,
    PostgreSQL's shared_buffers). Including them in a steady-state QPS figure
    understates every engine, and understates the ones with larger caches most.
    """

    def __init__(self, warmup: int = 0):
        self.warmup = max(0, warmup)
        self._samples: List[float] = []
        self._seen = 0
        self._wall_start: Optional[float] = None
        self._wall_end: Optional[float] = None

    def start(self) -> None:
        self._wall_start = time.perf_counter()

    def add(self, seconds: float) -> None:
        self._seen += 1
        if self._seen > self.warmup:
            self._samples.append(seconds)

    def stop(self) -> None:
        self._wall_end = time.perf_counter()

    @property
    def wall_seconds(self) -> float:
        if self._wall_start is None:
            return 0.0
        end = self._wall_end if self._wall_end is not None else time.perf_counter()
        return end - self._wall_start

    @property
    def measured_seconds(self) -> float:
        """Total time attributable to the measured (post-warm-up) queries."""
        return sum(self._samples)

    def qps(self, clients: int = 1) -> float:
        """Throughput over the measured window.

        With multiple clients the wall clock is shared, so QPS is measured
        queries divided by wall time — not by the sum of per-query latencies,
        which would count concurrent work multiple times.
        """
        wall = self.wall_seconds
        if wall <= 0 or not self._samples:
            return 0.0
        if clients > 1:
            return len(self._samples) / wall
        return len(self._samples) / max(self.measured_seconds, 1e-12)

    def stats(self) -> LatencyStats:
        return summarize(self._samples)

    def merge(self, other: "LatencyCollector") -> None:
        self._samples.extend(other._samples)
        self._seen += other._seen


def merge_collectors(collectors: Sequence[LatencyCollector],
                     wall_seconds: float) -> Dict[str, float]:
    """Combine per-client collectors into one concurrency data point."""
    samples: List[float] = []
    for c in collectors:
        samples.extend(c._samples)
    stats = summarize(samples)
    out = stats.as_dict()
    out["qps"] = len(samples) / wall_seconds if wall_seconds > 0 else 0.0
    return out
