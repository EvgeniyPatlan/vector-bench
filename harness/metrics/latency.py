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
from typing import Dict, List, NamedTuple, Optional, Sequence

#: Key under a concurrency record's `extra` counting the queries that were
#: still running when the measured window closed. One per client at most --
#: a client issues its queries one at a time -- and they are not counted,
#: because the wall clock in the denominator stopped at the window's end.
INFLIGHT_DISCARDED = "inflight_discarded"

#: Key counting the queries that had already started when the window opened.
#: Zero on every run, and structurally so: the window opens at the instant the
#: clients' barrier releases, which precedes every measured query. It is
#: recorded rather than assumed because the first attempt at this field was not
#: zero at all -- the window opened when the parent thread next read the clock,
#: which six of eight clients beat to their first query. A number here means
#: the window's two ends disagree about when it began.
PREWINDOW_DISCARDED = "prewindow_discarded"


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


class Sample(NamedTuple):
    """One query's latency, with the instants bracketing it when known.

    `started` and `finished` are `time.perf_counter()` readings and are None
    for a sample recorded by duration alone. That is not a degenerate case: it
    is what the filtered and churn workloads produce, and they measure to
    completion rather than to a deadline, so there is no window for them to
    fall outside of and nothing for a cutoff to compare against.

    `seconds` is derived once, at construction, rather than recomputed from the
    instants on the way out. Two spellings of one duration is how they come to
    disagree.
    """

    seconds: float
    started: Optional[float] = None
    finished: Optional[float] = None


def span_disposition(started: Optional[float], finished: Optional[float],
                     window_start: Optional[float],
                     cutoff: Optional[float]) -> Optional[str]:
    """Why a query does not belong to a measured window, or None if it does.

    ONE definition, used both by `merge_collectors` to decide which latencies
    to count and by the concurrency workload to decide which returned ids to
    score. Those two decisions have to be the same decision: the window's QPS
    and the recall printed beside it would otherwise describe different sets of
    queries, and `extra.query_reuse` -- executed over distinct -- would come
    out below 1.0 for a window in which nothing was repeated.

    The cutoff is tested first, so a query that both started before the window
    and outlived it counts once, as in-flight. Every span therefore lands in
    exactly one of three buckets and the two discard counts plus the kept
    count add up to what was recorded.
    """
    if cutoff is not None and finished is not None and finished > cutoff:
        # `>` and not `>=`: a query that completed at the instant the clock
        # stopped was paid for inside the window the denominator measures.
        return INFLIGHT_DISCARDED
    if (window_start is not None and started is not None
            and started < window_start):
        return PREWINDOW_DISCARDED
    return None


class LatencyCollector:
    """Collects per-query latencies, optionally discarding a warm-up prefix.

    Warm-up matters here: the first queries against a freshly started engine pay
    for cold graph caches (MHNSW's per-TABLE_SHARE cache, VIDX's share cache,
    PostgreSQL's shared_buffers). Including them in a steady-state QPS figure
    understates every engine, and understates the ones with larger caches most.
    """

    def __init__(self, warmup: int = 0):
        self.warmup = max(0, warmup)
        self._samples: List[Sample] = []
        self._seen = 0
        self._wall_start: Optional[float] = None
        self._wall_end: Optional[float] = None

    def start(self) -> None:
        self._wall_start = time.perf_counter()

    def add(self, seconds: float) -> None:
        """Record a query by its duration alone.

        What the build, filtered and churn workloads call. Those loops run to
        completion, so a duration is the whole truth about the query and there
        is no window it could fall outside.
        """
        self._record(Sample(seconds))

    def add_span(self, started: float, finished: float) -> None:
        """Record a query by the two instants that bracket it.

        What a fixed-duration window needs, because whether a query counts
        depends on WHEN it ran and not only on how long it took. Both
        arguments must be `time.perf_counter()` readings on the same clock as
        the window's own start and cutoff.

        The cost of keeping the instants is two more floats per query -- at a
        fast engine's 20,000 q/s over a 20 s window that is tens of megabytes
        in the harness container, which holds the dataset in NumPy anyway and
        is deliberately a different container from the one whose memory the
        benchmark measures.
        """
        self._record(Sample(finished - started, started, finished))

    def _record(self, sample: Sample) -> None:
        """The one place a sample enters, so the warm-up prefix is discarded
        identically however the caller spelled the query."""
        self._seen += 1
        if self._seen > self.warmup:
            self._samples.append(sample)

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
        return sum(s.seconds for s in self._samples)

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
        return summarize([s.seconds for s in self._samples])

    def merge(self, other: "LatencyCollector") -> None:
        self._samples.extend(other._samples)
        self._seen += other._seen


def merge_collectors(collectors: Sequence[LatencyCollector],
                     wall_seconds: float,
                     cutoff: Optional[float] = None,
                     window_start: Optional[float] = None) -> Dict[str, float]:
    """Combine per-client collectors into one concurrency data point.

    `window_start` and `cutoff` are the instants the measured window opened
    and closed. Given them, only the queries that both began and ended inside
    that window are counted -- which is the only set of queries the wall time
    in the denominator actually covers.

    WHY THAT MATTERS, because the correction looks like a rounding. A
    fixed-duration window ends by setting a flag; each client then finishes the
    query it is holding and records it, so the numerator used to include one
    query per client that the denominator did not.

    The relative error is that one query over the queries a client did finish:
    `1 / queries-completed-per-client`, which in published record fields is
    `inflight_discarded / queries_executed` and is about
    `per-query latency / window length`. **The client count cancels.** Each
    client contributes one spurious query and roughly `window / latency` real
    ones, so the ratio is the same at 1 client as at 32 -- measured flat at
    2.564% across 1, 2, 4, 8, 16 and 32 clients over a two-second window of
    50 ms queries. What the error does depend on is a SLOW ENGINE OVER A SHORT
    WINDOW: it is 0.005% for a 1.4 ms query in a 30-second window and 2.5% for
    a 50 ms query in two seconds, whatever the concurrency.

    It also broke Little's law, reporting more queries in flight on average
    than there were clients to hold them, which no client count can do.

    Both bounds are optional and default to off, so a caller that measures to
    completion -- the filtered and churn workloads -- keeps every sample. A
    sample recorded by duration alone is never dropped: it carries no instants
    to compare, and dropping it would empty those workloads' percentiles rather
    than trim them.

    Returns the latency summary plus `qps` and the two discard counts, which
    the concurrency workload moves into the record's `extra`. A window in which
    nothing finished in time reports `queries_executed` 0, a QPS of 0 and no
    latency figures at all -- the honest answer for a configuration whose
    queries outlast the window, rather than one in-flight query's throughput.
    """
    kept: List[float] = []
    discarded = {INFLIGHT_DISCARDED: 0, PREWINDOW_DISCARDED: 0}
    for c in collectors:
        for sample in c._samples:
            reason = span_disposition(sample.started, sample.finished,
                                      window_start, cutoff)
            if reason is None:
                kept.append(sample.seconds)
            else:
                discarded[reason] += 1

    stats = summarize(kept)
    out = stats.as_dict()
    if not kept:
        # None, not the NaN `summarize([])` returns. `json.dumps` writes a NaN
        # as the bare literal `NaN`, which is not JSON: Python's `json.loads`
        # accepts it as a documented extension and the browser's `JSON.parse`
        # rejects it, so a records file holding one reads back perfectly in
        # `report/loaders.py` and throws in the web UI. One consumer silently
        # fine and another broken is this repository's signature fault, and a
        # measurement framework should not have a path that produces it.
        #
        # Trimming the window newly created that path: before it, every client
        # completed at least one query, so the sample set was never empty.
        # Now a configuration whose queries outlast the window keeps nothing.
        #
        # Nulled HERE and not in `summarize`, which predates this window and
        # is also what the build, filtered and churn workloads reach through:
        # those loops run to completion, so an empty sample set there means
        # nothing was asked of the engine at all, and widening the change
        # widens the blast radius for no gain.
        #
        # Derived from `as_dict`'s own keys rather than from a list of field
        # names kept beside it, so a field added to `LatencyStats` is nulled
        # here by default instead of being the one that still reports NaN.
        out = {key: None for key in out}
        # The count is the one figure an empty window does know, and 0 is a
        # measurement rather than a missing one.
        out["queries_executed"] = 0
    # Zero completions over the window, which is what was observed. Unlike a
    # null it plots, and unlike a NaN it serialises.
    out["qps"] = len(kept) / wall_seconds if wall_seconds > 0 else 0.0
    out.update(discarded)
    return out
