"""Latency accumulation, percentile reporting, and what a window achieved.

Percentiles are computed from every recorded sample rather than from a running
approximation. At the query counts used here (10k test vectors per run) keeping
all samples costs a few megabytes and removes a whole class of "is the p99
real?" doubt. `nearest-rank` is used rather than interpolation so a reported p99
is always a latency that actually occurred.

Keeping every span is also what makes `WindowConcurrency` possible: how many
queries a multi-client window actually had outstanding, and how many it had in
service, are both sums over the spans this module already holds. They are here
rather than in `harness/workloads/clients.py`, where the pool that reports them
lives, for two reasons -- `span_disposition`'s definition of "inside the
window" has to be the SAME one the percentiles and the throughput use, and
`clients.py` was 846 lines against this repository's 800-line ceiling with the
policy inlined. `SERVED_CONCURRENCY_FLOOR` and `scaling_status` follow the
quantity they judge for the same reason `churn_readers.py` exists.
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


class WindowConcurrency(NamedTuple):
    """How many queries a measured window actually had in it, two ways.

    Both are derived from the spans the window already keeps, by the SAME
    `span_disposition` call that decides which latencies and which returned
    ids count -- so they describe exactly the queries the QPS beside them
    counts. Neither needs a distributional assumption and neither needs a
    field the record does not already carry.

    `outstanding` is Little's law computed exactly: the integral of the
    occupancy function over the window, divided by the window. Sum the spans
    and you have the area; there is no estimator in it. It answers "how many
    queries did the clients have in flight on average".

    `served` is Little's law with the MEDIAN span rather than the mean. It
    answers a different question -- "how many queries were in SERVICE on
    average" -- and the two diverge by exactly the time a span spends waiting
    for Python's interpreter lock instead of waiting for the engine.

    **Which one a guard must use was settled by measurement, not by
    preference, and the answer is not the exact one.** Five synthetic drivers
    of identical nominal query cost (0.325 ms, the smoke run's observed p50)
    were run through the real `ClientPool` at 1 and 4 clients
    (CPython 3.12.3, 3 windows of 2 s per point, median window reported;
    `tests/test_g1_client_concurrency.py` rebuilds four of the five regimes
    and asserts the verdicts rather than the decimals):

        driver                     efficiency   outstanding/n   served/n
        sleep (releases the lock)        1.00           0.99        0.96
        saturated server                 0.28           1.00        0.98
        atomic C call holding the lock   0.27           0.99        0.22
        Python spin holding the lock     0.25           0.98        0.25
        half released, half held         0.52           1.00        0.96

    `outstanding/n` is 0.98-1.00 on all five, across efficiencies from 0.25 to
    1.00. It is BLIND, and the reason is mechanical rather than a matter of
    noise: a thread that is waiting for the lock is still inside
    `driver.query`, because the span's start instant is read in Python and the
    interpreter can drop the lock at any bytecode boundary after it. The wait
    lands inside the span, the stretched spans tile the wall, and the sum
    approaches the client count however little work was served. The signature
    in the same rows is a mean far above the median -- 1.025 ms against
    0.227 ms for the atomic driver -- and a p99 of 16.7 ms where the p50 did
    not move.

    `served/n` separates the two regimes by a factor of four: 0.22-0.25 where
    the pool served about one query, 0.96-0.98 where it served four. A
    genuinely saturated server raises the WHOLE distribution, median
    included -- 0.389 ms to 1.377 ms at four clients here -- so its `served`
    stays at the client count and it is not flagged. That is the distinction
    a guard has to get right.

    `None`, never 0.0, when the window counted no query: zero is the claim
    "the clients had nothing in flight", which cannot be true of a window
    that executed queries, and it is the value a guard would read as
    serialization on a window nobody measured.
    """

    outstanding: Optional[float]
    served: Optional[float]


def window_concurrency(collectors: Sequence["LatencyCollector"],
                       wall_seconds: float,
                       cutoff: Optional[float] = None,
                       window_start: Optional[float] = None
                       ) -> WindowConcurrency:
    """The concurrency a window achieved, from the spans it already kept.

    Beside `merge_collectors` and not inside it, because `merge_collectors`
    returns `Record` columns and these two are not columns -- they belong in
    the concurrency record's `extra` the way the client-CPU figures do. In
    this module rather than in the workload, because the kept-span rule has to
    be the SAME rule: one `span_disposition` call decides which spans the QPS
    counts, which latencies the percentiles cover, which ids recall scores and
    now which spans these two sum. Four definitions of "inside the window" is
    how a published recall comes to describe a different query set from the
    throughput beside it, which this module's own history records.

    The median is `percentile(sorted, 0.50)` -- the same nearest-rank function
    and the same definition `latency_p50_ms` is reported with -- so a reader
    can check `served` against the record with one multiplication:
    `queries_executed x latency_p50_ms / 1000 / window`. An interpolating
    median here would make that arithmetic disagree with the published p50 and
    the check would read as a bug in the guard.
    """
    kept: List[float] = []
    for c in collectors:
        for sample in c._samples:
            if span_disposition(sample.started, sample.finished,
                                window_start, cutoff) is None:
                kept.append(sample.seconds)
    if not kept or wall_seconds <= 0:
        return WindowConcurrency(None, None)
    kept.sort()
    return WindowConcurrency(
        outstanding=sum(kept) / wall_seconds,
        served=len(kept) * percentile(kept, 0.50) / wall_seconds,
    )


#: Share of the client count that must have been IN SERVICE before a scaling
#: efficiency is reportable as a property of the engine. Below it the harness
#: records a status instead and the report refuses to render the figure.
#: `scaling_status` below says what the statuses mean; `WindowConcurrency`
#: above says why `served` and not the exact `outstanding` is compared.
#:
#: 0.5 IS MEASURED, and the measurement is the GAP it sits in. Five synthetic
#: drivers of identical 0.325 ms query cost through the real `ClientPool` at 1
#: and 4 clients, three 2 s windows each, median window
#: (CPython 3.12.3; `docs/05-methodology.md` §7 has the full table and
#: `tests/test_g1_client_concurrency.py` the reproducible form):
#:
#:     driver                     efficiency   served/clients   flagged?
#:     releases the lock                1.00             0.96   no
#:     saturated server                 0.28             0.98   no
#:     holds the lock (two forms)  0.25-0.27        0.22-0.25   YES
#:     half released, half held         0.52             0.96   no
#:     any driver at 1 client           1.00        0.94-1.00   no
#:
#: Two populations, 0.22-0.25 and 0.94-1.00, with nothing measured between:
#: 0.5 is a factor of two from each side and no plausible move of it changes
#: a verdict. The midpoint of a measured gap is the whole justification.
#:
#: **The saturated-server row is the one this has to get right.** That driver
#: IS the bottleneck -- latency rises with the client count, throughput is
#: flat and efficiency falls to 0.28, the shape a reader would call an engine
#: ceiling, and it is one. Its median span rose with it, 0.389 ms to
#: 1.377 ms, so its `served` stayed at the client count and it is NOT
#: flagged. A real queue raises the whole distribution; a lock stall raises
#: only the tail.
#:
#: **"A REAL QUEUE RAISES THE WHOLE DISTRIBUTION" IS TRUE OF THE FIVE DRIVERS
#: ABOVE AND FALSE IN GENERAL, AND THAT IS WHY `efficiency` WAS ADMITTED TO
#: THE RULE BELOW.** All five have a NARROW service time, so for all five
#: `served / clients` and `p50 / mean` are the same number. They are the same
#: number for a mechanical reason -- `served / outstanding == p50 / mean`
#: identically, both being sums over one kept set, and `outstanding / clients`
#: is 0.84-1.00 on every regime ever measured here -- so `served / clients`
#: below a half IS `p50` below half the mean, which is a statement about
#: LATENCY SKEW and not about how many clients ran. A two-mode service time
#: with miss rate p and penalty s/f drives it under the floor whenever
#: `s/f > (1+p)/p`: 4.3x at p=0.3, 11x at p=0.1, 21x at p=0.05. A graph cache
#: smaller than the graph produces exactly that, and this comparison sizes
#: those caches deliberately (`mhnsw_max_cache_size`, `vidx_hnsw_cache_size`).
#:
#: Measured on this tree through the real `ClientPool`, 4 clients, 2 s
#: windows, better of two by qps, synthetic drivers that SLEEP -- releasing
#: the lock, no shared queue, so the server is explicitly not the bottleneck
#: (CPython 3.12.3, on a box whose other 20 cores were busy throughout; the
#: lock-holding rows are the retired thread pool, which is the only arm that
#: can still produce them, and the whole table is rebuilt by
#: `tests/test_skewed_latency_is_not_serialization.py`):
#:
#:     regime                        efficiency   served/clients   p50/mean
#:     narrow latency (control)           0.998            0.973      0.988
#:     30% of queries at 20x cost         0.984            0.163      0.164
#:     10% at 50x                         0.959            0.191      0.193
#:      5% at 50x                         1.000            0.327      0.331
#:     saturated server, narrow           0.277            0.989      0.994
#:     saturated AND 30% at 20x           0.233            0.155      0.156
#:     clients took turns (threads)       0.248            0.246      0.252
#:
#: Rows two to four SCALED LINEARLY and were flagged `concurrency_not
#: achieved` on `served / clients` alone -- the published reading of which is
#: "the clients serialized, or the engine answers one query at a time", and
#: neither was true of them. The column that holds is `efficiency`: 0.96-1.00
#: where the pool scaled and 0.22-0.28 where it did not, with nothing
#: between them at this window length, and the same 0.5 is again the
#: midpoint of the measured gap -- a
#: factor of 1.9 below the lowest scaling row and 1.8 above the highest
#: non-scaling one. So ONE floor judges both estimators and there is no
#: second constant to choose. A floor at 0.35 would publish a pool that
#: delivered 40% of its clients' throughput, which is the refusal's own
#: definition of not achieving it; a floor at 0.75 would refuse the
#: 10%-at-50x row's 0.959 only on noise, and at 0.6 s windows that row
#: measured as low as 0.789 for want of samples.
SERVED_CONCURRENCY_FLOOR = 0.5

#: The pool served about as many queries as it had clients, so a scaling
#: efficiency measured over it describes the engine.
SCALING_OK = "ok"

#: The pool served materially fewer queries at once than it had clients, so
#: whatever `scaling_efficiency` came out of it is not n-client scaling of
#: this engine. Named for the OBSERVATION and not for a cause, deliberately:
#: see `scaling_status` for the two causes a client's own clock cannot
#: separate.
SCALING_CONCURRENCY_NOT_ACHIEVED = "concurrency_not_achieved"

#: Both statuses, for a report or a test that wants to assert it has handled
#: every one. A tuple rather than a set so the order is the order they are
#: documented in.
SCALING_STATUSES = (SCALING_OK, SCALING_CONCURRENCY_NOT_ACHIEVED)


def served_fraction(served: Optional[float],
                    clients: int) -> Optional[float]:
    """`served` as a share of the client count.

    The quantity the floor is compared against, as one expression in one
    place, so `scaling_status` and anything printing the evidence cannot
    divide by two different denominators. `None` propagates: a window that
    counted no query has no fraction, and a 0.0 here reads as total
    serialization to every consumer.
    """
    if served is None or clients <= 0:
        return None
    return served / clients


def throughput_fraction(efficiency: Optional[float]) -> Optional[float]:
    """`scaling_efficiency` read as the share of the clients IN SERVICE.

    THE SAME QUANTITY `served_fraction` ESTIMATES, which is why it can be
    compared against the same floor, and the identity is not a coincidence:
    at one client a span has no queue and no lock in it, so the one-client
    throughput IS the reciprocal of the per-query service cost. Little's law
    then gives the queries in service at n clients as
    `qps_n x service_cost == qps_n / qps_1`, and dividing by n is exactly
    `scaling_efficiency`. One estimator uses the n-client MEDIAN SPAN for the
    service cost and the other the ONE-CLIENT MEAN, and that is the whole
    difference between them.

    Which makes their failure modes disjoint, which is the point.
    `served_fraction` is biased DOWN by latency skew, because a median span
    is not a mean service cost when the distribution has two modes; this one
    is not, because a throughput is a count and a count has no quantile in
    it. This one is instead biased down by a SERVER-SIDE queue -- the
    saturated row's 0.281, where the engine really did answer about one query
    at a time -- and `served_fraction` is not, because a queue raises the
    median along with everything else.

    A pass-through, not arithmetic, and deliberately so: the name is the
    argument, and a reader of `scaling_status` should not have to work out
    that the efficiency beside it is a concurrency fraction. `None`
    propagates, for the reason it propagates everywhere here: a point with no
    one-client baseline to divide by has no efficiency, and 0.0 there would
    read as total serialization.
    """
    return efficiency


def achieved_fraction(served: Optional[float], clients: int,
                      efficiency: Optional[float] = None) -> Optional[float]:
    """The best estimate of the share of `clients` the pool had in service.

    THE LARGER of the two estimators above, and the choice of `max` is the
    whole content of this function. Each is exact in the regime the other is
    biased in -- see `throughput_fraction` -- and both are biased in ONE
    direction only, downwards: skew cannot inflate a median above a mean, and
    a server-side queue cannot make a pool deliver throughput it did not
    deliver. Two lower bounds on one quantity, so the larger is the better
    bound and the smaller carries no information the larger lacks.

    `None` when `served_fraction` is None, and NOT the efficiency on its own:
    a window that counted no query has nothing in it to have been in
    service, and "nobody could look" has to stay distinguishable from a
    verdict -- see `scaling_status`. An efficiency without a window is a
    ratio of two throughputs one of which was never measured.
    """
    fraction = served_fraction(served, clients)
    if fraction is None:
        return None
    throughput = throughput_fraction(efficiency)
    if throughput is None:
        return fraction
    return max(fraction, throughput)


def scaling_status(served: Optional[float],
                   clients: int,
                   efficiency: Optional[float] = None) -> Optional[str]:
    """Whether a scaling efficiency over this window is reportable.

    Here rather than in `harness/workloads/clients.py`, where it was written
    and where `PoolWindow.scaling_status` still exposes it: this module owns
    the quantity, the floor belongs beside the quantity it judges, and
    `clients.py` was 846 lines against this repository's 800-line ceiling
    with the policy inlined. `churn_readers.py` exists for the same reason.

    `efficiency` is OPTIONAL AND THE DEFAULT IS THE OLD RULE, which is a
    migration state and not a design: the two callers that have an efficiency
    to pass are in `harness/workloads/`, and a caller that does not pass one
    gets a status computed from the median span alone and therefore gets the
    false positive the floor's docstring measures. `PoolWindow.scaling_status`
    is one such caller -- the window does not know the one-client baseline
    that `concurrency.run` divides by -- and the churn readers are the other,
    and they have no one-client baseline at all. Until those are routed this
    argument changes nothing about any published figure.

    `None` -- not `SCALING_OK` -- when there is nothing to judge: a window
    that counted no query has no spans to sum, and "the pool achieved its
    concurrency" and "nobody could look" are different statements, exactly as
    `PoolWindow.client_bound` distinguishes them for an unreadable CPU
    counter. The report must render the two differently or the absence reads
    as a pass.

    `SCALING_CONCURRENCY_NOT_ACHIEVED` NAMES THE OBSERVATION, NOT A CAUSE,
    deliberately. Two things produce it and a client's own clock cannot tell
    them apart: the CLIENTS serialized on Python's interpreter lock, the case
    the live smoke run exposed, which makes the efficiency a property of the
    harness; or the SERVER serialized, in which case there is a real finding
    -- but it is "this engine serializes queries", not "this engine scales at
    0.25", which is a different claim needing a different presentation. Both
    make the published number wrong, so both are refused and the report
    states both readings. `client_cpu_seconds` separates them in practice,
    roughly one core against roughly none, and travels beside the status in
    the record. Putting the CPU figure INSIDE the condition was rejected for
    the reason `clients.CLIENT_BOUND_UTIL` is a cautionary tale: a CPU term
    reintroduces the blindness this quantity was added to fix.

    **WHICH IS WHY NO THIRD STATUS WAS ADDED, having been considered.** The
    residue of the rule -- a pool that neither scaled nor kept its median at
    its client count -- holds BOTH causes, and they are indistinguishable
    from this window: measured on this tree, clients taking turns on the
    interpreter lock, a saturated server whose service time is also two-moded,
    and a server whose cache misses queue on one shared resource all deliver a
    throughput that does NOT rise with the client count, at an efficiency of
    0.22-0.25 and a `served / clients` of 0.05-0.25 -- and the median span
    failed to grow with the client count in two of the three, so even that
    does not sort them. A value
    meaning "withheld, cannot attribute" would therefore be a synonym for
    this one, which already means exactly that, and it would have to be
    plumbed through `report/` to say nothing new.

    **What the `efficiency` term is for, then, is the case where NEITHER
    reading is true**: a pool that scaled linearly and whose latency is
    merely skewed. `SERVED_CONCURRENCY_FLOOR`'s docstring has that
    measurement -- three regimes at efficiency 0.96-1.00 and
    `served / clients` 0.16-0.33, every one of them flagged by the
    median-span rule alone. A pool that delivered n times a one-client
    throughput cannot have been starved, whatever shape its latency has, and
    this is the term that says so.
    """
    fraction = achieved_fraction(served, clients, efficiency)
    if fraction is None:
        return None
    if fraction < SERVED_CONCURRENCY_FLOOR:
        return SCALING_CONCURRENCY_NOT_ACHIEVED
    return SCALING_OK


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
