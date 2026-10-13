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

Two things this workload records that are easy to leave out, and both of which
made the chart above misleading until they were added:

1. **Recall.** The ids every query returned used to be discarded, so a
   concurrency point had QPS and latency but no quality figure beside them.
   Two engines at `ef_search=100` can sit at recall 0.98 and 0.80 — that is
   the normal spread of these implementations at a shared ef, not a pathology
   — and a throughput chart with no recall column presents them as comparable
   when one is simply doing less work per query.

2. **The size of the query pool.** The pool used to be whatever
   `ops.max_queries` was, a key that exists to cap the cost of the filtered
   and churn workloads (500 in `main.yml`). Each client cycles its own shard
   for the whole window, so at 32 clients that is about fifteen queries per
   client repeated for thirty seconds, and every graph cache in the
   comparison sees a hot set small enough to hold entirely. `ops.
   concurrency_queries` separates the two, defaulting to the whole test set.

3. **How busy the clients themselves were.** The clients are Python threads
   in a container pinned to a few CPUs, and they serialize vectors and hold
   the GIL between socket calls. For a fast engine at 16-32 clients the
   client container can run out of CPU before the server does, and a
   throughput curve that flattens for that reason is indistinguishable from
   one that flattens because the engine stopped scaling -- except that the
   first is a property of this harness and the second is the finding the
   chart exists to report. Every window therefore records the client
   container's own cgroup CPU over the window, as `extra.client_cpu_seconds`,
   `extra.client_cpus`, `extra.client_cpu_util` and the derived
   `extra.client_bound`. The report draws a bound point hollow and the run
   says so on stderr while it happens. See `CLIENT_BOUND_UTIL` for what that
   flag can and cannot see -- it catches a saturated cpuset and not a
   saturated interpreter lock, and the difference has been measured.
"""

from __future__ import annotations

import sys
import threading
import time
from typing import (Callable, Dict, List, NamedTuple, Optional, Sequence,
                    Tuple)

import numpy

from ..datasets import Dataset, cached_ground_truth, recall_at_k
from ..drivers.base import EngineDriver, IndexSpec
from ..metrics.latency import (INFLIGHT_DISCARDED, PREWINDOW_DISCARDED,
                               LatencyCollector, merge_collectors,
                               span_disposition)
from ..metrics.records import PHASE_CONCURRENCY, Record
from ..metrics.resources import own_cpu_count, read_own_cpu_usec
from ..progress import Heartbeat
from .context import RunContext

DriverFactory = Callable[[], EngineDriver]

#: `max_queries` sentinel meaning "every query the dataset ships". The pool is
#: a property of the dataset, not of the cap the filtered and churn workloads
#: need, so the default asks for all of it rather than a number someone has to
#: remember to raise.
WHOLE_TEST_SET = 0

#: How far two windows' recall may differ before the run says so on stderr.
#: Recall is a property of the index and the ef, not of the window, so two
#: windows of the same configuration measuring different recall means the
#: offsets-to-truth mapping is wrong or the index moved underneath the
#: measurement. Either way the QPS beside it is not what it claims to be.
RECALL_SPREAD_LIMIT = 0.005

#: Share of the client container's CPUs above which the point is reported as
#: client-bound. Not a measured constant and not claimed to be one: it is the
#: level at which this harness stops being able to say whose ceiling the curve
#: shows. Python threads serializing vectors do not scale smoothly to the last
#: percent of a cpuset, so a point at 0.8 is already one where the client may
#: be contributing to the flattening, and the honest report is "this number is
#: not evidence about the engine" rather than a correction factor nobody can
#: justify.
#:
#: SUFFICIENT, NOT NECESSARY, and the gap is structural rather than a question
#: of where the limit was set. **A CPU-utilisation threshold cannot see a GIL
#: ceiling, because the GIL prevents the CPU consumption the threshold looks
#: for.** A thread waiting its turn to hold the lock is not running, so it
#: costs nothing this counter can measure: the harness stops scaling while its
#: utilisation stays flat, and no ratio of CPU-seconds to CPUs can distinguish
#: that from an engine that stopped scaling. Raising or lowering 0.80 does not
#: help, because the quantity itself is blind to the case.
#:
#: The signature is throughput falling with more clients while the utilisation
#: does not move, and it is measured rather than supposed. Run against
#: `tests/fake_driver.py` on a twenty-CPU box, this workload's own CPU flattens
#: at 2.06 cores from four clients upwards -- 0.51 of a four-CPU client
#: container at 4, 8, 16 and 32 clients alike -- while aggregate throughput
#: peaks at two clients (16,614 q/s) and falls to 8,127 by thirty-two. That is
#: the exact shape a reader would call an engine ceiling, and this flag would
#: not have fired on any of those points.
#:
#: So a marked point is evidence that the harness was part of the limit, and an
#: unmarked one is not evidence that the engine was. The fake's queries are
#: numpy-bound in-process, whereas a real driver spends most of a query blocked
#: on a socket with the lock released, so the figure above overstates a real
#: client's cost and the synthetic shape may not reproduce -- which is why the
#: decision to run the clients as processes waits on a real dev or smoke run
#: rather than on this measurement.
CLIENT_BOUND_UTIL = 0.80


class _Answer(NamedTuple):
    """One query's returned ids, with the span that produced them.

    The ids and the span travel together because the window is trimmed to the
    queries that finished inside it and recall must be scored over exactly the
    queries that were counted. Keeping the span beside the ids means the same
    `span_disposition` call decides both, rather than the latency being
    trimmed by one rule and the result set by another.
    """

    ids: List[int]
    started: float
    finished: float


class _Window(NamedTuple):
    """One measured window's results.

    A named tuple rather than a bare tuple because `run()` sorts these by
    throughput, takes the median and then reads four other fields off it: with
    positional access that is `s[0]` in the sort, `s[2]` in the spread check
    and a five-way unpack in between, and adding a field means finding all
    three. `merged` deliberately holds nothing but `Record` columns, so
    `Record(**merged)` stays valid -- the discard counts belong in `extra` and
    are therefore carried here instead.
    """

    qps: float
    merged: Dict[str, float]
    recall: Optional[float]
    recall_queries: int
    inflight_discarded: int
    prewindow_discarded: int
    #: The client container's own CPU over this window, the size of the cpuset
    #: it was measured against, and their ratio. All three are carried rather
    #: than the ratio alone: the ratio is a cgroup counter divided by a wall
    #: clock and a CPU count, each of which can be wrong while the quotient
    #: stays plausibly between 0 and 1, so the record keeps the numerator and
    #: the denominator beside it.
    client_cpu_seconds: Optional[float]
    client_cpus: int
    client_cpu_util: Optional[float]


class _Client(threading.Thread):
    """One benchmark client: own connection, own query slice, own latencies."""

    def __init__(self, driver_factory: DriverFactory, queries: numpy.ndarray,
                 query_offsets: numpy.ndarray,
                 k: int, ef_search: int, index: IndexSpec, warmup: int,
                 ready: threading.Barrier, stop: threading.Event):
        super().__init__(daemon=True)
        self._factory = driver_factory
        self._queries = queries
        # Where this client's queries sit in the full query array. Needed
        # because recall scores a result against the ground truth of the query
        # that produced it, and a shard is a strided view: `shards[1]` under
        # four clients is queries 1, 5, 9, ... and its results belong against
        # those rows of the truth, not against rows 0, 1, 2, ...
        self._offsets = query_offsets
        self._k = k
        self._ef_search = ef_search
        self._index = index
        self._warmup = warmup
        self._ready = ready
        self._stop_event = stop
        self.collector = LatencyCollector(warmup=0)
        #: Query position -> that query's answer and span, first pass only.
        self.results: Dict[int, _Answer] = {}
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
                ids = driver.query(q, self._k)
                # Both instants, not the duration: whether this query counts
                # depends on when it ran. The window ends by setting a flag,
                # and this client is holding a query when that happens -- a
                # query that finishes after the wall clock in the denominator
                # stopped. `merge_collectors` drops it on the strength of
                # `finished`, so the instant has to survive to the merge.
                finished = time.perf_counter()
                self.collector.add_span(started, finished)
                # Only the first pass over the shard. The window is a fixed
                # duration, so a fast engine makes thousands of passes and
                # keeping every result would grow without bound and charge the
                # measurement for the allocation; every pass answers the same
                # query, so later ones add no recall information. This bounds
                # the memory to one result list per query however long the
                # window runs.
                #
                # After `collector.add_span`, deliberately: the timer brackets
                # the query alone, so recording the result cannot be charged
                # to the latency the record reports.
                #
                # The span is stored with the ids, not just in the collector.
                # Whether this query's latency counts is decided after the
                # window closes, and its result has to be decided the same way
                # -- recall over a query whose timing was discarded would
                # describe a different set of queries from the QPS beside it.
                if i < n:
                    self.results[int(self._offsets[i])] = _Answer(
                        ids, started, finished)
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
        max_queries: int = WHOLE_TEST_SET, repeats: int = 1,
        indexed_rows: Optional[int] = None) -> None:
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

    `max_queries` caps the query pool; `WHOLE_TEST_SET` asks for all of it,
    which is the default and what `ops.concurrency_queries` resolves to when a
    profile does not set the key.

    `indexed_rows` must be the number of rows actually loaded, exactly as the
    filtered and churn workloads need it: a profile that loaded a subset of the
    training set would otherwise have its recall scored against neighbours
    pointing at rows the engine never received, understating every engine at
    once in a way that reads as a result.
    """
    queries = _query_pool(dataset.test, max_queries)

    # Once per run, not once per window and not once per client count: the
    # answer depends on the dataset, k and how many rows were loaded, none of
    # which move across the sweep, and recomputing it over a subset corpus is
    # minutes of brute force.
    truth = cached_ground_truth(ctx.cache_dir, dataset, ctx.k,
                                indexed_rows=indexed_rows)

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
                warmup, duration_s, truth))

        # The median window, not an average of several: latency percentiles
        # from one measured window mean something, and the mean of several
        # engines' percentiles means nothing. Recall rides inside the sample
        # tuple rather than being selected by a second indexing expression, so
        # "the recall of the median window" is true by construction and cannot
        # drift into being some other window's recall.
        samples.sort(key=lambda s: s.qps)
        window = samples[len(samples) // 2]
        qps, merged, recall = window.qps, window.merged, window.recall
        recall_queries = window.recall_queries
        spread = (((samples[-1].qps - samples[0].qps) / qps * 100.0)
                  if qps else 0.0)

        recall_spread = _check_recall_spread(samples, clients)

        # How many times the average query was repeated inside the window.
        # A cache-sensitive throughput number cannot be judged without it: the
        # same QPS over a pool of 10,000 and a pool of 15 are measurements of
        # two different things.
        executed = merged.get("queries_executed") or 0
        reuse = (executed / recall_queries) if recall_queries else None

        if clients == 1:
            baseline_qps = qps
        efficiency = (qps / (clients * baseline_qps)) if baseline_qps else None

        # The median window's, by the same rule as its recall: the figure
        # beside a throughput number has to come from the window that
        # produced it. `None` -- not False -- when there was no counter to
        # read, because "the clients were not the limit" and "nobody looked"
        # are different statements and only one of them is a measurement.
        client_bound = (None if window.client_cpu_util is None
                        else window.client_cpu_util > CLIENT_BOUND_UTIL)
        if client_bound:
            _warn_client_bound(clients, window.client_cpu_util,
                               window.client_cpus, window.client_cpu_seconds)

        _print_point(common["engine"], clients, ctx.k, qps, recall, merged,
                     efficiency, spread, len(samples))

        ctx.recorder.write(Record(
            **common,
            phase=PHASE_CONCURRENCY,
            ef_search=ef_search,
            clients=clients,
            qps=round(qps, 2),
            # An existing Record field, not a new one: this is the same
            # quantity the recall/QPS path and the filtered and churn
            # workloads already report, so it belongs in the column the report
            # already reads rather than in `extra`.
            recall_at_k=round(recall, 6) if recall is not None else None,
            **merged,
            extra={
                "scaling_efficiency": round(efficiency, 4) if efficiency is not None else None,
                "duration_s": duration_s,
                "warmup_queries_per_client": warmup,
                "repeats": len(samples),
                # Every sample, so a reader can see the spread rather than
                # trust a summary of it.
                "qps_samples": [round(s.qps, 1) for s in samples],
                "qps_spread_pct": round(spread, 1),
                # How many distinct queries the recall above covers. A window
                # too short for one pass over a shard leaves part of the pool
                # unmeasured, and a recall over 200 of 10,000 queries is a
                # different claim from one over all of them.
                "recall_queries": recall_queries,
                "recall_spread": (round(recall_spread, 6)
                                  if recall_spread is not None else None),
                # The pool the clients drew from and how hard they reused it,
                # so a reader can see what the engines' caches were shown.
                "query_pool": len(queries),
                "query_reuse": round(reuse, 2) if reuse is not None else None,
                # Queries the window cut off, and the evidence that both ends
                # of the window are honest. `inflight_discarded` is at most
                # one per client -- the one each was holding when the clock
                # stopped -- and those are the queries the QPS above no longer
                # counts. Their share of it is
                # `inflight_discarded / queries_executed`, about
                # `latency / window` and independent of the client count: a
                # few per cent for a slow engine over a short window, 0.005%
                # for a 1.4 ms query over thirty seconds.
                # `prewindow_discarded` is zero by
                # construction, because the window opens at the barrier
                # release and no measured query can precede that, so a number
                # here means the window's two ends disagree about when it
                # began.
                INFLIGHT_DISCARDED: window.inflight_discarded,
                PREWINDOW_DISCARDED: window.prewindow_discarded,
                # Whose ceiling this point shows. The clients are Python
                # threads in a container pinned to a few CPUs; for a fast
                # engine at 16-32 clients they can run out of CPU first, and
                # a curve that flattens for that reason looks exactly like
                # one that flattens because the engine stopped scaling. The
                # numerator and the denominator are recorded beside the
                # ratio so a reader can see which of the three moved.
                "client_cpu_seconds": (round(window.client_cpu_seconds, 3)
                                       if window.client_cpu_seconds is not None
                                       else None),
                "client_cpus": window.client_cpus,
                "client_cpu_util": (round(window.client_cpu_util, 4)
                                    if window.client_cpu_util is not None
                                    else None),
                "client_bound": client_bound,
            },
        ))


def _query_pool(test: numpy.ndarray, max_queries: int) -> numpy.ndarray:
    """The queries the clients may draw from.

    `WHOLE_TEST_SET` is a sentinel, not a cap: treating it as one leaves the
    pool empty, every client count skipped by the rule below and a silently
    empty concurrency chart. The whole point of `ops.concurrency_queries` is
    that the pool stops being whatever `ops.max_queries` happened to be — a
    key that exists to bound the filtered and churn workloads' cost — so the
    default has to mean "all of them" rather than a number a profile has to
    remember to raise.

    MUST RETURN A PREFIX OF `test`, and this is load-bearing rather than
    incidental. `run()` scores results with `truth[offsets]`, where `offsets`
    are positions into the pool this returns and `truth` is indexed by
    position into the full test set. Those two agree only while position *i*
    of the pool is position *i* of `test`. Narrow the pool any other way — a
    random sample, a stratified draw, every other query — and recall is
    silently computed against the wrong rows: the mapping failure
    `_shard_queries` warns about, one level up and with nothing in the record
    to show it happened. A non-prefix pool therefore has to translate its own
    positions back to test-set positions and hand those to `_shard_queries`.
    """
    if max_queries <= WHOLE_TEST_SET:
        return test
    return test[:max_queries]


def _shard_queries(queries: numpy.ndarray,
                   clients: int) -> List[Tuple[numpy.ndarray, numpy.ndarray]]:
    """Each client's query slice, paired with those queries' positions.

    The slice and the offsets are built in one place and returned together
    because they are two spellings of one fact: `queries[c::clients]` is a
    strided view and `arange(c, n, clients)` is where its elements came from.
    Let them drift apart and recall is scored against the wrong rows of the
    ground truth — client 1 of 4 answering queries 1, 5, 9, ... while its
    results are compared against the neighbours of queries 0, 1, 2, ... That
    produces a plausible-looking low recall rather than an error, which is the
    exact failure mode this repository opens by warning about, and it would be
    read as the engine's ANN quality.

    Strided rather than contiguous blocks, which is the older decision this
    preserves: two clients handed adjacent slices of a sorted query set warm
    each other's caches, and part of what the window measured would be that
    rather than the engine.
    """
    n = len(queries)
    return [(queries[c::clients], numpy.arange(c, n, clients))
            for c in range(clients)]


def _window_recall(results: Dict[int, _Answer], truth: numpy.ndarray,
                   k: int) -> Tuple[Optional[float], int]:
    """Recall over the queries this window actually executed, and how many.

    Only the queries present in `results` are scored. A window shorter than
    one pass over a shard leaves the rest of the pool unexecuted, and padding
    those with empty result sets would report a recall collapse that is the
    window's length rather than the engine's quality.

    The caller has already dropped the answers whose queries fell outside the
    measured window, so this counts the same queries the QPS beside it counts.
    """
    offsets = sorted(results)
    if not offsets:
        return None, 0
    returned = [results[offset].ids for offset in offsets]
    want = truth[numpy.array(offsets, dtype=numpy.int64)]
    return recall_at_k(returned, want, k), len(offsets)


def _warn_nothing_finished(clients: int, duration_s: float,
                           inflight: int) -> None:
    """Say so when a window completed no query at all.

    Reachable only for a configuration whose queries outlast the window, which
    no shipped profile sets up — but the record it writes has no latency
    figures and a QPS of zero, and a gap in a chart discovered a day later is
    the expensive way to learn that `ops.concurrency_duration_s` was shorter
    than one query. Loud, and on stderr, like the recall-spread check: this one
    is a profile fault rather than a framework bug, so it says what to change.
    """
    print(
        f"[concurrency] NOTHING FINISHED: no query completed inside the "
        f"{duration_s}s window at {clients} client(s); {inflight} were still "
        f"running when it closed. The record carries a QPS of 0 and no "
        f"latency figures, because there is nothing to report one over. "
        f"Raise ops.concurrency_duration_s above this configuration's query "
        f"latency — a window shorter than a single query cannot measure "
        f"throughput.",
        file=sys.stderr, flush=True)


def _client_cpu(cpu_start: Optional[int], cpu_end: Optional[int],
                wall_s: float,
                cpus: int) -> Tuple[Optional[float], Optional[float]]:
    """The clients' own CPU over the window, in seconds and as a utilisation.

    Both numbers from one place because the second is the first divided by
    `wall_s * cpus`, and a record that carried only the ratio would leave a
    reader unable to tell a busy window from a short one or from a narrow
    cpuset. Each of those three inputs can be wrong in a way that still
    produces a number between 0 and 1: the counter in the wrong unit (cgroup
    v1 publishes nanoseconds), the wall taken around the wrong interval, or
    the CPU count taken from the host rather than from the cpuset. So all
    three are recorded and the arithmetic is one expression in one function
    that a test can calibrate against a known number of saturated cores.

    `(None, None)`, never `(0.0, 0.0)`, when the counter is unavailable or the
    window is degenerate. Zero is the claim "the clients did nothing", which
    is the one thing that cannot be true of a window that executed queries,
    and it would make the client-bound flag read false on a run where nothing
    was ever measured.
    """
    if cpu_start is None or cpu_end is None or wall_s <= 0 or cpus <= 0:
        return None, None
    # Clamped at zero: the counter is monotonic, so a negative delta means the
    # two readings did not come from the same cgroup and the magnitude is
    # meaningless either way.
    seconds = max(0.0, (cpu_end - cpu_start) / 1e6)
    return seconds, seconds / (wall_s * cpus)


def _warn_client_bound(clients: int, util: float, cpus: int,
                       cpu_seconds: Optional[float]) -> None:
    """Say so when the clients, and not the engine, may be the ceiling.

    On stderr and while the run is happening, like the recall-spread check,
    because the remedy is a different machine or a different client count and
    neither is available to someone reading the report a day later. The point
    is still recorded: it is a real measurement of this harness against this
    engine, and the flag says what it is evidence about.
    """
    print(
        f"[concurrency] CLIENT CPU-BOUND: at {clients} client(s) the client "
        f"container used {util:.0%} of its {cpus} CPU(s)"
        + (f" ({cpu_seconds:.1f} CPU-seconds)" if cpu_seconds is not None
           else "")
        + f", above the {CLIENT_BOUND_UTIL:.0%} limit. The throughput at this "
        f"point is bounded by the clients as well as by the engine, so it is "
        f"not evidence about where the engine saturates. The report draws it "
        f"hollow. Re-measure with more client CPUs if the point matters.",
        file=sys.stderr, flush=True)


def _ms(value: Optional[float]) -> str:
    """A latency for the operator's line, or `n/a` where there is none.

    None arrives from a window that completed no query, which is the one case
    with no percentiles to print; formatting it with `:7.3f` raises.
    """
    return f"{value:7.3f}ms" if value is not None else f"{'n/a':>7}  "


def _print_point(engine: str, clients: int, k: int, qps: float,
                 recall: Optional[float], merged: Dict[str, float],
                 efficiency: Optional[float], qps_spread: float,
                 windows: int) -> None:
    """The operator's one line per data point.

    Separate from `run` because it is presentation, and because every one of
    these figures is optional in a different way: efficiency is undefined
    until the one-client baseline exists, recall is None when no query
    completed, and a spread means nothing from a single window. Formatting
    that beside the measurement is what pushed `run` past the length this
    repository asks of a function.
    """
    eff_text = f"{efficiency:.2f}" if efficiency is not None else "n/a"
    spread_text = f" spread={qps_spread:.1f}%" if windows > 1 else ""
    recall_text = f"{recall:.4f}" if recall is not None else "n/a"
    print(
        f"[concurrency] {engine:>8} clients={clients:>3} "
        f"qps={qps:11,.1f} "
        f"recall@{k}={recall_text} "
        f"p50={_ms(merged['latency_p50_ms'])} "
        f"p95={_ms(merged['latency_p95_ms'])} "
        f"p99={_ms(merged['latency_p99_ms'])} "
        f"efficiency={eff_text}{spread_text}"
    )


def _check_recall_spread(samples: Sequence[_Window],
                         clients: int) -> Optional[float]:
    """How far the windows' recall figures spread, complaining if too far.

    None, not 0.0, for a single window. A lone sample has no spread in the
    same sense that it has no error bar, and 0.0 would claim the stronger
    thing — that two windows were compared and agreed exactly, which is
    precisely the evidence this field exists to carry. `report/render.py`
    already draws that distinction for `qps_spread_pct` by consulting
    `repeats`; this keeps the recall field honest without a reader having to.
    """
    recalls = [s.recall for s in samples if s.recall is not None]
    if len(recalls) < 2:
        return None
    spread = max(recalls) - min(recalls)
    if spread > RECALL_SPREAD_LIMIT:
        # Loud, and on stderr, because this is not measurement noise. Recall
        # is a property of the index and the ef, so two windows of one
        # configuration disagreeing about it means either the query-to-truth
        # mapping is wrong or the index moved underneath the measurement — and
        # in both cases the QPS recorded beside it is not measuring what the
        # record says it is.
        print(
            f"[concurrency] BUG: recall varied by {spread:.4f} across "
            f"{len(recalls)} windows at {clients} client(s), above the "
            f"{RECALL_SPREAD_LIMIT} limit. Recall cannot depend on the "
            f"window, so either the query-to-ground-truth mapping is wrong "
            f"or the index changed under the measurement; the QPS recorded "
            f"beside it is suspect.",
            file=sys.stderr, flush=True)
    return spread


def _one_window(driver_factory: DriverFactory, queries, clients: int, k: int,
                ef_search: int, index: IndexSpec, warmup: int,
                duration_s: float, truth: numpy.ndarray) -> _Window:
    """One warm-up and one measured window.

    Recall is computed here, inside the window that produced the results, so
    the median window's throughput and the recall reported beside it are the
    same window's numbers rather than two independently chosen ones.

    The window is a closed interval `[wall_start, stop_at]` and only the
    queries that both began and ended inside it are counted -- in latency, in
    throughput and in recall. See `merge_collectors` for what counting the
    query each client was still holding did to the QPS.
    """
    shards = _shard_queries(queries, clients)

    # The instant the window opens, read by the barrier itself.
    #
    # NOT `perf_counter()` after `ready.wait()` returns in this thread, which
    # is the obvious place and is wrong. A barrier releases every party at
    # once, so a client can issue its first query before this thread is next
    # scheduled to read the clock — with eight clients on one GIL, six of them
    # did. Those queries then look as though they began before the window, and
    # dropping them takes the first query of each client out of the window's
    # recall coverage: `extra.recall_queries` came in at 198 of a 200-query
    # pool however long the window ran, because no duration can change what
    # happened at its start.
    #
    # `Barrier(action=...)` runs in the last party to arrive, after all of them
    # are waiting and before any is released, so this reading precedes every
    # measured query by construction. It is also the honest start: from the
    # release onwards the clients are querying, so the time this thread spends
    # getting back to `time.sleep` is time the window contains and its queries
    # are counted in.
    # The CPU counter is read in the same barrier action, immediately before
    # the wall instant, so the two readings bracket one interval instead of
    # two. Reading it in this thread after `ready.wait()` returns would put
    # the clients' first queries' CPU outside a window whose denominator
    # covers them -- the same gap that took the first query of each client out
    # of the recall coverage, in the other direction. Reading it before the
    # barrier would include every client's warm-up, which is hundreds of
    # queries of CPU charged to a window that did not run them.
    #
    # CPU first, then the wall: the file read costs microseconds, and spending
    # them before the wall starts means the window contains no CPU that its
    # start reading has already counted.
    opened: List[Tuple[Optional[int], float]] = []

    def _open_window() -> None:
        opened.append((read_own_cpu_usec(), time.perf_counter()))

    ready = threading.Barrier(clients + 1, action=_open_window)
    stop = threading.Event()

    workers = [
        _Client(driver_factory, shard, offsets, k, ef_search, index,
                warmup, ready, stop)
        for shard, offsets in shards
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

    cpu_start, wall_start = opened[0]
    time.sleep(duration_s)
    # The stop instant, read BEFORE the flag is set and used as both the
    # denominator's end and the numerator's cutoff. Those were different
    # instants until this was recorded: the wall ended here, while every
    # client went on to finish the query it was holding and record it, so QPS
    # counted up to one query per client that its own wall did not cover.
    stop_at = time.perf_counter()
    # And the CPU counter immediately after, for the same reason in the same
    # direction: read after `stop.set()`, it would include whatever the
    # clients burned finishing the query each was holding -- work the wall
    # above deliberately excludes.
    cpu_end = read_own_cpu_usec()
    stop.set()
    wall = stop_at - wall_start
    client_cpus = own_cpu_count()
    cpu_seconds, cpu_util = _client_cpu(cpu_start, cpu_end, wall, client_cpus)

    for w in workers:
        w.join(timeout=120)

    failures = [w.error for w in workers if w.error is not None]
    if failures:
        raise RuntimeError(
            f"{len(failures)} of {clients} clients failed; "
            f"first error: {failures[0]}")

    merged = merge_collectors([w.collector for w in workers], wall,
                              cutoff=stop_at, window_start=wall_start)
    if not merged["queries_executed"]:
        _warn_nothing_finished(clients, duration_s, merged[INFLIGHT_DISCARDED])

    # One dict across every client. The shards are disjoint and the offsets
    # are absolute positions, so no two clients can write the same key and the
    # merge cannot lose a result.
    #
    # Trimmed by the same predicate and the same window as the latencies, so
    # the recall reported beside this window's QPS is over the queries that
    # QPS counted. Letting the two diverge puts `queries_executed` below
    # `extra.recall_queries` for any window inside its first pass, and
    # `extra.query_reuse` -- executed over distinct -- below 1.0 for a window
    # in which nothing was repeated.
    results: Dict[int, _Answer] = {}
    for w in workers:
        for offset, answer in w.results.items():
            if span_disposition(answer.started, answer.finished,
                                wall_start, stop_at) is None:
                results[offset] = answer
    recall, recall_queries = _window_recall(results, truth, k)

    return _Window(
        qps=merged.pop("qps"),
        inflight_discarded=merged.pop(INFLIGHT_DISCARDED),
        prewindow_discarded=merged.pop(PREWINDOW_DISCARDED),
        merged=merged,
        recall=recall,
        recall_queries=recall_queries,
        client_cpu_seconds=cpu_seconds,
        client_cpus=client_cpus,
        client_cpu_util=cpu_util,
    )
