"""One client pool, shared by the concurrency and churn workloads.

PR 4 §4.3 item 3 asks for this module by name, and the reason is that churn now
needs readers running while the writer works. Every piece of machinery that
takes is already in `concurrency.py` and was argued for there over three
separate corrections, so the choice was to lift it or to write a second
implementation that would start out agreeing and then drift:

* **The barrier that opens the window.** `threading.Barrier(action=...)` reads
  the clock in the last party to arrive, after every party is waiting and
  before any is released, so the window's start instant provably precedes
  every measured query. Reading it in the parent thread after `wait()` returns
  is the obvious place and is wrong: with eight clients on one GIL, six of them
  issued their first query first, those queries looked as though they began
  before the window, and dropping them pinned `extra.recall_queries` at 198 of
  a 200-query pool however long the window ran.

* **PR 1's window trimming.** A fixed-duration window ends by setting a flag;
  each client then finishes the query it is holding. Counting those put more
  queries in the numerator than the denominator's wall covered, by
  `inflight_discarded / queries_executed` -- measured flat at 2.564% from 1 to
  32 clients, because the client count cancels, and ~0.005% at `main`'s 30 s
  windows. It also reported more queries in flight on average than there were
  clients to hold them, which no client count can do. One
  `span_disposition` call decides which latencies AND which returned ids
  count, so the recall beside a throughput figure is over the queries that
  figure counted.

* **PR 1's CPU accounting.** The clients are Python threads in a container
  pinned to a few CPUs. `client_cpu` divides a cgroup counter by a wall clock
  and a CPU count, and the record keeps all three, because each of the three
  can be wrong while the quotient stays plausibly between 0 and 1.

**What a pool's numbers are NOT evidence about, and this matters more now that
a second workload inherits it.** `CLIENT_BOUND_UTIL` is sufficient and not
necessary, and the gap is structural rather than a question of where the limit
was set: a CPU-utilisation threshold cannot see a GIL ceiling, because the lock
prevents the CPU consumption the threshold looks for. Measured against
`tests/fake_driver.py` on a twenty-CPU box, the concurrency workload's own CPU
flattens at 2.06 cores -- 0.51 of a four-CPU client container at 4, 8, 16 and
32 clients alike -- while aggregate throughput peaks at two clients
(16,614 q/s) and falls to 8,127 by thirty-two. That is the exact shape a reader
would call an engine ceiling and the flag fired on none of those points. So a
churn reader pool's QPS is a measurement of this harness against this engine
under writes, and it is not evidence about where either one saturates.

**And the synthetic figure above overstates a real client's cost, so the
synthetic SHAPE may not reproduce.** The fake's queries are numpy-bound
in-process, whereas a real driver spends most of a query blocked on a socket
with the lock released -- which is why the decision to run the clients as
processes waits on a real dev or smoke run rather than on this measurement.
This paragraph lived at `concurrency.py:118` before PR 4 §4.3 lifted the flag
into this module, which deleted it; it is restored here because two workloads
now inherit the flag and because it is the qualitative form of a measurement
PR 4 went on to make.

That measurement, for the one quantity where the two regimes were compared
directly: a reader pool's aggregate QPS divided by a single-client baseline --
the spurious retention `report/churn_state.py` exists to keep out of the churn
chart -- came out **3.74 to 4.26** over 9 runs with readers whose query
releases the lock (a `latency_s` sleep, the socket-like regime) and **0.89 to
2.62** over 9 runs with numpy-bound readers, which is the regime the 2.06-core
figure above was taken in. Two independent measurements agree on both rows:
a second party measured 3.83-4.24 and 1.06-1.93, each inside the range above.

So "may not reproduce" is not hedging. The socket-like regime puts that
quantity at roughly the reader count and the numpy-bound one puts it near 1,
which are different answers to "do four readers deliver four readers' worth of
throughput" rather than two estimates of one answer. The ranges overlap at
their edges, so no single factor separates them and none is quoted here. The
numpy-bound regime is the pessimistic one for scaling and the optimistic one
for this flag, which is the direction that matters: a flag calibrated there
under-reports what a real client costs.

`_one_window` deliberately stays in `concurrency.py`: it owns the fixed
duration, the recall scoring and the repeats, none of which churn wants.
"""

from __future__ import annotations

import threading
import time
from typing import (Any, Callable, Dict, List, NamedTuple, Optional, Tuple)

import numpy

from ..drivers.base import EngineDriver, IndexSpec
from ..metrics.latency import (INFLIGHT_DISCARDED, PREWINDOW_DISCARDED,
                               LatencyCollector, merge_collectors,
                               span_disposition)
from ..metrics.resources import own_cpu_count, read_own_cpu_usec
from ..progress import Heartbeat

DriverFactory = Callable[[], EngineDriver]

#: Share of the client container's CPUs above which a point is reported as
#: client-bound. Not a measured constant and not claimed to be one: it is the
#: level at which this harness stops being able to say whose ceiling a figure
#: shows. Python threads serializing vectors do not scale smoothly to the last
#: percent of a cpuset, so a point at 0.8 is already one where the clients may
#: be contributing, and the honest report is "this number is not evidence about
#: the engine" rather than a correction factor nobody can justify.
#:
#: SUFFICIENT, NOT NECESSARY. See this module's docstring for the measurement:
#: raising or lowering 0.80 does not help, because the quantity itself is blind
#: to a GIL ceiling -- and for the regime that measurement was taken in, which
#: is the numpy-bound one and not the socket-bound one a real driver runs in.
CLIENT_BOUND_UTIL = 0.80

#: Seconds the PARENT waits at the barrier, and it does **not** bind. Kept at
#: the value it had before the lift so nothing about the window moved, and
#: documented honestly: a client that is still warming up when
#: `CLIENT_BARRIER_TIMEOUT_S` expires breaks the barrier from its own side,
#: which makes this wait raise `BrokenBarrierError` immediately. So warm-up is
#: bounded at roughly `CLIENT_BARRIER_TIMEOUT_S` after the first client
#: arrives, and 300 s is a ceiling that cannot be reached while the client
#: side is the smaller of the two.
#:
#: An earlier revision of this comment called it "the only thing that stops a
#: stuck driver holding the machine". That was the 120 below, unnamed at the
#: time, and naming it is what made the inconsistency visible.
BARRIER_TIMEOUT_S = 300.0

#: Seconds a CLIENT waits at the barrier, which is the one that binds. A
#: client reaching the barrier waits for its siblings; the slowest warm-up
#: therefore has this long from the first arrival, not `BARRIER_TIMEOUT_S`.
#:
#: Named rather than left as the bare `timeout=120` it was before the lift,
#: because the two numbers only make sense read together: the smaller one is
#: the bound, and a future edit that raised this above 300 would silently move
#: the bound to the parent without either comment changing.
CLIENT_BARRIER_TIMEOUT_S = 120.0

#: Seconds a client is given to notice the stop flag and return. One query's
#: worth of slack over any latency this framework has measured; a client still
#: running after it is reported as a failure rather than waited on, because a
#: join with no timeout is how a reader exception becomes a hang.
JOIN_TIMEOUT_S = 120.0


class Answer(NamedTuple):
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


class PoolSpec(NamedTuple):
    """What identifies one client pool, for comparing two of its windows.

    This exists because of the shape of PR 4 §4.3 item 5's headline number:
    `p99 under writes / baseline p99`, a ratio of two separately measured
    percentiles. This repository has published two wrong ratios of exactly
    that shape -- the filter-strategy ratio divided a narrow point from one
    ops unit by a wide point from another, and `charts.churn_retention` pairs
    records from two different maintenance states -- and both read as findings
    rather than as bugs. So a ratio across two windows is computed only when
    the two windows agree on every axis that could make them measurements of
    different things, and this tuple is that list.

    `unit` carries the ops unit: (M, build mode, storage engine, resource
    pass). It is `()` for the concurrency workload, which takes no such ratio.
    It is deliberately a tuple the CALLER builds rather than a field list this
    module knows, because `harness/` must not import `report/` -- see
    `churn.OPS_UNIT_KEYS`, which names the axes and is pinned against
    `report.generate.OPS_UNIT_FIELDS` by a test.
    """

    clients: int
    #: How many distinct queries the clients drew from. Two windows over pools
    #: of 10,000 and of 15 queries are measurements of two different things:
    #: every graph cache in this comparison is sensitive to the size of the
    #: hot set, which is the whole reason `ops.concurrency_queries` exists.
    queries: int
    k: int
    ef_search: int
    unit: Tuple[Any, ...] = ()


class PoolWindow(NamedTuple):
    """One measured window of one pool.

    `merged` holds nothing but `Record` columns, so `Record(**merged)` stays
    valid; the discard counts and the throughput are carried separately for
    that reason rather than left inside it.
    """

    qps: float
    merged: Dict[str, float]
    inflight_discarded: int
    prewindow_discarded: int
    #: The client container's own CPU over this window, the size of the cpuset
    #: it was measured against, and their ratio. All three, never the ratio
    #: alone: the ratio is a cgroup counter divided by a wall clock and a CPU
    #: count, each of which can be wrong while the quotient stays plausibly
    #: between 0 and 1.
    client_cpu_seconds: Optional[float]
    client_cpus: int
    client_cpu_util: Optional[float]
    #: The wall the throughput above is over, kept because a reader cannot
    #: otherwise tell a busy window from a short one.
    wall_s: float
    #: Which pool measured this. Two windows with different specs are not two
    #: samples of one quantity -- see `PoolSpec`.
    spec: PoolSpec
    #: Query position -> that query's answer, trimmed to this window. Empty
    #: when the pool was built with `collect_results=False`, which is what the
    #: churn readers do: §4.3 item 6 says they compute no recall, because the
    #: corpus is mid-change and no ground truth describes it.
    results: Dict[int, Answer]

    @property
    def p99_ms(self) -> Optional[float]:
        return self.merged.get("latency_p99_ms")

    @property
    def queries_executed(self) -> int:
        return int(self.merged.get("queries_executed") or 0)

    @property
    def client_bound(self) -> Optional[bool]:
        """Whether the clients may have been part of the ceiling.

        `None` -- not False -- when there was no counter to read, because "the
        clients were not the limit" and "nobody looked" are different
        statements and only one of them is a measurement.
        """
        if self.client_cpu_util is None:
            return None
        return self.client_cpu_util > CLIENT_BOUND_UTIL


class Instruments(NamedTuple):
    """The clocks, counters and primitives a pool measures itself with.

    Injectable, and the reason is not abstraction for its own sake: five
    existing concurrency tests calibrate a window by replacing exactly these
    names **on `harness.workloads.concurrency`** -- a CPU counter that
    advances by one CPU-second per CPU per wall second so the expected
    utilisation is 1.0 by construction, a `merge_collectors` that runs twice
    to show what the untrimmed merge counted, and a `threading` whose `Event`
    takes a visible amount of time to `set()` so the gap between the window's
    end and the flag is measurable.

    Rebinding a name on the concurrency module does **not** reach a function
    this module captured at import time -- that is how three mutants got a
    fully green suite in an earlier task here -- so the pool cannot simply
    import them. `concurrency._one_window` therefore passes its own
    module-level names in, which keeps the concurrency module the one place a
    test replaces an instrument, and keeps PR 4's acceptance ("concurrency
    tests unchanged") meaningful rather than something the lift quietly broke.

    `threading` needs only `Event`, `Barrier` and `BrokenBarrierError`: the
    thread class itself is resolved in this module and a test that wanted to
    replace it would be replacing the client, not an instrument.

    **`time` is deliberately NOT in this bundle, and the asymmetry is worth
    knowing about.** Every window instant -- the barrier action's
    `perf_counter`, the stop instant, each client's per-query span -- is read
    through THIS module's `time`. `concurrency.time` is still live because
    `_one_window` sleeps on it, so a test that patched `concurrency.time`
    would reach the sleep and not the measurement, where before the lift it
    reached both. Nobody does that today. Two things did get safer in the same
    move, in the direction that fails loudly: `span_disposition` and
    `LatencyCollector` are no longer attributes of `concurrency` at all, so a
    `monkeypatch.setattr` on either now raises `AttributeError` instead of
    passing green against a patch that reached nothing.
    """

    threading: Any = threading
    merge: Callable[..., Dict[str, float]] = merge_collectors
    cpu_usec: Callable[[], Optional[int]] = read_own_cpu_usec
    cpu_count: Callable[[], int] = own_cpu_count


DEFAULT_INSTRUMENTS = Instruments()


class PoolError(RuntimeError):
    """A client failed. Raised rather than returned, and never swallowed.

    PR 4 §4.4 asks that a reader exception fail the record instead of hanging,
    and the hang is the specific hazard: a client that dies before the barrier
    leaves every other party waiting on it. `Client.run` releases the barrier
    in a `finally` for that reason, `ClientPool.start` aborts the barrier on
    `BrokenBarrierError`, and `ClientPool.stop` joins with a timeout and then
    raises this. A pool that measured nothing raises too: a QPS of 0 from a
    pool whose clients all died reads as "the engine was slow", which is the
    one thing it is not evidence of.
    """


class Client(threading.Thread):
    """One benchmark client: own connection, own query slice, own latencies."""

    def __init__(self, driver_factory: DriverFactory, queries: numpy.ndarray,
                 query_offsets: numpy.ndarray,
                 k: int, ef_search: int, index: IndexSpec, warmup: int,
                 ready: threading.Barrier, stop: threading.Event,
                 collect_results: bool = True):
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
        # False for the churn readers. §4.3 item 6: they compute no recall,
        # because the corpus is mid-change and no ground truth describes it,
        # so retaining a result list per query would allocate inside the
        # window for nothing a reader can use.
        self._collect_results = collect_results
        self.collector = LatencyCollector(warmup=0)
        #: Query position -> that query's answer and span, first pass only.
        self.results: Dict[int, Answer] = {}
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
            # Always release the barrier, or a failing client deadlocks the
            # run. `CLIENT_BARRIER_TIMEOUT_S` and not a bare literal: it is
            # smaller than the parent's `BARRIER_TIMEOUT_S` and is therefore
            # what actually bounds a warm-up, which is a fact about the pair
            # rather than about either number.
            try:
                self._ready.wait(timeout=CLIENT_BARRIER_TIMEOUT_S)
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
                if self._collect_results and i < n:
                    self.results[int(self._offsets[i])] = Answer(
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


def shard_queries(queries: numpy.ndarray,
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


def client_cpu(cpu_start: Optional[int], cpu_end: Optional[int],
               wall_s: float,
               cpus: int) -> Tuple[Optional[float], Optional[float]]:
    """The clients' own CPU over a window, in seconds and as a utilisation.

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


class ClientPool:
    """A pool of query clients and one measured window over it.

    Two calls, in order: `start()` runs the warm-up and opens the window at
    the barrier; `stop()` closes it and returns a `PoolWindow`. The caller
    decides what happens in between -- `concurrency._one_window` sleeps for a
    fixed duration, `churn` waits for the writer -- and that is the only
    difference between the two workloads' use of this machinery.

    A pool measures ONE window. Threads cannot be restarted, and the churn
    workload needs its reader windows either side of a delete step it must not
    be reading through (§4.3 item 4 starts the readers after the delete), so
    it builds a second pool from the same `PoolSpec` instead. `PoolSpec`
    equality is what makes "the same pool" checkable rather than asserted --
    see `PoolSpec` for the two published-wrong ratios that motivated it.
    """

    def __init__(self, driver_factory: DriverFactory, queries: numpy.ndarray,
                 clients: int, k: int, ef_search: int, index: IndexSpec,
                 warmup: int, collect_results: bool = True,
                 unit: Tuple[Any, ...] = (),
                 instruments: Instruments = DEFAULT_INSTRUMENTS):
        if clients <= 0:
            raise ValueError(f"a pool needs at least one client, got {clients}")
        self.spec = PoolSpec(clients=clients, queries=len(queries), k=k,
                             ef_search=ef_search, unit=tuple(unit))
        self._instruments = instruments
        self._stop = instruments.threading.Event()
        # The instant the window opens, read by the barrier itself -- see this
        # module's docstring for what reading it in the parent thread did to
        # `extra.recall_queries`.
        #
        # The CPU counter is read in the same barrier action, immediately
        # before the wall instant, so the two readings bracket one interval
        # instead of two. Reading it in the parent after `wait()` returns
        # would put the clients' first queries' CPU outside a window whose
        # denominator covers them; reading it before the barrier would include
        # every client's warm-up, which is hundreds of queries of CPU charged
        # to a window that did not run them.
        #
        # CPU first, then the wall: the file read costs microseconds, and
        # spending them before the wall starts means the window contains no
        # CPU that its start reading has already counted.
        self._opened: List[Tuple[Optional[int], float]] = []
        self._ready = instruments.threading.Barrier(
            clients + 1, action=self._open_window)
        self._workers = [
            Client(driver_factory, shard, offsets, k, ef_search, index,
                   warmup, self._ready, self._stop,
                   collect_results=collect_results)
            for shard, offsets in shard_queries(queries, clients)
        ]
        self._started = False

    def _open_window(self) -> None:
        self._opened.append(
            (self._instruments.cpu_usec(), time.perf_counter()))

    def start(self, label: str, prefix: str) -> None:
        """Run the warm-up and open the measured window at the barrier.

        `label` and `prefix` are the heartbeat's, because warm-up is unbounded
        from the operator's point of view: every client runs its warm-up
        queries before the barrier releases, and on a slow configuration that
        is minutes of silence before the window even starts.
        """
        for worker in self._workers:
            worker.start()
        self._started = True
        try:
            with Heartbeat(label, prefix=prefix):
                self._ready.wait(timeout=BARRIER_TIMEOUT_S)
        except self._instruments.threading.BrokenBarrierError:
            self._ready.abort()
            self._stop.set()
            raise PoolError(
                f"a client failed during warm-up: "
                f"{self._first_error() or 'barrier timed out'}")

    def stop(self) -> PoolWindow:
        """Close the window and return it, or raise if a client failed."""
        if not self._started or not self._opened:
            raise PoolError("the window was never opened")
        cpu_start, wall_start = self._opened[0]
        # The stop instant, read BEFORE the flag is set and used as both the
        # denominator's end and the numerator's cutoff. Those were different
        # instants until this was recorded: the wall ended here, while every
        # client went on to finish the query it was holding and record it, so
        # QPS counted up to one query per client that its own wall did not
        # cover.
        stop_at = time.perf_counter()
        # And the CPU counter immediately after, for the same reason in the
        # same direction: read after `stop.set()`, it would include whatever
        # the clients burned finishing the query each was holding -- work the
        # wall above deliberately excludes.
        cpu_end = self._instruments.cpu_usec()
        self._stop.set()
        wall = stop_at - wall_start
        cpus = self._instruments.cpu_count()
        cpu_seconds, cpu_util = client_cpu(cpu_start, cpu_end, wall, cpus)

        for worker in self._workers:
            worker.join(timeout=JOIN_TIMEOUT_S)

        failures = [w.error for w in self._workers if w.error is not None]
        if failures:
            raise PoolError(
                f"{len(failures)} of {self.spec.clients} clients failed; "
                f"first error: {failures[0]}")
        still_running = [w for w in self._workers if w.is_alive()]
        if still_running:
            raise PoolError(
                f"{len(still_running)} of {self.spec.clients} clients did not "
                f"return within {JOIN_TIMEOUT_S:.0f}s of the window closing")

        merged = self._instruments.merge(
            [w.collector for w in self._workers], wall,
            cutoff=stop_at, window_start=wall_start)

        # One dict across every client. The shards are disjoint and the
        # offsets are absolute positions, so no two clients can write the same
        # key and the merge cannot lose a result.
        #
        # Trimmed by the same predicate and the same window as the latencies,
        # so a recall reported beside this window's QPS is over the queries
        # that QPS counted. Letting the two diverge puts `queries_executed`
        # below `extra.recall_queries` for any window inside its first pass,
        # and `extra.query_reuse` -- executed over distinct -- below 1.0 for a
        # window in which nothing was repeated.
        results: Dict[int, Answer] = {}
        for worker in self._workers:
            for offset, answer in worker.results.items():
                if span_disposition(answer.started, answer.finished,
                                    wall_start, stop_at) is None:
                    results[offset] = answer

        return PoolWindow(
            qps=merged.pop("qps"),
            inflight_discarded=merged.pop(INFLIGHT_DISCARDED),
            prewindow_discarded=merged.pop(PREWINDOW_DISCARDED),
            merged=merged,
            client_cpu_seconds=cpu_seconds,
            client_cpus=cpus,
            client_cpu_util=cpu_util,
            wall_s=wall,
            spec=self.spec,
            results=results,
        )

    def _first_error(self) -> Optional[BaseException]:
        for worker in self._workers:
            if worker.error is not None:
                return worker.error
        return None


def window_ratio(numerator: PoolWindow, denominator: PoolWindow,
                 field: str) -> Optional[float]:
    """One field's ratio across two windows of the SAME pool.

    The whole value of this function is the refusal. PR 4 §4.3 item 5's
    headline number is `p99 under writes / baseline p99`, and a ratio of two
    separately measured quantities is the shape that has produced two
    published-wrong results in this repository: the filter-strategy ratio
    divided a narrow point from one ops unit by a wide point from another, and
    `charts.churn_retention` pairs records from two different maintenance
    states. Both looked like findings. So the two windows must agree on every
    axis in `PoolSpec` -- client count, query pool, k, ef_search and the ops
    unit -- and a caller that crosses any one of them gets a `PoolError`
    naming the field rather than a plausible number.

    `None` when either window has no value for `field`, which is what a window
    that completed no query records. A ratio over a missing percentile would
    be a `TypeError` at best and a 0.0 published as "no slowdown" at worst.
    """
    if numerator.spec != denominator.spec:
        differing = [name for name, a, b
                     in zip(PoolSpec._fields, numerator.spec, denominator.spec)
                     if a != b]
        raise PoolError(
            f"refusing to divide two windows measured by different pools: "
            f"they differ on {', '.join(differing)} "
            f"({numerator.spec} vs {denominator.spec}). A ratio across two "
            f"pools is not a measurement of either.")
    top = numerator.merged.get(field)
    bottom = denominator.merged.get(field)
    if not top or not bottom:
        return None
    return top / bottom


__all__ = [
    "Answer", "BARRIER_TIMEOUT_S", "CLIENT_BARRIER_TIMEOUT_S",
    "CLIENT_BOUND_UTIL", "Client", "ClientPool", "DEFAULT_INSTRUMENTS",
    "DriverFactory", "Instruments", "JOIN_TIMEOUT_S", "PoolError", "PoolSpec",
    "PoolWindow", "client_cpu", "shard_queries", "window_ratio",
]
