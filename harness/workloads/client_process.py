"""One benchmark client, as an OS process, and what crosses the fork.

Split out of `harness/workloads/clients.py`, which owns the pool, the window
and the measurement policy. The reason is this repository's 800-line ceiling
and the reason the ceiling exists: `clients.py` stood at 791 lines before G3,
so there was no substantive change to it that would have fitted. The same
argument produced `metrics/latency.py` and `churn_readers.py`. **It produced
`transfer_deadline.py` too, and the cut was forced the same way**: this file
reached 799 lines of 800 and the next change to it was a three-line
correctness fix that did not fit, so the span-transfer deadline -- the
watchdog, its two flags and the accessor `ClientPool.stop` reads -- went
there. `TransferDeadline` is mixed into `Client` precisely so none of the
patch targets moved; that module's docstring has the argument.

**WHAT THE SPLIT HAD TO BE CAREFUL ABOUT, because this repository has been
caught by it twice.** A module that re-exports a name does not SHARE it:
rebinding `clients.SERVED_CONCURRENCY_FLOOR` does not reach
`latency.scaling_status`, and `tests/test_g1_client_concurrency.py` asserts
that asymmetry so a mutation aimed at the wrong module is not taken for a
clean result. Five existing tests patch a name ON `clients` and expect it to
reach the pool:

* `clients.window_concurrency` and `Instruments.merge` are read by
  `ClientPool.stop`, which stayed in `clients.py`. Unaffected.
* `clients.BARRIER_TIMEOUT_S`, `clients.CLIENT_BARRIER_TIMEOUT_S` and
  `clients.JOIN_TIMEOUT_S` are the pool's protocol and stayed there too --
  but `Client` needs all three. It does **not** import them, because an
  imported copy would freeze at import and a patch on `clients` would reach
  nothing. `ClientPool.__init__` reads them out of its own module globals and
  hands them over as a `Timeouts` bundle, so `monkeypatch.setattr(cl,
  "CLIENT_BARRIER_TIMEOUT_S", 0.3)` before a pool is built still reaches the
  client that waits on it -- which is what `tests/test_t4b_readers.py` does.
* `clients.Client.launch` and `clients.Client._tell` are patched as class
  attributes, which reaches the one class object however it is imported.

So the only name whose patch target moved is `CLIENT_START_METHOD`, which
nothing patches, and the only behaviour change is that the three timeouts are
captured when a pool is CONSTRUCTED rather than when each client reads them.
Every existing patch site already sets them before construction.

**THE CLIENTS ARE PROCESSES, AND THE RUN THAT SETTLED IT.** The paragraph
this one replaces said running the clients as processes was "the fix for the
underlying limit" and "deliberately NOT attempted", and named the next real
run as the thing that would settle whether the limit was the client library.
That run finished on 2026-10-14 -- a six-engine `smoke` pass on a Xeon Gold
6230, `SMOKE/results/smoke-20261014-075119/` -- and it settled it in the
direction the paragraph feared. `served = qps x p50` (Little's law, the
quantity `served_concurrency` computes) per client, at 1 client -> 4 clients:

    mariadb      0.95 -> 0.238      pgvector   0.98 -> 0.974
    mariadb123   0.95 -> 0.237      valkey     0.97 -> 0.969
    alisql       0.97 -> 0.242      mongodb    0.99 -> 0.984

Exactly the three engines that share MariaDB Connector/Python are pinned near
one query in service however many clients ask; the three that do not are at
the client count. That is the per-library split the paragraph predicted,
measured on live engines, and it is not a property of this pool: the same
pool put pgvector at 2,466.9 -> 8,754.7 q/s over the same four-fold rise. The
same run's churn writer managed **11.991 rows/s** against four readers --
`extra.reinsert_rows_per_s`, 2,000 rows in 166.794 s -- where the SAME run's
single-threaded load of the SAME corpus recorded `ingest_rows_per_s` 769.9 at
`ingest_threads` 1. A factor of 64, inside one run, with nothing changed but
whether four readers were querying. That comparison is stronger than §7's
against a modelled ~150 rows/s, which is a 1.18M-row glove figure from
`orchestrator/estimate.py` and not this corpus.

So the clients are now OS PROCESSES and there is no shared interpreter lock in
the measurement path. What that costs, and what it did not cost, each measured
rather than accepted (the §7 list this answers is in `docs/05-methodology.md`):

* **fork, not spawn, and it is not a preference.** `driver_factory` is a
  callable the caller supplies, and every real one is unpicklable:
  `harness/main.py:270` is a `def` nested inside `main()`, whose
  `__qualname__` is `main.<locals>.driver_factory`, and pickle resolves a
  function by qualified name -- measured, `AttributeError: Can't pickle local
  object`. The test suite's factories are a `lambda` and a bound
  `FakeDriver.clone`, which fail the same way. Under spawn the pool would
  therefore have to be handed something picklable, which is a change to every
  caller's contract; under fork nothing is pickled at all, because the child
  IS the parent's address space. `CLIENT_START_METHOD` pins it rather than
  inheriting the interpreter default, which is `fork` on Linux today and
  becomes `forkserver` in 3.14.
* **Memory: the cost §7 states is NOT REAL, and the DIRECTION is a
  measurement.** §7 says "each client becomes an OS process with its own copy
  of the query set, so the harness container's memory grows with the client
  count". Under fork it does not: the query set is read-only in a client,
  `shard_queries` hands it out as a strided VIEW rather than a copy, and
  copy-on-write never copies a page nobody writes. Which is not the same as
  a child leaving the pages alone -- a shard is strided across the whole
  array, so every child READS pages spread over the entire query set and
  only the write is absent. §7 carries the figures and their label: SUPPLIED,
  and a sum over eight children against one 104.9 MB array rather than a
  per-client one. They are not restated here, because two copies of a number
  is how the last set drifted.
* **The clock the processes agree on is still `time.perf_counter()`, and this
  was verified rather than assumed.** It is `clock_gettime(CLOCK_MONOTONIC)`
  on this platform, which is system-wide rather than per-process, so the
  reference point is shared across a fork -- unlike `time.process_time`.
  Measured over 200 forks, the child's `perf_counter - CLOCK_MONOTONIC`
  offset matches the parent's to a few microseconds and its reading fell
  inside the parent's bracketing pair 200 times out of 200; §7 has the
  readings. `tests/test_g3_process_clients.py` runs both probes at that same
  200 forks and asserts **10 ms, not those microseconds**: the quantity is a
  difference of two offsets each taken across two consecutive clock reads,
  so a descheduled thread widens it for a reason that is not the clock, and
  10 ms still separates the two clocks by four orders of magnitude --
  `time.process_time` is the control and a child reads near zero.
* **Client CPU accounting keeps working, for a structural reason.** `fork`
  never changes cgroup membership -- only an explicit write to
  `cgroup.procs` does -- so every client process is in the harness
  container's cgroup and `read_own_cpu_usec` still counts all of them.
  Verified by reading `/proc/self/cgroup` in parent and children and
  comparing the strings, which is exact; the MAGNITUDE cannot be checked on
  a developer box for the reason §2.3 gives, that outside a container the
  same file is the root cgroup and reads the whole machine.
* **`/dev/shm` needs no raising, measured.** `multiprocessing` semaphores
  are POSIX semaphores, one page each, and are unlinked immediately after
  `sem_open`, so they never appear in a directory listing -- measure the
  tmpfs, not the listing. One pool's whole footprint is **40,960 bytes** and
  is independent of the client count, because the go and halt flags are two
  Events for the pool rather than two per client (that design measured
  1,310,720 bytes at 32 clients). Pipes cost zero. The harness container is
  started with `shm_size` from the resource pass -- 2g normalized, 8g tuned,
  `orchestrator/ops_pass.py:138` -- so 40 KiB is five orders of magnitude
  inside it and nothing outside this module has to change.

**What did NOT survive, and a reader has to be told.** The clients no longer
share an interpreter, so a SYNTHETIC driver that models the defect by holding
the GIL -- `tests/test_g1_client_concurrency.py`'s `holding` family, a
pure-Python spin -- no longer serializes this pool and no longer produces
`concurrency_not_achieved`. The instrument is unchanged and still fires on a
pool that really serves one query at a time; what changed is that Python's own
lock can no longer put it there. The quantity is kept and the guard is kept,
because the SERVER serializing is the other cause `scaling_status` names and
this change does nothing about that one.

This also changes what every published multi-client number means, so it
invalidates comparison with every run measured before it, including the
2026-10-14 smoke run above -- `OPS_MEASUREMENT_VERSION` 6, bumped in the same
commit by the rule version 3's query-pool change earned one.

`ClientPool`, `PoolWindow`, `PoolSpec`, `Instruments`, the client-CPU
accounting and the sharding all stay in `clients.py`: they are the pool's, and
a client is a participant in the protocol rather than its owner.
"""

from __future__ import annotations

import multiprocessing
import threading
import time
from typing import (Any, Callable, Dict, List, NamedTuple, Optional)

import numpy

from ..drivers.base import EngineDriver, IndexSpec
from ..metrics.latency import LatencyCollector
# The span-transfer deadline, lifted out when this file reached the ceiling.
# `TRANSFER_GRACE_S` is imported so `client_process.TRANSFER_GRACE_S` still
# READS -- which is all anything does with it -- but `_recv_by` reads it out
# of `transfer_deadline`'s globals, so rebinding it here reaches nothing.
from .transfer_deadline import TRANSFER_GRACE_S  # noqa: F401
from .transfer_deadline import TransferDeadline

DriverFactory = Callable[[], EngineDriver]

#: The `multiprocessing` start method the pool forks its clients with, pinned
#: rather than inherited. See this module's docstring: spawn and forkserver
#: both pickle the target and its arguments, and every `driver_factory` a real
#: caller passes is a closure or a bound method that cannot be pickled, so
#: choosing them would be choosing to change every caller's contract. Pinned
#: and not left as the default because the Linux default is `fork` today and
#: becomes `forkserver` in CPython 3.14 -- a release upgrade would otherwise
#: turn a working pool into `AttributeError: Can't pickle local object` at the
#: start of a measured window.
CLIENT_START_METHOD = "fork"


#: The context the clients, the pipes and the two window flags come from.
#: One context object for the module, so nothing can half-fork: mixing
#: primitives from two contexts raises at use rather than at construction.
_MP = multiprocessing.get_context(CLIENT_START_METHOD)


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


class Timeouts(NamedTuple):
    """The three waits a client needs, handed over by the pool that owns them.

    Handed over rather than imported, and that is the split's one load-bearing
    decision -- see this module's docstring. `ClientPool.__init__` reads
    `clients.BARRIER_TIMEOUT_S`, `clients.CLIENT_BARRIER_TIMEOUT_S` and
    `clients.JOIN_TIMEOUT_S` out of its own module globals and builds one of
    these, so a test that rebinds any of them on `clients` before building a
    pool still reaches the client that waits on it. An `import` here would
    have frozen all three at module load and quietly defeated
    `tests/test_t4b_readers.py`'s barrier-timeout measurement, which is the
    test that establishes which of the two barrier numbers actually binds.
    """

    #: What the PARENT waits at the barrier; the client only needs it to bound
    #: how long it waits for the window to open after reporting warm.
    barrier_s: float
    #: What a CLIENT waits at the barrier, and the bound on its warm-up.
    client_barrier_s: float
    #: What the span transport and the join are given, for the message.
    join_s: float


class PoolError(RuntimeError):
    """A client failed. Raised rather than returned, and never swallowed.

    PR 4 §4.4 asks that a reader exception fail the record instead of hanging,
    and the hang is the specific hazard: a client that dies before the barrier
    leaves every other party waiting on it. `Client.attend` releases the
    barrier in a `finally` for that reason, `ClientPool.start` aborts the
    barrier on `BrokenBarrierError`, and `ClientPool.stop` joins with a
    timeout and then raises this. A pool that measured nothing raises too: a
    QPS of 0 from a pool whose clients all died reads as "the engine was
    slow", which is the one thing it is not evidence of.

    **AGAINST PROCESS DEATH RATHER THAN THREAD EXCEPTIONS, which is a wider
    set of failures and needed re-establishing.** A thread could only stop by
    raising, and the raise was an object in the same memory as the pool. A
    client process can also be killed outright -- `SIGKILL` from the OOM
    killer is the realistic one in a container with a memory limit, and it
    runs no handler, unwinds nothing and writes no error anywhere. The three
    ways that is caught, each with its own test in
    `tests/test_g3_process_clients.py`:

    * **Before the barrier.** The client's end of its pipe closes when it
      dies, so the parent's `recv` raises `EOFError` instead of blocking
      forever. `Client.attend` turns that into a `ClientDied` and ABORTS the
      barrier rather than joining it, so `start` raises at once instead of
      after the caller's whole window length. A thread could not reach this
      state at all.
    * **Mid-window.** The same `EOFError`, on the payload `stop` waits for,
      becomes the client's `error` and the existing failure path raises.
    * **After reporting.** A client that sent its spans and was then killed
      leaves `error` None and a window that looks complete, so `stop` checks
      `exitcode` as well: anything that is neither 0 nor None is a failure,
      and a negative one is the signal that killed it. Without this check a
      pool whose clients were OOM-killed at the end of a window would publish
      the partial window as a measurement.

    None of these is `qps == 0`. A window in which nothing finished is a
    legitimate measurement of a configuration whose queries outlast
    `ops.concurrency_duration_s`, and `concurrency._warn_nothing_finished`
    reports it on stderr; it must not be confused with a pool that died, and
    the two are distinguished by whether any client failed rather than by the
    throughput.
    """


class ClientFailed(PoolError):
    """One client's exception, carried back across the process boundary.

    TEXT and not the exception object, and the reason is not convenience: a
    driver's exception can hold a socket, a connection or a C extension's
    state, and pickling it either fails or reconstructs something that is not
    the same object. `str(ClientFailed(text)) == text`, so the message
    `ClientPool.stop` builds reads exactly as it did when the client was a
    thread and `extra.readers_failed` still carries the driver's own wording.

    A subclass of `PoolError` so a caller that already catches `PoolError` --
    `churn_readers` does, around both halves of its ratio -- catches this
    without being changed.
    """


class ClientDied(PoolError):
    """A client process ended without reporting anything.

    The state a thread could not reach. Named separately from `ClientFailed`
    because the two say different things to whoever reads the run's stderr: a
    `ClientFailed` is the driver's own error and the message is actionable,
    while a `ClientDied` means something outside the harness ended the
    process and the exit status is the only evidence there is.
    """


def _exit_description(exitcode: Optional[int]) -> str:
    """How a client process ended, in the words a reader needs.

    A negative `exitcode` is `-signum`, and the signal is the whole finding:
    `-9` is the OOM killer in a container with a memory limit and `-11` is a
    segfault in a driver's C extension. Spelling it out rather than printing
    the raw negative number, because "-9" in a run log is the kind of thing
    that gets read as an error code.
    """
    if exitcode is None:
        return "it is still running"
    if exitcode < 0:
        import signal
        try:
            name = signal.Signals(-exitcode).name
        except ValueError:  # pragma: no cover - unknown signal number
            name = f"signal {-exitcode}"
        return f"it was killed by {name}"
    return f"it exited with status {exitcode}"


class Client(TransferDeadline, _MP.Process):
    """One benchmark client: own connection, own query slice, own latencies.

    A process and no longer a thread, which is the whole of G3: MariaDB
    Connector/Python holds the interpreter lock across its socket wait, so
    three of this study's seven engines measured one query in service however
    many clients asked (the module docstring has the run). An OS process has
    its own interpreter and there is no shared lock left in the measurement
    path.

    **THE HAZARD FORK CREATES, AND IT IS THE ONLY ONE THAT MATTERS HERE.**
    `_MP.Process` with the fork start method copies the parent's address
    space, so `self` exists on both sides and `self.error`, `self.collector`
    and `self.results` are TWO objects after `start()`. Everything `run`
    assigns is in the child and the parent will never see it; everything the
    parent reads has to have crossed the pipe. That is a silent fault rather
    than a crash -- a pool whose clients all failed would report a clean
    window of zero queries -- so the two sides are separated by name here:
    `run` and the `_child_*` helpers execute in the CHILD and may only write
    down the pipe, while `attend`, `absorb` and the attributes the pool reads
    belong to the PARENT and are filled from what arrives on it.

    Deliberately NOT subclassing `multiprocessing.Process`: that is the
    default context's class, and the default start method on Linux becomes
    `forkserver` in CPython 3.14, which pickles. `_MP` pins fork. See
    `CLIENT_START_METHOD`.
    """

    def __init__(self, driver_factory: DriverFactory, queries: numpy.ndarray,
                 query_offsets: numpy.ndarray,
                 k: int, ef_search: int, index: IndexSpec, warmup: int,
                 ready: threading.Barrier, stop: threading.Event,
                 collect_results: bool = True,
                 go: Optional[Any] = None, halt: Optional[Any] = None,
                 broken_barrier: type = threading.BrokenBarrierError,
                 timeouts: Optional[Timeouts] = None):
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
        # Kept under its old name although the CHILD no longer watches it: it
        # is the parent-side flag `ClientPool.stop` sets first, and one
        # concurrency test replaces it with an `Event` whose `set()` is slow
        # in order to prove the window's end is read before the flag goes up.
        # The child watches `_halt`, which is set immediately after it.
        self._stop_event = stop
        self._go = go if go is not None else _MP.Event()
        self._halt = halt if halt is not None else _MP.Event()
        self._broken_barrier = broken_barrier
        if timeouts is None:
            # Refused rather than defaulted. A fallback here would be a second
            # copy of three numbers whose whole point is that `clients.py`
            # owns them and a test can move them -- and the copy would be the
            # one that silently won on any path that forgot to pass them.
            raise ValueError(
                "a client needs the pool's Timeouts; see client_process."
                "Timeouts for why they are passed and not imported")
        #: See `Timeouts`: the pool's numbers, read when the pool was built.
        self._timeouts = timeouts
        # One pipe per client, child -> parent, and one message per phase. A
        # Queue would do the same job with a feeder THREAD inside the child,
        # which is a thread in the measurement path for no gain.
        #
        # Opened by `launch` rather than here, and the ordering is what makes
        # a dead client detectable at all. See `launch`.
        self._from_client: Any = None
        self._to_parent: Any = None
        # False for the churn readers. §4.3 item 6: they compute no recall,
        # because the corpus is mid-change and no ground truth describes it,
        # so retaining a result list per query would allocate inside the
        # window for nothing a reader can use.
        self._collect_results = collect_results
        self.collector = LatencyCollector(warmup=0)
        #: Query position -> that query's answer and span, first pass only.
        self.results: Dict[int, Answer] = {}
        self.error: Optional[BaseException] = None
        #: Whether this client's window payload has been read off the pipe.
        #: Parent-side; `absorb` is idempotent so a second `stop` cannot hang.
        self._absorbed = False
        #: The span-transfer deadline's two flags, set here because this is
        #: where every other parent-side field is. INTENT: the deadline
        #: fired. FACT: this pool killed a LIVE client. They are two because
        #: only the fact may suppress `ClientPool.stop`'s exit-status report
        #: -- `transfer_deadline.py` has why, and what one flag silenced.
        self._transfer_cut = False
        self._killed_here = False

    # -- starting it -------------------------------------------------------

    def launch(self) -> None:
        """Open this client's pipe, fork it, and close the parent's write end.

        **ALL THREE, IN THIS ORDER, AND THE ORDER IS A CORRECTNESS
        PROPERTY.** `Connection.recv` raises `EOFError` only when EVERY copy
        of the pipe's write end is closed, and that is the whole mechanism by
        which the parent discovers a client that died without reporting --
        `PoolError`'s discipline rests on it. Two ways to keep a stray copy
        open, both of which turn a dead client into a pool that hangs until a
        timeout instead of failing at once:

        * the PARENT's own copy of the write end. Closed on the line below,
          immediately after the fork.
        * a SIBLING CHILD's copy. A fork duplicates every open descriptor, so
          a client forked while another client's pipe already existed would
          hold that pipe's write end for its whole life. This is why the pipe
          is opened here, one client at a time, instead of for all of them in
          `__init__`: by the time client *i* is forked, every earlier
          client's write end has already been closed in the parent, so there
          is nothing left for *i* to inherit.

        `tests/test_g3_process_clients.py` kills one client of several and
        reads back how long the pool took to fail, which is the measurement
        that distinguishes this from the version that hangs.
        """
        self._from_client, self._to_parent = _MP.Pipe(duplex=False)
        self.start()
        try:
            self._to_parent.close()
        except OSError:  # pragma: no cover
            pass

    # -- the child ---------------------------------------------------------

    def run(self) -> None:
        """The client, IN THE CHILD PROCESS. Writes only down the pipe.

        Nothing this method assigns to `self` reaches the pool. The structure
        is the thread version's, in two phases with a report after each, so
        the parent can tell "failed to warm up" from "failed in the window"
        from "never spoke again".
        """
        self._from_client.close()  # the child never reads its own pipe
        driver = None
        error: Optional[BaseException] = None
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
            error = exc

        # Report before the barrier, always -- the parent's `attend` is
        # blocked on this and a client that dies instead of sending it is why
        # `attend` has to treat `EOFError` as a death rather than waiting.
        self._tell("warmup", error, None, None)
        if error is not None or driver is None:
            return

        try:
            # The window opens in the parent, in the barrier's action, before
            # any client is released. This wait is what "released" means now,
            # and it is why `extra.prewindow_discarded` is still zero by
            # construction: the flag cannot be set until the opening instant
            # has been read.
            if not self._go.wait(timeout=self._timeouts.barrier_s):
                self._tell("window", ClientDied(
                    "the window never opened for this client"), None, None)
                return
            self.collector.start()
            i = 0
            n = len(self._queries)
            while not self._halt.is_set():
                q = self._queries[i % n]
                started = time.perf_counter()
                ids = driver.query(q, self._k)
                # Both instants, not the duration: whether this query counts
                # depends on when it ran. The window ends by setting a flag,
                # and this client is holding a query when that happens -- a
                # query that finishes after the wall clock in the denominator
                # stopped. `merge_collectors` drops it on the strength of
                # `finished`, so the instant has to survive to the merge --
                # and now also across the pipe, which is why the spans are
                # shipped as instants rather than as durations.
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
                        list(ids), started, finished)
                i += 1
            self.collector.stop()
        except BaseException as exc:  # noqa: BLE001
            error = exc
        finally:
            try:
                driver.close()
            except Exception:
                pass

        self._tell("window", error, self.collector, self.results)

    def _tell(self, phase: str, error: Optional[BaseException],
              collector: Optional[LatencyCollector],
              results: Optional[Dict[int, Answer]]) -> None:
        """Send one phase's outcome to the parent. CHILD side.

        The spans go as a flat list of `(started, finished)` floats rather
        than as `Sample` objects or as a `LatencyCollector`: the parent
        rebuilds them through `add_span`, which is the same call the child
        made, so `Sample.seconds` is derived once on each side by the same
        expression and the two cannot disagree. Shipping the collector itself
        would pickle its private state and make the transport sensitive to a
        field being added to it.

        The error is `repr`-free text. See `ClientFailed`.
        """
        spans = ([(s.started, s.finished) for s in collector._samples]
                 if collector is not None else [])
        try:
            self._to_parent.send(
                (phase, None if error is None else f"{error}", spans,
                 results or {}))
        except (BrokenPipeError, OSError):  # pragma: no cover - parent gone
            pass

    # -- the parent --------------------------------------------------------

    def attend(self) -> None:
        """Stand in for this client at the barrier. PARENT side, own thread.

        The barrier is a THREAD barrier in the parent, because its action is
        what reads the window's opening instant and that reading has to land
        in the parent's memory -- a `multiprocessing.Barrier`'s action runs in
        whichever process arrives last, so the instant would be written into
        a forked copy of `ClientPool._opened` and lost. See
        `clients.Instruments` for the second reason, which is that the barrier
        class is an injected instrument and the injected value is the
        `threading` module.

        So one parent thread per client waits for that client to say it is
        warm and then takes its place at the barrier. This thread does nothing
        during the measured window: it is gone before the window opens, and
        the GIL it holds while it waits is not in any client's query path.

        The `finally` is the thread version's, for the same reason: a client
        that failed still has to release the barrier, or every sibling waits
        out `clients.CLIENT_BARRIER_TIMEOUT_S` on a client that never comes.
        """
        try:
            self.error = self._await_warmup()
        finally:
            if isinstance(self.error, ClientDied):
                # A client that DIED cannot be waited for and must not be
                # waited for. Aborting breaks the barrier from this client's
                # side, which makes the parent's `wait` raise at once --
                # `start` then fails the record instead of the caller sleeping
                # out a whole window over a pool that is one client short.
                # A client that merely RAISED still joins the barrier, which
                # is what the thread version did: `start` returns, `stop`
                # collects the error and raises there.
                self._ready.abort()
            else:
                try:
                    self._ready.wait(timeout=self._timeouts.client_barrier_s)
                except self._broken_barrier:
                    pass

    def _await_warmup(self) -> Optional[BaseException]:
        """Wait for this client to report that it is warm. PARENT side.

        Three outcomes, and the pool behaves differently for each:
        the client reported (its error or None), the client died without
        reporting (`EOFError` on a closed pipe -- `ClientDied`), or the client
        is alive and still has not reported inside
        `clients.CLIENT_BARRIER_TIMEOUT_S`, which is the stuck-driver case
        the constant is named for.
        """
        try:
            if not self._from_client.poll(self._timeouts.client_barrier_s):
                return ClientDied(
                    f"client {self.name} did not finish its warm-up within "
                    f"{self._timeouts.client_barrier_s:.0f}s")
            phase, text, _spans, _results = self._from_client.recv()
        except (EOFError, OSError):
            return ClientDied(
                f"client {self.name} died during warm-up: "
                f"{_exit_description(self.exitcode)}")
        if phase != "warmup":  # pragma: no cover - protocol desync
            return ClientDied(
                f"client {self.name} sent a {phase!r} payload before the "
                f"window opened")
        return None if text is None else ClientFailed(text)

    def absorb(self, deadline: float) -> None:
        """Take this client's window payload off the pipe. PARENT side.

        Called before any client is joined, because a client blocks writing a
        payload larger than the pipe's 64 KiB buffer until somebody reads it
        -- a 30 s window of a fast engine is tens of megabytes of spans -- so
        joining first would deadlock the pool at exactly the configurations
        that matter most.

        `deadline` is a `time.monotonic` instant and is shared across the
        whole pool, so a slow first client cannot buy the last one a fresh
        `clients.JOIN_TIMEOUT_S`.

        **THE DEADLINE BOUNDS THE TRANSFER AND NOT ONLY THE WAIT FOR THE
        FIRST BYTE, and an earlier revision bounded only the wait.** `poll`
        respects `deadline`; `recv` takes no timeout, so a client that
        started sending and then stalled while STILL ALIVE blocked the parent
        inside `recv` for as long as it lived, taking `stop` past
        `JOIN_TIMEOUT_S` without limit on the one call an operator cannot
        interrupt. One that DIES mid-send always raised. `_recv_by` closes it.

        **NO PUBLISHED NUMBER CHANGES AND AN OUTCOME CAN**, and the
        distinction is the point: "neither fix changes a measurement" is
        exactly true of published NUMBERS -- every figure is fixed before
        this runs -- and not of OUTCOMES, because a bound can FAIL a window
        the unbounded wait would have published late. Measured: a
        1,000,000-span / 100,000-result payload takes **0.606 s** to `recv`
        against `TRANSFER_GRACE_S`'s 1.0 s floor, 61% of it, where a
        realistic ~100,000-span one takes **0.057 s** and leaves ~17x margin.
        Remote rather than absent, and the right way round -- a `stop` that
        never returns is not a result either.
        `tests/test_g4_pool_teardown.py` has the pairs.
        """
        if self._absorbed:
            return
        self._absorbed = True
        if self.error is not None:
            # It never ran a window, so there is no payload coming and
            # waiting for one would spend the whole deadline. The error is
            # already the finding.
            return
        try:
            left = max(0.0, deadline - time.monotonic())
            if not self._from_client.poll(left):
                self.error = ClientDied(
                    f"client {self.name} did not return its spans within "
                    f"{self._timeouts.join_s:.0f}s of the window closing")
                return
            phase, text, spans, results = self._recv_by(deadline)
        except (EOFError, OSError):
            if self._transfer_cut:
                # The deadline ended it, so say that rather than reporting
                # the SIGKILL this method sent as the client's own failure.
                self.error = ClientDied(
                    f"client {self.name} stopped partway through sending its "
                    f"spans and was ended at the pool's "
                    f"{self._timeouts.join_s:.0f}s deadline; a partial "
                    f"payload is not a measurement of the window")
            else:
                self.error = ClientDied(
                    f"client {self.name} died during the measured window: "
                    f"{_exit_description(self.exitcode)}")
            return
        if phase != "window":  # pragma: no cover - protocol desync
            self.error = ClientDied(
                f"client {self.name} sent a {phase!r} payload after the "
                f"window closed")
            return
        if text is not None:
            self.error = ClientFailed(text)
        # Rebuilt through `add_span` -- the call the child made -- so the
        # duration is derived by one expression on both sides. A collector
        # assembled by assigning to `_samples` would be a second definition
        # of `Sample.seconds`.
        for started, finished in spans:
            self.collector.add_span(started, finished)
        self.results = {int(offset): Answer(list(ids), began, ended)
                        for offset, (ids, began, ended) in results.items()}

    def reap(self) -> None:
        """End this client and close the parent's end of its pipe.

        `terminate` then `kill`, because a client blocked in a driver's C
        extension can ignore `SIGTERM` and a leaked client process would go on
        querying the server that the next configuration is about to measure.
        That is worse than a leaked thread: a thread dies with the
        interpreter, a process does not.
        """
        try:
            if self.is_alive():
                self.terminate()
                self.join(timeout=5.0)
            if self.is_alive():  # pragma: no cover - ignores SIGTERM
                self.kill()
                self.join(timeout=5.0)
        except (OSError, ValueError, AssertionError):  # pragma: no cover
            pass
        for end in (self._from_client, self._to_parent):
            if end is None:
                continue
            try:
                end.close()
            except OSError:  # pragma: no cover
                pass




__all__ = [
    "Answer", "CLIENT_START_METHOD", "Client", "ClientDied", "ClientFailed",
    "DriverFactory", "PoolError", "TRANSFER_GRACE_S", "Timeouts",
]
