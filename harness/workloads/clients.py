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
processes waited on a real dev or smoke run rather than on this measurement.
This paragraph lived at `concurrency.py:118` before PR 4 §4.3 lifted the flag
into this module, which deleted it; it is restored here because two workloads
now inherit the flag and because it is the qualitative form of a measurement
PR 4 went on to make. **That run has now happened -- see THE CLIENTS ARE
PROCESSES below -- and it did not reproduce the synthetic shape: it was
worse.**

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

**THE MECHANISM, AND WHICH PARTS OF IT ARE MEASURED.** A live `smoke` run
exposed this rather than a reading of the code, so the claims are separated by
how each one is known.

*Observed (SUPPLIED -- a `smoke` run of MariaDB on fashion-mnist-784 at
M=16, not reproduced here).* Throughput flat across a four-fold rise in
clients -- 2,920.7 q/s at one and 2,927.5 at four -- with every percentile
unchanged: p50 0.325 ms, p95 0.442/0.439, p99 0.552/0.544. `qps x p50` is
0.95, so about one query was in service. The same run's churn writer managed
12.0 rows/s against four readers at 2,819 q/s, against a modelled ~150.

*Measured here.* A saturated server and a serializing client ARE
distinguishable from the client's own spans, and `served_concurrency` is the
quantity that does it -- see `SERVED_CONCURRENCY_FLOOR` for the five-driver
calibration and `metrics/latency.WindowConcurrency` for why the exact
span-sum does not. The pool itself was never the fault: driven by a
lock-releasing driver it scales 1.02 at four clients. `client_cpu_util` on
the serializing point reads 0.177 against a 0.80 threshold.

*Read, not measured, and it CANNOT be measured here -- this host starts no
server.* Whether a client library releases the interpreter lock around its
socket wait is a property of that library's C extension, and the engines do
not share one: `drivers/postgres.py` imports **psycopg** (psycopg3) while
`drivers/mysql_family.py` imports **mariadb** (MariaDB Connector/Python over
Connector/C), which `docker/_shared/requirements-mysql.txt` says is used for
BOTH MariaDB and AliSQL on purpose. The question is per-library, not
per-engine, and nothing in this repository settles it.

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
* **Memory: the cost §7 states is NOT REAL, and that is a measurement.** §7
  says "each client becomes an OS process with its own copy of the query set,
  so the harness container's memory grows with the client count". Under fork
  it does not: the query set is read-only in a client, `shard_queries` hands
  it out as a strided VIEW rather than a copy, and copy-on-write never
  copies a page nobody writes. Measured, eight children over a 104.9 MB
  float32 array: summed `Private_Dirty` 13.8 MB, 0.13x the array, against
  838.9 MB if each had copied it. The control arm is what makes that a
  measurement and not a blind probe -- eight children that WRITE the array
  measure 851.7 MB, 8.12x, so the probe can see a copy when there is one.
  What a client does cost is its own interpreter's private pages, ~1.7 MB.
* **The clock the processes agree on is still `time.perf_counter()`, and this
  was verified rather than assumed.** It is `clock_gettime(CLOCK_MONOTONIC)`
  on this platform, which is system-wide rather than per-process, so the
  reference point is shared across a fork -- unlike `time.process_time`.
  Measured over 200 forks: the child's `perf_counter - CLOCK_MONOTONIC`
  offset matches the parent's to within **4.2 us**, which is the cost of
  reading two clocks back to back rather than an offset, and the child's
  reading fell inside the parent's bracketing pair **200 times out of 200**.
  `tests/test_g3_process_clients.py` asserts both, so a platform where
  `perf_counter` is process-local fails loudly instead of producing spans
  that cannot be compared with the window they are inside.
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

`_one_window` deliberately stays in `concurrency.py`: it owns the fixed
duration, the recall scoring and the repeats, none of which churn wants.
"""

from __future__ import annotations

import multiprocessing
import threading
import time
from typing import (Any, Callable, Dict, List, NamedTuple, Optional, Tuple)

import numpy

from ..drivers.base import EngineDriver, IndexSpec
# The scaling-status policy lives in `metrics/latency.py`, beside the
# quantity it judges and because this module was 846 lines against the repo's
# 800-line ceiling with it inlined -- the reason `churn_readers.py` exists.
# Re-exported under the names it was written with, because that is how
# `concurrency.py`, `churn_readers.py` and the tests reach them. **A
# re-export does not SHARE the name**: rebinding `clients.SERVED_CONCURRENCY_FLOOR`
# does not reach `scaling_status`, which reads it out of `latency`'s globals,
# and `tests/test_g1_client_concurrency.py` asserts that asymmetry so a
# mutation aimed here is not taken for a clean result.
from ..metrics.latency import (INFLIGHT_DISCARDED,  # noqa: F401
                               PREWINDOW_DISCARDED,
                               SCALING_CONCURRENCY_NOT_ACHIEVED, SCALING_OK,
                               SCALING_STATUSES, SERVED_CONCURRENCY_FLOOR,
                               LatencyCollector, merge_collectors,
                               scaling_status, served_fraction,
                               span_disposition, window_concurrency)
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
#:
#: THAT BLINDNESS IS NOW A MEASUREMENT AND NOT AN ARGUMENT. Four GIL-holding
#: clients at 0.325 ms per query delivered a quarter of this pool's throughput
#: -- `scaling_efficiency` 0.26 against 1.02 for the same cost released --
#: while `client_cpu_util` read **0.177** against this 0.80 and the flag
#: stayed `False`. The 0.177 is SUPPLIED: that counter is a cgroup's and reads
#: the whole machine outside a container.
#: `metrics/latency.SERVED_CONCURRENCY_FLOOR` is the quantity that sees it.
#:
#: Kept unchanged by the move to client PROCESSES. A client container that
#: genuinely runs out of CPU is still a real case and is the case this flag
#: catches; what the move removed is the OTHER case, the one it never saw.
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
#:
#: It is also the bound on how long the parent waits for a client PROCESS to
#: say it is warm, which is what made the pair bind in the one-client case
#: too. A single hanging thread used to leave the parent's 300 s as the only
#: limit, because there was no sibling to time out at the barrier and break
#: it; `Client.attend` aborts the barrier itself when its client has not
#: reported inside this, so the smaller number now binds at every client
#: count. That is the direction the comment above already claimed.
CLIENT_BARRIER_TIMEOUT_S = 120.0

#: Seconds a client is given to notice the stop flag and return. One query's
#: worth of slack over any latency this framework has measured; a client still
#: running after it is reported as a failure rather than waited on, because a
#: join with no timeout is how a reader exception becomes a hang.
#:
#: It bounds the span transport as well as the join, and the order matters: a
#: client PROCESS blocks writing its spans into a 64 KiB pipe until the parent
#: reads them, so `stop` reads every client's payload BEFORE it joins any of
#: them. Joining first is a deadlock rather than a slow path.
JOIN_TIMEOUT_S = 120.0

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
    #: The two concurrency figures, over exactly the spans the QPS above
    #: counts: the exact time-average number of queries the clients had
    #: OUTSTANDING, and the median-span estimate of how many were in SERVICE.
    #: `metrics/latency.WindowConcurrency` has the measurement that says why
    #: both are kept and why only the second can carry a guard.
    #:
    #: Defaulted rather than required, and the default is the honest absence:
    #: `tests/test_t4b_readers.py` hand-builds a `PoolWindow` to exercise
    #: `window_ratio`'s refusals, which are a property of `PoolSpec` alone.
    #: A None there says nobody measured it. `stop` passes both explicitly.
    outstanding_concurrency: Optional[float] = None
    served_concurrency: Optional[float] = None

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

        **WHAT THIS CANNOT SEE, as a measurement rather than an argument.**
        Four clients were run through this pool on a driver costing 0.325 ms
        per query and holding the interpreter lock for it, against the same
        driver releasing the lock -- the pair that
        `tests/test_g1_client_concurrency.py` rebuilds. The releasing one
        scaled -- `scaling_efficiency` 1.02 at four clients -- and the
        holding one delivered 0.26. On that 0.26 point `client_cpu_util` read
        **0.177** (SUPPLIED: that counter is a cgroup's and reads the whole
        machine outside a container, so only a containerised run makes it)
        against `CLIENT_BOUND_UTIL` of **0.80**, and this property returned
        `False` -- three of four clients' throughput went missing and the flag
        said the clients were fine. Not a threshold merely set too high: a
        lock ceiling PREVENTS the CPU consumption the quotient looks for,
        since a thread waiting its turn is not running and costs the counter
        nothing, and moving 0.80 into (0, 0.177] would fire on every point of
        every run. Read `scaling_status` instead, computed from the spans.
        This flag is kept because it catches a different and real case -- a
        client container that genuinely runs out of CPU at 16-32 clients with
        a driver that does release the lock, which `scaling_status` cannot.

        The clients are processes now, so the case above can no longer be
        reached through Python's own lock -- but the paragraph stays, because
        the figures in it are what a reader needs to understand why this flag
        is not the instrument, and because a client LIBRARY that serializes
        inside one process is not the only way a pool fails to achieve its
        concurrency. This flag's own case is unaffected by the move: the
        cgroup counter it divides still covers every client, since fork does
        not change cgroup membership.
        """
        if self.client_cpu_util is None:
            return None
        return self.client_cpu_util > CLIENT_BOUND_UTIL

    @property
    def served_fraction(self) -> Optional[float]:
        """`served_concurrency` as a share of this pool's client count.

        NOT recursive, though it reads like it: a class body is not an
        enclosing scope for the functions defined in it, so the call below
        resolves to `metrics/latency.served_fraction` through this module's
        globals and never to this property. Said out loud because the obvious
        "fix" -- renaming one of them -- would be undoing a deliberate choice
        to keep the window's view of the quantity under the quantity's own
        name. `scaling_status` below is the same shape.
        """
        return served_fraction(self.served_concurrency, self.spec.clients)

    @property
    def scaling_status(self) -> Optional[str]:
        """Whether a scaling efficiency over this window is reportable.

        `metrics/latency.scaling_status` holds the rule, the floor and the
        two statuses, and carries the measurement that chose the floor. This
        property is the window's view of it and nothing else, so there is one
        definition a test can move.
        """
        return scaling_status(self.served_concurrency, self.spec.clients)


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

    **IT IS STILL `threading`, AND THAT IS DELIBERATE NOW THAT THE CLIENTS ARE
    PROCESSES.** The obvious reading of "make the clients processes" is to
    hand this field a `multiprocessing` context, and it is wrong twice. It
    would not even import: `multiprocessing` and its contexts have **no**
    `BrokenBarrierError` attribute -- measured, `hasattr` is False on both --
    because `multiprocessing.synchronize.Barrier` SUBCLASSES
    `threading.Barrier` and raises `threading.BrokenBarrierError` itself, the
    same class object. And `concurrency._instruments()` passes
    `threading=threading` literally, so a pool that needed a multiprocessing
    bundle here would be broken by its own caller. So the barrier and the stop
    flag stay THREAD primitives in the PARENT, where the barrier's action
    still reads the window's opening instant in the last party to arrive;
    `Client.attend` stands in for each client process at that barrier and
    `ClientPool._go` / `_halt` are the two inter-process flags that carry the
    decision across the fork. `tests/test_g3_process_clients.py` asserts the
    `BrokenBarrierError` fact, so a future edit that reaches for a context
    here fails on the reason rather than on a traceback.

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

    A client process reads `time` out of its own copy of this module, which is
    the same module object the parent forked -- and the readings are
    comparable because `perf_counter` is `clock_gettime(CLOCK_MONOTONIC)`,
    whose reference point is the machine's rather than the process's. That is
    verified rather than assumed; the module docstring has the 200-fork
    measurement.
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


class Client(_MP.Process):
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
                 instruments: Instruments = DEFAULT_INSTRUMENTS):
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
        self._broken_barrier = instruments.threading.BrokenBarrierError
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
            if not self._go.wait(timeout=BARRIER_TIMEOUT_S):
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
        a forked copy of `ClientPool._opened` and lost. See `Instruments` for
        the second reason, which is that the barrier class is an injected
        instrument and the injected value is the `threading` module.

        So one parent thread per client waits for that client to say it is
        warm and then takes its place at the barrier. This thread does nothing
        during the measured window: it is gone before the window opens, and
        the GIL it holds while it waits is not in any client's query path.

        The `finally` is the thread version's, for the same reason: a client
        that failed still has to release the barrier, or every sibling waits
        out `CLIENT_BARRIER_TIMEOUT_S` on a client that is never coming.
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
                    self._ready.wait(timeout=CLIENT_BARRIER_TIMEOUT_S)
                except self._broken_barrier:
                    pass

    def _await_warmup(self) -> Optional[BaseException]:
        """Wait for this client to report that it is warm. PARENT side.

        Three outcomes, and the pool behaves differently for each:
        the client reported (its error or None), the client died without
        reporting (`EOFError` on a closed pipe -- `ClientDied`), or the client
        is alive and still has not reported inside
        `CLIENT_BARRIER_TIMEOUT_S`, which is the stuck-driver case the
        constant is named for.
        """
        try:
            if not self._from_client.poll(CLIENT_BARRIER_TIMEOUT_S):
                return ClientDied(
                    f"client {self.name} did not finish its warm-up within "
                    f"{CLIENT_BARRIER_TIMEOUT_S:.0f}s")
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
        `JOIN_TIMEOUT_S`.
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
                    f"{JOIN_TIMEOUT_S:.0f}s of the window closing")
                return
            phase, text, spans, results = self._from_client.recv()
        except (EOFError, OSError):
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

    A VIEW and not a copy, which is now load-bearing for a second reason.
    `docs/05-methodology.md` §7 states as a cost of client processes that
    "each client becomes an OS process with its own copy of the query set".
    It does not: these are strided views of one array, a client never writes
    to one, and fork's copy-on-write therefore copies no page of it. Measured
    at 0.13x the array across eight children, against 8.12x for children that
    write to it -- the module docstring has both arms.
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

    Still a measurement of the clients now that they are separate processes,
    and the reason is structural rather than empirical: `fork` does not change
    cgroup membership -- only an explicit write to `cgroup.procs` does -- so
    every client is inside the harness container's cgroup and the one counter
    `read_own_cpu_usec` reads covers all of them. Verified by comparing
    `/proc/self/cgroup` between parent and children, which is exact; the
    magnitude cannot be verified outside a container, because there the same
    file is the root cgroup and reads the whole machine (§2.3).
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

    A pool measures ONE window. Processes cannot be restarted any more than
    threads could, and the churn workload needs its reader windows either side
    of a delete step it must not be reading through (§4.3 item 4 starts the
    readers after the delete), so it builds a second pool from the same
    `PoolSpec` instead. `PoolSpec` equality is what makes "the same pool"
    checkable rather than asserted -- see `PoolSpec` for the two
    published-wrong ratios that motivated it.

    **THE THREE THINGS THAT CROSS THE FORK, and why they are not more.** The
    clients are processes, so the barrier, the stop flag and the per-client
    collectors can no longer all be one kind of object:

    * the **barrier** stays a thread barrier in the parent, with one
      `Client.attend` thread standing in for each client, because its action
      reads the window's opening instant into parent memory and because it is
      an injected instrument (`Instruments`);
    * **`_go` and `_halt`** are the two inter-process `Event`s that carry the
      barrier's decision and the stop flag's decision to the children. Two
      for the POOL and not two per client, which is what keeps the `/dev/shm`
      footprint at a measured 40,960 bytes whatever the client count;
    * the **spans** come back over one pipe per client and are rebuilt into
      the same `LatencyCollector` objects the merge has always been handed,
      so `merge_collectors` and `window_concurrency` see exactly what they
      saw -- one shared wall, one `span_disposition` rule, no new definition
      of "inside the window".
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
        # The two flags the CLIENTS watch. The parent's `_stop` above is the
        # instrument a test replaces and the witness that the window's end is
        # read before any flag goes up; `_halt` is what actually ends a
        # client's loop, and it is set immediately after `_stop`. Setting
        # `_halt` first would let a slow `_stop.set()` -- which one
        # concurrency test injects on purpose -- stop the clients before the
        # ordering it exists to demonstrate could be observed.
        self._go = _MP.Event()
        self._halt = _MP.Event()
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
                   collect_results=collect_results,
                   go=self._go, halt=self._halt, instruments=instruments)
            for shard, offsets in shard_queries(queries, clients)
        ]
        #: The parent threads that stand in for the clients at the barrier.
        #: Empty until `start`; they are finished before the window opens.
        self._attendants: List[threading.Thread] = []
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

        The clients are FORKED FIRST, before any attendant thread and before
        the heartbeat, and the order is deliberate. A fork copies only the
        calling thread, so a child inherits any lock a sibling thread happened
        to be holding at that instant -- already locked, with no thread left
        to unlock it. Forking before this method starts a thread of its own
        keeps the window of exposure to threads the CALLER already had.

        **THE CALLER'S OWN THREADS ARE NOT THIS METHOD'S TO REMOVE, AND
        CPYTHON NOW SAYS SO OUT LOUD.** Since 3.12 `os.fork()` raises
        `DeprecationWarning: This process is multi-threaded, use of fork()
        may lead to deadlocks in the child` whenever the forking process has
        more than one thread. A real run reaches here from
        `harness/main.py` with no thread of its own, so the warning does not
        fire; the test suite reaches here from fixtures that hold a
        `threading.Lock` to count queries, and 53 of those warnings are in
        the suite's output. They are not noise to filter: each one names a
        fork that COULD have copied a held lock, and the reason none of them
        deadlocks is that the locks in question are released between queries.
        Nothing here suppresses the warning, because a measurement framework
        that silenced a deadlock warning would be choosing a quiet log over a
        run that finishes.
        """
        for worker in self._workers:
            worker.launch()
        self._started = True
        # One attendant per client, started after the fork so no child
        # inherits them. Each waits for its client to report warm and then
        # takes that client's place at the barrier.
        self._attendants = [
            threading.Thread(target=worker.attend, daemon=True,
                             name=f"client-attendant-{i}")
            for i, worker in enumerate(self._workers)]
        for attendant in self._attendants:
            attendant.start()
        try:
            with Heartbeat(label, prefix=prefix):
                self._ready.wait(timeout=BARRIER_TIMEOUT_S)
        except self._instruments.threading.BrokenBarrierError:
            self._ready.abort()
            self._stop.set()
            self._halt.set()
            first = self._first_error()
            # The clients are ended rather than left to notice a flag. A
            # thread that never noticed died with the interpreter; a process
            # that never notices goes on querying the server the next
            # configuration is about to measure, which would put one engine's
            # concurrency figure on top of the previous one's leftovers.
            for worker in self._workers:
                worker.reap()
            raise PoolError(
                f"a client failed during warm-up: "
                f"{first or 'barrier timed out'}")
        # The window is open: its instant has been read in the action above,
        # before this line, so no measured query can precede it and
        # `extra.prewindow_discarded` is still zero by construction.
        self._go.set()

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
        # The flag the clients actually watch, set immediately after the
        # instrument above and never before it. See `__init__`.
        self._halt.set()
        wall = stop_at - wall_start
        cpus = self._instruments.cpu_count()
        cpu_seconds, cpu_util = client_cpu(cpu_start, cpu_end, wall, cpus)

        # Every payload BEFORE any join. A client blocks writing more than the
        # pipe's buffer until the parent reads it, so joining first deadlocks
        # the pool on exactly the windows that produced the most spans. One
        # deadline across the pool, so a slow first client cannot hand the
        # last one a fresh timeout.
        deadline = time.monotonic() + JOIN_TIMEOUT_S
        for worker in self._workers:
            worker.absorb(deadline)
        for worker in self._workers:
            worker.join(timeout=max(0.0, deadline - time.monotonic()))

        failures = [w.error for w in self._workers if w.error is not None]
        if failures:
            for worker in self._workers:
                worker.reap()
            raise PoolError(
                f"{len(failures)} of {self.spec.clients} clients failed; "
                f"first error: {failures[0]}")
        still_running = [w for w in self._workers if w.is_alive()]
        if still_running:
            for worker in self._workers:
                worker.reap()
            raise PoolError(
                f"{len(still_running)} of {self.spec.clients} clients did not "
                f"return within {JOIN_TIMEOUT_S:.0f}s of the window closing")
        # A client that sent its spans and was THEN killed leaves no error and
        # a window that looks complete. Only the exit status says otherwise,
        # and the realistic case is a container's memory limit reaping a
        # client at the end of a long window: without this check the pool
        # would publish the partial window as a measurement of the engine.
        # Checked after the two above so the more specific message wins.
        killed = [(w, w.exitcode) for w in self._workers
                  if w.exitcode not in (0, None)]
        if killed:
            worker, code = killed[0]
            for each in self._workers:
                each.reap()
            raise PoolError(
                f"{len(killed)} of {self.spec.clients} client processes ended "
                f"abnormally after reporting; first: client {worker.name}, "
                f"{_exit_description(code)}. The spans it sent cover only "
                f"part of the window, so this is not a measurement.")

        merged = self._instruments.merge(
            [w.collector for w in self._workers], wall,
            cutoff=stop_at, window_start=wall_start)

        # The concurrency this pool actually achieved, over the same spans
        # and by the same `span_disposition` rule the merge above used. NOT
        # routed through `Instruments.merge`: that bundle exists so a test can
        # replace the merge to show what an untrimmed one counted, and a guard
        # computed by the replaced instrument would be calibrated against the
        # test's fixture rather than against the window.
        outstanding, served = window_concurrency(
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

        for worker in self._workers:
            worker.reap()

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
            outstanding_concurrency=outstanding,
            served_concurrency=served,
        )

    def _first_error(self) -> Optional[BaseException]:
        """The first client error the parent knows about.

        **It does NOT wait for the attendants, and an earlier revision of
        this method did.** The reasoning for waiting was that an attendant
        might be one `poll` return away from naming the driver's own
        exception, which is worth more to an operator than 'barrier timed
        out'. It was wrong twice. An attendant that has not reported is
        blocked on its client's PIPE, not on the barrier, so breaking the
        barrier does not wake it and the wait runs to its full length every
        time. And the wait lands inside the failure path, where it delays the
        very timeout it is reporting: measured through
        `tests/test_t4b_readers.py`'s inverted barrier pair -- client 6.0 s,
        parent 1.5 s -- a one-second grace made the pool fail after **2.51 s**
        against a 1.5 s timeout, which is a `PoolError` whose message
        misstates its own cause by two thirds.

        Nothing is lost by not waiting, because the attendant assigns
        `worker.error` BEFORE it touches the barrier: `attend` sets it in the
        `try` and aborts or waits in the `finally`. So every error that could
        have broken the barrier is already readable here, and the only case
        that reaches the fallback is the one that has no error to name -- a
        client still warming up when the barrier timed out.
        """
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
    "CLIENT_BOUND_UTIL", "CLIENT_START_METHOD", "Client", "ClientDied",
    "ClientFailed", "ClientPool", "DEFAULT_INSTRUMENTS", "DriverFactory",
    "Instruments", "JOIN_TIMEOUT_S", "PoolError", "PoolSpec", "PoolWindow",
    "SCALING_CONCURRENCY_NOT_ACHIEVED", "SCALING_OK", "SCALING_STATUSES",
    "SERVED_CONCURRENCY_FLOOR", "client_cpu", "shard_queries", "window_ratio",
]
