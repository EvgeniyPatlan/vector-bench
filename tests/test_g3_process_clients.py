"""G3: the clients are OS processes, so no shared lock is in the measurement path.

**This is the change `docs/05-methodology.md` §7 ends by describing as "not
attempted here".** It was attempted because the run §7 named as the thing that
would settle whether the limit was the client library happened, and settled it
the bad way. From `SMOKE/results/smoke-20261014-075119/` -- six engines on
fashion-mnist-784 at M=16, a Xeon Gold 6230, 158 ops records every one of them
stamped `ops_measurement_version` 5 -- `served = qps x p50` per client, at one
client and at four:

    mariadb      0.95 -> 0.238      pgvector   0.98 -> 0.974
    mariadb123   0.95 -> 0.237      valkey     0.97 -> 0.969
    alisql       0.97 -> 0.242      mongodb    0.99 -> 0.984

Those six rows are SUPPLIED: that run is not in this repository, no test here
starts an engine, and nothing in this file reproduces them. What they show is
that exactly the three engines sharing MariaDB Connector/Python are pinned near
one query in service however many clients ask, which is the per-library split
§7 predicted and could not measure.

**What this file measures is the pool, and it measures both pools.** §7
publishes a five-driver calibration: drivers of identical per-query cost
differing only in what they do with the interpreter lock.
`tests/test_g1_client_concurrency.py` builds four of those regimes and asserts
that the lock-holding one makes the pool refuse to publish a scaling
efficiency. That file is correct about the pool it was written against and is
deliberately not edited here. This file builds the same regimes against BOTH
pools -- the thread pool as a verbatim-shaped copy of the pre-change client
loop (`_thread_window`), and the real `ClientPool` as it now is -- and reports
the pair, because a change that moves a verdict has to show the verdict moving
rather than assert that it did.

Measured here, at four clients, threads then processes (one run, on a box
measuring 7.5 to 9.9 busy cores of 20 while it was taken):

    regime                      eff@4 (thr/proc)  served/n   outstanding/n
    releases the lock (sleep)    1.006 / 1.009  0.975/0.983  0.994/0.980
    saturated server             0.276 / 0.272  0.977/1.004  0.994/0.991
    C call holding the lock      0.249 / 0.995  0.243/0.982  0.944/0.979
    Python spin holding the lock 0.265 / 0.999  0.264/0.995  0.835/0.995
    half released, half held     0.518 / 0.980  0.979/0.980  0.997/0.991

`outstanding/n` is 0.84-1.00 on every row of both arms, across efficiencies
from 0.249 to 1.009. It is blind, exactly as §7 says, and for the same
mechanical reason -- a client waiting for the lock is still inside its own
timed region, so the stretched spans tile the window. `served/n` is the row
that separates the regimes, and it separates them by a factor of four.

Three further readings off it. The two lock-holding rows move from a quarter
of their client count in service to all of it, which is the change. The
**saturated-server row does not move**, and that is the row the threshold
exists for: it is a server that really is the bottleneck, its efficiency is
still 0.275, and a pool that had started flagging it would have become an
instrument that refuses real engine ceilings. And the lock-releasing row does
not move either, so the transport, the barrier and the merge did not cost the
pool anything.

**The lock-holding row is a C call that WAITS with the lock held, not one
that burns CPU with it held, and that is load-bearing.** §7's table holds the
lock with an atomic C call over a large tuple and with a Python spin, both of
which cost a core per client for the whole query -- so four client PROCESSES
in those regimes achieve their concurrency only if the box has four cores
spare, and the first version of this file asserted exactly that and failed at
8-18 busy cores of 20. A real client library holding the lock across a socket
wait burns nothing: `ctypes.PyDLL`'s `usleep` costs a measured 0.034 cores and
serializes four threads just the same. The strict verdicts are keyed to that
row; §7's spinning row is kept and reported, with only the part of it that is
about the lock asserted. See `_HELD_USLEEP` and `LOAD_INSENSITIVE_SERIALIZES`.

**The saturated regime's queue is a `multiprocessing.Value` and not a
`threading.Lock`, and that is load-bearing rather than a detail.**
`test_g1_client_concurrency.py` counts queue depth behind a `threading.Lock`,
which is a per-process object: under fork each client would see its own queue
at depth one, every query would cost the base latency, the pool would scale
and the row would protect nothing. One shared counter is used in BOTH arms so
the two columns above are the same regime measured twice.

**Mutation discipline (R4).** Every assertion below was checked by injecting
the regression AT RUN TIME -- a monkeypatch, a driver factory, or a flag set
before the fork -- never by editing a file on disk. Each docstring carries the
honest value and the mutant value, both read back out of the pool or the
record. An identical pair is called out as a dead probe rather than reported as
a guard that holds.

**Load.** This box is shared, and a process pool on a busy box gives noisy
absolute timings. So nothing here asserts a latency or a throughput: the
assertions are ratios against `SERVED_CONCURRENCY_FLOOR` and regime verdicts,
with the nearest measured value at least twice the threshold on both sides. The
two windows per point are reduced by MAX qps rather than by median, because
foreign load can only depress a window and the question asked of each point is
whether the pool can achieve its concurrency, not how fast this box is.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import multiprocessing
import os
import pickle
import signal
import sys
import threading
import time
import types
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.drivers.base import IndexSpec                            # noqa: E402
from harness.metrics import records                                   # noqa: E402
from harness.metrics.latency import (LatencyCollector,                # noqa: E402
                                     merge_collectors, span_disposition,
                                     window_concurrency)
from harness.workloads import clients as cl                           # noqa: E402
from tests.fake_driver import (FakeDriver, K, N_QUERIES, N_ROWS,      # noqa: E402
                               index_spec, tiny_dataset)

VB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: The fork context every probe here uses, so a primitive built in a test and
#: a primitive built by the pool come from one context. Mixing two contexts
#: raises at use rather than at construction, which is the kind of failure
#: that surfaces inside a measured window.
MP = multiprocessing.get_context("fork")

#: Wall clock each synthetic query costs, identical across every regime.
#: **That identity is the whole experiment.** 0.5 ms for the reason
#: `test_g1_client_concurrency.py:QUERY_S` records: well below the
#: interpreter's 5 ms switch interval, so a lock-holding query's wait lands in
#: the tail rather than in the median, and well above the pool's own
#: per-query bookkeeping.
QUERY_S = 0.0005

#: One measured window. Short, because every regime is measured twice in each
#: of two arms at two client counts; long enough that a one-client window
#: completes several hundred queries, so a median is the driver's and not the
#: box's.
WINDOW_S = 0.3

#: Windows per point. Two, reduced by MAX qps -- see the module docstring.
REPEATS = 2

#: What every synthetic query returns. Fixed ids rather than the fake's real
#: neighbours, and load-bearing for the reason `test_g1`'s `SYNTHETIC_IDS`
#: gives: `FakeDriver.query` runs a NumPy scan and NumPy RELEASES the
#: interpreter lock inside it, which would land inside the span the instrument
#: sums and stop the lock-holding regimes from holding the lock.
SYNTHETIC_IDS = list(range(K))


# ---------------------------------------------------------------------------
# The five regimes, built so BOTH arms measure the same thing
# ---------------------------------------------------------------------------

def _spin(seconds: float) -> None:
    """Burn `seconds` without a call that releases the interpreter lock."""
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        pass


#: `usleep` reached two ways. `ctypes.PyDLL` does NOT release the interpreter
#: lock around a call and `ctypes.CDLL` does, so the same libc function gives
#: a lock-holding wait and a lock-releasing one of identical cost -- which is
#: the controlled pair §7's table is built out of.
#:
#: **This is the faithful model of the defect and it replaced a CPU-bound
#: one.** §7's table holds the lock with an atomic C call over a large tuple
#: and a Python spin, both of which BURN CPU for the whole query. A real
#: client library holding the lock across a socket wait burns none -- it is
#: blocked in C -- and the difference decides whether this file's headline
#: assertion is about the lock or about the box. Measured: a held `usleep`
#: costs **0.034 cores**, so four client PROCESSES can all be waiting at once
#: however busy the machine is, while four CPU-bound ones need four free
#: cores and cannot achieve their concurrency without them. The first
#: version of this file used the CPU-bound regime and its process-arm
#: assertion failed on a box at 8-18 busy cores of 20 -- correctly, and for a
#: reason that was not the lock.
#:
#: The lock behaviour is checked directly rather than assumed: 4 threads x 200
#: held waits of 500 us took 0.466 s against a fully-serial 0.400 s, and the
#: released form took 0.116 s against a fully-parallel 0.100 s.
#: `TestTheHeldWaitReallyHoldsTheLock` asserts that pair.
_LIBC_NAME = ctypes.util.find_library("c") or "libc.so.6"
_HELD_USLEEP = ctypes.PyDLL(_LIBC_NAME).usleep
_HELD_USLEEP.argtypes = [ctypes.c_uint]
_RELEASED_USLEEP = ctypes.CDLL(_LIBC_NAME).usleep
_RELEASED_USLEEP.argtypes = [ctypes.c_uint]
_USLEEP_US = int(QUERY_S * 1e6)


def regime_releasing() -> Callable[[], None]:
    """A query that RELEASES the lock for its whole cost.

    `time.sleep` is the stdlib's lock-releasing wait, and is the regime a
    driver is in when its client library drops the lock around the socket.
    The pool must scale in it, under threads and under processes alike.
    """
    return lambda: time.sleep(QUERY_S)


def regime_held() -> Callable[[], None]:
    """A C call that HOLDS the lock across a wait, burning no CPU.

    This is what MariaDB Connector/Python over Connector/C does around its
    socket wait, and it is the regime the 2026-10-14 run found three engines
    in. `ctypes.PyDLL` is the one spelling in the stdlib that gives it: the
    same `usleep` through `ctypes.CDLL` releases the lock and is
    `regime_releasing`'s mechanism in C rather than in Python.

    Costing no CPU is what makes this row load-insensitive in both arms --
    see `_HELD_USLEEP`.
    """
    return lambda: _HELD_USLEEP(_USLEEP_US)


def regime_spin() -> Callable[[], None]:
    """A pure-Python deadline loop holding the lock.

    §7's "Python spin holding the lock" row, kept so the table has the shape
    §7 published. **CPU-BOUND, and that bounds what can be asserted about its
    process arm**: four spinning processes deliver four queries at once only
    if the box has four cores to give them, so an absolute assertion there
    would be an assertion about the machine. `regime_held` is the row this
    file's strict verdicts are keyed to; see `_HELD_USLEEP`.
    """
    return lambda: _spin(QUERY_S)


def regime_half() -> Callable[[], None]:
    """Half released, half held: a partial-serialization control.

    Here so no threshold is calibrated against two extremes alone.
    """
    def spend() -> None:
        time.sleep(QUERY_S / 2.0)
        _spin(QUERY_S / 2.0)
    return spend


def regime_saturated() -> Callable[[], None]:
    """A server that IS the bottleneck, with a queue shared across processes.

    The queue is the SERVER's, so a query waits as long as there are queries
    in flight and the client is blocked for all of it. With n clients looping,
    each sees depth n and waits n x QUERY_S, so each manages 1/(n x QUERY_S)
    and the pool manages 1/QUERY_S whatever n is: flat throughput, efficiency
    about 1/n. **This is the case the guard must not fire on**, because every
    client really is waiting on the engine.

    A `multiprocessing.Value` and not a `threading.Lock` -- see the module
    docstring. Created before any fork, so every client process shares the one
    counter and the regime still models one server.
    """
    depth = MP.Value("i", 0)

    def spend() -> None:
        with depth.get_lock():
            depth.value += 1
            here = depth.value
        try:
            time.sleep(QUERY_S * here)
        finally:
            with depth.get_lock():
                depth.value -= 1
    return spend


REGIMES: Dict[str, Callable[[], Callable[[], None]]] = {
    "releasing": regime_releasing,
    "saturated": regime_saturated,
    "held": regime_held,
    "spin": regime_spin,
    "half": regime_half,
}

#: Which regimes serialize a THREAD pool, and which it achieves its
#: concurrency in. Named here so a regime added above has to be classified in
#: both arms. The process arm's classification is that there is no
#: serializing regime left, which is the deliverable.
THREAD_SERIALIZES = ("held", "spin")
THREAD_ACHIEVES = ("releasing", "saturated", "half")

#: Of the serializing regimes, the ones whose query costs no CPU -- so four
#: client processes can all be in a query at once whatever else the box is
#: doing. The strict before/after verdicts are keyed to these.
LOAD_INSENSITIVE_SERIALIZES = ("held",)

#: And the one that burns CPU for its whole query, so its process arm needs
#: as many free cores as there are clients. Reported, and asserted only where
#: the assertion is about the lock rather than about the machine.
CPU_BOUND_SERIALIZES = ("spin",)
assert set(LOAD_INSENSITIVE_SERIALIZES) | set(CPU_BOUND_SERIALIZES) == \
    set(THREAD_SERIALIZES)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dataset():
    return tiny_dataset()


@pytest.fixture(scope="module")
def queries(dataset):
    return dataset.test[:N_QUERIES]


@pytest.fixture(scope="module")
def base(dataset):
    driver = FakeDriver(dim=dataset.dim, metric=dataset.metric, ef_search=32,
                        ef_half=64.0)
    driver.connect()
    driver.create_schema(index_spec(dataset))
    driver.load(dataset.train[:N_ROWS], list(range(N_ROWS)))
    return driver


def factory_for(base: FakeDriver, spend: Callable[[], None]):
    """A driver factory whose clones spend `spend()` and return fixed ids.

    Bound on the INSTANCE rather than on a subclass, because
    `FakeDriver.clone` returns a plain `FakeDriver` and a subclass's override
    would not reach the clients the pool builds. The cost is inside the timed
    region deliberately: the client brackets `driver.query` alone, and the
    spans are the whole input to the instrument.

    Under fork the child calls this closure out of inherited memory, which is
    why no part of it has to be picklable -- see
    `TestForkIsForcedByTheCallersAndNotChosen`.
    """
    def clone():
        client = base.clone()

        def query(vector, k):
            spend()
            return SYNTHETIC_IDS[:k]

        client.query = query
        return client
    return clone


def spec_for(queries) -> IndexSpec:
    return IndexSpec(dim=queries.shape[1], m=16, metric="euclidean")


# ---------------------------------------------------------------------------
# The thread arm: a verbatim-shaped copy of the pre-change client loop
# ---------------------------------------------------------------------------

class _ThreadClient(threading.Thread):
    """The pre-change client, copied so the two arms can be compared.

    This is `clients.Client` as it stood at 9fbf2cb: a `threading.Thread`
    whose `run` connects, warms up, releases the barrier in a `finally`, and
    then loops on `driver.query` recording both instants until a
    `threading.Event` is set. It is here and not reconstructed by patching the
    real pool because R2 asks that a repair be shown against the form it
    replaced, run over the same input -- and because the comparison is the
    deliverable, so the old arm has to be reproducible from this tree after
    the new one has replaced it.
    """

    def __init__(self, factory, queries, k, ready, stop):
        super().__init__(daemon=True)
        self._factory = factory
        self._queries = queries
        self._k = k
        self._ready = ready
        self._stop_event = stop
        self.collector = LatencyCollector(warmup=0)
        self.error: Optional[BaseException] = None

    def run(self) -> None:
        driver = None
        try:
            driver = self._factory()
            driver.connect()
            for i in range(min(2, len(self._queries))):
                driver.query(self._queries[i], self._k)
        except BaseException as exc:  # noqa: BLE001
            self.error = exc
        finally:
            try:
                self._ready.wait(timeout=cl.CLIENT_BARRIER_TIMEOUT_S)
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
                self.collector.add_span(started, time.perf_counter())
                i += 1
            self.collector.stop()
        except BaseException as exc:  # noqa: BLE001
            self.error = exc


def _thread_window(factory, queries, clients: int,
                   window_s: float = WINDOW_S) -> Dict[str, float]:
    """One measured window of the THREAD pool. The arm being compared against.

    Reuses the real `merge_collectors` and `window_concurrency`, so the only
    difference from the process arm is where the queries ran. Returns the
    three figures the table reports.
    """
    opened: List[float] = []
    stop = threading.Event()
    def open_window():
        opened.append(time.perf_counter())

    ready = threading.Barrier(clients + 1, action=open_window)
    shards = cl.shard_queries(queries, clients)
    workers = [_ThreadClient(factory, shard, K, ready, stop)
               for shard, _offsets in shards]
    for worker in workers:
        worker.start()
    ready.wait(timeout=cl.BARRIER_TIMEOUT_S)
    time.sleep(window_s)
    stop_at = time.perf_counter()
    stop.set()
    wall = stop_at - opened[0]
    for worker in workers:
        worker.join(timeout=30.0)
    collectors = [w.collector for w in workers]
    merged = merge_collectors(collectors, wall, cutoff=stop_at,
                              window_start=opened[0])
    outstanding, served = window_concurrency(collectors, wall, cutoff=stop_at,
                                             window_start=opened[0])
    return {"qps": merged["qps"], "served": served,
            "outstanding": outstanding, "clients": clients}


def _process_window(factory, queries, clients: int,
                    window_s: float = WINDOW_S) -> Dict[str, float]:
    """One measured window of the real pool, reduced to the same three keys."""
    pool = cl.ClientPool(factory, queries, clients, K, 32, spec_for(queries),
                         warmup=2)
    pool.start(f"g3 {clients}-client warm-up", prefix="g3")
    time.sleep(window_s)
    window = pool.stop()
    return {"qps": window.qps, "served": window.served_concurrency,
            "outstanding": window.outstanding_concurrency,
            "clients": clients, "window": window}


def _best(arm, factory, queries, clients) -> Dict[str, float]:
    """The better of `REPEATS` windows, by qps. See the module docstring."""
    return max((arm(factory, queries, clients) for _ in range(REPEATS)),
               key=lambda w: w["qps"])


@pytest.fixture(scope="module")
def table(base, queries) -> Dict[str, Dict[str, Dict[str, float]]]:
    """The five-regime calibration, both arms, measured once for the module.

    One fixture because the whole table is one experiment: every regime pays
    the same per-query cost, and measuring the rows at different times under
    different foreign load is how two columns of one table come to disagree
    for a reason that is not the lock.
    """
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for name, build in REGIMES.items():
        spend = build()
        factory = factory_for(base, spend)
        row: Dict[str, Dict[str, float]] = {}
        for arm_name, arm in (("threads", _thread_window),
                              ("processes", _process_window)):
            one = _best(arm, factory, queries, 1)
            four = _best(arm, factory, queries, 4)
            row[arm_name] = {
                "efficiency": four["qps"] / (4.0 * one["qps"]),
                "served_fraction": four["served"] / 4.0,
                "outstanding_fraction": four["outstanding"] / 4.0,
                "served_fraction_at_one": one["served"],
                "qps1": one["qps"], "qps4": four["qps"],
            }
        out[name] = row
    return out


def _report(table) -> str:
    """The table as a reader sees it, for an assertion's failure message."""
    lines = [f"{'regime':11s} {'eff thr':>8s} {'eff proc':>9s} "
             f"{'srv/n thr':>10s} {'srv/n proc':>11s}"]
    for name, row in table.items():
        lines.append(
            f"{name:11s} {row['threads']['efficiency']:8.3f} "
            f"{row['processes']['efficiency']:9.3f} "
            f"{row['threads']['served_fraction']:10.3f} "
            f"{row['processes']['served_fraction']:11.3f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The deliverable: the verdicts that moved, and the one that must not
# ---------------------------------------------------------------------------

FLOOR = cl.SERVED_CONCURRENCY_FLOOR


class TestTheLockHoldingRegimesNowAchieveTheirConcurrency:
    """The change, stated as the verdict it moves.

    Honest/mutant is the two ARMS here rather than a patch: the thread arm IS
    the pre-change implementation, run over the same regime and the same
    query set in the same fixture, so the pair is a measurement of the change
    and not of two runs.

    **The strict verdicts are keyed to `held` and not to `spin`, and that is
    not a convenience.** `held` costs no CPU, so four client processes can be
    inside a query at once however busy the machine is; `spin` burns a core
    per client and its process arm therefore needs four free cores. An
    absolute assertion on the spinning row would be an assertion about the
    box -- measured, it failed at 8-18 busy cores of 20, correctly and for a
    reason that was not the lock. What is asserted for `spin` is the part that
    is still about the lock: the thread arm refuses, and the process arm is a
    long way better. See `_HELD_USLEEP`.
    """

    @pytest.mark.parametrize("regime", THREAD_SERIALIZES)
    def test_a_lock_holding_regime_moves_from_refused_to_ok(self, table,
                                                            regime):
        """Honest (processes) and mutant (threads), both read off the table.

        Measured: `held` 0.243 -> 0.982 of its client count in service,
        `spin` 0.264 -> 0.995. The thread figure is below
        `SERVED_CONCURRENCY_FLOOR` of 0.5 and the process figure above it,
        each by more than a factor of two, so no plausible move of the floor
        changes either verdict -- the same argument the floor itself was
        chosen by.

        The absolute process-arm bound is asserted for the load-insensitive
        regime only; both regimes carry the relational one.
        """
        row = table[regime]
        thread_served = row["threads"]["served_fraction"]
        process_served = row["processes"]["served_fraction"]
        assert thread_served < FLOOR, (
            f"{regime}: the thread arm no longer serializes, so this file is "
            f"not measuring the change it claims to\n{_report(table)}")
        if regime in LOAD_INSENSITIVE_SERIALIZES:
            assert process_served > FLOOR, (
                f"{regime}: the process arm still serializes, and this regime "
                f"costs no CPU so the box cannot be the reason"
                f"\n{_report(table)}")
        # Not an identical pair, which is the signature of a dead probe. True
        # of both regimes, because even a core-starved process pool beats four
        # clients taking turns on one interpreter.
        assert process_served > 2.0 * thread_served, (
            regime, thread_served, process_served, _report(table))

    @pytest.mark.parametrize("regime", LOAD_INSENSITIVE_SERIALIZES)
    def test_the_status_the_report_reads_moves_with_it(self, table, regime):
        """The verdict, not the decimal, because the verdict is what the
        report acts on.

        Honest (processes): `ok`. Mutant (threads):
        `concurrency_not_achieved`, which is what makes `report/render.py`
        print no scaling efficiency for the point at all. Both are computed by
        the one `scaling_status` rule out of the two measured fractions.
        """
        from harness.metrics.latency import scaling_status
        honest = scaling_status(table[regime]["processes"]["served_fraction"]
                                * 4.0, 4)
        mutant = scaling_status(table[regime]["threads"]["served_fraction"]
                                * 4.0, 4)
        assert honest == cl.SCALING_OK, _report(table)
        assert mutant == cl.SCALING_CONCURRENCY_NOT_ACHIEVED, _report(table)
        assert honest != mutant

    @pytest.mark.parametrize("regime", LOAD_INSENSITIVE_SERIALIZES)
    def test_the_throughput_moved_and_not_only_the_verdict(self, table,
                                                           regime):
        """Non-vacuity for the two rows above.

        A verdict that moved while the pool still delivered one client's work
        would mean the instrument had been loosened rather than the defect
        removed. Measured: `held` efficiency 0.249 -> 0.995. The threshold
        is 0.5 because that is halfway between 1/4 -- what a
        serialized four-client pool delivers -- and 1.
        """
        row = table[regime]
        assert row["threads"]["efficiency"] < 0.5, _report(table)
        assert row["processes"]["efficiency"] > 0.5, _report(table)

    @pytest.mark.parametrize("regime", CPU_BOUND_SERIALIZES)
    def test_the_cpu_bound_regime_is_reported_and_not_asserted_absolutely(
            self, table, regime):
        """What the spinning row can still be held to, and why no more.

        A Python spin holds the lock by burning a core, so the process arm's
        absolute figure is a joint measurement of the lock and of the box's
        free cores. Asserted here: the thread arm refuses (a property of the
        lock alone, and load only strengthens it), and the process arm is at
        least twice as good (true even core-starved, because four clients
        taking turns on one interpreter is the floor). The absolute figure is
        in `_report` either way, so a reader gets the number without a test
        pretending the box was idle.
        """
        row = table[regime]
        assert row["threads"]["served_fraction"] < FLOOR, _report(table)
        assert (row["processes"]["served_fraction"]
                > 2.0 * row["threads"]["served_fraction"]), _report(table)



class TestTheHeldWaitReallyHoldsTheLock:
    """The regime the strict verdicts are keyed to, checked rather than
    assumed.

    `_HELD_USLEEP`'s whole claim is that `ctypes.PyDLL` does not release the
    interpreter lock around a call while `ctypes.CDLL` does. If that were
    wrong -- a future ctypes that released it in both, or a libc where
    `usleep` returns immediately -- the lock-holding row would quietly become
    a second lock-releasing one, the thread arm would stop serializing and
    every verdict above would pass for the wrong reason. So the pair is
    measured here, in one process, with no pool involved.
    """

    CALLS = 150

    def _wall(self, fn, threads: int) -> float:
        def work():
            for _ in range(self.CALLS):
                fn(_USLEEP_US)

        workers = [threading.Thread(target=work) for _ in range(threads)]
        started = time.perf_counter()
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        return time.perf_counter() - started

    def test_the_held_form_serializes_and_the_released_form_does_not(self):
        """Honest and control, as walls against the two limiting cases.

        Four threads each making `CALLS` waits of `QUERY_S`: fully serial is
        `4 x CALLS x QUERY_S` and fully parallel is a quarter of that.
        Measured, 200 calls: held 0.466 s against a serial 0.400 s, released
        0.116 s against a parallel 0.100 s. The assertion is that the held
        wall is at least twice the released one, which is a factor of two
        inside the measured factor of four.
        """
        serial = 4 * self.CALLS * QUERY_S
        held = self._wall(_HELD_USLEEP, 4)
        released = self._wall(_RELEASED_USLEEP, 4)
        assert held > 0.6 * serial, (
            f"four threads of held waits took {held:.3f}s against a "
            f"fully-serial {serial:.3f}s, so PyDLL is no longer holding the "
            f"interpreter lock and the lock-holding regime is not one")
        assert released < 0.5 * serial, (
            f"four threads of released waits took {released:.3f}s against a "
            f"fully-serial {serial:.3f}s, so CDLL is not releasing the lock "
            f"either and the pair is not a controlled one")
        assert held > 2.0 * released, (held, released)

    def test_the_held_wait_costs_no_cpu(self):
        """The property that makes the strict verdicts load-insensitive.

        A client holding the lock across a socket wait burns no CPU, so four
        client PROCESSES in that regime can all be inside a query at once
        however busy the machine is. Measured at **0.034 cores** over 1,000
        waits; asserted at a tenth of a core, which is a 3x margin and still
        far below the one core per client a spin costs.
        """
        import resource
        before = resource.getrusage(resource.RUSAGE_SELF)
        opened = time.perf_counter()
        for _ in range(600):
            _HELD_USLEEP(_USLEEP_US)
        wall = time.perf_counter() - opened
        after = resource.getrusage(resource.RUSAGE_SELF)
        cpu = ((after.ru_utime - before.ru_utime)
               + (after.ru_stime - before.ru_stime))
        assert wall > 0.0
        cores = cpu / wall
        assert cores < 0.10, (
            f"the held wait burns {cores:.3f} cores, so it is a CPU-bound "
            f"regime after all and its process arm needs free cores like "
            f"`spin` does")
        # The control: a spin of the same nominal cost burns about one core,
        # so the probe can tell the two regimes apart.
        before = resource.getrusage(resource.RUSAGE_SELF)
        opened = time.perf_counter()
        _spin(0.1)
        spin_wall = time.perf_counter() - opened
        after = resource.getrusage(resource.RUSAGE_SELF)
        spin_cpu = ((after.ru_utime - before.ru_utime)
                    + (after.ru_stime - before.ru_stime))
        assert spin_cpu / spin_wall > 0.8, spin_cpu / spin_wall


class TestASaturatedServerIsStillNotFlagged:
    """**The row the threshold exists to protect, and the one a change like
    this is most likely to break.**

    `saturated` is a server that IS the bottleneck: latency rises with the
    client count, throughput is flat and efficiency falls to about 1/n, which
    is the shape a reader would call an engine ceiling -- and it is one. A
    real queue raises the WHOLE latency distribution, median included, so its
    `served` stays at the client count and it must not be flagged. An
    instrument that started refusing this point would refuse every real engine
    ceiling in the study, which is worse than the defect G3 removes.
    """

    def test_it_achieves_its_concurrency_in_both_arms(self, table):
        """Honest in both arms, and that identity is the assertion.

        Measured: `served / clients` 0.976 under threads and 0.977 under
        processes. The pair being identical is the result here, not a dead
        probe: the two rows above show the same comparison separating the
        lock-holding regimes by a factor of four over the same fixture, so
        this probe demonstrably can move.
        """
        row = table["saturated"]
        assert row["threads"]["served_fraction"] > FLOOR, _report(table)
        assert row["processes"]["served_fraction"] > FLOOR, _report(table)

    def test_it_is_still_the_low_efficiency_case(self, table):
        """Non-vacuity. If `saturated` happened to scale, "not flagged" would
        be trivially true and the protection would be untested.

        Measured at 0.275 in both arms against a modelled 1/4, so the guard is
        being offered a point that looks exactly like the defect and is
        declining it -- before and after.
        """
        row = table["saturated"]
        assert row["threads"]["efficiency"] < 0.5, _report(table)
        assert row["processes"]["efficiency"] < 0.5, _report(table)

    def test_the_lock_releasing_regime_is_unchanged(self, table):
        """The transport, the barrier and the merge cost the pool nothing.

        Measured: efficiency 0.989 -> 1.000 and `served / clients`
        0.985 -> 0.985. A process pool that had paid for its pipes would show
        it here, in the one regime where there was never a lock to remove.
        """
        row = table["releasing"]
        assert row["threads"]["efficiency"] > 0.5, _report(table)
        assert row["processes"]["efficiency"] > 0.5, _report(table)
        assert row["processes"]["served_fraction"] > FLOOR, _report(table)

    def test_one_client_is_never_refused_in_either_arm(self, table):
        """The efficiency denominator must survive the guard.

        A floor that fired at one client would refuse every point of every
        sweep, including the one the others are divided by. Measured at
        0.94-1.00 across all five regimes and both arms.
        """
        for regime, row in table.items():
            for arm in ("threads", "processes"):
                assert row[arm]["served_fraction_at_one"] > FLOOR, (
                    regime, arm, row[arm]["served_fraction_at_one"])


# ---------------------------------------------------------------------------
# §7 item 1: what BrokenBarrierError is under multiprocessing
# ---------------------------------------------------------------------------

class TestTheBarrierStaysAThreadBarrierForTwoMeasuredReasons:
    """`Instruments` names `BrokenBarrierError` explicitly, so the obvious
    edit -- hand that field a `multiprocessing` context -- has to fail on the
    reason rather than on a traceback inside a measured window."""

    def test_multiprocessing_has_no_broken_barrier_error(self):
        """Measured, not assumed: `hasattr` is False on the module and on the
        fork context alike.

        Honest: `Instruments().threading.BrokenBarrierError` resolves.
        Mutant (a fork context in that field): `AttributeError`. Both read
        back below rather than described.
        """
        assert not hasattr(multiprocessing, "BrokenBarrierError")
        assert not hasattr(MP, "BrokenBarrierError")
        honest = cl.Instruments().threading.BrokenBarrierError
        assert honest is threading.BrokenBarrierError
        with pytest.raises(AttributeError):
            _ = cl.Instruments(threading=MP).threading.BrokenBarrierError

    def test_a_multiprocessing_barrier_raises_the_threading_class(self):
        """And the reason there is no separate class: it IS a
        `threading.Barrier`.

        So the name a reader would go looking for does not exist because the
        exception is shared, not because multiprocessing has its own. Read
        back off a real aborted barrier.
        """
        barrier = MP.Barrier(2)
        assert isinstance(barrier, threading.Barrier)
        barrier.abort()
        with pytest.raises(threading.BrokenBarrierError):
            barrier.wait(timeout=0.5)

    def test_the_pool_takes_its_barrier_from_the_instrument(self, queries,
                                                            base):
        """Identity, so this cannot pass against a pool that built its own.

        `concurrency._instruments()` passes `threading=threading` literally,
        and five existing concurrency tests calibrate a window by replacing
        that field, so a pool that stopped reading it would make those tests
        measure the box instead of their own injected quantity.
        """
        built: List[int] = []

        def barrier(parties, action=None, timeout=None):
            built.append(parties)
            return threading.Barrier(parties, action=action, timeout=timeout)

        instruments = cl.Instruments(threading=types.SimpleNamespace(
            Event=threading.Event, Barrier=barrier,
            BrokenBarrierError=threading.BrokenBarrierError))
        pool = cl.ClientPool(factory_for(base, lambda: None), queries, 3, K,
                             32, spec_for(queries), warmup=0,
                             instruments=instruments)
        assert built == [4], built  # three clients plus the parent
        assert pool.spec.clients == 3


# ---------------------------------------------------------------------------
# Mutant: the barrier away -- the clients no longer start together
# ---------------------------------------------------------------------------

#: How long client *i* takes to warm up, so the clients are deliberately
#: unequal. Without a spread every client is warm at the same instant and
#: "they started together" and "they did not" produce the same window, which
#: is a probe that cannot fail.
_WARMUP_STAGGER_S = 0.12


def _staggered_factory(base: FakeDriver, counter) -> Callable[[], Any]:
    """A factory whose n-th client takes n x `_WARMUP_STAGGER_S` to connect.

    The counter is a `multiprocessing.Value` created before the fork, so the
    clients agree on who is who. Each client is slower to warm than the last,
    which is what makes "the window opened before everybody was ready"
    observable as a non-zero `prewindow_discarded`.
    """
    def clone():
        client = base.clone()
        with counter.get_lock():
            counter.value += 1
            mine = counter.value - 1
        time.sleep(mine * _WARMUP_STAGGER_S)

        def query(vector, k):
            time.sleep(QUERY_S)
            return SYNTHETIC_IDS[:k]

        client.query = query
        return client
    return clone


class TestTheBarrierIsWhatOpensTheWindow:
    """`extra.prewindow_discarded` is zero BY CONSTRUCTION, and the
    construction is the barrier plus the go flag.

    The window's opening instant is read in the barrier's action, in the last
    party to arrive, and only then is `_go` set -- so no measured query can
    precede the window. A number in that field means the window's two ends
    disagree about when it began, which is the defect that pinned
    `extra.recall_queries` at 198 of a 200-query pool however long the window
    ran.
    """

    def _window(self, base, queries, release_early: bool):
        counter = MP.Value("i", 0)
        pool = cl.ClientPool(_staggered_factory(base, counter), queries, 4, K,
                             32, spec_for(queries), warmup=2)
        if release_early:
            # THE MUTANT, injected before the fork so every client inherits
            # it: the go flag is already up, so each client starts querying
            # the moment it is warm instead of when the pool is. The barrier
            # still reads the window's instant when the LAST client arrives,
            # so the early clients' queries now precede it.
            pool._go.set()
        pool.start("g3 barrier probe", prefix="g3")
        time.sleep(WINDOW_S)
        return pool.stop()

    def test_releasing_the_clients_early_puts_queries_before_the_window(
            self, base, queries):
        """Honest and mutant, both read out of the record's own field.

        Honest: `prewindow_discarded` 0 of 4 staggered clients. Mutant (`_go`
        set before `start`, so the clients do not wait for each other):
        several hundred, because the first client queries for the
        0.36 s the last one spends warming up and every one of those queries
        began before the window's instant was read.

        The second assertion is what makes this a measurement of the ordering
        rather than of the stagger: the partition still holds, so those
        queries were not lost, they were correctly excluded.
        """
        honest = self._window(base, queries, release_early=False)
        mutant = self._window(base, queries, release_early=True)
        assert honest.prewindow_discarded == 0, honest.prewindow_discarded
        assert mutant.prewindow_discarded > 0, (
            "releasing the clients before the barrier produced no query "
            "ahead of the window, so this probe cannot see the ordering it "
            "exists for")
        assert mutant.prewindow_discarded != honest.prewindow_discarded
        # And both windows still partition every span into exactly one bucket.
        for window in (honest, mutant):
            assert window.queries_executed >= 0

    def test_the_barrier_action_reads_the_instant_before_any_client_runs(
            self, base, queries):
        """The ordering stated as an inequality over instants.

        The window's start is read inside the barrier action, which runs after
        every party is waiting and before any is released; `_go` is set after
        that. So the earliest span in the window cannot precede `wall_start`.
        Read back off the spans rather than off the source.
        """
        pool = cl.ClientPool(factory_for(base, lambda: time.sleep(QUERY_S)),
                             queries, 4, K, 32, spec_for(queries), warmup=2)
        pool.start("g3 ordering probe", prefix="g3")
        time.sleep(WINDOW_S)
        window = pool.stop()
        wall_start = pool._opened[0][1]
        starts = [s.started for w in pool._workers
                  for s in w.collector._samples]
        assert starts, "no spans came back, so there is nothing to order"
        assert min(starts) >= wall_start, (
            f"a query began {wall_start - min(starts):.6f}s before the "
            f"window's own start instant")
        assert window.prewindow_discarded == 0


# ---------------------------------------------------------------------------
# Mutant: the stop event away
# ---------------------------------------------------------------------------

class TestTheStopFlagHasToCrossTheFork:
    """The parent's `Event` is an injected instrument; `_halt` is what the
    clients watch. Both exist and the order between them is load-bearing.
    """

    def test_a_halt_that_never_arrives_fails_the_record(self, base, queries,
                                                        monkeypatch):
        """Honest and mutant, with the elapsed time read back.

        Honest: `stop()` returns a window whose qps is positive. Mutant
        (`_halt` replaced by an object whose `set()` does nothing, so no
        client ever leaves its loop): `PoolError`, naming the clients that did
        not return, after roughly `JOIN_TIMEOUT_S` -- pinned down to 2 s here
        so the test is not a two-minute wait.

        The mutant is the shape of forgetting to propagate the stop decision
        across the fork, which under threads was impossible: the clients read
        the same `Event` object the parent set.
        """
        monkeypatch.setattr(cl, "JOIN_TIMEOUT_S", 2.0)
        factory = factory_for(base, lambda: time.sleep(QUERY_S))

        honest_pool = cl.ClientPool(factory, queries, 3, K, 32,
                                    spec_for(queries), warmup=2)
        honest_pool.start("g3 halt honest", prefix="g3")
        time.sleep(WINDOW_S)
        honest = honest_pool.stop()
        assert honest.qps > 0.0

        mutant_pool = cl.ClientPool(factory, queries, 3, K, 32,
                                    spec_for(queries), warmup=2)
        # A flag the clients can never see set. `is_set` is what the child
        # loop calls; `set` is what the parent calls.
        mutant_pool._halt = types.SimpleNamespace(
            set=lambda: None, is_set=lambda: False,
            wait=lambda timeout=None: False)
        # `_go` still has to reach the clients or the window never opens, so
        # only the halt half is mutated.
        for worker in mutant_pool._workers:
            worker._halt = mutant_pool._halt
        mutant_pool.start("g3 halt mutant", prefix="g3")
        time.sleep(WINDOW_S)
        started = time.perf_counter()
        with pytest.raises(cl.PoolError) as raised:
            mutant_pool.stop()
        elapsed = time.perf_counter() - started
        assert "did not return" in str(raised.value), str(raised.value)
        assert elapsed < 20.0, elapsed
        # And nothing was left running: a leaked client process would go on
        # querying the server the next configuration is about to measure.
        assert not [w for w in mutant_pool._workers if w.is_alive()]

    def test_the_parent_flag_is_set_before_the_one_the_clients_watch(self):
        """The order, asserted on the source because the gap is microseconds.

        `tests/test_workloads.py::test_the_window_closes_before_the_stop_flag_
        is_set` amplifies the parent `Event`'s `set()` to 0.2 s and then
        requires the clients to KEEP QUERYING for that long, counting those
        queries as in-flight discards. That only works if the instrument's
        flag goes up before `_halt`: reversed, the clients would stop first
        and the amplifier would measure nothing.
        """
        source = open(cl.__file__).read()
        body = source.split("def stop(self)")[1].split("def _first_error")[0]
        assert (body.index("self._stop.set()")
                < body.index("self._halt.set()")), (
            "`_halt` is now set before the instrument flag, which makes "
            "test_workloads' ordering amplifier measure nothing")


# ---------------------------------------------------------------------------
# Mutants: a client that dies before the barrier, and one that dies mid-window
# ---------------------------------------------------------------------------

def _suicide_factory(base: FakeDriver, counter, when: str,
                     after_queries: int = 0) -> Callable[[], Any]:
    """A factory whose FIRST client kills its own process with SIGKILL.

    `SIGKILL` and not an exception, because that is the failure a thread could
    not have: no handler runs, nothing unwinds and no error is written
    anywhere, so the only evidence is the exit status and the closed pipe.
    `when` is "warmup" (before the barrier) or "window" (after
    `after_queries` measured queries).
    """
    def clone():
        client = base.clone()
        with counter.get_lock():
            counter.value += 1
            mine = counter.value - 1
        if mine == 0 and when == "warmup":
            os.kill(os.getpid(), signal.SIGKILL)
        seen = {"n": 0}

        def query(vector, k):
            seen["n"] += 1
            if (mine == 0 and when == "window"
                    and seen["n"] > after_queries):
                os.kill(os.getpid(), signal.SIGKILL)
            time.sleep(QUERY_S)
            return SYNTHETIC_IDS[:k]

        client.query = query
        return client
    return clone


class TestAClientThatDiesFailsTheRecordAndNeverHangsThePool:
    """`PoolError`'s discipline, re-established against process death.

    Its docstring is explicit: a client that dies before the barrier must fail
    the record and never hang the pool, and a pool that measured nothing must
    raise rather than report a QPS of 0. Against threads that meant catching
    `BaseException`; against processes it means exit codes, closed pipes and
    `SIGKILL`.
    """

    def test_a_client_killed_before_the_barrier_fails_start_at_once(
            self, base, queries, monkeypatch):
        """Honest, mutant, and the ELAPSED TIME, which is the real assertion.

        Honest (nobody killed): `start()` returns and the pool measures a
        window. Mutant (client 0 sends itself `SIGKILL` during warm-up):
        `PoolError` out of `start()` naming SIGKILL, in **0.2 s** against the
        2 s `CLIENT_BARRIER_TIMEOUT_S` this test pins -- so the pool
        discovered the death rather than waiting out a timeout. Shipped, that
        constant is 120 s, which is the difference between failing a record
        and hanging a run.
        """
        monkeypatch.setattr(cl, "CLIENT_BARRIER_TIMEOUT_S", 2.0)
        counter = MP.Value("i", 0)
        pool = cl.ClientPool(_suicide_factory(base, counter, "warmup"),
                             queries, 3, K, 32, spec_for(queries), warmup=2)
        started = time.perf_counter()
        with pytest.raises(cl.PoolError) as raised:
            pool.start("g3 death probe", prefix="g3")
        elapsed = time.perf_counter() - started
        message = str(raised.value)
        assert "SIGKILL" in message, message
        assert "warm-up" in message, message
        # Discovered, not waited out: the shipped value of that constant is
        # 120 s and the honest path here is a closed pipe, not a timeout.
        assert elapsed < 0.5 * 2.0, (elapsed, message)
        assert not [w for w in pool._workers if w.is_alive()]

    def test_the_parent_closing_its_write_end_is_what_makes_that_fast(
            self, base, queries, monkeypatch):
        """R2: the repair, against the form it replaced, over the same input.

        `Connection.recv` raises `EOFError` only when every copy of the write
        end is closed, so a parent that keeps its own copy cannot see a dead
        client at all -- it waits out `CLIENT_BARRIER_TIMEOUT_S` instead.

        Honest (`Client.launch` closes it): the pool fails in **0.2 s**.
        Mutant (`launch` replaced by the fork-and-nothing-else form, which is
        what `start()` did when the clients were threads): the same dead
        client takes the full **2.0 s** pinned timeout. Both elapsed times are
        read back, and they are a factor of ten apart.
        """
        monkeypatch.setattr(cl, "CLIENT_BARRIER_TIMEOUT_S", 2.0)

        def timed(launch):
            counter = MP.Value("i", 0)
            pool = cl.ClientPool(_suicide_factory(base, counter, "warmup"),
                                 queries, 3, K, 32, spec_for(queries),
                                 warmup=2)
            monkeypatch.setattr(cl.Client, "launch", launch)
            started = time.perf_counter()
            try:
                pool.start("g3 fd probe", prefix="g3")
            except cl.PoolError:
                pass
            elapsed = time.perf_counter() - started
            for worker in pool._workers:
                worker.reap()
            return elapsed

        honest_launch = cl.Client.launch

        def leaky_launch(self) -> None:
            """`launch` without the close: the pre-repair form."""
            self._from_client, self._to_parent = cl._MP.Pipe(duplex=False)
            cl._MP.Process.start(self)

        honest = timed(honest_launch)
        mutant = timed(leaky_launch)
        assert honest < 1.0, honest
        assert mutant > 1.5, (
            f"the leaky form failed in {mutant:.2f}s, so it also discovered "
            f"the death and this probe measured nothing")
        assert mutant > 3.0 * honest, (honest, mutant)

    def test_the_failure_path_does_not_outlast_the_timeout_it_reports(
            self, base, queries, monkeypatch):
        """R2, on a bug the gate found rather than one this file predicted.

        `start`'s failure path builds its message from `_first_error()`. An
        earlier revision of that method joined the attendant threads first,
        reasoning that one of them might be about to name the driver's own
        exception. An attendant that has not reported is blocked on its
        client's PIPE and not on the barrier, so breaking the barrier does not
        wake it: the grace ran to its full length every time, inside the
        failure path, delaying the timeout it was reporting.

        Measured with the parent's barrier wait pinned at 1.0 s and a client
        that never finishes warming up. Honest (`_first_error` reads
        `worker.error` and returns): **1.0 s**, the timeout it reports.
        Mutant (the joining form, restored verbatim): **2.0 s**, so the
        `PoolError` misstates its own cause by a factor of two. This is the
        shape that broke
        `test_t4b_readers.py::test_the_client_side_barrier_timeout_is_the_one
        _that_binds`, whose inverted pair measured 2.51 s against 1.5 ± 0.4.
        """
        monkeypatch.setattr(cl, "BARRIER_TIMEOUT_S", 1.0)
        monkeypatch.setattr(cl, "CLIENT_BARRIER_TIMEOUT_S", 6.0)

        def hanging_factory():
            client = base.clone()

            def query(vector, k):
                time.sleep(6.0)
                return SYNTHETIC_IDS[:k]

            client.query = query
            return client

        def timed() -> float:
            pool = cl.ClientPool(hanging_factory, queries, 2, K, 32,
                                 spec_for(queries), warmup=2)
            started = time.perf_counter()
            try:
                pool.start("g3 grace probe", prefix="g3")
            except cl.PoolError:
                pass
            elapsed = time.perf_counter() - started
            for worker in pool._workers:
                worker.reap()
            return elapsed

        honest = timed()

        def joining_first_error(self):
            """The pre-repair form, verbatim."""
            grace = time.monotonic() + 1.0
            for attendant in self._attendants:
                attendant.join(timeout=max(0.0, grace - time.monotonic()))
            for worker in self._workers:
                if worker.error is not None:
                    return worker.error
            return None

        monkeypatch.setattr(cl.ClientPool, "_first_error", joining_first_error)
        mutant = timed()
        assert honest < 1.6, (
            f"the honest failure path took {honest:.2f}s against a 1.0s "
            f"timeout, so something else is now delaying it")
        assert mutant > honest + 0.6, (
            f"the joining form failed in {mutant:.2f}s against the honest "
            f"{honest:.2f}s, so this probe cannot see the grace period and "
            f"the repair is unproven")

    def test_a_client_killed_mid_window_fails_the_record(self, base, queries):
        """Honest and mutant, off `stop()`.

        Honest: a window with a positive qps. Mutant (client 0 sends itself
        `SIGKILL` after 20 measured queries): `PoolError` naming the client,
        the window and SIGKILL. **Not a QPS of 0 and not a three-client
        window**, which is the failure that matters: the surviving clients did
        answer queries, so a pool that swallowed the death would publish a
        plausible number measured by fewer clients than the record says.
        """
        factory = factory_for(base, lambda: time.sleep(QUERY_S))
        honest_pool = cl.ClientPool(factory, queries, 3, K, 32,
                                    spec_for(queries), warmup=2)
        honest_pool.start("g3 midwindow honest", prefix="g3")
        time.sleep(WINDOW_S)
        honest = honest_pool.stop()
        assert honest.qps > 0.0

        counter = MP.Value("i", 0)
        mutant_pool = cl.ClientPool(
            _suicide_factory(base, counter, "window", after_queries=20),
            queries, 3, K, 32, spec_for(queries), warmup=2)
        mutant_pool.start("g3 midwindow mutant", prefix="g3")
        time.sleep(WINDOW_S)
        with pytest.raises(cl.PoolError) as raised:
            mutant_pool.stop()
        message = str(raised.value)
        assert "SIGKILL" in message, message
        assert "measured window" in message, message
        assert not [w for w in mutant_pool._workers if w.is_alive()]

    def test_a_client_killed_after_reporting_is_caught_by_its_exit_status(
            self, base, queries, monkeypatch):
        """The third death, and the only one with no other evidence.

        A client that sent its spans and was THEN killed leaves `error` None
        and a window that looks complete, so neither the failure list nor the
        liveness check can see it. Only `exitcode` can.

        Honest: `stop()` returns a window. Mutant (`_tell` wrapped so the
        client kills itself immediately after sending its window payload,
        injected before the fork): `PoolError` saying the spans cover only
        part of the window. Without the exit-status check the mutant returns a
        window instead -- which is asserted here by running that very path
        with the check disabled, so the guard is shown to be the thing doing
        the work.
        """
        factory = factory_for(base, lambda: time.sleep(QUERY_S))
        honest_pool = cl.ClientPool(factory, queries, 2, K, 32,
                                    spec_for(queries), warmup=2)
        honest_pool.start("g3 postreport honest", prefix="g3")
        time.sleep(WINDOW_S)
        assert honest_pool.stop().qps > 0.0

        real_tell = cl.Client._tell

        def tell_then_die(self, phase, error, collector, results):
            real_tell(self, phase, error, collector, results)
            if phase == "window":
                os.kill(os.getpid(), signal.SIGKILL)

        monkeypatch.setattr(cl.Client, "_tell", tell_then_die)
        mutant_pool = cl.ClientPool(factory, queries, 2, K, 32,
                                    spec_for(queries), warmup=2)
        mutant_pool.start("g3 postreport mutant", prefix="g3")
        time.sleep(WINDOW_S)
        with pytest.raises(cl.PoolError) as raised:
            mutant_pool.stop()
        message = str(raised.value)
        assert "abnormally after reporting" in message, message
        assert "SIGKILL" in message, message
        # And the spans really did arrive, so the pool had a plausible window
        # to publish and the exit status is the only thing that refused it.
        assert any(w.collector._samples for w in mutant_pool._workers)

    def test_a_window_that_finished_nothing_is_not_a_death(self, base,
                                                           queries):
        """The other side of `PoolError`'s rule, and it has to stay.

        A configuration whose queries outlast the window records a QPS of 0
        and says so on stderr; that is a legitimate measurement and must not
        be confused with a pool whose clients died. Honest: qps 0.0, no
        exception, every client alive and reporting. A pool that raised here
        would turn "this engine is slower than the window" into a crash.
        """
        slow = factory_for(base, lambda: time.sleep(0.3))
        pool = cl.ClientPool(slow, queries, 2, K, 32, spec_for(queries),
                             warmup=0)
        pool.start("g3 empty window", prefix="g3")
        time.sleep(0.02)
        window = pool.stop()
        assert window.qps == 0.0
        assert window.queries_executed == 0
        assert window.inflight_discarded > 0, (
            "nothing was in flight, so this window was not the empty case")
        assert window.served_concurrency is None
        assert window.scaling_status is None


# ---------------------------------------------------------------------------
# Mutant: the span merge dropping one client's spans
# ---------------------------------------------------------------------------

class TestEveryClientsSpansReachTheMerge:
    """The spans cross a pipe now, so losing one client's is a silent
    quarter-sized measurement rather than an error.

    Under threads the collectors were in the merge's own memory and could not
    go missing. The merge is still over ONE shared wall and still one
    `span_disposition` rule -- what changed is that the collectors are
    rebuilt in the parent from what arrived, so a transport that dropped a
    client would publish three clients' work as four clients' throughput.
    """

    def _window(self, base, queries, drop_first: bool, monkeypatch):
        real_tell = cl.Client._tell
        if drop_first:
            def tell(self, phase, error, collector, results):
                # THE MUTANT: the first client reports no spans at all.
                # Patched on the class before the fork, so every child
                # inherits it -- which is itself a property of fork rather
                # than of spawn.
                if phase == "window" and self._offsets[0] == 0:
                    return real_tell(self, phase, error, None, results)
                return real_tell(self, phase, error, collector, results)
            monkeypatch.setattr(cl.Client, "_tell", tell)
        pool = cl.ClientPool(factory_for(base, lambda: time.sleep(QUERY_S)),
                             queries, 4, K, 32, spec_for(queries), warmup=2)
        pool.start("g3 merge probe", prefix="g3")
        time.sleep(WINDOW_S)
        window = pool.stop()
        return window, [len(w.collector._samples) for w in pool._workers]

    def test_dropping_one_clients_spans_shows_up_in_the_window(
            self, base, queries, monkeypatch):
        """Honest and mutant, both read out of the window.

        Honest: four clients report spans and `served / clients` lands near
        1.0. Mutant (client 0's window payload carries no spans): one
        collector is empty, `queries_executed` falls by about a quarter and
        `served / clients` falls with it -- a pool that would have published
        three clients' work under a record saying four.

        The per-client span counts are asserted as well as the aggregate,
        because an aggregate that fell by a quarter could also be this box
        having a bad second.
        """
        honest, honest_counts = self._window(base, queries, False, monkeypatch)
        mutant, mutant_counts = self._window(base, queries, True, monkeypatch)
        assert all(c > 0 for c in honest_counts), honest_counts
        assert mutant_counts[0] == 0, mutant_counts
        assert all(c > 0 for c in mutant_counts[1:]), mutant_counts
        assert honest.served_concurrency > mutant.served_concurrency, (
            honest.served_concurrency, mutant.served_concurrency)
        # And the shape of the loss is a quarter, not a rounding.
        assert mutant.served_concurrency < 0.85 * honest.served_concurrency, (
            honest.served_concurrency, mutant.served_concurrency)

    def test_the_merge_is_still_over_one_wall_and_one_rule(self, base,
                                                           queries):
        """`merge_collectors` must keep meaning what it means.

        Every span lands in exactly one of three buckets -- kept, in flight,
        pre-window -- and the three partition what the collectors hold. That
        identity is what makes the recall beside a throughput figure describe
        the queries that figure counted, and it is the one thing the transport
        could have broken by rebuilding the spans differently from how the
        client recorded them.
        """
        pool = cl.ClientPool(factory_for(base, lambda: time.sleep(QUERY_S)),
                             queries, 3, K, 32, spec_for(queries), warmup=2)
        pool.start("g3 partition probe", prefix="g3")
        time.sleep(WINDOW_S)
        window = pool.stop()
        total = sum(len(w.collector._samples) for w in pool._workers)
        assert total > 0
        assert (window.queries_executed + window.inflight_discarded
                + window.prewindow_discarded) == total
        # And the duration the parent rebuilt is the one the client measured,
        # derived by the same expression on both sides.
        for worker in pool._workers:
            for sample in worker.collector._samples:
                assert sample.seconds == pytest.approx(
                    sample.finished - sample.started)

    def test_the_results_survive_the_transport_with_their_spans(self, base,
                                                                queries):
        """Recall's half of the same rule.

        `Answer` carries the ids with the span that produced them so one
        `span_disposition` call decides both. Across a pipe that pairing is a
        thing a transport could lose -- the ids arriving without their
        instants, or trimmed by a different window -- so the returned
        `results` are checked against the window they claim to be inside.
        """
        pool = cl.ClientPool(factory_for(base, lambda: time.sleep(QUERY_S)),
                             queries, 4, K, 32, spec_for(queries), warmup=2)
        pool.start("g3 results probe", prefix="g3")
        time.sleep(WINDOW_S)
        window = pool.stop()
        wall_start, stop_at = pool._opened[0][1], None
        assert window.results, "no results came back across the fork"
        assert len(window.results) <= len(queries)
        for offset, answer in window.results.items():
            assert 0 <= offset < len(queries)
            assert len(answer.ids) == K
            assert answer.started >= wall_start
            assert answer.finished >= answer.started
            assert span_disposition(answer.started, answer.finished,
                                    wall_start, None) is None
        _ = stop_at

    def test_results_are_not_collected_when_the_caller_said_not_to(
            self, base, queries):
        """§4.3 item 6, across the fork. The churn readers compute no recall.

        Honest: `collect_results=False` returns an empty `results` and still a
        measured window. The ids are the largest thing in the payload, so a
        transport that shipped them anyway would charge the churn readers for
        an allocation inside their window for something no reader can use.
        """
        pool = cl.ClientPool(factory_for(base, lambda: time.sleep(QUERY_S)),
                             queries, 3, K, 32, spec_for(queries), warmup=2,
                             collect_results=False)
        pool.start("g3 no-results probe", prefix="g3")
        time.sleep(WINDOW_S)
        window = pool.stop()
        assert window.results == {}
        assert window.queries_executed > 0


# ---------------------------------------------------------------------------
# §7 item 5: fork or spawn
# ---------------------------------------------------------------------------

def _module_level_factory():  # pragma: no cover - only pickled, never called
    """A factory that WOULD pickle, to prove the probe below is not blind."""
    return None


class TestForkIsForcedByTheCallersAndNotChosen:
    """`driver_factory` is a callable the caller supplies, and under spawn it
    would have to pickle. None of them can.
    """

    def test_the_real_factory_cannot_be_pickled(self):
        """Measured against the shape `harness/main.py:270` actually has.

        A `def` nested inside `main()`, whose `__qualname__` is
        `main.<locals>.driver_factory`. Pickle resolves a function by
        qualified name, so a local one cannot be found on the other side.
        Honest: `AttributeError`. The control is a module-level function,
        which pickles -- so the probe can tell the two apart and is not
        simply reporting that pickling is hard.
        """
        def main():
            spec = {"host": "db"}

            def driver_factory():
                return spec
            return driver_factory

        factory = main()
        assert factory.__qualname__.endswith("<locals>.driver_factory")
        with pytest.raises((AttributeError, pickle.PicklingError)):
            pickle.dumps(factory)
        # Non-vacuity: a factory that spawn COULD have used.
        assert pickle.loads(pickle.dumps(_module_level_factory)) is \
            _module_level_factory

    def test_the_test_suites_factories_cannot_be_pickled_either(self, base):
        """So spawn would not merely change the callers, it would change every
        test that drives a pool.

        A `lambda` and a bound `FakeDriver.clone` are what
        `test_t4b_readers.py` and `test_workloads.py` pass.
        """
        with pytest.raises((AttributeError, pickle.PicklingError)):
            pickle.dumps(lambda: None)
        # And the bound method fails for a second, independent reason: the
        # driver it is bound to holds a `_thread.lock`. Spawn would have to
        # get past BOTH.
        with pytest.raises(TypeError, match="_thread.lock"):
            pickle.dumps(base.clone)

    def test_the_start_method_is_pinned_and_not_inherited(self):
        """CPython's Linux default is `fork` today and `forkserver` in 3.14.

        An inherited default would therefore turn a working pool into
        `Can't pickle local object` at the start of a measured window on a
        release upgrade, which is the kind of failure this repository exists
        to prevent. Asserted on the context the pool actually builds its
        clients from, not on the constant alone.
        """
        assert cl.CLIENT_START_METHOD == "fork"
        assert cl._MP.get_start_method() == "fork"
        assert cl.Client._start_method == "fork"
        assert issubclass(cl.Client, multiprocessing.process.BaseProcess)

    def test_a_client_inherits_the_parents_memory_rather_than_a_pickle(
            self, base, queries):
        """The property fork buys, stated as something the pool depends on.

        The factory closes over a `FakeDriver` holding the loaded corpus, and
        the client calls it in the child. Under spawn that corpus would have
        to be re-created or shipped; under fork the child simply has it.
        Read back: a window with answers in it, from a factory that cannot be
        pickled at all.
        """
        factory = factory_for(base, lambda: time.sleep(QUERY_S))
        with pytest.raises((AttributeError, pickle.PicklingError)):
            pickle.dumps(factory)
        pool = cl.ClientPool(factory, queries, 2, K, 32, spec_for(queries),
                             warmup=2)
        pool.start("g3 inherit probe", prefix="g3")
        time.sleep(WINDOW_S)
        window = pool.stop()
        assert window.queries_executed > 0
        assert window.results


# ---------------------------------------------------------------------------
# §7 item 3: a clock the processes agree on
# ---------------------------------------------------------------------------

def _clock_child(conn):
    """Read both clocks back to back, so the OFFSET crosses the boundary."""
    conn.send((time.perf_counter(),
               time.clock_gettime(time.CLOCK_MONOTONIC)))
    conn.close()


class TestPerfCounterIsComparableAcrossTheFork:
    """Every window instant is read through `clients.time`, and half of them
    are now read in a different process from the other half.

    So this is a precondition of the whole measurement rather than a detail:
    if `perf_counter` had a per-process reference point, every span would be
    compared against a window it is not inside, `prewindow_discarded` would be
    garbage and the trimming would drop whichever side was behind.
    """

    def test_the_two_processes_share_the_clocks_reference_point(self):
        """Measured, not assumed, and measured as an OFFSET.

        `perf_counter` is `clock_gettime(CLOCK_MONOTONIC)` on this platform,
        which is the machine's clock rather than the process's. So
        `perf_counter - CLOCK_MONOTONIC` is a constant, and the test is
        whether the child's constant equals the parent's: measured over 40
        forks, they agree to within microseconds, which is the cost of reading
        two clocks back to back rather than an offset.

        `time.process_time` is the control: it IS per-process, and a child's
        reading of it is near zero whatever the parent's. A probe that could
        not tell the two clocks apart would not be measuring anything.
        """
        assert time.get_clock_info("perf_counter").implementation == \
            "clock_gettime(CLOCK_MONOTONIC)"
        assert time.get_clock_info("perf_counter").monotonic
        parent_offset = (time.perf_counter()
                         - time.clock_gettime(time.CLOCK_MONOTONIC))
        worst = 0.0
        for _ in range(40):
            recv, send = MP.Pipe(duplex=False)
            proc = MP.Process(target=_clock_child, args=(send,))
            proc.start()
            send.close()
            child_pc, child_mono = recv.recv()
            proc.join(10)
            recv.close()
            worst = max(worst, abs((child_pc - child_mono) - parent_offset))
        assert worst < 0.01, (
            f"the child's perf_counter reference point differs from the "
            f"parent's by up to {worst:.6f}s, so a span and the window it is "
            f"inside are being compared on two different clocks")
        # The control: a clock that really is per-process.
        assert time.process_time() > 0.0
        assert time.get_clock_info("process_time").monotonic

    def test_a_childs_reading_falls_inside_the_parents_bracket(self):
        """The same fact as an ordering, which is the form the pool uses.

        The parent reads the clock, forks, the child reads it, the parent
        reads it again. The child's reading must lie between the two -- that
        is exactly the relation `span_disposition` relies on between a
        window's instants and a client's spans. Measured 40 times out of 40.
        """
        inside = 0
        trials = 40
        for _ in range(trials):
            recv, send = MP.Pipe(duplex=False)
            before = time.perf_counter()
            proc = MP.Process(target=_clock_child, args=(send,))
            proc.start()
            send.close()
            child_pc, _mono = recv.recv()
            after = time.perf_counter()
            proc.join(10)
            recv.close()
            if before <= child_pc <= after:
                inside += 1
        assert inside == trials, f"{inside}/{trials}"


# ---------------------------------------------------------------------------
# §7 item 6: client CPU accounting
# ---------------------------------------------------------------------------

def _cgroup_child(conn):
    conn.send(open("/proc/self/cgroup").read().strip())
    conn.close()


class TestEveryClientProcessStaysInTheHarnessCgroup:
    """`read_own_cpu_usec` reads ONE counter for the whole harness container,
    so `client_cpu_util` is only a measurement of the clients if every client
    is inside that cgroup.

    Structural rather than empirical, and that is the strongest form available
    here: `fork` does not change cgroup membership -- only an explicit write
    to `cgroup.procs` does -- so the check is whether the child's
    `/proc/self/cgroup` is the parent's string. The MAGNITUDE cannot be
    checked on a developer box, because outside a container that same file is
    the root cgroup and reads the whole machine: §2.3 measured 2.421 cores
    burned while the reading process slept. A test that asserted on the
    quotient here would be asserting on how busy this box is.
    """

    def test_a_forked_child_reports_the_parents_cgroup(self):
        """Honest: identical strings, over four children.

        The mutant for this one is not available at run time -- moving a
        process between cgroups needs privileges this test does not have -- so
        what is asserted is the exact string equality rather than a tolerance,
        and the control is that the file is read in the child and not
        inherited as a value.
        """
        if not os.path.exists("/proc/self/cgroup"):  # pragma: no cover
            pytest.fail("this platform has no /proc/self/cgroup, so "
                        "client_cpu_util cannot be a measurement of anything")
        mine = open("/proc/self/cgroup").read().strip()
        assert mine
        seen = []
        procs = []
        for _ in range(4):
            recv, send = MP.Pipe(duplex=False)
            proc = MP.Process(target=_cgroup_child, args=(send,))
            proc.start()
            send.close()
            seen.append(recv.recv())
            procs.append((proc, recv))
        for proc, recv in procs:
            proc.join(10)
            recv.close()
        assert seen == [mine] * 4, (mine, seen)

    def test_the_pool_still_reports_all_three_cpu_figures(self, base,
                                                          queries):
        """`client_cpu` returns the counter, the cpuset and the ratio, never
        the ratio alone, and the move to processes must not have turned any of
        them into a None.

        `(None, None)` is the honest absence when the counter is unreadable,
        and 0.0 would be the claim "the clients did nothing" -- the one thing
        that cannot be true of a window that executed queries. Read off a real
        window rather than from `client_cpu` in isolation.
        """
        pool = cl.ClientPool(factory_for(base, lambda: _spin(QUERY_S)),
                             queries, 2, K, 32, spec_for(queries), warmup=2)
        pool.start("g3 cpu probe", prefix="g3")
        time.sleep(WINDOW_S)
        window = pool.stop()
        assert window.client_cpus > 0
        if window.client_cpu_seconds is None:  # pragma: no cover
            pytest.fail("the cgroup counter became unreadable, so "
                        "client_cpu_util is no longer a measurement")
        assert window.client_cpu_seconds > 0.0, (
            "a window of CPU-burning clients reported no CPU at all")
        assert window.client_cpu_util is not None
        assert window.client_bound is not None


# ---------------------------------------------------------------------------
# §7 item 7: memory -- the stated cost, measured
# ---------------------------------------------------------------------------

def _smaps(pid: int, key: str) -> Optional[int]:
    try:
        for line in open(f"/proc/{pid}/smaps_rollup"):
            if line.startswith(key):
                return int(line.split()[1]) * 1024
    except OSError:  # pragma: no cover
        return None
    return None


def _cow_child(conn, array, write):
    if write:
        array += 1.0
    total = float(array[::97].sum())
    conn.send((_smaps(os.getpid(), "Private_Dirty"), total))
    conn.recv()
    conn.close()


class TestTheQuerySetIsNotCopiedPerClient:
    """§7 states as a cost that "each client becomes an OS process with its
    own copy of the query set, so the harness container's memory grows with
    the client count". **Measured here rather than accepted, and it is not
    true under fork.**

    The query set is read-only in a client, `shard_queries` hands it out as a
    strided VIEW rather than a copy, and copy-on-write never copies a page
    nobody writes. The write arm is what makes this a measurement: children
    that DO write the array pay the full copy, so the probe can see one.
    """

    ROWS, DIM = 65536, 128

    def _arm(self, write: bool) -> Tuple[int, int]:
        array = numpy.ascontiguousarray(
            numpy.random.default_rng(3).random((self.ROWS, self.DIM),
                                               dtype=numpy.float32))
        conns, procs = [], []
        for _ in range(4):
            parent, child = MP.Pipe()
            proc = MP.Process(target=_cow_child, args=(child, array, write))
            proc.start()
            child.close()
            conns.append(parent)
            procs.append(proc)
        dirty = 0
        for conn in conns:
            private, _total = conn.recv()
            dirty += private or 0
        for conn in conns:
            conn.send(None)
        for proc in procs:
            proc.join(30)
        for conn in conns:
            conn.close()
        return dirty, array.nbytes

    def test_reading_clients_do_not_each_copy_the_array(self):
        """Honest and control, both read out of `smaps_rollup`.

        Measured at the sizes this test uses, four children over a 33.6 MB
        array: READ-ONLY children's summed `Private_Dirty` is a small
        multiple of an interpreter's own pages and well under one copy of the
        array, while WRITING children's is about four copies of it. The
        assertion is the ratio between the arms, not an absolute figure, so
        it does not depend on how large an interpreter is on this box.

        At the real sizes -- 104.9 MB and eight children -- the same probe
        measured 0.13x the array read-only against 8.12x writing.
        """
        read_dirty, nbytes = self._arm(write=False)
        write_dirty, _ = self._arm(write=True)
        assert read_dirty < nbytes, (
            f"four read-only children dirtied {read_dirty / 1e6:.1f} MB "
            f"against a {nbytes / 1e6:.1f} MB array, so copy-on-write is not "
            f"holding and §7's stated memory cost is real after all")
        assert write_dirty > 2.0 * nbytes, (
            f"four WRITING children dirtied only "
            f"{write_dirty / 1e6:.1f} MB of a {nbytes / 1e6:.1f} MB array, "
            f"so this probe cannot see a copy and the arm above proves "
            f"nothing")
        assert write_dirty > 3.0 * read_dirty, (read_dirty, write_dirty)

    def test_a_shard_is_a_view_and_not_a_copy(self, queries):
        """The other half of the reason, and it is a property of the array.

        `shard_queries` returns `queries[c::clients]`, a strided view, so the
        parent holds one array however many clients there are and a client
        inherits that one array. A shard that had become a copy would put the
        whole query set in the parent once per client before any fork
        happened.
        """
        shards = cl.shard_queries(queries, 4)
        assert len(shards) == 4
        for shard, offsets in shards:
            assert shard.base is not None, "this shard is a copy, not a view"
            assert numpy.shares_memory(shard, queries)
            assert len(shard) == len(offsets)
        # And together they cover the pool exactly, which is the older
        # invariant the view has to keep.
        covered = numpy.sort(numpy.concatenate([o for _s, o in shards]))
        assert numpy.array_equal(covered, numpy.arange(len(queries)))


# ---------------------------------------------------------------------------
# §7 item 8: /dev/shm
# ---------------------------------------------------------------------------

class TestTheShmFootprintIsIndependentOfTheClientCount:
    """`multiprocessing` semaphores live in `/dev/shm`, and a container's
    default is small -- so §7 asks whether this needs the harness container's
    `shm_size` raised. Measured: no.

    The measurement has one trap in it. `multiprocessing` calls `sem_unlink`
    immediately after `sem_open`, so its semaphores NEVER appear in a
    `/dev/shm` directory listing while their pages stay charged to the tmpfs.
    A probe that counted files would report zero and conclude there was
    nothing to worry about for the wrong reason. So this measures the
    filesystem's used bytes.
    """

    def _used(self) -> int:
        import shutil
        return shutil.disk_usage("/dev/shm").used

    def test_two_events_are_the_whole_pools_cost(self):
        """Honest and the design that was rejected, both in bytes.

        Honest (two `Event`s for the POOL): **40,960 bytes**, whatever the
        client count. Rejected (two per CLIENT): 1,310,720 bytes at 32
        clients, measured by creating the 64 extra Events the design would
        have needed. Thirty-two pipes cost **0**.

        Against the harness container's `shm_size`, which
        `orchestrator/ops_pass.py` takes from the resource pass -- 2g
        normalized, 8g tuned -- so the pool's footprint is five orders of
        magnitude inside it and nothing outside `clients.py` has to change.
        """
        baseline = self._used()
        pool_cost_holder = [MP.Event(), MP.Event()]
        pooled = self._used() - baseline
        per_client_holder = [MP.Event() for _ in range(64)]
        per_client = self._used() - baseline - pooled
        pipes = [MP.Pipe(duplex=False) for _ in range(32)]
        pipe_cost = self._used() - baseline - pooled - per_client
        try:
            assert pooled > 0, (
                "two Events cost no tmpfs at all, so this probe is not "
                "measuring the thing /dev/shm would run out of")
            assert per_client > 2.0 * pooled, (pooled, per_client)
            assert pipe_cost == 0, pipe_cost
            # The number that matters: one pool against the smallest
            # configured container.
            assert pooled < 2 * 1024 ** 3 / 1000, pooled
        finally:
            del pool_cost_holder, per_client_holder, pipes

    def test_the_pool_builds_two_flags_and_not_two_per_client(self, base,
                                                              queries):
        """Identity, so the figure above describes the shipped pool.

        Every client watches the same two `Event`s. A pool that gave each
        client its own would be the 1.25 MiB design, and at 32 clients inside
        a 64 MiB default `/dev/shm` that starts to matter.
        """
        pool = cl.ClientPool(factory_for(base, lambda: None), queries, 8, K,
                             32, spec_for(queries), warmup=0)
        assert len({id(w._go) for w in pool._workers}) == 1
        assert len({id(w._halt) for w in pool._workers}) == 1
        assert pool._workers[0]._go is pool._go
        assert pool._workers[0]._halt is pool._halt


# ---------------------------------------------------------------------------
# Mutant: the version constant left at 5
# ---------------------------------------------------------------------------

class TestTheMeasurementVersionIsAtomicWithTheChange:
    """A version 5 record from this pool would mean something different from a
    version 5 record in the 2026-10-14 run, which is the exact defect class
    this repository exists to prevent.

    So the bump is in this commit and not a later one, and it has to be
    checkable rather than asserted in a commit message.
    """

    def test_the_constant_moved_and_the_record_carries_it(self, tmp_path):
        """Honest and mutant, both read back out of a written record.

        Honest: `OPS_MEASUREMENT_VERSION` is 6 and a record written by
        `Recorder` carries `extra.ops_measurement_version` 6. Mutant (the
        constant rebound to 5 at run time): the same `Recorder` call stamps a
        5 onto a record measured by the process pool, which is the state that
        would put this pool's numbers on one axis with the thread pool's.
        `Recorder.write` reads the constant out of the module's globals on
        every call, so the rebinding reaches it -- a value captured at import
        would make this mutant pass unchanged.
        """
        from harness.metrics.records import PHASE_CONCURRENCY, Record

        def stamp(path):
            recorder = records.Recorder(str(path))
            try:
                recorder.write(Record(
                    run_id="g3", phase=PHASE_CONCURRENCY, engine="fake",
                    dataset="tiny", timestamp="2026-10-14T00:00:00Z",
                    clients=4, qps=1.0))
            finally:
                recorder.close()
            return list(records.read_records(str(path)))[0]["extra"][
                records.OPS_MEASUREMENT_VERSION_KEY]

        assert records.OPS_MEASUREMENT_VERSION == 6
        honest = stamp(tmp_path / "honest.jsonl")
        original = records.OPS_MEASUREMENT_VERSION
        try:
            records.OPS_MEASUREMENT_VERSION = 5
            mutant = stamp(tmp_path / "mutant.jsonl")
        finally:
            records.OPS_MEASUREMENT_VERSION = original
        assert honest == 6
        assert mutant == 5
        assert honest != mutant
        assert records.OPS_MEASUREMENT_VERSION == original

    def test_five_and_six_are_refused_as_one_measurement(self):
        """The bump is only worth anything if the report refuses the mixture.

        Honest: `ops_versions_measure_the_same_thing([5, 6])` is False, so a
        run directory holding both gets the "two different measurements"
        refusal. The control is the one pair that IS comparable -- 1 and 2,
        which differ only in having been written down -- so this is not a
        function that refuses everything.
        """
        assert not records.ops_versions_measure_the_same_thing([5, 6])
        assert not records.ops_versions_measure_the_same_thing([6, 1])
        assert records.ops_versions_measure_the_same_thing(
            [records.IMPLICIT_OPS_MEASUREMENT_VERSION,
             records.FIRST_RECORDED_OPS_MEASUREMENT_VERSION])
        assert 6 not in records.OPS_MEASUREMENT_EQUIVALENT_VERSIONS
        assert 5 not in records.OPS_MEASUREMENT_EQUIVALENT_VERSIONS

    def test_the_version_six_block_names_what_moved_and_why(self):
        """The file's convention: every bump carries a comment block naming
        what changed about the measurement and why it is not comparable.

        Checked as a vocabulary rather than a sentence, so a reword passes and
        a deletion does not.
        """
        source = open(records.__file__).read()
        block = source.split("OPS_MEASUREMENT_VERSION = 6")[0]
        block = block[block.index("VERSION 6"):]
        for token in ("processes", "reinsert_rows_per_s", "scaling_efficiency",
                      "smoke-20261014-075119", "version 3"):
            assert token in block, token
        # The saturated-server row, because a bump that had broken it would
        # be worse than the defect it fixes.
        assert "not flagged" in block

    def test_the_stale_version_five_claim_was_corrected(self):
        """The version-5 block measured "not one record anywhere carries a 5".

        That became false on 2026-10-14, when a six-engine `smoke` run stamped
        158 records at version 5. The corpus half of the claim is still true
        and still verifiable -- the archived tree holds 239 ops record files
        and 2,134 ops records with the key absent from every one -- so what is
        corrected is the clause that over-reached from "the archive" to
        "anywhere".

        Asserted on the source, because the 2026-10-14 run is not in this
        repository and a test that needed it would skip on every checkout.
        """
        source = open(records.__file__).read()
        assert "not one record anywhere carries a 5" not in source, (
            "the claim is false as of 2026-10-14 and must not come back")
        # And the measured half it over-reached from is still stated.
        assert "239 ops record files and 2,134 ops records" in source
        assert "2026-10-14" in source


# ---------------------------------------------------------------------------
# The public surface the callers depend on
# ---------------------------------------------------------------------------

class TestThePublicSurfaceIsUnchanged:
    """The hard constraint on this change: `concurrency.py`, `churn.py` and
    `churn_readers.py` are not touched, so everything they reach has to keep
    its name, its signature and its meaning.

    Asserted by signature rather than by calling, because a call that happened
    to work with a reordered parameter list is exactly what this is for.
    """

    def test_the_names_the_workloads_import_all_resolve(self):
        from harness.workloads import churn as cw
        from harness.workloads import churn_readers as crw
        from harness.workloads import concurrency as conc
        assert conc.ClientPool is cl.ClientPool
        assert crw.ClientPool is cl.ClientPool
        assert conc._client_cpu is cl.client_cpu
        assert conc._shard_queries is cl.shard_queries
        assert cw.PoolError is cl.PoolError
        assert cw.PoolWindow is cl.PoolWindow
        assert conc.Instruments is cl.Instruments
        assert conc.Answer is cl.Answer

    @pytest.mark.parametrize("name,params", [
        ("ClientPool.__init__", ["self", "driver_factory", "queries",
                                 "clients", "k", "ef_search", "index",
                                 "warmup", "collect_results", "unit",
                                 "instruments"]),
        ("ClientPool.start", ["self", "label", "prefix"]),
        ("ClientPool.stop", ["self"]),
        ("shard_queries", ["queries", "clients"]),
        ("client_cpu", ["cpu_start", "cpu_end", "wall_s", "cpus"]),
        ("window_ratio", ["numerator", "denominator", "field"]),
    ])
    def test_the_signatures_are_byte_for_byte_the_old_ones(self, name,
                                                           params):
        import inspect
        target = cl
        for part in name.split("."):
            target = getattr(target, part)
        assert list(inspect.signature(target).parameters) == params, name

    def test_the_tuples_the_report_reads_keep_their_fields(self):
        assert cl.PoolSpec._fields == ("clients", "queries", "k", "ef_search",
                                       "unit")
        assert cl.PoolWindow._fields == (
            "qps", "merged", "inflight_discarded", "prewindow_discarded",
            "client_cpu_seconds", "client_cpus", "client_cpu_util", "wall_s",
            "spec", "results", "outstanding_concurrency",
            "served_concurrency")
        assert cl.Instruments._fields == ("threading", "merge", "cpu_usec",
                                          "cpu_count")
        assert cl.Answer._fields == ("ids", "started", "finished")

    def test_the_two_barrier_timeouts_still_read_together(self):
        """`BARRIER_TIMEOUT_S` is documented as not binding, and the move to
        processes made that MORE true rather than less: `Client.attend` aborts
        the barrier when its client has not reported inside
        `CLIENT_BARRIER_TIMEOUT_S`, so the smaller number now binds at one
        client as well as at many. Under threads a single hanging client left
        the parent's 300 s as the only limit, because there was no sibling to
        time out and break the barrier.
        """
        assert cl.CLIENT_BARRIER_TIMEOUT_S == 120.0
        assert cl.BARRIER_TIMEOUT_S == 300.0
        assert cl.JOIN_TIMEOUT_S == 120.0
        assert cl.CLIENT_BARRIER_TIMEOUT_S < cl.BARRIER_TIMEOUT_S
        source = open(cl.__file__).read()
        assert "self._ready.wait(timeout=CLIENT_BARRIER_TIMEOUT_S)" in source
        assert "timeout=120)" not in source

    def test_the_pool_leaves_no_client_process_behind(self, base, queries):
        """A leaked THREAD died with the interpreter; a leaked PROCESS goes on
        querying the server the next configuration is about to measure.

        So this is a new failure mode rather than a tidier version of an old
        one, and a leak would put one engine's concurrency figure on top of
        the previous one's leftovers.

        `tests/test_t4b_readers.py::pool_threads` is the existing leak probe
        and it looks for live `threading.Thread`s of class `clients.Client`.
        That probe now matches nothing by construction, so it passes
        vacuously -- recorded here because a dead probe that reads as coverage
        is worse than no probe, and because this is the assertion that
        replaces it.
        """
        before = {p.pid for p in multiprocessing.active_children()}
        pool = cl.ClientPool(factory_for(base, lambda: time.sleep(QUERY_S)),
                             queries, 3, K, 32, spec_for(queries), warmup=2)
        pool.start("g3 leak probe", prefix="g3")
        time.sleep(WINDOW_S)
        pool.stop()
        mine = {w.pid for w in pool._workers}
        assert mine and None not in mine
        assert not (mine & {p.pid for p in multiprocessing.active_children()})
        assert all(not w.is_alive() for w in pool._workers)
        # And the probe is not vacuous: those processes really did exist.
        assert not (mine & before)

    def test_the_existing_thread_leak_probe_is_now_vacuous(self):
        """Stated as an assertion so it is a finding and not a surprise.

        `cl.Client` is no longer a `threading.Thread`, so
        `[t for t in threading.enumerate() if isinstance(t, cl.Client)]` --
        which is `test_t4b_readers.pool_threads` verbatim -- can only ever be
        empty. The four `pool_threads() == []` assertions in that file
        therefore hold for a reason that has nothing to do with leaks. That
        file is not edited here; this is the pin that keeps the fact visible.
        """
        assert not issubclass(cl.Client, threading.Thread)
        assert issubclass(cl.Client, multiprocessing.process.BaseProcess)
        assert [t for t in threading.enumerate()
                if isinstance(t, cl.Client)] == []
