"""The thread pool G3 retired, kept so a retired failure mode stays
demonstrable.

**This is not dead code and it is not a mock.** Until G3 the ops harness ran
its concurrency clients, and the churn workload's readers, as Python threads
in one interpreter. MariaDB Connector/Python holds the interpreter lock across
its socket wait, so those clients took turns -- and three of this study's
seven engines measured about one query in service however many clients asked
(`SMOKE/results/smoke-20261014-075119/`, 2026-10-14). The clients are OS
processes now and that cannot happen through Python's own lock any more.

Which creates a problem for the instrument that found it.
`metrics/latency.scaling_status` exists to refuse a scaling efficiency
measured over a pool that did not achieve its concurrency, and
**`concurrency_not_achieved` is now close to unreachable**: a serializing
SERVER raises the median along with everything else, so `served` stays near
the client count; it was the client-side lock that produced the low reading.
The guard has therefore stopped being an expected finding and become a
**tripwire** -- for a revert to threads, for a driver that serializes inside
one process at the C level, for a future pool bug. A tripwire nothing can
trip is indistinguishable from a tripwire that is broken.

So this module is the one thing in the tree that can still produce the shape,
and it is deliberately a VERBATIM-SHAPED copy of `clients.Client` as it stood
at 9fbf2cb: connect, warm up, release the barrier in a `finally`, then loop
on `driver.query` recording both instants until a `threading.Event` is set.
It reuses the REAL `merge_collectors` and `window_concurrency`, so the only
difference from the live pool is where the queries ran -- which is what makes
"the old shape is refused and the new shape is served" one measurement rather
than two.

Shared by `tests/test_g1_client_concurrency.py` (which asserts the verdicts
the instrument reaches) and `tests/test_g3_process_clients.py` (which reports
the five-regime table in both arms). One copy, because two would start out
agreeing and then drift -- the argument `harness/workloads/clients.py` was
itself written out of.

It is a TEST helper and nothing in `harness/` imports it. The timeouts come
from `clients.py` so the two arms wait the same way.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.metrics.latency import (LatencyCollector,  # noqa: E402
                                     merge_collectors, window_concurrency)
from harness.workloads.clients import (BARRIER_TIMEOUT_S,  # noqa: E402
                                       CLIENT_BARRIER_TIMEOUT_S,
                                       shard_queries)

#: Warm-up queries a retired client runs before the barrier. Two, which is
#: what every caller of the live pool in the test suite passes.
WARMUP = 2


class ThreadClient(threading.Thread):
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
            for i in range(min(WARMUP, len(self._queries))):
                driver.query(self._queries[i], self._k)
        except BaseException as exc:  # noqa: BLE001
            self.error = exc
        finally:
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
                driver.query(q, self._k)
                self.collector.add_span(started, time.perf_counter())
                i += 1
            self.collector.stop()
        except BaseException as exc:  # noqa: BLE001
            self.error = exc


def thread_window(factory, queries, clients: int, k: int,
                  window_s: float) -> Dict[str, float]:
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
    shards = shard_queries(queries, clients)
    workers = [ThreadClient(factory, shard, k, ready, stop)
               for shard, _offsets in shards]
    for worker in workers:
        worker.start()
    ready.wait(timeout=BARRIER_TIMEOUT_S)
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
            "outstanding": outstanding, "clients": clients,
            # The collectors, the wall and the two window instants, so a
            # caller can re-run a DIFFERENT instrument over exactly these
            # spans. `tests/test_g1_client_concurrency.py` needs that: its
            # median-versus-mean mutation has to compare two readings of one
            # window rather than two windows.
            "collectors": collectors, "wall": wall,
            "window_start": opened[0], "cutoff": stop_at}


