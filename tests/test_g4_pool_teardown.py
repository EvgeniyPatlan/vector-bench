"""G4: the two teardown paths that could outlive their own timeout.

**Why a new file (R3).** `tests/test_g3_process_clients.py` owns the move to
process clients and is already at the length this repository argues about. The
two defects below were found by reviewing that change after it merged, they are
behavioural rather than prose, and each needs its own synthetic client shape --
so they go here and that file keeps its subject.

The two:

* **`ClientPool.start` did not join its attendants on the failure path.**
  `stop` has always joined them, for the reason `ClientPool._join_attendants`
  records: a sweep builds one pool per window, and the next pool's `launch`
  forking beside the previous pool's live threads is a fork from a
  multi-threaded parent -- 54 CPython `DeprecationWarning`s in a full suite
  run before that join and 0 after. `start`'s `BrokenBarrierError` path raises
  `PoolError` and never reaches `stop`, so on that path the join did not
  happen; `harness/workloads/churn.py` catches `PoolError` at three sites and
  goes on to the next churn fraction, which forks the next pool.

* **`Client.absorb` bounded the wait for the first byte and not the
  transfer.** `poll(left)` respects the pool's deadline and `recv()` takes no
  timeout at all, so a client that started sending and then stalled while
  STILL ALIVE blocked `ClientPool.stop` for as long as it stayed alive --
  past `JOIN_TIMEOUT_S` by an unbounded amount. A client that DIES mid-send
  was already handled: its pipe closes and `recv` raises.

**Mutation discipline (R4).** Both repairs are probed by patching the system
under test AT RUN TIME -- never by editing a file on disk -- and each
docstring carries the honest reading and the mutant reading, both measured.
Two of the three mutants are a `monkeypatch.setattr` on a `ClientPool` or
`Client` method, which is why those methods exist as named methods at all: a
repair spelled inline in `start` and again in `stop` could only be mutated by
editing the file, and a copy is how one of the two came to be missing in the
first place.

**No measured throughput or latency is asserted here**, for the reason
`test_g3_process_clients.py` gives about a shared box. What is asserted is a
thread count, a warning count, and whether a bounded call returned inside its
own bound -- none of which a busy machine can move in the direction that
would make a broken repair pass.
"""

from __future__ import annotations

import os
import struct
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Tuple

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.drivers.base import IndexSpec                            # noqa: E402
from harness.workloads import client_process as cp                    # noqa: E402
from harness.workloads import clients as cl                           # noqa: E402
from tests.fake_driver import (FakeDriver, K, N_QUERIES,              # noqa: E402
                               N_ROWS, index_spec, tiny_dataset)

#: Wall clock each synthetic query costs. Nothing here times a query; this
#: only has to be small enough that a window closes promptly and large enough
#: that a client is not spinning on the stop flag.
QUERY_S = 0.001

#: How long the attendant probe's synthetic attendant stays alive after it
#: aborts the barrier. It is the quantity that separates the two arms in TIME
#: as well as in thread count: the honest `start` cannot return before it, and
#: the mutant returns well inside it. Large enough that scheduling noise on a
#: loaded box cannot close the gap, small enough to stay out of the way.
#:
#: **A `sleep` in the attendant and NOT an `Event` a timer opens, and that is
#: this file eating its own cooking.** The first version of this probe held a
#: `threading.Timer` to open a gate, started before `pool.start()` -- so every
#: pool it built forked beside that timer and the probe emitted six CPython
#: multi-threaded-fork `DeprecationWarning`s of its own. Measured: 6 before
#: this change and 0 after. A test for the 54 -> 0 repair that added six back
#: is worse than no test, because the next person reads the count.
GATE_S = 0.5

#: The pool deadline the transfer probe runs against, in place of the shipped
#: `JOIN_TIMEOUT_S` of 120 s. Patched onto `clients` BEFORE the pool is built,
#: because `ClientPool._timeouts()` captures the three waits at construction
#: -- `client_process.py`'s docstring is the authority on that and five other
#: tests depend on the same ordering.
PROBE_JOIN_S = 2.0

#: Clients in the transfer probe's pool, and every one of them stalls. It is
#: named because the honest arm's bound is arithmetic in it: the shared
#: deadline is spent by the first stalled client and each one after it falls
#: back on `client_process.TRANSFER_GRACE_S`.
STALLING_CLIENTS = 2

#: How long the mutant arm of the transfer probe is allowed to run before it
#: is called unbounded. The honest arm returns inside `PROBE_JOIN_S` plus one
#: grace per client after the first; this is several times that, so "did not
#: return" is a statement about the mutant and not about the box.
UNBOUNDED_AFTER_S = 12.0


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


def spec_for(queries) -> IndexSpec:
    return IndexSpec(dim=queries.shape[1], m=16, metric="euclidean")


def factory_for(base: FakeDriver) -> Callable[[], Any]:
    """A driver factory whose clones sleep instead of scanning the corpus.

    The same shape `test_g3_process_clients.py` uses and for the same reason:
    the cost is bound on the INSTANCE, because `FakeDriver.clone` returns a
    plain `FakeDriver` and a subclass override would not reach the clients the
    pool forks. Under fork the child calls this closure out of inherited
    memory, so nothing here has to be picklable.
    """
    def clone():
        client = base.clone()

        def query(vector, k, ef_search=None):
            time.sleep(QUERY_S)
            return list(range(K))

        client.query = query
        return client
    return clone


def live_attendants(pool: cl.ClientPool) -> List[str]:
    """The pool's own attendant threads that are still running.

    By NAME and not by `pool._attendants`, deliberately. `_attendants` is the
    list the pool built, and a thread that has finished is still in it -- so
    reading the list alone cannot tell a joined thread from a leaked one.
    `ClientPool.start` names them `client-attendant-N`, which makes this
    probe exact and makes it ignore every other thread in the interpreter:
    the suite is known to carry a leaked `webui/jobs.py` worker, and a probe
    that counted non-main threads would have measured that instead.
    """
    return sorted(t.name for t in threading.enumerate()
                  if t.name.startswith("client-attendant-") and t.is_alive())


class TestStartJoinsItsAttendantsWhenTheBarrierBreaks:
    """`start`'s failure path, which raises instead of reaching `stop`.

    The synthetic client is the barrier abort itself: `Client.attend` is
    replaced with one that records an error, aborts the barrier -- exactly
    what the real `attend` does in its `finally` for a client that DIED -- and
    then sleeps for `GATE_S`. So the attendant is provably alive at the
    instant `start` decides to raise, and provably finishes a known time
    later, which is what makes the two arms differ by a thread count rather
    than by a race.

    Patching `attend` rather than killing a real child is also what keeps the
    probe off the clock: a dying child would leave the parent's `poll` to
    return whenever the kernel got round to closing the pipe.
    """

    def _pool_that_fails(self, base, queries, monkeypatch):
        """A real pool whose attendants abort the barrier and then linger."""
        def attend(self):
            self.error = cp.ClientDied(
                f"synthetic: {self.name} never reported warm")
            self._ready.abort()
            time.sleep(GATE_S)

        monkeypatch.setattr(cl.Client, "attend", attend)
        return cl.ClientPool(factory_for(base), queries, 3, K, 32,
                             spec_for(queries), warmup=1)

    def _run(self, base, queries, monkeypatch) -> Tuple[float, List[str]]:
        """Start a doomed pool and report (elapsed, attendants still alive).

        Nothing in here starts a thread of its own before `pool.start()`, and
        that is deliberate: `start` forks, and a helper thread alive at that
        instant would make this probe emit the very warning it is about. See
        `GATE_S`. The attendants are joined in a `finally` so a failing
        assertion cannot leave three of them parked for the next test to fork
        beside either.
        """
        pool = self._pool_that_fails(base, queries, monkeypatch)
        started = time.perf_counter()
        try:
            with pytest.raises(cl.PoolError) as raised:
                pool.start("g4 barrier abort", prefix="g4")
            elapsed = time.perf_counter() - started
            assert "during warm-up" in str(raised.value), str(raised.value)
            return elapsed, live_attendants(pool)
        finally:
            for attendant in pool._attendants:
                attendant.join(timeout=10.0)
            for worker in pool._workers:
                worker.reap()

    def test_no_attendant_outlives_the_failure(self, base, queries,
                                               monkeypatch):
        """Honest and mutant, as a thread count and as a wall clock.

        HONEST: `start` raises after **0 live attendants** and not before the
        gate opens -- measured 0 attendants at 0.50 s against a 0.5 s gate,
        because `_join_attendants` waits for every one of the three.

        MUTANT (`_join_attendants` replaced with a no-op, which is exactly the
        state `start` was in before this repair): **3 live attendants**, and
        `start` returns in 0.00 s because it waits for nothing. Those three
        are the threads a following pool forks beside.

        The pair is not identical in either quantity, so this is a probe and
        not a dead one. The thread count is the assertion; the elapsed
        readings are reported because a repair that joined a thread which had
        already finished would show the same count and a different clock.
        """
        honest_elapsed, honest_live = self._run(base, queries, monkeypatch)

        monkeypatch.setattr(cl.ClientPool, "_join_attendants",
                            lambda self: None)
        mutant_elapsed, mutant_live = self._run(base, queries, monkeypatch)

        assert honest_live == [], (
            f"{len(honest_live)} attendants outlived `start`'s failure path "
            f"({honest_live}), so the next pool forks beside them")
        assert mutant_live, (
            "removing the join changed nothing, so this probe cannot see it "
            "and the repair is unproven")
        assert honest_live != mutant_live
        # And the clock agrees with the count, which is what says the honest
        # arm really waited rather than finding nothing to wait for.
        assert honest_elapsed >= GATE_S, (honest_elapsed, GATE_S)
        assert mutant_elapsed < GATE_S, (mutant_elapsed, GATE_S)

    def test_the_repair_is_one_method_and_both_paths_call_it(self):
        """Structural, and it has no mutant -- the claim is the call graph.

        A repair spelled out at both call sites could drift apart, and that is
        not hypothetical here: `stop` had the join and `start` did not, which
        is how the failure path came to be missing it. Named so a future edit
        that inlines one copy fails here.
        """
        source = open(cl.__file__).read()
        assert source.count("def _join_attendants") == 1
        assert source.count("self._join_attendants()") == 2
        # Both of them, identified by the method they sit in.
        start_body = source.split("def start(self")[1].split("def stop(self")[0]
        stop_body = source.split("def stop(self")[1].split(
            "def _first_error")[0]
        assert "self._join_attendants()" in start_body
        assert "self._join_attendants()" in stop_body
        # And the reap comes FIRST in the failure path, which is the ordering
        # `_join_attendants` and `_first_error` both argue for: an attendant
        # blocked on its client's pipe is not woken by breaking the barrier.
        assert (start_body.index("worker.reap()")
                < start_body.index("self._join_attendants()")), (
            "`start` now joins the attendants before reaping the clients, so "
            "an attendant blocked in `_await_warmup` holds the failure path "
            "for its whole `CLIENT_BARRIER_TIMEOUT_S`")


#: A bare interpreter that forks once with a live thread and once without,
#: reporting the CPython multi-threaded-fork warning count for each. Run as a
#: SUBPROCESS and not inline, and that is the only way it can be honest: the
#: warning is a property of the whole process, and this suite is known to
#: leave a `webui/jobs.py` worker thread alive, so an inline control arm would
#: measure that thread rather than the absence of this one.
_FORK_WARNING_PROBE = r"""
import multiprocessing, threading, warnings
MP = multiprocessing.get_context("fork")


def noop():
    pass


def warnings_on_one_fork():
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        proc = MP.Process(target=noop)
        proc.start()
        proc.join(30)
    return len([w for w in seen if "multi-threaded" in str(w.message)])


gate = threading.Event()
thread = threading.Thread(target=lambda: gate.wait(30), daemon=True)
thread.start()
not_joined = warnings_on_one_fork()
gate.set()
thread.join()
joined = warnings_on_one_fork()
print(joined, not_joined)
"""


class TestTheLeakedAttendantIsWhatCPythonWarnsAbout:
    """Why the thread count above is the damage and not a tidiness point.

    `ClientPool._join_attendants` cites 54 warnings before the `stop` join and
    0 after. That measurement is not reproducible from a checkout -- it is a
    whole-suite count -- so what is reproduced here is the mechanism it rests
    on: one live thread at fork time is the difference between a warning and
    no warning.
    """

    def test_one_live_thread_at_fork_time_is_the_whole_difference(self):
        """Honest and mutant in one clean interpreter, as warning counts.

        HONEST (the thread joined before the fork, which is what both call
        sites of `_join_attendants` now guarantee): **0 warnings**.
        MUTANT (the same fork with that thread still alive, which is the state
        `start`'s failure path left behind): **1 warning** --
        `DeprecationWarning: This process (pid=...) is multi-threaded, use of
        fork() may lead to deadlocks in the child.`

        Measured on CPython 3.12.3. Skipped below 3.12, which does not warn at
        all: a pair of zeros there is a dead probe and is reported as one
        rather than passing.
        """
        if sys.version_info < (3, 12):  # pragma: no cover - platform gate
            pytest.skip("CPython warns about a multi-threaded fork from 3.12")
        done = subprocess.run([sys.executable, "-I", "-c",
                               _FORK_WARNING_PROBE],
                              capture_output=True, text=True, timeout=120)
        assert done.returncode == 0, done.stderr
        joined, not_joined = (int(n) for n in done.stdout.split())
        assert joined == 0, (
            f"a fork from a single-threaded parent warned {joined} time(s), "
            f"so this probe is not measuring the thread")
        assert not_joined >= 1, (
            f"a fork beside a live thread warned {not_joined} time(s); this "
            f"interpreter does not warn, so the 54 -> 0 measurement "
            f"`_join_attendants` cites cannot be read off it")
        assert joined != not_joined


def _stalling_tell(real_tell):
    """A `_tell` that half-sends the window payload and then stays alive.

    IN THE CHILD. It writes a frame header promising 4 MiB, then 1 KiB of it,
    and then sleeps -- which is precisely the shape `absorb`'s `poll` passes
    and its `recv` has no timeout for. A client that EXITS here instead would
    close its pipe and raise `EOFError` in the parent, which the pool already
    handled; staying alive is the whole point.

    The header is written with `os.write` on the raw descriptor rather than
    through `Connection.send`, because `send` is atomic over one message and
    there is no way to ask it for half of one.
    """
    def _tell(self, phase, error, collector, results):
        if phase != "window":
            return real_tell(self, phase, error, collector, results)
        descriptor = self._to_parent.fileno()
        os.write(descriptor, struct.pack("!i", 4 * 1024 * 1024))
        os.write(descriptor, b"x" * 1024)
        time.sleep(600)
    return _tell


class TestAStalledTransferCannotOutlastThePoolDeadline:
    """`Client.absorb`, against a client that is mid-send and still alive.

    `ClientPool.stop` is the call an operator cannot interrupt: it is inside
    the measured window's teardown, between the engine's last query and the
    record being written. An unbounded `recv` there is a run that stops making
    progress with no log line and no timeout, which is why this is a
    correctness fix and not a robustness nicety.
    """

    def _pool(self, base, queries, monkeypatch) -> cl.ClientPool:
        monkeypatch.setattr(cl, "JOIN_TIMEOUT_S", PROBE_JOIN_S)
        monkeypatch.setattr(cp.Client, "_tell", _stalling_tell(cp.Client._tell))
        # Built AFTER the patch, because `_timeouts()` captures the three
        # waits at construction -- so a pool built first would carry 120 s.
        return cl.ClientPool(factory_for(base), queries, STALLING_CLIENTS, K,
                             32, spec_for(queries), warmup=1)

    def _stop_in_a_thread(self, pool, outcome: Dict[str, Any]):
        """Start `stop` on its own thread and return it, unjoined.

        A direct call would hang the test rather than measure the mutant, and
        "the suite timed out" is not a reading. Returned rather than joined
        here so the caller can reap the clients FIRST and only then join: the
        mutant's `recv` is unblocked by the client dying, so reaping is what
        lets this thread finish at all -- and a thread still alive when the
        next pool forks would make this file emit the warning it is about.

        Started after `pool.start()` has forked, for the same reason.
        """
        def run():
            try:
                pool.stop()
                outcome["raised"] = None
            except BaseException as exc:  # noqa: BLE001 - reported
                outcome["raised"] = exc
            outcome["at"] = time.perf_counter()

        runner = threading.Thread(target=run, daemon=True, name="g4-stop")
        outcome["started"] = time.perf_counter()
        runner.start()
        return runner

    def _run(self, base, queries, monkeypatch):
        pool = self._pool(base, queries, monkeypatch)
        pool.start("g4 stalled transfer", prefix="g4")
        time.sleep(0.05)
        outcome: Dict[str, Any] = {}
        runner = self._stop_in_a_thread(pool, outcome)
        try:
            runner.join(timeout=UNBOUNDED_AFTER_S)
            # Read BEFORE the reap below. The reap is what unblocks an
            # unbounded `recv`, so a reading taken after it would show the
            # mutant returning too and this probe would be dead -- which is
            # exactly what the first version of it did.
            returned_on_its_own = not runner.is_alive()
        finally:
            for worker in pool._workers:
                worker.reap()
            # Only now can the mutant's `recv` return, so join again: a live
            # `g4-stop` thread at the next fork is this file's own defect.
            runner.join(timeout=30.0)
            assert not runner.is_alive(), (
                "the `stop` thread outlived its reaped clients, so a later "
                "fork in this file happens beside it")
        if not returned_on_its_own:
            return None, None
        return outcome["at"] - outcome["started"], outcome["raised"]

    def test_the_transfer_is_bounded_and_not_only_the_first_byte(
            self, base, queries, monkeypatch):
        """Honest and mutant, as the seconds `stop` took to return.

        HONEST: `stop` raised `PoolError` after **3.004 s**, and the message
        names the deadline rather than a signal -- "stopped partway through
        sending its spans and was ended at the pool's 2s deadline". The
        watchdog `_recv_by` arms is what ends it: killing the client is what
        unblocks a parent already inside `recv`, because closing the parent's
        own descriptor does not wake a reader already blocked on that open
        file description.

        **3.0 s and not 2.0 s, and the extra second is `TRANSFER_GRACE_S`
        doing exactly what it says.** Both of this pool's two clients stall.
        The first is cut off at the shared 2.0 s deadline; the second then
        reaches `_recv_by` with nothing left on that deadline and gets the
        1.0 s floor, which is there so a client whose payload is already
        complete is never raced against a zero-length watchdog. So the bound
        `stop` carries is `JOIN_TIMEOUT_S + (stalled clients - 1) x
        TRANSFER_GRACE_S`, and `CEILING_S` below is that arithmetic rather
        than a number picked to fit.

        MUTANT (`_recv_by` replaced with the unbounded `self._from_client.
        recv()` this method used to call directly): `stop` **did not return at
        all** inside 12.0 s. The pair is 3.0 s against unbounded, which is the
        defect.

        What is asserted exactly is that one arm returned and the other did
        not, plus that the honest arm landed inside the arithmetic above:
        a machine at 20 busy cores of 20 can add scheduling delay to the
        raise, so the window has slack on top, and none of that slack can
        make an unbounded `recv` pass.
        """
        honest_elapsed, honest_raised = self._run(base, queries, monkeypatch)

        monkeypatch.setattr(cp.Client, "_recv_by",
                            lambda self, deadline: self._from_client.recv())
        mutant_elapsed, _mutant_raised = self._run(base, queries, monkeypatch)

        assert honest_elapsed is not None, (
            f"`stop` did not return within {UNBOUNDED_AFTER_S}s against a "
            f"{PROBE_JOIN_S}s deadline, so the transfer is still unbounded")
        assert mutant_elapsed is None, (
            f"the unbounded `recv` returned after {mutant_elapsed}s, so this "
            f"probe does not reach the thing the bound protects and the "
            f"repair is unproven")
        assert isinstance(honest_raised, cl.PoolError), honest_raised
        # The bound it was bounded BY, spelled as the arithmetic rather than
        # as a constant: the shared deadline plus one grace per client that
        # had to fall back on the floor. Slack on top for a loaded box, in
        # the one direction that cannot make an unbounded `recv` pass.
        ceiling = PROBE_JOIN_S + (STALLING_CLIENTS - 1) * cp.TRANSFER_GRACE_S
        assert honest_elapsed < ceiling + 3.0, (
            f"`stop` returned after {honest_elapsed:.3f}s against a bound of "
            f"{ceiling:.1f}s ({PROBE_JOIN_S}s deadline + "
            f"{STALLING_CLIENTS - 1} x {cp.TRANSFER_GRACE_S}s grace)")
        assert honest_elapsed >= PROBE_JOIN_S, (
            f"`stop` returned after {honest_elapsed:.3f}s, inside its own "
            f"{PROBE_JOIN_S}s deadline -- so it is not the deadline that "
            f"ended the transfer and this probe is measuring something else")

    def test_the_deadline_names_itself_rather_than_the_signal_it_sent(
            self, base, queries, monkeypatch):
        """The message, because the two causes have different remedies.

        `_cut_off_transfer` kills the client, so without `_transfer_cut` the
        existing handler would report `_exit_description(-9)` -- "it was
        killed by SIGKILL" -- and send an operator looking for an OOM killer
        that was this method.

        HONEST: "stopped partway through sending its spans and was ended at
        the pool's 2s deadline". MUTANT (`_transfer_cut` forced back to False
        after the watchdog sets it, which is the pre-repair message path):
        "died during the measured window: it was killed by SIGKILL".
        """
        _elapsed, raised = self._run(base, queries, monkeypatch)
        honest = str(raised)
        assert "partway through sending its spans" in honest, honest
        assert "SIGKILL" not in honest, honest

        # The mutant: the flag the message branches on, held at False.
        real_cut = cp.Client._cut_off_transfer

        def cut_without_recording(self):
            real_cut(self)
            self._transfer_cut = False

        monkeypatch.setattr(cp.Client, "_cut_off_transfer",
                            cut_without_recording)
        _mutant_elapsed, mutant_raised = self._run(base, queries, monkeypatch)
        mutant = str(mutant_raised)
        assert "SIGKILL" in mutant, mutant
        assert "partway through sending its spans" not in mutant, mutant
        assert honest != mutant


class TestEveryDriverFamilyIsStillClassified:
    """The duty `test_g1_client_concurrency.py`'s `SERIALIZES = ()` claimed.

    That tuple carried the comment "Named here so a family added above has to
    be classified" and was referenced by no `parametrize` -- and an empty
    tuple parametrizes nothing, so even a reference would have run no test.
    It is removed, and this is the assertion that discharges what it claimed.
    Here rather than there because it is a new check (R3), and `test_g1`'s
    grant covers corrections only.
    """

    def test_no_driver_family_is_left_unclassified(self):
        from tests import test_g1_client_concurrency as g1
        assert set(g1.DRIVERS) == set(g1.SCALES), (
            f"these driver families are not classified by `SCALES`: "
            f"{sorted(set(g1.DRIVERS) ^ set(g1.SCALES))}")
        # Non-vacuity: the classification is a real list and not empty, and
        # the thread-arm list is a strict subset of it rather than a second
        # unrelated set.
        assert len(g1.SCALES) >= 4
        assert set(g1.SERIALIZES_UNDER_THREADS) <= set(g1.DRIVERS)
        assert not hasattr(g1, "SERIALIZES"), (
            "`SERIALIZES` is back; if it is meant to classify a family, wire "
            "it to a `parametrize` -- an empty tuple runs no test")
