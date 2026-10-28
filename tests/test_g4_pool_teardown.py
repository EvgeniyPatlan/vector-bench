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

* **`_transfer_cut` was set and never read on `stop`'s exit-status path.**
  The watchdog can fire while a `recv` of an already-complete payload is
  landing -- `cancel()` is then a no-op, `recv` returns, `absorb` records no
  error, and the exit code is `-SIGKILL`. `ClientPool.stop` read that as a
  client killed from outside and failed a window whose spans are entire,
  claiming they "cover only part of the window". `TRANSFER_GRACE_S` narrows
  that race; `Client.ended_by_this_pool` is what closes it.

* **And reading ONE flag there swallowed the opposite case.**
  `_cut_off_transfer` sets the intent before its liveness check, so a client
  already dead from an OUTSIDE kill sets it while this pool sends nothing --
  and suppressing `stop`'s report on that flag hid a container memory limit
  reaping a client, which is the case the branch was written for. The spans
  are entire there too, so no wrong number is published and only the OOM
  event goes silent. Two flags now: `_transfer_cut` the intent,
  `_killed_here` the fact, and only the fact may suppress.

**Mutation discipline (R4), counted rather than described.** Every repair is
probed by patching the system under test AT RUN TIME -- never by editing a
file on disk -- and each docstring carries the honest reading and the mutant
reading, both measured. There are **six** mutants and **five** of them are a
`monkeypatch.setattr` on a `ClientPool` or `Client` attribute:

1. `ClientPool._join_attendants` -> a no-op (monkeypatch).
2. one live thread at fork time, against the same fork after joining it --
   a **subprocess with and without the thread, not a monkeypatch**, because
   the warning is a property of the whole process and this suite is known to
   leak one. Taken `FORK_PROBE_SAMPLES` times, for the reason that constant
   records: `join()` returns before the kernel reaps the OS thread, so the
   CONTROL arm -- not the mutant arm -- raced, and this probe failed on
   0.2-1.3% of runs depending on load before that was repaired.
3. `Client._recv_by` -> the unbounded `self._from_client.recv()` (monkeypatch).
4. `Client._cut_off_transfer` -> one that kills without recording
   (monkeypatch).
5. `Client.ended_by_this_pool` -> `False` (monkeypatch).
6. `Client.ended_by_this_pool` -> the INTENT flag instead of the fact, which
   is the ordering before the two flags were separated (monkeypatch).

An earlier revision of this paragraph said "two of the three mutants", which
was wrong in both numbers and is the class of inaccuracy the first commit of
this branch is about; it then said five and four, which this round made six
and five. The monkeypatch targets are why those five exist as NAMED
attributes at all: a repair spelled inline in `start` and again in `stop`
could only be mutated by editing the file, and a copy is how one of the two
came to be missing in the first place.

**One mutation in this file was silently clobbered and the probe's own state
assertion is what caught it** -- `_pool` set `_recv_by` after the caller did,
so the outside-kill arm measured the wrong interleaving and came back with
the fact flag True. `_recv_by` is a parameter of `_pool` now. That is the
hazard R4 names, met in this file rather than quoted from it.

**No measured throughput or latency is asserted here**, for the reason
`test_g3_process_clients.py` gives about a shared box. What is asserted is a
thread count, a warning count, an exit code, and whether a bounded call
returned inside its own bound -- none of which a busy machine can move in the
direction that would make a broken repair pass.
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


#: How many independent live-thread/joined-thread PAIRS the probe below takes
#: in one interpreter, and the reason it samples at all.
#:
#: **`Thread.join()` does not deliver the state the control arm claims.** On
#: CPython 3.12.3, where all of this was measured, `join()` returns when the
#: dying thread's `_tstate_lock` drops, which is BEFORE the OS thread is
#: reaped, while the fork warning reads the KERNEL's thread count. Later
#: versions rework `join()` and may not have the window at all; the settle
#: below makes that difference moot rather than resting on it. So a fork can
#: land in a window
#: where `threading` reports one thread and the kernel still reports two, and
#: warn from what the arm believes is a single-threaded parent. Measured over
#: 400 runs of an instrumented copy: every run whose joined arm warned read
#: `active_count() == 1`, `is_alive() == False` and `/proc/self/status`
#: `Threads: 2` at fork time, and of the 390 runs that read `Threads: 1` not
#: one warned. (10 read 2 and 3 of those stayed quiet -- the READ races the
#: fork too, and that is the harmless direction.) That window was this
#: file's own flake -- the single-pair probe this replaces returned
#: `joined=1 not_joined=1` on 4 of 2000 quiet runs (0.20%), 22 of 2000 at
#: ten-way load (1.10%) and 39 of 3000 at twenty-way (1.30%), always on the
#: joined arm alone and never on the mutant arm.
#:
#: Two repairs, independent on purpose. `wait_until_single()` closes the window
#: at its SOURCE, and soundly: after `join()` nothing creates a thread, so the
#: count is monotone and a reading of 1 cannot go stale. The SAMPLING is what
#: the assertions are phrased against, so the test does not come to depend on
#: a Linux-only file that a legitimate box may not have. Over 2000 runs each at
#: ten-way load: sampling alone left 197 of 10000 joined samples warning (and
#: 0 runs with every sample warning); sampling over the settled probe left 0
#: of 10000.
#:
#: Five pairs and not one. Without the settle the dirty samples came back
#: OVER-DISPERSED on the run that chose k -- one descheduled instant dirties
#: several -- so `p**5` could understate the tail. That dispersion is
#: load-dependent rather than a property of the window: an independent
#: ten-way sample came back binomial-consistent. Either way k was picked
#: against a measured distribution and not a model (dirty
#: samples per run over 2000 runs: 1845 none, 122 one, 26 two, 5 three, 2
#: four, 0 five).
FORK_PROBE_SAMPLES = 5

#: Bound on the wait for the kernel to agree the probe is single-threaded.
#:
#: **The warrant is the MARGIN, not a count of clean runs.** Worst settle
#: observed: 3.12 ms over 2000 settles at ten-way, and 4.81 ms with the probe
#: pinned to ONE CPU at `nice 19` against the box's existing load -- so the
#: tightest margin measured is **416x** this bound. The shape is what makes
#: that travel: cutting CPU share roughly tenfold worsened the worst case only
#: 1.5x, because what is being bounded is a thread-exit completion costing
#: microseconds of KERNEL work, not a quantity that scales with contention.
#: Reaching 2 s needs some 400x more degradation than could be produced here.
#:
#: A zero-expiry rate is the weaker half and is quoted second for that reason:
#: 75000 samples with no expiry (35000 here, 40000 independently) bounds a rate
#: under the conditions sampled and does not generalise to another machine,
#: whereas the margin is an argument about the mechanism. An earlier revision
#: led with 2400x against a QUIET worst case, which overstated it.
#:
#: Generous on purpose, because EXCEEDING IT IS A FAILURE and not a relaxation
#: -- a thread still counted after 2 s did not leave. It is also what a mutant
#: that leaves the thread alive pays per sample, which is why it is seconds and
#: not minutes. That mutant is `one_pair` with its `gate.set()` and
#: `thread.join()` deleted, so the thread is still on `gate.wait` and there is
#: no `join` to block in: five samples pay the bound, cost 10 s, and then fail.
FORK_PROBE_SETTLE_S = 2.0

#: A bare interpreter that forks beside a live thread and again after joining
#: it, `FORK_PROBE_SAMPLES` times, reporting the CPython multi-threaded-fork
#: warning count for each fork. Run as a SUBPROCESS and not inline, and that is
#: the only way it can be honest: the warning is a property of the whole
#: process, and this suite is known to leave a `webui/jobs.py` worker thread
#: alive, so an inline control arm would measure that thread rather than the
#: absence of this one.
#:
#: Each pair builds a FRESH thread, so each joined-arm fork races the window
#: `FORK_PROBE_SAMPLES` documents independently rather than re-reading one
#: outcome. Prints one `joined,not_joined,settled` triple per pair, where
#: `settled` is 1 where the kernel confirmed one thread before the joined-arm
#: fork, 0 where `FORK_PROBE_SETTLE_S` expired with it still reporting more,
#: and -1 where there was no count to read -- three values and not a flag,
#: because "could not read" must not report itself as "confirmed".
_FORK_WARNING_PROBE = r"""
import multiprocessing, threading, time, warnings
MP = multiprocessing.get_context("fork")

SAMPLES = __SAMPLES__
SETTLE_S = __SETTLE_S__


def noop():
    pass


def os_thread_count():
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("Threads:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return None


def wait_until_single():
    deadline = time.monotonic() + SETTLE_S
    while True:
        count = os_thread_count()
        if count is None:
            return -1
        if count == 1:
            return 1
        if time.monotonic() > deadline:
            return 0
        time.sleep(0.0005)


def warnings_on_one_fork():
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        proc = MP.Process(target=noop)
        proc.start()
        proc.join(30)
    return len([w for w in seen if "multi-threaded" in str(w.message)])


def one_pair():
    gate = threading.Event()
    thread = threading.Thread(target=lambda: gate.wait(30), daemon=True)
    thread.start()
    not_joined = warnings_on_one_fork()
    gate.set()
    thread.join()
    settled = wait_until_single()
    return warnings_on_one_fork(), not_joined, settled


pairs = [one_pair() for _ in range(SAMPLES)]
print(" ".join(",".join(str(n) for n in pair) for pair in pairs))
""".replace("__SAMPLES__", str(FORK_PROBE_SAMPLES)) \
   .replace("__SETTLE_S__", str(FORK_PROBE_SETTLE_S))


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

        **Two tiers over `FORK_PROBE_SAMPLES` pairs, and the kernel's own
        answer picks which.** Where `wait_until_single` confirmed one thread
        before EVERY joined-arm fork, the bound is `max(joined) == 0`: all five
        must be silent, which is STRICTER than the single sample this replaces.
        Nothing is being tolerated there -- a reading of 1 cannot go stale,
        because after `join()` nothing creates a thread -- so a warning in that
        tier would refute that reasoning rather than report a busy box.

        **And it is DOMINANCE rather than a shared counter that makes the tier
        sound.** `wait_until_single` reads `/proc/self/status`, which is not
        what the warning reads: CPython 3.12.3 contains no `/proc/self/status`
        string at all. It does not need to be. A dying thread unlinks its
        Python thread state before its OS thread can exit, so the kernel's
        count is the LAST of the candidate counters to drop; a kernel count of
        1 therefore implies every counter the warning could consult reads 1
        too, whichever one it is. Measured: 0 warnings in 35000 confirmed
        samples here, and 0 in 40000 independently.

        An EXPIRY is not a third tier, it is a failure, for the reason the
        assertion records: only a box with no kernel count to read -- no
        `/proc` -- drops to the FLOOR, `min(joined) == 0`. One-sidedness is
        what makes the floor sound there: an OS thread the kernel has not
        finished reaping is one MORE thread at fork time, so the window can
        only ADD a warning and never remove one the live thread would have
        caused, and the minimum is the noise-free reading.

        **A floor alone would not catch a stray thread, which is why it is
        confined to the no-`/proc` box.** One-sidedness justifies a floor
        against noise that is RARE per sample; it says nothing about a stray
        warning on MOST samples and not all. One lingering past `join()` on 80%
        of samples would have failed the old single-sample test 80% of the time
        and a floor only where all five coincided.

        Where there is a kernel count the two shapes of that stray are handled
        DIFFERENTLY, and only one of them is *caught*. A stray shorter-lived
        than `FORK_PROBE_SETTLE_S` is ELIMINATED: the settle waits it out, the
        parent really is single-threaded by the time it forks, no warning is
        produced and no assertion fires -- such a case passes silently, and
        correctly, because there is nothing left to detect. Verified: a 50 ms
        stray on four of five samples passes 5 of 5. A stray that outlives the
        bound is CAUGHT, by the expiry. And a stray that dirties a parent the
        kernel did confirm is caught by the strict bound, which is the case the
        flat floor used to pass. Claiming the strict bound "catches" the
        short-lived shape would be claiming detection where there is
        elimination; the distinction is the whole reason the settle exists.

        The strict bound also stops a settle that silently stopped waiting,
        which still reports `confirmed` and is then held to five silent forks
        under the very race it stopped closing.

        Either tier refutes what the single sample refuted. An interpreter that
        warned on EVERY fork -- the "this probe is not measuring the thread"
        failure -- has no clean sample to offer and fails both, and so does the
        live-thread state itself, which warns on all five.

        The live arm is a floor too, which makes it STRICTER than the single
        sample it replaces: `min(not_joined) >= 1` demands that **every** one
        of the five forks beside a live thread warned, as 2000 runs (10000
        forks) did. That arm has no race to tolerate -- `start()` has returned,
        so the thread is alive and counted -- and that is what lets a floor
        stand there rather than an average.

        **In the fallback tier, not the mode and not "most samples clean".**
        Those tolerate two of the five forks from a single-threaded parent
        warning, which would itself be evidence the mechanism had changed, and
        they need only ceil(k/2) coincident dirty samples rather than k: on the
        measured unsettled distribution above, a majority rule would have
        failed on the 7 runs in 2000 that went three-or-more dirty (0.35%),
        where the floor failed on 0. In the confirmed tier the rule is stricter
        than any of the three, so neither is the better shape in either tier.
        """
        if sys.version_info < (3, 12):  # pragma: no cover - platform gate
            pytest.skip("CPython warns about a multi-threaded fork from 3.12")
        done = subprocess.run([sys.executable, "-I", "-c",
                               _FORK_WARNING_PROBE],
                              capture_output=True, text=True, timeout=120)
        assert done.returncode == 0, done.stderr
        pairs = [tuple(int(n) for n in field.split(","))
                 for field in done.stdout.split()]
        assert len(pairs) == FORK_PROBE_SAMPLES, (
            f"the probe reported {len(pairs)} pair(s) and not "
            f"{FORK_PROBE_SAMPLES}: {done.stdout!r}")
        joined = [pair[0] for pair in pairs]
        not_joined = [pair[1] for pair in pairs]
        # How the settle went before each joined-arm fork -- 1 the kernel
        # confirmed one thread, 0 the bound expired with it reporting more, -1
        # there was no count to read.
        settle = {1: "confirmed", 0: "expired", -1: "unreadable"}
        settled = ", ".join(
            f"{sum(1 for pair in pairs if pair[2] == value)} {name}"
            for value, name in settle.items())
        # An expiry is a DEFECT, not a platform fact, so it FAILS here instead
        # of relaxing the bound: the bound carries a 416x margin over the worst
        # settle measured, so a thread that `join()` had already returned on,
        # and that the kernel still counted that long afterwards, did not
        # leave. Only `unreadable` -- no kernel count to read at all -- is a
        # box this test accommodates.
        #
        # **That dichotomy is not total, and the limit is worth naming.**
        # `unreadable` keys on `OSError` or a missing `Threads:` line, so a
        # `/proc` that HAS `Threads:` but updates it lazily -- gVisor-style
        # sandboxes, WSL1, some emulations -- reads as an expiry and fails with
        # a message blaming a thread that never left. Every sample behind these
        # figures came from one kernel and one CPython 3.12.3, so they are no
        # evidence at all about that case. The exposure is small because this
        # repo already requires a real Linux `/proc` (`read_own_cpu_usec` and
        # `tests/test_harness.py` read cgroup files directly), but a reader on
        # such a box should be told to suspect the reader and not the thread.
        #
        # **Demoting on an expiry is what the first version of this rule did,
        # and it inverted the whole point.** A stray thread OUTLIVING the bound
        # made the settle expire, the expiry dropped the run to the floor, and
        # the floor then passed on whichever sample the stray had left clean --
        # so the `/proc` visibility that was supposed to enable the catch was
        # exactly what disabled it. Measured on that version: `joined=[0, 1, 1,
        # 1, 1]` with `settles=[1, 0, 0, 0, 0]` PASSED 3 of 3 runs, ignoring 12
        # dirty samples of 15, where the single sample it replaced would have
        # failed on 80% of runs. A pass prints nothing, so it was silent too.
        expired = [i for i, pair in enumerate(pairs) if pair[2] == 0]
        assert not expired, (
            f"the wait for the kernel to report one thread expired on "
            f"sample(s) {expired} of {FORK_PROBE_SAMPLES} after "
            f"{FORK_PROBE_SETTLE_S}s each, so a thread `join()` had returned "
            f"on was still counted that long afterwards and did not leave. "
            f"Warnings were joined={joined} not_joined={not_joined}. (If this "
            f"box is pathologically slow, or reports `Threads:` lazily, then "
            f"the reader is the suspect and not the thread: the bound carried "
            f"a 416x margin over the worst settle measured, but every such "
            f"sample came from ONE box, one Linux kernel and CPython 3.12.3.)")
        # No expiry, so the kernel either confirmed every parent or could not
        # be read at all. Confirmed -> every sample must be silent, which is
        # stricter than the one sample this replaces. Unreadable -> the floor,
        # which is all the one-sided argument can honestly support.
        confirmed = all(pair[2] == 1 for pair in pairs)
        rule = ("every sample, because the kernel confirmed a single-threaded "
                "parent before each" if confirmed else
                "the quietest sample, because this box has no kernel thread "
                "count to read")
        assert (max(joined) if confirmed else min(joined)) == 0, (
            f"a fork from a single-threaded parent warned: {joined} over "
            f"{FORK_PROBE_SAMPLES} samples, read across {rule}; the wait for "
            f"the kernel to report one thread went {settled}. So this probe "
            f"is not measuring the thread")
        assert min(not_joined) >= 1, (
            f"a fork beside a live thread warned {not_joined} time(s) per "
            f"sample; this interpreter does not warn, so the 54 -> 0 "
            f"measurement `_join_attendants` cites cannot be read off it")
        # Documentation and not detection: implied by the two bounds above in
        # either tier, and deliberately kept to state the separation the test
        # is named for, as the single-sample `joined != not_joined` it replaces
        # also did.
        assert min(joined) < min(not_joined)


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


def _lingering_tell(real_tell):
    """A `_tell` that sends the window payload in full and then stays alive.

    IN THE CHILD, and the opposite shape to `_stalling_tell`: the payload is
    COMPLETE and buffered in the pipe, so a parent `recv` succeeds whenever it
    runs. The child then sleeps, which is what leaves it killable -- without
    that it would exit 0 the instant `run` returned and there would be no
    non-zero exit code for `stop` to misread.
    """
    def _tell(self, phase, error, collector, results):
        real_tell(self, phase, error, collector, results)
        if phase == "window":
            time.sleep(600)
    return _tell


class TestAKillAfterACompleteTransferIsNotAnAbnormalDeath:
    """The race `TRANSFER_GRACE_S` narrows and `ended_by_this_pool` closes.

    `_cut_off_transfer` sets the flag and kills. If it fires while a `recv` of
    an already-complete payload is landing, `watchdog.cancel()` is a no-op,
    `recv` returns normally and `absorb` records no error -- but the client's
    exit code is now `-SIGKILL`. `ClientPool.stop`'s exit-status check reads
    any non-zero code as a client killed from outside and says "the spans it
    sent cover only part of the window, so this is not a measurement".

    Both halves of that are false here: the spans are entire, and the signal
    came from this pool. `_transfer_cut` was set and never read on that path,
    so the flag looked like a guard while only `absorb`'s message branch used
    it. Reachable in a real run only once the pool's shared `JOIN_TIMEOUT_S`
    is spent, which is why it is driven directly here rather than waited for.

    The race is driven, not raced: `_recv_by` is replaced with one that calls
    the real `_cut_off_transfer` FIRST and then does the real `recv`. That is
    the losing interleaving every time instead of once in a while, which is
    the only way a probe for a race can be a probe rather than a lottery.
    """

    def _pool(self, base, queries, monkeypatch, recv_by=None) -> cl.ClientPool:
        """A pool whose clients send in full, linger, and are then cut off.

        `recv_by` is a PARAMETER and not something a caller patches around
        this method, because patching `_recv_by` before calling it was
        silently clobbered here: this method set its own and the caller's
        mutation never reached the pool. The probe's own state assertion
        caught it -- the fact flag came back True when the test needed the
        outside-kill path -- which is why both flags are read back.
        """
        monkeypatch.setattr(cl, "JOIN_TIMEOUT_S", PROBE_JOIN_S)
        monkeypatch.setattr(cp.Client, "_tell",
                            _lingering_tell(cp.Client._tell))

        def cut_then_recv(self, deadline):
            self._cut_off_transfer()
            return self._from_client.recv()

        monkeypatch.setattr(cp.Client, "_recv_by", recv_by or cut_then_recv)
        return cl.ClientPool(factory_for(base), queries, 2, K, 32,
                             spec_for(queries), warmup=1)

    def _run(self, base, queries, monkeypatch, recv_by=None):
        """What `stop` did, and the state both arms have to share.

        Returns `(window, raised, exitcodes, facts, intents, errors)`. BOTH
        flags, because the whole fix is that they differ: `intents` is
        `_transfer_cut` (the deadline fired) and `facts` is `_killed_here`
        (this pool killed a live client), and a probe that read only one
        could not show which of them `stop` consults.
        """
        pool = self._pool(base, queries, monkeypatch, recv_by=recv_by)
        pool.start("g4 kill after a complete transfer", prefix="g4")
        time.sleep(0.05)
        try:
            try:
                window = pool.stop()
                raised = None
            except cl.PoolError as exc:
                window, raised = None, exc
            codes = [w.exitcode for w in pool._workers]
            facts = [w._killed_here for w in pool._workers]
            intents = [w._transfer_cut for w in pool._workers]
            errors = [w.error for w in pool._workers]
            return window, raised, codes, facts, intents, errors
        finally:
            for worker in pool._workers:
                worker.reap()

    def test_the_window_survives_a_kill_this_pool_sent(self, base, queries,
                                                       monkeypatch):
        """Honest and mutant, as whether `stop` published or raised.

        Both arms see the same pool state, which is what makes this one
        experiment: exit codes **[-9, -9]**, `ended_by_this_pool` **[True,
        True]**, and `Client.error` **[None, None]** -- every payload arrived
        complete and no client failed.

        HONEST: `stop` returns a `PoolWindow` and the window is published.
        MUTANT (`ended_by_this_pool` forced to False, which is the state
        before this repair, with `_transfer_cut` set and unread): `stop`
        raises `PoolError` "2 of 2 client processes ended abnormally after
        reporting; first: client ..., it was killed by SIGKILL. The spans it
        sent cover only part of the window, so this is not a measurement" --
        a sound window failed, over a signal the harness sent, with a claim
        about the payload that is factually wrong.
        """
        window, raised, codes, facts, intents, errors = self._run(
            base, queries, monkeypatch)
        assert raised is None, (
            f"a window whose payloads all arrived complete was failed: "
            f"{raised}")
        assert window is not None and window.qps is not None
        # The pool state both arms share, asserted so neither arm can be
        # passing for a reason other than the branch under test. Here this
        # pool really did send the signal, so BOTH flags are set.
        assert all(code == -9 for code in codes), codes
        assert all(facts), facts
        assert all(intents), intents
        assert errors == [None, None], errors

        monkeypatch.setattr(cp.Client, "ended_by_this_pool",
                            property(lambda self: False))
        _w, mutant, m_codes, _m_f, _m_i, m_errors = self._run(
            base, queries, monkeypatch)
        assert isinstance(mutant, cl.PoolError), (
            "forcing `ended_by_this_pool` False changed nothing, so `stop` "
            "does not read it and this repair is unproven")
        assert "SIGKILL" in str(mutant), str(mutant)
        assert "cover only part of the window" in str(mutant), str(mutant)
        # Same state, opposite outcome: that is the pair.
        assert all(code == -9 for code in m_codes), m_codes
        assert m_errors == [None, None], m_errors

    def test_an_outside_kill_after_a_complete_transfer_is_still_reported(
            self, base, queries, monkeypatch):
        """The exclusion must not swallow the case the branch exists for.

        **This is the probe that would have caught the LOW the exclusion
        introduced.** `_cut_off_transfer` sets the INTENT flag before its
        `is_alive()` check, so a client already dead from an OUTSIDE kill
        sets it while this pool sends nothing. Suppressing `stop`'s report on
        that flag therefore hid a container memory limit reaping a client --
        and in that case the spans really are entire, so no wrong number is
        published and the OOM event is simply silent. In a framework where
        the container's memory limit is part of the measurement, that is the
        event a reader most needs.

        Driven, not raced: `_recv_by` is replaced with one that kills the
        client from OUTSIDE the deadline path (a bare `self.kill()`, which is
        what an OOM reap looks like to the parent), waits for it to die, then
        calls the real `_cut_off_transfer` -- whose `is_alive()` is now False
        -- and then the real `recv`, which still succeeds because the payload
        is buffered.

        Both arms see: exit code **-9**, `_transfer_cut` **True** (the intent
        fired), `_killed_here` **False** (this pool sent nothing).

        HONEST (`ended_by_this_pool` returns the FACT): `stop` raises
        `PoolError` naming the abnormal exit, so the reap is reported.
        MUTANT (the property returning the INTENT, which is the ordering
        before this fix): `stop` returns a window and the kill is swallowed.
        """
        def kill_from_outside_then_cut(self, deadline):
            self.kill()
            self.join(timeout=10.0)
            self._cut_off_transfer()
            return self._from_client.recv()

        def run():
            return self._run(base, queries, monkeypatch,
                             recv_by=kill_from_outside_then_cut)

        window, raised, codes, facts, intents, errors = run()
        assert all(code == -9 for code in codes), codes
        assert errors == [None, None], errors
        # The state that makes this the right experiment: the deadline fired
        # (intent set) and this pool killed nothing (fact clear). If the two
        # agreed here there would be nothing for the fix to distinguish.
        assert all(intents), intents
        assert not any(facts), facts
        assert isinstance(raised, cl.PoolError), (
            "an outside kill after a complete transfer was not reported, so "
            "the `killed` exclusion swallows the OOM reap it exists for")
        assert "SIGKILL" in str(raised), str(raised)
        assert window is None

        monkeypatch.setattr(cp.Client, "ended_by_this_pool",
                            property(lambda self: self._transfer_cut))
        m_window, m_raised, m_codes, _m_f, _m_i, _m_err = run()
        assert m_raised is None, (
            "reading the INTENT still reported the kill, so this probe does "
            "not reach the ordering and the fix is unproven")
        assert m_window is not None
        assert all(code == -9 for code in m_codes), m_codes
        assert m_window is not None and window is None


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
