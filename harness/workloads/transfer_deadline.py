"""The span-transfer deadline: the watchdog, its two flags, and what reads them.

Split out of `harness/workloads/client_process.py`, and the reason is the one
that module's own docstring gave for the cut before it was taken: it stood at
799 lines of this repository's 800-line ceiling, and the next change to it was
a correctness fix that did not fit. The alternative was to keep shaving the
rationale out of a measurement module to make room for a three-line repair,
which is how a file ends up holding code nobody can explain.

**WHAT THE SPLIT HAD TO BE CAREFUL ABOUT, for the reason `client_process.py`
records twice over: a module that re-exports a name does not SHARE it.** The
three members below are reached as `Client` attributes and nothing reaches
them through this module, so a `monkeypatch.setattr(cp.Client, "_recv_by", f)`
sets the name ON `Client`, shadows the mixin through the MRO, and still
reaches the pool. `tests/test_g4_pool_teardown.py` patches all three that way
and its pairs are the proof the targets did not move.

`TRANSFER_GRACE_S` is the one name whose MODULE moved. `client_process.py`
imports it so `cp.TRANSFER_GRACE_S` still reads, which is all anything does
with it -- but `_recv_by` reads it out of THIS module's globals, so rebinding
`client_process.TRANSFER_GRACE_S` would reach nothing. Nothing rebinds it; if
anything ever needs to, it has to patch it here.

**THE DEADLINE, AND WHY A BOUND WAS NEEDED AT ALL.** `Client.absorb` waits
for a client's window payload with `poll(deadline)`, which bounds the wait for
the FIRST BYTE and nothing after it: `Connection.recv` takes no timeout. A
client that started sending and then stalled while STILL ALIVE therefore
passed the `poll` and blocked the parent inside `recv` for as long as it
lived, taking `ClientPool.stop` past `JOIN_TIMEOUT_S` without limit, on the
one call an operator cannot interrupt. A client that DIES mid-send always
raised, because its pipe closes.

**THE FLAGS ARE TWO BECAUSE THEY ANSWER TWO QUESTIONS, and one flag answering
both silenced an event this framework exists to report.**

* `_transfer_cut` is the INTENT: the deadline fired. `_cut_off_transfer` sets
  it BEFORE the liveness check, and that ordering is load-bearing -- the kill
  is what makes the parent's `recv` raise, so `absorb`'s message branch has
  to find the flag already set or it reports the harness's own SIGKILL as the
  client's death.
* `_killed_here` is the FACT: this pool killed a LIVE client. It is set only
  after a `kill` that actually happened.

Only the FACT may suppress `ClientPool.stop`'s exit-status report. Reading the
intent there suppressed it whenever the deadline merely fired -- including for
a client ALREADY DEAD from an outside kill, which sets the intent and sends no
signal. That is exactly a container's memory limit reaping a client, which is
the case `stop`'s branch was written for: the spans in that case really are
entire, so no wrong number is published, and the OOM event goes unreported
instead. In a framework where the container's memory limit is part of the
measurement, a silently reaped client is not a small loss.
"""

from __future__ import annotations

import threading
import time
from typing import Any

#: Floor on how long a client that has STARTED sending its window payload is
#: given to finish, used by `_recv_by` when the pool's SHARED deadline has
#: already run out on earlier clients.
#:
#: **A floor, not a second timeout**: a zero-length watchdog would race a
#: `recv` of an already complete payload, publishing the window on one run
#: and failing it on the next. One second, because what it beats is a pipe
#: read of bytes already there, and it costs `stop` at most this per client
#: in an already pathological window against no limit before. It NARROWS
#: that race; `TransferDeadline.ended_by_this_pool` closes it.
TRANSFER_GRACE_S = 1.0


class TransferDeadline:
    """Bounds a client's span transfer. Mixed into `Client`, PARENT side.

    A mixin rather than three module-level functions taking a client, because
    every one of them is reached as a `Client` attribute by the pool and by
    the tests that mutate it, and turning them into functions would have moved
    those patch targets for no gain. It owns no state of its own: `Client`
    initialises both flags, because `Client.__init__` is where every other
    parent-side field is set and splitting that would be worse than the split.
    """

    def _recv_by(self, deadline: float) -> Any:
        """`recv` this client's payload, bounded by the pool's deadline.

        Killing the client is what unblocks a parent already inside `recv`:
        closing the parent's own descriptor does not, because on Linux a
        `close` never wakes a reader already blocked in `read` on that open
        file description. Measured on a child that half-sent and then slept,
        `recv` returned in 2.001 s against a 2.000 s watchdog where without
        it it does not return at all.

        A `threading.Timer` and not `signal.alarm`, which fires only in the
        main thread while `stop` runs on whichever thread the workload is on;
        and not a non-blocking read, because a `Connection` message is framed
        and a partial read loses the frame.

        The timer is cancelled AND JOINED rather than left to expire: a
        lingering timer thread is a live thread in the parent at the instant
        the next pool forks, which is the condition
        `ClientPool._join_attendants` exists for arriving from a second
        direction. **The 5 s on that join is the unstated second term in
        `stop`'s worst case** -- in principle `JOIN_TIMEOUT_S` plus one
        `TRANSFER_GRACE_S` and one 5 s join per client -- and it cannot be
        spent: the timer's body sets a flag and calls `kill`, so the thread
        this waits on either never ran or is microseconds from returning. The
        5 s is there so a pathological `kill` cannot hang the pool, not as a
        budget.
        """
        left = max(TRANSFER_GRACE_S, deadline - time.monotonic())
        watchdog = threading.Timer(left, self._cut_off_transfer)
        watchdog.daemon = True
        watchdog.start()
        try:
            return self._from_client.recv()
        finally:
            watchdog.cancel()
            watchdog.join(timeout=5.0)

    @property
    def ended_by_this_pool(self) -> bool:
        """Whether THIS pool's span-transfer deadline killed the client.

        Read by `ClientPool.stop`'s exit-status check, which otherwise reads
        every non-zero exit code as a client killed from OUTSIDE and says
        "the spans it sent cover only part of the window". **Both halves are
        wrong for a client the watchdog killed after a COMPLETE transfer**:
        `recv` returns, `absorb` records no error, the exit code is
        `-SIGKILL`, and `stop` would fail a window whose spans are entire
        over a signal the harness itself sent.

        **It returns the FACT and not the intent.** This module's docstring
        has why: the intent is set before the liveness check, so a client
        already dead from an outside kill sets it without this pool sending
        anything, and suppressing the report on the intent hid the container
        memory limit reaping a client.
        """
        return self._killed_here

    def _cut_off_transfer(self) -> None:
        """End a client that is mid-send at the pool's deadline. PARENT side.

        Runs on the watchdog thread. INTENT before the liveness check, FACT
        only after a `kill` this pool really sent -- the module docstring has
        the ordering and what each one is allowed to decide.

        `kill` and not `terminate`: a client blocked inside a `send` to a full
        pipe is in a syscall its own handler would not run promptly, and the
        point of this path is a bound. `reap` runs afterwards and is
        idempotent about an already dead client.
        """
        self._transfer_cut = True
        try:
            if self.is_alive():
                self.kill()
                self._killed_here = True
        except (OSError, ValueError, AssertionError):  # pragma: no cover
            pass


__all__ = ["TRANSFER_GRACE_S", "TransferDeadline"]
