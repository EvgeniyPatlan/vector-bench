"""Tell the orchestrator when the one profiled concurrency window runs.

`ops.profile_window_clients` asks for a `perf record -g` of the server over
ONE concurrency window per engine. perf runs on the host, against the server
container's cgroup (`orchestrator/perf_window.py`); the harness runs in a
container and is the only party that knows when that window begins. The two
already share one channel the orchestrator reads line by line -- the harness's
stdout, which `docker_ctl.run_foreground` streams through a `line_filter` --
and one directory both can write, the ops results mount. So:

1. before the pool starts, the harness prints `BEGIN_MARK` with a token and
   waits for `ack-<token>` to appear in the ack directory;
2. the orchestrator, on reading that line, starts perf (or decides it cannot)
   and writes the ack file either way;
3. after the pool has stopped, the harness prints `END_MARK` and the
   orchestrator stops perf.

**Every step is outside the timed span.** The wait for the ack happens before
`ClientPool.start`, whose barrier is where the window opens; the end mark is
printed after `pool.stop()`. Nothing inside the window prints, sleeps or
polls. The profile therefore also covers the pool's warm-up queries, which
are the same queries against the same index, and its teardown; the profile's
own metadata records the window's wall-clock ends beside perf's, so a reader
can see the margin.

**A missing ack costs a bounded wait and nothing else.** An orchestrator that
never answers (an older host, a lab session) leaves the harness waiting
`ACK_TIMEOUT_S` and then measuring as normal; the window is not skipped and
nothing is recorded as profiled.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Optional

BEGIN_MARK = "[vb-profile] begin"
END_MARK = "[vb-profile] end"

#: Long enough for a `docker inspect` and a perf start on a loaded host,
#: short enough that a host that will never answer costs a rounding error
#: against a phase measured in minutes.
ACK_TIMEOUT_S = 30.0
ACK_POLL_S = 0.1


class ProfileWindow:
    """One designated window's begin/end handshake."""

    def __init__(self, ack_dir: Optional[str], clients: int,
                 timeout_s: float = ACK_TIMEOUT_S):
        self.ack_dir = ack_dir
        self.clients = int(clients)
        self.timeout_s = float(timeout_s)
        self.token = uuid.uuid4().hex[:12]
        self.acknowledged: Optional[bool] = None

    def begin(self) -> bool:
        print(f"{BEGIN_MARK} clients={self.clients} token={self.token}",
              flush=True)
        if not self.ack_dir:
            self.acknowledged = False
            return False
        path = os.path.join(self.ack_dir, f"ack-{self.token}")
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            if os.path.exists(path):
                self.acknowledged = True
                try:
                    os.remove(path)
                except OSError:
                    pass
                return True
            time.sleep(ACK_POLL_S)
        print(f"[vb-profile] no acknowledgement within {self.timeout_s:.0f}s; "
              f"measuring the window unprofiled", flush=True)
        self.acknowledged = False
        return False

    def end(self) -> None:
        print(f"{END_MARK} clients={self.clients} token={self.token}",
              flush=True)


__all__ = ["ACK_TIMEOUT_S", "BEGIN_MARK", "END_MARK", "ProfileWindow"]
