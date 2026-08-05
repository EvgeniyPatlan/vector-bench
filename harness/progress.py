"""Time-based progress reporting for long-running harness phases.

Every long operation in this harness had the same defect independently: it ran
for minutes or hours and printed nothing, so a working phase was
indistinguishable from a hung one and "how long will this take" could only be
answered by waiting. Fixing that per-site produced four near-identical loops, so
it lives here once.

Reporting is driven by elapsed TIME, not by an iteration count. A count-based
trigger is wrong for exactly the workloads that need it most: at 5 rows/s a
"every 10,000 rows" trigger is silent for half an hour, and the operations here
range over four orders of magnitude in throughput.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional

DEFAULT_INTERVAL_S = float(os.environ.get("VB_PROGRESS_INTERVAL", "20"))


class Progress:
    """Emit a rate-and-ETA line at most once per interval.

    Usage:

        p = Progress(len(queries), "filtered 10%", prefix="mariadb")
        for q in queries:
            ...
            p.step()
        p.finish()
    """

    def __init__(self, total: int, label: str = "", prefix: str = "",
                 interval_s: Optional[float] = None, stream=sys.stderr):
        self.total = max(0, int(total))
        self.label = label
        self.prefix = prefix
        self.interval = DEFAULT_INTERVAL_S if interval_s is None else interval_s
        self.stream = stream
        self.done = 0
        self._started = time.perf_counter()
        self._next = self._started + self.interval

    # ------------------------------------------------------------------

    def step(self, n: int = 1) -> None:
        self.done += n
        now = time.perf_counter()
        if now >= self._next:
            self._emit(now)
            self._next = now + self.interval

    def _emit(self, now: float) -> None:
        elapsed = max(now - self._started, 1e-9)
        rate = self.done / elapsed
        head = f"[{self.prefix}] " if self.prefix else ""
        tail = ""
        if self.total and rate > 0:
            eta = (self.total - self.done) / rate
            tail = f", ETA {eta / 60:.1f} min"
        total = f"/{self.total:,}" if self.total else ""
        print(f"{head}  {self.label}{' ' if self.label else ''}"
              f"{self.done:,}{total}, {rate:,.1f}/s{tail}",
              file=self.stream, flush=True)

    def finish(self) -> float:
        """Report the total and return elapsed seconds."""
        elapsed = time.perf_counter() - self._started
        # Only summarise work that ran long enough to have been worth watching;
        # a sub-interval operation needs no epitaph.
        if elapsed >= self.interval:
            head = f"[{self.prefix}] " if self.prefix else ""
            rate = self.done / max(elapsed, 1e-9)
            print(f"{head}  {self.label}{' ' if self.label else ''}"
                  f"complete: {self.done:,} in {elapsed:.1f}s ({rate:,.1f}/s)",
                  file=self.stream, flush=True)
        return elapsed


class Heartbeat:
    """Announce that a single long, unsplittable operation is still running.

    Some steps cannot report fractional progress — a bulk `CREATE INDEX`, or a
    brute-force ground-truth pass inside one NumPy call. Silence during those is
    still indistinguishable from a hang, so at least say that time is passing
    and how much.
    """

    def __init__(self, label: str, prefix: str = "",
                 interval_s: Optional[float] = None, stream=sys.stderr):
        self.label = label
        self.prefix = prefix
        self.interval = DEFAULT_INTERVAL_S if interval_s is None else interval_s
        self.stream = stream
        self._started = 0.0
        self._stop = None
        self._thread = None

    def __enter__(self) -> "Heartbeat":
        import threading

        self._started = time.perf_counter()
        self._stop = threading.Event()

        def beat() -> None:
            while not self._stop.wait(self.interval):
                head = f"[{self.prefix}] " if self.prefix else ""
                print(f"{head}  {self.label} — still running, "
                      f"{time.perf_counter() - self._started:.0f}s elapsed",
                      file=self.stream, flush=True)

        self._thread = threading.Thread(target=beat, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        if self._stop is not None:
            self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        elapsed = time.perf_counter() - self._started
        if elapsed >= self.interval:
            head = f"[{self.prefix}] " if self.prefix else ""
            print(f"{head}  {self.label} — done in {elapsed:.1f}s",
                  file=self.stream, flush=True)
