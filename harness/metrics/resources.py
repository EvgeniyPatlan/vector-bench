"""Container resource accounting: peak memory, CPU time, on-disk sizes.

Peak RSS is read from the container's cgroup rather than sampled from the host,
because sampling misses transient peaks — and index construction is exactly the
kind of workload whose peak lasts seconds. cgroup v2 exposes `memory.peak`
directly; on v1 (and on kernels whose v2 lacks memory.peak) we fall back to
polling `memory.current`/`usage_in_bytes` on a background thread and keep the
maximum, which is explicitly recorded as an approximation.

**Two transports, and the difference is load-bearing.** Everything that
measures the *server* reaches its cgroup by `docker exec`-ing into the server
container (`_exec`, `_cat`), because the harness runs in a second container and
has the Docker CLI for exactly this. `read_own_cpu_usec` and `own_cpu_count`
measure the *client* — the harness's own container — and so open the local path
directly. Routing those through `_cat` would hand `docker exec` a container
name the client image has no Docker CLI to resolve: every call returns None,
`client_cpu_util` is None on every record, and the client-bound flag is
permanently false. A measurement that is silently absent reads exactly like a
measurement that came back negative, which is the failure this module's whole
defensive style exists to avoid.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class ResourceSample:
    peak_rss_bytes: Optional[int] = None
    cpu_seconds: Optional[float] = None
    # True when peak_rss_bytes came from polling rather than a kernel-maintained
    # high-water mark, and so may have missed a short-lived peak.
    peak_is_approximate: bool = False


def _exec(container: str, *args: str, timeout: int = 15) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["docker", "exec", container, *args],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _cat(container: str, path: str) -> Optional[str]:
    return _exec(container, "cat", path)


def read_peak_rss(container: str) -> Optional[int]:
    """Kernel-maintained memory high-water mark, if the cgroup exposes one."""
    for path in (
        "/sys/fs/cgroup/memory.peak",                      # cgroup v2
        "/sys/fs/cgroup/memory/memory.max_usage_in_bytes",  # cgroup v1
    ):
        value = _cat(container, path)
        if value and value.isdigit():
            return int(value)
    return None


def read_current_rss(container: str) -> Optional[int]:
    for path in (
        "/sys/fs/cgroup/memory.current",
        "/sys/fs/cgroup/memory/memory.usage_in_bytes",
    ):
        value = _cat(container, path)
        if value and value.isdigit():
            return int(value)
    return None


#: Where a process finds its OWN cgroup's CPU accounting. Module-level so a
#: test can point them at fixtures: the cgroup v1 branch is unreachable on a
#: v2 host, and an unreachable branch that converts units is exactly where a
#: factor of a thousand survives review.
OWN_CPU_STAT = "/sys/fs/cgroup/cpu.stat"                     # cgroup v2, usec
OWN_CPUACCT_USAGE = "/sys/fs/cgroup/cpuacct/cpuacct.usage"   # cgroup v1, nsec


def _usage_usec(cpu_stat: str) -> Optional[int]:
    """`usage_usec` out of a cgroup v2 `cpu.stat` body, in microseconds.

    Factored out so the client's reader and the server's reader, both below,
    parse one format in one place; they differ only in how the bytes arrive.
    """
    for line in cpu_stat.splitlines():
        if line.startswith("usage_usec"):
            try:
                return int(line.split()[1])
            except (IndexError, ValueError):
                return None
    return None


def _read_own(path: str) -> Optional[str]:
    """Read a path in THIS container, by opening it rather than exec-ing."""
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return None


def read_own_cpu_usec() -> Optional[int]:
    """This process's own cgroup CPU usage, in microseconds.

    Microseconds rather than seconds because that is the unit cgroup v2
    publishes, and the caller subtracts two readings: keeping the raw unit
    means the subtraction is exact and the one division happens where the
    ratio is formed. The cgroup v1 file reports NANOseconds under a different
    name, and is converted here so a caller never has to know which kernel it
    is on — getting that conversion wrong would read as 1000x utilisation,
    which clamps to "client-bound" on every point of every run.

    None when neither file can be read, which the caller must propagate rather
    than turn into a zero: zero utilisation is a claim, and "no counter here"
    is not that claim.
    """
    text = _read_own(OWN_CPU_STAT)
    if text:
        usec = _usage_usec(text)
        if usec is not None:
            return usec
    text = _read_own(OWN_CPUACCT_USAGE)
    if text and text.isdigit():
        return int(text) // 1000
    return None


def own_cpu_count() -> int:
    """How many CPUs this process may actually run on.

    `sched_getaffinity`, not `cpu_count`: the harness container is started
    with `--cpuset-cpus` from the resource pass, which is `cpu.client_cpus: 4`
    in both shipped passes, while `os.cpu_count()` reports the whole host.
    Utilisation computed against the wrong denominator is wrong by exactly
    that ratio — 20% where the truth is 100% on a twenty-CPU box — which is
    the one error in this calculation that still produces a plausible number
    between 0 and 1.
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # not Linux; no cpuset to respect either
        return os.cpu_count() or 1


def read_cpu_seconds(container: str) -> Optional[float]:
    value = _cat(container, "/sys/fs/cgroup/cpu.stat")
    if value:
        usec = _usage_usec(value)
        if usec is not None:
            return usec / 1e6
    value = _cat(container, "/sys/fs/cgroup/cpuacct/cpuacct.usage")
    if value and value.isdigit():
        return int(value) / 1e9
    return None


def reset_peak_rss(container: str) -> bool:
    """Zero the high-water mark so a phase measures only its own peak.

    cgroup v2 accepts a write of "0" to memory.peak on Linux 6.x; v1 accepts a
    write to memory.max_usage_in_bytes. Where neither works the caller must fall
    back to the polling monitor, so the return value matters.
    """
    for path in ("/sys/fs/cgroup/memory.peak",
                 "/sys/fs/cgroup/memory/memory.max_usage_in_bytes"):
        if _exec(container, "sh", "-c", f"echo 0 > {path}") is not None:
            after = _cat(container, path)
            if after and after.isdigit():
                return True
    return False


class PeakMemoryMonitor:
    """Track a container's peak memory across a phase.

    Prefers the kernel high-water mark. Falls back to polling only when the mark
    cannot be reset, and flags the result as approximate so the report never
    presents a possibly-missed peak as if it were exact.
    """

    def __init__(self, container: str, poll_interval: float = 0.2):
        self.container = container
        self.poll_interval = poll_interval
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._polled_peak = 0
        self._use_kernel_mark = False
        self._cpu_start: Optional[float] = None

    def __enter__(self) -> "PeakMemoryMonitor":
        self._cpu_start = read_cpu_seconds(self.container)
        self._use_kernel_mark = reset_peak_rss(self.container)
        if not self._use_kernel_mark:
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        return self

    def _poll(self) -> None:
        while not self._stop.is_set():
            current = read_current_rss(self.container)
            if current is not None:
                self._polled_peak = max(self._polled_peak, current)
            self._stop.wait(self.poll_interval)

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def result(self) -> ResourceSample:
        cpu_end = read_cpu_seconds(self.container)
        cpu_delta = None
        if cpu_end is not None and self._cpu_start is not None:
            cpu_delta = round(max(0.0, cpu_end - self._cpu_start), 3)

        if self._use_kernel_mark:
            return ResourceSample(
                peak_rss_bytes=read_peak_rss(self.container),
                cpu_seconds=cpu_delta,
                peak_is_approximate=False,
            )
        return ResourceSample(
            peak_rss_bytes=self._polled_peak or None,
            cpu_seconds=cpu_delta,
            peak_is_approximate=True,
        )


def directory_bytes(container: str, path: str, pattern: str = "*") -> int:
    """Total size of files under `path` in the container matching `pattern`.

    Used for index/table sizing on the MySQL-family engines, whose HNSW graphs
    live in companion tables on disk rather than in an index segment a catalog
    function could size.
    """
    out = _exec(
        container, "sh", "-c",
        f"find {path} -type f -name '{pattern}' -printf '%s\\n' 2>/dev/null "
        f"| awk '{{s+=$1}} END {{print s+0}}'",
        timeout=60,
    )
    try:
        return int(out) if out else 0
    except ValueError:
        return 0


def wait_for_stable_size(container: str, path: str, pattern: str = "*",
                         settle_s: float = 2.0, timeout_s: float = 120.0) -> int:
    """Poll a directory's size until it stops changing.

    Both MySQL-family engines flush index pages asynchronously, so measuring the
    moment a build returns can undercount. This waits for the size to hold
    steady before believing it.
    """
    deadline = time.time() + timeout_s
    last = -1
    stable_since = None
    while time.time() < deadline:
        size = directory_bytes(container, path, pattern)
        if size == last:
            if stable_since is None:
                stable_since = time.time()
            elif time.time() - stable_since >= settle_s:
                return size
        else:
            last = size
            stable_since = None
        time.sleep(0.5)
    return max(last, 0)
