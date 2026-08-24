"""Docker container, network and volume management for the orchestrator.

Uses the docker CLI rather than the Python SDK so the orchestrator's only
dependencies are python3 and pyyaml. That matters because the orchestrator runs
on the host, and a benchmark framework that requires pip installs on the machine
under test is a framework that changes the machine under test.

Everything created here is tagged with the run id and torn down by
`cleanup_run`, so an interrupted run leaves nothing behind that would perturb
the next one.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

LABEL = "vector-bench"


class DockerError(RuntimeError):
    pass


def _run(args: Sequence[str], check: bool = True, timeout: int = 120,
         capture: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        list(args),
        capture_output=capture, text=True, timeout=timeout, check=False,
    )
    if check and proc.returncode != 0:
        raise DockerError(
            f"command failed ({proc.returncode}): {' '.join(shlex.quote(a) for a in args)}\n"
            f"stdout: {(proc.stdout or '').strip()}\n"
            f"stderr: {(proc.stderr or '').strip()}"
        )
    return proc


def docker_available() -> bool:
    try:
        _run(["docker", "info"], timeout=30)
        return True
    except (DockerError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def image_exists(image: str) -> bool:
    try:
        _run(["docker", "image", "inspect", image], timeout=30)
        return True
    except DockerError:
        return False


def image_id(image: str) -> str:
    try:
        proc = _run(["docker", "image", "inspect", "--format", "{{.Id}}", image], timeout=30)
        return proc.stdout.strip()
    except DockerError:
        return "unknown"


# ---------------------------------------------------------------------------
# Networks and volumes
# ---------------------------------------------------------------------------

def create_network(name: str, internal: bool = True) -> str:
    if _network_exists(name):
        return name
    args = ["docker", "network", "create", "--label", f"{LABEL}=1"]
    if internal:
        # No route off the host. The engines never need outbound access, and
        # removing it eliminates a source of variance (and of surprise).
        args.append("--internal")
    args.append(name)
    _run(args)
    return name


def _network_exists(name: str) -> bool:
    proc = _run(["docker", "network", "ls", "--format", "{{.Name}}"], check=False)
    return name in (proc.stdout or "").split()


def remove_network(name: str) -> None:
    _run(["docker", "network", "rm", name], check=False)


def create_volume(name: str, device: Optional[str] = None) -> str:
    """Create a named volume, optionally backed by a specific host directory.

    Without `device`, Docker puts the volume under its own data-root, which on
    a default install is /var/lib/docker on the root filesystem. That is rarely
    where the space is: a benchmark box typically has a small root volume and a
    large data mount, and a million-vector corpus lands on the wrong one. The
    failure is not subtle but it is misleading — initdb reports "No space left
    on device" from inside the container and it reads as an engine problem.
    """
    cmd = ["docker", "volume", "create", "--label", f"{LABEL}=1"]
    if device:
        os.makedirs(device, exist_ok=True)
        cmd += ["--driver", "local",
                "--opt", "type=none", "--opt", "o=bind", "--opt", f"device={device}"]
    cmd.append(name)
    _run(cmd)
    return name


def root_dir() -> Optional[str]:
    """Where the daemon stores images, containers and volumes."""
    try:
        proc = _run(["docker", "info", "-f", "{{.DockerRootDir}}"], check=False)
    except Exception:
        return None
    out = (proc.stdout or "").strip()
    return out or None


def remove_volume(name: str) -> None:
    _run(["docker", "volume", "rm", "-f", name], check=False, timeout=180)


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------

@dataclass
class ContainerSpec:
    name: str
    image: str
    network: Optional[str] = None
    cpuset: Optional[str] = None
    memory_bytes: Optional[int] = None
    shm_size: Optional[str] = None
    env: Dict[str, str] = None
    volumes: List[str] = None            # "src:dst[:mode]"
    command: List[str] = None
    entrypoint: Optional[str] = None
    user: Optional[str] = None
    workdir: Optional[str] = None
    detach: bool = True
    # Set only when the container must reach the internet (dataset fetch).
    allow_network: bool = False


def _spec_args(spec: ContainerSpec) -> List[str]:
    args = ["docker", "run", "--name", spec.name, "--label", f"{LABEL}=1"]
    if spec.detach:
        args.append("-d")
    else:
        args.append("--rm")
    if spec.network:
        args += ["--network", spec.network]
    if spec.cpuset:
        args += ["--cpuset-cpus", spec.cpuset]
    if spec.memory_bytes:
        # Memory and swap set to the same value: without this the container can
        # swap past its limit and the measurement silently becomes a disk
        # benchmark instead of an out-of-memory failure.
        args += ["--memory", str(spec.memory_bytes),
                 "--memory-swap", str(spec.memory_bytes)]
    if spec.shm_size:
        args += ["--shm-size", spec.shm_size]
    if spec.user:
        args += ["--user", spec.user]
    if spec.workdir:
        args += ["--workdir", spec.workdir]
    for key, value in (spec.env or {}).items():
        args += ["-e", f"{key}={value}"]
    for volume in (spec.volumes or []):
        args += ["-v", volume]
    if spec.entrypoint:
        args += ["--entrypoint", spec.entrypoint]
    args.append(spec.image)
    args += list(spec.command or [])
    return args


def start(spec: ContainerSpec) -> str:
    remove(spec.name)
    proc = _run(_spec_args(spec), timeout=300)
    return proc.stdout.strip()


def save_phase_log(run_dir: str, engine: str, phase: str, resource_pass: str,
                   lines: List[str]) -> Optional[str]:
    """Keep a phase's console output next to its measurements.

    The output of a phase is where a failure explains itself, and until now it
    existed only on the operator's terminal. A run that wrote no recall results
    and a churn that stopped after nothing both had their reason printed and
    then lost, and by the time anyone asked, the terminal was gone. The
    measurements are archived and copied between machines; their explanation
    should travel with them.
    """
    if not lines:
        return None
    directory = os.path.join(run_dir, "logs")
    try:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{engine}-{phase}-{resource_pass}.log")
        with open(path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        return path
    except OSError:
        return None


def run_foreground(spec: ContainerSpec, timeout: int = 24 * 3600,
                   stream: bool = True,
                   sink: Optional[List[str]] = None,
                   line_filter: Optional[Any] = None) -> int:
    """Run a container in the foreground, streaming its output.

    Used for the measurement containers, whose logs are the primary record of
    what happened and must reach the operator live rather than after the fact.

    `sink` collects the streamed lines. An exit code alone cannot distinguish
    "this tool failed" from "this tool considers having nothing to do an
    error", and at least one dependency here takes the latter view.

    `line_filter(line) -> list[str]` decides what actually reaches the operator.
    It returns the lines to print, so it can hold some back and release them
    later — which is what suppressing a known-benign multi-line traceback
    requires. Everything still reaches `sink` regardless.
    """
    spec.detach = False
    remove(spec.name)
    args = _spec_args(spec)
    if not stream:
        proc = _run(args, check=False, timeout=timeout)
        for chunk in (proc.stdout, proc.stderr):
            if chunk:
                print(chunk)
                if sink is not None:
                    sink.extend(chunk.splitlines())
        return proc.returncode

    process = subprocess.Popen(args, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        assert process.stdout is not None
        for line in process.stdout:
            line = line.rstrip()
            if sink is not None:
                sink.append(line)
            for shown in (line_filter(line) if line_filter else [line]):
                print(shown)
        if line_filter is not None:
            for shown in line_filter(None):      # flush anything held back
                print(shown)
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"[docker] timeout after {timeout}s; killing {spec.name}")
        process.kill()
        remove(spec.name)
        return 124
    finally:
        if process.poll() is None:
            process.kill()


def remove(name: str) -> None:
    _run(["docker", "rm", "-f", name], check=False, timeout=120)


def stop(name: str, timeout_s: int = 60) -> None:
    _run(["docker", "stop", "-t", str(timeout_s), name], check=False,
         timeout=timeout_s + 30)


def is_running(name: str) -> bool:
    proc = _run(["docker", "inspect", "--format", "{{.State.Running}}", name],
                check=False, timeout=30)
    return (proc.stdout or "").strip() == "true"


def logs(name: str, tail: int = 100) -> str:
    proc = _run(["docker", "logs", "--tail", str(tail), name], check=False, timeout=60)
    return ((proc.stdout or "") + (proc.stderr or "")).strip()


def exec_in(name: str, args: Sequence[str], timeout: int = 60,
            check: bool = False) -> subprocess.CompletedProcess:
    return _run(["docker", "exec", name, *args], check=check, timeout=timeout)


def wait_healthy(name: str, probe: Sequence[str], timeout_s: int = 180,
                 interval_s: float = 1.0) -> None:
    """Wait until `probe` succeeds inside the container.

    Fails with the container's own log tail attached, because "the server did
    not come up" is useless without the reason, and the reason is always in
    that log.
    """
    deadline = time.time() + timeout_s
    last = ""
    while time.time() < deadline:
        if not is_running(name):
            raise DockerError(
                f"container {name} exited during startup.\n--- logs ---\n{logs(name, 200)}"
            )
        proc = exec_in(name, probe, timeout=20)
        if proc.returncode == 0:
            return
        last = ((proc.stdout or "") + (proc.stderr or "")).strip()
        time.sleep(interval_s)
    raise DockerError(
        f"container {name} did not become ready within {timeout_s}s.\n"
        f"last probe output: {last}\n--- logs ---\n{logs(name, 200)}"
    )


# ---------------------------------------------------------------------------
# Resource sampling
# ---------------------------------------------------------------------------

class MemorySampler(threading.Thread):
    """Sample a container's memory and CPU into a JSONL timeseries.

    A timeseries rather than a single peak, for two reasons: the peak of each
    phase can be derived from it after the fact by intersecting with that
    phase's time window, and a memory curve shows things a scalar cannot — a
    graph cache filling up, a build spiking, an engine steadily leaking.
    """

    def __init__(self, container: str, output_path: str, interval_s: float = 0.25):
        super().__init__(daemon=True)
        self.container = container
        self.output_path = output_path
        self.interval_s = interval_s
        self._stop_event = threading.Event()
        self.samples = 0

    def _read(self, path: str) -> Optional[int]:
        proc = exec_in(self.container, ["cat", path], timeout=10)
        value = (proc.stdout or "").strip()
        return int(value) if value.isdigit() else None

    def _cpu_seconds(self) -> Optional[float]:
        proc = exec_in(self.container, ["cat", "/sys/fs/cgroup/cpu.stat"], timeout=10)
        for line in (proc.stdout or "").splitlines():
            if line.startswith("usage_usec"):
                try:
                    return int(line.split()[1]) / 1e6
                except (IndexError, ValueError):
                    return None
        return None

    def run(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
        with open(self.output_path, "a", buffering=1) as fh:
            while not self._stop_event.is_set():
                if not is_running(self.container):
                    break
                current = (self._read("/sys/fs/cgroup/memory.current")
                           or self._read("/sys/fs/cgroup/memory/memory.usage_in_bytes"))
                peak = (self._read("/sys/fs/cgroup/memory.peak")
                        or self._read("/sys/fs/cgroup/memory/memory.max_usage_in_bytes"))
                if current is not None:
                    fh.write(json.dumps({
                        "t": round(time.time(), 3),
                        "container": self.container,
                        "rss_bytes": current,
                        "peak_bytes": peak,
                        "cpu_seconds": self._cpu_seconds(),
                    }) + "\n")
                    self.samples += 1
                self._stop_event.wait(self.interval_s)

    def stop(self) -> None:
        self._stop_event.set()
        self.join(timeout=10)


def running_containers() -> List[str]:
    """Names of vector-bench containers currently running."""
    proc = _run(["docker", "ps", "--format", "{{.Names}}",
                 "--filter", f"label={LABEL}=1"], check=False, timeout=60)
    return [n for n in (proc.stdout or "").split() if n]


def remove_tree_as_root(path: str, image: str) -> bool:
    """Delete a host directory that a container filled as root.

    Engines run as root inside their containers, so everything they write to a
    bind mount is root-owned. shutil.rmtree cannot remove it, and with
    ignore_errors=True it fails silently, which is how a teardown that looked
    like it worked left tens of GB behind.

    Returns True if the path is gone afterwards.
    """
    if not os.path.exists(path):
        return True
    if os.getuid() == 0:
        shutil.rmtree(path, ignore_errors=True)
        return not os.path.exists(path)

    parent, name = os.path.dirname(path), os.path.basename(path)
    spec = ContainerSpec(
        name=f"vb-rm-{os.getpid()}",
        image=image,
        network="none",
        entrypoint="rm",
        command=["-rf", f"/target/{name}"],
        volumes=[f"{parent}:/target:rw"],
        detach=False,
    )
    try:
        run_foreground(spec, timeout=600, stream=False)
    except Exception as exc:  # pragma: no cover - depends on the daemon
        print(f"[clean] could not remove {path}: {exc}", file=sys.stderr)
    return not os.path.exists(path)


def cleanup_run(run_id: str = "") -> Dict[str, int]:
    """Remove every container, network and volume belonging to a run.

    With no run_id this removes everything the harness has ever created. Always
    scoped by the vector-bench label, so nothing else on a shared box is
    touched. Returns counts per kind.
    """
    removed: Dict[str, int] = {}
    for kind in ("container", "network", "volume"):
        filters = ["--filter", f"label={LABEL}=1"]
        # An empty --filter name= is not "match everything", it is a filter on
        # the empty string, and whether that matches is version-dependent.
        if run_id:
            filters += ["--filter", f"name={run_id}"]
        proc = _run(["docker", kind, "ls", "-q", *filters], check=False, timeout=60)
        ids = [i for i in (proc.stdout or "").split() if i]
        removed[kind] = len(ids)
        if not ids:
            continue
        force = ["-f"] if kind in ("container", "volume") else []
        _run(["docker", kind, "rm", *force, *ids], check=False, timeout=300)
    return removed
