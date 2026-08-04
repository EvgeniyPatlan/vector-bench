"""Host and container environment capture.

Every result carries its environment. A recall/QPS number without the CPU it was
produced on is not reproducible and not comparable — particularly here, where
both MariaDB MHNSW and AliSQL VIDX document AVX-512 distance kernels, so the
presence or absence of `avx512f` plausibly reorders the engines.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

# Flags that change which SIMD path a distance kernel takes. Recorded verbatim.
SIMD_FLAGS = (
    "avx512f", "avx512dq", "avx512bw", "avx512vl", "avx512vnni",
    "avx2", "avx", "fma", "avx_vnni", "f16c",
    "neon", "asimd", "sve",
)


@dataclass
class CpuInfo:
    model: str = "unknown"
    arch: str = ""
    logical_cpus: int = 0
    physical_cores: int = 0
    sockets: int = 1
    numa_nodes: int = 1
    threads_per_core: int = 1
    hybrid: bool = False
    performance_cpus: List[int] = field(default_factory=list)
    efficiency_cpus: List[int] = field(default_factory=list)
    simd_flags: List[str] = field(default_factory=list)
    has_avx512: bool = False


@dataclass
class SysInfo:
    cpu: CpuInfo
    total_ram_bytes: int = 0
    available_ram_bytes: int = 0
    kernel: str = ""
    os_release: str = ""
    docker_version: str = ""
    cgroup_version: str = ""
    hostname: str = ""
    python_version: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


def _read(path: str) -> Optional[str]:
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return None


def _cpuinfo_field(text: str, key: str) -> str:
    match = re.search(rf"^{re.escape(key)}\s*:\s*(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else ""


def _parse_cpulist(raw: Optional[str]) -> List[int]:
    """Parse a kernel cpulist string like '0-11,16,18-19'."""
    out: List[int] = []
    if not raw:
        return out
    for part in raw.strip().split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return sorted(out)


def detect_hybrid_topology() -> Dict[str, List[int]]:
    """Split logical CPUs into performance and efficiency cores.

    Intel hybrid parts (Alder Lake and later) mix P-cores and E-cores with very
    different throughput. Scheduling a benchmark across both produces run-to-run
    variance that easily exceeds the difference between the engines, so the
    normalized pass pins to one homogeneous set.

    Three detection strategies, most reliable first:

    1. ``/sys/devices/system/cpu/types/{intel_core,intel_atom}`` — authoritative,
       but only present on newer kernels.
    2. SMT asymmetry. On every shipped Intel hybrid part the P-cores have
       hyperthreading and the E-cores do not, so "has a thread sibling" cleanly
       separates the two classes.
    3. Frequency clustering, split at the largest *relative* gap between distinct
       max frequencies. Naively taking the single highest frequency is wrong:
       Turbo Boost Max 3.0 gives two favoured P-cores a slightly higher ceiling
       than their siblings, so a 6P+8E part reports three distinct frequencies
       and the top one covers only the two favoured cores.

    Returns empty lists on non-hybrid systems.
    """
    cpu_dir = "/sys/devices/system/cpu"
    if not os.path.isdir(cpu_dir):
        return {"performance": [], "efficiency": []}

    # --- 1. kernel-declared CPU types --------------------------------
    core = _parse_cpulist(_read(f"{cpu_dir}/types/intel_core/cpulist"))
    atom = _parse_cpulist(_read(f"{cpu_dir}/types/intel_atom/cpulist"))
    if core and atom:
        return {"performance": core, "efficiency": atom}

    # --- gather per-CPU facts for strategies 2 and 3 ------------------
    freqs: Dict[int, int] = {}
    for entry in sorted(os.listdir(cpu_dir)):
        match = re.fullmatch(r"cpu(\d+)", entry)
        if not match:
            continue
        cpu = int(match.group(1))
        value = (_read(f"{cpu_dir}/{entry}/cpu_capacity")
                 or _read(f"{cpu_dir}/{entry}/cpufreq/cpuinfo_max_freq"))
        if value is not None:
            try:
                freqs[cpu] = int(value)
            except ValueError:
                pass

    # --- 2. SMT asymmetry --------------------------------------------
    siblings = hyperthread_siblings()
    if siblings:
        smt = sorted(c for c, s in siblings.items() if len(s) > 1)
        non_smt = sorted(c for c, s in siblings.items() if len(s) == 1)
        if smt and non_smt:
            # Confirm with frequency where available: P-cores should not be the
            # slower group. Without this an unusual topology could invert the
            # classes silently.
            if freqs:
                mean_smt = sum(freqs.get(c, 0) for c in smt) / len(smt)
                mean_non = sum(freqs.get(c, 0) for c in non_smt) / len(non_smt)
                if mean_smt < mean_non:
                    return {"performance": non_smt, "efficiency": smt}
            return {"performance": smt, "efficiency": non_smt}

    # --- 3. frequency clustering at the largest relative gap ----------
    if not freqs:
        return {"performance": [], "efficiency": []}
    distinct = sorted(set(freqs.values()))
    if len(distinct) < 2:
        return {"performance": [], "efficiency": []}

    gaps = [
        ((distinct[i + 1] - distinct[i]) / distinct[i], i)
        for i in range(len(distinct) - 1)
    ]
    largest_gap, split_at = max(gaps)
    # A hybrid part separates its classes by a wide margin; small spreads are
    # just per-core turbo binning and must not be read as two core types.
    if largest_gap < 0.15:
        return {"performance": [], "efficiency": []}

    threshold = distinct[split_at]
    performance = sorted(c for c, v in freqs.items() if v > threshold)
    efficiency = sorted(c for c, v in freqs.items() if v <= threshold)
    return {"performance": performance, "efficiency": efficiency}


def hyperthread_siblings() -> Dict[int, List[int]]:
    """Map each logical CPU to its SMT siblings."""
    out: Dict[int, List[int]] = {}
    cpu_dir = "/sys/devices/system/cpu"
    if not os.path.isdir(cpu_dir):
        return out
    for entry in sorted(os.listdir(cpu_dir)):
        match = re.fullmatch(r"cpu(\d+)", entry)
        if not match:
            continue
        cpu = int(match.group(1))
        raw = _read(f"{cpu_dir}/{entry}/topology/thread_siblings_list")
        if not raw:
            continue
        siblings: List[int] = []
        for part in raw.split(","):
            if "-" in part:
                lo, hi = part.split("-")
                siblings.extend(range(int(lo), int(hi) + 1))
            else:
                siblings.append(int(part))
        out[cpu] = sorted(siblings)
    return out


def detect_cpu() -> CpuInfo:
    text = _read("/proc/cpuinfo") or ""
    flags = set((_cpuinfo_field(text, "flags") or _cpuinfo_field(text, "Features")).split())

    logical = os.cpu_count() or 0
    physical = len({
        (m.group(1), m.group(2))
        for m in re.finditer(
            r"physical id\s*:\s*(\d+)[\s\S]*?core id\s*:\s*(\d+)", text
        )
    }) or logical
    sockets = len(set(re.findall(r"^physical id\s*:\s*(\d+)$", text, re.MULTILINE))) or 1

    numa = 1
    node_dir = "/sys/devices/system/node"
    if os.path.isdir(node_dir):
        numa = len([d for d in os.listdir(node_dir) if re.fullmatch(r"node\d+", d)]) or 1

    hybrid = detect_hybrid_topology()

    return CpuInfo(
        model=_cpuinfo_field(text, "model name") or platform.processor() or "unknown",
        arch=platform.machine(),
        logical_cpus=logical,
        physical_cores=physical,
        sockets=sockets,
        numa_nodes=numa,
        threads_per_core=max(1, logical // physical) if physical else 1,
        hybrid=bool(hybrid["efficiency"]),
        performance_cpus=hybrid["performance"],
        efficiency_cpus=hybrid["efficiency"],
        simd_flags=sorted(f for f in SIMD_FLAGS if f in flags),
        has_avx512="avx512f" in flags,
    )


def detect_cgroup_version() -> str:
    if os.path.exists("/sys/fs/cgroup/cgroup.controllers"):
        return "v2"
    if os.path.isdir("/sys/fs/cgroup/memory"):
        return "v1"
    return "unknown"


def _meminfo(key: str) -> int:
    text = _read("/proc/meminfo") or ""
    match = re.search(rf"^{key}:\s+(\d+) kB$", text, re.MULTILINE)
    return int(match.group(1)) * 1024 if match else 0


def detect_docker_version() -> str:
    if not shutil.which("docker"):
        return ""
    try:
        return subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True, text=True, timeout=10, check=False,
        ).stdout.strip()
    except Exception:
        return ""


def collect() -> SysInfo:
    return SysInfo(
        cpu=detect_cpu(),
        total_ram_bytes=_meminfo("MemTotal"),
        available_ram_bytes=_meminfo("MemAvailable"),
        kernel=platform.release(),
        os_release=(_read("/etc/os-release") or "").split("\n")[0].replace('PRETTY_NAME=', '').strip('"'),
        docker_version=detect_docker_version(),
        cgroup_version=detect_cgroup_version(),
        hostname=platform.node(),
        python_version=platform.python_version(),
    )


def recommended_cpuset(want: int, prefer: str = "performance",
                       allow_smt: bool = False) -> List[int]:
    """Choose a homogeneous CPU set of `want` logical CPUs.

    Order of preference:
      1. Performance cores only, on hybrid CPUs.
      2. One logical CPU per physical core (no SMT siblings), unless allow_smt.
      3. Whatever remains.

    Returns fewer than `want` CPUs if the machine cannot supply them, rather
    than silently mixing core classes.
    """
    cpu = detect_cpu()
    if prefer == "performance" and cpu.performance_cpus:
        pool = list(cpu.performance_cpus)
    elif prefer == "efficiency" and cpu.efficiency_cpus:
        pool = list(cpu.efficiency_cpus)
    else:
        pool = list(range(cpu.logical_cpus))

    if not allow_smt:
        siblings = hyperthread_siblings()
        chosen: List[int] = []
        claimed = set()
        for c in pool:
            if c in claimed:
                continue
            chosen.append(c)
            claimed.update(siblings.get(c, [c]))
        pool = chosen

    return pool[:want]


def format_cpuset(cpus: List[int]) -> str:
    """Render a CPU list as a Docker --cpuset-cpus string, collapsing runs."""
    if not cpus:
        return ""
    cpus = sorted(cpus)
    parts, start, prev = [], cpus[0], cpus[0]
    for c in cpus[1:]:
        if c == prev + 1:
            prev = c
            continue
        parts.append(f"{start}-{prev}" if start != prev else f"{start}")
        start = prev = c
    parts.append(f"{start}-{prev}" if start != prev else f"{start}")
    return ",".join(parts)


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(collect().to_dict(), indent=2, sort_keys=True))
