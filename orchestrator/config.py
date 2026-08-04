"""Configuration loading and resolution for the orchestrator.

Three config layers combine into one resolved plan:

  profiles/<name>.yml    what to measure (datasets, grids, workloads)
  resources/<pass>.yml   how much machine each engine gets
  engines/<engine>.yml   how each engine is built, started and spoken to

Resolution turns fractions and "0 means auto" placeholders into concrete
numbers — cpusets, byte counts, server flags — because a manifest that records
"buffer_fraction: 0.35" tells a reader nothing, while one that records
"innodb_buffer_pool_size=6012954214" is reproducible.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

VB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(VB_ROOT, "config")

GB = 1024 ** 3


def _load(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"config not found: {path}")
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def load_profile(name: str) -> Dict[str, Any]:
    return _load(os.path.join(CONFIG_DIR, "profiles", f"{name}.yml"))


def load_resources(name: str) -> Dict[str, Any]:
    return _load(os.path.join(CONFIG_DIR, "resources", f"{name}.yml"))


def load_engine(name: str) -> Dict[str, Any]:
    return _load(os.path.join(CONFIG_DIR, "engines", f"{name}.yml"))


def available_profiles() -> List[str]:
    d = os.path.join(CONFIG_DIR, "profiles")
    return sorted(f[:-4] for f in os.listdir(d) if f.endswith(".yml"))


@dataclass
class ResolvedResources:
    """Concrete resource settings, ready to hand to Docker and to the servers."""

    name: str
    server_cpuset: str
    client_cpuset: str
    server_cpu_count: int
    server_memory_bytes: int
    client_memory_bytes: int
    buffer_bytes: int
    graph_cache_bytes: int
    maintenance_bytes: int
    build_threads: int
    shm_size: str
    transaction_isolation: str
    hybrid_cpu: bool
    core_class_used: str
    warnings: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "server_cpuset": self.server_cpuset,
            "client_cpuset": self.client_cpuset,
            "server_cpu_count": self.server_cpu_count,
            "server_memory_bytes": self.server_memory_bytes,
            "client_memory_bytes": self.client_memory_bytes,
            "buffer_bytes": self.buffer_bytes,
            "graph_cache_bytes": self.graph_cache_bytes,
            "maintenance_bytes": self.maintenance_bytes,
            "build_threads": self.build_threads,
            "shm_size": self.shm_size,
            "transaction_isolation": self.transaction_isolation,
            "hybrid_cpu": self.hybrid_cpu,
            "core_class_used": self.core_class_used,
            "warnings": self.warnings,
        }


def resolve_resources(resources: Dict[str, Any], engine: str,
                      sysinfo: Any) -> ResolvedResources:
    """Turn a resource profile plus the detected hardware into concrete limits."""
    from harness.metrics.sysinfo import format_cpuset, recommended_cpuset

    cpu_cfg = resources.get("cpu", {}) or {}
    mem_cfg = resources.get("memory", {}) or {}
    build_cfg = resources.get("build", {}) or {}
    docker_cfg = resources.get("docker", {}) or {}
    iso_cfg = resources.get("isolation", {}) or {}

    warnings: List[str] = []

    prefer = cpu_cfg.get("prefer_cores", "performance")
    allow_smt = bool(cpu_cfg.get("allow_smt", False))

    want_server = int(cpu_cfg.get("server_cpus", 0) or 0)
    want_client = int(cpu_cfg.get("client_cpus", 2) or 2)

    pool = recommended_cpuset(10 ** 6, prefer=prefer, allow_smt=allow_smt)
    if not pool:
        pool = list(range(sysinfo.cpu.logical_cpus))
        warnings.append("could not determine core topology; using all logical CPUs")

    if want_server <= 0:
        # "all available", minus what the client needs.
        want_server = max(1, len(pool) - want_client)

    if want_server + want_client > len(pool):
        warnings.append(
            f"requested {want_server} server + {want_client} client CPUs but only "
            f"{len(pool)} homogeneous CPUs are available "
            f"(prefer_cores={prefer}, allow_smt={allow_smt}); "
            f"server and client will share cores, which adds latency noise"
        )
        want_server = max(1, len(pool) - 1)
        want_client = max(1, len(pool) - want_server)

    server_cpus = pool[:want_server]
    client_cpus = pool[want_server:want_server + want_client] or pool[-want_client:]

    if sysinfo.cpu.hybrid and prefer == "performance" and not sysinfo.cpu.performance_cpus:
        warnings.append(
            "hybrid CPU detected but performance cores could not be identified; "
            "results will carry P-core/E-core scheduling noise"
        )

    # --- memory -------------------------------------------------------
    limit_gb = float(mem_cfg.get("server_limit_gb", 0) or 0)
    if limit_gb > 0:
        server_bytes = int(limit_gb * GB)
    else:
        fraction = float(mem_cfg.get("host_fraction", 0.6))
        server_bytes = int(sysinfo.total_ram_bytes * fraction)

    if server_bytes > sysinfo.total_ram_bytes:
        warnings.append(
            f"requested server memory ({server_bytes / GB:.1f} GB) exceeds host RAM "
            f"({sysinfo.total_ram_bytes / GB:.1f} GB); clamping"
        )
        server_bytes = int(sysinfo.total_ram_bytes * 0.8)

    client_bytes = int(float(mem_cfg.get("client_limit_gb", 8) or 8) * GB)

    buffer_fraction = float(mem_cfg.get("buffer_fraction", 0.35))
    graph_fraction = float(mem_cfg.get("graph_cache_fraction", 0.35))
    maint_fraction = float(mem_cfg.get("maintenance_fraction", 0.15))

    if engine == "pgvector" and mem_cfg.get("postgres_absorbs_graph_cache", True):
        # PostgreSQL has no vector-specific cache; graph pages live in
        # shared_buffers. Handing pgvector only the buffer fraction would give
        # it strictly less resident memory than the other two for the same
        # container limit, which would not be a fair normalization.
        buffer_fraction = buffer_fraction + graph_fraction
        graph_fraction = 0.0

    buffer_bytes = int(server_bytes * buffer_fraction)
    graph_cache_bytes = int(server_bytes * graph_fraction)
    maintenance_bytes = int(server_bytes * maint_fraction)

    total_allocated = buffer_bytes + graph_cache_bytes + maintenance_bytes
    if total_allocated > server_bytes * 0.9:
        warnings.append(
            "buffer + graph cache + maintenance exceeds 90% of the container "
            "memory limit, leaving little for connections and sorting; "
            "the engine may be OOM-killed"
        )

    build_threads = int(build_cfg.get("threads", 1) or 0)
    if build_threads <= 0:
        build_threads = len(server_cpus)

    return ResolvedResources(
        name=resources.get("name", "unnamed"),
        server_cpuset=format_cpuset(server_cpus),
        client_cpuset=format_cpuset(client_cpus),
        server_cpu_count=len(server_cpus),
        server_memory_bytes=server_bytes,
        client_memory_bytes=client_bytes,
        buffer_bytes=buffer_bytes,
        graph_cache_bytes=graph_cache_bytes,
        maintenance_bytes=maintenance_bytes,
        build_threads=build_threads,
        shm_size=str(docker_cfg.get("shm_size", "2g")),
        transaction_isolation=str(iso_cfg.get("transaction_isolation", "engine_default")),
        hybrid_cpu=bool(sysinfo.cpu.hybrid),
        core_class_used=prefer if sysinfo.cpu.hybrid else "uniform",
        warnings=warnings,
    )


def server_args(engine_cfg: Dict[str, Any], resource_pass: str,
                resolved: ResolvedResources) -> List[str]:
    """Render an engine's server flags with the resolved values substituted."""
    server = engine_cfg.get("server", {}) or {}
    args: List[str] = list(server.get("common", []) or [])
    args += list(server.get(resource_pass, []) or [])

    substitutions = {
        "buffer_bytes": resolved.buffer_bytes,
        "graph_cache_bytes": resolved.graph_cache_bytes,
        "maintenance_bytes": resolved.maintenance_bytes,
        "build_threads": resolved.build_threads,
        "server_cpu_count": resolved.server_cpu_count,
    }

    rendered: List[str] = []
    for arg in args:
        text = str(arg)
        for key, value in substitutions.items():
            text = text.replace("{" + key + "}", str(value))
        # A graph cache of zero is meaningless; drop the flag entirely rather
        # than passing 0 and having the engine reject it or disable caching.
        if resolved.graph_cache_bytes == 0 and "cache-size=0" in text.replace("_", "-"):
            continue
        rendered.append(text)
    return rendered
