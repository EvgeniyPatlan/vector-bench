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


def merge_resource_overrides(resources: Dict[str, Any],
                             profile: Dict[str, Any]) -> Dict[str, Any]:
    """Apply a profile's `resources:` block on top of the resource pass.

    Resource passes are sized for the common case, and the common case is a
    corpus that fits comfortably. A profile that loads a million 1536-dimension
    vectors needs a bigger budget than one that loads 60,000 at 784, and without
    this the only options were to edit the shared pass (changing every other
    profile) or to run under a budget that cannot hold the data.

    Returns a new dict; the inputs are not modified, so a caller can resolve
    several profiles against one loaded pass.
    """
    override = (profile or {}).get("resources") or {}
    if not override:
        return resources

    def deep_merge(base: Dict[str, Any], top: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(base)
        for key, value in top.items():
            if isinstance(value, dict) and isinstance(out.get(key), dict):
                out[key] = deep_merge(out[key], value)
            else:
                out[key] = value
        return out

    merged = deep_merge(resources, override)
    merged["_overridden_by_profile"] = sorted(override)
    return merged


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
    #: JVM heap for mongot. Zero for every engine that is a single process.
    mongot_heap_bytes: int
    #: Valkey's maxmemory. Zero for every engine that is not in-memory.
    maxmemory_bytes: int
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
            "mongot_heap_bytes": self.mongot_heap_bytes,
            "maxmemory_bytes": self.maxmemory_bytes,
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

    maxmemory_bytes = 0
    if engine == "valkey":
        # Nothing here lives on disk, so there is no share of a larger thing to
        # reserve. buffer_fraction and graph_cache_fraction both describe memory
        # set aside to hold part of something bigger; carving them out of an
        # in-memory store would leave the data less room than the container was
        # given, for no benefit.
        buffer_fraction = 0.0
        graph_fraction = 0.0
        maint_fraction = 0.0

        # maxmemory sits below the container limit on purpose. Reaching
        # maxmemory under noeviction returns an error the driver can report;
        # reaching the container limit gets the process OOM-killed, which
        # arrives as a crash with no cause attached to it.
        headroom = float((resources.get("memory", {}) or {}).get(
            "valkey_maxmemory_fraction", 0.9))
        maxmemory_bytes = int(server_bytes * headroom)

        # An in-memory engine given less memory than the dataset does not get
        # slower, it fails partway through the load. Saying so now beats
        # discovering it after an hour of writing.
        corpus_bytes = int((resources.get("memory", {}) or {}).get(
            "expected_corpus_bytes", 0) or 0)
        if not corpus_bytes:
            # 990k x 1536 float32 plus the HNSW graph and Valkey's per-key
            # overhead. Deliberately generous: a false warning costs a sentence
            # and a missed one costs the run.
            corpus_bytes = int(16 * GB)
        if maxmemory_bytes < corpus_bytes:
            warnings.append(
                f"the corpus does not fit: maxmemory resolves to "
                f"{maxmemory_bytes / GB:.1f} GB and an in-memory engine needs "
                f"roughly {corpus_bytes / GB:.0f} GB for 990k x 1536 vectors "
                f"plus the graph. Valkey will refuse writes partway through the "
                f"load rather than running slowly."
            )

    mongot_heap_bytes = 0
    if engine == "mongodb":
        # The opposite of the pgvector rule above, for the opposite reason.
        #
        # Percona Search is two processes: mongod holds the documents in its
        # WiredTiger cache, and mongot holds the index as memory-mapped Lucene
        # segments served from the OS filesystem cache. The filesystem cache is
        # unallocated memory by definition, so the graph share cannot be
        # absorbed into anything -- it has to be left free. Giving it to either
        # process starves the cache that actually answers queries, which is
        # what MongoDB's guidance means by warning against a heap above 50%.
        graph_fraction = 0.0

        # There is no maintenance_work_mem equivalent either. WiredTiger has no
        # such knob, and mongot's index build is heap work, already accounted
        # above. Reserving a maintenance share would take memory from the page
        # cache and, worse, size /dev/shm from it: that term exists for
        # pgvector's parallel HNSW build, which has no counterpart here.
        maint_fraction = 0.0

        # Heap scales with the number of indexed fields, not with the number of
        # vectors. This index has one vector field and one filter field, so a
        # large heap buys nothing and costs page cache.
        heap_gb = float(mem_cfg.get("mongot_heap_gb", 8) or 8)
        mongot_heap_bytes = int(heap_gb * GB)

        # Past roughly 30 GB the JVM drops compressed object pointers and every
        # reference doubles in width. MongoDB's advice is to stay below the
        # boundary or jump straight to 48 GB; nothing here needs 48.
        compressed_oops_ceiling = int(30 * GB)
        if mongot_heap_bytes > compressed_oops_ceiling:
            warnings.append(
                f"mongot heap of {heap_gb:.0f} GB is above the ~30 GB "
                f"compressed object pointer boundary, where every reference "
                f"doubles in width; capping at 30 GB. Heap scales with indexed "
                f"field count, not vector count, so this is very unlikely to "
                f"be the limit that matters."
            )
            mongot_heap_bytes = compressed_oops_ceiling

        # And never more than half the container, or there is nothing left for
        # the segments themselves.
        half = server_bytes // 2
        if mongot_heap_bytes > half:
            warnings.append(
                f"mongot heap of {mongot_heap_bytes / GB:.1f} GB is over half "
                f"the {server_bytes / GB:.1f} GB container budget, leaving "
                f"insufficient filesystem cache for the memory-mapped index "
                f"segments that serve queries; capping at {half / GB:.1f} GB"
            )
            mongot_heap_bytes = half

    buffer_bytes = int(server_bytes * buffer_fraction)
    graph_cache_bytes = int(server_bytes * graph_fraction)
    maintenance_bytes = int(server_bytes * maint_fraction)

    total_allocated = (buffer_bytes + graph_cache_bytes + maintenance_bytes
                       + mongot_heap_bytes)
    if total_allocated > server_bytes * 0.9:
        warnings.append(
            "buffer + graph cache + maintenance exceeds 90% of the container "
            "memory limit, leaving little for connections and sorting; "
            "the engine may be OOM-killed"
        )

    build_threads = int(build_cfg.get("threads", 1) or 0)
    if build_threads <= 0:
        build_threads = len(server_cpus)

    # Hard cap. "threads: 0" means "match the cpuset", which on a 40-core server
    # resolved to 76 parallel maintenance workers -- absurd for an index build,
    # and enough to exhaust the container's /dev/shm with dynamic shared memory
    # segments, which is exactly how pgvector's tuned pass died. Parallel index
    # build stops paying past a handful of workers regardless.
    max_threads = int(build_cfg.get("max_threads", 8) or 8)
    if build_threads > max_threads:
        warnings.append(
            f"build threads clamped from {build_threads} to {max_threads}; "
            f"parallel index build does not benefit beyond a handful of workers "
            f"and each one consumes shared memory (raise build.max_threads to override)"
        )
        build_threads = max_threads

    # Dynamic shared memory for parallel workers lives in /dev/shm. Size it from
    # the parallelism actually granted rather than trusting a fixed value that
    # silently becomes too small when build.threads is raised.
    shm_size = str(docker_cfg.get("shm_size", "2g"))
    try:
        shm_gb = float(shm_size.rstrip("gG"))

        # Two independent demands on /dev/shm, and the larger one wins.
        #
        # Per-worker segments scale with the worker count. That was the only
        # term here originally, and it is the smaller of the two by a wide
        # margin once maintenance_work_mem is generous.
        per_worker_gb = 1.0 + 0.5 * build_threads

        # pgvector's parallel HNSW build puts the graph in a dynamic shared
        # memory segment sized from maintenance_work_mem, so /dev/shm has to
        # hold the whole thing. With maintenance_work_mem at 11.25 GB and
        # shm at 8g the build died with
        #   could not resize shared memory segment ... to 12078927552 bytes:
        #   No space left on device
        # after 27 minutes. 12078927552 is exactly maintenance_work_mem, which
        # is what identifies the term that was missing.
        build_segment_gb = (maintenance_bytes / GB) * 1.1 + 1.0

        needed_gb = max(per_worker_gb, build_segment_gb)

        # /dev/shm is tmpfs and counts against the container's memory limit, so
        # it cannot be sized past the budget it is being carved out of.
        ceiling_gb = (server_bytes / GB) * 0.5
        if needed_gb > ceiling_gb:
            warnings.append(
                f"a parallel index build wants {needed_gb:.0f}g of /dev/shm but "
                f"that is over half the {server_bytes / GB:.0f}g container "
                f"budget; capping at {ceiling_gb:.0f}g. Lower "
                f"memory.maintenance_fraction or build.max_threads if the build "
                f"fails with 'No space left on device'."
            )
            needed_gb = ceiling_gb

        if shm_gb < needed_gb:
            warnings.append(
                f"shm_size raised from {shm_size} to {needed_gb:.0f}g: a "
                f"parallel index build needs /dev/shm to hold a segment the "
                f"size of maintenance_work_mem "
                f"({maintenance_bytes / GB:.1f}g), plus room for "
                f"{build_threads} worker(s)"
            )
            shm_size = f"{needed_gb:.0f}g"
    except ValueError:
        pass

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
        mongot_heap_bytes=mongot_heap_bytes,
        maxmemory_bytes=maxmemory_bytes,
        build_threads=build_threads,
        shm_size=shm_size,
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
