"""Recall/QPS measurement path: ann-benchmarks, run inside each engine image.

ann-benchmarks is used unmodified. Rather than letting its own Docker runner
launch containers (which would give us no control over env, cpuset or memory),
the orchestrator launches the container itself and runs `run.py --local` inside
it. Local mode executes the algorithm in-process, so ann-benchmarks still owns
definitions, ground truth, recall computation and result files, while the
orchestrator owns the container's resource limits. No patching of the upstream
project is required.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
from typing import Any, Dict, List, Optional

import yaml

from . import docker_ctl
from .config import ResolvedResources, server_args

# Constructor class names, as ann-benchmarks expects them in config.yml.
CONSTRUCTORS = {"mariadb": "MariaDB", "alisql": "AliSQL", "pgvector": "PGVector"}

# Where each engine's data directory lives inside its image. The ops path
# mounts a volume here; the ann path binds a host directory for the same reason.
DATA_MOUNT = {
    "mariadb": "/var/lib/vbench",
    "alisql": "/var/lib/vbench",
    "pgvector": "/var/lib/postgresql",
}

# ann-benchmarks raises this when every configuration is already done.
# See ann_benchmarks/main.py: `raise Exception("Nothing to run")`.
NOTHING_TO_RUN = "Nothing to run"


# Bump when the measurement path itself changes in a way that makes older
# numbers non-comparable, even at an identical resource configuration. Adding
# the per-configuration warmup was such a change: without it the first
# ef_search point paid for a cold cache and landed below the second.
ANN_MEASUREMENT_VERSION = 2


def ann_fingerprint(resolved: Any) -> str:
    """Short digest of everything that makes two ann runs incomparable.

    ann-benchmarks caches by algorithm and index parameters. It has no idea how
    much memory the engine was given or how many cores it had, so a result
    measured under one budget is silently reused under another.

    That is not hypothetical. A dbpedia run was re-launched at 64 GB after the
    first attempt had run at 16 GB, and every recall point came back
    byte-identical to the old ones because the files were already on disk. The
    report then carried a manifest saying 64 GB above a curve measured at 16.
    """
    # Accepts a ResolvedResources or the dict a manifest stores, so the report
    # can recompute the fingerprint without rebuilding the resource objects.
    get = (resolved.get if isinstance(resolved, dict)
           else lambda k, d=None: getattr(resolved, k, d))

    # Deliberately engine-invariant. The individual cache figures differ by
    # engine by design -- pgvector has no separate graph cache, so its buffer
    # absorbs that share -- and hashing them gave every engine its own results
    # tree. The report can only narrow to one tree, so a three-engine run
    # produced a recall chart containing a single engine.
    #
    # The *total* handed out is identical across engines, so it captures a
    # change to the fractions without fragmenting by engine.
    # Quantised to MiB. int(64GB * 0.30) * 2 and int(64GB * 0.60) differ by a
    # single byte from float rounding, which was enough to split the engines
    # into separate trees again.
    allocated = sum(int(get(k) or 0) for k in
                    ("buffer_bytes", "graph_cache_bytes", "maintenance_bytes"))
    allocated //= 1024 * 1024
    fields = (
        ANN_MEASUREMENT_VERSION,
        get("server_memory_bytes"),
        allocated,
        get("build_threads"),
        get("server_cpu_count"),
    )
    blob = "|".join(str(f) for f in fields).encode()
    return hashlib.sha256(blob).hexdigest()[:8]


def annb_results_dir(paths: Dict[str, str], resource_pass: str,
                     resolved: Optional[ResolvedResources] = None) -> str:
    """Result tree for one resource pass and one resource configuration.

    ann-benchmarks derives its result filenames from the algorithm and its
    parameters, with no notion of a resource pass. Sharing one tree would make
    the tuned pass skip every point the normalized pass already computed — and
    the report would then present normalized numbers as tuned ones. Separate
    trees keep the two passes independent and keep resumption working within
    each.

    The configuration fingerprint extends the same reasoning to the budget: a
    curve measured at 16 GB is not a curve measured at 64 GB, and resumption
    should not treat them as interchangeable. Runs at an unchanged
    configuration still resume normally, because the fingerprint is unchanged.
    """
    parts = [paths["annb_results"], resource_pass]
    if resolved is not None:
        parts.append(ann_fingerprint(resolved))
    path = os.path.join(*parts)
    os.makedirs(path, exist_ok=True)
    return path


def render_config(engine: str, profile: Dict[str, Any],
                  resources: Dict[str, Any], resource_pass: str) -> Dict[str, Any]:
    """Build the ann-benchmarks config.yml body for one engine and profile.

    Generated rather than hand-maintained so a profile change cannot leave one
    engine sweeping a different grid from the others — which would look like a
    performance difference and would not be one.
    """
    ann = profile.get("ann", {}) or {}
    m_values = list(ann.get("m_values", [16]))
    query_args = [list(ann.get("ef_search", [10, 40, 160]))]
    extras = (resources.get("extras", {}) or {}) if resource_pass == "tuned" else {}

    run_groups: Dict[str, Any] = {}

    if engine in ("mariadb", "alisql"):
        if engine == "mariadb":
            storage_engines = list(
                extras.get("mariadb_storage_engines")
                or ann.get("mariadb_storage_engines", ["InnoDB"])
            )
        else:
            # VIDX is InnoDB-only; offering anything else would fail at DDL.
            storage_engines = ["InnoDB"]

        for storage in storage_engines:
            run_groups[storage.lower()] = {
                "arg_groups": [{"M": m_values, "engine": storage}],
                "args": {},
                "query_args": query_args,
            }

    elif engine == "pgvector":
        if resource_pass == "tuned" and extras.get("pgvector_ef_construction"):
            ef_constructions = list(extras["pgvector_ef_construction"])
        else:
            # Pinned in the normalized pass: ef_construction is the one build
            # knob MariaDB and AliSQL do not expose, so sweeping it there would
            # hand pgvector a tuning axis the others lack.
            ef_constructions = [int(ann.get("pgvector_ef_construction", 200))]

        for build_mode in ann.get("pgvector_build_modes", ["post"]):
            run_groups[f"{build_mode}_build"] = {
                "arg_groups": [{
                    "M": m_values,
                    "efConstruction": ef_constructions,
                    "build_mode": build_mode,
                }],
                "args": {},
                "query_args": query_args,
            }
    else:
        raise ValueError(f"unknown engine: {engine}")

    return {
        "float": {
            "any": [{
                "base_args": ["@metric"],
                "constructor": CONSTRUCTORS[engine],
                "disabled": False,
                "docker_tag": f"vector-bench/{engine}-bench",
                "module": f"ann_benchmarks.algorithms.{engine}",
                "name": engine,
                "run_groups": run_groups,
            }]
        }
    }


def write_config(work_dir: str, engine: str, body: Dict[str, Any]) -> str:
    """Write the rendered config into the disposable ann-benchmarks working copy."""
    path = os.path.join(work_dir, "ann_benchmarks", "algorithms", engine, "config.yml")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = (
        f"# GENERATED by vector-bench for engine '{engine}'. Do not edit.\n"
        f"# Regenerate with: ./run-benchmark.sh render --profile <name>\n"
    )
    with open(path, "w") as fh:
        fh.write(header)
        yaml.safe_dump(body, fh, sort_keys=False, default_flow_style=False)
    return path


# In the ann pass the database server and the ann-benchmarks client share one
# container, so they also share one cgroup memory limit. The client's share is
# not small and is not constant: ann-benchmarks loads the corpus into RAM twice
# over — once in the parent process (main.py calls get_dataset() to learn the
# dimension) and again inside the forked worker (runner.run() calls it again on
# its own). Neither copy is shared; the second is a fresh h5py read.
#
# 2.0x for those two copies, 0.2x for the transient float32 conversion numpy
# makes while reading, and a flat 1 GB for the interpreter, the connector and
# the query set.
_CLIENT_COPIES = 2.2
_CLIENT_BASE_BYTES = 1 * 1024**3


def _host_ram_bytes() -> int:
    """Total host RAM, or 0 if it cannot be determined."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return 0


def client_memory_bytes(datasets_dir: str, dataset: str) -> int:
    """Estimate what the ann-benchmarks client needs resident for this dataset.

    Sized from the HDF5 file on disk rather than a lookup table so that
    generated corpora (the dbpedia family) are covered without anyone having to
    remember to add them. The train matrix is >98% of these files, which makes
    file size a good proxy and, unlike h5py, one the host orchestrator can read
    without any dependency at all.
    """
    path = os.path.join(datasets_dir, f"{dataset}.hdf5")
    try:
        size = os.path.getsize(path)
    except OSError:
        return _CLIENT_BASE_BYTES
    return int(size * _CLIENT_COPIES) + _CLIENT_BASE_BYTES


def run_engine(engine: str, dataset: str, profile: Dict[str, Any],
               engine_cfg: Dict[str, Any], resolved: ResolvedResources,
               resource_pass: str, paths: Dict[str, str], run_id: str,
               force: bool = False, timeout_s: int = 24 * 3600) -> int:
    """Run the ann-benchmarks sweep for one engine on one dataset."""
    image = engine_cfg.get("image", {}).get("bench", f"vector-bench/{engine}-bench")
    if not docker_ctl.image_exists(image):
        raise docker_ctl.DockerError(
            f"image {image} not found. Build it first:\n"
            f"  ./run-benchmark.sh build --engines {engine}"
        )

    flags = server_args(engine_cfg, resource_pass, resolved)
    container = f"{run_id}-annb-{engine}-{dataset}".replace("_", "-")[:60]

    # The engine's own budget stays exactly as the resource profile specifies —
    # that is what makes the normalized pass comparable across engines. The
    # client's copies of the corpus are not an engine resource, so they are
    # added on top of the container limit rather than taken out of it.
    #
    # Sizing the container to the engine budget alone silently capped the client
    # too: on dbpedia-openai-1000k (6.1 GB corpus) the worker was OOM-killed the
    # moment it finished loading, and because only the forked child died, the
    # parent exited 0 with no results and no error.
    client_bytes = client_memory_bytes(paths["datasets"], dataset)
    container_memory_bytes = resolved.server_memory_bytes + client_bytes

    # Without this the engine writes its data directory into the container's
    # writable layer, which lives under Docker's data-root — usually the root
    # filesystem, and usually not where the space is. A pgvector run died at
    # `initdb: could not create directory ".../pg_wal": No space left on
    # device` while the filesystem holding the checkout had 100+ GB free.
    #
    # Each configuration builds a fresh index, so the directory is cleared
    # rather than reused; leaving it would also make the ingest measurement
    # depend on whatever the previous run left behind.
    state_dir = os.path.join(paths["engine_state"], "annb", f"{resource_pass}-{engine}")
    shutil.rmtree(state_dir, ignore_errors=True)
    os.makedirs(state_dir, exist_ok=True)

    command = [
        "run.py", "--local",
        "--algorithm", engine,
        "--dataset", dataset,
        "--count", str(profile.get("k", 10)),
        "--runs", str(profile.get("runs", 1)),
        "--parallelism", "1",
    ]
    if profile.get("ann", {}).get("batch"):
        command.append("--batch")
    if force:
        command.append("--force")

    spec = docker_ctl.ContainerSpec(
        name=container,
        image=image,
        # No network needed: the dataset is mounted and the server is in-process.
        network="none",
        cpuset=resolved.server_cpuset,
        memory_bytes=container_memory_bytes,
        shm_size=resolved.shm_size,
        entrypoint="python3",
        workdir="/home/app",
        env={
            "VB_SERVER_ARGS": " ".join(flags),
            "VB_INSERT_THREADS": str(profile.get("ann", {}).get("insert_threads", 0)),
            "VB_RUN_ID": run_id,
            "VB_RESOURCE_PASS": resource_pass,
            "PYTHONUNBUFFERED": "1",
        },
        volumes=[
            f"{paths['work_annb']}:/home/app:rw",
            f"{paths['datasets']}:/home/app/data:rw",
            f"{annb_results_dir(paths, resource_pass, resolved)}:/home/app/results:rw",
            f"{state_dir}:{DATA_MOUNT[engine]}:rw",
        ],
        command=command,
        detach=False,
    )

    print(f"[ann] {engine} / {dataset}: cpuset={resolved.server_cpuset} "
          f"mem={container_memory_bytes / 1024**3:.1f}GB "
          f"(engine {resolved.server_memory_bytes / 1024**3:.1f}GB "
          f"+ client {client_bytes / 1024**3:.1f}GB for the in-RAM corpus)")
    print(f"[ann] server flags: {' '.join(flags)}")

    # Better to say this now than to have the OOM killer say it later, when it
    # will arrive as a silent zero-result exit rather than an error.
    host_ram = _host_ram_bytes()
    if host_ram and container_memory_bytes > host_ram * 0.9:
        print(
            f"[ann] WARNING: this container needs "
            f"{container_memory_bytes / 1024**3:.1f} GB but the host has only "
            f"{host_ram / 1024**3:.1f} GB. Expect the kernel to kill the client "
            f"mid-load. Lower memory.server_limit_gb in the resource profile, or "
            f"use a smaller corpus.",
            file=sys.stderr,
        )

    results_dir = annb_results_dir(paths, resource_pass, resolved)
    before = _count_results(results_dir, engine, dataset)

    # Warn before the fact, not after. ann-benchmarks signals "everything is
    # already done" by raising and printing a full traceback, which is
    # indistinguishable from a crash unless you already know to expect it.
    if before and not force:
        print(f"[ann] {before} existing result file(s) for {engine} / "
              f"{dataset}; already-computed configurations will be skipped")
    output: List[str] = []
    rc = docker_ctl.run_foreground(
        spec, timeout=timeout_s, sink=output,
        # Only filter when a benign traceback is actually possible.
        line_filter=_SuppressNothingToRun() if (before and not force) else None,
    )
    after = _count_results(results_dir, engine, dataset)
    text = "\n".join(output)

    # ann-benchmarks treats "every configuration already has results" as an
    # error: main() raises Exception("Nothing to run") and the process exits
    # non-zero. For us that is successful resumption, not failure — the whole
    # point of leaving the results tree in place between runs. Recognise it
    # rather than reporting three healthy engines as broken.
    if rc != 0 and NOTHING_TO_RUN in text:
        if after > 0:
            print(
                f"[ann] nothing to do: all {after} configuration(s) for "
                f"{engine} / {dataset} already have results in {results_dir}. "
                f"Pass --force to recompute them."
            )
            return 0
        print(
            f"[ann] FAILED: ann-benchmarks had nothing to run and there are no "
            f"existing results in {results_dir}. The rendered config matched no "
            f"definitions, or the algorithm module failed to import.",
            file=sys.stderr,
        )
        return 1

    # A clean exit that produced nothing at all, with nothing already present,
    # means every configuration errored inside the run loop — ann-benchmarks
    # logs those per-definition and still exits 0. Without this check a whole
    # engine disappears from the report as an apparent success.
    if rc == 0 and after == 0:
        # ann-benchmarks runs each definition in a forked worker and never
        # checks its exit code: main() joins, logs "Terminating N workers" from
        # a finally block, and returns 0. So a worker killed by the OOM killer
        # is indistinguishable from a clean run except that nothing was written.
        # That is by far the most common cause here, because the client holds
        # the whole corpus in RAM twice, so name it first and show the numbers.
        killed_silently = "Terminating" in text and "Traceback" not in text
        print(
            f"[ann] FAILED: {engine} / {dataset} exited 0 but produced no result "
            f"files at all in {results_dir}.",
            file=sys.stderr,
        )
        if killed_silently:
            print(
                f"[ann] The worker died without raising, which almost always means "
                f"the kernel OOM-killed it: ann-benchmarks does not check worker "
                f"exit codes, so a kill shows up as a silent zero-result success.\n"
                f"[ann]   container limit : {container_memory_bytes / 1024**3:.1f} GB\n"
                f"[ann]   engine budget   : {resolved.server_memory_bytes / 1024**3:.1f} GB "
                f"(buffer pool is allocated up front)\n"
                f"[ann]   client estimate : {client_bytes / 1024**3:.1f} GB "
                f"(corpus is held in RAM twice)\n"
                f"[ann] Confirm with: dmesg -T | tail  — look for Killed process ... python3",
                file=sys.stderr,
            )
        else:
            print(
                f"[ann] The algorithm module most likely failed to import, or every "
                f"configuration errored — check the container output above.",
                file=sys.stderr,
            )
        return 1
    if rc == 0:
        if after == before:
            print(f"[ann] nothing to do: all {after} configuration(s) for "
                  f"{engine} / {dataset} already have results "
                  f"(pass --force to recompute)")
        else:
            print(f"[ann] {after - before} new result file(s), {after} total")
    return rc


class _SuppressNothingToRun:
    """Hide ann-benchmarks' traceback when it only means "already done".

    ann-benchmarks reports "every configuration has results" by raising, so it
    prints a full Python traceback for a condition that is completely normal
    here. Explaining that in advance was not enough — a traceback in the output
    reads as a crash no matter what precedes it.

    Lines are held from "Traceback (most recent call last):" onwards and
    released only if the traceback turns out to be something else. A real
    failure is therefore never swallowed, merely delayed by a few lines.
    """

    def __init__(self) -> None:
        self._held: Optional[List[str]] = None

    def __call__(self, line: Optional[str]) -> List[str]:
        if line is None:                              # end of stream
            held, self._held = self._held, None
            return held or []

        if self._held is None:
            if line.startswith("Traceback (most recent call last)"):
                self._held = [line]
                return []
            return [line]

        self._held.append(line)
        if line.startswith("Exception: ") or line.startswith("  File "):
            if NOTHING_TO_RUN in line:
                self._held = None                     # benign: drop it entirely
                return []
            if line.startswith("Exception: "):
                held, self._held = self._held, None   # some other error: show it
                return held
        return []


def _count_results(results_dir: str, engine: str, dataset: str) -> int:
    """Number of ann-benchmarks result files for this engine and dataset."""
    root = os.path.join(results_dir, dataset)
    if not os.path.isdir(root):
        return 0
    return sum(
        len([f for f in files if f.endswith(".hdf5")])
        for base, _dirs, files in os.walk(root)
        if os.path.basename(base) == engine
    )


def fix_ownership(path: str, image: str) -> None:
    """Return root-created result files to the invoking user.

    The engines run as root inside their containers, so anything they write to a
    bind mount lands root-owned on the host. Left alone that makes the results
    directory unreadable to the user who started the run.
    """
    uid, gid = os.getuid(), os.getgid()
    if uid == 0:
        return
    spec = docker_ctl.ContainerSpec(
        name=f"vb-chown-{os.getpid()}",
        image=image,
        entrypoint="chown",
        command=["-R", f"{uid}:{gid}", "/target"],
        volumes=[f"{path}:/target:rw"],
        detach=False,
        network="none",
    )
    docker_ctl.run_foreground(spec, timeout=600, stream=False)
