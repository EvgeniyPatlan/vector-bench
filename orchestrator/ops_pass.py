"""Ops measurement path: build cost, concurrency, filtered search, churn.

Two containers, not one. The server runs alone in its own container so that its
cgroup accounting measures the server and nothing else — if the harness shared
that container, the several hundred megabytes of NumPy holding the dataset would
be charged to the engine and every peak-memory number would be wrong.

The harness runs in a second container on the same private network and connects
over TCP. That adds loopback network cost to every query, identically for all
three engines, which is why concurrency numbers here are compared against each
other rather than against the ann-benchmarks in-process numbers.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from typing import Any, Dict, List, Optional

from . import docker_ctl
from . import engines as engines_mod
from .config import ResolvedResources, server_args
from .manifest import utcnow


# How much of the server's own log to keep beside the measurements. Generous,
# because the lines that matter are the ones written while something was going
# wrong and there is no way to know in advance how much noise follows them.
# mongod is the verbose extreme at a few thousand lines an hour; the others
# write almost nothing.
SERVER_LOG_TAIL = 20000

# Readiness probes. Each performs a real query, not just a port check: all three
# servers accept connections before they are able to serve, and a premature
# start would charge initialisation time to the first measurement.

# Database account per engine. PostgreSQL's bootstrap superuser is `postgres`;
# the MySQL-family images create a `bench` account from their --init-file
# (--skip-grant-tables is unusable because on MySQL 8 it disables networking).



# The ops client loads the corpus once (unlike the ann client, which loads it
# twice) and then needs working space for the brute-force ground truth it
# computes over the qualifying subset. 1.5x the file plus a flat 2 GB covers
# both, and the caller takes the max against the configured client limit so
# small corpora keep the profile's value.
_OPS_CLIENT_COPIES = 1.5
_OPS_CLIENT_BASE_BYTES = 2 * 1024**3


def ops_client_memory_bytes(datasets_dir: str, dataset: str) -> int:
    """Floor for the ops client's container memory, sized to the corpus."""
    try:
        size = os.path.getsize(os.path.join(datasets_dir, f"{dataset}.hdf5"))
    except OSError:
        return _OPS_CLIENT_BASE_BYTES
    return int(size * _OPS_CLIENT_COPIES) + _OPS_CLIENT_BASE_BYTES


class OpsRun:
    """Manages the server/client container pair for one ops measurement."""

    def __init__(self, engine: str, engine_cfg: Dict[str, Any],
                 resolved: ResolvedResources, resource_pass: str,
                 paths: Dict[str, str], run_id: str, dataset: str, tag: str):
        self.engine = engine
        self.engine_cfg = engine_cfg
        self.resolved = resolved
        self.resource_pass = resource_pass
        self.paths = paths
        self.run_id = run_id
        self.dataset = dataset
        self.tag = tag

        safe = f"{run_id}-{engine}-{tag}".replace("_", "-").replace(".", "-")[:55]
        self.network = f"{safe}-net"
        self.volume = f"{safe}-data"
        self.server_name = f"{safe}-srv"
        self.client_name = f"{safe}-cli"
        self.port = int(engine_cfg.get("port", engines_mod.get(engine).port))

    # ------------------------------------------------------------------

    def __enter__(self) -> "OpsRun":
        try:
            docker_ctl.create_network(self.network, internal=True)
            # Backed by a directory under VB_ROOT rather than Docker's
            # data-root, so the corpus lands on the filesystem the checkout is
            # on. See docker_ctl.create_volume.
            docker_ctl.create_volume(
                self.volume,
                device=os.path.join(self.paths["engine_state"], "ops",
                                    self.volume),
            )
            self._start_server()
        except BaseException:
            # `with` calls __exit__ only if __enter__ *returned*. Setup happens
            # in three steps and the last two can fail -- a missing image, or a
            # server that never becomes healthy inside five minutes -- and
            # every failure after the first step leaves behind a network, a
            # volume, a root-owned directory under state/ops and sometimes a
            # server container, with nothing saying so. Reproduced by asking
            # for an engine whose image is not built.
            #
            # BaseException, not Exception: the widest window here is
            # wait_healthy's five-minute poll, and the likeliest thing to
            # arrive during it is the operator's Ctrl-C.
            self.teardown()
            raise
        return self

    def __exit__(self, *exc) -> None:
        self.teardown()

    def _start_server(self) -> None:
        image = self.engine_cfg.get("image", {}).get(
            "runtime", f"vector-bench/{self.engine}-runtime"
        )
        if not docker_ctl.image_exists(image):
            raise docker_ctl.DockerError(
                f"image {image} not found. Build it first:\n"
                f"  ./run-benchmark.sh build --engines {self.engine}"
            )

        flags = server_args(self.engine_cfg, self.resource_pass, self.resolved)
        data_mount = ("/var/lib/postgresql" if self.engine == "pgvector"
                      else "/var/lib/vbench")

        spec = docker_ctl.ContainerSpec(
            name=self.server_name,
            image=image,
            network=self.network,
            cpuset=self.resolved.server_cpuset,
            memory_bytes=self.resolved.server_memory_bytes,
            shm_size=self.resolved.shm_size,
            env={
                "VB_SERVER_ARGS": " ".join(flags),
                "VB_RUN_ID": self.run_id,
                # Sized by resolve_resources and passed through, so a tuned run
                # cannot silently fall back to the image default. Harmless for
                # the single-process engines, which never read it.
                "VB_MONGOT_HEAP_GB": str(
                    max(1, self.resolved.mongot_heap_bytes // (1024 ** 3))),
                "VB_MAXMEMORY_BYTES": str(self.resolved.maxmemory_bytes),
            },
            volumes=[f"{self.volume}:{data_mount}:rw"],
            command=["server"],
            detach=True,
        )
        print(f"[ops] starting {self.engine} server: cpuset={self.resolved.server_cpuset} "
              f"mem={self.resolved.server_memory_bytes / 1024**3:.1f}GB")
        print(f"[ops] server flags: {' '.join(flags)}")
        docker_ctl.start(spec)
        docker_ctl.wait_healthy(self.server_name, list(engines_mod.get(self.engine).probe), timeout_s=300)
        print(f"[ops] {self.engine} server ready")

    # ------------------------------------------------------------------

    def run_harness(self, args: List[str], output_path: str,
                    memory_timeseries: Optional[str] = None,
                    timeout_s: int = 12 * 3600) -> int:
        """Run the ops harness against the running server."""
        image = self.engine_cfg.get("image", {}).get(
            "bench", f"vector-bench/{self.engine}-bench"
        )

        volumes = [
            f"{self.paths['harness']}:/opt/harness:ro",
            f"{self.paths['datasets']}:/datasets:ro",
            f"{self.paths['ops_results']}:/results:rw",
        ]
        data_dir_arg: List[str] = []
        mount_point = engines_mod.get(self.engine).server_data_mount
        if mount_point:
            # Read-only view of the server's data directory so index files can
            # be sized exactly, rather than inferred from a catalog that does
            # not track companion tables.
            volumes.append(f"{self.volume}:/server-data:ro")
            data_dir_arg = ["--server-data-dir", mount_point]

        db_user, db_password = engines_mod.get(self.engine).credentials

        # The run directory is mounted at /results inside the client container,
        # so the harness must be given a container path. Passing the host path
        # would make Recorder create that directory inside the container and
        # write the records into a filesystem that disappears on exit.
        container_output = "/results/" + os.path.basename(output_path)

        command = [
            "/opt/harness/main.py",
            "--engine", self.engine,
            # Read from config/engines/*.yml here, because the harness container
            # mounts harness/ only and cannot read it for itself.
            "--driver", engines_mod.get(self.engine).driver,
            "--user", db_user,
            "--password", db_password,
            "--host", self.server_name,
            "--port", str(self.port),
            "--dataset", self.dataset,
            "--datasets-dir", "/datasets",
            "--run-id", self.run_id,
            "--resource-pass", self.resource_pass,
            "--output", container_output,
            "--cache-dir", "/results/.cache",
            *data_dir_arg,
            *args,
        ]

        spec = docker_ctl.ContainerSpec(
            name=self.client_name,
            image=image,
            network=self.network,
            cpuset=self.resolved.client_cpuset,
            # The ops client loads the corpus once and then computes ground
            # truth over it by brute force, which needs working space on top.
            # A fixed client_limit_gb is fine at 100 dimensions and far too
            # small at 1536, where the corpus alone is 6 GB — see the same
            # failure mode in ann_pass.client_memory_bytes().
            memory_bytes=max(self.resolved.client_memory_bytes,
                             ops_client_memory_bytes(self.paths['datasets'], self.dataset)),
            entrypoint="python3",
            workdir="/opt",
            env={
                "PYTHONUNBUFFERED": "1",
                "PYTHONPATH": "/opt",
                "VB_DB_USER": db_user,
                "VB_DB_PASSWORD": db_password,
                "VB_ENGINE_TAG": str(self.engine_cfg.get("source", {}).get("tag", "")),
            },
            volumes=volumes,
            command=command,
            detach=False,
        )

        sampler = None
        if memory_timeseries:
            sampler = docker_ctl.MemorySampler(self.server_name, memory_timeseries)
            sampler.start()
        try:
            captured: List[str] = []
            rc = docker_ctl.run_foreground(spec, timeout=timeout_s,
                                           sink=captured)
            saved = docker_ctl.save_phase_log(
                self.paths["run_dir"], self.engine, f"ops-{self.tag}",
                self.resource_pass, captured)
            if saved:
                print(f"[ops] log -> "
                      f"{os.path.relpath(saved, self.paths['run_dir'])}")
            return rc
        finally:
            # In the finally, not on the success path: a phase that timed out
            # or crashed is exactly the one whose server log is worth having.
            self._save_server_log()
            if sampler is not None:
                sampler.stop()
                print(f"[ops] captured {sampler.samples} memory samples "
                      f"-> {os.path.basename(memory_timeseries)}")

    def run_script(self, script: str, args: List[str],
                   interactive: bool = False,
                   client_memory_bytes: Optional[int] = None,
                   timeout_s: int = 4 * 3600) -> int:
        """Run one script from scripts/ against the running server.

        The diagnostic counterpart to run_harness. Same two-container shape,
        same server flags, same cpuset and memory -- because a probe that
        answers a question about a run has to run against the configuration
        that run used. The first version of this was a standalone shell script
        that started its own Valkey with hand-written flags, and it was already
        drifting: a 32 GB default where the run gives 101 GB, and
        maxmemory-policy re-specified by hand next to the copy in
        config/engines/valkey.yml. Two places to state one fact is how a probe
        ends up disproving something the run never did.

        What is deliberately *not* here is anywhere to write a measurement.
        run_harness mounts the run directory at /results; this mounts no
        writable path at all. A lab script generates its own vectors and its
        own ground truth, and those numbers must never be able to reach a
        report and sit in a table beside corpus numbers looking like a
        measurement.
        """
        image = self.engine_cfg.get("image", {}).get(
            "bench", f"vector-bench/{self.engine}-bench"
        )
        if not docker_ctl.image_exists(image):
            raise docker_ctl.DockerError(
                f"image {image} not found. Build it first:\n"
                f"  ./run-benchmark.sh build --engines {self.engine}"
            )

        db_user, db_password = engines_mod.get(self.engine).credentials
        volumes = [
            # Mounted rather than baked into the image, so editing a probe and
            # re-running it costs nothing. The whole point of the lab is a
            # turnaround measured in seconds.
            f"{self.paths['harness']}:/opt/harness:ro",
            f"{self.paths['scripts']}:/opt/scripts:ro",
            # Read-only, and only so a script *can* use the real corpus. Most
            # generate their own; the ones that need dbpedia should not have to
            # be run from inside a benchmark to reach it.
            f"{self.paths['datasets']}:/datasets:ro",
        ]

        env = {
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": "/opt",
            "VB_ENGINE": self.engine,
            "VB_DB_USER": db_user,
            "VB_DB_PASSWORD": db_password,
            "VB_HOST": self.server_name,
            "VB_PORT": str(self.port),
        }

        if interactive:
            entrypoint, command = "bash", []
        else:
            # --host and --port are passed for every script, so the contract a
            # lab script signs is those two flags. Credentials go through the
            # environment instead: an engine that needs them differs per
            # engine, and a script that does not take --user must still be
            # runnable.
            entrypoint = "python3"
            command = [f"/opt/scripts/{script}",
                       "--host", self.server_name,
                       "--port", str(self.port), *args]

        spec = docker_ctl.ContainerSpec(
            name=self.client_name,
            image=image,
            network=self.network,
            cpuset=self.resolved.client_cpuset,
            memory_bytes=client_memory_bytes or self.resolved.client_memory_bytes,
            entrypoint=entrypoint,
            workdir="/opt",
            env=env,
            volumes=volumes,
            command=command,
            detach=False,
        )

        try:
            if interactive:
                return docker_ctl.run_interactive(spec)
            return docker_ctl.run_foreground(spec, timeout=timeout_s)
        finally:
            # Same reasoning as run_harness: the server's half of the story is
            # worth most precisely when the client's half ended badly.
            self._save_server_log()

    def _save_server_log(self) -> None:
        """Archive what the server said, not only what the client saw.

        The ann path gets this for free: the engine and the benchmark share one
        container, so one stream carries both sides. The ops path splits them,
        and only the client's half was ever kept. A Valkey churn then stalled
        with the client blocked on a socket and the server at its idle CPU
        baseline, and the question of which one was wrong could not be answered
        from the run directory at all -- the half that would have said was
        discarded when the container was removed.
        """
        try:
            text = docker_ctl.logs(self.server_name, tail=SERVER_LOG_TAIL)
        except Exception:
            return
        if not text:
            return
        docker_ctl.save_phase_log(
            self.paths["run_dir"], self.engine, f"server-{self.tag}",
            self.resource_pass, text.splitlines())

    # ------------------------------------------------------------------

    def teardown(self) -> None:
        # Stop the server before removing the volume, or the removal races the
        # engine's own shutdown flush and leaves a dangling volume behind.
        docker_ctl.stop(self.server_name, timeout_s=120)
        docker_ctl.remove(self.server_name)
        docker_ctl.remove(self.client_name)
        docker_ctl.remove_network(self.network)
        docker_ctl.remove_volume(self.volume)
        # Removing a bind-backed volume leaves the host directory behind, so
        # without this every configuration would leak a full copy of the corpus
        # and the index onto disk.
        # Root-owned: the engine wrote it from inside the container, so this
        # has to go through a container too. shutil.rmtree cannot touch it and
        # fails silently, leaving a full corpus and index on disk.
        bind_dir = os.path.join(self.paths["engine_state"], "ops", self.volume)
        image = self.engine_cfg.get("image", {}).get(
            "runtime", f"vector-bench/{self.engine}-runtime")
        if not docker_ctl.remove_tree_as_root(bind_dir, image):
            print(f"[ops] WARNING: {bind_dir} survived teardown and is still "
                  f"using disk", file=sys.stderr)


def _quantization(profile: Dict[str, Any], resources: Dict[str, Any],
                  resource_pass: str) -> str:
    """Which quantization the ops build should use.

    The same rule render_config applies to the recall path: pinned off in the
    normalized pass so no engine gets an axis the others lack, and the vendor's
    recommendation in the tuned pass. Reading it in only one of the two paths
    meant a tuned run measured a quantized index for recall and an unquantized
    one for build cost and every ops workload, then reported them side by side
    as one configuration.
    """
    if resource_pass == "tuned":
        values = (resources.get("extras", {}) or {}).get("mongodb_quantization")
        if values:
            return str(list(values)[0])
    return str((profile.get("ann", {}) or {}).get("mongodb_quantization", "none"))


def harness_args(profile: Dict[str, Any], m: int, engine: str,
                 resolved: ResolvedResources,
                 resource_pass: str, resources: Dict[str, Any],
                 build_mode: str = "post",
                 storage_engine: str = "InnoDB",
                 iterative_scan: Optional[str] = None) -> List[str]:
    """Translate a profile into ops-harness command-line arguments."""
    ops = profile.get("ops", {}) or {}
    ann = profile.get("ann", {}) or {}

    args = [
        "--m", str(m),
        "--k", str(profile.get("k", 10)),
        "--ef-search", str(ops.get("ef_search", 100)),
        "--storage-engine", storage_engine,
        "--build-mode", build_mode,
        "--churn-budget", str(ops.get("churn_budget_s", 1800)),
        # Read from the same extras render_config uses for the recall path, so
        # both measurement paths in one run build the same index. Only Percona
        # Search has the knob; the flag is omitted for everything else.
        *(["--quantization", _quantization(profile, resources, resource_pass)]
          if engine == "mongodb" else []),
        "--load-threads", str(ops.get("load_threads", 1)),
        "--max-queries", str(ops.get("max_queries", 1000)),
        "--workloads", ",".join(ops.get("workloads", ["build"])),
        "--client-counts", ",".join(str(c) for c in ops.get("client_counts", [1])),
        "--concurrency-duration", str(ops.get("concurrency_duration_s", 20)),
        "--concurrency-repeats", str(ops.get("concurrency_repeats", 1)),
        "--selectivities", ",".join(str(s) for s in ops.get("selectivities", [0.1])),
        "--churn-fractions", ",".join(str(c) for c in ops.get("churn_fractions", [0.1])),
    ]
    if ops.get("subset_rows"):
        args += ["--subset-rows", str(ops["subset_rows"])]
    if engine == "pgvector":
        ef_construction = ann.get("pgvector_ef_construction", 200)
        args += ["--ef-construction", str(ef_construction)]
        if iterative_scan:
            args += ["--iterative-scan", iterative_scan]
    return args
