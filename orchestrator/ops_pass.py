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
from typing import Any, Dict, List, Optional, Sequence, Union

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
            #
            # THE LOGS COME OUT BEFORE TEARDOWN, because this is the one path
            # where the server's own log is most wanted and was least likely
            # to exist. `run_harness` is never entered when setup fails, so
            # neither capture ran, and `teardown` then removed the volume: a
            # server that refused to start under a tuned
            # `innodb_buffer_pool_size` left "did not become ready within
            # 300s" plus the three-line banner, and the assertion that
            # actually caused it was deleted. Cannot raise -- see
            # `_save_startup_log`.
            self._save_startup_log()
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
                self.paths["run_dir"], self.engine, self.dataset,
                f"ops-{self.tag}", self.resource_pass, captured)
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

        Only the client's half was ever kept. A Valkey churn then stalled with
        the client blocked on a socket and the server at its idle CPU
        baseline, and the question of which one was wrong could not be answered
        from the run directory at all -- the half that would have said was
        discarded when the container was removed.

        THE ANN PATH DOES NOT GET THIS FOR FREE, and that gap is still open.
        This docstring used to say it did, on the grounds that the engine and
        the benchmark share one container there. They do share one, but
        `overlay/ann-benchmarks/ann_benchmarks/vb_mysql.py` starts the engine
        with `Popen([ENTRYPOINT, "server"])` -- `/usr/local/bin/vb-entrypoint`,
        the same script, because every bench image is `FROM runtime AS bench`
        -- so mysqld is given the same `--log-error=<file>` and that one
        stream carries the client's half plus the banner, exactly as here.
        What the ann path has instead is that `ann_pass.run_engine`
        bind-mounts `state_dir` at the engine's `data_mount` read-write, so
        the file does land on the host: root-owned, keyed by (pass, engine) and
        NOT by dataset, and wiped by the next dataset's
        `shutil.rmtree(state_dir)`. Nothing copies it into the run directory
        yet.

        TWO CAPTURES, because for five of the seven engines `docker logs` is
        not all of the server's voice, and for four of those it is almost none
        of it. `mariadb`, `mariadb123`, `alisql` and `percona` are started
        with `--log-error=<file>`, so their stdout is the entrypoint banner --
        three lines -- and everything the server itself has to say goes into a
        file in the data volume that teardown deletes. That is precisely the
        half this method was added to keep, and for those engines it was
        capturing the wrong one: 13 of the 24 ops units in run
        full-20261015-215006 died inside the first INSERT with
        `mariadb.InterfaceError: Lost connection to server during query`, and
        all six server logs it archived are the banner. `mongodb` is the fifth
        and a partial case -- `mongod` does log to stdout, but `mongot`, which
        is where vector search actually happens, is redirected into the data
        volume -- so for it both captures carry something.
        `runtime.error_log` names the file; `_save_server_error_log` copies it
        out while the container is still up.

        Independent of each other, and neither can end the run: a server whose
        stdout is empty is exactly the one whose file log matters, so an
        exception from the first must not skip the second.
        """
        try:
            text = docker_ctl.logs(self.server_name, tail=SERVER_LOG_TAIL)
        except Exception:
            text = ""
        if text:
            docker_ctl.save_phase_log(
                self.paths["run_dir"], self.engine, self.dataset,
                f"server-{self.tag}", self.resource_pass, text.splitlines())
        self._save_server_error_log()

    def _save_startup_log(self) -> None:
        """`_save_server_log`, but only once there is a container to ask.

        Setup fails in three places and only the last one has started a
        server: a missing image raises before `docker run`, and
        `create_network` / `create_volume` raise before that. In those cases
        `docker logs` answers with the daemon's "No such container ..." on
        stderr, which `logs` folds into its return value -- and archiving that
        as `<engine>-<dataset>-server-<tag>-<pass>.log` would put a Docker
        error in the run directory under a name that reads as the server's own
        account of itself.

        Cannot raise: `container_exists` is a `check=False` inspect, and
        `_save_server_log` already swallows everything underneath it.
        """
        try:
            if not docker_ctl.container_exists(self.server_name):
                return
        except Exception:                        # pragma: no cover - defensive
            return
        self._save_server_log()

    def _save_server_error_log(self) -> None:
        """Copy the engine's own `--log-error` file out before teardown.

        Only for an engine whose config declares `runtime.error_log`; the
        engines that log to stderr are already covered by `docker logs` above
        and this is a no-op for them, so their behaviour is unchanged.

        Must run BEFORE teardown and does, because every caller reaches it from
        a `finally` inside a `with OpsRun(...)` block, and `teardown` -- which
        stops the container and then removes the volume the file lives in -- is
        `__exit__`. There is no second chance after that: the file is not on
        the host, and the volume is bind-backed under state/ops, which
        `remove_tree_as_root` then deletes as well.

        NOTHING HERE MAY RAISE. This is on the path a crashed phase takes, and
        a phase that has already lost its measurement must not also lose the
        run because its log could not be fetched. Every failure -- no such
        engine, no such file, an unreadable path, a container that has gone, no
        daemon -- degrades to a printed note and no file. `read_file_tail`
        returns None for all of them; the try is for the registry lookup and
        the write.
        """
        try:
            error_log = engines_mod.get(self.engine).error_log
            if not error_log:
                return
            text = docker_ctl.read_file_tail(self.server_name, error_log,
                                             lines=SERVER_LOG_TAIL)
            if not text:
                print(f"[ops] note: {self.engine} declares "
                      f"runtime.error_log={error_log} but nothing could be "
                      f"read from it; the server's own log is not in this run "
                      f"directory", file=sys.stderr)
                return
            saved = docker_ctl.save_phase_log(
                self.paths["run_dir"], self.engine, self.dataset,
                f"server-error-{self.tag}", self.resource_pass,
                text.splitlines())
            if saved:
                print(f"[ops] server error log -> "
                      f"{os.path.relpath(saved, self.paths['run_dir'])}")
        except Exception as exc:                 # pragma: no cover - defensive
            print(f"[ops] note: could not archive {self.engine}'s error log: "
                  f"{exc}", file=sys.stderr)

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


#: What `ops.concurrency_queries` means when a profile does not set it: every
#: query the dataset ships (10,000 for glove-100 and sift-128). The harness
#: reads 0 as a sentinel, not as a cap — see `_query_pool` in
#: harness/workloads/concurrency.py. The default is the new behaviour and the
#: way back is to set the key, which is the convention every switch here
#: follows.
DEFAULT_CONCURRENCY_QUERIES = 0


#: The ef policy, as `harness/workloads/calibrate.py` defines it. Declared
#: again here rather than imported, and that is deliberate: this module runs on
#: the host, whose only Python dependency is pyyaml, and
#: `harness.workloads.calibrate` reaches numpy through `harness.datasets`. The
#: same reason `cli.py` imports `harness.datasets` inside a function body
#: instead of at module scope. `tests/test_calibrate.py` pins the two
#: declarations together, which is how `ops.selectivities` and
#: `ops.churn_fractions` are already handled.
DEFAULT_EF_MODE = "calibrated"
DEFAULT_TARGET_RECALL = 0.95
DEFAULT_EF_CALIBRATION = (10, 16, 24, 32, 48, 64, 100, 160, 256, 400, 640,
                          1000)


def _concurrency_queries(ops: Dict[str, Any]) -> int:
    """How many queries the concurrency clients may draw from.

    Its own function rather than an inline `.get` because the value does not
    come from `ops.max_queries` and the reason is not obvious: that key caps
    the filtered and churn workloads, which answer each query once, and
    main.yml holds it at 500 to bound their cost. The concurrency workload
    cycles its shard for a fixed duration, so sharing the key gave 32 clients
    fifteen queries each and measured every engine's cache against a hot set
    that fitted in it.
    """
    return int(ops.get("concurrency_queries", DEFAULT_CONCURRENCY_QUERIES))


#: Seconds per (column, selectivity, iterative-scan mode) point in the
#: filtered workload. Declared here as well as in
#: `harness.workloads.filtered` for the reason DEFAULT_EF_MODE is: this module
#: runs on the host, whose only Python dependency is pyyaml, and the workload
#: reaches numpy through `harness.datasets`. `tests/test_t3c_filtered.py` pins
#: the two declarations together, as `tests/test_calibrate.py` already does
#: for the ef policy.
DEFAULT_FILTERED_BUDGET_S = 300

#: Seconds one churn re-insertion may spend. Matches
#: `harness/workloads/churn.py:DEFAULT_CHURN_BUDGET_S`, which is not
#: importable here -- the orchestrator runs on the host and the harness runs
#: in a container -- and was a bare literal in the argument list until the
#: `null` fix needed to pass it to `_budget_arg` as a default.
DEFAULT_CHURN_BUDGET_S = 1800

#: The filter columns the filtered workload measures when a profile says
#: nothing. Both, matching `harness.datasets.FILTER_COLUMNS`, and pinned
#: against it by the same test.
DEFAULT_FILTER_COLUMNS = ("tag", "ctag")

#: Readers querying the index while the churn writer re-inserts, per PR 4
#: §4.3 item 2. Matches `harness.workloads.churn_readers`'
#: `DEFAULT_CHURN_READERS`, which is not importable here -- the orchestrator
#: runs on the host and the harness runs in a container -- and is pinned
#: against it by `tests/test_t4b_readers.py`, the way `tests/test_calibrate.py`
#: already pins the ef policy and `tests/test_t3c_filtered.py` the filtered
#: budget.
#:
#: Emitted unconditionally, including the default, for the reason `--ef-mode`
#: and `--filtered-budget` are: the harness command line in a run's phase log
#: should say what read load the churn write rate was measured against rather
#: than leaving a reader to infer it from whatever the workload's default was
#: at the time. `ops.churn_readers: 0` is the write-only rate and is a real
#: choice, not an absence.
#:
#: NOT run through `_budget_arg`. That function turns a `null` into "0", which
#: is right for a budget -- null and 0 both mean "no cap" -- and wrong here: a
#: reader count of 0 means "no readers", so `null` would silently become
#: "turn the pool off" rather than "use the default". A profile that wants no
#: readers writes 0.
DEFAULT_CHURN_READERS = 4


def _budget_arg(ops: Dict[str, Any], key: str, default: Any) -> str:
    """A budget key as a command-line value, with YAML `null` meaning no cap.

    `str(ops.get(key, default))` -- the idiom every other numeric key here
    uses -- turns a key PRESENT with value null into the string "None", which
    argparse rejects with `invalid float value: 'None'` and `SystemExit(2)`.
    That kills the whole ops unit, not just the workload: `harness/main.py`
    wraps concurrency, filtered and churn in one `except`, so a reader who
    wrote `filtered_budget_s: null` to turn the cap off lost two other
    workloads with it.

    `null` and `0` both mean no cap, which is what `_filtered_search` in the
    estimate has always assumed (it treats `budget > 0` as the only bounded
    case) and what the workload now implements.

    Applied to `ops.churn_budget_s` as well, as of T4a. T3c left that one
    alone on purpose: the enforcement lives in `harness/workloads/churn.py`
    and fixing the emission without it would have moved the failure from
    argparse into a workload whose first deadline check fires immediately,
    which is the shape that made every filtered point record ZERO queries.
    `churn.run` now converts a non-positive budget into an infinite deadline,
    so both halves are in place and both spellings mean no cap.

    One place still raises on `null` and it is not reachable from here:
    `orchestrator/estimate.py:_churn_reinsert` reads
    `float(plan.ops.get("churn_budget_s", 1800))` with no `or 0.0`, so the
    estimate -- which runs on the host, before the run starts -- dies with
    `TypeError: float() argument must be a string or a real number, not
    'NoneType'`. `_filtered_search` two functions above it has the `or 0.0`.
    Fixing that is a one-line change in a file T4a does not own.
    """
    value = ops.get(key, default)
    return "0" if value is None else str(value)


def _churn_readers(ops: Dict[str, Any]) -> int:
    """How many readers run while the churn writer re-inserts.

    Its own function rather than an inline `.get` for the same reason
    `_concurrency_queries` is one: a `null` here must not mean what a `null`
    in a budget key means. `ops.churn_budget_s: null` means "no cap" and
    `_budget_arg` turns it into "0"; `ops.churn_readers: null` is a profile
    that cleared the key and wants whatever the default is, and emitting "0"
    for it would quietly turn the reader pool off -- which changes what
    `extra.reinsert_rows_per_s` measures while the record still looks the
    same. A profile that wants the write-only rate writes 0 explicitly.
    """
    value = ops.get("churn_readers", DEFAULT_CHURN_READERS)
    return int(DEFAULT_CHURN_READERS if value is None else value)


def _filter_columns(ops: Dict[str, Any]) -> List[str]:
    """The filter columns to measure, as the harness spells them.

    Its own function because a profile may legitimately narrow this -- a
    reproduction of a pre-PR-3 run wants `[tag]` alone -- and because an empty
    or absent key must mean "both" rather than "none". `[]` in YAML is falsy,
    and a profile that wrote it would otherwise silently measure no filtered
    point at all while the phase still reported success.
    """
    declared = ops.get("filter_columns")
    if not declared:
        return list(DEFAULT_FILTER_COLUMNS)
    return [str(column) for column in declared]


def _iterative_scan_modes(iterative_scan) -> Optional[str]:
    """The `--iterative-scan` value, from one mode or several.

    A plain string passes through verbatim, which is what keeps a single-mode
    call site emitting exactly what it asked for; a sequence is joined with
    commas and the workload splits it again. `str` is checked first on purpose
    -- a string IS a sequence, and joining one gives "o,f,f".
    """
    if iterative_scan is None:
        return None
    if isinstance(iterative_scan, str):
        return iterative_scan or None
    modes = [str(mode) for mode in iterative_scan if str(mode)]
    return ",".join(modes) or None


def harness_args(profile: Dict[str, Any], m: int, engine: str,
                 resolved: ResolvedResources,
                 resource_pass: str, resources: Dict[str, Any],
                 build_mode: str = "post",
                 storage_engine: str = "InnoDB",
                 iterative_scan: Optional[Union[str, Sequence[str]]] = None
                 ) -> List[str]:
    """Translate a profile into ops-harness command-line arguments.

    `iterative_scan` takes one mode or several. One is emitted verbatim, which
    is what the only call site in `cli.py` passes today -- it resolves the
    FIRST declared mode and pins it. Several are comma-joined for the workload
    to loop, which is the shape PR 3 §3.4 item 4 costs out; nothing in the
    orchestrator produces that shape yet, and turning it on is one line in
    `cli.py:_ops_iterative_scan` (return the whole declared list rather than
    `list(declared)[0]`). It is wired from this end so that the day it is
    turned on, the carriage is already tested.
    """
    ops = profile.get("ops", {}) or {}
    ann = profile.get("ann", {}) or {}

    args = [
        "--m", str(m),
        "--k", str(profile.get("k", 10)),
        "--ef-search", str(ops.get("ef_search", 100)),
        # The ef policy. Emitted unconditionally, including the default, so the
        # harness command line in a run's phase log says which policy produced
        # the numbers rather than leaving a reader to infer it from the
        # harness's own default at the time.
        "--ef-mode", str(ops.get("ef_mode", DEFAULT_EF_MODE)),
        "--target-recall", str(ops.get("target_recall", DEFAULT_TARGET_RECALL)),
        "--ef-grid", ",".join(
            str(int(ef)) for ef in
            (ops.get("ef_calibration") or DEFAULT_EF_CALIBRATION)),
        "--storage-engine", storage_engine,
        "--build-mode", build_mode,
        "--churn-budget", _budget_arg(ops, "churn_budget_s",
                                      DEFAULT_CHURN_BUDGET_S),
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
        "--concurrency-queries", str(_concurrency_queries(ops)),
        "--selectivities", ",".join(str(s) for s in ops.get("selectivities", [0.1])),
        "--filter-columns", ",".join(_filter_columns(ops)),
        # Emitted unconditionally, including the default, for the reason
        # --ef-mode is: the harness command line in a run's phase log should
        # say what bounded the filtered loop rather than leaving a reader to
        # infer it from whatever the workload's default was at the time.
        "--filtered-budget", _budget_arg(ops, "filtered_budget_s",
                                         DEFAULT_FILTERED_BUDGET_S),
        "--churn-fractions", ",".join(str(c) for c in ops.get("churn_fractions", [0.1])),
        "--churn-readers", str(_churn_readers(ops)),
    ]
    if ops.get("subset_rows"):
        args += ["--subset-rows", str(ops["subset_rows"])]
    if engine == "pgvector":
        ef_construction = ann.get("pgvector_ef_construction", 200)
        args += ["--ef-construction", str(ef_construction)]
        modes = _iterative_scan_modes(iterative_scan)
        if modes:
            args += ["--iterative-scan", modes]
    return args
