"""Launching and monitoring the long commands: run, fetch, build, report, render.

Every one of these is the same shape -- a command that takes minutes to days and
whose output you want to watch -- so they share one supervisor rather than each
growing its own. The UI used to print `./run-benchmark.sh fetch ...` and send
you to a terminal, which is not an interface, it is a reminder.

NOTHING RUNS ALONGSIDE A BENCHMARK. A 5 GB download or a compile during an
ingest measurement perturbs exactly what is being measured -- a competing build
distorted MariaDB's numbers by 2x once, which is why the README says to give the
machine to the benchmark. Setup jobs may overlap each other, because nothing is
being measured then.

Commands are built as argv lists and every token is validated against an
allowlist; nothing reaches a shell.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
PHASES = ("ann", "ops", "both")
PASSES = ("normalized", "tuned", "both")
TARGETS = ("all", "runtime", "bench")

#: kind -> whether it is a measurement. A measurement excludes everything;
#: everything else only excludes a measurement.
KINDS = {
    "run": {"measurement": True, "label": "benchmark run"},
    "fetch": {"measurement": False, "label": "dataset download"},
    "build": {"measurement": False, "label": "image build"},
    "report": {"measurement": False, "label": "report generation"},
    "render": {"measurement": False, "label": "config render"},
}


def _utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


class JobStore:
    def __init__(self, root: str, known_engines: Tuple[str, ...] = ()):
        self.root = root
        self.known_engines = tuple(known_engines)
        self.state_dir = os.path.join(root, "state", "webui")
        self.log_dir = os.path.join(self.state_dir, "logs")
        self.index_path = os.path.join(self.state_dir, "jobs.json")
        self._lock = threading.Lock()
        self._processes: Dict[str, subprocess.Popen] = {}
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._load()

    # -- persistence -----------------------------------------------------

    def _load(self) -> None:
        try:
            with open(self.index_path) as fh:
                stored = json.load(fh)
        except (OSError, json.JSONDecodeError):
            stored = []
        for job in stored:
            # Jobs recorded before there was more than one kind of job.
            job = {"kind": "run", **job}
            if job.get("status") == "running" and not _alive(job.get("pid")):
                job = {**job, "status": "orphaned"}
            self._jobs[job["id"]] = job

    def _persist(self) -> None:
        try:
            os.makedirs(self.state_dir, exist_ok=True)
            temporary = f"{self.index_path}.tmp"
            with open(temporary, "w") as fh:
                json.dump(list(self._jobs.values()), fh, indent=1)
            os.replace(temporary, self.index_path)
        except OSError:
            pass

    # -- queries ---------------------------------------------------------

    def list_jobs(self) -> List[Dict[str, Any]]:
        with self._lock:
            jobs = list(self._jobs.values())
        return sorted(jobs, key=lambda j: j.get("started_at") or "", reverse=True)

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._jobs.get(job_id)

    def active_jobs(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [job for job in self._jobs.values()
                    if job.get("status") in ("running", "stopping")
                    and _alive(job.get("pid"))]

    def active(self) -> Optional[Dict[str, Any]]:
        """The measurement in progress, else any running job, else None."""
        running = self.active_jobs()
        for job in running:
            if KINDS.get(job.get("kind", "run"), {}).get("measurement"):
                return job
        return running[0] if running else None

    # -- validation ------------------------------------------------------

    def validate(self, spec: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """(errors, argv-after-the-subcommand). `kind` defaults to a run."""
        kind = str(spec.get("kind") or "run")
        if kind not in KINDS:
            return [f"unknown job kind: {kind} "
                    f"(expected one of {', '.join(sorted(KINDS))})"], []
        return getattr(self, f"_validate_{kind}")(spec)

    # -- per-kind validation ---------------------------------------------

    def _validate_run(self, spec: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        errors: List[str] = []
        argv: List[str] = []

        profile = str(spec.get("profile") or "").strip()
        if not TOKEN_RE.match(profile):
            errors.append("profile is required and must be a plain name")
        elif not os.path.isfile(os.path.join(self.root, "config", "profiles",
                                             f"{profile}.yml")):
            errors.append(f"no such profile: {profile}")
        else:
            argv += ["--profile", profile]

        engines, engine_errors = self._engine_list(spec.get("engines"))
        errors += engine_errors
        if engines:
            argv += ["--engines", ",".join(engines)]

        datasets, dataset_errors = self._dataset_list(spec.get("datasets"))
        errors += dataset_errors
        if datasets:
            argv += ["--datasets", ",".join(datasets)]

        resource_pass = spec.get("resource_pass")
        if resource_pass:
            if resource_pass not in PASSES:
                errors.append(f"resource_pass must be one of {', '.join(PASSES)}")
            else:
                argv += ["--resource-pass", resource_pass]

        phases = spec.get("phases")
        if phases:
            if phases not in PHASES:
                errors.append(f"phases must be one of {', '.join(PHASES)}")
            else:
                argv += ["--phases", phases]

        run_id = str(spec.get("run_id") or "").strip()
        if run_id:
            if not TOKEN_RE.match(run_id):
                errors.append("run_id contains characters Docker will not accept")
            else:
                argv += ["--run-id", run_id]

        for flag, option in (("resume", "--resume"), ("force", "--force"),
                             ("fail_fast", "--fail-fast"),
                             ("no_report", "--no-report")):
            if spec.get(flag):
                argv.append(option)

        if spec.get("resume") and not run_id:
            errors.append("--resume needs the run_id of the run to continue")

        return errors, argv

    def _validate_fetch(self, spec: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        datasets, errors = self._dataset_list(spec.get("datasets"))
        if not datasets and not errors:
            errors.append("choose at least one dataset to download")
        argv = ["--datasets", ",".join(datasets)] if datasets else []
        return errors, argv

    def _validate_build(self, spec: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        errors: List[str] = []
        argv: List[str] = []

        engines, engine_errors = self._engine_list(spec.get("engines"))
        errors += engine_errors
        if not engines and not engine_errors:
            errors.append("choose at least one engine to build")
        if engines:
            argv += ["--engines", ",".join(engines)]

        target = spec.get("target")
        if target:
            if target not in TARGETS:
                errors.append(f"target must be one of {', '.join(TARGETS)}")
            else:
                argv += ["--target", target]

        march = str(spec.get("march") or "").strip()
        if march:
            # -march reaches a compiler, so it is the token most worth pinning
            # down. Never mix values between engines: that turns the benchmark
            # into a comparison of compiler flags.
            if not re.match(r"^[a-z0-9][a-z0-9=_.-]*$", march):
                errors.append("march must be a plain gcc -march value, "
                              "e.g. native or x86-64-v3")
            else:
                argv += ["--march", march]

        if spec.get("no_cache"):
            argv.append("--no-cache")
        return errors, argv

    def _validate_report(self, spec: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        errors: List[str] = []
        run_id = str(spec.get("run_id") or "").strip()
        if not TOKEN_RE.match(run_id):
            errors.append("run_id is required")
            return errors, []
        run_dir = os.path.join(self.root, "results", run_id)
        if not os.path.isdir(run_dir):
            errors.append(f"no such run: {run_id}")
            return errors, []

        argv = ["--run-dir", run_dir]
        datasets, dataset_errors = self._dataset_list(spec.get("datasets"))
        errors += dataset_errors
        if datasets:
            argv += ["--datasets", ",".join(datasets)]
        return errors, argv

    def _validate_render(self, spec: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        errors: List[str] = []
        argv: List[str] = []
        profile = str(spec.get("profile") or "").strip()
        if not TOKEN_RE.match(profile):
            errors.append("profile is required")
        else:
            argv += ["--profile", profile]

        resource_pass = spec.get("resource_pass")
        if resource_pass:
            if resource_pass not in ("normalized", "tuned"):
                errors.append("resource_pass must be normalized or tuned")
            else:
                argv += ["--resource-pass", resource_pass]
        return errors, argv

    # -- shared token checks ---------------------------------------------

    def _engine_list(self, value: Any) -> Tuple[List[str], List[str]]:
        engines = [str(e) for e in (value or [])]
        unknown = [e for e in engines
                   if e not in self.known_engines or not TOKEN_RE.match(e)]
        if unknown:
            return [], [f"unknown engines: {', '.join(unknown)}"]
        return engines, []

    def _dataset_list(self, value: Any) -> Tuple[List[str], List[str]]:
        datasets = [str(d) for d in (value or [])]
        bad = [d for d in datasets if not TOKEN_RE.match(d)]
        if bad:
            return [], [f"invalid dataset names: {', '.join(bad)}"]
        return datasets, []

    # -- launching -------------------------------------------------------

    def conflict_for(self, kind: str) -> Optional[str]:
        """Why this job may not start now, or None.

        A measurement excludes everything, because a download or a compile
        beside it perturbs what is being measured. Setup jobs only exclude a
        measurement, and an identical command already running.
        """
        for job in self.active_jobs():
            other = job.get("kind", "run")
            if KINDS.get(other, {}).get("measurement"):
                return (f"a {KINDS[other]['label']} is in progress "
                        f"({job['id']}: {job.get('command_display')}). Nothing "
                        f"else may run beside it, or it measures the "
                        f"interference as well as the engine.")
            if KINDS.get(kind, {}).get("measurement"):
                return (f"{KINDS.get(other, {}).get('label', other)} {job['id']} "
                        f"is still running ({job.get('command_display')}). A "
                        f"benchmark started beside it would measure it too.")
        return None

    def launch(self, spec: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        kind = str(spec.get("kind") or "run")
        if kind not in KINDS:
            return None, [f"unknown job kind: {kind}"]

        conflict = self.conflict_for(kind)
        if conflict:
            return None, [conflict]

        errors, argv = self.validate(spec)
        if errors:
            return None, errors

        command = [os.path.join(self.root, "run-benchmark.sh"), kind, *argv]
        display = " ".join(["./run-benchmark.sh", kind, *argv])
        for job in self.active_jobs():
            if job.get("command_display") == display:
                return None, [f"exactly this is already running: {job['id']}"]

        job_id = f"job-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}"
        os.makedirs(self.log_dir, exist_ok=True)
        log_path = os.path.join(self.log_dir, f"{job_id}.log")

        try:
            log_file = open(log_path, "wb", buffering=0)
        except OSError as exc:
            return None, [f"could not open log: {exc}"]

        try:
            process = subprocess.Popen(
                command, cwd=self.root, stdout=log_file,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                start_new_session=True,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
        except OSError as exc:
            log_file.close()
            return None, [f"could not start run: {exc}"]

        job = {
            "id": job_id,
            "kind": kind,
            "status": "running",
            "pid": process.pid,
            "command": command,
            "command_display": display,
            "spec": spec,
            "log": log_path,
            "started_at": _utcnow(),
            "finished_at": None,
            "exit_code": None,
        }

        with self._lock:
            self._jobs[job_id] = job
            self._processes[job_id] = process
            self._persist()

        threading.Thread(target=self._reap, args=(job_id, process, log_file),
                         daemon=True).start()
        return job, []

    def _reap(self, job_id: str, process: subprocess.Popen, log_file) -> None:
        exit_code = process.wait()
        try:
            log_file.close()
        except OSError:
            pass
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                if exit_code == 0:
                    status = "completed"
                elif job.get("status") == "stopping":
                    # Asked to stop, so a non-zero exit is the answer, not a
                    # failure. Reporting it as one makes the log look alarming
                    # for something the operator did on purpose.
                    status = "stopped"
                else:
                    status = "failed"
                self._jobs[job_id] = {
                    **job,
                    "status": status,
                    "exit_code": exit_code,
                    "finished_at": _utcnow(),
                }
            self._processes.pop(job_id, None)
            self._persist()

    def stop(self, job_id: str) -> Tuple[bool, List[str]]:
        job = self.get(job_id)
        if not job:
            return False, [f"no such job: {job_id}"]
        if job.get("status") != "running":
            return False, [f"job is not running: {job.get('status')}"]

        pid = job.get("pid")
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (OSError, ProcessLookupError) as exc:
            return False, [f"could not signal {pid}: {exc}"]

        with self._lock:
            current = self._jobs.get(job_id)
            if current:
                self._jobs[job_id] = {**current, "status": "stopping"}
                self._persist()
        return True, []

    # -- log -------------------------------------------------------------

    def read_log(self, job_id: str, offset: int = 0,
                 limit: int = 256 * 1024) -> Optional[Dict[str, Any]]:
        job = self.get(job_id)
        if not job:
            return None
        path = job.get("log")
        try:
            size = os.path.getsize(path)
        except OSError:
            return {"offset": 0, "size": 0, "data": "", "status": job.get("status")}

        start = max(0, min(int(offset), size))
        try:
            with open(path, "rb") as fh:
                fh.seek(start)
                chunk = fh.read(limit)
        except OSError:
            chunk = b""

        return {
            "offset": start + len(chunk),
            "size": size,
            "data": chunk.decode("utf-8", "replace"),
            "status": job.get("status"),
            "exit_code": job.get("exit_code"),
        }
