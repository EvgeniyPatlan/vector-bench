"""Launching and monitoring benchmark runs.

One run at a time. Two concurrent runs would compete for the same cores and
invalidate both sets of measurements, which is a correctness constraint rather
than a resource one, so a second launch is refused rather than queued.

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

    def active(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            for job in self._jobs.values():
                if job.get("status") == "running" and _alive(job.get("pid")):
                    return job
        return None

    # -- validation ------------------------------------------------------

    def validate(self, spec: Dict[str, Any]) -> Tuple[List[str], List[str]]:
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

        engines = spec.get("engines") or []
        if engines:
            unknown = [e for e in engines
                       if e not in self.known_engines or not TOKEN_RE.match(str(e))]
            if unknown:
                errors.append(f"unknown engines: {', '.join(map(str, unknown))}")
            else:
                argv += ["--engines", ",".join(engines)]

        datasets = spec.get("datasets") or []
        if datasets:
            bad = [d for d in datasets if not TOKEN_RE.match(str(d))]
            if bad:
                errors.append(f"invalid dataset names: {', '.join(map(str, bad))}")
            else:
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

    # -- launching -------------------------------------------------------

    def launch(self, spec: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        running = self.active()
        if running:
            return None, [f"a run is already in progress: {running['id']} "
                          f"({running.get('command_display')})"]

        errors, argv = self.validate(spec)
        if errors:
            return None, errors

        job_id = f"job-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}"
        os.makedirs(self.log_dir, exist_ok=True)
        log_path = os.path.join(self.log_dir, f"{job_id}.log")
        command = [os.path.join(self.root, "run-benchmark.sh"), "run", *argv]

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
            "status": "running",
            "pid": process.pid,
            "command": command,
            "command_display": " ".join(["./run-benchmark.sh", "run", *argv]),
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
                self._jobs[job_id] = {
                    **job,
                    "status": "completed" if exit_code == 0 else "failed",
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
