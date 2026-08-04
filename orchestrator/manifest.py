"""Run manifest: the provenance that makes a number interpretable.

A result without its manifest is not a result. The report generator refuses to
emit a report for a run directory that has no manifest, because every one of the
conclusions this framework can produce depends on facts recorded here — which
CPU, which SIMD flags, which source commit, which resource limits, which image.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import subprocess
from typing import Any, Dict, List, Optional

MANIFEST_NAME = "run-manifest.json"


def utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}-{_dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


class Manifest:
    def __init__(self, run_dir: str, run_id: str):
        self.run_dir = run_dir
        self.path = os.path.join(run_dir, MANIFEST_NAME)
        os.makedirs(run_dir, exist_ok=True)
        self.data: Dict[str, Any] = {
            "run_id": run_id,
            "started_at": utcnow(),
            "finished_at": None,
            "status": "running",
            "framework": {
                "name": "vector-bench",
                "commit": _git_commit(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__)))),
            },
            "host": {},
            "engines": {},
            "phases": [],
            "config": {},
            "warnings": [],
        }
        self.save()

    # -- population -----------------------------------------------------

    def set_host(self, sysinfo: Any) -> None:
        self.data["host"] = sysinfo.to_dict()
        self.data["host"]["platform"] = platform.platform()
        cpu = sysinfo.cpu
        if not cpu.has_avx512:
            # Not a defect — but both MHNSW and VIDX document AVX-512 distance
            # kernels, so a run without it is measuring their fallback paths and
            # any conclusion is scoped to hardware like this.
            self.add_warning(
                "This CPU has no AVX-512. MariaDB MHNSW and AliSQL VIDX both "
                "document AVX-512 distance kernels, so both are running narrower "
                "SIMD paths here. Results are valid for this class of hardware "
                "and should not be extrapolated to AVX-512 machines."
            )
        if cpu.hybrid:
            self.add_warning(
                f"Hybrid CPU detected ({len(cpu.performance_cpus)} performance / "
                f"{len(cpu.efficiency_cpus)} efficiency logical CPUs). Runs are "
                f"pinned to one core class; unpinned numbers from this machine "
                f"would carry scheduling noise larger than the effects measured."
            )

    def set_config(self, profile: Dict[str, Any], resource_pass: str,
                   resolved: Any, extra: Optional[Dict[str, Any]] = None) -> None:
        self.data["config"] = {
            "profile": profile,
            "resource_pass": resource_pass,
            "resolved_resources": resolved.as_dict() if hasattr(resolved, "as_dict") else resolved,
            **(extra or {}),
        }
        for warning in getattr(resolved, "warnings", []) or []:
            self.add_warning(warning)

    def set_engine(self, engine: str, sources_dir: str,
                   image_runtime: str, image_bench: str,
                   image_ids: Dict[str, str]) -> None:
        source = _read_json(os.path.join(sources_dir, f"{engine}.source.json")) or {}
        build = _read_json(os.path.join(sources_dir, f"{engine}.image.json")) or {}
        self.data["engines"][engine] = {
            "source": source,
            "build": build,
            "images": {
                "runtime": {"ref": image_runtime, "id": image_ids.get("runtime", "unknown")},
                "bench": {"ref": image_bench, "id": image_ids.get("bench", "unknown")},
            },
        }
        self.save()

    def set_harness(self, sources_dir: str) -> None:
        annb = _read_json(os.path.join(sources_dir, "annbench.source.json"))
        if annb:
            self.data["ann_benchmarks"] = annb

    def add_phase(self, name: str, engine: str, dataset: str,
                  status: str, started_at: str, finished_at: str,
                  detail: Optional[Dict[str, Any]] = None) -> None:
        self.data["phases"].append({
            "phase": name,
            "engine": engine,
            "dataset": dataset,
            "status": status,
            "started_at": started_at,
            "finished_at": finished_at,
            **(detail or {}),
        })
        self.save()

    def add_warning(self, message: str) -> None:
        if message not in self.data["warnings"]:
            self.data["warnings"].append(message)
            self.save()

    def finish(self, status: str = "completed") -> None:
        self.data["finished_at"] = utcnow()
        self.data["status"] = status
        self.save()

    # -- persistence ----------------------------------------------------

    def save(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self.data, fh, indent=2, sort_keys=True, default=str)
            fh.write("\n")
        os.replace(tmp, self.path)

    @staticmethod
    def load(run_dir: str) -> Dict[str, Any]:
        path = os.path.join(run_dir, MANIFEST_NAME)
        data = _read_json(path)
        if data is None:
            raise FileNotFoundError(
                f"no manifest at {path}. Results without a manifest cannot be "
                f"reported: the environment they were produced in is unknown."
            )
        return data


def _git_commit(path: str) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", path, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        return proc.stdout.strip() or "uncommitted"
    except Exception:
        return "unknown"
