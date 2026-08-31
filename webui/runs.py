"""Run discovery and summarisation.

A run directory is any directory under results/ holding a run-manifest.json.
results/annb is the shared ann-benchmarks tree, not a run, and is excluded by
that rule rather than by name.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

MANIFEST_NAME = "run-manifest.json"
RECORDS_NAME = os.path.join("report", "records.jsonl")
REPORT_NAME = os.path.join("report", "report.html")


def load_manifest(run_dir: str) -> Optional[Dict[str, Any]]:
    try:
        with open(os.path.join(run_dir, MANIFEST_NAME)) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def is_run_dir(path: str) -> bool:
    return os.path.isfile(os.path.join(path, MANIFEST_NAME))


def _count_lines(path: str) -> int:
    try:
        with open(path, "rb") as fh:
            return sum(1 for _ in fh)
    except OSError:
        return 0


def _duration_s(manifest: Dict[str, Any]) -> Optional[float]:
    phases = manifest.get("phases") or []
    total = sum(float(p.get("duration_s") or 0) for p in phases)
    return round(total, 1) if total else None


def _phase_tally(manifest: Dict[str, Any]) -> Dict[str, int]:
    tally: Dict[str, int] = {}
    for phase in manifest.get("phases") or []:
        status = str(phase.get("status") or "unknown")
        tally = {**tally, status: tally.get(status, 0) + 1}
    return tally


def _datasets(manifest: Dict[str, Any]) -> List[str]:
    seen = [str(p.get("dataset")) for p in (manifest.get("phases") or [])
            if p.get("dataset")]
    if seen:
        return sorted(set(seen))
    profile = (manifest.get("config") or {}).get("profile") or {}
    return list(profile.get("datasets") or [])


def summarize(run_id: str, run_dir: str, manifest: Dict[str, Any]) -> Dict[str, Any]:
    config = manifest.get("config") or {}
    profile = config.get("profile") or {}
    host = manifest.get("host") or {}
    cpu = host.get("cpu") or {}
    records_path = os.path.join(run_dir, RECORDS_NAME)

    return {
        "run_id": manifest.get("run_id") or run_id,
        "dir_name": run_id,
        "status": manifest.get("status") or "unknown",
        "started_at": manifest.get("started_at"),
        "finished_at": manifest.get("finished_at"),
        "duration_s": _duration_s(manifest),
        "profile": profile.get("name"),
        "description": profile.get("description"),
        "resource_pass": config.get("resource_pass"),
        "engines": sorted((manifest.get("engines") or {}).keys()),
        "datasets": _datasets(manifest),
        "cpu_model": cpu.get("model"),
        "has_avx512": cpu.get("has_avx512"),
        "hybrid_cpu": cpu.get("hybrid"),
        "warning_count": len(manifest.get("warnings") or []),
        "phase_tally": _phase_tally(manifest),
        "has_report": os.path.isfile(os.path.join(run_dir, REPORT_NAME)),
        "has_records": os.path.isfile(records_path),
        "record_count": _count_lines(records_path),
    }


def discover_runs(results_dir: str) -> List[Dict[str, Any]]:
    """Every run under results_dir, newest first."""
    out: List[Dict[str, Any]] = []
    try:
        entries = sorted(os.listdir(results_dir))
    except OSError:
        return out

    for name in entries:
        run_dir = os.path.join(results_dir, name)
        if not os.path.isdir(run_dir) or not is_run_dir(run_dir):
            continue
        manifest = load_manifest(run_dir)
        if manifest is None:
            continue
        out.append(summarize(name, run_dir, manifest))

    return sorted(out, key=lambda r: (r.get("started_at") or "", r["dir_name"]),
                  reverse=True)


def resolve_run_dir(results_dir: str, run_id: str) -> Optional[str]:
    """Map a run id to its directory, refusing anything outside results_dir."""
    if not run_id or os.path.sep in run_id or run_id in (".", ".."):
        return None
    run_dir = os.path.join(results_dir, run_id)
    if not is_run_dir(run_dir):
        return None
    return run_dir
