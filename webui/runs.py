"""Run discovery and summarisation.

A run directory is any directory under results/ holding a run-manifest.json.
results/annb is the shared ann-benchmarks tree, not a run, and is excluded by
that rule rather than by name.
"""

from __future__ import annotations

import json
import os
import socket
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


def report_inputs(results_dir: str, datasets_dir: str, run_dir: str,
                  manifest: Dict[str, Any]) -> Dict[str, Any]:
    """What regenerating this run's report would have to work from.

    A run directory is self-contained to *view*: report.html inlines its charts.
    Regenerating is different. The recall measurements live in the
    ann-benchmarks tree at results/annb/<pass>/<fingerprint>/, which is a
    sibling of the run rather than inside it, and scoring them needs the dataset
    to compute ground truth against. Copy a run from another machine and
    neither travels, so regenerating there silently produces a report with the
    ops measurements and no recall curves.
    """
    config = manifest.get("config") or {}
    resource_pass = config.get("resource_pass")
    fingerprint = config.get("ann_fingerprint")
    if resource_pass and not fingerprint and config.get("resolved_resources"):
        try:
            from orchestrator.ann_pass import ann_fingerprint
            fingerprint = ann_fingerprint(config["resolved_resources"])
        except Exception:  # noqa: BLE001
            fingerprint = None

    # Mirror what generate_report actually does: prefer the tree keyed by this
    # run's resource configuration, and fall back to everything under annb/ when
    # that path does not exist -- which is the case for every run recorded
    # before the fingerprint was introduced. Checking only the keyed path
    # reported "recall will be lost" for legacy runs whose recall is right
    # there.
    annb_root = os.path.join(results_dir, "annb")
    tree = None
    narrowed = False
    if resource_pass and fingerprint:
        candidate = os.path.join(annb_root, resource_pass, fingerprint)
        if os.path.isdir(candidate):
            tree, narrowed = candidate, True
    if tree is None:
        tree = annb_root
    tree_files = _count_hdf5(tree)

    records = _records_summary(run_dir)
    missing_datasets = sorted(
        name for name in records["recall_datasets"]
        if not os.path.isfile(os.path.join(datasets_dir, f"{name}.hdf5")))

    measured_on = ((manifest.get("host") or {}).get("hostname") or "")
    try:
        elsewhere = bool(measured_on) and measured_on != socket.gethostname()
    except OSError:
        elsewhere = False

    # Three outcomes, not two. Losing the recall section is the obvious one;
    # quietly gaining someone else's is worse, and happens whenever the tree
    # cannot be narrowed to this run's own configuration.
    if narrowed:
        risk, note = "none", ""
    elif tree_files == 0 and records["recall"]:
        risk = "loses_recall"
        note = (f"This run's report has {records['recall']} recall measurements. "
                f"They live in results/annb/, which is a sibling of the run "
                f"directory and does not travel with it, and there are none on "
                f"this machine. Regenerating would produce a report with the "
                f"ops measurements and no recall curves.")
    elif tree_files == 0:
        risk, note = "none", ""
    else:
        risk = "unnarrowed"
        note = (f"This run does not record which ann results are its own, so "
                f"regenerating reads every ann result on this machine "
                f"({tree_files} files) rather than only this run's. "
                + (f"This run was measured on {measured_on}, not here, so those "
                   f"are somebody else's measurements. "
                   if elsewhere else "")
                + "The report flags what it could not attribute in its Validity "
                  "section.")

    if missing_datasets and risk != "loses_recall":
        note = (note + " " if note else "") + (
            f"Ground truth is recomputed from the dataset files, and "
            f"{', '.join(missing_datasets)} is not here, so those recall "
            f"points would be skipped.")
        risk = risk if risk != "none" else "missing_datasets"

    return {
        "ann_tree": tree,
        "ann_tree_narrowed": narrowed,
        "ann_results_present": tree_files,
        "recall_records": records["recall"],
        "recall_datasets": sorted(records["recall_datasets"]),
        "missing_datasets": missing_datasets,
        "measured_on": measured_on,
        "measured_elsewhere": elsewhere,
        "regenerate_risk": risk,
        "regenerate_note": note,
    }


def _count_hdf5(path: str) -> int:
    total = 0
    for _root, _dirs, files in os.walk(path):
        total += sum(1 for f in files if f.endswith(".hdf5"))
    return total


def _records_summary(run_dir: str) -> Dict[str, Any]:
    path = os.path.join(run_dir, RECORDS_NAME)
    recall = 0
    datasets = set()
    try:
        with open(path) as fh:
            for line in fh:
                if '"recall_qps"' not in line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("phase") == "recall_qps":
                    recall += 1
                    if record.get("dataset"):
                        datasets.add(str(record["dataset"]))
    except OSError:
        pass
    return {"recall": recall, "recall_datasets": datasets}


def resolve_run_dir(results_dir: str, run_id: str) -> Optional[str]:
    """Map a run id to its directory, refusing anything outside results_dir."""
    if not run_id or os.path.sep in run_id or run_id in (".", ".."):
        return None
    run_dir = os.path.join(results_dir, run_id)
    if not is_run_dir(run_dir):
        return None
    return run_dir
