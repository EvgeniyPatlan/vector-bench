#!/usr/bin/env python3
"""Generate the vector-bench report: charts, Markdown and self-contained HTML.

Runs inside an engine bench image (which already carries numpy, h5py,
matplotlib and jinja2), so producing a report does not require installing a
scientific Python stack on the machine under test.

Refuses to emit a report for a run directory with no manifest. Every conclusion
this framework can support depends on facts the manifest records — CPU model,
SIMD flags, source commits, resource limits — and a chart without them invites
exactly the over-generalisation docs/05-methodology.md warns against.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional

VB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for candidate in (VB_ROOT, "/opt", os.path.dirname(os.path.abspath(__file__))):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from report import charts, loaders  # noqa: E402
from report.render import render_html, render_markdown  # noqa: E402

ENGINE_LABEL = {
    "mariadb": "MariaDB (MHNSW)",
    "alisql": "AliSQL (VIDX)",
    "pgvector": "PostgreSQL (pgvector)",
}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="generate the vector-bench report")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--annb-results", default=None,
                   help="ann-benchmarks results tree (default: <run-dir>/../annb)")
    p.add_argument("--datasets-dir", default=None,
                   help="dataset HDF5 directory, needed to recompute recall")
    p.add_argument("--output-dir", default=None,
                   help="where to write the report (default: <run-dir>/report)")
    p.add_argument("--title", default="Vector search benchmark")
    return p.parse_args(argv)


def load_manifest(run_dir: str) -> Dict[str, Any]:
    path = os.path.join(run_dir, "run-manifest.json")
    if not os.path.exists(path):
        raise SystemExit(
            f"no run-manifest.json in {run_dir}.\n"
            "A results directory without a manifest cannot be reported: the "
            "hardware, engine versions and resource limits behind the numbers "
            "are unknown, and every conclusion depends on them."
        )
    with open(path) as fh:
        return json.load(fh)


def _parse_ts(value: Any) -> Optional[float]:
    """Manifest timestamps are ISO-8601 Z; return epoch seconds or None."""
    if not value:
        return None
    try:
        import datetime
        return datetime.datetime.fromisoformat(
            str(value).replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return None


def summarize(records: List[Dict[str, Any]],
              manifest: Optional[Dict[str, Any]] = None,
              memory: Optional[Dict[str, List[Dict[str, Any]]]] = None
              ) -> Dict[str, Any]:
    """Headline figures the narrative sections are written from."""
    from report.loaders import pareto_frontier

    recall_records = [r for r in records if r.get("phase") == "recall_qps"]
    datasets = sorted({r["dataset"] for r in records if r.get("dataset")})
    engines = sorted({r["engine"] for r in records if r.get("engine")})

    summary: Dict[str, Any] = {
        "datasets": datasets,
        "engines": engines,
        "per_dataset": {},
        "plan_failures": [],
        "short_result_cases": [],
        "failed_phases": [],
        "memory_pressure": [],
        "stale_ann": [],
        "duplicate_ann": [],
    }

    # Two results for one configuration means the report is reading more than
    # one measurement tree. Timestamps can be lost by copying a run between
    # machines, so this catches the same problem without depending on them: a
    # 16 GB curve and a 64 GB curve were merged into one chart, and the charts
    # silently showed whichever was faster at each point.
    by_config: Dict[Any, List[Dict[str, Any]]] = {}
    for r in recall_records:
        # The key must name every axis a profile is allowed to sweep, or a
        # legitimate curve looks like a duplicate. The tuned pass sweeps
        # storage engine for MariaDB (InnoDB and MyISAM) and ef_construction
        # for pgvector, and omitting those flagged 16 real measurements as
        # accidental repeats.
        by_config.setdefault(
            (r.get("engine"), r.get("dataset"), r.get("m"), r.get("ef_search"),
             r.get("build_mode"), r.get("storage_engine"),
             r.get("ef_construction")), []).append(r)
    for key, group in sorted(by_config.items(), key=lambda kv: str(kv[0])):
        if len(group) > 1:
            summary["duplicate_ann"].append({
                "engine": key[0], "dataset": key[1], "m": key[2],
                "ef_search": key[3], "storage_engine": key[5],
                "ef_construction": key[6], "count": len(group),
                "qps": sorted(round(g.get("qps") or 0, 1) for g in group),
            })

    # ann-benchmarks skips configurations that already have result files, and
    # reports that as success. A re-run after a config change therefore returns
    # instantly with the previous numbers, and nothing in the records says so.
    started = _parse_ts((manifest or {}).get("started_at"))
    if started:
        for r in recall_records:
            # The loader stores this under `extra`, alongside source_file.
            # Reading it from the top level meant the check never fired once.
            mtime = (r.get("extra") or {}).get("source_mtime") or r.get("source_mtime")
            if mtime and mtime < started:
                summary["stale_ann"].append({
                    "engine": r.get("engine"),
                    "dataset": r.get("dataset"),
                    "measured_at": mtime,
                    "run_started": started,
                    "source_file": (r.get("extra") or {}).get("source_file"),
                })

    # An engine that spent the phase against its cgroup limit was reclaiming
    # continuously, so its numbers describe the memory budget and not the
    # implementation. Nothing in the records shows this.
    limit = (((manifest or {}).get("config") or {})
             .get("resolved_resources") or {}).get("server_memory_bytes")
    for name, rows in sorted((memory or {}).items()):
        from report.loaders import ceiling_pressure
        pressure = ceiling_pressure(rows, limit)
        if not pressure:
            continue
        engine = name.split("-", 1)[0]
        pressure["engine"] = engine
        pressure["series"] = name
        summary["memory_pressure"].append(pressure)

    # Any measurement taken while the vector index was NOT in the plan is a
    # full scan, and must be surfaced rather than averaged into a curve.
    for r in records:
        if r.get("vector_index_used") is False:
            summary["plan_failures"].append({
                "engine": r.get("engine"), "dataset": r.get("dataset"),
                "phase": r.get("phase"), "m": r.get("m"),
                "ef_search": r.get("ef_search"),
                "selectivity": r.get("selectivity"),
            })
        if (r.get("extra") or {}).get("returned_fewer_than_k"):
            summary["short_result_cases"].append({
                "engine": r.get("engine"), "dataset": r.get("dataset"),
                "selectivity": r.get("selectivity"),
                "queries": (r.get("extra") or {}).get("short_result_queries"),
            })

    for dataset in datasets:
        per_engine: Dict[str, Any] = {}
        for engine in engines:
            points = [r for r in recall_records
                      if r["dataset"] == dataset and r["engine"] == engine]
            frontier = pareto_frontier(points)
            if not frontier:
                continue
            # QPS at a recall floor is the comparison an operator actually
            # makes: "how fast is it, at accuracy I can accept?"
            entry: Dict[str, Any] = {"points": len(points)}
            for floor in (0.90, 0.95, 0.99):
                qualifying = [p for p in frontier if (p.get("recall_at_k") or 0) >= floor]
                entry[f"qps_at_recall_{int(floor * 100)}"] = (
                    max(p["qps"] for p in qualifying) if qualifying else None
                )
            entry["max_recall"] = max((p.get("recall_at_k") or 0) for p in frontier)
            per_engine[engine] = entry
        if per_engine:
            summary["per_dataset"][dataset] = per_engine

    # Build-cost and concurrency headlines.
    for phase, key in (("index_build", "build"), ("concurrency", "concurrency"),
                       ("filtered", "filtered"), ("churn", "churn"),
                       ("ingest", "ingest")):
        summary[key] = [r for r in records if r.get("phase") == phase]

    # A phase that failed leaves a hole in the results. Without this the report
    # simply omits that engine/pass and a reader cannot distinguish "not
    # measured" from "measured and unremarkable".
    for phase in (manifest or {}).get("phases", []):
        if phase.get("status") != "completed":
            summary["failed_phases"].append(phase)

    return summary


def _narrow_to_this_run(annb_dir: str, manifest: Dict[str, Any]) -> str:
    """Point the loader at the tree this run actually wrote, if it exists.

    Results are stored under <pass>/<config fingerprint>. Walking the parent
    would pull in every configuration ever measured on this machine, which is
    how a 16 GB curve ended up in a report whose manifest said 64 GB. Falls
    back to the whole tree for older runs that predate the fingerprint, where
    the staleness check below is the only guard.
    """
    cfg = manifest.get("config") or {}
    pass_name = cfg.get("resource_pass")
    # Taken from the manifest, not recomputed. This module runs inside a bench
    # image that mounts report/ and harness/ only, so importing the orchestrator
    # to derive it is not an option.
    fingerprint = cfg.get("ann_fingerprint")
    if not (pass_name and fingerprint):
        return annb_dir
    candidate = os.path.join(annb_dir, pass_name, fingerprint)
    if os.path.isdir(candidate):
        print(f"[report] ann-bench tree: {pass_name}/{fingerprint}")
        return candidate
    print(f"[report] no tree at {pass_name}/{fingerprint}; reading the whole "
          f"ann-benchmarks directory. Results older than this run will be "
          f"flagged in Validity.")
    return annb_dir


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    run_dir = os.path.abspath(args.run_dir)
    manifest = load_manifest(run_dir)
    run_id = manifest.get("run_id", os.path.basename(run_dir))

    annb_dir = args.annb_results or os.path.join(os.path.dirname(run_dir), "annb")
    annb_dir = _narrow_to_this_run(annb_dir, manifest)
    datasets_dir = args.datasets_dir or os.path.join(VB_ROOT, "datasets")
    if not os.path.isdir(datasets_dir):
        datasets_dir = "/datasets"
    out_dir = args.output_dir or os.path.join(run_dir, "report")
    chart_dir = os.path.join(out_dir, "charts")
    os.makedirs(chart_dir, exist_ok=True)

    print(f"[report] run       : {run_id}")
    print(f"[report] ann-bench : {annb_dir}")
    print(f"[report] datasets  : {datasets_dir}")
    print(f"[report] output    : {out_dir}")

    records: List[Dict[str, Any]] = []
    records += loaders.load_ops_records(run_dir)
    print(f"[report] ops records: {len(records)}")

    annb = loaders.load_annb_results(annb_dir, datasets_dir, run_id)
    records += annb
    print(f"[report] ann-benchmarks records: {len(annb)}")

    if not records:
        print("[report] no records found — nothing to report", file=sys.stderr)
        return 1

    memory = loaders.load_memory_series(run_dir)

    # Fill in peak memory from the timeseries where the harness could not read
    # a kernel high-water mark itself.
    peaks = {name: loaders.peak_from_series(rows) for name, rows in memory.items()}
    for r in records:
        if r.get("phase") == "index_build" and not r.get("peak_rss_bytes"):
            prefix = f"{r.get('engine')}-{r.get('dataset')}-{r.get('resource_pass')}"
            match = next((v for name, v in peaks.items()
                          if name.startswith(prefix) and v), None)
            if match:
                r["peak_rss_bytes"] = match

    # Persisted AFTER the enrichment above, not before: writing it earlier left
    # records.jsonl missing the peak-RSS values the rendered report displayed,
    # so the raw data and the report disagreed.
    merged_path = os.path.join(out_dir, "records.jsonl")
    with open(merged_path, "w") as fh:
        for r in records:
            fh.write(json.dumps(r, sort_keys=True, default=str) + "\n")
    print(f"[report] merged records -> {merged_path}")

    summary = summarize(records, manifest, memory)
    chart_paths: Dict[str, Dict[str, str]] = {}

    for dataset in summary["datasets"]:
        by_engine: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in records:
            if r.get("phase") == "recall_qps" and r.get("dataset") == dataset:
                by_engine[r["engine"]].append(r)
        k = next((r.get("k") for r in records
                  if r.get("dataset") == dataset and r.get("k")), 10)

        safe = dataset.replace(".", "_")

        # A profile chooses which workloads to run. Drawing an empty axis for
        # one that was never enabled has confused every reader of these
        # reports so far -- an absent chart reads as a broken report, when it
        # only means `workloads:` did not include it. Charts whose phase
        # produced no records are skipped entirely, and the section says so.
        phases_present = {r.get("phase") for r in records
                          if r.get("dataset") == dataset}
        needs = {
            "latency": "concurrency",
            "concurrency": "concurrency",
            "filtered": "filtered",
            "churn": "churn",
            "churnimpact": "churn",
        }

        for name, fn, payload in (
            ("pareto", charts.pareto, (dict(by_engine), dataset, k)),
            ("paretozoom", lambda *a, **kw: charts.pareto(*a, recall_floor=0.85, **kw),
             (dict(by_engine), dataset, k)),
            ("qpsatrecall", charts.qps_at_recall, (summary, dataset)),
            ("latency", charts.latency_percentiles, (records, dataset)),
            ("build", charts.build_cost, (records, dataset)),
            ("storage", charts.storage_breakdown, (records, dataset)),
            ("concurrency", charts.concurrency, (records, dataset)),
            ("filtered", charts.filtered, (records, dataset)),
            ("churn", charts.churn, (records, dataset)),
            ("churnimpact", charts.churn_impact, (records, dataset)),
            ("passcompare", charts.pass_comparison, (records, dataset)),
        ):
            required = needs.get(name)
            if required and required not in phases_present:
                print(f"[report] skipping {name}: no {required} records "
                      f"(not in this profile's workloads)")
                continue
            stem = f"{name}-{safe}"
            try:
                paths = fn(*payload, chart_dir, stem)
            except Exception as exc:  # noqa: BLE001 - one bad chart must not kill the report
                print(f"[report] chart {stem} failed: {exc}")
                paths = None
            if paths:
                chart_paths[stem] = paths
                print(f"[report] chart -> {os.path.basename(paths['svg'])}")

    if memory:
        paths = charts.memory_timeline(memory, chart_dir, "memory-timeline")
        if paths:
            chart_paths["memory-timeline"] = paths

    md_path = os.path.join(out_dir, "report.md")
    html_path = os.path.join(out_dir, "report.html")

    markdown = render_markdown(manifest, summary, records, chart_paths, args.title)
    with open(md_path, "w") as fh:
        fh.write(markdown)
    print(f"[report] markdown -> {md_path}")

    html = render_html(manifest, summary, records, chart_paths, args.title)
    with open(html_path, "w") as fh:
        fh.write(html)
    print(f"[report] html -> {html_path}")

    if summary["plan_failures"]:
        print(f"\n[report] WARNING: {len(summary['plan_failures'])} measurement(s) ran "
              f"WITHOUT the vector index in the query plan. Those are full scans, "
              f"not ANN search — see the 'Validity' section of the report.")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
