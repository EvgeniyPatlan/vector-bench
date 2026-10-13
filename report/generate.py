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
from report import measurement_versions as versions  # noqa: E402
from report.render import (adopt_presentation as render_adopt_presentation,  # noqa: E402
                           render_html, render_markdown)

from report.render import ENGINE_LABEL  # noqa: E402  (one table, not two)


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
    p.add_argument("--datasets", default=None,
                   help="comma-separated dataset names to report on. The ann "
                        "results tree is keyed by resource configuration, not "
                        "by corpus, so a machine that has measured two corpora "
                        "under one configuration reports both. Both are "
                        "correct and separated, but a report meant to describe "
                        "one of them should say which.")
    p.add_argument("--from-records", default=None,
                   help="regenerate from a previously merged records.jsonl "
                        "instead of reading the ops and ann-benchmarks trees. "
                        "Use when a run directory has been archived or copied "
                        "away from the machine that produced it.")
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


#: Which phase lands in which `summary` bucket, as a named constant rather
#: than a literal inside `summarize`, for two reasons.
#:
#: The first is that `ef_calibration` has to be ABSENT from it and absence is
#: invisible in an inline tuple. A calibration point carries `clients=1` and a
#: `qps`, so a sixth entry mapping it to "concurrency" would put up to twelve
#: single-client probes per engine into `summary["concurrency"]` and from there
#: into both the concurrency table and the concurrency chart, beside the
#: numbers this report publishes as throughput. The two are not the same
#: quantity: a calibration point is one client over `--max-queries` fresh
#: queries, a concurrency point is N clients over the shared query pool, and
#: `extra.query_pool` and `extra.query_reuse` are recorded precisely because
#: throughput over a small hot set and throughput over the whole corpus are
#: measurements of different things. The sweep's records reach the report
#: through `operating_points` instead, which is a join over them rather than a
#: phase bucket.
#:
#: The second is that the absence is then testable: a test can add that entry
#: at run time and watch the published QPS move, which is the only way to show
#: the exclusion is doing work rather than being a comment.
_PHASE_GROUPS = (
    ("index_build", "build"),
    ("concurrency", "concurrency"),
    ("filtered", "filtered"),
    ("churn", "churn"),
    ("ingest", "ingest"),
)


#: What `extra.filter_column` means when it is absent, which it is on every
#: filtered record written before PR 3: all 69 archived pgvector filtered
#: records, every record `tests/synthetic.py` builds, and anything a
#: pre-T3c harness produced. Those runs had one filter column and it was
#: `tag`, so reading absence as `tag` is a statement of fact about them
#: rather than a default -- and the alternative, bucketing them as "unknown",
#: would split one engine's `tag` curve across two series on every chart that
#: reads an archived run.
FILTER_COLUMN_WHEN_ABSENT = "tag"


def filter_column(record: Dict[str, Any]) -> str:
    """The filter column a filtered record measured."""
    return ((record.get("extra") or {}).get("filter_column")
            or FILTER_COLUMN_WHEN_ABSENT)


def filtered_chart_entries(records: List[Dict[str, Any]], dataset: str,
                           draw=None) -> List[Any]:
    """The `(name, fn, payload)` chart entries for the filtered section.

    One per filter column present, or -- when there are no filtered records
    at all -- the single `filtered` entry this always had, so that the `needs`
    table still matches it and the shared "not in this profile's workloads"
    skip line still prints. Without that fallback a profile running no
    filtered workload would generate no chart and say nothing about why.

    `functools.partial`, not a lambda. A lambda written
    `lambda *a, **kw: charts.filtered(*a, column=column, **kw)` closes over
    the NAME `column`, not its value, so every entry in the list would plot
    whichever column the comprehension finished on: two filenames, one chart,
    and the `tag` curve silently gone from a report that still looks
    complete. `partial` binds the value at construction and the mistake
    cannot be written. A module-level function rather than an expression
    inside `main()` so the binding is testable without a run directory --
    `draw` is the chart function, injectable for exactly that.
    """
    from functools import partial

    from report import charts as charts_mod
    draw = draw or charts_mod.filtered
    columns = _filter_columns_present(records, dataset)
    if not columns:
        return [("filtered", draw, (records, dataset))]
    return [(f"filtered-{column}", partial(draw, column=column),
             (records, dataset))
            for column in columns]


#: The axes `cli.py:_run_unit` loops over. One ops unit is one
#: (M, build_mode, storage_engine) combination inside one resource pass, and
#: each unit reloads the corpus, rebuilds the index and writes its OWN full
#: column x selectivity grid of filtered records. `main` gives pgvector two
#: units per (pass, dataset); `full` gives mariadb three and pgvector six;
#: `--resource-pass both` doubles all of them.
OPS_UNIT_FIELDS = ("m", "build_mode", "storage_engine", "resource_pass")


def ops_unit(record: Dict[str, Any]) -> tuple:
    """Which ops unit a record was measured in.

    Any derivation that compares two filtered points -- a latency ratio
    across a selectivity curve, most of all -- has to hold these constant or
    it is comparing two indexes built differently on two different machine
    allocations. Measured on a two-build-mode fixture: `post` alone gives a
    0.20x ratio and the verdict "pre-filter", `incremental` alone gives 1.00x
    and "post-filter, abandons the query", and pooling them publishes ONE row
    at 50.0x and "post-filter, exhaustive" -- a verdict neither unit
    produced, obtained by dividing one unit's 1% latency by the other's 50%.
    """
    return tuple(record.get(field) for field in OPS_UNIT_FIELDS)


def _filter_columns_present(records: List[Dict[str, Any]],
                            dataset: str) -> List[str]:
    """The filter columns this dataset has filtered records for, in order.

    `FILTER_COLUMNS`' order rather than first-seen order, so the list does
    not depend on whatever order the JSONL happens to be in -- which is what
    makes the generated file set reproducible for a given run.

    It does NOT decide the order they appear in the report: `render.figures()`
    iterates `sorted(chart_paths.items())`, so `filtered-ctag-<dataset>`
    renders before `filtered-tag-<dataset>` alphabetically. Changing that
    means changing how every figure in every section is ordered, which is a
    larger decision than this function should make; both charts carry the
    column in their title and caption, so a reader is not misled.
    """
    present = {filter_column(r) for r in records
               if r.get("phase") == "filtered" and r.get("dataset") == dataset}
    ordered = [c for c in ("tag", "ctag") if c in present]
    # Anything a future column adds is appended rather than dropped: a chart
    # missing entirely is harder to notice than one with an unexpected name.
    return ordered + sorted(present - set(ordered))


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
        "recall_floor_gaps": [],
        "silent_ann_failures": [],
        "qps_inversions": [],
        "contended_phases": [],
        "filter_strategies": [],
        # Points the filtered budget stopped early. A record that measured 93
        # of 500 planned queries is a real measurement of a slow
        # configuration, not a failure -- but its recall carries the variance
        # of 93 queries and the report must not draw it as though it carried
        # 500. `insufficient_sample` is the stricter flag of the two: fewer
        # than 50 queries finished.
        "time_bounded_points": [],
        "passes": sorted({r.get("resource_pass") for r in records
                          if r.get("resource_pass")}),
    }

    # Two results for one configuration means the report is reading more than
    # one measurement tree. Timestamps can be lost by copying a run between
    # machines, so this catches the same problem without depending on them: a
    # 16 GB curve and a 64 GB curve were merged into one chart, and the charts
    # silently showed whichever was faster at each point.
    from report.loaders import build_signature
    by_config: Dict[Any, List[Dict[str, Any]]] = {}
    for r in recall_records:
        # The key must name every axis a profile is allowed to sweep, or a
        # legitimate curve looks like a duplicate. Listing them by hand failed
        # twice: once when the tuned pass began sweeping storage engine and
        # ef_construction, flagging 16 real measurements as repeats, and again
        # when Percona Search gained a quantization axis that is not a column
        # at all, flagging 8 more. build_signature takes the configuration from
        # the result filename instead, which covers axes this code has never
        # heard of.
        by_config.setdefault(
            (build_signature(r), r.get("ef_search")), []).append(r)
    for key, group in sorted(by_config.items(), key=lambda kv: str(kv[0])):
        if len(group) > 1:
            first = group[0]
            summary["duplicate_ann"].append({
                "engine": first.get("engine"), "dataset": first.get("dataset"),
                "m": first.get("m"), "ef_search": key[1],
                "storage_engine": first.get("storage_engine"),
                "ef_construction": first.get("ef_construction"),
                "count": len(group),
                "qps": sorted(round(g.get("qps") or 0, 1) for g in group),
            })

    # Throughput that rises as query effort rises. Examining more candidates
    # cannot be faster on the same index, so an inversion is never the engine
    # -- it is the machine, and the measurement describes whatever else was on
    # it. AliSQL measured 2.6 QPS at ef_search 400 and 23.3 at 800 in one
    # process against one index, and nobody noticed for five days because
    # nothing looked at the series as a series.
    #
    # The warm-up artifact lands here too, and should: the first configuration
    # of a sweep pays for a cold cache and reads as an inversion of about
    # 1.8x. Both are reported, with the ratio, because the ratio is what
    # separates a cold cache from a contaminated window.
    by_series: Dict[Any, List[Dict[str, Any]]] = {}
    for r in recall_records:
        if r.get("qps") is None or r.get("ef_search") is None:
            continue
        by_series.setdefault(build_signature(r), []).append(r)
    for key, group in sorted(by_series.items(), key=lambda kv: str(kv[0])):
        ordered = sorted(group, key=lambda r: r["ef_search"])
        for previous, current in zip(ordered, ordered[1:]):
            if current["qps"] <= previous["qps"]:
                continue
            summary["qps_inversions"].append({
                "engine": previous.get("engine"),
                "dataset": previous.get("dataset"),
                "storage_engine": previous.get("storage_engine"),
                "ef_construction": previous.get("ef_construction"),
                "from_ef": previous["ef_search"], "to_ef": current["ef_search"],
                "from_qps": round(previous["qps"], 1),
                "to_qps": round(current["qps"], 1),
                "ratio": round(current["qps"] / previous["qps"], 2),
                "first_in_sweep": previous["ef_search"] == ordered[0]["ef_search"],
            })
    summary["qps_inversions"].sort(key=lambda i: -i["ratio"])

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

    # CPU the machine spent that the container being measured did not. A phase
    # that ran under someone else's job is not a measurement of the engine, and
    # until both counters were sampled there was no way to say so.
    for name, rows in sorted((memory or {}).items()):
        from report.loaders import foreign_cpu
        contention = foreign_cpu(rows)
        if not contention:
            continue
        contention["series"] = name
        contention["engine"] = name.split("-", 1)[0]
        summary["contended_phases"].append(contention)
    summary["contended_phases"].sort(key=lambda c: -c["foreign_cores"])

    # Any measurement taken while the vector index was NOT in the plan is a
    # full scan, and must be surfaced rather than averaged into a curve.
    for r in records:
        if r.get("vector_index_used") is False:
            summary["plan_failures"].append({
                "engine": r.get("engine"), "dataset": r.get("dataset"),
                "phase": r.get("phase"), "m": r.get("m"),
                "ef_search": r.get("ef_search"),
                "selectivity": r.get("selectivity"),
                # `explain_uses_vector_index` runs per (column, selectivity)
                # since PR 3 §3.3 item 2, because the MySQL-family optimizers
                # can switch to `ctag_idx` or to a full scan at a different
                # threshold than they do for `tag`. A plan failure that does
                # not say which column it was asked about names the wrong
                # index to go and look at.
                "filter_column": (filter_column(r)
                                  if r.get("phase") == "filtered" else None),
            })
        if (r.get("extra") or {}).get("time_bounded"):
            extra = r.get("extra") or {}
            summary["time_bounded_points"].append({
                "engine": r.get("engine"), "dataset": r.get("dataset"),
                "filter_column": filter_column(r),
                "selectivity": r.get("selectivity"),
                "resource_pass": r.get("resource_pass"),
                "build_mode": r.get("build_mode"),
                "queries_executed": r.get("queries_executed"),
                "queries_planned": extra.get("queries_planned"),
                "budget_s": extra.get("filtered_budget_s"),
                "insufficient_sample": bool(extra.get("insufficient_sample")),
            })
        if (r.get("extra") or {}).get("returned_fewer_than_k"):
            summary["short_result_cases"].append({
                "engine": r.get("engine"), "dataset": r.get("dataset"),
                "selectivity": r.get("selectivity"),
                # Without the column, `tag` at 10% and `ctag` at 10.85% are
                # two rows differing only in a selectivity a reader cannot
                # attribute to a predicate.
                "filter_column": filter_column(r),
                # Without these, four measurements of one engine are one row
                # printed four times and the table reads as broken.
                "resource_pass": r.get("resource_pass"),
                "build_mode": r.get("build_mode"),
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
            # Which storage engine reached each floor, not only how fast. The
            # frontier spans every configuration swept, so on the tuned pass
            # "MariaDB 12.3: 1,176 QPS" is a MyISAM result -- printing it
            # unattributed credits it to the default build.
            entry["storage_engines"] = sorted(
                {p.get("storage_engine") for p in points if p.get("storage_engine")})
            for floor in (0.90, 0.95, 0.99):
                qualifying = [p for p in frontier if (p.get("recall_at_k") or 0) >= floor]
                best = max(qualifying, key=lambda p: p["qps"]) if qualifying else None
                entry[f"qps_at_recall_{int(floor * 100)}"] = best["qps"] if best else None
                entry[f"qps_at_recall_{int(floor * 100)}_storage"] = (
                    best.get("storage_engine") if best else None)
                entry[f"qps_at_recall_{int(floor * 100)}_recall"] = (
                    best.get("recall_at_k") if best else None)
            entry["max_recall"] = max((p.get("recall_at_k") or 0) for p in frontier)
            # The floor columns compare engines at an accuracy the operator
            # accepts, which is only a comparison if both engines have
            # measurements near it. ef_search cannot go below k and MHNSW
            # exposes no ef_construction, so with M pinned there is no MariaDB
            # configuration returning recall below about 0.975: its 0.90 and
            # 0.95 figures are one measurement printed twice. Recording the
            # lowest recall reached is what lets the report say so.
            entry["min_recall"] = min((p.get("recall_at_k") or 0) for p in points)
            for floor in (0.90, 0.95, 0.99):
                if entry[f"qps_at_recall_{int(floor * 100)}"] is None:
                    continue
                if entry["min_recall"] > floor:
                    summary["recall_floor_gaps"].append({
                        "engine": engine, "dataset": dataset, "floor": floor,
                        "lowest_recall": entry["min_recall"],
                        "measured_at": entry[f"qps_at_recall_{int(floor * 100)}_recall"],
                        "qps": entry[f"qps_at_recall_{int(floor * 100)}"],
                    })
            per_engine[engine] = entry
        if per_engine:
            summary["per_dataset"][dataset] = per_engine

    # Which filtering strategy each engine actually used, derived from how its
    # latency moves when the predicate narrows.
    #
    # A pre-filter compares only the qualifying vectors, so a ten-times
    # narrower predicate is roughly ten times less work. Anything that walks
    # the graph and applies the predicate during or after the walk pays more,
    # because it has to travel further to find k survivors. The direction of
    # the change separates them, and it is the one thing here that does not
    # depend on trusting a vendor's description.
    #
    # This exists because the report asserted for weeks that Valkey "chooses
    # its filtering strategy per query", taken from its documentation. The run
    # showed 18.2 ms at 10% selectivity and 120.6 ms at 1% -- six times more
    # work for a tenth of the candidates, which is inline filtering and not a
    # planner choosing to pre-filter.
    # The key has to hold constant everything that changes the answer, because
    # the derivation takes the narrowest and the widest point of a series and
    # divides their latencies. Anything pooled into one series is a pair of
    # points that may never have run together, and the result is published as
    # a verdict about an engine.
    #
    # Four kinds of confound, all reachable in configured profiles:
    #
    # * The filter COLUMN. A series holding both can pair a `ctag` point
    #   against a `tag` one -- two different predicates over two different
    #   qualifying sets. `tag` at 1% and `ctag` at 1.9% (the share a nominal
    #   1% actually delivers on the 2,000-row fixture) are not two points on
    #   one curve.
    # * The iterative-scan MODE. pgvector with index iteration off and with it
    #   on walk different amounts of graph, which is the entire reason the
    #   mode is recorded.
    # * The OPS UNIT -- M, build mode, storage engine. `_run_unit` loops
    #   `m_values x build_modes x storage_engines` and each pass through it
    #   reloads the corpus, rebuilds the index and writes its own full grid.
    #   pgvector's `post` and `incremental` builds are different indexes;
    #   MariaDB's InnoDB and MyISAM are different storage. Measured, one
    #   engine and one column over two build modes: 0.20x "pre-filter" and
    #   1.00x "post-filter, abandons the query" separately, against a pooled
    #   50.0x "post-filter, exhaustive" that neither produced.
    # * The RESOURCE PASS. normalized and tuned give the engine different
    #   memory and cpusets; pairing a normalized 1% point against a tuned 50%
    #   one published a ratio of 100.0x on the same fixture.
    #
    # The report has already been wrong about exactly this class of thing
    # once, asserting for weeks that Valkey "chooses its filtering strategy
    # per query" on the strength of its documentation.
    by_engine: Dict[Any, List[Dict[str, Any]]] = {}
    for r in records:
        if (r.get("phase") == "filtered" and r.get("selectivity")
                and r.get("latency_p50_ms")):
            key = ((r.get("engine"), r.get("dataset"), filter_column(r),
                    (r.get("extra") or {}).get("iterative_scan"))
                   + ops_unit(r))
            by_engine.setdefault(key, []).append(r)
    for key, points in sorted(by_engine.items(), key=lambda kv: str(kv[0])):
        engine, dataset, column, scan_mode = key[:4]
        # Zipped rather than unpacked positionally, so the key can carry
        # however many axes `OPS_UNIT_FIELDS` names. A fixed-length unpack
        # here made the obvious regression -- dropping an axis from the key --
        # raise a ValueError instead of publishing a wrong ratio, which is a
        # mutant that cannot be injected and therefore a guard that cannot be
        # proven.
        unit = dict(zip(OPS_UNIT_FIELDS, key[4:]))
        if len(points) < 2:
            continue
        points.sort(key=lambda r: r["selectivity"])
        narrow, wide = points[0], points[-1]
        ratio = narrow["latency_p50_ms"] / wide["latency_p50_ms"]
        span = wide["selectivity"] / narrow["selectivity"]
        # Recall is what separates "gives up" from "keeps looking": an engine
        # that returns quickly because it stopped early shows it here.
        collapsed = (narrow.get("recall_at_k") or 1) < 0.5
        if ratio < 0.9:
            observed = "pre-filter"
        elif collapsed:
            observed = "post-filter, abandons the query"
        elif ratio < 10:
            observed = "filters during the graph walk"
        else:
            observed = "post-filter, exhaustive"
        summary["filter_strategies"].append({
            "engine": engine, "dataset": dataset,
            "filter_column": column, "iterative_scan": scan_mode,
            # Carried so the rendered row can say which unit it describes.
            # A table that cannot tell two units apart prints the same row
            # twice and reads as a rendering fault -- which is the reason the
            # `short_result_cases` block beside this one carries them.
            # `.get`, not `[...]`: an axis absent from OPS_UNIT_FIELDS is
            # reported as None rather than crashing, so the rendered table
            # shows a blank cell where the pooling happened.
            "m": unit.get("m"), "build_mode": unit.get("build_mode"),
            "storage_engine": unit.get("storage_engine"),
            "resource_pass": unit.get("resource_pass"),
            "narrow": narrow["selectivity"], "wide": wide["selectivity"],
            "narrow_p50": narrow["latency_p50_ms"],
            "wide_p50": wide["latency_p50_ms"],
            "narrow_recall": narrow.get("recall_at_k"),
            "ratio": round(ratio, 2), "span": round(span, 1),
            "observed": observed,
        })
    summary["filter_strategies"].sort(
        key=lambda s: (s["ratio"], s["engine"] or "", s["filter_column"],
                       str(s["resource_pass"]),
                       # Numeric, so M 8/16/32 does not come out 16, 32, 8 --
                       # which `full`'s `m_values: [8, 16, 32]` would produce.
                       s["m"] or 0,
                       str(s["build_mode"]), str(s["storage_engine"])))

    # Build-cost and concurrency headlines.
    for phase, key in _PHASE_GROUPS:
        summary[key] = [r for r in records if r.get("phase") == phase]

    # PR 2 §2.3.1: the ef each calibrated unit settled on, with the recall and
    # the status that justify it. Built here rather than in `render.py` so the
    # concurrency chart and the operating-point table are drawn from one
    # derivation -- a chart marking an engine `floor_above_target` beside a
    # table that does not is worse than neither.
    from report.loaders import calibration_targets, operating_points
    summary["operating_points"] = operating_points(records)
    summary["calibration_targets"] = calibration_targets(records)

    # An engine whose ann phase ran and wrote nothing.
    #
    # ann-benchmarks catches a per-algorithm exception and exits zero, so a
    # module that raises leaves the phase marked completed and the engine
    # simply absent from the recall comparison. Three runs of Percona Search
    # failed that way before anyone noticed the engine was missing rather than
    # slow, because nothing distinguished "did not run" from "ran and produced
    # no measurement".
    measured = {r.get("engine") for r in recall_records}
    for phase in (manifest or {}).get("phases", []):
        engine = phase.get("engine")
        if (phase.get("phase") == "ann" and phase.get("status") == "completed"
                and engine and engine not in measured):
            summary["silent_ann_failures"].append({
                "engine": engine,
                "dataset": phase.get("dataset"),
                "resource_pass": phase.get("resource_pass"),
                "duration_s": phase.get("duration_s"),
            })

    # A phase that failed leaves a hole in the results. Without this the report
    # simply omits that engine/pass and a reader cannot distinguish "not
    # measured" from "measured and unremarkable".
    for phase in (manifest or {}).get("phases", []):
        if phase.get("status") != "completed":
            summary["failed_phases"].append(phase)

    return summary


def attach_peak_rss(records: List[Dict[str, Any]],
                    peaks: Dict[str, Optional[int]]) -> None:
    """Fill in peak server memory from the sampled timeseries, in place.

    Matched on the exact series name rather than a prefix. A prefix of
    engine-dataset-pass matches both `...-m16-post` and `...-m16-post-myisam`,
    so MariaDB's InnoDB and MyISAM builds were both given whichever series came
    first: the InnoDB row was published at 13.9 GiB against a true 28.7.

    A value derived here is tagged as such, so regenerating a report from a
    merged records.jsonl recomputes it rather than trusting it. A value the
    harness read from the kernel's high-water mark carries no tag and is left
    alone: polling a timeseries can miss a peak that memory.peak recorded.
    """
    from report.loaders import memory_stem

    for r in records:
        if r.get("phase") != "index_build":
            continue
        derived = (r.get("extra") or {}).get("peak_rss_source") == "sampled"
        if r.get("peak_rss_bytes") and not derived:
            continue
        peak = peaks.get(memory_stem(r))
        if peak:
            r["peak_rss_bytes"] = peak
            r.setdefault("extra", {})["peak_rss_source"] = "sampled"


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
    # Engine colours and labels come from the run that produced the numbers, so
    # an engine added after this code was written still names and draws itself.
    charts.adopt_presentation(manifest)
    render_adopt_presentation(manifest)
    run_id = manifest.get("run_id", os.path.basename(run_dir))

    annb_dir = args.annb_results or os.path.join(os.path.dirname(run_dir), "annb")
    annb_dir = _narrow_to_this_run(annb_dir, manifest)
    datasets_dir = args.datasets_dir or os.path.join(VB_ROOT, "datasets")
    if not os.path.isdir(datasets_dir):
        datasets_dir = "/datasets"
    out_dir = args.output_dir or os.path.join(run_dir, "report")
    chart_dir = os.path.join(out_dir, "charts")

    print(f"[report] run       : {run_id}")
    print(f"[report] ann-bench : {annb_dir}")
    print(f"[report] datasets  : {datasets_dir}")
    print(f"[report] output    : {out_dir}")

    # Loaded before anything is created or deleted under out_dir. The chart
    # cleanup below used to run first, so a regeneration that then refused --
    # or simply found no records -- left the previous report as a page of
    # broken images with index.html still pointing at the files it had just
    # removed. A report that cannot be rebuilt should leave the one that
    # already works alone.
    records: List[Dict[str, Any]] = []
    try:
        if args.from_records:
            # A run directory is routinely copied off the machine that produced
            # it, and the ann-benchmarks HDF5 tree is not copied with it. The
            # merged records file is self-contained, so a report can be rebuilt
            # from an archived run without re-running anything.
            with open(args.from_records) as fh:
                records = [json.loads(line) for line in fh if line.strip()]
            print(f"[report] records from {args.from_records}: {len(records)}")
            # The same guard load_ops_records applies, because this path does
            # not go through it. records.jsonl is written by this generator and
            # read back by the web UI, so a mixed set admitted here does not
            # just draw one bad chart -- it is persisted below as the file
            # webui/records.py prefers over the raw ops records, and every
            # later reader inherits it.
            # `declared` as well, so this path gets the whole guard and not
            # half of it. The manifest that will be printed above these numbers
            # is run_dir's, whatever file the records came out of -- so it is
            # run_dir's declared version they have to agree with.
            versions.refuse_mixed_measurement_versions(
                records, source=args.from_records,
                declared=versions.declared_ops_version(run_dir))
        else:
            records += loaders.load_ops_records(run_dir)
            print(f"[report] ops records: {len(records)}")

            annb = loaders.load_annb_results(annb_dir, datasets_dir, run_id)
            records += annb
            print(f"[report] ann-benchmarks records: {len(annb)}")
    except versions.MixedMeasurementVersions as exc:
        # Reported the way this module reports every other refusal to produce a
        # report -- a message and a non-zero exit, not a traceback. The message
        # is multi-line and names what to do; a stack trace above it would bury
        # that.
        print(f"[report] {exc}", file=sys.stderr)
        return 2

    if args.datasets:
        wanted = {d.strip() for d in args.datasets.split(",") if d.strip()}
        present = {r.get("dataset") for r in records if r.get("dataset")}
        unknown = wanted - present
        if unknown:
            print(f"[report] --datasets names nothing measured here: "
                  f"{sorted(unknown)}. Present: {sorted(present)}",
                  file=sys.stderr)
            return 2
        before = len(records)
        records = [r for r in records
                   if not r.get("dataset") or r["dataset"] in wanted]
        print(f"[report] narrowed to {sorted(wanted)}: "
              f"{len(records)} of {before} records")

    if not records:
        print("[report] no records found — nothing to report", file=sys.stderr)
        return 1

    # Only now, with records in hand and every refusal behind us, is the output
    # directory touched. Every chart is redrawn from the records on each run, so
    # anything already here is from a previous one. Left in place they
    # accumulate: narrowing a report to one corpus with --datasets leaves the
    # other corpus's charts sitting beside the new ones, named after a dataset
    # the report no longer covers. Nothing references them, which is exactly
    # what makes them dangerous -- a stale file that nothing points at is one
    # somebody opens by hand and believes.
    os.makedirs(chart_dir, exist_ok=True)
    removed = 0
    for name in os.listdir(chart_dir):
        if name.endswith((".png", ".svg")):
            try:
                os.remove(os.path.join(chart_dir, name))
                removed += 1
            except OSError:
                pass
    if removed:
        print(f"[report] cleared {removed} chart(s) from the previous report")

    memory = loaders.load_memory_series(run_dir)

    # Fill in peak memory from the timeseries where the harness could not read
    # a kernel high-water mark itself.
    peaks = {name: loaders.peak_from_series(rows) for name, rows in memory.items()}
    attach_peak_rss(records, peaks)

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
            # PR 2 §2.3.4 calls the calibration chart optional. This is what
            # makes it so, and through the existing mechanism rather than a
            # second one: `ef_mode: fixed` runs no sweep and writes no
            # `ef_calibration` records, so the chart is skipped.
            #
            # Two caveats on the shared skip line below, both of which are
            # about this entry and not about the mechanism. It says "not in
            # this profile's workloads", which is wrong for this one: the
            # cause is `ops.ef_mode`, not `ops.workloads`. And it is §6 that
            # explains the absence to a reader, not the appendix -- there IS
            # no appendix on a fixed-ef run, which `render_markdown` enforces
            # and `test_there_is_no_appendix_without_a_chart` pins. Fixing the
            # console wording means giving this entry its own skip branch; it
            # is a log line, the report itself is unambiguous, and that branch
            # belongs with whatever next needs a second reason for a skip.
            "calibration": "ef_calibration",
        }

        # The pass comparison exists to put normalized next to tuned. With one
        # pass it draws pale bars against nothing, and its per-workload panels
        # are blank for whatever the profile did not run -- four panels, two of
        # them empty, on a chart that could not say anything either way.
        passes_present = {r.get("resource_pass") for r in records
                          if r.get("dataset") == dataset and r.get("resource_pass")}

        # One filtered chart per column, or -- when there are no filtered
        # records at all -- the single `filtered` entry this always had, so
        # that `needs` below still matches it and still prints the shared
        # "not in this profile's workloads" skip line. Without the fallback a
        # profile that runs no filtered workload would generate no chart and
        # say nothing about why, which is the one case the log line exists
        # for.
        filtered_charts = filtered_chart_entries(records, dataset)

        for name, fn, payload in (
            ("pareto", charts.pareto, (dict(by_engine), dataset, k)),
            # 0.95, not 0.85. With the main chart on a logit axis the whole
            # range is already legible, so the companion's job is the region
            # people actually deploy at -- and at 0.85 every engine's curve was
            # still mostly below the floor anyone would ship.
            ("paretozoom", lambda *a, **kw: charts.pareto(*a, recall_floor=0.95, **kw),
             (dict(by_engine), dataset, k)),
            ("qpsatrecall", charts.qps_at_recall, (summary, dataset)),
            ("latency", charts.latency_percentiles, (records, dataset)),
            ("build", charts.build_cost, (records, dataset)),
            ("storage", charts.storage_breakdown, (records, dataset)),
            ("concurrency", charts.concurrency, (records, dataset)),
            # One per filter column present. PR 3's acceptance asks for "a
            # column selector (or two charts)"; two charts is the option that
            # survives being rendered to Markdown, which this report also is.
            # Derived from the records rather than from FILTER_COLUMNS: an
            # archived run carries no `filter_column` and must keep producing
            # the one chart it always produced, and a profile narrowed to
            # `filter_columns: [tag]` must not emit an empty `ctag` panel.
            *filtered_charts,
            ("churn", charts.churn, (records, dataset)),
            ("churnimpact", charts.churn_impact, (records, dataset)),
            ("passcompare", charts.pass_comparison, (records, dataset)),
            # The whole record list, not the calibration records alone: the
            # chart rings the chosen point on each curve, and which point that
            # is lives on the downstream records (see
            # `loaders.operating_points`). Handed only the sweep's own records
            # it would draw twelve points per engine and mark none of them.
            ("calibration", charts.calibration_curves, (records, dataset)),
        ):
            if name == "passcompare" and len(passes_present) < 2:
                print(f"[report] skipping passcompare: only the "
                      f"{', '.join(sorted(passes_present)) or 'one'} pass ran "
                      f"(needs normalized and tuned)")
                continue
            # Split on "-" so `filtered-tag` and `filtered-ctag` find the
            # `filtered` entry. No other chart name contains a hyphen, and a
            # future one that does would want the same prefix behaviour.
            required = needs.get(name) or needs.get(name.split("-")[0])
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
        # One chart per phase. The ann phase loads a corpus and sweeps a grid;
        # the ops phase loads it again and runs four workloads. Different
        # durations, different questions, and twelve series on one axis was
        # crowded before any of them held two measurements.
        for stem, phase in (("memory-timeline", None),
                            ("memory-timeline-ann", "ann")):
            paths = charts.memory_timeline(memory, chart_dir, stem, phase=phase)
            if paths:
                chart_paths[stem] = paths

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
