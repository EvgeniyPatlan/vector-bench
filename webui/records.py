"""Measurement access for one run.

report/records.jsonl is the merged artifact the report generator writes and is
preferred when present. A run whose report has not been generated yet still has
its ops records, so those are read directly; recall/QPS records only exist once
`run-benchmark.sh report` has merged the ann tree.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

RECORDS_NAME = os.path.join("report", "records.jsonl")

#: Fields the UI filters and groups by.
FACET_FIELDS: Tuple[str, ...] = (
    "engine", "dataset", "phase", "resource_pass", "metric_space",
    "storage_engine", "build_mode", "m", "ef_construction", "ef_search",
    "k", "clients", "selectivity", "churn_fraction",
)

#: Numeric fields the UI can plot.
MEASURE_FIELDS: Tuple[str, ...] = (
    "recall_at_k", "qps", "latency_p50_ms", "latency_p95_ms", "latency_p99_ms",
    "build_wall_s", "build_cpu_s", "peak_rss_bytes", "index_bytes",
    "ingest_rows_per_s", "ingest_wall_s",
)


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return out


def load_records(run_dir: str) -> Tuple[List[Dict[str, Any]], str]:
    """Records for a run, with the source they came from."""
    merged = os.path.join(run_dir, RECORDS_NAME)
    if os.path.isfile(merged):
        return _read_jsonl(merged), "records.jsonl"

    from report.loaders import load_ops_records
    return load_ops_records(run_dir), "ops-*.jsonl"


def facets(records: Sequence[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """Distinct values per facet, sorted, nulls dropped."""
    collected: Dict[str, set] = {name: set() for name in FACET_FIELDS}
    for record in records:
        for name in FACET_FIELDS:
            value = record.get(name)
            if value is not None:
                collected[name].add(value)

    out: Dict[str, List[Any]] = {}
    for name, values in collected.items():
        if not values:
            continue
        try:
            out[name] = sorted(values)
        except TypeError:
            out[name] = sorted(values, key=str)
    return out


def available_measures(records: Sequence[Dict[str, Any]]) -> List[str]:
    present = {name for record in records for name in MEASURE_FIELDS
               if isinstance(record.get(name), (int, float))}
    return [name for name in MEASURE_FIELDS if name in present]


def _coerce(sample: Any, raw: str) -> Any:
    if isinstance(sample, bool):
        return raw.lower() in ("1", "true", "yes")
    if isinstance(sample, int):
        try:
            return int(raw)
        except ValueError:
            return raw
    if isinstance(sample, float):
        try:
            return float(raw)
        except ValueError:
            return raw
    return raw


def filter_records(records: Sequence[Dict[str, Any]],
                   selection: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    """Keep records matching every non-empty facet selection.

    Values arrive as strings from the query string and are coerced against the
    first non-null value seen for that field, so ?m=16 matches the integer 16.
    """
    active = {name: values for name, values in selection.items()
              if name in FACET_FIELDS and values}
    if not active:
        return list(records)

    wanted: Dict[str, set] = {}
    for name, raws in active.items():
        sample = next((r.get(name) for r in records if r.get(name) is not None), "")
        wanted[name] = {_coerce(sample, raw) for raw in raws}

    return [r for r in records
            if all(r.get(name) in values for name, values in wanted.items())]


def series(records: Sequence[Dict[str, Any]], x: str, y: str,
           group_by: Sequence[str] = ("engine",)) -> List[Dict[str, Any]]:
    """Group records into plottable [x, y] series, each sorted by x."""
    # A local accumulator: nothing the caller owns is mutated, and rebuilding the
    # dict and its lists per record made this quadratic in the record count.
    grouped: Dict[Tuple[Any, ...], List[Tuple[float, float]]] = {}
    for record in records:
        xv, yv = record.get(x), record.get(y)
        if not isinstance(xv, (int, float)) or not isinstance(yv, (int, float)):
            continue
        key = tuple(record.get(field) for field in group_by)
        grouped.setdefault(key, []).append((float(xv), float(yv)))

    out: List[Dict[str, Any]] = []
    for key, points in grouped.items():
        ordered = sorted(points)
        out.append({
            "key": " / ".join(str(part) for part in key if part is not None),
            "group": dict(zip(group_by, key)),
            "x": [p[0] for p in ordered],
            "y": [p[1] for p in ordered],
        })
    return sorted(out, key=lambda s: s["key"])
