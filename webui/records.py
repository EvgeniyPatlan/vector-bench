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
#:
#: Every entry must be a field something in this repo actually writes, because
#: `available_measures` below is the ceiling the Explore tab's axis pickers are
#: filtered against: a name here that no writer ever populates becomes a
#: dropdown option that resolves to an empty chart, which reads as "this run
#: measured nothing" rather than as "nothing measures this".
#:
#: `build_cpu_s` was exactly that and is why this note exists. It is declared
#: on `Record` in `harness/metrics/records.py` and was listed here and offered
#: by the Explore tab's index-build view, and nothing in `harness/`,
#: `report/`, `orchestrator/` or `webui/` ever assigned it. Measured on the
#: archive of completed runs kept beside this repo -- 239 `ops-*.jsonl` files
#: plus 41 merged `report/records.jsonl`, and 484 index_build records between
#: them, 239 raw and 245 merged -- it is on 0 of those 484, against 484 of 484
#: for `build_wall_s` beside it.
#:
#: The schema column stays: a `Record` field nothing writes is a question about
#: the schema, with an ops-measurement-version argument attached, and
#: `docs/01-architecture.md` documents the row shape rather than this tuple.
#: Removing it from this tuple is enough to take the dead axis away, and the
#: day something measures build CPU time it goes back in the same change.
#:
#: Note that an entry here need not be written by the harness. `peak_rss_bytes`
#: is merged in from the `mem-*.jsonl` timeseries by `report/generate.py`'s
#: `attach_peak_rss`, so it is on none of the 239 raw index_build records and
#: on 238 of the 245 merged ones -- a legitimately partial measure, which is a
#: different thing from one nothing writes.
MEASURE_FIELDS: Tuple[str, ...] = (
    "recall_at_k", "qps", "latency_p50_ms", "latency_p95_ms", "latency_p99_ms",
    "build_wall_s", "peak_rss_bytes", "index_bytes",
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
    """Records for a run, with the source they came from.

    Raises report.loaders.MixedMeasurementVersions when the run holds ops
    records from more than one measurement version. The server turns that into
    a 500 naming the fault, which is the right answer for a run that cannot
    honestly be plotted: `series()` below groups by engine and would otherwise
    draw two different measurements as one line.

    The merged file needs the check as much as the raw records do -- more, in
    fact. It is preferred here whenever it exists, which is true of almost
    every archived run, so a mixed set that reached records.jsonl would make
    this the one reader that never consults the guard at all.
    """
    from report import measurement_versions as versions
    from report.loaders import load_ops_records

    merged = os.path.join(run_dir, RECORDS_NAME)
    if os.path.isfile(merged):
        rows = _read_jsonl(merged)
        # `declared` too: the UI shows this run's manifest beside these
        # numbers, so records a version behind it are mislabelled here exactly
        # as they would be in a rendered report.
        versions.refuse_mixed_measurement_versions(
            rows, source=merged,
            declared=versions.declared_ops_version(run_dir))
        return rows, "records.jsonl"

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
