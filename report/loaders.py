"""Load both measurement paths into the one flat record schema.

ann-benchmarks writes HDF5 result files; the ops harness writes JSON lines. The
report generator should not care about that difference, so everything is
normalized here.

Recall is computed exactly the way ann-benchmarks computes it — a distance
threshold rather than an id intersection — so numbers produced by this framework
are directly comparable to published ann-benchmarks results. Using an id
intersection would disagree whenever a dataset contains ties at the k-th
distance, which is common in quantised embeddings.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterator, List, Optional

import numpy


def knn_recall(dataset_distances: numpy.ndarray, run_distances: numpy.ndarray,
               count: int, epsilon: float = 1e-3) -> float:
    """ann-benchmarks' knn recall: fraction within the true k-th distance."""
    total = 0.0
    n = len(run_distances)
    if n == 0:
        return float("nan")
    for i in range(n):
        threshold = dataset_distances[i][count - 1] + epsilon
        actual = int(numpy.count_nonzero(numpy.asarray(run_distances[i])[:count] <= threshold))
        total += actual / count
    return total / n


def load_annb_results(annb_dir: str, datasets_dir: str,
                      run_id: str, resource_pass_default: str = "unknown"
                      ) -> List[Dict[str, Any]]:
    """Walk ann-benchmarks HDF5 results and emit recall/QPS records."""
    records: List[Dict[str, Any]] = []
    if not os.path.isdir(annb_dir):
        return records

    # Imported only once there is something to read, so an ops-only run can be
    # reported on a machine without h5py.
    import h5py

    from harness.metrics.records import PHASE_RECALL

    truth_cache: Dict[str, numpy.ndarray] = {}

    for root, _dirs, files in os.walk(annb_dir, followlinks=True):
        for filename in files:
            if not filename.endswith(".hdf5"):
                continue
            path = os.path.join(root, filename)
            try:
                with h5py.File(path, "r") as fh:
                    attrs = dict(fh.attrs)
                    dataset_name = str(attrs.get("dataset", ""))
                    count = int(attrs.get("count", 10))

                    if dataset_name not in truth_cache:
                        truth_path = os.path.join(datasets_dir, f"{dataset_name}.hdf5")
                        if not os.path.exists(truth_path):
                            print(f"[report] skipping {path}: dataset "
                                  f"{dataset_name} not available for ground truth")
                            continue
                        with h5py.File(truth_path, "r") as tf:
                            truth_cache[dataset_name] = numpy.asarray(tf["distances"])

                    truth = truth_cache[dataset_name]
                    run_distances = numpy.asarray(fh["distances"])
                    times = numpy.asarray(fh["times"]) if "times" in fh else None

                    recall = knn_recall(truth, run_distances, count)
                    best = float(attrs.get("best_search_time", float("nan")))
                    qps = 1.0 / best if best and best > 0 else float("nan")

                    record: Dict[str, Any] = {
                        "run_id": run_id,
                        "phase": PHASE_RECALL,
                        # The directory ann-benchmarks wrote the file into is
                        # named for the algorithm and is authoritative. The
                        # `engine` attribute is written by the module, which
                        # reported its dialect name -- shared between MariaDB
                        # 11.8 and 12.3 -- so 12.3 results arrived labelled
                        # "mariadb" despite sitting in a mariadb123 directory.
                        "engine": (os.path.basename(os.path.dirname(path))
                                   or str(attrs.get("engine",
                                                    attrs.get("algo", "unknown")))),
                        "engine_version": _maybe_str(attrs.get("engine_version")),
                        "dataset": dataset_name,
                        "metric_space": str(attrs.get("distance", "")),
                        # The pass is recorded by the module when it can read
                        # VB_RESOURCE_PASS; otherwise it is recovered from the
                        # directory the results were written into.
                        "resource_pass": str(
                            attrs.get("resource_pass")
                            or _pass_from_path(path, annb_dir)
                            or resource_pass_default),
                        "storage_engine": _maybe_str(attrs.get("storage_engine")),
                        "k": count,
                        "recall_at_k": round(float(recall), 6),
                        "qps": round(float(qps), 3) if qps == qps else None,
                        "clients": 1,
                        "timestamp": _maybe_str(attrs.get("timestamp")) or "",
                        "m": _maybe_int(attrs.get("M")),
                        "ef_search": _maybe_int(attrs.get("ef_search")),
                        "ef_construction": _maybe_int(attrs.get("ef_construction")),
                        "build_mode": _maybe_str(attrs.get("build_mode")),
                        "index_bytes": _maybe_int(attrs.get("index_bytes")),
                        "march": _maybe_str(attrs.get("march")),
                        "vector_index_used": _maybe_bool(attrs.get("vector_index_used")),
                        "extra": {
                            "parameters": _maybe_str(attrs.get("name")),
                            "batch_mode": bool(attrs.get("batch_mode", False)),
                            "run_count": _maybe_int(attrs.get("run_count")),
                            "source_file": os.path.relpath(path, annb_dir),
                            # Used to detect a result file that predates the
                            # run it is being reported under. ann-benchmarks
                            # skips configurations that already have results,
                            # so an unchanged tree silently reappears in a new
                            # report.
                            "source_mtime": os.path.getmtime(path),
                        },
                    }
                    if attrs.get("index_size") is not None:
                        # ann-benchmarks records index size in kilobytes.
                        record.setdefault("index_bytes",
                                          int(float(attrs["index_size"]) * 1024))
                    if times is not None and len(times):
                        record["latency_p50_ms"] = float(numpy.percentile(times, 50) * 1000)
                        record["latency_p95_ms"] = float(numpy.percentile(times, 95) * 1000)
                        record["latency_p99_ms"] = float(numpy.percentile(times, 99) * 1000)
                        record["queries_executed"] = int(len(times))
                    records.append(record)
            except Exception as exc:  # noqa: BLE001 - one bad file must not stop the report
                print(f"[report] could not read {path}: {exc}")

    return records


def _pass_from_path(path: str, annb_dir: str) -> Optional[str]:
    """Recover the resource pass from the results tree layout."""
    try:
        relative = os.path.relpath(path, annb_dir)
    except ValueError:
        return None
    head = relative.split(os.sep)[0]
    return head if head in ("normalized", "tuned") else None


def load_ops_records(run_dir: str) -> List[Dict[str, Any]]:
    """Read every ops-harness JSONL file in a run directory."""
    from harness.metrics.records import read_records

    out: List[Dict[str, Any]] = []
    if not os.path.isdir(run_dir):
        return out
    for filename in sorted(os.listdir(run_dir)):
        if not (filename.startswith("ops-") and filename.endswith(".jsonl")):
            continue
        try:
            # Materialised: read_records is a generator, and the relabelling
            # below walks it before out.extend does.
            records = list(read_records(os.path.join(run_dir, filename)))
        except Exception as exc:  # noqa: BLE001
            print(f"[report] could not read {filename}: {exc}")
            continue
        # The orchestrator writes ops-<engine>-<dataset>-<pass>-<tag>.jsonl, so
        # the filename is authoritative about which engine produced the file.
        # The records themselves were not: the harness recorded the driver's
        # own name, and mariadb123 shares MariaDBDriver, so every 12.3
        # measurement arrived labelled "mariadb". Trusting the filename repairs
        # runs recorded before that was fixed, without re-measuring them.
        engine = _engine_from_ops_filename(filename)
        if engine:
            for r in records:
                r["engine"] = engine
        out.extend(records)
    return out


def _engine_from_ops_filename(filename: str) -> Optional[str]:
    """Engine name out of ops-<engine>-<dataset>-<pass>-<tag>.jsonl.

    Datasets contain hyphens, so match the longest known engine prefix rather
    than splitting on the separator.
    """
    stem = filename[len("ops-"):-len(".jsonl")]
    known = ("mariadb123", "mariadb", "alisql", "pgvector")
    for name in known:
        if stem.startswith(name + "-"):
            return name
    return None


def load_memory_series(run_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    """Read the per-container memory timeseries files.

    Keyed by the filename stem so a phase's peak can be derived by intersecting
    with that phase's time window, rather than trusting a single scalar that may
    have been sampled at the wrong moment.
    """
    import json

    series: Dict[str, List[Dict[str, Any]]] = {}
    if not os.path.isdir(run_dir):
        return series
    for filename in sorted(os.listdir(run_dir)):
        if not (filename.startswith("mem-") and filename.endswith(".jsonl")):
            continue
        rows: List[Dict[str, Any]] = []
        with open(os.path.join(run_dir, filename)) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    break  # truncated tail from an interrupted run
        if rows:
            series[filename[4:-6]] = rows
    return series


def peak_from_series(rows: List[Dict[str, Any]]) -> Optional[int]:
    values = [r.get("rss_bytes") for r in rows if r.get("rss_bytes")]
    return max(values) if values else None


def memory_stem(record: Dict[str, Any]) -> str:
    """Name of the memory series belonging to one ops unit.

    Mirrors the stem `orchestrator/cli.py` gives the `mem-*.jsonl` file it
    writes, including the rule that InnoDB stays unsuffixed so checkpoints from
    runs that predate the storage-engine sweep are still honoured. Kept here
    rather than imported from the orchestrator because this package runs inside
    a bench image that mounts only report/ and harness/.

    Matching on a prefix instead matched both storage engines and took
    whichever the dict happened to yield first, which published MyISAM's peak
    memory as InnoDB's.
    """
    stem = (f"{record.get('engine')}-{record.get('dataset')}"
            f"-{record.get('resource_pass')}-m{record.get('m')}"
            f"-{record.get('build_mode')}")
    storage = (record.get("storage_engine") or "").lower()
    if storage and storage != "innodb":
        stem += f"-{storage}"
    return stem


# Within this of the container limit counts as "at the ceiling". cgroup peak
# accounting cannot exceed memory.max, so an engine under sustained pressure
# reports a value just below the limit rather than above it, and an exact
# comparison would never fire.
_CEILING_TOLERANCE = 0.98


def ceiling_pressure(rows: List[Dict[str, Any]],
                     limit_bytes: Optional[int]) -> Optional[Dict[str, Any]]:
    """Detect a phase that spent significant time against its memory limit.

    An engine pinned at its cgroup limit is reclaiming continuously, so its
    throughput and latency describe the budget rather than the implementation.
    That is invisible in the records themselves: the phase either succeeds with
    depressed numbers or is OOM-killed and shows up only as a non-zero exit.

    Returns None when there is nothing to report, so callers can filter falsy.
    """
    if not limit_bytes:
        return None
    values = [r.get("rss_bytes") for r in rows if r.get("rss_bytes")]
    if not values:
        return None
    threshold = limit_bytes * _CEILING_TOLERANCE
    at_ceiling = sum(1 for v in values if v >= threshold)
    if not at_ceiling:
        return None
    first = next(i for i, v in enumerate(values) if v >= threshold)
    return {
        "peak_bytes": max(values),
        "limit_bytes": limit_bytes,
        "samples": len(values),
        "samples_at_ceiling": at_ceiling,
        "fraction_at_ceiling": at_ceiling / len(values),
        "first_hit_fraction": first / len(values),
    }


def _maybe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _maybe_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def _maybe_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, (bool, numpy.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("true", "1"):
        return True
    if text in ("false", "0"):
        return False
    return None


def pareto_frontier(points: List[Dict[str, Any]], x: str = "recall_at_k",
                    y: str = "qps") -> List[Dict[str, Any]]:
    """Upper-left envelope: the points not beaten on both recall and QPS.

    Comparing raw scatter points is misleading — an engine with a badly chosen
    parameter looks slow at a recall where a different parameter of the same
    engine is fast. The frontier is what an operator could actually achieve.
    """
    usable = [p for p in points if p.get(x) is not None and p.get(y) is not None]
    if not usable:
        return []
    ordered = sorted(usable, key=lambda p: (p[x], p[y]))
    frontier: List[Dict[str, Any]] = []
    best_qps = float("-inf")
    for point in reversed(ordered):        # highest recall first
        if point[y] > best_qps:
            frontier.append(point)
            best_qps = point[y]
    return list(reversed(frontier))

# A phase is "contended" when the machine burned meaningfully more CPU than the
# container we were measuring did. Some gap is always there -- the kernel, the
# sampler itself, the other cpuset -- so the threshold is a share of a core,
# not zero.
FOREIGN_CPU_CORES = 2.0


def foreign_cpu(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """CPU the host spent that the measured container did not.

    Three AliSQL recall points were taken while something outside the harness
    was on the machine. Nothing recorded it, so it surfaced five days later as
    a latency ratio somebody happened to compare by hand. Both counters are in
    every sample now, and the difference between them over the phase is the
    answer without anybody comparing anything.

    Returns None when the series cannot answer -- an older run without the
    host fields, or too few samples to difference.
    """
    usable = [r for r in rows
              if r.get("host_cpu_seconds") is not None
              and r.get("cpu_seconds") is not None]
    if len(usable) < 2:
        return None
    first, last = usable[0], usable[-1]
    elapsed = (last.get("t") or 0) - (first.get("t") or 0)
    if elapsed <= 0:
        return None
    host = last["host_cpu_seconds"] - first["host_cpu_seconds"]
    ours = last["cpu_seconds"] - first["cpu_seconds"]
    foreign = host - ours
    cores = foreign / elapsed
    if cores < FOREIGN_CPU_CORES:
        return None
    loads = [r["host_load1"] for r in usable if r.get("host_load1") is not None]
    return {
        "elapsed_s": round(elapsed, 1),
        "container_cores": round(ours / elapsed, 2),
        "foreign_cores": round(cores, 2),
        "peak_load1": round(max(loads), 1) if loads else None,
    }

def build_signature(record: Dict[str, Any]) -> Any:
    """Everything about a measurement except the query effort.

    Two points belong to the same curve when they differ only in ef_search.
    Deciding that from named columns needs the key to list every axis a
    profile may sweep, and it silently fails the moment an engine gains one:
    Percona Search's quantization is not a column, so its unquantized and
    scalar curves collapsed into a single series -- which made eight legitimate
    measurements look like duplicates and invented six inversions comparing
    ef_search 10 against ef_search 10.

    ann-benchmarks already solves this. It names each result file after the
    build arguments with the query argument appended:

        angular_M_16_quantization_none_400.hdf5
        angular_M_16_build_mode_post_efConstruction_200_400.hdf5

    Dropping the trailing query argument leaves the build configuration, for
    any engine, including axes this code has never heard of. The named columns
    stay as the fallback for records loaded without a source file.
    """
    source = (record.get("extra") or {}).get("source_file")
    if source:
        stem = os.path.basename(str(source))
        if stem.endswith(".hdf5"):
            stem = stem[:-len(".hdf5")]
        head, _, tail = stem.rpartition("_")
        # Only strip a trailing query argument, never part of the build name.
        # A stem that does not end in one is still a configuration identifier
        # and is kept whole: falling back to the columns here would discard
        # the very information the filename was consulted for.
        build = head if (head and tail.isdigit()) else stem
        return ("file", record.get("engine"), record.get("dataset"), build)
    return ("cols", record.get("engine"), record.get("dataset"), record.get("m"),
            record.get("build_mode"), record.get("storage_engine"),
            record.get("ef_construction"),
            (record.get("extra") or {}).get("quantization"))

