"""The single flat record schema every measurement lands in.

Both measurement paths — ann-benchmarks (recall/QPS) and this ops harness
(build cost, concurrency, filtered search, churn) — emit the same record shape.
Keeping one flat schema means the report generator has no per-source special
cases and the raw data can be queried with anything that reads JSON lines.

Unused fields stay None. A record is never partially written: `Recorder` writes
one complete JSON object per line and flushes, so a killed run leaves valid
data for everything that completed.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional

SCHEMA_VERSION = 1

# Phases a record can belong to. Used by the report generator to route records
# to the right chart without inspecting which fields happen to be populated.
PHASE_RECALL = "recall_qps"       # ann-benchmarks Pareto point
PHASE_INGEST = "ingest"           # rows loaded, no index or index maintained
PHASE_BUILD = "index_build"       # index construction cost
PHASE_CONCURRENCY = "concurrency"  # multi-client QPS / latency
PHASE_FILTERED = "filtered"       # vector search with a scalar predicate
PHASE_CHURN = "churn"             # recall after update/delete cycles

ALL_PHASES = (
    PHASE_RECALL, PHASE_INGEST, PHASE_BUILD,
    PHASE_CONCURRENCY, PHASE_FILTERED, PHASE_CHURN,
)


@dataclass
class Record:
    # --- identity -----------------------------------------------------
    run_id: str
    engine: str
    phase: str
    dataset: str
    timestamp: str

    engine_version: Optional[str] = None
    engine_tag: Optional[str] = None
    resource_pass: Optional[str] = None      # "normalized" | "tuned"
    storage_engine: Optional[str] = None
    metric_space: Optional[str] = None       # "angular" | "euclidean"
    march: Optional[str] = None

    # --- index configuration -----------------------------------------
    m: Optional[int] = None
    ef_construction: Optional[int] = None
    ef_search: Optional[int] = None
    build_mode: Optional[str] = None         # pgvector: "post" | "incremental"
    k: Optional[int] = None

    # --- quality ------------------------------------------------------
    recall_at_k: Optional[float] = None
    vector_index_used: Optional[bool] = None

    # --- throughput / latency ----------------------------------------
    clients: Optional[int] = None
    qps: Optional[float] = None
    latency_mean_ms: Optional[float] = None
    latency_p50_ms: Optional[float] = None
    latency_p95_ms: Optional[float] = None
    latency_p99_ms: Optional[float] = None
    queries_executed: Optional[int] = None

    # --- build cost ---------------------------------------------------
    ingest_rows_per_s: Optional[float] = None
    ingest_wall_s: Optional[float] = None
    ingest_threads: Optional[int] = None
    build_wall_s: Optional[float] = None
    build_cpu_s: Optional[float] = None
    peak_rss_bytes: Optional[int] = None
    index_bytes: Optional[int] = None
    table_bytes: Optional[int] = None
    rows: Optional[int] = None

    # --- workload shape ----------------------------------------------
    selectivity: Optional[float] = None      # fraction of rows passing the filter
    churn_fraction: Optional[float] = None

    # --- escape hatch -------------------------------------------------
    notes: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None and v != {}}


class Recorder:
    """Append-only JSON-lines writer, safe for use from multiple threads."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._lock = threading.Lock()
        self._fh = open(path, "a", buffering=1, encoding="utf-8")

    def write(self, record: Record) -> None:
        if record.phase not in ALL_PHASES:
            raise ValueError(f"unknown phase: {record.phase}")
        line = json.dumps(record.to_dict(), sort_keys=True)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()
            os.fsync(self._fh.fileno())

    def write_many(self, records: Iterable[Record]) -> None:
        for record in records:
            self.write(record)

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.close()

    def __enter__(self) -> "Recorder":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_records(path: str) -> Iterator[Dict[str, Any]]:
    """Read a JSONL file, skipping any truncated trailing line.

    A run killed mid-write can leave one partial line. Everything before it is
    still valid data and must remain readable.
    """
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # Only tolerate damage at the very end of the file.
                remainder = fh.read().strip()
                if remainder:
                    raise ValueError(
                        f"{path}: malformed JSON at line {lineno} with data following it"
                    )
                return


def load_all(paths: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for path in paths:
        if os.path.exists(path):
            out.extend(read_records(path))
    return out
