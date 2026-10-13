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

# Bump this in any change that alters what an ops number *means* — the shape of
# the record is `SCHEMA_VERSION`'s business, this is the measurement's. A
# concurrency figure measured with a per-client warmup is not the same quantity
# as one measured without; until this constant existed nothing in the record
# said which it was, so two such figures were indistinguishable and the report
# had no way to keep them off one series.
#
# It is the ops counterpart of ANN_MEASUREMENT_VERSION
# (`orchestrator/ann_pass.py`), which does the same job for the recall/QPS path
# by feeding the fingerprint that keeps a 16 GB curve out of a 64 GB report.
# The two are separate numbers on purpose: a fix to the churn workload must not
# invalidate every recall curve ever measured.
#
# WHY IT IS NOT BESIDE ITS ANN COUNTERPART, which is the obvious place and the
# first thing a reader will want to "fix": ops records are written from inside
# the harness container, which bind-mounts `harness/` and nothing else.
# `harness/` importing `orchestrator` raises ModuleNotFoundError seconds after
# the server comes up and kills the run — it has happened, and
# tests/test_config.py::TestHarnessIsolation exists to stop it happening again.
# So the constant the ops writer needs lives on the harness side of that
# boundary, and `orchestrator/manifest.py` imports it from here. That is the one
# direction this dependency is allowed to point.
#
# Version 1 is implicit. Records written before this constant existed carry no
# `extra.ops_measurement_version` at all, so every reader must treat an absent
# value as 1 — see IMPLICIT_OPS_MEASUREMENT_VERSION — rather than as "unknown"
# and therefore comparable with anything.
#
# VERSION 3 separated the concurrency query pool from `ops.max_queries`. That
# key caps the filtered and churn workloads, which answer each query once, and
# main.yml holds it at 500 to bound their cost; the concurrency workload
# cycles its shard for a fixed duration instead, so it was giving 32 clients
# about fifteen queries each for thirty seconds. Every graph cache in the
# comparison — MHNSW's per-TABLE_SHARE cache, VIDX's shared cache, pgvector's
# shared_buffers — was being measured against a hot set small enough to hold
# entirely, which is the one thing the concurrency chart exists to tell apart.
# The pool now defaults to the whole test set, so a version 2 concurrency QPS
# and a version 3 one are throughput over a 500-query and a 10,000-query
# working set: the same workload over two different amounts of the corpus, and
# not the same quantity. Hence a bump, and hence 3 is deliberately absent from
# OPS_MEASUREMENT_EQUIVALENT_VERSIONS below.
#
# The recall that version 3 also added to concurrency records is new
# information rather than a changed number and would not on its own have
# earned a bump — a field that was None is now populated, and nothing a
# version 2 reader already trusted moves. The pool is what earned it.
#
# Version 3 ALSO counts only the queries that finish inside each concurrency
# window, which sits inside this version rather than earning a fourth one
# because no run has ever been measured at version 3 — the value was
# introduced and then extended inside one change set, so there is no version 3
# figure anywhere for a trimmed one to be incomparable with.
#
# VERSION 4 chose ef_search instead of declaring it. The ops pass now
# calibrates: it sweeps a grid on the index it has just built, picks the
# smallest ef_search reaching a target recall (0.95 by default), and runs
# concurrency, filtered and churn at that value instead of at the profile's
# `ops.ef_search`. A version 3 concurrency QPS is throughput at a declared ef
# — 100 for every engine — and a version 4 one is throughput at whatever ef
# puts THAT engine at the target. Two engines at ef_search 100 sit at recall
# 0.98 and 0.80, which is the ordinary spread of these implementations, so the
# version 3 figure and the version 4 figure are not two samples of one
# quantity: one holds the knob fixed and lets quality vary, the other holds
# quality fixed and lets the knob vary. That is the whole point of PR 2 and it
# is exactly what this constant exists to keep off one series.
#
# WHY A BUMP AND NOT JUST extra.ef_mode, which the records also now carry and
# which is the obvious reason to think one is unnecessary: this constant is
# what makes the REPORT REFUSE TO MIX, not what lets a reader tell two numbers
# apart by hand. Two concurrency records both stamped version 3, one `fixed` at
# ef_search 100 and one `calibrated` at 98, load together with no refusal and
# land on one chart — the comparison PR 2 exists to prevent. `extra.ef_mode`
# is the right diagnostic for which policy produced a number; it is not a
# guard, and `report/measurement_versions.py` reads this.
#
# It covers `--ef-mode fixed` too, which measures what version 3 measured. The
# version names the harness that wrote a record, not the flags it was given:
# per-run is the granularity the whole scheme is built on (see the per-phase
# argument below), and a constant that varied with a flag would have to be
# computed per record and could no longer be stamped from one place. A fixed
# run under version 4 is therefore labelled conservatively — reported as a
# different measurement from a version 3 run that it in fact matches. That is
# the safe direction of the two: this set is about refusing to merge, and the
# cost of an unnecessary refusal is one report per version, while the cost of a
# missing one is a published chart that is wrong.
#
# The cost this scheme knowingly pays: the bump invalidates comparison across
# every ops phase, including build cost and churn, which this change does not
# touch. The version is per-run, not per-phase, because a record names the
# harness that wrote it and splitting it per phase would mean four constants
# and four ways to forget one.
OPS_MEASUREMENT_VERSION = 4

#: Key under `Record.extra` that `Recorder.write` stamps the version into.
#: Lives in `extra` rather than as a column because SCHEMA_VERSION stays 1.
OPS_MEASUREMENT_VERSION_KEY = "ops_measurement_version"

#: What an ops record with no recorded version was measured by.
IMPLICIT_OPS_MEASUREMENT_VERSION = 1

#: The first version that was ever written into a record. It changed nothing
#: measured -- it exists because version 1 was never written down, and a
#: version nobody records cannot be compared against a later one.
#:
#: Distinct from OPS_MEASUREMENT_VERSION, which is whatever the harness is
#: stamping today and moves on every bump. This one never moves: it names a
#: historical fact about version 2, not the current state of the code.
FIRST_RECORDED_OPS_MEASUREMENT_VERSION = 2

# Versions that denote the SAME measurement, despite being different numbers.
# A version 1 number and a version 2 number are the same quantity, so a run
# holding both needs a report per version, not a re-measurement.
# `docs/05-methodology.md` says exactly this and `tests/test_config.py` pins
# the two together -- the refusal text used to tell operators to re-measure
# data the docs call comparable, on the only version pair that occurs today.
#
# DERIVED from the two constants above rather than written as a literal, and
# that is the whole safety property. A set spelled `frozenset({1, 2})` is an
# extension point whose only lock is a comment asking people not to use it:
# `{1, 2, 4}` would be a one-character edit asserting that version 4 measures
# what version 1 did, and a subset-style test cannot see a set growing. Built
# this way, a third member cannot be added without first naming a third
# concept and saying in its own docstring why that version changed nothing --
# which is the deliberateness the comment used to merely request.
#
# So: a bump that changes what a number means -- the per-client warmup, a
# different filtered predicate -- simply cannot appear here, and the refusal
# gives every pair this set does not cover the "two different measurements"
# wording.
OPS_MEASUREMENT_EQUIVALENT_VERSIONS = frozenset({
    IMPLICIT_OPS_MEASUREMENT_VERSION,
    FIRST_RECORDED_OPS_MEASUREMENT_VERSION,
})


def ops_versions_measure_the_same_thing(versions: Iterable[Any]) -> bool:
    """Whether every version given denotes the same measurement.

    False for anything unrecognised, including a malformed value, so the
    general "these are different measurements" wording is what an unknown
    version gets. Saying two numbers are comparable is a claim; saying they
    may not be is the safe default.
    """
    found = list(versions)
    if not found:
        return False
    for version in found:
        # A hand-edited manifest can hold a list or a dict here, and `in`
        # against a frozenset hashes what it is given.
        if not isinstance(version, int) or isinstance(version, bool):
            return False
        if version not in OPS_MEASUREMENT_EQUIVALENT_VERSIONS:
            return False
    return True

# Phases a record can belong to. Used by the report generator to route records
# to the right chart without inspecting which fields happen to be populated.
PHASE_RECALL = "recall_qps"       # ann-benchmarks Pareto point
PHASE_INGEST = "ingest"           # rows loaded, no index or index maintained
PHASE_BUILD = "index_build"       # index construction cost
PHASE_CONCURRENCY = "concurrency"  # multi-client QPS / latency
PHASE_FILTERED = "filtered"       # vector search with a scalar predicate
PHASE_CHURN = "churn"             # recall after update/delete cycles
# The ef_search calibration sweep: recall and QPS at each grid point, written
# so the choice of ef is auditable. Its own phase and not PHASE_RECALL, because
# these are not Pareto points: they are single-client probes against the index
# the ops pass just built, and a chart that drew them as recall/QPS results
# would put twelve points per engine on a curve measured with a different
# method, a different container and a different query count.
PHASE_CALIBRATION = "ef_calibration"

ALL_PHASES = (
    PHASE_RECALL, PHASE_INGEST, PHASE_BUILD,
    PHASE_CALIBRATION, PHASE_CONCURRENCY, PHASE_FILTERED, PHASE_CHURN,
)

#: Keys under `extra` that name the ef policy a record was measured under.
#: Stamped by `Recorder`, not written by the workloads -- see
#: `stamp_run_extra`.
EF_MODE_KEY = "ef_mode"
TARGET_RECALL_KEY = "target_recall"
CALIBRATION_STATUS_KEY = "calibration_status"


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


def stamp_measurement_version(payload: Dict[str, Any]) -> Dict[str, Any]:
    """A copy of a record dict carrying the version that measured it.

    Applied in `Recorder.write`, not in `RunContext.record_defaults`. The
    defaults dict is the obvious home — it is the choke point every workload's
    provenance already travels through, and it exists precisely so a new
    workload cannot drop a field. But four of the workloads build their record
    as `Record(**common, ..., extra={...})`, so an `extra` key in `common`
    raises `TypeError: got multiple values for keyword argument 'extra'`.
    Merging the two would mean editing every workload, and then a fifth
    workload added later would still be free to forget.

    `Recorder.write` is the only point every ops record provably passes
    through, which is why the ALL_PHASES check lives there too. Stamping here
    makes "every ops record carries the version" true by construction rather
    than by convention.

    Takes and returns a plain dict, and copies both it and its `extra` rather
    than writing into either. Note what that does and does not protect: the
    `Record` the caller still holds is protected by `to_dict()`, which goes
    through `dataclasses.asdict` and therefore deep-copies `extra` before this
    function ever sees it — not by anything here. What the copy here buys is
    that this function is safe for any caller, including a future `to_dict`
    that stops deep-copying to save the allocation. `TestOpsMeasurementVersion`
    guards both halves separately, because an earlier version of that test
    asserted the record was untouched and could not fail: asdict was doing the
    work and a stamp written straight into `payload` passed it.
    """
    extra = dict(payload.get("extra") or {})
    # Unconditional, not setdefault. A record reaching this Recorder was
    # produced by this code, so this code's version is the honest answer; a
    # caller-supplied value would be a claim about someone else's measurement,
    # and a merge tool replaying old records through a new Recorder would
    # relabel them as new ones if it were allowed to carry its own version in.
    extra[OPS_MEASUREMENT_VERSION_KEY] = OPS_MEASUREMENT_VERSION
    return {**payload, "extra": extra}


def stamp_run_extra(payload: Dict[str, Any],
                    run_extra: Dict[str, Any]) -> Dict[str, Any]:
    """A copy of a record dict carrying facts that are true of the whole run.

    The second stamp applied in `Recorder.write`, and it exists for exactly the
    reason `stamp_measurement_version` does: `RunContext.record_defaults` is
    the obvious home for run-wide provenance and cannot carry an `extra` key,
    because four workloads splat that dict into `Record(**common, ...,
    extra={...})` and a duplicate keyword argument is a TypeError.

    What travels this way is the ef policy -- `ef_mode`, `target_recall`,
    `calibration_status`. A concurrency figure measured at a calibrated
    ef_search and one measured at the profile's literal are not the same
    quantity, and the record has to say which it is without the reader having
    to find the calibration records that preceded it. Stamping here makes that
    true for every workload, including the three this change does not touch.

    Unconditional, like the version stamp and for the same reason: a record
    reaching this Recorder was produced by this run, so this run's policy is
    the honest answer. Keys whose value is None are left out rather than
    written as nulls -- in `fixed` mode no target was pursued and no
    calibration ran, and a record carrying `target_recall: null` beside an ef
    nobody chose would be a claim about a calibration that never happened.
    `ops.pgvector_iterative_scan` spent 69 records saying null; absence is the
    honest encoding of "there was none".
    """
    if not run_extra:
        return payload
    extra = dict(payload.get("extra") or {})
    extra.update({k: v for k, v in run_extra.items() if v is not None})
    return {**payload, "extra": extra}


class Recorder:
    """Append-only JSON-lines writer, safe for use from multiple threads."""

    def __init__(self, path: str, run_extra: Optional[Dict[str, Any]] = None):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._lock = threading.Lock()
        self._fh = open(path, "a", buffering=1, encoding="utf-8")
        self._run_extra: Dict[str, Any] = dict(run_extra or {})

    def declare_run_extra(self, **values: Any) -> None:
        """Add facts every record written from now on should carry.

        Deliberately not a constructor-only argument: `calibration_status` is
        not known until the calibration sweep has finished, and the sweep's own
        records are written before it exists. A per-point record therefore
        carries no status, which is correct -- at the time it was written there
        was not one.
        """
        with self._lock:
            self._run_extra.update(values)

    def write(self, record: Record) -> None:
        if record.phase not in ALL_PHASES:
            raise ValueError(f"unknown phase: {record.phase}")
        # After to_dict(), which drops an empty `extra`, so the stamps survive
        # that filter instead of being removed by it.
        with self._lock:
            run_extra = dict(self._run_extra)
        line = json.dumps(
            stamp_run_extra(stamp_measurement_version(record.to_dict()),
                            run_extra),
            sort_keys=True)
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
