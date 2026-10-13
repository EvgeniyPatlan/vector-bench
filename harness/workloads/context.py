"""Shared run context for the ops-harness workloads."""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..datasets import Dataset
from ..drivers.base import EngineDriver, IndexSpec
from ..metrics.records import Recorder


def utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class RunContext:
    run_id: str
    resource_pass: str          # "normalized" | "tuned"
    recorder: Recorder
    # The engine name the orchestrator asked for, which is not always the
    # driver's own name: mariadb and mariadb123 are the same server at
    # different tags and share MariaDBDriver. Recording driver.name labelled
    # every 12.3 measurement as "mariadb", so 12.3 vanished from the report and
    # 11.8 appeared to have measured everything twice.
    engine: Optional[str] = None
    engine_tag: Optional[str] = None
    march: Optional[str] = None
    k: int = 10
    cache_dir: str = "/results/.cache"

    def record_defaults(self, driver: EngineDriver, dataset: Dataset,
                        index: IndexSpec) -> Dict[str, Any]:
        """Fields every record from this run shares.

        Centralised so a new workload cannot accidentally omit the provenance
        that makes a number interpretable.

        One piece of provenance is deliberately *not* here, and a reader
        looking for it will look here first: `extra.ops_measurement_version`.
        It cannot be, because this dict is splatted into `Record(**common,
        ...)` and build, concurrency, filtered and churn each pass `extra=`
        explicitly alongside it — so an `extra` key here is a duplicate
        keyword argument. The failure is worse than it sounds for being
        partial: build.py's ingest record passes no `extra` and writes fine,
        and the index_build record immediately after it raises TypeError, so
        the run dies one record into the first phase with a file that already
        looks like it is working.

        It is stamped in `Recorder.write` instead
        (`harness/metrics/records.py`), which is the only point every ops
        record provably passes through and is already where the ALL_PHASES
        check lives.
        """
        return {
            "run_id": self.run_id,
            "engine": self.engine or driver.name,
            "engine_version": driver.server_version(),
            "engine_tag": self.engine_tag,
            "resource_pass": self.resource_pass,
            "dataset": dataset.name,
            "metric_space": dataset.metric,
            "storage_engine": index.storage_engine,
            "march": self.march,
            "m": index.m,
            "ef_construction": index.ef_construction,
            "build_mode": index.build_mode,
            "k": self.k,
            "timestamp": utcnow(),
        }
