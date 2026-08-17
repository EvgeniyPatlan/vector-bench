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
