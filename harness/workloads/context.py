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
            "engine": driver.name,
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
