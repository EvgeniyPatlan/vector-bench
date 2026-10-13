"""Engine driver interface for the ops harness.

The ops harness measures what ann-benchmarks does not: index build cost,
concurrency scaling, filtered search and churn. Those need operations
ann-benchmarks' BaseANN has no concept of (delete rows, size an index, open N
independent connections), so they get their own interface.

A driver talks to a server that is already running in its own container. The
harness never starts or configures the server — the orchestrator does, so that
resource limits and server flags are applied identically for every engine and
are recorded in the manifest.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy

TABLE = "t1"
DATABASE = "ann"


@dataclass
class ConnectionSpec:
    host: str
    port: int
    user: str = "bench"
    password: str = "bench"
    database: str = DATABASE
    # Path where this process can read the server's data directory, when the
    # orchestrator has shared it read-only. Enables exact on-disk index sizing
    # for engines whose catalogs cannot report it.
    data_dir: Optional[str] = None


@dataclass
class IndexSpec:
    dim: int
    m: int
    metric: str                       # "angular" | "euclidean"
    ef_construction: Optional[int] = None
    storage_engine: str = "InnoDB"
    build_mode: str = "post"          # "post" | "incremental"
    # Percona Search is the only engine here that compresses indexed vectors.
    # "none" | "scalar" | "binary"; pinned off in the normalized pass so it is
    # not an axis the others lack, and set to the vendor's recommendation in
    # the tuned pass, where following each vendor's own advice is the point.
    quantization: Optional[str] = None


@dataclass
class LoadResult:
    rows: int
    wall_seconds: float
    threads: int

    @property
    def rows_per_second(self) -> float:
        return self.rows / self.wall_seconds if self.wall_seconds > 0 else 0.0


class EngineDriver(abc.ABC):
    """One connection-owning driver instance per client."""

    name: str = "unknown"
    #: True when the engine maintains its graph during INSERT and therefore has
    #: no separable bulk build step. Drives how build cost is attributed.
    incremental_index: bool = True
    #: True when the index is built by another process after the statement
    #: returns, so build cost is load time plus time-to-ready and the two
    #: overlap. Neither value of incremental_index describes that.
    async_index_build: bool = False

    def __init__(self, spec: ConnectionSpec):
        self.spec = spec
        self._conn: Any = None

    # -- lifecycle ------------------------------------------------------

    @abc.abstractmethod
    def connect(self) -> None:
        """Open a connection and apply session setup."""

    @abc.abstractmethod
    def close(self) -> None: ...

    def __enter__(self) -> "EngineDriver":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- schema ---------------------------------------------------------

    @abc.abstractmethod
    def drop_schema(self) -> None: ...

    @abc.abstractmethod
    def create_schema(self, index: IndexSpec) -> None:
        """Create the table, and the index too if the engine builds it inline."""

    @abc.abstractmethod
    def create_index(self, index: IndexSpec) -> None:
        """Create the vector index. No-op where create_schema already did it."""

    # -- data -----------------------------------------------------------

    @abc.abstractmethod
    def load(self, vectors: numpy.ndarray, tags: numpy.ndarray,
             threads: int = 1, start_id: int = 0) -> LoadResult: ...

    @abc.abstractmethod
    def delete_ids(self, ids: Sequence[int]) -> None: ...

    @abc.abstractmethod
    def insert_rows(self, ids: Sequence[int], vectors: numpy.ndarray,
                    tags: Sequence[int]) -> None: ...

    @abc.abstractmethod
    def count_rows(self) -> int: ...

    # -- query ----------------------------------------------------------

    @abc.abstractmethod
    def set_ef_search(self, ef_search: int) -> None: ...

    def effective_ef(self, ef: int, k: int) -> int:
        """How many candidates this engine will really search for `ef`.

        For every engine here except one the answer is `ef`, which is why this
        is a concrete method rather than another abstract one. MongoDB clamps:
        `numCandidates` below `limit` cannot return `limit` rows, so it raises
        the low end of the grid to k and the calibration records have to say so
        -- a reader comparing two engines at "ef 10" needs to know that one of
        them looked at 10 candidates and the other at k.

        Takes `k` because the only clamp in this codebase is against k. The
        spec names the shape `effective_ef(ef)`, and a one-argument version
        would have to keep its own copy of k: k arrives per query
        (`query(vector, k)`) and no driver stores it, so the value would be
        stale exactly when a profile changed it.

        Reports rather than enforces. The engine does the clamping; this says
        what it did, and a driver that returned a value its queries do not use
        would be worse than no method at all.
        """
        return int(ef)

    @abc.abstractmethod
    def query(self, vector, k: int) -> List[int]: ...

    @abc.abstractmethod
    def query_filtered(self, vector, k: int, tag_threshold: int) -> List[int]: ...

    @abc.abstractmethod
    def explain_uses_vector_index(self, vector, k: int,
                                  tag_threshold: Optional[int] = None) -> bool:
        """Whether the planner actually chose the vector index.

        Not a nicety. Every one of these engines can fall back to a full scan,
        which returns exact results slowly — indistinguishable in the output
        from "high recall, low QPS" unless it is checked explicitly.
        """

    # -- sizing / metadata ----------------------------------------------

    @abc.abstractmethod
    def index_bytes(self) -> int: ...

    @abc.abstractmethod
    def table_bytes(self) -> int: ...

    @abc.abstractmethod
    def server_version(self) -> str: ...

    def capabilities(self) -> Dict[str, Any]:
        return {
            "incremental_index": self.incremental_index,
            "async_index_build": self.async_index_build,
            "ef_construction_tunable": False,
        }


def batched(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]
