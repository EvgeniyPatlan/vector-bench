"""Will an engine's index build fit in the server's memory limit?

One function, used in three places so they cannot disagree: the orchestrator
refuses a unit up front with it (`orchestrator/cli.py:_memory_refusal`), the
report states the refusal from the figures it returns, and
`docs/04-engine-notes.md`'s MyVector memory table is checked against it by
`tests/test_e25_myvector_memory_fit.py`.

Pure arithmetic, no numpy: it runs on the host, where the orchestrator lives
and the framework's host requirements are python3 and pyyaml only.

Only MyVector has a model here (`capabilities.memory_fit: myvector` in its
engine config). Its HNSW graph is hnswlib memory in mysqld's heap -- resident,
unpageable, outside `innodb_buffer_pool_size` -- and its build additionally
holds the whole (id, vector) result set client-side while it inserts
(`BuildMyVectorIndexSQL`: `mysql_store_result`). A graph that does not fit is
not a slow run, as it would be for MariaDB's or AliSQL's graph cache; the
container's OOM killer ends it mid-build.
"""

from __future__ import annotations

from typing import Dict, Optional

GIB = 1024 ** 3

#: Per-element bytes beyond `Element Data Size`: hnswlib's per-element lock
#: (`link_list_locks_`, a std::mutex), level and link-list-pointer arrays and
#: the label map entry. An allowance, not a measurement -- stated as one.
PER_ELEMENT_OVERHEAD = 100


def element_bytes(dim: int, m: int) -> int:
    """hnswlib's `size_data_per_element_` for float vectors, plus overhead.

    Vector (4 x dim) + level-0 link list (2M ids of 4 bytes plus a 4-byte
    count) + the 8-byte label. Measured 3,276 B before overhead at 784 dims
    and M=16 (`MYVECTOR_INDEX_STATUS`'s `Element Data Size`), which this
    reproduces exactly.
    """
    return 4 * int(dim) + (2 * int(m) * 4 + 4) + 8 + PER_ELEMENT_OVERHEAD


def myvector_build_peak(dim: int, rows: int, m: int,
                        buffer_bytes: int) -> Dict[str, int]:
    """Bytes at MyVector's build peak, by part.

    `graph` is rows x `element_bytes`; `transient` is the vectors held
    client-side by the build, 4 x dim x rows; `buffer` is the InnoDB buffer
    pool the run gives the server, which the load has filled by then.
    """
    graph = int(rows) * element_bytes(dim, m)
    transient = 4 * int(dim) * int(rows)
    buffer = int(buffer_bytes)
    return {"graph": graph, "transient": transient, "buffer": buffer,
            "peak": graph + transient + buffer}


def gib(nbytes: float) -> str:
    return f"{nbytes / GIB:.1f} GiB"


def refusal_reason(peak: int, limit: int) -> str:
    """The sentence a refused unit carries, in the manifest and the report."""
    return (f"does not fit: estimated peak {gib(peak)} > limit {gib(limit)} "
            f"(buffer pool + HNSW graph + the build's transient copy of the "
            f"vectors; harness/memory_fit.py)")


def check(model: Optional[str], dim: int, rows: int, m: int,
          buffer_bytes: int, limit_bytes: int) -> Optional[Dict[str, object]]:
    """A refusal record for a unit that cannot fit, or None.

    None when the engine declares no model, when the corpus size or the limit
    is unknown, or when the peak fits. An unknown is never a refusal: this
    exists to stop a certain OOM, not to guess.
    """
    if model != "myvector" or not dim or not rows or not limit_bytes:
        return None
    parts = myvector_build_peak(dim, rows, m, buffer_bytes)
    if parts["peak"] <= limit_bytes:
        return None
    return {"reason": refusal_reason(parts["peak"], limit_bytes),
            "estimated_peak_bytes": parts["peak"],
            "limit_bytes": int(limit_bytes),
            "estimate": parts, "rows": int(rows), "dim": int(dim), "m": int(m)}
