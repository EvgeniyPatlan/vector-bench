"""Ops-harness drivers for MariaDB (MHNSW) and AliSQL (VIDX).

Both are driven through MariaDB Connector/C via the `mariadb` Python package.
Connector/C speaks the MySQL protocol, so one client library serves both
servers — which is the point: if the two engines were driven through different
clients, part of any measured difference would belong to the clients.

Behavioural differences between the two are confined to the `Dialect` values at
the bottom of this file.
"""

from __future__ import annotations

import glob
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy

from .base import DATABASE, TABLE, ConnectionSpec, EngineDriver, IndexSpec, LoadResult

PROGRESS_INTERVAL_S = float(os.environ.get("VB_PROGRESS_INTERVAL", "20"))

try:
    import mariadb as _connector
except ImportError:  # pragma: no cover - only the bench image has it
    _connector = None


def to_binary_f32(vector) -> bytes:
    """Pack a vector as little-endian float32.

    Accepted directly by MariaDB's VECTOR and by AliSQL's Field_vector, which
    validates that the string is exactly 4*dim bytes. Using the binary form for
    both avoids charging AliSQL for VEC_FROMTEXT() text parsing that MariaDB
    would not pay — a client-side cost that has nothing to do with the index.
    """
    return numpy.asarray(vector, dtype="<f4").tobytes()


@dataclass
class MySQLDialect:
    name: str
    session_setup: Tuple[str, ...] = ()
    global_setup: Tuple[str, ...] = ()
    set_ef_search: str = ""
    index_name: str = "vi"
    index_file_globs: Tuple[str, ...] = ()
    metric_names: Dict[str, str] = field(default_factory=dict)
    # Statement that reports the size of the vector index from the catalog,
    # used when the data directory is not readable from this process.
    index_size_sql: Optional[str] = None


class MySQLFamilyDriver(EngineDriver):
    dialect: MySQLDialect
    incremental_index = True

    def __init__(self, spec: ConnectionSpec, dialect: Optional[MySQLDialect] = None):
        super().__init__(spec)
        if dialect is not None:
            self.dialect = dialect
        self.name = self.dialect.name
        self._cur = None
        self._index: Optional[IndexSpec] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        if _connector is None:
            raise RuntimeError("the 'mariadb' Python connector is not installed")
        self._conn = _connector.connect(
            host=self.spec.host,
            port=self.spec.port,
            user=self.spec.user,
            password=self.spec.password,
            autocommit=True,
            connect_timeout=30,
        )
        self._cur = self._conn.cursor()
        for stmt in self.dialect.global_setup:
            try:
                self._cur.execute(stmt)
            except Exception as exc:
                # GLOBAL statements need SUPER; the bench account has it, but a
                # server someone else started may not grant it. Report rather
                # than fail, because the server flag usually covers this anyway.
                print(f"[{self.name}] global setup '{stmt}' failed: {exc}")
        self._use_database(create=True)

    def _use_database(self, create: bool = False) -> None:
        if create:
            self._cur.execute(f"CREATE DATABASE IF NOT EXISTS {DATABASE}")
        self._cur.execute(f"USE {DATABASE}")
        for stmt in self.dialect.session_setup:
            self._cur.execute(stmt)

    def close(self) -> None:
        for closer in (self._cur, self._conn):
            try:
                if closer is not None:
                    closer.close()
            except Exception:
                pass
        self._cur = self._conn = None

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def drop_schema(self) -> None:
        self._cur.execute(f"DROP DATABASE IF EXISTS {DATABASE}")
        self._use_database(create=True)

    def create_schema(self, index: IndexSpec) -> None:
        self._index = index
        metric = self.dialect.metric_names[index.metric]
        ddl = (
            f"CREATE TABLE {TABLE} (\n"
            f"  id INT PRIMARY KEY,\n"
            f"  tag INT NOT NULL,\n"
            f"  v VECTOR({index.dim}) NOT NULL,\n"
            f"  KEY tag_idx (tag),\n"
            f"  VECTOR INDEX {self.dialect.index_name} (v) "
            f"M={index.m} DISTANCE={metric}\n"
            f") ENGINE={index.storage_engine}"
        )
        print(f"[{self.name}] {ddl}")
        self._cur.execute(ddl)

    def create_index(self, index: IndexSpec) -> None:
        # MHNSW and VIDX both build the graph incrementally as rows are inserted;
        # the index is declared in CREATE TABLE and there is nothing to do here.
        # This asymmetry against pgvector's bulk build is measured explicitly
        # rather than hidden — see docs/05-methodology.md.
        self._index = index

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def load(self, vectors: numpy.ndarray, tags: numpy.ndarray,
             threads: int = 1, start_id: int = 0) -> LoadResult:
        total = len(vectors)
        threads = max(1, threads)
        start = time.perf_counter()

        if threads == 1:
            self._load_range(self._cur, vectors, tags, start_id, 0, total,
                             report=True)
        else:
            bounds = [(total * i // threads, total * (i + 1) // threads)
                      for i in range(threads)]

            def worker(bound: Tuple[int, int]) -> None:
                lo, hi = bound
                conn = _connector.connect(
                    host=self.spec.host, port=self.spec.port,
                    user=self.spec.user, password=self.spec.password,
                    autocommit=True, connect_timeout=30,
                )
                try:
                    cur = conn.cursor()
                    cur.execute(f"USE {DATABASE}")
                    for stmt in self.dialect.session_setup:
                        cur.execute(stmt)
                    # Only the first shard reports, or the threads interleave.
                    self._load_range(cur, vectors, tags, start_id, lo, hi,
                                     report=(lo == 0))
                    cur.close()
                finally:
                    conn.close()

            with ThreadPoolExecutor(max_workers=threads) as pool:
                list(pool.map(worker, bounds))

        elapsed = time.perf_counter() - start
        return LoadResult(rows=total, wall_seconds=elapsed, threads=threads)

    def _load_range(self, cur, vectors, tags, start_id: int, lo: int, hi: int,
                    batch: int = 500, report: bool = False) -> None:
        """Load a contiguous range in batches.

        Reports progress on a TIME interval when `report` is set. These engines
        maintain the HNSW graph on every INSERT, so a million-row load runs for
        a long time at a rate that falls as the graph grows; without periodic
        output there is no way to tell a working load from a hung one, and no
        way to answer "how long will this take" except by waiting.
        """
        sql = f"INSERT INTO {TABLE} (id, tag, v) VALUES (%s, %s, %s)"
        rows: List[Tuple[int, int, bytes]] = []
        total = hi - lo
        started = time.perf_counter()
        next_report = started + PROGRESS_INTERVAL_S

        for i in range(lo, hi):
            rows.append((start_id + i, int(tags[i]), to_binary_f32(vectors[i])))
            if len(rows) >= batch:
                cur.executemany(sql, rows)
                # Rebind rather than clear(): the connector receives the list
                # itself, and mutating the object it was handed is a needless
                # bet on executemany() having fully consumed it.
                rows = []
                now = time.perf_counter()
                if report and now >= next_report:
                    done = i - lo + 1
                    rate = done / max(now - started, 1e-9)
                    eta = (total - done) / max(rate, 1e-9)
                    print(f"[{self.name}]   {done:,}/{total:,} rows, "
                          f"{rate:,.0f} rows/s, ETA {eta / 60:.1f} min", flush=True)
                    next_report = now + PROGRESS_INTERVAL_S
        if rows:
            cur.executemany(sql, rows)
        cur.execute("COMMIT")

    def delete_ids(self, ids: Sequence[int]) -> None:
        if not ids:
            return
        for chunk_start in range(0, len(ids), 1000):
            chunk = ids[chunk_start:chunk_start + 1000]
            placeholders = ",".join(["%s"] * len(chunk))
            self._cur.execute(
                f"DELETE FROM {TABLE} WHERE id IN ({placeholders})",
                tuple(int(i) for i in chunk),
            )
        self._cur.execute("COMMIT")

    def insert_rows(self, ids: Sequence[int], vectors: numpy.ndarray,
                    tags: Sequence[int]) -> None:
        sql = f"INSERT INTO {TABLE} (id, tag, v) VALUES (%s, %s, %s)"
        rows = [
            (int(i), int(t), to_binary_f32(v))
            for i, v, t in zip(ids, vectors, tags)
        ]
        for chunk_start in range(0, len(rows), 500):
            self._cur.executemany(sql, rows[chunk_start:chunk_start + 500])
        self._cur.execute("COMMIT")

    def count_rows(self) -> int:
        self._cur.execute(f"SELECT COUNT(*) FROM {TABLE}")
        return int(self._cur.fetchone()[0])

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def set_ef_search(self, ef_search: int) -> None:
        self._cur.execute(self.dialect.set_ef_search.format(ef_search=int(ef_search)))

    def _metric_fn(self) -> str:
        assert self._index is not None, "create_schema must run before querying"
        return self.dialect.metric_names[self._index.metric].lower()

    def _select(self, k: int, filtered: bool) -> str:
        where = "WHERE tag < %s " if filtered else ""
        return (
            f"SELECT id FROM {TABLE} {where}"
            f"ORDER BY vec_distance_{self._metric_fn()}(v, %s) LIMIT {int(k)}"
        )

    def query(self, vector, k: int) -> List[int]:
        self._cur.execute(self._select(k, False), (to_binary_f32(vector),))
        return [row[0] for row in self._cur.fetchall()]

    def query_filtered(self, vector, k: int, tag_threshold: int) -> List[int]:
        self._cur.execute(
            self._select(k, True), (int(tag_threshold), to_binary_f32(vector))
        )
        return [row[0] for row in self._cur.fetchall()]

    def explain_uses_vector_index(self, vector, k: int,
                                  tag_threshold: Optional[int] = None) -> bool:
        filtered = tag_threshold is not None
        params: Tuple[Any, ...] = (
            (int(tag_threshold), to_binary_f32(vector)) if filtered
            else (to_binary_f32(vector),)
        )
        try:
            self._cur.execute("EXPLAIN " + self._select(k, filtered), params)
            plan = " | ".join(str(c) for row in self._cur.fetchall() for c in row)
        except Exception as exc:
            print(f"[{self.name}] EXPLAIN failed: {exc}")
            return False
        used = self.dialect.index_name in plan
        if not used:
            print(
                f"[{self.name}] WARNING: vector index NOT used "
                f"(k={k}, filtered={filtered}). Plan: {plan}"
            )
        return used

    # ------------------------------------------------------------------
    # Sizing
    # ------------------------------------------------------------------

    def _glob_bytes(self, patterns: Sequence[str]) -> int:
        if not self.spec.data_dir:
            return 0
        base = os.path.join(self.spec.data_dir, DATABASE)
        total, seen = 0, set()
        for pattern in patterns:
            for path in glob.glob(os.path.join(base, pattern)):
                real = os.path.realpath(path)
                if real in seen:
                    continue
                seen.add(real)
                try:
                    total += os.stat(real).st_size
                except OSError:
                    pass
        return total

    def index_bytes(self) -> int:
        """On-disk size of the HNSW graph.

        Both engines keep the graph in a companion table, so a filesystem
        measurement against the shared data directory is authoritative. The
        catalog query is a fallback for when the data directory is not shared.
        """
        size = self._glob_bytes(self.dialect.index_file_globs)
        if size:
            return size
        if self.dialect.index_size_sql:
            try:
                self._cur.execute(self.dialect.index_size_sql)
                row = self._cur.fetchone()
                if row and row[0]:
                    return int(row[0])
            except Exception:
                pass
        return 0

    def table_bytes(self) -> int:
        size = self._glob_bytes((f"{TABLE}.ibd", f"{TABLE}.MYD", f"{TABLE}.MYI"))
        if size:
            return size
        try:
            self._cur.execute(
                "SELECT DATA_LENGTH + INDEX_LENGTH FROM information_schema.TABLES "
                "WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s",
                (DATABASE, TABLE),
            )
            row = self._cur.fetchone()
            return int(row[0]) if row and row[0] else 0
        except Exception:
            return 0

    def server_version(self) -> str:
        try:
            self._cur.execute("SELECT VERSION()")
            return str(self._cur.fetchone()[0])
        except Exception:
            return "unknown"

    def capabilities(self) -> Dict[str, Any]:
        return {"incremental_index": True, "ef_construction_tunable": False}


# ---------------------------------------------------------------------------
# Concrete engines
# ---------------------------------------------------------------------------

MARIADB_DIALECT = MySQLDialect(
    name="mariadb",
    session_setup=("SET SESSION default_storage_engine = InnoDB",),
    global_setup=(),
    set_ef_search="SET mhnsw_ef_search = {ef_search}",
    index_name="vi",
    index_file_globs=(
        "t1#i#*.ibd", "t1#i#*.MYI", "t1#i#*.MYD",
        "t1@0023i@0023*.ibd", "t1@0023i@0023*.MYI", "t1@0023i@0023*.MYD",
    ),
    metric_names={"angular": "cosine", "euclidean": "euclidean"},
)

ALISQL_DIALECT = MySQLDialect(
    name="alisql",
    # RC is mandatory: any other isolation level raises ER_NOT_SUPPORTED_YET on
    # every vector operation. vidx_disabled must be OFF or DDL fails outright.
    session_setup=(
        "SET SESSION transaction_isolation = 'READ-COMMITTED'",
        "SET SESSION default_storage_engine = InnoDB",
    ),
    global_setup=("SET GLOBAL vidx_disabled = OFF",),
    set_ef_search="SET vidx_hnsw_ef_search = {ef_search}",
    index_name="vi",
    # VIDX names its auxiliary table vidx_%016lx_%02x — see VIDX_NAME in
    # sql/vidx/vidx_index.cc. The table id is not knowable in advance.
    index_file_globs=("vidx_*.ibd",),
    metric_names={"angular": "COSINE", "euclidean": "EUCLIDEAN"},
)


class MariaDBDriver(MySQLFamilyDriver):
    dialect = MARIADB_DIALECT
    name = "mariadb"


class AliSQLDriver(MySQLFamilyDriver):
    dialect = ALISQL_DIALECT
    name = "alisql"
