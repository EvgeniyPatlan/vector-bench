"""Ops-harness driver for PostgreSQL + pgvector.

PostgreSQL necessarily uses a different client stack (psycopg3) from the
MySQL-family engines. That asymmetry cannot be removed and is stated in
docs/05-methodology.md; psycopg3's binary protocol is the nearest equivalent to
Connector/C's, so it is what is used here.

The substantive difference from MHNSW and VIDX is that pgvector builds its HNSW
index as a bulk operation after the data is loaded, rather than incrementally on
INSERT. `build_mode` selects which of the two comparisons is being made.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy

from .base import DATABASE, TABLE, ConnectionSpec, EngineDriver, IndexSpec, LoadResult

try:
    import psycopg
    import pgvector.psycopg
except ImportError:  # pragma: no cover - only the bench image has these
    psycopg = None

PROGRESS_INTERVAL_S = float(os.environ.get("VB_PROGRESS_INTERVAL", "20"))

INDEX = f"{TABLE}_embedding_idx"
OPCLASS = {"angular": "vector_cosine_ops", "euclidean": "vector_l2_ops"}
OPERATOR = {"angular": "<=>", "euclidean": "<->"}


class PostgresDriver(EngineDriver):
    name = "pgvector"
    incremental_index = False

    def __init__(self, spec: ConnectionSpec):
        super().__init__(spec)
        self._cur = None
        self._index: Optional[IndexSpec] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _open(self):
        conn = psycopg.connect(
            host=self.spec.host, port=self.spec.port,
            user=self.spec.user, dbname=self.spec.database,
            password=self.spec.password or None,
            autocommit=True, connect_timeout=30,
        )
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        pgvector.psycopg.register_vector(conn)
        # JIT compiles the distance expression per query; at these row counts
        # that costs more than it saves, and vector deployments routinely
        # disable it for exactly this reason.
        conn.execute("SET jit = off")
        return conn

    def connect(self) -> None:
        if psycopg is None:
            raise RuntimeError("psycopg/pgvector are not installed")
        self._conn = self._open()
        self._cur = self._conn.cursor()

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
        self._cur.execute(f"DROP TABLE IF EXISTS {TABLE}")

    def create_schema(self, index: IndexSpec) -> None:
        self._index = index
        self._cur.execute(
            f"CREATE TABLE {TABLE} (id int PRIMARY KEY, tag int NOT NULL, "
            f"embedding vector({index.dim}))"
        )
        # Keep vectors inline. TOASTed vectors would add a detoast to every
        # distance comparison, which pgvector's own documentation warns against.
        self._cur.execute(
            f"ALTER TABLE {TABLE} ALTER COLUMN embedding SET STORAGE PLAIN"
        )
        self._cur.execute(f"CREATE INDEX {TABLE}_tag_idx ON {TABLE} (tag)")
        if index.build_mode == "incremental":
            self.create_index(index)

    def create_index(self, index: IndexSpec) -> None:
        self._index = index
        ef_construction = index.ef_construction or 200
        sql = (
            f"CREATE INDEX {INDEX} ON {TABLE} USING hnsw "
            f"(embedding {OPCLASS[index.metric]}) "
            f"WITH (m = {index.m}, ef_construction = {ef_construction})"
        )
        print(f"[pgvector] {sql}")
        self._cur.execute(sql)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def load(self, vectors: numpy.ndarray, tags: numpy.ndarray,
             threads: int = 1, start_id: int = 0) -> LoadResult:
        total = len(vectors)
        threads = max(1, threads)
        start = time.perf_counter()

        def copy_range(lo: int, hi: int, cur, report: bool = False) -> None:
            span = hi - lo
            began = time.perf_counter()
            next_report = began + PROGRESS_INTERVAL_S
            with cur.copy(f"COPY {TABLE} (id, tag, embedding) FROM STDIN") as copy:
                for i in range(lo, hi):
                    copy.write_row((start_id + i, int(tags[i]), vectors[i]))
                    if report and (i - lo) % 5000 == 0:
                        now = time.perf_counter()
                        if now >= next_report:
                            done = i - lo + 1
                            rate = done / max(now - began, 1e-9)
                            eta = (span - done) / max(rate, 1e-9)
                            print(f"[pgvector]   {done:,}/{span:,} rows, "
                                  f"{rate:,.0f} rows/s, ETA {eta / 60:.1f} min",
                                  flush=True)
                            next_report = now + PROGRESS_INTERVAL_S

        if threads == 1:
            copy_range(0, total, self._cur, report=True)
        else:
            bounds = [(total * i // threads, total * (i + 1) // threads)
                      for i in range(threads)]

            def worker(bound: Tuple[int, int]) -> None:
                lo, hi = bound
                conn = self._open()
                try:
                    copy_range(lo, hi, conn.cursor(), report=(lo == 0))
                finally:
                    conn.close()

            with ThreadPoolExecutor(max_workers=threads) as pool:
                list(pool.map(worker, bounds))

        elapsed = time.perf_counter() - start
        return LoadResult(rows=total, wall_seconds=elapsed, threads=threads)

    def delete_ids(self, ids: Sequence[int]) -> None:
        if not ids:
            return
        self._cur.execute(
            f"DELETE FROM {TABLE} WHERE id = ANY(%s)", ([int(i) for i in ids],)
        )

    def insert_rows(self, ids: Sequence[int], vectors: numpy.ndarray,
                    tags: Sequence[int]) -> None:
        with self._cur.copy(f"COPY {TABLE} (id, tag, embedding) FROM STDIN") as copy:
            for i, v, t in zip(ids, vectors, tags):
                copy.write_row((int(i), int(t), v))

    def count_rows(self) -> int:
        self._cur.execute(f"SELECT count(*) FROM {TABLE}")
        return int(self._cur.fetchone()[0])

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def set_ef_search(self, ef_search: int) -> None:
        self._cur.execute(f"SET hnsw.ef_search = {int(ef_search)}")

    def _op(self) -> str:
        assert self._index is not None, "create_schema must run before querying"
        return OPERATOR[self._index.metric]

    def query(self, vector, k: int) -> List[int]:
        self._cur.execute(
            f"SELECT id FROM {TABLE} ORDER BY embedding {self._op()} %s LIMIT %s",
            (vector, k), binary=True, prepare=True,
        )
        return [row[0] for row in self._cur.fetchall()]

    def query_filtered(self, vector, k: int, tag_threshold: int) -> List[int]:
        self._cur.execute(
            f"SELECT id FROM {TABLE} WHERE tag < %s "
            f"ORDER BY embedding {self._op()} %s LIMIT %s",
            (int(tag_threshold), vector, k), binary=True, prepare=True,
        )
        return [row[0] for row in self._cur.fetchall()]

    def set_iterative_scan(self, mode: str) -> None:
        """pgvector 0.8 iterative index scan mode: off | relaxed_order | strict_order.

        Materially changes filtered-search recall: without it, a selective filter
        can exhaust the ef_search candidate list before finding k qualifying rows
        and simply return fewer results. Recorded per measurement so a filtered
        number is never ambiguous about which mode produced it.
        """
        self._cur.execute(f"SET hnsw.iterative_scan = {mode}")

    def explain_uses_vector_index(self, vector, k: int,
                                  tag_threshold: Optional[int] = None) -> bool:
        try:
            if tag_threshold is None:
                sql = f"SELECT id FROM {TABLE} ORDER BY embedding {self._op()} %s LIMIT %s"
                params: Tuple[Any, ...] = (vector, k)
            else:
                sql = (f"SELECT id FROM {TABLE} WHERE tag < %s "
                       f"ORDER BY embedding {self._op()} %s LIMIT %s")
                params = (int(tag_threshold), vector, k)
            self._cur.execute("EXPLAIN (FORMAT TEXT) " + sql, params)
            plan = " ".join(str(r[0]) for r in self._cur.fetchall())
        except Exception as exc:
            print(f"[pgvector] EXPLAIN failed: {exc}")
            return False
        used = INDEX in plan
        if not used:
            print(
                f"[pgvector] WARNING: HNSW index NOT used "
                f"(k={k}, filtered={tag_threshold is not None}). Plan: {plan}"
            )
        return used

    # ------------------------------------------------------------------
    # Sizing
    # ------------------------------------------------------------------

    def index_bytes(self) -> int:
        try:
            self._cur.execute("SELECT pg_relation_size(%s)", (INDEX,))
            return int(self._cur.fetchone()[0])
        except Exception:
            return 0

    def table_bytes(self) -> int:
        try:
            self._cur.execute("SELECT pg_table_size(%s)", (TABLE,))
            return int(self._cur.fetchone()[0])
        except Exception:
            return 0

    def server_version(self) -> str:
        try:
            self._cur.execute(
                "SELECT version() || ' / pgvector ' || "
                "coalesce((SELECT extversion FROM pg_extension WHERE extname='vector'),'?')"
            )
            return str(self._cur.fetchone()[0])
        except Exception:
            return "unknown"

    def capabilities(self) -> Dict[str, Any]:
        return {"incremental_index": False, "ef_construction_tunable": True}


DRIVERS = {}


def _driver_table() -> Dict[str, Any]:
    """Engine name -> driver class.

    The single source of truth for what this harness can drive. The argument
    parser reads it too, so adding an engine here is enough: a mismatch between
    the two lists is how a run got as far as starting the server and then died
    on `argument --engine: invalid choice`.
    """
    from .mongo import MongoDriver
    from .mysql_family import AliSQLDriver, MariaDBDriver

    return {
        "mariadb": MariaDBDriver,
        # Same server software at a different tag; the driver is identical.
        "mariadb123": MariaDBDriver,
        "alisql": AliSQLDriver,
        "pgvector": PostgresDriver,
        "mongodb": MongoDriver,
    }


def known_engines() -> Tuple[str, ...]:
    """Engine names this harness accepts, for argparse choices."""
    return tuple(_driver_table())


def get_driver(engine: str, spec: ConnectionSpec) -> EngineDriver:
    """Resolve an engine name to a connected-capable driver instance."""
    table = _driver_table()
    if engine not in table:
        raise ValueError(f"unknown engine: {engine} (expected one of {sorted(table)})")
    return table[engine](spec)
