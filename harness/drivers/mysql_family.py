"""Ops-harness drivers for MariaDB (MHNSW), AliSQL (VIDX) and Percona (InnoDB
HNSW).

All three are driven through MariaDB Connector/C via the `mariadb` Python
package. Connector/C speaks the MySQL protocol, so one client library serves
every server here — which is the point: if the engines were driven through
different clients, part of any measured difference would belong to the clients.

**Differences live in the `Dialect` values at the bottom of this file wherever
they can.** For MariaDB and AliSQL that is all of them, and Percona moved three
more into values that `MySQLFamilyDriver` used to spell out — the primary-key
type, the index clause and the distance expression — so the next engine finds
them as fields rather than as methods to override.

Four of Percona's differences are not values and could not be: `capabilities()`
varies with `IndexSpec.build_mode` because it is the first engine here that can
build its graph either way; `effective_ef` clamps; `maintain()` runs a real
repair path and has to read the index's options back off the server to do it
honestly; and `create_schema`/`create_index` split the index out of the table
definition in bulk-build mode. Those four are `PerconaDriver`, and they are
what makes it an architecture rather than a variant.
"""

from __future__ import annotations

import glob
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy

from .base import (DATABASE, FILTER_CTAG, FILTER_TAG, TABLE, ConnectionSpec,
                   EngineDriver, IndexSpec, LoadResult, RowAttrs,
                   checked_filter_column)

PROGRESS_INTERVAL_S = float(os.environ.get("VB_PROGRESS_INTERVAL", "20"))

try:
    import mariadb as _connector
except ImportError:  # pragma: no cover - only the bench image has it
    _connector = None

# One INSERT for both write paths. The bulk load and churn's re-insert each
# held their own copy of this string, naming the same columns by coincidence
# rather than by construction; PR 3 adds a column to both, which is exactly
# the edit where one copy gets missed. The column names come from
# `datasets.FILTER_COLUMNS` via base.py rather than being spelled here, so the
# statement cannot name a column the dataset module does not compute.
INSERT_SQL = (
    f"INSERT INTO {TABLE} (id, {FILTER_TAG}, {FILTER_CTAG}, v) "
    f"VALUES (%s, %s, %s, %s)"
)


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

    # --- the five fields Percona needed, each defaulted to what MariaDB and
    # --- AliSQL already did, so neither of those engines changes behaviour.
    #
    # Before these, three of Percona's differences were not VALUES at all --
    # `MySQLFamilyDriver` spelled the statements out, so a second spelling
    # meant a second method rather than a second dialect. That is the
    # arrangement where two methods drift.

    #: Primary-key column type. Percona REQUIRES a `BIGINT UNSIGNED`
    #: single-column primary key on any table carrying a vector index, and
    #: refuses `BIGINT`, `INT UNSIGNED` and composite keys with one message:
    #: "vector indexes on tables lacking a BIGINT UNSIGNED single column
    #: primary key". So Percona rows are 4 bytes wider in the clustered index
    #: and in the aux table's `base_pk` than MariaDB's and AliSQL's -- by the
    #: server's requirement, not by our choice. About 4.7 MB at glove's
    #: 1.18M rows: far below the resolution of any figure here, but
    #: `index_bytes` and `table_bytes` are reported and someone will notice.
    pk_type: str = "INT"

    #: The index clause inside CREATE TABLE. `{name}`, `{m}`, `{metric}` and
    #: `{ef_construction}` are interpolated. MariaDB and AliSQL take the
    #: `M=n DISTANCE=m` spelling; Percona takes a parenthesised option list
    #: after `TYPE hnsw`, where an option value is a bare identifier or an
    #: integer and never a quoted string.
    index_clause: str = "VECTOR INDEX {name} (v) M={m} DISTANCE={metric}"

    #: The ORDER BY distance expression, with `%s` for the bound query vector.
    #: `{metric}` is the dialect's lowercased metric name and `{metric_upper}`
    #: its uppercase form. MariaDB and AliSQL have one function per metric;
    #: Percona has one function with the metric as a mandatory third argument
    #: (`sql/item_create.cc` registers DISTANCE with arity 3).
    #:
    #: The metric stays a LITERAL and does not become a bound parameter. The
    #: parameter order is therefore the same for both spellings -- filtered
    #: queries bind `(threshold, vector)` because the predicate precedes the
    #: ORDER BY -- and making it a parameter would add a third placeholder and
    #: reorder every call site for no gain.
    distance_expr: str = "vec_distance_{metric}(v, %s)"

    #: What an EXPLAIN must contain for the vector path to have been taken,
    #: beyond the index name. `""` means "the index name is the whole test",
    #: which is sound for MariaDB and AliSQL.
    #:
    #: IT IS NOT SOUND FOR PERCONA, and that is demonstrated rather than
    #: argued. `…/percona_vector/r/optimizer.result:491-492` records a plan
    #: with access type `ALL`, `key=NULL` and `Extra=Using filesort` -- a full
    #: table scan -- on a table whose only index is a `VECTOR KEY`, with the
    #: index name sitting in `possible_keys`. A substring test against the
    #: joined EXPLAIN text therefore returns True for brute force published as
    #: ANN search, which is the exact vacuity shape this repo keeps finding.
    #: `:610` is the second shape -- the vector index offered and a different
    #: one taken -- and a filtered query is exactly that shape.
    #:
    #: `vector_ann` is an access TYPE and appears only when the vector path
    #: was activated.
    plan_marker: str = ""

    #: Build-time effort knob, for the one engine here that has one in the
    #: MySQL family. `None` means the engine does not expose it and the
    #: `{ef_construction}` slot is never reached. Percona's value comes from
    #: `IndexSpec.ef_construction` per build, not from this default.
    ef_construction: Optional[int] = None


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
        ddl = self._create_table_ddl(index, with_index=True)
        print(f"[{self.name}] {ddl}")
        self._cur.execute(ddl)

    def _create_table_ddl(self, index: IndexSpec, *,
                          with_index: bool) -> str:
        """The CREATE TABLE, with or without the vector index inline.

        One builder rather than one per driver, and `with_index` is the only
        thing a caller may vary. `PerconaDriver` needs the index left out in
        bulk-build mode and everything else identical; a second copy of this
        string in that subclass would be a second place a column has to be
        added, which is the arrangement `INSERT_SQL` above was consolidated
        out of for the same reason.

        `ctag` carries a plain KEY exactly as `tag` does, and that parity is
        the measurement rather than tidiness: the two columns exist to be
        compared, and an optimiser offered an index on one and not the other
        is choosing between different plans, not between different filter
        correlations. `tests/test_t3b_drivers.py` pins the index kinds on
        the two columns as equal sets rather than checking that a `ctag`
        index exists somewhere in the DDL.
        """
        columns = [
            f"  id {self.dialect.pk_type} PRIMARY KEY",
            "  tag INT NOT NULL",
            "  ctag INT NOT NULL",
            f"  v VECTOR({index.dim}) NOT NULL",
            "  KEY tag_idx (tag)",
            "  KEY ctag_idx (ctag)",
        ]
        if with_index:
            metric = self.dialect.metric_names[index.metric]
            columns.append(f"  {self._index_clause(index, metric)}")
        body = ",\n".join(columns)
        return (f"CREATE TABLE {TABLE} (\n{body}\n"
                f") ENGINE={index.storage_engine}")

    def _index_clause(self, index: IndexSpec, metric: str) -> str:
        """The vector-index clause, in this dialect's spelling.

        Its own method rather than an inline `.format()` because `maintain()`
        has to rebuild the same index with the same options and must not
        re-derive them from a second expression -- an index rebuilt at
        different options is a different index, and the recall measured after
        it would belong to that one.
        """
        return self.dialect.index_clause.format(
            name=self.dialect.index_name,
            m=int(index.m),
            metric=metric,
            ef_construction=(index.ef_construction
                             if index.ef_construction is not None
                             else self.dialect.ef_construction),
        )

    def create_index(self, index: IndexSpec) -> None:
        # MHNSW and VIDX both build the graph incrementally as rows are inserted;
        # the index is declared in CREATE TABLE and there is nothing to do here.
        # This asymmetry against pgvector's bulk build is measured explicitly
        # rather than hidden — see docs/05-methodology.md.
        self._index = index

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def load(self, vectors: numpy.ndarray,
             attrs: Union[RowAttrs, Sequence[int], numpy.ndarray],
             threads: int = 1, start_id: int = 0) -> LoadResult:
        total = len(vectors)
        threads = max(1, threads)
        # Coerced once, here, rather than inside `_load_range`: the worker
        # threads below each get a slice of the same object, and a per-worker
        # coercion would be per-worker mirroring of a bare array.
        attrs = RowAttrs.coerce(attrs)
        start = time.perf_counter()

        if threads == 1:
            self._load_range(self._cur, vectors, attrs, start_id, 0, total,
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
                    self._load_range(cur, vectors, attrs, start_id, lo, hi,
                                     report=(lo == 0))
                    cur.close()
                finally:
                    conn.close()

            with ThreadPoolExecutor(max_workers=threads) as pool:
                list(pool.map(worker, bounds))

        elapsed = time.perf_counter() - start
        return LoadResult(rows=total, wall_seconds=elapsed, threads=threads)

    def _load_range(self, cur, vectors, attrs: RowAttrs, start_id: int,
                    lo: int, hi: int,
                    batch: int = 500, report: bool = False) -> None:
        """Load a contiguous range in batches.

        Reports progress on a TIME interval when `report` is set. These engines
        maintain the HNSW graph on every INSERT, so a million-row load runs for
        a long time at a rate that falls as the graph grows; without periodic
        output there is no way to tell a working load from a hung one, and no
        way to answer "how long will this take" except by waiting.
        """
        rows: List[Tuple[Any, ...]] = []
        total = hi - lo
        started = time.perf_counter()
        next_report = started + PROGRESS_INTERVAL_S

        for i in range(lo, hi):
            rows.append((start_id + i, *attrs.row(i), to_binary_f32(vectors[i])))
            if len(rows) >= batch:
                cur.executemany(INSERT_SQL, rows)
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
            cur.executemany(INSERT_SQL, rows)
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
                    attrs: Union[RowAttrs, Sequence[int], numpy.ndarray]
                    ) -> None:
        attrs = RowAttrs.coerce(attrs)
        rows = [
            (int(row_id), *attrs.row(i), to_binary_f32(vectors[i]))
            for i, row_id in enumerate(ids)
        ]
        for chunk_start in range(0, len(rows), 500):
            self._cur.executemany(INSERT_SQL, rows[chunk_start:chunk_start + 500])
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

    def _select(self, k: int, filtered: bool, *,
                column: str = FILTER_TAG) -> str:
        """The one SELECT every query path in this driver runs.

        `query`, `query_filtered` and `explain_uses_vector_index` all come
        through here, so there is exactly one place a predicate is built and
        exactly one place the column name is checked. Two sites checking the
        same thing is the arrangement where one of them is later edited and
        the other is not.

        The column is whitelisted unconditionally, not only on the filtered
        branch: a `column` that cannot be queried is a caller's mistake
        whether or not this particular call puts it in the WHERE clause, and
        reporting it on the first call rather than the first *filtered* call
        is the difference between a clear error and a plan check that passed
        before a measurement that could not.

        `column` is keyword-only for the reason `postgres._select` gives at
        length: it is the argument that decides which column a measurement
        names, and it must not be identified by position. Same marker, same
        guard, same shape -- the two drivers differ only in the `k` they
        interpolate into LIMIT.
        """
        # The flag guard, argued in full beside `postgres._select`'s copy:
        # the `*` above stops a column arriving in `column`'s position, and
        # only this stops one arriving in the FLAG's -- `_select(k, "ctag")`
        # is truthy and silently builds `WHERE tag < %s`. It also survives
        # someone deleting the marker.
        if not isinstance(filtered, bool):
            raise TypeError(
                f"_select's filtered argument is a flag, not a column; got "
                f"{filtered!r}. Did you mean column={filtered!r}?")
        where = f"WHERE {checked_filter_column(column)} < %s " if filtered else ""
        metric = self._metric_fn()
        distance = self.dialect.distance_expr.format(
            metric=metric, metric_upper=metric.upper())
        return (
            f"SELECT id FROM {TABLE} {where}"
            f"ORDER BY {distance} LIMIT {int(k)}"
        )

    def query(self, vector, k: int) -> List[int]:
        self._cur.execute(self._select(k, False), (to_binary_f32(vector),))
        return [row[0] for row in self._cur.fetchall()]

    def query_filtered(self, vector, k: int, threshold: int,
                       column: str = FILTER_TAG) -> List[int]:
        self._cur.execute(
            self._select(k, True, column=column),
            (int(threshold), to_binary_f32(vector)),
        )
        return [row[0] for row in self._cur.fetchall()]

    #: Zero-based position of the `type` column in `EXPLAIN
    #: FORMAT=TRADITIONAL` output, whose twelve columns are id, select_type,
    #: table, partitions, type, possible_keys, key, key_len, ref, rows,
    #: filtered, Extra. Named rather than inlined because a bare `row[4]` in
    #: the middle of a plan check is unreadable and un-greppable.
    _EXPLAIN_TYPE_COLUMN = 4
    _EXPLAIN_COLUMNS = 12

    def _marker_in_access_type(self, rows: Sequence[Sequence[Any]]) -> bool:
        """Is `plan_marker` the ACCESS TYPE of some row, not merely present?

        The joined-text form of this test -- `plan_marker in plan` -- is the
        shape this repo keeps being bitten by: a substring satisfied by
        something other than the thing being asserted. It is not reachable for
        the statements this driver sends (the table is `t1`, the columns are
        fixed, and nothing else in the plan text spells `vector_ann`), so this
        is closing a class rather than fixing a live defect. It is worth
        closing because the whole argument for `plan_marker` existing is that
        a substring check accepted a brute-force scan, and leaving the new
        check as a substring would be making that argument and then not
        applying it.

        Falls back to the joined text when the row shape is not the twelve
        tabular columns, which is the honest behaviour rather than a `False`:
        a `False` here is published as "this engine did not use its index",
        and an unexpected EXPLAIN shape is a statement about the format, not
        about the engine. The fallback is no weaker than what this method did
        before, so no plan that passed can start failing.

        Compared case-insensitively on a stripped value. MySQL emits access
        types lowercase, and a server that capitalised one would otherwise
        turn a used index into a reported scan.
        """
        marker = self.dialect.plan_marker.strip().lower()
        shaped = [r for r in rows if len(r) >= self._EXPLAIN_COLUMNS]
        if not shaped:
            plan = " | ".join(str(c) for row in rows for c in row)
            print(f"[{self.name}] EXPLAIN returned "
                  f"{len(rows)} row(s) with no {self._EXPLAIN_COLUMNS}-column "
                  f"shape; falling back to a text match for "
                  f"{self.dialect.plan_marker!r}")
            return marker in plan.lower()
        return any(
            str(row[self._EXPLAIN_TYPE_COLUMN]).strip().lower() == marker
            for row in shaped)

    def explain_uses_vector_index(self, vector, k: int,
                                  tag_threshold: Optional[int] = None,
                                  column: str = FILTER_TAG) -> bool:
        # Checked here, above the try, and the placement is the point. The
        # body below swallows every exception on purpose -- a plan check that
        # fails must not end a run -- so a column rejected inside the try
        # would come back as a plain False, which every caller reads as a
        # statement about the engine. It would be a statement about the
        # caller. The measurement that followed would then fail anyway, one
        # step later in `query_filtered`, with this check already recorded in
        # the report's Validity section as an index that was not used.
        column = checked_filter_column(column)
        filtered = tag_threshold is not None
        params: Tuple[Any, ...] = (
            (int(tag_threshold), to_binary_f32(vector)) if filtered
            else (to_binary_f32(vector),)
        )
        try:
            # FORMAT=TRADITIONAL IS NOT DECORATION, and it is the one piece of
            # hardening here that is free. `Dialect.plan_marker` is a value in
            # the `type` column, which only the tabular format has.
            # `explain_format` defaults to TRADITIONAL so a bare `EXPLAIN`
            # worked -- but the variable is settable and reachable through
            # `VB_SERVER_ARGS`, and under TREE or JSON the marker disappears
            # and this method returns False about a server that DID use the
            # index. That records a false negative in the report's Validity
            # section and sends someone hunting an engine bug that is not
            # there. Spelling the format out removes the whole failure mode.
            self._cur.execute(
                "EXPLAIN FORMAT=TRADITIONAL "
                + self._select(k, filtered, column=column),
                params)
            rows = list(self._cur.fetchall())
            plan = " | ".join(str(c) for row in rows for c in row)
        except Exception as exc:
            print(f"[{self.name}] EXPLAIN failed: {exc}")
            return False
        # BOTH when a marker is declared, and the order of the two conjuncts
        # is the order of their importance. The access type is the
        # load-bearing half: the index name alone is satisfied by a name
        # sitting in `possible_keys` on a plan that took a full table scan,
        # which is recorded in Percona's own MTR output (see
        # `MySQLDialect.plan_marker`). The name is kept beside it so that a
        # second vector index on the table could not satisfy the check for
        # the wrong one.
        used = self.dialect.index_name in plan
        if self.dialect.plan_marker:
            used = used and self._marker_in_access_type(rows)
        if not used:
            # The column is named, not just `filtered=True`: `tag` and `ctag`
            # carry the same kind of index and different value distributions,
            # and an optimiser that takes the vector index for one and a range
            # scan for the other is the single most likely way this check
            # fires in PR 3. "filtered" alone would not say which.
            # The marker is named when there is one, because "not used" over
            # two conjuncts does not say which failed, and the two mean
            # different things: a missing index name is the optimiser taking
            # another index, a missing access type with the name present is
            # the brute-force scan this check exists to catch.
            looked_for = (f"access type {self.dialect.plan_marker!r} and key "
                          f"{self.dialect.index_name!r}"
                          if self.dialect.plan_marker
                          else f"key {self.dialect.index_name!r}")
            print(
                f"[{self.name}] WARNING: vector index NOT used "
                f"(k={k}, filtered={filtered}, column={column!r}); "
                f"looked for {looked_for}. Plan: {plan}"
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


PERCONA_DIALECT = MySQLDialect(
    name="percona",
    # Nothing to enable. Unlike AliSQL, whose VIDX ships disabled behind
    # `vidx_disabled` and refuses every vector operation outside READ
    # COMMITTED, Percona's HNSW index needs no global switch and no isolation
    # level: no test in the pinned tree's percona_vector suite sets either to
    # make vector DML work.
    session_setup=("SET SESSION default_storage_engine = InnoDB",),
    global_setup=(),
    # SESSION and GLOBAL, default 40, range 1..100000, integers only, read per
    # query inside the scan's open (`ha_innodb.cc` passes
    # `THDVAR(m_user_thd, hnsw_ef_search)` to `vec_ann_open`). SESSION is
    # spelled out: the variable has both scopes and the global one would
    # outlive the calibration sweep that set it.
    set_ef_search="SET SESSION innodb_hnsw_ef_search = {ef_search}",
    index_name="vi",
    # Percona keeps the graph in an InnoDB auxiliary table per vector index,
    # named percona_vec_<type>_<table_id>_<index_id>. The ids are not knowable
    # in advance, so match the prefix -- which is what the suite itself does
    # when it checks the files are cleaned up.
    index_file_globs=("percona_vec_*.ibd",),
    # Left of the colon is the dataset's metric name; right is Percona's.
    # Only these two exist as INDEX metrics: `euclidean_squared`, `dot` and
    # `manhattan` are accepted by DISTANCE() but cannot be indexed at all
    # ("HNSW indexes on anything but the EUCLIDEAN or COSINE metric"), so
    # this benchmark can never show a Percona inner-product frontier.
    # Lowercase because an index option value is a bare identifier and the
    # data dictionary stores the canonical lowercase form.
    metric_names={"angular": "cosine", "euclidean": "euclidean"},
    # Required by the server on any table carrying a vector index. See
    # `MySQLDialect.pk_type`.
    pk_type="BIGINT UNSIGNED",
    # `VECTOR KEY … TYPE hnsw (opts)`. An option value is a bare identifier or
    # an integer and never a quoted string -- `metric = 'cosine'`, `M = 6.0`
    # and `M = "6"` are all parse errors. Exactly one vector index per table
    # is allowed, so there is nothing here to repeat.
    index_clause=("VECTOR KEY {name} (v) TYPE hnsw "
                  "(M = {m}, ef_construction = {ef_construction}, "
                  "metric = {metric})"),
    # Arity exactly 3; the metric argument is mandatory and matched
    # case-insensitively. Uppercase here is for the reader, not the parser.
    #
    # A cosine index serves only 'COSINE' and an L2 index only 'EUCLIDEAN' --
    # never the cross product, because the optimizer reads the metric back
    # out of the stored index option. Querying with the other one gets a
    # SILENT FULL SCAN rather than an error, which is why `metric_names` above
    # is the single source for both the DDL and this expression.
    distance_expr="DISTANCE(v, %s, '{metric_upper}')",
    # The access type, and the reason this field exists at all. See
    # `MySQLDialect.plan_marker`.
    plan_marker="vector_ann",
    # Not a default to build at: `IndexSpec.ef_construction` supplies the
    # value per build and `_index_clause` prefers it. This is what a
    # by-hand reproduction gets, and it is the server's own default.
    ef_construction=200,
)


class PerconaDriver(MySQLFamilyDriver):
    """Percona Server's InnoDB HNSW.

    A class of its own rather than a third `MySQLDialect` constant, because
    four things about this engine are behaviour and not values: `capabilities`
    varies with the build mode, there is a real repair path, `effective_ef`
    clamps, and `maintain()` has to read the index options back off the server.
    `_driver_classes()` is "one entry per architecture, not per engine", and
    those four are what make this one.
    """

    dialect = PERCONA_DIALECT
    name = "percona"

    #: Declared, not inferred: the churn workload stamps it onto the "before
    #: maintenance" record BEFORE `maintain()` runs. Percona is the only
    #: MySQL-family engine here that has a repair path at all.
    repair_path = "index-replace"

    def effective_ef(self, ef: int, k: int) -> int:
        """`max(ef, k)` -- the clamp is the variable's own contract.

        `innodb_hnsw_ef_search`'s help text states it ("the effective width is
        max(ef, LIMIT)") and the call site passes `ef_search` beside
        `std::max<size_t>(limit, 1)`, so the searched width is never below
        `LIMIT k`.

        At k=10 every grid value this repo sweeps is above 10, so this returns
        the grid verbatim and today it changes nothing. It is implemented
        rather than inherited because the inherited identity would be wrong at
        the one grid point where it mattered and silently right everywhere
        else -- and a calibration record that says "ef 5" when the engine
        looked at 10 candidates is the kind of figure a reader compares across
        engines. MongoDB clamps the same way and this hook exists for it.
        """
        return max(int(ef), int(k))

    def capabilities(self) -> Dict[str, Any]:
        """Percona is the first engine here whose build mode is a CHOICE.

        It maintains the graph on every INSERT *and* has a real bulk path
        (`ALTER TABLE … ADD VECTOR KEY` builds from one clustered scan), so
        `incremental_index` is a property of the configuration rather than of
        the engine, and it has to be reported that way.

        **This is load-bearing, and `harness/workloads/build.py` is the
        reason.** Its `incremental` flag is
        `driver.capabilities().get("incremental_index", True)` with an
        override that forces it True for `build_mode == "incremental"` -- and
        there is NO counterpart forcing it False for `build_mode == "post"`.
        That flag then decides whether `create_index` is called at all and
        what `build_wall_s` means. A class-level `incremental_index = True`
        here would make a `build_mode: post` Percona run load an
        already-indexed table, never call `create_index`, and record the load
        time as the build time -- a record reading "incremental" while the
        profile asked for a bulk build, with nothing anywhere saying so.

        Deriving it from the spec is the smaller of the two available fixes:
        the other is a line in `build.py`, which governs every engine's build
        accounting and is not this driver's to change.

        Before `create_schema` has run there is no spec to read. The honest
        answer then is the engine's unconfigured behaviour, which is
        incremental -- the graph is maintained on the write path unless a
        profile asks otherwise.
        """
        incremental = (self._index is None
                       or self._index.build_mode == "incremental")
        return {"incremental_index": incremental,
                "ef_construction_tunable": True}

    def create_schema(self, index: IndexSpec) -> None:
        """CREATE TABLE, with the vector index inline only in incremental mode.

        The split is what makes `build_mode` mean anything here. In `post`
        mode the table is created without the index and `create_index` adds it
        afterwards, so `build_wall_s` times a bulk build; in `incremental`
        mode the index is declared inline and the graph is maintained inside
        every INSERT, so the build cost IS the load cost. Creating the index
        inline in both modes would make the two modes the same measurement
        under two labels.

        pgvector's driver splits the same way and for the same reason.
        """
        self._index = index
        inline = index.build_mode == "incremental"
        ddl = self._create_table_ddl(index, with_index=inline)
        print(f"[{self.name}] {ddl}")
        self._cur.execute(ddl)

    def create_index(self, index: IndexSpec) -> None:
        """The bulk build: one `ALTER TABLE … ADD VECTOR KEY`.

        A no-op in incremental mode, where `create_schema` already declared
        the index and `build.py` does not call this at all.

        ALGORITHM=INPLACE is spelled out because the two algorithms build
        different things: INPLACE builds the graph in memory and bulk-loads
        the aux table, while a COPY-built aux is correct but slower to build
        and larger. Leaving the choice to the server would let the measured
        "bulk build" be either one.
        """
        self._index = index
        if index.build_mode == "incremental":
            return
        metric = self.dialect.metric_names[index.metric]
        sql = (f"ALTER TABLE {TABLE} ADD "
               f"{self._index_clause(index, metric)}, ALGORITHM=INPLACE")
        print(f"[{self.name}] {sql}")
        self._cur.execute(sql)

    def maintain(self) -> Optional[str]:
        """Replace the vector index in one ALTER, which compacts tombstones.

        A DELETEd row leaves a graph tombstone and there is no background
        process that collects it -- Percona has no vacuum equivalent, which is
        why `prepare_churn` and `wait_index_synced` are both inherited no-ops
        here. The rebuild collects from the live rows only, so the tombstones
        go.

        `OPTIMIZE TABLE` would also compact, and is available. It is not what
        this runs, because it rewrites the whole base table -- every row,
        every secondary index -- so its cost is dominated by work that has
        nothing to do with the vector index and the churn phase would be
        timing a table rebuild. The index replace touches only the aux table
        and reuses each row's existing label, which the suite proves by
        asserting `TABLE_ID` is unchanged across it. That is the narrower
        repair and the one whose cost means something.

        **The option list is read back off the server rather than re-derived
        from the profile, and that is the whole correctness argument.** The
        DDL that rebuilds the index decides the new graph's `M`,
        `ef_construction` and `metric`; omitting them rebuilds at the server's
        defaults -- `M=25`, not the profile's -- and the post-maintenance
        recall would then be a different index's recall, reported as the
        repaired one's. `SHOW CREATE TABLE` returns the full canonical option
        set because defaults are resolved at create time and stored, so the
        server's own answer to "what was this graph built with" is available
        and is better than trusting the DDL we sent.

        Not timed here: `churn.py` times the call. It rebuilds the whole
        graph, so on a 1M-row corpus this costs on the order of a bulk build,
        and the ALTER takes a SHARED metadata lock -- readers continue, every
        writer stops for the duration.
        """
        clause = self._index_clause_from_server()
        sql = (f"ALTER TABLE {TABLE} DROP KEY {self.dialect.index_name}, "
               f"ADD {clause}, ALGORITHM=INPLACE")
        print(f"[{self.name}] {sql}")
        self._cur.execute(sql)
        self._cur.execute("COMMIT")
        return "ALTER TABLE (index replace)"

    def _index_clause_from_server(self) -> str:
        """The live index definition, taken out of `SHOW CREATE TABLE`.

        Returns a `VECTOR KEY …` clause ready to follow `ADD`. The server
        emits the canonical form with backticks and `USING` rather than
        `TYPE` -- both spellings are accepted on the way back in, so the
        line is re-sent as the server wrote it rather than reformatted into
        the spelling this module happens to use. Reformatting is where an
        option would get dropped.

        Falls back to rebuilding the clause from the `IndexSpec` only if the
        server's definition cannot be found, and says so on the way past: a
        silent fall-back here is exactly the "rebuilt at the defaults"
        failure this method exists to avoid, except with our own numbers
        instead of the server's.
        """
        try:
            self._cur.execute(f"SHOW CREATE TABLE {TABLE}")
            ddl = str(self._cur.fetchone()[1])
            for line in ddl.splitlines():
                stripped = line.strip().rstrip(",")
                if stripped.upper().startswith("VECTOR KEY"):
                    return stripped
        except Exception as exc:
            print(f"[{self.name}] SHOW CREATE TABLE failed: {exc}")
        assert self._index is not None, "create_schema must run before maintain"
        metric = self.dialect.metric_names[self._index.metric]
        rebuilt = self._index_clause(self._index, metric)
        print(f"[{self.name}] WARNING: could not read the vector index back "
              f"out of SHOW CREATE TABLE; rebuilding from the profile as "
              f"{rebuilt!r}. If the live index was built with other options, "
              f"the post-maintenance recall is a different index's.")
        return rebuilt
