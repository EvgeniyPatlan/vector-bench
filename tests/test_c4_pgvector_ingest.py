"""C4: pgvector writes through INSERT wherever its graph is being maintained.

Two paths changed and one deliberately did not. The INCREMENTAL-mode load and
churn's re-insert both maintain the HNSW graph inside the write, so both now go
through batched INSERT at the MySQL family's own batch size; the BULK-mode load
does not, so it keeps COPY.

`build_mode` decides which comparison the build workload is making, and until
now it decided only that. The ingest underneath it was a COPY either way --
Postgres's bulk fast-path -- while MariaDB's MHNSW and AliSQL's VIDX are
ingested through `executemany(INSERT INTO ...)`. In INCREMENTAL mode that is
not a detail of the client: `workloads/build.py` records the load's wall time
AS the build time, because the graph is maintained inside those writes, so the
ingest mechanism was inside the number the report sets against MHNSW's and
VIDX's. In BULK mode it is the opposite: the index is created after the load,
the load is pure ingest, and COPY is the path anyone loading pgvector actually
uses.

Churn's re-insert is the incremental case by construction and had no bulk
exception to claim: it runs after `build.run`, so the index exists in BOTH
build modes by the time anything re-inserts, and unlike the bulk load there is
no later CREATE INDEX for a COPY to be the cheap prelude to. It fed
`churn.py`'s `reinsert_rows_per_s`, so that figure carried the same fast path.

Every assertion here runs against a recording cursor, because no engine is
installed on the host and because the thing being changed IS the statement.
What can be checked without a server is exactly what this change consists of:
which statement each mode sends, how many of them, how wide each row is, and
which values are in it. No rate, no wall time, no row-per-second figure appears
in this file -- those need a server, and a number measured against a fake
cursor would be a number about the fake cursor.

Three habits are borrowed from `tests/test_t3b_drivers.py`, each because of a
way an assertion in this repo has looked like coverage and was not:

**Nothing asserts a substring where a longer string would satisfy it.** `tag`
is a substring of `ctag`, so `"tag" in sql` is satisfied by a statement naming
only `ctag`, and -- the shape that actually matters here -- an INSERT whose two
scalar columns have been SWAPPED contains every substring the honest one does.
The statement assertions are therefore `re.fullmatch` against a literal, and
the column list is compared as an ordered list.

**No fixture hands back what it was given.** The cursor records; it never
echoes. No expected statement is read out of `postgres.py` -- the literals are
written out here, so a mutation of `INSERT_SQL` cannot drag the expectation
along with it.

**No assertion runs over an empty collection.** A load that sent nothing would
satisfy "every batch is 500 rows" and "no COPY was streamed" at the same time,
so the batch count and the row total are asserted before anything is said
about their contents.

`TestTheOldFormsDoNotCatchThese` at the bottom closes the loop the other way:
it reconstructs the weaker check each assertion replaces and shows it passing
on the mutant the strong one refuses.
"""

from __future__ import annotations

import inspect
import io
import os
import re
import sys

import numpy
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import datasets as ds  # noqa: E402
from harness.drivers import mysql_family as mysql_mod  # noqa: E402
from harness.drivers import postgres as pg_mod  # noqa: E402
from harness.drivers.base import ConnectionSpec, IndexSpec, RowAttrs  # noqa: E402

DIM = 8
TAG = ds.FILTER_TAG
CTAG = ds.FILTER_CTAG
COLUMNS = ds.FILTER_COLUMNS

#: The statements, written out rather than imported. The point of a literal
#: here is that a mutation of `postgres.INSERT_SQL` must fail this file; an
#: expectation read out of the module under test would follow the mutation.
EXPECTED_INSERT = (
    r"INSERT INTO t1 \(id, tag, ctag, embedding\) "
    r"VALUES \(%s, %s, %s, %s\)")
EXPECTED_COPY = r"COPY t1 \(id, tag, ctag, embedding\) FROM STDIN"

#: Rows per batch, here as a literal for the same reason. The guard that this
#: is the MySQL family's number reads the family's own signature; this one only
#: has to be a number the module cannot talk out of.
FAMILY_BATCH = 500

#: 1,200 rows against a 500-row batch: two full batches and a short one. A row
#: count that divided evenly would say nothing about the tail, and the tail is
#: a separate `executemany` call.
ROWS = 1200
EXPECTED_BATCH_SIZES = [500, 500, 200]


# ---------------------------------------------------------------------------
# Fakes. They record; they do not answer.
# ---------------------------------------------------------------------------

class RecordingCursor:
    """Enough of a psycopg3 cursor to record both write mechanisms.

    `executemany` and `copy` land in two separate lists on purpose: the whole
    change is which of the two a mode uses, so "it sent the right statement"
    and "it sent nothing through the other mechanism" have to be two readable
    facts rather than one. A fake that funnelled both into one list would pass
    a load that did both.
    """

    def __init__(self):
        self.statements = []        # (sql, params)
        self.many = []              # (sql, rows)
        self.copies = []            # (sql, rows)

    def execute(self, sql, params=None, **kw):
        self.statements.append((sql, params))

    def executemany(self, sql, rows):
        self.many.append((sql, list(rows)))

    def copy(self, sql):
        return _RecordingCopy(self, sql)

    def close(self):
        pass

    @property
    def sql(self):
        return [sql for sql, _ in self.statements]


class _RecordingCopy:
    def __init__(self, cursor, sql):
        self._cursor = cursor
        self._sql = sql
        self._rows = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._cursor.copies.append((self._sql, self._rows))
        return False

    def write_row(self, row):
        self._rows.append(tuple(row))


class _RecordingConnection:
    """What a worker thread's `self._open()` hands back.

    One cursor per connection, kept on the connection, so a threaded load's
    writes can be read back per worker rather than merged -- a merged view
    cannot tell "every worker inserted" from "one worker inserted four times".
    """

    def __init__(self):
        self.cursor_ = RecordingCursor()
        self.closed = False

    def cursor(self):
        return self.cursor_

    def close(self):
        self.closed = True


def spec():
    return ConnectionSpec(host="h", port=5432, user="u", database="d")


def pg_driver(build_mode="post", index=True):
    """A real `PostgresDriver` with its cursor faked.

    `index=False` leaves `_index` as the constructor leaves it -- None --
    which is the state a diagnostic script that never called `create_schema`
    is in, and a state the load path has to have an answer for.
    """
    driver = pg_mod.PostgresDriver(spec())
    driver._cur = RecordingCursor()
    if index:
        driver._index = IndexSpec(dim=DIM, m=16, metric="euclidean",
                                  build_mode=build_mode)
    return driver


def distinct_attrs(rows):
    """`tag` and `ctag` values that are never equal and never each other.

    `tag` counts up from 0 and `ctag` from 900, so a mirrored column
    (`ctag = tag`), a swapped pair and a column written from the wrong array
    are each visible in a single row rather than only in aggregate.
    """
    return RowAttrs(tag=numpy.arange(rows, dtype=numpy.int64),
                    ctag=numpy.arange(rows, dtype=numpy.int64) + 900)


def reinsert_ids(rows):
    """Fresh ids, offset from the row positions so the two cannot be confused.

    Churn re-inserts under new ids while the vectors and scalars keep their
    original positions, so an id taken from the loop counter instead of the
    caller's list is a live defect class rather than a hypothetical one --
    and ids equal to positions would hide it.
    """
    return list(range(5000, 5000 + rows))


def marked_vectors(rows):
    """Row `i` is a vector of `i`s, so a row paired with another row's vector
    is visible. Zeros would make every row's vector interchangeable."""
    return numpy.arange(rows, dtype=numpy.float32).repeat(DIM).reshape(rows, DIM)


def insert_columns_and_placeholders(sql):
    """(column names, placeholders) for an INSERT ... VALUES statement."""
    found = re.search(r"\(([^)]*)\)\s*VALUES\s*\(([^)]*)\)", sql, re.IGNORECASE)
    assert found, f"not an INSERT ... VALUES: {sql!r}"
    columns = [c.strip() for c in found.group(1).split(",")]
    placeholders = [p.strip() for p in found.group(2).split(",")]
    assert columns and placeholders, sql
    return columns, placeholders


def batches(cursor):
    """The rows of each `executemany`, with the statement checked once.

    Asserts the list is non-empty: "every batch carried 500 rows" is true of
    no batches at all, which is precisely what a load that fell back to COPY
    produces.
    """
    assert cursor.many, "nothing was sent through executemany"
    for sql, _rows in cursor.many:
        assert re.fullmatch(EXPECTED_INSERT, sql), sql
    return [rows for _sql, rows in cursor.many]


def source_of(module):
    return io.open(inspect.getsourcefile(module), encoding="utf-8").read()


# ===========================================================================
# The incremental load: batched INSERT, at the family's batch size
# ===========================================================================

class TestTheIncrementalLoadInserts:
    def test_it_sends_insert_batches_and_streams_no_copy(self):
        """The change itself.

        Honest: 3 `executemany` calls, 0 COPY streams. Mutant (`load_range`
        pinned to `copy_range`, i.e. the pre-C4 file): 0 and 1 -- and the
        number the report publishes as pgvector's incremental build cost is
        then a COPY's.
        """
        driver = pg_driver(build_mode="incremental")
        driver.load(marked_vectors(ROWS), distinct_attrs(ROWS))
        assert len(driver._cur.many) == len(EXPECTED_BATCH_SIZES)
        assert driver._cur.copies == []
        # And nothing was smuggled through plain `execute` either -- a load
        # that sent one INSERT per `execute` call would leave `many` empty and
        # could not be called batched.
        assert driver._cur.sql == []

    def test_the_insert_is_exactly_the_four_column_statement(self):
        """Regression: a column dropped, added, or the two scalars swapped.

        `re.fullmatch` against a literal, because every weaker form is
        satisfied by the swap: `INSERT INTO t1 (id, ctag, tag, embedding)`
        contains `tag`, contains `ctag`, and contains `INSERT INTO t1` --
        while the value tuples still arrive in `(id, tag, ctag, vector)`
        order, so every row's two scalar columns would be exchanged. Both
        columns are INT and NOT NULL, so the server accepts it.

        Honest: columns `['id', 'tag', 'ctag', 'embedding']`, 4 placeholders.
        Mutant (the two scalars swapped in `INSERT_SQL`):
        `['id', 'ctag', 'tag', 'embedding']`.
        """
        driver = pg_driver(build_mode="incremental")
        driver.load(marked_vectors(600), distinct_attrs(600))
        sent = {sql for sql, _rows in driver._cur.many}
        assert len(sent) == 1, sent
        sql = sent.pop()
        assert re.fullmatch(EXPECTED_INSERT, sql), sql
        columns, placeholders = insert_columns_and_placeholders(sql)
        assert columns == ["id", TAG, CTAG, "embedding"]
        assert [c for c in columns if c in COLUMNS] == list(COLUMNS)
        assert len(placeholders) == len(columns) == 2 + len(COLUMNS)
        assert set(placeholders) == {"%s"}

    def test_every_batch_but_the_last_carries_the_family_s_batch_size(self):
        """Regression: the batch size drifting from the engines this is
        compared against -- the fork this was ported from used 1000.

        Honest: [500, 500, 200] for 1,200 rows. Mutant
        (`INSERT_BATCH_ROWS = 1000`): [1000, 200], i.e. half the round trips
        for the same rows, which is a share of the advantage this change
        exists to remove.
        """
        driver = pg_driver(build_mode="incremental")
        driver.load(marked_vectors(ROWS), distinct_attrs(ROWS))
        sizes = [len(rows) for rows in batches(driver._cur)]
        assert sizes == EXPECTED_BATCH_SIZES
        assert sum(sizes) == ROWS, "rows were dropped or duplicated"

    def test_the_batch_size_is_read_from_the_mysql_family_not_chosen(self):
        """The anti-drift guard, and the only test here that reads another
        driver: the number's whole justification is that it equals the
        family's, so the family is the thing to compare it against. A literal
        500 on both sides of an assertion would pin a coincidence.

        Honest: 500 == 500. Mutant (`INSERT_BATCH_ROWS = 1000`, which is what
        the fork used and stated was the family's batch): 1000 != 500.
        """
        family = inspect.signature(
            mysql_mod.MySQLFamilyDriver._load_range).parameters["batch"].default
        assert isinstance(family, int) and family > 0, family
        assert pg_mod.INSERT_BATCH_ROWS == family
        # Pinned to the literal as well, so a change made to BOTH drivers at
        # once still gets read by a human: the two engines' batch sizes are a
        # fairness decision, not a tuning knob.
        assert family == FAMILY_BATCH

    def test_each_row_carries_its_own_id_tag_ctag_and_vector(self):
        """Regression: `ctag` written from the tag array, the two swapped, or
        a row paired with another row's vector.

        Honest: row 501 is `(501, 501, 1401, [501.0 ...])`. Mutant
        (`RowAttrs.row` returning `(tag, tag)`): `(501, 501, 501, ...)`.
        Mutant (the value tuple built `*reversed(attrs.row(i))`):
        `(501, 1401, 501, ...)`.
        """
        attrs = distinct_attrs(ROWS)
        vectors = marked_vectors(ROWS)
        driver = pg_driver(build_mode="incremental")
        driver.load(vectors, attrs)
        sent = [row for rows in batches(driver._cur) for row in rows]
        assert len(sent) == ROWS
        for i, row in enumerate(sent):
            assert len(row) == 2 + len(COLUMNS), row
            assert row[0] == i
            assert row[1] == int(attrs.tag[i]), i
            assert row[2] == int(attrs.ctag[i]), i
            assert numpy.array_equal(row[3], vectors[i]), i

    def test_start_id_still_offsets_the_ids_it_writes(self):
        """`load(start_id=...)` is how a second corpus is appended, and the id
        is now built in a second place. Honest: ids 7000..7011. Mutant (`i`
        written instead of `start_id + i` in the INSERT path): 0..11, which
        collides with the primary key of rows already there."""
        driver = pg_driver(build_mode="incremental")
        driver.load(marked_vectors(12), distinct_attrs(12), start_id=7000)
        ids = [row[0] for rows in batches(driver._cur) for row in rows]
        assert ids == list(range(7000, 7012))

    def test_a_threaded_incremental_load_inserts_on_every_worker(self):
        """Regression: the port applied to the single-threaded call site only.

        `load_threads: 8` is live in `config/profiles/tuned-complete.yml` and
        `mariadb-blog-repro.yml`, so a half-applied change is not hypothetical
        -- it is those profiles measuring COPY under a record that says
        INSERT.

        Honest: 4 worker cursors, 4 INSERT batches, 1,200 rows, 0 COPY
        streams. Mutant (the worker left calling `copy_range`): 4 COPY
        streams and no batches.
        """
        driver = pg_driver(build_mode="incremental")
        opened = []

        def fake_open():
            conn = _RecordingConnection()
            opened.append(conn)
            return conn

        driver._open = fake_open
        driver.load(marked_vectors(ROWS), distinct_attrs(ROWS), threads=4)

        assert len(opened) == 4, "one connection per worker"
        assert all(conn.closed for conn in opened), "a worker leaked a connection"
        assert driver._cur.many == [] and driver._cur.copies == [], (
            "the threaded path wrote through the driver's own cursor")
        ids = []
        for conn in opened:
            assert conn.cursor_.copies == [], "a worker streamed a COPY"
            for rows in batches(conn.cursor_):
                ids += [row[0] for row in rows]
        assert sorted(ids) == list(range(ROWS)), "rows lost, doubled or renumbered"


# ===========================================================================
# The bulk load: still COPY, and that is the other half of the correction
# ===========================================================================

class TestTheBulkLoadStillCopies:
    def test_bulk_mode_streams_one_copy_and_sends_no_insert(self):
        """The distinction, from the other side. In bulk mode the index is
        created after the load, so the load is pure ingest and COPY is
        pgvector's idiomatic path; an INSERT here would measure an ingest
        nobody uses and would make build.py's bulk-versus-incremental claim a
        comparison of two statement kinds.

        Honest: 1 COPY stream carrying 1,200 rows, 0 `executemany`. Mutant
        (`load_range = insert_range` unconditionally): 0 and 3.
        """
        driver = pg_driver(build_mode="post")
        driver.load(marked_vectors(ROWS), distinct_attrs(ROWS))
        assert driver._cur.many == []
        assert len(driver._cur.copies) == 1
        sql, written = driver._cur.copies[0]
        assert re.fullmatch(EXPECTED_COPY, sql), sql
        assert len(written) == ROWS

    def test_the_default_build_mode_copies(self):
        """`IndexSpec.build_mode` defaults to "post", and a load that reached
        the INSERT path by default would change every bulk measurement in the
        suite rather than the incremental ones. Honest: 1 COPY. Mutant (the
        test read as `!= "post"` instead of `== "incremental"`): 0 COPY and
        3 batches, for a spec nobody set a mode on."""
        driver = pg_driver()
        driver._index = IndexSpec(dim=DIM, m=16, metric="euclidean")
        driver.load(marked_vectors(600), distinct_attrs(600))
        assert len(driver._cur.copies) == 1
        assert driver._cur.many == []

    def test_a_load_with_no_index_spec_copies(self):
        """The defensive branch, pinned as defensive.

        An earlier version of this docstring said
        `scripts/probe-valkey-filter.py` and `verify-valkey-churn.py` load
        without a schema. That was wrong twice over, and measured: both call
        `create_schema(index)` immediately before `load()`
        (`probe-valkey-filter.py:143`, `verify-valkey-churn.py:190`), and
        both construct a `ValkeyDriver` -- nothing in `scripts/` builds this
        driver at all. `build.run` creates the schema first and `main.py`'s
        no-build path assigns `_index` directly, so no in-tree caller reaches
        this branch.

        It is pinned anyway, because a state with no index spec is a state
        with no HNSW index to maintain, and COPY is the right answer for it.
        Honest: 1 COPY, 0 batches. Mutant (`self._index is None or ...`):
        0 COPY and 2 batches of INSERT -- a load that treats "I was told
        nothing" as "the graph is live".
        """
        driver = pg_driver(index=False)
        assert driver._index is None, "the fixture pre-loaded a spec"
        driver.load(marked_vectors(600), distinct_attrs(600))
        assert len(driver._cur.copies) == 1
        assert driver._cur.many == []

    def test_the_two_modes_do_not_send_the_same_statement(self):
        """Stated as one fact, because the pair is the deliverable: the same
        driver, the same rows, two ingest mechanisms chosen by `build_mode`
        alone. Honest: one INSERT, one COPY, and nothing in common. Mutant
        (either branch pinned to the other): the two are equal."""
        rows = 600
        incremental = pg_driver(build_mode="incremental")
        incremental.load(marked_vectors(rows), distinct_attrs(rows))
        bulk = pg_driver(build_mode="post")
        bulk.load(marked_vectors(rows), distinct_attrs(rows))

        assert [sql for sql, _ in incremental._cur.many]
        assert [sql for sql, _ in bulk._cur.copies]
        assert incremental._cur.copies == []
        assert bulk._cur.many == []
        insert_sql = incremental._cur.many[0][0]
        copy_sql = bulk._cur.copies[0][0]
        assert insert_sql != copy_sql
        assert insert_sql.startswith("INSERT INTO t1 (")
        assert copy_sql.startswith("COPY t1 (")
        # Same rows, same width, whichever mechanism carried them: a change
        # that made one path write a different number of fields would have
        # failed in the engine, after an unknown number of rows.
        assert (len(incremental._cur.many[0][1][0])
                == len(bulk._cur.copies[0][1][0]) == 2 + len(COLUMNS))


# ===========================================================================
# Churn's re-insert: the same correction in the second place it was needed
# ===========================================================================

class TestTheChurnReinsertInserts:
    def test_it_sends_insert_batches_and_streams_no_copy(self):
        """Honest: 3 `executemany` calls, 0 COPY streams. Mutant (the shipped
        form, `insert_rows` back on `COPY_SQL`): 0 and 1, and
        `reinsert_rows_per_s` is a COPY's rate against the MySQL family's
        batched-INSERT rate."""
        driver = pg_driver(build_mode="incremental")
        driver.insert_rows(reinsert_ids(ROWS), marked_vectors(ROWS),
                           distinct_attrs(ROWS))
        assert len(driver._cur.many) == len(EXPECTED_BATCH_SIZES)
        assert driver._cur.copies == []
        # `insert_rows` sends no COMMIT: the connection autocommits, and a
        # statement here would be a statement the load path does not send.
        assert driver._cur.sql == []

    def test_it_batches_at_the_same_size_the_load_does(self):
        """Honest: [500, 500, 200]. Mutant (one `executemany` for the whole
        slice): [1200] -- a third of the round trips, 1 call where the
        family sends 3, on the path whose rate is the measurement."""
        driver = pg_driver(build_mode="incremental")
        driver.insert_rows(reinsert_ids(ROWS), marked_vectors(ROWS),
                           distinct_attrs(ROWS))
        sizes = [len(rows) for rows in batches(driver._cur)]
        assert sizes == EXPECTED_BATCH_SIZES
        assert sum(sizes) == ROWS, "rows were dropped or duplicated"

    def test_it_inserts_in_both_build_modes(self):
        """The claim that the re-insert is the incremental case BY
        CONSTRUCTION, which is why this path is not gated on `build_mode`:
        churn runs after `build.run`, so the index exists either way.

        Honest: both modes send 2 INSERT batches and stream no COPY. Mutant
        (`insert_rows` gated on `build_mode` the way `load` is): bulk mode
        streams a COPY into a live HNSW index, and only half the engines'
        churn rates are comparable.
        """
        for build_mode in ("post", "incremental"):
            driver = pg_driver(build_mode=build_mode)
            driver.insert_rows(reinsert_ids(600), marked_vectors(600),
                               distinct_attrs(600))
            assert driver._cur.copies == [], build_mode
            assert len(driver._cur.many) == 2, build_mode

    def test_it_sends_the_statement_the_incremental_load_sends(self):
        """One statement across the two INSERT paths, which is what
        `INSERT_SQL` is for -- the property the old
        `test_both_write_paths_use_one_copy_statement` held for the two COPY
        paths. Honest: character-identical. Mutant (either path given its own
        string): they differ, and the re-insert fails against a four-column
        table after the churn delete."""
        load = pg_driver(build_mode="incremental")
        load.load(marked_vectors(600), distinct_attrs(600))
        churn = pg_driver(build_mode="incremental")
        churn.insert_rows(reinsert_ids(600), marked_vectors(600),
                          distinct_attrs(600))
        assert load._cur.many[0][0] == churn._cur.many[0][0]
        assert re.fullmatch(EXPECTED_INSERT, churn._cur.many[0][0])

    def test_each_row_keeps_the_callers_id_and_its_own_tag_and_ctag(self):
        """Regression: `ctag` written from the tag array (the churn mirror),
        the two swapped, or the caller's ids replaced by positions.

        Honest: row 501 is `(5501, 501, 1401, [501.0 ...])` -- the id from
        the caller's list, the scalars and the vector from position 501 of
        the arrays handed in beside it. Mutant (ctag mirrored):
        `(5501, 501, 501, ...)`. Mutant (`i` used as the id): `(501, ...)`.
        """
        attrs = distinct_attrs(ROWS)
        vectors = marked_vectors(ROWS)
        ids = reinsert_ids(ROWS)
        driver = pg_driver(build_mode="incremental")
        driver.insert_rows(ids, vectors, attrs)
        sent = [row for rows in batches(driver._cur) for row in rows]
        assert len(sent) == ROWS
        for i, row in enumerate(sent):
            assert len(row) == 2 + len(COLUMNS), row
            assert row[0] == ids[i]
            assert row[1] == int(attrs.tag[i]), i
            assert row[2] == int(attrs.ctag[i]), i
            assert numpy.array_equal(row[3], vectors[i]), i


# ===========================================================================
# The prose, which is part of the deliverable
# ===========================================================================

class TestTheReasoningIsWrittenDown:
    def test_the_bulk_exception_is_argued_where_the_statement_is(self):
        """A reader who deletes the `build_mode` test because "both paths may
        as well use INSERT" has to find the argument against it in the file,
        not in a commit message. Two phrases, both load-bearing: that bulk
        keeps COPY deliberately, and where the defect was found."""
        source = source_of(pg_mod)
        assert "BULK mode keeps COPY" in source
        assert "villagesql/vector-bench" in source

    def test_the_copy_comment_says_it_has_one_caller_left(self):
        """`COPY_SQL`'s comment said it served "the bulk load and churn's
        re-insert" and justified itself by the sharing: one string cannot
        disagree with itself about a column list. Both of those sentences are
        now false -- the re-insert sends INSERTs, so COPY has one caller and
        the parity it used to provide has to be asserted between two strings
        instead. A comment left making the old argument would be the kind of
        wrong prose that outlives the code."""
        source = source_of(pg_mod)
        assert "and now its only caller" in source
        # Single-line phrases, because the comment is reflowed at 79 columns
        # and `#: ` sits between the lines: a phrase that reads continuously
        # on the page is not a substring of the file.
        assert "What moved rather than went away is the PARITY" in source
        assert "One COPY statement for the bulk load and" not in source

    def test_what_the_change_does_not_control_for_is_stated(self):
        """The bound on the published claim: the paragraph a reader will
        quote, and the one most likely to be softened into a hedge.

        **This asserts the FACTS and not the headings, and the difference was
        measured.** The first version of this test asserted
        `"CONTROLLED, after this change"`, `"NOT CONTROLLED"` and
        `"wire encoding"` -- three memberships, all of them on labels. A
        review replaced the whole paragraph with "broadly speaking, things
        are now more comparable than they were" and "there may be minor
        residual differences, including possibly something to do with the
        wire encoding, but these are expected to be small", kept those three
        substrings, and this test PASSED: 3 passed in that class, 832 passed
        across every file that touches `postgres.py`. That is the
        membership-not-value shape, in the guard over the one paragraph that
        bounds what may be published.

        So each assertion below names a fact that a hedge cannot carry: the
        two client libraries by name, the call whose semantics the claim
        turns on, the batch size the parity rests on, the word the round-trip
        claim is pinned to, the quantity that is NOT controlled, and which
        way the residual leans. The phrases are single-line, because the
        comment is reflowed at 79 columns with `#: ` between the lines.

        **And they are asserted per SIDE of the paragraph, not against the
        paragraph or the file.** Three layers of the same vacuity were
        measured in this one guard, each a weaker claim than it looked:

        1. headings only -- the hedge above kept all three and passed;
        2. seven facts over the whole module -- `"psycopg3"`,
           `"executemany"` and `"Connector/C"` are in the module docstring
           and in the code below, so three of seven were falsely satisfied
           with every fact in the paragraph deleted (it caught the other
           four);
        3. seven facts over the paragraph -- which pins VOCABULARY and
           cannot tell CONTROLLED from NOT CONTROLLED, i.e. cannot see the
           distinction the paragraph exists to make. Two reviews got through
           it with all seven phrases present and 24 of 24 passing: one
           name-dropping them as "questions readers sometimes ask", and one
           **restoring the round-0 defect** -- `SERVER-SIDE EXECUTIONS`
           moved back into CONTROLLED ("so the server is asked to do the
           same number of things on each side") with NOT CONTROLLED reduced
           to "Nothing in that leans AGAINST pgvector in any way that would
           survive rounding". That is the conflation and the reversed sign,
           reinstated and green.

        So the block is split at `NOT CONTROLLED` and each fact is required
        on the side that makes it true. The assertion that does the work is
        the ABSENCE one: `SERVER-SIDE EXECUTIONS` must not appear in the
        controlled half, because claiming it there is precisely the error
        that shipped in round 0 and the error the reviewer could re-introduce
        at will.
        """
        source = source_of(pg_mod)
        opening = "#: **What this controls for and what it cannot**"
        assert opening in source, "the paragraph's opening is gone"
        start = source.index(opening)
        block = source[start:source.index("INSERT_SQL = (", start)]
        controlled, boundary, uncontrolled = block.partition("NOT CONTROLLED")
        assert boundary, "the paragraph no longer has two sides"
        # **`partition` splits on the FIRST occurrence, so without this line
        # the author chooses where the split is.** Measured: a decoy planted
        # mid-sentence in the summary -- "Very little is NOT CONTROLLED once
        # that holds." -- after the two controlled-side facts and before the
        # conflation leaves a 333-character prefix holding `500 on both
        # sides` and `ROUND TRIPS` and nothing else, so the claim that the
        # batch size fixes the per-row SERVER-SIDE EXECUTIONS lands in the
        # `uncontrolled` slice, where that phrase is REQUIRED rather than
        # refused. Round 0's conflation and its flattened sign then read as
        # CONTROLLED text and every assertion below passes: 24 of 24.
        assert block.count("NOT CONTROLLED") == 1, (
            "the paragraph says NOT CONTROLLED %d times, so which text counts "
            "as controlled is the author's choice rather than the structure's"
            % block.count("NOT CONTROLLED"))
        # What may be claimed as controlled, and in what terms.
        for fact in ("500 on both sides", "ROUND TRIPS"):
            assert fact in controlled, fact
        # What may NOT be claimed there. One line, and it is the one that
        # refuses the defect: round 0 said the batch size controls "the
        # number of client/server exchanges per row" while the next paragraph
        # said psycopg3 executes per row -- true of round trips, false of
        # executions, and the text never said which it meant.
        assert "SERVER-SIDE EXECUTIONS" not in controlled, (
            "SERVER-SIDE EXECUTIONS are claimed on the CONTROLLED side; "
            "psycopg3 executes once per row whatever the batch size, so only "
            "round trips are controlled. The half that says otherwise:\n"
            + controlled)
        # What is uncontrolled: the quantity, the two clients by name, the
        # call the claim turns on, and which way the residual leans.
        for fact in ("SERVER-SIDE EXECUTIONS", "psycopg3", "executemany",
                     "Connector/C", "AGAINST pgvector"):
            assert fact in uncontrolled, fact
        # A floor last, so a hedge fails naming the fact it dropped rather
        # than its length. It ran first once and failed a measured hedge on
        # the length alone -- a true failure with a diagnostic about size
        # instead of about the missing claim. No number is quoted for that,
        # because the slice it was measured against has since been widened to
        # span the whole three-paragraph block and the figure no longer
        # reproduces.
        assert len(block) > 400, len(block)


# ===========================================================================
# R2: the old forms do not catch these
# ===========================================================================

class TestTheOldFormsDoNotCatchThese:
    """A repair is only a repair if the old form passes the mutant the new
    form catches. Each mutant is built here as a string or a list, so the
    comparison is between two assertion FORMS over the same input -- no
    mutation of the module is needed to show that the weak form is blind."""

    #: The swap: both scalar columns present, both spellings present, every
    #: substring of the honest statement present, and every row's `tag` and
    #: `ctag` exchanged on the way into the table.
    SWAPPED = ("INSERT INTO t1 (id, ctag, tag, embedding) "
               "VALUES (%s, %s, %s, %s)")

    def test_a_substring_check_passes_the_swapped_column_list(self):
        # The weak form, as it is usually written.
        assert "tag" in self.SWAPPED
        assert "ctag" in self.SWAPPED
        assert "INSERT INTO t1" in self.SWAPPED
        # Even with word boundaries, which is the fix for the `tag`/`ctag`
        # substring trap and does nothing about order.
        assert re.search(r"\btag\b", self.SWAPPED)
        assert re.search(r"\bctag\b", self.SWAPPED)
        # The form this file uses.
        assert not re.fullmatch(EXPECTED_INSERT, self.SWAPPED)
        columns, _ = insert_columns_and_placeholders(self.SWAPPED)
        assert columns != ["id", TAG, CTAG, "embedding"]
        # And the set-wise check, which is the other weak form: the swap is a
        # permutation, so comparing sets cannot see it.
        assert set(columns) == {"id", TAG, CTAG, "embedding"}

    def test_a_membership_check_passes_the_drifted_batch_size(self):
        """`INSERT_BATCH_ROWS = 1000` against a family that batches at 500.
        The weak form asks whether the load batched at all; the strong form
        asks what the batch was."""
        honest = [500, 500, 200]
        mutant = [1000, 200]
        # Weak: it batched, and no row was lost.
        assert len(mutant) > 1 and sum(mutant) == ROWS == sum(honest)
        # Weak: every batch is at most the size the driver names.
        assert all(size <= 1000 for size in mutant)
        # Strong, which is what the test above asserts.
        assert mutant != EXPECTED_BATCH_SIZES
        assert honest == EXPECTED_BATCH_SIZES

    def test_counting_statements_passes_a_load_that_did_both(self):
        """Why `copies == []` is asserted next to the batch count rather than
        instead of it: a load that streamed a COPY *and* sent the INSERTs
        satisfies "it inserted"."""
        cursor = RecordingCursor()
        cursor.executemany("INSERT INTO t1 (id, tag, ctag, embedding) "
                           "VALUES (%s, %s, %s, %s)", [(0, 0, 900, None)])
        with cursor.copy("COPY t1 (id, tag, ctag, embedding) FROM STDIN") as c:
            c.write_row((0, 0, 900, None))
        assert cursor.many, "the weak form: something was inserted"
        assert cursor.copies != [], "and a COPY went out as well"

    def test_the_old_both_paths_copy_form_passes_the_shipped_defect(self):
        """The assertion this change replaced in `test_t3b_drivers.py`:
        `bulk._cur.copies[0][0] == churn._cur.copies[0][0]`. It compared the
        two write paths and was SATISFIED BY the unfairness -- it required
        both paths to be COPYs, so the shipped form passed it by definition
        and any fix had to fail it. The replacement compares the two column
        LISTS across two statement kinds, which is the property that was
        actually wanted."""
        shipped_copy = "COPY t1 (id, tag, ctag, embedding) FROM STDIN"
        # The old form, over the shipped behaviour: both paths streamed this.
        assert shipped_copy == shipped_copy
        # The new form, over the same two statements, still holds -- and also
        # holds once one of them becomes an INSERT, which the old form cannot
        # express at all.
        insert = "INSERT INTO t1 (id, tag, ctag, embedding) VALUES (%s, %s, %s, %s)"
        copy_cols = re.search(r"COPY \w+ \(([^)]*)\)", shipped_copy).group(1)
        assert [c.strip() for c in copy_cols.split(",")] == \
            insert_columns_and_placeholders(insert)[0]

    def test_an_empty_load_satisfies_every_claim_about_its_batches(self):
        """Why `batches()` asserts its own input is non-empty. A driver that
        wrote nothing passes "every batch is 500 rows" and "no COPY was
        streamed" simultaneously."""
        cursor = RecordingCursor()
        assert all(len(rows) == FAMILY_BATCH for _sql, rows in cursor.many)
        assert cursor.copies == []
        with pytest.raises(AssertionError):
            batches(cursor)
