"""Guards for wiring Percona Server in as the seventh engine (docs/09 §4).

`tests/test_e1_percona_design.py` guards the DESIGN -- the document, the
long-lead artefacts it wrote, and every `file:line` it cites. This file guards
the WIRING: the dialect fields, the driver, the registry entries, the two
scripts, the render branch and the estimator change that make the engine
reachable from a command line.

WHAT THESE TESTS CAN AND CANNOT ESTABLISH
-----------------------------------------
Nothing here establishes anything about Percona Server. No image has been built
and no server has been started; every claim about the engine's behaviour is
sourced to a file:line on commit 02a8b80c70b0ccf6c9035d8a2307e7fa988cc304 in
docs/09 and is settled by reading that tree. What these establish is that the
statements this repo would SEND are the ones the document says it sends, and
that the seven-engine wiring agrees with itself across the files that have to.

Two shapes recur below and both are deliberate:

* **Driver behaviour is exercised, not grepped.** The drivers are instantiated
  against a recording fake cursor and the SQL they emit is read back out of it.
  A substring test for `"BIGINT UNSIGNED"` passes against a file that also
  still says `id INT PRIMARY KEY` somewhere, which is exactly the DDL bug it
  would be looking for.
* **Every assertion about Percona is paired with the same assertion about
  MariaDB or AliSQL.** The dialect fields all default to what those two
  already did, so "Percona emits X" is only half the claim; "and MariaDB still
  emits what it emitted" is the half that says no existing engine moved.
"""

from __future__ import annotations

import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.drivers.base import (ConnectionSpec, EngineDriver,  # noqa: E402
                                  IndexSpec)
from harness.drivers.mysql_family import (ALISQL_DIALECT,  # noqa: E402
                                          MARIADB_DIALECT, PERCONA_DIALECT,
                                          AliSQLDriver, MariaDBDriver,
                                          MySQLDialect, PerconaDriver)
from harness.drivers.postgres import _driver_classes, _driver_table  # noqa: E402

VB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PINNED_SHA = "02a8b80c70b0ccf6c9035d8a2307e7fa988cc304"


# ---------------------------------------------------------------------------
# Prose guards: a vocabulary, not a sentence
# ---------------------------------------------------------------------------
#
# Every prose claim below is pinned as a REGEX over whitespace-normalised text,
# fixing subject, mechanism and polarity, with the rewordings it must accept
# and the attacks it must reject written beside it and asserted. This is the
# shape `tests/test_e3_rollout_doc.py` uses, and it is here because the first
# version of these guards matched exact phrases and **fired on correct prose**:
# `"all seven engines"` rejected "all seven of the engines", "every one of the
# seven engines" and "all 7 engines"; `"six of those seven"` rejected "covers
# six engines of the seven". A guard that fires on a correct rewrite teaches
# the next author to delete it, which is how the vacuity shapes this repo keeps
# finding got here in the first place.
#
# Normalising whitespace is necessary and not sufficient: a reflow was the
# first failure mode found, an exact phrase is the second.

PROSE_CLAIMS = {
    "the engine count is seven": dict(
        pattern=r"\b(all|every one of|each of)\b[^.]{0,24}?"
                r"\b(seven|7)\b[^.]{0,24}?\bengines?\b",
        accepts=[
            "all seven engines, which is what that command measures",
            "all seven of the engines",
            "every one of the seven engines",
            "all 7 engines",
            "each of the seven engines is measured",
        ],
        rejects=[
            "all six engines, which is what that command measures",
            "all six of the engines",
            "every one of the six engines",
        ]),
    "the ingest figure covers six of the seven": dict(
        pattern=r"\bsix\b[^.]{0,40}?\bseven\b"
                r"|\bseven\b[^.]{0,20}?\bminus one\b"
                r"|\bsix of\b[^.]{0,30}?\bengines?\b",
        accepts=[
            "The ingest figure covers six of those seven",
            "the ingest figure covers six engines of the seven",
            "six of the seven engines have a measured ingest rate",
            "that figure is six of seven engines",
        ],
        rejects=[
            "The ingest figure covers all seven engines",
            "nothing has measured any engine's ingest rate",
        ]),
    "percona does either build mode": dict(
        pattern=r"Percona(\s+Server)?\b[^.]{0,80}?"
                r"\b(does either|either mode|both|can do both|does both)\b",
        accepts=[
            "Percona Server does either, so its two build modes are a "
            "comparison within one engine",
            "Percona Server can do both, so build mode is an axis within "
            "one engine",
            "uniquely, Percona Server does both",
            "Percona Server builds either mode on request",
        ],
        rejects=[
            "pgvector bulk-builds; MHNSW and VIDX build on every INSERT.",
            "Percona Server builds incrementally like MHNSW and VIDX.",
        ]),
    "the total covers only the engines with a rate": dict(
        pattern=r"\b(\d+) of (\d+) engine",
        accepts=[
            "the total above is 6 of 7 engine(s) and excludes it",
            "that total covers 6 of 7 engines",
        ],
        rejects=[
            "the total above excludes percona",
            "no measured ingest rate for percona.",
        ]),
    "the real cost is higher than the printed total": dict(
        pattern=r"\b(higher|greater|larger|more)\b[^.]{0,40}?"
                r"\bthan\b[^.]{0,40}?\b(number|figure|total)\b"
                r"|\b(number|figure|total)\b[^.]{0,40}?"
                r"\bunderstates?\b",
        accepts=[
            "the ingest cost of this plan is higher than the number above, "
            "by an amount nothing here knows",
            "the real ingest cost is greater than the figure above",
            "the total above understates this plan by an unknown amount",
        ],
        rejects=[
            "No measured ingest rate for percona.",
            "percona is excluded from the total.",
        ]),
}


def _flat(text: str) -> str:
    """Whitespace-normalised, so a reflow cannot defeat a prose guard."""
    return " ".join(text.split())


def _prose(name: str, haystack: str):
    """Assert a named claim is still made, however it is worded."""
    entry = PROSE_CLAIMS[name]
    found = re.search(entry["pattern"], _flat(haystack), re.I)
    assert found, (
        f"this claim is no longer made: {name}\n"
        f"  pattern: {entry['pattern']}\n"
        f"  the pattern is a vocabulary, not a sentence -- if the claim is "
        f"still true and only the wording moved, widen PROSE_CLAIMS rather "
        f"than restoring the old phrasing")
    return found


class TestTheProseGuardsAreNeitherBrittleNorVacuous:
    """Each vocabulary against its own accepts and rejects.

    Without this the patterns are untested assertions about assertions. It is
    the check that caught the exact-phrase versions: three legitimate
    rewordings of the engine count and one of the ingest scope were rejected
    by guards that had looked fine in review.
    """

    @pytest.mark.parametrize("name", sorted(PROSE_CLAIMS))
    def test_every_accepted_wording_matches(self, name):
        entry = PROSE_CLAIMS[name]
        assert entry["accepts"], name
        for wording in entry["accepts"]:
            assert re.search(entry["pattern"], _flat(wording), re.I), (
                f"{name}: fires on correct prose: {wording!r}")

    @pytest.mark.parametrize("name", sorted(PROSE_CLAIMS))
    def test_every_rejected_wording_fails(self, name):
        entry = PROSE_CLAIMS[name]
        assert entry["rejects"], name
        for wording in entry["rejects"]:
            assert not re.search(entry["pattern"], _flat(wording), re.I), (
                f"{name}: accepts the attack: {wording!r}")


# ---------------------------------------------------------------------------
# A recording cursor, so the SQL a driver sends is the thing under test
# ---------------------------------------------------------------------------

class FakeCursor:
    """Records every statement and replays canned answers.

    `fetchone` returns whatever was queued, so `SHOW CREATE TABLE` can be made
    to answer the way a server would without one being reachable.
    """

    def __init__(self, answers=None):
        self.statements = []
        self.params = []
        self._answers = list(answers or [])
        self._last = None

    def execute(self, sql, params=None):
        self.statements.append(sql)
        self.params.append(params)
        self._last = self._answers.pop(0) if self._answers else None

    def executemany(self, sql, rows):
        self.statements.append(sql)

    def fetchone(self):
        return self._last

    def fetchall(self):
        return self._last or []

    def close(self):
        pass

    @property
    def sql(self):
        return "\n".join(self.statements)


def driver_with(cls, answers=None, index=None):
    driver = cls(ConnectionSpec(host="h", port=3306))
    driver._cur = FakeCursor(answers)
    if index is not None:
        driver._index = index
    return driver


def spec(**kw):
    base = dict(dim=8, m=16, metric="euclidean", build_mode="post")
    base.update(kw)
    return IndexSpec(**base)


# ---------------------------------------------------------------------------
# §4.1 -- the five new Dialect fields, each defaulted so nothing else moves
# ---------------------------------------------------------------------------

class TestTheNewDialectFieldsDefaultToTodaysBehaviour:
    """Five fields were added to a dataclass two live engines share.

    The risk is not that Percona gets them wrong; it is that MariaDB or AliSQL
    silently gets a different statement because a default was chosen to suit
    the new engine. Each test below therefore pins the OLD engines' value.
    """

    NEW_FIELDS = ("pk_type", "index_clause", "distance_expr", "plan_marker",
                  "ef_construction")

    @pytest.mark.parametrize("field", NEW_FIELDS)
    def test_the_field_exists_and_a_bare_dialect_gets_the_default(self, field):
        bare = MySQLDialect(name="probe")
        assert hasattr(bare, field), field

    def test_a_bare_dialect_is_mariadbs_spelling(self):
        """The defaults ARE MariaDB's statements, which is what makes adding
        the fields a no-op for the engines that predate them."""
        bare = MySQLDialect(name="probe")
        assert bare.pk_type == "INT"
        assert bare.index_clause == (
            "VECTOR INDEX {name} (v) M={m} DISTANCE={metric}")
        assert bare.distance_expr == "vec_distance_{metric}(v, %s)"
        assert bare.plan_marker == ""
        assert bare.ef_construction is None

    @pytest.mark.parametrize("dialect", [MARIADB_DIALECT, ALISQL_DIALECT])
    def test_neither_existing_engine_sets_any_of_them(self, dialect):
        """Not "the defaults are right" but "neither dialect overrides them",
        which is the stronger statement: a dialect that set `pk_type="INT"`
        explicitly would pass a value check and would still be a file someone
        has to keep in step."""
        bare = MySQLDialect(name="probe")
        for field in self.NEW_FIELDS:
            assert getattr(dialect, field) == getattr(bare, field), field

    def test_percona_sets_every_one_of_them(self):
        """The mirror image, and the reason the fields exist at all."""
        bare = MySQLDialect(name="probe")
        differs = [f for f in self.NEW_FIELDS
                   if getattr(PERCONA_DIALECT, f) != getattr(bare, f)]
        assert sorted(differs) == sorted(self.NEW_FIELDS), differs


class TestTheDDLEachEngineActuallySends:
    """`create_schema` is exercised and its statement read back."""

    @pytest.mark.parametrize("cls,expected_pk", [(MariaDBDriver, "id INT"),
                                                 (AliSQLDriver, "id INT")])
    def test_the_existing_engines_still_declare_an_int_key(self, cls,
                                                           expected_pk):
        driver = driver_with(cls)
        driver.create_schema(spec())
        assert expected_pk in driver._cur.sql
        assert "BIGINT UNSIGNED" not in driver._cur.sql

    def test_percona_declares_the_key_the_server_requires(self):
        """`BIGINT UNSIGNED`, single column. The server refuses `BIGINT`,
        `INT UNSIGNED` and composite keys on any table with a vector index."""
        driver = driver_with(PerconaDriver)
        driver.create_schema(spec(build_mode="incremental"))
        sql = driver._cur.sql
        assert "id BIGINT UNSIGNED PRIMARY KEY" in sql
        # The negative that matters: the old literal must be gone, not merely
        # joined by a new one.
        assert "id INT PRIMARY KEY" not in sql

    def test_the_existing_engines_still_send_the_mariadb_index_clause(self):
        driver = driver_with(MariaDBDriver)
        driver.create_schema(spec(m=24, metric="angular"))
        assert "VECTOR INDEX vi (v) M=24 DISTANCE=cosine" in driver._cur.sql

    def test_percona_sends_a_parenthesised_option_list(self):
        """`TYPE hnsw (M = n, ef_construction = n, metric = name)`.

        An option value is a bare identifier or an integer and never a quoted
        string -- `metric = 'cosine'`, `M = 6.0` and `M = "6"` are all parse
        errors on this server -- so the absence of quotes is asserted too.
        """
        driver = driver_with(PerconaDriver)
        driver.create_schema(spec(m=24, metric="angular",
                                  ef_construction=320,
                                  build_mode="incremental"))
        sql = driver._cur.sql
        assert ("VECTOR KEY vi (v) TYPE hnsw "
                "(M = 24, ef_construction = 320, metric = cosine)") in sql
        assert "metric = 'cosine'" not in sql
        assert 'metric = "cosine"' not in sql

    def test_percona_takes_ef_construction_from_the_spec_not_the_dialect(self):
        """The dialect's 200 is what a by-hand reproduction gets; a profile
        that asks for something else must reach the DDL, or an
        ef_construction sweep would silently run one value seven times."""
        assert PERCONA_DIALECT.ef_construction == 200
        driver = driver_with(PerconaDriver)
        driver.create_schema(spec(ef_construction=64,
                                  build_mode="incremental"))
        assert "ef_construction = 64" in driver._cur.sql
        assert "ef_construction = 200" not in driver._cur.sql

    def test_percona_falls_back_to_the_dialect_when_the_spec_says_nothing(self):
        driver = driver_with(PerconaDriver)
        driver.create_schema(spec(ef_construction=None,
                                  build_mode="incremental"))
        assert "ef_construction = 200" in driver._cur.sql

    @pytest.mark.parametrize("cls", [MariaDBDriver, AliSQLDriver,
                                     PerconaDriver])
    def test_every_engine_indexes_both_filter_columns_the_same_way(self, cls):
        """The parity `tag` and `ctag` have is the measurement, not tidiness:
        the two columns exist to be compared, and an optimiser offered an
        index on one and not the other is choosing between plans rather than
        between filter correlations."""
        driver = driver_with(cls)
        driver.create_schema(spec(build_mode="incremental"))
        sql = driver._cur.sql
        assert "KEY tag_idx (tag)" in sql
        assert "KEY ctag_idx (ctag)" in sql


class TestTheSelectEachEngineActuallySends:

    def test_the_existing_engines_still_use_one_function_per_metric(self):
        for cls, want in ((MariaDBDriver, "vec_distance_cosine(v, %s)"),
                          (AliSQLDriver, "vec_distance_cosine(v, %s)")):
            driver = driver_with(cls, index=spec(metric="angular"))
            assert want in driver._select(10, False)

    def test_percona_passes_the_metric_as_a_third_argument(self):
        """`DISTANCE()` is registered with arity 3 and the metric is
        mandatory. Uppercase here is for the reader -- the server matches it
        case-insensitively."""
        driver = driver_with(PerconaDriver, index=spec(metric="angular"))
        assert "DISTANCE(v, %s, 'COSINE')" in driver._select(10, False)

    def test_the_metric_stays_a_literal_and_not_a_bound_parameter(self):
        """One placeholder in the unfiltered form and two in the filtered one,
        for every engine. Making the metric a parameter would add a third and
        reorder every call site -- `query_filtered` binds
        `(threshold, vector)` because the predicate precedes the ORDER BY."""
        for cls in (MariaDBDriver, AliSQLDriver, PerconaDriver):
            driver = driver_with(cls, index=spec(metric="euclidean"))
            assert driver._select(10, False).count("%s") == 1, cls
            assert driver._select(10, True).count("%s") == 2, cls

    def test_percona_builds_the_index_and_queries_it_with_one_metric_name(self):
        """A cosine index serves only 'COSINE' and an L2 index only
        'EUCLIDEAN' -- never the cross product, because the optimizer reads
        the metric back out of the stored index option. A mismatch is a
        SILENT FULL SCAN rather than an error, so both statements must come
        from the same `metric_names` entry.
        """
        for metric, want in (("angular", "cosine"), ("euclidean", "euclidean")):
            driver = driver_with(PerconaDriver)
            driver.create_schema(spec(metric=metric, build_mode="incremental"))
            assert f"metric = {want}" in driver._cur.sql
            assert f"'{want.upper()}'" in driver._select(10, False)

    def test_only_the_two_indexable_metrics_are_offered(self):
        """`euclidean_squared`, `dot` and `manhattan` are accepted by
        DISTANCE() and cannot be indexed, so this benchmark can never show a
        Percona inner-product frontier. The map is the place that decides."""
        assert set(PERCONA_DIALECT.metric_names) == {"angular", "euclidean"}
        assert set(PERCONA_DIALECT.metric_names.values()) == {"cosine",
                                                              "euclidean"}


# ---------------------------------------------------------------------------
# §4.1 -- the plan check, which is vacuous for this engine without a marker
# ---------------------------------------------------------------------------

class TestThePlanCheckIsNotSatisfiedByTheIndexNameAlone:
    """The defect this field exists for, reproduced against the driver.

    Percona's own recorded output (`…/percona_vector/r/optimizer.result:
    491-492`) contains a plan with access type `ALL`, `key=NULL` and
    `Extra=Using filesort` -- a full table scan -- on a table whose only index
    is a `VECTOR KEY`, with the index name sitting in `possible_keys`. A
    substring test against the joined EXPLAIN text returns True for it.

    The rows below are that plan and its passing counterpart (`:95-96`),
    transcribed, and fed to the real `explain_uses_vector_index`.
    """

    #: A brute-force scan whose plan text still contains the index name.
    SCAN = [(1, "SIMPLE", "t1", None, "ALL", "vi", None, None, None,
             100, 100.00, "Using filesort")]
    #: The vector path actually taken.
    ANN = [(1, "SIMPLE", "t1", None, "vector_ann", "vi", "vi", None, None,
            10, 100.00, "Using where")]

    def _ask(self, cls, rows):
        driver = driver_with(cls, answers=[rows], index=spec())
        return driver.explain_uses_vector_index([0.0] * 8, 10)

    def test_percona_rejects_the_full_scan_that_names_the_index(self):
        assert self._ask(PerconaDriver, self.SCAN) is False

    def test_percona_accepts_the_plan_that_took_the_vector_path(self):
        assert self._ask(PerconaDriver, self.ANN) is True

    def test_the_index_name_alone_would_have_accepted_the_scan(self):
        """NON-VACUITY, and the reason `plan_marker` is not cosmetic.

        Measured against the engines that have no marker: given the very same
        brute-force plan, MariaDB's and AliSQL's check -- which is the check
        Percona would have inherited -- returns True. That is brute force
        published as ANN search, and it is the difference the marker makes.
        """
        assert self._ask(MariaDBDriver, self.SCAN) is True
        assert self._ask(AliSQLDriver, self.SCAN) is True
        assert self._ask(PerconaDriver, self.SCAN) is False

    def test_the_marker_alone_is_not_enough_either(self):
        """Both conjuncts. The access type is the load-bearing half; the name
        is kept so a second vector index on the table could not satisfy the
        check for the wrong one."""
        other_index = [(1, "SIMPLE", "t1", None, "vector_ann", "other",
                        "other", None, None, 10, 100.00, "Using where")]
        assert self._ask(PerconaDriver, other_index) is False

    @pytest.mark.parametrize("cls", [MariaDBDriver, AliSQLDriver,
                                     PerconaDriver])
    def test_every_engine_pins_the_traditional_explain_format(self, cls):
        """`plan_marker` is a value in the `type` column, which only the
        tabular format has. `explain_format` defaults to TRADITIONAL so a bare
        EXPLAIN worked -- but it is settable and reachable through
        `VB_SERVER_ARGS`, and under TREE or JSON the marker vanishes and the
        check records a FALSE NEGATIVE about a server that did use the index.
        That lands in the report's Validity section.
        """
        driver = driver_with(cls, answers=[self.ANN], index=spec())
        driver.explain_uses_vector_index([0.0] * 8, 10)
        assert driver._cur.statements[0].startswith(
            "EXPLAIN FORMAT=TRADITIONAL ")

    def test_the_marker_is_read_from_the_type_column_and_not_the_text(self):
        """The last substring in the chain, closed.

        `plan_marker in plan` over the joined row text is satisfied by the
        marker appearing ANYWHERE -- a table named `vector_ann`, the string in
        `Extra`, an index called `vector_ann_idx`. None of those is reachable
        for the statements this driver sends (`t1`, fixed columns), so this is
        a class being closed rather than a live defect being fixed. It is
        worth closing because the argument for `plan_marker` existing at all
        is that a substring check accepted a brute-force scan.

        Each row below is a full table scan -- type `ALL`, `key` NULL -- with
        the marker smuggled into a different column.
        """
        smuggled = {
            "table name": (1, "SIMPLE", "vector_ann", None, "ALL", "vi",
                            None, None, None, 100, 100.00, "Using filesort"),
            "possible_keys": (1, "SIMPLE", "t1", None, "ALL", "vi,vector_ann",
                               None, None, None, 100, 100.00, "Using filesort"),
            "Extra": (1, "SIMPLE", "t1", None, "ALL", "vi", None, None, None,
                       100, 100.00, "Using filesort; vector_ann unusable"),
            "ref": (1, "SIMPLE", "t1", None, "ALL", "vi", None, None,
                     "vector_ann", 100, 100.00, "Using filesort"),
        }
        for where, row in smuggled.items():
            driver = driver_with(PerconaDriver, answers=[[row]], index=spec())
            assert driver.explain_uses_vector_index([0.0] * 8, 10) is False, \
                where
            # Non-vacuity for each: the text form WOULD have accepted it, so
            # these rows really do exercise the difference.
            plan = " | ".join(str(c) for c in row)
            assert "vector_ann" in plan, where

    def test_the_access_type_is_matched_case_insensitively(self):
        """A server that capitalised an access type would otherwise turn a
        used index into a reported full scan."""
        row = (1, "SIMPLE", "t1", None, " VECTOR_ANN ", "vi", "vi", None,
               None, 10, 100.00, "Using where")
        driver = driver_with(PerconaDriver, answers=[[row]], index=spec())
        assert driver.explain_uses_vector_index([0.0] * 8, 10) is True

    def test_an_unexpected_row_shape_falls_back_rather_than_reporting_false(
            self, capsys):
        """A `False` here is published as "this engine did not use its index".

        An EXPLAIN shape with fewer than twelve columns is a statement about
        the FORMAT, not about the engine, so the method falls back to the text
        match it used before -- no plan that passed can start failing -- and
        says on stderr that it did.
        """
        short = (1, "SIMPLE", "t1", "vector_ann", "vi")
        driver = driver_with(PerconaDriver, answers=[[short]], index=spec())
        assert driver.explain_uses_vector_index([0.0] * 8, 10) is True
        out = " ".join(capsys.readouterr().out.split())
        assert "falling back to a text match" in out
        # And a short row that does NOT carry the marker is still False, so
        # the fallback is a fallback and not an unconditional pass.
        driver = driver_with(PerconaDriver,
                             answers=[[(1, "SIMPLE", "t1", "ALL", "vi")]],
                             index=spec())
        assert driver.explain_uses_vector_index([0.0] * 8, 10) is False

    def test_the_markerless_engines_are_untouched_by_any_of_this(self):
        """MariaDB and AliSQL declare no marker, so the access-type path is
        never entered and their check is exactly what it was."""
        for cls in (MariaDBDriver, AliSQLDriver):
            assert cls.dialect.plan_marker == ""
            short = (1, "SIMPLE", "t1", "vi")
            driver = driver_with(cls, answers=[[short]], index=spec())
            assert driver.explain_uses_vector_index([0.0] * 8, 10) is True

    def test_a_failed_check_names_what_it_looked_for(self, capsys):
        """"Not used" over two conjuncts does not say which failed, and the
        two mean different things: a missing index name is the optimiser
        taking another index, a missing access type with the name present is
        the brute-force scan."""
        self._ask(PerconaDriver, self.SCAN)
        # Whitespace-normalised: the message is built from several adjacent
        # f-string fragments, and re-wrapping those fragments is an edit that
        # must not be able to defeat this guard.
        out = " ".join(capsys.readouterr().out.split())
        assert "vector_ann" in out and "'vi'" in out


# ---------------------------------------------------------------------------
# §2.5 / §4.1 -- capabilities must follow the build mode, not the class
# ---------------------------------------------------------------------------

class TestCapabilitiesFollowTheBuildMode:
    """`harness/workloads/build.py` is the reason this is not a constant.

    Its flag is `capabilities().get("incremental_index", True)` with an
    override forcing it True for `build_mode == "incremental"` -- and no
    counterpart forcing it False for `post`. The flag then decides whether
    `create_index` is called at all and what `build_wall_s` means.
    """

    def test_build_py_still_has_no_post_counterpart(self):
        """The premise, checked against the file rather than assumed. If
        `build.py` ever grows an `elif build_mode == "post"`, this driver's
        derivation becomes belt-and-braces and someone should know."""
        source = open(os.path.join(VB_ROOT, "harness", "workloads",
                                   "build.py")).read()
        assert 'index.build_mode == "incremental"' in source   # findable
        assert 'build_mode == "post"' not in source

    def test_post_mode_reports_a_separable_build(self):
        driver = driver_with(PerconaDriver)
        driver.create_schema(spec(build_mode="post"))
        assert driver.capabilities()["incremental_index"] is False

    def test_incremental_mode_reports_no_separable_build(self):
        driver = driver_with(PerconaDriver)
        driver.create_schema(spec(build_mode="incremental"))
        assert driver.capabilities()["incremental_index"] is True

    def test_before_a_schema_exists_the_answer_is_the_engines_own_default(self):
        """Percona maintains the graph on the write path unless a profile asks
        otherwise, so `True` is the honest answer with no spec to read."""
        assert driver_with(PerconaDriver).capabilities()[
            "incremental_index"] is True

    def test_the_two_modes_do_not_produce_the_same_capabilities(self):
        """An identical pair here would mean the derivation is dead."""
        post = driver_with(PerconaDriver)
        post.create_schema(spec(build_mode="post"))
        inc = driver_with(PerconaDriver)
        inc.create_schema(spec(build_mode="incremental"))
        assert post.capabilities() != inc.capabilities()

    def test_the_existing_engines_are_unaffected_by_build_mode(self):
        """MariaDB and AliSQL have no separable build, so both modes must
        still report incremental -- a `post` profile on them calls a
        documented no-op and the load IS the build."""
        for cls in (MariaDBDriver, AliSQLDriver):
            for mode in ("post", "incremental"):
                driver = driver_with(cls)
                driver.create_schema(spec(build_mode=mode))
                assert driver.capabilities()["incremental_index"] is True

    def test_percona_is_the_only_mysql_family_engine_with_the_build_knob(self):
        for cls, want in ((MariaDBDriver, False), (AliSQLDriver, False),
                          (PerconaDriver, True)):
            driver = driver_with(cls)
            assert driver.capabilities()["ef_construction_tunable"] is want


class TestTheIndexLeavesTheTableDefinitionInBulkMode:
    """What `build_mode` actually changes, which is the DDL and not a label."""

    def test_post_mode_creates_the_table_without_the_index(self):
        driver = driver_with(PerconaDriver)
        driver.create_schema(spec(build_mode="post"))
        assert "VECTOR KEY" not in driver._cur.sql
        assert "CREATE TABLE" in driver._cur.sql

    def test_post_mode_adds_the_index_in_create_index(self):
        driver = driver_with(PerconaDriver)
        driver.create_schema(spec(build_mode="post"))
        before = len(driver._cur.statements)
        driver.create_index(spec(build_mode="post"))
        added = driver._cur.statements[before:]
        assert len(added) == 1
        assert added[0].startswith(f"ALTER TABLE ")
        assert "ADD VECTOR KEY vi (v) TYPE hnsw" in added[0]
        # INPLACE builds the graph in memory and bulk-loads the aux table; a
        # COPY-built aux is correct but slower to build and larger. Leaving
        # the choice to the server would let "bulk build" be either one.
        assert "ALGORITHM=INPLACE" in added[0]

    def test_incremental_mode_declares_the_index_inline(self):
        driver = driver_with(PerconaDriver)
        driver.create_schema(spec(build_mode="incremental"))
        assert "VECTOR KEY vi (v) TYPE hnsw" in driver._cur.sql

    def test_incremental_mode_makes_create_index_a_no_op(self):
        driver = driver_with(PerconaDriver)
        driver.create_schema(spec(build_mode="incremental"))
        before = len(driver._cur.statements)
        driver.create_index(spec(build_mode="incremental"))
        assert driver._cur.statements[before:] == []


# ---------------------------------------------------------------------------
# §2.7 -- the repair path, and the options it must preserve
# ---------------------------------------------------------------------------

class TestTheRepairPath:

    SHOW_CREATE = (
        "t1",
        "CREATE TABLE `t1` (\n"
        "  `id` bigint unsigned NOT NULL,\n"
        "  `v` vector(8) NOT NULL,\n"
        "  PRIMARY KEY (`id`),\n"
        "  VECTOR KEY `vi` (`v`) USING `hnsw` "
        "(`M`=24, `ef_construction`=320, `metric`=`cosine`)\n"
        ") ENGINE=InnoDB",
    )

    def test_percona_declares_its_repair_path(self):
        assert PerconaDriver.repair_path == "index-replace"

    def test_the_existing_engines_still_declare_none(self):
        assert MariaDBDriver.repair_path == "none"
        assert AliSQLDriver.repair_path == "none"

    def test_maintain_replaces_the_index_in_one_alter(self):
        driver = driver_with(PerconaDriver, answers=[self.SHOW_CREATE],
                             index=spec())
        what = driver.maintain()
        alter = [s for s in driver._cur.statements if s.startswith("ALTER")]
        assert len(alter) == 1
        assert "DROP KEY vi" in alter[0] and "ADD VECTOR KEY" in alter[0]
        assert what == "ALTER TABLE (index replace)"

    def test_maintain_is_not_optimize_table(self):
        """`OPTIMIZE TABLE` also compacts and is deliberately not used: it
        rewrites the whole base table, so its cost is dominated by work that
        has nothing to do with the vector index and the churn phase would be
        timing a table rebuild."""
        driver = driver_with(PerconaDriver, answers=[self.SHOW_CREATE],
                             index=spec())
        driver.maintain()
        assert "OPTIMIZE" not in driver._cur.sql.upper()

    def test_the_rebuild_reuses_the_servers_own_option_list(self):
        """THE CORRECTNESS ARGUMENT FOR READING IT BACK.

        The DDL that rebuilds the index decides the new graph's M,
        ef_construction and metric. Omitting them rebuilds at the server's
        defaults -- M=25, not the profile's -- and the post-maintenance recall
        would be a different index's recall reported as the repaired one's.

        The spec here deliberately carries DIFFERENT values (M=16,
        ef_construction unset) from the live index the fake server reports
        (M=24, ef_construction=320), so a rebuild derived from the profile
        instead of from the server fails this.
        """
        driver = driver_with(PerconaDriver, answers=[self.SHOW_CREATE],
                             index=spec(m=16, ef_construction=None))
        driver.maintain()
        alter = [s for s in driver._cur.statements if s.startswith("ALTER")][0]
        assert "`M`=24" in alter
        assert "`ef_construction`=320" in alter
        assert "`metric`=`cosine`" in alter
        # And not the profile's values, which is the half that bites.
        assert "M = 16" not in alter

    def test_it_asks_the_server_before_it_rebuilds(self):
        driver = driver_with(PerconaDriver, answers=[self.SHOW_CREATE],
                             index=spec())
        driver.maintain()
        assert driver._cur.statements[0].startswith("SHOW CREATE TABLE")

    def test_a_server_that_cannot_answer_is_warned_about_not_silently_guessed(
            self, capsys):
        """A silent fall-back here is the "rebuilt at the defaults" failure
        this method exists to avoid, except with our numbers instead of the
        server's -- so it happens loudly or not at all."""
        driver = driver_with(PerconaDriver, answers=[("t1", "CREATE TABLE t1")],
                             index=spec(m=16))
        driver.maintain()
        out = " ".join(capsys.readouterr().out.split())
        assert "WARNING" in out
        assert "post-maintenance recall is a different index's" in out
        alter = [s for s in driver._cur.statements if s.startswith("ALTER")][0]
        assert "M = 16" in alter

    def test_the_churn_hooks_percona_does_not_need_are_inherited(self):
        """`prepare_churn` and `wait_index_synced` stay the base class's.

        Percona has no background graph maintenance at all, so there is no
        knob whose state could differ between the baseline and the post-churn
        measurement, and nothing to wait for: the graph is maintained inside
        the INSERT and the honest wait is zero. Returning a small positive
        number "to be safe" would be a fabricated measurement in a field
        documented as a measurement.
        """
        assert PerconaDriver.prepare_churn is EngineDriver.prepare_churn
        assert PerconaDriver.wait_index_synced is EngineDriver.wait_index_synced
        assert driver_with(PerconaDriver).wait_index_synced(1000) == 0.0


class TestTheEfSearchClamp:

    def test_percona_clamps_up_to_k(self):
        driver = driver_with(PerconaDriver)
        assert [driver.effective_ef(ef, 100) for ef in (10, 40, 160)] == \
            [100, 100, 160]

    def test_it_changes_nothing_at_the_k_every_profile_uses(self):
        """Honest about its own reach: at k=10 every grid value this repo
        sweeps is above 10, so this override moves no figure today. It is
        implemented because the inherited identity would be wrong at the one
        grid point where it mattered and silently right everywhere else."""
        driver = driver_with(PerconaDriver)
        assert [driver.effective_ef(ef, 10) for ef in (10, 40, 160)] == \
            [10, 40, 160]

    def test_the_existing_mysql_engines_do_not_clamp(self):
        for cls in (MariaDBDriver, AliSQLDriver):
            driver = driver_with(cls)
            assert driver.effective_ef(10, 100) == 10

    def test_ef_search_is_set_per_session_and_not_globally(self):
        """The variable has both scopes; a GLOBAL write would outlive the
        calibration sweep that set it and reach every later phase."""
        driver = driver_with(PerconaDriver)
        driver.set_ef_search(200)
        stmt = driver._cur.statements[0]
        assert stmt == "SET SESSION innodb_hnsw_ef_search = 200"
        assert "GLOBAL" not in stmt


# ---------------------------------------------------------------------------
# §4.2 -- the driver registries
# ---------------------------------------------------------------------------

class TestTheDriverIsReachable:

    def test_the_class_registry_carries_it(self):
        assert _driver_classes()["PerconaDriver"] is PerconaDriver

    def test_the_engine_table_carries_it(self):
        assert _driver_table()["percona"] is PerconaDriver

    def test_the_config_names_the_class_the_registry_knows(self):
        """The two halves of `--driver`: the orchestrator reads the name out of
        config/engines/percona.yml and the harness looks it up here. A typo in
        either is `argument --engine: invalid choice` an hour into a run."""
        from orchestrator import engines as engines_mod
        declared = engines_mod.registry(refresh=True)["percona"].driver
        assert declared in _driver_classes()

    def test_every_registered_engine_has_a_driver_class(self):
        from orchestrator import engines as engines_mod
        classes = _driver_classes()
        for name, runtime in engines_mod.registry(refresh=True).items():
            assert runtime.driver in classes, name


# ---------------------------------------------------------------------------
# §4.3 / §4.5 -- the config, and the hand-kept copy of two of its values
# ---------------------------------------------------------------------------

class TestTheEngineConfig:

    @pytest.fixture(scope="class")
    def cfg(self):
        import yaml
        with open(os.path.join(VB_ROOT, "config", "engines",
                               "percona.yml")) as fh:
            return yaml.safe_load(fh)

    def test_the_source_pin_is_the_sha_under_the_tag_key(self, cfg):
        """`_source_tag` reads `tag` or `version` and there is no `sha` key, so
        a `source: sha:` block would make it return "" and
        prepare-sources.sh fall back to a hard-coded default."""
        assert cfg["source"]["tag"] == PINNED_SHA
        assert "sha" not in cfg["source"]

    def test_the_registry_carries_that_pin_into_the_manifest(self):
        from orchestrator import engines as engines_mod
        assert engines_mod.registry(refresh=True)["percona"].tag == PINNED_SHA

    def test_no_submodules_key(self, cfg):
        """All four of the pin's submodules are switched off in the build
        instead. Listing them here would make it fetch from three more hosts
        for code that is never compiled."""
        assert "submodules" not in cfg["source"]

    def test_no_sql_block(self, cfg):
        assert "sql" not in cfg

    def test_the_graph_memory_knob_is_unlimited_in_both_passes(self, cfg):
        """Decided policy, not a default. §3: Percona's is a hard allocation
        ceiling and not a cache, so feeding it `graph_cache_bytes` would
        compare a cache budget against a wall."""
        for pass_name in ("normalized", "tuned"):
            flags = cfg["server"][pass_name]
            assert "--innodb-hnsw-max-memory=0" in flags, pass_name
            assert not any("graph_cache_bytes" in f for f in flags), pass_name

    def test_it_is_not_given_the_other_engines_cache_knobs(self, cfg):
        """A different variable name with the same intent would reintroduce
        the same error under another spelling.

        Comments are stripped first, because the file EXPLAINS the absence by
        naming both knobs -- and the first version of this test matched its
        own explanation. That is the inverse of the usual vacuity: a negative
        assertion failing on prose that agrees with it.
        """
        text = open(os.path.join(VB_ROOT, "config", "engines",
                                 "percona.yml")).read()
        keys = "\n".join(ln.split("#", 1)[0] for ln in text.splitlines())
        for knob in ("mhnsw_max_cache_size", "vidx_hnsw_cache_size",
                     "mhnsw-max-cache-size", "vidx-hnsw-cache-size"):
            assert knob not in keys, knob
        # ...and the explanation really is still there, so this is not
        # passing because the file went silent on the question.
        assert "mhnsw_max_cache_size" in text

    def test_the_march_matches_every_engine_that_compiles_one(self, cfg):
        """"Never mix -march between engines" is a stated invariant.

        Compared against the other configs rather than against a literal, so
        a repo-wide change of the flag does not need this test edited to stay
        true.

        `march: none` is excluded and that is not a loophole: mongodb and
        valkey compile nothing -- they are installed from published packages
        -- so they have no SIMD kernels of ours to mismatch. The engines that
        DO compile are the ones the invariant is about, and `none` is checked
        to be exactly the non-compiling set rather than waved through.
        """
        import yaml
        marches = {}
        for name in sorted(os.listdir(os.path.join(VB_ROOT, "config",
                                                   "engines"))):
            if not name.endswith(".yml"):
                continue
            with open(os.path.join(VB_ROOT, "config", "engines", name)) as fh:
                build = yaml.safe_load(fh).get("build") or {}
            if build.get("march"):
                marches[name[:-4]] = build["march"]
        compiling = {k: v for k, v in marches.items() if v != "none"}
        assert len(compiling) >= 4, marches             # non-vacuity
        assert "percona" in compiling
        assert len(set(compiling.values())) == 1, compiling
        # The ones that opt out are exactly the ones that build nothing.
        assert {k for k, v in marches.items() if v == "none"} <= {"mongodb",
                                                                  "valkey"}

    def test_the_ann_constructor_is_unique_and_is_not_the_mongodb_one(self):
        """ann-benchmarks keys result files on this name, so a duplicate makes
        the second engine report the first's recall -- the defect `mariadb123`
        exists to prevent. `PerconaSearch` is the MongoDB engine."""
        from orchestrator import engines as engines_mod
        registry = engines_mod.registry(refresh=True)
        names = [e.ann_constructor for e in registry.values()]
        assert len(names) == len(set(names)), names
        assert registry["percona"].ann_constructor == "PerconaServer"
        assert registry["mongodb"].ann_constructor == "PerconaSearch"

    def test_the_image_tags_are_unique_too(self):
        import yaml
        tags = []
        for name in sorted(os.listdir(os.path.join(VB_ROOT, "config",
                                                   "engines"))):
            if not name.endswith(".yml"):
                continue
            with open(os.path.join(VB_ROOT, "config", "engines", name)) as fh:
                image = yaml.safe_load(fh).get("image") or {}
            tags.extend(v for k, v in image.items() if k in ("runtime", "bench"))
        assert len(tags) == len(set(tags)), sorted(tags)

    def test_percona_took_an_unused_colour_and_an_unused_marker(self):
        """Scoped to this engine, because markers are NOT globally unique and
        pretending otherwise would be a false claim: mariadb and mariadb123
        deliberately share `s`, being the same software at two versions, and
        are told apart by colour. Colours ARE unique and that is asserted
        across the whole registry."""
        from orchestrator import engines as engines_mod
        registry = engines_mod.registry(refresh=True)
        colours = [e.color for e in registry.values()]
        assert len(colours) == len(set(colours)), colours
        others = [e for name, e in registry.items() if name != "percona"]
        assert others                                   # non-vacuity
        mine = registry["percona"]
        assert mine.color == "#ff7f0e"
        assert mine.marker == "P"
        assert mine.color not in {e.color for e in others}
        assert mine.marker not in {e.marker for e in others}


class TestTheWebUiCopyMatchesTheRegistryCharacterForCharacter:
    """The front end is static and cannot read the engine registry, so
    `webui/static/fields.js` holds a second copy of every engine's colour and
    label. This is the one place in the addition where two files are kept in
    step by hand, which is why it is checked for EVERY engine and not only for
    the new one."""

    @pytest.fixture(scope="class")
    def maps(self):
        source = open(os.path.join(VB_ROOT, "webui", "static",
                                   "fields.js")).read()
        out = {}
        for const in ("ENGINE_COLOR", "ENGINE_LABEL"):
            body = source.split(f"const {const} = {{", 1)[1].split("};", 1)[0]
            out[const] = dict(re.findall(r'(\w+)\s*:\s*"([^"]*)"', body))
        return out

    def test_the_parser_found_both_maps(self, maps):
        """Non-vacuity: an empty dict would make every loop below pass."""
        assert len(maps["ENGINE_COLOR"]) >= 7, maps["ENGINE_COLOR"]
        assert len(maps["ENGINE_LABEL"]) >= 7, maps["ENGINE_LABEL"]

    def test_every_engine_appears_in_both_maps(self, maps):
        from orchestrator import engines as engines_mod
        registry = engines_mod.registry(refresh=True)
        assert set(maps["ENGINE_COLOR"]) == set(registry)
        assert set(maps["ENGINE_LABEL"]) == set(registry)

    def test_every_colour_and_label_matches_the_yaml_exactly(self, maps):
        """`chart_label` and not `label`: these maps feed charts, and the two
        differ for mongodb, whose chart says "Percona Search (mongot)" while
        its report label spells out "for MongoDB". For percona the two are the
        same string, so getting this wrong would have passed for the new
        engine and failed for an old one -- which is how it was found."""
        from orchestrator import engines as engines_mod
        for name, runtime in engines_mod.registry(refresh=True).items():
            assert maps["ENGINE_COLOR"][name] == runtime.color, name
            assert maps["ENGINE_LABEL"][name] == runtime.chart_label, name
        # Percona's two labels are identical, which is what makes the single
        # fields.js entry sufficient for it.
        percona = engines_mod.registry(refresh=True)["percona"]
        assert percona.chart_label == percona.label
        assert percona.label == "Percona Server (InnoDB HNSW)"

    def test_the_build_chart_hint_mentions_the_engine_that_does_both(self):
        """The hint read "pgvector bulk-builds; MHNSW and VIDX build on every
        INSERT", which became incomplete with a third behaviour in the chart.

        Matched on whitespace-normalised text, because a hint long enough to
        describe three behaviours is long enough for an editor to wrap.
        """
        source = open(os.path.join(VB_ROOT, "webui", "static",
                                   "fields.js")).read()
        assert "pgvector bulk-builds" in _flat(source)   # needle findable
        _prose("percona does either build mode", source)


# ---------------------------------------------------------------------------
# §4.10 -- the render branch, and the key it refuses
# ---------------------------------------------------------------------------

class TestTheAnnRenderBranch:

    @staticmethod
    def _render(ann=None, resource_pass="normalized", resources=None):
        from orchestrator import ann_pass
        profile = {"name": "t", "ann": ann or {}}
        return ann_pass.render_config("percona", profile, resources or {},
                                      resource_pass)

    def test_percona_no_longer_lands_on_the_raise(self):
        body = self._render()
        assert body["float"]["any"][0]["constructor"] == "PerconaServer"

    def test_an_unknown_engine_still_does(self):
        """Non-vacuity: the `raise` is still reachable, so the test above is
        about the branch and not about the raise having been deleted."""
        from orchestrator import ann_pass
        with pytest.raises(ValueError, match="unknown engine"):
            ann_pass.render_config("nope", {"name": "t"}, {}, "normalized")

    def test_it_emits_ef_construction_which_no_mysql_branch_does(self):
        groups = self._render()["float"]["any"][0]["run_groups"]
        args = list(groups.values())[0]["arg_groups"][0]
        assert args["efConstruction"] == [200]

    def test_the_mysql_family_branch_still_does_not(self):
        """The reason Percona needed a branch of its own rather than a line in
        the condition at the top."""
        from orchestrator import ann_pass
        body = ann_pass.render_config(
            "mariadb", {"name": "t", "ann": {}}, {}, "normalized")
        for group in body["float"]["any"][0]["run_groups"].values():
            for args in group["arg_groups"]:
                assert "efConstruction" not in args

    def test_the_tuned_pass_can_sweep_it(self):
        body = self._render(
            resource_pass="tuned",
            resources={"extras": {"percona_ef_construction": [64, 200, 512]}})
        args = list(body["float"]["any"][0]["run_groups"].values())[0]
        assert args["arg_groups"][0]["efConstruction"] == [64, 200, 512]

    def test_the_normalized_pass_ignores_the_tuned_extra(self):
        """Pinned in the normalized pass for the same reason pgvector's and
        Valkey's are: MariaDB and AliSQL lack the knob, and sweeping it where
        only one engine has it hands that engine a tuning axis the others
        cannot answer."""
        body = self._render(
            resources={"extras": {"percona_ef_construction": [64, 512]}})
        args = list(body["float"]["any"][0]["run_groups"].values())[0]
        assert args["arg_groups"][0]["efConstruction"] == [200]

    def test_one_run_group_and_innodb_only(self):
        """The vector index is InnoDB's, implemented in storage/innobase, so
        there is no second storage engine to curve against -- the same shape
        as alisql's branch and not mariadb's."""
        groups = self._render()["float"]["any"][0]["run_groups"]
        assert list(groups) == ["innodb"]
        assert groups["innodb"]["arg_groups"][0]["engine"] == "InnoDB"

    def test_a_build_mode_sweep_is_refused_loudly_not_emitted(self):
        """THE ONE DOCUMENTED INTENTION THAT COULD NOT BE DELIVERED.

        docs/09 §2.5 and §4.10 propose `percona_build_modes` as a within-engine
        ann comparison. The ann overlay module has no `build_mode` --
        pgvector's and valkey's each read it, validate it and branch on it in
        `fit`; `algorithms/percona/module.py` contains the string zero times --
        so emitting the key would produce two run groups differing only in a
        `method_param` nothing reads: two IDENTICAL recall curves under two
        labels, with no field in the record to say they were the same
        measurement.

        A `ValueError` at render time, before anything is built, is the good
        kind of failure. This pins that it is refused rather than honoured,
        and the message has to name where the gap is.
        """
        with pytest.raises(ValueError) as excinfo:
            self._render(ann={"percona_build_modes": ["post", "incremental"]})
        message = str(excinfo.value)
        assert "percona_build_modes" in message
        assert "build_mode" in message
        assert "module.py" in message

    def test_the_default_single_mode_is_accepted(self):
        """`["post"]` is what every profile without the key resolves to, so
        refusing it would refuse every run."""
        assert self._render(ann={"percona_build_modes": ["post"]})

    def test_the_premise_of_that_refusal_is_checked_not_assumed(self):
        """If `percona/module.py` ever learns `build_mode`, this fails and the
        refusal above should be replaced by a build-mode loop rather than
        left in place."""
        overlay = os.path.join(VB_ROOT, "overlay", "ann-benchmarks",
                               "ann_benchmarks")
        percona = open(os.path.join(overlay, "algorithms", "percona",
                                    "module.py")).read()
        shared = open(os.path.join(overlay, "vb_mysql.py")).read()
        assert "build_mode" not in percona
        assert "build_mode" not in shared
        # Non-vacuity: the two engines that DO support it really do say so,
        # so "the string is absent" is a fact about these files and not about
        # the spelling being wrong.
        for engine in ("pgvector", "valkey"):
            other = open(os.path.join(overlay, "algorithms", engine,
                                      "module.py")).read()
            assert "build_mode" in other, engine


class TestTheOpsPathDoesSupportBothBuildModes:
    """The half that works, and where the comparison actually comes from.

    `orchestrator/cli.py` hard-coded `["post"]` for every engine but pgvector,
    which docs/09 §4 never mentions -- so without this the Percona-only
    profile's headline axis would have been silently inert on both paths.
    """

    def test_the_profile_declares_the_one_measurable_mode(self):
        """It declared `[post, incremental]` until `post` was found to kill
        the server at the pinned version -- see
        `capabilities.bulk_vector_index_build` in
        `config/engines/percona.yml` for the reproduction. The axis is down to
        its measurable half, and this assertion is what stops the broken half
        being reinstated without the capability being flipped first.
        """
        from orchestrator.config import load_profile
        from orchestrator.config import load_engine
        assert load_profile("percona")["ops"]["percona_build_modes"] == [
            "incremental"]
        # Tied to the reason rather than to the value: if someone restores
        # `post` because the server was fixed, the capability has to say so.
        caps = load_engine("percona").get("capabilities", {}) or {}
        assert caps.get("bulk_vector_index_build") is False

    def test_the_key_lives_under_ops_where_the_ann_renderer_cannot_see_it(self):
        """**This assertion was inverted, and the inversion was a defect that
        cost this profile both of its axes.**

        It first read `"percona_build_modes" in profile["ann"]`, following
        `pgvector_build_modes`. That precedent is only safe because pgvector's
        overlay module reads `build_mode`; Percona's does not, which is why
        `render_config` POLICES `ann.percona_build_modes` and raises on any
        value but `["post"]`. With the key under `ann:` the profile tripped
        its own refusal: `run --profile percona` failed its ann phase, and
        `render --profile percona` raised uncaught out of `cmd_render` -- so
        `percona_ef_construction`, which only `render_config` reads, was never
        swept either.

        Under `ops:` the ann renderer structurally cannot see it. The readers
        carry the block in the key, so this is where they look.
        """
        from orchestrator.config import load_profile
        profile = load_profile("percona")
        assert "percona_build_modes" in profile["ops"]
        assert "percona_build_modes" not in profile.get("ann", {})
        # And the `ann:` block still explains the absence, so the next author
        # does not "fix" it by moving the key back.
        text = open(os.path.join(VB_ROOT, "config", "profiles",
                                 "percona.yml")).read()
        ann_block = text.split("ann:", 1)[1].split("\nops:", 1)[0]
        assert "percona_build_modes" in ann_block

    def test_pgvectors_key_still_lives_under_ann_because_it_drives_both(self):
        """The contrast that makes the split meaningful rather than untidy:
        pgvector's list really does drive the ann render as well as the ops
        loop, so it belongs with the ann configuration."""
        from orchestrator.config import load_profile
        for name in ("main", "smoke"):
            profile = load_profile(name)
            assert "pgvector_build_modes" in profile["ann"], name
            assert "pgvector_build_modes" not in profile.get("ops", {}), name

    def test_the_estimate_counts_both_modes_as_two_ops_units(self):
        """If the runner executes a second unit the estimator does not count,
        the printed figure is short by a whole unit of every query term."""
        from orchestrator import estimate as est_mod
        one = {"name": "t",
               "ops": {"m_values": [16], "percona_build_modes": ["post"]}}
        two = {"name": "t",
               "ops": {"m_values": [16],
                       "percona_build_modes": ["post", "incremental"]}}
        assert est_mod._ops_units(one, "percona") == 1
        assert est_mod._ops_units(two, "percona") == 2
        # And no other engine is affected by the key.
        assert est_mod._ops_units(two, "mariadb") == 1
        assert est_mod._ops_units(two, "alisql") == 1
        # The BLOCK is load-bearing, not cosmetic: the same list under `ann:`
        # is not read, which is what keeps it out of `render_config`'s way.
        under_ann = {"name": "t", "ops": {"m_values": [16]},
                     "ann": {"percona_build_modes": ["post", "incremental"]}}
        assert est_mod._ops_units(under_ann, "percona") == 1
        # ...and pgvector's really is read from `ann:`, which is the contrast
        # that makes the split a decision rather than an inconsistency.
        pg = {"name": "t", "ops": {"m_values": [16]},
              "ann": {"pgvector_build_modes": ["post", "incremental"]}}
        assert est_mod._ops_units(pg, "pgvector") == 2

    def test_the_cli_and_the_estimate_use_the_same_engine_list(self):
        """Two copies of "which engines have a separable build, and where the
        list lives", in two files that must agree.

        Compared against each other rather than each pinned to a literal, so a
        third engine added to one and not the other fails here -- and the
        BLOCK is part of what is compared, because a key read from `ann:` in
        one file and `ops:` in the other would silently give the runner two
        units and the estimate one.
        """
        from orchestrator import estimate as est_mod
        pattern = r'"(\w+)":\s*\(\s*"(\w+)",\s*"(\w+)"\s*\)'
        source = open(os.path.join(VB_ROOT, "orchestrator", "cli.py")).read()
        block = source.split("_OPS_BUILD_MODE_KEYS = {", 1)[1].split("}", 1)[0]
        in_cli = {m[0]: (m[1], m[2]) for m in re.findall(pattern, block)}
        assert in_cli, "the cli lookup did not parse"
        est_source = open(os.path.join(VB_ROOT, "orchestrator",
                                       "estimate.py")).read()
        est_block = est_source.split("mode_ref = {", 1)[1].split("}", 1)[0]
        in_est = {m[0]: (m[1], m[2]) for m in re.findall(pattern, est_block)}
        assert in_est == in_cli, (in_cli, in_est)
        assert in_cli == {"pgvector": ("ann", "pgvector_build_modes"),
                          "percona": ("ops", "percona_build_modes")}

    def test_the_ef_construction_sweep_is_real_and_not_only_claimed(self):
        """The profile's FIRST axis, read out of `render_config`.

        `render_config`'s normalized branch coerces the profile key to a
        single int by construction, so a scalar under `ann:` with
        sweep-shaped prose beside it would have claimed a measurement the
        profile never takes. The sweep is a `resources.extras` override that
        the TUNED branch reads, and both ends are asserted here -- one value
        normalized, three tuned -- because asserting only the tuned one would
        pass against a profile that swept in both passes and lost the pin.
        """
        from orchestrator import ann_pass
        from orchestrator.config import (load_profile, load_resources,
                                         merge_resource_overrides)
        profile = load_profile("percona")
        got = {}
        for resource_pass in ("normalized", "tuned"):
            resources = merge_resource_overrides(
                load_resources(resource_pass), profile)
            body = ann_pass.render_config("percona", profile, resources,
                                          resource_pass)
            group, = body["float"]["any"][0]["run_groups"].values()
            got[resource_pass] = group["arg_groups"][0]["efConstruction"]
        assert got["normalized"] == [200], got
        assert len(got["tuned"]) == 3, got
        assert got["tuned"] == [50, 200, 800], got
        # Every value inside the range the server accepts, checked against the
        # overlay module's own bounds rather than against numbers repeated
        # here: out-of-range arrives as "Illegal index construction parameter
        # value: 0" with no mention of which parameter.
        source = open(os.path.join(
            VB_ROOT, "overlay", "ann-benchmarks", "ann_benchmarks",
            "algorithms", "percona", "module.py")).read()
        lo = int(re.search(r"MIN_EF_CONSTRUCTION = (\d+)", source).group(1))
        hi = int(re.search(r"MAX_EF_CONSTRUCTION = (\d+)", source).group(1))
        assert (lo, hi) == (1, 4096)
        for value in got["tuned"]:
            assert lo <= value <= hi, value


# ---------------------------------------------------------------------------
# The estimator: an engine with no measured rate is named, not guessed
# ---------------------------------------------------------------------------

class TestAnUnmeasuredIngestRateIsNamedAndNotModelled:
    """`_effective_rate` ended in `.get(engine, 300)`, which contradicted the
    comment four lines above the table it read from: "an estimate that is
    optimistic by 3x is worse than none". 300 rows/s is 2x the measured
    MariaDB figure and 5.5x the measured AliSQL one, and it reached published
    output -- `webui/control.py` passes `KNOWN_ENGINES` straight in.
    """

    @staticmethod
    def _estimate(engines):
        from orchestrator.estimate import estimate_load_hours
        return estimate_load_hours(
            {"name": "t", "ann": {"m_values": [16]}, "ops": {"m_values": [16]}},
            list(engines), ["glove-100-angular"], ["normalized"],
            ["ann", "ops"])

    def test_an_engine_with_no_rate_returns_none_rather_than_a_number(self):
        from orchestrator.estimate import _effective_rate
        assert _effective_rate("percona", "glove-100-angular") is None
        # Non-vacuity: a measured engine still gets its number.
        assert _effective_rate("mariadb", "glove-100-angular") == 147.0

    def test_it_is_named(self):
        e = self._estimate(["mariadb", "percona"])
        assert e["engines_without_ingest_rate"] == ["percona"]

    def test_it_contributes_nothing_to_the_total(self):
        """The alternative -- a key worth 0.0 -- reads as "costs nothing",
        which is the opposite of what is known about an engine building an
        HNSW graph inside every INSERT."""
        one = self._estimate(["mariadb"])
        two = self._estimate(["mariadb", "percona"])
        assert two["total_hours"] == one["total_hours"]
        assert "percona" not in two["per_engine_hours"]

    def test_a_measured_engine_is_not_named(self):
        e = self._estimate(["mariadb", "alisql", "pgvector"])
        assert e["engines_without_ingest_rate"] == []

    def test_the_six_measured_rates_did_not_move(self):
        """The constraint on the repair: another task re-baselined these
        figures, and if the six moved its numbers and this change would
        fight."""
        from orchestrator import estimate as est_mod
        assert est_mod._INGEST_ROWS_PER_S == {
            "mariadb": 150, "mariadb123": 110, "alisql": 55,
            "pgvector": 3000, "valkey": 40000, "mongodb": 60000}

    def test_no_rate_was_invented_for_percona(self):
        """An estimate for an unmeasured engine is a guess presented as a
        plan. The smoke run supplies the figure."""
        from orchestrator import estimate as est_mod
        assert "percona" not in est_mod._INGEST_ROWS_PER_S
        assert not [k for k in est_mod._MEASURED_ROWS_PER_S
                    if k[0] == "percona"]

    def test_a_dataset_with_no_row_count_does_not_make_every_engine_a_gap(self):
        """The care `engines_without_query_rate` takes, taken here too: the
        gap list is keyed on the RATE, not on "contributed no hours". `dev`'s
        only dataset has no row count, and every engine would be listed if
        the test were the latter."""
        from orchestrator.estimate import estimate_load_hours
        e = estimate_load_hours(
            {"name": "t", "ann": {}, "ops": {}},
            ["mariadb", "percona"], ["tiny-16-euclidean"], ["normalized"],
            ["ann", "ops"])
        assert e["engines_without_ingest_rate"] == ["percona"]
        assert e["datasets_without_estimate"] == ["tiny-16-euclidean"]
        assert e["per_engine_hours"] == {"mariadb": 0.0}

    def test_the_printed_output_says_which_engines_the_total_covers(self):
        """A reader must not have to reconcile a seven-engine plan against a
        six-engine total by noticing a name missing from a list."""
        import io
        from contextlib import redirect_stdout
        from orchestrator.config import load_profile
        from orchestrator.cli import KNOWN_ENGINES
        from orchestrator.estimate import print_load_estimate
        profile = load_profile("main")
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_load_estimate(profile, list(KNOWN_ENGINES),
                                list(profile["datasets"]), ["normalized"],
                                ["ann", "ops"])
        out = buf.getvalue()
        assert "No measured ingest rate for percona" in _flat(out)
        _prose("the total covers only the engines with a rate", out)
        # And it says what a reader should CONCLUDE, not merely that a figure
        # is missing -- the attack this catches is a line that announces the
        # gap and drops the consequence.
        _prose("the real cost is higher than the printed total", out)

    def test_the_churn_term_does_not_charge_an_unmeasured_engine_a_duration(
            self):
        """`typical` stays 0.0 and the basis says why, which is what
        `_filtered_search` already does for a missing query rate. A budget
        says what work cannot exceed, not what it will take."""
        from orchestrator.estimate import estimate_load_hours
        profile = {"name": "t", "ops": {"enabled": True,
                                        "workloads": ["churn"],
                                        "m_values": [16],
                                        "subset_rows": 20000,
                                        "churn_fractions": [0.1],
                                        "churn_budget_s": 1800,
                                        "max_queries": 200}}
        e = estimate_load_hours(profile, ["percona"], ["glove-100-angular"],
                                ["normalized"], ["ops"])
        churn = [t for t in e["query_terms"] if t["name"] == "churn re-insert"]
        assert len(churn) == 1
        # Seconds, not hours: 2,000 rows is a few seconds of work even at
        # MariaDB's measured 147 rows/s, so `typical_hours` rounds to 0.0 for
        # BOTH engines and a comparison on it would be the identical
        # honest/mutant pair that signals a dead probe.
        assert churn[0]["typical_seconds"] == 0.0
        assert churn[0]["worst_seconds"] > 0
        assert "at a rate nothing here has measured" in churn[0]["basis"]
        # Non-vacuity: a measured engine gets a duration and a rate.
        m = estimate_load_hours(profile, ["mariadb"], ["glove-100-angular"],
                                ["normalized"], ["ops"])
        mchurn = [t for t in m["query_terms"]
                  if t["name"] == "churn re-insert"][0]
        assert mchurn["typical_seconds"] > 0
        assert "rows/s" in mchurn["basis"]
        assert "at a rate nothing here has measured" not in mchurn["basis"]

    def test_the_repair_path_disclosure_names_both_engines_that_have_one(self):
        """It said "pgvector of the six", which was a count and an engine list
        at once and became wrong in both halves. Percona's repair rebuilds the
        whole graph, so it is the larger of the two unmeasured repairs, not a
        footnote to pgvector's VACUUM."""
        from orchestrator.estimate import estimate_load_hours
        from orchestrator.config import load_profile
        from orchestrator.cli import KNOWN_ENGINES
        profile = load_profile("main")
        e = estimate_load_hours(profile, list(KNOWN_ENGINES),
                                list(profile["datasets"]), ["normalized"],
                                ["ann", "ops"])
        repair = [n for n in e["query_costs_not_modelled"]
                  if n.startswith("churn repair path:")]
        assert len(repair) == 1, repair
        flat = " ".join(repair[0].split())
        assert "pgvector" in flat and "Percona Server" in flat
        assert "of the six" not in flat


# ---------------------------------------------------------------------------
# §4.7 -- the scripts
# ---------------------------------------------------------------------------

class TestTheBuildScripts:

    @pytest.fixture(scope="class")
    def build_images(self):
        return open(os.path.join(VB_ROOT, "scripts", "build-images.sh")).read()

    @pytest.fixture(scope="class")
    def prepare(self):
        return open(os.path.join(VB_ROOT, "scripts",
                                 "prepare-sources.sh")).read()

    @pytest.fixture(scope="class")
    def lib(self):
        return open(os.path.join(VB_ROOT, "scripts", "lib.sh")).read()

    def test_build_images_has_both_cases_and_not_only_one(self, build_images):
        """There are TWO dispatches: the outer one on `$ENGINE` that decides
        whether the engine can be built at all, and the inner one on the
        alias base that supplies the tag build-arg. Adding only the inner one
        fails with "unknown engine" before the build-arg is ever read."""
        outer = build_images.rsplit('case "$ENGINE" in', 1)[1]
        assert "percona)" in outer
        inner = build_images.split('case "$base" in', 1)[1].split("esac", 1)[0]
        assert 'percona)' in inner
        assert "PERCONA_TAG" in inner

    def test_the_dockerfile_declares_the_arg_the_script_passes(self):
        dockerfile = open(os.path.join(VB_ROOT, "docker", "percona",
                                       "Dockerfile")).read()
        assert "ARG PERCONA_TAG" in dockerfile

    def test_prepare_sources_can_build_a_percona_context(self, prepare):
        assert "prepare_percona()" in prepare
        dispatch = prepare.rsplit('case "$ENGINE" in', 1)[1]
        assert "percona)" in dispatch

    def test_it_archives_rather_than_clones_with_submodules(self, prepare):
        """`git archive` of the pinned commit, and no submodule init -- all
        four are switched off by cmake flags instead."""
        body = prepare.split("prepare_percona()", 1)[1].split("\n}", 1)[0]
        assert "git archive" in body
        assert "submodule" not in body

    def test_it_verifies_the_commit_it_resolved_is_the_pinned_one(self, prepare):
        """A SHA that is passed along without being checked is a string, not a
        pin."""
        body = prepare.split("prepare_percona()", 1)[1].split("\n}", 1)[0]
        assert "rev-parse" in body
        assert 'resolved" == "$sha' in body

    def test_the_vendor_repo_guard_covers_the_percona_checkout(self, lib):
        """`assert_not_vendor_repo` loops over an explicit list. A repo absent
        from it gets NO protection, while the ones beside it make the rule
        look enforced. CLAUDE.md names these as read-only inputs."""
        assert "VB_REPO_PERCONA=" in lib
        loop = lib.split("assert_not_vendor_repo()", 1)[1].split("\n}", 1)[0]
        for var in ("VB_REPO_MARIADB", "VB_REPO_ALISQL", "VB_REPO_ANNB",
                    "VB_REPO_PERCONA"):
            assert var in loop, var

    def test_every_exported_vendor_repo_is_in_the_guard_loop(self, lib):
        """The general form, so the eighth engine fails here rather than
        shipping an unprotected checkout."""
        exported = set(re.findall(r"^export (VB_REPO_\w+)=", lib, re.M))
        loop = lib.split("assert_not_vendor_repo()", 1)[1].split("\n}", 1)[0]
        in_loop = set(re.findall(r"\$(VB_REPO_\w+)", loop))
        assert exported, "no vendor repos are exported; this test is stale"
        assert exported == in_loop, exported ^ in_loop


# ---------------------------------------------------------------------------
# §4.6 -- the profiles
# ---------------------------------------------------------------------------

class TestTheProfiles:

    def test_smoke_no_longer_claims_six(self):
        text = open(os.path.join(VB_ROOT, "config", "profiles",
                                 "smoke.yml")).read()
        _prose("the engine count is seven", text)

    def test_smoke_names_no_engines_so_percona_joins_automatically(self):
        from orchestrator.config import load_profile
        assert "engines" not in load_profile("smoke")

    def test_the_percona_profile_runs_only_percona(self):
        from orchestrator.config import load_profile
        assert load_profile("percona")["engines"] == ["percona"]

    def test_every_profile_still_loads(self):
        """A new YAML file in config/profiles/ is picked up by name, so a
        malformed one fails only when someone runs it."""
        from orchestrator.config import load_profile
        directory = os.path.join(VB_ROOT, "config", "profiles")
        names = sorted(f[:-4] for f in os.listdir(directory)
                       if f.endswith(".yml"))
        assert "percona" in names
        for name in names:
            assert load_profile(name)["name"] == name


class TestEveryShippedProfileRenders:
    """`render_config` called against every profile in the repo, for every
    engine that profile would run, in both resource passes.

    **This class exists because its absence let a shipped profile trip a
    refusal added in the same change.** `config/profiles/percona.yml` set
    `ann.percona_build_modes: [post, incremental]`, which
    `ann_pass.render_config` policed -- so the Percona-only profile's ann
    phase raised, and nothing noticed: every other `render_config` test in
    this repo either uses the `quick` profile or builds a synthetic dict, and
    `test_every_profile_still_loads` only parses the YAML.

    Both call sites are covered by this, and the second is the louder one:
    `cli._run_unit` catches and marks a unit failed with a traceback, while
    `cli.cmd_render` has NO handler and loops over KNOWN_ENGINES -- so a
    render of a one-engine profile tracebacks out of the process.
    """

    @staticmethod
    def _profiles():
        directory = os.path.join(VB_ROOT, "config", "profiles")
        return sorted(f[:-4] for f in os.listdir(directory)
                      if f.endswith(".yml"))

    def test_the_profile_list_is_not_empty(self):
        """Non-vacuity: an empty listing would make the parametrized tests
        below collect nothing and the suite report success."""
        names = self._profiles()
        assert len(names) >= 9, names
        assert "percona" in names

    @pytest.mark.parametrize("resource_pass", ["normalized", "tuned"])
    def test_every_profile_renders_for_every_engine_it_runs(self,
                                                             resource_pass):
        from orchestrator import ann_pass
        from orchestrator.cli import KNOWN_ENGINES
        from orchestrator.config import (load_profile, load_resources,
                                         merge_resource_overrides)
        checked = 0
        for name in self._profiles():
            profile = load_profile(name)
            # `engines:` when the profile names them, every known engine when
            # it does not -- which is what `config.profile_engines` and
            # `cli.cmd_run` resolve.
            #
            # **NOT what `cmd_render` does, though this comment said so until
            # the key became load-bearing.** `cmd_render` loops `KNOWN_ENGINES`
            # whatever the profile declares, deliberately, and
            # `test_cmd_render_renders_every_known_engine_when_told_none` below
            # asserts the literal expression that makes it so -- which is why a
            # profile naming one engine can still be asked to render seven, and
            # why a render failure for any of them is a whole-command failure.
            # One file holding a comment and a test that contradicted each
            # other about one function left a reader to conclude the pin was
            # the stale half.
            engines = profile.get("engines") or list(KNOWN_ENGINES)
            resources = merge_resource_overrides(
                load_resources(resource_pass), profile)
            for engine in engines:
                body = ann_pass.render_config(engine, profile, resources,
                                              resource_pass)
                groups = body["float"]["any"][0]["run_groups"]
                assert groups, f"{name}/{engine}: no run groups"
                for group in groups.values():
                    assert group["arg_groups"], f"{name}/{engine}"
                checked += 1
        assert checked >= 40, checked

    def test_cmd_render_renders_every_known_engine_when_told_none(self):
        """The property that turns a one-engine profile's render error into a
        whole-command failure, read out of `cli.py` rather than assumed."""
        source = open(os.path.join(VB_ROOT, "orchestrator", "cli.py")).read()
        body = source.split("def cmd_render", 1)[1].split("\ndef ", 1)[0]
        assert "args.engines.split(\",\") if args.engines else KNOWN_ENGINES" \
            in body
        # ...and it does not guard the call, which is why the profile had to
        # be the thing that changed. Stated so that a future `try` here is a
        # deliberate decision rather than a silent one.
        assert "try:" not in body

    def test_the_percona_profile_renders_both_of_its_axes(self):
        """The regression itself, named: this is the call that raised."""
        from orchestrator import ann_pass
        from orchestrator.config import (load_profile, load_resources,
                                         merge_resource_overrides)
        from orchestrator import estimate as est_mod
        profile = load_profile("percona")
        for resource_pass in ("normalized", "tuned"):
            resources = merge_resource_overrides(
                load_resources(resource_pass), profile)
            body = ann_pass.render_config("percona", profile, resources,
                                          resource_pass)
            assert sorted(body["float"]["any"][0]["run_groups"]) == ["innodb"]
        # And the ops axis is one unit, not because the key move failed to
        # preserve two -- it did, and `test_the_estimate_counts_both_modes_as_
        # two_ops_units` still proves the estimator counts two when two are
        # asked for -- but because the profile now asks for one. `post` kills
        # the server at this pin; the capability in
        # `config/engines/percona.yml` carries the evidence.
        assert est_mod._ops_units(profile, "percona") == 1


# ---------------------------------------------------------------------------
# Coverage this addition would otherwise have lost
# ---------------------------------------------------------------------------

class TestTheInvariantsKeyedOnHardCodedEngineTuples:
    """Three existing tests iterate a hand-written engine list rather than the
    registry, so a seventh engine is silently NOT covered by them -- they go
    on passing while checking six of seven. Those tests live in suites this
    task may not add to, so the missing coverage is supplied here, derived
    from the registry so the eighth engine cannot slip through the same way.
    """

    @staticmethod
    def _resolved(engine, resource_pass):
        from orchestrator.config import load_resources, resolve_resources
        info = type("I", (), {})()
        info.cpu = type("C", (), {"arch": "x86_64", "hybrid": False,
                                  "performance_cpus": list(range(40)),
                                  "efficiency_cpus": [], "logical_cpus": 80,
                                  "physical_cores": 40, "threads_per_core": 2,
                                  "model": "Xeon"})()
        info.total_ram_bytes = int(192 * 1024 ** 3)
        return resolve_resources(load_resources(resource_pass), engine, info)

    def test_the_ann_fingerprint_is_still_engine_invariant(self):
        """`test_config.py`'s version iterates a six-name tuple.

        The fingerprint keys the shared `results/annb/` tree by the resource
        pass's declared knobs and is deliberately engine-INVARIANT: making it
        engine-specific fragments the tree and produces a seven-engine recall
        chart containing one engine. It has broken twice before, both times
        by hashing something that differs per engine.
        """
        from orchestrator import engines as engines_mod
        from orchestrator.ann_pass import ann_fingerprint
        names = sorted(engines_mod.registry(refresh=True))
        assert "percona" in names and len(names) >= 7, names
        for resource_pass in ("normalized", "tuned"):
            prints = {e: ann_fingerprint(self._resolved(e, resource_pass))
                      for e in names}
            assert len(set(prints.values())) == 1, (resource_pass, prints)

    def test_the_two_passes_still_do_not_collide_for_the_new_engine(self):
        from orchestrator.ann_pass import ann_fingerprint
        assert (ann_fingerprint(self._resolved("percona", "normalized"))
                != ann_fingerprint(self._resolved("percona", "tuned")))

    def test_percona_resolves_resources_at_all(self):
        """An engine the resource layer cannot resolve dies after the image is
        built and the server is up."""
        resolved = self._resolved("percona", "normalized")
        assert resolved.buffer_bytes > 0
        assert resolved.server_memory_bytes > 0

    def test_the_overlay_module_reports_nothing_hdf5_cannot_store(self):
        """`test_config.py`'s version iterates a six-name tuple too.

        ann-benchmarks' `store_results` writes `get_additional()` into HDF5
        attributes, and `None`, a list or a dict raises `TypeError: Object
        dtype dtype('O') has no native HDF5 equivalent` -- which ann-benchmarks
        catches per worker and exits ZERO for, so the phase is reported
        complete and writes no results. Percona Search failed that way four
        times across two datasets while its measurements ran perfectly.
        """
        path = os.path.join(VB_ROOT, "overlay", "ann-benchmarks",
                            "ann_benchmarks", "algorithms", "percona",
                            "module.py")
        source = open(path).read()
        assert "def get_additional" in source           # needle findable
        block = source.split("def get_additional", 1)[1].split("\n    def ",
                                                                1)[0]
        offenders = re.findall(r'"(\w+)":\s*(None\b|\[|\{)', block)
        assert not offenders, offenders
