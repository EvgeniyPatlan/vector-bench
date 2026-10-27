"""E19 -- §3's asymmetry rows must be about the engines that ran.

THE DEFECT
----------
`report/render.py:_known_asymmetries` says it is "conditional on what actually
ran", and two of its rows are not.

* The `ef_construction` row computed its subject from the literal set
  `{"pgvector", "valkey"}`. `percona` is not in it, so a run whose engine set
  is Percona Server alone intersected to nothing, fell through to the
  single-engine arm and published

      `ef_construction` is exposed only by pgvector

  in a report about the one engine in this study that has BOTH a build-time
  and a query-time effort knob. `config/engines/percona.yml` declares
  `capabilities.ef_construction_tunable: true`, `PerconaDriver.capabilities`
  returns it, and `CLAUDE.md`'s invariants bullet names pgvector, Valkey
  **and** Percona Server -- so the report contradicted the repository, the
  engine config and the driver at once, and the sentence was false in the one
  report where the knob was the subject of the investigation.

* The next row, "pgvector bulk-builds after load; MHNSW and VIDX build
  incrementally", was unconditional. In a Percona-only report it named three
  engines, none of which ran, and said nothing about the one that did --
  whose `build_mode` is the only within-engine bulk/incremental comparison in
  the MySQL family (`docs/09-adding-percona-server.md` §2, "Separable bulk
  build: **yes**").

`docs/05-methodology.md` §4's static table carried the same claim and a second
one beside it: its consequence column said the build-quality sweep "can only
vary `M` for the MySQL-family engines", which Percona Server -- a MySQL-family
engine with the knob -- makes false rather than merely incomplete.

WHAT IS PINNED HERE
-------------------
* `TestAPerconaRunIsNotToldTheKnobBelongsToPgvector` -- the headline case.
  The two-engine set passed before and after the fix, so the only test that
  can fail on the defect is one whose engine set contains Percona.
* `TestEveryEngineThatExposesTheKnobIsNamed` -- the general shape: the row is
  a list derived from the records, not a sentence with a length.
* `TestARunWithNoTunableEngineSaysSo` -- the arm that rules out the obvious
  wrong discriminator. `harness/main.py` sets `IndexSpec.ef_construction`
  from `--ef-construction` for every engine, so `Record.ef_construction` is
  populated for MariaDB too; a `tunable` derived from that field being
  non-null would name MariaDB and AliSQL here.
* `TestTheBuildModeRowNamesTheEnginesThatBuilt` -- the second row, including
  the asynchronous case, which `separable_build` alone cannot tell from a
  bulk build.
* `TestTheMethodologyTableSaysTheSameThing` -- the static table in
  `docs/05-methodology.md` §4, and the one cross-check that can hold it to
  the generated list.

Nothing here reads `results/`: every summary below is built from record dicts
shaped the way `harness/workloads/build.py` writes them.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any, Dict, List

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from report import generate, render  # noqa: E402

VB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METHODOLOGY_DOC = os.path.join(VB_ROOT, "docs", "05-methodology.md")
ENGINES_DIR = os.path.join(VB_ROOT, "config", "engines")

#: The knob's name as §3 and §4 both spell it, and the needle every row
#: lookup below uses. Backticked, because `ef_construction` unbackticked also
#: appears in prose about the sweep.
EF = "`ef_construction`"


def _records(spec: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """`ingest` + `index_build` pairs, in the shape `build.py` writes them.

    Per engine, `spec` takes `tunable` (what the driver's `capabilities()`
    reported, omitted entirely to model a record written before
    `_driver_capabilities` existed), `separable` (one `index_build` record per
    value, which is how an engine swept over both build modes appears) and
    `async_build`.

    `ef_construction` is populated on every record on purpose: `main.py` in
    the harness sets it from `--ef-construction` for every engine, so a
    discriminator read off that field would pass
    `TestEveryEngineThatExposesTheKnobIsNamed` and fail
    `TestARunWithNoTunableEngineSaysSo`. Leaving it None would hide the
    difference between the right discriminator and that wrong one.
    """
    out: List[Dict[str, Any]] = []
    for engine, how in sorted(spec.items()):
        for separable in (how.get("separable") or (False,)):
            extra: Dict[str, Any] = {}
            if "tunable" in how:
                extra["ef_construction_tunable"] = how["tunable"]
            if how.get("async_build"):
                extra["async_index_build"] = True
            if how.get("separable") is not None:
                extra["separable_build"] = separable
            base = {
                "engine": engine, "dataset": "d", "resource_pass": "tuned",
                "m": 16, "storage_engine": "heap", "ef_construction": 200,
                "build_mode": ("async" if how.get("async_build")
                               else "post" if separable else "incremental"),
            }
            out.append({**base, "phase": "ingest",
                        "ingest_rows_per_s": 900.0})
            out.append({**base, "phase": "index_build", "build_wall_s": 300.0,
                        "index_bytes": 7 * 1024 ** 3, "extra": extra})
    return out


def _text(spec: Dict[str, Dict[str, Any]]) -> str:
    return render._known_asymmetries(generate.summarize(_records(spec), {}))


def _table_rows(text: str) -> List[str]:
    """The body rows of the one Markdown table in the section."""
    return [line for line in text.splitlines()
            if line.startswith("|") and not re.match(r"^\|\s*-{3}", line)
            and not line.startswith("| Asymmetry")]


def _row(text: str, needle: str) -> str:
    """The single row carrying `needle`, both cells of it.

    Both cells, because a table row is both cells: the subject of the claim
    is in the first and what it costs these results is in the second, and a
    guard that reads only one of them can be satisfied by a row that
    contradicts itself across the delimiter.
    """
    found = [line for line in _table_rows(text) if needle in line]
    assert len(found) == 1, (
        f"{needle!r} matched {len(found)} rows, expected 1:\n"
        + "\n".join(found))
    return found[0]


def _exposed_by(text: str) -> str:
    """The clause naming the engines the row says DO expose the knob.

    Scoped, and the scope is load-bearing. "`Percona` is somewhere in the
    row" is satisfied by a row reading "can be tuned for pgvector and Valkey,
    and not for Percona Server" -- which is the false claim this unit exists
    to withdraw, and an unscoped assertion accepts it. Both of the mutants
    below were measured against the unscoped form first and one of them
    survived it.
    """
    row = _row(text, EF)
    clause = re.search(r"exposed (?:only )?by ([^|.]*)", row)
    assert clause, f"the row makes no 'exposed by' claim at all:\n{row}"
    return clause.group(1)


def _rows_about_build_mode(text: str) -> str:
    """Every row about how the index came to exist, joined.

    Two rows rather than one: an engine that did both modes earns its own
    row, so an assertion about "the bulk/incremental row" has to look at the
    pair or it passes by reading the wrong half.
    """
    return "\n".join(line for line in _table_rows(text)
                     if "bulk" in line.lower() or "incrementally" in line)


def _methodology_section_4() -> str:
    text = open(METHODOLOGY_DOC).read()
    section = text.split("## 4. Known asymmetries")[1].split("## 5.")[0]
    assert section.strip(), "§4 is empty"
    return section


def _configured_tunable_engines() -> List[str]:
    """Engines whose config declares the knob, straight from `config/engines`.

    `KNOWN_ENGINES` is whatever that directory holds and there is no list in
    Python to keep in step with it (`CLAUDE.md`), so the set of engines §4 has
    to name is read from the same place the drivers' own tests read it.
    """
    found = []
    for name in sorted(os.listdir(ENGINES_DIR)):
        if not name.endswith(".yml"):
            continue
        cfg = yaml.safe_load(open(os.path.join(ENGINES_DIR, name))) or {}
        if (cfg.get("capabilities") or {}).get("ef_construction_tunable"):
            found.append(os.path.splitext(name)[0])
    assert found, "no engine config declares ef_construction_tunable"
    return found


#: How §4 names, in prose, each engine that exposes the knob. Three entries
#: and a completeness assertion rather than a derivation: the document writes
#: "pgvector" where the chart legend writes "PostgreSQL (pgvector)", so there
#: is nothing to derive it from -- and the assertion is what makes a fourth
#: tunable engine fail this file instead of being silently unchecked.
DOC_NAME = {"pgvector": "pgvector", "valkey": "Valkey",
            "percona": "Percona"}


class TestAPerconaRunIsNotToldTheKnobBelongsToPgvector:
    """The run the old literal got wrong, and the only shape that can fail.

    A Percona-only run at the current pin: one `index_build` record,
    incremental, `ef_construction_tunable: True` -- which is exactly what
    `percona-20261021-134142`'s ops record carries.
    """

    SPEC = {"percona": {"tunable": True, "separable": [False]}}

    def test_the_knob_is_not_attributed_to_an_engine_that_did_not_run(self):
        assert "only by pgvector" not in _text(self.SPEC)

    def test_the_row_names_percona(self):
        assert "Percona" in _exposed_by(_text(self.SPEC))

    def test_the_row_does_not_name_pgvector_at_all(self):
        """Not merely "not alone": pgvector is absent from the run, so the
        row has nothing to say about it in either cell."""
        assert "pgvector" not in _row(_text(self.SPEC), EF)

    def test_it_says_percona_has_both_knobs(self):
        """`CLAUDE.md`'s invariant: the build-time knob is only half of what
        makes this engine the interesting one, and a reader of a report about
        `ef_construction` has to be told the query-time knob exists too."""
        row = _row(_text(self.SPEC), EF)
        assert "build-time" in row and "query-time" in row

    def test_the_pin_policy_survives(self):
        """The rule the false sentence was justifying is unchanged: the knob
        is pinned to its default in the normalized pass."""
        row = _row(_text(self.SPEC), EF)
        assert "normalized" in row and "pinned" in row


class TestEveryEngineThatExposesTheKnobIsNamed:

    SPEC = {
        "percona": {"tunable": True, "separable": [False]},
        "pgvector": {"tunable": True, "separable": [True]},
        "valkey": {"tunable": True, "separable": [True]},
    }

    def test_all_three_are_named_in_the_row(self):
        exposed = _exposed_by(_text(self.SPEC)).lower()
        for engine in ("percona", "pgvector", "valkey"):
            assert engine in exposed, engine

    def test_it_no_longer_claims_one_engine_has_it_alone(self):
        assert "only by" not in _row(_text(self.SPEC), EF)

    def test_a_two_engine_run_still_reads_as_it_did(self):
        """The set that was right before this change stays right after it:
        the fix is a widening, not a rewrite."""
        spec = {k: v for k, v in self.SPEC.items() if k != "percona"}
        exposed = _exposed_by(_text(spec))
        assert "pgvector" in exposed and "Valkey" in exposed
        assert "percona" not in exposed.lower()

    def test_a_single_engine_run_says_only(self):
        """One engine with the knob is still an "only", and the sentence is
        parameterised by the engine found rather than by pgvector -- which is
        what keeps `tests/test_config.py`'s pgvector-only assertion true."""
        spec = {"pgvector": {"tunable": True, "separable": [True]}}
        assert "only by pgvector" in _row(_text(spec), EF)

    def test_an_engine_whose_records_predate_the_capability_key_is_placed(
            self):
        """A record written before `_driver_capabilities` existed states
        nothing, and the three engines that expose the knob did so before the
        key was recorded. Reading that silence as "no" would withdraw a true
        claim from every archived run."""
        spec = {"pgvector": {"separable": [True]},
                "valkey": {"separable": [True]}}
        exposed = _exposed_by(_text(spec))
        assert "pgvector" in exposed and "Valkey" in exposed

    def test_a_recorded_false_beats_the_fallback(self):
        """The records are the authority, not the fallback: an engine whose
        driver reports False is not tunable however it is spelled here.

        Naming it as an engine that LACKS the knob is correct and is what the
        row then does, so the assertion is on the claim and not on the token.
        """
        exposed = _exposed_by(_text({"pgvector": {"tunable": False,
                                                  "separable": [True]}}))
        assert "none" in exposed.lower()
        assert "pgvector" not in exposed


class TestARunWithNoTunableEngineSaysSo:
    """MariaDB and AliSQL, which is the arm the wrong discriminator fails."""

    SPEC = {"mariadb": {"tunable": False, "separable": [False]},
            "alisql": {"tunable": False, "separable": [False]}}

    def test_the_row_says_no_engine_here_exposes_it(self):
        assert "none" in _exposed_by(_text(self.SPEC)).lower()

    def test_it_does_not_name_pgvector_as_present(self):
        assert "pgvector" not in _exposed_by(_text(self.SPEC))
        assert "pgvector" not in _row(_text(self.SPEC), EF)
        assert "only by pgvector" not in _text(self.SPEC)

    def test_the_sweep_consequence_is_stated(self):
        """A run with no knob to vary sweeps `M` alone, and the row has to
        say so or it is a row about an absence with no consequence."""
        assert "`M`" in _row(_text(self.SPEC), EF)


class TestTheBuildModeRowNamesTheEnginesThatBuilt:

    def test_it_names_percona_when_percona_bulk_built(self):
        text = _text({"percona": {"tunable": True, "separable": [True]}})
        assert "Percona" in _rows_about_build_mode(text)

    def test_it_does_not_name_percona_when_only_mariadb_ran(self):
        text = _text({"mariadb": {"tunable": False, "separable": [False]}})
        rows = _rows_about_build_mode(text)
        assert rows, "nothing was said about how MariaDB's index was built"
        assert "Percona" not in rows
        assert "pgvector" not in rows

    def test_both_sides_are_named_from_the_records(self):
        text = _text({"pgvector": {"tunable": True, "separable": [True]},
                      "mariadb": {"tunable": False, "separable": [False]}})
        rows = _rows_about_build_mode(text)
        assert "pgvector" in rows and "MariaDB" in rows

    def test_an_engine_that_did_both_modes_gets_its_own_row(self):
        """Percona swept over both modes is a comparison within one engine,
        and the two halves are not the same operation: its first *ADD VECTOR
        KEY* rebuilds the base table where pgvector's *CREATE INDEX* does
        not, so one "the bulk build is faster" figure hides a table rebuild
        inside one engine's number."""
        text = _text({"percona": {"tunable": True,
                                  "separable": [True, False]}})
        row = _row(text, "either")
        assert "within one engine" in row
        assert "ADD VECTOR KEY" in row
        assert "rebuild" in row.lower()

    def test_one_mode_earns_no_within_engine_row(self):
        """At this pin `capabilities.bulk_vector_index_build` is false and
        `config/profiles/percona.yml` runs incremental alone, so a report of
        such a run must not claim the engine did either: the row is about
        what ran."""
        text = _text({"percona": {"tunable": True, "separable": [False]}})
        assert "does either" not in text

    def test_an_async_build_is_not_filed_as_a_bulk_build(self):
        """mongot's `separable_build` is True and its build is neither bulk
        nor incremental -- `_index_build_kind` already splits the four kinds
        for the build table, and this row has to split them the same way or
        it contradicts the asynchronous row below it."""
        text = _text({"mongodb": {"separable": [True],
                                  "async_build": True}})
        assert not _rows_about_build_mode(text)
        assert "asynchronously" in text

    def test_an_unrecorded_build_is_not_guessed(self):
        """A record with no `separable_build` at all cannot be placed on
        either side -- the same three-way split `generate.build_kind` makes
        for the chart and the build table."""
        text = _text({"mariadb": {"tunable": False, "separable": None}})
        assert not _rows_about_build_mode(text)


class TestTheMethodologyTableSaysTheSameThing:
    """§4's static table, and how far a test can hold it to §3's generated one.

    The generated list is per run and the document's table is about the study,
    so the two cannot be asserted equal: a Percona-only report legitimately
    names one engine where the document names three. What a test can pin is
    the study-level claim they both make -- which engines expose the knob at
    all -- and that is the claim that went stale in both places at once.
    """

    def test_the_row_names_percona_and_valkey(self):
        row = [line for line in _methodology_section_4().splitlines()
               if EF in line]
        assert len(row) == 1, row
        assert "Percona" in row[0] and "Valkey" in row[0]

    def test_it_no_longer_says_only_m_can_vary_for_the_mysql_family(self):
        """The second false claim in the same cell. Percona Server is a
        MySQL-family engine with the knob, so this is wrong rather than
        incomplete."""
        doc = open(METHODOLOGY_DOC).read()
        assert "can only vary `M` for the MySQL-family engines" not in doc

    def test_every_engine_whose_config_declares_the_knob_is_named(self):
        """The ratchet. Both stale sentences were stale because an engine
        landed and the claim about the engine set was not revisited."""
        row = [line for line in _methodology_section_4().splitlines()
               if EF in line][0]
        for engine in _configured_tunable_engines():
            assert engine in DOC_NAME, (
                f"{engine} declares ef_construction_tunable and this file "
                f"does not know how §4 names it")
            assert DOC_NAME[engine] in row, engine

    def test_the_generated_row_names_the_same_engine_set(self):
        """The cross-check, in the one direction that is a claim about the
        study: a run that contains every engine the document names must
        generate a row naming every engine the document names."""
        spec = {engine: {"tunable": True, "separable": [True]}
                for engine in _configured_tunable_engines()}
        exposed = _exposed_by(_text(spec)).lower()
        for engine in _configured_tunable_engines():
            assert DOC_NAME[engine].lower() in exposed, engine

    def test_neither_side_claims_the_knob_for_mhnsw_or_vidx(self):
        """The half of the old sentence that was true, in both places."""
        doc_row = [line for line in _methodology_section_4().splitlines()
                   if EF in line][0]
        assert "MHNSW" in doc_row and "VIDX" in doc_row
        text = _text({"mariadb": {"tunable": False, "separable": [False]},
                      "alisql": {"tunable": False, "separable": [False]},
                      "percona": {"tunable": True, "separable": [False]}})
        generated, exposed = _row(text, EF), _exposed_by(text)
        assert "MariaDB" in generated and "AliSQL" in generated
        assert "MariaDB" not in exposed and "AliSQL" not in exposed
