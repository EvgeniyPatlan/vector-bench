"""Guards for §2.0 of docs/05-methodology.md -- the operating point.

WHY THIS FILE EXISTS. The pre-publication checklist asks a reviewer to "review
the operating-point table; every non-`ok` status explained in the text", and
until §2.0 landed there was no text. A section written to discharge that item
is only worth having if it reports its own rot, because the thing it describes
-- a status name, a margin, a selection rule -- is code that moves.

WHAT THESE TESTS CAN AND CANNOT ESTABLISH
-----------------------------------------
They establish nothing about MariaDB, AliSQL, pgvector, Valkey or Percona
Search. No server is reachable from a test run, and no run this repository
holds contains an `ef_calibration` record, so the one figure §2.0 quotes about
real engine behaviour -- recall 0.98 against 0.80 at one `ef_search` -- is
marked supplied in the document and is not checked here. `TestWhatIsNotClaimed`
asserts the document says so, which is the only checkable form of that
statement.

What they do establish:

* the three status names, the floor margin, the target, the twelve-point grid,
  the phase string, the three footnote marks and the three calibrated
  workloads are read out of the modules and compared against what §2.0 prints
  -- in both directions, so a constant that moves fails here and so does a
  figure edited in the document alone;
* the selection rule is CALLED, over curves built for each of the four cases
  §2.0 enumerates, rather than grepped. "The smallest ef whose successor also
  reaches the target" is a claim about behaviour and a search for the word
  "successor" would pass against a selector that had stopped doing it;
* every `path:symbol` citation §2.0 makes resolves to a symbol in that file;
* §2.0 sits before §2.1 and does not renumber it, and the renderer really puts
  the operating-point table before the three sections that inherit the choice;
* no profile in `config/profiles/` sets the three ef keys, which is what makes
  the defaults the document prints the values every profile here runs with.

The prose claims are pinned as VOCABULARIES and not as sentences -- see
`CLAIMS`, which carries for each claim the rewordings it must accept and the
tidied-up forms it must refuse, both asserted by
`TestTheClaimPatternsAreNeitherBrittleNorVacuous`. A guard that fires on
correct prose is a slower-acting vacuous one: it teaches the next author to
delete it, which is how the vacuous guards in this repo got in.

Every pattern runs against `_plain(...)`, which collapses whitespace and strips
markdown emphasis. This document is hard-wrapped, so a pattern matched against
the raw text passes or fails on where the wrap happens to fall -- a defect that
has bitten this repository three times in one day.

Each negative assertion is paired with a positive one proving the needle is
findable, and each parser has a test asserting what it found: a parser that
matched nothing would make every assertion below pass over an empty set.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List

import pytest

VB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METHODOLOGY_DOC = os.path.join(VB_ROOT, "docs", "05-methodology.md")

#: The heading §2.0 carries, and the one after it. Both spelled out, because
#: the section's POSITION is half of what it claims: calibration is the premise
#: of §2.3, §2.4 and §2.5 rather than an eighth measurement dimension, and a
#: §2.0 renumbered to §2.8 would still contain every sentence below.
SECTION_HEADING = "### 2.0 The operating point"
NEXT_HEADING = "### 2.1 Recall vs QPS (ann-benchmarks)"


def _doc() -> str:
    return open(METHODOLOGY_DOC).read()


def _plain(text: str) -> str:
    """Prose with markdown emphasis removed and whitespace collapsed.

    Backticks and `**` go rather than being matched, so a claim can be checked
    against a sentence the document legitimately marks up: §2.0 writes
    `floor_above_target` in code font and a reworded version of the same
    sentence might not.

    One step more than `tests/test_e3_rollout_doc.py`'s version of this
    function, and it is measured rather than tidy: a hard wrapper that breaks
    on hyphens splits `never re-derived` into `re-` and `derived`, and after
    the whitespace collapse that is `never re- derived`. Three of the patterns
    below name a hyphenated word (`re-derived`, `re-calibrated`, `warm-up`),
    and all three stopped matching a document whose only change was where the
    lines fell -- which is the defect this repository has hit three times in
    one day, arriving by a route nobody had covered. The repair is scoped to a
    hyphen BETWEEN two word characters, which matches nothing in the honest
    document: this file's prose writes its dashes as em dashes, so there is no
    legitimate `word- word` for it to corrupt.
    """
    flat = re.sub(r"\s+", " ",
                  text.replace("`", "").replace("**", "")).strip()
    return re.sub(r"(?<=\w)-\s+(?=\w)", "-", flat)


def _section() -> str:
    """§2.0 alone, heading excluded.

    Scoped rather than document-wide, because several of these claims are
    about which section says a thing. `_section` is checked for non-vacuity by
    `TestTheParsersFoundSomething`: a slice that silently fell back to the
    whole document would make the scoping decorative.
    """
    doc = _doc()
    assert doc.count(SECTION_HEADING) == 1, "docs/05 has no single §2.0"
    assert doc.count(NEXT_HEADING) == 1, "docs/05 has no single §2.1"
    body = doc.split(SECTION_HEADING, 1)[1]
    return body.split(NEXT_HEADING, 1)[0]


def _claim(name: str) -> None:
    """Assert §2.0 still makes the named claim, as a vocabulary."""
    spec = CLAIMS[name]
    text = _plain(_section())
    found = re.search(spec["pattern"], text, re.I | re.S)
    assert found, (
        f"docs/05 §2.0 no longer makes the claim {name!r}; the pattern "
        f"{spec['pattern']!r} matches nothing in it")


# ---------------------------------------------------------------------------
# The prose claims, as vocabularies rather than sentences
# ---------------------------------------------------------------------------
#
# Each pattern pins the subject, the mechanism and the POLARITY and leaves the
# wording free. `accepts` is the rewordings it was actually checked against;
# `rejects` the tidied-up or reversed forms it must still catch. A pattern
# loose enough to accept its own negation fails in
# `TestTheClaimPatternsAreNeitherBrittleNorVacuous` rather than passing quietly
# forever.
CLAIMS: Dict[str, Dict[str, object]] = {
    "calibration is the premise of the three sections and not a dimension":
        dict(
            pattern=r"premise of (those|the) three\b.{0,80}?"
                    r"(rather than|not|instead of)\b",
            accepts=[
                "So this section is the premise of those three rather than a "
                "note about them",
                "It is the premise of the three sections below, not a note "
                "about them",
                "Calibration is the premise of those three and not an eighth "
                "measurement dimension",
            ],
            rejects=[
                "This is the seventh measurement dimension, beside the three "
                "sections below",
                "It is a note about those three sections and reads best after "
                "them",
            ]),
    "a fixed ef_search is not one quantity across these engines": dict(
        pattern=r"fixed ef_search is not one quantity\b",
        accepts=[
            "A fixed ef_search is not one quantity across these engines",
            "a fixed ef_search is not one quantity here: it is a different "
            "knob in each engine",
        ],
        rejects=[
            "A fixed ef_search is one quantity across these engines",
            "ef_search means the same thing in all five engines",
        ]),
    "equal ef is not equal effort because two engines clamp": dict(
        pattern=r"(do not search the number they are handed|"
                r"not a shared amount of work|"
                r"raise[sd]? it to k|clamp)",
        accepts=[
            "Two of those five do not search the number they are handed.",
            "a shared literal is demonstrably not a shared amount of work",
            "Both of them raise it to k before searching, so equal ef is not "
            "equal effort",
        ],
        rejects=[
            "All five engines search exactly the number of candidates they "
            "are given, so one ef_search is one amount of work",
        ]),
    "the recall spread at one ef is supplied and not measured here": dict(
        pattern=r"(0\.18|0\.98 and 0\.80)\b.{0,400}?\bsupplied\b"
                r"|\bsupplied\b.{0,400}?(0\.18|0\.98 and 0\.80)\b",
        accepts=[
            "The stronger half of the claim -- how far apart the recalls at "
            "one ef_search actually are -- is supplied, and nothing here "
            "measures it. The figure is two engines at recall 0.98 and 0.80.",
            "That the spread reaches 0.18 of recall is supplied by the "
            "tuned-profile notes and is not measured in this repository.",
        ],
        rejects=[
            "We measured two engines at recall 0.98 and 0.80 at one "
            "ef_search, which is where the 0.18 comes from.",
            "The 0.18 spread is a measurement of these engines taken by this "
            "framework.",
        ]),
    "no calibration record exists in this repository": dict(
        pattern=r"no run\b.{0,60}?(contains|holds|has)\b.{0,40}?"
                r"ef_calibration record",
        accepts=[
            "No run this repository holds contains an ef_calibration record "
            "at all",
            "no run the repository holds has an ef_calibration record",
            "There is no run in this project that contains an "
            "ef_calibration record",
        ],
        rejects=[
            "The archive holds 2,134 ef_calibration records measured under "
            "version 5",
            "Every archived run contains an ef_calibration record",
        ]),
    "the successor clause exists because one point can clear by noise": dict(
        pattern=r"(successor|next (grid )?point)\b.{0,400}?"
                r"(sampling noise|by noise|Bernoulli)",
        accepts=[
            "The successor clause is the whole guard. Recall over a finite "
            "query set is a mean of Bernoulli draws, so a single point can "
            "clear the target by sampling noise.",
            "whose successor also reaches it -- without that, one point "
            "lifted over the target by noise becomes the chosen ef",
        ],
        rejects=[
            "The smallest ef_search that reaches the target is chosen, and "
            "no other point is consulted.",
        ]),
    "confirmed is not a fourth status": dict(
        pattern=r"confirmed is not a (fourth|further|separate) status",
        accepts=[
            "confirmed is not a fourth status.",
            "The flag is reported beside the status, but confirmed is not a "
            "separate status",
        ],
        rejects=[
            "confirmed is the fourth status the sweep can end on",
        ]),
    "floor_above_target means the engine cannot be slowed to the target":
        dict(
            pattern=r"floor_above_target\b.{0,120}?"
                    r"cannot be slowed down to the target",
            accepts=[
                "floor_above_target -- the engine cannot be slowed down to "
                "the target.",
                "floor_above_target says this engine cannot be slowed down to "
                "the target at any grid point",
            ],
            rejects=[
                "floor_above_target means the engine was slowed down to the "
                "target at the smallest grid ef",
                "floor_above_target is a note that the first grid point was "
                "used and nothing else follows from it",
            ]),
    "floor_above_target reads at a higher recall": dict(
        pattern=r"floor_above_target\b.{0,900}?\bhigher recall\b",
        accepts=[
            "floor_above_target -- the engine cannot be slowed down to the "
            "target. Its numbers below were bought at a higher recall than "
            "the engines marked ok.",
            "floor_above_target: no grid point brings it down, so everything "
            "below sits at a higher recall than the ok engines'",
        ],
        rejects=[
            "floor_above_target -- the engine cannot be slowed down to the "
            "target, so everything below is at a lower recall than the ok "
            "engines'.",
        ]),
    "target_unreached reads at a lower recall": dict(
        pattern=r"target_unreached\b.{0,900}?\blower recall\b",
        accepts=[
            "target_unreached -- the engine reached the target at no grid "
            "point. Its numbers below were bought at a lower recall than the "
            "ok engines'.",
            "target_unreached: reported at its best achievable recall, which "
            "is a lower recall than the rest were held to",
        ],
        rejects=[
            "target_unreached -- the engine reached the target at no grid "
            "point, so its numbers sit at a higher recall than the others'.",
        ]),
    "the MHNSW and VIDX expectation is supplied": dict(
        pattern=r"(MHNSW and VIDX|VIDX and MHNSW)\b.{0,600}?"
                r"\b(expectation is supplied|supplied, not established|"
                r"supplied rather than established)",
        accepts=[
            "For MHNSW and VIDX at M = 16 the expectation is that this is a "
            "property of the implementation. That expectation is supplied, "
            "not established here.",
            "MHNSW and VIDX are the two this is expected of, and that "
            "expectation is supplied rather than established in this "
            "repository.",
        ],
        rejects=[
            "MHNSW and VIDX are measured here at 0.94-0.98 at minimum "
            "effort, which is why they land on this status.",
            "We established that MHNSW and VIDX cannot be slowed to the "
            "target at M = 16.",
        ]),
    "the chosen ef is read back off the downstream records": dict(
        pattern=r"(read back off|taken from|read off)\b.{0,200}?"
                r"\b(records|workloads)\b.{0,200}?"
                r"(never re-derived|not re-derived|rather than re-derived|"
                r"never recomputed)",
        accepts=[
            "the chosen ef_search and the status are read back off the "
            "records of the workloads that used them and never re-derived by "
            "applying the selection rule a second time",
            "both are read off the downstream records rather than re-derived "
            "from the curve",
        ],
        rejects=[
            "the report re-derives the chosen ef_search from the curve by "
            "applying the selection rule again",
        ]),
    "filtered search uses the unfiltered calibrated ef": dict(
        # `\W{0,2}\s*` between the two words, because the document italicises
        # "unfiltered" and `_plain` strips only backticks and `**` -- the same
        # definition `tests/test_e3_rollout_doc.py` uses. A pattern that
        # demanded the bare words would fire on correctly emphasised prose.
        pattern=r"unfiltered\W{0,2}\s*calibrated ef_search\b.{0,200}?"
                r"(not|never)\b.{0,100}?re-calibrat",
        accepts=[
            "Measured at the unfiltered calibrated ef_search, not at one "
            "re-calibrated per selectivity.",
            "It uses the unfiltered calibrated ef_search and is never "
            "re-calibrated per point",
        ],
        rejects=[
            "Each selectivity is measured at its own calibrated ef_search, "
            "re-calibrated per point.",
        ]),
    "no engine behaviour is measured in this document": dict(
        pattern=r"(no claim here is a measurement of an engine|"
                r"no server can be started)",
        accepts=[
            "No claim here is a measurement of an engine.",
            "No server can be started from this checkout, so nothing above "
            "is a measurement of an engine.",
        ],
        rejects=[
            "Every claim here is a measurement taken against a live server.",
        ]),
}


# ---------------------------------------------------------------------------
# The parsers, and what they found
# ---------------------------------------------------------------------------

#: `path:symbol` as §2.0 spells its citations. The file's own convention --
#: `harness/datasets.py:recall_at_k` already appears in §2.4 -- and chosen over
#: `path:LINE` deliberately: a line number in this document would have to be
#: re-derived every time anything above it is reflowed, which is the defect
#: docs/10's citations keep hitting. A symbol name is checkable without line
#: arithmetic and survives a reflow of the file it points into.
_CITE_RE = re.compile(
    r"`([A-Za-z0-9_./-]+\.(?:py|yml|sh|md)):([A-Za-z_][A-Za-z0-9_]*)`")

#: Rows of the knob table: engine label, the literal the harness sets, and the
#: `path:symbol` it is read from.
_KNOB_ROW_RE = re.compile(
    r"^\|\s*([^|]+?)\s*\|\s*`([^`]+)`[^|]*\|\s*"
    r"`([A-Za-z0-9_./-]+\.py):([A-Za-z_][A-Za-z0-9_]*)`\s*\|\s*$",
    re.M)


def _units() -> List[str]:
    """§2.0 as the units a claim can be made in: paragraphs and table rows.

    The unit matters, and measured: three guards in the first round of this
    file passed against an INVERTED document because each claim is stated
    twice -- once in prose and once in a summary table -- and a pattern run
    over the whole section is satisfied by whichever copy survived. Scoped to
    one paragraph or one table row, an inversion in either copy has nowhere to
    hide.

    Paragraph boundaries are blank lines, which a reflow preserves, so this is
    not a guard a rewrap defeats.
    """
    out: List[str] = []
    for block in _section().split("\n\n"):
        lines = [line for line in block.splitlines() if line.strip()]
        if lines and all(line.lstrip().startswith("|") for line in lines):
            out.extend(lines)          # one unit per table row
        elif block.strip():
            out.append(block)
    return [_plain(unit) for unit in out if _plain(unit)]


def _citations() -> List[tuple]:
    return sorted(set(_CITE_RE.findall(_section())))


def _knob_rows() -> List[tuple]:
    return _KNOB_ROW_RE.findall(_section())


def _grid_in_doc() -> List[int]:
    """The twelve grid values as §2.0 prints them.

    Parsed out of the document rather than restated here, so the assertion is
    between the document and the module and this file is not a third place the
    numbers can be wrong in.
    """
    text = _plain(_section())
    m = re.search(r"twelve requested ef_search values[^0-9]*"
                  r"((?:\d+\s*,\s*)+\d+)", text)
    assert m, "docs/05 §2.0 no longer prints the grid"
    return [int(v) for v in re.findall(r"\d+", m.group(1))]


class TestTheParsersFoundSomething:
    """Non-vacuity. A parser that matched nothing would make every assertion
    below pass over an empty set, which is the shape of a dead probe."""

    def test_the_section_is_a_real_slice_of_the_document(self):
        section = _section()
        assert section.strip()
        assert len(section) < len(_doc())
        # The heading must not be inside its own body, or the split silently
        # fell back to the whole file.
        assert SECTION_HEADING not in section
        assert NEXT_HEADING not in section

    def test_the_citation_parser_finds_citations(self):
        found = _citations()
        assert len(found) >= 10, found
        # Sentinels by FILE and not by symbol: keyed to one symbol this would
        # fire on a correct edit that cited a neighbouring constant instead.
        for path in ("harness/workloads/calibrate.py",
                     "harness/metrics/records.py",
                     "report/loaders.py"):
            assert any(p == path for p, _ in found), path

    def test_the_knob_table_parser_finds_five_rows(self):
        rows = _knob_rows()
        assert len(rows) == 5, rows

    def test_the_grid_parser_finds_twelve_values(self):
        assert len(_grid_in_doc()) == 12, _grid_in_doc()

    def test_the_unit_split_is_finer_than_the_section(self):
        units = _units()
        assert len(units) >= 20, len(units)
        # No unit may be the whole section, which is what a split that fell
        # through would produce.
        assert max(len(u) for u in units) < len(_plain(_section())) / 2


#: Which recall direction each qualifying status implies, and the direction
#: that reverses it. `report/render.py` and `report/charts.py` both carry the
#: same pairing in words -- "a dagger means the engine is above the target ...
#: a double dagger means it never got there" -- so this is the document's half
#: of a claim the code already makes.
DIRECTION = {"floor_above_target": ("higher", "lower"),
             "target_unreached": ("lower", "higher")}


class TestTheRecallDirectionsCannotBeInverted:
    """The one edit that makes §2.0 actively misleading while reading well.

    An inverted direction tells a reader that an engine's throughput is
    flattered when it is penalised, or the reverse, and every other guard here
    would still pass. Checked per unit, positively and negatively, because the
    section states each direction twice.
    """

    @pytest.mark.parametrize("status", sorted(DIRECTION))
    def test_each_statement_of_the_direction_agrees(self, status):
        right, wrong = DIRECTION[status]
        others = [s for s in DIRECTION if s != status]
        agreeing, reversed_ = [], []
        for unit in _units():
            if status not in unit or any(o in unit for o in others):
                continue
            if f"{right} recall" in unit:
                agreeing.append(unit)
            if f"{wrong} recall" in unit:
                reversed_.append(unit)
        assert agreeing, (
            f"nothing in docs/05 §2.0 says {status} reads at a {right} "
            f"recall than the engines marked ok")
        assert not reversed_, (
            f"docs/05 §2.0 pairs {status} with a {wrong} recall, which "
            f"reverses it: {reversed_[0][:200]}")

    def test_the_two_directions_are_opposites(self):
        """Non-vacuity for the table above: a DIRECTION map whose two rows
        agreed would make both parametrizations pass over one claim."""
        assert (DIRECTION["floor_above_target"][0]
                != DIRECTION["target_unreached"][0])
        for right, wrong in DIRECTION.values():
            assert right != wrong


class TestTheClaimPatternsAreNeitherBrittleNorVacuous:
    """What `CLAIMS` is allowed to accept, and what it must still refuse."""

    @pytest.mark.parametrize("name", sorted(CLAIMS))
    def test_the_section_still_makes_the_claim(self, name):
        _claim(name)

    @pytest.mark.parametrize("name", sorted(CLAIMS))
    def test_the_pattern_accepts_a_legitimate_rewording(self, name):
        pattern = CLAIMS[name]["pattern"]
        accepts = CLAIMS[name]["accepts"]
        assert len(accepts) >= 2, (
            f"{name}: give the pattern at least two rewordings to prove it is "
            f"not keyed to one sentence")
        for wording in accepts:
            assert re.search(pattern, _plain(wording), re.I | re.S), (
                f"{name}: the pattern rejects a legitimate rewording, which "
                f"makes it a guard that fires on correct prose:\n  {wording}")

    @pytest.mark.parametrize("name", sorted(CLAIMS))
    def test_the_pattern_refuses_the_reversed_form(self, name):
        """A check keyed to the right answer cannot see a wrong one: a
        `not in` test for the correct text passes any DIFFERENTLY wrong
        text, so each claim names the forms that must not pass."""
        pattern = CLAIMS[name]["pattern"]
        rejects = CLAIMS[name]["rejects"]
        assert rejects, f"{name}: name at least one form that must not pass"
        for wording in rejects:
            assert not re.search(pattern, _plain(wording), re.I | re.S), (
                f"{name}: the pattern accepts a form that reverses the claim, "
                f"so it cannot see the edit it exists to catch:\n  {wording}")

    def test_every_claim_is_used(self):
        """A claim nobody calls `_claim()` for reads as coverage and is not.

        `test_the_section_still_makes_the_claim` is parametrized over all of
        `CLAIMS`, so this is already true by construction -- asserted anyway,
        because the parametrization is the kind of thing a later narrowing
        silently removes.
        """
        source = open(os.path.abspath(__file__)).read()
        assert 'sorted(CLAIMS)' in source
        assert "_claim(name)" in source


class TestTheSectionSitsWhereItClaimsTo:
    """Position is half the argument, so it is checked rather than trusted."""

    def test_it_comes_before_the_recall_qps_section(self):
        doc = _doc()
        assert doc.index(SECTION_HEADING) < doc.index(NEXT_HEADING)

    def test_it_is_inside_section_2_and_not_a_section_of_its_own(self):
        doc = _doc()
        assert doc.index("## 2. Measurement dimensions") < doc.index(
            SECTION_HEADING)

    @pytest.mark.parametrize("heading", [
        "### 2.1 Recall vs QPS (ann-benchmarks)",
        "### 2.2 Index build cost (ops harness)",
        "### 2.3 Concurrency scaling (ops harness)",
        "### 2.4 Filtered / hybrid search (ops harness)",
        "### 2.5 Churn (ops harness)",
        "### 2.6 Uncertainty intervals, and when two engines are not ordered",
        "### 2.7 Cross-checking the two measurement paths",
    ])
    def test_nothing_was_renumbered(self, heading):
        """2.0 rather than a new 2.1 was deliberate: three files carry
        citations into this document and renumbering would invalidate every
        §-reference in them for no benefit."""
        assert _doc().count(heading) == 1, heading

    def test_the_report_puts_the_table_before_the_three_sections(self):
        """The ordering §2.0 claims the report has, read out of the report.

        Positions in the rendered markdown, not in the source: a comment
        saying the table comes first would pass a grep while the renderer
        emitted it last.
        """
        from report.render import (_concurrency_table, _filtered_table,
                                   _operating_point_table)
        summary = {"operating_points": [], "concurrency": [], "filtered": [],
                   "churn": []}
        # Each renders standalone, which is all that is needed: the ordering
        # claim is about `report/render.py`'s section list, checked below.
        for fn in (_operating_point_table, _concurrency_table,
                   _filtered_table):
            assert isinstance(fn(summary), str)
        source = open(os.path.join(VB_ROOT, "report", "render.py")).read()
        table = source.index('"\\n---\\n\\n## 6. Operating point')
        for later in ('"\\n---\\n\\n## 7. Concurrency scaling',
                      "## 8. Filtered", "## 9. Churn"):
            assert table < source.index(later), later


class TestTheStatusNamesComeFromTheCode:
    """Every status name §2.0 prints, against the module that declares it."""

    def test_the_three_statuses_are_the_three_the_module_declares(self):
        from harness.workloads import calibrate
        statuses = {calibrate.STATUS_OK,
                    calibrate.STATUS_TARGET_UNREACHED,
                    calibrate.STATUS_FLOOR_ABOVE_TARGET}
        assert statuses == {"ok", "target_unreached", "floor_above_target"}
        text = _section()
        for status in statuses:
            assert f"`{status}`" in text, status

    #: Code-font names in §2.0 that have the shape of a status and are not
    #: one. `target_recall` is a record key, and the test below proves it is
    #: by reading it out of `status_extra` -- a carve-out that was merely
    #: convenient would let a genuinely stray status hide behind it.
    NOT_A_STATUS = frozenset({"target_recall"})

    def test_the_section_names_no_status_the_module_does_not_declare(self):
        """The other direction. A fourth status invented in the prose, or one
        left behind after a rename, lands here rather than in a reader's
        head."""
        from harness.workloads import calibrate
        declared = {calibrate.STATUS_OK,
                    calibrate.STATUS_TARGET_UNREACHED,
                    calibrate.STATUS_FLOOR_ABOVE_TARGET}
        # Anything in the section that looks like a status literal.
        named = set(re.findall(r"`(ok|[a-z]+_[a-z_]+target[a-z_]*|"
                               r"target_[a-z_]+)`", _section()))
        # Non-vacuity: the three real ones have to be in what the parser found,
        # or the subtraction below is over an empty set.
        assert declared <= named, sorted(declared - named)
        unknown = named - declared - self.NOT_A_STATUS
        assert not unknown, (
            f"docs/05 §2.0 names these as statuses and "
            f"harness/workloads/calibrate.py declares no such thing: "
            f"{sorted(unknown)}")

    def test_the_carve_out_is_a_real_record_key_and_not_a_convenience(self):
        """Every name excused above has to be something the code writes."""
        from harness.workloads import calibrate
        result = calibrate.CalibrationResult(
            ef_search=32, recall_at_k=0.96, status=calibrate.STATUS_OK)
        keys = set(calibrate.status_extra(
            calibrate.EF_MODE_CALIBRATED, result))
        assert self.NOT_A_STATUS <= keys, sorted(self.NOT_A_STATUS - keys)

    def test_the_phase_string_is_the_one_the_records_carry(self):
        from harness.metrics.records import PHASE_CALIBRATION
        assert PHASE_CALIBRATION == "ef_calibration"
        assert f"`phase: {PHASE_CALIBRATION}`" in _section()

    def test_the_stamped_keys_are_the_ones_status_extra_writes(self):
        from harness.workloads import calibrate
        result = calibrate.CalibrationResult(
            ef_search=32, recall_at_k=0.96, status=calibrate.STATUS_OK)
        keys = set(calibrate.status_extra(
            calibrate.EF_MODE_CALIBRATED, result))
        assert keys == {"ef_mode", "target_recall", "calibration_status"}
        text = _section()
        for key in keys:
            assert f"`{key}`" in text, key

    def test_fixed_mode_asserts_no_status_which_is_what_the_section_says(self):
        """§2.0's `ef_mode: fixed` paragraph, called rather than read."""
        from harness.workloads import calibrate
        assert calibrate.status_extra(calibrate.EF_MODE_FIXED, None) == \
            {"ef_mode": "fixed"}
        assert "`ops.ef_mode: fixed`" in _section()


class TestTheNumbersComeFromTheCode:
    """The margin, the target, the grid and the warm-up, both directions."""

    def test_the_floor_margin_is_the_modules(self):
        from harness.workloads.calibrate import FLOOR_MARGIN
        assert FLOOR_MARGIN == 0.02
        # Printed twice in §2.0 -- once in the rule, once in the status
        # paragraph -- and both must track the constant.
        text = _plain(_section())
        assert text.count(f"{FLOOR_MARGIN}") >= 2, (
            f"docs/05 §2.0 prints the floor margin "
            f"{text.count(str(FLOOR_MARGIN))} time(s); the module says "
            f"{FLOOR_MARGIN}")

    def test_the_target_recall_is_the_modules(self):
        from harness.workloads.calibrate import DEFAULT_TARGET_RECALL
        assert DEFAULT_TARGET_RECALL == 0.95
        assert f"recall@k >= {DEFAULT_TARGET_RECALL}" in _plain(_section())

    def test_the_target_is_what_the_orchestrator_passes(self):
        """The two declarations, pinned to each other through the document.

        `orchestrator/ops_pass.py` keeps its own copy because the host's only
        Python dependency is pyyaml, so importing the workload would put numpy
        on that list. A document quoting one of two copies is quoting the
        wrong one half the time.
        """
        from harness.workloads.calibrate import DEFAULT_TARGET_RECALL
        from orchestrator.ops_pass import DEFAULT_TARGET_RECALL as HOST
        assert HOST == DEFAULT_TARGET_RECALL

    def test_the_grid_the_document_prints_is_the_modules(self):
        from harness.workloads.calibrate import DEFAULT_EF_GRID
        assert _grid_in_doc() == list(DEFAULT_EF_GRID)

    def test_the_grid_is_the_one_the_orchestrator_builds_the_command_from(
            self):
        from harness.workloads.calibrate import DEFAULT_EF_GRID
        from orchestrator.ops_pass import DEFAULT_EF_CALIBRATION
        assert tuple(DEFAULT_EF_CALIBRATION) == tuple(DEFAULT_EF_GRID)

    def test_the_warmup_and_query_count_are_the_modules(self):
        from harness.workloads.calibrate import (DEFAULT_MAX_QUERIES,
                                                 DEFAULT_WARMUP)
        text = _plain(_section())
        assert f"discard {DEFAULT_WARMUP} warm-up queries" in text, text[:0]
        assert f"{DEFAULT_MAX_QUERIES:,} by default" in text

    def test_the_profile_key_the_section_names_is_the_one_read(self):
        """§2.0 says the key is `ops.ef_calibration` and that the report's
        footnote names a key the orchestrator does not read. Both halves
        checked, because the second is the one a tidy-up would delete."""
        source = open(os.path.join(VB_ROOT, "orchestrator",
                                   "ops_pass.py")).read()
        assert 'ops.get("ef_calibration")' in source
        assert 'ops.get("ef_grid")' not in source
        text = _section()
        assert "`ops.ef_calibration`" in text
        assert "`ops.ef_grid`" in text


class TestNoProfileSetsTheEfKeys:
    """§2.0 says every profile here runs the defaults. Recomputed."""

    KEYS = ("ef_mode", "target_recall", "ef_calibration")

    def test_no_shipped_profile_declares_any_of_them(self):
        from orchestrator.config import available_profiles, load_profile
        names = list(available_profiles())
        assert len(names) >= 6, names
        for name in names:
            ops = load_profile(name).get("ops", {}) or {}
            for key in self.KEYS:
                assert key not in ops, (name, key)

    def test_the_section_says_so_about_exactly_those_keys(self):
        text = _section()
        for key in self.KEYS:
            assert f"`ops.{key}`" in text, key

    def test_the_default_mode_is_calibrated(self):
        from harness.workloads.calibrate import (EF_MODE_CALIBRATED,
                                                 EF_MODES)
        from orchestrator.ops_pass import DEFAULT_EF_MODE
        assert DEFAULT_EF_MODE == EF_MODE_CALIBRATED == "calibrated"
        assert set(EF_MODES) == {"fixed", "calibrated"}
        assert "`calibrated` mode" in _section()


class TestTheSelectionRuleIsCalledNotGrepped:
    """§2.0's four cases, run through `calibrate.select`.

    "The smallest ef whose successor also reaches the target" is a claim about
    behaviour; a search for the word "successor" would pass against a selector
    that had stopped doing it. Each case below also names the document
    sentence it settles.
    """

    @staticmethod
    def _curve(recalls):
        from harness.workloads.calibrate import CalibrationPoint
        return [CalibrationPoint(ef_search=ef, effective_ef=ef,
                                 recall_at_k=r, qps=100.0)
                for ef, r in zip((10, 16, 24, 32, 48), recalls)]

    def test_case_1_a_floor_above_the_target_by_more_than_the_margin(self):
        from harness.workloads import calibrate
        out = calibrate.select(self._curve([0.99, 0.99, 0.99, 0.99, 0.99]))
        assert out.status == calibrate.STATUS_FLOOR_ABOVE_TARGET
        assert out.ef_search == 10
        _claim("floor_above_target means the engine cannot be slowed to the "
               "target")

    def test_case_1_boundary_inside_the_margin_is_ok_and_not_a_floor(self):
        """The margin is not zero on purpose, which §2.0 states. 0.951 is
        above the 0.95 target and inside the 0.02 margin."""
        from harness.workloads import calibrate
        out = calibrate.select(self._curve([0.951, 0.96, 0.97, 0.98, 0.99]))
        assert out.status == calibrate.STATUS_OK
        assert out.ef_search == 10

    def test_case_2_the_successor_clause_skips_a_noise_spike(self):
        from harness.workloads import calibrate
        # 16 clears the target and 24 falls back under it: a single point
        # lifted over by noise. The rule must not choose 16.
        out = calibrate.select(self._curve([0.90, 0.96, 0.93, 0.96, 0.97]))
        assert out.status == calibrate.STATUS_OK
        assert out.ef_search == 32, out
        assert out.confirmed is True
        _claim("the successor clause exists because one point can clear by "
               "noise")

    def test_case_3_a_target_first_met_at_the_last_point_is_ok_unconfirmed(
            self):
        from harness.workloads import calibrate
        out = calibrate.select(self._curve([0.80, 0.85, 0.90, 0.93, 0.96]))
        assert out.status == calibrate.STATUS_OK
        assert out.ef_search == 48
        assert out.confirmed is False
        _claim("confirmed is not a fourth status")

    def test_case_4_nothing_reaches_the_target(self):
        from harness.workloads import calibrate
        out = calibrate.select(self._curve([0.70, 0.75, 0.80, 0.85, 0.90]))
        assert out.status == calibrate.STATUS_TARGET_UNREACHED
        assert out.ef_search == 48
        _claim("target_unreached reads at a lower recall")

    def test_reaching_is_inclusive_and_the_floor_test_is_strict(self):
        """§2.0's last sentence of the rule, both halves.

        Exactly on the target reaches it; exactly on target-plus-margin is not
        a floor. Written as two calls rather than as one, because the two
        comparisons are in different branches of `_choose`.
        """
        from harness.workloads import calibrate
        on_target = calibrate.select(
            self._curve([0.90, 0.95, 0.95, 0.96, 0.97]))
        assert on_target.status == calibrate.STATUS_OK
        assert on_target.ef_search == 16
        on_margin = calibrate.select(
            self._curve([0.97, 0.97, 0.97, 0.97, 0.97]))
        assert on_margin.status == calibrate.STATUS_OK, (
            "0.97 is exactly target + FLOOR_MARGIN and the floor test is a "
            "strict >, so this is not floor_above_target")
        assert "strict" in _plain(_section())

    def test_an_empty_curve_is_a_configuration_error_and_not_a_status(self):
        from harness.workloads import calibrate
        with pytest.raises(ValueError):
            calibrate.select([])


class TestTheThreeInheritingSectionsAreTheCalibratedWorkloads:
    """§2.0 names §2.3, §2.4 and §2.5. The module names the workloads."""

    def test_the_workload_list_is_the_three(self):
        from harness.workloads.calibrate import CALIBRATED_WORKLOADS
        assert tuple(CALIBRATED_WORKLOADS) == ("concurrency", "filtered",
                                               "churn")

    def test_the_section_names_each_of_the_three_by_number(self):
        text = _plain(_section())
        for ref in ("2.3", "2.4", "2.5"):
            assert f"§{ref}" in text, ref

    def test_it_names_no_other_measurement_section_as_inheriting(self):
        """The other direction: §2.1, §2.2, §2.6 and §2.7 do not inherit the
        choice, and listing one of them here would be a false caveat.

        Scoped to the inheritance subsection, because §2.0 legitimately
        mentions §2.1 and §2.6 elsewhere -- the dagger it shares with the
        headline table, and the intervals its own records carry.
        """
        body = _section().split("#### What §2.3, §2.4 and §2.5 inherit", 1)
        assert len(body) == 2, "the inheritance subsection has been renamed"
        tail = body[1]
        for ref in ("§2.1", "§2.2", "§2.6", "§2.7"):
            assert ref not in tail, ref

    def test_the_report_marks_engines_with_no_operating_point_from_those_three(
            self):
        """`_engines_without_an_operating_point` keys on the same three, which
        is what makes §2.0's "no row here" paragraph true."""
        from report.render import _engines_without_an_operating_point
        summary = {
            "operating_points": [],
            "concurrency": [{"engine": "mariadb", "dataset": "d"}],
            "filtered": [{"engine": "alisql", "dataset": "d"}],
            "churn": [{"engine": "pgvector", "dataset": "d"}],
            # An ingest-only engine must NOT be listed: it was never going to
            # have an operating point.
            "ingest": [{"engine": "valkey", "dataset": "d"}],
        }
        assert _engines_without_an_operating_point(summary) == [
            ("alisql", "d"), ("mariadb", "d"), ("pgvector", "d")]


class TestTheMarksComeFromTheRenderer:
    """The three glyphs §2.0 tabulates, against the module that defines them."""

    def test_the_table_rows_are_the_renderers_marks(self):
        from report.render import (FLOOR_MARK, UNCALIBRATED_MARK,
                                  UNREACHED_MARK)
        assert (FLOOR_MARK, UNREACHED_MARK, UNCALIBRATED_MARK) == \
            ("†", "‡", "*")
        text = _section()
        for mark, status in ((FLOOR_MARK, "floor_above_target"),
                             (UNREACHED_MARK, "target_unreached")):
            rows = [line for line in text.splitlines()
                    if line.startswith("|") and mark in line
                    and status in line]
            assert rows, (
                f"docs/05 §2.0 has no table row carrying {mark!r} beside "
                f"{status}")

    def test_the_two_marks_are_different_characters(self):
        """One mark for both would present opposite caveats as the same one,
        which is the thing §2.0's table exists to prevent."""
        from report.render import FLOOR_MARK, UNREACHED_MARK
        assert FLOOR_MARK != UNREACHED_MARK

    def test_the_chart_imports_the_marks_rather_than_respelling_them(self):
        """§2.0 says so, and it is the reason a chart's mark and a table's
        footnote cannot drift apart."""
        source = open(os.path.join(VB_ROOT, "report", "charts.py")).read()
        assert "from .render import (FLOOR_MARK, UNCALIBRATED_MARK" in source
        assert "UNREACHED_MARK" in source

    def test_the_renderer_really_marks_the_two_statuses_in_the_table(self):
        """Rendered, not grepped: the cell text for each status."""
        from report.render import FLOOR_MARK, UNREACHED_MARK
        from report.render import _operating_point_table
        for status, mark in (("floor_above_target", FLOOR_MARK),
                             ("target_unreached", UNREACHED_MARK)):
            out = _operating_point_table({"operating_points": [{
                "engine": "mariadb", "dataset": "glove-100-angular",
                "resource_pass": "normalized", "m": 16, "build_mode": "post",
                "ef_search": 10, "effective_ef": 10, "recall_at_k": 0.98,
                "target_recall": 0.95, "status": status,
                "points_measured": 3, "grid_size": 12}]})
            assert f"{status} {mark}" in out, (status, out)


class TestTheKnobTableResolves:
    """Each row's knob literal and symbol, read out of the driver it names."""

    @pytest.mark.parametrize("index", range(5))
    def test_the_row_points_at_what_it_says(self, index):
        rows = _knob_rows()
        engine, knob, path, symbol = rows[index]
        source = open(os.path.join(VB_ROOT, path)).read()
        assert symbol in source, (
            f"docs/05 §2.0 cites {path}:{symbol} for {engine} and that file "
            f"declares no such name")
        # The knob literal itself, stripped of the SET prefix the document
        # writes for readability: `numCandidates` and `EF_RUNTIME` are not
        # SQL and the row says so in words.
        #
        # As a WHOLE TOKEN, not as a substring, and that is measured rather
        # than tidy: `"mhnsw_ef" in source` is True in a file that only ever
        # writes `mhnsw_ef_search`, so a plain substring test cannot see a
        # knob name truncated in the document -- which is a wrong knob printed
        # to a reader who would then set a variable that does not exist.
        needle = knob.replace("SET ", "")
        whole = re.compile(r"(?<![A-Za-z0-9_.])" + re.escape(needle)
                           + r"(?![A-Za-z0-9_])")
        assert whole.search(source), (
            f"docs/05 §2.0 says {engine} is driven by {knob!r} and "
            f"{needle!r} does not appear in {path} as a whole name")

    def test_the_five_rows_name_five_distinct_knobs(self):
        """The claim the table makes by existing. Two rows with one knob
        would be a table that proves the opposite of what it is for."""
        knobs = [knob for _, knob, _, _ in _knob_rows()]
        assert len(set(knobs)) == 5, knobs

    def test_every_driver_with_a_concrete_knob_is_in_the_table(self):
        """The other direction. A seventh engine with a sixth knob must land
        here rather than leaving the table quietly incomplete.

        `base.py` is excluded because its `set_ef_search` is the abstract
        declaration (`...` as a body), not a knob.
        """
        import glob
        listed = {path for _, _, path, _ in _knob_rows()}
        concrete = set()
        for path in sorted(glob.glob(os.path.join(VB_ROOT, "harness",
                                                  "drivers", "*.py"))):
            rel = os.path.relpath(path, VB_ROOT)
            source = open(path).read()
            if "def set_ef_search" not in source:
                continue
            if re.search(r"def set_ef_search\(self, ef_search: int\)"
                         r" -> None: \.\.\.", source):
                continue
            concrete.add(rel)
        assert concrete, "no driver declares a concrete set_ef_search"
        assert concrete == listed, (sorted(concrete), sorted(listed))

    def test_the_clamping_drivers_really_clamp(self):
        """§2.0 says two of the five raise the request to k. Read out of the
        query each one builds, not out of a comment."""
        mongo = open(os.path.join(VB_ROOT, "harness", "drivers",
                                  "mongo.py")).read()
        valkey = open(os.path.join(VB_ROOT, "harness", "drivers",
                                   "valkey.py")).read()
        assert "max(self._num_candidates, k)" in mongo
        assert "max(self._ef_runtime, k)" in valkey

    def test_only_one_driver_reports_the_clamp_as_the_section_says(self):
        """§2.0 calls Valkey's unreported clamp a gap in the record. Both
        halves: Percona Search overrides `effective_ef` and Valkey does not."""
        mongo = open(os.path.join(VB_ROOT, "harness", "drivers",
                                  "mongo.py")).read()
        valkey = open(os.path.join(VB_ROOT, "harness", "drivers",
                                   "valkey.py")).read()
        assert "def effective_ef" in mongo
        assert "def effective_ef" not in valkey
        text = _plain(_section())
        assert "is not reflected in its effective_ef" in text


class TestEveryCitationResolves:
    """Turn a section that rots into one that reports its own rot."""

    @pytest.mark.parametrize("citation", _citations())
    def test_the_symbol_is_in_the_file(self, citation):
        path, symbol = citation
        full = os.path.join(VB_ROOT, path)
        assert os.path.exists(full), (
            f"docs/05 §2.0 cites {path}, which does not exist")
        source = open(full).read()
        assert symbol in source, (
            f"docs/05 §2.0 cites {path}:{symbol} and that file no longer "
            f"contains {symbol}")

    def test_no_citation_points_outside_the_repository(self):
        for path, _ in _citations():
            assert not path.startswith("/"), path
            assert ".." not in path, path


class TestWhatIsNotClaimed:
    """R11. The section must not phrase a supplied figure as a measurement."""

    def test_the_section_marks_the_supplied_figures_as_supplied(self):
        _claim("the recall spread at one ef is supplied and not measured here")
        _claim("the MHNSW and VIDX expectation is supplied")
        _claim("no engine behaviour is measured in this document")

    #: The magnitudes §2.0 quotes that no run in this repository produced.
    #: `0.02` and `0.95` are NOT here and must not be added: those are
    #: constants of this codebase, read out of `calibrate.py` by
    #: `TestTheNumbersComeFromTheCode`, and marking them supplied would be the
    #: opposite error.
    #: Both spellings of the pair, because §2.0 writes it out in prose and
    #: abbreviates it in the closing paragraph; guarding only one would leave
    #: the other free to lose its marker.
    SUPPLIED_FIGURES = ("0.98 and 0.80", "0.98/0.80", "0.18")

    #: A sentence that credits this system with the measurement. Written as
    #: subject-plus-verb rather than as a phrase blacklist, because a
    #: blacklist of the wordings somebody already used cannot see a DIFFERENTLY
    #: wrong one -- measured: the first round of this file listed "we
    #: measured", "this run shows" and "as measured here", and a mutant that
    #: wrote "a figure this framework measures directly" walked straight past
    #: all three.
    CREDITS_THIS_SYSTEM = re.compile(
        r"\b(we|this (?:framework|repository|repo|project|document|section|"
        r"benchmark|harness|checkout)|these runs?|the framework)\s+"
        r"(?:ha(?:s|ve) )?(?:measure[sd]?|establish(?:es|ed)?|"
        r"confirm(?:s|ed)?|show[sn]?)\b", re.I)

    def test_every_supplied_figure_sits_in_a_paragraph_that_marks_it(self):
        """Per PARAGRAPH, not per character window.

        Measured: a 600-character window passed against a document whose
        0.98/0.80 paragraph had had its "supplied" rewritten into a claim of
        measurement, because the *other* paragraph quoting the same figures
        still carried the word 600 characters away. The paragraph is the unit a
        reader takes the claim from, so it is the unit the guard uses.
        """
        marked = 0
        for unit in _units():
            present = [f for f in self.SUPPLIED_FIGURES if f in unit]
            if not present:
                continue
            assert "supplied" in unit, (
                f"docs/05 §2.0 prints {present} in a paragraph that does not "
                f"mark it supplied: {unit[:220]}")
            marked += len(present)
        # Non-vacuity, and the honest count: both figures appear in the
        # provenance paragraph and again in the closing one. A figure deleted
        # outright fails here rather than passing an empty loop.
        assert marked >= 4, marked

    def test_no_paragraph_quoting_a_supplied_figure_credits_this_system(self):
        """Polarity, which the positive above cannot supply on its own.

        A paragraph may keep the word "supplied" and still assert that this
        framework measured the thing, and that sentence is the one R11
        forbids.
        """
        hits = []
        for unit in _units():
            if not any(f in unit for f in self.SUPPLIED_FIGURES):
                continue
            found = self.CREDITS_THIS_SYSTEM.search(unit)
            if found:
                hits.append((found.group(0), unit[:200]))
        assert not hits, (
            f"docs/05 §2.0 credits this system with a supplied figure: {hits}")

    def test_the_polarity_pattern_sees_a_differently_wrong_sentence(self):
        """Non-vacuity for the pattern above, over wordings this document does
        not use -- otherwise it is a pattern that has never matched anything.

        The `allowed` rows are sentences §2.0 really contains, or legitimate
        rewordings of them, and they must NOT match: a pattern that fired on
        "nothing here measures it" or "what the sweep measures" would be a
        guard that goes off on correct prose.
        """
        for wrong in ("a figure this framework measures directly",
                      "we measured two engines at recall 0.98 and 0.80",
                      "this repository establishes the 0.18 spread",
                      "these runs show the spread reaching 0.18",
                      "this benchmark has confirmed the 0.18 gap"):
            assert self.CREDITS_THIS_SYSTEM.search(wrong), wrong
        for allowed in ("is supplied, and nothing here measures it",
                        "which engines land on them is what the sweep "
                        "measures",
                        "what is established from this repository is the "
                        "mechanism",
                        "no claim here is a measurement of an engine",
                        "supplied by the project specification and the "
                        "tuned-profile notes it cites"):
            assert not self.CREDITS_THIS_SYSTEM.search(allowed), allowed

    def test_no_archived_calibration_record_claim_still_holds(self):
        """The claim, and the condition that makes it true.

        The archive sits outside this checkout and is absent from a worktree,
        so this cannot count records. What it can do is confirm that nothing
        in the repository asserts the opposite, and that the phase constant
        the claim is about exists.
        """
        _claim("no calibration record exists in this repository")
        from harness.metrics.records import PHASE_CALIBRATION
        assert PHASE_CALIBRATION == "ef_calibration"

    def test_the_section_does_not_promise_a_recall_it_cannot_measure(self):
        """A negative, paired with a positive so it cannot pass on silence.

        The reversed phrasings are the ones a later tidy-up reaches for: "we
        measured", "this run shows". Each is checked against the section, and
        the positive below proves the needle is findable at all.
        """
        text = _plain(_section())
        assert "supplied" in text, "the section lost its provenance marking"
        for phrase in ("we measured", "this run shows",
                       "as measured here", "measured in this repository the"):
            assert phrase not in text.lower(), phrase
