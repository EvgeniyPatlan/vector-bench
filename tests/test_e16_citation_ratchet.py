"""A ratchet on line citations written into Python comments and docstrings.

THE DEFECT THIS IS FOR
----------------------
A comment that points at another file by line number is wrong the moment
anything is inserted above that line, and nothing notices. Six such citations
were found wrong in two days, independently, by two different readers. Four of
them had been wrong before any of the changes that found them. **Two were
created by changes whose own subject was citation rot** -- #41 and #43 repaired
line citations in `docs/09` and `docs/10` and moved others in the same commit,
and #47, which re-derived a list of citations and recorded three more it had
checked and found unmoved, moved the one in
`tests/test_t3c_filtered.py::TestTheReportKeepsTheTwoColumnsApart` and listed it
in neither.

`docs/09-adding-percona-server.md` and `docs/10-rollout.md` USED TO cite by
line, and were each guarded: `tests/test_e1_percona_design.py` and
`tests/test_e3_rollout_doc.py` carried a `CITED_IN_REPO` table that said, for
every citation in the document, what the cited line was expected to contain,
and re-opened the file to check it. That mechanism worked. **Both documents
have since been converted to symbol citations and both tables are gone**, so
the documents no longer make the citations this file counts and
`CITED_IN_REPO` no longer exists in either guard.

The reason the mechanism could not be extended here is unchanged and is why
this file still exists: it needed a table. A comment has no table. There is
nowhere to write down what a comment's citation claims is at that line except
beside the citation itself, in prose, where nothing can read it. What HAS
changed is the standing: a line citation in a Python comment is now the last
place in this repository where the form survives at all, so the census below
is no longer a floor under a practice the documents share -- it is the whole
of it.

WHY RANGE-CHECKING IS NOT THE GUARD, MEASURED BEFORE THIS FILE WAS WRITTEN
--------------------------------------------------------------------------
Every line citation in this tree's Python comments resolves to a line that
exists. Not one is past the end of its file. The rotten ones are wrong about
the *content* of a line that is perfectly in range -- the citation repaired in
`tests/test_t3c_filtered.py` named a line holding a `print` of the engine list
and claimed it held a `for` loop over resource passes, and the one repaired in
`tests/test_t5b_stats.py` offered two lines as the *evidence* for the sentence
around it and pointed at a memory-ceiling lookup. An out-of-range check would
have passed every one of them.

The census on `c8de6a5`, the commit this file was written against: **173 live
line citations across 125 Python files**, 136 naming a path and 37 continuing
one with a bare range. Zero out of range.

So this file does not check citations. **It counts them, and refuses an
increase.** Existing rot is grandfathered -- repairing 100-odd citations is a
separate change, and a guard that demanded it on day one would have been
deleted on day one -- while a new one fails here with the alternative named.
That is all a ratchet can do and it is the part that compounds: the census can
only come down.

WHAT COUNTS, AND WHAT DOES NOT
------------------------------
**Comments and docstrings only.** A line citation inside an ordinary string
literal is not counted, and the reason is not convenience:

* `CITED_IN_REPO` in the two document guards WAS a dict of citations, each
  with the anchor it was expected to match and a test that re-opened the line.
  Counting those would have made this file fail whenever `docs/09` or
  `docs/10` legitimately gained a guarded citation -- punishing the one
  citation form in this repository that was checked. Both tables are gone now,
  replaced by `CITED_SYMBOLS` keyed on `(path, symbol)`, so there are no
  guarded line citations left to punish; the rule is kept because the
  reasoning still applies to any table of the same shape, and because the
  third bullet below needs it.
* `tests/test_e15_methodology_roadmap_doc.py` builds a deliberately bad
  document containing the banned form, to prove its detector can see one.
  Planting a failure is how guards here are shown to work; a census that
  counted plants would tax exactly that practice.
* **And this file could not hold its own test data.** The sample table below
  spells the forms the matcher must catch. If literals counted, this file would
  have to be excluded from its own census -- and a guard that must exempt
  itself is the shape `docs/11` §0 and
  `tests/test_e15_methodology_roadmap_doc.py` both record as worse than no
  guard at all. Under the rule as written no exclusion is needed, and
  `test_this_file_makes_none_of_the_citations_it_counts` asserts there is none.

WHAT THE CENSUS CANNOT SEE, IN FULL
-----------------------------------
The first version of this file called the string-literal boundary "a leak",
singular. A review planted five forms on the live tree and all five passed,
**two of them written by this file's own commit** -- which is the defect this
series keeps finding, a guard understating its own reach. The list, so that a
green census is read for what it is:

* **A bare number continuing a path named earlier** -- a path citation, then a
  second range written as a colon and digits alone, a few words or a wrapped
  line later. `MATCHER_SAMPLES` spells the shape. Correct the day it is
  written, dead to any edit above it, and invisible to a matcher that wants a
  path in front. **This one is now counted** -- `_CONTINUATION_RE`, with the
  continuation inheriting the live or frozen status of the path it hangs off.
  It is listed here because it was a hole, and because closing it is why the
  comment blocks below are joined rather than taken a line at a time.
* **A line number written as words.** "…until #47 moved the loop down to line
  508." Not counted, and not closable by a pattern that would not also fire on
  every sentence containing the word *line* and a number. The commit that
  added this file wrote two of these and re-spelled them as revision-scoped
  citations once the review found them.
* **An extension outside the set.** `run-manifest.json:12`, `Dockerfile:40`.
  Not counted, deliberately: the set is the one the two document guards use,
  and widening it here alone would leave three parsers disagreeing about what
  a line citation is. Widening all three is a change of its own.
* **A citation inside an assertion message.** A string literal, so outside the
  boundary argued above, and it rots exactly as a comment does.
* **A citation in a file that is not Python.** `tests/make-tiny-dataset.sh`,
  `scripts/*.sh`, `webui/static/*.js`. The census is `.py` only; its name says
  so.

A ratchet is a floor on discipline, not a proof. These are the places it does
not reach, and a citation written in one of these forms is uncounted rather
than permitted -- `POLICY` says so in every failure.

HISTORICAL QUOTATIONS, AND WHY THE EXEMPTION IS A REVISION
----------------------------------------------------------
Several comments here quote dead line numbers **on purpose**, as the record of
where something used to be and what used to be there -- it is how the rot gets
diagnosed at all. `tests/test_g3_process_clients.py` does it for three numbers
measured at `ffcb3a7`; `tests/test_t5b_stats.py`,
`tests/test_estimate.py` and `tests/test_e1_percona_design.py` each do it
somewhere too. Counting those as live citations is a false positive, and it is
not hypothetical: an audit run before this file existed hit it.

**The exemption is the `rev:path:line` form** -- the spelling `git show` takes,
with a commit or tag in front of the path. `tests/test_t5b_stats.py` was
already writing citations that way and `tests/test_e7_percona_fortify_level.py`
and `tests/test_e8_package_flags_overwrite.py` were already doing it with
vendor tags. Naming that form is recognising a convention this repository had,
not inventing one.

Why it is the least fragile of the three options:

* **It is meaning-bearing, not a dodge.** A revision freezes the tree the
  number is true in, so the citation is permanently correct AND verifiable:
  `git show ffcb3a7:tests/test_config.py | sed -n 6988p` prints what the
  comment says was there. A bare line number has no tree attached and is true
  only until the next edit. The exemption therefore tracks the real property --
  "can this rot?" -- rather than a promise that it will not.
* **It is per-citation and stateless.** Nothing has to be kept in step with
  the tree. An allowlist keyed to (file, citation text) needs an entry per
  quotation and fails on a reword; keyed to (file, line) it rots on any edit
  above it, which is the precise failure this whole file exists to stop. A
  marker scoped to a comment *block* is worse again: it silently exempts
  citations added into that block later.
* **It degrades safely.** A nonsense revision prefix exempts a citation it
  should not -- but it is conspicuous, it reads as a lie to any human, and
  `git show` refutes it in one command. An allowlist entry is invisible.

NON-VACUITY
-----------
Every assertion here is over a set a parser produced, and one guard in this
series shipped green because it scanned zero files. So: the scan asserts how
many files it read and how many citations it found; the matcher is run against
a table of forms it must catch and forms it must not; a citation planted into a
synthetic tree is asserted to be counted, so the file walk and the comment
extraction are exercised end to end and not only the regex; and both ways of
breaking the scan -- a matcher that finds nothing, a walk that reads nothing --
are asserted to be reported rather than to pass quietly.
"""

from __future__ import annotations

import ast
import io
import os
import re
import tokenize
from typing import Dict, List, Tuple

import pytest

VB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SELF_REL = "tests/test_e16_citation_ratchet.py"

#: Directories never walked, with the reason. `.git` and `__pycache__` hold no
#: source; the rest are generated and gitignored per CLAUDE.md, so a census
#: that read them would differ between a fresh checkout and a machine that had
#: run a benchmark. `.claude` is gitignored too and holds other agents' git
#: worktrees -- walking it would count whole second copies of this repository.
SKIP_DIRS = ("__pycache__", ".git", ".pytest_cache", ".claude", ".cache",
             ".venv", "venv", "sources", "work", "results", "datasets",
             "state")

#: A line citation, as these comments actually spell one. Deliberately the same
#: extension set as `tests/test_e3_rollout_doc.py` and
#: `tests/test_e15_methodology_roadmap_doc.py` use to find citations in the two
#: documents, so that "a line citation" is one pattern in this repository
#: rather than three that might disagree.
#:
#: The path is NOT required to resolve. The repaired citation in
#: `tests/test_t3c_filtered.py` was spelled as a bare basename, which resolves
#: to nothing from the repository root, and a census that only counted
#: resolvable paths would have missed the very citation that prompted it.
#: Citations into the vendor trees are counted for the same reason: a number
#: into Percona's `CMakeLists.txt` rots when the source pin moves, which is a
#: slower clock than an edit above it but not a stopped one.
_CITATION_RE = re.compile(
    r"([A-Za-z0-9_][A-Za-z0-9_./-]*"
    r"\.(?:py|sh|js|yml|md|cc|h|txt)):(\d+)(?:-(\d+))?\b")

#: The token immediately before a cited path, when there is one: the `rev`
#: part of the `rev:path` form `git show` takes. Matched against the text to
#: the LEFT of a citation and anchored at its end, so only a prefix that is
#: actually adjacent can exempt anything.
_ADJACENT_TOKEN_RE = re.compile(r"([A-Za-z0-9][A-Za-z0-9_.-]*):$")

#: An abbreviated or full commit sha.
_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")


def _looks_like_revision(token: str) -> bool:
    """Whether `token` could be a revision this repository would name.

    **NARROWED AFTER REVIEW, and the hole it closes was an accident rather
    than a dodge.** The first version accepted any word abutting a colon, so a
    comment marker written tight against a path -- a to-do or a note label
    with no space after its colon; `MATCHER_SAMPLES` spells both -- read as
    revision-scoped and vanished from the census. The file's defence of the
    exemption, that a nonsense revision is conspicuous and `git show` refutes
    it, answers a deliberate evasion. It does not answer a prefix that looks
    like ordinary punctuation and that nobody would read as a claim about a
    frozen tree at all.

    A revision here is a sha, or a tag: every tag this repository pins carries
    a digit and a separator (`mariadb-11.8.8`, `AliSQL-8.0.44-2`, `v0.8.6`,
    `valkey-91`), and no comment marker does.
    """
    if _SHA_RE.match(token):
        return True
    return bool(re.search(r"\d", token) and re.search(r"[.-]", token))


#: A bare colon-and-number continuation, which is how a comment writes the
#: second and third citation of a sentence once the path has been named --
#: `MATCHER_SAMPLES` spells the shape. The lookbehind keeps it off the colon
#: INSIDE a path citation (always preceded by a word character), off a comment
#: marker, and off a clock time or a ratio, all of which put a digit in front
#: of the colon.
#:
#: **ADDED AFTER REVIEW, because the first version of this file both missed
#: this form and wrote one.** The census's own commit cited a line of
#: `tests/test_t3c_filtered.py` and then a second range as a continuation of
#: it -- correct that day, invisible to the count, and dead to any edit above
#: it. The sample table's reason for not counting a bare number was that it
#: "is already unreadable as one", which is true standing alone and plainly
#: false a few words after its path.
#: `tests/test_e1_percona_design.py` had already found and closed this exact
#: hole for the DOCUMENT parser; the Python census had no equivalent until
#: now.
_CONTINUATION_RE = re.compile(r"(?<![A-Za-z0-9_#]):(\d+)(?:-(\d+))?\b")

#: A matcher that finds nothing, for the arm that proves a dud one is reported.
_MATCHES_NOTHING = re.compile(r"(?!)")


def _is_revision_scoped(text: str, start: int) -> bool:
    m = _ADJACENT_TOKEN_RE.search(text[:start])
    return bool(m) and _looks_like_revision(m.group(1))


def live_citations(text: str,
                   cite_re: re.Pattern = _CITATION_RE) -> List[str]:
    """The citations in `text` that are not pinned to a revision.

    Takes text rather than a file, so the sample table and the planted arms
    run the same function the census runs.

    A continuation inherits the status of the nearest path citation before it,
    which is the only reading that matches what the comments mean. The dead
    numbers in `tests/test_g3_process_clients.py` are written
    `ffcb3a7:tests/test_config.py:6988`/`:7003`, and `:7003` is as frozen as
    the citation it hangs off; `:2053-2055` after a live path citation is as
    live as that one. A continuation with no path before it is not counted at
    all, which is the case the original rationale was about.
    """
    out: List[str] = []
    anchors: List[Tuple[int, int, bool]] = []
    for m in cite_re.finditer(text):
        live = not _is_revision_scoped(text, m.start())
        anchors.append((m.start(), m.end(), live))
        if live:
            out.append(m.group(0))
    for m in _CONTINUATION_RE.finditer(text):
        before = [a for a in anchors if a[1] <= m.start()]
        if before and before[-1][2]:
            out.append(m.group(0))
    return out


def prose_spans(source: str) -> List[str]:
    """Every comment BLOCK and every docstring in `source`.

    Comments come from `tokenize` and docstrings from the AST, because the two
    are different objects and neither tool sees both. A file this cannot parse
    contributes its comments and no docstrings rather than nothing, so a syntax
    error somewhere does not quietly shrink the census.

    **Consecutive comment lines are joined into one span**, and that is not
    cosmetic: `tokenize` hands back one token per line, and a continuation is
    routinely on a different line from the path it continues -- the one this
    file's own commit wrote was two lines below its path. Per-line spans would
    have lost every anchor that wraps. Joining cannot create or destroy a path
    citation, because the path character class spans neither a newline nor the
    `#` that starts the next line.
    """
    spans: List[str] = []
    block: List[str] = []
    previous = None
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type != tokenize.COMMENT:
                continue
            if previous is not None and tok.start[0] != previous + 1:
                spans.append("\n".join(block))
                block = []
            block.append(tok.string)
            previous = tok.start[0]
    except (tokenize.TokenError, IndentationError, SyntaxError, ValueError):
        pass
    if block:
        spans.append("\n".join(block))
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return spans
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef,
                             ast.AsyncFunctionDef, ast.ClassDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                spans.append(doc)
    return spans


def python_files(root: str) -> List[str]:
    out: List[str] = []
    for where, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            if name.endswith(".py"):
                out.append(os.path.join(where, name))
    return sorted(out)


def scan(root: str, cite_re: re.Pattern = _CITATION_RE
         ) -> Tuple[List[str], Dict[str, int]]:
    """`(files scanned, {relative path: live citations in its prose})`.

    The whole pipeline in one function so the planted arms exercise exactly
    what the census exercises: a different root plants a citation, a different
    matcher breaks the detector, and neither needs a copy of the walk.
    """
    files = python_files(root)
    counts: Dict[str, int] = {}
    for path in files:
        try:
            source = open(path, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        found = 0
        for span in prose_spans(source):
            found += len(live_citations(span, cite_re))
        if found:
            counts[os.path.relpath(path, root).replace(os.sep, "/")] = found
    return files, counts


_CACHE: Dict[str, Tuple[List[str], Dict[str, int]]] = {}


def census() -> Tuple[List[str], Dict[str, int]]:
    if "live" not in _CACHE:
        _CACHE["live"] = scan(VB_ROOT)
    return _CACHE["live"]


# ---------------------------------------------------------------------------
# Non-vacuity, as a function so the broken-scan arms can call it
# ---------------------------------------------------------------------------

#: Floors on what the scan must have seen. The file floor is the load-bearing
#: one and cannot be cleaned away -- this tree has well over a hundred Python
#: files and will not lose them -- so it is what catches the walk that reads
#: nothing, which is how one guard in this series shipped green.
#:
#: The citation floor is a floor on NON-VACUITY and not a target. It sits far
#: below the current census on purpose: the census is supposed to fall, and a
#: floor set just under today's number would turn the first real cleanup into a
#: failure. If a cleanup ever brings the census near it, lower it in the same
#: change. What actually proves the matcher is not this number but
#: `TestTheMatcherRecognisesEachForm` and the planted tree.
MIN_FILES = 100
MIN_CITATIONS = 50
MIN_FILES_WITH_CITATIONS = 15


def vacuity_problems(files: List[str], counts: Dict[str, int]) -> List[str]:
    problems: List[str] = []
    if len(files) < MIN_FILES:
        problems.append(
            f"the walk read {len(files)} Python files, fewer than the "
            f"{MIN_FILES} this tree holds -- the census is over an empty or "
            f"truncated set and every ceiling below it is vacuous")
    total = sum(counts.values())
    if total < MIN_CITATIONS:
        problems.append(
            f"the matcher found {total} line citations, fewer than the "
            f"{MIN_CITATIONS} known to be in this tree's comments -- it is "
            f"matching less than it should and a new citation would slip "
            f"under every ceiling")
    if len(counts) < MIN_FILES_WITH_CITATIONS:
        problems.append(
            f"only {len(counts)} files carry a citation, fewer than the "
            f"{MIN_FILES_WITH_CITATIONS} known to -- the census is "
            f"concentrated in a way that suggests the walk is partial")
    return problems


# ---------------------------------------------------------------------------
# The ratchet itself
# ---------------------------------------------------------------------------

#: THE CENSUS, PER FILE, AS MEASURED ON THIS COMMIT. Derived by running the
#: scan above, not written from memory, and every entry is a CEILING: a file
#: may hold fewer and the test passes, which is what makes repairing a citation
#: free and adding one expensive. A file absent from this map must hold none.
#:
#: Per file rather than one total so the failure can say WHICH file gained one.
#: A total alone would report an arithmetic difference and leave the author to
#: find it, and it would also let one file shed a citation while another
#: acquired one -- a net zero that is exactly the regression this is for.
#:
#: These numbers do not go stale in the way a pinned measurement does, and
#: that distinction is the whole reason they are allowed to be hand-written in
#: a repository whose recurring defect is a pinned figure rotting on its own. A
#: measurement is falsified by the tree changing underneath it. An upper bound
#: is not: nothing makes one of these wrong except somebody adding a citation,
#: which is the event it exists to report. Renaming a file fails here with its
#: old name, loudly, and the fix is to move the entry.
#:
#: **Every entry equals its file's exact count, so the slack the design leans
#: on is zero everywhere.** That was first written as a caveat awaiting its
#: first real repair, and the repair has happened: converting `docs/09` and
#: `docs/10` to symbol citations took 13 counted citations out of the two
#: document guards, 23 -> 14 and 6 -> 2.
#:
#: **It did not leave slack, and that was a deliberate choice rather than the
#: design working.** `test_no_ceiling_is_slack_by_more_than_the_file_holds`
#: tolerates an entry above its file and only refuses once the file holds none
#: at all, so 23 over 14 would have passed -- nine free citations in a file
#: whose own guard now refuses every one in the document it checks. "A repair
#: is free" is a statement about the comparison, not a licence to bank the
#: difference. A repair should come with its ceiling.
CITATION_CEILINGS: Dict[str, int] = {
    "harness/drivers/base.py": 9,
    "harness/drivers/mongo.py": 1,
    "harness/drivers/mysql_family.py": 1,
    "harness/drivers/postgres.py": 3,
    "harness/drivers/valkey.py": 4,
    "harness/metrics/records.py": 1,
    "harness/workloads/churn.py": 3,
    "harness/workloads/client_process.py": 2,
    "harness/workloads/clients.py": 1,
    "harness/workloads/filtered.py": 2,
    "orchestrator/ann_pass.py": 2,
    "overlay/ann-benchmarks/ann_benchmarks/algorithms/percona/module.py": 22,
    "report/crosscheck.py": 3,
    "tests/fake_driver.py": 2,
    "tests/test_c4_pgvector_ingest.py": 2,
    "tests/test_e10_report_unit_join_keys.py": 2,
    "tests/test_e13_percona_churn_repair_is_a_noop.py": 1,
    # 23 -> 14 when docs/09 was converted: the table of citations the
    # document made went with it, and three more became symbols or took
    # the revision-scoped form. Lowered in the same change, because a
    # ceiling left at 23 over a file holding 14 is nine free citations in
    # a file whose whole job is now to refuse them, and
    # `test_no_ceiling_is_slack_by_more_than_the_file_holds` only fires
    # at zero.
    "tests/test_e1_percona_design.py": 14,
    # 6 -> 2, same change, same reason.
    "tests/test_e3_rollout_doc.py": 2,
    "tests/test_e4_percona_build_flags.py": 21,
    "tests/test_e5_percona_make_output_sync.py": 1,
    "tests/test_e6_percona_vendor_alignment.py": 1,
    "tests/test_e7_percona_fortify_level.py": 3,
    "tests/test_e8_package_flags_overwrite.py": 24,
    "tests/test_estimate.py": 14,
    "tests/test_g3_process_clients.py": 1,
    "tests/test_t3b_drivers.py": 1,
    "tests/test_t3c_filtered.py": 1,
    "tests/test_t4a_churn.py": 3,
    "tests/test_t4b_readers.py": 4,
    "tests/test_t5b_stats.py": 1,
}

#: The two guards whose own documents now ratchet to ZERO line citations, and
#: for which a ceiling above the file is therefore not grandfathered slack but
#: a reopened hole.
#:
#: WHY THESE TWO ARE DIFFERENT from every other entry above. The census
#: tolerates slack on purpose: a repair should not fail this file, and
#: `test_no_ceiling_is_slack_by_more_than_the_file_holds` only refuses once a
#: file holds none at all. That is right for a file whose citations are
#: incidental. It is wrong for these two, because each one IS the guard that
#: refuses every line citation in the document it checks -- banked allowance
#: there is free citations in the one place the repository has decided there
#: will be none.
#:
#: Measured, not supposed: raising `tests/test_e1_percona_design.py` back from
#: 14 to 23 after the conversion passed the whole suite. That mutant survived
#: until this list existed.
RATCHETED_TO_ZERO = (
    "tests/test_e1_percona_design.py",
    "tests/test_e3_rollout_doc.py",
)

TOTAL_CEILING = sum(CITATION_CEILINGS.values())

#: Named in every failure, because a guard that only says no teaches nothing.
#: `docs/11-methodology-roadmap.md` §0 is where this policy is written down and
#: `tests/test_e15_methodology_roadmap_doc.py` is what holds that document to
#: it; this file is the same policy applied to Python prose, minus the ability
#: to check what a name resolves to.
POLICY = (
    "Cite the SYMBOL, not the line: the function, class, constant, config "
    "key or test id. A line number dies to any edit above it, including an "
    "edit that changes nothing the citation is about; a name dies only to a "
    "rename, which is a deliberate act a reader can follow. If the exact "
    "text matters, quote it and say which symbol it sits in. If the number "
    "is a historical record of where something used to be, write it in "
    "git's own rev-then-path form with a commit or tag in front, so it names "
    "the frozen tree it is true in and stops being a live citation. "
    "docs/11-methodology-roadmap.md section 0 states the policy; "
    "tests/test_e15_methodology_roadmap_doc.py enforces it for docs/11."
    "\n\nAnd do not route around the count. This census sees a path-and-line "
    "citation and a bare range continuing one, in a Python comment or "
    "docstring. It does NOT see a line number spelled out in words, an "
    "extension outside its own set, a citation inside an assertion message, "
    "or anything in a file that is not Python. Those forms are uncounted, "
    "not permitted -- the module docstring lists all four and says why each "
    "is left open."
)


class TestTheScanIsNotVacuous:
    """Floors first, because every ceiling below is an assertion over a set."""

    def test_the_walk_read_the_tree(self):
        files, counts = census()
        problems = vacuity_problems(files, counts)
        assert not problems, problems

    def test_it_read_the_files_the_census_turns_on(self):
        """Sentinels. The floors above are satisfied by a hundred of the wrong
        files; these are the ones whose comments the ceilings are about."""
        files, _ = census()
        rel = {os.path.relpath(f, VB_ROOT).replace(os.sep, "/")
               for f in files}
        for expected in ("orchestrator/cli.py",
                         "harness/drivers/base.py",
                         "tests/test_e3_rollout_doc.py",
                         "tests/test_t3c_filtered.py",
                         "overlay/ann-benchmarks/ann_benchmarks/algorithms/"
                         "percona/module.py",
                         SELF_REL):
            assert expected in rel, expected

    def test_a_walk_that_reads_nothing_is_reported(self, tmp_path):
        """The failure mode one guard in this series shipped with: a scan over
        no files, passing every assertion it makes."""
        files, counts = scan(str(tmp_path))
        assert files == []
        problems = vacuity_problems(files, counts)
        assert problems, "an empty tree passed the non-vacuity check"
        assert any("Python files" in p for p in problems), problems

    def test_a_matcher_that_finds_nothing_is_reported(self):
        """The other direction, over the real tree: the walk works, the
        detector does not. Without this the ceilings below would all pass on a
        regex that had been broken by a typo."""
        files, counts = scan(VB_ROOT, cite_re=_MATCHES_NOTHING)
        assert len(files) >= MIN_FILES, len(files)
        assert counts == {}, counts
        problems = vacuity_problems(files, counts)
        assert problems, "a matcher that finds nothing passed the check"
        assert any("line citations" in p for p in problems), problems


#: `(text, expected matches)`. Lives as data rather than as prose so that the
#: forms the matcher must catch can be spelled literally without this file
#: appearing in its own census -- see the module docstring.
MATCHER_SAMPLES: Tuple[Tuple[str, List[str]], ...] = (
    # The plain forms, with and without code font, with and without a
    # directory, with and without a range. All four appear in this tree.
    ("# `orchestrator/cli.py:508` runs the pass loop",
     ["orchestrator/cli.py:508"]),
    ("# cli.py:508 runs the pass loop", ["cli.py:508"]),
    ("# see harness/workloads/build.py:93-95 for the condition",
     ["harness/workloads/build.py:93-95"]),
    ("# two of them: `a/b.py:1` and `c.md:22`", ["a/b.py:1", "c.md:22"]),
    # Every extension the document guards recognise.
    ("# x.sh:1 y.js:2 z.yml:3 w.md:4 v.cc:5 u.h:6 t.txt:7",
     ["x.sh:1", "y.js:2", "z.yml:3", "w.md:4", "v.cc:5", "u.h:6", "t.txt:7"]),
    # Revision-scoped: a frozen tree, so not a live citation. Both spellings
    # this tree already uses -- a short sha and a vendor tag.
    ("# `6ee0b40:report/generate.py:497` said so at the time", []),
    ("# `mariadb-11.8.8:CMakeLists.txt:290` at the pin", []),
    # A comment marker is NOT a revision, however much it looks like one to a
    # regex. Both of these count.
    ("# TODO:orchestrator/cli.py:508 needs a name", ["orchestrator/cli.py:508"]),
    ("# NOTE:base.py:40 is the default", ["base.py:40"]),
    # Continuations: a bare number after a path, which is how a sentence
    # writes its second and third citation. Counted, and counted with the
    # status of the path they hang off -- a continuation of a frozen citation
    # is frozen too.
    ("# `postgres.py:689` and the dict at `:701-707`",
     ["postgres.py:689", ":701-707"]),
    ("# `a/b.py:1`, `:2` and `:3-4`", ["a/b.py:1", ":2", ":3-4"]),
    ("# `ffcb3a7:tests/test_config.py:6988`/`:7003` at that commit", []),
    # The same thing across a wrapped comment block, which is where this file
    # wrote one and could not see it.
    ("#: `harness/workloads/churn.py:244` bounds its slices, and\n"
     "#: `:250-251` states why",
     ["harness/workloads/churn.py:244", ":250-251"]),
    # Not citations. A bare number with NO path before it is genuinely
    # unreadable as one -- that, and not the bare form in general, is what the
    # original rationale was about. A path with no line is the form being
    # asked for; a version, a clock time and a ratio are none of the above.
    ("# the dict at :701-707", []),
    ("# `harness/metrics/records.py` and `SCHEMA_VERSION`", []),
    ("# mariadb-11.8.8 and AliSQL-8.0.44-2", []),
    ("# 11.8.8:12 is not a path", []),
    ("# #47 merged at 10:35 and #46 at 16:17", []),
    ("# a 1:2 split of the budget", []),
)


class TestTheMatcherRecognisesEachForm:
    """What the detector must catch and what it must leave alone.

    This is the real non-vacuity proof. The floors above establish that the
    scan read something; this establishes that what it read was understood,
    and it is the only arm that still fails if the census is ever cleaned down
    to nothing.
    """

    @pytest.mark.parametrize("text,expected", MATCHER_SAMPLES)
    def test_the_sample_is_read_as_intended(self, text, expected):
        assert live_citations(text) == expected, text

    def test_the_samples_cover_both_verdicts(self):
        """A table of positives only would pass a matcher that matched
        everything, and a table of negatives only one that matched nothing."""
        positive = [s for s in MATCHER_SAMPLES if s[1]]
        negative = [s for s in MATCHER_SAMPLES if not s[1]]
        assert len(positive) >= 4, len(positive)
        assert len(negative) >= 4, len(negative)

    def test_the_revision_exemption_is_only_the_prefix(self):
        """The exemption must attach to the citation beside the revision and
        not to the sentence it sits in -- otherwise one historical quotation
        would exempt every live citation in the same comment."""
        text = ("# `6ee0b40:report/generate.py:497` moved; it is now at "
                "`report/generate.py:612`")
        assert live_citations(text) == ["report/generate.py:612"], text

    def test_a_comment_marker_is_not_a_revision(self):
        """`_looks_like_revision`, directly, because the hole it closes was
        invisible in the census: no file in this tree writes `TODO:` against a
        path today, so the samples above are the only place the distinction is
        exercised and they would not say WHY it failed."""
        for revision in ("c8de6a5", "ffcb3a7", "6ee0b40", "a695c6d",
                         "mariadb-11.8.8", "AliSQL-8.0.44-2", "v0.8.6",
                         "valkey-91"):
            assert _looks_like_revision(revision), revision
        for marker in ("TODO", "NOTE", "FIXME", "XXX", "see", "at", "base",
                       "HACK", "WARNING"):
            assert not _looks_like_revision(marker), marker

    def test_the_exemption_is_in_use_in_the_tree(self):
        """Non-vacuity for the exemption itself: an exemption nothing uses is
        an untested branch, and this one carries the whole historical-quotation
        decision."""
        scoped = 0
        for path in python_files(VB_ROOT):
            source = open(path, encoding="utf-8", errors="replace").read()
            for span in prose_spans(source):
                for m in _CITATION_RE.finditer(span):
                    if _is_revision_scoped(span, m.start()):
                        scoped += 1
        assert scoped >= 3, scoped

    def test_the_continuation_form_is_in_use_in_the_tree(self):
        """Non-vacuity for the half added after review. A continuation matcher
        that found nothing in the tree would leave the hole open while reading
        as coverage, which is the state this file shipped in."""
        found = 0
        for path in python_files(VB_ROOT):
            source = open(path, encoding="utf-8", errors="replace").read()
            for span in prose_spans(source):
                found += sum(1 for c in live_citations(span)
                             if c.startswith(":"))
        assert found >= 3, found

    def test_comment_blocks_are_joined(self):
        """The grouping the continuation half depends on, asserted on its own.
        `tokenize` yields one token per line; a continuation two lines below
        its path would lose its anchor without this, which is precisely the
        instance this file's own commit wrote."""
        source = ("#: first line\n"
                  "#: second line\n"
                  "\n"
                  "x = 1  # a separate block\n")
        spans = prose_spans(source)
        assert "#: first line\n#: second line" in spans, spans
        assert "# a separate block" in spans, spans


class TestNoFileGainedALineCitation:
    """The ratchet. Grandfathered above, refused below."""

    def test_no_file_holds_more_than_its_ceiling(self):
        _files, counts = census()
        risen = []
        for rel, found in sorted(counts.items()):
            ceiling = CITATION_CEILINGS.get(rel, 0)
            if found > ceiling:
                risen.append(f"{rel}: {found} now, ceiling {ceiling}")
        assert not risen, (
            "a line citation was added to a Python comment or docstring in:\n  "
            + "\n  ".join(risen)
            + "\n\n" + POLICY
            + "\n\nIf the citation is genuinely the clearest thing to write, "
              "raise that file's entry in CITATION_CEILINGS in the same "
              "commit and say in the message why a symbol would not do.")

    def test_the_total_has_not_risen(self):
        """The same statement in one number, so a failure reports the size of
        the regression and not only its location."""
        _files, counts = census()
        total = sum(counts.values())
        assert total <= TOTAL_CEILING, (
            f"{total} line citations in Python prose, up from "
            f"{TOTAL_CEILING}. {POLICY}")

    def test_every_ceiling_is_for_a_file_that_exists(self):
        """A stale entry is an exemption nobody can see: it would let a deleted
        file's allowance silently cover a new citation under the same name."""
        missing = [rel for rel in CITATION_CEILINGS
                   if not os.path.isfile(os.path.join(VB_ROOT, rel))]
        assert not missing, missing

    def test_no_ceiling_is_slack_by_more_than_the_file_holds(self):
        """A ceiling far above its file is a hole. Entries are allowed to be
        higher than the count -- repairs should not fail this file -- but an
        entry for a file that now holds NONE is dead and has to go, or the
        next citation added there passes."""
        dead = [rel for rel, ceiling in sorted(CITATION_CEILINGS.items())
                if ceiling > 0 and rel not in census()[1]]
        assert not dead, (
            "these files no longer hold any line citation, so their entries "
            "are pure allowance, so remove them\n  "
            + "\n  ".join(dead))

    def test_a_document_guards_ceiling_is_exact(self):
        """No banked slack in the two files that refuse citations for a living.

        The test above tolerates a ceiling above its file and only refuses at
        zero, which is the right rule for a file whose citations are
        incidental and the wrong one for these two: each is the guard that
        ratchets its document to zero, so allowance there is free citations in
        the one place the repository has decided there will be none.

        This is a mutant that survived. Converting docs/09 and docs/10 took 13
        counted citations out of these two files; the ceilings were lowered in
        the same change, and raising
        `tests/test_e1_percona_design.py` back from 14 to 23 afterwards passed
        the whole suite. "A repair is free" is a statement about the
        comparison, not a licence to bank the difference -- so for these two a
        repair comes with its ceiling or it fails here.
        """
        _files, counts = census()
        wrong = []
        for rel in RATCHETED_TO_ZERO:
            ceiling = CITATION_CEILINGS.get(rel)
            live = counts.get(rel, 0)
            if ceiling != live:
                wrong.append(f"{rel}: ceiling {ceiling}, file holds {live}")
        assert not wrong, (
            "a document guard's ceiling must equal its exact count, because a "
            "ceiling above it is allowance in the file that refuses every "
            "citation in the document it checks. Lower it in the same commit "
            "as the repair:\n  " + "\n  ".join(wrong))

    def test_the_ratcheted_files_are_really_the_document_guards(self):
        """Non-vacuity for the list above: an empty or stale tuple would make
        the exactness check pass over nothing."""
        assert len(RATCHETED_TO_ZERO) == 2, RATCHETED_TO_ZERO
        for rel in RATCHETED_TO_ZERO:
            assert os.path.isfile(os.path.join(VB_ROOT, rel)), rel
            assert rel in CITATION_CEILINGS, rel
            # Each must actually contain a zero-ratchet assertion, or it is
            # not a file this rule is about.
            source = open(os.path.join(VB_ROOT, rel)).read()
            assert "contains_no_line_citation" in source, rel

    def test_this_file_makes_none_of_the_citations_it_counts(self):
        """The ban is unconditional, which is only possible because this file
        does not need to break it: every literal form it has to spell lives in
        `MATCHER_SAMPLES`, which is data and not prose.

        `docs/11` §0 made the same choice for the same reason -- it writes its
        one illustration of the banned form in italics rather than carving out
        an allowlist entry for itself -- and
        `tests/test_e15_methodology_roadmap_doc.py` records what happens when a
        guard's index does include the guard: it certifies names that exist
        nowhere else.
        """
        _files, counts = census()
        assert counts.get(SELF_REL, 0) == 0, (
            f"{SELF_REL} now makes {counts[SELF_REL]} of the citations it "
            f"counts; spell the form in MATCHER_SAMPLES instead")
        assert SELF_REL not in CITATION_CEILINGS


class TestAPlantedCitationIsCounted:
    """The census, proved end to end on a tree built for the purpose.

    The sample table proves the regex. This proves the rest: the walk finds a
    file, the comment and the docstring are both extracted from it, the string
    literal beside them is not, and the count lands under the right path. A
    regex test alone would pass against a walk that read no comments.
    """

    def _plant(self, tmp_path, body: str) -> Dict[str, int]:
        pkg = tmp_path / "harness"
        pkg.mkdir()
        (pkg / "planted.py").write_text(body, encoding="utf-8")
        _files, counts = scan(str(tmp_path))
        return counts

    def test_a_citation_in_a_comment_is_counted(self, tmp_path):
        counts = self._plant(
            tmp_path, "x = 1  # see orchestrator/cli.py:508 for the loop\n")
        assert counts == {"harness/planted.py": 1}, counts

    def test_a_citation_in_a_docstring_is_counted(self, tmp_path):
        counts = self._plant(
            tmp_path,
            '"""Module.\n\nAs orchestrator/cli.py:508 has it.\n"""\n')
        assert counts == {"harness/planted.py": 1}, counts

    def test_a_citation_in_a_function_docstring_is_counted(self, tmp_path):
        counts = self._plant(
            tmp_path,
            'def f():\n    """As orchestrator/cli.py:508 has it."""\n'
            '    return 1\n')
        assert counts == {"harness/planted.py": 1}, counts

    def test_a_citation_in_an_ordinary_literal_is_not(self, tmp_path):
        """The boundary the module docstring argues for, as a test rather than
        a claim: this is the shape of a `CITED_IN_REPO` row and of a planted
        bad document."""
        counts = self._plant(
            tmp_path, 'CITED = {"orchestrator/cli.py:508": "the pass loop"}\n')
        assert counts == {}, counts

    def test_a_revision_scoped_citation_is_not_counted(self, tmp_path):
        counts = self._plant(
            tmp_path, "x = 1  # a695c6d:orchestrator/cli.py:508 had the loop\n")
        assert counts == {}, counts

    def test_a_continuation_across_a_wrapped_comment_is_counted(self,
                                                                tmp_path):
        """End to end, in the exact shape this file's own commit wrote: the
        path on one comment line and the continuation two lines below it. The
        sample table proves the regex on one joined string; this proves the
        join happens when the text comes off disk."""
        counts = self._plant(
            tmp_path,
            "#: the cell read is at orchestrator/cli.py:508 --\n"
            "#: `cmd_run`, and the second one is at\n"
            "#: `:512-514` in the same body\n"
            "x = 1\n")
        assert counts == {"harness/planted.py": 2}, counts

    def test_a_continuation_of_a_frozen_citation_is_not_counted(self,
                                                                tmp_path):
        """The inheritance rule, which is what keeps the historical quotations
        in `tests/test_g3_process_clients.py` out of the census: the bare
        numbers there hang off a sha-scoped path and are as frozen as it is."""
        counts = self._plant(
            tmp_path,
            "x = 1  # a695c6d:orchestrator/cli.py:508 and :512-514 then\n")
        assert counts == {}, counts

    def test_a_clean_file_contributes_nothing(self, tmp_path):
        """The paired negative: the planted arms above would pass a counter
        that returned 1 for every file it read."""
        counts = self._plant(
            tmp_path,
            '"""Module.\n\nAs `cmd_run`\'s pass loop has it.\n"""\n'
            "x = 1  # and orchestrator/cli.py names it\n")
        assert counts == {}, counts
