"""Guards for the methodology roadmap in docs/11-methodology-roadmap.md.

This file is the checkable half of that document, and it guards a DIFFERENT
citation form from the two documents that came before it.

WHY THE FORM CHANGED. `docs/09-adding-percona-server.md` and
`docs/10-rollout.md` USED TO cite `path:line`, each under a guard that
re-opened the cited line and compared it against what the document claimed was
there (`tests/test_e1_percona_design.py` and `tests/test_e3_rollout_doc.py`).
Those guards worked -- they are the reason the rot was found at all -- but the
rot kept happening: line citations in this repository went stale twice in two
days and put `master` red both times, repaired in #41 and #43. Neither was a
careless edit to a document. In both cases a later round of edits inside the
SAME change moved lines below the point where the citations had been derived.
Both documents have since been converted, so this page's form is now the
repository's only citation form rather than a departure from the other two.

So docs/11 cites SYMBOLS. A line number is invalidated by any edit above it,
including one that changes nothing the citation is about; a symbol name is
invalidated only by a rename. The document says so in its own §0, and this file
is what stops that from being a promise.

WHAT THIS FILE ESTABLISHES

* Every code-font path in docs/11 names a file that exists in this tree --
  except the ones declared `ABSENT_PATHS`, which are asserted to be ABSENT, so
  that the day `harness/workloads/rate.py` lands the document saying it does
  not exist fails here rather than misleading a reader.
* Every code-font symbol in docs/11 still appears somewhere in the tree, and a
  floor of them resolve to a DEFINITION (a `def`, a `class`, a module- or
  class-level assignment, or a YAML key) rather than to a passing mention in a
  comment. The second floor is what keeps the first from being satisfied by
  prose.
* docs/11 contains NO `path:line` citation at all. That is the policy itself,
  executable -- and the same detector is run against a frozen fixture that is
  made of them, so a regex that stopped matching fails here instead of
  certifying every document in the repo as clean. The fixture used to be
  docs/10, which was "full of them"; it is not any more, because docs/09 and
  docs/10 were converted to symbol citations under this page's policy. A
  non-vacuity fixture must not be something another document can stop being.
* docs/11 cites nothing under a generated, gitignored tree (`sources/`,
  `work/`, `results/`, `datasets/`, `state/`). A path under one of those does
  not exist on a fresh checkout, so it is a citation nobody but the author can
  check. The document names vendor material in prose instead, and §0 says so.

WHAT IT CANNOT ESTABLISH, and the honest limit of the whole scheme: that a
symbol still MEANS what the document says it means. A renamed-away symbol fails
here. A symbol that kept its name and changed its behaviour does not. That is
the price of a citation form that does not rot weekly, and it is why the
document quotes the passages it leans on rather than only pointing at them.

NON-VACUITY, which this series has already got wrong once -- a guard shipped
green because it scanned zero files. Every parser here has a floor on what it
found and a sentinel it must find, and `TestThePlantedBadReferenceFails` runs
the whole classify-and-resolve pipeline against a synthetic document carrying
one absent symbol and one absent path and asserts both are reported. The
pipeline takes TEXT rather than reading the document itself for exactly that
reason: a mutant arm is only possible if the bad document can be constructed.
"""

from __future__ import annotations

import ast
import os
import re
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import pytest

VB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROADMAP_DOC = os.path.join(VB_ROOT, "docs", "11-methodology-roadmap.md")

#: Where a cited symbol is allowed to live. `docs/` is deliberately NOT here:
#: indexing the documentation would let docs/11 cite a symbol that exists in no
#: code at all and resolve it against another document's prose, which is the
#: one outcome that would make this whole file decorative.
#:
#: `tests/` IS here, and that is a real widening worth naming: a symbol that
#: exists only in a test resolves. Tests are this repository's code and several
#: things docs/11 points at -- a guard class, a fake driver -- live nowhere
#: else, so excluding them would force the document to stop naming them.
SOURCE_DIRS = ("harness", "report", "orchestrator", "overlay", "config",
               "webui", "scripts", "docker", "tests")

#: Root-level files that are part of the tree a citation may resolve against.
ROOT_SOURCES = ("run-benchmark.sh",)

#: Suffixes read into the index. Text only: a `.tar` or an `.hdf5` under a
#: generated tree would be read as bytes and is not a place a symbol is
#: declared.
INDEXED_SUFFIXES = (".py", ".yml", ".yaml", ".sh", ".js", ".json", ".html",
                    ".cnf", ".conf", ".txt")

#: THIS FILE IS EXCLUDED FROM ITS OWN INDEX, and the exclusion is the whole
#: reason the planted arm works. It was not foreseen: the first run of
#: `TestThePlantedBadReferenceFails` PASSED the live document and FAILED the
#: plant, because the planted names are spelled here, `tests/` is indexed, and
#: a symbol this file mentions therefore resolved. Left in, that is worse than
#: a vacuous guard -- docs/11 could cite a symbol that exists nowhere but in
#: the file guarding docs/11, and the citation would resolve.
SELF = os.path.abspath(__file__)

#: Generated and gitignored, per CLAUDE.md. A citation under one of these is
#: uncheckable on a fresh checkout, so docs/11 must not make one.
GENERATED_PREFIXES = ("sources/", "work/", "results/", "datasets/", "state/")

#: Paths docs/11 names BECAUSE THEY DO NOT EXIST, with the reason. Asserted
#: absent rather than skipped: the document's §6 turns on `rate.py` having been
#: specified and deferred, so the day it lands, the sentence saying it has not
#: is wrong and has to fail somewhere.
ABSENT_PATHS: Dict[str, str] = {
    "harness/workloads/rate.py":
        "the open-loop latency workload, specified and deferred. docs/10 "
        "records it as optional and absent and the suite asserts its absence; "
        "docs/11 §6 is the design for it. If this file appears, §6 is no "
        "longer a roadmap item and the document has to say what landed.",
}

#: Code-font spans that look like a symbol, are not one, and could not be
#: reworded out of the document without making a sentence worse. Each needs a
#: reason, and each is asserted NOT to resolve -- an entry that resolves is a
#: real symbol being excused, which is how an escape hatch becomes a hole.
#:
#: Empty, and that is a finding rather than a default: every identifier in
#: code font in docs/11 is a thing in this tree. Vendor type names and vendor
#: document sections are in italics there, which §0 states as a rule, so none
#: of them needed an entry here.
VOCABULARY: Dict[str, str] = {}

#: A code-font span that is a path. Anchored at both ends, so `extra.quantization`
#: is not mistaken for a file and `harness/main.py` is not mistaken for a symbol.
_PATH_RE = re.compile(
    r"^[A-Za-z0-9_][A-Za-z0-9_./-]*\.(?:py|yml|yaml|md|sh|js|cc|h|txt|json)$")

#: A code-font span that is a symbol: an identifier, optionally dotted, with an
#: optional empty call. Deliberately narrow -- `vector(dim)`, `--quantization`,
#: `p50/p95/max` and `x86-64-v3` all fail it and are classified as prose, which
#: is what they are.
_SYMBOL_RE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*(?:\(\))?$")

_CODE_SPAN_RE = re.compile(r"`([^`\n]+)`")

#: The citation form every document in this repository bans, and THE ONLY
#: COPY OF IT. It used to be one of three: this one, and one each in
#: `tests/test_e1_percona_design.py` and `tests/test_e3_rollout_doc.py`, with
#: a comment here saying it was "kept deliberately identical" to one of them.
#: Three patterns that must agree are three chances to disagree, and they did
#: disagree -- the other two were narrower than they needed to be, and the
#: narrow form could not see MTR `.result`/`.test` paths, `.yy` grammars,
#: extensionless files, bare `:N` continuations or a span carrying several
#: ranges. On `8a82da8` that was 130 of docs/09's 199 line references
#: invisible, so a ratchet built on the narrow form would have certified a
#: document carrying 130 live citations.
#:
#: It lives HERE, in the guard for the document that states the policy, and
#: the other two import it. The direction matters and is forced: those two
#: already import `classify` from this module, so a pattern defined there and
#: imported here would be a circular import between test modules.
#:
#: WIDENED AT FOUR POINTS, and docs/11 is now ratcheted over all of them
#: rather than over the narrow form alone:
#:
#:   * any extension, not a fixed list;
#:   * any extensionless file, not `Dockerfile` alone;
#:   * a bare `:N` continuation of a path named earlier in the sentence;
#:   * a span carrying several ranges (`foo.result:3-5, :13-17`).
#:
#: Deliberately generous: `max(ef, k)` and `p50/p95` do not match, because the
#: tail must be a colon and digits at the end of the span or before a comma,
#: but `tag:12` would. A false positive costs one reworded sentence; a false
#: negative costs the whole guard. `CITATION_FIXTURE` is what proves it still
#: sees each form.
_LINE_CITATION_RE = re.compile(
    r"`([^`\n]*?):(\d+)(?:-(\d+))?((?:\s*,\s*:\d+(?:-\d+)?)*)\s*`")


def line_citations(text):
    """Every line citation in `text`, as the text spells it.

    The WHOLE text, fences included: a gutter number inside a fenced block is
    not in a code-font span and this pattern cannot see one -- that class has
    its own check in `tests/test_e1_percona_design.py` -- but a backticked
    citation smuggled into a fence is visible here, and docs/09 §4.10's own
    history is why that matters.
    """
    return ["%s:%s%s" % (m.group(1), m.group(2),
                         ("-" + m.group(3) if m.group(3) else "") + m.group(4))
            for m in _LINE_CITATION_RE.finditer(text)]


#: The frozen fixture the pattern is proved against, carrying ONE OF EACH form
#: above plus spans that must NOT match. A literal and not a slice of a
#: document, and that is the correction of a real defect rather than a
#: preference: this file used to prove its detector by running it against
#: `docs/10-rollout.md` and asserting at least thirty matches, which was true
#: and was a dependency on another document's citation style. Converting
#: docs/10 under this page's own policy would have reddened this guard for no
#: defect at all. A fixture must not be something another document can stop
#: being.
CITATION_FIXTURE = (
    "a `foo.py:12` and a `bar/baz.result:13-15` and a `sql_yacc.yy:9` "
    "and a `docker/percona/Dockerfile:725` and a bare `:701-707` "
    "and a multi `r/x.result:3-5, :13-17, :23-26` "
    "but not `max(ef, k)` and not `harness/main.py` and not `p50/p95/max`"
)

CITATION_FIXTURE_EXPECTED = 6


def _doc() -> str:
    return open(ROADMAP_DOC).read()


# ---------------------------------------------------------------------------
# Extraction: what the document cites, as the document spells it
# ---------------------------------------------------------------------------

def _code_spans(text: str) -> List[str]:
    return [m.group(1).strip() for m in _CODE_SPAN_RE.finditer(text)]


def classify(text: str) -> Tuple[Set[str], Set[str], Set[str]]:
    """The paths, the symbols and the prose in `text`'s code-font spans.

    Takes text rather than reading the document, so that the planted-failure
    arm can build a bad document and run the real classifier over it.
    """
    paths: Set[str] = set()
    symbols: Set[str] = set()
    prose: Set[str] = set()
    for span in _code_spans(text):
        bare = span[:-2] if span.endswith("()") else span
        if _PATH_RE.match(span):
            paths.add(span)
        elif _SYMBOL_RE.match(span):
            symbols.add(bare.rsplit(".", 1)[-1])
        else:
            prose.add(span)
    return paths, symbols, prose


# ---------------------------------------------------------------------------
# The index a symbol resolves against, built once
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_YAML_KEY_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:", re.M)

_INDEX: Dict[str, Set[str]] = {}


def _indexed_files() -> List[str]:
    out: List[str] = []
    for name in ROOT_SOURCES:
        full = os.path.join(VB_ROOT, name)
        if os.path.isfile(full):
            out.append(full)
    for top in SOURCE_DIRS:
        root_dir = os.path.join(VB_ROOT, top)
        for root, dirs, files in os.walk(root_dir):
            dirs[:] = [d for d in dirs
                       if d not in ("__pycache__", ".git", ".pytest_cache")]
            for filename in files:
                if not filename.endswith(INDEXED_SUFFIXES):
                    continue
                full = os.path.join(root, filename)
                if os.path.abspath(full) == SELF:
                    continue
                out.append(full)
    return out


def _declared_in_python(source: str) -> Iterable[str]:
    """Every name a module DECLARES: def, class, and assignment targets.

    Parsed rather than grepped, because the point of this half is to be
    stronger than a word match -- a symbol that appears only inside a comment
    must not reach it.

    DELIBERATELY GENEROUS about what counts as a declaration: an attribute
    assigned anywhere (`self.foo = ...`) and an imported name both land here,
    not only module-level definitions. That is the right side to err on for
    what this index is FOR -- it is the stronger of two floors, and a floor
    that refused instance attributes would push the document away from naming
    record fields, which are most of what it has to name. It is not a claim
    that every name here is a public API.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ()
    names: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                for sub in ast.walk(target):
                    if isinstance(sub, ast.Name):
                        names.add(sub.id)
                    elif isinstance(sub, ast.Attribute):
                        names.add(sub.attr)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[-1])
    return names


def _index() -> Dict[str, Set[str]]:
    """`{"mentioned": ..., "declared": ..., "files": ...}`, built once."""
    if _INDEX:
        return _INDEX
    mentioned: Set[str] = set()
    declared: Set[str] = set()
    files = _indexed_files()
    for path in files:
        try:
            source = open(path, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        mentioned.update(_WORD_RE.findall(source))
        if path.endswith(".py"):
            declared.update(_declared_in_python(source))
        elif path.endswith((".yml", ".yaml")):
            declared.update(_YAML_KEY_RE.findall(source))
    _INDEX["mentioned"] = mentioned
    _INDEX["declared"] = declared
    _INDEX["files"] = set(files)
    return _INDEX


def unresolved(text: str) -> Tuple[List[str], List[str]]:
    """The symbols and the paths in `text` that this tree cannot account for.

    The whole pipeline in one function, so the planted-failure arm exercises
    exactly what the live assertions exercise and not a copy of it.
    """
    paths, symbols, _prose = classify(text)
    idx = _index()
    bad_symbols = sorted(s for s in symbols
                         if s not in VOCABULARY and s not in idx["mentioned"])
    bad_paths = sorted(p for p in paths
                       if p not in ABSENT_PATHS
                       and not os.path.exists(os.path.join(VB_ROOT, p)))
    return bad_symbols, bad_paths


# ---------------------------------------------------------------------------
# The parsers, before anything is asserted with them
# ---------------------------------------------------------------------------

class TestTheParsersFoundSomething:
    """Floors and sentinels, because every assertion below is over a set.

    A classifier that matched nothing would make this entire file pass against
    an empty document, which is the exact failure one guard in this series
    shipped with. The floors carry a little slack and no more: enough that
    tightening a paragraph does not fail here, not enough that a parser could
    lose most of what it finds and stay green.
    """

    def test_the_index_read_real_files(self):
        idx = _index()
        assert len(idx["files"]) >= 150, len(idx["files"])
        assert len(idx["mentioned"]) >= 5000, len(idx["mentioned"])
        assert len(idx["declared"]) >= 1500, len(idx["declared"])

    def test_the_index_holds_things_it_must_hold(self):
        """Sentinels by KIND, one per route into the index, so that a route
        quietly producing nothing is visible. Chosen as names that cannot
        plausibly be renamed away together: a function, a constant, a class,
        a YAML key and a shell file's presence."""
        idx = _index()
        for name in ("wait_index_synced", "OPS_MEASUREMENT_VERSION",
                     "PoolSpec", "mongodb_quantization"):
            assert name in idx["declared"], name
            assert name in idx["mentioned"], name

    def test_the_index_does_not_read_this_file(self):
        """Measured, not supposed. With this file in the index the planted arm
        below passes vacuously, because the plant's names are spelled here."""
        idx = _index()
        assert SELF not in {os.path.abspath(f) for f in idx["files"]}

    def test_the_index_does_not_read_the_documentation(self):
        """`docs/` is excluded on purpose: a symbol that exists only in prose
        must not resolve, or this file becomes a check that docs/11 agrees
        with docs/05.

        TWO INDEPENDENT MECHANISMS EXCLUDE IT, and an earlier revision of this
        test named the wrong one. It said it protected against `"docs"` being
        added to `SOURCE_DIRS` -- and planting exactly that left it PASSING,
        because `.md` is not in `INDEXED_SUFFIXES`, so walking `docs/` yields
        no indexed file and the path assertion is satisfied by an empty set.
        The suffix list is the real protection and the directory list is the
        belt; the mutation needs BOTH edits to land, and a test that names one
        of two mechanisms is dead against a change to the other. Both are
        asserted now.
        """
        idx = _index()
        assert not any(os.sep + "docs" + os.sep in f for f in idx["files"])
        assert ".md" not in INDEXED_SUFFIXES, (
            "Markdown is now indexed, so a symbol docs/11 cites could resolve "
            "against another document's prose instead of against code")
        assert "docs" not in SOURCE_DIRS

    def test_the_classifier_found_paths_symbols_and_prose(self):
        """Floors with about a tenth of slack below what the document carries.

        NO EXACT COUNTS HERE, and that is the second time this change learned
        the lesson it is about. This docstring used to print the three figures
        it was calibrated against. They were true at the commit that added
        them and false two commits later -- one prose span and one symbol
        arrived with the follow-ups, and nothing told the author -- which is
        the #41/#43 mechanism exactly, inside the file written to stop it. A
        floor is checked on every run and cannot drift; a figure in a comment
        is checked by nobody. The assertions below are the calibration.
        """
        paths, symbols, prose = classify(_doc())
        assert len(paths) >= 34, sorted(paths)
        assert len(symbols) >= 70, len(symbols)
        # Prose in code font is expected and is not a failure -- a flag, a SQL
        # fragment, a version string. It is asserted non-empty so that a
        # classifier which started calling everything a symbol fails here.
        assert prose, "nothing classified as prose; the classifier widened"

    def test_the_classifier_puts_each_kind_where_it_belongs(self):
        """Non-vacuity for the two regexes, as a table rather than as faith.

        A classifier that called paths symbols would resolve `main.py` against
        the word `main` and pass every assertion below.
        """
        paths, symbols, prose = classify(
            "`harness/main.py` `OPS_MEASUREMENT_VERSION` `wait_index_synced()` "
            "`extra.quantization` `--quantization` `vector(dim)` `x86-64-v3` "
            "`p50/p95/max`")
        assert paths == {"harness/main.py"}
        assert symbols == {"OPS_MEASUREMENT_VERSION", "wait_index_synced",
                           "quantization"}
        assert prose == {"--quantization", "vector(dim)", "x86-64-v3",
                         "p50/p95/max"}


# ---------------------------------------------------------------------------
# The citations themselves
# ---------------------------------------------------------------------------

class TestEverySymbolTheDocumentNamesStillResolves:
    """The guard the document's §0 promises."""

    def test_every_symbol_appears_in_the_tree(self):
        bad_symbols, _ = unresolved(_doc())
        assert not bad_symbols, (
            "docs/11 names these symbols and nothing in the tree has them -- "
            "renamed, or never existed: " + ", ".join(bad_symbols))

    def test_every_path_exists(self):
        _, bad_paths = unresolved(_doc())
        assert not bad_paths, (
            "docs/11 cites these paths and they are not in this checkout: "
            + ", ".join(bad_paths))

    def test_enough_symbols_resolve_to_a_declaration(self):
        """The half that keeps the half above from being satisfied by prose.

        A word match accepts a symbol that survives only inside a comment, and
        a document about work that has not been done is unusually likely to
        name one. So a floor on the symbols that resolve to a real `def`,
        `class`, assignment or YAML key -- stated as a count rather than a
        fraction, because a fraction moves when the document gains a sentence
        about a dict key.
        """
        _paths, symbols, _prose = classify(_doc())
        idx = _index()
        declared = sorted(s for s in symbols if s in idx["declared"])
        assert len(declared) >= 58, (
            f"only {len(declared)} of {len(symbols)} cited symbols resolve to "
            f"a declaration: {declared}")

    def test_the_vocabulary_escapes_are_named_and_do_not_resolve(self):
        """An escape hatch that excuses a real symbol is a hole, not a hatch."""
        idx = _index()
        for span, reason in VOCABULARY.items():
            assert reason.strip(), span
            assert span not in idx["mentioned"], (
                f"{span} is in VOCABULARY but resolves in the tree; it is a "
                f"citation, not vocabulary -- delete the entry")
        assert VOCABULARY == {}, (
            "VOCABULARY has entries; name them in the round report rather "
            "than letting them sit unchecked: " + ", ".join(VOCABULARY))

    @pytest.mark.parametrize("path", sorted(ABSENT_PATHS))
    def test_a_deliberately_absent_path_is_still_absent(self, path):
        assert ABSENT_PATHS[path].strip(), path
        assert not os.path.exists(os.path.join(VB_ROOT, path)), (
            f"{path} now exists; docs/11 describes it as absent and has to be "
            f"updated: {ABSENT_PATHS[path]}")

    @pytest.mark.parametrize("path", sorted(ABSENT_PATHS))
    def test_a_deliberately_absent_path_is_actually_cited(self, path):
        """A row for a path the document stopped naming reads as coverage and
        is not -- the other direction, as docs/10's citation table has."""
        paths, _symbols, _prose = classify(_doc())
        assert path in paths, (
            f"ABSENT_PATHS carries {path} and docs/11 no longer names it")

    def test_the_document_cites_the_guard_that_checks_it(self):
        """§0 claims a guard exists. This is the claim, and this is the file."""
        paths, _symbols, _prose = classify(_doc())
        assert "tests/test_e15_methodology_roadmap_doc.py" in paths


class TestTheGuardsOwnLimitsAreRecordedInTheDocument:
    """§0 has to state what this file cannot do, including its own past bug.

    Pinned because it is the paragraph a later tidy-up deletes first: it is
    self-critical, it is about the test file rather than about the methodology,
    and it reads as an aside. It is neither. The self-poisoning failure is the
    characteristic bug of every guard that builds a corpus and then searches
    it, and the next author to write one of those will read this page before
    reading this file -- so the warning has to survive in the page.
    """

    def test_the_meaning_limit_is_stated(self):
        """The structural limit: a rename fails, a behaviour change does not."""
        text = _plain(_doc())
        assert re.search(r"cannot establish", text, re.I)
        assert re.search(r"kept its name and\s*changed its behaviour", text, re.I)

    def test_the_self_poisoning_bug_is_recorded(self):
        """The defect, named as a defect rather than softened into a design
        note -- and named with its mechanism, because "the guard had a bug" is
        not transferable and "a guard inside its own corpus checks itself" is.
        """
        text = _plain(_doc())
        assert re.search(
            r"index of the tree includes\s*the guard itself", text, re.I), (
            "§0 must record that the guard's index once included the guard, "
            "which is what let a planted bad symbol resolve")
        assert re.search(r"passed the live document and \*{0,2}failed the\s*plant",
                         text, re.I) or "failed the plant" in text
        assert re.search(r"excludes itself from its own index", text, re.I)
        assert re.search(r"checking itself", text, re.I)

    def test_the_directory_form_scope_limit_is_recorded(self):
        """The third limit: `_PATH_RE` is anchored on an extension, so a
        trailing-slash citation into a generated tree is read as prose and
        escapes both the existence and the generated-tree checks.

        Recorded rather than closed, because the document depends on the
        behaviour -- it names `sources/`, `work/` and `tests/` in code font --
        and asserted here so the page cannot stop warning about a hole the
        guard still has. The behaviour itself is pinned below.
        """
        text = _plain(_doc())
        assert re.search(r"file extension", text, re.I)
        assert re.search(
            r"directory citation into a generated\s*tree is exactly as "
            r"uncheckable", text, re.I)

    def test_the_directory_form_really_does_classify_as_prose(self):
        """The behaviour the paragraph above describes, measured. If a later
        change made `_PATH_RE` accept a directory, the page's warning would be
        obsolete and this fails rather than the page going quietly stale."""
        paths, symbols, prose = classify("`work/ann-benchmarks/some/dir/`")
        assert paths == set() and symbols == set()
        assert prose == {"work/ann-benchmarks/some/dir/"}

    def test_the_order_dependent_assertion_is_recorded_too(self):
        """The companion failure, from the same probe. Kept beside the first
        because the pair is the point: neither was visible to reading."""
        text = _plain(_doc())
        assert re.search(r"passed in isolation and failed under the full suite",
                         text, re.I)
        assert "state/" in _doc()

    def test_the_file_really_does_exclude_itself(self):
        """The claim above, verified against the index rather than trusted to
        the prose -- so a document saying the hole is closed and a guard that
        reopened it cannot coexist."""
        assert SELF not in {os.path.abspath(f) for f in _index()["files"]}


class TestTheDocumentMakesNoLineCitation:
    """The citation policy, executable.

    This is the keystone: the document's §0 says it cites no line numbers, and
    a policy stated only in prose is a policy until the first hurry.
    """

    def test_docs_11_contains_no_path_line_citation(self):
        found = _LINE_CITATION_RE.findall(_doc())
        assert not found, (
            "docs/11 cites a line number, which its own §0 forbids -- line "
            "citations in this repo rotted twice in two days (#41, #43). Name "
            "the symbol instead: " + ", ".join(m[0] + ":" + m[1] for m in found))

    def test_the_detector_can_see_a_line_citation(self):
        """Non-vacuity, and not optional: the assertion above is a NEGATIVE
        one, and a negative assertion passes just as happily against a broken
        regex as against a clean document.

        THE FIXTURE IS A FROZEN LITERAL, AND IT USED TO BE docs/10. That read
        "docs/10 is made of these" and asserted at least thirty matches in
        it, which was true and was a dependency between two documents'
        citation styles: when docs/10 was converted to symbol citations under
        the policy this page sets out, this test went red on a page that had
        done exactly what this page asks for. A fixture cannot be converted
        out from under a test.
        """
        found = line_citations(CITATION_FIXTURE)
        assert len(found) == CITATION_FIXTURE_EXPECTED, found
        joined = " ".join(found)
        # One assertion per widening, so a pattern that lost ONE of them fails
        # naming the form it lost.
        assert "foo.py:12" in joined
        assert "bar/baz.result:13-15" in joined           # any extension
        assert "sql_yacc.yy:9" in joined                  # a grammar
        assert "docker/percona/Dockerfile:725" in joined  # no extension
        assert ":701-707" in joined                       # bare continuation
        assert "r/x.result:3-5, :13-17, :23-26" in joined  # several ranges
        # And it must not fire on correct prose, which is the other failure
        # mode: a guard that reddens a correct document gets deleted.
        assert not line_citations(
            "`max(ef, k)` `p50/p95/max` `harness/main.py` `x86-64-v3`")

    def test_the_policy_is_stated_in_the_document(self):
        """The reason, not just the rule. A rule whose reason is only in a test
        file gets relaxed by the next author to find it inconvenient."""
        text = _plain(_doc())
        assert "#41" in text and "#43" in text, (
            "docs/11 §0 justifies the policy with the two pull requests that "
            "repaired the rot; without them it is a style preference")
        assert re.search(r"invalidated by (?:any|every) edit", text, re.I)
        assert re.search(r"only by a rename", text, re.I)

    def test_the_illustration_of_the_banned_form_is_not_in_code_font(self):
        """The one place docs/11 has to SHOW a line citation is the paragraph
        banning it, and the ban is a pattern over code-font spans. Italics
        rather than an allowlist: a carve-out in the ban is a carve-out, and
        the document would then be the one file permitted to make the citation
        it forbids."""
        assert re.search(r"\*some/file\.py:123\*", _doc()), (
            "docs/11 §0 must illustrate the banned form in italics; in code "
            "font it trips its own ban")


class TestNoCitationIntoAGeneratedTree:
    """`sources/`, `work/`, `results/` are generated and gitignored.

    A path under one of them resolves only on the machine that fetched it, so
    it is a citation nobody reviewing the document can check. docs/11 names
    vendor material in prose for this reason and §0 says so.
    """

    def test_no_cited_path_is_under_a_generated_tree(self):
        paths, _symbols, _prose = classify(_doc())
        bad = sorted(p for p in paths
                     if p.startswith(GENERATED_PREFIXES))
        assert not bad, (
            "docs/11 cites a path under a generated, gitignored tree, which "
            "does not exist on a fresh checkout: " + ", ".join(bad))

    def test_every_prefix_is_actually_gitignored(self):
        """The reason above, checked against `.gitignore` rather than against
        the filesystem.

        It was the filesystem for one revision, asserting that none of these
        directories exists in the checkout -- which passed this file in
        isolation and FAILED the full suite, because some other test creates
        `state/` while it runs. The assertion was true of a fresh worktree and
        false of a worktree that had run the suite once, which makes it a test
        of execution order rather than of anything about the document. What
        the document's claim actually rests on is that these paths are ignored,
        so a fresh clone does not have them; that is a fact about `.gitignore`
        and nothing else can move it.
        """
        ignore = open(os.path.join(VB_ROOT, ".gitignore")).read()
        rules = {line.strip() for line in ignore.splitlines()}
        for prefix in GENERATED_PREFIXES:
            assert "/" + prefix in rules, (
                f"{prefix} is treated here as generated and uncheckable, but "
                f".gitignore does not ignore it -- either the list is wrong or "
                f"a generated tree became a committed one")
        # Non-vacuity: a `rules` set that had come back empty, or a membership
        # test that always passed, would certify any prefix at all.
        assert "/sources/" in rules and "/nonexistent-zq/" not in rules

    def test_the_document_says_why_vendor_material_is_named_in_prose(self):
        text = _doc()
        assert re.search(r"generated and gitignored", text)
        assert re.search(r"named in prose", text, re.I)


class TestThePlantedBadReferenceFails:
    """The mutant arm: the pipeline, run against a document that is wrong.

    Verified by planting rather than assumed. A guard that has never been shown
    to fail is a guard nobody has tested, and this file's whole value is that
    it fires when a symbol is renamed away -- which cannot be demonstrated by
    the live document, because the live document is correct.
    """

    def test_an_absent_symbol_is_reported(self):
        bad_symbols, bad_paths = unresolved(
            "A sentence citing `wait_index_synced` and "
            "`no_such_symbol_exists_anywhere_zq`.")
        assert bad_symbols == ["no_such_symbol_exists_anywhere_zq"]
        assert bad_paths == []

    def test_an_absent_path_is_reported(self):
        bad_symbols, bad_paths = unresolved(
            "A sentence citing `harness/main.py` and "
            "`harness/workloads/no_such_file_zq.py`.")
        assert bad_paths == ["harness/workloads/no_such_file_zq.py"]
        assert bad_symbols == []

    def test_a_renamed_symbol_is_reported(self):
        """The failure mode the policy is FOR: the symbol was real and the
        rename happened. Spelled as a near-miss on a live name, because a
        plausible typo is what actually ships."""
        bad_symbols, _ = unresolved("`baseline_is_credable`")
        assert bad_symbols == ["baseline_is_credable"]

    def test_a_planted_line_citation_is_reported(self):
        """The same for the policy check, which is the only assertion here
        that is purely negative against the live document."""
        assert _LINE_CITATION_RE.findall(
            "as `harness/metrics/records.py:413` has it")

    def test_a_good_document_reports_nothing(self):
        """The paired positive. Without it, every assertion above would pass
        against a pipeline that reported everything as broken."""
        bad_symbols, bad_paths = unresolved(
            "`harness/main.py` `wait_index_synced` `OPS_MEASUREMENT_VERSION` "
            "`config/resources/tuned.yml` `PoolSpec`")
        assert bad_symbols == []
        assert bad_paths == []


# ---------------------------------------------------------------------------
# The claims the document must keep making
# ---------------------------------------------------------------------------
#
# A small set, and deliberately small. docs/10's `CLAIMS` table exists because
# that document is a runbook whose steps can be dropped; this one is a design
# document, and pinning its prose sentence by sentence would fire on every
# legitimate rewrite. What is pinned here is only what a reader would be
# MISLED by if it went missing: the provenance distinctions, the refutations
# leading rather than trailing, and the version-bump rule.

def _plain(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("`", "").replace("**", "")).strip()


class TestTheProvenanceDistinctionsSurvive:
    """The document's value is that it says which claims are measurements.

    Flatten these and the page becomes a list of assertions of uniform
    confidence, which is the thing it was written to avoid: three of the four
    hypotheses behind it were wrong, and the fourth is partly a queueing
    argument.
    """

    def test_the_provenance_table_names_every_kind_it_uses(self):
        text = _plain(_doc())
        for mark in ("checked", "studied", "reasoned", "supplied",
                     "unconfirmed"):
            assert mark in text, mark

    @pytest.mark.parametrize("mark,floor", [
        ("checked", 4), ("studied", 3), ("reasoned", 4), ("supplied", 4),
        ("unconfirmed", 4),
    ])
    def test_each_mark_is_used_and_not_merely_defined(self, mark, floor):
        """A provenance vocabulary defined in a table and used twice is
        decoration. The floors are low on purpose -- they catch a rewrite that
        drops a whole category, not one that re-words a paragraph.

        `studied` has the lowest floor and is the one most worth having: it is
        the mark for a finding this page reports and did not re-derive, and the
        temptation when writing a page like this is to let every such finding
        read as `checked`. A document that defines the mark and never applies
        it has done exactly that."""
        count = len(re.findall(mark, _plain(_doc()), re.I))
        assert count >= floor, f"{mark}: {count}"

    def test_studied_is_applied_and_not_only_defined(self):
        """`studied` is the mark most at risk of becoming decoration.

        It is down to ONE live application -- §2.4's warm-up invariance --
        because §7's two halves were reclassified to `checked` once the
        ann-benchmarks runner turned out to be readable in the documented
        read-only vendor input rather than only in a generated tree. That is
        the right direction to move a claim in, but it leaves the occurrence
        floor above being satisfied mostly by §1's table and the bullet that
        explains the table. So this asserts APPLICATION: the mark has to
        appear somewhere after §1, on an actual claim, or the vocabulary is
        defined and unused and the page's provenance scheme is theatre.
        """
        after_table = _doc().split("## 2.", 1)[1]
        assert re.search(r"\bstudied\b", _plain(after_table), re.I), (
            "no claim after §1 is marked `studied`; either a finding that was "
            "not re-derived has been mis-marked `checked`, or the mark should "
            "leave the provenance table")

    def test_the_unconfirmed_items_are_named_rather_than_counted(self):
        """The two engines whose compression support cannot be checked here
        are the test case: no vendor tree, so the claim has to be scoped."""
        text = _plain(_doc())
        assert re.search(r"unconfirmed", text, re.I)
        assert "valkey-search" in text
        assert re.search(r"Percona Server", text)

    def test_the_refutations_come_before_the_gaps(self):
        """Lead with the gap list and a reader misjudges the instrument. This
        is an ORDER claim, so it is checked as one rather than as a word."""
        text = _doc()
        refuted = text.index("What the instrument already does")
        tier1 = text.index("Tier 1 ")
        assert refuted < tier1, (refuted, tier1)

    def test_the_scoreboard_carries_every_verdict(self):
        """Five hypotheses, four verdicts between them, all present.

        Pinned because the whole ordering argument rests on it: drop the
        refuted rows and §2 becomes an apology instead of a correction, and a
        reader who sees only "confirmed" in a scoreboard will read §4 as the
        instrument's condition rather than as its bug list. The counting is
        pinned too -- five against four -- because an earlier draft of this
        page wrote one confirmed plus one half-refuted plus three refuted
        against four hypotheses, which is five of four.
        """
        text = _plain(_doc())
        for verdict in ("confirmed", "refuted", "half refuted"):
            assert verdict in text, verdict
        assert re.search(r"Five hypotheses, of which four", text), (
            "§2's scoreboard must state the counting; four verdicts over "
            "five hypotheses is the thing that gets garbled")
        # Non-vacuity for the row set: the scoreboard is a table, so count the
        # verdict cells rather than trusting that the words appear somewhere.
        rows = [line for line in _doc().splitlines()
                if line.startswith("|") and "**" in line]
        assert len(rows) >= 5, rows

    def test_the_only_size_figure_is_recomputed(self):
        """docs/11 prints one magnitude about the tree: how big `stats.py` is.

        Printed as a floor rather than as an exact line count, which is the
        same decision §0 makes about line citations for the same reason -- an
        exact 357 is invalidated by one added line, and this page would then be
        wrong about the module it holds up as evidence. A floor is invalidated
        only by the module shrinking, which is a real event worth failing on.
        """
        stats = os.path.join(VB_ROOT, "harness", "metrics", "stats.py")
        lines = len(open(stats).read().splitlines())
        assert lines >= 350, (
            f"harness/metrics/stats.py is now {lines} lines; docs/11 §2.1 "
            f"holds it up as over 350 and has to be re-worded")
        assert re.search(r"over 350\s+lines", _plain(_doc())), (
            "docs/11 §2.1 must state the floor this test recomputes")
        # Non-vacuity: an exact figure must not creep back in.
        assert "357 lines" not in _doc()

    def test_the_dispersion_refutation_names_the_raise_and_not_a_comment(self):
        """The strongest single piece of evidence that the dispersion
        hypothesis was wrong is that the refusal is an exception, not a note.
        A rewrite that softens it to "a comment forbids" inverts the finding."""
        text = _plain(_doc())
        assert re.search(r"raises\b.*ValueError", text)
        assert "NON_INDEPENDENT_PHASES" in _doc()
        assert not re.search(r"a comment forbids", text, re.I)


class TestTheVersionBumpRuleIsStatedOnceAndCompletely:
    """It governs almost every item, so both halves have to be present.

    Half of it is the trap: "additive, so no bump" is only safe if the reader
    also knows that a bump is TOTAL. A document that states the first without
    the second invites exactly the change that costs a 20-hour run its report.
    """

    def test_both_halves_of_the_rule_are_stated(self):
        text = _plain(_doc())
        assert re.search(
            r"absent or None becoming populated does not earn a bump", text)
        assert re.search(r"published value changing does earn one", text)

    def test_the_refusal_is_described_as_total(self):
        text = _plain(_doc())
        assert re.search(r"there is no report at all", text, re.I)
        assert "MixedMeasurementVersions" in _doc()
        assert re.search(r"returns 2", text)

    def test_the_rule_is_recomputable_from_the_modules_it_names(self):
        """The two symbols the rule hangs on, read back out of the tree rather
        than trusted to the document -- the one place where a symbol citation
        can be strengthened into a value check cheaply."""
        from harness.metrics import records
        from report import measurement_versions

        assert isinstance(records.OPS_MEASUREMENT_VERSION, int)
        assert records.IMPLICIT_OPS_MEASUREMENT_VERSION == 1
        assert issubclass(measurement_versions.MixedMeasurementVersions,
                          Exception)

    def test_the_open_loop_trap_is_named_with_the_column_it_would_break(self):
        """§6's one non-additive move. Named by the column, because "do not
        change published values" is advice and "do not write it into
        latency_p99_ms" is a rule."""
        assert "latency_p99_ms" in _doc()
        assert re.search(r"off by default", _plain(_doc()), re.I)


class TestTheTiersAreAllPresentAndOrdered:
    """A roadmap that loses a tier loses the sequencing argument with it."""

    TIERS: Sequence[str] = (
        "Tier 1 ", "Tier 2 ", "Tier 3 ", "Tier 4 ", "Tier 5 ", "Tier 6 ",
    )

    def test_every_tier_has_a_heading(self):
        text = _doc()
        for tier in self.TIERS:
            assert tier in text, tier

    def test_the_tiers_appear_in_order(self):
        text = _doc()
        positions = [text.index(t) for t in self.TIERS]
        assert positions == sorted(positions), positions

    def test_the_sequence_section_puts_the_blocking_item_first(self):
        """§4.2 blocks §5 because the new ratios are built on the same tuple.
        If the sequencing list stops saying so, the ordering is arbitrary."""
        text = _doc()
        sequencing = text.split("## 10. Sequencing", 1)[1]
        for item in ("4.1", "4.2", "4.3", "§5", "§6", "§9"):
            assert item in sequencing, item
        # The ordering claims, as orderings. §4.2 is inside the first bullet's
        # range and §5 is downstream of it; §9 is last because it depends on
        # both of the workloads above it.
        assert sequencing.index("4.1") < sequencing.index("§5")
        assert sequencing.index("§5") < sequencing.index("§9")

    def test_the_paper_is_framed_as_support_and_not_as_a_commitment(self):
        text = _plain(_doc())
        assert re.search(r"would support", text, re.I)
        assert re.search(r"Nothing below is a result", text, re.I)


# ---------------------------------------------------------------------------
# §4.1's evidence, checked against the tree rather than against its position
# ---------------------------------------------------------------------------
#
# WHY THIS CLASS EXISTS, and it is the same defect #63 fixed for the citation
# guards. Before it, the only thing anything asserted about §4.1 was that the
# string "4.1" appears in §10's sequencing list ahead of "§5" -- its POSITION.
# So the section could be rewritten freely and the suite stayed green:
# demonstrated, by rewriting six of its sentences, including its heading and
# both of its two search claims, and watching all 51 tests pass.
#
# That matters more here than for an ordinary stale sentence, because §4.1 does
# not merely describe a gap -- it offers EVIDENCE for one, a pair of greps
# reported to return nothing, and an implementation of §4.1 is exactly the
# event that makes that evidence false. A page whose stated purpose is to
# record what the instrument cannot see is at its least trustworthy when it
# claims a gap that has been closed.
#
# The guard computes from the tree whether a collection site exists, and that
# half is complete. What it checks in the DOCUMENT is far narrower: four fixed
# phrases (`PRESENT_TENSE_ABSENCE`), whether a site's file name appears, and
# whether `extra.server_status` is described with no collector behind it. So
# a collector added or removed is always noticed, but a present-tense gap
# claim is caught only if it uses one of those four exact strings -- reworded,
# it passes. That beats a one-sided check -- "the document names the
# collector" -- which would pass forever once written and would say nothing if
# the collector were removed; it is not a reading of the section.

#: The status variable §4.1 is about. Capitalised exactly as the server
#: registers it, because the document's claim is about a case-sensitive search.
ROADMAP_GAUGE = "Innodb_hnsw_memory_used"

#: The three directories §4.1's second search names, in its own order. A
#: status variable recorded anywhere else would not satisfy the claim the
#: document makes, which is specifically about these.
COLLECTOR_DIRS = ("harness", "report", "orchestrator")

#: Phrases that assert the gap in the PRESENT tense. The past-tense forms are
#: deliberately absent from this tuple: the corrected §4.1 still quotes what
#: the searches used to return, because that is how the gap was found, and a
#: guard that could not tell "returns nothing" from "returned nothing" would
#: force the document to delete its own evidence trail.
PRESENT_TENSE_ABSENCE = (
    "is collected nowhere",
    "returns no hit at all",
    "There is no collection site",
    "Nothing in it has been implemented",
)


def _collector_sites(needle: str = ROADMAP_GAUGE) -> List[str]:
    """Every non-test source file under §4.1's three directories naming `needle`.

    Case-sensitive, which is the claim's own terms. Tests are excluded: §4.1's
    first search covers every `.py` in the tree and explicitly discounts a
    docstring in `tests/test_e1_percona_design.py` as "not a collection site
    either", so a mention under `tests/` -- including in THIS file, which names
    the gauge in `ROADMAP_GAUGE` -- must not satisfy the document's claim. That
    is the same self-poisoning trap `SELF` exists for above, reached by a
    different route.
    """
    found = []
    for directory in COLLECTOR_DIRS:
        base = os.path.join(VB_ROOT, directory)
        for dirpath, _dirnames, filenames in os.walk(base):
            for name in sorted(filenames):
                if not name.endswith(".py"):
                    continue
                path = os.path.join(dirpath, name)
                try:
                    with open(path) as handle:
                        if needle in handle.read():
                            found.append(os.path.relpath(path, VB_ROOT))
                except OSError:
                    continue
    return sorted(found)


def stale_gap_claims(text: str, sites: Sequence[str]) -> List[str]:
    """What §4.1 claims that the tree contradicts, as a list of problems.

    A pure function of the document text and the measured facts, so the
    planted arms below can run it over a mutated copy without touching disk --
    the discipline `TestThePlantedBadReferenceFails` uses, and for the reason
    given there: a mutant written to a file can leave stale bytecode valid.

    WHAT THIS DOES NOT CHECK, stated because a reviewer is the terminus for
    the rest and that is only acceptable if the limit is written down. It
    does not read §4.1's collection claim or its two search claims as claims.
    It checks the tree for collection sites, and checks the document for four
    literal strings (`PRESENT_TENSE_ABSENCE`), for a site's file name, and for
    a description of `extra.server_status`. Stale-gap detection is ONLY those
    four literal strings: a present-tense gap claim worded any other way --
    "nothing collects it", "no code reads this gauge" -- passes silently. More
    phrases would not close that, so none are added; the limit is the
    statement. It does not check that §4.1's description of
    WHICH phases are collected is right (the words "build phase" could be
    wrong about `harness/workloads/` and this would pass), nor the node-map
    rider's 30-50 bytes, nor the claim about `docs/09`'s mentions, nor whether
    the named symbols are the ones doing the work rather than merely existing
    -- `TestEverySymbolTheDocumentNamesStillResolves` establishes only that
    they resolve somewhere. Those are judgements, and a human makes them.
    """
    problems = []
    plain = _plain(text)
    collected = bool(sites)

    for phrase in PRESENT_TENSE_ABSENCE:
        if _plain(phrase) in plain:
            if collected:
                problems.append(
                    f"the document still asserts {phrase!r} in the present "
                    f"tense, but {len(sites)} file(s) under "
                    f"{list(COLLECTOR_DIRS)} name {ROADMAP_GAUGE}: {sites}")
        elif not collected and phrase == "is collected nowhere":
            problems.append(
                f"no file under {list(COLLECTOR_DIRS)} names "
                f"{ROADMAP_GAUGE}, so the gap is open, but the document no "
                f"longer says {phrase!r}")

    if collected:
        # The document has to name where the collection happens, or a reader
        # cannot check the claim that replaced the gap. One real site is
        # enough; requiring all of them would fail on a refactor that merged
        # two modules, which is not a stale document.
        if not any(site in text for site in sites):
            problems.append(
                f"the gauge is collected in {sites} and the document names "
                f"none of those files")
    else:
        for site in ("harness/drivers/mysql_family.py",
                     "harness/workloads/build.py"):
            if f"{site}`'s `_server_status" in text or "extra.server_status" in text:
                problems.append(
                    f"the document describes a collector "
                    f"(`extra.server_status`) that no file under "
                    f"{list(COLLECTOR_DIRS)} implements")
                break
    return problems


class TestSection41AgreesWithTheTree:
    def test_the_live_document_makes_no_stale_gap_claim(self):
        sites = _collector_sites()
        problems = stale_gap_claims(_doc(), sites)
        print(f"collection sites = {sites}")
        assert problems == [], problems

    def test_the_gauge_really_is_collected_right_now(self):
        """Non-vacuity, and the arm that makes the class worth running. Every
        assertion above is conditional on `_collector_sites()`, so a finder
        that silently returned `[]` -- a renamed directory, a changed suffix
        filter -- would take the "gap is open" branch and pass against a
        document that no longer claims a gap. This pins the fact itself."""
        sites = _collector_sites()
        assert sites, (
            f"no file under {list(COLLECTOR_DIRS)} names {ROADMAP_GAUGE}; "
            f"either the collector was removed or this finder is broken")
        assert "harness/drivers/mysql_family.py" in sites

    def test_a_mention_under_tests_does_not_count_as_a_collection_site(self):
        """This file names the gauge. If `tests/` were indexed, §4.1 could
        claim a collector that exists only in the guard checking §4.1 --
        `SELF`'s bug, by another route."""
        assert _collector_sites() == _collector_sites(ROADMAP_GAUGE)
        assert not any(site.startswith("tests/")
                       for site in _collector_sites())

    @pytest.mark.parametrize("phrase", PRESENT_TENSE_ABSENCE)
    def test_a_planted_present_tense_gap_claim_is_reported(self, phrase):
        """The mutant, run-time and per phrase: re-assert the closed gap in
        the present tense and the guard must fire. Parametrised so that a
        phrase which has quietly stopped being detectable -- a reworded
        document, a `_plain` that stops normalising -- is named rather than
        hidden behind a sibling that still works."""
        planted = _doc() + f"\n\nAnd so {phrase}.\n"
        problems = stale_gap_claims(planted, _collector_sites())
        print(f"MUTANT ({phrase!r}) -> {problems}")
        assert problems, phrase

    def test_a_document_that_forgets_the_collection_site_is_reported(self):
        """The other direction: the gap is closed, the document says so, but
        it names no file a reader could check."""
        stripped = _doc()
        for site in _collector_sites():
            stripped = stripped.replace(site, "somewhere in the harness")
        problems = stale_gap_claims(stripped, _collector_sites())
        print(f"MUTANT (sites unnamed) -> {problems}")
        assert problems

    def test_the_checker_fires_the_other_way_on_a_tree_with_no_collector(self):
        """**The bidirectionality, which is the whole design.** Run the LIVE
        document against measured facts in which the collector does not exist
        -- the state this page described for its whole life before this work.
        The guard must complain that the document stopped claiming a gap that
        is open, which is what would happen if the collector were reverted and
        §4.1 were left as it now reads."""
        problems = stale_gap_claims(_doc(), [])
        print(f"MUTANT (no collector in tree) -> {problems}")
        assert problems
        assert any("the gap is open" in p for p in problems), problems

    def test_the_document_records_which_of_the_two_options_was_taken(self):
        """§4.1 offered two fixes -- collect it, or strike the claim from all
        four places. A reader has to be able to tell which happened, because
        the two leave the tree in opposite states."""
        plain = _plain(_doc())
        assert "the first was taken" in plain, (
            "§4.1 does not say which of its two options was taken")
        # And it must not overclaim: the build phase only.
        assert "at the build phase only" in plain, (
            "§4.1 does not scope the collection to the phase that was built")

    def test_the_opening_no_longer_claims_nothing_is_implemented(self):
        """The page's own first paragraph carried the blanket claim, which an
        implementation of any §4 item falsifies. It is listed in
        `PRESENT_TENSE_ABSENCE` and so is covered by the planted arm above;
        this asserts the live text directly, because that sentence is the one
        a reader meets first and the one `CLAUDE.md`'s documentation map
        repeats."""
        plain = _plain(_doc())
        assert "Nothing in it has been implemented" not in plain
        assert "One item has since been implemented" in plain
