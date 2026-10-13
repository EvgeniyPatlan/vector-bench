"""Cleanup C2: the Explore tab must be able to reach every phase a run writes.

`explore.js` narrows a non-raw view to `view.phase` and builds the Y picker by
filtering `view.ys`, so `webui/static/fields.js` decides what the tab can show
at all. It declared five views against seven phases: `ingest` had never had
one, and `ef_calibration` had none because T2b added the phase, the records
and the report section and stopped at the report. Both phases were therefore
unreachable -- the data was on disk, correct and invisible, and the only way
to it was the Raw view, which is the hand reconstruction the view list exists
to have already done.

Two things are pinned here, and the second is the one that was hard.

1. **Coverage, in the schema's own order.** The non-raw views' phases are
   checked against `records.ALL_PHASES` as an ordered tuple, so the next
   `PHASE_*` added to the schema fails here until it has a view, which is the
   defect this file closes rather than merely the two instances of it.

2. **Every offered axis resolves against records of that phase.** Membership
   is not the property that matters: a view exists is not the same claim as
   its metrics draw something. `available_measures` (`webui/records.py`) is
   computed over the WHOLE run, so an axis populated only by some other phase
   survives `explore.js`'s filter, reaches the dropdown and resolves to an
   empty chart. `TestAvailableMeasuresIsRunGlobal` below measures that hazard
   instead of asserting it, and the two allowlist classes close it by deriving
   the expected `ys` from records the production writers actually emit.

The records those two classes read are written by `harness/workloads/build.py`
and `harness/workloads/calibrate.py` themselves, driven by `FakeDriver` -- the
precedent is `tests/test_harness.py`'s own real-workload check. They are not
hand-built dicts: a fixture that listed the keys would be this file asserting
its own opinion of an ingest record twice.

THE CORPUS, named once because three different ones got mixed into one set of
figures while this was being written: every count quoted in this file is
measured on the archive of completed runs kept beside the repo, which holds
239 `ops-*.jsonl` files plus 41 merged `report/records.jsonl` and 7,284
records in all -- 2,986 recall_qps, 1,862 concurrency, 818 churn, 650
filtered, 484 ingest, 484 index_build. The archive agrees with the in-process
writers wherever it can be consulted: all 484 `ingest` records carry
`ingest_rows_per_s` and `ingest_wall_s` and no other `MEASURE_FIELDS` entry.
It sits one directory above the repo and is absent from a worktree, so nothing
here depends on it being present -- and it must not become the oracle for
these guards even where it is, because it predates fields the writers now
emit. `recall_at_k` on the concurrency view is the live example: 0 of the
1,862 archived concurrency records carry it, while
`harness/workloads/concurrency.py` writes it on every point it records. An
archive-backed version of these guards would call that view's first quality
axis dead. There is no archived `ef_calibration` record at all, which is the
same fact as the view having been missed.

The same defect was found one view over and closed in the same change. The
`build` view offered `build_cpu_s`, a `Record` column that nothing in
`harness/`, `report/`, `orchestrator/` or `webui/` ever assigns -- declared,
labelled, listed in `MEASURE_FIELDS`, offered as an axis, and carried by 0 of
the archive's 484 index_build records. It is gone from `MEASURE_FIELDS` and
from that view's `ys`; the schema column stays, because a field nothing writes
is a question about the schema rather than about a dropdown.
`TestNoViewOffersAnAxisNothingCanFill` generalises it: no view may offer a
name outside `MEASURE_FIELDS`, and no `MEASURE_FIELDS` entry may be one that
no file in the measurement trees ever mentions. Only `recall_qps` genuinely
cannot be produced in process -- it comes out of ann-benchmarks -- and the
concurrency, filtered and churn writers are ones this cleanup must not drive,
which `tests/test_workloads.py` already does. Either way, the pair of checks
above needs no records at all, and it is what caught `build_cpu_s`.

Two things about the `build` view are reported rather than enforced, because
enforcing them would mean changing a view on a judgement nobody has recorded:
`ingest_wall_s` is on 484 of 484 index_build records and is deliberately not
offered (the ingest view owns the load cost), and `peak_rss_bytes` is a
legitimate partial -- none of the 239 raw ops records, 238 of the 245 merged
ones, because `report/generate.py`'s `attach_peak_rss` fills it from the
`mem-*.jsonl` timeseries. So the build view gets the at-least-one-record form
of the allowlist check, and the two new views keep the closed-both-ways and
every-record forms.
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.drivers.base import IndexSpec  # noqa: E402
from harness.metrics import records as records_mod  # noqa: E402
from harness.metrics.records import Recorder  # noqa: E402
from harness.workloads import build as build_workload  # noqa: E402
from harness.workloads import calibrate as calibrate_workload  # noqa: E402
from harness.workloads.context import RunContext  # noqa: E402
from webui import records as webui_records  # noqa: E402

from .fake_driver import (K, FakeDriver, index_spec,  # noqa: E402
                          loaded_driver, tiny_dataset)

VB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Module constants and not inline paths, so a mutation harness can point the
#: readers below at a patched copy of either file at run time. The read itself
#: is deliberately the same mechanism `tests/test_config.py`'s
#: `TestTheWebUiCanPlotConcurrencyRecall` uses -- source text, split on the
#: view's own `id:`, regex per key -- rather than a second parser that could
#: disagree with it about what the file says.
FIELDS_JS = os.path.join(VB_ROOT, "webui", "static", "fields.js")
EXPLORE_JS = os.path.join(VB_ROOT, "webui", "static", "explore.js")
API_PY = os.path.join(VB_ROOT, "webui", "api.py")
APP_JS = os.path.join(VB_ROOT, "webui", "static", "app.js")

#: The `ef_construction` the calibration fixture builds under. 200 because
#: that is what every archived pgvector record carries, so the fixture is the
#: shape of the one engine that reports the field at all rather than a value
#: invented here.
SPEC_EF_CONSTRUCTION = 200

#: Trees a measure's value can legitimately be assigned in. Searched as source
#: text rather than through imports, because these names are populated in
#: several shapes -- keyword arguments to `Record(...)`, dict keys inside
#: `LatencyStats.as_dict`, an item assignment in `report/generate.py`'s memory
#: merge -- and the only question asked of them is whether anything in the
#: tree ever puts a number there at all.
#: `webui` is in here deliberately, and it is what makes the exclusion below
#: load-bearing rather than decorative. With only the three measurement trees
#: searched, excluding `webui/records.py` could never match anything, so one
#: of the two excludes did nothing while the comment claimed both mattered.
WRITER_TREES = ("harness", "report", "orchestrator", "webui")

#: Left out of that search on purpose: both of these NAME a measure rather
#: than populate it, and "declared in the schema, listed in the UI's plottable
#: tuple, written by nothing" is the exact state `build_cpu_s` was in. Both
#: now earn their place -- `webui/records.py` discusses `build_cpu_s` in the
#: note above `MEASURE_FIELDS`, so without this entry the field's own
#: postmortem would count as a writer and the guard would pass on it.
WRITER_SEARCH_EXCLUDES = (
    os.path.join("harness", "metrics", "records.py"),
    os.path.join("webui", "records.py"),
)

#: Every key lookup below is anchored on this. The shortest key read here is
#: `x`, so a bare `x:` is a substring of any identifier ending in x -- and
#: this is not hypothetical: `fields.js` did contain `ys:` inside the word
#: `stays:`, in the build view's own comment above that view's real `ys:` key.
#: It survived an unanchored read only by luck, because prose followed the
#: colon rather than a `[`. That sentence has since been reworded for other
#: reasons and the file now has no such collision, which is exactly why
#: `test_the_key_anchor_reads_past_a_colon_inside_a_word` tests the reader
#: rather than the prose.
#:
#: What it does NOT buy: `ys` standing alone as a word is not glued to
#: anything, so a comment reading `the ys: ["..."]` defeats this reader as
#: completely as an unanchored one. That is pinned in the same test rather
#: than left to be discovered, and it is the one sentence not to write in
#: `fields.js`.
_KEY_START = r"(?<![\w$.])"


# ---------------------------------------------------------------------------
# Readers. Each returns a value rather than asserting, so the mutation harness
# can print what moved instead of reporting a verdict.
# ---------------------------------------------------------------------------

def source(path: str) -> str:
    with open(path) as fh:
        return fh.read()


def view_chunk(view_id: str, text: Optional[str] = None) -> str:
    """The text of one view's own object literal.

    Scoped the way the existing test scopes it: everything from this view's
    `id:` up to the next one. A whole-file search for a field name is
    satisfied by any other view that happens to list it, which is how a
    per-view allowlist check passes without the view being touched.
    """
    text = source(FIELDS_JS) if text is None else text
    chunks = text.split('id: "')
    for chunk in chunks[1:]:
        if chunk.startswith(view_id + '"'):
            return chunk
    raise AssertionError(f"no view with id {view_id!r} in {FIELDS_JS}")


def view_array(key: str, chunk: str) -> Optional[List[str]]:
    """A view's `key: [...]`, or None where it is `null`."""
    found = re.search(_KEY_START + key + r":\s*(?:\[(.*?)\]|null)", chunk, re.S)
    assert found, f"no {key} in this view"
    if found.group(1) is None:
        return None
    return re.findall(r'"([a-z_0-9]+)"', found.group(1))


def view_flag(key: str, chunk: str) -> str:
    """A view's `key: true|false`, as written. Bare booleans, not strings."""
    found = re.search(_KEY_START + key + r":\s*(true|false)", chunk)
    assert found, f"no boolean {key} in this view"
    return found.group(1)


def view_string(key: str, chunk: str) -> Optional[str]:
    """A view's `key: "..."`, or None where it is `null`."""
    found = re.search(_KEY_START + key + r':\s*(?:"([^"]*)"|null)', chunk)
    assert found, f"no {key} in this view"
    return found.group(1)


def view_ids(text: Optional[str] = None) -> List[str]:
    text = source(FIELDS_JS) if text is None else text
    return [chunk.split('"')[0] for chunk in text.split('id: "')[1:]]


def view_phases(text: Optional[str] = None) -> List[Optional[str]]:
    """Every view's phase, in file order, `None` for the raw escape hatch."""
    text = source(FIELDS_JS) if text is None else text
    return [view_string("phase", view_chunk(vid, text)) for vid in view_ids(text)]


def field_labels(text: Optional[str] = None) -> Dict[str, str]:
    text = source(FIELDS_JS) if text is None else text
    block = re.search(r"const FIELD_LABEL = \{(.*?)\n\};", text, re.S)
    assert block, "FIELD_LABEL is not where fields.js keeps it"
    return dict(re.findall(r"(\w+):\s*\"([^\"]*)\"", block.group(1)))


def populated_measures(rows: Sequence[Dict[str, Any]]) -> List[str]:
    """The `MEASURE_FIELDS` every one of `rows` carries a number for.

    `Record.to_dict` drops None, so a measure a phase does not produce is
    absent from the record rather than null -- but both are checked, because
    the merged `report/records.jsonl` is written by a different writer and an
    explicit null there must not read as populated either.
    """
    assert rows, "populated_measures was handed no records to read"
    return [name for name in webui_records.MEASURE_FIELDS
            if all(isinstance(row.get(name), (int, float))
                   and not isinstance(row.get(name), bool) for row in rows)]


def writer_files(field: str,
                 excludes: Optional[Sequence[str]] = None) -> List[str]:
    """Files under `WRITER_TREES` that mention `field` at all.

    Deliberately a mention and not an assignment: a stricter pattern would
    have to understand `**stats.as_dict()`, and the weaker question is the one
    with teeth, because the field this exists to catch is mentioned in exactly
    none of these files.

    `excludes` is a parameter rather than only a constant so that
    `test_each_exclude_is_load_bearing` can ask what each one is actually
    keeping out. An exclusion nobody can see the effect of is a comment
    asserting a reason that may not exist.
    """
    skip = set(WRITER_SEARCH_EXCLUDES if excludes is None else excludes)
    found: List[str] = []
    for tree in WRITER_TREES:
        for dirpath, _dirs, names in os.walk(os.path.join(VB_ROOT, tree)):
            for name in sorted(names):
                if not name.endswith(".py"):
                    continue
                rel = os.path.relpath(os.path.join(dirpath, name), VB_ROOT)
                if rel in skip:
                    continue
                if field in source(os.path.join(VB_ROOT, rel)):
                    found.append(rel)
    return sorted(found)


def facet_coverage(rows: Sequence[Dict[str, Any]], field: str) -> int:
    """How many of `rows` carry a non-null value for `field`."""
    return sum(1 for row in rows if row.get(field) is not None)


def distinct(rows: Sequence[Dict[str, Any]], field: str) -> List[Any]:
    return sorted({row[field] for row in rows if row.get(field) is not None},
                  key=str)


# ---------------------------------------------------------------------------
# Record producers: the real writers, with a fake engine under them.
# ---------------------------------------------------------------------------

def _context(tmp, recorder: Recorder) -> RunContext:
    return RunContext(run_id="c2", resource_pass="normalized",
                      recorder=recorder, engine="fake", engine_tag="none",
                      march="x86-64-v3", k=K,
                      cache_dir=os.path.join(str(tmp), "cache"))


def ingest_records(tmp, m_values: Sequence[int] = (16, 32)) -> List[Dict[str, Any]]:
    """Real `PHASE_INGEST` records, one per M, from `build.py` itself.

    More than one M because `m` is the view's x axis and a single value draws
    one point: a check that the axis is carried would pass on a corpus that
    could never produce a line.
    """
    out = os.path.join(str(tmp), "ops-c2-build.jsonl")
    recorder = Recorder(out)
    try:
        ctx = _context(tmp, recorder)
        dataset = tiny_dataset()
        for m in m_values:
            driver = FakeDriver(dim=dataset.dim, metric=dataset.metric)
            driver.connect()
            build_workload.run(ctx, driver, dataset,
                               index_spec(dataset, m=m))
    finally:
        recorder.close()
    return [row for row in records_mod.read_records(out)
            if row["phase"] == records_mod.PHASE_INGEST]


#: Two samples per memory series, the larger second, so `peak_from_series`
#: has something to pick a maximum out of rather than a single value it
#: returns whatever it does with.
MEM_SAMPLES = (291_545_088, 308_809_728)


def index_build_records(tmp, m_values: Sequence[int] = (16, 32),
                        attach_memory: bool = True) -> List[Dict[str, Any]]:
    """Real `PHASE_BUILD` records, read back the way a report reads them.

    `peak_rss_bytes` is not a harness field at all. The orchestrator polls the
    server container's cgroup into `mem-<stem>.jsonl` beside the ops file, and
    `report/generate.py` attaches the maximum onto the index_build record --
    which is why the field is absent from all 247 archived raw ops records and
    present on 238 of 249 merged ones. So the fixture writes that timeseries
    in the shape the orchestrator writes it and performs the same join through
    `report.loaders`' public helpers.

    Which is the right level for this: the alternative is putting
    `peak_rss_bytes` into a dict by hand, and a fixture that supplies the
    thing under test proves only that the test can type. Here the input is
    what the orchestrator really produces, and the attach is `report`'s own
    `attach_peak_rss` -- called rather than reimplemented, so the join
    (`memory_stem` reconstructing the filename the orchestrator chose), the
    `peak_rss_source` tag and the "leave a harness-read value alone"
    early-return are all the real code's. `attach_memory=False` is what the
    mutation harness uses to show the axis going dead when the merge does not
    happen.
    """
    from report import generate, loaders

    out = os.path.join(str(tmp), "ops-fake-tiny-normalized-m.jsonl")
    recorder = Recorder(out)
    try:
        ctx = _context(tmp, recorder)
        dataset = tiny_dataset()
        for m in m_values:
            driver = FakeDriver(dim=dataset.dim, metric=dataset.metric)
            driver.connect()
            build_workload.run(ctx, driver, dataset, index_spec(dataset, m=m))
    finally:
        recorder.close()

    rows = [row for row in loaders.load_ops_records(str(tmp))
            if row["phase"] == records_mod.PHASE_BUILD]
    if not attach_memory:
        return rows

    for row in rows:
        stem = loaders.memory_stem(row)
        with open(os.path.join(str(tmp), f"mem-{stem}.jsonl"), "w") as fh:
            for value in MEM_SAMPLES:
                fh.write(json.dumps({"t": 0.0, "container": stem,
                                     "rss_bytes": value}) + "\n")
    peaks = {name: loaders.peak_from_series(series) for name, series
             in loaders.load_memory_series(str(tmp)).items()}
    generate.attach_peak_rss(rows, peaks)
    return rows


def calibration_records(tmp, ef_grid: Sequence[int] = (10, 16, 24)
                        ) -> List[Dict[str, Any]]:
    """Real `PHASE_CALIBRATION` records from `calibrate.py` itself.

    The target is set above anything `FakeDriver` reaches on this grid, so the
    sweep's early stop does not fire and every grid point is recorded. Recall
    here is a property of the fake's error model and is not asserted on; what
    is read off these records is which keys a calibration record carries.

    The spec sets `ef_construction`, which makes these the shape pgvector
    writes rather than the shape the MySQL family writes. That is the harder
    fixture on purpose: it is the only one on which "the view does not filter
    on `ef_construction`" is a judgement about a single-valued facet instead
    of a tautology about a field nothing populated.
    """
    out = os.path.join(str(tmp), "ops-c2-calib.jsonl")
    recorder = Recorder(out)
    try:
        ctx = _context(tmp, recorder)
        dataset = tiny_dataset()
        driver = loaded_driver(dataset)
        spec = IndexSpec(dim=dataset.dim, m=16, metric=dataset.metric,
                         ef_construction=SPEC_EF_CONSTRUCTION)
        calibrate_workload.run(ctx, driver, dataset, spec,
                               ef_grid=ef_grid, target_recall=0.999,
                               max_queries=20, warmup=2,
                               indexed_rows=len(dataset.train))
    finally:
        recorder.close()
    return [row for row in records_mod.read_records(out)
            if row["phase"] == records_mod.PHASE_CALIBRATION]


# ---------------------------------------------------------------------------
# 1. Coverage
# ---------------------------------------------------------------------------

class TestEveryPhaseHasAView:
    """A phase with no view cannot be reached, so coverage is the fix."""

    def test_the_non_raw_views_are_the_schemas_phases_in_its_own_order(self):
        """An ordered tuple and not a set membership check.

        Order because `ALL_PHASES` is the measurement pipeline's order and the
        picker is rendered in file order, and a tuple because it catches the
        two failures a set cannot: a phase listed twice, and a phase added to
        `records.py` with no view here. The latter is the actual defect --
        `ef_calibration` was exactly that -- so the guard has to be the
        general one rather than two spot checks for the two instances.
        """
        assert tuple(p for p in view_phases() if p is not None) == \
            records_mod.ALL_PHASES

    def test_both_new_phases_name_the_phase_constants_and_not_a_near_miss(self):
        """Read off the constants, so a typo'd phase string cannot pass.

        `"ef_calibration"` is also a value no other view carries, which the
        order check above would not notice if two views shared a phase.
        """
        by_phase = {view_string("phase", view_chunk(vid)): vid
                    for vid in view_ids()}
        assert by_phase[records_mod.PHASE_INGEST] == "ingest"
        assert by_phase[records_mod.PHASE_CALIBRATION] == "calibration"

    def test_only_the_raw_view_has_no_phase(self):
        assert [vid for vid in view_ids()
                if view_string("phase", view_chunk(vid)) is None] == ["raw"]

    def test_explore_narrows_a_non_raw_view_to_its_phase(self):
        """Why a missing view means unreachable, stated against the consumer.

        Anchored on the whole expression including its trailing comma: a
        substring of it would be satisfied by a longer line that filtered on
        something else.
        """
        assert "phaseFilter: { phase: [view.phase] }," in source(EXPLORE_JS)

    def test_the_picker_is_built_from_the_view_list_itself(self):
        assert "for (const view of VIEWS) {" in source(EXPLORE_JS)

    def test_the_key_anchor_reads_past_a_colon_inside_a_word(self):
        """Tested as a property of the reader, not as a word in the file.

        The collision that prompted the anchor was real and was in this file:
        the build view's comment read "...and stays: the harness never...",
        and `stays:` ends in `ys:`. It survived the unanchored form only by
        luck -- prose followed that colon, the full pattern wanted a `[` next
        and backtracked past it -- and it is gone now, removed in passing by
        an unrelated edit to that same sentence. Measured after that edit: 0
        key-position collisions in the file for any of the four keys read
        here.

        So pinning the word would pin prose, and the next one to appear would
        be a surprise rather than a caught bug. What is pinned instead is the
        behaviour under the dangerous form of that same collision: the colon
        inside a word, with a bracketed list after it instead of prose.

        And the limit is pinned beside it, because finding it is the only way
        to know what the anchor buys. A lookbehind rejects `ys` glued to a
        preceding letter; it cannot reject `ys` standing alone as a word, so a
        comment reading `the ys: ["..."]` defeats the anchored reader exactly
        as it defeats the bare one. Writing that in a comment in this file
        would break these tests, which is the only defence short of a real
        parser -- and is why the second half of this test exists rather than
        being left as a surprise for whoever writes that sentence.
        """
        real_ys = view_array("ys", view_chunk("build"))
        in_word = 'stays: ["decoy_metric"] and so'
        chunk = view_chunk("build").replace("is kept:", in_word)
        assert in_word in chunk
        # Position taken from the reader rather than a literal copy of the
        # allowlist, so this test does not fire on every edit to that view.
        assert chunk.index(in_word) < re.search(
            _KEY_START + r"ys:\s*\[", chunk).start()
        assert view_array("ys", chunk) == real_ys
        unanchored = re.search(r"ys:\s*(?:\[(.*?)\]|null)", chunk, re.S)
        assert re.findall(r'"([a-z_0-9]+)"', unanchored.group(1)) == [
            "decoy_metric"]

        standalone = 'kept, and the ys: ["decoy_metric"] below says why:'
        loose = view_chunk("build").replace("is kept:", standalone)
        assert view_array("ys", loose) == ["decoy_metric"]


# ---------------------------------------------------------------------------
# 2. The hazard the allowlists exist to avoid, measured
# ---------------------------------------------------------------------------

class TestAvailableMeasuresIsRunGlobal:
    """Why `explore.js`'s own filter cannot catch a dead axis.

    `webui/api.py` hands the browser `available_measures(recs)` over every
    record in the run, and `explore.js` filters each view's `ys` against that.
    So the filter answers "does anything in this run populate this field",
    never "does anything in this PHASE populate it". A view offering a field
    some other phase produces therefore passes every check the front end
    makes and still draws nothing -- which is the shape of bug that makes a
    populated run look empty.

    This is measured rather than asserted: the mixed list below is real ingest
    and real calibration records, and `qps` comes back available.
    """

    def test_a_measure_no_ingest_record_carries_is_still_offered(self, tmp_path):
        rows = ingest_records(tmp_path) + calibration_records(tmp_path)
        available = webui_records.available_measures(rows)
        assert "qps" in available
        ingest_only = [r for r in rows
                       if r["phase"] == records_mod.PHASE_INGEST]
        assert facet_coverage(ingest_only, "qps") == 0
        assert "qps" not in populated_measures(ingest_only)

    def test_the_chain_from_measure_fields_to_the_picker_is_unbroken(self):
        """Four files, read end to end, because the allowlists below are
        derived from `MEASURE_FIELDS` and that is only the right ceiling if
        `MEASURE_FIELDS` is what the picker ends up filtering against."""
        assert '"measures": records_mod.available_measures(recs),' in \
            source(API_PY)
        assert "S.measures = meta.measures;" in source(APP_JS)
        assert "ys: view.ys.filter((f) => S.measures.includes(f))," in \
            source(EXPLORE_JS)
        assert "ingest_rows_per_s" in webui_records.MEASURE_FIELDS


# ---------------------------------------------------------------------------
# 3. The universal form: no view may offer an axis nothing can fill
# ---------------------------------------------------------------------------

class TestNoViewOffersAnAxisNothingCanFill:
    """Every view, including the four whose records no test can produce.

    The per-phase checks further down are the strong form, and they can only
    reach the three phases this file drives a writer for. `recall_qps` comes
    out of ann-benchmarks and `concurrency`, `filtered` and `churn` out of
    workloads this cleanup must not drive, so a per-record rule would leave
    four views unguarded -- and it was the fifth, `index_build`, that turned
    out to be offering `build_cpu_s`.

    Two cheap claims cover all of them. A view may only name a field the UI
    can plot, and the UI's plottable list may only name a field something in
    the measurement trees mentions. Nothing that fails either can ever draw.
    """

    def test_no_view_offers_a_field_outside_the_plottable_list(self):
        for view_id in view_ids():
            offered = view_array("ys", view_chunk(view_id))
            if offered is None:          # the raw view takes the run's own
                continue
            for field in offered:
                assert field in webui_records.MEASURE_FIELDS, (view_id, field)

    def test_every_plottable_field_is_mentioned_by_some_writer(self):
        """Reported as the mapping, so a failure names the dead field rather
        than only its count."""
        counts = {field: len(writer_files(field))
                  for field in webui_records.MEASURE_FIELDS}
        assert [field for field, n in counts.items() if n == 0] == []

    def test_the_search_finds_a_writer_that_is_known_to_exist(self):
        """Non-vacuity in the direction that matters: a search that matched
        nothing anywhere would satisfy the test above only by also reporting
        zero, but one that matched EVERYTHING would satisfy it silently."""
        assert any(path.endswith(os.path.join("workloads", "build.py"))
                   for path in writer_files("ingest_rows_per_s"))
        assert any(path.endswith(os.path.join("metrics", "latency.py"))
                   for path in writer_files("latency_p95_ms"))

    def test_each_exclude_is_load_bearing(self):
        """Every exclusion must change the answer, or it is decoration.

        This is the guard on `WRITER_TREES` as much as on the excludes. The
        first version of this search covered only the three measurement trees,
        which made `webui/records.py` unexcludable-in-effect: dropping that
        entry gave an identical result, so one of the two excludes did nothing
        while the comment claimed both mattered. Adding `webui` to the trees
        is what repaired it -- `webui/records.py` carries the note explaining
        `build_cpu_s`, so without the exclude the field's own postmortem
        counts as a writer -- and this test is what stops the tree quietly
        going away again, because then there would be nothing for that exclude
        to keep out and this fails.
        """
        for excluded in WRITER_SEARCH_EXCLUDES:
            others = [e for e in WRITER_SEARCH_EXCLUDES if e != excluded]
            assert excluded in writer_files("build_cpu_s", excludes=others), \
                excluded

    def test_the_field_this_cleanup_removed_is_what_the_search_returns_empty(
            self):
        """`build_cpu_s` is the live control, and it is still a real name.

        It remains a `Record` column and still has a reader's label in
        `fields.js`, so this is not a search for a string that does not occur
        in the repo -- it occurs in both the places that name a measure
        without populating it, which is why those two are excluded. If some
        change ever starts writing build CPU time, this test fails and says
        so, and the field can go back into the plottable list in that change.
        """
        assert writer_files("build_cpu_s") == []
        assert "build_cpu_s" not in webui_records.MEASURE_FIELDS
        assert "build_cpu_s" in field_labels()
        for view_id in view_ids():
            offered = view_array("ys", view_chunk(view_id)) or []
            assert "build_cpu_s" not in offered, view_id


# ---------------------------------------------------------------------------
# 4. The build view, now that it can meet the record-backed form
# ---------------------------------------------------------------------------

class TestTheBuildViewsAxesResolve:
    """Widened to here because `build_cpu_s` is gone.

    The at-least-one-record form rather than the every-record form the two new
    views get, and that is a measurement and not a concession:
    `peak_rss_bytes` is on 0 of 247 archived raw ops records and 238 of 249
    merged ones, so no honest every-record rule can be written for this view.
    """

    @pytest.fixture(scope="class")
    def rows(self, tmp_path_factory):
        return index_build_records(tmp_path_factory.mktemp("c2-build"))

    def test_there_are_records_to_derive_from(self, rows):
        assert len(rows) == 2
        assert {r["phase"] for r in rows} == {records_mod.PHASE_BUILD}

    def test_every_offered_axis_is_populated_on_at_least_one_record(self, rows):
        for field in view_array("ys", view_chunk("build")):
            assert facet_coverage(rows, field) > 0, field

    def test_the_removed_axis_is_gone_and_could_not_have_been_filled(
            self, rows):
        """The defect, in the view it was found in."""
        assert "build_cpu_s" not in view_array("ys", view_chunk("build"))
        assert facet_coverage(rows, "build_cpu_s") == 0
        assert writer_files("build_cpu_s") == []

    def test_the_partial_axis_is_partial_and_not_dead(self, rows):
        """`peak_rss_bytes` tells the two cases apart.

        The harness writes it no more than it writes `build_cpu_s` -- the
        fixture's own records carry none until the memory series is joined on.
        What makes it an axis rather than a dead option is that something does
        populate it, so both halves are read here: zero without the merge, and
        a value on every record with it.
        """
        assert "peak_rss_bytes" in view_array("ys", view_chunk("build"))
        assert facet_coverage(rows, "peak_rss_bytes") == len(rows)
        assert {r["peak_rss_bytes"] for r in rows} == {max(MEM_SAMPLES)}
        assert any("generate.py" in path
                   for path in writer_files("peak_rss_bytes"))
        # The tag `attach_peak_rss` stamps, read back: it is what tells a
        # sampled value from one the harness read off the kernel, and reading
        # it is how this fixture proves the real attach ran rather than an
        # assignment of its own.
        for row in rows:
            assert row["extra"]["peak_rss_source"] == "sampled"

    def test_the_load_cost_is_carried_here_and_offered_by_the_ingest_view(
            self, rows):
        """The one omission from this view that is a judgement.

        Every index_build record carries `ingest_wall_s` -- 496 of 496
        archived -- because `build.py` writes the same seconds onto both
        records it emits. It belongs to the ingest view, and showing one
        number under two headings invites reading them as costs that add up.
        """
        assert facet_coverage(rows, "ingest_wall_s") == len(rows)
        assert "ingest_wall_s" not in view_array("ys", view_chunk("build"))
        assert "ingest_wall_s" in view_array("ys", view_chunk("ingest"))


# ---------------------------------------------------------------------------
# 5. The ingest view's axes against real ingest records
# ---------------------------------------------------------------------------

class TestTheIngestViewsAxesResolve:
    """Derived from what `build.py` writes, closed in both directions."""

    @pytest.fixture(scope="class")
    def rows(self, tmp_path_factory):
        return ingest_records(tmp_path_factory.mktemp("c2-ingest"))

    def test_there_are_records_to_derive_from(self, rows):
        """The guards below are all of the form "every offered axis is
        populated on every record", which is vacuously true of no records."""
        assert len(rows) == 2
        assert {r["phase"] for r in rows} == {records_mod.PHASE_INGEST}

    def test_the_allowlist_is_exactly_the_measures_an_ingest_record_carries(
            self, rows):
        """Closed both ways in one assertion.

        A superset is the dead-dropdown-entry defect; a subset is the
        unreachable-metric defect this file is fixing. Comparing sets rather
        than lists because the order is the default-axis question below, which
        is a separate claim about a separate consumer.
        """
        assert set(view_array("ys", view_chunk("ingest"))) == \
            set(populated_measures(rows))

    def test_every_offered_axis_has_a_value_on_every_record(self, rows):
        """Value and not membership. A key present and null renders the em
        dash that reads as a report bug, and `explore.js` plots it as a gap."""
        for field in view_array("ys", view_chunk("ingest")):
            assert facet_coverage(rows, field) == len(rows), field
            for row in rows:
                assert isinstance(row[field], float), (field, row["m"])

    def test_the_rate_is_the_default_axis(self, rows):
        """`explore.js` opens a view on `ys[0]`, so the order is the default."""
        assert view_array("ys", view_chunk("ingest"))[0] == "ingest_rows_per_s"

    def test_the_y_axis_defaults_to_log(self):
        """Pinned because a default that nothing reads can revert silently.

        The number behind it is not re-derived here -- it is a property of the
        corpus named in this module's docstring, not of a two-row fixture:
        across those 484 ingest records `ingest_rows_per_s` runs 38.9 to
        71,372.8 rows/s, a factor of 1,837, and `ingest_wall_s` spans 0.407 s
        to 25,479.3 s. On a linear axis the three slowest engines are a flat
        line on the x axis.

        The `recall` view is checked alongside, because the comment in
        `fields.js` cites it as the precedent and a cited precedent that turns
        out not to hold is the prose defect this series keeps finding.
        """
        assert view_flag("logY", view_chunk("ingest")) == "true"
        assert view_flag("logY", view_chunk("recall")) == "true"
        # And the ops views whose axes do not span orders of magnitude are
        # left linear, so this is a decision about this view and not a blanket
        # change to the tab.
        for view_id in ("build", "calibration", "concurrency", "filtered",
                        "churn"):
            assert view_flag("logY", view_chunk(view_id)) == "false", view_id

    def test_the_x_axis_is_a_facet_every_record_carries_and_more_than_one_of(
            self, rows):
        x = view_string("x", view_chunk("ingest"))
        assert x in webui_records.FACET_FIELDS
        assert facet_coverage(rows, x) == len(rows)
        assert len(distinct(rows, x)) > 1

    def test_every_filter_is_a_facet_an_ingest_record_carries(self, rows):
        """`explore.js` hides a facet with one value, so a foreign facet is
        invisible in an ingest-only run and appears the moment the run also
        holds the phase that does carry it -- a checkbox that narrows to
        nothing."""
        for field in view_array("filters", view_chunk("ingest")):
            assert field in webui_records.FACET_FIELDS, field
            assert facet_coverage(rows, field) == len(rows), field

    def test_each_offered_axis_has_a_readers_label(self):
        labels = field_labels()
        for field in view_array("ys", view_chunk("ingest")) + \
                [view_string("x", view_chunk("ingest"))]:
            assert field in labels, field
            assert labels[field] != field, field

    def test_the_omitted_measures_are_omitted_because_nothing_carries_them(
            self, rows):
        """Justifying the omissions, not only the inclusions.

        Build cost, index size and peak RSS are on the `index_build` record
        written beside this one, so offering them here would be the dead
        option this class exists to prevent.
        """
        offered = set(view_array("ys", view_chunk("ingest")))
        for field in ("build_wall_s", "build_cpu_s", "index_bytes",
                      "peak_rss_bytes", "qps", "recall_at_k"):
            assert field not in offered, field
            assert facet_coverage(rows, field) == 0, field

    def test_the_two_keys_left_out_are_not_plottable_at_all(self, rows):
        """`rows` and `ingest_threads` are on every ingest record and are in
        neither `MEASURE_FIELDS` nor `FACET_FIELDS`, so the front end has
        nowhere to put them: naming either would add an axis `explore.js`
        discards. Pinned so that adding one to `MEASURE_FIELDS` later shows up
        here as a decision rather than as a silent omission."""
        for field in ("rows", "ingest_threads"):
            assert facet_coverage(rows, field) == len(rows), field
            assert field not in webui_records.MEASURE_FIELDS, field
            assert field not in webui_records.FACET_FIELDS, field


# ---------------------------------------------------------------------------
# 6. The calibration view's axes against real calibration records
# ---------------------------------------------------------------------------

class TestTheCalibrationViewsAxesResolve:
    """Derived from what `calibrate.py` writes.

    No archived run in this project holds an `ef_calibration` record -- the
    phase is newer than every archive -- so the sweep is driven here and the
    keys are read off its output. That is the authoritative source either
    way: the archive would only ever be a copy of what this writer emits.
    """

    @pytest.fixture(scope="class")
    def rows(self, tmp_path_factory):
        return calibration_records(tmp_path_factory.mktemp("c2-calib"))

    def test_there_are_records_to_derive_from(self, rows):
        assert len(rows) == 3
        assert {r["phase"] for r in rows} == {records_mod.PHASE_CALIBRATION}

    def test_the_allowlist_is_exactly_the_measures_a_calibration_record_carries(
            self, rows):
        assert set(view_array("ys", view_chunk("calibration"))) == \
            set(populated_measures(rows))

    def test_every_offered_axis_has_a_value_on_every_record(self, rows):
        for field in view_array("ys", view_chunk("calibration")):
            assert facet_coverage(rows, field) == len(rows), field
            for row in rows:
                assert isinstance(row[field], float), (field, row["ef_search"])

    def test_recall_is_the_default_axis(self, rows):
        """The sweep exists to pick the ef that reaches a target recall, so
        recall is the question the tab should open on; `explore.js` takes
        `ys[0]`."""
        assert view_array("ys", view_chunk("calibration"))[0] == "recall_at_k"

    def test_the_x_axis_is_the_swept_knob_and_it_varies(self, rows):
        """`> 1` as well as one-value-per-record, and the pair is not
        redundant: a sweep of a single grid point satisfies "as many distinct
        ef values as records" at 1 == 1, which is a chart of one point. The
        mutation that shortens the grid to `(10,)` moves the distinct values
        from `[10, 16, 24]` to `[10]` and only this line reads it."""
        x = view_string("x", view_chunk("calibration"))
        assert x == "ef_search"
        assert x in webui_records.FACET_FIELDS
        assert facet_coverage(rows, x) == len(rows)
        assert len(distinct(rows, x)) > 1
        assert len(distinct(rows, x)) == len(rows)

    def test_every_filter_is_a_facet_a_calibration_record_carries(self, rows):
        for field in view_array("filters", view_chunk("calibration")):
            assert field in webui_records.FACET_FIELDS, field
            assert facet_coverage(rows, field) == len(rows), field

    def test_the_build_effort_is_not_offered_as_a_filter(self, rows):
        """The one omission that is a judgement rather than an absence.

        `ef_construction` is on the record and the recall view does filter on
        it, so leaving it out here needs saying. The whole sweep for one
        engine runs under one `IndexSpec`, so every ops record an engine
        writes carries the same value. Measured on the archive's 484 ingest
        records, written by the same ops path from the same spec: pgvector
        reports 200 on all 114 of its own and the other five engines report
        nothing on the remaining 370.

        A one-value facet is not then quietly dropped, which was the first and
        wrong reason given here. `S.facets` is run-global, so a run whose
        recall records sweep ef_construction hands the facet 64, 200 and 400,
        `length > 1` holds, and picking 64 narrows this view to zero records
        -- the shape of 33 of the 39 archived merged runs that record the
        field at all. The fixture below is the engine-reports-it shape, so
        this is a check that a single-valued facet was left out, not that an
        absent one was.
        """
        assert "ef_construction" not in view_array(
            "filters", view_chunk("calibration"))
        assert facet_coverage(rows, "ef_construction") == len(rows)
        assert distinct(rows, "ef_construction") == [SPEC_EF_CONSTRUCTION]

    def test_each_offered_axis_has_a_readers_label(self):
        labels = field_labels()
        for field in view_array("ys", view_chunk("calibration")) + \
                [view_string("x", view_chunk("calibration"))]:
            assert field in labels, field
            assert labels[field] != field, field

    def test_the_omitted_measures_are_omitted_because_nothing_carries_them(
            self, rows):
        """`index_bytes` is the trap: the recall and churn records carry it,
        so a run holding either makes it `available` and the offer would
        survive `explore.js` and draw nothing."""
        offered = set(view_array("ys", view_chunk("calibration")))
        for field in ("index_bytes", "peak_rss_bytes", "build_wall_s",
                      "build_cpu_s", "ingest_rows_per_s", "ingest_wall_s"):
            assert field not in offered, field
            assert facet_coverage(rows, field) == 0, field

    def test_the_mean_latency_on_the_record_is_not_plottable(self, rows):
        """`latency_mean_ms` is populated on every calibration record and is
        absent from `MEASURE_FIELDS`, so it is left out for the same reason
        `rows` is left out of the ingest view and not because the sweep does
        not measure it."""
        assert facet_coverage(rows, "latency_mean_ms") == len(rows)
        assert "latency_mean_ms" not in webui_records.MEASURE_FIELDS
        assert "latency_mean_ms" not in view_array("ys",
                                                   view_chunk("calibration"))

    def test_the_single_client_count_is_not_offered_as_a_filter(self, rows):
        """Every calibration point is a single-client probe, which is why the
        phase is not `recall_qps`.

        And the reason to leave the facet out is NOT that `explore.js` hides a
        one-value facet, which was the first answer here and is false:
        `S.facets` is run-global by the same chain as `S.measures`, so the
        run's concurrency records put 1, 2, 4, ... into the `clients` facet,
        `length > 1` holds, the checkbox renders, and picking anything but 1
        narrows this view to zero records. Measured: 32 of the archive's 41
        merged runs have a run-global `clients` facet with more than one value
        while the single-client phases carry only 1. The reason is the one the
        ingest view gives for `selectivity` -- a facet this phase does not
        vary is a checkbox that narrows to nothing as soon as the run holds a
        phase that does vary it.
        """
        assert distinct(rows, "clients") == [1]
        assert "clients" not in view_array("filters", view_chunk("calibration"))
