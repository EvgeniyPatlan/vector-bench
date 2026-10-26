"""E14 -- a run that executed both resource passes must be reported as one.

The defect. `Manifest.set_config` is called once per (pass, engine) and
rebuilds `config` from scratch every time, so `config.resource_pass`,
`config.resolved_resources` and `config.ann_fingerprint` all described
whichever pass was configured last. A run of `--resource-pass both` therefore
left a manifest that described a run of one pass, and every reader believed it.

Measured on `full-20261015-215006`, a `full` run that executed `normalized`
then `tuned`: 16 phases, 8 per pass, 41.9 h of measurement in total.

1. `_narrow_to_this_run` built one path out of that single pass name and
   fingerprint -- `annb/tuned/a9349676` -- so the loader never saw the
   normalized tree. Of 397 records in `report/records.jsonl`, none carried
   `phase: recall_qps` with `resource_pass: normalized`, while the ops records
   carried both passes. The three completed normalized ANN phases ran
   0.50 h + 4.80 h + 7.78 h = 13.1 hours, and were absent from the report.
2. §1 stated one machine -- cpuset `0-75`, 112.5 GiB -- above tables holding
   normalized rows measured with 8 CPUs and 16 GB.
3. §11 printed `--resource-pass tuned` as the command to reproduce a run that
   did neither pass alone.

Nothing crashed, nothing looked wrong, and 13.1 hours of recall measurement
went unpublished. That is the failure mode this repository exists to prevent,
so the assertions here are about what reaches the artifact: a tree the loader
was handed, a cell in the rendered table, a line in the rendered command.

Two things must not break, and each has its own class below:

* `TestASinglePassRunIsUnchanged` -- a run of one pass must produce the same
  narrowed tree, the same §1 rows and the same §11 command as before. Proven
  by asserting the values, and by the mutation note on each test saying what
  the honest and the mutant forms produce.
* `TestAnArchivedManifestIsNotReStamped` -- `set_config` is reached by
  `run --resume --run-id <old-run>` even when every unit is skipped, so the
  accumulation has to be additive. Writing today's limits into a directory
  measured under other ones is the provenance bug the long comment in
  `set_config` exists to describe.

No test here reads `results/`, which is gitignored: every run directory and
ann results tree below is built under `tmp_path`, the way `tests/test_webui*.py`
build theirs.
"""

from __future__ import annotations

import ast
import json
import os
import platform
import socket
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.metrics import records as records_mod  # noqa: E402
from orchestrator import manifest as manifest_mod  # noqa: E402
from orchestrator.manifest import Manifest  # noqa: E402
from report import charts, generate, loaders, render  # noqa: E402

#: Skips the three tests that draw. `report/charts.py` guards its matplotlib
#: import and the stack lives in the bench images, so a host built from the
#: documented prerequisites has no matplotlib -- and a test that fails there
#: makes a correct install look broken.
needs_matplotlib = pytest.mark.skipif(
    not charts.have_matplotlib(),
    reason="draws a chart; matplotlib lives in the bench images, not on the host")

VB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: The two passes and the fingerprints their declared knobs hash to. Distinct
#: strings on purpose: a tree keyed on the wrong pass's fingerprint is the
#: whole defect, and two identical fingerprints could not tell the two apart.
#: Distinguishes "caller passed nothing" from "caller passed None",
#: which for `ops_measurement_version` are different manifests: today's
#: constant against a run that predates the field.
_UNSET = object()

NORMALIZED_FP = "1f2e3d4c"
TUNED_FP = "a9349676"

#: Pass-level limits that differ the way the two passes differ: 8 CPUs / 16 GB
#: against 76 / 112.5 GiB. The cpuset, memory limit and build threads are the
#: real figures -- `config/resources/normalized.yml` declares `server_cpus: 8`,
#: `server_limit_gb: 16` and `threads: 1`, and `TUNED_LIMITS` is lifted whole
#: from `full-20261015-215006`'s own manifest. So a row carrying the wrong
#: pass's limits is recognisable on sight rather than merely unequal; the two
#: buffer and graph-cache figures are only kept mutually distinct, because a
#: fixture whose cells repeat cannot tell a correct table from a transposed one.
NORMALIZED_LIMITS = {
    "name": "normalized", "server_cpuset": "0-7",
    "server_memory_bytes": 17179869184, "buffer_bytes": 5153960755,
    "graph_cache_bytes": 4294967296, "build_threads": 1,
    "pass_signature": "normalized|0|0.3|0.3|0.1|0|4|0|1",
}
TUNED_LIMITS = {
    "name": "tuned", "server_cpuset": "0-75",
    "server_memory_bytes": 120819381043, "buffer_bytes": 42286783365,
    "graph_cache_bytes": 30204845260, "build_threads": 8,
    "pass_signature": "tuned|0|0.6|0.35|0.25|0.1|0|4|0|8",
}


# ---------------------------------------------------------------------------
# Fixtures: synthetic manifests and synthetic ann results trees
# ---------------------------------------------------------------------------

def _manifest(passes: Sequence[Tuple[str, Optional[str], Dict[str, Any]]],
              *, ran: Optional[Sequence[str]] = None,
              accumulate: bool = True, hostname: Optional[str] = None,
              version: Any = _UNSET,
              run_id: str = "full-20261015-215006") -> Dict[str, Any]:
    """A manifest describing a run of `passes`, in the order they ran.

    `accumulate=False` reconstructs the pre-change shape: the single-valued
    keys describe the LAST pass and there is no per-pass list, which is what
    every archived manifest looks like. That is the fixture the "before" half
    of each measured pair runs against, so the pair is a real difference in
    behaviour rather than two spellings of the same one.

    `ran` defaults to the passes themselves. Passing it separately is how a
    run whose phase log disagrees with its config block is built -- an
    archived both-pass run is exactly that.
    """
    entries = [{"resource_pass": name, "resolved_resources": limits,
                **({"ann_fingerprint": fp} if fp is not None else {})}
               for name, fp, limits in passes]
    last_name, last_fp, last_limits = passes[-1]
    config: Dict[str, Any] = {
        "profile": {"name": "full", "description": "d",
                    "ops": {"workloads": ["concurrency"]}},
        "resource_pass": last_name,
        "resolved_resources": last_limits,
        "ops_measurement_version": (records_mod.OPS_MEASUREMENT_VERSION
                                    if version is _UNSET else version),
    }
    if version is None:
        # A run older than the field. Absent, not null -- `report/loaders.py`
        # reads a missing key as "nothing is declared, check nothing".
        config.pop("ops_measurement_version")
    if last_fp is not None:
        config["ann_fingerprint"] = last_fp
    if accumulate:
        config["passes"] = entries
    names = list(ran) if ran is not None else [p[0] for p in passes]
    return {
        "run_id": run_id, "started_at": "2026-10-15T21:50:06Z",
        "finished_at": "2026-10-17T15:45:00Z", "status": "completed",
        "host": ({"cpu": {}} if hostname is None
                 else {"cpu": {}, "hostname": hostname}),
        "engines": {}, "warnings": [],
        # The corpus the records below carry, not a placeholder. `phases[]`
        # is one of the two things `report/scope.py` reads to decide which
        # corpora a run measured, so a phase log naming `d1` beside records
        # naming `glove-100-angular` describes a run that measured neither --
        # and the report then correctly excludes every record this file
        # builds. A fixture has to be self-consistent on an axis another
        # reader keys off.
        "phases": [{"phase": "ann", "engine": "percona",
                    "dataset": "glove-100-angular",
                    "status": "completed", "resource_pass": name}
                   for name in names],
        "config": config,
    }


BOTH_PASSES = ((("normalized", NORMALIZED_FP, NORMALIZED_LIMITS),
                ("tuned", TUNED_FP, TUNED_LIMITS)))
TUNED_ONLY = ((("tuned", TUNED_FP, TUNED_LIMITS),))


def _trees(tmp_path, *relative: str) -> str:
    """An annb root with the given `<pass>/<fingerprint>` trees present.

    Directories only. `load_annb_results` imports h5py as soon as the
    directory exists, and h5py lives in the bench images rather than on a host
    built from the documented prerequisites -- so the loader is captured
    below rather than run, and what is asserted is which directory it was
    handed. Creating the directories is what makes the `os.path.isdir` check
    under test a real check.
    """
    root = tmp_path / "results" / "annb"
    for part in relative:
        (root / part).mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    return str(root)


def _run_dir(tmp_path, manifest: Dict[str, Any],
             records: Sequence[Dict[str, Any]] = ()) -> str:
    """A run directory `generate.main` will accept, beside the annb root.

    The ops filename matters: `load_ops_records` relabels every record with
    the engine the filename names. The declared ops measurement version has to
    match the records' or the directory is refused before anything is read,
    so both come off the constant.
    """
    run_dir = tmp_path / "results" / manifest["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    version = records_mod.OPS_MEASUREMENT_VERSION
    stem = "ops-mariadb-glove-100-angular-normalized-m16-post.jsonl"
    with open(run_dir / stem, "w") as fh:
        for r in records or [_ops_record()]:
            extra = dict(r.get("extra") or {})
            extra[records_mod.OPS_MEASUREMENT_VERSION_KEY] = version
            fh.write(json.dumps({**r, "extra": extra}) + "\n")
    with open(run_dir / "run-manifest.json", "w") as fh:
        json.dump(manifest, fh)
    return str(run_dir)


def _ops_record(resource_pass: str = "normalized") -> Dict[str, Any]:
    return {"run_id": "full-20261015-215006", "engine": "mariadb",
            "phase": "concurrency", "dataset": "glove-100-angular", "m": 16,
            "build_mode": "post", "storage_engine": "InnoDB",
            "resource_pass": resource_pass, "k": 10, "clients": 8,
            "qps": 1234.0, "recall_at_k": 0.95, "ef_search": 100,
            "timestamp": "2026-10-16T00:00:00Z"}


#: The `source_file` BOTH trees produce for one build configuration, and the
#: single most important value in this file.
#:
#: `report/loaders.py` sets `source_file = os.path.relpath(path, annb_dir)`,
#: and `annb_dir` is the per-pass tree -- so the same configuration measured
#: in `normalized/<fp>/` and in `tuned/<fp>/` yields the SAME relative path,
#: hence the same `build_signature`, which omits the resource pass. Every
#: downstream grouping that keys on that signature therefore merged the two
#: passes into one.
#:
#: The first version of this fixture handed each tree `"1.hdf5"` and
#: `"2.hdf5"`, which made the signatures differ and the collision impossible
#: -- non-representative on precisely the axis under test, and the reason no
#: test in this file saw a defect that a real two-pass run hits on its normal
#: path. One literal, shared, is what makes the fixture honest.
COLLIDING_SOURCE_FILE = "glove-100-angular/10/percona/percona_M-16_ef-200.hdf5"


class _LoaderCapture:
    """Records every `(directory, resource_pass_default)` the loader is given.

    The subject of most assertions below. `_narrow_to_this_run` deciding
    correctly is not the property that matters -- the property that matters is
    that each tree it decided on is then actually read, and with the pass name
    the records will be labelled by.

    Its records carry `COLLIDING_SOURCE_FILE` and a rising (ef_search, qps,
    recall) ladder, so two trees produce exactly what a real two-pass run
    produces: colliding build signatures, and a normalized point that pairs
    against a tuned point at higher effort.
    """

    def __init__(self, per_tree: int = 1):
        self.calls: List[Tuple[str, str]] = []
        self.per_tree = per_tree

    def __call__(self, annb_dir, datasets_dir, run_id,
                 resource_pass_default="unknown"):
        self.calls.append((annb_dir, resource_pass_default))
        return [dict(_ops_record(resource_pass_default),
                     phase="recall_qps", engine="percona",
                     ef_search=10 * (i + 1), qps=300.0 - 100.0 * i,
                     recall_at_k=0.90 + 0.01 * i,
                     extra={"source_file": COLLIDING_SOURCE_FILE})
                for i in range(self.per_tree)]

    @property
    def dirs(self) -> List[str]:
        return [d for d, _ in self.calls]

    @property
    def passes(self) -> List[str]:
        return [p for _, p in self.calls]


def _run_report(tmp_path, manifest, annb_root, monkeypatch,
                per_tree: int = 1) -> Tuple[int, _LoaderCapture]:
    """`generate.main` end to end, with the ann loader captured.

    Everything else is the shipped code: the manifest is read off disk, the
    ops records go through `load_ops_records`, and the report is rendered and
    written.
    """
    capture = _LoaderCapture(per_tree=per_tree)
    monkeypatch.setattr(loaders, "load_annb_results", capture)
    run_dir = _run_dir(tmp_path, manifest)
    datasets = tmp_path / "datasets"
    datasets.mkdir(exist_ok=True)
    code = generate.main(["--run-dir", run_dir, "--annb-results", annb_root,
                          "--datasets-dir", str(datasets)])
    return code, capture


def _table_rows(markdown: str, header: Sequence[str]) -> List[List[str]]:
    """Cells of the Markdown table whose header row is exactly `header`.

    Parses the rendered artifact rather than recomputing it, and compares
    whole cells: `"0-7" in report` is satisfied by a table that says `0-75`,
    which is precisely the confusion under test here.
    """
    want = list(header)
    rows: List[List[str]] = []
    collecting = False
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            collecting = False
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if cells == want:
            collecting = True
            continue
        if collecting and set("".join(cells)) <= {"-"}:
            continue
        if collecting:
            rows.append(cells)
    return rows


def _render(manifest, records=()) -> str:
    records = list(records) or [_ops_record()]
    summary = generate.summarize(records, manifest)
    return render.render_markdown(manifest, summary, records, {}, "E14")


def _page(manifest, records=()) -> str:
    records = list(records) or [_ops_record()]
    summary = generate.summarize(records, manifest)
    return render.render_html(manifest, summary, records, {}, "E14")


ENVIRONMENT_HEADER = ["Property", "Value"]

#: §4's header, in order. No `Pass` column: the pass is in the table's
#: heading, because every figure in a row is derived across that table's
#: whole point set. See `render._headline_tables`.
HEADLINE_HEADER = ["Engine", "QPS @ recall≥0.90", "QPS @ recall≥0.95",
                   "QPS @ recall≥0.99", "Recall range",
                   "Points measured"]


def _env_cell(markdown: str, prop: str) -> List[str]:
    return [r[1] for r in _table_rows(markdown, ENVIRONMENT_HEADER)
            if r[0] == prop]


# ---------------------------------------------------------------------------
# The manifest shape
# ---------------------------------------------------------------------------

class TestTheManifestRecordsEveryPassThatWasConfigured:
    """One entry per pass, accumulated, beside the single-valued keys.

    `set_config` overwrote the single-valued keys on every call and that is
    left alone: archived runs and every current reader -- `webui/runs.py`,
    `orchestrator/export.py`, `cli.generate_report` -- read them. What is
    added is a list the multi-pass readers can use.
    """

    def _both_passes(self, tmp_path, engines=("mariadb", "alisql")):
        m = Manifest(str(tmp_path / "run-1"), "run-1")
        for name, limits in (("normalized", NORMALIZED_LIMITS),
                             ("tuned", TUNED_LIMITS)):
            fingerprint = NORMALIZED_FP if name == "normalized" else TUNED_FP
            for engine in engines:
                m.set_config({"name": "full"}, name,
                             dict(limits, engine_specific=engine),
                             extra={"ann_fingerprint": fingerprint})
        return m

    def test_a_run_of_both_passes_records_both(self, tmp_path):
        m = self._both_passes(tmp_path)
        entries = m.data["config"]["passes"]
        assert [e["resource_pass"] for e in entries] == ["normalized", "tuned"]
        assert [e["ann_fingerprint"] for e in entries] == [NORMALIZED_FP,
                                                           TUNED_FP]

    def test_each_entry_carries_its_own_pass_limits(self, tmp_path):
        """The 8-CPU/16 GB pass and the 76-CPU/112.5 GiB pass, side by side.

        Honest: `0-7` and `0-75`. Mutant -- have the accumulator record
        `prior["resolved_resources"]` instead of this call's -- both entries
        read `0-7`, because the block still describes the previous pass.
        """
        entries = {e["resource_pass"]: e["resolved_resources"]
                   for e in self._both_passes(tmp_path).data["config"]["passes"]}
        assert entries["normalized"]["server_cpuset"] == "0-7"
        assert entries["tuned"]["server_cpuset"] == "0-75"
        assert entries["normalized"]["server_memory_bytes"] == 17179869184
        assert entries["tuned"]["server_memory_bytes"] == 120819381043

    def test_the_single_valued_keys_are_left_exactly_where_they_are(
            self, tmp_path):
        """Unchanged, deliberately: last writer still wins there.

        Archived manifests have only these keys, and three live readers reach
        for them. Moving them would have made this fix a migration.
        """
        config = self._both_passes(tmp_path).data["config"]
        assert config["resource_pass"] == "tuned"
        assert config["ann_fingerprint"] == TUNED_FP
        assert config["resolved_resources"]["server_cpuset"] == "0-75"

    def test_seven_engines_across_two_passes_make_two_entries(self, tmp_path):
        """`set_config` runs per (pass, engine); `full` runs seven engines.

        Honest: 2. Mutant -- append unconditionally -- 14, and §1 would print
        the same two limit blocks seven times over.
        """
        m = self._both_passes(
            tmp_path, engines=("mariadb", "mariadb123", "alisql", "pgvector",
                               "mongodb", "valkey", "percona"))
        assert len(m.data["config"]["passes"]) == 2

    def test_an_absent_fingerprint_stays_absent(self, tmp_path):
        """A caller that computes no fingerprint records no fingerprint.

        `None` there would be indistinguishable from a tree keyed on the
        string "None". What the key's absence does is stop the report
        narrowing to a tree -- whether it then SAYS anything depends on
        whether the un-narrowed read picked the pass's recall up anyway; see
        `TestAPassAddedOnAnotherMachineGetsNoFingerprint`, which states both
        branches.
        """
        m = Manifest(str(tmp_path / "run-2"), "run-2")
        m.set_config({}, "normalized", NORMALIZED_LIMITS)
        entry = m.data["config"]["passes"][0]
        assert "ann_fingerprint" not in entry
        assert entry["resource_pass"] == "normalized"

    def test_extra_cannot_hand_write_the_list(self, tmp_path):
        """`extra` comes from a caller and describes one pass; the list is the
        run's whole history and is not a caller's to supply.

        What makes that hold is the unconditional assignment at
        `orchestrator/manifest.py:269-277`: `_accumulate_passes` reads `prior`
        -- what the directory already said -- rather than `config`, and its
        result overwrites whatever the splat put there. There is no pop, and
        this docstring used to say there was one, describing a mechanism the
        commit that added this test later removed as dead code. The
        `ops_measurement_version` pop one line above is NOT the same case: its
        follow-up assignment is conditional, so a smuggled value survives on a
        manifest that declares nothing, which is every archived run.
        """
        m = Manifest(str(tmp_path / "run-3"), "run-3")
        m.set_config({}, "tuned", TUNED_LIMITS,
                     extra={"ann_fingerprint": TUNED_FP,
                            "passes": [{"resource_pass": "fabricated"}]})
        assert [e["resource_pass"] for e in m.data["config"]["passes"]] == ["tuned"]

    def test_the_list_survives_a_reload_of_the_directory(self, tmp_path):
        """A second `Manifest` over the same run id continues that run.

        `cli.py` builds one per invocation, so the accumulation has to live in
        the file and not in the process.
        """
        self._both_passes(tmp_path)
        again = Manifest(str(tmp_path / "run-1"), "run-1")
        assert [e["resource_pass"] for e in again.data["config"]["passes"]] \
            == ["normalized", "tuned"]

    def test_the_file_on_disk_holds_it(self, tmp_path):
        """`set_config` saves, because a run that dies between populators used
        to leave no config block at all."""
        self._both_passes(tmp_path)
        with open(tmp_path / "run-1" / "run-manifest.json") as fh:
            on_disk = json.load(fh)
        assert len(on_disk["config"]["passes"]) == 2


class TestAnArchivedManifestIsNotReStamped:
    """`run --resume --run-id <old-run>` must not rewrite what it finds.

    `cli.py` calls `set_config` per (pass, engine) *above* the `[skip]`
    checkpoint test, so a resume reaches it even when every unit is skipped
    and nothing is measured. That is what makes the no-re-stamp rule load
    bearing rather than theoretical, and it is also what makes the recovery
    possible: a resume of a 42-hour run records both passes' fingerprints and
    measures nothing.
    """

    def _archived(self, tmp_path, **kw) -> str:
        """`full-20261015-215006` as it is on disk: tuned in the config block,
        both passes in the phase log, and no per-pass list.

        `hostname` defaults to this machine's, because the recovery case is a
        resume where the run was measured, and a fixture that silently looked
        foreign would make every assertion below about the wrong branch.
        """
        kw.setdefault("hostname", platform.node())
        run_dir = tmp_path / "full-20261015-215006"
        run_dir.mkdir(exist_ok=True)
        manifest = _manifest(BOTH_PASSES, accumulate=False, **kw)
        with open(run_dir / "run-manifest.json", "w") as fh:
            json.dump(manifest, fh)
        return str(run_dir)

    def _resume(self, tmp_path, elsewhere: Dict[str, Any], **kw):
        """A resume on a machine whose resolution differs from the archive's."""
        m = Manifest(self._archived(tmp_path, **kw), "full-20261015-215006")
        for name in ("normalized", "tuned"):
            m.set_config({"name": "full"}, name, elsewhere,
                         extra={"ann_fingerprint": "ffffffff"})
        return m

    def test_the_measured_pass_keeps_the_limits_it_was_measured_under(
            self, tmp_path):
        """Honest: tuned's entry reads `0-75` / 112.5 GiB, adopted from the
        block the original run wrote. Mutant -- let a later call overwrite an
        existing entry -- it reads `0-3` / 1 byte, which is this machine and
        not the one that spent 42 hours measuring.
        """
        m = self._resume(tmp_path, {"server_cpuset": "0-3",
                                    "server_memory_bytes": 1})
        entries = {e["resource_pass"]: e for e in m.data["config"]["passes"]}
        assert entries["tuned"]["resolved_resources"]["server_cpuset"] == "0-75"
        assert entries["tuned"]["resolved_resources"][
            "server_memory_bytes"] == 120819381043

    def test_the_measured_pass_keeps_the_fingerprint_it_was_keyed_by(
            self, tmp_path):
        """The tree on disk is `annb/tuned/a9349676`. An entry re-stamped with
        this machine's fingerprint would point the report at a tree that does
        not exist, and the 14.3 h of tuned recall measurement would join the
        13.1 h of normalized recall measurement in being silently absent."""
        m = self._resume(tmp_path, {"server_cpuset": "0-3"})
        entries = {e["resource_pass"]: e for e in m.data["config"]["passes"]}
        assert entries["tuned"]["ann_fingerprint"] == TUNED_FP
        assert entries["tuned"]["ann_fingerprint"] != "ffffffff"

    def test_the_pass_that_was_never_recorded_is_added(self, tmp_path):
        """The additive half. Adopting the block alone would leave the resume
        no better off than the report it is meant to fix."""
        m = self._resume(tmp_path, {"server_cpuset": "0-3"})
        named = [e["resource_pass"] for e in m.data["config"]["passes"]]
        assert sorted(named) == ["normalized", "tuned"]

    def test_a_second_resume_changes_nothing(self, tmp_path):
        """Idempotent, which is the operational form of "never re-stamped": a
        resume that is interrupted and repeated must converge."""
        first = self._resume(tmp_path, {"server_cpuset": "0-3"})
        snapshot = json.dumps(first.data["config"]["passes"], sort_keys=True)
        again = Manifest(first.run_dir, "full-20261015-215006")
        for name in ("tuned", "normalized"):
            again.set_config({"name": "full"}, name, {"server_cpuset": "9-9"},
                             extra={"ann_fingerprint": "00000000"})
        assert json.dumps(again.data["config"]["passes"],
                          sort_keys=True) == snapshot

    def test_an_older_declared_ops_version_is_carried_not_stamped(
            self, tmp_path):
        """The invariant this change had to be additive around. A resume that
        rewrote the version made the report refuse the directory forever.

        The archive declares `1`, not today's constant. This test asserted the
        constant against a fixture that already declared it, so carrying and
        stamping produced the same value and the assertion could not fail:
        replacing the carry-forward with an unconditional stamp left all 57
        tests passing (honest 57, mutant 57 -- dead). Declaring an older
        version is what makes the two outcomes different.
        """
        m = self._resume(tmp_path, {"server_cpuset": "0-3"}, version=1)
        assert m.data["config"]["ops_measurement_version"] == 1
        assert m.data["config"]["ops_measurement_version"] != \
            records_mod.OPS_MEASUREMENT_VERSION

    def test_an_absent_ops_version_stays_absent(self, tmp_path):
        """The other half, and the case the `pop` in `set_config` exists for:
        a run recorded before the key existed was measured under a version
        this code cannot know, and inventing today's constant would be a claim
        about a measurement nobody witnessed."""
        m = self._resume(tmp_path, {"server_cpuset": "0-3"}, version=None)
        assert "ops_measurement_version" not in m.data["config"]

    def test_the_resume_recovers_the_report_without_re_measuring(
            self, tmp_path, monkeypatch):
        """End to end: the resumed manifest now narrows to both trees.

        This is the whole point of the adoption rule. `full-20261015-215006`
        cost 42 hours; the normalized tree is still on the build host at
        `results/annb/normalized/<its own fingerprint>/`, and this is how the
        report reaches it.
        """
        m = self._resume(tmp_path, dict(NORMALIZED_LIMITS))
        # The fingerprint the resume recorded for the pass it did not measure
        # here is whatever this machine's knobs hash to -- "ffffffff" in this
        # fixture. The tree is keyed by the same function, so they agree.
        root = _trees(tmp_path, "tuned/" + TUNED_FP, "normalized/ffffffff")
        code, capture = _run_report(tmp_path, m.data, root, monkeypatch)
        assert code == 0
        assert sorted(capture.passes) == ["normalized", "tuned"]


class TestAPassAddedOnAnotherMachineGetsNoFingerprint:
    """The wrong-machine hole, which loudness alone does not close.

    A newly appended pass carries this machine's fingerprint, which is right
    when the resume runs where the run ran. When it does not, two branches
    diverge. If this machine has never measured that pass, the tree is absent,
    the report says "cannot be located", and nothing is published wrongly --
    that is the loud branch. If it HAS, `annb/<pass>/<its own fingerprint>/`
    exists and the report would merge **another machine's recall curve** as
    this run's, under a §1 block carrying that machine's limits. The staleness
    check does not catch it: that compares against the run's start, and this
    tree was measured after.
    """

    def _resumed_elsewhere(self, tmp_path, hostname):
        run_dir = tmp_path / "full-20261015-215006"
        run_dir.mkdir(exist_ok=True)
        manifest = _manifest(BOTH_PASSES, accumulate=False, hostname=hostname)
        with open(run_dir / "run-manifest.json", "w") as fh:
            json.dump(manifest, fh)
        m = Manifest(str(run_dir), "full-20261015-215006")
        for name in ("normalized", "tuned"):
            m.set_config({"name": "full"}, name, NORMALIZED_LIMITS,
                         extra={"ann_fingerprint": "ffffffff"})
        return {e["resource_pass"]: e for e in m.data["config"]["passes"]}

    def test_a_foreign_host_withholds_the_fingerprint(self, tmp_path):
        """Honest: `normalized` carries no `ann_fingerprint`. Mutant -- record
        it regardless of host -- the entry carries `ffffffff`, and a tree this
        machine happens to hold under that key is published as the archived
        run's recall under a §1 block carrying this machine's limits.

        **What withholding it does NOT do is make the report say the tree
        could not be located.** It closes the case where this host has never
        measured that pass: there is no directory, nothing is read, and §2
        names the pass. Where this host HAS measured it, withholding the
        fingerprint sends `_narrow_to_this_run` into its whole-tree fallback,
        which reads that very directory; §4 then labels those records with the
        pass name taken from the directory, the intersection in `summarize`
        sees recall for the pass and suppresses §2's notice, and nothing in the
        document says the tree was not this run's. That half is not closed --
        see the class docstring, which states both branches, and the backlog
        item on telling a narrowed read from a fallback one by the `source_file`
        prefix the records already carry.
        """
        entries = self._resumed_elsewhere(tmp_path, "bench-rig-2")
        assert "ann_fingerprint" not in entries["normalized"]
        assert entries["normalized"]["resource_pass"] == "normalized"

    def test_the_adopted_pass_keeps_its_own_fingerprint_anyway(self, tmp_path):
        """The withholding applies only to a pass this code derived. `tuned`
        was recorded by the run that measured it, so a foreign resume carries
        it forward untouched -- otherwise moving a run directory between
        machines would erase provenance it already had."""
        entries = self._resumed_elsewhere(tmp_path, "bench-rig-2")
        assert entries["tuned"]["ann_fingerprint"] == TUNED_FP

    def test_this_machine_records_it(self, tmp_path):
        """The case the recovery is for, and the other half of the pair: on
        the host that measured the run, the fingerprint is recorded."""
        entries = self._resumed_elsewhere(tmp_path, platform.node())
        assert entries["normalized"]["ann_fingerprint"] == "ffffffff"

    def test_both_host_spellings_are_accepted(self, tmp_path, monkeypatch):
        """`platform.node()` writes the field and `webui/runs.py` reads it
        with `socket.gethostname()`. On this box they are the same string, so
        deleting either arm of the check left all 77 tests passing -- the
        probe was dead for an environmental reason rather than a written one.
        Forcing them apart is what makes the second arm observable: with
        `socket.gethostname()` reporting something else entirely, a manifest
        recorded under `platform.node()` is still this machine's.
        """
        monkeypatch.setattr(manifest_mod.socket, "gethostname",
                            lambda: "something-else-entirely")
        entries = self._resumed_elsewhere(tmp_path, platform.node())
        assert entries["normalized"]["ann_fingerprint"] == "ffffffff"

    def test_a_host_matching_only_the_socket_spelling_is_accepted(
            self, tmp_path, monkeypatch):
        """The mirror image, which is the arm that would have been deleted:
        the manifest's host matches `socket.gethostname()` and not
        `platform.node()`."""
        monkeypatch.setattr(manifest_mod.platform, "node",
                            lambda: "something-else-entirely")
        entries = self._resumed_elsewhere(tmp_path, socket.gethostname())
        assert entries["normalized"]["ann_fingerprint"] == "ffffffff"

    def test_a_manifest_with_no_recorded_host_records_it(self, tmp_path):
        """`cli.py` calls `set_host` before `set_config`, so a fresh run
        always has a hostname by then. Treating an absent one as foreign would
        block the recovery for a manifest written by something that never
        called `set_host`, with no stated reason."""
        entries = self._resumed_elsewhere(tmp_path, None)
        assert entries["normalized"]["ann_fingerprint"] == "ffffffff"


class TestThePassOrderIsTheRunOrder:
    """§1 prints the passes in order, so the order has to be the run's.

    `_accumulate_passes` adopts the pass the config block carried before the
    one it is called for, and `cli.py` expands `both` to
    `["normalized", "tuned"]`. So resuming the archived manifest, whose
    carried pass is `tuned`, records `['tuned', 'normalized']` for a run that
    measured normalized first. No figure is misattributed -- every block is
    labelled -- but §1's order reads as the run order, and `phases[]` is the
    record of what that was.
    """

    def _resumed(self, tmp_path):
        run_dir = tmp_path / "full-20261015-215006"
        run_dir.mkdir(exist_ok=True)
        with open(run_dir / "run-manifest.json", "w") as fh:
            json.dump(_manifest(BOTH_PASSES, accumulate=False,
                                hostname=platform.node()), fh)
        m = Manifest(str(run_dir), "full-20261015-215006")
        for name, limits in (("normalized", NORMALIZED_LIMITS),
                             ("tuned", TUNED_LIMITS)):
            m.set_config({"name": "full"}, name, limits,
                         extra={"ann_fingerprint": NORMALIZED_FP
                                if name == "normalized" else TUNED_FP})
        return m.data

    def test_the_recorded_entry_order_really_is_the_other_one(self, tmp_path):
        """Non-vacuity: without this, a test asserting run order could be
        passing because the two orders happen to agree."""
        recorded = [e["resource_pass"]
                    for e in self._resumed(tmp_path)["config"]["passes"]]
        assert recorded == ["tuned", "normalized"]

    def test_the_reader_returns_them_in_phase_log_order(self, tmp_path):
        manifest = self._resumed(tmp_path)
        assert [name for name, _fp in render.recorded_passes(manifest)] == \
            ["normalized", "tuned"]

    def test_the_environment_section_lists_them_in_run_order(self, tmp_path):
        """Honest: `normalized, tuned`, normalized's block first. Mutant --
        return the entries in recorded order -- `tuned, normalized`, for a run
        whose phase log says normalized ran first.
        """
        report = _render(self._resumed(tmp_path))
        assert _env_cell(report, "Resource pass") == ["normalized, tuned"]
        labelled = [r[0] for r in _table_rows(report, ENVIRONMENT_HEADER)
                    if " — " in r[0]]
        assert labelled[0] == "normalized — server cpuset"
        assert labelled[5] == "tuned — server cpuset"

    def test_a_pass_the_phase_log_does_not_mention_keeps_its_place(
            self, tmp_path):
        """A pass configured and then never reached -- the run died before its
        first unit -- has no phases. It is still recorded, and it sorts after
        the ones that ran rather than being dropped or floated to the front.
        """
        manifest = _manifest(BOTH_PASSES, ran=["tuned"])
        assert [name for name, _fp in render.recorded_passes(manifest)] == \
            ["tuned", "normalized"]


class TestARunWhoseSecondPassNeverStarted:
    """Only the passes that reached `set_config` are claimed.

    A run killed, or failed, during its first pass configured one pass and
    measured one pass. Claiming two would be the original defect with the
    sign flipped: a report naming a machine nothing ran on.
    """

    def _interrupted(self, tmp_path) -> Manifest:
        m = Manifest(str(tmp_path / "run-stopped"), "run-stopped")
        m.set_config({"name": "full"}, "normalized", NORMALIZED_LIMITS,
                     extra={"ann_fingerprint": NORMALIZED_FP})
        return m

    def test_only_the_pass_that_started_is_recorded(self, tmp_path):
        entries = self._interrupted(tmp_path).data["config"]["passes"]
        assert [e["resource_pass"] for e in entries] == ["normalized"]

    def test_it_narrows_to_that_one_tree(self, tmp_path, monkeypatch):
        manifest = self._interrupted(tmp_path).data
        manifest["run_id"] = "run-stopped"
        manifest.setdefault("phases", []).append(
            {"phase": "ann", "resource_pass": "normalized",
             "status": "completed", "engine": "percona",
             "dataset": "glove-100-angular"})
        root = _trees(tmp_path, f"normalized/{NORMALIZED_FP}",
                      f"tuned/{TUNED_FP}")
        code, capture = _run_report(tmp_path, manifest, root, monkeypatch)
        assert code == 0
        assert capture.dirs == [os.path.join(root, "normalized",
                                             NORMALIZED_FP)]

    def test_the_sections_name_one_pass(self, tmp_path):
        """§1 and §11 describe the run that happened, not the one that was
        planned. `--resource-pass normalized`, not `both`."""
        manifest = _manifest(
            ((("normalized", NORMALIZED_FP, NORMALIZED_LIMITS)),),
            ran=["normalized"])
        report = _render(manifest)
        assert _env_cell(report, "Resource pass") == ["normalized"]
        assert "--resource-pass normalized\n" in report
        assert "--resource-pass both" not in report


# ---------------------------------------------------------------------------
# The trees the loader is handed
# ---------------------------------------------------------------------------

class TestEveryTreeThisRunWroteIsRead:

    def test_both_trees_are_read(self, tmp_path, monkeypatch):
        """Honest: two calls, `normalized/1f2e3d4c` and `tuned/a9349676`.
        Mutant -- return only the last recorded pass, which is what the
        single-valued keys yield -- one call, `tuned/a9349676`, and 13.1 hours
        of measurement never reach the report.
        """
        root = _trees(tmp_path, f"normalized/{NORMALIZED_FP}",
                      f"tuned/{TUNED_FP}")
        code, capture = _run_report(tmp_path, _manifest(BOTH_PASSES), root,
                                    monkeypatch)
        assert code == 0
        assert capture.dirs == [
            os.path.join(root, "normalized", NORMALIZED_FP),
            os.path.join(root, "tuned", TUNED_FP),
        ]

    def test_the_records_from_both_trees_reach_the_report(
            self, tmp_path, monkeypatch):
        """Not just read -- merged. The records file the web UI prefers over
        the raw ops records is written from this list."""
        root = _trees(tmp_path, f"normalized/{NORMALIZED_FP}",
                      f"tuned/{TUNED_FP}")
        code, _capture = _run_report(tmp_path, _manifest(BOTH_PASSES), root,
                                     monkeypatch, per_tree=3)
        assert code == 0
        merged = tmp_path / "results" / "full-20261015-215006" / "report" \
            / "records.jsonl"
        rows = [json.loads(line) for line in open(merged) if line.strip()]
        recall = [r for r in rows if r.get("phase") == "recall_qps"]
        assert len(recall) == 6
        assert sorted({r["resource_pass"] for r in recall}) == ["normalized",
                                                                "tuned"]

    def test_the_pass_name_travels_with_the_directory(
            self, tmp_path, monkeypatch):
        """Inside a narrowed tree `_pass_from_path` has nothing left to read:
        the relative path starts at the dataset, not at the pass. Without the
        default label every attribute-less result file in both trees would
        arrive as `unknown` and the two passes would merge into one nameless
        column.
        """
        root = _trees(tmp_path, f"normalized/{NORMALIZED_FP}",
                      f"tuned/{TUNED_FP}")
        _code, capture = _run_report(tmp_path, _manifest(BOTH_PASSES), root,
                                     monkeypatch)
        assert capture.passes == ["normalized", "tuned"]
        assert "unknown" not in capture.passes

    def test_a_tree_the_cli_already_narrowed_still_finds_the_other_pass(
            self, tmp_path, monkeypatch):
        """`cli.generate_report` narrows `--annb-results` on the host, where
        `ann_fingerprint` is importable, and it can only name one tree because
        it reads the single-valued keys. The whole results directory is
        mounted, so the other tree is two levels up.

        Honest: both trees. Mutant -- treat the argument as the root -- the
        loader is handed `.../tuned/a9349676/normalized/1f2e3d4c`, which does
        not exist, and the report falls back to reading the already-narrowed
        tuned tree while announcing a fallback that did not happen.
        """
        root = _trees(tmp_path, f"normalized/{NORMALIZED_FP}",
                      f"tuned/{TUNED_FP}")
        narrowed = os.path.join(root, "tuned", TUNED_FP)
        code, capture = _run_report(tmp_path, _manifest(BOTH_PASSES), narrowed,
                                    monkeypatch)
        assert code == 0
        assert sorted(capture.dirs) == sorted([
            os.path.join(root, "normalized", NORMALIZED_FP), narrowed])

    def test_a_genuine_root_is_never_mistaken_for_a_narrowed_path(
            self, tmp_path, monkeypatch):
        """A root is what is normally passed, and climbing out of it would
        read the results directory itself."""
        root = _trees(tmp_path, f"tuned/{TUNED_FP}")
        _code, capture = _run_report(tmp_path, _manifest(TUNED_ONLY), root,
                                     monkeypatch)
        assert capture.dirs == [os.path.join(root, "tuned", TUNED_FP)]

    def test_a_root_spelled_like_a_narrowed_tree_is_not_climbed_out_of(
            self, tmp_path, monkeypatch):
        """The first guard is on the path SPELLING, and a root can be spelled
        that way.

        `cli.py` never produces one, so this needs a hand-passed
        `--annb-results` — but `.../normalized/1f2e3d4c/` holding
        `normalized/` and `tuned/` satisfies the spelling test, and climbing
        out of it made the report declare a pass whose tree was right there
        absent. So the contents are consulted too: a real root has a child
        named for a pass, a narrowed tree's children are datasets.

        Honest: both trees under that root are read. Mutant -- climb on the
        spelling alone -- neither is found, and the report announces a
        fallback that did not happen.
        """
        root = tmp_path / "hand" / "normalized" / NORMALIZED_FP
        for part in (f"normalized/{NORMALIZED_FP}", f"tuned/{TUNED_FP}"):
            (root / part).mkdir(parents=True, exist_ok=True)
        code, capture = _run_report(tmp_path, _manifest(BOTH_PASSES),
                                    str(root), monkeypatch)
        assert code == 0
        assert sorted(capture.dirs) == sorted([
            str(root / "normalized" / NORMALIZED_FP),
            str(root / "tuned" / TUNED_FP)])


class TestOneMissingTreeDoesNotCostTheOthers:

    def test_the_present_tree_is_still_read(self, tmp_path, monkeypatch):
        """Honest: the tuned tree is read. Mutant -- fall back to the whole
        directory as soon as any tree is missing -- the loader is handed the
        annb root instead, which pulls in every configuration ever measured on
        the machine: the 16 GB curve under a 64 GB manifest, again.
        """
        root = _trees(tmp_path, f"tuned/{TUNED_FP}")
        code, capture = _run_report(tmp_path, _manifest(BOTH_PASSES), root,
                                    monkeypatch)
        assert code == 0
        assert capture.dirs == [os.path.join(root, "tuned", TUNED_FP)]

    def test_it_says_which_tree_is_missing(self, tmp_path, monkeypatch,
                                           capsys):
        """A silently shorter report is the defect, not the fix."""
        root = _trees(tmp_path, f"tuned/{TUNED_FP}")
        _run_report(tmp_path, _manifest(BOTH_PASSES), root, monkeypatch)
        out = capsys.readouterr().out
        assert f"normalized/{NORMALIZED_FP}" in out
        assert "absent from this report" in out

    def test_every_tree_missing_falls_back_to_the_whole_directory(
            self, tmp_path, monkeypatch):
        """Unchanged behaviour, and the only case that could reach it before.
        A manifest whose trees are all gone has nothing to narrow to, and the
        staleness check in Validity takes over from there.
        """
        root = _trees(tmp_path)
        code, capture = _run_report(tmp_path, _manifest(BOTH_PASSES), root,
                                    monkeypatch)
        assert code == 0
        assert capture.calls == [(root, "unknown")]

    def test_a_manifest_predating_the_fingerprint_reads_the_whole_tree(
            self, tmp_path, monkeypatch):
        """No fingerprint anywhere means no tree can be named, and the whole
        directory is the documented fallback for those runs."""
        manifest = _manifest(((("tuned", None, TUNED_LIMITS)),),
                             accumulate=False)
        root = _trees(tmp_path, f"tuned/{TUNED_FP}")
        _code, capture = _run_report(tmp_path, manifest, root, monkeypatch)
        assert capture.calls == [(root, "unknown")]


class TestAnArchivedManifestSaysWhatItCannotLocate:
    """The 13.1 hours, named rather than dropped.

    An archived both-pass manifest records one pass's fingerprint and both
    passes in `phases[]`. The other fingerprint was never written down and
    nothing in `report/` can compute it -- that needs `ann_fingerprint`, which
    lives in the orchestrator. Listing `annb/<pass>/*` and picking one would
    be worse than saying nothing: that tree is shared across runs, so any
    directory under it may belong to another configuration entirely.
    """

    def _archived_report(self, tmp_path, monkeypatch, capsys):
        root = _trees(tmp_path, f"normalized/{NORMALIZED_FP}",
                      f"tuned/{TUNED_FP}")
        manifest = _manifest(BOTH_PASSES, accumulate=False)
        code, capture = _run_report(tmp_path, manifest, root, monkeypatch)
        return code, capture, capsys.readouterr().out

    def test_it_still_reads_only_the_tree_it_can_name(
            self, tmp_path, monkeypatch, capsys):
        """Guessing is not an option, so the archived report is no larger than
        it was. What changes is that it stops being silent."""
        _code, capture, _out = self._archived_report(tmp_path, monkeypatch,
                                                     capsys)
        assert capture.passes == ["tuned"]

    def test_it_names_the_pass_whose_recall_is_absent(
            self, tmp_path, monkeypatch, capsys):
        """Honest: the output names `normalized`. Mutant -- derive the passes
        that ran from the config block instead of from `phases[]` -- the
        output says nothing at all, because the config block is exactly the
        thing that forgot.
        """
        _code, _capture, out = self._archived_report(tmp_path, monkeypatch,
                                                     capsys)
        assert "this run also executed normalized" in out
        # The narrowing claim, which is all this notice can know: it runs
        # before a record is read. §2 Validity makes the absence claim, and
        # only about passes whose recall the report really does not carry.
        assert "cannot narrow to" in out

    def test_it_says_how_to_recover_without_re_measuring(
            self, tmp_path, monkeypatch, capsys):
        """42 hours. The instruction has to be the cheap one."""
        _code, _capture, out = self._archived_report(tmp_path, monkeypatch,
                                                     capsys)
        assert "--resume --run-id full-20261015-215006" in out

    def test_it_warns_even_when_no_tree_is_on_this_machine_at_all(
            self, tmp_path, monkeypatch, capsys):
        """The state the real bundle is in: a manifest naming a pass whose
        tree is on the build host and not here.

        That branch returns the whole annb directory, and the warning used to
        sit BELOW the return, so rendering the bundle that motivated this
        change printed nothing. No test saw it because `_archived_report`
        always builds both trees. Honest: the pass is named. Mutant -- move
        the call back below the return -- silence.
        """
        root = _trees(tmp_path)
        manifest = _manifest(BOTH_PASSES, accumulate=False)
        code, _capture = _run_report(tmp_path, manifest, root, monkeypatch)
        out = capsys.readouterr().out
        assert code == 0
        assert "this run also executed normalized" in out

    def test_a_complete_manifest_says_nothing_of_the_kind(
            self, tmp_path, monkeypatch, capsys):
        """A warning that fires on healthy runs is a warning nobody reads."""
        root = _trees(tmp_path, f"normalized/{NORMALIZED_FP}",
                      f"tuned/{TUNED_FP}")
        _run_report(tmp_path, _manifest(BOTH_PASSES), root, monkeypatch)
        assert "also executed" not in capsys.readouterr().out


class TestTheArchivedRunsOwnReportSaysWhatItDid:
    """The exhibit, rendered. This is the test the first version lacked.

    `TestAnArchivedManifestSaysWhatItCannotLocate` above asserts only on the
    generator's stdout, and `TestARunWhoseSecondPassNeverStarted` covers a run
    that genuinely executed one pass -- a different state. Neither looked at
    the document. Measured on the first version of this fix: regenerating
    `full-20261015-215006`'s report before and after the commit produced a
    BYTE-IDENTICAL `report.md` and `report.html`. §1 still read
    `Resource pass | tuned`, cpuset `0-75`, 112.5 GiB above 52 rows whose pass
    cell was `normalized`; §11 still printed `--resource-pass tuned`; and
    nothing in either artefact mentioned the 13.1 hours. The only notice was a
    `print()` on stdout, which `./run-benchmark.sh export` does not bundle.

    None of what this class asserts needs a fingerprint. `phases[]` is enough
    to name a pass and to reproduce a run; only locating its results tree
    needs the hash, and that part stays honestly absent.
    """

    ARCHIVED = _manifest(BOTH_PASSES, accumulate=False)

    def test_section_1_names_both_passes(self):
        """Honest: `normalized, tuned`. Mutant -- derive the pass list from
        the config block -- `tuned`, above the normalized rows."""
        report = _render(self.ARCHIVED)
        assert _env_cell(report, "Resource pass") == ["normalized, tuned"]

    def test_section_1_does_not_lend_tuneds_limits_to_normalized(self):
        """The misattribution itself. `0-75` and 112.5 GiB belong to the pass
        the config block names and to no other.

        Honest: normalized's cells read `?` and `—`, because the manifest
        genuinely does not state them. Mutant -- fall back to the
        single-valued `resolved_resources` for a pass with no entry -- they
        read `0-75` and `112.5 GiB`, which is the published defect with a
        label on it, and worse than before because the label makes it look
        checked.
        """
        report = _render(self.ARCHIVED)
        assert _env_cell(report, "normalized — server cpuset") == ["?"]
        assert _env_cell(report, "normalized — server memory limit") == ["—"]
        assert _env_cell(report, "tuned — server cpuset") == ["0-75"]
        assert _env_cell(report, "tuned — server memory limit") == \
            ["112.5 GiB"]
        assert _env_cell(report, "Server cpuset") == []

    def test_section_11_reproduces_both_passes(self):
        """Honest: `--resource-pass both`. Mutant -- keep the single-valued
        key -- `--resource-pass tuned`, a command that does neither of the
        two things this run did."""
        report = _render(self.ARCHIVED)
        assert "--resource-pass both\n" in report
        assert "--resource-pass tuned" not in report

    def test_section_2_validity_names_what_is_missing(self):
        """The half that survives `export`, and the one the first version of
        this fix did not write. Asserted on the rendered document, not on
        stdout."""
        report = _render(self.ARCHIVED)
        assert "### Recall measurements this report could not locate" in report
        section = report.split(
            "### Recall measurements this report could not locate", 1)[1]
        section = section.split("\n### ", 1)[0]
        assert "`normalized`" in section
        assert "absent from every chart and table below" in section
        assert "full-20261015-215006" in section

    def test_it_reaches_the_html_page(self):
        """`report.html` is what a reader opens, and it is the file `export`
        bundles. A notice only in the Markdown would be a notice in the file
        nobody looks at."""
        page = _page(self.ARCHIVED)
        assert "Recall measurements this report could not locate" in page
        assert "--resource-pass both" in page
        assert "normalized, tuned" in page

    def test_the_notice_is_not_an_empty_heading(self):
        """A section that renders its heading and no content on a healthy run
        is the empty-chart problem with extra steps."""
        report = _render(_manifest(BOTH_PASSES))
        assert "Recall measurements this report could not locate" \
            not in report

    def test_the_notices_coincide_when_nothing_was_published(self):
        """The two notices answer different questions, and agree only here.

        stdout names every pass it could not narrow to -- it runs before a
        record is read, so that is all it can know. §2 names the subset whose
        recall this report does not carry, which is the intersection with the
        recall records. With no recall records the subset is the whole set and
        the two coincide, which is what this asserts.

        It used to be titled "the document and stdout cannot disagree" and to
        claim both read `passes_without_a_tree`. They do not: §2 reads the
        intersected `summary["passes_without_a_tree"]` and stdout reads the
        un-intersected `render.passes_without_a_tree(manifest)`. The test
        passed because this fixture has no recall records, so the intersection
        is a no-op -- and the claim it made in passing is exactly the belief
        that let §2 assert the absence of recall §4 was printing. See
        `generate.summarize`, which records the same thing from the production
        side.
        """
        summary = generate.summarize([_ops_record()], self.ARCHIVED)
        assert summary["passes_without_a_tree"] == \
            render.passes_without_a_tree(self.ARCHIVED) == ["normalized"]


class TestTwoPassesAreNotOneConfigurationMeasuredTwice:
    """Reading both trees hands two passes to layers built for one.

    `build_signature` takes the configuration from `extra.source_file`, which
    is the result file's path relative to the tree it came from -- and the
    trees are per pass. So one build configuration measured in both passes
    produces an identical signature. Before the report read both trees that
    state was reachable only on an error path; afterwards it is the normal
    path for exactly the run shape this change exists to support, and three
    places keyed on that signature were wrong at once:

    * §4 pooled two machines into one frontier, publishing the right QPS over
      a doubled sample count (Percona 24 -> 48, pgvector 18 -> 36) and a
      recall range spanned across both, with no column saying so.
    * §2 reported the run's own two passes as 123 duplicate configurations,
      under prose calling them "an older resource budget" and advising the
      reader to delete one of the trees.
    * §2 reported 97 QPS inversions, 71 past the first step, as
      "measurements of something other than the engine" -- each one a
      normalized point paired against a tuned point at higher effort.

    The fixture's `source_file` is shared between the trees for this reason;
    see `COLLIDING_SOURCE_FILE`.
    """

    def _summary(self, passes=("normalized", "tuned"), per_tree=3):
        """A record set shaped like what two trees actually return, with the
        manifest that describes the same run.

        The manifest has to match: `described_passes` is the union of what the
        manifest says ran and what the records carry, so handing a both-pass
        manifest to a single-pass case asserts the single-pass invariant
        against a two-pass run.
        """
        records = []
        for name in passes:
            records += [
                dict(_ops_record(name), phase="recall_qps", engine="percona",
                     ef_search=10 * (i + 1),
                     # FALLING with effort, which is what a correct sweep
                     # looks like: examining more candidates cannot be
                     # faster on one index. A rising ladder would make the
                     # fixture itself an inversion.
                     qps=(300.0 - 100.0 * i) * (2 if name == "tuned" else 1),
                     recall_at_k=0.90 + 0.01 * i,
                     extra={"source_file": COLLIDING_SOURCE_FILE})
                for i in range(per_tree)]
        manifest = _manifest(BOTH_PASSES if len(set(passes)) > 1
                             else ((passes[0],
                                    TUNED_FP if passes[0] == "tuned"
                                    else NORMALIZED_FP,
                                    TUNED_LIMITS if passes[0] == "tuned"
                                    else NORMALIZED_LIMITS),))
        return generate.summarize(records, manifest), records

    def test_the_fixture_really_does_collide(self):
        """Non-vacuity, and the assertion the old fixture would have failed:
        the two passes' records are indistinguishable to `build_signature`."""
        from report.loaders import build_signature
        _summary, records = self._summary(per_tree=1)
        signatures = {build_signature(r) for r in records}
        assert len(signatures) == 1, signatures
        assert len({r["resource_pass"] for r in records}) == 2

    def test_no_duplicate_configurations_are_reported(self):
        """Honest: none. Mutant -- drop the pass from `by_config`'s key --
        every configuration is reported as measured twice, and §2 tells the
        reader to delete one of their passes."""
        summary, _records = self._summary()
        assert summary["duplicate_ann"] == []

    def test_no_qps_inversions_are_reported(self):
        """Honest: none. The tuned ladder is uniformly faster at every
        ef_search, so within a pass throughput falls as effort rises, which is
        correct. Mutant -- drop the pass from `by_series`'s key -- the merged
        series pairs normalized against tuned and reports rising throughput as
        a contaminated measurement.
        """
        summary, _records = self._summary()
        assert summary["qps_inversions"] == []

    def test_a_genuine_duplicate_within_one_pass_is_still_caught(self):
        """The other half. Keying on the pass must not blind the check to what
        it exists for: one pass's tree holding two copies of a configuration.
        """
        summary, _records = self._summary(passes=("normalized", "normalized"))
        assert len(summary["duplicate_ann"]) == 3
        assert {d["resource_pass"] for d in summary["duplicate_ann"]} == \
            {"normalized"}

    def test_a_genuine_inversion_within_one_pass_is_still_caught(self):
        """And the same for inversions: throughput rising with effort inside
        one pass is still the machine rather than the engine."""
        records = [
            dict(_ops_record("tuned"), phase="recall_qps", engine="percona",
                 ef_search=ef, qps=qps, recall_at_k=0.95,
                 extra={"source_file": COLLIDING_SOURCE_FILE})
            for ef, qps in ((10, 2.6), (20, 23.3))]
        summary = generate.summarize(records, _manifest(BOTH_PASSES))
        assert len(summary["qps_inversions"]) == 1

    def test_section_4_keeps_each_passs_sample_count_separate(self):
        """The published figure that was wrong: `Points measured`. Honest: 3
        and 3. Mutant -- pool the passes -- one row reading 6, a count no
        machine produced."""
        summary, _records = self._summary()
        by_pass = summary["per_dataset"]["glove-100-angular"]
        assert sorted(by_pass) == ["normalized", "tuned"]
        assert by_pass["normalized"]["percona"]["points"] == 3
        assert by_pass["tuned"]["percona"]["points"] == 3

    def test_section_4_frontiers_are_per_pass(self):
        """A frontier across two machines takes the faster of them at every
        recall and draws a curve no configuration produced. The tuned ladder
        is 2x the normalized one here, so a pooled frontier would report
        tuned's throughput under both."""
        summary, _records = self._summary()
        by_pass = summary["per_dataset"]["glove-100-angular"]
        normalized = by_pass["normalized"]["percona"]["qps_at_recall_90"]
        tuned = by_pass["tuned"]["percona"]["qps_at_recall_90"]
        assert normalized == 300.0
        assert tuned == 600.0

    def test_section_4_renders_one_table_per_pass(self):
        """The pass is in the heading, not in a column -- see
        `_headline_tables`' docstring for why §4 differs from §7-§10."""
        summary, records = self._summary()
        report = render.render_markdown(_manifest(BOTH_PASSES), summary,
                                        records, {}, "E14")
        assert "#### glove-100-angular — normalized" in report
        assert "#### glove-100-angular — tuned" in report
        rows = _table_rows(report, HEADLINE_HEADER)
        assert [r[-1] for r in rows] == ["3", "3"]
        assert "Pass" not in HEADLINE_HEADER

    def test_a_single_pass_run_keeps_the_bare_heading(self):
        """Published single-pass reports must regenerate identically, and the
        heading is the part this change could have disturbed."""
        summary, records = self._summary(passes=("tuned",))
        report = render.render_markdown(_manifest(TUNED_ONLY), summary,
                                        records, {}, "E14")
        assert summary["described_passes"] == ["tuned"]
        assert "#### glove-100-angular\n" in report
        assert "— tuned" not in report

    def test_the_floor_bar_chart_is_emitted_once_per_pass(self, tmp_path):
        """`qps_at_recall` compares engines at ONE budget, so two passes on
        one axis would be fourteen bars per floor paired by legend. Entry
        names, not a drawn chart, so this runs without matplotlib."""
        summary, _records = self._summary()
        drawn = []
        entries = generate.qps_at_recall_entries(
            summary, "glove-100-angular", draw=lambda *a, **kw: drawn.append(kw))
        assert [name for name, _fn, _payload in entries] == [
            "qpsatrecall-normalized", "qpsatrecall-tuned"]
        for _name, fn, payload in entries:
            fn(*payload, str(tmp_path), "stem")
        assert [kw["resource_pass"] for kw in drawn] == ["normalized", "tuned"]

    def test_a_single_pass_run_keeps_the_one_chart_and_its_stem(self):
        """`qpsatrecall-<dataset>` is the filename every published report
        carries; a per-pass stem on a one-pass run would rename it."""
        summary, _records = self._summary(passes=("tuned",))
        entries = generate.qps_at_recall_entries(summary, "glove-100-angular",
                                                 draw=lambda *a, **kw: None)
        assert [name for name, _fn, _payload in entries] == ["qpsatrecall"]

    def test_the_chart_is_handed_one_group_per_pass(self, tmp_path,
                                                    monkeypatch):
        """The other half of the Pareto fix, and the half
        `pareto_series_style` cannot see: `generate.main` has to GROUP by
        (engine, pass) before the chart is called. Styling correctly off a key
        that was never split is no fix at all -- and a mutant keying the
        grouping by engine alone left every style assertion above passing.

        `charts.pareto` is captured rather than drawn, so this says nothing
        about matplotlib. The records come from the real two-tree read, so the
        keys are the ones a two-pass run produces.
        """
        keys = []

        def capture(records_by_engine, *a, **kw):
            keys.append(sorted(records_by_engine, key=str))
            return None

        monkeypatch.setattr(charts, "pareto", capture)
        root = _trees(tmp_path, f"normalized/{NORMALIZED_FP}",
                      f"tuned/{TUNED_FP}")
        code, _capture = _run_report(tmp_path, _manifest(BOTH_PASSES), root,
                                     monkeypatch, per_tree=3)
        assert code == 0
        assert keys, "charts.pareto was never called"
        # Once for `pareto`, once for `paretozoom`; both get the split keys.
        for seen in keys:
            assert seen == [("percona", "normalized"), ("percona", "tuned")], \
                seen

    def test_the_pareto_legend_names_each_pass(self):
        """Honest: two labels, each naming its pass. Mutant -- key the chart
        by engine alone -- one label and one frontier drawn through two
        machines.

        Asserted on `pareto_series_style`, which is where the decision lives,
        rather than on a drawn figure: a test that has to draw can only check
        that the call returned, and that is exactly why two mutants of this
        logic left the suite green.
        """
        from report import charts
        order = ["normalized", "tuned"]
        labels = [charts.pareto_series_style(("percona", p), order)[1]
                  for p in order]
        assert labels == [f"{charts.style_for('percona')['label']} / normalized",
                          f"{charts.style_for('percona')['label']} / tuned"]

    def test_the_two_passes_of_one_engine_are_drawn_differently(self):
        """Colour and marker are the engine's, so the pass has to take the
        linestyle or the two lines are indistinguishable without reading the
        legend. Honest: different linestyles, same colour and marker. Mutant
        -- do not vary the linestyle -- all three are equal and the chart has
        two identical lines.
        """
        from report import charts
        order = ["normalized", "tuned"]
        first, _ = charts.pareto_series_style(("percona", "normalized"), order)
        second, _ = charts.pareto_series_style(("percona", "tuned"), order)
        assert first["linestyle"] != second["linestyle"]
        assert first["color"] == second["color"]
        assert first["marker"] == second["marker"]

    def test_a_single_pass_chart_is_styled_and_labelled_as_before(self):
        """The constraint: one pass means the engine's own style and a bare
        label, byte-identical to what this chart drew before the key grew."""
        from report import charts
        plain = charts.style_for("percona")
        style, label = charts.pareto_series_style(("percona", "tuned"),
                                                  ["tuned"])
        assert label == plain["label"]
        assert style["linestyle"] == plain["linestyle"]

    def test_a_bare_engine_key_is_still_accepted(self):
        """`--from-records` over an archive predating the field has no pass to
        offer, and must not be required to invent one."""
        from report import charts
        style, label = charts.pareto_series_style("percona", [])
        assert label == charts.style_for("percona")["label"]
        assert charts.pareto_engine_of("percona") == "percona"
        assert charts.pareto_pass_of("percona") is None


class TestValidityNeverDeniesRecallTheReportPublishes:
    """§2's absence claim has to be about what the report actually carries.

    §2 says, in bold, that a named pass's "recall and throughput measurements
    are absent from every chart and table below". That was derived from the
    manifest alone -- which knows only that a tree could not be located, not
    what the report ended up reading. Both whole-tree fallbacks in
    `_narrow_to_this_run` hand the loader the un-narrowed root, and
    `_pass_from_path` then recovers the real pass from the directory layout,
    so §4 published the very recall §2 denied.

    Worse than the original defect, because Validity vouched for the
    misreading: a reader who believes §2 reads a normalized table as tuned's
    and attributes 8-core throughput to the 76-core machine.

    Two states, both reached without any hand-passed flag:

    * the recorded tree is not on this machine (`if not trees`), which is
      every copied or archived bundle;
    * the manifest predates `ann_fingerprint` entirely (`if not named`),
      which is the legacy class `_narrow_to_this_run`'s own docstring
      provides for.
    """

    #: What the un-narrowed read returns: real pass labels, recovered from the
    #: directory layout by `_pass_from_path` rather than from the manifest.
    def _published(self, resource_pass="normalized", count=3):
        return [dict(_ops_record(resource_pass), phase="recall_qps",
                     engine="percona", ef_search=10 * (i + 1),
                     qps=300.0 - 100.0 * i, recall_at_k=0.90 + 0.01 * i,
                     extra={"source_file": COLLIDING_SOURCE_FILE})
                for i in range(count)]

    def _rendered(self, *, fingerprint, records):
        manifest = _manifest(BOTH_PASSES, accumulate=False)
        if not fingerprint:
            manifest["config"].pop("ann_fingerprint")
        summary = generate.summarize(records, manifest)
        return summary, render.render_markdown(manifest, summary, records,
                                               {}, "E14")

    def test_a_published_pass_is_not_declared_absent(self):
        """State one, and the CRITICAL. Honest: §2 is silent, because
        normalized's recall is in §4. Mutant -- derive the list from the
        manifest alone -- §2 states that normalized's recall is absent from
        every chart and table, directly above the table printing it.
        """
        summary, report = self._rendered(fingerprint=True,
                                         records=self._published())
        assert summary["passes_without_a_tree"] == []
        assert "Recall measurements this report could not locate" not in report
        # And the recall really is published, so the absence of the notice is
        # not the absence of the data.
        assert "#### glove-100-angular — normalized" in report

    def test_the_legacy_state_names_only_what_is_really_missing(self):
        """State two, and the HIGH: a manifest predating `ann_fingerprint`.

        Honest: §2 names `tuned` alone -- it ran and contributed no recall --
        and says nothing about `normalized`, which §4 prints. Mutant: both are
        declared absent, and a reader following Validity discards the entire
        recall half of a report that is carrying half of it.
        """
        summary, report = self._rendered(fingerprint=False,
                                         records=self._published())
        assert summary["passes_without_a_tree"] == ["tuned"]
        assert "Recall measurements this report could not locate" in report
        section = report.split(
            "### Recall measurements this report could not locate", 1)[1]
        section = section.split("\n### ", 1)[0]
        assert "`tuned`" in section
        assert "`normalized`" not in section

    def test_a_pass_that_really_has_no_recall_is_still_named(self):
        """The intersection must not silence the notice it exists for. With no
        recall records at all, both passes ran and neither is published."""
        summary, report = self._rendered(fingerprint=False, records=[])
        assert summary["passes_without_a_tree"] == ["normalized", "tuned"]
        assert "Recall measurements this report could not locate" in report

    def test_an_ops_record_does_not_silence_it(self):
        """`recall_records`, not `records`: the notice's next sentence says the
        ops measurements are unaffected, so an ops record for the pass must not
        be read as its recall being present."""
        summary, _report = self._rendered(
            fingerprint=False, records=[_ops_record("normalized")])
        assert summary["passes_without_a_tree"] == ["normalized", "tuned"]

    def test_the_stdout_warning_claims_only_what_it_can_know(self, capsys):
        """It runs before a record is read, so it can say the tree could not
        be narrowed to and not that the measurements are absent.

        Asserted on what it prints rather than on its source: the old wording
        was asserted by grepping the function text, and an f-string split
        across two lines defeats that whichever way the words fall.
        """
        generate._warn_about_unrecorded_passes(
            _manifest(BOTH_PASSES, accumulate=False))
        out = capsys.readouterr().out
        assert "cannot narrow to" in out
        assert "absent from this report" not in out
        assert "§2 Validity names any pass whose recall is genuinely " \
               "missing" in out

    @pytest.mark.parametrize("one_pass,phrase", [
        (True, "narrow to that pass's results tree"),
        (False, "narrow to those passes' results tree"),
    ])
    def test_the_possessive_agrees_with_the_number_of_passes(
            self, capsys, one_pass, phrase):
        """A whole phrase, not a substring, so a mutant has to change the
        words rather than the line count.

        A line-count-preserving mutant restoring "those pass's results tree"
        passed the entire suite, and the non-preserving variant failed only
        two citation rows -- reacting to a 2-to-1 line collapse rather than to
        the wording. This file's own history says why that matters: the §2
        notice next door spells the possessive correctly and carries a comment
        recording the lesson, and the fix had been applied in the document and
        not on stdout.
        """
        # One orphan needs a one-pass run; `ran` alone is not enough, because
        # the single-valued pair contributes its own pass whether or not
        # `phases[]` mentions it.
        manifest = _manifest(TUNED_ONLY if one_pass else BOTH_PASSES,
                             accumulate=False)
        manifest["config"].pop("ann_fingerprint")
        generate._warn_about_unrecorded_passes(manifest)
        assert phrase in capsys.readouterr().out

    def test_the_legacy_branch_is_not_silent(self, tmp_path, monkeypatch,
                                             capsys):
        """`if not named` -- a manifest with no fingerprint anywhere -- used to
        return without a word, so stdout said nothing while §2 spoke. That
        branch reads the whole shared tree, which is the cross-configuration
        risk worth hearing about.

        Driven through `generate.main`, because the point is which branch is
        taken and not which line is present.
        """
        manifest = _manifest(BOTH_PASSES, accumulate=False)
        manifest["config"].pop("ann_fingerprint")
        root = _trees(tmp_path, f"normalized/{NORMALIZED_FP}")
        code, capture = _run_report(tmp_path, manifest, root, monkeypatch)
        out = capsys.readouterr().out
        assert code == 0
        # The whole tree, un-narrowed: that is the branch under test.
        assert capture.calls == [(root, "unknown")]
        assert "cannot narrow to" in out
        assert "normalized, tuned" in out


class TestOnePassCountDescribesTheWholeReport:
    """§1, §2, §4's headings and the floor chart's stem must agree.

    They did not. §4 counted the passes in `per_dataset`, which holds only
    those that produced recall for one dataset, so a report whose §1 said
    "normalized, tuned" printed §4 under a bare single-pass heading -- and a
    reader took that table for the other pass's. The floor chart split its
    stem on the same per-dataset count, so with glove measured in both passes
    and sift in one you got two qualified glove stems and a bare sift one,
    whose figcaption then discriminates nothing because the on-figure title
    does not name the pass.

    One number, computed once in `summarize` and carried.
    """

    def test_the_count_is_the_union_of_the_manifest_and_the_records(self):
        """Honest: two, from `phases[]`, even though only one pass produced
        recall. Mutant -- count `per_dataset` -- one, and §4 goes bare."""
        records = [dict(_ops_record("normalized"), phase="recall_qps",
                        engine="percona", qps=1.0, recall_at_k=0.9,
                        ef_search=10)]
        summary = generate.summarize(records,
                                     _manifest(BOTH_PASSES, accumulate=False))
        assert summary["described_passes"] == ["normalized", "tuned"]
        assert sorted(summary["per_dataset"]["glove-100-angular"]) == \
            ["normalized"]

    def test_records_alone_qualify_the_headings(self):
        """`--from-records` over an archive carries no manifest at all."""
        records = [dict(_ops_record(p), phase="recall_qps", engine="percona",
                        qps=1.0, recall_at_k=0.9, ef_search=10)
                   for p in ("normalized", "tuned")]
        summary = generate.summarize(records, {})
        assert summary["described_passes"] == ["normalized", "tuned"]

    #: The reference bundle's own record shape: ops measured in both passes,
    #: recall in `tuned` only -- one dataset's ANN phase failing in one pass is
    #: the normal shape, and the commit message describes exactly this.
    OPS_BOTH_RECALL_ONE = "ops in both passes, recall in tuned only"

    def _bundle_records(self):
        out = [dict(_ops_record(p), phase="concurrency", clients=8, qps=1.0)
               for p in ("normalized", "tuned")]
        out.append(dict(_ops_record("tuned"), phase="recall_qps",
                        engine="percona", ef_search=10, qps=10.0,
                        recall_at_k=0.95,
                        extra={"source_file": COLLIDING_SOURCE_FILE}))
        return out

    @pytest.mark.parametrize("shape,manifest", [
        ("legacy single-valued pair, no phases[]",
         {"run_id": "r", "host": {"cpu": {}}, "engines": {}, "warnings": [],
          "config": {"resource_pass": "tuned", "resolved_resources": {}}}),
        ("no manifest passes at all",
         {"run_id": "r", "host": {"cpu": {}}, "engines": {}, "warnings": [],
          "phases": [], "config": {}}),
        ("phases[] present but empty",
         {"run_id": "r", "host": {"cpu": {}}, "engines": {}, "warnings": [],
          "phases": [], "config": {"resolved_resources": {}}}),
    ])
    def test_an_ops_record_is_evidence_the_run_had_two_passes(self, shape,
                                                              manifest):
        """A record of ANY phase naming a pass counts towards the number.

        On a manifest that names no passes -- a legacy single-valued pair with
        no `phases[]`, no manifest passes at all, or an empty `phases[]` --
        the ops records are the only evidence the run had two, and the
        reference bundle measures ops in both passes with recall in one. A
        narrowing of this union meant only to normalise the unnamed pass
        restricted it to `recall_qps` records and sent §4's heading and the
        floor-chart stem bare on exactly these three shapes, which is the
        defect this change is named for.

        Honest: two passes, a qualified heading, a qualified stem. Mutant --
        keep only the `recall_qps` term -- one pass, `#### d`, `qpsatrecall`.
        """
        records = self._bundle_records()
        summary = generate.summarize(records, manifest)
        assert summary["described_passes"] == ["normalized", "tuned"], shape
        report = render.render_markdown(manifest, summary, records, {}, "E14")
        assert "#### glove-100-angular — tuned" in report, shape
        names = [n for n, _f, _p in generate.qps_at_recall_entries(
            summary, "glove-100-angular", draw=lambda *a, **kw: None)]
        assert names == ["qpsatrecall-tuned"], shape

    def test_it_agrees_with_the_chart_that_derives_its_own(self):
        """`main`'s `passes_present` counts records of any phase for the
        dataset, so restricting this union to recall records made the two
        disagree: `passcompare` drew a normalized-vs-tuned chart above a bare
        single-pass §4 heading. They agree again on all three shapes."""
        records = self._bundle_records()
        manifest = {"run_id": "r", "host": {"cpu": {}}, "engines": {},
                    "warnings": [], "phases": [], "config": {}}
        summary = generate.summarize(records, manifest)
        passes_present = sorted({r.get("resource_pass") for r in records
                                 if r.get("dataset") == "glove-100-angular"
                                 and r.get("resource_pass")})
        assert summary["described_passes"] == passes_present

    def test_a_single_pass_manifest_still_counts_one(self):
        """The hard invariant: the byte-identical single-pass path."""
        records = [dict(_ops_record("tuned"), phase="recall_qps",
                        engine="percona", qps=1.0, recall_at_k=0.9,
                        ef_search=10)]
        for accumulate in (True, False):
            summary = generate.summarize(
                records, _manifest(TUNED_ONLY, accumulate=accumulate))
            assert summary["described_passes"] == ["tuned"], accumulate
            assert len(render.described_passes(summary)) == 1

    def test_a_hand_built_summary_falls_back_to_per_dataset(self):
        """Several tests build a summary without `described_passes`; the
        reader must still answer, and answer 1 for a single pass."""
        assert len(render.described_passes(
            {"per_dataset": {"d": {None: {"mariadb": {}}}}})) == 1
        assert len(render.described_passes(
            {"per_dataset": {"d": {"normalized": {}, "tuned": {}}}})) == 2

    def test_the_floor_chart_stem_splits_on_the_run_not_the_dataset(self):
        """glove in both passes, sift in tuned only. Honest: both datasets get
        qualified stems. Mutant -- split per dataset -- sift gets a bare
        `qpsatrecall` stem whose figcaption names no pass at all.
        """
        records = []
        for dataset, passes in (("glove-100-angular", ("normalized", "tuned")),
                                ("sift-128-euclidean", ("tuned",))):
            for p in passes:
                records.append(dict(_ops_record(p), phase="recall_qps",
                                    engine="percona", dataset=dataset,
                                    qps=1.0, recall_at_k=0.9, ef_search=10))
        summary = generate.summarize(records, _manifest(BOTH_PASSES))
        glove = [n for n, _f, _p in generate.qps_at_recall_entries(
            summary, "glove-100-angular", draw=lambda *a, **kw: None)]
        sift = [n for n, _f, _p in generate.qps_at_recall_entries(
            summary, "sift-128-euclidean", draw=lambda *a, **kw: None)]
        assert glove == ["qpsatrecall-normalized", "qpsatrecall-tuned"]
        assert sift == ["qpsatrecall-tuned"]
        assert "qpsatrecall" not in sift


#: Packages whose `.py` files are parsed by an interpreter this repo does not
#: choose.
#:
#: `report/` and `harness/` are bind-mounted into the bench images and parsed
#: by whatever `python3` those carry. `overlay/ann-benchmarks/` is copied over
#: `work/ann-benchmarks/` and run by `run.py --local` INSIDE those same
#: images -- pgvector's bench entrypoint is `python3 -u run_algorithm.py` on
#: `postgres:17-bookworm` -- so every engine's `algorithms/<e>/module.py` is
#: parsed by the same Python 3.11 this guard exists for, and a construct there
#: would kill that engine's ann pass mid-run. `orchestrator/`, `webui/`,
#: `tests/` and `scripts/` run on the host, whose only documented requirement
#: is `python3`.
#:
#: `overlay` and `scripts` were omitted at first while the docstring's own
#: rule plainly covered them. Both are clean, so that was a future-proofing
#: gap rather than a live defect -- but the guard would have been green while
#: the overlay failed to import, which is the incident class the guard cites.
_PARSED_ELSEWHERE = ("report", "harness", "orchestrator", "overlay",
                     "scripts", "tests", "webui")

#: The four delimiters a string token can open with, longest first so `"""`
#: is recognised before `"`.
_QUOTES = ('"""', "'''", '"', "'")


def _delimiter(text: str) -> str:
    """The quote delimiter a string or f-string token opens with."""
    body = text.lstrip("rRbBuUfF")
    for quote in _QUOTES:
        if body.startswith(quote):
            return quote
    return ""


def _reuses(inner: str, outer: str) -> bool:
    """Whether 3.11 would refuse `inner` nested inside an `outer` f-string.

    Inside a single-delimiter f-string no occurrence of that quote character
    may appear at all; inside a triple-quoted one a lone quote is fine and
    only the triple collides.
    """
    if not inner or not outer:
        return False
    return inner[0] == outer[0] and (len(outer) == 1 or inner == outer)


def _fstring_quote_reuse(source: str):
    """(line, token) for each replacement field reusing its f-string's quote.

    Tokenised, NOT parsed. `ast.parse(src, feature_version=(3, 11))` accepts
    the construct -- measured, and pinned below -- because `feature_version`
    does not downgrade the 3.12 tokenizer. A parse-based guard is therefore a
    false negative that reads exactly like a clean run.

    Returns `None` when the tokenizer itself predates PEP 701, the one case
    this cannot answer: on such an interpreter a file containing the construct
    does not import at all, so the suite is already red.
    """
    import io
    import tokenize

    start = getattr(tokenize, "FSTRING_START", None)
    end = getattr(tokenize, "FSTRING_END", None)
    if start is None:
        return None

    hits, stack = [], []
    for tok in tokenize.tokenize(io.BytesIO(source.encode()).readline):
        if tok.type == start:
            inner = _delimiter(tok.string)
            # A nested f-string reusing the quote is the same construct, and
            # it arrives as FSTRING_START rather than STRING.
            if stack and _reuses(inner, stack[-1]):
                hits.append((tok.start[0], tok.string))
            stack.append(inner)
        elif tok.type == end:
            if stack:
                stack.pop()
        elif stack and tok.type == tokenize.STRING:
            if _reuses(_delimiter(tok.string), stack[-1]):
                hits.append((tok.start[0], tok.string))
    return hits


class TestEveryFileParsesOnTheInterpreterThatWillParseIt:
    """No PEP 701 f-strings: this repo's floor is Python 3.11, undeclared.

    `report/` is bind-mounted into a bench image and parsed by whatever
    `python3` that image has. `docker/pgvector/Dockerfile` builds from
    `postgres:17-bookworm` and `docker/valkey/Dockerfile` from
    `debian:bookworm-slim`; both bench stages `apt-get install python3` and
    install no newer interpreter, and Debian 12 ships 3.11. There is no
    `pyproject.toml`, no `requires-python` and no declared floor anywhere, so
    there is nothing to lean on -- the floor is whatever the documented
    prerequisites give you.

    A replacement field reusing its f-string's own quote shipped in
    `report/generate.py` and would have been a SyntaxError at module parse
    time: a report dying at the end of a long run, which is the incident
    `CLAUDE.md` records for this very package, and a documented-prerequisites
    host where `pytest` cannot collect at all, because `tests/test_config.py`
    imports `report.generate` dozens of times. It was unconditional, not
    gated on the warning it sat in ever firing, and the whole repository was
    3.11-compatible grammar until that line.

    This host has only 3.12, so nobody working here can run 3.11 to find out.
    The guard is the only thing holding the line, which is why it is a guard
    and not a note.
    """

    def _sources(self):
        for package in _PARSED_ELSEWHERE:
            root_dir = os.path.join(VB_ROOT, package)
            if not os.path.isdir(root_dir):
                continue
            for root, _dirs, files in os.walk(root_dir):
                if "__pycache__" in root:
                    continue
                for name in sorted(files):
                    if name.endswith(".py"):
                        yield os.path.join(root, name)

    def test_the_tree_is_free_of_them(self):
        import tokenize
        if getattr(tokenize, "FSTRING_START", None) is None:
            pytest.skip("tokenizer predates PEP 701; a file containing the "
                        "construct would not import here at all")
        offenders = []
        for path in self._sources():
            for line, token in _fstring_quote_reuse(open(path).read()) or []:
                offenders.append(
                    f"{os.path.relpath(path, VB_ROOT)}:{line} {token!r}")
        assert not offenders, (
            "f-string replacement fields reusing the enclosing quote are PEP "
            "701 (Python 3.12+) and this tree must parse on 3.11. Hoist the "
            "value to a local above the f-string:\n  " + "\n  ".join(offenders))

    def test_the_scan_covers_every_package_it_claims_to(self):
        """Non-vacuity: a walk that found nothing would pass the test above
        however broken the detector was.

        Three assertions, because each catches a different way of weakening
        the guard and no one of them catches the others.

        **The membership is pinned**, which is the only thing that catches a
        package being REMOVED from `_PARSED_ELSEWHERE`. Deriving the loop from
        the tuple does not: a tuple with `webui` gone is a loop that never
        looks for `webui`. The total does not either -- `webui` is 12 files
        and `scripts` is 2, against a walk of 122, so both removals sit well
        inside any sane floor. The set is a fact about where this repo's code
        is parsed, so it is written down with a reason per member in the tuple's
        own comment.

        **The loop is derived** from the tuple rather than hand-listed, which
        catches a package being ADDED without being found. The hand-written
        version named four of the then-five members and left `webui`
        unasserted.

        **The total is a deliberate backstop, not a pin.** The walk finds 122
        files (report 8, harness 29, orchestrator 11, overlay 10, scripts 2,
        tests 50, webui 12), which is every tracked `.py` in the repo. `> 100`
        is kept rather than raised to just under 122 on purpose: pinned tight
        it would go red every time someone deletes a test file, which is a
        guard that cries wolf, and the two assertions above already catch
        every way the SCOPE can change. What the total is for is the walk
        itself silently finding nothing -- a wrong `VB_ROOT`, a renamed
        package, an `os.walk` that stops recursing -- and 100 is far enough
        below 122 to tolerate ordinary churn while still catching that.
        """
        assert set(_PARSED_ELSEWHERE) == {
            # bind-mounted into the bench images
            "report", "harness",
            # copied into the bench image and imported by run.py --local
            "overlay",
            # run on the host, whose only requirement is `python3`
            "orchestrator", "webui", "scripts", "tests",
        }, sorted(_PARSED_ELSEWHERE)
        scanned = list(self._sources())
        assert len(scanned) > 100, len(scanned)
        for package in _PARSED_ELSEWHERE:
            assert any(f"{os.sep}{package}{os.sep}" in p for p in scanned), \
                package
        # Two named files, one in a bind-mounted package and one in the
        # overlay the bench image imports, so the walk cannot be passing on
        # host-side packages alone.
        assert any(p.endswith(f"report{os.sep}generate.py") for p in scanned)
        assert any(p.endswith(f"pgvector{os.sep}module.py") for p in scanned)

    @pytest.mark.parametrize("source,flagged,why", [
        ('x = f"{"a" if c else "b"}"', True, "the construct that shipped"),
        ("x = f'{'a' if c else 'b'}'", True, "single-quoted, same reuse"),
        ('x = f"{f"{y}"}"', True, "a nested f-string reusing the quote"),
        ('x = f"{a!r} and {b}"', False, "an ordinary f-string"),
        ('x = f"{chr(39)} and {b}"', False, "a call in a replacement field"),
        ('x = f"""{"a"}"""', False, "a lone quote inside a triple-quoted one"),
        ('x = "plain" + f"{y}"', False, "a string beside an f-string"),
    ])
    def test_the_detector_tells_the_two_apart(self, source, flagged, why):
        """The detector's own probe: a guard whose predicate stopped matching
        is indistinguishable from a clean tree.

        Counts are not pinned -- the first case holds two reusing tokens, and
        the only question is whether the file would parse.
        """
        hits = _fstring_quote_reuse(source)
        if hits is None:
            pytest.skip("tokenizer predates PEP 701")
        assert bool(hits) is flagged, (why, hits)

    def test_a_parse_based_guard_would_not_have_caught_it(self):
        """Why this tokenises rather than parses. The obvious implementation
        reports success on the exact construct that shipped."""
        import ast
        ast.parse('x = f"{"a" if c else "b"}"', feature_version=(3, 11))


class TestARecordNamingNoPassIsSpelledOneWay:
    """`unknown`, everywhere, and never `None`.

    Reachable only from a hand-assembled or third-party `records.jsonl`:
    `load_annb_results` has set `resource_pass` unconditionally since the
    field existed and the value is always truthy, so nothing this pipeline
    writes produces the state. It existed anyway, because `described_passes`
    filtered the falsy name out while `per_dataset` kept it as a key -- and
    the two states that produced are different, which an earlier version of
    this docstring braided into one wrong sentence:

    * **One unnamed pass and one named one.** The count said one, the keys
      said two, so §4 drew two tables for one engine -- 2x apart in
      throughput on this fixture -- under an IDENTICAL BARE `#### d`, and the
      floor chart drew no bars at all because `qps_at_recall` returned `None`
      and the caller swallowed it without a `[report] skipping` line.
    * **An unnamed pass beside TWO named ones.** The count reaches two, so
      headings qualify -- and the unnamed one qualifies as `#### d — None`,
      with a file called `qpsatrecall-None`. That is the Python repr in a
      published artefact.

    Both were real and both are closed; only the sentence describing them was
    wrong. The §1 detail was wrong too: with these fixtures' empty manifest §1
    prints `Resource pass | ?`, naming no pass rather than one.
    """

    def _records(self, *passes):
        out = []
        for i, name in enumerate(passes):
            row = dict(_ops_record("x"), phase="recall_qps", engine="percona",
                       ef_search=10, qps=100.0 * (i + 1), recall_at_k=0.95,
                       extra={"source_file": COLLIDING_SOURCE_FILE})
            if name is None:
                row.pop("resource_pass")
            else:
                row["resource_pass"] = name
            out.append(row)
        return out

    def test_the_count_and_the_table_keys_agree(self):
        """Honest: both say `unknown`, so one engine gets one table. Mutant --
        filter the falsy name out of the count while keying the table on the
        raw value -- the count is 1 and the keys are 2, and §4 prints two
        tables for one engine under one heading."""
        summary = generate.summarize(self._records(None, "tuned"), {})
        assert summary["described_passes"] == ["tuned", "unknown"]
        assert sorted(summary["per_dataset"]["glove-100-angular"]) == \
            ["tuned", "unknown"]

    def test_no_python_repr_reaches_the_document(self):
        """`#### d — None` is a Python repr in a published artefact."""
        records = self._records(None, "tuned")
        summary = generate.summarize(records, {})
        report = render.render_markdown({"config": {}}, summary, records,
                                        {}, "E14")
        assert "— unknown" in report
        assert "None" not in report

    def test_no_chart_stem_is_named_none(self):
        """The stem is a filename, and `qpsatrecall-None` is one a reader
        would have to recognise as a bug rather than a pass."""
        summary = generate.summarize(self._records(None, "tuned"), {})
        names = [n for n, _f, _p in generate.qps_at_recall_entries(
            summary, "glove-100-angular", draw=lambda *a, **kw: None)]
        assert names == ["qpsatrecall-tuned", "qpsatrecall-unknown"]

    def test_the_legend_uses_the_same_word(self):
        """The fourth spelling, already correct, now delegated so it cannot
        drift from the other three."""
        assert charts._pass_label(None) == "unknown"
        assert charts._pass_label("") == "unknown"
        assert charts._pass_label("tuned") == "tuned"

    def test_a_named_pass_is_untouched(self):
        """The normaliser must not rename anything real."""
        from report.loaders import pass_or_unknown
        assert pass_or_unknown("normalized") == "normalized"
        assert pass_or_unknown("tuned") == "tuned"


class TestThePassReachesTheDrawnFigure:
    """Three behaviours that only a drawn figure can confirm.

    `pareto_series_style` returning the right label and linestyle is not the
    same as `pareto` USING them, and a mutant that drew `style_for(engine)`'s
    label instead survived the whole suite -- the pass simply vanished from a
    two-pass legend. Same for a hard-coded linestyle. `matplotlib` lives in
    the bench images, so these skip on a bare host.
    """

    def _points(self, resource_pass):
        return [{"engine": "percona", "resource_pass": resource_pass,
                 "recall_at_k": 0.90 + 0.01 * i, "qps": 300.0 - 100.0 * i,
                 "dataset": "d", "ef_search": 10 * (i + 1)}
                for i in range(3)]

    def _lines(self, by_key):
        """The lines `pareto` actually drew, captured through `charts._save`.

        `plt.gcf()` is no good here: `_save` closes the figure, so the current
        figure afterwards is a fresh empty one and `axes[0]` raises. The spy is
        the idiom `tests/test_config.py` already uses for this chart.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import tempfile
        captured = {}
        original = charts._save

        def spy(fig, out_dir, stem):
            captured["ax"] = fig.axes[0]
            return original(fig, out_dir, stem)

        charts._save = spy
        try:
            with tempfile.TemporaryDirectory() as d:
                charts.pareto(by_key, "d", 10, d, "pareto-d")
        finally:
            charts._save = original
        lines = [(str(l.get_label()), l.get_linestyle())
                 for l in captured["ax"].get_lines()]
        plt.close("all")
        # Only the frontier lines carry a pass suffix; the scatter does not.
        return [(label, style) for label, style in lines if "/ " in label]

    @needs_matplotlib
    def test_the_legend_names_both_passes(self):
        """Honest: two entries, each suffixed. Mutant -- plot
        `style_for(engine)["label"]` -- one name twice and no pass anywhere.
        """
        drawn = self._lines(
            {("percona", "normalized"): self._points("normalized"),
             ("percona", "tuned"): self._points("tuned")})
        assert sorted(label for label, _style in drawn) == [
            f"{charts.style_for('percona')['label']} / normalized",
            f"{charts.style_for('percona')['label']} / tuned"], drawn

    @needs_matplotlib
    def test_the_two_frontiers_are_drawn_with_different_strokes(self):
        """Honest: two linestyles. Mutant -- a fixed `linestyle="-"` -- both
        passes of one engine are visually identical and the legend is the only
        way to tell them apart."""
        drawn = self._lines(
            {("percona", "normalized"): self._points("normalized"),
             ("percona", "tuned"): self._points("tuned")})
        assert len({style for _label, style in drawn}) == 2, drawn

    @needs_matplotlib
    def test_a_pass_without_a_name_is_still_told_apart(self):
        """A records file hand-merged across the introduction of
        `resource_pass` holds records with the field and without. Honest: two
        strokes and a named `unknown`. Mutant -- filter `None` out of the
        ordering -- one engine draws two frontiers under one label and one
        stroke."""
        drawn = self._lines({("percona", None): self._points(None),
                             ("percona", "tuned"): self._points("tuned")})
        assert len({style for _label, style in drawn}) == 2, drawn
        assert any(label.endswith("/ unknown") for label, _s in drawn), drawn

    def test_the_any_pass_sentinel_is_not_none(self, tmp_path):
        """`None` is a real pass key -- records predating the field group under
        it -- so a `None` default would make "no pass recorded" and "caller did
        not say" the same request. Honest: asking for `None` against a summary
        holding only `tuned` draws nothing. Mutant -- `_ANY_PASS = None` --
        it draws tuned's chart under a caller's request for a different pass.
        """
        summary = {"per_dataset": {"d": {"tuned": {"percona": {
            "qps_at_recall_90": 1.0}}}}}
        # Into `tmp_path`, because the honest path draws nothing but the
        # mutant does, and a probe run must not leave a chart in the repo.
        assert charts.qps_at_recall(summary, "d", str(tmp_path), "s",
                                    resource_pass=None) is None
        assert charts._ANY_PASS is not None


# ---------------------------------------------------------------------------
# The two sections that describe the run
# ---------------------------------------------------------------------------

class TestTheEnvironmentSectionNamesEachPass:

    def test_both_passes_are_named(self, tmp_path):
        """Honest: `normalized, tuned`. Mutant -- read the single-valued
        `resource_pass` -- `tuned`, above tables holding normalized rows.
        """
        report = _render(_manifest(BOTH_PASSES))
        assert _env_cell(report, "Resource pass") == ["normalized, tuned"]

    def test_each_pass_carries_its_own_limits(self, tmp_path):
        """The published failure: cpuset `0-75` and 112.5 GiB printed above
        rows measured with 8 CPUs and 16 GB. Cells compared whole, because
        `"0-7" in report` is satisfied by `0-75`.
        """
        report = _render(_manifest(BOTH_PASSES))
        assert _env_cell(report, "normalized — server cpuset") == ["0-7"]
        assert _env_cell(report, "tuned — server cpuset") == ["0-75"]
        assert _env_cell(report, "normalized — server memory limit") == \
            ["16.0 GiB"]
        assert _env_cell(report, "tuned — server memory limit") == \
            ["112.5 GiB"]

    def test_the_undifferentiated_rows_are_gone_when_there_are_two_passes(
            self, tmp_path):
        """A `Server cpuset` row beside two labelled ones would be a third
        claim, and a reader would have to guess which pass it belonged to."""
        report = _render(_manifest(BOTH_PASSES))
        assert _env_cell(report, "Server cpuset") == []
        assert _env_cell(report, "Server memory limit") == []

    def test_every_pass_limit_appears_once_per_pass(self, tmp_path):
        rows = _table_rows(_render(_manifest(BOTH_PASSES)),
                           ENVIRONMENT_HEADER)
        labelled = [r[0] for r in rows if " — " in r[0]]
        assert len(labelled) == 10
        assert len(set(labelled)) == 10

    def test_it_reaches_the_html_page_too(self, tmp_path):
        """The page is the artifact most reports are read as."""
        page = _page(_manifest(BOTH_PASSES))
        assert "normalized — server cpuset" in page
        assert "tuned — server memory limit" in page

    def test_the_headline_cards_do_not_pick_one_pass(self, tmp_path):
        """The cards sit above §1 and made the same single claim it did.

        Honest: both, each qualified by its pass. Mutant -- keep the
        single-valued read -- the first thing the page says is `0-75` and
        112.5 GiB, above tables holding 8-CPU, 16 GB rows.
        """
        page = _page(_manifest(BOTH_PASSES))
        assert "<dd>0-7 (normalized), 0-75 (tuned)</dd>" in page
        assert "<dd>16.0 GiB (normalized), 112.5 GiB (tuned)</dd>" in page
        assert "<dd>normalized, tuned</dd>" in page
        assert "<dd>0-75</dd>" not in page

    def test_a_limit_the_manifest_does_not_state_is_not_invented(self,
                                                                 tmp_path):
        """`?`, matching the single-pass rows, rather than `None`."""
        manifest = _manifest(
            (("normalized", NORMALIZED_FP, {"server_memory_bytes": 1}),
             ("tuned", TUNED_FP, TUNED_LIMITS)))
        report = _render(manifest)
        assert _env_cell(report, "normalized — server cpuset") == ["?"]
        assert _env_cell(report, "normalized — build threads") == ["?"]
        # A byte count the manifest DOES state still goes through
        # `_fmt_bytes`, so "not invented" is not "not formatted".
        assert _env_cell(report, "normalized — server memory limit") == \
            ["1.0 B"]


class TestTheReproductionCommandNamesThePassesThatRan:

    def test_both_passes_reproduce_as_both(self, tmp_path):
        """`--resource-pass both` is the flag `cli.py` expands into the pass
        list. Honest: `both`. Mutant -- keep the single-valued key -- `tuned`,
        a command that does neither of the two things the run did.
        """
        report = _render(_manifest(BOTH_PASSES))
        assert "--resource-pass both\n" in report
        assert "--resource-pass tuned" not in report

    def test_it_is_not_a_comma_separated_list(self, tmp_path):
        """`cli.py` tests `requested == "both"` and otherwise takes the value
        as a single pass name, so `normalized,tuned` would send the reader to
        an invalid-choice error."""
        report = _render(_manifest(BOTH_PASSES))
        assert "--resource-pass normalized,tuned" not in report

    def test_the_html_command_agrees_with_the_markdown_one(self, tmp_path):
        """Two renderers, one run. They disagreed by construction before:
        both read the same key, but only one would have been fixed by a patch
        to the markdown.
        """
        page = _page(_manifest(BOTH_PASSES))
        assert "--resource-pass both" in page


# ---------------------------------------------------------------------------
# The constraint the fix had to hold: a single-pass run is untouched
# ---------------------------------------------------------------------------

class TestASinglePassRunIsUnchanged:
    """Proven, not asserted by inspection.

    Every published single-pass report has to keep reading the way it reads.
    The fixture is the archived shape -- single-valued keys, no per-pass list
    -- because that is what every existing run directory on disk looks like.
    """

    @pytest.mark.parametrize("accumulate", [True, False],
                             ids=["recorded", "archived"])
    def test_it_narrows_to_the_same_single_tree(self, tmp_path, monkeypatch,
                                                accumulate):
        """Both shapes of a one-pass manifest yield the one tree, with the
        same `[report] ann-bench tree:` line as before."""
        manifest = _manifest(TUNED_ONLY, accumulate=accumulate)
        root = _trees(tmp_path, f"tuned/{TUNED_FP}")
        code, capture = _run_report(tmp_path, manifest, root, monkeypatch)
        assert code == 0
        assert capture.dirs == [os.path.join(root, "tuned", TUNED_FP)]

    def test_the_narrowing_line_is_unchanged(self, tmp_path, monkeypatch,
                                             capsys):
        root = _trees(tmp_path, f"tuned/{TUNED_FP}")
        _run_report(tmp_path, _manifest(TUNED_ONLY), root, monkeypatch)
        assert f"[report] ann-bench tree: tuned/{TUNED_FP}" in \
            capsys.readouterr().out

    def test_the_missing_tree_message_is_unchanged(self, tmp_path,
                                                   monkeypatch, capsys):
        """The exact sentence a one-pass run has always printed when its tree
        is gone, including the pointer to Validity."""
        root = _trees(tmp_path)
        _run_report(tmp_path, _manifest(TUNED_ONLY), root, monkeypatch)
        assert (f"[report] no tree at tuned/{TUNED_FP}; reading the whole "
                f"ann-benchmarks directory. Results older than this run will "
                f"be flagged in Validity.") in capsys.readouterr().out

    def test_it_does_not_announce_a_per_tree_record_count(
            self, tmp_path, monkeypatch, capsys):
        """One tree, one count line, worded as before."""
        root = _trees(tmp_path, f"tuned/{TUNED_FP}")
        _run_report(tmp_path, _manifest(TUNED_ONLY), root, monkeypatch)
        out = capsys.readouterr().out
        assert "[report] ann-benchmarks records: 1" in out
        assert "records from tuned" not in out

    @pytest.mark.parametrize("accumulate", [True, False],
                             ids=["recorded", "archived"])
    def test_the_environment_rows_are_exactly_the_old_ones(self, accumulate):
        """The five limit rows, undifferentiated, in their original order and
        with their original labels. A labelled block here would mean every
        published single-pass report disagrees with a regenerated one.
        """
        manifest = _manifest(TUNED_ONLY, accumulate=accumulate)
        rows = render._environment_rows(manifest)
        assert rows[-5:] == [
            ["Server cpuset", "0-75"],
            ["Server memory limit", "112.5 GiB"],
            ["Buffer pool / shared_buffers", "39.4 GiB"],
            ["Graph cache", "28.1 GiB"],
            ["Build threads", "8"],
        ]
        assert ["Resource pass", "tuned"] in rows
        assert not [r for r in rows if " — " in r[0]]

    @pytest.mark.parametrize("accumulate", [True, False],
                             ids=["recorded", "archived"])
    def test_the_reproduction_command_is_unchanged(self, accumulate):
        manifest = _manifest(TUNED_ONLY, accumulate=accumulate)
        assert "--resource-pass tuned\n" in _render(manifest)
        assert "--resource-pass tuned" in _page(manifest)

    @pytest.mark.parametrize("accumulate", [True, False],
                             ids=["recorded", "archived"])
    def test_the_headline_cards_are_unchanged(self, accumulate):
        """Bare values, unqualified, as every published page has them."""
        page = _page(_manifest(TUNED_ONLY, accumulate=accumulate))
        assert "<dt>Resource pass</dt><dd>tuned</dd>" in page
        assert "<dt>Server cpuset</dt><dd>0-75</dd>" in page
        assert "<dt>Server memory</dt><dd>112.5 GiB</dd>" in page

    def test_a_manifest_whose_config_block_names_no_pass_still_renders(self):
        """A config block with no `resource_pass`, which no real run produces
        and the report refuses elsewhere. §1's cell still reads `?`, because
        that cell is the config block's own statement.

        §11 is the one deliberate behaviour change outside a multi-pass run,
        and it is an improvement rather than a regression: the flag now comes
        from `phases[]`, so it prints the pass the run actually executed
        instead of the hard-coded `normalized` this used to guess. It cannot
        affect a published single-pass report, where the config block names
        the pass `phases[]` names; the four assertions in
        `test_the_reproduction_command_is_unchanged` cover that case.
        """
        manifest = _manifest(TUNED_ONLY, accumulate=False)
        manifest["config"].pop("resource_pass")
        manifest["config"].pop("ann_fingerprint")
        report = _render(manifest)
        assert _env_cell(report, "Resource pass") == ["?"]
        assert "--resource-pass tuned\n" in report
        assert "--resource-pass normalized" not in report

    def test_a_manifest_with_neither_a_pass_nor_a_phase_log_falls_back(self):
        """Nothing anywhere to read a pass out of: §11 keeps its documented
        default rather than emitting an empty flag."""
        manifest = _manifest(TUNED_ONLY, accumulate=False)
        manifest["config"].pop("resource_pass")
        manifest["config"].pop("ann_fingerprint")
        manifest["phases"] = []
        report = _render(manifest)
        assert _env_cell(report, "Resource pass") == ["?"]
        assert "--resource-pass normalized\n" in report


# ---------------------------------------------------------------------------
# Where the fingerprint comes from
# ---------------------------------------------------------------------------

class TestTheFingerprintIsRecordedAndNeverRecomputedByTheReport:
    """`report/` runs in a container that mounts report/ and harness/ only.

    `ann_fingerprint` lives in `orchestrator/ann_pass.py`. Importing the
    orchestrator from the report once crashed generation at the end of a
    twenty-hour run, so the fingerprint has to arrive through the manifest --
    exactly as `engines.<name>.presentation` does. A second copy of the hash
    in `report/` would be worse than an import: it would drift silently, and
    the report would narrow to a tree nothing ever wrote.
    """

    def test_the_report_reads_the_fingerprint_out_of_the_manifest(self):
        assert render.recorded_passes(_manifest(BOTH_PASSES)) == [
            ("normalized", NORMALIZED_FP), ("tuned", TUNED_FP)]

    def test_a_pass_recorded_without_one_yields_none_rather_than_a_guess(self):
        manifest = _manifest(
            (("normalized", None, NORMALIZED_LIMITS),
             ("tuned", TUNED_FP, TUNED_LIMITS)))
        assert render.recorded_passes(manifest) == [
            ("normalized", None), ("tuned", TUNED_FP)]

    def test_a_list_that_omits_the_carried_pass_still_reads_its_fingerprint(
            self):
        """`_pass_resources` adopts the pass the single-valued block names
        whether or not the list mentions it, so `recorded_passes` has to as
        well.

        Otherwise the two readers disagree about one pass: §1 prints its real
        limits -- adopted from the block -- while §2 declares its recall
        unlocatable, because only one of them read its recorded fingerprint.
        Unreachable from the fixed orchestrator, reachable from a hand-edited
        manifest, and a disagreement between two sections of one report is the
        whole subject of this file.

        Honest: both passes, tuned carrying `a9349676`. Mutant -- consult the
        pair only when the list is empty -- tuned appears with no fingerprint
        and §2 names it as unlocatable beside §1's `0-75`.
        """
        manifest = _manifest(BOTH_PASSES, accumulate=False)
        manifest["config"]["passes"] = [
            {"resource_pass": "normalized",
             "resolved_resources": NORMALIZED_LIMITS,
             "ann_fingerprint": NORMALIZED_FP}]
        assert render.recorded_passes(manifest) == [
            ("normalized", NORMALIZED_FP), ("tuned", TUNED_FP)]
        assert render.passes_without_a_tree(manifest) == []

    def test_the_report_package_imports_no_hash_at_all(self):
        """A source guard, because the subtly wrong alternative to recording
        is re-deriving: the hash is eight hex characters of a five-field
        digest and a copy of it would look right until a field moved.

        Parsed rather than grepped. This read `"import hashlib" not in line`,
        which does not match `from hashlib import sha256` -- so the obvious
        mutant imported the hash and the test stayed green (honest 1 passed,
        mutant 1 passed: a dead probe). Walking `Import` and `ImportFrom`
        catches every spelling, and it drops the need to skip comment lines,
        which was the other half of why a string search was fragile here.
        """
        for name in sorted(os.listdir(os.path.join(VB_ROOT, "report"))):
            if not name.endswith(".py"):
                continue
            path = os.path.join(VB_ROOT, "report", name)
            tree = ast.parse(open(path).read(), filename=path)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert alias.name.split(".")[0] != "hashlib", \
                            (name, alias.name)
                elif isinstance(node, ast.ImportFrom):
                    assert (node.module or "").split(".")[0] != "hashlib", \
                        (name, node.module)

    def test_the_report_package_calls_no_fingerprint_function(self):
        """The other spelling of the same mistake: a copy of the digest
        written inline, or a call smuggled in through an import of the
        orchestrator (which its own guard already refuses)."""
        for name in sorted(os.listdir(os.path.join(VB_ROOT, "report"))):
            if not name.endswith(".py"):
                continue
            path = os.path.join(VB_ROOT, "report", name)
            tree = ast.parse(open(path).read(), filename=path)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                called = node.func
                named = (called.id if isinstance(called, ast.Name)
                         else getattr(called, "attr", ""))
                assert named != "ann_fingerprint", (name, named)

    def test_the_unknown_pass_sentinel_is_defined_once(self):
        """`generate.py` has to hand the loader a default explicitly for the
        whole-tree fallback, so the sentinel was spelled in both files.

        Asserted as the claim actually is -- the sentinel is defined once and
        nothing else in the package spells it by hand -- rather than as a
        spelling pin on one signature. A value test cannot fail here at all:
        respelling the default back to `"unknown"` is behaviour-neutral, and a
        probe that does it left all 76 tests passing. Counting the literal
        also survives a rename of the constant, where an `ast` check for a
        `Name` node called `UNKNOWN_PASS` would not.

        Counted per file and NOT pinned by line: a line number in an assertion
        is the rot this file has already had to repair twice.
        """
        from report import loaders as loaders_mod
        assert loaders_mod.UNKNOWN_PASS == generate.UNKNOWN_PASS == "unknown"

        remaining: Dict[str, int] = {}
        definitions = 0
        for name in sorted(os.listdir(os.path.join(VB_ROOT, "report"))):
            if not name.endswith(".py"):
                continue
            path = os.path.join(VB_ROOT, "report", name)
            tree = ast.parse(open(path).read(), filename=path)
            defined_here = set()
            for node in ast.walk(tree):
                if (isinstance(node, ast.Assign)
                        and any(isinstance(t, ast.Name)
                                and t.id == "UNKNOWN_PASS"
                                for t in node.targets)):
                    defined_here.add(id(node.value))
                    definitions += 1
            for node in ast.walk(tree):
                if (isinstance(node, ast.Constant)
                        and node.value == "unknown"
                        and id(node) not in defined_here):
                    remaining[name] = remaining.get(name, 0) + 1
        assert definitions == 1, "the sentinel must be defined exactly once"
        # The two accounted-for fallbacks, measured rather than assumed:
        # `loaders.py`'s is ann-benchmarks' `algo` fallback and `charts.py`'s
        # is the FALLBACK style's label. Neither is a resource pass. Comments
        # and docstrings mentioning the word are not string literals and so
        # do not count.
        assert remaining == {"charts.py": 1, "loaders.py": 1}, remaining

    def test_the_orchestrator_computes_it_where_the_function_lives(self):
        """The counterpart: `cli.py` passes it into `set_config` as `extra`,
        on the host, from the same function that keys the tree. Asserted on
        the call rather than on a string, so moving the call out of the
        orchestrator fails here.
        """
        source = open(os.path.join(VB_ROOT, "orchestrator", "cli.py")).read()
        at_set_config = source.index("manifest.set_config(")
        window = source[at_set_config:at_set_config + 400]
        assert "ann_pass.ann_fingerprint(resolved)" in window

    def test_the_recorded_fingerprint_is_the_one_that_keys_the_tree(
            self, tmp_path):
        """Not a string comparison: the entry is built by `set_config` from
        `ann_fingerprint`'s own output, and `annb_results_dir` keys the
        directory with the same call. If the two ever disagreed the report
        would narrow to a path nothing wrote.
        """
        from orchestrator import ann_pass
        m = Manifest(str(tmp_path / "run-fp"), "run-fp")
        for name, limits in (("normalized", NORMALIZED_LIMITS),
                             ("tuned", TUNED_LIMITS)):
            m.set_config({}, name, limits,
                         extra={"ann_fingerprint":
                                ann_pass.ann_fingerprint(limits)})
        paths = {"annb_results": str(tmp_path / "annb")}
        for entry in m.data["config"]["passes"]:
            name = entry["resource_pass"]
            limits = NORMALIZED_LIMITS if name == "normalized" else TUNED_LIMITS
            written = ann_pass.annb_results_dir(paths, name, limits)
            assert os.path.basename(written) == entry["ann_fingerprint"]
        prints = {e["ann_fingerprint"] for e in m.data["config"]["passes"]}
        assert len(prints) == 2, "the two passes must key different trees"
