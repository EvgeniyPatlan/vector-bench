"""E22 -- the bundle meant to be emailed would carry a multi-gigabyte core.

THE DEFECT
----------
`orchestrator/export.py::write_bundle` collected a run directory with one
unfiltered call -- `archive.add(run_dir, arcname=run_id)` -- and `tarfile.add`
recurses by default and applies no filter, so every byte under the run
directory went into the bundle.

That was harmless only for as long as nothing large was written there. Crash
capture changes that: on a failed phase the server's core dump is copied into
`<run_dir>/cores/<engine>-<dataset>-<phase>-<pass>/`, under the capture side's
own `core*` glob, and a core is a full memory image of the server process. The
measured peak RSS for the `percona` engine in this project's archives is
**54.7 GiB** (gist-960, tuned, ann pass; the ops-phase peak is 48.5 GiB). Gzip
shrinks that and does not make it emailable, and `write_bundle` would have sat
there compressing it saying nothing about what it was doing.

The second half is the one that matters more, and it is why these tests weigh
the README as heavily as the member list. A recipient holds the bundle and has
no way to compare it against the run directory it was made from. An exclusion
they are not told about leaves them with a credible-looking copy of a run that
is quietly missing something -- which is the failure this repository is built
to avoid, rather than the crash it would be if the tar simply broke.

WHAT THESE TESTS PIN
--------------------
* A core goes, whether it is in the run root or nested -- the filter is applied
  to every member `add` walks, which is the reason it is a `filter=` callable
  and not a pre-walk.
* The directories around it stay. Dropping `cores/` as a directory would be
  cheaper, and `add` would then never walk into it, so there would be nothing
  left to name or size in the README. It also keeps a directory called
  `core-something` from taking its contents with it.
* Everything else in a run directory still travels, asserted as set equality
  against a fixture shaped like the twelve archived runs under
  `ALL_RESULTS/results/` -- which catches a predicate that grew as well as one
  that shrank.
* The README names each exclusion and its size, relative to the run directory,
  and says where the core still is and what to ask for instead.
* A run with no core produces the README it always produced, byte for byte.
* `detail` on success stays a bare path, because `cli.cmd_export` calls
  `os.path.getsize` on it. See `TestTheDetailIsStillAPath`.
"""

from __future__ import annotations

import json
import os
import sys
import tarfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator import export as export_mod  # noqa: E402

RUN_ID = "percona-20261021-134142"

# Complete on purpose: every field the README fills in is populated, so the
# word "unknown" cannot reach the text from the manifest side and a test that
# refuses to see it in the exclusion block is testing the exclusion block.
MANIFEST = {
    "run_id": RUN_ID,
    "status": "failed",
    "started_at": "2026-10-21T13:41:42Z",
    "engines": {"percona": {"build": {"tag": "vector-mvp", "march": "x86-64-v3"}}},
    "host": {"hostname": "bench-rig-2", "kernel": "6.8.0-147-generic",
             "total_ram_bytes": 64 * 1024 ** 3,
             "cpu": {"model": "AMD EPYC 9554", "physical_cores": 64,
                     "logical_cpus": 128, "has_avx512": True,
                     "simd_flags": ["avx2", "avx512f"]}},
    "config": {"resource_pass": "tuned",
               "profile": {"name": "percona", "description": "the seventh engine"}},
    "phases": [{"dataset": "gist-960-euclidean"}],
}

# The quadruple the capture side keys a core's directory on -- engine, dataset,
# phase, resource pass -- and a second phase of the same run for the case where
# more than one of them crashed.
QUAD = "percona-gist-960-euclidean-ann-tuned"
QUAD_2 = "percona-gist-960-euclidean-m32-post-tuned"
CORE_DIR = f"cores/{QUAD}"


@pytest.fixture
def run_dir(tmp_path):
    """A run directory shaped like the archived ones, holding no core.

    The inventory is the union of what the twelve run directories under
    `ALL_RESULTS/results/` actually contain: `run-manifest.json`,
    `checkpoints.json`, `mem-*.jsonl`, `ops-*.jsonl`, `logs/*.log`,
    `.cache/*.npy`, `report/` with its charts -- plus `vb-label.json`, which a
    web UI import writes. Not one entry in any of them begins with `core`,
    which is what makes the predicate under test safe to apply by basename.
    """
    run = tmp_path / "results" / RUN_ID
    (run / "report" / "charts").mkdir(parents=True)
    (run / "logs").mkdir()
    (run / ".cache").mkdir()
    (run / "run-manifest.json").write_text(json.dumps(MANIFEST))
    (run / "checkpoints.json").write_text('{"done": []}')
    (run / "vb-label.json").write_text('{"label": "sent to the server team"}')
    (run / "ops-percona-gist-960-euclidean-tuned-m16-post.jsonl").write_text("{}\n")
    (run / "mem-percona-gist-960-euclidean-tuned-m16-post.jsonl").write_text("{}\n")
    (run / "logs" / "percona-gist-960-euclidean-tuned-ann.log").write_text("ready\n")
    (run / ".cache" / "gist-960-euclidean-filtered.npy").write_bytes(b"\0" * 32)
    (run / "report" / "report.html").write_text("<html>report</html>")
    (run / "report" / "report.md").write_text("# report\n")
    (run / "report" / "records.jsonl").write_text('{"phase":"ingest"}\n')
    (run / "report" / "charts" / "pareto-gist-960-euclidean.svg").write_text("<svg/>")
    return run


# What that fixture must look like inside the archive, directory members
# included. Set equality against this is the test for "a non-core file of every
# other kind is kept", and it fails just as loudly if the predicate starts
# keeping something it should not.
BASE_MEMBERS = {
    RUN_ID,
    f"{RUN_ID}/README.txt",
    f"{RUN_ID}/run-manifest.json",
    f"{RUN_ID}/checkpoints.json",
    f"{RUN_ID}/vb-label.json",
    f"{RUN_ID}/ops-percona-gist-960-euclidean-tuned-m16-post.jsonl",
    f"{RUN_ID}/mem-percona-gist-960-euclidean-tuned-m16-post.jsonl",
    f"{RUN_ID}/logs",
    f"{RUN_ID}/logs/percona-gist-960-euclidean-tuned-ann.log",
    f"{RUN_ID}/.cache",
    f"{RUN_ID}/.cache/gist-960-euclidean-filtered.npy",
    f"{RUN_ID}/report",
    f"{RUN_ID}/report/report.html",
    f"{RUN_ID}/report/report.md",
    f"{RUN_ID}/report/records.jsonl",
    f"{RUN_ID}/report/charts",
    f"{RUN_ID}/report/charts/pareto-gist-960-euclidean.svg",
}


def _write_core(run, relative, size=4096):
    """Put a file of `size` bytes at `relative` under the run directory."""
    path = run / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * size)
    return path


def _bundle(run, tmp_path):
    """Write a real tarball and hand back (members, readme, detail)."""
    out = str(tmp_path / "bundle.tar.gz")
    ok, detail = export_mod.write_bundle(str(run), out)
    assert ok, detail
    with tarfile.open(out) as archive:
        members = set(archive.getnames())
        readme = archive.extractfile(f"{RUN_ID}/README.txt").read().decode()
    return members, readme, detail


def _section(readme):
    """Just the disclosure block, so a match cannot come from elsewhere."""
    head = "NOT IN THIS BUNDLE"
    if head not in readme:
        return ""
    rest = readme[readme.index(head):]
    tail = "IF YOU HAVE VECTOR-BENCH"
    return rest[:rest.index(tail)] if tail in rest else rest


class TestTheCoreDoesNotTravel:
    def test_a_core_in_the_run_root_is_left_out(self, run_dir, tmp_path):
        """Where a core lands when the crashing process's cwd is the run dir."""
        _write_core(run_dir, "core.4211")
        members, _readme, _detail = _bundle(run_dir, tmp_path)
        assert f"{RUN_ID}/core.4211" not in members
        assert members == BASE_MEMBERS

    def test_a_core_in_a_nested_directory_is_left_out(self, run_dir, tmp_path):
        """Where capture puts it, two levels down, under the quadruple."""
        _write_core(run_dir, f"{CORE_DIR}/core.4211", size=7 * 1024 ** 2)
        members, _readme, _detail = _bundle(run_dir, tmp_path)
        assert f"{RUN_ID}/{CORE_DIR}/core.4211" not in members
        assert not [n for n in members if "core." in n]

    def test_nothing_but_the_cores_was_left_behind(self, run_dir, tmp_path):
        """Set equality, with a core in both places a core can be.

        A non-core file of every kind a run directory holds is in
        `BASE_MEMBERS`, so this fails as loudly for a predicate that grew as
        for one that shrank -- a filter that quietly stopped shipping
        `.cache/*.npy` or `logs/*.log` would otherwise look exactly like a
        working one from the member list of a bundle nobody diffed.
        """
        _write_core(run_dir, "core.9981")
        _write_core(run_dir, f"{CORE_DIR}/core.4211")
        members, _readme, _detail = _bundle(run_dir, tmp_path)
        assert members == BASE_MEMBERS | {f"{RUN_ID}/cores",
                                          f"{RUN_ID}/{CORE_DIR}"}

    def test_the_directories_around_it_still_travel(self, run_dir, tmp_path):
        """The subtree is walked, not skipped.

        Excluding `cores/` as a directory would be cheaper and `add` would
        never walk into it -- which would leave nothing to name or size in the
        README, and the disclosure is the half that matters.
        """
        _write_core(run_dir, f"{CORE_DIR}/core.4211")
        members, _readme, _detail = _bundle(run_dir, tmp_path)
        assert f"{RUN_ID}/cores" in members
        assert f"{RUN_ID}/{CORE_DIR}" in members

    def test_a_directory_whose_name_begins_with_core_keeps_its_contents(
            self, run_dir, tmp_path):
        """A basename predicate that also ate directories would take this."""
        (run_dir / "core-notes").mkdir()
        (run_dir / "core-notes" / "mysqld-stderr.log").write_text("sigsegv\n")
        members, _readme, _detail = _bundle(run_dir, tmp_path)
        assert f"{RUN_ID}/core-notes" in members
        assert f"{RUN_ID}/core-notes/mysqld-stderr.log" in members

    def test_every_core_goes_when_there_is_more_than_one(self, run_dir, tmp_path):
        """A phase per M value crashing the same way leaves several."""
        _write_core(run_dir, f"{CORE_DIR}/core.4211")
        _write_core(run_dir, f"cores/{QUAD_2}/core.5100")
        _write_core(run_dir, "core.9981")
        members, readme, _detail = _bundle(run_dir, tmp_path)
        assert not [n for n in members if os.path.basename(n).startswith("core.")]
        section = _section(readme)
        for name in (f"{CORE_DIR}/core.4211", f"cores/{QUAD_2}/core.5100",
                     "core.9981"):
            assert name in section

    def test_a_non_core_file_whose_name_begins_with_core_goes_too_and_is_named(
            self, run_dir, tmp_path):
        """The predicate's known false positive, pinned rather than discovered.

        `core*` is the capture side's glob and matching anything narrower here
        would keep a file that half had collected *because* it is a core. The
        cost is that a `core-notes.txt` nobody writes today would go as well.
        That is acceptable only because it cannot happen silently, which is
        what this asserts: whatever is dropped is named.
        """
        (run_dir / "core-notes.txt").write_text("not actually a core\n")
        members, readme, _detail = _bundle(run_dir, tmp_path)
        assert f"{RUN_ID}/core-notes.txt" not in members
        assert "core-notes.txt" in _section(readme)


class TestTheReadmeSaysWhatIsMissing:
    def test_it_names_the_excluded_file_and_its_size(self, run_dir, tmp_path):
        _write_core(run_dir, f"{CORE_DIR}/core.4211", size=3 * 1024 ** 3)
        _members, readme, _detail = _bundle(run_dir, tmp_path)
        section = _section(readme)
        assert f"{CORE_DIR}/core.4211" in section
        assert "3.0 GiB" in section

    def test_the_path_it_names_is_relative_to_the_run_directory(
            self, run_dir, tmp_path):
        """Not the tar member name, and not an absolute host path.

        `tarfile` hands the filter `<run_id>/cores/...`, and the run directory
        is somewhere on the exporting machine. Neither is a path the recipient
        can act on: they will be looking at an extracted directory, where the
        FILES section above names things like `report/report.html`.
        """
        _write_core(run_dir, f"{CORE_DIR}/core.4211")
        _members, readme, _detail = _bundle(run_dir, tmp_path)
        section = _section(readme)
        assert f"{CORE_DIR}/core.4211" in section
        assert f"{RUN_ID}/{CORE_DIR}" not in section
        assert str(run_dir) not in readme

    def test_it_says_where_the_core_still_is_and_what_to_ask_for(
            self, run_dir, tmp_path):
        """A core is unreadable off its machine; the backtrace is the artefact."""
        _write_core(run_dir, f"{CORE_DIR}/core.4211")
        _members, readme, _detail = _bundle(run_dir, tmp_path)
        section = " ".join(_section(readme).split())
        assert "machine that measured" in section
        assert "BACKTRACE" in section
        assert "gdb-backtrace.py" in section

    def test_a_run_with_no_core_gets_no_stray_section(self, run_dir, tmp_path):
        """And gets the README it always got, byte for byte.

        Twelve archived bundles carry the unexcluded text and
        `tests/test_e14_per_pass_manifest.py` measures lines of it. An empty
        heading, or one reading "(none)", would send a reader of the ordinary
        case looking for something that was never taken.
        """
        _members, readme, _detail = _bundle(run_dir, tmp_path)
        assert "NOT IN THIS BUNDLE" not in readme
        assert readme == export_mod.readme_text(RUN_ID, MANIFEST)

    def test_a_truncated_core_is_reported_as_zero_and_not_as_unknown(
            self, run_dir, tmp_path):
        """`_fmt_bytes` answers a falsy count with "unknown". Zero is known.

        The name is the kernel's own default `core_pattern`, a bare `core`,
        which is also the shortest thing the glob has to match.
        """
        _write_core(run_dir, "core", size=0)
        members, readme, _detail = _bundle(run_dir, tmp_path)
        section = _section(readme)
        assert f"{RUN_ID}/core" not in members
        assert "0 B" in section
        assert "unknown" not in section

    def test_the_size_is_the_size_the_file_actually_had(self, run_dir, tmp_path):
        """Reported off the member, so it cannot drift from what was dropped."""
        _write_core(run_dir, "core.4211", size=5 * 1024 ** 2)
        _members, readme, _detail = _bundle(run_dir, tmp_path)
        assert "5.0 MiB" in _section(readme)


class TestTheDetailIsStillAPath:
    def test_the_success_detail_is_the_path_and_nothing_else(
            self, run_dir, tmp_path):
        """And the bundle's size does not belong in it.

        `cli.cmd_export` does `size = os.path.getsize(detail)` and prints the
        size itself, so decorating `detail` -- with a size, or with anything --
        raises `FileNotFoundError` in a caller that has just written a correct
        bundle, and no test in this suite covers `cmd_export`. Measured on
        `38c496a`: patching `write_bundle` to append `"  (4.2 MiB)"` turns
        `cmd_export`'s `rc = 0` into
        `FileNotFoundError: ... 'b.tar.gz  (4.2 MiB)'`, raised from that
        `os.path.getsize` call in `cmd_export`.
        """
        out = str(tmp_path / "bundle.tar.gz")
        ok, detail = export_mod.write_bundle(str(run_dir), out)
        assert ok, detail
        assert detail == out
        assert os.path.getsize(detail) > 0


class TestTheGlobIsTheCaptureSides:
    def test_it_agrees_with_the_constant_capture_matches_on(self):
        """The duplication, recorded where it can be checked.

        `export.CORE_FILE_GLOB` is a copy of `ops_pass.CORE_FILE_GLOB` and must
        stay in step with it: the capture half decides what a core on disk is
        called, and this half decides what does not travel. It is a copy
        because `export.py` is a leaf that imports the standard library and
        nothing else -- `webui/server.py` imports it inside a request handler --
        while `ops_pass` pulls in `docker_ctl`, `engines`, `config` and
        `manifest`.

        This skips until the capture half lands, and becomes a real check the
        moment it does.
        """
        from orchestrator import ops_pass

        theirs = getattr(ops_pass, "CORE_FILE_GLOB", None)
        if theirs is None:
            pytest.skip("ops_pass.CORE_FILE_GLOB arrives with crash capture; "
                        "until then export.py's copy is the only one")
        assert export_mod.CORE_FILE_GLOB == theirs
