"""Tests for importing a run measured elsewhere.

Extraction is the part worth testing hard. A tar archive can name absolute
paths, walk out with .., or carry a symlink pointing anywhere, and the image
this runs in has Python 3.11 without tarfile.data_filter — so every member is
checked here rather than trusted to a version-dependent guard.
"""

from __future__ import annotations

import io
import json
import os
import sys
import tarfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from webui import importing as importing_mod  # noqa: E402

MANIFEST = {"run_id": "r1", "status": "completed",
            "host": {"hostname": "rig-2", "cpu": {"model": "EPYC"}},
            "config": {"profile": {"name": "main"}}}


def build_archive(path, entries):
    with tarfile.open(path, "w:gz") as archive:
        for name, kind, *rest in entries:
            info = tarfile.TarInfo(name)
            if kind == "file":
                payload = (rest[0] if rest else "{}").encode()
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
            elif kind == "dir":
                info.type = tarfile.DIRTYPE
                archive.addfile(info)
            elif kind == "sym":
                info.type = tarfile.SYMTYPE
                info.linkname = rest[0] if rest else "/etc/passwd"
                archive.addfile(info)
            elif kind == "link":
                info.type = tarfile.LNKTYPE
                info.linkname = rest[0]
                archive.addfile(info)
            elif kind == "fifo":
                info.type = tarfile.FIFOTYPE
                archive.addfile(info)
    return path


def good_archive(tmp_path, top="r1"):
    return build_archive(str(tmp_path / "b.tar.gz"), [
        (f"{top}/", "dir"),
        (f"{top}/run-manifest.json", "file", json.dumps(MANIFEST)),
        (f"{top}/report/", "dir"),
        (f"{top}/report/report.html", "file", "<html>ok</html>"),
        (f"{top}/report/records.jsonl", "file", '{"phase":"ingest"}'),
    ])


class TestArchiveIsRefusedWhenHostile:
    @pytest.mark.parametrize("entries,expected", [
        ([("/etc/passwd", "file")], "not a usable run id"),
        ([("r1/../../evil", "file")], "escapes"),
        ([("r1/run-manifest.json", "file"), ("r1/x", "sym")], "link"),
        ([("r1/run-manifest.json", "file"), ("r1/x", "link", "r1/y")], "link"),
        ([("r1/run-manifest.json", "file"), ("r1/x", "fifo")], "device or fifo"),
        ([("r1/run-manifest.json", "file"), ("other/x", "file")], "more than one"),
        ([("r1/data.jsonl", "file")], "run-manifest.json"),
    ])
    def test_rejected(self, tmp_path, entries, expected):
        path = build_archive(str(tmp_path / "bad.tar.gz"), entries)
        with pytest.raises(importing_mod.RejectedArchive) as excinfo:
            importing_mod.inspect(path)
        assert expected in str(excinfo.value)

    def test_not_a_tar_at_all(self, tmp_path):
        path = tmp_path / "nope.tar.gz"
        path.write_bytes(b"this is not a tar")
        with pytest.raises(importing_mod.RejectedArchive):
            importing_mod.inspect(str(path))

    def test_empty_archive(self, tmp_path):
        path = build_archive(str(tmp_path / "empty.tar.gz"), [])
        with pytest.raises(importing_mod.RejectedArchive):
            importing_mod.inspect(str(path))

    def test_nothing_escaped_onto_disk(self, tmp_path):
        """A refused archive must leave no trace, not a partial extraction."""
        results = tmp_path / "results"
        results.mkdir()
        path = build_archive(str(tmp_path / "bad.tar.gz"),
                             [("r1/run-manifest.json", "file"), ("r1/x", "sym")])
        run_id, errors = importing_mod.import_bundle(str(results), path)
        assert run_id is None and errors
        assert os.listdir(results) == []


class TestImport:
    def test_unpacks_under_its_own_name(self, tmp_path):
        results = tmp_path / "results"
        results.mkdir()
        run_id, errors = importing_mod.import_bundle(
            str(results), good_archive(tmp_path))
        assert errors == [] and run_id == "r1"
        assert (results / "r1" / "run-manifest.json").is_file()
        assert (results / "r1" / "report" / "report.html").is_file()

    def test_can_be_renamed_on_the_way_in(self, tmp_path):
        """Two machines running the same profile produce the same run id."""
        results = tmp_path / "results"
        results.mkdir()
        run_id, errors = importing_mod.import_bundle(
            str(results), good_archive(tmp_path), run_id="rig2-main")
        assert errors == [] and run_id == "rig2-main"
        assert (results / "rig2-main" / "run-manifest.json").is_file()

    def test_refuses_to_overwrite(self, tmp_path):
        results = tmp_path / "results"
        results.mkdir()
        importing_mod.import_bundle(str(results), good_archive(tmp_path))
        _run_id, errors = importing_mod.import_bundle(
            str(results), good_archive(tmp_path))
        assert any("already exists" in e for e in errors)

    @pytest.mark.parametrize("name", ["../evil", "with space", "x" * 70, "a/b"])
    def test_refuses_an_unusable_name(self, tmp_path, name):
        results = tmp_path / "results"
        results.mkdir()
        _run_id, errors = importing_mod.import_bundle(
            str(results), good_archive(tmp_path), run_id=name)
        assert errors and "usable run id" in errors[0]

    def test_blank_name_uses_the_archive(self, tmp_path):
        results = tmp_path / "results"
        results.mkdir()
        run_id, errors = importing_mod.import_bundle(
            str(results), good_archive(tmp_path), run_id="")
        assert errors == [] and run_id == "r1"

    def test_extracted_files_get_sane_modes(self, tmp_path):
        """An archive's modes are its author's choice, not this machine's."""
        results = tmp_path / "results"
        results.mkdir()
        importing_mod.import_bundle(str(results), good_archive(tmp_path))
        manifest = results / "r1" / "run-manifest.json"
        assert oct(manifest.stat().st_mode)[-3:] == "644"
        assert oct((results / "r1" / "report").stat().st_mode)[-3:] == "755"

    def test_leaves_no_staging_directory_behind(self, tmp_path):
        results = tmp_path / "results"
        results.mkdir()
        importing_mod.import_bundle(str(results), good_archive(tmp_path))
        assert [p for p in os.listdir(results) if p.startswith(".import-")] == []

    def test_records_the_label_and_where_it_came_from(self, tmp_path):
        results = tmp_path / "results"
        results.mkdir()
        importing_mod.import_bundle(str(results), good_archive(tmp_path),
                                    label="EPYC rig", source="bench-rig-2")
        found = importing_mod.read_label(str(results / "r1"))
        assert found["label"] == "EPYC rig"
        assert found["source"] == "bench-rig-2"
        assert found["imported_at"].endswith("Z")

    def test_the_manifest_is_not_edited(self, tmp_path):
        """It is the provenance of a measurement, not a place for notes."""
        results = tmp_path / "results"
        results.mkdir()
        importing_mod.import_bundle(str(results), good_archive(tmp_path),
                                    label="a nickname")
        with open(results / "r1" / "run-manifest.json") as fh:
            assert json.load(fh) == MANIFEST


class TestFreshCheckout:
    """results/ is generated and gitignored, so a new machine has none.

    Creating the staging directory inside it raised FileNotFoundError out of
    import_bundle, killed the request handler, and closed the connection with
    no reply -- which the browser reported as the upload dropping after every
    byte of it had arrived.
    """

    def test_import_creates_results_if_it_is_missing(self, tmp_path):
        results = tmp_path / "results"          # deliberately not created
        assert not results.exists()
        run_id, errors = importing_mod.import_bundle(
            str(results), good_archive(tmp_path))
        assert errors == [] and run_id == "r1"
        assert (results / "r1" / "run-manifest.json").is_file()

    def test_an_unwritable_destination_is_an_error_not_a_crash(self, tmp_path):
        blocked = tmp_path / "blocked"
        blocked.mkdir(mode=0o500)
        try:
            run_id, errors = importing_mod.import_bundle(
                str(blocked / "results"), good_archive(tmp_path))
            assert run_id is None
            assert any("cannot write to" in e for e in errors)
        finally:
            blocked.chmod(0o700)


class TestLabels:
    def test_set_read_and_clear(self, tmp_path):
        run = tmp_path / "r1"
        run.mkdir()
        assert importing_mod.read_label(str(run)) == {}
        importing_mod.set_label(str(run), "nickname", source="rig-2")
        assert importing_mod.read_label(str(run))["label"] == "nickname"
        importing_mod.clear_label(str(run))
        assert importing_mod.read_label(str(run)) == {}

    def test_clearing_what_was_never_set_is_fine(self, tmp_path):
        run = tmp_path / "r1"
        run.mkdir()
        importing_mod.clear_label(str(run))

    def test_a_long_label_is_truncated(self, tmp_path):
        run = tmp_path / "r1"
        run.mkdir()
        importing_mod.set_label(str(run), "x" * 500)
        assert len(importing_mod.read_label(str(run))["label"]) == 200

    def test_unreadable_label_file_is_ignored(self, tmp_path):
        run = tmp_path / "r1"
        run.mkdir()
        (run / importing_mod.LABEL_NAME).write_text("{not json")
        assert importing_mod.read_label(str(run)) == {}

    def test_a_label_reaches_the_run_summary(self, tmp_path):
        from webui import runs as runs_mod
        results = tmp_path / "results"
        results.mkdir()
        importing_mod.import_bundle(str(results), good_archive(tmp_path),
                                    label="EPYC rig", source="bench-rig-2")
        summary = runs_mod.discover_runs(str(results))[0]
        assert summary["label"] == "EPYC rig"
        assert summary["source"] == "bench-rig-2"
