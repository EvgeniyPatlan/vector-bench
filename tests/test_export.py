"""Tests for packaging a run to send to someone.

A bundle that arrives without its provenance is not worth sending: the whole
point of this framework is that a number means something only alongside the
machine and configuration that produced it.
"""

from __future__ import annotations

import json
import os
import sys
import tarfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator import export as export_mod  # noqa: E402

MANIFEST = {
    "run_id": "r1",
    "status": "completed",
    "started_at": "2026-08-04T05:50:09Z",
    "engines": {"mariadb": {"build": {"tag": "mariadb-11.8.8", "march": "x86-64-v3"}}},
    "host": {"hostname": "bench-rig-2", "kernel": "6.8.0", "total_ram_bytes": 64 * 1024 ** 3,
             "cpu": {"model": "AMD EPYC 9554", "physical_cores": 64, "logical_cpus": 128,
                     "has_avx512": True, "simd_flags": ["avx2", "avx512f"]}},
    "config": {"resource_pass": "normalized",
               "profile": {"name": "main", "description": "the real one",
                           "datasets": ["glove-100-angular"]}},
}


@pytest.fixture
def run_dir(tmp_path):
    run = tmp_path / "results" / "r1"
    (run / "report" / "charts").mkdir(parents=True)
    (run / "run-manifest.json").write_text(json.dumps(MANIFEST))
    (run / "report" / "report.html").write_text("<html>report</html>")
    (run / "report" / "records.jsonl").write_text('{"phase":"ingest"}\n')
    (run / "ops-mariadb-d1-normalized-m16-post.jsonl").write_text("{}\n")
    return str(run)


class TestReadme:
    def test_names_the_machine_the_numbers_belong_to(self):
        text = export_mod.readme_text("r1", MANIFEST)
        assert "bench-rig-2" in text
        assert "AMD EPYC 9554" in text
        assert "64 physical / 128 logical" in text

    def test_states_the_avx512_caveat(self):
        text = export_mod.readme_text("r1", MANIFEST)
        assert "AVX-512" in text
        assert "do not transfer" in text

    def test_says_how_to_read_it_without_the_framework(self):
        text = export_mod.readme_text("r1", MANIFEST)
        assert "report/report.html" in text
        assert "self-contained" in text

    def test_warns_that_regenerating_is_not_the_same(self):
        """The recall data lives outside the run directory and does not travel."""
        # The README is hard-wrapped prose, so match on the words, not the lines.
        text = " ".join(export_mod.readme_text("r1", MANIFEST).split())
        assert "results/annb/" in text
        assert "loses its recall section" in text
        assert "reads those instead" in text

    def test_lists_the_engines_and_their_tags(self):
        text = export_mod.readme_text("r1", MANIFEST)
        assert "mariadb-11.8.8" in text
        assert "-march=x86-64-v3" in text

    def test_survives_a_sparse_manifest(self):
        text = export_mod.readme_text("r1", {})
        assert "unknown" in text
        assert "r1" in text


class TestBundle:
    def test_writes_an_archive(self, run_dir, tmp_path):
        out = str(tmp_path / "b.tar.gz")
        ok, detail = export_mod.write_bundle(run_dir, out)
        assert ok, detail
        assert tarfile.is_tarfile(out)

    def test_contains_the_run_and_a_readme(self, run_dir, tmp_path):
        out = str(tmp_path / "b.tar.gz")
        export_mod.write_bundle(run_dir, out)
        with tarfile.open(out) as archive:
            names = archive.getnames()
        assert "r1/README.txt" in names
        assert "r1/run-manifest.json" in names
        assert "r1/report/report.html" in names
        assert "r1/report/records.jsonl" in names

    def test_everything_is_under_one_directory(self, run_dir, tmp_path):
        """So extracting into results/ cannot scatter files."""
        out = str(tmp_path / "b.tar.gz")
        export_mod.write_bundle(run_dir, out)
        with tarfile.open(out) as archive:
            assert all(n == "r1" or n.startswith("r1/") for n in archive.getnames())

    def test_readme_is_readable_from_the_archive(self, run_dir, tmp_path):
        out = str(tmp_path / "b.tar.gz")
        export_mod.write_bundle(run_dir, out)
        with tarfile.open(out) as archive:
            text = archive.extractfile("r1/README.txt").read().decode()
        assert "bench-rig-2" in text

    def test_refuses_a_run_without_a_manifest(self, tmp_path):
        bare = tmp_path / "results" / "nomanifest"
        bare.mkdir(parents=True)
        ok, detail = export_mod.write_bundle(str(bare), str(tmp_path / "b.tar.gz"))
        assert not ok
        assert "provenance" in detail

    def test_refuses_a_missing_directory(self, tmp_path):
        ok, detail = export_mod.write_bundle(str(tmp_path / "nope"),
                                             str(tmp_path / "b.tar.gz"))
        assert not ok and "no such run directory" in detail

    def test_filename_carries_the_run_id(self):
        assert export_mod.bundle_filename("main-20260825") == \
            "vector-bench-main-20260825.tar.gz"
