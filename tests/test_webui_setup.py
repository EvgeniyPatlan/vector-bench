"""Tests for the setup checklist.

The step order is the point: images, a corpus, the smoke gate, then a
measurement. The -march check is the one that earns its keep — every engine
compiles SIMD distance kernels, so a mixed build compares compiler flags and
nothing about the resulting numbers looks wrong.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from webui import setup as setup_mod  # noqa: E402


def engine(name, built=True, march="x86-64-v3", group="original"):
    return {"name": name, "label": name, "color": "#000", "group": group,
            "tag": "t", "runtime_built": built, "bench_built": built,
            "march": march}


class TestMarchAgreement:
    def test_agreed_when_all_built_images_match(self):
        found = setup_mod.march_in_use([engine("a"), engine("b")])
        assert found["agreed"] == "x86-64-v3"
        assert found["mixed"] is False

    def test_mixed_is_flagged(self):
        found = setup_mod.march_in_use([engine("a"), engine("b", march="native")])
        assert found["mixed"] is True
        assert found["values"] == ["native", "x86-64-v3"]

    def test_unbuilt_images_do_not_count(self):
        """A missing image has no -march to disagree with."""
        found = setup_mod.march_in_use(
            [engine("a"), engine("b", built=False, march="native")])
        assert found["agreed"] == "x86-64-v3"
        assert found["mixed"] is False

    def test_nothing_built_agrees_on_nothing(self):
        found = setup_mod.march_in_use([engine("a", built=False)])
        assert found["agreed"] is None and found["mixed"] is False


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "datasets").mkdir()
    (tmp_path / "sources").mkdir()
    return tmp_path


def plan_for(workspace, monkeypatch, engines, runs=()):
    monkeypatch.setattr(setup_mod, "engine_state", lambda root: engines)
    from webui import runs as runs_mod
    monkeypatch.setattr(runs_mod, "discover_runs", lambda d: list(runs))
    return setup_mod.plan(str(workspace), str(workspace / "results"),
                          str(workspace / "datasets"))


class TestPlan:
    def test_step_order_is_the_order_you_do_them(self, workspace, monkeypatch):
        found = plan_for(workspace, monkeypatch, [engine("a")])
        assert [s["id"] for s in found["steps"]] == \
            ["images", "datasets", "smoke", "measure"]

    def test_next_is_the_first_unfinished_step(self, workspace, monkeypatch):
        found = plan_for(workspace, monkeypatch, [engine("a", built=False)])
        assert found["next"] == "images"

    def test_images_done_when_the_original_three_are_built(self, workspace, monkeypatch):
        engines = [engine("a"), engine("b"),
                   engine("c", built=False, group="extra")]
        found = plan_for(workspace, monkeypatch, engines)
        assert found["steps"][0]["done"] is True, "an unbuilt extra must not block"

    def test_images_not_done_when_an_original_is_missing(self, workspace, monkeypatch):
        found = plan_for(workspace, monkeypatch,
                         [engine("a"), engine("b", built=False)])
        assert found["steps"][0]["done"] is False

    def test_a_synthetic_corpus_does_not_count_as_a_dataset(self, workspace, monkeypatch):
        """tiny-* prove the framework works and measure nothing about an engine."""
        (workspace / "datasets" / "tiny-16-euclidean.hdf5").write_bytes(b"x")
        found = plan_for(workspace, monkeypatch, [engine("a")])
        assert found["steps"][1]["done"] is False

    def test_a_real_corpus_counts(self, workspace, monkeypatch):
        (workspace / "datasets" / "fashion-mnist-784-euclidean.hdf5").write_bytes(b"x")
        found = plan_for(workspace, monkeypatch, [engine("a")])
        step = found["steps"][1]
        assert step["done"] is True
        assert step["detail"]["smoke_dataset_present"] is True

    def test_smoke_step_needs_a_completed_smoke_run(self, workspace, monkeypatch):
        found = plan_for(workspace, monkeypatch, [engine("a")],
                         runs=[{"dir_name": "r1", "profile": "smoke",
                                "status": "completed"}])
        assert found["steps"][2]["done"] is True

    def test_a_failed_smoke_run_does_not_count(self, workspace, monkeypatch):
        found = plan_for(workspace, monkeypatch, [engine("a")],
                         runs=[{"dir_name": "r1", "profile": "smoke",
                                "status": "completed_with_failures"}])
        assert found["steps"][2]["done"] is False

    def test_another_profile_is_not_the_smoke_gate(self, workspace, monkeypatch):
        found = plan_for(workspace, monkeypatch, [engine("a")],
                         runs=[{"dir_name": "r1", "profile": "main",
                                "status": "completed"}])
        assert found["steps"][2]["done"] is False

    def test_measure_is_never_done(self, workspace, monkeypatch):
        """It is the destination, not a box to tick."""
        found = plan_for(workspace, monkeypatch, [engine("a")])
        assert found["steps"][3]["done"] is False

    def test_ready_ignores_the_measure_step(self, workspace, monkeypatch):
        (workspace / "datasets" / "glove-100-angular.hdf5").write_bytes(b"x")
        found = plan_for(workspace, monkeypatch, [engine("a")],
                         runs=[{"dir_name": "r1", "profile": "smoke",
                                "status": "completed"}])
        assert found["ready"] is True

    def test_disk_shortfall_is_reported(self, workspace, monkeypatch):
        found = plan_for(workspace, monkeypatch, [engine("a")])
        assert "free_bytes" in found["disk"]
        assert found["disk"]["wanted_bytes"] == setup_mod.WANTED_DISK_BYTES


class TestEngineState:
    def test_reads_march_from_the_build_record(self, tmp_path, monkeypatch):
        sources = tmp_path / "sources"
        sources.mkdir()
        (sources / "mariadb.image.json").write_text(
            json.dumps({"engine": "mariadb", "march": "native"}))

        from orchestrator import docker_ctl
        from orchestrator import engines as engines_mod
        import orchestrator.config as config_mod

        monkeypatch.setattr(docker_ctl, "image_exists", lambda ref: True)
        monkeypatch.setattr(config_mod, "load_engine",
                            lambda n: {"image": {"runtime": "r", "bench": "b"}})
        monkeypatch.setattr(engines_mod, "registry",
                            lambda refresh=False: {"mariadb": engines_mod.EngineRuntime(
                                name="mariadb", display_name="MariaDB", driver="D",
                                ann_constructor="C", port=1, data_mount="/d",
                                server_data_mount=None, user="u", password="p",
                                group="original", order=1, color="#000", marker="o",
                                linestyle="-", chart_label="MariaDB", label="MariaDB",
                                tag="t", probe=("sh",))})
        found = setup_mod.engine_state(str(tmp_path))
        assert found[0]["march"] == "native"
        assert found[0]["bench_built"] is True
