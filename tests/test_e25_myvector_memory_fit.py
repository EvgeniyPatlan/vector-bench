"""MyVector's build must fit the server's memory limit, or not be started.

The plugin's HNSW graph is resident hnswlib memory outside the buffer pool,
and its build holds a second copy of every vector while it inserts. Past the
container limit that is an OOM kill mid-build, so the orchestrator refuses
the unit up front -- with a reason that reaches the manifest and the report --
rather than paying for the load and then losing the server.

One estimator (`harness/memory_fit.py`) feeds the refusal, the report and
docs/04's table; the tests below hold all three to it.
"""

from __future__ import annotations

import os
import re
import sys

import pytest

VB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, VB_ROOT)

from harness import memory_fit  # noqa: E402
from harness.metrics import sysinfo  # noqa: E402
from orchestrator import cli  # noqa: E402
from orchestrator.config import (load_engine, load_profile,  # noqa: E402
                                 load_resources, merge_resource_overrides,
                                 resolve_resources)

GIB = 1024 ** 3


@pytest.fixture(scope="module")
def info():
    return sysinfo.collect()


def _resolved(info, resource_pass, profile, engine="myvector"):
    resources = merge_resource_overrides(load_resources(resource_pass), profile)
    return resolve_resources(resources, engine, info)


def _refusal(info, engine, dataset, phase, profile, resource_pass="normalized",
             resources=None):
    """`cli._memory_refusal` against the resolution `cmd_run` would make."""
    if resources is None:
        resources = merge_resource_overrides(load_resources(resource_pass),
                                             profile)
    resolved = resolve_resources(resources, engine, info)
    return cli._memory_refusal(engine, load_engine(engine), dataset, phase,
                               profile, resources, resolved, resource_pass)


def _profile(dataset, ann_m=(16,), ops_m=(16,), **ops):
    return {"name": "e25", "datasets": [dataset],
            "ann": {"m_values": list(ann_m)},
            "ops": {"m_values": list(ops_m), **ops}}


class TestTheEstimator:
    def test_element_size_matches_the_measured_status(self):
        # MYVECTOR_INDEX_STATUS printed `Element Data Size : 3276` at 784 dims
        # and M=16 in the smoke run; the formula adds the stated overhead.
        assert memory_fit.element_bytes(784, 16) == \
            3276 + memory_fit.PER_ELEMENT_OVERHEAD

    @pytest.mark.parametrize("dim,rows,peak_gib", [
        (960, 1_000_000, 12.2),   # gist-960
        (1536, 990_000, 16.4),    # dbpedia-openai-1000k
    ])
    def test_the_figures_the_docs_quote(self, dim, rows, peak_gib):
        buffer = int(16 * GIB * 0.30)
        parts = memory_fit.myvector_build_peak(dim, rows, 16, buffer)
        assert round(parts["peak"] / GIB, 1) == peak_gib


class TestTheRefusal:
    """The three cases the change was specified by, and the edges around them."""

    def test_dbpedia_under_normalized_16_gib_is_refused(self, info):
        profile = _profile("dbpedia-openai-1000k-angular")
        assert _resolved(info, "normalized", profile).server_memory_bytes \
            == 16 * GIB
        for phase in ("ann", "ops"):
            refusal = _refusal(info, "myvector",
                               "dbpedia-openai-1000k-angular", phase, profile)
            assert refusal is not None, phase
            assert refusal["reason"].startswith(
                "does not fit: estimated peak 16.4 GiB > limit 16.0 GiB")
            assert "at M=16 (largest of [16])" in refusal["reason"]

    def test_gist_under_normalized_16_gib_is_allowed(self, info):
        profile = _profile("gist-960-euclidean")
        for phase in ("ann", "ops"):
            assert _refusal(info, "myvector", "gist-960-euclidean", phase,
                            profile) is None, phase

    def test_dbpedia_under_mariadb_blogs_64_gib_is_allowed(self, info):
        profile = load_profile("mariadb-blog")
        pass_ = profile.get("default_resource_pass", "normalized")
        assert _resolved(info, pass_, profile).server_memory_bytes >= 32 * GIB, \
            "mariadb-blog's own 64 GiB override did not reach the resolution"
        for phase in ("ann", "ops"):
            assert _refusal(info, "myvector", "dbpedia-openai-1000k-angular",
                            phase, profile, pass_) is None, phase

    def test_other_engines_are_never_refused(self, info):
        profile = _profile("dbpedia-openai-1000k-angular")
        for engine in ("mariadb", "percona", "pgvector", "valkey"):
            assert _refusal(info, engine, "dbpedia-openai-1000k-angular",
                            "ops", profile) is None, engine

    def test_a_subset_is_estimated_at_its_own_size(self, info):
        profile = _profile("dbpedia-openai-1000k-angular", subset_rows=20_000)
        assert _refusal(info, "myvector", "dbpedia-openai-1000k-angular",
                        "ops", profile) is None

    def test_an_unknown_corpus_is_not_refused(self, info):
        profile = _profile("tiny-16-euclidean")
        assert _refusal(info, "myvector", "tiny-16-euclidean", "ops",
                        profile) is None

    def test_the_default_budget_is_untouched(self):
        assert load_resources("normalized")["memory"]["server_limit_gb"] == 16

    def test_the_engine_config_opts_in(self):
        assert load_engine("myvector")["capabilities"]["memory_fit"] == "myvector"


class TestTheMValuesAndRowsArePerPhase:
    """gist-960 under a 10.62 GiB limit: M=8 and M=16 fit and M=32 does not
    (`test_the_limit_sits_between_m16_and_m32` proves the premise), so a
    `max` replaced by `min` or a constant fails here."""

    LIMIT_GB = 10.62    # between the limits gist-960 needs at M=16 and at M=32

    def _tight(self, profile):
        profile = dict(profile)
        profile["resources"] = {"memory": {"server_limit_gb": self.LIMIT_GB}}
        return profile

    def test_the_limit_sits_between_m16_and_m32(self):
        buffer = int(self.LIMIT_GB * GIB * 0.30)
        at = {m: memory_fit.myvector_build_peak(960, 1_000_000, m, buffer)["peak"]
              for m in (8, 16, 32)}
        limit = self.LIMIT_GB * GIB
        assert at[8] <= limit and at[16] <= limit < at[32], \
            {m: v / GIB for m, v in at.items()}

    def test_the_largest_ann_m_decides(self, info):
        profile = self._tight(_profile("gist-960-euclidean", ann_m=(8, 16, 32)))
        refusal = _refusal(info, "myvector", "gist-960-euclidean", "ann",
                           profile)
        assert refusal is not None, "max(M) not used: [8, 16, 32] fitted"
        assert "at M=32 (largest of [8, 16, 32])" in refusal["reason"]
        fits = self._tight(_profile("gist-960-euclidean", ann_m=(8, 16)))
        assert _refusal(info, "myvector", "gist-960-euclidean", "ann",
                        fits) is None

    def test_ann_does_not_read_ops_m_and_ops_does_not_read_ann_m(self, info):
        ann_big = self._tight(_profile("gist-960-euclidean", ann_m=(32,),
                                       ops_m=(16,)))
        assert _refusal(info, "myvector", "gist-960-euclidean", "ann",
                        ann_big) is not None
        assert _refusal(info, "myvector", "gist-960-euclidean", "ops",
                        ann_big) is None
        ops_big = self._tight(_profile("gist-960-euclidean", ann_m=(16,),
                                       ops_m=(32,)))
        assert _refusal(info, "myvector", "gist-960-euclidean", "ann",
                        ops_big) is None
        assert _refusal(info, "myvector", "gist-960-euclidean", "ops",
                        ops_big) is not None

    def test_subset_rows_bind_ops_only(self, info):
        profile = _profile("dbpedia-openai-1000k-angular", subset_rows=20_000)
        assert _refusal(info, "myvector", "dbpedia-openai-1000k-angular",
                        "ops", profile) is None
        assert _refusal(info, "myvector", "dbpedia-openai-1000k-angular",
                        "ann", profile) is not None

    def test_a_tuned_m_values_by_engine_override_is_honoured(self, info):
        from orchestrator import ann_pass
        profile = self._tight(_profile("gist-960-euclidean", ann_m=(16,)))
        resources = merge_resource_overrides(load_resources("tuned"), profile)
        resources["memory"]["server_limit_gb"] = self.LIMIT_GB
        resources.setdefault("extras", {})["m_values_by_engine"] = {
            "myvector": [32]}
        assert ann_pass.ann_m_values("myvector", profile, resources,
                                     "tuned") == [32]
        refusal = _refusal(info, "myvector", "gist-960-euclidean", "ann",
                           profile, "tuned", resources)
        assert refusal is not None
        assert "at M=32 (largest of [32])" in refusal["reason"]

    def test_render_config_and_the_check_share_one_m_function(self):
        source = open(os.path.join(VB_ROOT, "orchestrator", "ann_pass.py")).read()
        body = source.split("def render_config(", 1)[1].split("\ndef ", 1)[0]
        assert "ann_m_values(engine, profile, resources, resource_pass)" in body


class TestTheDocsStateTheTwoLimitsOfTheCheck:
    def test_whole_unit_and_server_limit_only_are_stated(self):
        doc = open(os.path.join(VB_ROOT, "docs", "04-engine-notes.md")).read()
        flat = " ".join(doc.split())
        assert "A unit is refused as a whole when its largest M does not fit" \
            in flat
        assert "The check compares against the server container's limit only" \
            in flat


class TestADisabledPhaseIsNotRefused:
    def test_ann_disabled(self, info):
        profile = _profile("dbpedia-openai-1000k-angular")
        profile["ann"]["enabled"] = False
        assert _refusal(info, "myvector", "dbpedia-openai-1000k-angular",
                        "ann", profile) is None
        assert _refusal(info, "myvector", "dbpedia-openai-1000k-angular",
                        "ops", profile) is not None

    def test_ops_disabled(self, info):
        profile = _profile("dbpedia-openai-1000k-angular")
        profile["ops"]["enabled"] = False
        assert _refusal(info, "myvector", "dbpedia-openai-1000k-angular",
                        "ops", profile) is None


class TestTheRunLoopRefusesForReal:
    """`cmd_run` driven end to end with `_run_unit` stubbed: the refused unit
    is never started, carries no checkpoint, and the rest of the run goes on.
    The previous source grep passed with the `continue` deleted."""

    DATASET = "dbpedia-openai-1000k-angular"

    def _run(self, monkeypatch, tmp_path, capsys):
        import argparse
        from orchestrator import cli as cli_mod
        root = str(tmp_path / "vb")
        real_paths = cli_mod.paths_for
        monkeypatch.setattr(cli_mod, "paths_for", lambda run_id: {
            k: v.replace(VB_ROOT, root) if isinstance(v, str) else v
            for k, v in real_paths(run_id).items()})
        profile = {"name": "e25run", "description": "", "k": 10,
                   "datasets": [self.DATASET],
                   "ann": {"m_values": [16]},
                   "ops": {"m_values": [16], "workloads": ["build"]}}
        monkeypatch.setattr(cli_mod, "load_profile", lambda name: profile)
        monkeypatch.setattr(cli_mod.docker_ctl, "docker_available",
                            lambda: True)
        monkeypatch.setattr(cli_mod.docker_ctl, "image_id", lambda ref: "x")
        monkeypatch.setattr(cli_mod.docker_ctl, "image_exists",
                            lambda ref: False)
        monkeypatch.setattr(cli_mod, "print_load_estimate",
                            lambda *a, **k: None)
        monkeypatch.setattr(cli_mod, "_check_free_disk", lambda *a: False)
        ran = []

        def _unit(phase, engine, dataset, *a, **k):
            ran.append((engine, dataset, phase))
            return 0
        monkeypatch.setattr(cli_mod, "_run_unit", _unit)
        args = argparse.Namespace(
            profile="e25run", engines="myvector,pgvector", datasets=None,
            resource_pass="normalized", phases="both",
            run_id="e25run-20260101-000000", resume=False, force=False,
            fail_fast=False, no_report=True)
        paths = cli_mod.paths_for(args.run_id)
        os.makedirs(paths["datasets"], exist_ok=True)
        open(os.path.join(paths["datasets"], f"{self.DATASET}.hdf5"),
             "wb").close()
        rc = cli_mod.cmd_run(args)
        return rc, ran, paths, capsys.readouterr()

    def test_refused_units_never_run_and_others_do(self, monkeypatch,
                                                    tmp_path, capsys):
        import json
        rc, ran, paths, out = self._run(monkeypatch, tmp_path, capsys)
        assert rc == 0
        assert not [u for u in ran if u[0] == "myvector"], ran
        assert sorted(ran) == [("pgvector", self.DATASET, "ann"),
                               ("pgvector", self.DATASET, "ops")]
        manifest = json.load(open(os.path.join(paths["run_dir"],
                                               "run-manifest.json")))
        mine = [p for p in manifest["phases"] if p["engine"] == "myvector"]
        assert {p["phase"] for p in mine} == {"ann", "ops"}
        assert all(p["status"] == "refused" for p in mine), mine
        assert all(p["reason"].startswith("does not fit") for p in mine)
        assert manifest["status"] == "completed"
        checkpoints = ""
        if os.path.exists(paths["checkpoints"]):
            checkpoints = open(paths["checkpoints"]).read()
        assert "myvector" not in checkpoints
        assert "pgvector" in checkpoints

    def test_the_finish_banner_summarises_the_refusals(self, monkeypatch,
                                                       tmp_path, capsys):
        rc, ran, paths, out = self._run(monkeypatch, tmp_path, capsys)
        text = out.out + out.err
        finished = text.index("finished ===")
        summary = text[finished:]
        assert ("2 unit(s) refused before they ran: "
                f"normalized/myvector/{self.DATASET}/ann, "
                f"normalized/myvector/{self.DATASET}/ops") in summary
        assert "see the report's Validity" in summary


class TestTheReportStatesIt:
    def _page(self):
        from report import generate, render
        manifest = {"run_id": "e25", "status": "completed",
                    "config": {"profile": {"name": "e25", "ops": {
                        "workloads": ["build"]}}},
                    "engines": {},
                    "phases": [{
                        "phase": "ops", "engine": "myvector",
                        "dataset": "dbpedia-openai-1000k-angular",
                        "status": "refused", "resource_pass": "normalized",
                        "reason": memory_fit.refusal_reason(
                            int(16.4 * GIB), 16 * GIB)}]}
        records = [{"run_id": "e25", "engine": "pgvector", "phase": "concurrency",
                    "dataset": "dbpedia-openai-1000k-angular", "m": 16,
                    "build_mode": "post", "resource_pass": "normalized",
                    "k": 10, "clients": 1, "qps": 1.0, "recall_at_k": 0.9,
                    "ef_search": 40, "timestamp": "2026-10-30T00:00:00Z"}]
        summary = generate.summarize(records, manifest)
        return summary, render.render_markdown(manifest, summary, records, {},
                                               "E25")

    def test_validity_names_the_refused_unit_and_why(self):
        summary, page = self._page()
        validity = page.split("## 2. Validity", 1)[1].split("\n---\n", 1)[0]
        assert "Units refused before they ran" in validity
        assert "does not fit: estimated peak 16.4 GiB > limit 16.0 GiB" in validity
        assert "dbpedia-openai-1000k-angular" in validity

    def test_a_refusal_is_not_reported_as_a_failure(self):
        summary, page = self._page()
        assert summary["failed_phases"] == []
        assert len(summary["refused_phases"]) == 1
        assert "FAILED" not in page.split("## 3.", 1)[0]


class TestTheDocsQuoteTheEstimator:
    def test_docs_04_table_matches_the_function(self):
        doc = open(os.path.join(VB_ROOT, "docs", "04-engine-notes.md")).read()
        buffer = int(16 * GIB * 0.30)
        for name, dim, rows in (("gist-960", 960, 1_000_000),
                                ("dbpedia-openai-1000k", 1536, 990_000)):
            row = next(line for line in doc.splitlines()
                       if line.startswith(f"| {name}"))
            cells = [c.strip() for c in row.strip("|").split("|")]
            parts = memory_fit.myvector_build_peak(dim, rows, 16, buffer)
            want = [memory_fit.gib(parts[k]) for k in
                    ("graph", "transient", "buffer", "peak")]
            got = [re.sub(r"[*~]", "", c) for c in cells[1:5]]
            assert got == want, (name, got, want)


class TestAStaleAnnResultStillGetsTheFootnote:
    """An ann result cached before 970a55a has no query-path attributes."""

    def _ann(self, **extra):
        return {"run_id": "e25", "engine": "myvector", "phase": "recall_qps",
                "dataset": "glove-100-angular", "metric_space": "angular",
                "m": 16, "resource_pass": "normalized", "k": 10,
                "recall_at_k": 0.95, "qps": 3000.0, "extra": dict(extra)}

    def test_the_footnote_is_derived_and_the_record_flagged(self):
        from report import disclosures
        text = disclosures.structural_lines([self._ann()], str)
        assert "2 statements per query" in text
        assert "scalar (computeCosineDistance)" in text
        assert "1 recall/QPS result(s) predate these attributes" in text

    def test_a_current_record_is_not_flagged(self):
        from report import disclosures
        rec = self._ann(round_trips_per_query=2,
                        distance_impl="scalar (computeCosineDistance)")
        text = disclosures.structural_lines([rec], str)
        assert "2 statements per query" in text
        assert "predate" not in text

    def test_euclidean_stale_record_is_not_called_scalar(self):
        from report import disclosures
        rec = self._ann()
        rec["metric_space"] = "euclidean"
        text = disclosures.structural_lines([rec], str)
        assert "computes this corpus's distance" not in text

    def test_other_engines_ann_records_get_nothing(self):
        from report import disclosures
        rec = self._ann()
        rec["engine"] = "mariadb"
        assert disclosures.structural_lines([rec], str) == ""

    def test_the_three_distance_tables_agree(self):
        from harness.drivers.myvector import METRIC_NAMES, MyVectorDriver
        from report import disclosures
        derived = disclosures.ENGINE_QUERY_PATH_FACTS["myvector"]
        assert derived["round_trips_per_query"] == 2
        assert derived["distance_impl"] == {
            metric: MyVectorDriver.DISTANCE_IMPL[dist]
            for metric, dist in METRIC_NAMES.items()}
