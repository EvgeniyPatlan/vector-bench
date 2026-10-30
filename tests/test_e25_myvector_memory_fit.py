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


def _profile(dataset, **ops):
    return {"name": "e25", "datasets": [dataset],
            "ann": {"m_values": [16]},
            "ops": {"m_values": [16], **ops}}


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
    """The three cases the change was specified by."""

    def test_dbpedia_under_normalized_16_gib_is_refused(self, info):
        profile = _profile("dbpedia-openai-1000k-angular")
        resolved = _resolved(info, "normalized", profile)
        assert resolved.server_memory_bytes == 16 * GIB
        for phase in ("ann", "ops"):
            refusal = cli._memory_refusal(
                "myvector", load_engine("myvector"),
                "dbpedia-openai-1000k-angular", phase, profile, resolved)
            assert refusal is not None, phase
            assert refusal["reason"].startswith(
                "does not fit: estimated peak 16.4 GiB > limit 16.0 GiB")

    def test_gist_under_normalized_16_gib_is_allowed(self, info):
        profile = _profile("gist-960-euclidean")
        resolved = _resolved(info, "normalized", profile)
        for phase in ("ann", "ops"):
            assert cli._memory_refusal(
                "myvector", load_engine("myvector"), "gist-960-euclidean",
                phase, profile, resolved) is None, phase

    def test_dbpedia_under_mariadb_blogs_64_gib_is_allowed(self, info):
        profile = load_profile("mariadb-blog")
        resolved = _resolved(info, profile.get("default_resource_pass",
                                               "normalized"), profile)
        assert resolved.server_memory_bytes >= 32 * GIB, \
            "mariadb-blog's own 64 GiB override did not reach the resolution"
        assert cli._memory_refusal(
            "myvector", load_engine("myvector"),
            "dbpedia-openai-1000k-angular", "ops", profile, resolved) is None

    def test_other_engines_are_never_refused(self, info):
        profile = _profile("dbpedia-openai-1000k-angular")
        for engine in ("mariadb", "percona", "pgvector", "valkey"):
            resolved = _resolved(info, "normalized", profile, engine)
            assert cli._memory_refusal(
                engine, load_engine(engine), "dbpedia-openai-1000k-angular",
                "ops", profile, resolved) is None, engine

    def test_a_subset_is_estimated_at_its_own_size(self, info):
        profile = _profile("dbpedia-openai-1000k-angular", subset_rows=20_000)
        resolved = _resolved(info, "normalized", profile)
        assert cli._memory_refusal(
            "myvector", load_engine("myvector"),
            "dbpedia-openai-1000k-angular", "ops", profile, resolved) is None

    def test_an_unknown_corpus_is_not_refused(self, info):
        profile = _profile("tiny-16-euclidean")
        resolved = _resolved(info, "normalized", profile)
        assert cli._memory_refusal(
            "myvector", load_engine("myvector"), "tiny-16-euclidean", "ops",
            profile, resolved) is None

    def test_the_default_budget_is_untouched(self):
        assert load_resources("normalized")["memory"]["server_limit_gb"] == 16

    def test_the_engine_config_opts_in(self):
        assert load_engine("myvector")["capabilities"]["memory_fit"] == "myvector"


class TestTheRunLoopRecordsIt:
    def test_the_phase_loop_consults_the_check_and_records_refused(self):
        source = open(os.path.join(VB_ROOT, "orchestrator", "cli.py")).read()
        loop = source.split("def cmd_run", 1)[1].split("\ndef ", 1)[0]
        assert "_memory_refusal(" in loop
        assert '"refused"' in loop


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
