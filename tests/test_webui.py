"""Tests for the web UI: discovery, record access, API dispatch and the server.

Fixtures build synthetic run directories rather than reading results/, which is
generated and gitignored, so the suite behaves the same on a fresh checkout.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import urllib.error
import urllib.request

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from webui import api as api_mod  # noqa: E402
from webui import records as records_mod  # noqa: E402
from webui import runs as runs_mod  # noqa: E402
from webui import server as server_mod  # noqa: E402


def write_run(results_dir, run_id, *, manifest=None, records=None, ops=None,
              report_html=False):
    run_dir = os.path.join(results_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    if manifest is not None:
        with open(os.path.join(run_dir, "run-manifest.json"), "w") as fh:
            json.dump(manifest, fh)
    if records is not None:
        os.makedirs(os.path.join(run_dir, "report"), exist_ok=True)
        with open(os.path.join(run_dir, "report", "records.jsonl"), "w") as fh:
            for record in records:
                fh.write(json.dumps(record) + "\n")
    if ops is not None:
        for filename, rows in ops.items():
            with open(os.path.join(run_dir, filename), "w") as fh:
                for row in rows:
                    fh.write(json.dumps(row) + "\n")
    if report_html:
        os.makedirs(os.path.join(run_dir, "report"), exist_ok=True)
        with open(os.path.join(run_dir, "report", "report.html"), "w") as fh:
            fh.write("<html><body>report</body></html>")
    return run_dir


def basic_manifest(run_id, **over):
    manifest = {
        "run_id": run_id,
        "status": "completed",
        "started_at": "2026-08-04T05:50:09Z",
        "finished_at": "2026-08-04T05:52:11Z",
        "engines": {"mariadb": {"build": {"tag": "mariadb-11.8.8", "march": "x86-64-v3"}},
                    "pgvector": {"build": {"tag": "v0.8.6", "march": "x86-64-v3"}}},
        "host": {"cpu": {"model": "Test CPU", "has_avx512": False, "hybrid": True},
                 "total_ram_bytes": 64 * 1024 ** 3},
        "config": {"profile": {"name": "dev", "datasets": ["tiny-16-euclidean"]},
                   "resource_pass": "normalized"},
        "phases": [{"phase": "ops", "engine": "mariadb", "dataset": "tiny-16-euclidean",
                    "resource_pass": "normalized", "status": "completed", "duration_s": 22.0}],
        "warnings": ["no AVX-512"],
    }
    return {**manifest, **over}


SAMPLE_RECORDS = [
    {"engine": "mariadb", "dataset": "d1", "phase": "recall_qps", "m": 16,
     "ef_search": 10, "recall_at_k": 0.91, "qps": 900.0, "resource_pass": "normalized"},
    {"engine": "mariadb", "dataset": "d1", "phase": "recall_qps", "m": 16,
     "ef_search": 40, "recall_at_k": 0.98, "qps": 400.0, "resource_pass": "normalized"},
    {"engine": "pgvector", "dataset": "d1", "phase": "recall_qps", "m": 16,
     "ef_search": 10, "recall_at_k": 0.85, "qps": 1500.0, "resource_pass": "normalized"},
    {"engine": "pgvector", "dataset": "d2", "phase": "index_build", "m": 16,
     "build_wall_s": 120.0, "index_bytes": 1024, "resource_pass": "normalized"},
]


@pytest.fixture
def results_dir(tmp_path):
    root = tmp_path / "results"
    root.mkdir()
    write_run(str(root), "run-a", manifest=basic_manifest("run-a"),
              records=SAMPLE_RECORDS, report_html=True)
    write_run(str(root), "run-b",
              manifest=basic_manifest("run-b", started_at="2026-08-01T00:00:00Z"),
              ops={"ops-mariadb-d1-normalized-m16-post.jsonl": [
                  {"engine": "mariadb", "dataset": "d1", "phase": "ingest",
                   "rows": 4000, "ingest_rows_per_s": 93.9}]})
    # The shared ann tree is not a run: it has no manifest.
    (root / "annb").mkdir()
    return str(root)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

class TestDiscovery:
    def test_finds_runs_and_skips_annb(self, results_dir):
        found = runs_mod.discover_runs(results_dir)
        assert [r["dir_name"] for r in found] == ["run-a", "run-b"]

    def test_newest_first(self, results_dir):
        found = runs_mod.discover_runs(results_dir)
        assert found[0]["started_at"] > found[1]["started_at"]

    def test_missing_results_dir_is_empty(self, tmp_path):
        assert runs_mod.discover_runs(str(tmp_path / "nope")) == []

    def test_summary_fields(self, results_dir):
        summary = next(r for r in runs_mod.discover_runs(results_dir)
                       if r["dir_name"] == "run-a")
        assert summary["engines"] == ["mariadb", "pgvector"]
        assert summary["profile"] == "dev"
        assert summary["resource_pass"] == "normalized"
        assert summary["warning_count"] == 1
        assert summary["has_report"] is True
        assert summary["record_count"] == len(SAMPLE_RECORDS)
        assert summary["duration_s"] == 22.0

    def test_datasets_come_from_phases(self, results_dir):
        summary = next(r for r in runs_mod.discover_runs(results_dir)
                       if r["dir_name"] == "run-a")
        assert summary["datasets"] == ["tiny-16-euclidean"]

    def test_corrupt_manifest_is_skipped(self, results_dir):
        run_dir = os.path.join(results_dir, "broken")
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, "run-manifest.json"), "w") as fh:
            fh.write("{not json")
        assert "broken" not in [r["dir_name"] for r in runs_mod.discover_runs(results_dir)]

    @pytest.mark.parametrize("bad", ["../etc", "a/b", "..", ".", ""])
    def test_resolve_run_dir_rejects_traversal(self, results_dir, bad):
        assert runs_mod.resolve_run_dir(results_dir, bad) is None

    def test_resolve_run_dir_accepts_real_run(self, results_dir):
        assert runs_mod.resolve_run_dir(results_dir, "run-a") is not None


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

class TestRecords:
    def test_prefers_merged_records(self, results_dir):
        recs, source = records_mod.load_records(os.path.join(results_dir, "run-a"))
        assert source == "records.jsonl"
        assert len(recs) == len(SAMPLE_RECORDS)

    def test_falls_back_to_ops_jsonl(self, results_dir):
        recs, source = records_mod.load_records(os.path.join(results_dir, "run-b"))
        assert source == "ops-*.jsonl"
        assert [r["phase"] for r in recs] == ["ingest"]

    def test_facets_drop_nulls_and_sort(self):
        facets = records_mod.facets(SAMPLE_RECORDS)
        assert facets["engine"] == ["mariadb", "pgvector"]
        assert facets["dataset"] == ["d1", "d2"]
        assert "ef_construction" not in facets

    def test_available_measures(self):
        measures = records_mod.available_measures(SAMPLE_RECORDS)
        assert "recall_at_k" in measures and "qps" in measures
        assert "latency_p99_ms" not in measures

    def test_filter_coerces_numeric_strings(self):
        kept = records_mod.filter_records(SAMPLE_RECORDS, {"ef_search": ["10"]})
        assert len(kept) == 2
        assert {r["engine"] for r in kept} == {"mariadb", "pgvector"}

    def test_filter_combines_fields_conjunctively(self):
        kept = records_mod.filter_records(
            SAMPLE_RECORDS, {"engine": ["mariadb"], "phase": ["recall_qps"]})
        assert len(kept) == 2

    def test_empty_selection_keeps_everything(self):
        assert len(records_mod.filter_records(SAMPLE_RECORDS, {"engine": []})) == 4

    def test_unknown_field_is_ignored(self):
        assert len(records_mod.filter_records(SAMPLE_RECORDS, {"nope": ["x"]})) == 4

    def test_series_groups_and_sorts_by_x(self):
        out = records_mod.series(SAMPLE_RECORDS, "recall_at_k", "qps", ("engine",))
        by_key = {s["key"]: s for s in out}
        assert by_key["mariadb"]["x"] == [0.91, 0.98]
        assert by_key["mariadb"]["y"] == [900.0, 400.0]

    def test_series_skips_non_numeric(self):
        out = records_mod.series(SAMPLE_RECORDS, "recall_at_k", "build_wall_s")
        assert all(len(s["x"]) == len(s["y"]) for s in out)
        assert "mariadb" not in {s["key"] for s in out}

    def test_series_multi_field_group(self):
        out = records_mod.series(SAMPLE_RECORDS, "ef_search", "qps",
                                 ("engine", "dataset"))
        assert "mariadb / d1" in {s["key"] for s in out}


# ---------------------------------------------------------------------------
# API dispatch
# ---------------------------------------------------------------------------

@pytest.fixture
def api(results_dir, tmp_path):
    instance = api_mod.Api(str(tmp_path))
    instance.results_dir = results_dir
    return instance


class TestApi:
    def test_list_runs(self, api):
        status, body = api_mod.dispatch(api, "GET", "/api/runs", {})
        assert status == 200 and len(body["runs"]) == 2

    def test_get_run(self, api):
        status, body = api_mod.dispatch(api, "GET", "/api/runs/run-a", {})
        assert status == 200
        assert body["manifest"]["run_id"] == "run-a"
        assert body["summary"]["has_report"] is True

    def test_missing_run_is_404(self, api):
        status, _ = api_mod.dispatch(api, "GET", "/api/runs/nope", {})
        assert status == 404

    def test_traversal_run_id_is_404(self, api):
        status, _ = api_mod.dispatch(api, "GET", "/api/runs/..", {})
        assert status == 404

    def test_facets_endpoint(self, api):
        status, body = api_mod.dispatch(api, "GET", "/api/runs/run-a/facets", {})
        assert status == 200 and body["total"] == 4
        assert body["facets"]["engine"] == ["mariadb", "pgvector"]

    def test_records_filtered(self, api):
        status, body = api_mod.dispatch(
            api, "GET", "/api/runs/run-a/records", {"engine": ["pgvector"]})
        assert status == 200 and body["matched"] == 2
        assert body["truncated"] is False

    def test_series_endpoint(self, api):
        status, body = api_mod.dispatch(
            api, "GET", "/api/runs/run-a/series",
            {"x": ["recall_at_k"], "y": ["qps"], "phase": ["recall_qps"]})
        assert status == 200 and body["x"] == "recall_at_k"
        assert {s["key"] for s in body["series"]} == {"mariadb", "pgvector"}

    def test_series_filters_do_not_leak_axis_params(self, api):
        _status, body = api_mod.dispatch(
            api, "GET", "/api/runs/run-a/series",
            {"x": ["recall_at_k"], "y": ["qps"], "group_by": ["engine"]})
        assert body["matched"] == 4

    def test_unknown_endpoint_is_none(self, api):
        assert api_mod.dispatch(api, "GET", "/api/bogus", {}) is None

    def test_wrong_method_is_405(self, api):
        status, _ = api_mod.dispatch(api, "POST", "/api/runs", {})
        assert status == 405


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

@pytest.fixture
def live_server(results_dir, tmp_path):
    server = server_mod.make_server(str(tmp_path), "127.0.0.1", 0, False)
    server.RequestHandlerClass.api.results_dir = results_dir
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def get(url, headers=None):
    request = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


class TestFrontEndWiring:
    """Element ids are a contract between three files that never import each other.

    A rename in index.html shows up as a silently dead button, not an error.
    """

    STATIC = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "webui", "static")

    def _read(self, name):
        with open(os.path.join(self.STATIC, name)) as fh:
            return fh.read()

    @pytest.mark.parametrize("element_id", [
        "section-list", "run-list", "run-filter",
        "runs-group", "runs-toggle", "runs-body", "runs-count",
        "tabs", "context", "control-badge", "main",
    ])
    def test_ids_used_by_app_js_exist_in_the_page(self, element_id):
        assert f'id="{element_id}"' in self._read("index.html")
        assert element_id in self._read("app.js")

    @pytest.mark.parametrize("panel", [
        "status", "datasets", "overview", "explore", "report",
        "profiles", "engines", "jobs",
    ])
    def test_every_route_has_a_panel(self, panel):
        assert f'id="panel-{panel}"' in self._read("index.html")

    def test_every_script_the_page_loads_exists(self):
        import re
        for src in re.findall(r'<script src="/([^"]+)"', self._read("index.html")):
            assert os.path.isfile(os.path.join(self.STATIC, src)), src

    def test_setup_comes_before_runs_in_the_sidebar(self):
        """Configuration is what you reach for first; results are the archive."""
        page = self._read("index.html")
        assert page.index("Set up") < page.index(">Runs<")


class TestServer:
    def test_index_is_served(self, live_server):
        status, body = get(f"{live_server}/")
        assert status == 200 and b"vector-bench" in body

    def test_static_assets(self, live_server):
        for path in ("/app.js", "/explore.js", "/style.css",
                     "/vendor/uPlot.iife.min.js"):
            status, _ = get(live_server + path)
            assert status == 200, path

    def test_health_over_http(self, live_server):
        status, body = get(f"{live_server}/api/health")
        payload = json.loads(body)
        assert status == 200 and payload["ok"] is True
        assert payload["auth_enabled"] is False
        assert payload["control_enabled"] is False

    def test_api_over_http(self, live_server):
        status, body = get(f"{live_server}/api/runs")
        assert status == 200 and len(json.loads(body)["runs"]) == 2

    def test_run_report_is_served(self, live_server):
        status, body = get(f"{live_server}/runs/run-a/report/report.html")
        assert status == 200 and b"report" in body

    def test_report_of_unknown_run_is_404(self, live_server):
        status, _ = get(f"{live_server}/runs/nope/report/report.html")
        assert status == 404

    def test_path_traversal_is_refused(self, live_server):
        for path in ("/../../etc/passwd", "/runs/run-a/../../../etc/passwd"):
            status, _ = get(live_server + path)
            assert status == 404, path

    def test_foreign_host_header_is_refused(self, live_server):
        status, _ = get(f"{live_server}/api/health",
                        headers={"Host": "evil.example.com"})
        assert status == 403

    def test_unknown_path_is_404(self, live_server):
        status, _ = get(f"{live_server}/nope")
        assert status == 404
