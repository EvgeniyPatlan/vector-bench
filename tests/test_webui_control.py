"""Tests for the control surface: profile writing, job validation, guards.

The launch path is exercised with a harmless command rather than a real
benchmark, so the suite stays fast and starts no containers.
"""

from __future__ import annotations

import json
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from webui import api as api_mod  # noqa: E402
from webui import control as control_mod  # noqa: E402
from webui import jobs as jobs_mod  # noqa: E402
from webui import profiles as profiles_mod  # noqa: E402

# Registering twice would duplicate routes across test modules.
if not any(p.pattern == r"^/api/jobs$" for _m, p, _h in api_mod.ROUTES):
    api_mod.extend(control_mod.ROUTES)


VALID = """name: sample
description: a sample
datasets:
  - tiny-16-euclidean
k: 10
ann:
  enabled: false
ops:
  enabled: true
  m_values: [16]
"""


@pytest.fixture
def root(tmp_path):
    os.makedirs(tmp_path / "config" / "profiles")
    os.makedirs(tmp_path / "results")
    os.makedirs(tmp_path / "datasets")
    with open(tmp_path / "config" / "profiles" / "sample.yml", "w") as fh:
        fh.write(VALID)
    return str(tmp_path)


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

class TestProfiles:
    def test_list(self, root):
        listed = profiles_mod.list_profiles(os.path.join(root, "config"))
        assert [p["name"] for p in listed] == ["sample"]
        assert listed[0]["datasets"] == ["tiny-16-euclidean"]
        assert listed[0]["ann_enabled"] is False

    def test_read_roundtrip(self, root):
        got = profiles_mod.read_profile(os.path.join(root, "config"), "sample")
        assert got["parsed"]["name"] == "sample"
        assert got["errors"] == []

    def test_read_missing(self, root):
        assert profiles_mod.read_profile(os.path.join(root, "config"), "nope") is None

    @pytest.mark.parametrize("name", ["../etc", "Bad Name", "", "a/b", "x" * 70])
    def test_invalid_names_rejected(self, name):
        errors, _warnings, _parsed = profiles_mod.validate(name, VALID)
        assert any("profile name" in e for e in errors)

    def test_path_traversal_never_resolves(self, root):
        assert profiles_mod.profile_path(os.path.join(root, "config"), "../evil") is None

    def test_name_mismatch_is_an_error(self):
        errors, _w, _p = profiles_mod.validate("other", VALID)
        assert any("expected 'other'" in e for e in errors)

    def test_missing_required_keys(self):
        errors, _w, _p = profiles_mod.validate("sample", "description: x\n")
        assert "missing required key: name" in errors
        assert "missing required key: datasets" in errors

    def test_broken_yaml(self):
        errors, _w, _p = profiles_mod.validate("sample", "name: [unclosed\n")
        assert any("YAML error" in e for e in errors)

    def test_empty_dataset_list_rejected(self):
        errors, _w, _p = profiles_mod.validate("sample", "name: sample\ndatasets: []\n")
        assert "datasets must be a non-empty list" in errors

    def test_empty_map_override_warns(self):
        _e, warnings, _p = profiles_mod.validate(
            "sample", "name: sample\ndatasets: [a]\nresources:\n  cpu: {}\n")
        assert any("use null instead" in w for w in warnings)

    def test_many_m_values_warns(self):
        _e, warnings, _p = profiles_mod.validate(
            "sample",
            "name: sample\ndatasets: [a]\nann:\n  m_values: [4, 8, 16, 24, 32]\n")
        assert any("reloads the whole corpus" in w for w in warnings)

    def test_unknown_top_level_key_warns(self):
        _e, warnings, _p = profiles_mod.validate(
            "sample", "name: sample\ndatasets: [a]\nnonsense: 1\n")
        assert any("unrecognised top-level key: nonsense" in w for w in warnings)

    def test_write_creates_file(self, root):
        ok, errors, _w = profiles_mod.write_profile(
            os.path.join(root, "config"), "fresh", VALID.replace("sample", "fresh"))
        assert ok and not errors
        assert os.path.isfile(os.path.join(root, "config", "profiles", "fresh.yml"))

    def test_write_refuses_invalid(self, root):
        ok, errors, _w = profiles_mod.write_profile(
            os.path.join(root, "config"), "fresh", "datasets: []")
        assert not ok and errors
        assert not os.path.exists(os.path.join(root, "config", "profiles", "fresh.yml"))


# ---------------------------------------------------------------------------
# Job validation
# ---------------------------------------------------------------------------

@pytest.fixture
def store(root):
    return jobs_mod.JobStore(root, ("mariadb", "alisql", "pgvector"))


class TestJobValidation:
    def test_valid_plan_becomes_argv(self, store):
        errors, argv = store.validate({
            "profile": "sample", "engines": ["mariadb", "alisql"],
            "phases": "ops", "resource_pass": "normalized"})
        assert errors == []
        assert argv == ["--profile", "sample", "--engines", "mariadb,alisql",
                        "--resource-pass", "normalized", "--phases", "ops"]

    def test_unknown_profile(self, store):
        errors, _argv = store.validate({"profile": "nope"})
        assert errors == ["no such profile: nope"]

    @pytest.mark.parametrize("engine", [
        "mariadb; touch /tmp/x", "$(id)", "--engines=evil", "../../etc"])
    def test_engine_injection_refused(self, store, engine):
        errors, _argv = store.validate({"profile": "sample", "engines": [engine]})
        assert any("unknown engines" in e for e in errors)

    @pytest.mark.parametrize("dataset", ["a b", "x;y", "-leading", "../escape"])
    def test_dataset_injection_refused(self, store, dataset):
        errors, _argv = store.validate({"profile": "sample", "datasets": [dataset]})
        assert any("invalid dataset names" in e for e in errors)

    def test_run_id_with_escape_sequence_refused(self, store):
        errors, _argv = store.validate({"profile": "sample", "run_id": "main-2026\x1b[A"})
        assert any("Docker will not accept" in e for e in errors)

    def test_bad_phase_refused(self, store):
        errors, _argv = store.validate({"profile": "sample", "phases": "everything"})
        assert any("phases must be one of" in e for e in errors)

    def test_bad_resource_pass_refused(self, store):
        errors, _argv = store.validate({"profile": "sample", "resource_pass": "fast"})
        assert any("resource_pass must be one of" in e for e in errors)

    def test_resume_needs_run_id(self, store):
        errors, _argv = store.validate({"profile": "sample", "resume": True})
        assert any("--resume needs the run_id" in e for e in errors)

    def test_flags_become_options(self, store):
        _errors, argv = store.validate({
            "profile": "sample", "run_id": "r1", "resume": True,
            "force": True, "fail_fast": True, "no_report": True})
        for option in ("--resume", "--force", "--fail-fast", "--no-report"):
            assert option in argv


# ---------------------------------------------------------------------------
# The other long commands
# ---------------------------------------------------------------------------

class TestJobKinds:
    """fetch and build used to be printed as advice and left to the terminal."""

    def test_fetch(self, store):
        errors, argv = store.validate({"kind": "fetch",
                                       "datasets": ["sift-128-euclidean"]})
        assert errors == []
        assert argv == ["--datasets", "sift-128-euclidean"]

    def test_fetch_needs_a_dataset(self, store):
        errors, _argv = store.validate({"kind": "fetch", "datasets": []})
        assert any("at least one dataset" in e for e in errors)

    def test_build(self, store):
        errors, argv = store.validate({"kind": "build", "engines": ["mariadb"],
                                       "target": "bench", "march": "x86-64-v3"})
        assert errors == []
        assert argv == ["--engines", "mariadb", "--target", "bench",
                        "--march", "x86-64-v3"]

    def test_build_needs_an_engine(self, store):
        errors, _argv = store.validate({"kind": "build", "engines": []})
        assert any("at least one engine" in e for e in errors)

    @pytest.mark.parametrize("march", ["native; id", "$(uname -m)", "a b", "--x"])
    def test_march_injection_refused(self, store, march):
        """-march reaches a compiler; it is the token most worth pinning down."""
        errors, _argv = store.validate({"kind": "build", "engines": ["mariadb"],
                                        "march": march})
        assert any("plain gcc -march value" in e for e in errors)

    def test_bad_target_refused(self, store):
        errors, _argv = store.validate({"kind": "build", "engines": ["mariadb"],
                                        "target": "everything"})
        assert any("target must be one of" in e for e in errors)

    def test_report_needs_an_existing_run(self, store, root):
        errors, _argv = store.validate({"kind": "report", "run_id": "nope"})
        assert any("no such run" in e for e in errors)

    def test_report_of_a_real_run(self, store, root):
        os.makedirs(os.path.join(root, "results", "r1"), exist_ok=True)
        errors, argv = store.validate({"kind": "report", "run_id": "r1"})
        assert errors == []
        assert argv[0] == "--run-dir" and argv[1].endswith("results/r1")

    def test_render(self, store):
        errors, argv = store.validate({"kind": "render", "profile": "sample",
                                       "resource_pass": "tuned"})
        assert errors == []
        assert argv == ["--profile", "sample", "--resource-pass", "tuned"]

    def test_unknown_kind_names_the_alternatives(self, store):
        errors, _argv = store.validate({"kind": "destroy"})
        assert any("unknown job kind" in e and "fetch" in e for e in errors)

    def test_kind_defaults_to_run(self, store):
        errors, argv = store.validate({"profile": "sample"})
        assert errors == [] and argv == ["--profile", "sample"]


class TestExclusivity:
    """Nothing runs alongside a benchmark.

    A 5 GB download or a compile during an ingest measurement perturbs exactly
    what is being measured; a competing build distorted MariaDB's numbers by 2x
    once. Setup jobs may overlap each other, because nothing is being measured
    then.
    """

    @pytest.fixture
    def slow(self, root):
        script = os.path.join(root, "run-benchmark.sh")
        with open(script, "w") as fh:
            fh.write("#!/bin/sh\nsleep 10\n")
        os.chmod(script, 0o755)
        os.makedirs(os.path.join(root, "results", "r1"), exist_ok=True)
        return jobs_mod.JobStore(root, ("mariadb",))

    def test_a_benchmark_blocks_a_download(self, slow):
        run, errors = slow.launch({"kind": "run", "profile": "sample"})
        assert errors == []
        _job, errors = slow.launch({"kind": "fetch", "datasets": ["glove-25-angular"]})
        assert any(run["id"] in e and "measures the interference" in e
                   for e in errors), errors
        slow.stop(run["id"])

    def test_a_download_blocks_a_benchmark(self, slow):
        fetch, errors = slow.launch({"kind": "fetch", "datasets": ["glove-25-angular"]})
        assert errors == []
        _job, errors = slow.launch({"kind": "run", "profile": "sample"})
        assert any(fetch["id"] in e and "would measure it too" in e
                   for e in errors), errors
        slow.stop(fetch["id"])

    def test_setup_jobs_may_overlap(self, slow):
        fetch, errors = slow.launch({"kind": "fetch", "datasets": ["glove-25-angular"]})
        assert errors == []
        build, errors = slow.launch({"kind": "build", "engines": ["mariadb"]})
        assert errors == [], errors
        assert build is not None
        slow.stop(fetch["id"])
        slow.stop(build["id"])

    def test_the_identical_command_is_refused(self, slow):
        first, errors = slow.launch({"kind": "fetch", "datasets": ["glove-25-angular"]})
        assert errors == []
        _job, errors = slow.launch({"kind": "fetch", "datasets": ["glove-25-angular"]})
        assert any("already running" in e for e in errors)
        slow.stop(first["id"])

    def test_the_job_records_its_kind(self, slow):
        job, _errors = slow.launch({"kind": "fetch", "datasets": ["glove-25-angular"]})
        assert job["kind"] == "fetch"
        assert job["command_display"].startswith("./run-benchmark.sh fetch")
        slow.stop(job["id"])


# ---------------------------------------------------------------------------
# Job lifecycle
# ---------------------------------------------------------------------------

def wait_for(store, job_id, statuses, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = store.get(job_id)
        if job and job["status"] in statuses:
            return job
        time.sleep(0.05)
    return store.get(job_id)


class TestJobLifecycle:
    @pytest.fixture
    def runnable(self, root):
        """A JobStore whose run-benchmark.sh is a harmless stand-in."""
        script = os.path.join(root, "run-benchmark.sh")
        with open(script, "w") as fh:
            fh.write("#!/bin/sh\necho \"args: $*\"\necho done\nexit 0\n")
        os.chmod(script, 0o755)
        return jobs_mod.JobStore(root, ("mariadb",))

    def test_launch_records_and_completes(self, runnable):
        job, errors = runnable.launch({"profile": "sample", "engines": ["mariadb"]})
        assert errors == [] and job["status"] == "running"
        finished = wait_for(runnable, job["id"], {"completed", "failed"})
        assert finished["status"] == "completed"
        assert finished["exit_code"] == 0

    def test_log_is_captured_and_offset_advances(self, runnable):
        job, _errors = runnable.launch({"profile": "sample"})
        wait_for(runnable, job["id"], {"completed", "failed"})
        first = runnable.read_log(job["id"], 0)
        assert "args: run --profile sample" in first["data"]
        assert first["offset"] == first["size"]
        assert runnable.read_log(job["id"], first["offset"])["data"] == ""

    def test_second_launch_refused_while_running(self, root):
        script = os.path.join(root, "run-benchmark.sh")
        with open(script, "w") as fh:
            fh.write("#!/bin/sh\nsleep 5\n")
        os.chmod(script, 0o755)
        store = jobs_mod.JobStore(root, ("mariadb",))
        first, errors = store.launch({"profile": "sample"})
        assert errors == []
        second, errors = store.launch({"profile": "sample"})
        assert second is None
        # The refusal has to name the job in the way, or it is just a "no".
        assert any(first["id"] in e for e in errors), errors
        store.stop(first["id"])

    def test_stop_terminates(self, root):
        script = os.path.join(root, "run-benchmark.sh")
        with open(script, "w") as fh:
            fh.write("#!/bin/sh\nsleep 30\n")
        os.chmod(script, 0o755)
        store = jobs_mod.JobStore(root, ("mariadb",))
        job, _errors = store.launch({"profile": "sample"})
        ok, errors = store.stop(job["id"])
        assert ok and errors == []
        finished = wait_for(store, job["id"], {"completed", "failed", "stopped"})
        # Asked to stop, so this is the answer rather than a failure.
        assert finished["status"] == "stopped"

    def test_stop_unknown_job(self, runnable):
        ok, errors = runnable.stop("nope")
        assert not ok and errors

    def test_older_records_without_a_kind_still_load(self, root):
        """jobs.json predates having more than one kind of job."""
        state = os.path.join(root, "state", "webui")
        os.makedirs(state, exist_ok=True)
        with open(os.path.join(state, "jobs.json"), "w") as fh:
            json.dump([{"id": "old", "status": "completed",
                        "started_at": "2026-01-01T00:00:00Z"}], fh)
        store = jobs_mod.JobStore(root, ("mariadb",))
        assert store.get("old")["kind"] == "run"

    def test_jobs_survive_restart_as_orphaned(self, root):
        state = os.path.join(root, "state", "webui")
        os.makedirs(state, exist_ok=True)
        with open(os.path.join(state, "jobs.json"), "w") as fh:
            json.dump([{"id": "old", "status": "running", "pid": 999999,
                        "started_at": "2026-01-01T00:00:00Z"}], fh)
        store = jobs_mod.JobStore(root, ("mariadb",))
        assert store.get("old")["status"] == "orphaned"
        assert store.active() is None


# ---------------------------------------------------------------------------
# Starting the server
# ---------------------------------------------------------------------------

class TestWebCommandDiagnostics:
    """Failures must not arrive underneath a line claiming success.

    `web` printed the URL before starting the container, so a port clash read
    as: "web UI on http://127.0.0.1:8080" followed by Docker's own message
    about an endpoint id and a network driver.
    """

    @pytest.mark.parametrize("host,expected", [
        ("0.0.0.0", "127.0.0.1"), ("", "127.0.0.1"), ("::", "127.0.0.1"),
        ("127.0.0.1", "127.0.0.1"), ("192.168.1.10", "192.168.1.10"),
    ])
    def test_reachable_host(self, host, expected):
        from orchestrator.cli import _webui_reachable_host
        assert _webui_reachable_host(host) == expected

    def test_free_port_has_no_holder(self):
        import socket as socket_mod
        from orchestrator.cli import _port_holder
        with socket_mod.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            free = probe.getsockname()[1]
        assert _port_holder("127.0.0.1", free) is None

    def test_taken_port_is_detected(self):
        import socket as socket_mod
        from orchestrator.cli import _port_holder
        with socket_mod.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            port = listener.getsockname()[1]
            assert _port_holder("127.0.0.1", port) is not None

    def test_a_container_holding_the_port_is_named(self, monkeypatch):
        import socket as socket_mod
        from orchestrator import cli as cli_mod
        from orchestrator import docker_ctl

        monkeypatch.setattr(docker_ctl, "containers_publishing",
                            lambda port: ["vb-webui-42"])
        with socket_mod.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            port = listener.getsockname()[1]
            assert "vb-webui-42" in cli_mod._port_holder("127.0.0.1", port)

    def test_containers_publishing_reads_the_port_column(self, monkeypatch):
        from orchestrator import docker_ctl

        class Result:
            stdout = ("other\t0.0.0.0:9000->9000/tcp\n"
                      "mine\t127.0.0.1:8080->8080/tcp\n"
                      "noports\t\n")

        monkeypatch.setattr(docker_ctl, "_run", lambda *a, **k: Result())
        assert docker_ctl.containers_publishing(8080) == ["mine"]
        assert docker_ctl.containers_publishing(9000) == ["other"]
        assert docker_ctl.containers_publishing(1234) == []

    def test_the_container_banner_claims_no_url(self, capsys, tmp_path, monkeypatch):
        """Behind a publish the port here is the internal one.

        Printing it beside the real address is worse than not printing it.
        """
        from webui import server as server_mod

        started = {}

        class FakeServer:
            def serve_forever(self):
                started["ran"] = True

            def server_close(self):
                pass

        monkeypatch.setattr(server_mod, "make_server",
                            lambda *a, **k: FakeServer())
        server_mod.serve(str(tmp_path), host="0.0.0.0", port=8080,
                         published_host="127.0.0.1")
        out = capsys.readouterr().out
        assert "in-container" in out
        assert "http://0.0.0.0:8080" not in out

    def test_the_direct_banner_does_name_itself(self, capsys, tmp_path, monkeypatch):
        from webui import server as server_mod

        class FakeServer:
            def serve_forever(self):
                pass

            def server_close(self):
                pass

        monkeypatch.setattr(server_mod, "make_server", lambda *a, **k: FakeServer())
        server_mod.serve(str(tmp_path), host="127.0.0.1", port=8452)
        assert "http://127.0.0.1:8452" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Control guards
# ---------------------------------------------------------------------------

@pytest.fixture
def readonly_api(root):
    return api_mod.Api(root, allow_control=False)


@pytest.fixture
def control_api(root):
    return api_mod.Api(root, allow_control=True)


class TestControlGuards:
    MUTATIONS = [
        ("PUT", "/api/profiles/sample", {"text": VALID}),
        ("POST", "/api/profiles/sample/validate", {"text": VALID}),
        ("POST", "/api/jobs", {"profile": "sample"}),
        ("POST", "/api/jobs/job-1/stop", None),
    ]

    @pytest.mark.parametrize("method,path,body", MUTATIONS)
    def test_refused_without_allow_control(self, readonly_api, method, path, body):
        status, payload = api_mod.dispatch(readonly_api, method, path, {}, body)
        assert status == 403
        assert "control is disabled" in payload["error"]

    def test_reads_allowed_without_control(self, readonly_api):
        for path in ("/api/profiles", "/api/profiles/sample", "/api/jobs"):
            status, _payload = api_mod.dispatch(readonly_api, "GET", path, {})
            assert status == 200, path

    def test_put_profile_with_control(self, control_api, root):
        status, payload = api_mod.dispatch(
            control_api, "PUT", "/api/profiles/written", {},
            {"text": VALID.replace("name: sample", "name: written")})
        assert status == 200 and payload["ok"] is True
        assert os.path.isfile(os.path.join(root, "config", "profiles", "written.yml"))

    def test_put_profile_rejects_bad_yaml(self, control_api):
        status, payload = api_mod.dispatch(
            control_api, "PUT", "/api/profiles/written", {}, {"text": "datasets: []"})
        assert status == 400 and payload["ok"] is False

    def test_put_profile_requires_text(self, control_api):
        status, _payload = api_mod.dispatch(
            control_api, "PUT", "/api/profiles/written", {}, {})
        assert status == 400

    def test_missing_profile_is_404(self, control_api):
        status, _payload = api_mod.dispatch(control_api, "GET", "/api/profiles/nope", {})
        assert status == 404

    def test_concurrent_launch_is_409(self, control_api, root):
        script = os.path.join(root, "run-benchmark.sh")
        with open(script, "w") as fh:
            fh.write("#!/bin/sh\nsleep 5\n")
        os.chmod(script, 0o755)

        status, payload = api_mod.dispatch(
            control_api, "POST", "/api/jobs", {}, {"profile": "sample"})
        assert status == 201

        status, payload = api_mod.dispatch(
            control_api, "POST", "/api/jobs", {}, {"profile": "sample"})
        assert status == 409
        assert any("measure" in e or "running" in e for e in payload["errors"])

        control_api.jobs.stop(control_api.jobs.active()["id"])

    def test_invalid_plan_is_400(self, control_api, root):
        script = os.path.join(root, "run-benchmark.sh")
        with open(script, "w") as fh:
            fh.write("#!/bin/sh\nexit 0\n")
        os.chmod(script, 0o755)
        status, payload = api_mod.dispatch(
            control_api, "POST", "/api/jobs", {}, {"profile": "nope"})
        assert status == 400 and payload["ok"] is False

    def test_job_log_unknown_is_404(self, control_api):
        status, _payload = api_mod.dispatch(
            control_api, "GET", "/api/jobs/nope/log", {"offset": ["0"]})
        assert status == 404
