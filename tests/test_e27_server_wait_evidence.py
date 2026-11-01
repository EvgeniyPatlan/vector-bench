"""E27 -- server-side evidence of what limited a concurrency point.

The published six-engine post attributed the MySQL-family ceiling first to
the server and then to the client, with no server-side evidence either way.
Ops version 9 added the server's CPU and cgroup pressure per window and its
own guard says it cannot tell a server that serializes from one that is
saturated. This file covers what was added to tell them apart, beside the
window-timing and MySQL-family tests in
`tests/test_e20_server_status_capture.py` §6:

1. pgvector's `pg_stat_activity` sampler -- a thread in the harness process,
   on its own connection, that stops when the window does;
2. the opt-in `ops.server_wait_instruments`: off by default, and when on,
   rendered as server flags and recorded in `engines_resolved`, its `applied`
   list and on every record;
3. the optional perf window: unavailable is a recorded reason and the run
   goes on; available writes files for exactly the designated window;
4. the report's "what limited throughput" row, with and without wait data;
5. the estimate's mongot index-size settle term;
6. the per-engine snapshot parsers for Valkey and MongoDB.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from typing import Any, Dict, List
from unittest import mock

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from harness.drivers import engine_waits                           # noqa: E402
from harness.drivers.base import ConnectionSpec                    # noqa: E402
from harness.drivers.postgres import PostgresDriver                # noqa: E402
from harness.metrics import records                                # noqa: E402
from harness.workloads import concurrency as concurrency_workload  # noqa: E402
from harness.workloads import profile_window                       # noqa: E402
from harness.workloads.context import RunContext                   # noqa: E402
from harness.workloads.server_waits import WaitCapture             # noqa: E402
from orchestrator import config as config_mod                      # noqa: E402
from orchestrator import estimate as est_mod                       # noqa: E402
from orchestrator import ops_pass                                  # noqa: E402
from orchestrator import perf_window                               # noqa: E402
from report import limiters                                        # noqa: E402
from report import render                                          # noqa: E402
from tests.fake_driver import (K, FakeDriver, index_spec,           # noqa: E402
                               loaded_driver, tiny_dataset)

MYSQL_FAMILY = ("mariadb", "mariadb123", "alisql", "percona", "myvector")


# ---------------------------------------------------------------------------
# 1. pgvector's side sampler
# ---------------------------------------------------------------------------

class PgCursor:
    """`pg_stat_activity` as a list of (type, event) rows per call."""

    def __init__(self, answers: List[List[tuple]]):
        self.answers = list(answers)
        self.statements: List[str] = []
        self.closed = False
        self._answer: List[tuple] = []

    def execute(self, sql, params=None):
        self.statements.append(sql)
        self._answer = self.answers.pop(0) if len(self.answers) > 1 \
            else list(self.answers[0])

    def fetchall(self):
        return list(self._answer)

    def close(self):
        self.closed = True


class SidePg(PostgresDriver):
    """A real `PostgresDriver` whose connection is a recording cursor."""

    opened: List["SidePg"] = []

    def connect(self):
        self._conn = None
        self._cur = PgCursor([[("LWLock", "BufferMapping"), (None, None)],
                              [("LWLock", "BufferMapping")]])
        SidePg.opened.append(self)

    def close(self):
        if self._cur is not None:
            self._cur.close()
        self._cur = None


class TestPostgresSampler:
    def test_it_samples_on_its_own_connection_and_stops(self, monkeypatch):
        """The sampler is a forked process with its own connection, gone when
        the window is. Mutant (sampling on the main/client driver): the
        client cursor below records the sampler's statements."""
        SidePg.opened = []
        monkeypatch.setattr(SidePg, "wait_sample_interval_s", 0.05)
        client = SidePg(ConnectionSpec(host="h", port=5432))
        client.connect()
        capture = WaitCapture(
            lambda: SidePg(ConnectionSpec(host="h", port=5432))).begin()
        assert capture.side_driver is not client
        assert capture.sampling
        # The parent never connected a side driver: the connection lives in
        # the sampler process alone.
        assert SidePg.opened == [client]
        time.sleep(0.3)
        keys = capture.finish()
        assert not capture.sampling, "the sampler outlived the window"
        assert capture.side_driver is None
        # The client's connection was never used by the sampler.
        assert client._cur.statements == []
        waits = keys["server_waits"]
        assert keys["server_waits_reason"] is None
        assert waits["ranked_by"] == "count"
        assert waits["samples"] >= 3
        top = waits["events"][0]
        assert top["event"] == "LWLock:BufferMapping"
        assert top["count"] == waits["samples"]
        assert top["wait_s"] is None
        assert any(e["event"] == engine_waits.ON_CPU for e in waits["events"])
        assert not [t for t in threading.enumerate()
                    if t.name == "vb-wait-sampler"]

    def test_a_sampling_engine_leaves_no_thread_alive_at_the_client_fork(
            self, tmp_path):
        """The clients fork after the sampler starts. A sampler THREAD would
        be inherited by that fork, which CPython warns about and
        `tests/test_g4_pool_teardown.py` forbids; as an error here, a thread
        sampler fails this test."""
        import warnings

        class Sampling(FakeDriver):
            wait_sample_interval_s = 0.05

            def clone(self):
                sibling = super().clone()
                sibling.__class__ = Sampling
                return sibling

            def wait_snapshot(self):
                return {"events": {"LWLock:x": {"count": 1, "wait_s": None}},
                        "samples": 1}, None

            def wait_delta(self, before, after):
                return engine_waits.postgres_delta(before, after, 0.05)

        recorder = records.Recorder(str(tmp_path / "ops.jsonl"))
        ctx = RunContext(run_id="e27s", resource_pass="normalized",
                         recorder=recorder, engine="fake", engine_tag="x",
                         march="x86-64-v3", k=K,
                         cache_dir=str(tmp_path / "cache"))
        dataset = tiny_dataset()
        driver = loaded_driver(dataset, ef_half=64.0, ef_search=32)
        driver.__class__ = Sampling
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", DeprecationWarning)
                concurrency_workload.run(
                    ctx, driver.clone, dataset, index_spec(dataset),
                    ef_search=32, client_counts=(2,), duration_s=0.2,
                    warmup=1, max_queries=50)
        finally:
            recorder.close()
        waits = list(records.read_records(recorder.path))[0]["extra"][
            "server_waits"]
        assert waits["events"][0]["event"] == "LWLock:x"
        assert waits["samples"] >= 2

    def test_the_sample_query_excludes_the_sampler_itself(self):
        assert "pg_backend_pid()" in engine_waits.PG_SAMPLE_SQL
        assert "state = 'active'" in engine_waits.PG_SAMPLE_SQL

    def test_no_sample_is_a_reason_not_an_empty_profile(self):
        waits, reason = engine_waits.postgres_delta(
            {"events": {}, "samples": 0}, {"events": {}, "samples": 0}, 1.0)
        assert waits is None and "shorter than" in reason

    def test_a_side_connection_that_fails_is_a_reason(self):
        def broken():
            raise ConnectionError("refused")
        keys = WaitCapture(broken).finish() if False else \
            WaitCapture(broken).begin().finish()
        assert keys["server_waits"] is None
        assert "refused" in keys["server_waits_reason"]


class TestChurnReaderWindows:
    def test_both_reader_windows_carry_their_own_waits(self, tmp_path,
                                                       monkeypatch):
        """The during-writes record carries the baseline window's waits and
        the during-writes window's, each under its own prefix, so the
        writer's contention shows up beside the ratio it explains."""
        from harness.drivers.server_waits import counter_delta
        from harness.workloads import churn as cw
        from tests import test_t4b_readers as t4b

        class Counting(t4b.SlowWriter):
            calls = 0

            def clone(self):
                sibling = super().clone()
                sibling.__class__ = Counting
                return sibling

            def wait_snapshot(self):
                Counting.calls += 1
                n = Counting.calls
                return {"events": {"lock": {"count": n, "wait_s": n * 0.5}}
                        }, None

            def wait_delta(self, before, after):
                return counter_delta(before, after, "fake")

        monkeypatch.setattr(cw, "READER_BASELINE_WINDOW_S",
                            t4b.TEST_BASELINE_WINDOW_S)
        recorder = records.Recorder(str(tmp_path / "ops.jsonl"))
        ctx = RunContext(run_id="e27c", resource_pass="normalized",
                         recorder=recorder, engine="fake", engine_tag="x",
                         march="x86-64-v3", k=K,
                         cache_dir=str(tmp_path / "cache"))
        dataset = tiny_dataset()
        driver = t4b.reader_driver(dataset, cls=Counting)
        try:
            t4b.run_churn(ctx, dataset, driver)
        finally:
            recorder.close()
        rows = [r for r in records.read_records(recorder.path)
                if (r.get("extra") or {}).get("during_writes")]
        assert len(rows) == 1
        extra = rows[0]["extra"]
        assert Counting.calls == 4, "two snapshots per reader window"
        for prefix in ("reader_baseline_", "reader_under_writes_"):
            waits = extra[prefix + "server_waits"]
            assert extra[prefix + "server_waits_reason"] is None
            assert waits["events"] == [{"event": "lock", "count": 1,
                                        "wait_s": 0.5}]


# ---------------------------------------------------------------------------
# 2. ops.server_wait_instruments
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def resolved():
    from harness.metrics import sysinfo
    info = sysinfo.collect()
    return {e: config_mod.resolve_resources(
        config_mod.load_resources("normalized"), e, info)
        for e in MYSQL_FAMILY + ("pgvector",)}


class TestWaitInstruments:
    def test_off_by_default_in_every_shipped_profile(self):
        for name in config_mod.available_profiles():
            assert config_mod.ops_wait_instruments(
                config_mod.load_profile(name)) is False, name

    def test_a_non_boolean_is_refused_not_coerced(self):
        with pytest.raises(ValueError):
            config_mod.ops_wait_instruments(
                {"ops": {"server_wait_instruments": "false"}})

    @pytest.mark.parametrize("engine", MYSQL_FAMILY)
    def test_on_renders_the_flags_and_off_renders_none(self, engine,
                                                       resolved):
        cfg = config_mod.load_engine(engine)
        off = config_mod.server_args(cfg, "normalized", resolved[engine])
        on = config_mod.server_args(cfg, "normalized", resolved[engine],
                                    wait_instruments=True)
        flags = config_mod.engine_wait_instruments(cfg)
        assert flags and not any("performance-schema" in f for f in off)
        assert on == off + flags
        assert any("wait/synch/mutex/" in f for f in flags)
        assert any("wait/synch/rwlock/" in f for f in flags)

    @pytest.mark.parametrize("engine", MYSQL_FAMILY)
    def test_engines_resolved_records_it_only_when_on(self, engine,
                                                      resolved):
        """Off: the entry is the pre-feature entry, byte for byte, so a
        resumed run recorded before the key existed is not refused as
        `ResolutionChanged`. On: `applied` and its own block say so."""
        cfg = config_mod.load_engine(engine)
        off = config_mod.engine_resolution(cfg, "normalized",
                                           resolved[engine])
        on = config_mod.engine_resolution(cfg, "normalized",
                                          resolved[engine],
                                          wait_instruments=True)
        assert "server_wait_instruments" not in off
        assert "server_wait_instruments" not in off["applied"]
        assert on["applied"] == off["applied"] + ["server_wait_instruments"]
        block = on["server_wait_instruments"]
        assert block["enabled"] is True and block["applies_to"] == ["ops"]
        assert block["flags"] == config_mod.engine_wait_instruments(cfg)

    def test_an_engine_without_instruments_records_nothing(self, resolved):
        cfg = config_mod.load_engine("pgvector")
        off = config_mod.engine_resolution(cfg, "normalized",
                                           resolved["pgvector"])
        on = config_mod.engine_resolution(cfg, "normalized",
                                          resolved["pgvector"],
                                          wait_instruments=True)
        assert on == off

    def test_the_ann_path_never_gets_them(self):
        import inspect
        from orchestrator import ann_pass
        assert "wait_instruments" not in inspect.getsource(ann_pass)

    def test_harness_args_carry_the_flag_only_when_on(self, resolved):
        profile = config_mod.load_profile("smoke")
        resources = config_mod.load_resources("normalized")
        off = ops_pass.harness_args(profile, 16, "percona",
                                    resolved["percona"], "normalized",
                                    resources)
        on_profile = {**profile, "ops": {**profile["ops"],
                                         "server_wait_instruments": True}}
        on = ops_pass.harness_args(on_profile, 16, "percona",
                                   resolved["percona"], "normalized",
                                   resources)
        pg = ops_pass.harness_args(on_profile, 16, "pgvector",
                                   resolved["pgvector"], "normalized",
                                   resources)
        assert "--server-wait-instruments" not in off
        assert on == off + ["--server-wait-instruments"]
        assert "--server-wait-instruments" not in pg

    def test_every_record_carries_it_when_on_and_none_when_off(
            self, tmp_path):
        from harness import main as main_mod
        dataset = tiny_dataset()

        def factory(*a, **kw):
            return FakeDriver(dim=dataset.dim, metric=dataset.metric)

        def run(flag):
            out = tmp_path / f"ops-{flag}.jsonl"
            argv = ["--engine", "fake", "--host", "h", "--port", "1",
                    "--dataset", "tiny", "--m", "16", "--run-id", "e27",
                    "--output", str(out), "--cache-dir",
                    str(tmp_path / f"c-{flag}"), "--workloads", "build",
                    "--ef-mode", "fixed"]
            if flag:
                argv.append("--server-wait-instruments")
            with mock.patch.object(main_mod, "load_dataset",
                                   lambda *a, **kw: dataset), \
                 mock.patch.object(main_mod, "get_driver", factory):
                assert main_mod.main(argv) == 0
            return list(records.read_records(str(out)))

        on, off = run(True), run(False)
        assert on and off
        assert all(r["extra"]["server_wait_instruments"] is True for r in on)
        assert all("server_wait_instruments" not in (r.get("extra") or {})
                   for r in off)


# ---------------------------------------------------------------------------
# 3. The perf window
# ---------------------------------------------------------------------------

def _watch(tmp_path, factory=perf_window.PerfWindow):
    return perf_window.ProfileWatcher(
        "srv", str(tmp_path / "profiles"), "percona-d-normalized-m16-post",
        str(tmp_path / "ack"), {"engine": "percona"}, bench_image=None,
        window_factory=factory)


def _profile_json(tmp_path, clients):
    path = (tmp_path / "profiles" /
            f"percona-d-normalized-m16-post-c{clients}" / "profile.json")
    return json.loads(path.read_text())


class TestPerfWindow:
    def test_unavailable_records_the_reason_and_acks(self, tmp_path,
                                                     monkeypatch):
        monkeypatch.setattr(perf_window, "perf_unavailable_reason",
                            lambda *a, **k: "perf is not installed on the "
                                            "host")
        watch = _watch(tmp_path)
        line = f"{profile_window.BEGIN_MARK} clients=4 token=abc123"
        assert watch(line) == [line]
        assert (tmp_path / "ack" / "ack-abc123").exists(), \
            "the harness would wait out its timeout"
        assert watch(f"{profile_window.END_MARK} clients=4 token=abc123")
        assert watch(None) == []
        meta = _profile_json(tmp_path, 4)
        assert meta["status"] == "unavailable"
        assert meta["reason"] == "perf is not installed on the host"
        assert "files" not in meta

    def test_ordinary_lines_pass_through_untouched(self, tmp_path):
        watch = _watch(tmp_path)
        assert watch("[concurrency] 4 client(s)") == [
            "[concurrency] 4 client(s)"]
        assert not (tmp_path / "profiles").exists()

    def test_a_mocked_perf_writes_files_for_exactly_that_window(
            self, tmp_path, monkeypatch):
        """Available (mocked): perf starts at the begin mark, stops at the end
        mark, and leaves perf.data, folded stacks and profile.json in the
        designated window's directory -- and in no other."""
        events: List[str] = []

        class FakeProc:
            def __init__(self, argv, **kw):
                self.argv = argv
                self.out = argv[argv.index("-o") + 1]
                events.append("start")
                self._code = None

            def poll(self):
                return self._code

            def send_signal(self, sig):
                events.append("stop")
                with open(self.out, "w") as fh:
                    fh.write("PERFDATA")
                self._code = 0

            def wait(self, timeout=None):
                return 0

            def kill(self):
                pass

        script = ("mysqld 1 [000] 1.0: cpu-clock:\n"
                  "\tffff row_search_mvcc+0x10 (/usr/sbin/mysqld)\n"
                  "\tffff do_command (/usr/sbin/mysqld)\n\n"
                  "mysqld 1 [000] 1.1: cpu-clock:\n"
                  "\tffff row_search_mvcc+0x22 (/usr/sbin/mysqld)\n"
                  "\tffff do_command (/usr/sbin/mysqld)\n\n")

        monkeypatch.setattr(perf_window, "perf_unavailable_reason",
                            lambda *a, **k: None)
        monkeypatch.setattr(perf_window, "container_target", lambda c: {
            "method": "cgroup", "target": "system.slice/docker-x.scope",
            "args": ["-a", "-e", "cpu-clock", "-G",
                     "system.slice/docker-x.scope"]})
        monkeypatch.setattr(perf_window, "PERF_START_GRACE_S", 0.0)
        monkeypatch.setattr(perf_window.subprocess, "Popen", FakeProc)
        monkeypatch.setattr(
            perf_window.subprocess, "run",
            lambda *a, **k: mock.Mock(stdout=script, stderr="",
                                      returncode=0))
        watch = _watch(tmp_path)
        watch("[concurrency] 1 client(s): measuring")
        assert events == []
        watch(f"{profile_window.BEGIN_MARK} clients=8 token=feed")
        assert events == ["start"]
        watch("[concurrency] something inside the window")
        assert events == ["start"]
        watch(f"{profile_window.END_MARK} clients=8 token=feed")
        assert events == ["start", "stop"]
        watch(None)
        assert events == ["start", "stop"], "stopped twice"
        meta = _profile_json(tmp_path, 8)
        assert meta["status"] == "recorded"
        assert meta["method"] == "cgroup"
        assert meta["files"] == {"perf_data": "perf.data",
                                 "folded": "folded.txt"}
        assert "svg_reason" in meta
        assert meta["perf_started_at"] <= meta["perf_ended_at"]
        folded = (tmp_path / "profiles" / "percona-d-normalized-m16-post-c8" /
                  "folded.txt").read_text()
        assert folded == "mysqld;do_command;row_search_mvcc 2\n"
        dirs = sorted(os.listdir(tmp_path / "profiles"))
        assert dirs == ["percona-d-normalized-m16-post-c8"]

    def test_perf_refusing_at_start_is_a_reason_and_the_run_goes_on(
            self, tmp_path, monkeypatch):
        class Refused:
            def __init__(self, argv, stderr=None, **kw):
                stderr.write("Access to performance monitoring and "
                             "observability operations is limited.\n")
                stderr.flush()

            def poll(self):
                return 255

        monkeypatch.setattr(perf_window, "perf_unavailable_reason",
                            lambda *a, **k: None)
        monkeypatch.setattr(perf_window, "container_target", lambda c: {
            "method": "pid", "target": "1", "args": ["-p", "1"]})
        monkeypatch.setattr(perf_window, "PERF_START_GRACE_S", 0.0)
        monkeypatch.setattr(perf_window.subprocess, "Popen", Refused)
        monkeypatch.setattr(perf_window, "paranoid_level", lambda: 2)
        watch = _watch(tmp_path)
        watch(f"{profile_window.BEGIN_MARK} clients=2 token=aa")
        assert (tmp_path / "ack" / "ack-aa").exists()
        meta = _profile_json(tmp_path, 2)
        assert meta["status"] == "unavailable"
        assert "perf_event_paranoid is 2" in meta["reason"]
        assert "limited" in meta["reason"]

    def test_the_folder_handles_symbols_without_offsets_or_dso(self):
        folded = perf_window.fold_perf_script([
            "postgres 7 [1] 2.0: cpu-clock:",
            "\t  aa [unknown] ([unknown])",
            "\t  bb PostgresMain+0x1 (/usr/lib/postgresql/bin/postgres)",
            ""])
        assert dict(folded) == {"postgres;PostgresMain;[unknown]": 1}

    def test_no_perf_binary_is_the_certain_refusal(self):
        assert perf_window.perf_unavailable_reason(lambda name: None) == \
            "perf is not installed on the host"
        assert perf_window.perf_unavailable_reason(
            lambda name: "/usr/bin/perf") is None


class TestTheHarnessSignalsOneWindow:
    def test_only_the_first_repeat_of_the_designated_count_is_marked(
            self, tmp_path, monkeypatch):
        calls: List[int] = []
        def begin(self):
            calls.append(self.clients)
            self.acknowledged = True
            return True

        monkeypatch.setattr(profile_window.ProfileWindow, "begin", begin)
        monkeypatch.setattr(profile_window.ProfileWindow, "end",
                            lambda self: calls.append(-self.clients))
        recorder = records.Recorder(str(tmp_path / "ops.jsonl"))
        ctx = RunContext(run_id="e27", resource_pass="normalized",
                         recorder=recorder, engine="fake", engine_tag="x",
                         march="x86-64-v3", k=K,
                         cache_dir=str(tmp_path / "cache"))
        dataset = tiny_dataset()
        driver = loaded_driver(dataset, ef_half=64.0, ef_search=32)
        try:
            concurrency_workload.run(
                ctx, driver.clone, dataset, index_spec(dataset),
                ef_search=32, client_counts=(1, 2), duration_s=0.05,
                warmup=1, max_queries=50, repeats=2,
                profile_window_clients=2, profile_ack_dir=str(tmp_path))
        finally:
            recorder.close()
        assert calls == [2, -2]
        rows = list(records.read_records(recorder.path))
        assert "profiled_repeat" not in rows[0]["extra"]
        assert rows[1]["extra"]["profiled_repeat"] == 1

    def test_an_unacknowledged_window_is_not_claimed_as_profiled(
            self, tmp_path, monkeypatch):
        monkeypatch.setattr(profile_window, "ACK_TIMEOUT_S", 0.05)
        recorder = records.Recorder(str(tmp_path / "ops.jsonl"))
        ctx = RunContext(run_id="e27n", resource_pass="normalized",
                         recorder=recorder, engine="fake", engine_tag="x",
                         march="x86-64-v3", k=K,
                         cache_dir=str(tmp_path / "cache"))
        dataset = tiny_dataset()
        driver = loaded_driver(dataset, ef_half=64.0, ef_search=32)
        try:
            concurrency_workload.run(
                ctx, driver.clone, dataset, index_spec(dataset),
                ef_search=32, client_counts=(1,), duration_s=0.05,
                warmup=1, max_queries=50, profile_window_clients=1,
                profile_ack_dir=None)
        finally:
            recorder.close()
        row = list(records.read_records(recorder.path))[0]
        assert "profiled_repeat" not in row["extra"]

    def test_the_handshake_waits_for_the_ack_and_no_longer(self, tmp_path):
        window = profile_window.ProfileWindow(str(tmp_path), 4, timeout_s=5)
        timer = threading.Timer(
            0.2, lambda: (tmp_path / f"ack-{window.token}").write_text("ok"))
        timer.start()
        started = time.monotonic()
        assert window.begin() is True
        assert time.monotonic() - started < 2

    def test_a_host_that_never_answers_costs_the_timeout_only(self, tmp_path):
        window = profile_window.ProfileWindow(str(tmp_path), 4,
                                              timeout_s=0.2)
        assert window.begin() is False

    def test_profile_window_must_be_a_measured_count(self):
        ops = {"client_counts": [1, 4, 16]}
        assert config_mod.profile_window_clients({"ops": ops}) is None
        assert config_mod.profile_window_clients(
            {"ops": {**ops, "profile_window_clients": 4}}) == 4
        with pytest.raises(ValueError):
            config_mod.profile_window_clients(
                {"ops": {**ops, "profile_window_clients": 8}})
        with pytest.raises(ValueError):
            config_mod.profile_window_clients(
                {"ops": {**ops, "profile_window_clients": True}})

    def test_the_marks_the_orchestrator_parses_are_the_harness_marks(self):
        line = (f"{profile_window.BEGIN_MARK} clients=16 "
                f"token={profile_window.ProfileWindow(None, 16).token}")
        match = perf_window._MARK_RE.search(line)
        assert match and match.group(1) == "16"


# ---------------------------------------------------------------------------
# 4. The report's limiter row
# ---------------------------------------------------------------------------

def _record(**extra: Any) -> Dict[str, Any]:
    return {"phase": "concurrency", "engine": "percona", "dataset": "d",
            "resource_pass": "normalized", "m": 16, "build_mode": "post",
            "storage_engine": "InnoDB", "clients": 8, "qps": 100.0,
            "extra": {"client_cpu_util": 0.4, **extra}}


def _limiter_rows(summary):
    table = render._limiter_table(summary)
    lines = [ln for ln in table.splitlines() if ln.startswith("| ")]
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    body = [[c.strip() for c in ln.strip("|").split("|")]
            for ln in lines[2:]]
    return header, body


class TestLimiterRow:
    WAITS = {"source": "ps", "ranked_by": "wait_s", "total_wait_s": 4.0,
             "events": [{"event": "wait/synch/mutex/x", "count": 120,
                         "wait_s": 3.0}], "events_total": 1}

    def test_with_wait_data_names_the_top_wait_and_its_share(self):
        rec = _record(server_waits=self.WAITS, server_waits_reason=None)
        from report.server_windows import window_key
        summary = {"concurrency": [rec], "server_windows": {
            window_key(rec): {"cpu_cores": 7.6, "cpu_util": 0.95,
                              "cpu_pressure": 0.1, "io_pressure": None}},
            "profiles": {"percona-d-normalized-m16-post-c8": {
                "href": "../profiles/p/folded.txt"}}}
        header, body = _limiter_rows(summary)
        row = dict(zip(header, body[0]))
        assert row["Top server wait"].startswith("wait/synch/mutex/x: 3.000 s")
        assert "75% of waited time" in row["Top server wait"]
        assert row["Profile"] == "[profile](../profiles/p/folded.txt)"
        assert row["Largest measured"] == "server CPU (95%)"
        assert row["Client CPU"] == "0.40"

    def test_without_wait_data_says_why(self):
        rec = _record(server_waits=None,
                      server_waits_reason="performance_schema is OFF")
        header, body = _limiter_rows({"concurrency": [rec]})
        row = dict(zip(header, body[0]))
        assert row["Top server wait"] == "— (performance_schema is OFF)"
        assert row["Profile"] == "—"
        # Only the client was measured, so only it can be largest.
        assert row["Largest measured"] == "client CPU (40%)"

    def test_a_record_from_before_the_feature_says_not_recorded(self):
        header, body = _limiter_rows({"concurrency": [_record()]})
        assert dict(zip(header, body[0]))["Top server wait"] == \
            "— (not recorded)"

    def test_a_measured_empty_profile_is_not_a_dash(self):
        assert limiters.top_wait_cell({"server_waits": {"events": []}}) == \
            "none measured"

    def test_a_sampled_engine_ranks_by_count(self):
        cell = limiters.top_wait_cell({"server_waits": {
            "ranked_by": "count", "samples": 20,
            "events": [{"event": "LWLock:BufferMapping", "count": 12,
                        "wait_s": None}]}})
        assert cell == "LWLock:BufferMapping: 12 of 20 samples"

    def test_an_unavailable_profile_shows_its_reason(self):
        cell = limiters.profile_cell(_record(), {
            "percona-d-normalized-m16-post-c8": {
                "status": "unavailable", "reason": "perf is not installed"}})
        assert cell == "— (perf is not installed)"

    def test_the_profile_index_links_relative_to_the_report(self, tmp_path):
        out = tmp_path / "profiles" / "percona-d-normalized-m16-post-c8"
        out.mkdir(parents=True)
        (out / "profile.json").write_text(json.dumps(
            {"status": "recorded", "files": {"perf_data": "perf.data",
                                             "folded": "folded.txt"}}))
        index = limiters.profile_index(str(tmp_path),
                                       str(tmp_path / "report"))
        entry = index["percona-d-normalized-m16-post-c8"]
        assert entry["href"] == os.path.join(
            "..", "profiles", "percona-d-normalized-m16-post-c8",
            "folded.txt")

    def test_the_html_converter_renders_the_link(self):
        assert '<a href="x/y.txt">profile</a>' in render._inline_md(
            "[profile](x/y.txt)")


# ---------------------------------------------------------------------------
# 5. The estimate counts mongot's settle
# ---------------------------------------------------------------------------

class TestMongotSettleInTheEstimate:
    def _est(self, engines):
        profile = config_mod.load_profile("smoke")
        return est_mod.estimate_load_hours(
            profile, engines, list(profile["datasets"])[:1], ["normalized"],
            ["ops"])

    def test_the_worst_case_includes_the_cap_for_mongodb_only(self):
        mongo = {t["name"]: t for t in self._est(["mongodb"])["query_terms"]}
        pg = {t["name"]: t for t in self._est(["pgvector"])["query_terms"]}
        term = mongo["mongot index-size settle"]
        assert term["worst_seconds"] == est_mod.MONGOT_SETTLE_CAP_S
        assert term["typical_seconds"] == est_mod.MONGOT_SETTLE_INTERVAL_S
        assert term["bounded"] is True
        assert "mongot index-size settle" not in pg

    def test_the_constants_match_the_harness_defaults(self):
        from harness.drivers import mongo
        assert est_mod.MONGOT_SETTLE_CAP_S == mongo.SETTLE_CAP_S
        assert est_mod.MONGOT_SETTLE_INTERVAL_S == mongo.SETTLE_INTERVAL_S


# ---------------------------------------------------------------------------
# 6. Valkey and MongoDB snapshot parsers
# ---------------------------------------------------------------------------

class FakeValkey:
    def __init__(self, calls, threshold=0, latest=()):
        self._calls, self._threshold, self._latest = calls, threshold, latest

    def info(self, section):
        assert section == "commandstats"
        return {"cmdstat_FT.SEARCH": {"calls": self._calls,
                                      "usec": self._calls * 100}}

    def execute_command(self, *args):
        return list(self._latest)

    def config_get(self, key):
        return {key: str(self._threshold)}


class TestValkeyAndMongo:
    def test_valkey_commandstats_delta(self):
        before, _ = engine_waits.valkey_snapshot(FakeValkey(10))
        after, _ = engine_waits.valkey_snapshot(FakeValkey(1010))
        waits, reason = engine_waits.valkey_delta(before, after)
        assert reason is None
        assert waits["events"][0] == {"event": "cmd:FT.SEARCH",
                                      "count": 1000, "wait_s": 0.1}
        assert "latency-monitor-threshold is 0" in \
            waits["latency_latest_reason"]

    def test_valkey_latency_spikes_inside_the_window_only(self):
        before, _ = engine_waits.valkey_snapshot(FakeValkey(1))
        now = int(time.time()) + 5
        after, _ = engine_waits.valkey_snapshot(FakeValkey(
            2, threshold=10, latest=[[b"command", now, 12, 30],
                                     [b"fork", 1, 5, 5]]))
        waits, _ = engine_waits.valkey_delta(before, after)
        assert waits["latency_latest"] == [
            {"event": "command", "at": now, "latest_ms": 12, "max_ms": 30}]

    def test_mongo_lock_waits_and_an_unreachable_mongot(self, monkeypatch):
        def status(waits):
            return {"locks": {"Global": {"acquireWaitCount": {"r": waits},
                                         "timeAcquiringMicros":
                                             {"r": waits * 1000}}},
                    "globalLock": {"currentQueue": {"total": 0}},
                    "queues": {"execution": {"read": {"out": 3}}}}

        class Client:
            def __init__(self, waits):
                self.admin = mock.Mock()
                self.admin.command = lambda name: status(waits)

        def unreachable(url, timeout=None):
            raise OSError("connection refused")

        monkeypatch.setattr(engine_waits.urllib.request, "urlopen",
                            unreachable)
        before, _ = engine_waits.mongo_snapshot(Client(2), "srv")
        after, _ = engine_waits.mongo_snapshot(Client(7), "srv")
        waits, reason = engine_waits.mongo_delta(before, after)
        assert reason is None
        assert waits["events"][0] == {"event": "locks.Global.r", "count": 5,
                                      "wait_s": 0.005}
        assert waits["gauges_after"]["tickets"] == {"read": {"out": 3}}
        assert "not reachable" in waits["mongot_metrics_reason"]
