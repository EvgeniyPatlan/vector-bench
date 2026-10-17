"""E10 -- a 42-hour run failed 9 of its 16 phases and the run directory could
not say why, for three separate reasons.

THE RUN: `full-20261015-215006`, percona only, `--profile full
--resource-pass both`, four datasets, 16 phases. What it left behind, all of
it checked by hand before this file was written:

  * 13 of its 24 `ops-percona-*.jsonl` files are zero bytes. Each of those
    units died inside the first INSERT with `mariadb.InterfaceError: Lost
    connection to server during query` from
    `harness/drivers/mysql_family.py`'s `_load_range`, within 0.5-5.5 s, with
    server RSS flat at the buffer-pool baseline and CPU at 0.16 cores.
  * All six `logs/percona-server-*.log` files are THREE LINES: the entrypoint
    banner. Whether the server took a signal, hit an assertion or refused the
    statement is not recoverable from that run, and cannot be.
  * `logs/` holds 14 files for 16 phases, and only ONE `percona-ann-*.log`
    per resource pass. The surviving `percona-ann-normalized.log` is gist's
    (`Got a train set of size (1000000 * 960)`); the sift-128 ANN phase had
    failed in six seconds in both passes and its output had been overwritten.
  * `percona-ann-normalized.log` completes 16 `Running query argument group`
    blocks where the rendered config asks for 24, ends at line 4171 with
    `mariadb.InterfaceError: Server has gone away` raised from the forked
    worker's `_insert_parallel`, then `Terminating 1 workers` -- and
    `checkpoints.json` nevertheless contains
    `normalized/percona/gist-960-euclidean/ann`. The phase ran 7 h 46 m
    (06:36:18 -> 14:22:58), exited 0, and `--resume` will never return to it.

THREE DEFECTS, one per heading below:

  D1  the engine's own error log never leaves the container. Four engines are
      started with `--log-error=<file>`, so `docker logs` is the banner and
      the file is in the volume that teardown deletes. mongodb redirects
      mongot's log into that volume too, with a shell `>` rather than a flag,
      which is why the guard here looks for every spelling and not just one.
  D2  the phase-log filename had no dataset in it and is opened "w", so a
      four-dataset profile wrote one name four times.
  D3  `after > 0` was taken as a complete sweep. ann-benchmarks runs each
      definition in a forked worker and never checks its exit code, so a
      sweep that lost one curve exits 0.

None of the three changes a measured number. All three change what is
recorded about a failure.

PROBE STYLE, per this suite's convention (see
`tests/test_e9_percona_bulk_build_crash.py`): every decision test reports its
HONEST and its MUTANT value and asserts they differ, because an identical
pair means the probe is dead and is measuring nothing. The mutant arm is
always an INPUT -- a different engine, a repeated dataset, a different result
count -- never a patched file, so nothing here can leave residue.

NO LIVE ENGINES. Nothing in this file builds an image, starts a container or
runs a profile. `docker exec` is faked at `docker_ctl.exec_in` and the
measurement container at `docker_ctl.run_foreground`; both fakes record their
argv so the tests can assert what WOULD have been run.
"""

import os
import re
import subprocess

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: The run this file was written from. Quoted so the evidence above can be
#: traced back to a directory rather than to a memory.
RUN_ID = "full-20261015-215006"

#: Engines whose entrypoint sends a log to a FILE, and the file each one
#: names. `mariadb123` is `alias_of: mariadb`, so it is started by the same
#: entrypoint and logs to the same path.
#:
#: `mongodb` is here and it is the odd one: `mongod` itself runs in the
#: foreground unredirected, so `docker logs` has it, but `start_mongot`
#: redirects MONGOT's log into the data volume -- and vector search on this
#: engine happens in mongot. The key is still `error_log` because that is the
#: mechanism it needs; the config comment says what it actually is.
FILE_LOGGING = {
    "percona": ("percona", "/var/lib/vbench/percona.err"),
    "mariadb": ("mariadb", "/var/lib/vbench/mariadb.err"),
    "mariadb123": ("mariadb", "/var/lib/vbench/mariadb.err"),
    "alisql": ("alisql", "/var/lib/vbench/alisql.err"),
    "mongodb": ("mongodb", "/var/lib/vbench/mongot/mongot.log"),
}

#: Engines whose every log reaches the container's stdout, so `docker logs`
#: already carries all of it and a second copy would be noise.
STDERR_LOGGING = ("pgvector", "valkey")

#: EVERY WAY A SERVER HERE COULD BE TOLD TO LOG TO A FILE, not just the
#: MySQL-family flag. The first version of the guard below grepped
#: `--log-error` alone, and its failure message -- "{engine}'s entrypoint now
#: redirects the server's log to a file" -- was ALREADY UNTRUE OF MONGODB when
#: it was written: `start_mongot` redirects with a plain shell `>`. The test
#: passed only because it was looking for a flag mongod does not take. A guard
#: that can only see one vendor's spelling is a guard that reports what it
#: happens to understand.
LOG_TO_FILE_MARKERS = ("--log-error", "--logpath", "--logfile",
                       "logging_collector", "log_directory", ".log\"",
                       ".log'", ".log ")


def _entrypoint(name):
    path = os.path.join(REPO, "docker", name, f"entrypoint-{name}.sh")
    with open(path) as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# D1 -- the engine's own error log never leaves the container
# ---------------------------------------------------------------------------

class TestTheFileLoggingEnginesDeclareTheirErrorLog:
    """`runtime.error_log` is the only thing that knows the file exists.

    It is not derivable: the path is chosen by the entrypoint, the orchestrator
    never sees that shell script, and `docker logs` gives no hint that the
    interesting stream went elsewhere. Five engines need it and two must not
    have it, because for those two the existing capture is already complete
    and a second copy of the same lines would be noise.
    """

    @pytest.mark.parametrize("engine", sorted(FILE_LOGGING))
    def test_it_is_declared(self, engine):
        from orchestrator import engines as engines_mod
        want = FILE_LOGGING[engine][1]
        got = engines_mod.registry(refresh=True)[engine].error_log
        assert got == want, (
            f"{engine}'s entrypoint sends a log to a file, so without "
            f"runtime.error_log those diagnostics stay in the container and "
            f"go with the volume; expected {want!r}, got {got!r}")

    @pytest.mark.parametrize("engine", sorted(set(FILE_LOGGING) - {"mongodb"}))
    def test_it_matches_the_entrypoint_that_starts_the_server(self, engine):
        """The config and the shell script must name the same file.

        A LINE CITATION WOULD ROT HERE, so this reads the value instead: the
        entrypoint's `LOG_FILE` default is the single place the path is chosen
        and the only value mysqld is given, so comparing against it is a
        check that survives the file being reorganised.
        """
        image, declared = FILE_LOGGING[engine]
        source = _entrypoint(image)
        match = re.search(r'LOG_FILE="\$\{VB_LOG_FILE:-([^}]+)\}"', source)
        assert match, (
            f"docker/{image}/entrypoint-{image}.sh no longer sets LOG_FILE "
            f"from VB_LOG_FILE; config/engines/{engine}.yml's error_log is "
            f"derived from that default and can no longer be checked")
        assert match.group(1) == declared, (
            f"config/engines/{engine}.yml declares error_log={declared!r} but "
            f"the entrypoint logs to {match.group(1)!r}; the orchestrator "
            f"would copy out a file the server never wrote")
        assert '--log-error="$LOG_FILE"' in source, (
            f"docker/{image}/entrypoint-{image}.sh no longer passes LOG_FILE "
            f"to --log-error, so the declared error_log is not where the "
            f"server logs")

    def test_the_mongot_path_is_composed_from_the_entrypoint(self):
        """mongodb spells it differently, so it is checked differently.

        No flag and no LOG_FILE: `start_mongot` redirects with a shell `>` to
        `${MONGOT_DATA}/mongot.log`. Both halves are read here, so a change to
        either the variable's default or the redirection fails rather than
        leaving the config pointing at a file nothing writes.
        """
        source = _entrypoint("mongodb")
        match = re.search(r'MONGOT_DATA="\$\{VB_MONGOT_DATA:-([^}]+)\}"', source)
        assert match, "entrypoint-mongodb.sh no longer sets MONGOT_DATA"
        assert '>"${MONGOT_DATA}/mongot.log"' in source, (
            "entrypoint-mongodb.sh no longer redirects mongot into "
            "MONGOT_DATA/mongot.log")
        assert f"{match.group(1)}/mongot.log" == FILE_LOGGING["mongodb"][1]

    def test_mongod_itself_still_reaches_stdout(self):
        """Which is why the declared path is mongot's and not mongod's, and why
        the stdout capture is still worth keeping for this engine."""
        source = _entrypoint("mongodb")
        assert re.search(r"^\s*mongod --dbpath", source, re.MULTILINE), source[:0]
        assert "--logpath" not in source

    @pytest.mark.parametrize("engine", STDERR_LOGGING)
    def test_the_stdout_only_engines_declare_nothing(self, engine):
        """Absence is the statement that `docker logs` is already complete.

        Checked against every spelling in `LOG_TO_FILE_MARKERS`, not just the
        MySQL flag -- see the note on that tuple for why a narrower guard
        passed while being wrong.
        """
        from orchestrator import engines as engines_mod
        assert engines_mod.registry(refresh=True)[engine].error_log is None
        source = _entrypoint(engine)
        found = [m for m in LOG_TO_FILE_MARKERS if m in source]
        assert not found, (
            f"{engine}'s entrypoint now sends a log to a file ({found}), so it "
            f"needs a runtime.error_log like the other five")

    def test_every_engine_is_accounted_for_one_way_or_the_other(self):
        """A new engine must land in one of the two tables, not neither.

        Discovery is by directory, so an engine added later appears in the
        registry without appearing here, and would silently inherit the
        stdout-only treatment whether or not that is right for it.
        """
        from orchestrator import engines as engines_mod
        known = set(engines_mod.known_engines(refresh=True))
        classified = set(FILE_LOGGING) | set(STDERR_LOGGING)
        assert known == classified, (
            f"unclassified: {sorted(known - classified)}; "
            f"stale: {sorted(classified - known)}")


class _FakeExec:
    """A stand-in for `docker_ctl.exec_in` holding a fake container filesystem.

    Records every argv, so a test can assert what `docker exec` WOULD have
    been asked to run without running it.
    """

    def __init__(self, files=None, error="No such file or directory"):
        self.files = dict(files or {})
        self.error = error
        self.calls = []

    def __call__(self, name, args, timeout=60, check=False):
        args = list(args)
        self.calls.append((name, args))
        path = args[-1]
        if args[:2] == ["tail", "-n"] and path in self.files:
            return subprocess.CompletedProcess(args, 0, self.files[path], "")
        return subprocess.CompletedProcess(args, 1, "", f"tail: {path}: {self.error}")


class TestReadingTheErrorLogOutOfTheContainerCannotFail:
    """`read_file_tail` runs in a `finally` on the path a crashed phase takes.

    A phase that has already lost its measurement must not also lose the run
    because its log could not be fetched, so every failure has to come back as
    "no log" rather than as an exception.
    """

    def test_it_returns_the_tail_of_the_file(self, monkeypatch):
        from orchestrator import docker_ctl
        fake = _FakeExec({"/var/lib/vbench/percona.err": "[ERROR] assertion\n"})
        monkeypatch.setattr(docker_ctl, "exec_in", fake)
        got = docker_ctl.read_file_tail("srv", "/var/lib/vbench/percona.err",
                                        lines=20000)
        assert got == "[ERROR] assertion"
        assert fake.calls == [("srv", ["tail", "-n", "20000",
                                       "/var/lib/vbench/percona.err"])], (
            "the file has no bound -- mongod writes thousands of lines an "
            "hour -- so this must be a bounded tail and not a cat")

    def test_an_absent_file_is_not_an_error(self, monkeypatch):
        from orchestrator import docker_ctl
        monkeypatch.setattr(docker_ctl, "exec_in", _FakeExec({}))
        assert docker_ctl.read_file_tail("srv", "/nope.err") is None

    def test_an_unreadable_file_is_not_an_error(self, monkeypatch):
        from orchestrator import docker_ctl
        monkeypatch.setattr(docker_ctl, "exec_in",
                            _FakeExec({}, error="Permission denied"))
        assert docker_ctl.read_file_tail("srv", "/var/lib/vbench/x.err") is None

    def test_an_empty_file_reads_as_no_log(self, monkeypatch):
        from orchestrator import docker_ctl
        monkeypatch.setattr(docker_ctl, "exec_in", _FakeExec({"/x.err": "  \n"}))
        assert docker_ctl.read_file_tail("srv", "/x.err") is None

    def test_a_daemon_that_is_not_answering_is_not_an_error(self, monkeypatch):
        """`docker exec` against a container that has gone, or with no docker
        on PATH at all, must not propagate."""
        from orchestrator import docker_ctl

        def explode(*a, **k):
            raise FileNotFoundError("docker")

        monkeypatch.setattr(docker_ctl, "exec_in", explode)
        assert docker_ctl.read_file_tail("gone", "/x.err") is None


def _ops_run(tmp_path, engine, dataset="gist-960-euclidean", tag="m8-incremental"):
    """An OpsRun with no containers behind it.

    `__init__` only computes names and reads the registry, so constructing one
    touches neither docker nor an engine.
    """
    from orchestrator import ops_pass
    from orchestrator.config import load_engine
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    return ops_pass.OpsRun(
        engine, load_engine(engine), _Resolved(), "normalized",
        {"run_dir": str(run_dir), "engine_state": str(tmp_path / "state")},
        "r1", dataset, tag)


def _log_names(tmp_path):
    directory = tmp_path / "run" / "logs"
    return sorted(os.listdir(directory)) if directory.is_dir() else []


class TestTheServerLogThatMattersIsTheOneInTheContainer:
    """D1's probe, and the before-teardown requirement.

    `_save_server_log`'s own docstring already made the right argument -- "the
    half that would have said was discarded when the container was removed" --
    and then captured the container's stdout, which for the four MySQL-family
    engines is the entrypoint banner. Six such files in run
    full-20261015-215006 are three lines each.
    """

    def test_a_file_logging_engine_gets_its_error_log_archived(
            self, tmp_path, monkeypatch):
        """HONEST arm: percona, which declares `runtime.error_log`.
        MUTANT arm: valkey, which does not and must keep today's behaviour.
        """
        from orchestrator import docker_ctl, ops_pass
        monkeypatch.setattr(ops_pass.docker_ctl, "logs",
                            lambda *a, **k: "[vb-percona] exec mysqld ...")

        honest_dir = tmp_path / "honest"
        honest_dir.mkdir()
        monkeypatch.setattr(docker_ctl, "exec_in", _FakeExec(
            {"/var/lib/vbench/percona.err":
             "[ERROR] [MY-013183] InnoDB: Assertion failure\n"}))
        _ops_run(honest_dir, "percona")._save_server_log()
        honest = [n for n in _log_names(honest_dir) if "server-error" in n]

        mutant_dir = tmp_path / "mutant"
        mutant_dir.mkdir()
        monkeypatch.setattr(docker_ctl, "exec_in", _FakeExec(
            {"/var/lib/vbench/percona.err": "would not be read\n"}))
        _ops_run(mutant_dir, "valkey")._save_server_log()
        mutant = [n for n in _log_names(mutant_dir) if "server-error" in n]

        print(f"HONEST (percona, error_log declared) = {honest}")
        print(f"MUTANT (valkey, no error_log)        = {mutant}")
        assert honest != mutant, (
            "DEAD PROBE: declaring runtime.error_log made no difference to "
            f"what was archived, so this test is measuring nothing. Both arms "
            f"produced {honest}")
        assert len(honest) == 1, honest
        assert mutant == [], (
            "an engine without runtime.error_log must keep today's behaviour "
            f"exactly; valkey produced {mutant}")
        text = (honest_dir / "run" / "logs" / honest[0]).read_text()
        assert "Assertion failure" in text

    def test_it_asks_for_the_path_the_engine_config_names(self, tmp_path,
                                                          monkeypatch):
        """The whole fix is worthless if it fetches the wrong file."""
        from orchestrator import docker_ctl, ops_pass
        monkeypatch.setattr(ops_pass.docker_ctl, "logs", lambda *a, **k: "")
        fake = _FakeExec({"/var/lib/vbench/alisql.err": "[Note] shutdown\n"})
        monkeypatch.setattr(docker_ctl, "exec_in", fake)
        _ops_run(tmp_path, "alisql")._save_server_log()
        assert any(argv[-1] == "/var/lib/vbench/alisql.err"
                   for _name, argv in fake.calls), fake.calls

    def test_the_two_server_streams_land_in_two_files(self, tmp_path,
                                                       monkeypatch):
        """The stdout capture STAYS, and does not become the error log.

        The banner it holds records the exact argv mysqld was started with,
        which is how run full-20261015-215006's buffer-pool size and its
        `--innodb-hnsw-max-memory=0` were confirmed after the fact. Two
        streams, two names.
        """
        from orchestrator import docker_ctl, ops_pass
        monkeypatch.setattr(ops_pass.docker_ctl, "logs",
                            lambda *a, **k: "[vb-percona] exec mysqld ...")
        monkeypatch.setattr(docker_ctl, "exec_in", _FakeExec(
            {"/var/lib/vbench/percona.err": "[ERROR] gone\n"}))
        _ops_run(tmp_path, "percona")._save_server_log()
        names = _log_names(tmp_path)
        assert len(names) == 2, names
        assert any("-server-error-m8-incremental-" in n for n in names), names
        assert any("-server-m8-incremental-" in n for n in names), names

    def test_an_empty_stdout_does_not_skip_the_error_log(self, tmp_path,
                                                         monkeypatch):
        """A server that said nothing on stdout is exactly the one whose file
        log matters, so the two captures cannot be chained."""
        from orchestrator import docker_ctl, ops_pass
        monkeypatch.setattr(ops_pass.docker_ctl, "logs", lambda *a, **k: "")
        monkeypatch.setattr(docker_ctl, "exec_in", _FakeExec(
            {"/var/lib/vbench/percona.err": "[ERROR] signal 11\n"}))
        _ops_run(tmp_path, "percona")._save_server_log()
        assert [n for n in _log_names(tmp_path) if "server-error" in n]

    def test_a_missing_error_log_is_reported_and_not_fatal(self, tmp_path,
                                                            monkeypatch, capsys):
        """Losing the log is bad. Losing the run because the log could not be
        fetched would be worse -- this is called from a `finally` on the path a
        crashed phase takes."""
        from orchestrator import docker_ctl, ops_pass
        monkeypatch.setattr(ops_pass.docker_ctl, "logs", lambda *a, **k: "")
        monkeypatch.setattr(docker_ctl, "exec_in", _FakeExec({}))
        _ops_run(tmp_path, "percona")._save_server_log()
        assert [n for n in _log_names(tmp_path) if "server-error" in n] == []
        assert "error_log" in capsys.readouterr().err, (
            "an engine that declares an error log and then has none is worth "
            "a line; silence here reads as 'the server had nothing to say'")

    def test_neither_capture_can_end_the_run(self, tmp_path, monkeypatch):
        from orchestrator import docker_ctl, ops_pass

        def explode(*a, **k):
            raise RuntimeError("no such container")

        monkeypatch.setattr(ops_pass.docker_ctl, "logs", explode)
        monkeypatch.setattr(docker_ctl, "exec_in", explode)
        _ops_run(tmp_path, "percona")._save_server_log()      # must not raise

    def test_it_runs_before_teardown_on_both_container_paths(self):
        """`teardown` removes the volume the file lives in, so there is no
        second chance. Both `run_harness` and `run_script` reach the capture
        from a `finally`, inside the `with` whose `__exit__` is `teardown`.
        """
        with open(os.path.join(REPO, "orchestrator", "ops_pass.py")) as fh:
            source = fh.read()
        for method in ("run_harness", "run_script"):
            block = source.split(f"def {method}")[1].split("\n    def ")[0]
            assert "_save_server_log()" in block.split("finally:")[1], method
        teardown = source.split("def teardown")[1].split("\n    def ")[0]
        assert "remove_volume" in teardown, (
            "if teardown no longer removes the volume, the before-teardown "
            "requirement this test guards has changed shape")



class TestSetupThatNeverReachedTheHarnessStillExplainsItself:
    """The one path where the server's log was most wanted and least likely.

    `__enter__` runs three steps and the last two can fail -- a missing image,
    or a server that never answers inside `wait_healthy`'s five minutes. On
    failure it tears down and re-raises, and `run_harness` is never entered,
    so neither capture ran and `teardown` removed the volume the engine's own
    log lives in. A percona that refuses to start under a tuned
    `innodb_buffer_pool_size` left "did not become ready within 300s" and the
    three-line banner, and the assertion that caused it was deleted.
    """

    def _wire(self, monkeypatch, tmp_path, exists=True, error_log=True):
        """Every docker call `__enter__` and `teardown` make, replaced.

        `teardown` records whether the logs were already on disk when it ran,
        which is how the before-teardown ordering is asserted rather than
        assumed.
        """
        from orchestrator import docker_ctl, ops_pass
        seen = {}

        def remove_volume(name):
            seen["logs_at_teardown"] = _log_names(tmp_path)

        for name, value in (
            ("create_network", lambda *a, **k: "net"),
            ("create_volume", lambda *a, **k: "vol"),
            ("image_exists", lambda *a, **k: True),
            ("start", lambda *a, **k: "cid"),
            ("stop", lambda *a, **k: None),
            ("remove", lambda *a, **k: None),
            ("remove_network", lambda *a, **k: None),
            ("remove_volume", remove_volume),
            ("remove_tree_as_root", lambda *a, **k: True),
            ("container_exists", lambda *a, **k: exists),
            ("logs", lambda *a, **k: "[vb-percona] exec mysqld ..."),
        ):
            monkeypatch.setattr(ops_pass.docker_ctl, name, value)

        def never_ready(*a, **k):
            raise docker_ctl.DockerError(
                "container ...-srv did not become ready within 300s")

        monkeypatch.setattr(ops_pass.docker_ctl, "wait_healthy", never_ready)
        monkeypatch.setattr(docker_ctl, "exec_in", _FakeExec(
            {"/var/lib/vbench/percona.err":
             "[ERROR] [MY-012271] InnoDB: Cannot allocate memory for the "
             "buffer pool\n"} if error_log else {}))
        return seen

    def test_the_error_log_comes_out_before_the_volume_is_removed(
            self, tmp_path, monkeypatch):
        from orchestrator import docker_ctl, ops_pass
        seen = self._wire(monkeypatch, tmp_path)
        run = _ops_run(tmp_path, "percona")
        with pytest.raises(docker_ctl.DockerError):
            run.__enter__()
        names = _log_names(tmp_path)
        assert [n for n in names if "server-error" in n], names
        text = (tmp_path / "run" / "logs"
                / [n for n in names if "server-error" in n][0]).read_text()
        assert "Cannot allocate memory" in text, (
            "the reason the server refused to start is the whole point of "
            "this path")
        assert seen["logs_at_teardown"] == names, (
            "teardown removes the volume the file lives in, so a capture that "
            "runs after it captures nothing: at teardown the directory held "
            f"{seen['logs_at_teardown']} and afterwards {names}")

    def test_a_setup_that_never_made_a_container_writes_no_log(
            self, tmp_path, monkeypatch):
        """A missing image raises before `docker run`, and `docker logs` then
        answers with the daemon's "No such container" on stderr. Archiving
        that under a `-server-` name would put a Docker error in the run
        directory as though the server had said it."""
        self._wire(monkeypatch, tmp_path, exists=False)
        from orchestrator import docker_ctl
        run = _ops_run(tmp_path, "percona")
        with pytest.raises(docker_ctl.DockerError):
            run.__enter__()
        assert _log_names(tmp_path) == []

    def test_it_still_tears_down(self, tmp_path, monkeypatch):
        """The capture is added in front of teardown, not instead of it: a
        failed setup must still not leak a volume, a network and a root-owned
        corpus directory."""
        from orchestrator import docker_ctl
        seen = self._wire(monkeypatch, tmp_path)
        with pytest.raises(docker_ctl.DockerError):
            _ops_run(tmp_path, "percona").__enter__()
        assert "logs_at_teardown" in seen, "remove_volume was never reached"

    def test_a_capture_that_fails_does_not_mask_the_real_error(
            self, tmp_path, monkeypatch):
        """The setup failure is the diagnosis; losing the log must not replace
        it with an exception about logging."""
        from orchestrator import docker_ctl, ops_pass
        self._wire(monkeypatch, tmp_path, error_log=False)

        def explode(*a, **k):
            raise RuntimeError("daemon went away")

        monkeypatch.setattr(ops_pass.docker_ctl, "container_exists", explode)
        with pytest.raises(docker_ctl.DockerError) as caught:
            _ops_run(tmp_path, "percona").__enter__()
        assert "did not become ready" in str(caught.value)

# ---------------------------------------------------------------------------
# D2 -- each dataset overwrote the previous dataset's log
# ---------------------------------------------------------------------------

class TestEachDatasetKeepsItsOwnPhaseLog:

    def test_two_datasets_do_not_collide(self, tmp_path):
        """HONEST arm: two datasets, as a four-dataset profile runs them.
        MUTANT arm: the same dataset twice, which is what the name used to
        express for every dataset in the profile.
        """
        from orchestrator.docker_ctl import save_phase_log
        honest_dir, mutant_dir = tmp_path / "h", tmp_path / "m"
        for directory, second in ((honest_dir, "sift-128-euclidean"),
                                  (mutant_dir, "gist-960-euclidean")):
            save_phase_log(str(directory), "percona", "gist-960-euclidean",
                           "ann", "normalized", ["gist output"])
            save_phase_log(str(directory), "percona", second,
                           "ann", "normalized", ["second output"])
        honest = sorted(os.listdir(honest_dir / "logs"))
        mutant = sorted(os.listdir(mutant_dir / "logs"))
        print(f"HONEST (two datasets)  = {honest}")
        print(f"MUTANT (one, repeated) = {mutant}")
        assert honest != mutant, (
            "DEAD PROBE: the dataset made no difference to the filename, so "
            f"the second phase still overwrote the first. Both arms left "
            f"{honest}")
        assert len(honest) == 2, honest
        assert len(mutant) == 1, mutant
        kept = (honest_dir / "logs" / [n for n in honest if "gist" in n][0])
        assert kept.read_text().strip() == "gist output", (
            "in run full-20261015-215006 this is the file that was lost: the "
            "surviving percona-ann-normalized.log was gist's and sift's had "
            "been overwritten, so the six-second sift failure has no record")

    def test_the_dataset_is_required_and_cannot_be_defaulted(self):
        """A keyword with a default is how this comes back.

        Both call sites have the dataset in scope -- `ann_pass` as a local and
        `ops_pass` as `self.dataset` -- so there is nothing to default to, and
        a default would let the next caller silently reintroduce the collision.
        """
        import inspect
        from orchestrator.docker_ctl import save_phase_log
        parameter = inspect.signature(save_phase_log).parameters["dataset"]
        assert parameter.default is inspect.Parameter.empty
        assert parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD

    def test_every_other_field_still_separates(self, tmp_path):
        """The dataset is added to the name, not substituted for anything."""
        from orchestrator.docker_ctl import save_phase_log
        written = {
            save_phase_log(str(tmp_path), *args, ["x"]) for args in (
                ("percona", "gist-960-euclidean", "ann", "normalized"),
                ("percona", "gist-960-euclidean", "ann", "tuned"),
                ("percona", "gist-960-euclidean", "ops-m8-incremental",
                 "normalized"),
                ("percona", "gist-960-euclidean", "server-m8-incremental",
                 "normalized"),
                ("mariadb", "gist-960-euclidean", "ann", "normalized"),
            )
        }
        assert len(written) == 5, written

    def test_the_lab_shape_stays_sane(self, tmp_path):
        """`cmd_lab` drives OpsRun with dataset="lab", which is not a corpus.

        It still has to produce a legible, collision-free name, because a lab
        probe and a measurement can share a run directory.
        """
        from orchestrator.docker_ctl import save_phase_log
        path = save_phase_log(str(tmp_path), "valkey", "lab", "server-lab",
                              "tuned", ["probe output"])
        assert os.path.basename(path) == "valkey-lab-server-lab-tuned.log"

    def test_both_measurement_paths_pass_their_dataset(self):
        """A grep, because the defect was a caller that had the value in scope
        and did not pass it."""
        for module, expression in (("ann_pass", "engine, dataset,"),
                                   ("ops_pass", "self.engine, self.dataset,")):
            with open(os.path.join(REPO, "orchestrator",
                                   f"{module}.py")) as fh:
                source = fh.read()
            for call in source.split("save_phase_log(")[1:]:
                head = call.split(")")[0]
                assert expression in head, (
                    f"orchestrator/{module}.py has a save_phase_log call that "
                    f"does not pass its dataset: {head!r}")


# ---------------------------------------------------------------------------
# D3 -- a partially crashed ANN sweep was checkpointed as complete
# ---------------------------------------------------------------------------

#: What each engine's `full` grid asks for, per resource pass. MEASURED, by
#: rendering the config and counting the file names it leads to.
#:
#: THE WHOLE TABLE EXISTS BECAUSE ONE ROW IS NOT ENOUGH. Every `run_engine`
#: test below drives percona under `full`, and percona is 24 in BOTH passes --
#: so four separate defects were invisible: reading the count for a hardcoded
#: engine, returning a hardcoded 24 whenever the config parses, and getting
#: either aggregation wrong in a way that happens to land on 24 for this one
#: grid. mariadb is where it bites, at 24 normalized and 16 tuned, because the
#: tuned pass gives it its own `m_values_by_engine`.
FULL_PROFILE_COUNTS = {
    ("percona", "normalized"): 24, ("percona", "tuned"): 24,
    ("mariadb", "normalized"): 24, ("mariadb", "tuned"): 16,
    ("mariadb123", "normalized"): 24, ("mariadb123", "tuned"): 16,
    ("alisql", "normalized"): 24, ("alisql", "tuned"): 16,
    ("pgvector", "normalized"): 48, ("pgvector", "tuned"): 144,
    ("valkey", "normalized"): 24, ("valkey", "tuned"): 72,
    ("mongodb", "normalized"): 8, ("mongodb", "tuned"): 16,
}

#: Result file names an ACTUAL ann-benchmarks run left in
#: `results/annb/normalized/fashion-mnist-784-euclidean/10/`, with the
#: arguments that produced them.
#:
#: This is the one part of the reproduction in `ann_config` that cannot be
#: settled by reading upstream: it has to agree with `json.dumps`' spacing and
#: key order and with `re`'s `\W` class, exactly. These literals are the only
#: external ground truth available -- `results/` is gitignored, so they are
#: copied in rather than read, and the listing they came from is
#: `euclidean_M_16_engine_InnoDB_{10,40,160}.hdf5` under `mariadb/` and
#: `alisql/` and `euclidean_M_16_build_mode_post_efConstruction_200_{...}.hdf5`
#: under `pgvector/`.
REAL_RESULT_STEMS = [
    (["euclidean", {"M": 16, "engine": "InnoDB"}, 10],
     "euclidean_M_16_engine_InnoDB_10"),
    (["euclidean", {"M": 16, "engine": "InnoDB"}, 40],
     "euclidean_M_16_engine_InnoDB_40"),
    (["euclidean", {"M": 16, "engine": "InnoDB"}, 160],
     "euclidean_M_16_engine_InnoDB_160"),
    (["euclidean", {"M": 16, "build_mode": "post", "efConstruction": 200}, 10],
     "euclidean_M_16_build_mode_post_efConstruction_200_10"),
    (["euclidean", {"M": 16, "build_mode": "post", "efConstruction": 200}, 40],
     "euclidean_M_16_build_mode_post_efConstruction_200_40"),
    (["euclidean", {"M": 16, "build_mode": "post", "efConstruction": 200}, 160],
     "euclidean_M_16_build_mode_post_efConstruction_200_160"),
]


def _render(engine, profile_name, resource_pass):
    from orchestrator import ann_pass
    from orchestrator.config import load_profile, load_resources
    return ann_pass.render_config(engine, load_profile(profile_name),
                                  load_resources(resource_pass), resource_pass)


def _body(run_groups, name="e"):
    """A minimal rendered body, for the cases no profile produces."""
    return {"float": {"any": [{"name": name, "base_args": ["@metric"],
                               "run_groups": run_groups}]}}


class TestTheResultFileNameIsWhatGetsCounted:
    """ann-benchmarks names the file from the ARGUMENTS and nothing else.

    `results.build_result_filepath` is handed
    `definition.arguments + query_arguments`; the run group's name is not in
    the path. So two definitions with equal argument values write ONE file,
    and multiplying grid sizes over-counts -- which would make a healthy phase
    fail the shortfall check and blame a worker that never died. Counting the
    set of file names cannot make that mistake.
    """

    @pytest.mark.parametrize("data,stem", REAL_RESULT_STEMS,
                             ids=[s for _d, s in REAL_RESULT_STEMS])
    def test_it_reproduces_a_name_a_real_run_wrote(self, data, stem):
        from orchestrator import ann_config
        assert ann_config.result_stem(data) == stem

    def test_a_repeated_index_value_is_one_file(self):
        from orchestrator import ann_config
        body = _body({"g": {"arg_groups": [{"M": [16, 16]}],
                            "query_args": [[10, 20]]}})
        assert ann_config.expected_result_count(body) == 2, (
            "M: [16, 16] is two definitions writing the same two file names; "
            "the arithmetic this replaces said 4")

    def test_a_repeated_query_value_is_one_file(self):
        from orchestrator import ann_config
        body = _body({"g": {"arg_groups": [{"M": [16]}],
                            "query_args": [[10, 10, 20]]}})
        assert ann_config.expected_result_count(body) == 2

    def test_values_the_sanitiser_cannot_tell_apart_are_one_file(self):
        """`\\W+` -> `_` collapses the quotes, so 10 and "10" name one file."""
        from orchestrator import ann_config
        body = _body({"g": {"arg_groups": [{"M": [16]}],
                            "query_args": [[10, "10"]]}})
        assert ann_config.expected_result_count(body) == 1

    def test_two_run_groups_that_differ_only_by_name_share_their_files(self):
        """The case that would have broken a future engine branch.

        Every branch of `render_config` today happens to put its
        distinguishing value inside the arg dict -- `build_mode` for pgvector
        and valkey, `quantization` for mongodb, `engine` for the MySQL family.
        A branch distinguishing its run groups by NAME alone would write one
        set of files and be charged for two, and every complete phase of that
        engine would fail after hours with a message about a dead worker.
        """
        from orchestrator import ann_config
        body = _body({"a": {"arg_groups": [{"M": [8, 16, 32]}],
                            "query_args": [[10, 20]]},
                      "b": {"arg_groups": [{"M": [16, 32]}],
                            "query_args": [[10, 20]]}})
        assert ann_config.expected_result_count(body) == 6, (
            "six distinct (M, ef) pairs across the two groups; the arithmetic "
            "this replaces said 10")

    def test_equal_stems_under_two_algorithms_are_two_files(self):
        """Upstream writes under `results/<dataset>/<count>/<algorithm>/`, so
        the algorithm IS in the path even though the run group is not. Merging
        them would under-count, and under-counting hides a lost curve."""
        from orchestrator import ann_config
        body = {"float": {"any": [
            {"name": "one", "base_args": ["@metric"],
             "run_groups": {"g": {"arg_groups": [{"M": [16]}],
                                  "query_args": [[10]]}}},
            {"name": "two", "base_args": ["@metric"],
             "run_groups": {"g": {"arg_groups": [{"M": [16]}],
                                  "query_args": [[10]]}}},
        ]}}
        assert ann_config.expected_result_count(body) == 2

    def test_arg_group_entries_are_multiplied_not_added(self):
        """W10. `generate_arg_combinations` expands each `arg_groups` entry and
        then takes the cartesian PRODUCT of the expansions. Summing them --
        which is what this helper did first, and what its docstring wrongly
        claimed upstream does -- gives 5 where 6 files are written.
        """
        from orchestrator import ann_config
        body = _body({"g": {"arg_groups": [{"M": [8, 16, 32]},
                                           {"ef": [100, 200]}],
                            "query_args": [[10]]}})
        assert ann_config.expected_result_count(body) == 6

    def test_query_arg_lists_are_multiplied_not_replaced(self):
        """W11. `query_args` is expanded by the same cartesian product, so two
        lists are every pair of them, not the last list."""
        from orchestrator import ann_config
        body = _body({"g": {"arg_groups": [{"M": [16]}],
                            "query_args": [[10, 20], [1, 2, 3]]}})
        assert ann_config.expected_result_count(body) == 6

    def test_a_definition_with_no_query_args_still_writes_one_file(self):
        """`definition.query_argument_groups or [[]]` -- one iteration and one
        file, not zero."""
        from orchestrator import ann_config
        assert ann_config.expected_result_count(
            _body({"g": {"arg_groups": [{"M": [8, 16]}]}})) == 2

    def test_a_scalar_index_parameter_is_not_a_sweep(self):
        """`engine: "InnoDB"` is one value carried, not a list to multiply by."""
        from orchestrator import ann_config
        assert ann_config.expected_result_count(
            _body({"g": {"arg_groups": [{"M": [8, 16, 32],
                                         "efConstruction": [200],
                                         "engine": "InnoDB"}],
                         "query_args": [[10]]}})) == 3


class TestTheExpectedResultCountComesFromTheRenderedConfig:
    """The rendered config is the only source of truth for the grid.

    The profile, the resource pass and `render_config`'s per-engine branch all
    contribute to it and none of them reaches the result tree, so there is
    nothing else to compare `after` against.
    """

    def test_the_full_profile_asks_percona_for_twenty_four(self):
        """3 M x 1 efConstruction x 8 ef_search, in both passes.

        Which is the number run full-20261015-215006 is measured against: its
        normalized gist phase completed 16 `Running query argument group`
        blocks and its tuned gist phase completed 24.
        """
        from orchestrator import ann_config
        for resource_pass in ("normalized", "tuned"):
            body = _render("percona", "full", resource_pass)
            group = body["float"]["any"][0]["run_groups"]["innodb"]
            assert len(group["arg_groups"][0]["M"]) == 3
            assert len(group["arg_groups"][0]["efConstruction"]) == 1
            assert len(group["query_args"][0]) == 8
            assert ann_config.expected_result_count(body) == 24, resource_pass

    @pytest.mark.parametrize("engine,resource_pass",
                             sorted(FULL_PROFILE_COUNTS),
                             ids=[f"{e}-{p}"
                                  for e, p in sorted(FULL_PROFILE_COUNTS)])
    def test_every_engine_and_pass_gets_its_own_number(self, engine,
                                                       resource_pass,
                                                       tmp_path):
        """W6 and W7, through `write_config` -- the path `run_engine` uses.

        Reading the count for a fixed engine, or returning a fixed 24 whenever
        the config parses, passes every percona case and fails here.
        """
        from orchestrator import ann_config
        body = _render(engine, "full", resource_pass)
        ann_config.write_config(str(tmp_path), engine, body)
        assert (ann_config.expected_results(str(tmp_path), engine)
                == FULL_PROFILE_COUNTS[(engine, resource_pass)])

    def test_the_passes_disagree_for_at_least_one_engine(self):
        """Non-vacuity for the table above. If every engine were 24 in both
        passes the parametrised case could not see the pass being ignored --
        which is exactly how it was invisible while only percona was driven."""
        differing = {engine for (engine, _p) in FULL_PROFILE_COUNTS
                     if FULL_PROFILE_COUNTS[(engine, "normalized")]
                     != FULL_PROFILE_COUNTS[(engine, "tuned")]}
        assert "mariadb" in differing, differing
        assert len(differing) >= 4, differing

    def test_it_moves_with_the_grid(self):
        """HONEST arm: the `full` grid. MUTANT arm: one M value removed.

        A count that does not move with the config is a constant, and a
        constant cannot tell a short sweep from a complete one.
        """
        from orchestrator import ann_config
        honest_body = _render("percona", "full", "normalized")
        mutant_body = _render("percona", "full", "normalized")
        mutant_body["float"]["any"][0]["run_groups"]["innodb"][
            "arg_groups"][0]["M"] = [16]
        honest = ann_config.expected_result_count(honest_body)
        mutant = ann_config.expected_result_count(mutant_body)
        print(f"HONEST (M=[8,16,32]) = {honest}")
        print(f"MUTANT (M=[16])      = {mutant}")
        assert honest != mutant, (
            "DEAD PROBE: the expected count did not move when the grid did, "
            f"so it is not derived from the rendered config. Both arms "
            f"returned {honest}")
        assert (honest, mutant) == (24, 8)

    def test_every_profile_in_the_repo_renders_and_counts(self):
        """No profile may produce a body the counter reads as zero.

        Zero switches the shortfall check off silently, so a profile whose
        body this helper chokes on would lose the guard without saying so.
        `render_config` refusing outright is the only acceptable way for a
        profile and an engine not to go together.
        """
        from orchestrator import ann_config, ann_pass
        from orchestrator import engines as engines_mod
        from orchestrator.config import CONFIG_DIR, load_profile, load_resources
        names = sorted(f[:-4] for f in
                       os.listdir(os.path.join(CONFIG_DIR, "profiles"))
                       if f.endswith(".yml"))
        assert len(names) >= 8, names
        checked = 0
        for profile_name in names:
            profile = load_profile(profile_name)
            for resource_pass in ("normalized", "tuned"):
                resources = load_resources(resource_pass)
                for engine in engines_mod.known_engines(refresh=True):
                    try:
                        body = ann_pass.render_config(engine, profile,
                                                      resources, resource_pass)
                    except ValueError:
                        continue          # a refusal is a legitimate answer
                    count = ann_config.expected_result_count(body)
                    assert count > 0, (profile_name, resource_pass, engine)
                    checked += 1
        assert checked >= 100, checked

    def test_an_unreadable_config_switches_the_check_off(self, tmp_path):
        """Zero, not a guess. A rendered config that cannot be read is a
        reason to fall back to today's behaviour, not a reason to fail a phase
        whose grid is unknown.
        """
        from orchestrator import ann_config
        assert ann_config.expected_results(str(tmp_path), "percona") == 0
        path = ann_config.config_path(str(tmp_path), "percona")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            fh.write("float: [this: is\n  not: valid\n")
        assert ann_config.expected_results(str(tmp_path), "percona") == 0

    def test_a_body_with_an_unrunnable_run_group_counts_zero(self):
        """`prepare_args` raises on a run group carrying neither `args` nor
        `arg_groups`, so the run would not start at all. Counting only the
        groups that DO parse would under-count and hide a real shortfall."""
        from orchestrator import ann_config
        assert ann_config.expected_result_count(
            {"float": {"any": [{"name": "e", "base_args": [],
                                "run_groups": {
                                    "ok": {"arg_groups": [{"M": [8]}],
                                           "query_args": [[10]]},
                                    "bad": {"query_args": [[10]]}}}]}}) == 0

    def test_it_reads_the_file_the_container_will_read(self, tmp_path):
        from orchestrator import ann_config, ann_pass
        body = _render("percona", "full", "normalized")
        written = ann_pass.write_config(str(tmp_path), "percona", body)
        assert written == ann_config.config_path(str(tmp_path), "percona")
        assert ann_config.expected_results(str(tmp_path), "percona") == 24

    def test_the_names_are_still_reachable_through_ann_pass(self):
        """`cli.py` and a dozen existing tests call these on `ann_pass`. The
        split moved where they live, not who may call them."""
        from orchestrator import ann_config, ann_pass
        for name in ("config_path", "write_config", "expected_result_count",
                     "expected_results", "result_stem", "result_stems"):
            assert getattr(ann_pass, name) is getattr(ann_config, name), name


class _Resolved:
    """The resource figures `server_args` and `ann_fingerprint` substitute.

    A plain object rather than a real `ResolvedResources`, because nothing
    here depends on how they were derived -- only that they are stable, so
    the fingerprint and therefore the results tree is.
    """

    server_cpuset = "0-7"
    client_cpuset = "8-15"
    server_memory_bytes = 8 * 1024 ** 3
    client_memory_bytes = 2 * 1024 ** 3
    shm_size = "1g"
    buffer_bytes = 4 * 1024 ** 3
    graph_cache_bytes = 0
    maintenance_bytes = 512 * 1024 ** 2
    maxmemory_bytes = 0
    mongot_heap_bytes = 1 * 1024 ** 3
    build_threads = 8
    server_cpu_count = 8
    io_threads = 4
    pass_signature = "e10-fixed-signature"


class _Sampler:
    """`MemorySampler` without the thread or the container."""

    samples = 0

    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        pass

    def stop(self):
        pass


class _AnnHarness:
    """Everything `run_engine` needs, with the container replaced by a count.

    The fake `run_foreground` writes `files` result files where
    ann-benchmarks would and returns `rc`, which is the whole of the shape
    this defect lives in: ann-benchmarks forks a worker per definition, never
    checks its exit code, and returns 0 whether every definition finished or
    one of them died.
    """

    def __init__(self, tmp_path, monkeypatch, engine="percona",
                 dataset="gist-960-euclidean", resource_pass="normalized",
                 profile_name="full"):
        from orchestrator import ann_pass
        from orchestrator.config import load_engine, load_profile, load_resources

        self.ann_pass = ann_pass
        self.engine = engine
        self.dataset = dataset
        self.resource_pass = resource_pass
        self.engine_cfg = load_engine(engine)
        self.profile = load_profile(profile_name)
        self.resolved = _Resolved()
        self.specs = []

        self.paths = {}
        for key in ("run_dir", "datasets", "annb_results", "engine_state",
                    "work_annb"):
            path = tmp_path / key
            path.mkdir(parents=True, exist_ok=True)
            self.paths[key] = str(path)

        self.body = ann_pass.render_config(
            engine, self.profile, load_resources(resource_pass), resource_pass)
        ann_pass.write_config(self.paths["work_annb"], engine, self.body)

        self.results_dir = ann_pass.annb_results_dir(
            self.paths, resource_pass, self.resolved)

        monkeypatch.setattr(ann_pass.docker_ctl, "image_exists",
                            lambda image: True)
        monkeypatch.setattr(ann_pass.docker_ctl, "MemorySampler", _Sampler)
        monkeypatch.setattr(ann_pass, "prune_empty_results",
                            lambda directory, image: 0)
        self.monkeypatch = monkeypatch

    def seed(self, count):
        """Result files already on disk, i.e. `before`."""
        self._write(count)
        return self

    def _write(self, count):
        target = os.path.join(self.results_dir, self.dataset, "10", self.engine)
        os.makedirs(target, exist_ok=True)
        for index in range(count):
            open(os.path.join(target, f"r{index:02d}.hdf5"), "w").close()

    def drop_rendered_config(self):
        os.remove(self.ann_pass.config_path(self.paths["work_annb"],
                                            self.engine))
        return self

    def run(self, files, rc, lines=(), force=False):
        """Run the phase with a container that writes `files` and exits `rc`."""
        def run_foreground(spec, timeout=None, sink=None, line_filter=None,
                           stream=True):
            self.specs.append(spec)
            self._write(files)
            if sink is not None:
                sink.extend(lines)
            return rc

        self.monkeypatch.setattr(self.ann_pass.docker_ctl, "run_foreground",
                                 run_foreground)
        return self.ann_pass.run_engine(
            self.engine, self.dataset, self.profile, self.engine_cfg,
            self.resolved, self.resource_pass, self.paths, "r1", force=force)

    def kept_results(self):
        return self.ann_pass._count_results(self.results_dir, self.engine,
                                            self.dataset)


class TestAPartialSweepIsNotACompleteOne:
    """D3's probe and the three resume paths that already worked.

    What happened in run full-20261015-215006: the normalized gist-960 phase
    ran 7 h 46 m, lost its M=8 curve to `mariadb.InterfaceError: Server has
    gone away` during the parallel load, wrote 16 of 24 files, exited 0, and
    was checkpointed. `cli.py` saves a checkpoint only `if rc == 0`, so
    returning non-zero is what brings `--resume` back to the missing curve.
    """

    @pytest.mark.parametrize("engine,resource_pass",
                             sorted(FULL_PROFILE_COUNTS),
                             ids=[f"{e}-{p}"
                                  for e, p in sorted(FULL_PROFILE_COUNTS)])
    def test_every_engine_and_pass_is_judged_against_its_own_grid(
            self, engine, resource_pass, tmp_path, monkeypatch):
        """W6, and the reason every other case here cannot see it.

        `_AnnHarness` defaults to percona under `full`, and percona asks for 24
        in BOTH passes -- so `run_engine` reading the expectation for a
        hardcoded engine, or for a hardcoded pass, passed every test in this
        file. It is caught here and only here, because this drives the real
        `run_engine` for an engine whose number is different: mariadb is 24
        normalized and 16 tuned, so a complete 16-file tuned mariadb sweep
        judged against percona's 24 would be reported as a lost curve and
        checkpointed out of the run.

        Both directions, for all fourteen: the exact count passes, one file
        short fails.
        """
        want = FULL_PROFILE_COUNTS[(engine, resource_pass)]
        complete = _AnnHarness(tmp_path / "c", monkeypatch, engine=engine,
                               resource_pass=resource_pass)
        assert complete.run(want, 0) == 0, (
            f"{engine}/{resource_pass} wrote all {want} files and was still "
            f"reported as short")
        short = _AnnHarness(tmp_path / "s", monkeypatch, engine=engine,
                            resource_pass=resource_pass)
        assert short.run(want - 1, 0) == 1, (
            f"{engine}/{resource_pass} wrote {want - 1} of {want} files and "
            f"was reported as complete")

    def test_a_sweep_that_lost_a_curve_is_reported(self, tmp_path,
                                                    monkeypatch, capsys):
        """HONEST arm: 16 of the 24 the config asks for, exit 0 -- this run.
        MUTANT arm: all 24, exit 0 -- the same phase, complete.
        """
        honest = _AnnHarness(tmp_path / "h", monkeypatch).run(
            16, 0, lines=["mariadb.InterfaceError: Server has gone away",
                          "Terminating 1 workers"])
        message = capsys.readouterr().err
        mutant = _AnnHarness(tmp_path / "m", monkeypatch).run(24, 0)
        print(f"HONEST (16 of 24 files, exit 0) = {honest}")
        print(f"MUTANT (24 of 24 files, exit 0) = {mutant}")
        assert honest != mutant, (
            "DEAD PROBE: a sweep missing a third of its result files returned "
            f"the same code as a complete one, so --resume will never come "
            f"back for the missing curve. Both arms returned {honest}")
        assert honest == 1
        assert mutant == 0
        assert "16 of 24" in message, (
            f"the message has to name how many of how many, or the operator "
            f"cannot tell a lost curve from a narrower grid: {message!r}")

    def test_nothing_to_do_stays_a_success(self, tmp_path, monkeypatch):
        """`after == expected == before`: a complete tree, re-run. The whole
        point of leaving the results tree in place between runs."""
        harness = _AnnHarness(tmp_path, monkeypatch).seed(24)
        assert harness.run(24, 0) == 0
        assert harness.kept_results() == 24

    def test_nothing_to_run_stays_a_success(self, tmp_path, monkeypatch):
        """ann-benchmarks signals "every configuration is done" by raising, so
        the process exits non-zero for what is successful resumption here."""
        from orchestrator import ann_pass
        harness = _AnnHarness(tmp_path, monkeypatch).seed(24)
        assert harness.run(0, 1, lines=[
            "Traceback (most recent call last):",
            f"Exception: {ann_pass.NOTHING_TO_RUN}"]) == 0

    def test_nothing_to_run_over_a_short_tree_is_not_a_success(
            self, tmp_path, monkeypatch, capsys):
        """HONEST arm: 16 of 24 on disk. MUTANT arm: 24 of 24. Same exit, same
        output from ann-benchmarks; only the tree differs.

        "Nothing to run" does NOT mean "everything is done". `main()` raises
        it whenever the definition list is empty when it looks, and the resume
        filter is only one of the things that empties it: in --local mode
        `check_module_import_and_constructor` drops a definition whose module
        will not import with a logged WARNING and no raise, and it runs AFTER
        the resume filter. So a bench image whose algorithm module lost a
        dependency in a rebuild, over a tree holding 16 of 24, left 8
        definitions to run, dropped all 8, raised -- and this returned 0, the
        phase was checkpointed, and the report carried 16 of 24 curves. That
        is the defect the shortfall check exists to prevent, arriving through
        the one branch that was not gated on it.
        """
        from orchestrator import ann_pass
        said = [
            "WARNING ann_benchmarks.algorithms.percona.module could not be "
            "loaded; skipping",
            "Traceback (most recent call last):",
            f"Exception: {ann_pass.NOTHING_TO_RUN}",
        ]
        honest = _AnnHarness(tmp_path / "h", monkeypatch).seed(16).run(0, 1,
                                                                      lines=said)
        message = capsys.readouterr().err
        mutant = _AnnHarness(tmp_path / "m", monkeypatch).seed(24).run(0, 1,
                                                                      lines=said)
        print(f"HONEST (16 of 24 on disk, Nothing to run) = {honest}")
        print(f"MUTANT (24 of 24 on disk, Nothing to run) = {mutant}")
        assert honest != mutant, (
            "DEAD PROBE: a 'Nothing to run' over a tree missing a third of "
            f"its files returned the same code as one over a complete tree, "
            f"so --resume will never come back for it. Both arms returned "
            f"{honest}")
        assert honest == 1
        assert mutant == 0
        assert "16 result file(s)" in message and "24" in message, message
        assert "could not be loaded" in message, (
            "the message has to name the import failure as a cause, because "
            f"that is the one an operator would not guess: {message!r}")

    def test_nothing_to_run_with_an_unknown_grid_stays_a_success(
            self, tmp_path, monkeypatch):
        """`expected == 0` switches the gate off here too. Without a grid to
        compare against there is no evidence of a shortfall, and failing a
        resumption on no evidence would break the path that works."""
        from orchestrator import ann_pass
        harness = _AnnHarness(tmp_path, monkeypatch).seed(16)
        harness.drop_rendered_config()
        assert harness.run(0, 1, lines=[
            f"Exception: {ann_pass.NOTHING_TO_RUN}"]) == 0

    def test_a_tree_holding_more_than_this_profile_asks_for_is_not_a_failure(
            self, tmp_path, monkeypatch):
        """The results tree is shared across runs and keyed only by the
        resource configuration, so a wider grid measured earlier at the same
        budget is legitimately still in it."""
        harness = _AnnHarness(tmp_path, monkeypatch).seed(40)
        assert harness.run(40, 0) == 0
        assert harness.kept_results() == 40

    def test_force_still_recomputes_and_still_passes(self, tmp_path,
                                                      monkeypatch):
        harness = _AnnHarness(tmp_path, monkeypatch).seed(24)
        assert harness.run(24, 0, force=True) == 0
        assert "--force" in harness.specs[0].command

    def test_a_sweep_that_produced_nothing_is_still_reported(self, tmp_path,
                                                              monkeypatch):
        """The total loss was already caught; adding the partial case must not
        move it."""
        harness = _AnnHarness(tmp_path, monkeypatch)
        assert harness.run(0, 0, lines=["Terminating 1 workers"]) == 1

    def test_an_unknown_grid_falls_back_to_todays_behaviour(self, tmp_path,
                                                             monkeypatch):
        """With no rendered config to read, `expected` is 0 and the check is
        off -- a phase is not failed over a grid nothing could establish."""
        harness = _AnnHarness(tmp_path, monkeypatch).drop_rendered_config()
        assert harness.run(16, 0) == 0

    def test_a_non_zero_exit_is_still_a_non_zero_exit(self, tmp_path,
                                                       monkeypatch):
        harness = _AnnHarness(tmp_path, monkeypatch).seed(24)
        assert harness.run(24, 7) == 7

    def test_the_phase_log_lands_under_this_dataset(self, tmp_path,
                                                     monkeypatch):
        """D2 and D3 meet here: the log naming the dead worker is the one that
        used to be overwritten by the next dataset."""
        harness = _AnnHarness(tmp_path, monkeypatch)
        harness.run(16, 0, lines=["mariadb.InterfaceError: Server has gone away"])
        logs = os.listdir(os.path.join(harness.paths["run_dir"], "logs"))
        assert logs == ["percona-gist-960-euclidean-ann-normalized.log"], logs

    def test_a_short_sweep_is_not_also_reported_as_done(self, tmp_path,
                                                        monkeypatch, capsys):
        """The shortfall test runs before anything is called complete.

        It used to run after, so a seven-hour phase that lost a curve ended
        with "nothing to do: all 16 configuration(s) ... already have results"
        on stdout and "FAILED: 16 of 24" on stderr -- a log arguing with
        itself about what had just happened, in the place an operator looks
        first.
        """
        harness = _AnnHarness(tmp_path, monkeypatch).seed(16)
        assert harness.run(16, 0) == 1
        captured = capsys.readouterr()
        assert "nothing to do" not in captured.out, captured.out
        assert "new result file" not in captured.out, captured.out
        assert "16 of 24" in captured.err

    def test_a_complete_sweep_is_still_reported_as_done(self, tmp_path,
                                                        monkeypatch, capsys):
        """Non-vacuity for the test above: the reporting still happens, it is
        only gated behind the shortfall test."""
        harness = _AnnHarness(tmp_path, monkeypatch).seed(24)
        assert harness.run(24, 0) == 0
        assert "nothing to do" in capsys.readouterr().out

    def test_new_results_are_still_counted_out(self, tmp_path, monkeypatch,
                                               capsys):
        harness = _AnnHarness(tmp_path, monkeypatch).seed(8)
        assert harness.run(24, 0) == 0
        assert "16 new result file(s), 24 total" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# The citations this change moved, and could not see moving
# ---------------------------------------------------------------------------

#: What each bare `:N` in docs/09 §4.10 is pointing at, read out of the
#: section's own prose.
#:
#: WHY THIS IS HERE AT ALL. `tests/test_e1_percona_design.py` guards every
#: citation in that document of the form `path:N` -- but §4.10 writes nine of
#: them as a BARE `:N`, taking the file from its heading, and the guard's
#: pattern needs a path. So the one section devoted to `ann_pass.py` was the
#: one section whose pointers into it were unchecked, and all nine went stale
#: the moment this change put an import block above `render_config`. They were
#: found by hand, which is the thing this table exists to stop happening
#: again. Numbers are not asserted here: the anchor is, and the test re-derives
#: the number the way a reader would.
#:
#: Each anchor must occur ONCE in `ann_pass.py` -- that is what makes "the
#: section cites it" a checkable claim. (How often the SECTION cites it is a
#: different count and does not matter: `:150` is quoted three times.) The two
#: `efConstruction` citations have no unique anchor and get their own test.
SECTION_4_10_ANCHORS = (
    "def render_config",
    'if engine in MARIADB_ENGINES + ("alisql",)',
    'elif engine == "pgvector":',
    'ann.get("pgvector_build_modes"',
    'elif engine == "valkey":',
    'ann.get("valkey_build_modes"',
    'elif engine == "percona":',
)


def _section_4_10():
    with open(os.path.join(REPO, "docs", "09-adding-percona-server.md")) as fh:
        doc = fh.read()
    return doc.split("### 4.10")[1].split("\n### ")[0]


def _ann_pass_lines():
    with open(os.path.join(REPO, "orchestrator", "ann_pass.py")) as fh:
        return fh.read().splitlines()


class TestSection410StillPointsIntoAnnPass:
    """Nine pointers into a file this change reorganised, none of them guarded.

    The section is `### 4.10 orchestrator/ann_pass.py`, so its bare `:N`
    citations mean lines of that file. Two of them are the same string
    (`"efConstruction": ef_constructions`, which occurs three times in the
    file), so those are checked by position within their branch rather than by
    uniqueness.
    """

    def test_the_section_still_cites_the_file_by_line(self):
        """Non-vacuity: if the citations were ever rewritten into another form
        the tests below would pass by finding nothing."""
        cited = re.findall(r"`:(\d+)`", _section_4_10())
        assert len(cited) >= 9, cited
        assert len(set(cited)) == 9, sorted(set(cited))

    @pytest.mark.parametrize("anchor", sorted(SECTION_4_10_ANCHORS))
    def test_each_cited_line_holds_what_the_section_says(self, anchor):
        """Every anchor the section names must sit on a line it cites."""
        lines = _ann_pass_lines()
        cited = {int(n) for n in re.findall(r"`:(\d+)`", _section_4_10())}
        found = {n for n, line in enumerate(lines, 1) if anchor in line}
        assert len(found) == 1, (
            f"{anchor!r} occurs {len(found)} times in ann_pass.py; §4.10's "
            f"citation of it is only checkable while it is unique")
        assert found & cited, (
            f"docs/09 §4.10 describes {anchor!r} at one of {sorted(cited)}, "
            f"but it is at {sorted(found)}. Re-derive the citation with "
            f"grep -n against orchestrator/ann_pass.py as it now stands.")

    def test_the_two_ef_construction_citations_are_inside_their_branches(self):
        """`"efConstruction": ef_constructions` occurs three times -- pgvector,
        valkey and percona. §4.10 cites the first two, as the keys the MySQL
        family cannot emit, so each cited line must fall after its own branch
        and before the next one."""
        lines = _ann_pass_lines()
        cited = {int(n) for n in re.findall(r"`:(\d+)`", _section_4_10())}
        at = [n for n, line in enumerate(lines, 1)
              if '"efConstruction": ef_constructions' in line]
        assert len(at) == 3, at
        branches = {name: n for name, n in (
            (name, n) for n, line in enumerate(lines, 1)
            for name in ("pgvector", "valkey", "percona")
            if f'elif engine == "{name}":' in line)}
        for name, nxt in (("pgvector", "valkey"), ("valkey", "percona")):
            inside = [n for n in at
                      if branches[name] < n < branches[nxt]]
            assert len(inside) == 1, (name, inside)
            assert inside[0] in cited, (
                f"§4.10 cites the {name} efConstruction at one of "
                f"{sorted(cited)}; it is at {inside[0]}")
