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
      the file is in the volume that teardown deletes.
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

#: Engines started with `--log-error=<file>`, and the file each one names.
#: `mariadb123` is `alias_of: mariadb`, so it is started by the same
#: entrypoint and logs to the same path.
FILE_LOGGING = {
    "percona": ("percona", "/var/lib/vbench/percona.err"),
    "mariadb": ("mariadb", "/var/lib/vbench/mariadb.err"),
    "mariadb123": ("mariadb", "/var/lib/vbench/mariadb.err"),
    "alisql": ("alisql", "/var/lib/vbench/alisql.err"),
}

#: Engines that log to stderr, where `docker logs` already carries it all.
STDERR_LOGGING = ("pgvector", "mongodb", "valkey")


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
    interesting stream went elsewhere. Four engines need it and three must not
    have it, because for those three the existing capture is already complete
    and a second copy of the same lines would be noise.
    """

    @pytest.mark.parametrize("engine", sorted(FILE_LOGGING))
    def test_it_is_declared(self, engine):
        from orchestrator import engines as engines_mod
        want = FILE_LOGGING[engine][1]
        got = engines_mod.registry(refresh=True)[engine].error_log
        assert got == want, (
            f"{engine} is started with --log-error, so without "
            f"runtime.error_log its own diagnostics stay in the container and "
            f"go with the volume; expected {want!r}, got {got!r}")

    @pytest.mark.parametrize("engine", sorted(FILE_LOGGING))
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

    @pytest.mark.parametrize("engine", STDERR_LOGGING)
    def test_the_stderr_engines_declare_nothing(self, engine):
        """Absence is the statement that `docker logs` is already complete."""
        from orchestrator import engines as engines_mod
        assert engines_mod.registry(refresh=True)[engine].error_log is None
        assert "--log-error" not in _entrypoint(engine), (
            f"{engine}'s entrypoint now redirects the server's log to a file, "
            f"so it needs a runtime.error_log like the MySQL family's")


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
    and then captured the container's stdout, which for these four engines is
    the entrypoint banner. Six such files in run full-20261015-215006 are
    three lines each.
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

class TestTheExpectedResultCountComesFromTheRenderedConfig:
    """The rendered config is the only source of truth for the grid.

    The profile, the resource pass and `render_config`'s per-engine branch all
    contribute to it and none of them reaches the result tree, so there is
    nothing else to compare `after` against.
    """

    def _body(self, engine, profile_name, resource_pass):
        from orchestrator import ann_pass
        from orchestrator.config import load_profile, load_resources
        return ann_pass.render_config(engine, load_profile(profile_name),
                                      load_resources(resource_pass),
                                      resource_pass)

    def test_the_full_profile_asks_percona_for_twenty_four(self):
        """3 M x 1 efConstruction x 8 ef_search, in both passes.

        Which is the number run full-20261015-215006 is measured against: its
        normalized gist phase completed 16 `Running query argument group`
        blocks and its tuned gist phase completed 24.
        """
        from orchestrator import ann_pass
        for resource_pass in ("normalized", "tuned"):
            body = self._body("percona", "full", resource_pass)
            group = body["float"]["any"][0]["run_groups"]["innodb"]
            assert len(group["arg_groups"][0]["M"]) == 3
            assert len(group["arg_groups"][0]["efConstruction"]) == 1
            assert len(group["query_args"][0]) == 8
            assert ann_pass.expected_result_count(body) == 24, resource_pass

    def test_it_moves_with_the_grid(self, tmp_path):
        """HONEST arm: the `full` grid. MUTANT arm: one M value removed.

        A count that does not move with the config is a constant, and a
        constant cannot tell a short sweep from a complete one.
        """
        from orchestrator import ann_pass
        honest_body = self._body("percona", "full", "normalized")
        mutant_body = self._body("percona", "full", "normalized")
        mutant_body["float"]["any"][0]["run_groups"]["innodb"][
            "arg_groups"][0]["M"] = [16]
        honest = ann_pass.expected_result_count(honest_body)
        mutant = ann_pass.expected_result_count(mutant_body)
        print(f"HONEST (M=[8,16,32]) = {honest}")
        print(f"MUTANT (M=[16])      = {mutant}")
        assert honest != mutant, (
            "DEAD PROBE: the expected count did not move when the grid did, "
            f"so it is not derived from the rendered config. Both arms "
            f"returned {honest}")
        assert (honest, mutant) == (24, 8)

    def test_multiple_run_groups_are_summed(self):
        """pgvector renders two build modes as two run groups, and a definition
        in either writes result files."""
        from orchestrator import ann_pass
        body = self._body("pgvector", "full", "normalized")
        groups = body["float"]["any"][0]["run_groups"]
        assert len(groups) == 2, sorted(groups)
        assert ann_pass.expected_result_count(body) == 48

    def test_a_scalar_index_parameter_is_not_a_sweep(self):
        """`engine: "InnoDB"` is one value carried, not a list to multiply by.
        Counting it as a sweep would inflate the expectation and fail every
        complete phase."""
        from orchestrator import ann_pass
        assert ann_pass._combination_count(
            {"M": [8, 16, 32], "efConstruction": [200], "engine": "InnoDB"}) == 3

    def test_an_unreadable_config_switches_the_check_off(self, tmp_path):
        """Zero, not a guess. A rendered config that cannot be read is a
        reason to fall back to today's behaviour, not a reason to fail a
        phase whose grid is unknown.
        """
        from orchestrator import ann_pass
        assert ann_pass.expected_results(str(tmp_path), "percona") == 0
        path = ann_pass.config_path(str(tmp_path), "percona")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            fh.write("float: [this: is\n  not: valid\n")
        assert ann_pass.expected_results(str(tmp_path), "percona") == 0

    def test_it_reads_the_file_the_container_will_read(self, tmp_path):
        from orchestrator import ann_pass
        body = self._body("percona", "full", "normalized")
        written = ann_pass.write_config(str(tmp_path), "percona", body)
        assert written == ann_pass.config_path(str(tmp_path), "percona")
        assert ann_pass.expected_results(str(tmp_path), "percona") == 24


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
