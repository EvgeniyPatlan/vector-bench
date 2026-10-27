"""E21 -- the core a crash would have left, and the exit state nothing read.

THE RUN THIS IS FOR: `full-20261015-215006`, percona, `--profile full
--resource-pass both`. Twelve of its twenty-four `ops-percona-*.jsonl` files
hold no record at all; each of those units died inside the first INSERT with
`mariadb.InterfaceError: Lost connection to server during query` within
0.5-5.5 s. `tests/test_e10_failures_explain_themselves.py` fixed one half of
why that was unanswerable -- the engine's own error log never left the
container. This file is the other half, and it is four separate gaps, each
checked against the tree before being written:

  G1  nothing recorded the server container's EXIT STATE. `docker_ctl` had
      `is_running` and `container_exists`; both run `docker inspect` and throw
      the rest away. *OOMKilled* is the fact that settles a cgroup OOM against
      an in-process crash with no cooperation from the server at all, and it
      was never read. The reference run's peak server RSS was 48.5 GiB against
      a 112.5 GiB container cap, so the cgroup-OOM hypothesis was live and
      could not be closed.
  G2  `ContainerSpec` had no ulimit field and `_spec_args` emitted no
      `--ulimit`, so the core size limit was whatever the daemon defaulted to.
  G3  `config/engines/percona.yml`'s `server` blocks did not ask mysqld for a
      core, so one was never written whatever the ulimit said.
  G4  `OpsRun.teardown` stops the container, removes the volume and then
      `remove_tree_as_root`s the bind directory the volume lived in -- and
      `__exit__` is `teardown`. A core in the data mount died with the run.
      `run_script`, the lab route that would open one, mounts
      `harness/`, `scripts/` and `datasets/` and deliberately no writable
      path, so it could not see a core either.

WHAT IS NOT PROVEN HERE, stated first because it is the limit of this file.
The two mysqld flags G3 adds -- `--core-file` and
`--innodb-buffer-pool-in-core-file=OFF` -- have NEVER BEEN ACCEPTED BY A
RUNNING SERVER. An unknown server variable makes mysqld exit at startup, so
whether this base takes both spellings is a question for the rebuild and the
first `dev` run on the build host, not for a test on a machine with no image.
Everything below asserts what the orchestrator WOULD pass and what the config
DECLARES. Nothing here builds an image, starts a server, runs a profile or
reads a core.

PROBE STYLE, per this suite's convention
(`tests/test_e9_percona_bulk_build_crash.py`,
`tests/test_e10_failures_explain_themselves.py`): a decision test reports its
HONEST and its MUTANT value and asserts they differ, because an identical pair
means the probe is dead. The mutant arm is always an INPUT -- a different
engine, a spec with the field unset, a different inspect output -- never a
patched file, so nothing here can leave residue.

PROSE GUARDS ASSERT POLARITY AND PLACEMENT, NOT PRESENCE, which is the house
standard `tests/test_e1_percona_design.py` states in
`test_it_answers_the_ef_search_question_affirmatively`: "the affirmative
answer has to come before the evidence, and the negative answer must be
absent. Presence alone is what the hedge attack exploits." A guard here that
merely required the string `--core-file` to appear in the engine config would
be satisfied by a comment saying the flag was considered and rejected, and a
guard that merely required `"13 of 24"` to be absent would be satisfied by
"not 13 of 24". Both are written so the inverted form fails.
"""


import os
import re
import subprocess

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE_PATH = os.path.join(REPO, "config", "engines", "percona.yml")

#: The two flags G3 adds, in the spelling the config must carry.
CORE_FLAG = "--core-file"
BUFFER_POOL_FLAG = "--innodb-buffer-pool-in-core-file=OFF"


def _engine_text():
    with open(ENGINE_PATH) as fh:
        return fh.read()


class _Resolved:
    """The resource figures `server_args` substitutes.

    A plain object rather than a real `ResolvedResources`, for the reason
    `tests/test_e10_failures_explain_themselves.py`'s copy gives: nothing here
    depends on how they were derived, only that they are stable.
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
    pass_signature = "e21-fixed-signature"


def _ops_run(tmp_path, engine, dataset="gist-960-euclidean",
             tag="m8-incremental", resource_pass="normalized"):
    """An OpsRun with no containers behind it.

    `__init__` only computes names and reads the registry, so constructing one
    touches neither docker nor an engine.
    """
    from orchestrator import ops_pass
    from orchestrator.config import load_engine
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    return ops_pass.OpsRun(
        engine, load_engine(engine), _Resolved(), resource_pass,
        {"run_dir": str(run_dir),
         "engine_state": str(tmp_path / "state"),
         "harness": str(tmp_path / "harness"),
         "scripts": str(tmp_path / "scripts"),
         "datasets": str(tmp_path / "datasets"),
         "ops_results": str(run_dir)},
        "r1", dataset, tag)


def _make_bind_dir(tmp_path, run):
    """The host side of the bind-backed volume, as `create_volume` leaves it.

    The capture refuses to spawn a container when this does not exist -- there
    is nothing to look in, and `remove_tree_as_root` sets that precedent with
    its own `if not os.path.exists(path)`. So a test that wants the copy
    ATTEMPTED has to create it, and that is itself the proof the guard is
    load-bearing.
    """
    bind = tmp_path / "state" / "ops" / run.volume
    bind.mkdir(parents=True, exist_ok=True)
    return str(bind)


# ---------------------------------------------------------------------------
# G2 -- the ulimit reaches the argv, and nothing else does
# ---------------------------------------------------------------------------

#: `_spec_args` for a server spec, EXACTLY as it reads with no ulimits. A
#: literal rather than a comparison against a second call, because "identical
#: to today's" is a claim about today and a test that compares the code to
#: itself cannot make it. If a future change reorders these, this list is what
#: has to be re-derived deliberately.
TODAYS_SERVER_ARGV = [
    "docker", "run", "--name", "r1-percona-m8-srv",
    "--label", "vector-bench=1", "-d",
    "--network", "r1-percona-m8-net",
    "--cpuset-cpus", "0-7",
    "--memory", "8589934592", "--memory-swap", "8589934592",
    "--shm-size", "1g",
    "-e", "VB_SERVER_ARGS=--skip-name-resolve",
    "-v", "r1-percona-m8-data:/var/lib/vbench:rw",
    "vector-bench/percona-runtime", "server",
]


def _server_spec(**extra):
    from orchestrator.docker_ctl import ContainerSpec
    kwargs = dict(
        name="r1-percona-m8-srv", image="vector-bench/percona-runtime",
        network="r1-percona-m8-net", cpuset="0-7",
        memory_bytes=8 * 1024 ** 3, shm_size="1g",
        env={"VB_SERVER_ARGS": "--skip-name-resolve"},
        volumes=["r1-percona-m8-data:/var/lib/vbench:rw"],
        command=["server"], detach=True,
    )
    kwargs.update(extra)
    return ContainerSpec(**kwargs)


class TestTheUlimitReachesTheArgvAndNothingElseMoves:
    """G2. An unlimited core size is a `docker run` flag, not a server flag.

    mysqld's `--core-file` asks the process to dump; the process can still only
    dump as much as RLIMIT_CORE allows, and a container inherits the daemon's
    default, which on a stock install is the hard limit of the daemon's own
    process. Asking for the core without raising the limit produces a truncated
    core or none, and a truncated core is worse than no core: gdb opens it and
    prints a backtrace that stops where the file does.
    """

    def test_a_core_ulimit_becomes_a_docker_flag(self):
        """HONEST arm: `ulimits={"core": "-1"}`. MUTANT arm: the field unset,
        which is every other engine and what the tree did before this."""
        from orchestrator.docker_ctl import _spec_args
        honest = _spec_args(_server_spec(ulimits={"core": "-1"}))
        mutant = _spec_args(_server_spec(ulimits=None))
        print(f"HONEST (ulimits={{'core': '-1'}}) = "
              f"{[a for a in honest if a not in mutant]}")
        print(f"MUTANT (ulimits=None)            = "
              f"{[a for a in mutant if a not in honest]}")
        assert honest != mutant, (
            "DEAD PROBE: setting ulimits made no difference to the argv, so "
            f"this test is measuring nothing. Both arms produced {honest}")
        # Adjacent, in that order: `--ulimit` and its value are one option.
        assert "--ulimit" in honest, honest
        index = honest.index("--ulimit")
        assert honest[index + 1] == "core=-1", honest
        # Before the image, or docker reads it as an argument to the command.
        assert index < honest.index("vector-bench/percona-runtime"), honest

    def test_more_than_one_ulimit_each_get_their_own_flag(self):
        """`--ulimit a=1 --ulimit b=2`, not one flag with two values.

        Nothing in the tree needs a second ulimit today. The shape is asserted
        anyway because the field is a dict, so the next caller WILL put two in
        it, and a loop that emitted `--ulimit` once would make the second
        value an argument to the image.
        """
        from orchestrator.docker_ctl import _spec_args
        args = _spec_args(_server_spec(
            ulimits={"core": "-1", "nofile": "65536:65536"}))
        assert args.count("--ulimit") == 2, args
        pairs = [args[i + 1] for i, a in enumerate(args) if a == "--ulimit"]
        assert pairs == ["core=-1", "nofile=65536:65536"], pairs

    def test_a_spec_without_ulimits_is_byte_identical_to_todays_argv(self):
        """"The six other engines are untouched" is only checkable this way.

        Six of the seven engines declare no ulimits and must keep exactly the
        command line they had. A new field with a default is the quiet way to
        change every one of them -- an empty dict rendered as `--ulimit` with
        no value, or a `None` rendered as the string "None" -- and both would
        start a container that fails in a way that reads as an engine problem.
        """
        from orchestrator.docker_ctl import _spec_args
        assert _spec_args(_server_spec()) == TODAYS_SERVER_ARGV
        assert _spec_args(_server_spec(ulimits=None)) == TODAYS_SERVER_ARGV
        assert _spec_args(_server_spec(ulimits={})) == TODAYS_SERVER_ARGV

    def test_the_field_is_last_in_the_dataclass(self):
        """Where it has to be, and the reason is Python's, not taste.

        `EngineRuntime.error_log` records it: a field without a default cannot
        follow one that has it, so a defaulted field goes at the end rather
        than beside the fields it reads best next to. The same applies here,
        and `ContainerSpec` is constructed BY KEYWORD everywhere, so the
        position is invisible to callers and only this test notices if it
        moves to where it would break the class.
        """
        import dataclasses
        from orchestrator.docker_ctl import ContainerSpec
        names = [f.name for f in dataclasses.fields(ContainerSpec)]
        assert names[-1] == "ulimits", names


# ---------------------------------------------------------------------------
# G2 -- and the engine that asks for it is the only one that gets it
# ---------------------------------------------------------------------------

class TestTheEngineConfigDeclaresTheCoreUlimit:
    """Per-engine rather than global, so six configs stay as they are and the
    value lands in `run-manifest.json` with the rest of the runtime block."""

    def test_the_config_declares_it(self):
        from orchestrator.config import load_engine
        runtime = load_engine("percona")["runtime"]
        assert runtime["ulimits"]["core"] == "-1", runtime.get("ulimits")

    def test_the_registry_carries_it_and_only_for_this_engine(self):
        """HONEST arm: percona. MUTANT arm: mariadb, which declares none."""
        from orchestrator import engines as engines_mod
        honest = dict(engines_mod.get("percona").ulimits)
        mutant = dict(engines_mod.get("mariadb").ulimits)
        print(f"HONEST (percona) = {honest}")
        print(f"MUTANT (mariadb) = {mutant}")
        assert honest != mutant, (
            "DEAD PROBE: the registry reports the same ulimits for an engine "
            f"that declares them and one that does not. Both gave {honest}")
        assert honest == {"core": "-1"}, honest
        assert mutant == {}, mutant

    def test_the_registry_entry_is_still_hashable(self):
        """Why `ulimits` is a tuple of pairs and not the dict it reads as.

        `EngineRuntime` is `@dataclass(frozen=True)`, so Python generates
        `__hash__` from its fields, and `hash()` on an instance works today --
        measured, not assumed. One `Dict` field makes every engine in the
        registry unhashable, and the failure would surface somewhere that puts
        one in a set, not here. `probe` is a tuple for the same reason.
        """
        from orchestrator import engines as engines_mod
        assert hash(engines_mod.get("percona")) is not None
        assert hash(engines_mod.get("mariadb")) is not None

    def test_start_server_applies_them(self, tmp_path, monkeypatch):
        """The declaration is worthless if `_start_server` does not pass it.

        HONEST arm: percona. MUTANT arm: mariadb, whose spec must carry no
        ulimits at all rather than an empty dict -- see
        `test_a_spec_without_ulimits_is_byte_identical_to_todays_argv`.
        """
        from orchestrator import ops_pass

        def wire(engine):
            captured = {}
            monkeypatch.setattr(ops_pass.docker_ctl, "image_exists",
                                lambda *a, **k: True)
            monkeypatch.setattr(ops_pass.docker_ctl, "start",
                                lambda spec: captured.setdefault("spec", spec))
            monkeypatch.setattr(ops_pass.docker_ctl, "wait_healthy",
                                lambda *a, **k: None)
            _ops_run(tmp_path / engine, engine)._start_server()
            return captured["spec"].ulimits

        honest, mutant = wire("percona"), wire("mariadb")
        print(f"HONEST (percona) = {honest!r}")
        print(f"MUTANT (mariadb) = {mutant!r}")
        assert honest != mutant, (
            "DEAD PROBE: both engines' server specs carry the same ulimits, "
            f"so declaring them changed nothing. Both gave {honest!r}")
        assert honest == {"core": "-1"}, honest
        assert not mutant, mutant


# ---------------------------------------------------------------------------
# G3 -- the server is asked for a core, and for one without the buffer pool
# ---------------------------------------------------------------------------

class TestTheServerFlagsAskForACore:
    """Both flags, in both passes, and the second is not optional.

    With `innodb_buffer_pool_in_core_file` at its default ON the core contains
    the buffer pool. `config/resources/tuned.yml` sets `buffer_fraction: 0.35`
    and the reference machine's tuned container cap is 112.5 GiB
    (`orchestrator/manifest.py` quotes that figure), so the buffer pool alone
    is ~39 GiB of core, written into the data volume under `state/` on the
    filesystem the checkout is on. That is the difference between a usable
    artefact and a disk incident.
    """

    @pytest.fixture(scope="class")
    def percona(self):
        from orchestrator.config import load_engine
        return load_engine("percona")

    @pytest.mark.parametrize("resource_pass", ("normalized", "tuned"))
    def test_both_flags_are_rendered_in_both_passes(self, percona,
                                                    resource_pass):
        """In `server.common`, so neither pass can be the one that forgot.

        A core only matters on the path where the measurement is already lost,
        and that path exists in both passes -- the reference run's failures
        were 7 normalized and 6 tuned.
        """
        from orchestrator.config import server_args
        flags = server_args(percona, resource_pass, _Resolved())
        assert CORE_FLAG in flags, flags
        assert BUFFER_POOL_FLAG in flags, flags

    def test_no_other_engine_gains_them(self, percona):
        """HONEST arm: percona. MUTANT arm: every other engine, none of which
        is under investigation and none of whose argv may move."""
        from orchestrator import engines as engines_mod
        from orchestrator.config import load_engine, server_args
        honest = [f for f in server_args(percona, "tuned", _Resolved())
                  if "core" in f]
        others = {}
        for name in engines_mod.known_engines(refresh=True):
            if name == "percona":
                continue
            for resource_pass in ("normalized", "tuned"):
                flags = server_args(load_engine(name), resource_pass,
                                    _Resolved())
                found = [f for f in flags if "core-file" in f]
                if found:
                    others[f"{name}/{resource_pass}"] = found
        print(f"HONEST (percona)      = {honest}")
        print(f"MUTANT (every other)  = {others}")
        assert honest != list(others.values()), (
            "DEAD PROBE: percona and the other engines render the same "
            "core-file flags")
        assert honest, honest
        assert others == {}, others

    def test_the_buffer_pool_flag_is_off_and_not_merely_present(self, percona):
        """Polarity, not presence. `=ON` is the server's own default and would
        be the one spelling that makes the artefact unusable, so the value is
        asserted rather than the key."""
        from orchestrator.config import server_args
        flags = server_args(percona, "tuned", _Resolved())
        matching = [f for f in flags if "innodb-buffer-pool-in-core-file" in f]
        assert matching == [BUFFER_POOL_FLAG], matching

    def test_the_comment_says_the_flags_have_never_been_accepted(self):
        """POLARITY AND PLACEMENT, not presence.

        The failure this guards is a later editor deleting the caveat because
        the flags "obviously work", and the only reason they are not verified
        is that verifying them needs the rebuild. A guard that required the
        word "unverified" to appear somewhere in a 350-line config would be
        satisfied by a sentence about something else entirely, so:

          * PLACEMENT -- the caveat has to be in the `server:` block, above
            the flags it is about, not in the file's header where a reader of
            the flag will not meet it.
          * POLARITY -- it has to say the flags have NOT been accepted by a
            running server. The inverted form, "both flags were accepted by
            the server", is the thing that must fail.
        """
        text = _engine_text()
        server_block = text.split("\nserver:", 1)[1].split(
            "\nstorage_engines:", 1)[0]
        assert CORE_FLAG in server_block, "the flag is not in the server block"
        flag_at = server_block.index(CORE_FLAG)

        # Polarity: the negative claim, in one of the spellings that makes it.
        caveat = re.search(
            r"(never been accepted by a running server"
            r"|has never run|not been verified)", server_block,
            re.IGNORECASE)
        assert caveat, (
            "the server block does not say the flags are unverified; it must, "
            "because an unknown server variable makes mysqld exit at startup "
            "and nothing here has started one")
        # Placement: the caveat leads, the flags follow.
        assert caveat.start() < flag_at, (
            "the caveat must come before the flags it is about: a reader who "
            "stops at the flag has to have read it")
        # Polarity: nothing may claim the opposite.
        lowered = server_block.lower()
        for wrong in ("both flags were accepted", "verified on the build host",
                      "confirmed by a running server",
                      "accepted by the server"):
            assert wrong not in lowered, wrong
        # The gate is named, so a reader knows what would settle it.
        assert "dev" in server_block, (
            "the caveat must name the run that settles it")

    def test_the_comment_carries_the_buffer_pool_arithmetic(self):
        """Why the second flag exists, in numbers rather than in adjectives.

        A reader who does not see the figures will read
        `--innodb-buffer-pool-in-core-file=OFF` as a tidy-up and delete it the
        first time a core looks incomplete.
        """
        server_block = _engine_text().split("\nserver:", 1)[1].split(
            "\nstorage_engines:", 1)[0]
        assert "112.5" in server_block, (
            "the tuned container cap is the figure that makes this a disk "
            "incident rather than a preference")
        assert "0.35" in server_block or "39" in server_block, (
            "the buffer-pool fraction or the resulting core size has to be "
            "stated, or the cap alone proves nothing")


# ---------------------------------------------------------------------------
# G1 -- the exit state, and it cannot raise
# ---------------------------------------------------------------------------

def _inspect(stdout, returncode=0):
    """A `docker_ctl._run` that answers one `docker inspect`."""
    def run(args, check=True, timeout=120, capture=True):
        return subprocess.CompletedProcess(list(args), returncode, stdout, "")
    return run


class TestExitStateReturnsNoneForEveryFailure:
    """The contract `read_file_tail` and `copy_file_out` already document.

    The only caller runs in a `finally` on the path a crashed phase takes, so
    a container that has already gone, a daemon that is not answering and an
    inspect format the daemon does not understand must all come back as "no
    exit state" rather than as an exception that converts a lost measurement
    into a lost run.
    """

    def test_a_failed_inspect_is_not_an_error(self, monkeypatch):
        from orchestrator import docker_ctl
        monkeypatch.setattr(docker_ctl, "_run", _inspect("", returncode=1))
        assert docker_ctl.exit_state("gone") is None

    def test_a_daemon_that_is_not_answering_is_not_an_error(self, monkeypatch):
        from orchestrator import docker_ctl

        def explode(*a, **k):
            raise FileNotFoundError("docker")

        monkeypatch.setattr(docker_ctl, "_run", explode)
        assert docker_ctl.exit_state("x") is None

    def test_a_timeout_is_not_an_error(self, monkeypatch):
        from orchestrator import docker_ctl

        def explode(*a, **k):
            raise subprocess.TimeoutExpired("docker", 30)

        monkeypatch.setattr(docker_ctl, "_run", explode)
        assert docker_ctl.exit_state("x") is None

    def test_output_in_a_shape_it_does_not_understand_is_not_an_error(
            self, monkeypatch):
        from orchestrator import docker_ctl
        for stdout in ("", "\n", "not\ttab\tseparated\tbut\ttoo\tmany",
                       "notanint\tfalse\t2026-10-15T22:22:05Z\t"):
            monkeypatch.setattr(docker_ctl, "_run", _inspect(stdout))
            got = docker_ctl.exit_state("x")
            assert got is None or isinstance(got, dict), (stdout, got)
            if stdout.startswith("notanint"):
                assert got is None, (stdout, got)


class TestExitStateParsesWhatTheDaemonPrints:
    """`docker inspect --format` prints Go's rendering of the field, so every
    value arrives as text. `OOMKilled` is the one that matters and `"false"` is
    a non-empty string: anything that coerces rather than compares turns the
    answer "this was not an OOM" into "this was an OOM"."""

    HONEST = "137\ttrue\t2026-10-15T22:22:05.123456789Z\t"
    MUTANT = "1\tfalse\t2026-10-16T15:12:17.000000000Z\tOCI runtime error"

    def test_oom_killed_is_a_bool_and_not_the_string(self, monkeypatch):
        """HONEST arm: `OOMKilled` true. MUTANT arm: false."""
        from orchestrator import docker_ctl
        monkeypatch.setattr(docker_ctl, "_run", _inspect(self.HONEST))
        honest = docker_ctl.exit_state("x")
        monkeypatch.setattr(docker_ctl, "_run", _inspect(self.MUTANT))
        mutant = docker_ctl.exit_state("x")
        print(f"HONEST (OOMKilled true)  = {honest}")
        print(f"MUTANT (OOMKilled false) = {mutant}")
        assert honest["oom_killed"] != mutant["oom_killed"], (
            "DEAD PROBE: `true` and `false` parse to the same value, so this "
            "field cannot tell a cgroup OOM from an in-process crash -- which "
            "is the one thing it was added for")
        assert honest["oom_killed"] is True, honest
        assert mutant["oom_killed"] is False, mutant

    def test_the_exit_code_is_an_int(self, monkeypatch):
        from orchestrator import docker_ctl
        monkeypatch.setattr(docker_ctl, "_run", _inspect(self.HONEST))
        state = docker_ctl.exit_state("x")
        assert state["exit_code"] == 137 and isinstance(state["exit_code"], int)

    def test_it_carries_finished_at_and_the_daemons_own_error(self,
                                                              monkeypatch):
        from orchestrator import docker_ctl
        monkeypatch.setattr(docker_ctl, "_run", _inspect(self.MUTANT))
        state = docker_ctl.exit_state("x")
        assert state["finished_at"] == "2026-10-16T15:12:17.000000000Z"
        assert state["error"] == "OCI runtime error"

    def test_it_asks_for_all_four_fields_in_one_inspect(self, monkeypatch):
        """One `docker inspect`, not four: the capture chain already sits on
        510 s of worst-case timeouts before a failed setup gives up."""
        from orchestrator import docker_ctl
        calls = []

        def run(args, check=True, timeout=120, capture=True):
            calls.append(list(args))
            return subprocess.CompletedProcess(list(args), 0, self.HONEST, "")

        monkeypatch.setattr(docker_ctl, "_run", run)
        docker_ctl.exit_state("srv")
        assert len(calls) == 1, calls
        argv = " ".join(calls[0])
        for field in (".State.ExitCode", ".State.OOMKilled",
                      ".State.FinishedAt", ".State.Error"):
            assert field in argv, (field, argv)


# ---------------------------------------------------------------------------
# G1 + G4 -- the two captures cannot raise, and they run before teardown
# ---------------------------------------------------------------------------

class _Exploding:
    """A `docker_ctl` whose every function raises.

    Not a subset: the point is that the capture survives a module that is
    broken in a way this test did not predict, which is the state a wedged
    daemon and a half-removed container actually produce.
    """

    def __init__(self, order=None):
        self.order = order

    def __getattr__(self, name):
        def explode(*a, **k):
            if self.order is not None:
                self.order.append(name)
            raise RuntimeError(f"docker_ctl.{name} is broken")
        return explode


class TestNeitherCaptureMayRaise:
    """`_save_server_error_log`'s docstring is the standard and the reason:
    "This is on the path a crashed phase takes, and a phase that has already
    lost its measurement must not also lose the run because its log could not
    be fetched."

    This is the assertion that matters most in this file, and its mutant is
    known: drop the `try` and it fails.
    """

    def test_the_exit_facts_capture_survives_a_broken_docker(self, tmp_path,
                                                             monkeypatch):
        from orchestrator import ops_pass
        run = _ops_run(tmp_path, "percona")
        monkeypatch.setattr(ops_pass, "docker_ctl", _Exploding())
        assert run._save_container_exit_facts() is None

    def test_the_core_capture_survives_a_broken_docker(self, tmp_path,
                                                       monkeypatch):
        from orchestrator import ops_pass
        run = _ops_run(tmp_path, "percona")
        _make_bind_dir(tmp_path, run)
        monkeypatch.setattr(ops_pass, "docker_ctl", _Exploding())
        assert run._save_server_core() is None

    def test_the_whole_capture_chain_survives_a_broken_docker(self, tmp_path,
                                                              monkeypatch):
        """And every one of them is attempted: an exception from the first
        must not skip the rest, which is the property `_save_server_log`
        already states for its two existing captures."""
        from orchestrator import ops_pass
        run = _ops_run(tmp_path, "percona")
        _make_bind_dir(tmp_path, run)
        order = []
        monkeypatch.setattr(ops_pass, "docker_ctl", _Exploding(order))
        assert run._save_server_log() is None
        print(f"attempted against a broken docker_ctl: {order}")
        assert "logs" in order, order
        assert "exit_state" in order, order
        assert "copy_glob_as_root" in order, order

    def test_an_unwritable_run_directory_is_not_an_error(self, tmp_path,
                                                         monkeypatch):
        """The other half of "nothing may raise": the fetch can succeed and
        the write still fail. `_read_error_log`'s own note -- "an unwritable
        scratch location is one more way to have no log" -- is the precedent.
        """
        from orchestrator import ops_pass
        run = _ops_run(tmp_path, "percona")
        _make_bind_dir(tmp_path, run)
        monkeypatch.setattr(ops_pass.docker_ctl, "exit_state",
                            lambda name: {"exit_code": 137, "oom_killed": True,
                                          "finished_at": "t", "error": ""})

        def refuse(*a, **k):
            raise OSError("read-only file system")

        monkeypatch.setattr(ops_pass.os, "makedirs", refuse)
        assert run._save_container_exit_facts() is None
        assert run._save_server_core() is None


class TestTheExitFactsAreWrittenWhereAFailedPhaseLeavesThem:
    """Not as record fields, and that is a decision rather than a shortcut.

    The case these facts exist for is a phase that wrote NO RECORD AT ALL --
    twelve of the reference run's twenty-four units. An `extra` key would
    therefore be absent in exactly the case it was added for. They go into the
    run directory beside the two log captures, through the same
    `save_phase_log` whose name carries the dataset for the reason that
    function's docstring gives.
    """

    def _wire(self, tmp_path, monkeypatch, state):
        from orchestrator import ops_pass
        monkeypatch.setattr(ops_pass.docker_ctl, "logs", lambda *a, **k: "")
        monkeypatch.setattr(ops_pass.docker_ctl, "read_file_tail",
                            lambda *a, **k: None)
        monkeypatch.setattr(ops_pass.docker_ctl, "copy_file_out",
                            lambda *a, **k: None)
        monkeypatch.setattr(ops_pass.docker_ctl, "copy_glob_as_root",
                            lambda *a, **k: [])
        monkeypatch.setattr(ops_pass.docker_ctl, "exit_state",
                            lambda name: state)

    def _exit_log(self, tmp_path):
        directory = tmp_path / "run" / "logs"
        if not directory.is_dir():
            return None
        names = [n for n in os.listdir(directory) if "server-exit-" in n]
        if not names:
            return None
        return (directory / names[0]).read_text()

    def test_the_three_facts_land_in_the_run_directory(self, tmp_path,
                                                       monkeypatch):
        self._wire(tmp_path, monkeypatch,
                   {"exit_code": 137, "oom_killed": True,
                    "finished_at": "2026-10-15T22:22:05Z", "error": ""})
        _ops_run(tmp_path, "percona")._save_server_log()
        text = self._exit_log(tmp_path)
        assert text, "no server-exit artefact was written"
        assert "137" in text
        assert "2026-10-15T22:22:05Z" in text
        print(f"artefact:\n{text}")

    def test_the_oom_verdict_is_stated_and_not_left_to_the_reader(
            self, tmp_path, monkeypatch):
        """HONEST arm: OOMKilled true. MUTANT arm: false.

        Polarity, because the artefact exists to settle one question and a
        file that printed `OOMKilled` without its value would settle nothing.
        """
        honest_dir, mutant_dir = tmp_path / "h", tmp_path / "m"
        for directory, oom in ((honest_dir, True), (mutant_dir, False)):
            self._wire(directory, monkeypatch,
                       {"exit_code": 137, "oom_killed": oom,
                        "finished_at": "t", "error": ""})
            _ops_run(directory, "percona")._save_server_log()
        honest = self._exit_log(honest_dir)
        mutant = self._exit_log(mutant_dir)
        print(f"HONEST (OOMKilled true)  = {honest!r}")
        print(f"MUTANT (OOMKilled false) = {mutant!r}")
        assert honest != mutant, (
            "DEAD PROBE: the artefact reads the same whether the kernel killed "
            "the server or it died on its own, which is the distinction it "
            "was added to record")
        assert "True" in honest or "true" in honest.lower().split("oom", 1)[-1]

    def test_the_name_carries_the_dataset_and_the_pass(self, tmp_path,
                                                       monkeypatch):
        """HONEST arm: two datasets. MUTANT arm: the same one twice.

        `save_phase_log` is opened "w", and run full-20261015-215006 kept 14
        log files for 16 phases because the name had no dataset in it. A
        four-dataset profile would lose three of four exit artefacts the same
        way.
        """
        state = {"exit_code": 1, "oom_killed": False, "finished_at": "t",
                 "error": ""}
        honest_dir, mutant_dir = tmp_path / "h", tmp_path / "m"
        for directory, second in ((honest_dir, "sift-128-euclidean"),
                                  (mutant_dir, "gist-960-euclidean")):
            self._wire(directory, monkeypatch, state)
            for dataset in ("gist-960-euclidean", second):
                _ops_run(directory, "percona",
                         dataset=dataset)._save_server_log()
        def exits(directory):
            d = directory / "run" / "logs"
            return sorted(n for n in os.listdir(d) if "server-exit-" in n)
        honest, mutant = exits(honest_dir), exits(mutant_dir)
        print(f"HONEST (two datasets)  = {honest}")
        print(f"MUTANT (one, repeated) = {mutant}")
        assert honest != mutant, (
            "DEAD PROBE: the dataset made no difference to the filename")
        assert len(honest) == 2, honest
        assert len(mutant) == 1, mutant
        assert all("normalized" in n for n in honest), honest

    def test_no_artefact_when_there_is_no_exit_state_to_report(
            self, tmp_path, monkeypatch):
        """A file saying nothing is worse than no file: `export` ships it and
        a reader spends time on it. `exit_state` returns None for a container
        that never existed, which is the missing-image path."""
        self._wire(tmp_path, monkeypatch, None)
        _ops_run(tmp_path, "percona")._save_server_log()
        assert self._exit_log(tmp_path) is None

    def test_the_facts_are_not_added_to_any_record(self):
        """The decision, guarded. A later change that "tidied" these into
        `extra` would put them in the one place the failing phase cannot
        write, and would owe a measurement-version bump for a field that is
        absent exactly when it is wanted.
        """
        import harness.metrics.records as records_mod
        source = open(records_mod.__file__).read()
        for needle in ("oom_killed", "OOMKilled", "exit_code"):
            assert needle not in source, (
                f"{needle} has appeared in the record schema; the exit facts "
                f"are run-directory artefacts on purpose -- see this class's "
                f"docstring")


class TestTheCaptureRunsBeforeTeardown:
    """G4, and the defect the whole unit exists to prevent.

    `teardown` stops the container, removes it, removes the network, removes
    the volume and then `remove_tree_as_root`s the bind directory the volume
    lived in -- and `__exit__` is `teardown`. A core written into the data
    mount is gone after the first of those calls that matters, so the capture
    has to precede `stop`, which is where
    `tests/test_e10_failures_explain_themselves.py` already draws the line for
    the error log and for the same reason.
    """

    def _wire(self, monkeypatch, tmp_path):
        from orchestrator import docker_ctl, ops_pass
        order = []
        run = _ops_run(tmp_path, "percona")
        bind = _make_bind_dir(tmp_path, run)

        def record(label, fn=None):
            def call(*a, **k):
                order.append(label)
                return fn(*a, **k) if fn else None
            return call

        for name, value in (
            ("create_network", lambda *a, **k: "net"),
            ("create_volume", lambda *a, **k: "vol"),
            ("image_exists", lambda *a, **k: True),
            ("start", lambda *a, **k: "cid"),
            ("stop", record("stop")),
            ("remove", record("remove")),
            ("remove_network", record("remove_network")),
            ("remove_volume", record("remove_volume")),
            ("remove_tree_as_root", record("remove_tree",
                                           lambda *a, **k: True)),
            ("container_exists", record("inspect", lambda *a, **k: True)),
            ("logs", record("logs", lambda *a, **k: "banner")),
            ("exit_state", record("exit_state", lambda name: {
                "exit_code": 134, "oom_killed": False,
                "finished_at": "2026-10-15T22:22:05Z",
                "error": ""})),
            ("read_file_tail", record("read_file_tail",
                                      lambda *a, **k: "[ERROR] signal 6")),
            ("copy_glob_as_root", record("core_copy", lambda *a, **k: [])),
        ):
            monkeypatch.setattr(ops_pass.docker_ctl, name, value)

        def never_ready(*a, **k):
            raise docker_ctl.DockerError(
                "container ...-srv did not become ready within 300s")

        monkeypatch.setattr(ops_pass.docker_ctl, "wait_healthy", never_ready)
        return order, bind

    def test_all_three_captures_precede_the_first_teardown_call(
            self, tmp_path, monkeypatch):
        """The ordering assertion. Its mutant is moving `_save_server_core`
        into `__exit__` after `teardown`, which is the defect being fixed."""
        from orchestrator import docker_ctl
        order, _bind = self._wire(monkeypatch, tmp_path)
        with pytest.raises(docker_ctl.DockerError):
            _ops_run(tmp_path, "percona").__enter__()
        print(f"order = {order}")
        for label in ("exit_state", "read_file_tail", "core_copy", "stop",
                      "remove_volume"):
            assert label in order, (label, order)
        first_teardown = min(order.index("stop"), order.index("remove_volume"))
        for label in ("exit_state", "read_file_tail", "core_copy"):
            assert order.index(label) < first_teardown, (
                f"{label} ran after teardown had begun, so the artefact it "
                f"was fetching had already been deleted. Order was {order}")

    def test_it_still_tears_down(self, tmp_path, monkeypatch):
        """The captures are added in front of teardown, not instead of it."""
        from orchestrator import docker_ctl
        order, _bind = self._wire(monkeypatch, tmp_path)
        with pytest.raises(docker_ctl.DockerError):
            _ops_run(tmp_path, "percona").__enter__()
        for call in ("stop", "remove", "remove_network", "remove_volume",
                     "remove_tree"):
            assert call in order, (call, order)

    def test_the_core_capture_is_not_inside_teardown(self):
        """Unconditional, so no guard depends on which teardown call is the
        operative one. `tests/test_e10_failures_explain_themselves.py` makes
        the same assertion about the error-log capture and says why."""
        source = open(os.path.join(REPO, "orchestrator",
                                   "ops_pass.py")).read()
        teardown = source.split("def teardown")[1].split("\n    def ")[0]
        assert "_save_server_core" not in teardown, (
            "the core copy has moved inside teardown, which removes the "
            "volume the core lives in")
        assert "_save_container_exit_facts" not in teardown, teardown[:200]
        assert "remove_volume" in teardown, (
            "if teardown no longer removes the volume, the before-teardown "
            "requirement needs re-deriving rather than re-pointing")


class TestTheCoreCopyGoesThroughAContainerAndLandsWhereExportCanSeeIt:
    """Why `copy_glob_as_root` and not `shutil.copy`.

    The engine writes the data directory as root inside the container, so the
    host side of a bind-backed volume is typically root:root 0640 and a
    non-root orchestrator cannot even list it --
    `docker_ctl.remove_tree_as_root` exists for that and branches on
    `os.getuid() == 0` for it. `read_file_tail` needs a RUNNING container and
    `copy_file_out` needs an EXACT path, and a core's filename is decided by
    the host's `core_pattern`, which this framework must not change. So the
    enumeration has to happen inside a container too.
    """

    def test_the_copy_is_asked_for_and_lands_under_the_run_directory(
            self, tmp_path, monkeypatch):
        from orchestrator import ops_pass
        run = _ops_run(tmp_path, "percona")
        bind = _make_bind_dir(tmp_path, run)
        calls = []
        monkeypatch.setattr(ops_pass.docker_ctl, "copy_glob_as_root",
                            lambda *a, **k: calls.append((a, k)) or [])
        run._save_server_core()
        assert len(calls) == 1, calls
        args, _kwargs = calls[0]
        assert args[0] == bind, args
        assert "core" in args[1], args
        dest = args[2]
        assert dest.startswith(str(tmp_path / "run")), dest
        # The same quadruple `save_phase_log` keys on, for the same reason.
        for part in ("percona", "gist-960-euclidean", "m8-incremental",
                     "normalized"):
            assert part in dest, (part, dest)

    def test_an_engine_that_asked_for_no_core_spawns_no_container(
            self, tmp_path, monkeypatch):
        """HONEST arm: percona, which declares `runtime.ulimits.core`.
        MUTANT arm: valkey, which does not.

        Looking for a core costs one container per failed phase. Six engines
        were never asked for one, and `_save_server_error_log` sets the
        precedent for gating on the declaration: "this is a no-op for them, so
        their behaviour is unchanged".
        """
        from orchestrator import ops_pass
        results = {}
        for engine in ("percona", "valkey"):
            directory = tmp_path / engine
            run = _ops_run(directory, engine)
            _make_bind_dir(directory, run)
            calls = []
            monkeypatch.setattr(ops_pass.docker_ctl, "copy_glob_as_root",
                               lambda *a, **k: calls.append(a) or [])
            run._save_server_core()
            results[engine] = len(calls)
        print(f"HONEST (percona, core ulimit declared) = {results['percona']}")
        print(f"MUTANT (valkey, none declared)         = {results['valkey']}")
        assert results["percona"] != results["valkey"], (
            "DEAD PROBE: declaring a core ulimit made no difference to "
            "whether a container was spawned to look for a core")
        assert results["percona"] == 1
        assert results["valkey"] == 0

    def test_a_missing_bind_directory_spawns_no_container(self, tmp_path,
                                                          monkeypatch):
        """`remove_tree_as_root`'s own first line is `if not os.path.exists`,
        and the reason transfers: when setup failed before the server wrote
        anything, spawning a container to look needs the very image whose
        absence may have caused the failure."""
        from orchestrator import ops_pass
        calls = []
        monkeypatch.setattr(ops_pass.docker_ctl, "copy_glob_as_root",
                            lambda *a, **k: calls.append(a) or [])
        _ops_run(tmp_path, "percona")._save_server_core()
        assert calls == [], calls

    def test_copy_glob_as_root_returns_empty_rather_than_raising(
            self, monkeypatch, tmp_path):
        from orchestrator import docker_ctl

        def explode(*a, **k):
            raise RuntimeError("no daemon")

        monkeypatch.setattr(docker_ctl, "run_foreground", explode)
        got = docker_ctl.copy_glob_as_root(str(tmp_path), "core*",
                                           str(tmp_path / "dest"), "img")
        assert got == []

    def test_it_mounts_the_source_read_only(self, monkeypatch, tmp_path):
        """The data directory of a server that may be about to be re-examined
        is not something a copy may write to."""
        from orchestrator import docker_ctl
        captured = {}

        def run_foreground(spec, **kwargs):
            captured["spec"] = spec
            return 0

        monkeypatch.setattr(docker_ctl, "run_foreground", run_foreground)
        source = tmp_path / "src"
        source.mkdir()
        docker_ctl.copy_glob_as_root(str(source), "core*",
                                     str(tmp_path / "dest"), "img")
        volumes = captured["spec"].volumes
        assert any(v == f"{source}:/src:ro" for v in volumes), volumes
        assert not any(v.startswith(f"{source}:") and v.endswith(":rw")
                       for v in volumes), volumes

    def test_the_copy_is_chowned_back_to_the_invoking_user(self, monkeypatch,
                                                           tmp_path):
        """`ann_pass.fix_ownership` is the precedent and its reason applies:
        without it the artefact is unreadable to the operator and to `export`.
        """
        from orchestrator import docker_ctl
        captured = {}
        monkeypatch.setattr(docker_ctl, "run_foreground",
                            lambda spec, **k: captured.setdefault("spec", spec)
                            and 0 or 0)
        source = tmp_path / "src"
        source.mkdir()
        docker_ctl.copy_glob_as_root(str(source), "core*",
                                     str(tmp_path / "dest"), "img")
        script = " ".join(captured["spec"].command or [])
        assert "chown" in script, script
        assert f"{os.getuid()}:{os.getgid()}" in script, script


# ---------------------------------------------------------------------------
# G4 -- the lab route can be given a path to read, and only to read
# ---------------------------------------------------------------------------

#: `run_script`'s volume list EXACTLY as it reads with no artifacts directory.
#: A literal for the reason `TODAYS_SERVER_ARGV` is one.
def _todays_script_volumes(tmp_path):
    return [f"{tmp_path / 'harness'}:/opt/harness:ro",
            f"{tmp_path / 'scripts'}:/opt/scripts:ro",
            f"{tmp_path / 'datasets'}:/datasets:ro"]


class TestRunScriptStillMountsNoWritablePath:
    """The property `run_script`'s docstring states and must keep: "What is
    deliberately *not* here is anywhere to write a measurement ... A lab
    script generates its own vectors and its own ground truth, and those
    numbers must never be able to reach a report and sit in a table beside
    corpus numbers looking like a measurement."

    An artefacts mount is a READ of something a run already produced, which is
    the opposite of that hazard -- but only while it is read-only.
    """

    def _volumes(self, tmp_path, monkeypatch, **kwargs):
        from orchestrator import ops_pass
        captured = {}
        monkeypatch.setattr(ops_pass.docker_ctl, "image_exists",
                            lambda *a, **k: True)
        monkeypatch.setattr(ops_pass.docker_ctl, "run_foreground",
                            lambda spec, **k: captured.setdefault("spec", spec)
                            and 0 or 0)
        monkeypatch.setattr(ops_pass.OpsRun, "_save_server_log",
                            lambda self: None)
        run = _ops_run(tmp_path, "percona")
        run.run_script("probe.py", [], **kwargs)
        return list(captured["spec"].volumes)

    def test_without_an_artifacts_dir_the_mounts_are_what_they_were(
            self, tmp_path, monkeypatch):
        assert (self._volumes(tmp_path, monkeypatch)
                == _todays_script_volumes(tmp_path))
        assert (self._volumes(tmp_path, monkeypatch, artifacts_dir=None)
                == _todays_script_volumes(tmp_path))

    def test_an_artifacts_dir_adds_exactly_one_read_only_mount(
            self, tmp_path, monkeypatch):
        """HONEST arm: a path. MUTANT arm: None."""
        artifacts = tmp_path / "results" / "run-1" / "cores"
        artifacts.mkdir(parents=True)
        honest = self._volumes(tmp_path, monkeypatch,
                               artifacts_dir=str(artifacts))
        mutant = self._volumes(tmp_path, monkeypatch, artifacts_dir=None)
        print(f"HONEST (a path) = {honest}")
        print(f"MUTANT (None)   = {mutant}")
        assert honest != mutant, (
            "DEAD PROBE: passing an artifacts directory mounted nothing")
        assert len(honest) == len(mutant) + 1, (honest, mutant)
        added = [v for v in honest if v not in mutant]
        assert added == [f"{artifacts}:/artifacts:ro"], added

    def test_nothing_in_the_lab_container_is_writable(self, tmp_path,
                                                      monkeypatch):
        """Polarity, and the mutant the spec names: `:rw` here is what would
        let a lab script's invented numbers reach a report."""
        artifacts = tmp_path / "results" / "run-1" / "cores"
        artifacts.mkdir(parents=True)
        volumes = self._volumes(tmp_path, monkeypatch,
                                artifacts_dir=str(artifacts))
        writable = [v for v in volumes if not v.endswith(":ro")]
        assert writable == [], (
            "every mount in the lab container is read-only on purpose; "
            f"these are not: {writable}")


# ---------------------------------------------------------------------------
# The host's core_pattern, which decides whether any of this produces a file
# ---------------------------------------------------------------------------

class TestTheHostCorePatternIsReportedAndNeverWritten:
    """A pipe in `/proc/sys/kernel/core_pattern` -- apport, systemd-coredump --
    means the kernel hands the core to a helper that does not exist in the
    container's mount namespace, so NO FILE IS WRITTEN and every other part of
    this unit is inert. Ubuntu ships exactly that.

    The framework must not change it. "A benchmark framework must not modify
    the machine it measures" is CLAUDE.md's rule about host dependencies and it
    applies with more force to a kernel sysctl that affects every other process
    on the box. So this prints the value and says plainly what it means, and
    the one-line opt-in belongs in the operator's procedure.
    """

    def _start(self, tmp_path, monkeypatch, engine, pattern):
        from orchestrator import ops_pass
        monkeypatch.setattr(ops_pass.docker_ctl, "image_exists",
                            lambda *a, **k: True)
        monkeypatch.setattr(ops_pass.docker_ctl, "start", lambda spec: "cid")
        monkeypatch.setattr(ops_pass.docker_ctl, "wait_healthy",
                            lambda *a, **k: None)
        monkeypatch.setattr(ops_pass, "host_core_pattern",
                            lambda: pattern)
        _ops_run(tmp_path, engine)._start_server()

    def test_a_pipe_is_called_out_as_producing_no_core(self, tmp_path,
                                                       monkeypatch, capsys):
        """HONEST arm: a pipe, the Ubuntu default. MUTANT arm: `core`, the
        kernel default, which does produce a file."""
        self._start(tmp_path / "h", monkeypatch, "percona",
                    "|/usr/share/apport/apport -p%p -s%s -c%c -d%d -P%P")
        honest = capsys.readouterr().out
        self._start(tmp_path / "m", monkeypatch, "percona", "core")
        mutant = capsys.readouterr().out
        print(f"HONEST (pipe) = {honest!r}")
        print(f"MUTANT (core) = {mutant!r}")
        assert honest != mutant, (
            "DEAD PROBE: a piped core_pattern and a plain one print the same "
            "thing, so an operator reading this cannot tell that no core will "
            "be written")
        assert "no core" in honest.lower(), honest
        assert "no core" not in mutant.lower(), mutant

    def test_it_is_only_printed_for_an_engine_that_asked_for_a_core(
            self, tmp_path, monkeypatch, capsys):
        """HONEST arm: percona. MUTANT arm: mariadb, whose console output must
        not gain a line about a facility it does not use."""
        self._start(tmp_path / "h", monkeypatch, "percona", "core")
        honest = "core_pattern" in capsys.readouterr().out
        self._start(tmp_path / "m", monkeypatch, "mariadb", "core")
        mutant = "core_pattern" in capsys.readouterr().out
        print(f"HONEST (percona) = {honest}")
        print(f"MUTANT (mariadb) = {mutant}")
        assert honest != mutant, "DEAD PROBE"
        assert honest and not mutant

    def test_reading_it_never_writes_it(self):
        """The one assertion that cannot be made from behaviour, because the
        behaviour being forbidden is one no test may perform.

        A test that set `kernel.core_pattern` to prove the framework does not
        would be the thing being forbidden, so this reads the source: every
        `open()` naming the sysctl must take no mode argument, which is
        read-only, and nothing may reach for `sysctl` at all.
        """
        source = open(os.path.join(REPO, "orchestrator",
                                   "ops_pass.py")).read()
        assert "/proc/sys/kernel/core_pattern" in source, (
            "core_pattern is not read at all")
        opens = [m.group(1) for m in re.finditer(r"open\(([^)]*)\)", source)
                 if "CORE_PATTERN" in m.group(1).upper()
                 or "core_pattern" in m.group(1)]
        assert opens, f"nothing opens the sysctl: {opens}"
        for inner in opens:
            assert "," not in inner, (
                f"open({inner}) passes a mode; the sysctl is read-only here")
        # The word appears in the comment that explains why it must not be
        # done, so what is forbidden is an INVOCATION: the name as an argv
        # entry, or the flag that writes one.
        for forbidden in ('"sysctl"', "'sysctl'", "sysctl -w", "sysctl_set"):
            assert forbidden not in source, (
                f"{forbidden!r}: nothing in the ops path may set a kernel "
                f"parameter on the machine under measurement")

    def test_an_unreadable_core_pattern_is_not_an_error(self, monkeypatch):
        """A non-Linux host, a locked-down /proc: the server still has to
        start."""
        from orchestrator import ops_pass
        monkeypatch.setattr(ops_pass, "open", _refuse_open, raising=False)
        assert ops_pass.host_core_pattern() is None


def _refuse_open(*args, **kwargs):
    raise OSError("no /proc here")


# ---------------------------------------------------------------------------
# The figure in the engine config, which was a statement about a moving run
# ---------------------------------------------------------------------------

class TestTheReferenceRunIsCountedCorrectly:
    """`runtime.error_log`'s comment said "13 of 24 ops units died inside the
    first INSERT". The archived run holds TWELVE empty `ops-percona-*.jsonl`
    files today, and the comment is checkable against the archive, so a reader
    who checks finds it wrong.

    WHY IT CHANGED, which is the part worth recording: it was 13 during the
    original pass (2026-10-15T21:50 to 2026-10-17T15:42), and a `--resume` of
    the same run id on 2026-10-19 re-ran
    `fashion-mnist-784-euclidean / normalized / M=16` and it survived, 20
    records at 08:45-08:55Z. The percona-only export taken on 2026-10-18 still
    shows 13 empty files and no checkpoint for that unit; the seven-engine
    bundle shows 12 and has one. So the figure was not an arithmetic error --
    it is a statement about a run that was still moving.

    AND THE "12 of 18" THAT THE PLAN READS AS A CONTRADICTION IS NOT ONE. The
    eighteen million-row configurations (gist, glove, sift x two passes x three
    M values) held 12 empty files both before and after the resume; the
    recovered unit is fashion-mnist, which is 60,000 rows. 12 + 1 = 13 then
    and 12 + 0 = 12 now, so the two figures agreed with each other all along.
    """

    def test_the_count_bound_to_the_claim_is_twelve(self):
        """POLARITY AND PLACEMENT, not absence.

        A guard asserting only that `"13 of 24"` is gone would pass against
        "not 13 of 24 but 12", against a paragraph that dropped the count
        entirely, and against the figure moving into a different sentence
        about something else. So this finds every `<n> of 24 ops units died`
        in the file, requires exactly one, and requires its number to be 12.
        """
        text = _engine_text()
        found = re.findall(r"(\d+) of (?:the )?24 ops units died", text)
        print(f"counts bound to 'of 24 ops units died': {found}")
        assert found, (
            "no sentence in this file states how many ops units died; the "
            "count is the evidence runtime.error_log exists for")
        # Every one of them, not the first: a file that stated the figure
        # twice and disagreed with itself is the defect this replaced.
        assert set(found) == {"12"}, found

    def test_the_count_is_inside_the_error_log_comment(self):
        """Placement: it is evidence for `runtime.error_log` existing, so it
        belongs in that key's comment and nowhere else."""
        text = _engine_text()
        block = text.split("\nruntime:", 1)[1]
        assert re.search(r"12 of (?:the )?24 ops units died", block), (
            "the count is not in the runtime block, where the key it is "
            "evidence for lives")

    def test_the_stale_figure_is_not_the_headline_anywhere(self):
        text = _engine_text()
        assert not re.search(r"13 of (?:the )?24 ops units", text), (
            "the pre-resume count is back as a claim about the run")

    def test_the_resume_that_moved_it_is_recorded(self):
        """Without this the next reader re-derives 13 from the 2026-10-18
        export, decides the comment is wrong again, and changes it back."""
        block = _engine_text().split("\nruntime:", 1)[1]
        assert "2026-10-19" in block, (
            "the date of the resume that recovered the twelfth unit has to be "
            "in the comment, or the figure cannot be reconciled with the "
            "percona-only export")
        assert "resum" in block.lower(), block[:0] or "no mention of a resume"
