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
      was never read. The reference run's peak server RSS ACROSS ITS OPS
      PHASES was 48.5 GiB against a 112.5 GiB container cap, so the
      cgroup-OOM hypothesis was live and could not be closed. (54.7 GiB is
      that run's percona peak overall, in `gist-960-euclidean-tuned-ann`; the
      ops scope is the right one here because the ops units are what died,
      and saying "the run" claimed more than the figure supports.)
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

#: Any sentence that makes a CLAIM about whether a server has taken these
#: flags, by what it says rather than by one of a handful of spellings. The
#: four-string blacklist this replaces was defeated twice: "Update: the
#: rebuild brought mysqld up clean with both of these present" carried none of
#: the four words, and "SETTLED: a live server has taken both of these"
#: carried a fifth.
_CLAIM_RE = re.compile(
    r"\b(accept\w*|verif\w*|settl\w*|confirm\w*|validat\w*"
    r"|came up|come up|brought\s+\w+\s+up|up clean|ran clean"
    r"|has taken|have taken|took both|tested)\b", re.IGNORECASE)

#: What turns such a claim into the honest negative one. A claim sentence
#: carrying none of these is asserting that a server HAS taken the flags.
_NEGATION_RE = re.compile(
    r"\b(never|not|no|cannot|can't|un(verified|accepted|tested|known)"
    r"|proposed|until|would|nothing|neither)\b", re.IGNORECASE)

#: The caveat itself, in any of the spellings that make the negative claim.
_CAVEAT_RE = re.compile(
    r"never been accepted by a running server|has never run"
    r"|not been verified", re.IGNORECASE)


def _comment_prose(lines):
    """Comment lines joined back into prose, so sentences can be examined.

    A YAML comment is hard-wrapped, so a sentence is split across lines and
    neither half can be judged on its own.
    """
    out = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            out.append(stripped.lstrip("#").strip())
    return " ".join(part for part in out if part)


def _sentences(prose):
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", prose)
            if part.strip()]


def _block_above(text, needle):
    """The contiguous comment block directly above the line holding `needle`.

    This is "the prose a reader meets immediately before the flag", which is
    what the placement arm has to be about -- a caveat with a contradiction
    wedged between it and the flag is not a caveat the reader believes.
    """
    lines = text.split("\n")
    index = next(i for i, line in enumerate(lines) if needle in line
                 and not line.strip().startswith("#"))
    block = []
    i = index - 1
    while i >= 0 and lines[i].strip().startswith("#"):
        block.append(lines[i])
        i -= 1
    return list(reversed(block))



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


#: What `exit_state` returns for a container that has ended. Both captures
#: take it now rather than fetching it themselves: `_save_server_log` does one
#: `docker inspect` and passes it down, and for the core capture it is also
#: the gate -- see `test_a_live_server_is_not_searched_for_a_core`.
ENDED = {"exit_code": 134, "oom_killed": False,
         "finished_at": "2026-10-15T22:22:05Z", "error": ""}


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
    """Per-engine rather than global, so six configs stay as they are.

    NOT so the value reaches the manifest, which an earlier version of this
    docstring claimed and which is false: `Manifest.set_engine` writes an
    engine's `source`, `build`, `images` and `presentation` and no `runtime`
    block at all, `docker_ctl.start` does not print its argv, and nothing
    `tee`s the console, so the `docker run` line is archived nowhere. The
    limit in force when a core was written is whatever this file said at the
    time. `ContainerSpec.ulimits` and the config comment both say so; this one
    said the opposite and was the only site.
    """

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
        and that path exists in both passes. Counted from the seven-engine
        bundle, the twelve units that wrote no record are 7 normalized and 5
        tuned. (An earlier version of this docstring said 7 and 6, which sums
        to the PRE-resume thirteen while taking its normalized half from the
        post-resume twelve -- the exact confusion
        `TestTheReferenceRunIsCountedCorrectly` exists to stop. Pre-resume is
        8 and 5.)
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

    def test_the_caveat_is_the_first_prose_a_reader_of_the_flag_meets(self):
        """PLACEMENT, and stronger than "somewhere above the flag".

        The first version of this required only that the caveat appear before
        the flag. That left the whole comment block between them open, and a
        contradiction wedged in there -- "Update: the rebuild brought mysqld
        up clean with both of these present", immediately above the flag --
        passed. A caveat a reader passes and then reads a retraction of is not
        a caveat.

        So: the contiguous comment block directly above `--core-file` must
        BEGIN with the caveat. Anything inserted between them displaces it
        from first position and fails here.
        """
        block = _block_above(_engine_text(), CORE_FLAG)
        assert block, "there is no comment above --core-file at all"
        sentences = _sentences(_comment_prose(block))
        assert sentences, "the comment above --core-file carries no prose"
        print(f"first sentence above the flag: {sentences[0]!r}")
        assert _CAVEAT_RE.search(sentences[0]), (
            "the first thing said above --core-file must be that the flags "
            "have never been accepted by a running server; it currently says "
            f"{sentences[0]!r}")
        # The gate is named, so a reader knows what would settle it.
        assert "dev" in " ".join(sentences), (
            "the caveat must name the run that settles it")

    def test_no_sentence_in_the_server_block_claims_a_server_took_the_flags(
            self):
        """POLARITY, asserted on the CLAIM rather than on a list of spellings.

        WHAT THIS REPLACES AND WHY. The first version banned four fixed
        strings, and two rewordings walked past it while asserting exactly
        what the four were there to forbid:

            "Update: the rebuild brought mysqld up clean with both of these
             present"                                          -> 54 passed
            "SETTLED: a live server has taken both of these; the paragraph
             below is kept for history only"                   -> 54 passed

        A blacklist resists spellings. What has to be resisted is the claim.
        So every sentence in the `server:` block that says anything about
        acceptance, verification, settling, confirming or a server coming up
        must also carry a negation -- which the honest caveat does and an
        assertion that the flags work does not.

        THE NEGATION MUST COME BEFORE THE CLAIM, in the same sentence, and
        that is not a detail. A sentence-wide check was the version this file
        shipped first, and its author's own fifth attack walked past it:

            "In fact these two were accepted by a live server on the build
             host already, so no further check is needed here."
                                                               -> 56 passed

        "no further check" supplied the negation token for a sentence
        asserting the opposite, with the caveat still first in the block so
        the placement arm was satisfied too. Requiring the negation to
        PRECEDE the claim token rejects it: "have never been accepted" has
        one before, "were accepted ... so no further check" has none.

        WHAT THIS STILL CANNOT DO, said rather than left to be discovered by
        the next reviewer. It is lexical. A claim phrased entirely outside
        `_CLAIM_RE`'s vocabulary gets through, and so does a negation placed
        before the claim for the sake of it -- "No, these two were accepted on
        the build host" passes, and the author checked that rather than
        assuming it. What the three arms buy is that every hedge tried so far
        has to get past a placement rule that does not depend on vocabulary,
        a positional polarity rule, and the four spellings already used; what
        they do not buy is a decision about whether prose is true, which no
        text guard can make.
        """
        server_block = _engine_text().split("\nserver:", 1)[1].split(
            "\nstorage_engines:", 1)[0]
        sentences = _sentences(_comment_prose(server_block.split("\n")))
        found = 0
        for sentence in sentences:
            for claim in _CLAIM_RE.finditer(sentence):
                found += 1
                before = sentence[:claim.start()]
                print(f"  claim {claim.group(0)!r} after {before[-60:]!r}")
                assert _NEGATION_RE.search(before), (
                    "this sentence claims a server has taken the flags, and "
                    "none has -- no image has been built from this engine. "
                    "If a `dev` run has now accepted them, the caveat comes "
                    "out with the same change rather than being contradicted "
                    f"beside it:\n  {sentence!r}")
        print(f"claim occurrences checked: {found}")
        assert found, (
            "nothing in the server block says anything about whether these "
            "flags have been accepted; the caveat is gone")
        # The four spellings that have actually been used to walk past an
        # earlier version of this guard. Kept as a regression arm: the rule
        # above is the general one, and these are the ones already tried.
        lowered = " ".join(sentences).lower()
        for wrong in ("both flags were accepted", "verified on the build host",
                      "confirmed by a running server",
                      "accepted by the server"):
            assert wrong not in lowered, wrong

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
        for stdout in ("", "\n", "false\t1\tfalse",
                       "false\tnotanint\tfalse\t2026-10-15T22:22:05Z\t"):
            monkeypatch.setattr(docker_ctl, "_run", _inspect(stdout))
            assert docker_ctl.exit_state("x") is None, stdout


class TestExitStateParsesWhatTheDaemonPrints:
    """`docker inspect --format` prints Go's rendering of the field, so every
    value arrives as text. `OOMKilled` is the one that matters and `"false"` is
    a non-empty string: anything that coerces rather than compares turns the
    answer "this was not an OOM" into "this was an OOM"."""

    HONEST = "false\t137\ttrue\t2026-10-15T22:22:05.123456789Z\t"
    MUTANT = ("false\t1\tfalse\t2026-10-16T15:12:17.000000000Z\t"
              "OCI runtime error")
    #: A server that is STILL UP, which is what `run_harness`'s `finally`
    #: usually meets: the client failed and the server never stopped. The
    #: daemon answers ExitCode 0 and Go's zero time for it.
    RUNNING = "true\t0\tfalse\t0001-01-01T00:00:00Z\t"
    #: Created and never started -- not running, and still no exit.
    NEVER_STARTED = "false\t0\tfalse\t0001-01-01T00:00:00Z\t"
    #: STARTED AGAIN after a stop: running, and carrying a real `FinishedAt`
    #: from its previous life. The zero-time check cannot see this one, which
    #: is why `Running` is a separate check rather than a redundant one.
    RESTARTED = "true\t139\tfalse\t2026-10-16T15:12:17.000000000Z\t"

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

    def test_a_container_that_has_not_ended_has_no_exit_state(self,
                                                              monkeypatch):
        """HONEST arm: a stopped container. MUTANT arm: the same output with
        `Running` true.

        "exit_code: 0" for a server that is still up would say it exited
        cleanly when it had not exited at all, and a reader would close the
        question on it. `run_harness`'s `finally` runs after the CLIENT
        failed, so this is the common case rather than an edge one.
        """
        from orchestrator import docker_ctl
        monkeypatch.setattr(docker_ctl, "_run", _inspect(self.MUTANT))
        honest = docker_ctl.exit_state("x")
        monkeypatch.setattr(docker_ctl, "_run", _inspect(self.RUNNING))
        mutant = docker_ctl.exit_state("x")
        print(f"HONEST (Running false) = {honest}")
        print(f"MUTANT (Running true)  = {mutant}")
        assert honest != mutant, (
            "DEAD PROBE: a running container and a stopped one report the "
            "same exit state, so the artefact would claim an exit that never "
            "happened")
        assert honest is not None
        assert mutant is None

    def test_a_restarted_container_does_not_report_its_previous_exit(
            self, monkeypatch):
        """The case the zero-time check cannot see, and the reason `Running`
        is a second check and not a duplicate of the first.

        `docker start` after a stop leaves `FinishedAt` pointing at the
        previous exit, so a container that is up right now carries a real
        timestamp and a real exit code belonging to a life that is over. Both
        would be archived as the exit of the phase being captured.
        """
        from orchestrator import docker_ctl
        monkeypatch.setattr(docker_ctl, "_run", _inspect(self.MUTANT))
        honest = docker_ctl.exit_state("x")
        monkeypatch.setattr(docker_ctl, "_run", _inspect(self.RESTARTED))
        mutant = docker_ctl.exit_state("x")
        print(f"HONEST (stopped, exit 1)        = {honest}")
        print(f"MUTANT (running, stale exit 139) = {mutant}")
        assert honest != mutant, "DEAD PROBE"
        assert honest is not None
        assert mutant is None, (
            "a running container's FinishedAt belongs to a previous life; "
            f"reporting it would archive someone else's exit: {mutant}")

    def test_a_container_created_and_never_started_has_no_exit_state(
            self, monkeypatch):
        """Not running either, and still nothing to report: the daemon gives
        Go's zero time for `FinishedAt`, which is the only thing that tells
        this apart from a clean exit 0."""
        from orchestrator import docker_ctl
        monkeypatch.setattr(docker_ctl, "_run", _inspect(self.NEVER_STARTED))
        assert docker_ctl.exit_state("x") is None

    def test_it_asks_for_every_field_in_one_inspect(self, monkeypatch):
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
        for field in (".State.Running", ".State.ExitCode",
                      ".State.OOMKilled", ".State.FinishedAt",
                      ".State.Error"):
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
        assert run._save_container_exit_facts(ENDED) is None

    def test_the_core_capture_survives_a_broken_docker(self, tmp_path,
                                                       monkeypatch):
        from orchestrator import ops_pass
        run = _ops_run(tmp_path, "percona")
        _make_bind_dir(tmp_path, run)
        monkeypatch.setattr(ops_pass, "docker_ctl", _Exploding())
        assert run._save_server_core(ENDED) is None

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
        # NOT `copy_glob_as_root`, and that is the gate working rather than a
        # hole: with the daemon broken the exit state is unknown, and the core
        # lookup only runs for a container that has demonstrably ended.
        # Skipping it costs nothing -- the copy would have failed against the
        # same daemon -- and the two tests above are what prove the core
        # capture cannot raise when it does run.
        assert "copy_glob_as_root" not in order, order

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
        assert run._save_container_exit_facts(ENDED) is None
        assert run._save_server_core(ENDED) is None


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

        STRUCTURAL, NOT A GREP OF THE FILE, and the first version was the
        grep. `harness/metrics/records.py` is mostly prose -- the
        measurement-version charter runs to hundreds of comment lines -- so a
        future note in it explaining WHY `exit_code` is deliberately absent
        would have fired the guard that exists to protect that decision. This
        reads the class-level field declarations and the module-level string
        constants instead, which is what "appeared in the record schema"
        actually means.

        ALL FOUR FACTS, not three. The grep checked `oom_killed`, `OOMKilled`
        and `exit_code` and omitted `finished_at`, which is the one of the
        four a reader is most likely to think harmless.
        """
        import ast
        import harness.metrics.records as records_mod
        tree = ast.parse(open(records_mod.__file__).read())

        declared = set()
        constants = set()
        for node in ast.walk(tree):
            # Dataclass fields and any other annotated class attribute.
            if isinstance(node, ast.AnnAssign) and isinstance(node.target,
                                                              ast.Name):
                declared.add(node.target.id)
            # A named constant holding one of these as its VALUE would be the
            # other way the field arrives -- e.g. an `extra` key name.
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                constants.add(node.value)

        print(f"{len(declared)} declared names, {len(constants)} string "
              f"constants scanned")
        for needle in ("exit_code", "oom_killed", "finished_at",
                       "daemon_error"):
            assert needle not in declared, (
                f"{needle!r} is now a declared field of the record schema; "
                f"the exit facts are run-directory artefacts on purpose -- "
                f"see this class's docstring")
            assert needle not in constants, (
                f"{needle!r} appears as a string literal in records.py, which "
                f"is how an `extra` key name arrives; the exit facts are "
                f"run-directory artefacts on purpose")


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
        run._save_server_core(ENDED)
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

        Looking for a core costs a container start. Six engines were never
        asked for one, and `_save_server_error_log` sets the precedent for
        gating on the declaration: "this is a no-op for them, so their
        behaviour is unchanged". The second gate -- a container that has
        actually ended -- is `test_a_live_server_is_not_searched_for_a_core`.
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
            run._save_server_core(ENDED)
            results[engine] = len(calls)
        print(f"HONEST (percona, core ulimit declared) = {results['percona']}")
        print(f"MUTANT (valkey, none declared)         = {results['valkey']}")
        assert results["percona"] != results["valkey"], (
            "DEAD PROBE: declaring a core ulimit made no difference to "
            "whether a container was spawned to look for a core")
        assert results["percona"] == 1
        assert results["valkey"] == 0

    def test_a_live_server_is_not_searched_for_a_core(self, tmp_path,
                                                      monkeypatch):
        """HONEST arm: the container has ended. MUTANT arm: it has not.

        `_save_server_log` runs from `run_harness`'s UNCONDITIONAL `finally`,
        so without this gate every SUCCESSFUL phase would spawn a container to
        search a datadir the server is still writing to, and print a note
        about a core that was never going to exist. `--profile full` is 24
        percona ops units: 24 needless container starts and 24 misleading
        notes on a healthy run. `exit_state` is None while the server is up,
        which is the "nothing crashed here" signal already in hand.
        """
        from orchestrator import ops_pass
        results = {}
        for label, state in (("ended", ENDED), ("still running", None)):
            directory = tmp_path / label.replace(" ", "-")
            run = _ops_run(directory, "percona")
            _make_bind_dir(directory, run)
            calls = []
            monkeypatch.setattr(ops_pass.docker_ctl, "copy_glob_as_root",
                                lambda *a, **k: calls.append(a) or [])
            run._save_server_core(state)
            results[label] = len(calls)
        print(f"HONEST (container ended)      = {results['ended']}")
        print(f"MUTANT (server still running) = {results['still running']}")
        assert results["ended"] != results["still running"], (
            "DEAD PROBE: a live server and a dead one are searched alike, so "
            "every successful phase pays for a container and prints a note "
            "about a core that cannot exist")
        assert results["ended"] == 1
        assert results["still running"] == 0

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
        _ops_run(tmp_path, "percona")._save_server_core(ENDED)
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
    means the kernel PIPES THE DUMP TO THAT PROGRAM AND WRITES NO FILE on any
    filesystem, so every other part of this unit is inert. Ubuntu ships
    exactly that.

    THE MECHANISM, because an earlier version of this docstring had it wrong
    and the wrong version suggests a fix that does not work. It said the
    helper "does not exist in the container's mount namespace".
    `/usr/share/apport/apport` is present on this host, and `core(5)` says the
    program pathname "is interpreted with respect to the initial mount
    namespace as it is always executed there. It is not affected by the
    settings (e.g., root directory, mount namespace, current working
    directory) of the crashing process" -- which is why `%P` and `%I` exist.
    The helper runs. What leaves no core is the PIPE: there is no file to
    collect, so bind-mounting apport into the image or installing it there
    would change nothing.

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


# ---------------------------------------------------------------------------
# The seam between this unit and the one that reads the core (#67 / A6)
# ---------------------------------------------------------------------------

LAB_SCRIPT = os.path.join(REPO, "scripts", "gdb-backtrace.py")


def _script_constant(name):
    """A module-level string or tuple constant out of the lab script's AST.

    Read rather than imported: the script's `main()` expects to be inside the
    bench container, and importing it here would couple this file to whatever
    it does at import time.
    """
    import ast
    tree = ast.parse(open(LAB_SCRIPT).read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{LAB_SCRIPT} has no module-level {name}")


class TestTheTwoHalvesOfTheCoreRouteAgree:
    """This unit writes the core; #67's `scripts/gdb-backtrace.py` reads it.
    They were written by different units against the same seam, so every part
    of it is asserted against THAT FILE'S OWN CONSTANTS rather than against a
    comment here.

    The seam has three parts -- the container path, the keyword name, and the
    filename -- and the third does not fully agree. That is recorded here
    rather than smoothed over, because the disagreement decides whether a core
    is ever read.
    """

    def test_the_container_path_is_the_same_on_both_sides(self, tmp_path,
                                                          monkeypatch):
        """`run_script` mounts it; the script looks there. One string, two
        files, and nothing but this test holding them together."""
        from orchestrator import ops_pass
        theirs = _script_constant("ARTIFACTS")
        artifacts = tmp_path / "results" / "run-1"
        artifacts.mkdir(parents=True)

        captured = {}
        monkeypatch.setattr(ops_pass.docker_ctl, "image_exists",
                            lambda *a, **k: True)
        monkeypatch.setattr(ops_pass.docker_ctl, "run_foreground",
                            lambda spec, **k: captured.setdefault("spec", spec)
                            and 0 or 0)
        monkeypatch.setattr(ops_pass.OpsRun, "_save_server_log",
                            lambda self: None)
        run = _ops_run(tmp_path, "percona")
        run.run_script("probe.py", [], artifacts_dir=str(artifacts))
        mounts = [v for v in captured["spec"].volumes if str(artifacts) in v]
        print(f"theirs={theirs!r} ours={mounts}")
        assert mounts == [f"{artifacts}:{theirs}:ro"], (
            f"scripts/gdb-backtrace.py looks in {theirs!r}; this mounts "
            f"{mounts}")

    def test_the_keyword_name_is_the_one_cmd_lab_passes(self):
        """`cmd_lab` passes `artifacts_dir=`; `run_script` has to take it.

        This is the seam #67's own ratchet was watching, and the reason it
        watched by signature rather than by grep: a mismatch here is a
        TypeError raised minutes after a server came up.
        """
        import inspect
        from orchestrator import cli, ops_pass
        assert "artifacts_dir" in inspect.signature(
            ops_pass.OpsRun.run_script).parameters
        source = open(os.path.join(REPO, "orchestrator", "cli.py")).read()
        assert "artifacts_dir=artifacts_dir" in source, (
            "cmd_lab no longer passes artifacts_dir by that name")

    def test_the_recommended_core_pattern_works_end_to_end(self):
        """The one value both halves handle, asserted on BOTH halves.

        This is the assertion that matters in this class. The capture collects
        `core*`; the script searches `core`, `core.*`, `*.core`, `core-*`. The
        sets are different, so a `core_pattern` can satisfy one and not the
        other -- and the console tells the operator what to set, so what it
        recommends has to be in the intersection.
        """
        import fnmatch
        from orchestrator.ops_pass import (CORE_FILE_GLOB,
                                           RECOMMENDED_CORE_PATTERN)
        theirs = _script_constant("CORE_GLOBS")
        produced = RECOMMENDED_CORE_PATTERN.replace("%p", "12345")
        print(f"recommend {RECOMMENDED_CORE_PATTERN!r} -> {produced!r}; "
              f"ours={CORE_FILE_GLOB!r} theirs={theirs}")
        assert fnmatch.fnmatch(produced, CORE_FILE_GLOB), (
            f"the capture would not collect {produced!r}, which is what the "
            f"console tells the operator to produce")
        assert any(fnmatch.fnmatch(produced, pattern) for pattern in theirs), (
            f"scripts/gdb-backtrace.py would not find {produced!r}, which is "
            f"what the console tells the operator to produce; its globs are "
            f"{theirs}")

    def test_every_message_that_recommends_a_pattern_recommends_that_one(self):
        """Placement is not enough here -- a second message naming a different
        value is how an operator ends up on one of the two edges below. So no
        message may hard-code a pattern of its own."""
        import re
        from orchestrator.ops_pass import RECOMMENDED_CORE_PATTERN
        source = open(os.path.join(REPO, "orchestrator",
                                   "ops_pass.py")).read()
        reporter = source.split("def _report_core_pattern", 1)[1].split(
            "\n    def ", 1)[0]
        # The CODE only. The docstring has to be able to name `core` while
        # explaining which relative patterns are collected; what must not
        # hard-code a value is a message an operator reads.
        body = reporter.split('"""', 2)[2]
        literals = set(re.findall(r"`(core[.%\w-]*)`", body))
        print(f"pattern literals in the reporter: {sorted(literals)}")
        assert not literals, (
            f"these are hard-coded in the console messages instead of coming "
            f"from RECOMMENDED_CORE_PATTERN ({RECOMMENDED_CORE_PATTERN!r}): "
            f"{sorted(literals)}")
        assert "RECOMMENDED_CORE_PATTERN" in reporter

    def test_the_two_filename_sets_do_not_fully_agree_and_the_edges_are_known(
            self):
        """THE DISAGREEMENT, asserted so it cannot be forgotten or drift.

        `core*` and the script's four globs overlap but neither contains the
        other, and both edges are reachable from an ordinary `core_pattern`:

            core%p   -> `core12345`    collected here, the script cannot see it
            %e.core  -> `mysqld.core`  the script looks for it, this never
                                       collects it

        The second is mitigated: `_report_core_pattern` refuses to promise a
        copy-out for a relative pattern that does not start with `core`, so
        the operator is told at server start. The first is in
        `scripts/gdb-backtrace.py`'s globs and is not in this unit's file set
        -- widening them to `core*` would close it.

        This test FAILS IF EITHER EDGE CLOSES, which is the point: whoever
        closes one should delete the half of this that stops being true rather
        than leaving a note claiming a gap that is gone.
        """
        import fnmatch
        from orchestrator.ops_pass import CORE_FILE_GLOB
        theirs = _script_constant("CORE_GLOBS")

        def collected(name):
            return fnmatch.fnmatch(name, CORE_FILE_GLOB)

        def readable(name):
            return any(fnmatch.fnmatch(name, p) for p in theirs)

        assert collected("core12345") and not readable("core12345"), (
            "the `core%p` edge has changed; update this test and the comment "
            "in ops_pass beside CORE_FILE_GLOB")
        assert readable("mysqld.core") and not collected("mysqld.core"), (
            "the `%e.core` edge has changed; update this test and the comment "
            "in ops_pass beside CORE_FILE_GLOB")
        # And the agreed middle, which is what the recommendation uses.
        for name in ("core", "core.12345", "core-12345"):
            assert collected(name) and readable(name), name

    def test_a_core_in_this_units_layout_is_reachable_by_their_search(
            self, tmp_path):
        """The layout is one directory deeper than a flat `/artifacts/core*`.

        `_save_server_core` writes
        `<run_dir>/cores/<engine>-<dataset>-<phase>-<pass>/<name>`, and the
        operator passes `--artifacts results/<run-id>`, so the core is two
        levels below the mount point. The script's `find_cores` uses
        `os.walk`, which is unbounded in depth -- exercised here against a
        real directory tree rather than read off the source.
        """
        import fnmatch
        from orchestrator.ops_pass import (CORE_FILE_GLOB,
                                           RECOMMENDED_CORE_PATTERN)
        theirs = _script_constant("CORE_GLOBS")
        name = RECOMMENDED_CORE_PATTERN.replace("%p", "12345")
        deep = (tmp_path / "cores"
                / "percona-gist-960-euclidean-m8-incremental-normalized")
        deep.mkdir(parents=True)
        (deep / name).write_bytes(b"not really a core")
        assert fnmatch.fnmatch(name, CORE_FILE_GLOB)

        found = []
        for directory, _subdirs, names in os.walk(str(tmp_path)):
            for candidate in names:
                if any(fnmatch.fnmatch(candidate, p) for p in theirs):
                    found.append(os.path.join(directory, candidate))
        print(f"found by their search shape: {found}")
        assert found == [str(deep / name)], (
            "a core written in this unit's layout is not reachable by the "
            "recursive search the backtrace script performs")


class TestLabArtifactsReachesTheMountNowThatTheRefusalIsGone:
    """#67 refused `--artifacts` on a tree whose `run_script` lacked the
    parameter, because the alternative was a TypeError raised minutes after a
    server came up. That refusal is deleted in this change, which means the
    path it was standing in front of is now live and nothing had ever
    exercised it.

    So this drives `cmd_lab` itself -- the real parser, the real validation,
    the real path arithmetic -- with only the Docker layer replaced. It is
    the whole of `lab --artifacts` except the container.

    WHAT IT DOES NOT PROVE, and the first real invocation is what will: that
    the bench image has gdb in it, that the mount appears inside the
    container, and that a core is there to read. Those need an image, a
    server and a crash. What this proves is that the argument survives the
    journey from argv to the volume list, which is exactly what the deleted
    refusal existed to guarantee and is no longer there to.
    """

    def _run_lab(self, tmp_path, monkeypatch, argv):
        from orchestrator import cli, ops_pass

        root = tmp_path / "repo"
        (root / "scripts").mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(cli, "VB_ROOT", str(root))

        captured = {}

        class FakeRun:
            server_name, port = "srv", 3306

            def __init__(self, *a, **k):
                captured["paths"] = a[4] if len(a) > 4 else k.get("paths")

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def run_script(self, script, args, **kwargs):
                captured["script"] = script
                captured["kwargs"] = kwargs
                return 0

        monkeypatch.setattr(cli.ops_pass, "OpsRun", FakeRun)
        monkeypatch.setattr(cli.docker_ctl, "docker_available",
                            lambda: True)
        monkeypatch.setattr(cli, "measurement_in_progress", lambda: [])
        args = cli.build_parser().parse_args(argv)
        rc = args.func(args)
        return rc, captured

    def test_the_path_reaches_run_script_as_an_absolute_path(self, tmp_path,
                                                             monkeypatch):
        """HONEST arm: `--artifacts` given. MUTANT arm: omitted.

        Absolute because the orchestrator hands host paths to the Docker
        daemon; a relative one would mount nothing, which is the failure
        CLAUDE.md records for the web UI's bind.
        """
        root = tmp_path / "repo"
        cores = root / "results" / "run-1" / "cores"
        cores.mkdir(parents=True)

        rel = os.path.relpath(str(cores), os.getcwd())
        rc, honest = self._run_lab(
            tmp_path, monkeypatch,
            ["lab", "--engine", "percona", "--shell", "--artifacts", rel])
        assert rc == 0, rc
        rc, mutant = self._run_lab(
            tmp_path, monkeypatch, ["lab", "--engine", "percona", "--shell"])
        assert rc == 0, rc

        print(f"HONEST (--artifacts given) = {honest['kwargs']}")
        print(f"MUTANT (omitted)           = {mutant['kwargs']}")
        assert honest["kwargs"] != mutant["kwargs"], (
            "DEAD PROBE: passing --artifacts made no difference to what "
            "run_script was given")
        assert honest["kwargs"]["artifacts_dir"] == str(cores.resolve()), (
            honest["kwargs"])
        assert mutant["kwargs"]["artifacts_dir"] is None, mutant["kwargs"]

    def test_a_directory_outside_the_repo_is_still_refused(self, tmp_path,
                                                           monkeypatch):
        """The refusal that SURVIVES. `_artifacts_mount_problem` is gone;
        `_lab_artifacts_problem` is not, and it is the one that matters now --
        lab mounts this path into a container.
        """
        outside = tmp_path / "elsewhere"
        outside.mkdir()
        rc, captured = self._run_lab(
            tmp_path, monkeypatch,
            ["lab", "--engine", "percona", "--shell", "--artifacts",
             str(outside)])
        assert rc == 2, rc
        assert "kwargs" not in captured, (
            "the server was started despite the refusal")

    def test_a_missing_directory_is_refused_before_anything_starts(
            self, tmp_path, monkeypatch):
        rc, captured = self._run_lab(
            tmp_path, monkeypatch,
            ["lab", "--engine", "percona", "--shell", "--artifacts",
             str(tmp_path / "repo" / "nope")])
        assert rc == 2, rc
        assert "kwargs" not in captured
