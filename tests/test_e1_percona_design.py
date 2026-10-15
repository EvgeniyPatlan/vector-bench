"""Guards for the Percona Server addition designed in docs/09-adding-percona-server.md.

This file holds the checkable half of that document. It exists because three fix
rounds in this series were spent on prose alone — "code right, prose wrong" — and
a claim in a document is part of the deliverable.

WHAT THESE TESTS CAN AND CANNOT ESTABLISH
-----------------------------------------
They cannot establish anything about Percona Server. No server is reachable from
a test run, and a test asserting "Percona requires a BIGINT UNSIGNED primary key"
would only be asserting that a document contains a sentence. Every claim about
the engine is sourced to a file:line on the pinned SHA in docs/09 and is settled
by reading that tree, not here.

What they do establish is that the artefacts agree with each other and with the
document: that the DDL the overlay module emits is the DDL the document says it
emits, that docker/percona/ has the shape the other engines' directories have,
that init.sql does not carry the one line that would stop the container booting,
and that the arithmetic in docs/09 §3 reproduces from the formula it cites.

The module is loaded and CALLED rather than grepped. `tests/test_config.py`'s
existing overlay tests read module source as text because importing an overlay
module needs an upstream ann-benchmarks package that is not in this repo;
`_load_overlay()` below supplies the one missing piece so these tests can assert
on behaviour instead of on substrings. That matters here specifically: the
difference between MariaDB's DDL and Percona's is a handful of tokens, and a
substring test for `"BIGINT UNSIGNED"` would pass against a file that also still
said `id INT PRIMARY KEY` somewhere.

Each negative assertion below is paired with a positive one that proves the
needle is findable, because a negative assertion passes just as happily against
an empty string as against a correct file.
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
import types

import pytest

VB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCKER = os.path.join(VB_ROOT, "docker")
OVERLAY = os.path.join(VB_ROOT, "overlay", "ann-benchmarks")
ALGORITHMS = os.path.join(OVERLAY, "ann_benchmarks", "algorithms")
PERCONA_DOC = os.path.join(VB_ROOT, "docs", "09-adding-percona-server.md")

# The SHA docs/09 pins. Repeated here so that a test can check the document, the
# module docstring and this file all name the same commit -- the value is the
# whole point of pinning, and three copies that can drift is how a pin stops
# meaning anything.
PINNED_SHA = "02a8b80c70b0ccf6c9035d8a2307e7fa988cc304"


# ---------------------------------------------------------------------------
# Loading the overlay module
# ---------------------------------------------------------------------------

def _load_overlay():
    """Import `vb_mysql` and the percona module with the upstream piece stubbed.

    `ann_benchmarks/vb_mysql.py` does `from .algorithms.base.module import
    BaseANN`, and `algorithms/base/` belongs to upstream ann-benchmarks, which
    this repo clones into `work/` at run time rather than vendoring. So the
    overlay cannot be imported as-is from a test.

    The stub is one empty class. Nothing in `Dialect.create_table`,
    `_query_sql` or `_verify_plan` touches `BaseANN`, so stubbing it does not
    stand in for any behaviour under test -- which is the trap this repo calls
    "fixtures supplying the thing under test". If a future test here needed
    `BaseANN`'s own behaviour it would have to clone upstream instead.

    Returns (vb_mysql_module, percona_module).
    """
    cached = getattr(_load_overlay, "_cache", None)
    if cached is not None:
        return cached

    def _pkg(name: str, path: str | None = None) -> types.ModuleType:
        mod = types.ModuleType(name)
        mod.__path__ = [path] if path else []
        sys.modules[name] = mod
        return mod

    ab = _pkg("ann_benchmarks", os.path.join(OVERLAY, "ann_benchmarks"))
    algs = _pkg("ann_benchmarks.algorithms", ALGORITHMS)
    ab.algorithms = algs
    base = _pkg("ann_benchmarks.algorithms.base")
    algs.base = base

    base_module = types.ModuleType("ann_benchmarks.algorithms.base.module")

    class BaseANN:  # the whole stub
        pass

    base_module.BaseANN = BaseANN
    sys.modules["ann_benchmarks.algorithms.base.module"] = base_module
    base.module = base_module

    def _load(dotted: str, path: str) -> types.ModuleType:
        spec = importlib.util.spec_from_file_location(dotted, path)
        assert spec and spec.loader, f"cannot load {path}"
        mod = importlib.util.module_from_spec(spec)
        sys.modules[dotted] = mod
        spec.loader.exec_module(mod)
        return mod

    # Importing from `overlay/` would otherwise write `__pycache__/` into the
    # overlay tree, and `overlay/ann-benchmarks/` is copied wholesale over a
    # disposable clone on every `run` and `render`. A .pyc compiled by the
    # host's interpreter has no business travelling into the bench image
    # beside the source it was built from, so bytecode is off for these loads.
    was_writing = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        vb = _load("ann_benchmarks.vb_mysql",
                   os.path.join(OVERLAY, "ann_benchmarks", "vb_mysql.py"))
        ab.vb_mysql = vb
        percona_pkg = _pkg("ann_benchmarks.algorithms.percona",
                           os.path.join(ALGORITHMS, "percona"))
        algs.percona = percona_pkg
        percona = _load("ann_benchmarks.algorithms.percona.module",
                        os.path.join(ALGORITHMS, "percona", "module.py"))
    finally:
        sys.dont_write_bytecode = was_writing

    _load_overlay._cache = (vb, percona)
    return vb, percona


@pytest.fixture(scope="module")
def overlay():
    return _load_overlay()


def test_the_loader_does_not_write_bytecode_into_the_overlay():
    """`overlay/ann-benchmarks/` is copied wholesale over a disposable clone on
    every `run` and `render`, so a host-compiled `.pyc` sitting beside the
    source would travel into the bench image. Importing from there is this
    file's doing, so suppressing the write is this file's responsibility.

    Verified by running the full suite from a clean tree and confirming no
    `.pyc` appeared anywhere under `overlay/`. This test guards the mechanism
    rather than the directory state, because a directory assertion would fail
    on a stale file left by something other than the suite and would then look
    like a broken checkout.
    """
    import inspect
    source = inspect.getsource(_load_overlay)
    assert "sys.dont_write_bytecode = True" in source
    # Restored afterwards rather than left on: leaving a global interpreter
    # flag flipped would change the behaviour of every test that runs after
    # this one.
    assert "sys.dont_write_bytecode = was_writing" in source
    assert "finally:" in source


def _strip_sql_comments(text: str) -> str:
    """Drop `--` line comments, keeping only statements.

    docs/09 and docker/percona/init.sql both discuss `mysql_native_password` at
    length in comments, so a test asking whether the file "contains" the string
    would fail on a correct file. What matters is whether the server is ever
    asked for the plugin, which means the statements.
    """
    out = []
    for line in text.splitlines():
        head = line.split("--", 1)[0]
        if head.strip():
            out.append(head)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# docker/percona/ shape
# ---------------------------------------------------------------------------

class TestDockerDirectoryShape:
    """docker/percona/ must look like the other MySQL-family engines' directories.

    The expected shape is DERIVED from docker/mariadb/ and docker/alisql/ rather
    than spelled out, so that a change to the pattern fails here instead of
    leaving the newest engine as the only one following the old one.
    """

    @staticmethod
    def _kinds(engine: str) -> set:
        """The file roles present, with the engine name factored out."""
        out = set()
        for name in os.listdir(os.path.join(DOCKER, engine)):
            out.add(name.replace(engine, "<engine>"))
        return out

    def test_the_directory_exists(self):
        assert os.path.isdir(os.path.join(DOCKER, "percona"))

    def test_same_file_roles_as_the_two_engines_it_is_modelled_on(self):
        mariadb = self._kinds("mariadb")
        alisql = self._kinds("alisql")
        # Non-vacuity: the two templates must agree with each other, or
        # "matches the pattern" would not be a statement about anything.
        assert mariadb == alisql, (
            "the two MySQL-family docker dirs no longer share a shape; "
            f"mariadb={sorted(mariadb)} alisql={sorted(alisql)}")
        assert mariadb == {"Dockerfile", "entrypoint-<engine>.sh", "init.sql"}
        assert self._kinds("percona") == mariadb

    def test_dockerfile_declares_the_same_build_stages(self):
        """A missing `bench` stage fails only when the ann pass is launched."""
        def stages(engine):
            src = open(os.path.join(DOCKER, engine, "Dockerfile")).read()
            return set(re.findall(r"^FROM\s+\S+\s+AS\s+(\w+)", src,
                                  re.MULTILINE))
        assert stages("alisql") == {"builder", "runtime", "bench"}
        assert stages("percona") == stages("alisql")

    def test_the_dockerfile_copies_the_entrypoint_that_exists(self):
        """A COPY of a filename nobody wrote is a build failure, late."""
        src = open(os.path.join(DOCKER, "percona", "Dockerfile")).read()
        copied = re.findall(r"^COPY\s+(entrypoint-\S+)", src, re.MULTILINE)
        assert copied == ["entrypoint-percona.sh"]
        assert os.path.isfile(os.path.join(DOCKER, "percona", copied[0]))

    def test_the_entrypoint_offers_the_same_subcommands(self):
        """`vb_mysql._start_server` runs `vb-entrypoint server`; the ops path and
        docs/03 use `init` and `client`."""
        def subcommands(engine):
            path = os.path.join(DOCKER, engine, f"entrypoint-{engine}.sh")
            src = open(path).read()
            body = src.split("case ", 1)[-1]
            return set(re.findall(r"^\s{2}(\w+)\)", body, re.MULTILINE))
        assert {"server", "init", "client"} <= subcommands("alisql")
        assert subcommands("percona") == subcommands("alisql")

    def test_the_socket_path_agrees_between_dockerfile_and_entrypoint(self):
        """Two files name this path; a readiness probe waits on one of them.

        If they disagree the server comes up on one socket and everything that
        waits for it watches the other, which presents as a start-up timeout
        rather than as a typo.
        """
        dockerfile = open(os.path.join(DOCKER, "percona", "Dockerfile")).read()
        entrypoint = open(os.path.join(DOCKER, "percona",
                                       "entrypoint-percona.sh")).read()
        from_env = re.search(r"VB_SOCKET=(\S+)", dockerfile)
        from_default = re.search(r'SOCKET="\$\{VB_SOCKET:-([^}"]+)\}"',
                                 entrypoint)
        assert from_env and from_default
        assert from_env.group(1) == from_default.group(1)
        # And it must be engine-specific, or two servers sharing a data mount
        # would collide. Derived from the other engines' convention.
        assert from_env.group(1).endswith("/percona.sock")


class TestBuildArgConventions:
    """The Dockerfile has to receive the pin through the ARG the build script
    will pass, and the pin has to be under the key the orchestrator reads.

    Both of these were wrong in the first draft of docs/09, and neither would
    have failed visibly: a mismatched ARG leaves `VB_ENGINE_TAG=unknown` in the
    image, and the wrong config key makes `_source_tag()` return the empty
    string and `prepare-sources.sh` silently fall back to a default tag. The
    engine would then be built from whatever that default names, and the
    manifest would not record which commit was measured.
    """

    def test_the_dockerfile_arg_follows_the_build_script_convention(self):
        """build-images.sh:129-131 passes `<ENGINE>_TAG=${tag}`."""
        script = open(os.path.join(VB_ROOT, "scripts",
                                   "build-images.sh")).read()
        passed = dict(re.findall(r"^\s+(\w+)\)\s+bargs\+=\(--build-arg "
                                 r"\"(\w+_TAG)=", script, re.MULTILINE))
        # Non-vacuity: the convention must be visible for the existing engines.
        assert passed.get("mariadb") == "MARIADB_TAG"
        assert passed.get("alisql") == "ALISQL_TAG"
        dockerfile = open(os.path.join(DOCKER, "percona", "Dockerfile")).read()
        args = set(re.findall(r"^ARG (\w+_TAG)", dockerfile, re.MULTILINE))
        assert args == {"PERCONA_TAG"}, args

    def test_the_arg_is_wired_to_the_env_var_the_manifest_reads(self):
        """ops_pass.py:231 puts source.tag into VB_ENGINE_TAG; the image sets it
        too, and an ARG declared but never used leaves it `unknown`."""
        dockerfile = open(os.path.join(DOCKER, "percona", "Dockerfile")).read()
        assert "VB_ENGINE_TAG=${PERCONA_TAG}" in dockerfile

    def test_the_orchestrator_has_no_source_sha_key(self):
        """docs/09 §4.3 tells the follow-on task to write `source: tag:` with a
        SHA as the value. That instruction is only correct while this holds."""
        source = open(os.path.join(VB_ROOT, "orchestrator",
                                   "engines.py")).read()
        fn = source.split("def _source_tag", 1)[1].split("\ndef ", 1)[0]
        assert 'source.get("tag")' in fn          # needle findable
        assert 'source.get("sha")' not in fn

    def test_the_document_says_tag_and_not_sha(self):
        doc = open(PERCONA_DOC).read()
        skeleton = doc.split("```yaml", 1)[1].split("```", 1)[0]
        assert f"tag: {PINNED_SHA}" in skeleton
        assert f"sha: {PINNED_SHA}" not in skeleton

    def test_build_images_refuses_an_engine_it_has_no_case_for(self):
        """Which is why the missing case is a clear failure and not a silent
        one -- worth knowing before someone goes looking for a subtler bug."""
        script = open(os.path.join(VB_ROOT, "scripts",
                                   "build-images.sh")).read()
        assert 'die "engine' in script and "has no known build-arg" in script


class TestEverySubmoduleIsMitigated:
    """The build blocker review found: four submodules, four flags needed.

    `prepare-sources.sh` ships a `git archive`, which omits submodule paths, so
    cmake must be told not to descend into any of them. The first revision of
    the Dockerfile enumerated all four correctly, said "two", and passed three
    flags -- `extra/libkmip` was left out and cmake configure would have failed.

    These guards are self-consistency checks on the Dockerfile's own note: the
    note maps each submodule to the flag that handles it, and the flag has to
    appear in the cmake invocation. A count stated in prose and a list that
    disagrees with it is exactly what happened.
    """

    @pytest.fixture(scope="class")
    def dockerfile(self):
        return open(os.path.join(DOCKER, "percona", "Dockerfile")).read()

    @pytest.fixture(scope="class")
    def mapping(self, dockerfile):
        """submodule path -> cmake flag, parsed out of note 2's table."""
        note = dockerfile.split("NO SUBMODULE FETCH", 1)[1].split(
            "BOOST NEEDS NO FLAG", 1)[0]
        pairs = re.findall(r"^#\s+(\S+/\S+)\s+->\s+(-D\S+)", note,
                           re.MULTILINE)
        return dict(pairs)

    def test_all_four_submodules_are_named_with_a_flag_each(self, mapping):
        assert set(mapping) == {
            "storage/rocksdb/rocksdb",
            "extra/coredumper",
            "extra/jwt-cpp",
            "extra/libkmip",
        }, mapping

    def test_every_mapped_flag_is_actually_passed_to_cmake(self, dockerfile,
                                                           mapping):
        """The half that was wrong: the note listed four, cmake got three."""
        invocation = dockerfile.split("cmake /src/source", 1)[1].split(
            "make -j", 1)[0]
        for submodule, flag in sorted(mapping.items()):
            assert flag in invocation, f"{submodule} -> {flag} not passed"

    def test_libkmip_is_closed_by_either_accepted_spelling(self, mapping):
        """`extra/libkmip` must be gated, by one of the two spellings that work.

        BOTH work, and an earlier version of this test asserted otherwise. It
        required `startswith("-DWITHOUT_COMPONENT_KEYRING_KMIP")` and its
        docstring said `-DWITH_...=OFF` "would not close it" -- so a later
        engineer switching to the WITH_ spelling would have got a red test
        telling them something untrue, which is worse than no test at all.

        Measured under cmake 3.28.3, with CMakeLists.txt:2295-2298 transcribed
        verbatim into a standalone project:

            (no flags)                          -> WOULD_ADD_SUBDIRECTORY
            -DWITH_COMPONENT_KEYRING_KMIP=OFF   -> SKIPPED
            -DWITH_COMPONENT_KEYRING_KMIP=0     -> SKIPPED
            -DWITH_COMPONENT_KEYRING_KMIP=NO    -> SKIPPED
            -DWITH_COMPONENT_KEYRING_KMIP=ON    -> WOULD_ADD_SUBDIRECTORY
            -DWITHOUT_COMPONENT_KEYRING_KMIP=1  -> SKIPPED
            -DWITHOUT_COMPONENT_KEYRING_KMIP=0  -> SKIPPED

        The old reasoning conflated `DEFINED WITH_X` with `WITH_X`: defining it
        is exactly what makes the AND false, and the OR's second operand is
        `WITH_X` as a boolean, where OFF/0/NO are CMake false constants. The
        component's own guard keys off the WITH_ spelling too --
        components/keyrings/keyring_kmip/CMakeLists.txt:28 is
        `IF(NOT WITH_COMPONENT_KEYRING_KMIP) RETURN()`.

        What this asserts is the part that is load-bearing: the submodule is
        gated by a recognised spelling, and specifically not by the one value
        that re-opens it. The Dockerfile may prefer WITHOUT_ -- it does, because
        that gate tests definedness and never the value, so it cannot be broken
        by a value spelling -- but that is a preference, and a preference does
        not belong in an assertion.
        """
        flag = mapping["extra/libkmip"]
        symbol, _, value = flag.partition("=")
        assert symbol in ("-DWITH_COMPONENT_KEYRING_KMIP",
                          "-DWITHOUT_COMPONENT_KEYRING_KMIP"), flag
        if symbol == "-DWITH_COMPONENT_KEYRING_KMIP":
            assert value.upper() in ("OFF", "0", "NO", "FALSE", "N"), flag

    def test_the_note_does_not_claim_the_WITH_spelling_fails(self, dockerfile):
        """Polarity, on the prose this time.

        The falsehood lived in two places and the test enforced it. Deleting it
        from the assertion is not enough -- the note has to stop saying it, or
        the next reader believes the note and the test's silence looks like
        agreement.
        """
        note = dockerfile.split("NO SUBMODULE FETCH", 1)[1].split(
            "BOOST NEEDS NO FLAG", 1)[0]
        for wrong in ("would NOT help",
                      "so only the WITHOUT_ spelling closes it"):
            assert wrong not in note, wrong
        # ...and it carries the measurement instead of a replacement argument.
        assert "cmake 3.28.3" in note
        assert "-DWITH_COMPONENT_KEYRING_KMIP=OFF   -> SKIPPED" in note

    def test_the_note_does_not_undercount_its_own_list(self, dockerfile,
                                                       mapping):
        """The actual defect: prose said "two", the list had four.

        Anchored on ONE deterministic phrase in note 2's heading, "TAKES <n>
        FLAGS", and not on number words anywhere in the note. The first version
        of this test scanned the whole note for number words and compared the
        mapping length against the SET of them -- and the note's prose contains
        two, three, four and one, so `len(mapping) in {1,2,3,4}` held for any
        plausible mapping. Measured: with libkmip deleted from the note the
        mapping was 3, the word set was {1,2,3,4}, and the test passed.

        That is membership-not-value, which is on this repo's list of measured
        vacuity shapes. A count assertion has to compare against a single
        stated number.
        """
        heading = dockerfile.split("NO SUBMODULE FETCH", 1)[1].split("\n", 1)[0]
        words = {"ONE": 1, "TWO": 2, "THREE": 3, "FOUR": 4, "FIVE": 5}
        stated = re.search(r"TAKES (" + "|".join(words) + r") FLAGS", heading)
        assert stated, (
            f"note 2's heading states no flag count: {heading!r}")
        assert words[stated.group(1)] == len(mapping), (
            f"heading says {stated.group(1)} flags, mapping has {len(mapping)}")

    def test_no_submodules_key_is_proposed_for_the_engine_config(self):
        """The alternative path -- listing them in percona.yml's
        `source.submodules:` the way mariadb.yml lists libmariadb -- would make
        the build fetch from three more hosts. The document rules it out, so
        its skeleton must not quietly contain one.

        Comments are stripped first, because the skeleton *explains* the
        absence in a comment and the first version of this test matched its own
        explanation. That is the inverse of the usual vacuity: a negative
        assertion failing on prose that agrees with it.
        """
        doc = open(PERCONA_DOC).read()
        skeleton = doc.split("```yaml", 1)[1].split("```", 1)[0]
        keys = "\n".join(ln.split("#", 1)[0] for ln in skeleton.splitlines())
        assert "submodules:" not in keys
        # ...and the explanation is still there, so this is not passing because
        # the skeleton went silent on the question.
        assert "submodules:" in skeleton
        # Non-vacuity: mariadb.yml really does use that key, so the spelling
        # this is looking for is the right one, and the stripper does not eat
        # a real key.
        mariadb = open(os.path.join(VB_ROOT, "config", "engines",
                                    "mariadb.yml")).read()
        mariadb_keys = "\n".join(ln.split("#", 1)[0]
                                 for ln in mariadb.splitlines())
        assert "submodules:" in mariadb_keys


class TestGraphMemoryFallback:
    """The by-hand `docker run` must not get the 1 GiB compiled-in default, and
    the orchestrator's value must always win.

    docker/alisql/init.sql:13-15 sets the precedent -- "the server flag covers
    startup, this covers a server someone restarted without it" -- but the
    mechanism cannot be copied: --init-file runs AFTER the server flags, so a
    `SET GLOBAL` there would overwrite a measurement parameter and put a value
    in the manifest that was not in force.
    """

    @pytest.fixture(scope="class")
    def entrypoint(self):
        return open(os.path.join(DOCKER, "percona",
                                 "entrypoint-percona.sh")).read()

    def test_the_fallback_exists(self, entrypoint):
        assert "--innodb-hnsw-max-memory=0" in entrypoint

    def test_it_is_conditional_and_not_unconditional(self, entrypoint):
        """An unconditional append would override the resource pass."""
        start = entrypoint.split("start_server()", 1)[1]
        appended = start.split("--innodb-hnsw-max-memory=0", 1)[0]
        # The append sits inside a guard on whether one was already given.
        assert "have_max_memory" in appended
        assert re.search(r"if\s*\(\(\s*!\s*have_max_memory\s*\)\)", appended)

    def test_the_guard_matches_both_mysqld_spellings(self, entrypoint):
        """`--flag=value` and `--flag value` are both valid, and underscores and
        hyphens are interchangeable in mysqld option names. A guard matching
        only one spelling would silently double-specify the option."""
        start = entrypoint.split("start_server()", 1)[1]
        assert "${a//_/-}" in start
        assert "--innodb-hnsw-max-memory|--innodb-hnsw-max-memory=*" in start

    def test_init_sql_does_not_set_it(self):
        """The thing that must NOT be done, paired with the AliSQL precedent it
        would have been copied from."""
        alisql = _strip_sql_comments(
            open(os.path.join(DOCKER, "alisql", "init.sql")).read())
        percona = _strip_sql_comments(
            open(os.path.join(DOCKER, "percona", "init.sql")).read())
        # Non-vacuity: AliSQL really does repeat its global in the init file.
        assert "SET GLOBAL" in alisql
        assert "SET GLOBAL" not in percona
        assert "innodb_hnsw_max_memory" not in percona

    def test_init_sql_explains_the_asymmetry_rather_than_omitting_it(self):
        """A silent omission reads as an oversight; the next person adds it."""
        src = open(os.path.join(DOCKER, "percona", "init.sql")).read()
        assert "init.sql:13-15" in src
        assert "entrypoint-percona.sh" in src


class TestExplainFormatIsPinned:
    """`vector_ann` lives in the `type` column of TRADITIONAL EXPLAIN only.

    `explain_format` defaults to TRADITIONAL, so a bare EXPLAIN works today, but
    the variable is reachable through VB_SERVER_ARGS and under TREE or JSON the
    marker vanishes -- a false negative about a server that DID use the index.
    No MTR test could catch it, because MTR output is always TRADITIONAL.
    """

    def test_the_module_pins_the_format(self, overlay):
        _, percona = overlay
        body = open(os.path.join(ALGORITHMS, "percona", "module.py")).read()
        fn = body.split("def _verify_plan", 1)[1].split("\n    def ", 1)[0]
        # The executable line, not the docstring: look at the execute() call.
        call = fn.split("self._cur.execute(", 1)[1]
        assert '"EXPLAIN FORMAT=TRADITIONAL ' in call

    def test_the_shared_implementation_checks_the_name_and_not_the_access_type(
            self, overlay):
        """Non-vacuity: proves the module differs rather than matching.

        **Re-aimed from the EXPLAIN FORMAT at the access type, because the
        format version was a landmine.** It asserted
        `"FORMAT=TRADITIONAL" not in fn` for the SHARED reader, which pinned
        the ann path to *not* being hardened the way its ops sibling was -- so
        a later author hardening it would have read this failure as "you broke
        something". That is the shape of guard this file exists to remove, not
        to add.

        The load-bearing difference is the access type, and it cannot be
        acquired by accident: the shared reader asks only whether
        `dialect.index_name` appears anywhere in the joined plan text, which
        Percona's own recorded output satisfies for a full table scan.

        **Pinning the format in the shared reader would be safe and is not
        forbidden by anything here.** It is simply unnecessary: a name-only
        check is format-insensitive -- the index name appears in TREE and JSON
        output too -- so the keyword buys that reader nothing. It is pinned in
        the ops driver (`harness/drivers/mysql_family.py`) because that one
        tests `Dialect.plan_marker`, a value in the tabular `type` column that
        no other format has, and in `percona/module.py` for the same reason.
        All three servers the shared reader serves accept the keyword, so the
        choice is about need rather than risk.
        """
        shared = open(os.path.join(OVERLAY, "ann_benchmarks",
                                   "vb_mysql.py")).read()
        fn = shared.split("def _verify_plan", 1)[1].split("\n    def ", 1)[0]
        assert '"EXPLAIN ' in fn                      # needle findable
        # The shared reader does NOT look for an access type. This is what
        # makes the Percona override necessary rather than cosmetic.
        assert "vector_ann" not in fn
        assert "PLAN_ACCESS_TYPE" not in fn
        # ...and the Percona module does, which is the other half of the pair.
        percona = open(os.path.join(ALGORITHMS, "percona",
                                    "module.py")).read()
        own = percona.split("def _verify_plan", 1)[1].split("\n    def ", 1)[0]
        assert "PLAN_ACCESS_TYPE in plan" in own


class TestInitSqlDoesNotAskForARemovedPlugin:
    """The one line that must not be copied from docker/alisql/init.sql.

    docs/09 §7 item 2 cites mysql-test/r/mysql_native_password_removed.result:4-5
    on the pinned SHA: `CREATE USER ... IDENTIFIED WITH mysql_native_password`
    returns `ERROR HY000: Plugin 'mysql_native_password' is not loaded`. An
    --init-file statement that errors is fatal to start-up, so copying AliSQL's
    line would make the container fail to boot.
    """

    NEEDLE = "mysql_native_password"

    def _statements(self, engine: str) -> str:
        path = os.path.join(DOCKER, engine, "init.sql")
        return _strip_sql_comments(open(path).read())

    def test_the_needle_is_findable_by_this_test(self):
        """Non-vacuity, twice over.

        The negative assertion below would pass against an empty string, and it
        would also pass if `_strip_sql_comments` were over-eager enough to
        remove everything. AliSQL's init.sql really does request the plugin in a
        statement, so finding it there proves both the needle and the stripper.
        """
        alisql = self._statements("alisql")
        assert self.NEEDLE in alisql
        assert "CREATE USER" in alisql

    def test_percona_init_sql_never_requests_it_in_a_statement(self):
        assert self.NEEDLE not in self._statements("percona")

    def test_percona_init_sql_is_not_vacuously_empty(self):
        """A file whose statements were all deleted would pass the test above."""
        statements = self._statements("percona")
        assert statements.count("CREATE USER") == 2        # '%' and 'localhost'
        assert statements.count("GRANT ALL PRIVILEGES") == 2
        assert "FLUSH PRIVILEGES" in statements
        # The account the rest of the framework connects as.
        assert "'bench'" in statements

    def test_it_does_not_copy_alisqls_vector_enable_statement_either(self):
        """AliSQL's init.sql runs `SET GLOBAL vidx_disabled = OFF` because VIDX
        ships disabled. Percona's index needs no switch, and an unknown variable
        in an --init-file is also fatal."""
        assert "vidx_disabled" in self._statements("alisql")   # needle findable
        assert "vidx_disabled" not in self._statements("percona")


# ---------------------------------------------------------------------------
# The dialect the module actually emits
# ---------------------------------------------------------------------------

class TestEmittedDDL:
    """docs/09 §2.2 and §2.3, checked against the statement the code builds."""

    @pytest.fixture
    def ddl(self, overlay):
        _, percona = overlay
        dialect = percona.PERCONA
        return dialect.create_table(dim=128, m=16, metric="euclidean",
                                    storage_engine="InnoDB")

    def test_primary_key_is_bigint_unsigned(self, ddl, overlay):
        """index_syntax.result:186-204 records four refusals of anything else.

        Paired with the base dialect, which emits the INT key Percona refuses,
        so this asserts a difference rather than the presence of a string.
        """
        vb, percona = overlay
        base = vb.Dialect(name="probe").create_table(
            dim=128, m=16, metric="euclidean", storage_engine="InnoDB")
        assert "id INT PRIMARY KEY" in base
        assert "id BIGINT UNSIGNED PRIMARY KEY" in ddl
        assert "id INT PRIMARY KEY" not in ddl

    def test_the_index_clause_is_percona_spelling(self, ddl):
        assert "VECTOR KEY vi (v) TYPE hnsw (" in ddl
        # MariaDB's spelling must be gone, not merely joined by Percona's.
        assert "DISTANCE=" not in ddl
        assert "M=16" not in ddl          # Percona writes `M = 16`

    def test_all_three_index_options_are_stated(self, ddl):
        """Omitting one silently takes Percona's default -- M=25 rather than the
        profile's value (vec0vec.h:55). dd_params_resolved.result:8-19 shows the
        server filling in and storing whatever was not said."""
        opts = re.search(r"TYPE hnsw \(([^)]*)\)", ddl)
        assert opts, ddl
        names = {kv.split("=")[0].strip() for kv in opts.group(1).split(",")}
        assert names == {"M", "ef_construction", "metric"}

    def test_the_metric_is_a_bare_identifier(self, ddl):
        """`metric = 'euclidean'` is a parse error -- an option value is an
        identifier or an integer (Docs/percona-hnsw.md:1305-1307)."""
        assert "metric = euclidean" in ddl
        assert "metric = 'euclidean'" not in ddl
        assert 'metric = "euclidean"' not in ddl

    def test_m_and_ef_construction_are_unquoted_integers(self, ddl):
        """`M = 6.0` and `M = "6"` are both parse errors
        (index_syntax.test:356, :363)."""
        opts = re.search(r"TYPE hnsw \(([^)]*)\)", ddl).group(1)
        for pair in opts.split(","):
            name, _, value = (p.strip() for p in pair.partition("="))
            if name in ("M", "ef_construction"):
                assert re.fullmatch(r"\d+", value), pair

    def test_ef_construction_defaults_to_perconas_own_default(self, overlay):
        _, percona = overlay
        assert percona.DEFAULT_EF_CONSTRUCTION == 200       # vec0vec.h:56
        assert percona.PERCONA.ef_construction == 200
        ddl = percona.PERCONA.create_table(128, 16, "euclidean", "InnoDB")
        assert "ef_construction = 200" in ddl

    def test_the_vector_column_is_not_null(self, ddl):
        """An indexed vector column is NOT NULL (search.test:36)."""
        assert re.search(r"v VECTOR\(128\) NOT NULL", ddl)


class TestEmittedQuery:
    """docs/09 §2.4, checked against the SELECT the code builds."""

    @pytest.fixture
    def instance(self, overlay):
        """A PerconaServer with only the attributes `_query_sql` reads.

        Built without `__init__`, which starts a server. The attributes set here
        are the ones the real constructor would set; nothing under test is
        supplied by the fixture -- `_query_sql`'s output is computed, not given.
        """
        _, percona = overlay
        obj = object.__new__(percona.PerconaServer)
        obj.dialect = percona.PERCONA
        obj._metric_fn = "euclidean"
        return obj

    def test_it_is_a_three_argument_distance_call(self, instance):
        """sql/item_create.cc:1403 registers DISTANCE with arity 3."""
        sql = instance._query_sql(10)
        call = re.search(r"DISTANCE\(([^)]*)\)", sql)
        assert call, sql
        args = [a.strip() for a in call.group(1).split(",")]
        assert len(args) == 3, args
        assert args[0] == "v"
        assert args[1] == "%s"
        assert args[2] == "'EUCLIDEAN'"

    def test_mariadbs_per_metric_function_is_gone(self, instance, overlay):
        """Non-vacuity: the shared implementation emits it, so its absence here
        is a difference and not an empty assertion."""
        vb, percona = overlay
        shared = vb.VBMySQLBase._query_sql(instance, 10)
        assert "vec_distance_euclidean(v, %s)" in shared
        assert "vec_distance_" not in instance._query_sql(10)

    def test_exactly_one_bound_parameter(self, instance):
        """The caller passes `(vector,)`. A second placeholder would raise at
        execute time, inside a measured loop."""
        assert instance._query_sql(10).count("%s") == 1

    def test_ascending_order_and_a_limit(self, instance):
        """Both are required for the index to be used at all
        (search.test:49-55: no LIMIT and DESC are listed as shapes that must
        not use the index)."""
        sql = instance._query_sql(37)
        assert "ORDER BY DISTANCE(" in sql
        assert " DESC" not in sql.upper().replace("DESCRIBE", "")
        assert sql.rstrip().endswith("LIMIT 37")

    def test_the_metric_tracks_the_configured_one(self, instance):
        instance._metric_fn = "cosine"
        assert "'COSINE'" in instance._query_sql(10)
        assert "'EUCLIDEAN'" not in instance._query_sql(10)


class TestMetricMapping:
    """docs/09 §2.4: only euclidean and cosine can be index metrics."""

    def test_both_metrics_the_harness_uses_are_mapped(self, overlay):
        _, percona = overlay
        assert percona.PERCONA.metric_names == {"angular": "cosine",
                                                "euclidean": "euclidean"}

    def test_no_metric_that_cannot_be_indexed_is_offered(self, overlay):
        """index_syntax.result:789-808 refuses euclidean_squared, dot and
        manhattan as index metrics with ER_NOT_SUPPORTED_YET. Offering one here
        would build nothing and fall back to a full scan."""
        _, percona = overlay
        offered = set(percona.PERCONA.metric_names.values())
        assert offered.isdisjoint({"euclidean_squared", "dot", "manhattan"})

    def test_mapped_names_are_lowercase_for_the_ddl(self, overlay):
        """The DD stores the canonical lowercase form
        (dd_params_resolved.result:55)."""
        _, percona = overlay
        for value in percona.PERCONA.metric_names.values():
            assert value == value.lower()


class TestEfSearchStatement:
    """docs/09 §1: innodb_hnsw_ef_search, SESSION scope."""

    def test_it_formats_into_a_set_session_statement(self, overlay):
        _, percona = overlay
        stmt = percona.PERCONA.set_ef_search.format(ef_search=123)
        assert stmt == "SET SESSION innodb_hnsw_ef_search = 123"

    def test_it_names_perconas_variable_and_not_another_engines(self, overlay):
        _, percona = overlay
        template = percona.PERCONA.set_ef_search
        assert "innodb_hnsw_ef_search" in template
        for foreign in ("mhnsw_ef_search", "vidx_hnsw_ef_search"):
            assert foreign not in template

    def test_the_template_has_the_placeholder_the_caller_fills(self, overlay):
        """`vb_mysql.set_query_arguments` calls
        `.format(ef_search=...)`; a template without the field silently ignores
        the sweep and measures one operating point many times."""
        _, percona = overlay
        assert "{ef_search}" in percona.PERCONA.set_ef_search


class TestRangeValidation:
    """The module refuses out-of-range M and efConstruction itself.

    Only the raising path is exercised: a valid call reaches
    `super().__init__`, which starts a server.
    """

    def test_ef_construction_bounds_match_the_source(self, overlay):
        _, percona = overlay
        assert (percona.MIN_EF_CONSTRUCTION,
                percona.MAX_EF_CONSTRUCTION) == (1, 4096)   # vec0vec.h:71-72

    def test_m_bounds_match_the_source(self, overlay):
        _, percona = overlay
        assert (percona.MIN_M, percona.MAX_M) == (2, 200)   # vec0vec.h:64-65

    @pytest.mark.parametrize("ef", [0, 4097])
    def test_out_of_range_ef_construction_is_refused(self, overlay, ef):
        _, percona = overlay
        with pytest.raises(ValueError, match="efConstruction"):
            percona.PerconaServer("euclidean", {"M": 16, "efConstruction": ef})

    @pytest.mark.parametrize("m", [1, 201])
    def test_out_of_range_m_is_refused(self, overlay, m):
        _, percona = overlay
        with pytest.raises(ValueError, match="^M must be in"):
            percona.PerconaServer("euclidean", {"M": m})

    def test_the_swept_m_values_are_all_accepted(self, overlay):
        """The grid in the overlay config must be inside the engine's range, or
        a sweep dies partway through."""
        _, percona = overlay
        import yaml
        cfg = yaml.safe_load(open(os.path.join(ALGORITHMS, "percona",
                                               "config.yml")))
        groups = cfg["float"]["any"][0]["run_groups"]["innodb"]["arg_groups"]
        swept = [m for g in groups for m in g["M"]]
        assert swept, "no M values are swept"
        assert all(percona.MIN_M <= m <= percona.MAX_M for m in swept), swept


# ---------------------------------------------------------------------------
# The plan check
# ---------------------------------------------------------------------------

class _FakeCursor:
    """A cursor that returns one canned EXPLAIN row."""

    def __init__(self, row):
        self._row = row

    def execute(self, sql, params=None):
        self._last = sql

    def fetchall(self):
        return [self._row]

    def fetchone(self):
        return (128,)          # VECTOR_DIM(v)


# Two plans, both taken from the server's own recorded output in
# mysql-test/suite/percona_vector/r/optimizer.result on the pinned SHA.
PLAN_USED = (1, "SIMPLE", "t1", None, "vector_ann", "vi", "vi", 0,
             "const", 100, 10.00, "Using where")          # :95-96
PLAN_NOT_USED = (1, "SIMPLE", "t1", None, "ALL", None, None, None,
                 None, 100, 100.00, "Using filesort")     # :127-128
# The plan this guard exists for: the vector index offered and not taken. The
# index name is in `possible_keys`, so any check that only looks for the name
# passes; `key` is another index and the access type is a range scan.
PLAN_OFFERED_NOT_TAKEN = (1, "SIMPLE", "t1", None, "range", "vi,tag_idx",
                          "tag_idx", 4, None, 50, 11.11, "Using where")


class TestPlanCheckIsNotSatisfiedByTheIndexName:
    """docs/09 §4.1: the shared substring check is vacuous on Percona.

    This is the repo's recurring defect shape -- a substring satisfied by
    something other than the thing being asserted -- and it is worth a test
    because the consequence is a full table scan recorded as an ANN search:
    exact results, slowly, which no output distinguishes from "accurate but
    slow".
    """

    @pytest.fixture
    def instance(self, overlay):
        _, percona = overlay
        obj = object.__new__(percona.PerconaServer)
        obj.dialect = percona.PERCONA
        obj._metric_fn = "euclidean"
        return obj

    def _verify(self, instance, plan_row):
        instance._cur = _FakeCursor(plan_row)
        return instance._verify_plan(10)

    def test_it_accepts_the_plan_that_used_the_index(self, instance):
        assert self._verify(instance, PLAN_USED) is True

    def test_it_rejects_the_plain_scan(self, instance):
        assert self._verify(instance, PLAN_NOT_USED) is False

    def test_it_rejects_a_plan_that_only_offered_the_index(self, instance):
        """The whole point of the override."""
        assert self._verify(instance, PLAN_OFFERED_NOT_TAKEN) is False

    def test_the_shared_check_would_have_accepted_that_plan(self, instance,
                                                            overlay):
        """Non-vacuity: proves the override is load-bearing.

        Without this, the three tests above would pass equally well against an
        override that changed nothing, and there would be no evidence that the
        inherited implementation had a hole.
        """
        vb, _ = overlay
        instance._cur = _FakeCursor(PLAN_OFFERED_NOT_TAKEN)
        assert vb.VBMySQLBase._verify_plan(instance, 10) is True
        # ...and the override rejects the same plan through the same cursor.
        assert instance._verify_plan(10) is False

    def test_the_marker_is_an_access_type_and_not_a_name(self, overlay):
        _, percona = overlay
        assert percona.PLAN_ACCESS_TYPE == "vector_ann"
        assert percona.PLAN_ACCESS_TYPE != percona.PERCONA.index_name

    def test_the_check_is_switched_on(self, overlay):
        """`verify_index_used=False` would skip it entirely."""
        _, percona = overlay
        assert percona.PERCONA.verify_index_used is True


class TestMarchIsReportedAndNotLostToAnUnknown:
    """docs/09 §4.8: BOTH `-march` readers know this image's path.

    "Never mix -march between engines" is a stated invariant, and `march` is the
    field that would catch a mixed build. A reader that does not know an
    engine's path reports `unknown` for every one of that engine's records,
    which is the "hard-coded probe whose figure reaches a published artefact
    looking measured" shape.

    **THIS CLASS WAS INVERTED BY THE CHANGE IT DESCRIBES, AND THAT IS WHY IT
    READS THIS WAY NOW.** §4.8 prescribes two steps -- add
    `/opt/percona/.march` to both tuples, and *then* drop the per-engine
    override in `percona/module.py`, "since the next engine added will hit the
    same thing twice". Until that was done these guards asserted the state the
    DESIGN task left behind: the path absent from the shared tuple and present
    in an override. Performing the prescribed change therefore failed four of
    them. They assert the finished state instead, and they are strictly
    stronger for it -- there are two readers, the old wording checked one, and
    the one it could not check (`harness/main.py`, in the harness container) is
    the one every ops record's `march` comes from.
    """

    @staticmethod
    def _march_paths(path_parts, name="_image_march") -> list:
        """The string literals a march reader actually iterates.

        Parsed out of the AST rather than matched against the function's text.
        The text version of this test was vacuous: the method's own DOCSTRING
        names `/opt/percona/.march`, so deleting the path from the real tuple
        left the substring present and the guard green. Measured, not assumed --
        with the path removed from the tuple the old assertion still passed.

        This matters more than most guards here because §4.8's failure mode is
        a wrong `march` reaching a published artefact looking measured, and
        `march` is the field that would catch a mixed-SIMD build.
        """
        import ast
        tree = ast.parse(open(os.path.join(*path_parts)).read())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                # The docstring is an Expr statement, never a `for` iterable,
                # so walking the loops cannot pick it up.
                for inner in ast.walk(node):
                    if isinstance(inner, ast.For) and isinstance(
                            inner.iter, (ast.Tuple, ast.List)):
                        return [e.value for e in inner.iter.elts
                                if isinstance(e, ast.Constant)
                                and isinstance(e.value, str)]
        raise AssertionError(f"no {name} loop over a literal sequence")

    #: The two readers, in the two containers. Named as data so every test
    #: below covers both and one cannot be quietly dropped from a loop.
    ANN_READER = (OVERLAY, "ann_benchmarks", "vb_mysql.py")
    OPS_READER = (VB_ROOT, "harness", "main.py")
    READERS = ((ANN_READER, "_image_march"), (OPS_READER, "_detect_march"))

    @pytest.mark.parametrize("parts,name", READERS)
    def test_every_march_reader_knows_this_images_path(self, parts, name):
        """Both tuples, not one.

        `vb_mysql.py` serves the ann pass; `harness/main.py` serves the ops
        pass from a container that mounts no overlay, so no module override
        could ever have reached it and `orchestrator/ops_pass.py` passes no
        `--march` for its `or` to fall back from.
        """
        paths = self._march_paths(parts, name=name)
        assert paths, f"{name}: the tuple is empty"
        assert "/opt/mariadb/.march" in paths, paths   # needle findable
        assert "/opt/percona/.march" in paths, paths

    def test_the_per_engine_override_is_gone(self):
        """§4.8's second instruction, and the half that is easy to skip.

        Adding the path to the shared tuples while leaving the override in
        place would work, and would leave the next engine two places to miss
        instead of one. Asserted on the AST rather than on the text, because
        the comment that REPLACED the override names the method it replaced
        and a substring check would match that comment -- the same vacuity
        shape the docstring hole had.
        """
        import ast
        tree = ast.parse(open(os.path.join(ALGORITHMS, "percona",
                                           "module.py")).read())
        defined = [n.name for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef)]
        assert "_image_march" not in defined, (
            "percona/module.py still overrides _image_march; §4.8 says the "
            "path belongs in both shared tuples instead")
        # Non-vacuity: the parser does find this file's other methods, so an
        # empty list is not what made the line above pass.
        assert "get_additional" in defined, defined
        # And the removal is explained where the override was, rather than
        # leaving a silent hole for the next reader to 'fix' by adding it back.
        body = open(os.path.join(ALGORITHMS, "percona", "module.py")).read()
        assert "_image_march" in body, "the reason it is absent is not recorded"

    @pytest.mark.parametrize("parts,name", READERS)
    def test_no_march_reader_can_be_satisfied_by_prose(self, parts, name):
        """Non-vacuity, stated as a test rather than as a comment.

        Proves the assertions above read the tuples and not the surrounding
        text: in both files the path appears in the function's text more often
        than in its tuple, so a text-matching guard cannot tell the two apart.
        """
        body = open(os.path.join(*parts)).read()
        slice_ = body.split(f"def {name}", 1)[1]
        slice_ = slice_.split("\n    def ", 1)[0].split("\ndef ", 1)[0]
        in_tuple = len([p for p in self._march_paths(parts, name=name)
                        if p == "/opt/percona/.march"])
        assert slice_.count("/opt/percona/.march") > in_tuple, name

    def test_the_dockerfile_writes_the_path_both_readers_read(self):
        """The two halves of one measurement, in three files."""
        dockerfile = open(os.path.join(DOCKER, "percona", "Dockerfile")).read()
        assert "/opt/percona/.march" in dockerfile


# ---------------------------------------------------------------------------
# The overlay config
# ---------------------------------------------------------------------------

class TestOverlayConfig:

    @pytest.fixture(scope="class")
    def cfg(self):
        import yaml
        return yaml.safe_load(open(os.path.join(ALGORITHMS, "percona",
                                                "config.yml")))

    @pytest.fixture(scope="class")
    def entry(self, cfg):
        return cfg["float"]["any"][0]

    def test_the_constructor_it_names_is_defined_in_the_module(self, entry):
        """A config naming a class the module does not define fails only once
        the image is built and the corpus is loaded -- the same check
        tests/test_config.py makes for every other engine."""
        import ast
        tree = ast.parse(open(os.path.join(ALGORITHMS, "percona",
                                           "module.py")).read())
        classes = {n.name for n in ast.walk(tree)
                   if isinstance(n, ast.ClassDef)}
        assert entry["constructor"] in classes

    def test_the_module_path_matches_the_directory(self, entry):
        assert entry["module"] == "ann_benchmarks.algorithms.percona"
        assert os.path.isfile(os.path.join(ALGORITHMS, "percona", "module.py"))

    def test_the_ann_constructor_collides_with_no_existing_engine(self, entry):
        """ann-benchmarks keys result files on the constructor, so a duplicate
        makes the second engine report the first's recall. This is why
        mariadb123 exists as a separate name."""
        import ast
        others = set()
        for name in sorted(os.listdir(ALGORITHMS)):
            path = os.path.join(ALGORITHMS, name, "module.py")
            if name == "percona" or not os.path.isfile(path):
                continue
            tree = ast.parse(open(path).read())
            others |= {n.name for n in ast.walk(tree)
                       if isinstance(n, ast.ClassDef)}
        assert others, "no other overlay modules found to compare against"
        assert entry["constructor"] not in others

    def test_it_is_innodb_only(self, entry):
        """Percona's vector index is InnoDB's (storage/innobase), so there is no
        MyISAM curve -- the alisql shape, not the mariadb one."""
        groups = entry["run_groups"]
        assert set(groups) == {"innodb"}
        for g in groups["innodb"]["arg_groups"]:
            assert g["engine"] == "InnoDB"

    def test_the_ef_search_grid_is_inside_the_servers_range(self, entry):
        """1..100000, clamped with a warning outside
        (innodb_hnsw_ef_search_basic.result:31-50). A clamped point would be
        recorded under the value asked for, not the value used."""
        grid = entry["run_groups"]["innodb"]["query_args"][0]
        assert grid
        assert all(1 <= ef <= 100000 for ef in grid), grid

    def test_the_ef_search_grid_matches_the_other_mysql_family_engines(self,
                                                                      entry):
        """Three curves sampled at different points are not comparable."""
        import yaml
        grid = entry["run_groups"]["innodb"]["query_args"][0]
        other = yaml.safe_load(open(os.path.join(ALGORITHMS, "alisql",
                                                 "config.yml")))
        other_grid = (other["float"]["any"][0]["run_groups"]["innodb"]
                      ["query_args"][0])
        assert grid == other_grid

    def test_ef_construction_is_declared_and_pinned_to_the_vendor_default(
            self, entry, overlay):
        """Percona is the third engine here to expose the knob, and a curve that
        did not say which value built it could not be read beside MariaDB's.

        The name says "pinned" and not "swept" because the config pins it to a
        single value: the old name claimed a sweep while asserting only that the
        key was present, which would have passed on any grid including a
        one-element one. Checked both ways now.
        """
        _, percona = overlay
        groups = entry["run_groups"]["innodb"]["arg_groups"]
        assert groups
        for g in groups:
            assert "efConstruction" in g
            # Pinned, deliberately: a knob only one engine has adds a grid
            # dimension that cannot be compared across engines, so the fair
            # reading beside MariaDB is at the vendor default.
            assert g["efConstruction"] == [percona.DEFAULT_EF_CONSTRUCTION]

    def test_the_config_says_in_words_that_it_is_pinned_not_swept(self):
        """The comment is the only place a reader learns it was a choice."""
        src = open(os.path.join(ALGORITHMS, "percona", "config.yml")).read()
        assert "pinned" in src.lower()


# ---------------------------------------------------------------------------
# The document
# ---------------------------------------------------------------------------

class TestTheDocument:

    @pytest.fixture(scope="class")
    def doc(self):
        return open(PERCONA_DOC).read()

    def test_it_exists_and_is_the_file_docs_are_numbered_for(self, doc):
        assert doc.startswith("# Adding Percona Server as a seventh engine")

    def test_the_pinned_sha_is_the_same_in_every_file_that_names_one(self, doc):
        """A pin that differs between the document and the code it describes is
        not a pin."""
        module = open(os.path.join(ALGORITHMS, "percona", "module.py")).read()
        shas = set(re.findall(r"\b[0-9a-f]{40}\b", doc + module))
        assert shas == {PINNED_SHA}, shas

    @pytest.fixture(scope="class")
    def section1(self, doc):
        return doc.split("## 1.", 1)[1].split("## 2.", 1)[0]

    def test_it_answers_the_ef_search_question_affirmatively(self, section1):
        """The question this investigation existed to answer.

        An earlier version of this test required only that the section contain
        `innodb_hnsw_ef_search`, `**Yes.**` and `SESSION`. The whole section was
        replaced with waffle that kept those three tokens and deleted every
        citation, both bounds and both riders, and the suite stayed green.

        So this asserts POLARITY and PLACEMENT, not presence: the affirmative
        answer has to come before the evidence, and the negative answer must be
        absent. Presence alone is what the hedge attack exploits.
        """
        yes = re.search(r"\*\*Yes\.\*\*", section1)
        assert yes, "the section does not answer affirmatively"
        assert "innodb_hnsw_ef_search" in section1
        # Placement: the answer leads, the evidence follows. A section that
        # buried the verdict under hedging would fail here.
        first_citation = re.search(r"\.result:\d+|\.cc:\d+|\.test:\d+",
                                   section1)
        assert first_citation, "the section cites nothing"
        assert yes.start() < first_citation.start()
        # Polarity: none of the phrasings that would mean the opposite.
        lowered = section1.lower()
        for wrong in ("exposes nothing equivalent", "no ef_search",
                      "build-time only", "not settable per query",
                      "has no query-time"):
            assert wrong not in lowered, wrong

    def test_it_states_both_scopes_and_the_bounds(self, section1):
        """The facts a driver author needs, each of which the hedge attack
        deleted while keeping the three tokens the old test looked for."""
        assert "SESSION" in section1 and "GLOBAL" in section1
        for value in ("40", "100000"):
            assert value in section1, value
        # The clamp, which is why an out-of-range grid point would be recorded
        # under a value the server did not use.
        assert "clamp" in section1.lower()

    def test_it_carries_both_riders(self, section1):
        """Each rider changes driver code, so neither may be dropped.

        The two tokens below are the QUOTED forms, not paraphrases, and that is
        the point. A second hedge attack got a §1 past all 94 guards: it led
        with `**Yes.**`, carried every token then demanded, met the citation
        floor exactly, and avoided all five banned phrasings -- while telling
        the follow-on task to do the opposite of what the investigation
        established, on the two sentences that become driver code. It kept the
        token `max(ef_search, k)` while arguing against implementing the clamp,
        because the old regex accepted either spelling.

        `max(ef, LIMIT)` is the server's own help string (ha_innodb.cc:1208)
        and `base.py:374` is the hook that has to be overridden; the attack
        carried neither. Requiring the quoted forms is the same technique that
        already works for the bounds.
        """
        assert "effective_ef" in section1
        assert "max(ef, LIMIT)" in section1, (
            "the clamp must be quoted in the server's own spelling, not "
            "paraphrased")
        assert "base.py:374" in section1
        assert "ef_construction" in section1

    def test_it_recommends_implementing_the_clamp_rather_than_inheriting(
            self, section1):
        """Polarity on the first rider's CONCLUSION, not just its tokens.

        The attack's sentence was "`effective_ef` should therefore be left as
        the inherited identity: overriding it would double-apply a clamp the
        server has already applied". Banning that wording alone would be
        brittle, and "the inherited identity" appears in the correct text too
        ("...and not leave the inherited identity"), so a substring ban would
        collide with the thing it is protecting -- hence the longer phrase
        `be left as the inherited identity`, which the correct text does not
        contain.

        REQUIRE THE INSTRUCTION PAIRED WITH ITS VALUE, NOT ONE VERB. An earlier
        version demanded the literal `should implement`, and that fired on
        rewordings preserving every fact and both conclusions:

            should **override** `effective_ef` as `max(ef, k)`   -> FAILED
            **must** implement `effective_ef` as `max(ef, k)`    -> FAILED

        `override` is if anything the more precise verb -- `effective_ef` is an
        inherited concrete method with a default at `base.py:374`, so
        overriding is literally what happens -- and `must` is strictly stronger
        than `should`. A guard that fires on correct prose teaches the next
        author to weaken or delete it, which is how the vacuity shapes reached
        this repo in the first place: a false positive on a correct document is
        a slower-acting version of the same failure.

        So: either verb, any modal, any wording between, but the target AND the
        value must both be present. `[\\s\\S]` rather than `.` so a line wrap
        cannot fail it, and rather than `\\W` because the gap contains words
        ("as", "with", "returning") -- a `\\W` version of this pattern rejected
        all seven legitimate rewordings for exactly that reason before it was
        measured against them.
        """
        assert re.search(
            r"(implement|override)[\s\S]{0,30}effective_ef"
            r"[\s\S]{0,60}max\(ef,\s*k\)", section1), (
            "§1 must tell the follow-on task to implement or override "
            "effective_ef AS max(ef, k)")
        for wrong in ("double-apply", "be left as the inherited identity"):
            assert wrong not in section1, wrong

    def test_it_says_the_claude_md_asymmetry_sentence_becomes_false(
            self, section1):
        """Polarity on the second rider's conclusion.

        The attack wrote "`ef_construction` is a build-time concern rather than
        a knob this engine surfaces, so `CLAUDE.md`'s 'exposed only by pgvector
        and Valkey' sentence stands and needs no edit" -- the exact opposite of
        the finding, with the token `ef_construction` present throughout.

        BANS SCOPED TO THE SENTENCE CARRYING THE SUBJECT. An earlier version
        banned `needs no edit` anywhere in §1, and that is subject-blind: it
        fires on a true sentence about something else, for instance

            "`base.py:374`'s inherited default is correct for every other
             engine here and **needs no edit**"

        which is correct prose about a different subject, and "needs no edit"
        is exactly the phrase that recurs in a change-list document. A ban has
        to name the subject it is a ban about, so these look only at sentences
        mentioning `CLAUDE.md` or `ef_construction` -- which is where the
        attack's claim lived, and where a claim of this kind has to live to
        mean anything.
        """
        assert re.search(r"sentence false|sentence.{0,40}false", section1), (
            "§1 must say the CLAUDE.md asymmetry sentence becomes false")
        subject_sentences = [
            s for s in re.split(r"(?<=[.!?])\s+", section1)
            if "CLAUDE.md" in s or "ef_construction" in s
        ]
        assert subject_sentences, (
            "§1 says nothing about ef_construction or CLAUDE.md at all")
        for sentence in subject_sentences:
            for wrong in ("sentence stands", "needs no edit"):
                assert wrong not in sentence, f"{wrong!r} in: {sentence[:90]}"

    def test_it_cites_more_than_a_token(self, section1):
        """A citation floor, because the hedge attack's tell was that every
        file:line went away while the verdict stayed.

        Raised from 4 to 6. §1 carries 7, and the second attack met a `>= 4`
        floor exactly -- a threshold three below the real count is a licence to
        delete three citations silently. One of slack, not three.
        """
        citations = re.findall(r"[\w/.-]+\.(?:result|test|cc|h|md):\d+",
                               section1)
        assert len(citations) >= 6, citations

    def test_the_sysvar_declaration_is_quoted_not_paraphrased(self, section1):
        """ha_innodb.cc's declaration is the primary source for every number in
        this section -- scope, default and both bounds in one place -- so the
        section quotes it rather than restating it."""
        assert "MYSQL_THDVAR_ULONG" in section1
        assert "ha_innodb.cc:" in section1

    # -- the prose is cross-checked against that quote ------------------
    #
    # A fourth attack shape passed all 104 guards: every required form, no
    # banned form, `**Yes.**` before the first citation, enough citations, and
    # BOTH riders' conclusions stated correctly -- with the two facts a reader
    # actually looks up inverted. It claimed the variable was GLOBAL-only and
    # established once per connection, and that the default was 100000 with a
    # floor of 40.
    #
    # The detail that makes it fixable: the section quotes the correct
    # declaration -- `nullptr, nullptr, 40, 1, 100000, 0)` -- three lines above
    # the prose contradicting it. The guards required that quote and nothing
    # compared the two. These do, value against value, which is the technique
    # that makes the §3 arithmetic guard strong and which vocabulary cannot
    # satisfy.
    #
    # It is load-bearing because §1 is where the calibration-grid author and
    # the driver author read scope and range. "GLOBAL-only, once per
    # connection" sends the driver to `global_setup` instead of
    # `set_ef_search`, which is the whole calibration mechanism; "default
    # 100000, floor 40" licenses a grid below 40 that the server silently
    # clamps -- the exact failure the clamp sentence exists to warn about.

    @staticmethod
    def _declared(section1):
        """(default, min, max) parsed out of the quoted declaration."""
        decl = re.search(r"MYSQL_THDVAR_ULONG\(([\s\S]*?)\);", section1)
        assert decl, "§1 no longer quotes the MYSQL_THDVAR_ULONG declaration"
        tail = re.search(
            r"nullptr\s*,\s*nullptr\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)",
            decl.group(1))
        assert tail, (
            "the quoted declaration's trailing arguments do not parse; the "
            "cross-checks below have nothing to compare against")
        return tail.group(1), tail.group(2), tail.group(3)

    @staticmethod
    def _flat(section1):
        """Whitespace-normalised section.

        Every check below runs on this, not on the raw text. Measured reason:
        the fourth attack wrote "a read-only\\nreflection of it", and a raw
        substring ban on `read-only reflection` did not fire because the phrase
        straddled a line wrap. A prose guard that a reflow defeats is not a
        guard.
        """
        return " ".join(section1.split())

    def test_the_declaration_quote_parses_to_a_default_and_two_bounds(
            self, section1):
        """Pin the quote itself, so the cross-checks cannot pass vacuously.

        If the declaration were reformatted or trimmed to the point where the
        trailing arguments stop parsing, every value comparison below would
        have nothing to compare and would need to fail loudly rather than
        quietly succeed. `_declared` asserts, so this test is that failure.
        """
        default, mn, mx = self._declared(section1)
        assert (default, mn, mx) == ("40", "1", "100000"), (default, mn, mx)

    def test_the_stated_default_and_maximum_match_the_declaration(
            self, section1):
        """The prose's numbers must agree with the quote three lines above it.

        Four comparisons, each tying a number to a ROLE rather than checking
        that the number appears somewhere: the attack contained both 40 and
        100000, so presence proves nothing. It swapped which was which.
        """
        default, mn, mx = self._declared(section1)
        flat = self._flat(section1)
        assert re.search(r"default\D{0,20}\b%s\b" % default, flat, re.I), (
            f"§1 must state the default as {default}, as its own quote does")
        assert not re.search(r"default\D{0,20}\b%s\b" % mx, flat, re.I), (
            f"§1 calls {mx} the default; the declaration it quotes says "
            f"{default} is the default and {mx} is the maximum")
        assert re.search(r"(maximum|upper bound|range)\D{0,24}\b%s\b" % mx,
                         flat, re.I), (
            f"§1 must state {mx} as the maximum or the top of the range")
        assert not re.search(r"(floor|minimum)\D{0,20}\b%s\b" % default, flat,
                             re.I), (
            f"§1 calls {default} the floor; the declaration it quotes says "
            f"the minimum is {mn}")

    def test_the_stated_scope_matches_the_session_variable_macro(
            self, section1):
        """`MYSQL_THDVAR_*` is the session-variable macro, and the engine reads
        it per query through `THDVAR`. Prose saying otherwise sends the driver
        to the wrong hook.
        """
        flat = self._flat(section1)
        assert "MYSQL_THDVAR_ULONG" in flat
        assert "SET SESSION innodb_hnsw_ef_search" in flat, (
            "§1 must show the variable being set per session")
        assert "THDVAR" in flat, (
            "§1 must cite the per-query read, or 'per query' is an assertion")
        assert not re.search(r"\bGLOBAL\b[^.]{0,40}\b(only|in scope)\b", flat), (
            "§1 describes the variable as GLOBAL-only; MYSQL_THDVAR_ULONG is "
            "session and global")
        assert not re.search(r"read-only\s+reflection", flat), (
            "§1 calls the session value a read-only reflection of the global "
            "one; it is settable per session")

    def test_it_keeps_the_live_server_questions_in_their_own_section(self, doc):
        """docs/09's own promise: what needs a server is listed separately so a
        reader can tell reading from measurement."""
        assert "## 7. What only a live server can settle" in doc
        section = doc.split("## 7. What only a live server can settle", 1)[1]
        # Enumerated, not a paragraph of hedging. Floor raised from 5 to 7:
        # §7 carries 8, and a threshold three below the real count would let
        # three questions be dropped without a red test -- the same loose-floor
        # shape the second §1 hedge attack exploited.
        items = re.findall(r"^\d+\. \*\*", section, re.MULTILINE)
        assert len(items) >= 7, items
        # Contiguous numbering, so striking an item renumbers rather than
        # leaving a gap a reader reads as a missing question.
        numbers = [int(n) for n in re.findall(r"^(\d+)\. \*\*", section,
                                              re.MULTILINE)]
        assert numbers == list(range(1, len(numbers) + 1)), numbers

    def test_the_struck_items_heading_counts_its_own_bullets(self, doc):
        """Self-consistency, the same shape as the submodule-count guard.

        Caught by mutation: renaming the heading from "Three" to "Two" while
        leaving three bullets passed every other guard in this file. A count in
        a heading that disagrees with the list under it is how the submodule
        note came to say "two" over four entries, and the fix is the same --
        anchor on one deterministic phrase rather than scan for number words.
        """
        section = doc.split("## 7. What only a live server can settle", 1)[1]
        heading = re.search(r"^### (\w+) items struck from this list", section,
                            re.MULTILINE)
        assert heading, "the struck-items subsection is missing or renamed"
        words = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
        stated = words.get(heading.group(1))
        assert stated is not None, heading.group(1)
        struck = section.split(heading.group(0), 1)[1]
        bullets = re.findall(r"^- \*\*", struck, re.MULTILINE)
        assert len(bullets) == stated, (
            f"heading says {heading.group(1)}, {len(bullets)} bullets listed")

    def test_it_names_every_file_the_follow_on_task_must_change(self, doc):
        """The brief for this design lists these by name; a change list missing
        one of them gets as far as starting a server and then dies on
        `argument --engine: invalid choice`."""
        section = doc.split("## 4. File-by-file change list", 1)[1]
        for required in ("harness/drivers/mysql_family.py",
                         "_driver_classes()",
                         "harness/drivers/postgres.py",
                         "config/engines/percona.yml",
                         "GOLDEN_ORDER",
                         "tests/test_engine_registry.py",
                         "webui/static/fields.js",
                         "config/profiles/smoke.yml",
                         # Added after review found all three missing. Each is
                         # a file Percona breaks or silently degrades:
                         # ann_pass raises, main.py blanks `march`, lib.sh
                         # leaves a vendor checkout unprotected.
                         "orchestrator/ann_pass.py",
                         "harness/main.py",
                         "scripts/lib.sh"):
            assert required in section, required

    def test_it_records_the_family_wide_entrypoint_defect(self, doc):
        """§4.9. Found while matching the family's entrypoint, out of scope to
        fix, and the kind of thing that is lost if it is not written down.

        Reproduced rather than reasoned: the pattern extracted verbatim with
        `id -u` stubbed exits 1 before `exec` for a non-root uid and prints
        nothing. Latent only because no image here sets USER.
        """
        section = doc.split("### 4.9", 1)[1].split("### 4.10", 1)[0]
        # All three siblings named, so a fix cannot be applied to one.
        for path in ("entrypoint-percona.sh", "entrypoint-alisql.sh",
                     "entrypoint-mariadb.sh"):
            assert path in section, path
        assert "set -euo pipefail" in section
        assert "latent" in section.lower()
        # The reproduction, not just the claim.
        assert "FAKE_UID=1000" in section and "rc=1" in section

    def test_the_entrypoint_pattern_it_describes_is_really_there(self):
        """The claim is about three files in this checkout, so check them.

        If a sibling is ever fixed without the others, this fails and says so --
        which is the point of recording it as family-wide.
        """
        for engine in ("percona", "alisql", "mariadb"):
            path = os.path.join(DOCKER, engine, f"entrypoint-{engine}.sh")
            src = open(path).read()
            # The call site: this line IS byte-identical across all three.
            assert 'local u; u="$(user_args)"' in src, (
                f"{engine}: call-site pattern gone, update §4.9")
            assert "set -euo pipefail" in src, engine
            # The function body whose non-zero exit `set -e` acts on. Matched
            # on the test-and-printf rather than on a one-line definition:
            # mariadb's user_args() is formatted across four lines with a
            # comment, so the three are NOT byte-identical as functions even
            # though they behave identically and the call site is. A guard on
            # the one-line spelling passed for alisql and percona and failed
            # for mariadb, which would have read as "mariadb is already fixed".
            assert re.search(r'\[\[ "\$\(id -u\)" -eq 0 \]\] && printf', src), \
                engine

    def test_it_records_where_ann_pass_refuses_an_unknown_engine(self, doc):
        """§4.10's whole point is that Percona lands on this `raise`. If the line
        moves, the follow-on task is sent to the wrong place."""
        source = open(os.path.join(VB_ROOT, "orchestrator",
                                   "ann_pass.py")).read().splitlines()
        claimed = int(re.search(r"^(\d+)\s+raise ValueError\(f\"unknown engine",
                                doc, re.MULTILINE).group(1))
        assert "raise ValueError" in source[claimed - 1]
        assert "unknown engine" in source[claimed - 1]

    def test_no_mysql_family_branch_in_ann_pass_emits_ef_construction(self,
                                                                      doc):
        """§4.10 says Percona needs a branch of its own because the MySQL-family
        branch cannot emit `efConstruction`. Checked, not asserted."""
        source = open(os.path.join(VB_ROOT, "orchestrator",
                                   "ann_pass.py")).read()
        body = source.split("def render_config", 1)[1]
        mysql_branch = body.split('if engine in MARIADB_ENGINES', 1)[1]
        mysql_branch = mysql_branch.split('elif engine ==', 1)[0]
        # Non-vacuity: the pgvector branch right after it does emit the key.
        assert "efConstruction" in body
        assert "efConstruction" not in mysql_branch

    def test_the_ops_path_march_reader_covers_this_image(self, doc):
        """§4.8's second half, now done.

        A different container from the overlay, so no module override can
        reach it -- which is exactly why the path had to go in here and not
        only in `percona/module.py`. This assertion used to be `not in`,
        recording that the work was outstanding; it fired when the work was
        done, which is the right direction for it to fire in.
        """
        source = open(os.path.join(VB_ROOT, "harness", "main.py")).read()
        fn = source.split("def _detect_march", 1)[1].split("\ndef ", 1)[0]
        assert "/opt/mariadb/.march" in fn          # needle findable
        assert "/opt/percona/.march" in fn
        assert "harness/main.py" in doc
        # And nothing passes --march in from the orchestrator, so the detector
        # is what runs.
        ops = open(os.path.join(VB_ROOT, "orchestrator", "ops_pass.py")).read()
        assert "march" not in ops

    def test_it_does_not_claim_optimize_table_fails_to_compact(self, doc):
        """The review's HIGH 1. `handler0alter.cc:1393-1404` says a table with a
        vector index is rebuilt in place and the graph rebuilt from the copied
        rows; a DELETEd row is not copied, so OPTIMIZE compacts. An earlier
        revision said the opposite, having read a scare-quoted hedge in an MTR
        comment as a claim about the statement.

        Polarity, not presence: the section has to say it DOES compact.
        """
        section = doc.split("### 2.7", 1)[1].split("## 3.", 1)[0]
        assert "handler0alter.cc:1393-1404" in section
        assert "**does compact**" in section
        for wrong in ("does **not** compact graph tombstones",
                      "`OPTIMIZE TABLE` is not it"):
            assert wrong not in section, wrong
        # And it still explains why maintain() is the index replace anyway.
        assert "without rewriting the base table" in section

    def test_it_records_the_build_py_lines_it_calls_a_defect(self, doc):
        """§2.5's claim ABOUT build.py, beyond where its lines are.

        The line range itself is checked by the citation table below, as one
        row among all of them. What stays here is the part a line check cannot
        express: the document's claim that `build_mode == "post"` has no
        counterpart anywhere in the file. That is the defect §2.5 is about, and
        it would survive any amount of renumbering.
        """
        source = open(os.path.join(VB_ROOT, "harness", "workloads",
                                   "build.py")).read()
        assert 'index.build_mode == "incremental"' in source
        assert 'build_mode == "post"' not in source
        assert "Lines 93-95" in doc


#: Every `path:line` citation in docs/09 that points into THIS repo, with a
#: string that must appear in the cited line or range.
#:
#: WHY THIS TABLE EXISTS. The document carried about twenty-one such citations
#: and exactly two of them were checked. The other nineteen rotted silently,
#: and that is not theoretical: merging #12 and #13 invalidated two of them in
#: a day. `postgres.py:481` came to point at a sentence of prose, and the one
#: guard that existed caught it -- which is the whole argument for having the
#: rest. Three more were stale that nobody had flagged, found only by walking
#: every citation: `churn.py:310-311` (moved to 484-485 by T4b),
#: `docs/07-planning-a-run.md:478` (moved to 481), and the inline line numbers
#: inside §4.8's quoted `_detect_march` block.
#:
#: The key is the citation AS THE DOCUMENT SPELLS IT, so the completeness test
#: below can match the two against each other. Both directions are then
#: covered: a number that changes in the document without changing here fails
#: completeness, and a number that no longer matches the file fails its anchor.
#:
#: Anchors are chosen to be the thing the document is pointing AT, not merely
#: a string that happens to be nearby. Where a citation has no such thing it
#: goes in NO_ANCHOR with a reason rather than getting a weak anchor invented
#: for it to make a row pass.
CITED_IN_REPO = {
    # doc spelling                   -> (real path, anchor)
    "base.py:239-258":
        ("harness/drivers/base.py", 'repair_path: str = "none"'),
    "base.py:255-256":
        ("harness/drivers/base.py", "Valkey and Percona Search inherit"),
    "base.py:374":
        ("harness/drivers/base.py", "def effective_ef(self, ef: int, k: int)"),
    "harness/drivers/base.py:325-371":
        ("harness/drivers/base.py", "def prepare_churn"),
    "harness/drivers/base.py:374":
        ("harness/drivers/base.py", "def effective_ef(self, ef: int, k: int)"),
    "churn.py:525-526":
        ("harness/workloads/churn.py", "what = driver.maintain()"),
    "harness/workloads/build.py:12-16":
        ("harness/workloads/build.py", "ingest_wall_s"),
    "harness/datasets.py:492":
        ("harness/datasets.py", "1,183,514 training rows"),
    "harness/main.py:209-227":
        ("harness/main.py", "def _detect_march"),
    "harness/drivers/mysql_family.py:58":
        ("harness/drivers/mysql_family.py", "def to_binary_f32"),
    # The anchor moved WITH the code, and that is the citation's job. §4.1
    # replaced the hard-coded `id INT PRIMARY KEY` with `Dialect.pk_type`, so
    # the literal the document used to point at no longer exists in this file
    # -- it exists in the dialect default instead. Pointing the row at the
    # interpolation keeps it checking the line §2.2 is talking about.
    "harness/drivers/mysql_family.py:234":
        ("harness/drivers/mysql_family.py",
         "id {self.dialect.pk_type} PRIMARY KEY"),
    "mysql_family.py:162-172":
        ("harness/drivers/mysql_family.py", "host=self.spec.host"),
    "postgres.py:689":
        ("harness/drivers/postgres.py", "def _driver_classes()"),
    "ops_pass.py:231":
        ("orchestrator/ops_pass.py", "VB_ENGINE_TAG"),
    "orchestrator/engines.py:63-65":
        ("orchestrator/engines.py", "def _source_tag"),
    "docs/07-planning-a-run.md:505":
        ("docs/07-planning-a-run.md", "60,000-row smoke"),
    # Found only once `_CITE_RE` learned to see extensionless filenames --
    # see the note on that pattern. It is the citation that sat at `:113`,
    # 51 lines wrong, for this document's whole life.
    "docker/percona/Dockerfile:725":
        ("docker/percona/Dockerfile", "ARG PERCONA_TAG=unknown"),
    "docker/alisql/entrypoint-alisql.sh:94":
        ("docker/alisql/entrypoint-alisql.sh", 'local u; u="$(user_args)"'),
    "docker/mariadb/entrypoint-mariadb.sh:95":
        ("docker/mariadb/entrypoint-mariadb.sh", 'local u; u="$(user_args)"'),
    "vb_mysql.py:96":
        ("overlay/ann-benchmarks/ann_benchmarks/vb_mysql.py",
         "id INT PRIMARY KEY"),
    "vb_mysql.py:237":
        ("overlay/ann-benchmarks/ann_benchmarks/vb_mysql.py", "unix_socket"),
    "vb_mysql.py:400":
        ("overlay/ann-benchmarks/ann_benchmarks/vb_mysql.py",
         "force_index_hint"),
    "vb_mysql.py:428-436":
        ("overlay/ann-benchmarks/ann_benchmarks/vb_mysql.py",
         "CHARACTER_OCTET_LENGTH"),
    "overlay/.../vb_mysql.py:51":
        ("overlay/ann-benchmarks/ann_benchmarks/vb_mysql.py",
         "def to_binary_f32"),
    "overlay/.../vb_mysql.py:96":
        ("overlay/ann-benchmarks/ann_benchmarks/vb_mysql.py",
         "id INT PRIMARY KEY"),
    "overlay/.../vb_mysql.py:538-564":
        ("overlay/ann-benchmarks/ann_benchmarks/vb_mysql.py",
         "/opt/mariadb/.march"),
    "overlay/.../algorithms/alisql/module.py:22-24":
        ("overlay/ann-benchmarks/ann_benchmarks/algorithms/alisql/module.py",
         "Field_vector::store()"),
}

#: Citations the document makes as a bare `:N` continuation of the file named
#: in the sentence before -- "`postgres.py:689` … the dict at `:701-707`".
#: The auto-parser cannot see these, because the path is not in the backticks,
#: so they are covered deliberately rather than discovered. They are the four
#: that point into this repo; the rest are Percona sources.
#:
#: This is the honest limit of the completeness check: a NEW bare continuation
#: could be added without a row and nothing would notice. Naming the limit
#: here is the point of writing it down.
BARE_CONTINUATIONS = {
    ("harness/drivers/postgres.py", 698, 698):
        "from .mysql_family import",
    # NOTE the `and` in the old form of this value -- `'"PerconaDriver"' and
    # "return {"` evaluates to its RIGHT operand, so the anchor was only ever
    # `return {` and the PerconaDriver half was never checked. Both are
    # asserted now, as a tuple the test iterates.
    ("harness/drivers/postgres.py", 701, 713):
        ("return {", '"PerconaDriver": PerconaDriver'),
    ("harness/drivers/postgres.py", 710, 730):
        "def _driver_table()",
    ("harness/main.py", 283, 283):
        "args.march or _detect_march()",
}

#: Local citations with no sensible anchor, and why. Empty, and that is a
#: finding rather than a default: every citation the document makes into this
#: repo points at a named thing -- a def, a literal, a sentence -- so none
#: needed a weak anchor invented for it.
NO_ANCHOR: Dict[str, str] = {}

#: Paths that belong to the pinned Percona tree. They cannot be checked from
#: here and must not be: they are pinned by SHA
#: (02a8b80c70b0ccf6c9035d8a2307e7fa988cc304), so a local file of the same name
#: would be the wrong file, and checking against one would be worse than not
#: checking at all. Verified by reading that tree, which is what §0's
#: provenance table commits the document to.
#:
#: A deny-list by name, deliberately, and not "whatever does not resolve
#: locally". Resolving by existence would misclassify a Percona source that
#: happens to share a filename with a repo file as local, and then check the
#: document against the wrong file -- which is the one outcome worse than not
#: checking. The cost is that a newly cited Percona file must be added here,
#: and `test_every_local_citation_has_a_row` is what makes that cost visible:
#: it is how `ddl0loader.cc:273` was caught sitting unclassified in §0's commit
#: table the first time this guard ran.
FOREIGN_MARKERS = (
    "percona-hnsw", "ha_innodb", "handler0alter", "field.cc", "field.h",
    "item_create", "sys_vars.cc", "vec0vec", "vector_constants",
    "CMakeLists", "keyring_kmip", "mysql-test", "MYSQL_VERSION",
    # Cited in §4.1 for the three servers' EXPLAIN grammars. Invisible to
    # `_CITE_RE` today because `.yy` is not in its extension list; listed
    # anyway so that widening the parser classifies it instead of failing.
    "sql_yacc",
    "ddl0loader",
)

#: Every `path:N[-M]` citation in backticks. The second alternative is for
#: extensionless filenames, and `Dockerfile` is the one this document cites.
#:
#: **IT WAS ADDED BECAUSE THE OMISSION HID A LIVE ERROR.** The parser saw 60 of
#: this document's 72 citations; eleven of the misses were foreign `.result` and
#: `.test` paths that `_is_foreign` would have skipped anyway, and exactly one
#: was in-repo: the `docker/percona/Dockerfile` citation, which sat in §4.7
#: reading `:113` -- wrong by 51 lines, pointing at `ENV DEBIAN_FRONTEND`, with
#: nothing in this file able to see it. (No line number is given for it here on
#: purpose: this is a narrative about a citation that has since been corrected
#: and has since moved, and a number in prose that nothing checks is exactly
#: the rot the row below exists to stop. The live, checked number is in
#: CITED_IN_REPO.) A completeness guard that cannot see a whole
#: CLASS of citation reports coverage it does not have, which is the same shape
#: as the vacuity it exists to catch.
#:
#: Group numbering is unchanged: the inner alternation is non-capturing, so
#: group 1 is still the path and 2/3 the line range.
_CITE_RE = re.compile(
    r"`((?:[A-Za-z0-9_./-]+\.(?:py|sh|js|yml|md|cc|h|txt)"
    r"|[A-Za-z0-9_./-]*Dockerfile)):(\d+)(?:-(\d+))?`")


def _doc_citations():
    """Every `path:N[-M]` citation in docs/09, as the document spells it."""
    text = open(PERCONA_DOC).read()
    out = set()
    for m in _CITE_RE.finditer(text):
        lo, hi = m.group(2), m.group(3)
        out.add("%s:%s%s" % (m.group(1), lo, "-" + hi if hi else ""))
    return out


def _is_foreign(citation):
    return any(marker in citation for marker in FOREIGN_MARKERS)


class TestEveryCitationIntoThisRepoIsChecked:
    """Turn a document that rots into one that reports its own rot.

    The `march` AST guard replaced a docstring hole by reading the thing it
    was asserting about; this does the same for line numbers. Twenty-six
    citations, each opened and matched, instead of two.
    """

    def test_the_parser_finds_the_citations(self):
        """Non-vacuity: a regex that matched nothing would make every
        assertion below pass over an empty set."""
        found = _doc_citations()
        assert len(found) >= 50, len(found)
        assert "postgres.py:689" in found
        assert any(_is_foreign(c) for c in found)

    def test_every_local_citation_has_a_row(self):
        """Completeness, and the half that makes this closed rather than a
        snapshot: a citation added to the document later, or a number edited
        in the document without editing the table, lands here.
        """
        found = {c for c in _doc_citations() if not _is_foreign(c)}
        covered = set(CITED_IN_REPO) | set(NO_ANCHOR)
        missing = sorted(found - covered)
        assert not missing, (
            "docs/09 cites these repo locations and nothing checks them: "
            + ", ".join(missing))

    def test_no_row_cites_something_the_document_dropped(self):
        """The other direction: a row for a citation the document no longer
        makes is dead weight that reads as coverage."""
        found = _doc_citations()
        stale = sorted(c for c in set(CITED_IN_REPO) | set(NO_ANCHOR)
                       if c not in found)
        assert not stale, (
            "these rows cite locations docs/09 no longer mentions: "
            + ", ".join(stale))

    @pytest.mark.parametrize("citation", sorted(CITED_IN_REPO))
    def test_the_citation_points_at_what_the_document_says(self, citation):
        path, anchor = CITED_IN_REPO[citation]
        spec = citation.rsplit(":", 1)[1]
        lo, _, hi = spec.partition("-")
        lo, hi = int(lo), int(hi or lo)
        full = os.path.join(VB_ROOT, path)
        assert os.path.isfile(full), f"{path} does not exist"
        lines = open(full).read().splitlines()
        assert hi <= len(lines), (
            f"docs/09 cites {citation} but {path} has {len(lines)} lines")
        body = "\n".join(lines[lo - 1:hi])
        assert anchor in body, (
            f"docs/09 cites {citation} for {anchor!r}; "
            f"{path}:{lo}-{hi} now reads: {body.strip()[:120]!r}")

    @pytest.mark.parametrize("key", sorted(BARE_CONTINUATIONS))
    def test_the_bare_continuations_still_point_where_they_did(self, key):
        """The `:N` citations the auto-parser cannot see.

        A value may be one anchor or a tuple of them. The tuple form exists
        because the single-string form invited `'a' and 'b'`, which Python
        evaluates to `'b'` -- so a row that read as two assertions made one,
        and the half that was silently dropped was the half about the entry
        the follow-on task had to add.
        """
        path, lo, hi = key
        anchors = BARE_CONTINUATIONS[key]
        if isinstance(anchors, str):
            anchors = (anchors,)
        lines = open(os.path.join(VB_ROOT, path)).read().splitlines()
        assert hi <= len(lines), f"{path} has {len(lines)} lines"
        body = "\n".join(lines[lo - 1:hi])
        for anchor in anchors:
            assert anchor in body, (
                f"{path}:{lo}-{hi} no longer contains {anchor!r}; it reads "
                f"{body.strip()[:120]!r}")

    def test_foreign_citations_are_skipped_explicitly(self):
        """The Percona sources are pinned by SHA and are not in this checkout.

        Asserted rather than left implicit, because "we check every citation"
        would otherwise be false in a way nobody could see: a local file
        sharing a name with a Percona source would be the wrong file, and
        checking against it would be worse than not checking.
        """
        foreign = {c for c in _doc_citations() if _is_foreign(c)}
        assert len(foreign) >= 25, sorted(foreign)
        for citation in foreign:
            assert citation not in CITED_IN_REPO, citation
        doc = open(PERCONA_DOC).read()
        assert "02a8b80c70b0ccf6c9035d8a2307e7fa988cc304" in doc

    def test_no_anchor_set_is_documented_rather_than_silent(self):
        """If a citation ever has no sensible anchor, saying so is the finding.

        Currently empty, which is itself worth asserting: it means no citation
        was waved through.
        """
        for citation, reason in NO_ANCHOR.items():
            assert reason.strip(), citation
        assert NO_ANCHOR == {}, (
            "NO_ANCHOR has entries; name them in the round report rather than "
            "letting them sit unchecked: " + ", ".join(NO_ANCHOR))


class TestTheMemoryArithmeticReproduces:
    """docs/09 §3's table, recomputed from the formula it cites.

    The formula is the floor an MTR test asserts the server stays above:
    mysql-test/suite/percona_vector/r/hnsw_memory_used.result:63 is
    `SET @floor = 1000 * (64 * 4 + 2 * 16 * 8);` for 1000 nodes of VECTOR(64) at
    M = 16, generalising to rows * (dims * 4 + 2 * M * 8).

    These are NOT measurements and the document says so: rows, dims and M are
    all supplied, from this repo's dataset facts and profiles. What is checked
    here is only that the numbers printed are the ones the formula gives -- an
    arithmetic slip in a table that a reader will use to size a two-day run.
    """

    GIB = 1073741824

    @staticmethod
    def _floor(rows: int, dims: int, m: int) -> int:
        return rows * (dims * 4 + 2 * m * 8)

    @pytest.fixture(scope="class")
    def rows(self):
        """The §3 table, parsed out of the document."""
        doc = open(PERCONA_DOC).read()
        section = doc.split("## 3. `innodb_hnsw_max_memory`", 1)[1]
        out = []
        for line in section.splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) != 7 or not re.fullmatch(r"[\d,]+|per 1M", cells[1]):
                continue
            out.append(cells)
        assert out, "could not parse the §3 table"
        return out

    def test_the_table_was_found_and_has_every_row(self, rows):
        """Non-vacuity: a parser that matched nothing would make every
        assertion below pass silently."""
        assert len(rows) == 5

    def test_every_floor_matches_the_formula(self, rows):
        for cells in rows:
            _, raw_rows, dims, m, floor, _, _ = cells
            n = 1_000_000 if raw_rows == "per 1M" else int(raw_rows.replace(",", ""))
            want = self._floor(n, int(dims), int(m))
            got = int(re.sub(r"[^\d]", "", floor))
            assert got == want, f"{cells[0]}: doc says {got:,}, formula gives {want:,}"

    def test_every_ratio_matches_its_floor(self, rows):
        for cells in rows:
            floor = int(re.sub(r"[^\d]", "", cells[4]))
            claimed = float(re.sub(r"[^\d.]", "", cells[5]))
            assert round(floor / self.GIB, 3) == claimed, cells[0]

    def test_the_over_the_default_column_follows_from_the_numbers(self, rows):
        """A row whose floor exceeds the default and whose verdict says "no"
        would be the dangerous kind of error: it reads as reassurance."""
        for cells in rows:
            floor = int(re.sub(r"[^\d]", "", cells[4]))
            verdict = cells[6].lower()
            if floor > self.GIB:
                assert verdict.startswith("**yes") or verdict.startswith("yes"), cells
            else:
                assert "floor no" in verdict or verdict == "no", cells

    def test_section_3_states_a_decided_policy_not_a_recommendation(self):
        """The human chose policy (1). §3 has to read as settled.

        Polarity, because the failure mode is a reader relitigating a decision
        whose reasoning is still on the page: "two defensible policies, the
        choice is the user's" is exactly right *before* the choice and
        misleading after it.
        """
        doc = open(PERCONA_DOC).read()
        section = doc.split("## 3. `innodb_hnsw_max_memory`", 1)[1].split(
            "## 4.", 1)[0]
        assert "### The policy: `innodb_hnsw_max_memory = 0`" in section
        assert "**This is decided, not proposed.**" in section
        assert "CHOSEN" in section
        for stale in ("### The recommendation",
                      "This document recommends (1)",
                      "the choice is the user's because it is a fairness "
                      "judgement and not a fact"):
            assert stale not in section, stale
        # The justification survives the decision -- both policies still argued.
        assert "graph_cache_bytes`, accepting that some configurations will "\
               "fail" in section

    def test_the_node_map_caveat_is_attached_to_the_gauge(self):
        """The caveat has to sit where a reader takes a figure from the
        variable, not in a footnote elsewhere: the chosen policy's audit trail
        IS that variable, and it is short by one node map per vector index."""
        doc = open(PERCONA_DOC).read()
        section = doc.split("## 3. `innodb_hnsw_max_memory`", 1)[1].split(
            "## 4.", 1)[0]
        assert "Docs/percona-hnsw.md:1567-1568" in section
        assert "lower bound" in section
        # The projection, with both ends, and marked as a projection.
        assert "35.5 to 59.2 MB" in section
        assert "projection" in section
        assert "supplied" in section

    def test_the_node_map_projection_reproduces(self):
        """Arithmetic, from the repo's own row count and Percona's stated range.

        Not a measurement and §3 says so: the row count is supplied from
        harness/datasets.py:492 and 30-50 bytes/node is Percona's own estimate.
        """
        datasets = open(os.path.join(VB_ROOT, "harness", "datasets.py")).read()
        assert "1,183,514 training rows" in datasets
        rows = 1183514
        assert rows * 30 == 35_505_420
        assert rows * 50 == 59_175_700
        assert round(rows * 30 / 1e6, 1) == 35.5
        assert round(rows * 50 / 1e6, 1) == 59.2

    def test_section_3_keeps_the_line_that_the_gate_cannot_surface_this(self):
        """The most load-bearing sentence in §3 for anyone reading a green
        smoke run as evidence the memory setting is fine."""
        doc = open(PERCONA_DOC).read()
        section = doc.split("## 3. `innodb_hnsw_max_memory`", 1)[1].split(
            "## 4.", 1)[0]
        assert "the gate cannot surface this" in section
        assert "0.19" in section

    def test_the_default_is_the_one_the_server_ships(self):
        """innodb_hnsw_max_memory_basic.result:3-5 reads 1073741824."""
        doc = open(PERCONA_DOC).read()
        assert "1073741824" in doc
        assert self.GIB == 1073741824

    def test_the_smoke_profile_really_does_fit_under_the_default(self):
        """§3's conclusion that the gate will not surface the ceiling, taken
        from the profile rather than from the document."""
        import yaml
        profile = yaml.safe_load(open(os.path.join(
            VB_ROOT, "config", "profiles", "smoke.yml")))
        subset = profile["ops"]["subset_rows"]
        m_values = profile["ann"]["m_values"]
        assert subset == 20000 and m_values == [16]
        # fashion-mnist-784: dims from the dataset name the profile asks for.
        assert profile["datasets"] == ["fashion-mnist-784-euclidean"]
        assert self._floor(subset, 784, max(m_values)) < self.GIB
        # ...and the ann pass loads the whole 60,000-row training set.
        assert self._floor(60000, 784, max(m_values)) < self.GIB

    def test_the_main_profile_does_not(self):
        """The finding that changes what can be run: glove-100 at M=32 exceeds
        the default on the floor alone, so the load is guaranteed to stop.

        The row count is this repo's own, quoted at harness/datasets.py:492.
        """
        datasets = open(os.path.join(VB_ROOT, "harness", "datasets.py")).read()
        assert "1,183,514 training rows" in datasets
        assert self._floor(1183514, 100, 32) > self.GIB
        # At M=16 the floor fits but the MTR's own 2x ceiling does not, which is
        # why §3 calls that configuration a coin-flip rather than safe.
        assert self._floor(1183514, 100, 16) < self.GIB
        assert 2 * self._floor(1183514, 100, 16) > self.GIB
