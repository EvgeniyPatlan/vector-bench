"""What the Percona image takes from the vendor's own build, and what it must not.

WHY THIS FILE EXISTS
--------------------
2026-10-14. After five configure failures and three failed compiles, the
natural proposal was made: stop hand-deriving cmake flags and just use
`build-ps/build-binary.sh`, the vendor's own build script, since it
demonstrably works. It was audited against the pinned tree rather than
dismissed, and the audit is written up as note 5 in
`docker/percona/Dockerfile`. The conclusion was that the script cannot be
used, for reasons that are about the measurement rather than convenience:

  * it sets no `-march` on x86_64 -- its flags come from `rpm --eval
    %optflags` behind `if [ -n "$(command -v rpm)" ]`, and this image is
    noble, so CFLAGS would end up empty while four other source-built engines
    carry `x86-64-v3`. That is the first invariant in CLAUDE.md.
  * it enables PGO (a three-pass build) and LTO, so Percona would be the only
    engine in the study carrying either.
  * it enables Percona telemetry, which is periodic background work inside a
    server whose CPU and wall time this benchmark charges to query execution.
  * it needs the rocksdb submodule and a prebuilt V8 tree, and forces an
    in-source build.

Two of its flags are dead on this tree as well: `extra/libevent` and
`plugin/innodb_memcached` are both absent at this commit, and cmake warns
about unused `-D` variables rather than failing, so copying them would be
silent noise.

None of that is self-enforcing. The next person to hit a failed Percona
compile will reach for the same script, and the cost of the mistake is not a
broken build -- it is a build that SUCCEEDS and silently publishes a Percona
number produced with LTO, PGO and no `-march`, against six engines without
them. That failure has no symptom. This file is the symptom.

The flags this image DOES take from the vendor are asserted here too, because
a guard that only forbids is satisfied by taking nothing.
"""

from __future__ import annotations

import os
import re

import pytest
import yaml

VB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERCONA_YML = os.path.join(VB_ROOT, "config", "engines", "percona.yml")
PERCONA_DOCKERFILE = os.path.join(VB_ROOT, "docker", "percona", "Dockerfile")

CMAKE_ANCHOR = "cmake /src/source"
REGION_END_ANCHOR = "make -j"

# Every one of these is passed by build-ps/build-binary.sh at the pinned
# commit. The comment is the reason it may not be passed here; note 5 in the
# Dockerfile carries the line number in the vendor script for each.
FORBIDDEN_FLAGS = {
    "WITH_LTO": "link-time optimization; tree default is OFF and no other "
                "engine has it",
    "FPROFILE_GENERATE": "profile-guided optimization, first pass",
    "FPROFILE_USE": "profile-guided optimization, second pass",
    "WITH_PERCONA_TELEMETRY": "background work charged to query time; tree "
                              "default is OFF",
    "WITH_JS_LANG": "needs a prebuilt V8 tree",
    "V8_INCLUDE_DIR": "needs a prebuilt V8 tree",
    "V8_LIB_DIR": "needs a prebuilt V8 tree",
    "FORCE_INSOURCE_BUILD": "note 1: this image builds out of source",
    "WITH_PGO": "profile-guided optimization",
}

# Passed by the vendor script and meaningless on this tree: the directory and
# the plugin each one governs are absent at the pinned commit.
DEAD_FLAGS = ("WITH_LIBEVENT", "WITH_INNODB_MEMCACHED")

# Taken from the vendor: `bundled` for the three third-party libraries where
# this tree offers the choice. Asserted so that "forbid everything" cannot
# satisfy this file.
BUNDLED_LIBRARIES = ("WITH_CURL", "WITH_ZLIB", "WITH_ZSTD")

# Installed by the vendor's install_deps() and deliberately not taken: their
# packaging, test and documentation pipeline. A wholesale copy of that list
# brings these, and each one is a few hundred MB in an image that ships to a
# benchmark host.
VENDOR_ONLY_PACKAGES = (
    "valgrind", "gdb", "doxygen", "doxygen-gui", "graphviz", "devscripts",
    "debhelper", "fakeroot", "ccache", "patchelf", "python3-mysqldb",
    "libeatmydata", "dh-apparmor", "copyright-update",
)


# ---------------------------------------------------------------------------
# Predicates, as functions of the text
# ---------------------------------------------------------------------------

def cmake_region(text: str) -> str:
    """Only the cmake invocation.

    Bounded at both ends because note 5 quotes every forbidden flag by name in
    prose. A whole-file scan would fail on the documentation that explains why
    the flags are absent -- which is the one way this file could be worse than
    nothing.
    """
    if CMAKE_ANCHOR not in text:
        return ""
    after = text.split(CMAKE_ANCHOR, 1)[1]
    if REGION_END_ANCHOR not in after:
        return ""
    return after.split(REGION_END_ANCHOR, 1)[0]


def region_flags(text: str) -> list[str]:
    return re.findall(r"^\s*(-D\S+)", cmake_region(text), re.MULTILINE)


def cflags_line(text: str) -> str:
    """The builder's `export CFLAGS=` line, where -march lives."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("export CFLAGS="):
            return stripped.rstrip("\\").strip().rstrip(";").strip()
    return ""


def apt_packages(text: str) -> frozenset[str]:
    """Package names from every `apt-get install` in the file.

    Deliberately NOT per stage: this file only ever asks whether a package is
    absent everywhere, and a vendor-only package is as wrong in the runtime
    stage as in the builder. `tests/test_e4_percona_build_flags.py` owns the
    per-stage questions.
    """
    found: set[str] = set()
    for body in re.findall(r"apt-get install((?:[^\n]*\\\n)*[^\n]*)", text):
        for token in body.replace("\\", " ").split():
            if not token.startswith("-") and token not in {"&&", "install"}:
                found.add(token)
    return frozenset(found)


@pytest.fixture(scope="module")
def dockerfile_text() -> str:
    with open(PERCONA_DOCKERFILE) as handle:
        return handle.read()


@pytest.fixture(scope="module")
def engine_config() -> dict:
    with open(PERCONA_YML) as handle:
        return yaml.safe_load(handle)


# ---------------------------------------------------------------------------
# Non-vacuity first: every assertion below is "X is absent", and an empty
# input satisfies all of them.
# ---------------------------------------------------------------------------

class TestTheInputsAreActuallyThere:

    def test_both_files_exist(self):
        assert os.path.isfile(PERCONA_YML), PERCONA_YML
        assert os.path.isfile(PERCONA_DOCKERFILE), PERCONA_DOCKERFILE

    def test_the_cmake_region_is_not_empty(self, dockerfile_text):
        assert cmake_region(dockerfile_text).strip(), (
            f"the region between {CMAKE_ANCHOR!r} and {REGION_END_ANCHOR!r} "
            "is blank, so every absence asserted below is vacuous")

    def test_the_region_holds_a_plausible_number_of_flags(self,
                                                          dockerfile_text):
        flags = region_flags(dockerfile_text)
        assert len(flags) >= 10, f"only {len(flags)} flags parsed: {flags}"

    def test_the_cflags_line_was_found(self, dockerfile_text):
        assert cflags_line(dockerfile_text), (
            "no `export CFLAGS=` line in the builder stage; the -march "
            "assertions below cannot see anything")

    def test_the_apt_list_was_found(self, dockerfile_text):
        packages = apt_packages(dockerfile_text)
        assert len(packages) >= 10, sorted(packages)
        assert "build-essential" in packages, sorted(packages)


# ---------------------------------------------------------------------------
# The invariant the vendor script would break
# ---------------------------------------------------------------------------

class TestMarchIsSetAndMatchesTheConfig:
    """CLAUDE.md's first invariant, for the one engine most likely to lose it.

    build-binary.sh would leave CFLAGS empty on this image. An empty CFLAGS
    does not fail: it produces a working server compiled without the SIMD
    baseline every other source-built engine has.
    """

    def test_cflags_carries_an_march(self, dockerfile_text):
        line = cflags_line(dockerfile_text)
        assert "-march=" in line, (
            f"the builder's CFLAGS has no -march. {line!r}")

    def test_the_march_is_the_build_arg_and_not_a_literal(self,
                                                          dockerfile_text):
        """`-march=${MARCH}`, so one ARG governs it and `--march native`
        reaches the compiler. A literal here silently ignores the flag."""
        line = cflags_line(dockerfile_text)
        assert "-march=${MARCH}" in line, (
            f"-march is not driven by the MARCH build arg: {line!r}")

    def test_the_arg_default_matches_the_engine_config(self, dockerfile_text,
                                                       engine_config):
        declared = str((engine_config.get("build") or {}).get("march") or "")
        assert declared, "config/engines/percona.yml has no build.march"
        match = re.search(r"^ARG MARCH=(\S+)", dockerfile_text, re.MULTILINE)
        assert match is not None, "no `ARG MARCH=` in the Dockerfile"
        assert match.group(1) == declared, (
            f"Dockerfile ARG MARCH={match.group(1)} but percona.yml declares "
            f"build.march: {declared}. The orchestrator passes the config "
            f"value, so the two disagreeing means the ARG default is dead "
            f"text -- and dead text here is how an -march mismatch hides.")

    def test_no_second_march_contradicts_it(self, dockerfile_text):
        """One -march in the file's build commands, not two."""
        in_commands = [
            line for line in dockerfile_text.splitlines()
            if "-march=" in line and not line.lstrip().startswith("#")
        ]
        assert len(in_commands) == 1, (
            f"-march appears in {len(in_commands)} non-comment lines: "
            f"{in_commands}")


# ---------------------------------------------------------------------------
# What may not be taken from the vendor
# ---------------------------------------------------------------------------

class TestTheDisqualifyingVendorFlagsAreAbsent:

    @pytest.mark.parametrize("flag", sorted(FORBIDDEN_FLAGS))
    def test_the_flag_is_not_passed_to_cmake(self, flag, dockerfile_text):
        region = cmake_region(dockerfile_text)
        assert flag not in region, (
            f"-D{flag} reached the cmake invocation. "
            f"{FORBIDDEN_FLAGS[flag]}. build-ps/build-binary.sh passes it; "
            f"see note 5 in the Dockerfile for why this image must not.")

    def test_rocksdb_is_off_and_not_merely_absent(self, dockerfile_text):
        """The one vendor flag whose *name* has to be present here.

        The submodule is not in the tarball, so `-DWITH_ROCKSDB=OFF` is
        required (note 2) while the vendor's `=ON` is forbidden. Asserting
        absence of the name would contradict note 2, so the value is what is
        asserted.
        """
        flags = region_flags(dockerfile_text)
        assert "-DWITH_ROCKSDB=OFF" in flags, flags
        assert "-DWITH_ROCKSDB=ON" not in flags, flags

    @pytest.mark.parametrize("flag", DEAD_FLAGS)
    def test_the_dead_vendor_flags_are_absent(self, flag, dockerfile_text):
        """Neither symbol exists on the pinned tree.

        `extra/libevent` and `plugin/innodb_memcached` are both absent at the
        pinned commit, and neither name appears in the top-level
        CMakeLists.txt or under cmake/. cmake reports an unused `-D` as a
        warning, so these would not break the build -- they would sit in the
        flag list looking load-bearing.
        """
        assert flag not in cmake_region(dockerfile_text), (
            f"-D{flag} is passed, and this tree does not define it. It comes "
            f"from build-ps/build-binary.sh, where it is also dead.")


# ---------------------------------------------------------------------------
# What is taken from the vendor -- so "forbid everything" fails too
# ---------------------------------------------------------------------------

class TestTheVendorChoicesThatAreTakenAreTaken:

    @pytest.mark.parametrize("flag", BUNDLED_LIBRARIES)
    def test_the_library_is_bundled(self, flag, dockerfile_text):
        """Percona's own build uses `bundled` for all three, and every one of
        the five configure failures on 2026-10-14 was a missing system
        library. This tree offers the choice for exactly these three."""
        flags = region_flags(dockerfile_text)
        assert f"-D{flag}=bundled" in flags, (
            f"-D{flag} is not `bundled`. Flags: {flags}")
        assert f"-D{flag}=system" not in flags, flags

    @pytest.mark.parametrize("flag", BUNDLED_LIBRARIES)
    def test_the_engine_config_mirrors_it(self, flag, engine_config):
        """`tests/test_e4_percona_build_flags.py` compares the whole lists;
        this checks the three values this file is about, so a drift here fails
        with a message about the vendor alignment rather than about set
        difference."""
        declared = (engine_config.get("build") or {}).get("cmake_flags") or []
        assert f"-D{flag}=bundled" in declared, declared


class TestTheVendorPackagingPipelineIsNotInstalled:

    @pytest.mark.parametrize("package", VENDOR_ONLY_PACKAGES)
    def test_the_package_is_absent(self, package, dockerfile_text):
        assert package not in apt_packages(dockerfile_text), (
            f"{package} is installed. install_deps() in "
            f"build-ps/percona-server-9.0_builder.sh installs it for "
            f"Percona's packaging and test pipeline; nothing in this image "
            f"runs that pipeline.")


# ---------------------------------------------------------------------------
# Mutation probes
# ---------------------------------------------------------------------------
#
# Every predicate above is an absence check, which is the kind most likely to
# be dead. Each probe corrupts the text, asserts the corruption applied, and
# reports the honest value beside the mutant one.

class TestTheProbesDiscriminate:

    def test_adding_lto_is_seen(self, dockerfile_text):
        mutant = dockerfile_text.replace(
            "      -DWITH_ROCKSDB=OFF \\",
            "      -DWITH_ROCKSDB=OFF \\\n      -DWITH_LTO=ON \\")
        assert mutant != dockerfile_text, "mutation did not apply"
        assert "WITH_LTO" not in cmake_region(dockerfile_text)
        assert "WITH_LTO" in cmake_region(mutant)

    def test_naming_lto_in_a_comment_is_not_seen(self, dockerfile_text):
        """The false positive a whole-file grep would have.

        Note 5 names every forbidden flag in prose. If the region bound broke,
        this file would fail on its own documentation -- so the bound is
        asserted from the other side too.
        """
        assert "WITH_LTO" in dockerfile_text, (
            "note 5 no longer names WITH_LTO, so the comment this probe "
            "guards against mis-reading is gone")
        assert "WITH_LTO" not in cmake_region(dockerfile_text)

    def test_losing_march_is_seen(self, dockerfile_text):
        mutant = dockerfile_text.replace(" -march=${MARCH}", "")
        assert mutant != dockerfile_text, "mutation did not apply"
        honest, mutated = cflags_line(dockerfile_text), cflags_line(mutant)
        assert honest != mutated, (honest, mutated)
        assert "-march=" in honest
        assert "-march=" not in mutated

    def test_pinning_march_to_a_literal_is_seen(self, dockerfile_text):
        """The subtle one: a literal still carries an -march, so the first
        assertion in TestMarchIsSetAndMatchesTheConfig would pass. `--march
        native` would then be accepted and ignored."""
        mutant = dockerfile_text.replace("-march=${MARCH}", "-march=x86-64-v3")
        assert mutant != dockerfile_text, "mutation did not apply"
        assert "-march=" in cflags_line(mutant)
        assert "-march=${MARCH}" in cflags_line(dockerfile_text)
        assert "-march=${MARCH}" not in cflags_line(mutant)

    def test_switching_zlib_back_to_system_is_seen(self, dockerfile_text):
        mutant = dockerfile_text.replace("-DWITH_ZLIB=bundled",
                                         "-DWITH_ZLIB=system")
        assert mutant != dockerfile_text, "mutation did not apply"
        assert "-DWITH_ZLIB=bundled" in region_flags(dockerfile_text)
        assert "-DWITH_ZLIB=bundled" not in region_flags(mutant)
        assert "-DWITH_ZLIB=system" in region_flags(mutant)

    def test_installing_the_vendor_test_pipeline_is_seen(self,
                                                         dockerfile_text):
        mutant = dockerfile_text.replace(
            "      perl gawk unzip curl",
            "      perl gawk unzip curl valgrind gdb doxygen patchelf")
        assert mutant != dockerfile_text, "mutation did not apply"
        honest, mutated = apt_packages(dockerfile_text), apt_packages(mutant)
        assert "valgrind" not in honest
        assert "valgrind" in mutated
        assert len(mutated) == len(honest) + 4, (len(honest), len(mutated))

    def test_a_region_with_no_anchor_reads_as_empty_not_as_everything(
            self, dockerfile_text):
        """The failure mode that would make every absence above vacuous while
        the file stayed green: no anchor, so take the rest of the file."""
        mutant = dockerfile_text.replace(CMAKE_ANCHOR, "cmake /src/elsewhere")
        assert mutant != dockerfile_text, "mutation did not apply"
        assert cmake_region(dockerfile_text).strip() != ""
        assert cmake_region(mutant) == ""
        assert region_flags(mutant) == []
