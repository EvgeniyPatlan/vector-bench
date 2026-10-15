"""Why the Percona builder names a _FORTIFY_SOURCE level of its own.

WHY THIS FILE EXISTS
--------------------
2026-10-15. Three builds of the Percona image died with
`make: *** [Makefile:166: all] Error 2` and nothing in the tail naming a
file. The serial retry added in the previous commit put the real error last:

    extra/libfido2/libfido2-1.15.0/src/.../aes256.c.o
    <command-line>: error: "_FORTIFY_SOURCE" redefined
    <command-line>: note: this is the location of the previous definition

Two definitions with different values were reaching the preprocessor: the
tree asks for 2 (`CMakeLists.txt:1586` at the pin prepends
`-DNDEBUG -D_FORTIFY_SOURCE=2` for the non-debug build types) and gcc on
noble supplies 3 of its own. Ubuntu's default stands down the moment it sees
a plain `-D` for the macro, so the builder now passes one, and every
definition that remains says 2.

WHAT THIS FILE IS FOR
---------------------
The flag looks like decoration. It is load-bearing, it has an exact value,
and the only symptom of getting it wrong is a build that fails two thirds of
the way in with a diagnostic the log's tail does not show. A reader tidying
"redundant hardening flags" out of CFLAGS would reintroduce the failure, and
a reader "modernising" 2 to 3 would reintroduce it in the same breath --
because 3 against the tree's 2 is the conflict, not a stronger version of
the fix.

It also holds the level itself. `_FORTIFY_SOURCE=3` adds dynamic
object-size checks that 2 does not, so the level is a codegen setting. Both
MariaDB pins ask for 2 plainly and get it
(`mariadb-11.8.8:CMakeLists.txt:290`, `mariadb-12.3.2:CMakeLists.txt:303`,
read at both pins) while this image was compiling at 3 -- an engine-to-engine
difference in generated code, which is the class of defect
`tests/test_config.py` exists for. The value here is not free to drift.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess

import pytest

VB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERCONA_DOCKERFILE = os.path.join(VB_ROOT, "docker", "percona", "Dockerfile")

# The level the pinned tree itself asks for. Spelled out rather than derived,
# so that a tree that changes its mind fails this file with an explanation
# instead of silently re-deciding the hardening level of a published number.
TREE_FORTIFY_LEVEL = "2"

# The form that reproduces the observed diagnostic. The driver's scan for a
# user-supplied level does not see a `-Wp,` define, so its own default is
# added as well and the two disagree. The site emitting this in the pinned
# tree was never located -- this is the reproduction, not a citation.
TREE_DEFINE_AS_SEEN = "-Wp,-D_FORTIFY_SOURCE=2"

# libfido2's dialect and strictness at the pin, which is what makes the
# warning fatal there and nowhere else:
# extra/libfido2/libfido2-1.15.0/CMakeLists.txt:329-330 adds -pedantic and
# -pedantic-errors; lines 88-104 build the dialect flags.
LIBFIDO2_FLAGS = [
    "-std=c99", "-pedantic", "-pedantic-errors",
    "-D_POSIX_C_SOURCE=200809L", "-D_BSD_SOURCE", "-D_GNU_SOURCE",
    "-D_DEFAULT_SOURCE",
]


@pytest.fixture(scope="module")
def dockerfile_text() -> str:
    with open(PERCONA_DOCKERFILE, encoding="utf-8") as handle:
        return handle.read()


def build_commands(text: str) -> list[str]:
    """Non-comment lines only. The notes quote flags in prose at length."""
    return [line for line in text.splitlines()
            if not line.lstrip().startswith("#")]


def cflags_line(text: str) -> str:
    for line in build_commands(text):
        if line.strip().startswith("export CFLAGS="):
            return line
    raise AssertionError(
        "no `export CFLAGS=` line in the builder stage; this file's anchor "
        "is stale or the Dockerfile was restructured")


def cflags_value(text: str) -> str:
    line = cflags_line(text)
    match = re.search(r'export CFLAGS="([^"]*)"', line)
    assert match, f"could not read the CFLAGS value out of {line!r}"
    return match.group(1)


def expanded_cflags(text: str) -> list[str]:
    """CFLAGS as the shell would have it, with the ARG defaults applied."""
    value = cflags_value(text)
    args = {name: default.strip().strip('"').strip("'")
            for name, default in
            re.findall(r"^ARG (\w+)=(.*)$", text, re.MULTILINE)}
    value = value.replace("${MARCH}", args.get("MARCH", "x86-64-v3"))
    value = value.replace("${EXTRA_CXXFLAGS}", args.get("EXTRA_CXXFLAGS", ""))
    assert "${" not in value, (
        f"an unexpanded variable is left in CFLAGS: {value!r}. This file "
        f"compiles with these flags, so it has to know all of them.")
    return value.split()


# ---------------------------------------------------------------------------
# Non-vacuity. Every assertion below is about one token in one line, and a
# file that failed to parse would satisfy several of them by having nothing.
# ---------------------------------------------------------------------------

class TestTheFileWasActuallyRead:

    def test_the_dockerfile_exists(self):
        assert os.path.isfile(PERCONA_DOCKERFILE), PERCONA_DOCKERFILE

    def test_it_has_a_builder_cflags_line(self, dockerfile_text):
        assert cflags_line(dockerfile_text).strip()

    def test_the_cflags_carry_the_other_build_flags_too(self, dockerfile_text):
        """If this line stopped being the real CFLAGS, the rest is theatre."""
        value = cflags_value(dockerfile_text)
        for expected in ("-O3", "-march="):
            assert expected in value, (
                f"{expected} is not in the CFLAGS this file is guarding, so "
                f"it is reading the wrong line: {value!r}")


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------

class TestTheLevelIsNamedAndIsTheTreesLevel:

    def test_cflags_names_a_fortify_level(self, dockerfile_text):
        value = cflags_value(dockerfile_text)
        assert "-D_FORTIFY_SOURCE=" in value, (
            "the builder's CFLAGS no longer names a _FORTIFY_SOURCE level. "
            "Ubuntu's gcc then supplies its own 3, the tree's own 2 arrives "
            "in a form the driver cannot see, and bundled libfido2 fails the "
            "build at 69% under -pedantic-errors. CFLAGS: " + repr(value))

    def test_the_level_is_the_one_the_tree_asks_for(self, dockerfile_text):
        value = cflags_value(dockerfile_text)
        assert f"-D_FORTIFY_SOURCE={TREE_FORTIFY_LEVEL}" in value, (
            f"the level is not {TREE_FORTIFY_LEVEL}. It has to AGREE with "
            f"the level the pinned tree prepends for its non-debug build "
            f"types; a different number here is the redefinition conflict "
            f"again, not a stronger fix. CFLAGS: {value!r}")

    def test_no_second_level_contradicts_it(self, dockerfile_text):
        """One level in the build commands, not two."""
        carrying = [line for line in build_commands(dockerfile_text)
                    if "-D_FORTIFY_SOURCE" in line]
        assert len(carrying) == 1, (
            f"_FORTIFY_SOURCE is set in {len(carrying)} non-comment lines; "
            f"two levels in one build is the defect itself: {carrying}")

    def test_the_level_is_not_left_valueless(self, dockerfile_text):
        """`-D_FORTIFY_SOURCE` with no value means 1, and 1 against the
        tree's 2 is a redefinition exactly like 3 is."""
        value = cflags_value(dockerfile_text)
        assert not re.search(r"-D_FORTIFY_SOURCE(?![=\w])", value), (
            f"-D_FORTIFY_SOURCE is passed without a value: {value!r}")

    def test_cplusplus_inherits_it(self, dockerfile_text):
        """absl is C++ and shows the same redefinition, so CXXFLAGS has to
        carry the level too. It does by deriving from CFLAGS; if that ever
        becomes a separate literal, this file is not guarding C++."""
        commands = "\n".join(build_commands(dockerfile_text))
        assert 'export CXXFLAGS="$CFLAGS"' in commands, (
            "CXXFLAGS no longer derives from CFLAGS, so the fortify level "
            "this file guards may not reach the C++ half of the build")

    def test_the_level_is_in_cflags_and_not_a_cmake_argument(
            self, dockerfile_text):
        """It belongs to the compiler, not to cmake. Kept out of the cmake
        region on purpose: tests/test_e4_percona_build_flags.py collects
        line-anchored `-D` flags there and compares them against
        config/engines/percona.yml, which does not list this one."""
        after = dockerfile_text.split("cmake /src/source", 1)
        assert len(after) == 2, "the cmake invocation anchor is stale"
        region = after[1].split("make -j", 1)[0]
        anchored = re.findall(r"^\s*(-D\S+)", region, re.MULTILINE)
        assert not [f for f in anchored if "_FORTIFY_SOURCE" in f], (
            f"the fortify level is being passed as a cmake argument: "
            f"{anchored}")


# ---------------------------------------------------------------------------
# The live probe: does the flag actually do what the comment says?
# ---------------------------------------------------------------------------
#
# The assertions above are text matching, which cannot tell a load-bearing
# flag from a cargo-culted one. This compiles the libfido2 recipe with the
# Dockerfile's own CFLAGS and reports the honest result beside the result
# with the flag removed. It needs a compiler whose default fortify level is
# the hazard, which the documented host prerequisites do not include -- so it
# skips rather than failing a correctly set up machine.

CC = shutil.which("cc") or shutil.which("gcc")


def _compile(flags: list[str], tmp_path) -> tuple[int, int]:
    """Returns (exit status, number of redefinition diagnostics)."""
    src = tmp_path / "fortify_probe.c"
    src.write_text("int main(void){return 0;}\n")
    done = subprocess.run(
        [CC, *flags, "-c", str(src), "-o", os.devnull],
        capture_output=True, text=True, timeout=120)
    return done.returncode, done.stderr.count('"_FORTIFY_SOURCE" redefined')


@pytest.mark.skipif(CC is None, reason="no host C compiler; see the module "
                                       "docstring -- docker/python3/pyyaml/"
                                       "git/curl are the documented host "
                                       "prerequisites")
class TestTheFlagIsLoadBearingOnThisHost:

    @pytest.fixture(scope="class")
    def host_default_level(self) -> str:
        done = subprocess.run([CC, "-O3", "-dM", "-E", "-"], input="",
                              capture_output=True, text=True, timeout=120)
        found = re.search(r"#define _FORTIFY_SOURCE (\S+)", done.stdout)
        return found.group(1) if found else ""

    @pytest.fixture(scope="class")
    def honest_flags(self, dockerfile_text) -> list[str]:
        return [*LIBFIDO2_FLAGS, *expanded_cflags(dockerfile_text),
                TREE_DEFINE_AS_SEEN]

    @pytest.fixture(scope="class")
    def probe(self, honest_flags, host_default_level, tmp_path_factory):
        """The pair, or a skip explaining why this host cannot produce one."""
        if not host_default_level:
            pytest.skip("this compiler supplies no default _FORTIFY_SOURCE, "
                        "so the conflict this flag removes cannot arise here "
                        "and the probe would prove nothing")
        if host_default_level == TREE_FORTIFY_LEVEL:
            pytest.skip(f"this compiler defaults to the tree's own level "
                        f"({host_default_level}), so there is nothing for it "
                        f"to disagree with")
        tmp_path = tmp_path_factory.mktemp("fortify")

        # -march=x86-64-v3 and the rest have to be accepted here, or a
        # failure below would be about the flags and not about fortify.
        usable, _ = _compile(
            [f for f in honest_flags if f != TREE_DEFINE_AS_SEEN], tmp_path)
        if usable != 0:
            pytest.skip("this host's compiler will not take the builder's "
                        "own CFLAGS, so the probe cannot run here")

        mutant_flags = [f for f in honest_flags
                        if not f.startswith("-D_FORTIFY_SOURCE=")]
        assert len(mutant_flags) == len(honest_flags) - 1, (
            "the mutation did not apply: no -D_FORTIFY_SOURCE= to remove")
        return (_compile(honest_flags, tmp_path),
                _compile(mutant_flags, tmp_path),
                host_default_level)

    def test_honest_and_mutant_differ(self, probe):
        honest, mutant, level = probe
        report = (f"honest (exit, redefinitions) = {honest}; "
                  f"mutant with the flag removed = {mutant}; "
                  f"host default level = {level}")

        assert honest != mutant, (
            f"DEAD PROBE -- removing the flag changed nothing, so this test "
            f"is measuring nothing on this host. {report}")
        assert honest == (0, 0), (
            f"the builder's own CFLAGS do not compile the libfido2 recipe "
            f"cleanly, which is the failure this flag exists to remove. "
            f"{report}")
        assert mutant[1] >= 1, (
            f"without the flag the redefinition did not appear, so the "
            f"mechanism recorded in the Dockerfile's note is not the one "
            f"acting on this host. {report}")
        assert mutant[0] != 0, (
            f"without the flag the compile still succeeded, so "
            f"-pedantic-errors is not making the redefinition fatal here. "
            f"{report}")
