"""Why both MySQL-family builders pass `-DWITH_PACKAGE_FLAGS=OFF` and
`-DCMAKE_C_FLAGS=` on the cmake COMMAND LINE.

WHY THIS FILE EXISTS
--------------------
2026-10-15. `docker/percona/Dockerfile` and `docker/alisql/Dockerfile` both
build a MySQL-family tree by exporting `CFLAGS`/`CXXFLAGS` and running
`cmake`. On a Debian or Ubuntu host that tree THROWS THOSE FLAGS AWAY and
substitutes `dpkg-buildflags` output. The chain, READ from each engine's
pinned tree except where a step says MEASURED:

  1. Percona `CMakeLists.txt:750` / AliSQL `CMakeLists.txt:785` -- for
     `LINUX_STANDALONE AND (LINUX_DEB_PLATFORM OR LINUX_RPM_PLATFORM)` the
     tree offers a `WITH_PACKAGE_FLAGS` option, and Percona `:758` / AliSQL
     `:793` default it **ON** when the build type is `RELWITHDEBINFO`. Every
     other build type defaults it OFF. Both engines build `RelWithDebInfo`.
  2. Percona `:769-770` / AliSQL `:804-805` -- the escape hatch is
     `IF(MY_COMPILER_IS_GNU AND WITH_PACKAGE_FLAGS AND NOT
     INITIAL_CMAKE_C_FLAGS AND NOT INITIAL_CMAKE_CXX_FLAGS)`.
  3. Percona `:313-322` / AliSQL `:303-306` -- `INITIAL_CMAKE_C_FLAGS` is
     captured from `IF(DEFINED CMAKE_C_FLAGS)`, and that block runs BEFORE
     `PROJECT()` (Percona `:326` says why it must: "Cannot
     INCLUDE(CMakeDetermineSystem) prior to PROJECT initialization below").
     The tree's own comment calls it a capture of values "form the CACHE or
     from the command line" -- the ENVIRONMENT is not in that list, and cmake
     does not read `$CFLAGS` until the C language is enabled, inside
     `PROJECT()`. MEASURED, with that snippet alone on cmake 3.28.3: an
     exported `CFLAGS` leaves `CMAKE_C_FLAGS` UNDEFINED at that point so the
     hatch stays OPEN, while `-DCMAKE_C_FLAGS=<non-empty>` on the command line
     defines it and the hatch CLOSES. An EMPTY command-line value does NOT
     close it: the capture expands unquoted, so `SET(INITIAL_CMAKE_C_FLAGS
     ${CMAKE_C_FLAGS})` over an empty value is `SET()` with zero arguments and
     UNSETS the variable rather than setting it to `""`. MEASURED: with
     `-DCMAKE_C_FLAGS=""`, `DEFINED CMAKE_C_FLAGS` is true but
     `DEFINED INITIAL_CMAKE_C_FLAGS` is false. `IF(NOT ...)` reads true either
     way, which is why the value and not just the symbol is asserted below.
  4. Percona `cmake/compile_flags.cmake:119-136` / AliSQL `:120-137` --
     `ADD_LINUX_DEB_FLAGS()`. Its four `SET(... PARENT_SCOPE)` calls sit
     INSIDE the `FOREACH(flag CPPFLAGS CFLAGS CXXFLAGS LDFLAGS)` (Percona
     `:120-135`, AliSQL `:121-136`), so they run four times and only the last
     iteration has every `GET_*` populated; the net effect is the final values.
     The one that matters is `SET(CMAKE_C_FLAGS "${GET_CFLAGS}
     ${GET_CPPFLAGS}" PARENT_SCOPE)` (Percona `:130`, AliSQL `:131`), with the
     CXX equivalent beside it: an OVERWRITE, not an append. The three
     LINKER_FLAGS variables are replaced from dpkg's LDFLAGS the same way
     (Percona `:132-134`, AliSQL `:133-135`), which is what makes the LTO real.
  5. Both builder stages install `build-essential`, which depends on
     `dpkg-dev`, which ships `/usr/bin/dpkg-buildflags`, so
     `MY_DPKG_BUILDFLAGS` is found and step 4 fires.

TWO CONSEQUENCES, AND THE SECOND IS WHY THIS FILE IS NOT ONLY ABOUT A BUILD
FAILURE.

Percona's build DIED on it: `-D_FORTIFY_SOURCE=3` out of `dpkg-buildflags
--get CPPFLAGS` against the `-D_FORTIFY_SOURCE=2` the tree prepends at
`CMakeLists.txt:1583-1588` is the logged
`<command-line>: error: "_FORTIFY_SOURCE" redefined`, fatal under libfido2's
`-pedantic-errors`. AliSQL's build did not fail, for two reasons that are each
sufficient and neither of which is "its tree states no level": the single live
`-D_FORTIFY_SOURCE=2` in its tree sits behind an option defaulted OFF, and it
compiles no libfido2 at all. `docker/alisql/Dockerfile` carries that derivation
with the grep behind it.

The second consequence applied to BOTH, in silence, and is exactly two things:
`-march=x86-64-v3` is dropped, and dpkg's `-flto=auto -ffat-lto-objects
-fstack-protector-strong -fstack-clash-protection -fcf-protection` are gained.
`/opt/<engine>/.march` is written from the `MARCH` build ARG, not from what
cmake honoured, so every `march` reader in this repo reports `x86-64-v3` either
way. CLAUDE.md's "Never mix -march between engines" is what that violates, and
it is the kind of breach that publishes a number rather than failing a build.

THE OPTIMISATION LEVEL IS NOT ONE OF THE TWO, and no assertion here should be
read as touching it. `-O3` in the exported CFLAGS has never reached the
compiler for either engine: the compile line is `CMAKE_C_FLAGS
CMAKE_C_FLAGS_<configuration>` and RelWithDebInfo's own value ends it with
`-O2 -g -DNDEBUG`. Both trees say so themselves (Percona
`CMakeLists.txt:2959-2969`, AliSQL `:2709-2719`) and name the remedy neither
Dockerfile passes. MEASURED on cmake 3.28.3 and gcc 13: the emitted command is
`cc -O3 -g -march=x86-64-v3 -O2 -g -DNDEBUG`, and `gcc -O3 -O2 -Q
--help=optimizers` is identical to `gcc -O2`. That the study mixes -O2 and -O3
across engines is a real and separate defect, outside this file's scope.

WHAT THESE TESTS CAN AND CANNOT ESTABLISH
-----------------------------------------
Nothing here builds an image, starts a server or runs a profile. These are
text and YAML guards over two Dockerfiles and two engine configs, plus ONE
measurement of the host's own `dpkg-buildflags`. They establish that the two
flags are passed and mirrored, that the configuration the defect needs is
still the configuration we have, and that this host's distro really does
disagree with the tree about `_FORTIFY_SOURCE`. They establish nothing about
either server.

THE VACUITY SHAPES THESE GUARDS ARE BUILT TO AVOID
--------------------------------------------------
1. **Prose read as arguments.** Both Dockerfiles now explain this defect at
   length ABOVE the cmake call and name `-DWITH_PACKAGE_FLAGS=OFF` in that
   prose. A whole-file search is satisfied by the documentation. The parser
   here drops comment lines first and is bounded to the invocation, and
   `TestTheProbesDiscriminate` deletes the real flag while leaving the prose
   in place to prove it.
2. **An unbounded region that reads as everything.** If the terminating
   `make` line were lost, a naive parser would swallow the rest of the file
   and every presence assertion would stay green over a bound that no longer
   exists. `cmake_invocation()` returns the EMPTY string in that case, and
   that is asserted from both sides.
3. **A flag that is present but inert.** `-DCMAKE_C_FLAGS=` with an empty
   value closes nothing (step 3, measured). So the value is asserted, not the
   symbol -- and specifically asserted to resolve to a string carrying
   `-march=`.
4. **A probe that cannot fail.** Every mutation below reports the honest
   verdict beside the mutant one and fails loudly if the two are identical.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from typing import NamedTuple

import pytest
import yaml

VB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: The two engines built by exporting CFLAGS into a MySQL-family cmake tree.
#:
#: `mariadb` and `mariadb123` are NOT here and must not be added, for two
#: independent reasons. They already pass
#: `-DCMAKE_C_FLAGS="-march=${MARCH}"` on the cmake command line
#: (`docker/mariadb/Dockerfile:42-43`; `mariadb123` shares that Dockerfile),
#: and MariaDB's tree carries none of this machinery: MEASURED by grep over
#: both extracted pins under `sources/`, `WITH_PACKAGE_FLAGS`,
#: `ADD_LINUX_DEB_FLAGS` and `dpkg-buildflags` occur ZERO times in either.
#: (MariaDB does have a `cmake/compile_flags.cmake`, which is why that
#: filename is not the test -- it is a different file with none of these
#: symbols in it.) Both pins do ask for `-D_FORTIFY_SOURCE=2`, at
#: `mariadb-11.8.8:CMakeLists.txt:290` and
#: `mariadb-12.3.2:CMakeLists.txt:303`, READ at both pins -- which is the
#: level the two engines here agree with.
#:
#: pgvector is a Makefile build; mongodb and valkey are installed from
#: packages and declare `march: none`.
ENGINES = ("percona", "alisql")

#: Both halves of the fix, and both are required. The first closes the option
#: (step 1); the second closes the guard (step 3). Either alone would do on
#: the trees as pinned -- both are passed because they fail in different
#: directions, which is the argument the Dockerfile notes make.
PACKAGE_FLAGS_OFF = "-DWITH_PACKAGE_FLAGS=OFF"
C_FLAGS_SYMBOL = "-DCMAKE_C_FLAGS"
CXX_FLAGS_SYMBOL = "-DCMAKE_CXX_FLAGS"

#: The build type that makes `WITH_PACKAGE_FLAGS` default ON. Spelled out
#: rather than derived so that a config changing its mind fails here with an
#: explanation instead of silently making these flags look unnecessary.
HAZARDOUS_BUILD_TYPE = "relwithdebinfo"

#: The package that drags `dpkg-dev` (and therefore `dpkg-buildflags`) into
#: the builder stage. Its presence is what makes step 5 fire; asserted so
#: that a reader who removes it can see that these flags became moot, rather
#: than discovering it from a changed binary.
DPKG_BUILDFLAGS_CARRIER = "build-essential"

#: The level the two trees ask for. The point of the measured test below is
#: that the DISTRO disagrees with this, whatever the distro's number happens
#: to be -- not that the distro's number is any particular value.
TREE_FORTIFY_LEVEL = "2"


def engine_paths(engine: str) -> tuple[str, str]:
    return (os.path.join(VB_ROOT, "docker", engine, "Dockerfile"),
            os.path.join(VB_ROOT, "config", "engines", f"{engine}.yml"))


# ---------------------------------------------------------------------------
# The parser, as a function of the text
# ---------------------------------------------------------------------------

def _commands(text: str) -> list[str]:
    """Non-comment lines only.

    Both Dockerfiles document this defect at length and name the flags in
    that prose, so this is the first thing that has to happen.
    """
    return [line for line in text.splitlines()
            if not line.lstrip().startswith("#")]


def cmake_invocation(text: str) -> str:
    """The builder's cmake invocation: the `cmake` line to the next `make`.

    Bounded at BOTH ends, and empty rather than greedy when the closing bound
    is missing -- vacuity shape 2. `cmake` is matched at the start of a
    (possibly indented) line so that `cmake` inside an apt list or a comment
    cannot open the region, and the terminator is `make` at the start of a
    line so that `make install` or `make -j` closes it.
    """
    lines = _commands(text)
    start = next((i for i, line in enumerate(lines)
                  if re.match(r"^\s*cmake\s", line)), None)
    if start is None:
        return ""
    end = next((j for j in range(start + 1, len(lines))
                if re.match(r"^\s*make\b", lines[j])), None)
    if end is None:
        return ""
    return "\n".join(lines[start:end])


def invocation_flags(text: str) -> list[str]:
    r"""Flags as passed: one per continuation line, anchored at line start.

    Anchored for the same reason test_e4 anchors: an unanchored `-D\S+` also
    matches `-D` tokens sitting inside another flag's value, and
    `-DCMAKE_C_FLAGS="$CFLAGS"` is about to carry one
    (`-D_FORTIFY_SOURCE=2`, by way of the exported variable).

    THE BLIND SPOT THIS BUYS, stated because the failure is misleading rather
    than dangerous: `^\s*(-D\S+)` takes only the FIRST `-D` on each line. An
    invocation reformatted to put two flags on one line would have the second
    one read as absent, and this file would report "not passed" for a build
    that is in fact protected. The direction is safe -- a false negative, never
    a false positive -- but the message would send a reader looking for a
    missing flag that is right there. If you reformat the cmake call, keep one
    flag per line or widen this regex.
    """
    return re.findall(r"^\s*(-D\S+)", cmake_invocation(text), re.MULTILINE)


def exported(text: str, name: str) -> str:
    """The value of `export NAME="..."` in the builder stage, or "".

    This is where `-march` lives, and `-DCMAKE_C_FLAGS="$CFLAGS"` is a
    reference to it rather than a copy -- so reading the flag's value alone
    would say nothing about what reaches the compiler.
    """
    for line in _commands(text):
        match = re.search(r'export %s="([^"]*)"' % re.escape(name), line)
        if match:
            return match.group(1)
    return ""


def flag_value(text: str, symbol: str) -> str:
    """The value passed for one flag symbol, with `$VAR` resolved.

    `-DCMAKE_C_FLAGS="$CFLAGS"` is resolved against the `export CFLAGS=` line
    in the same RUN, because an unresolved value cannot be asked whether it
    carries `-march=`.
    """
    for flag in invocation_flags(text):
        name, _, value = flag.partition("=")
        if name != symbol:
            continue
        value = value.strip('"').strip("'")
        match = re.fullmatch(r"\$\{?(\w+)\}?", value)
        return exported(text, match.group(1)) if match else value
    return ""


class Verdict(NamedTuple):
    """What the parser decides, with everything it decided it from.

    A bool alone would make the probes below report `False != True` and
    nothing else; the fields are carried so a failure says WHY.
    """

    flag_count: int
    package_flags_off: bool
    c_flags_on_command_line: bool
    cxx_flags_on_command_line: bool
    c_flags_value_is_empty: bool

    @property
    def protected(self) -> bool:
        """Both defences, and the command-line value must be non-empty.

        An empty `-DCMAKE_C_FLAGS=` leaves `INITIAL_CMAKE_C_FLAGS` UNSET --
        the capture expands unquoted, so zero arguments reach `SET()` -- and
        the tree's `IF(NOT ...)` reads that as true, so the hatch stays open.
        MEASURED on cmake 3.28.3; vacuity shape 3.
        """
        return (self.package_flags_off
                and self.c_flags_on_command_line
                and not self.c_flags_value_is_empty)


def protection(text: str) -> Verdict:
    """Is this Dockerfile's build protected from the DEB package-flags
    overwrite? The whole predicate, so the probes can mutate one input and
    read one answer."""
    flags = invocation_flags(text)
    symbols = [f.partition("=")[0] for f in flags]
    return Verdict(
        flag_count=len(flags),
        package_flags_off=PACKAGE_FLAGS_OFF in flags,
        c_flags_on_command_line=C_FLAGS_SYMBOL in symbols,
        cxx_flags_on_command_line=CXX_FLAGS_SYMBOL in symbols,
        c_flags_value_is_empty=(C_FLAGS_SYMBOL in symbols
                                and not flag_value(text, C_FLAGS_SYMBOL)),
    )


def apt_packages(text: str) -> frozenset[str]:
    """Package names from every `apt-get install` in the file."""
    found: set[str] = set()
    for body in re.findall(r"apt-get install((?:[^\n]*\\\n)*[^\n]*)", text):
        for token in body.replace("\\", " ").split():
            if not token.startswith("-") and token not in {"&&", "install"}:
                found.add(token)
    return frozenset(found)


@pytest.fixture(scope="module")
def sources() -> dict[str, tuple[str, dict]]:
    """Each engine's Dockerfile text and parsed config, read once."""
    out = {}
    for engine in ENGINES:
        dockerfile, config = engine_paths(engine)
        with open(dockerfile, encoding="utf-8") as handle:
            text = handle.read()
        with open(config, encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle) or {}
        out[engine] = (text, cfg)
    return out


# ---------------------------------------------------------------------------
# Non-vacuity: the parser found something before anything asks what
# ---------------------------------------------------------------------------

class TestTheInputsAreActuallyThere:

    @pytest.mark.parametrize("engine", ENGINES)
    def test_both_files_exist(self, engine):
        for path in engine_paths(engine):
            assert os.path.isfile(path), path

    @pytest.mark.parametrize("engine", ENGINES)
    def test_the_invocation_was_found(self, engine, sources):
        text, _ = sources[engine]
        assert cmake_invocation(text).strip(), (
            f"{engine}: no cmake invocation parsed, so every assertion below "
            f"is vacuous")

    @pytest.mark.parametrize("engine", ENGINES)
    def test_the_invocation_is_a_proper_slice_of_the_file(self, engine,
                                                          sources):
        """Bounded at both ends. A 'region' the size of the file has not
        bounded anything, and the notes above it name the flags in prose."""
        text, _ = sources[engine]
        region = cmake_invocation(text)
        assert len(region) < len(text), f"{engine}: the region is everything"
        assert "RUN set -eux" not in region, (
            f"{engine}: the region reaches back over the RUN header")

    @pytest.mark.parametrize("engine", ENGINES)
    def test_a_plausible_number_of_flags_parsed(self, engine, sources):
        text, _ = sources[engine]
        flags = invocation_flags(text)
        assert len(flags) >= 5, f"{engine}: only {len(flags)} parsed: {flags}"

    @pytest.mark.parametrize("engine", ENGINES)
    def test_the_config_declares_a_flag_list(self, engine, sources):
        _, cfg = sources[engine]
        declared = (cfg.get("build") or {}).get("cmake_flags") or []
        assert declared, (
            f"config/engines/{engine}.yml build.cmake_flags parsed to "
            f"nothing; the mirror assertions below would be vacuous")


# ---------------------------------------------------------------------------
# The rule: both flags, both engines, both copies
# ---------------------------------------------------------------------------

class TestBothEnginesPassBothDefences:
    """The flags themselves. Three in the Dockerfile, one mirrored in YAML.

    `-DCMAKE_C_FLAGS` and `-DCMAKE_CXX_FLAGS` are deliberately NOT mirrored
    into the configs: `scripts/build-images.sh:119` appends `build.cmake_flags`
    as `${CMAKE_FLAGS}` AFTER the Dockerfile's own flags, so a YAML copy would
    be LAST and would win -- carrying whatever literal the config held instead
    of the `$CFLAGS` the builder actually exported.
    `tests/test_e4_percona_build_flags.py` allowlists them for Percona as
    Dockerfile-only for that reason.
    """

    @pytest.mark.parametrize("engine", ENGINES)
    def test_the_dockerfile_switches_the_option_off(self, engine, sources):
        text, _ = sources[engine]
        flags = invocation_flags(text)
        assert PACKAGE_FLAGS_OFF in flags, (
            f"{engine}: {PACKAGE_FLAGS_OFF} is not passed. The tree defaults "
            f"WITH_PACKAGE_FLAGS ON for RelWithDebInfo on a DEB platform and "
            f"then overwrites CMAKE_C_FLAGS with dpkg-buildflags output. "
            f"Parsed: {flags}")

    @pytest.mark.parametrize("symbol", (C_FLAGS_SYMBOL, CXX_FLAGS_SYMBOL))
    @pytest.mark.parametrize("engine", ENGINES)
    def test_the_dockerfile_passes_the_flags_on_the_command_line(
            self, engine, symbol, sources):
        """On the COMMAND LINE, which is the whole point.

        Exporting the variable is not equivalent and not a fallback: the
        tree's capture runs before `PROJECT()`, and cmake has not read the
        environment by then (measured; see the module docstring, step 3).
        """
        text, _ = sources[engine]
        symbols = [f.partition("=")[0] for f in invocation_flags(text)]
        assert symbol in symbols, (
            f"{engine}: {symbol}= is not on the cmake command line, so "
            f"INITIAL_CMAKE_C_FLAGS is empty when the tree checks it and the "
            f"package-flags overwrite is not blocked. Parsed: {symbols}")

    @pytest.mark.parametrize("engine", ENGINES)
    def test_the_command_line_value_is_not_empty(self, engine, sources):
        """Vacuity shape 3, measured rather than reasoned.

        `-DCMAKE_C_FLAGS=` with nothing after it leaves CMAKE_C_FLAGS DEFINED
        but INITIAL_CMAKE_C_FLAGS UNSET: `SET(INITIAL_CMAKE_C_FLAGS
        ${CMAKE_C_FLAGS})` expands unquoted, and an empty expansion is zero
        arguments, which unsets rather than empties. cmake's `IF(NOT ...)`
        reads that as TRUE, so the hatch stays open and the overwrite happens
        anyway, with the flag sitting there looking load-bearing.
        """
        text, _ = sources[engine]
        value = flag_value(text, C_FLAGS_SYMBOL)
        assert value.strip(), (
            f"{engine}: {C_FLAGS_SYMBOL} resolves to {value!r}. An empty "
            f"value does not arm the guard (measured on cmake 3.28.3)")

    @pytest.mark.parametrize("engine", ENGINES)
    def test_the_config_mirrors_the_option(self, engine, sources):
        _, cfg = sources[engine]
        declared = (cfg.get("build") or {}).get("cmake_flags") or []
        assert PACKAGE_FLAGS_OFF in declared, (
            f"config/engines/{engine}.yml build.cmake_flags does not carry "
            f"{PACKAGE_FLAGS_OFF}. That list is the live path -- "
            f"scripts/build-images.sh passes it as the CMAKE_FLAGS build-arg "
            f"-- and both files document carrying the same values. "
            f"Declared: {declared}")

    @pytest.mark.parametrize("engine", ENGINES)
    def test_the_config_does_not_mirror_the_flag_strings(self, engine,
                                                         sources):
        """The one entry that must NOT be duplicated into the YAML.

        Appended last, a literal `-DCMAKE_C_FLAGS=` from the config would
        REPLACE the builder's exported CFLAGS -- `-march` included -- with
        whatever this file happened to say. That is the same defect wearing
        the fix's clothes.
        """
        _, cfg = sources[engine]
        declared = (cfg.get("build") or {}).get("cmake_flags") or []
        offenders = [f for f in declared
                     if f.partition("=")[0] in (C_FLAGS_SYMBOL,
                                                CXX_FLAGS_SYMBOL)]
        assert not offenders, (
            f"config/engines/{engine}.yml declares {offenders}. "
            f"scripts/build-images.sh:119 appends this list AFTER the "
            f"Dockerfile's flags, so this value would win and the builder's "
            f"own exported CFLAGS would be discarded -- reinstating the bug "
            f"through the fix's own flag.")

    @pytest.mark.parametrize("engine", ENGINES)
    def test_the_whole_predicate_says_protected(self, engine, sources):
        """The parser's own verdict, so the probes below mutate one thing and
        read one answer."""
        text, _ = sources[engine]
        verdict = protection(text)
        assert verdict.protected, f"{engine}: {verdict}"


# ---------------------------------------------------------------------------
# The shape the defect needs in order to bite
# ---------------------------------------------------------------------------

class TestTheConfigurationTheDefectNeedsIsStillOurs:
    """So that a future reader can see why these flags are here.

    Each of these is a precondition of the overwrite. If one of them changes,
    the flags stop being load-bearing -- and the right response is to delete
    them deliberately, with this file's failure as the argument, rather than
    to leave two cmake flags nobody can account for.
    """

    @pytest.mark.parametrize("engine", ENGINES)
    def test_the_build_type_is_the_one_that_defaults_the_option_on(
            self, engine, sources):
        """RELWITHDEBINFO is the ONLY type for which the trees default
        `WITH_PACKAGE_FLAGS` ON. The OFF branches are, per tree: any sanitizer
        (Percona `:753-754`, AliSQL `:788-789`), DEBUG or WITH_DEBUG (Percona
        `:755-756`, AliSQL `:790-791`) and everything else including Release
        and MinSizeRel (Percona `:759-760`, AliSQL `:794-795`). The ON branch
        is Percona `:757-758` / AliSQL `:792-793`."""
        _, cfg = sources[engine]
        declared = str((cfg.get("build") or {}).get("type") or "")
        assert declared.lower() == HAZARDOUS_BUILD_TYPE, (
            f"config/engines/{engine}.yml build.type is {declared!r}, not "
            f"RelWithDebInfo. The tree only defaults WITH_PACKAGE_FLAGS ON "
            f"for RELWITHDEBINFO, so this change's flags may now be dead "
            f"text -- decide that deliberately rather than leaving them.")

    @pytest.mark.parametrize("engine", ENGINES)
    def test_the_dockerfile_arg_default_agrees_with_the_config(self, engine,
                                                              sources):
        """A by-hand `docker build` uses the ARG default, and the framework
        passes the config value. The two disagreeing means one of the two
        paths is not the build this file is reasoning about."""
        text, cfg = sources[engine]
        declared = str((cfg.get("build") or {}).get("type") or "")
        match = re.search(r"^ARG BUILD_TYPE=(\S+)", text, re.MULTILINE)
        assert match is not None, f"{engine}: no `ARG BUILD_TYPE=`"
        assert match.group(1).lower() == declared.lower(), (
            f"{engine}: ARG BUILD_TYPE={match.group(1)} but the config says "
            f"{declared}")

    @pytest.mark.parametrize("engine", ENGINES)
    def test_the_builder_still_installs_the_dpkg_buildflags_carrier(
            self, engine, sources):
        """`build-essential` depends on `dpkg-dev`, which ships
        `/usr/bin/dpkg-buildflags` (both measured on the build host). Its
        presence is what makes `MY_DPKG_BUILDFLAGS` findable, and therefore
        what makes `ADD_LINUX_DEB_FLAGS()` fire at all."""
        text, _ = sources[engine]
        packages = apt_packages(text)
        assert len(packages) >= 5, f"{engine}: parsed {sorted(packages)}"
        assert DPKG_BUILDFLAGS_CARRIER in packages, (
            f"{engine}: {DPKG_BUILDFLAGS_CARRIER} is no longer installed as a "
            f"whole token, so dpkg-buildflags may be absent and these flags "
            f"moot. Installed: {sorted(packages)}")


# ---------------------------------------------------------------------------
# -march must survive, and `.march` must not be the only evidence of it
# ---------------------------------------------------------------------------

class TestTheMarchIsInTheStringThatReachesCmake:
    """The consequence with no symptom, guarded where it can be seen.

    `echo "${MARCH}" > /opt/<engine>/.march` writes the build ARG. It is
    written whether or not cmake honoured the flag, so it is evidence of
    INTENT and not of codegen -- and before this change it was the only
    evidence either way. What these assert is that `-march=` is in the string
    now passed to `-DCMAKE_C_FLAGS`, which is the copy the tree cannot
    discard.
    """

    @pytest.mark.parametrize("engine", ENGINES)
    def test_the_resolved_c_flags_carry_an_march(self, engine, sources):
        text, _ = sources[engine]
        value = flag_value(text, C_FLAGS_SYMBOL)
        assert "-march=" in value, (
            f"{engine}: the string passed to {C_FLAGS_SYMBOL} has no -march. "
            f"/opt/{engine}/.march would still say x86-64-v3, because it is "
            f"written from the build ARG. Resolved value: {value!r}")

    @pytest.mark.parametrize("engine", ENGINES)
    def test_the_march_is_the_build_arg_and_not_a_literal(self, engine,
                                                          sources):
        text, _ = sources[engine]
        value = flag_value(text, C_FLAGS_SYMBOL)
        assert "-march=${MARCH}" in value, (
            f"{engine}: -march is not driven by the MARCH build arg, so "
            f"`--march native` would be accepted and ignored: {value!r}")

    @pytest.mark.parametrize("engine", ENGINES)
    def test_the_command_line_copy_and_the_exported_copy_are_the_same_string(
            self, engine, sources):
        """`-DCMAKE_C_FLAGS="$CFLAGS"`, a reference and not a second literal.

        The export stays because sub-builds that are not this cmake project
        read it from the environment. Two literals would be two places to
        edit, and the Dockerfile notes promise they cannot drift.
        """
        text, _ = sources[engine]
        raw = [f for f in invocation_flags(text)
               if f.partition("=")[0] == C_FLAGS_SYMBOL]
        assert raw, f"{engine}: {C_FLAGS_SYMBOL} not passed"
        assert raw[0].partition("=")[2].strip('"') == "$CFLAGS", (
            f"{engine}: {raw[0]} is a literal rather than a reference to the "
            f"exported CFLAGS; the two copies can now drift")

    @pytest.mark.parametrize("engine", ENGINES)
    def test_the_march_file_is_still_written_from_the_arg(self, engine,
                                                          sources):
        """Not a complaint -- a statement of what that file is, asserted so
        that nobody reads it as a measurement. It is the ARG, which is why the
        assertions above exist."""
        text, _ = sources[engine]
        commands = "\n".join(_commands(text))
        assert f'echo "${{MARCH}}" > /opt/{engine}/.march' in commands, (
            f"{engine}: the .march file is no longer written from the MARCH "
            f"ARG; if it became a measurement, say so and revisit this class")


# ---------------------------------------------------------------------------
# The one measurement: does this host's distro really disagree with the tree?
# ---------------------------------------------------------------------------

DPKG_BUILDFLAGS = shutil.which("dpkg-buildflags")


@pytest.mark.skipif(
    DPKG_BUILDFLAGS is None,
    reason="dpkg-buildflags is not on PATH, so the level this distro would "
           "substitute cannot be read here. It comes from dpkg-dev, which "
           "the documented host prerequisites do not include -- the IMAGES "
           "have it, via build-essential.")
class TestThisHostsDistroDisagreesWithTheTree:

    @pytest.fixture(scope="class")
    def cppflags(self) -> str:
        done = subprocess.run([DPKG_BUILDFLAGS, "--get", "CPPFLAGS"],
                              capture_output=True, text=True, timeout=60)
        assert done.returncode == 0, done.stderr
        return done.stdout.strip()

    def test_it_names_a_fortify_level_at_all(self, cppflags):
        """Non-vacuity: if the distro named none, the test below would be
        asserting about a string that is not there."""
        assert "_FORTIFY_SOURCE" in cppflags, (
            f"dpkg-buildflags --get CPPFLAGS names no _FORTIFY_SOURCE on this "
            f"host, so the redefinition this change removes could not arise "
            f"here. It read: {cppflags!r}")

    def test_the_level_is_not_the_one_the_tree_asks_for(self, cppflags):
        """THE POINT IS THE DISAGREEMENT, NOT THE NUMBER.

        On the host this was written on the value is 3 and the tree asks for
        2, which is the `"_FORTIFY_SOURCE" redefined` error the Percona build
        died with. A distro that moved to 4 would be the same defect; a distro
        that moved to 2 would mean this particular symptom stopped, which is
        worth failing on so it gets read rather than assumed.
        """
        found = re.search(r"-D_FORTIFY_SOURCE=(\S+)", cppflags)
        assert found is not None, (
            f"_FORTIFY_SOURCE is named but not with a -D=<level>: "
            f"{cppflags!r}")
        level = found.group(1)
        assert level != TREE_FORTIFY_LEVEL, (
            f"dpkg-buildflags --get CPPFLAGS on this host says "
            f"_FORTIFY_SOURCE={level}, which AGREES with the level the trees "
            f"prepend ({TREE_FORTIFY_LEVEL}). The overwrite would then be "
            f"silent rather than fatal -- consequence (b), the dropped "
            f"-march, is unchanged and these flags are still required, but "
            f"the Dockerfile notes' account of the build FAILURE no longer "
            f"describes this host. Read it before editing. Full CPPFLAGS: "
            f"{cppflags!r}")

    def test_the_measurement_is_reported_and_not_merely_asserted(self,
                                                                 cppflags):
        """A test that only passes tells the next reader nothing about what
        the machine actually said."""
        found = re.search(r"-D_FORTIFY_SOURCE=(\S+)", cppflags)
        print(f"\nMEASURED dpkg-buildflags --get CPPFLAGS: {cppflags!r}\n"
              f"MEASURED _FORTIFY_SOURCE level here: "
              f"{found.group(1) if found else '<none>'} "
              f"(the trees ask for {TREE_FORTIFY_LEVEL})")
        assert cppflags


# ---------------------------------------------------------------------------
# Mutation probes
# ---------------------------------------------------------------------------
#
# Every assertion above is a presence check over text, which is the kind most
# likely to be satisfied by documentation or by a parser that stopped
# bounding. Each probe below RUNTIME-PATCHES the real file on disk, re-reads
# it through the same `protection()` the tests use, restores it, and reports
# the honest verdict beside the mutant one. An identical pair is a dead probe
# and says so rather than passing.
#
# The patch is applied to the real path on purpose: these predicates read
# files, and a probe that only ever sees an in-memory string cannot catch a
# path that stopped resolving. The restore is in a `finally` and is then
# asserted byte-identical; the gate runs this suite serially.

def _announce(what: str, report: str) -> None:
    """Print the pair. A probe that only passes tells the next reader nothing
    about what it compared, and `-q` hides the docstring that would have."""
    print(f"\nPROBE {what}:\n{report}")


def _patched(path: str, old: str, new: str):
    """Write `new` in place of `old`, yield the file's fresh verdict, restore.

    Returns `(honest, mutant, restored, original_bytes_came_back)`.
    """
    with open(path, "rb") as handle:
        original = handle.read()
    text = original.decode()
    assert text.count(old) == 1, (
        f"the mutation anchor is not unique in {path}: "
        f"{text.count(old)} occurrences of {old!r}")
    honest = protection(text)
    try:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text.replace(old, new, 1))
        with open(path, encoding="utf-8") as handle:
            mutant_text = handle.read()
        assert mutant_text != text, "the patch did not land on disk"
        mutant = protection(mutant_text)
    finally:
        with open(path, "wb") as handle:
            handle.write(original)
    with open(path, "rb") as handle:
        came_back = handle.read()
    return honest, mutant, protection(came_back.decode()), came_back == original


@pytest.mark.parametrize("engine", ENGINES)
class TestTheProbesDiscriminate:

    def test_deleting_the_option_flips_the_verdict(self, engine):
        """The flag is deleted from the ARGUMENTS while the Dockerfile's note
        keeps naming it in prose -- vacuity shape 1, as a mutation."""
        dockerfile, _ = engine_paths(engine)
        honest, mutant, restored, same = _patched(
            dockerfile, f"      {PACKAGE_FLAGS_OFF} \\\n", "")
        report = (f"{engine}\n  honest   = {honest}\n  mutant    = {mutant}\n"
                  f"  restored = {restored}")

        _announce("delete " + PACKAGE_FLAGS_OFF, report)
        assert honest != mutant, f"DEAD PROBE -- deleting the flag changed "\
                                 f"nothing.\n{report}"
        assert honest.protected, report
        assert not mutant.protected, (
            f"the verdict still says protected with {PACKAGE_FLAGS_OFF} "
            f"deleted from the invocation. The notes above the cmake call "
            f"still name it, so the parser is reading prose.\n{report}")
        assert PACKAGE_FLAGS_OFF in open(dockerfile).read(), (
            "the flag is not named anywhere in the restored file, so this "
            "probe no longer demonstrates that prose is excluded")
        assert same and restored == honest, (
            f"THE FILE WAS NOT RESTORED. A copy of the original is still in "
            f"git; `git checkout -- {dockerfile}` puts it back.\n{report}")

    def test_deleting_the_command_line_flags_flips_the_verdict(self, engine):
        dockerfile, _ = engine_paths(engine)
        honest, mutant, restored, same = _patched(
            dockerfile, '      -DCMAKE_C_FLAGS="$CFLAGS" \\\n', "")
        report = (f"{engine}\n  honest   = {honest}\n  mutant    = {mutant}\n"
                  f"  restored = {restored}")

        _announce("delete " + C_FLAGS_SYMBOL, report)
        assert honest != mutant, f"DEAD PROBE -- deleting "\
                                 f"{C_FLAGS_SYMBOL} changed nothing.\n{report}"
        assert honest.c_flags_on_command_line, report
        assert not mutant.c_flags_on_command_line, report
        assert not mutant.protected, report
        assert same and restored == honest, (
            f"THE FILE WAS NOT RESTORED. `git checkout -- {dockerfile}` puts "
            f"it back.\n{report}")

    def test_emptying_the_command_line_value_flips_the_verdict(self, engine):
        """The subtle one. The flag is still there, the symbol check still
        passes, and the guard is still open (measured).

        Anchored on the whole ARGUMENT line, indentation and continuation
        included, because both Dockerfiles now quote this flag in their prose
        as well -- which is the ambiguity every anchor here has to survive.
        """
        dockerfile, _ = engine_paths(engine)
        honest, mutant, restored, same = _patched(
            dockerfile,
            '      -DCMAKE_C_FLAGS="$CFLAGS" \\\n',
            '      -DCMAKE_C_FLAGS="" \\\n')
        report = (f"{engine}\n  honest   = {honest}\n  mutant    = {mutant}\n"
                  f"  restored = {restored}")

        _announce("empty the " + C_FLAGS_SYMBOL + " value", report)
        assert honest != mutant, f"DEAD PROBE -- emptying the value changed "\
                                 f"nothing.\n{report}"
        assert mutant.c_flags_on_command_line, (
            f"the mutant lost the symbol too, so this probe is not testing "
            f"the empty-value hazard it exists for.\n{report}")
        assert mutant.c_flags_value_is_empty and not mutant.protected, report
        assert not honest.c_flags_value_is_empty and honest.protected, report
        assert same and restored == honest, (
            f"THE FILE WAS NOT RESTORED. `git checkout -- {dockerfile}` puts "
            f"it back.\n{report}")

    def test_losing_the_march_from_the_export_is_seen(self, engine):
        """The command-line flag is a reference, so the march assertions have
        to follow it through the export. If they did not, deleting -march
        there would change nothing."""
        dockerfile, _ = engine_paths(engine)
        with open(dockerfile, encoding="utf-8") as handle:
            text = handle.read()
        mutant_text = text.replace(" -march=${MARCH}", "", 1)
        assert mutant_text != text, "mutation did not apply"
        honest_value = flag_value(text, C_FLAGS_SYMBOL)
        mutant_value = flag_value(mutant_text, C_FLAGS_SYMBOL)
        report = (f"{engine}\n  honest = {honest_value!r}\n"
                  f"  mutant = {mutant_value!r}")

        _announce("delete -march=${MARCH} from the export", report)
        assert honest_value != mutant_value, (
            f"DEAD PROBE -- the resolved value did not change, so the "
            f"-march assertions are not following the export.\n{report}")
        assert "-march=" in honest_value, report
        assert "-march=" not in mutant_value, report

    def test_an_unbounded_region_reads_as_empty_and_not_as_everything(self,
                                                                      engine):
        """Vacuity shape 2. If the terminating `make` were lost, a greedy
        parser would swallow the rest of the file and keep every presence
        assertion green over a bound that no longer exists."""
        dockerfile, _ = engine_paths(engine)
        with open(dockerfile, encoding="utf-8") as handle:
            text = handle.read()
        mutant_text = re.sub(r"^(\s*)make\b", r"\1gmake", text,
                             flags=re.MULTILINE)
        assert mutant_text != text, "mutation did not apply"
        honest, mutant = cmake_invocation(text), cmake_invocation(mutant_text)
        report = (f"{engine}\n  honest length = {len(honest)}\n"
                  f"  mutant length = {len(mutant)}")

        _announce("rename the closing `make` (unbounded region)", report)
        assert honest.strip(), report
        assert mutant == "", (
            f"with no closing `make` the region is {len(mutant)} characters "
            f"instead of empty, so it has swallowed the file below the "
            f"invocation.\n{report}")
        assert protection(mutant_text).flag_count == 0, report
        assert not protection(mutant_text).protected, report

    def test_the_prose_alone_does_not_satisfy_the_predicate(self, engine):
        """The other side of vacuity shape 1, asserted rather than trusted:
        the note names the flag, and that is not enough."""
        dockerfile, _ = engine_paths(engine)
        with open(dockerfile, encoding="utf-8") as handle:
            text = handle.read()
        commented = [line for line in text.splitlines()
                     if line.lstrip().startswith("#")
                     and PACKAGE_FLAGS_OFF in line]
        assert commented, (
            f"{engine}: the Dockerfile's note no longer names "
            f"{PACKAGE_FLAGS_OFF}, so this probe has nothing to discriminate "
            f"against -- and the reasoning for the flag has gone with it")
        prose_only = "\n".join(line for line in text.splitlines()
                               if line.lstrip().startswith("#"))
        _announce("the comment block alone",
                  f"{engine}\n  honest (whole file) = {protection(text)}\n"
                  f"  mutant (prose only) = {protection(prose_only)}\n"
                  f"  comment lines naming the flag = {len(commented)}")
        assert PACKAGE_FLAGS_OFF in prose_only
        assert protection(prose_only).flag_count == 0, (
            f"{engine}: the comment block alone parses to "
            f"{protection(prose_only)}; the parser is reading documentation")
