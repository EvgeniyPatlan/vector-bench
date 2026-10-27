"""The Percona build inputs: the cmake flag list, its mirror, and the apt list.

WHY THIS FILE EXISTS
--------------------
2026-10-14. The first `./run-benchmark.sh build --engines percona` ever run
failed configure five times in a row, and the shape of the fix changed on the
way through. Each of the first two was an option left at a default that is
only correct on a machine carrying development packages this image does not
install, and each was "fixed" by switching the option off:

    CMake Error at CMakeLists.txt:2075 (MESSAGE):
      -DWITH_AUTHENTICATION_LDAP=ON, but missing system libraries
    CMake Error at CMakeLists.txt:2127 (MESSAGE):
      -DWITH_CURL=system, but missing system libraries.
      You can disable CURL with -DWITH_CURL=0

The third had no off switch at all -- `CMakeLists.txt:2224-2226` is
`IF(LINUX) MYSQL_CHECK_RPC() ENDIF()` and `cmake/rpc.cmake:106-110` is a bare
`IF(NOT RPC_INCLUDE_DIRS)` ending in FATAL_ERROR -- and the last two showed
why switching things off was the wrong strategy on this tree:

    CMake Error at plugin/auth_ldap/CMakeLists.txt:21 (MESSAGE):
      Required LDAP dev library not found.
    CMake Error at components/keyrings/keyring_vault/CMakeLists.txt:33:
      Not building Keyring Vault Component, could not find CURL library

The first of those is a PERCONA plugin keyed on a different symbol
(`WITH_PERCONA_AUTHENTICATION_LDAP`, CMakeLists.txt:2004-2011, default ON off
APPLE), which the LDAP flag never reached. The second was CAUSED by
`-DWITH_CURL=0`: the component is ON by default and tests `CURL_FOUND`, so
removing the target broke a Percona component that wanted it.

The fix for all of that landed separately, as commit 5fe565e (PR #22),
and this file is the half that commit did not carry: nothing in
`tests/` compared the two copies of the flag list, which is how they
could drift. What #22 changed, and what is asserted below:

  * the BUILDER stage's apt list gains thirteen packages, taken from the
    vendor's own `install_deps()` in
    `build-ps/percona-server-9.0_builder.sh` filtered to noble, which is this
    image's base. It is a superset and was not pruned -- see the Dockerfile's
    apt note. Eleven landed in 5fe565e; a8b8785 (PR #23) added the last two
    after a --no-cache rebuild with `libsasl2-dev` present STILL died at
    :2075, because `WARN_MISSING_SYSTEM_SASL` wants a SCRAM *plugin* from
    `libsasl2-modules` and not the dev package's headers. A check whose
    message says "development libraries" wanting a runtime plugin is the
    sharpest thing this episode turned up.
  * `-DWITH_CURL=bundled`, which is the value Percona's own
    `build-ps/build-binary.sh:304` uses. `=0` and `=system` are the two
    spellings that failed above.
  * NO authentication flag. Intermediate attempts at this fix passed
    `-DWITH_AUTHENTICATION_LDAP=OFF`, and #22 carries none: with libldap2-dev,
    libsasl2-dev and libkrb5-dev installed, the warnings that fed :2075 do not
    fire, so such a flag would gate nothing.
    `FORBIDDEN_FLAG_SYMBOLS` asserts it and its two obvious companions stay
    out, and says what would have to change to re-add one.

The package assertions are anchored to the BUILDER stage specifically. This
Dockerfile has three stages (builder, runtime, bench) and a whole-file grep
would be satisfied by a package sitting in the wrong one, which would not
build -- the mutation run for this file includes exactly that mutant.

THE FLAG LIST LIVES IN TWO PLACES THAT NOTHING KEPT IN STEP:

  * `config/engines/percona.yml` -> `build.cmake_flags`, which
    `scripts/build-images.sh:64 cmake_flags_for()` reads and passes as the
    CMAKE_FLAGS build-arg. This is the live path -- it is what
    `./run-benchmark.sh build` uses.
  * `docker/percona/Dockerfile`'s hardcoded list in the builder RUN, whose own
    comment says "Flags mirror config/engines/percona.yml". This is what a
    by-hand `docker build` uses, and docs/03-running-manually.md documents
    doing exactly that.

The Dockerfile appends `${CMAKE_FLAGS}` AFTER its own flags, so a value in the
YAML can confirm or override one above it but can never remove it. That makes
drift between the two lists silent in the worst direction: the framework path
and the by-hand path would compile different servers, and nothing would say so.
Before this file, `grep -rn cmake_flags tests/` returned nothing.

WHAT THESE TESTS CAN AND CANNOT ESTABLISH
-----------------------------------------
Nothing here builds anything, and no build of this image has ever succeeded.
These are agreement guards over two text files plus one YAML parse: that the
two copies of the flag list say the same thing, that the one flag this change
adds is in both with the vendor's value, that the removed flags stay removed,
and that the eleven packages are declared in the stage that compiles. They
establish nothing about Percona Server, and they cannot. Every claim about
what cmake does with these options is a file:line on commit
56124b03fdc5954e2401ab4d2c7bf7cb3a767158 plus the operator's own configure
output. Those readings were taken at the earlier pin and are carried forward
unverified: the base moved about four months at the re-pin to the `9.7`
series and no configure has been run against it. The comment above the builder's apt list in
`docker/percona/Dockerfile`, and the one beside `build.cmake_flags` in
`config/engines/percona.yml`, carry that reading at length.

THE VACUITY SHAPES THESE GUARDS ARE BUILT TO AVOID
-------------------------------------------------
Every one of these is live in this file rather than hypothetical, which is why
each is asserted rather than assumed:

1. **A symbol match that cannot tell a right value from a wrong one.** This is
   the big one here, because `-DWITH_CURL` has three spellings that all look
   like fixes: `=bundled` is correct, `=system` caused the
   CMakeLists.txt:2127 fatal and `=0` caused the keyring_vault:33 fatal. A
   guard written `assert "WITH_CURL" in text`, or one checking only that the
   symbol is passed, is satisfied by all three. `REQUIRED_FLAGS` therefore
   pins symbol AND value, and
   `test_matching_the_curl_symbol_alone_cannot_tell_right_from_wrong` asserts
   that the Dockerfile really does carry `-DWITH_CURL=0` in its prose while
   passing `=bundled` as an argument -- and that the correct value appears in
   the prose too, so even counting the right string over the whole file
   settles nothing.
2. **Prefix containment, where the needle prefixes a different flag.**
   `-DWITH_R` prefixes both `-DWITH_ROCKSDB=OFF` and `-DWITH_ROUTER=OFF`, and
   `-DWITH` prefixes `-DWITHOUT_COMPONENT_KEYRING_KMIP=1` as well as every
   `-DWITH_*`. So prefix matching cannot name a flag in this list; every
   comparison is exact equality on a parsed token, and
   `test_a_symbol_prefix_cannot_identify_a_flag` asserts the ambiguity.
   (A review that fed into this file offered `WITH_LDAP` as a substring of
   `WITH_AUTHENTICATION_LDAP`. It is not one -- after `WITH_` comes
   `AUTHENTICATION`. `test_the_reviewers_substring_example_is_not_the_real_trap`
   records that it was checked rather than believed. The shape is real; that
   instance of it is not.)
3. **A text search that finds the flag in the prose documenting it.** The
   Dockerfile header quotes all five cmake errors verbatim, so
   `-DWITH_AUTHENTICATION_LDAP=ON`, `-DWITH_CURL=0` and `-DWITH_CURL=system`
   -- every wrong value, two of them for the flag that IS passed -- are in the
   file, and so is the correct `-DWITH_CURL=bundled`. A whole-file grep
   therefore proves
   nothing about what is passed and can prove the opposite of the truth; it is
   also why the removed-flag assertion works on parsed tokens, since
   `assert "-DWITH_AUTHENTICATION_LDAP" not in dockerfile_text` would fail
   against a correct tree. Both parsers are bounded to the builder's cmake
   invocation and `TestTheProseAroundTheInvocationIsNotMistakenForArguments`
   asserts the bounding works.
4. **A package assertion that cannot see the wrong stage.** The Dockerfile has
   three stages and only `builder` runs cmake, so `libtirpc-dev` in `runtime`
   would satisfy a grep and still fail to build. The package tests parse
   per-stage apt lists, and the mutation run includes the move-to-runtime
   mutant specifically.
5. **A package name that is a prefix of another package name.**
   `libsasl2-modules` is a strict prefix of `libsasl2-modules-gssapi-mit`, and
   `libsasl2` prefixes both plus `libsasl2-dev`. A guard written
   `any(p.startswith("libsasl2-modules") ...)` is satisfied by the gssapi
   package alone -- leaving the SCRAM plugin missing, :2075 still firing, and
   the test green. Package assertions are whole-token membership of a parsed
   install list, each name separately, and
   `test_the_sasl_names_collide_by_prefix_in_both_directions` asserts the
   collision is real.
6. **A value vocabulary assumed to be `OFF`.** `-DWITH_CURL=bundled`,
   `-DWITHOUT_COMPONENT_KEYRING_KMIP=1`, `-DWITH_SSL=system` and
   `-DWITH_ZLIB=system` do not end in `OFF`. A check that split on `"=OFF"`
   would skip four flags including the one this change adds. Values are parsed
   with `partition("=")`, and `test_the_value_vocabulary_is_mixed` fails if
   that stops being necessary.
7. **An absence assertion that is really a policy in disguise.**
   `FORBIDDEN_FLAG_SYMBOLS` asserts three flags stay out. That is a claim
   about THIS pin -- their gates do not fire with the dev packages installed,
   and two of them route through a `WITH_INTERNAL` that is 0 because
   `internal/CMakeLists.txt` is absent from the published tree. It is not
   "these are wrong forever": the constant's comment says what would have to
   change to re-add one, so a future pin that needs them gets an argument to
   make rather than a red test to delete.
8. **`'"X"' and "Y"` evaluating to `"Y"`.** No assertion below is a
   conjunction of string literals. Expectations live in a dict and three
   frozensets, and the non-vacuity classes assert the parsers returned
   something before anything else asks what they returned.
"""

from __future__ import annotations

import os
import re

import pytest
import yaml

VB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERCONA_YML = os.path.join(VB_ROOT, "config", "engines", "percona.yml")
PERCONA_DOCKERFILE = os.path.join(VB_ROOT, "docker", "percona", "Dockerfile")

# The one flag this change adds, symbol -> value, spelled exactly as both
# copies must carry it. The value is Percona's own: build-ps/build-binary.sh:304
# is `-DWITH_CURL=bundled`, which satisfies
# components/keyrings/keyring_vault/CMakeLists.txt:32 by building extra/curl.
#
# THE VALUE IS THE WHOLE CLAIM HERE, which is why it is kept apart from the
# symbol. `-DWITH_CURL` has three spellings that all look like a fix and two of
# them break the build: `=system` is the default that produced the
# CMakeLists.txt:2127 fatal, and `=0` produced the keyring_vault:33 fatal. A
# guard matching the symbol, or the substring `WITH_CURL`, passes on all three.
REQUIRED_FLAGS = {
    "-DWITH_CURL": "bundled",
}

# Symbols that must NOT be in either list. Intermediate attempts at this fix
# passed `-DWITH_AUTHENTICATION_LDAP=OFF` and 5fe565e ships none of them,
# because the builder now installs libldap2-dev, libsasl2-dev and libkrb5-dev,
# so the warnings that fed the CMakeLists.txt:2075 fatal do not fire and such
# a flag would gate nothing.
# The other two are listed with it because they are the obvious companions to
# re-add, and they would be equally dead: both defaults route through
# `WITH_INTERNAL`, which is 0 whenever `internal/CMakeLists.txt` is absent, and
# it is absent from the published tree at this pin.
#
# This is an absence assertion, a shape worth justifying. It is not "these
# flags are wrong forever" -- it is "this list does not switch authentication
# plugins off, and a flag that gates nothing does not belong in a list whose
# header claims every entry cites the line that made it necessary". If a future
# pin makes one load-bearing, this constant is where that argument gets made,
# with the file:line that changed.
FORBIDDEN_FLAG_SYMBOLS = frozenset({
    "-DWITH_AUTHENTICATION_LDAP",
    "-DWITH_AUTHENTICATION_KERBEROS",
    "-DWITH_AUTHENTICATION_CLIENT_PLUGINS",
})

# Development headers and runtime plugins the builder stage installs, from the
# vendor's own `install_deps()` in build-ps/percona-server-9.0_builder.sh
# filtered to noble. It is a SUPERSET, not a minimal set -- the Dockerfile's
# apt note names the entries that are probably redundant and says why it was
# not pruned.
#
# THIRTEEN, AND THE LAST TWO ARE NOT DEV PACKAGES AT ALL. A --no-cache rebuild
# with libsasl2-dev installed and LDAP found still died at
# CMakeLists.txt:2075, because `WARN_MISSING_SYSTEM_SASL` returns quietly only
# when `FIND_LIBRARY(NAMES scram PATH_SUFFIXES sasl2)` finds a SCRAM *plugin*:
# libsasl2-dev ships sasl/sasl.h, libsasl2.so and libsasl2.a and nothing under
# sasl2/, while the unversioned sasl2/libscram.so symlink that FIND_LIBRARY
# needs comes from libsasl2-modules, a runtime package. A check whose message
# says "development libraries" wants a plugin, so the fix is two more packages
# rather than a different -dev one. (5fe565e had eleven; a8b8785 has thirteen.)
BUILDER_REQUIRED_PACKAGES = frozenset({
    "libldap2-dev", "libsasl2-dev", "libkrb5-dev", "libcurl4-openssl-dev",
    "libtirpc-dev", "libpam0g-dev", "liblz4-dev", "libzstd-dev",
    "libevent-dev", "libre2-dev", "libwrap0-dev",
    "libsasl2-modules", "libsasl2-modules-gssapi-mit",
})

# The three SASL package names, which collide by prefix in both directions and
# are the reason every package assertion here is whole-token.
#
#   libsasl2-dev                  headers; was present when :2075 still fired
#   libsasl2-modules              the SCRAM plugin the check actually wants
#   libsasl2-modules-gssapi-mit   a different plugin; CONTAINS the name above
#
# A check for the substring `libsasl2` is satisfied by any one of them. A
# check for `libsasl2-modules` as a substring or prefix is satisfied by
# `libsasl2-modules-gssapi-mit` alone -- which would leave the SCRAM plugin
# missing and :2075 still firing, with a green test. That is not hypothetical:
# the first of these three being present while the second was absent is
# exactly what cost the rebuild.
SASL_PACKAGES = (
    "libsasl2-dev",
    "libsasl2-modules",
    "libsasl2-modules-gssapi-mit",
)

# The spelling the vendor script uses, which must never appear here: it is
# virtual on noble (provided by libpam0g-dev), so an edit that "corrects" the
# list to match build-ps must fail here and not in a two-hour build.
VENDOR_VIRTUAL_PAM_NAME = "libpam-dev"

# Wrong values the Dockerfile header quotes out of the five cmake errors. Each
# must be findable in the FILE and unfindable in the ARGUMENTS; that gap is
# what says the parsers are bounded to the invocation rather than grepping
# documentation. Two of them are wrong values for a flag that IS passed, which
# is precisely what makes a substring check on `WITH_CURL` worthless.
WRONG_VALUES_QUOTED_IN_PROSE = (
    "-DWITH_AUTHENTICATION_LDAP=ON",
    "-DWITH_CURL=0",
    "-DWITH_CURL=system",
)

# The CORRECT value is quoted in the header too (the apt note explains the
# package list "with -DWITH_CURL=bundled and the vendor's bundled
# libevent/zstd"), so even counting occurrences of the right string across the
# whole file cannot establish what cmake is told. Only the bounded parse can.
RIGHT_VALUE_ALSO_QUOTED_IN_PROSE = "-DWITH_CURL=bundled"

# The Dockerfile legitimately passes seven flags the YAML does not, and this is
# the allowlist that says which. They are not policy: four name where this
# particular image puts things, one names the build it is, and two hand cmake
# the builder stage's own exported CFLAGS -- so none of them is a decision the
# YAML could sensibly express or a resource pass could vary.
#
# The last two are the ones with a trap behind them. `-DCMAKE_C_FLAGS=` and
# `-DCMAKE_CXX_FLAGS=` are half of the fix for the DEB package-flags overwrite
# (docker/percona/Dockerfile's note above its cmake line, and
# tests/test_e8_package_flags_overwrite.py): their value is `"$CFLAGS"`, a
# shell expansion of a variable exported two lines earlier in the same RUN. A
# YAML copy could only hold a literal, and because `${CMAKE_FLAGS}` is appended
# LAST that literal would WIN -- quietly replacing the flags the builder
# exported with whatever this file happened to say. So they belong here and the
# YAML must stay silent about them. `-DWITH_PACKAGE_FLAGS=OFF`, the other half,
# IS mirrored in the YAML, because a duplicate of it can only confirm.
#
# This set is asserted to be EXACTLY the Dockerfile-only symbols, not merely to
# contain them. An eighth Dockerfile-only flag added later therefore fails here
# rather than slipping in, which is the drift this file exists to stop; and an
# entry that goes stale because the Dockerfile stopped passing it fails too.
DOCKERFILE_ONLY_SYMBOLS = frozenset({
    "-DCMAKE_BUILD_TYPE",
    "-DCMAKE_INSTALL_PREFIX",
    "-DMYSQL_DATADIR",
    "-DSYSCONFDIR",
    "-DMYSQL_SERVER_SUFFIX",
    "-DCMAKE_C_FLAGS",
    "-DCMAKE_CXX_FLAGS",
})


def _split(flag: str) -> tuple[str, str]:
    """`-DWITH_SSL=system` -> `("-DWITH_SSL", "system")`.

    `partition` and not `split("=")`: a value may itself contain `=`, and a
    flag with no `=` at all (`-DFOO`) must come back with an empty value
    rather than raising, so the mirror can still report it. Nothing here
    assumes the value is `OFF` -- see vacuity shape 3 in the module docstring.
    """
    symbol, _, value = flag.partition("=")
    return symbol, value


@pytest.fixture(scope="module")
def yaml_flags() -> list[str]:
    """`build.cmake_flags`, parsed as YAML and not grepped as text.

    Grepping would pass against a flag that is present but commented out, and
    this config's own comment block quotes both failing values out of the cmake
    errors -- so a text search for either symbol finds a hit in the prose
    whichever way the real list is set.
    """
    with open(PERCONA_YML) as handle:
        cfg = yaml.safe_load(handle)
    return list((cfg or {}).get("build", {}).get("cmake_flags") or [])


@pytest.fixture(scope="module")
def dockerfile_text() -> str:
    with open(PERCONA_DOCKERFILE) as handle:
        return handle.read()


@pytest.fixture(scope="module")
def cmake_region(dockerfile_text: str) -> str:
    """Just the builder RUN's cmake invocation.

    Bounded at both ends on purpose. The header above it quotes `-D` flags in
    prose -- note 2's submodule table and the apt-list comment's account of
    the configure failures both do -- so a scan of the whole file would
    collect flags that are documentation and not arguments, including a
    value that caused one of the failures.
    """
    assert "cmake /src/source" in dockerfile_text, (
        "the builder's cmake invocation was not found; this test's anchor is "
        "stale or the Dockerfile was restructured")
    after = dockerfile_text.split("cmake /src/source", 1)[1]
    assert "make -j" in after, (
        "no 'make -j' after the cmake invocation to bound the region on")
    return after.split("make -j", 1)[0]


@pytest.fixture(scope="module")
def dockerfile_flags(cmake_region: str) -> list[str]:
    r"""Flags as passed: one per continuation line, anchored at line start.

    Anchoring matters. An unanchored `-D\S+` would also match inside
    `-march=${MARCH}` style text and inside any `-D` that appears mid-line.
    """
    return re.findall(r"^\s*(-D\S+)", cmake_region, re.MULTILINE)


# ---------------------------------------------------------------------------
# Non-vacuity: the parsers found something before anything asks what
# ---------------------------------------------------------------------------
#
# Most assertions further down are of the form "this set difference is empty",
# and two empty inputs satisfy all of them. A renamed file, a restructured
# Dockerfile or a `build:` key that parses to `{}` would otherwise turn this
# whole module green over nothing. These run first for that reason.

class TestTheInputsAreActuallyThere:

    def test_both_files_exist(self):
        assert os.path.isfile(PERCONA_YML), PERCONA_YML
        assert os.path.isfile(PERCONA_DOCKERFILE), PERCONA_DOCKERFILE

    def test_the_yaml_flag_list_is_not_empty(self, yaml_flags):
        assert yaml_flags, (
            "build.cmake_flags parsed to nothing -- an empty list, a renamed "
            "key or a file that is no longer the engine config")

    def test_the_dockerfile_text_is_not_empty(self, dockerfile_text):
        assert dockerfile_text.strip(), "the Dockerfile read back empty"

    def test_the_dockerfile_cmake_region_is_not_empty(self, cmake_region):
        assert cmake_region.strip(), (
            "the region between 'cmake /src/source' and 'make -j' is blank")

    def test_the_dockerfile_flag_list_is_not_empty(self, dockerfile_flags):
        assert dockerfile_flags, (
            "no -D flags parsed out of the builder's cmake invocation")

    def test_neither_list_is_implausibly_short(self, yaml_flags,
                                               dockerfile_flags):
        """A parse that returns one flag is not obviously broken the way one
        that returns zero is, so the floor is set above the trivial case.

        Ten is below both real counts and above anything a partial parse would
        plausibly produce; it is a smoke bound, not a pin on the contents.
        """
        assert len(yaml_flags) >= 10, yaml_flags
        assert len(dockerfile_flags) >= 10, dockerfile_flags

    def test_no_flag_is_duplicated_in_either_list(self, yaml_flags,
                                                  dockerfile_flags):
        """Set arithmetic below hides duplicates, and a duplicated flag with
        two different values is exactly the drift this file is about: the
        Dockerfile's `${CMAKE_FLAGS}` append means the LAST one wins, so two
        copies of one symbol make the effective value depend on ordering.
        """
        for label, flags in (("yaml", yaml_flags),
                             ("dockerfile", dockerfile_flags)):
            symbols = [_split(f)[0] for f in flags]
            dupes = sorted({s for s in symbols if symbols.count(s) > 1})
            assert not dupes, f"{label} repeats: {dupes}"


# ---------------------------------------------------------------------------
# The flags the two incidents required
# ---------------------------------------------------------------------------

class TestTheFlagsTheConfigureFailuresRequired:
    """Both flags, in both copies, with the right values. Asserted as exact
    membership of a parsed list and exact equality on the value."""

    @pytest.mark.parametrize("symbol,value", sorted(REQUIRED_FLAGS.items()))
    def test_the_yaml_carries_it(self, symbol, value, yaml_flags):
        assert f"{symbol}={value}" in yaml_flags, (
            f"{symbol}={value} missing from config/engines/percona.yml "
            f"build.cmake_flags; this is the live path that "
            f"./run-benchmark.sh build reads. Got: {yaml_flags}")

    @pytest.mark.parametrize("symbol,value", sorted(REQUIRED_FLAGS.items()))
    def test_the_dockerfile_carries_it(self, symbol, value, dockerfile_flags):
        assert f"{symbol}={value}" in dockerfile_flags, (
            f"{symbol}={value} missing from docker/percona/Dockerfile's "
            f"builder cmake invocation, so a by-hand `docker build` would "
            f"still fail configure. Got: {dockerfile_flags}")

    @pytest.mark.parametrize("source", ["yaml", "dockerfile"])
    @pytest.mark.parametrize("symbol,value", sorted(REQUIRED_FLAGS.items()))
    def test_the_value_is_the_disabling_one(self, symbol, value, source,
                                            yaml_flags, dockerfile_flags):
        """Presence is not the claim.

        `-DWITH_AUTHENTICATION_LDAP=ON` and `-DWITH_CURL=system` are the
        strings the two cmake errors print, and they are the defects. A test
        that only checked the symbol was present would pass on both.
        """
        flags = yaml_flags if source == "yaml" else dockerfile_flags
        values = [v for s, v in map(_split, flags) if s == symbol]
        assert values == [value], (
            f"{source}: expected exactly one {symbol} with value {value!r}, "
            f"got {values!r}")

    def test_the_value_vocabulary_is_mixed(self, yaml_flags):
        """Why `partition("=")` and not a trailing-`OFF` filter.

        Four of these flags do not end in `OFF`: `-DWITH_CURL=bundled`,
        `-DWITHOUT_COMPONENT_KEYRING_KMIP=1`, `-DWITH_SSL=system` and
        `-DWITH_ZLIB=system`. The first is the flag this change exists for, so
        a check that split on `"=OFF"` or filtered for a trailing `OFF` would
        miss precisely the regression it was written to catch. This fails if
        the vocabulary ever collapses to `OFF`, at which point the generality
        elsewhere can be dropped deliberately rather than forgotten.
        """
        values = {v for _, v in map(_split, yaml_flags)}
        non_off = sorted(values - {"OFF"})
        assert len(non_off) >= 3, (
            f"expected several non-OFF values, got {sorted(values)}")
        # ...and every required flag's value is among them, so the generality
        # is protecting the flag this change adds and not only its neighbours.
        for value in REQUIRED_FLAGS.values():
            assert value in values, (value, sorted(values))
            assert value != "OFF", REQUIRED_FLAGS

    def test_a_symbol_prefix_cannot_identify_a_flag(self, yaml_flags):
        """The prefix ambiguity that makes exact matching necessary, asserted.

        `-DWITH_R` is a prefix of both `-DWITH_ROCKSDB=OFF` and
        `-DWITH_ROUTER=OFF`, and `-DWITH` is a prefix of
        `-DWITHOUT_COMPONENT_KEYRING_KMIP=1` as well as of every `-DWITH_*`.
        So prefix matching cannot name a flag in this list, and this fails if
        that stops being true -- at which point the strictness everywhere else
        can be relaxed deliberately rather than by accident.
        """
        for prefix, least in (("-DWITH_R", 2), ("-DWITH", 2)):
            hits = sorted(f for f in yaml_flags if f.startswith(prefix))
            assert len(hits) >= least, (
                f"{prefix!r} no longer collides; hits={hits}")

    def test_matching_the_curl_symbol_alone_cannot_tell_right_from_wrong(
            self, yaml_flags, dockerfile_text):
        """Why `REQUIRED_FLAGS` pins a value and not a symbol.

        All three spellings of this flag appear in the Dockerfile -- `=bundled`
        as an argument, `=system` and `=0` quoted in the header out of the two
        fatals they caused. A guard written `assert "WITH_CURL" in text`, or
        one checking only that the symbol is passed, is satisfied by every one
        of them, including the two that break the build. This test asserts
        that the ambiguity is real, so the exactness elsewhere is visibly
        load-bearing rather than stylistic.
        """
        for wrong in ("-DWITH_CURL=0", "-DWITH_CURL=system"):
            assert wrong in dockerfile_text, (
                f"the header no longer quotes {wrong!r}, one of the two "
                f"values of this flag that failed a real configure, so this "
                f"test has stopped discriminating")
        # The right value is in the prose as well, so even a count of the
        # correct string over the whole file proves nothing about arguments.
        assert dockerfile_text.count(RIGHT_VALUE_ALSO_QUOTED_IN_PROSE) >= 2, (
            dockerfile_text.count(RIGHT_VALUE_ALSO_QUOTED_IN_PROSE))
        passed = [f for f in yaml_flags if _split(f)[0] == "-DWITH_CURL"]
        assert passed == ["-DWITH_CURL=bundled"], passed

    def test_no_other_flag_contains_the_curl_symbol(self, yaml_flags,
                                                    dockerfile_flags):
        """`-DWITH_CURL` is a substring of nothing else in either list, which
        is luck rather than design: adding a `-DWITH_CURL_PATH` later would
        make any substring check elsewhere ambiguous, and this makes that
        visible instead of silent."""
        for label, flags in (("yaml", yaml_flags),
                             ("dockerfile", dockerfile_flags)):
            carriers = sorted(f for f in flags if "-DWITH_CURL" in f)
            assert carriers == ["-DWITH_CURL=bundled"], (
                f"{label}: more than one flag now carries the text "
                f"'-DWITH_CURL': {carriers}")

    @pytest.mark.parametrize("symbol", sorted(FORBIDDEN_FLAG_SYMBOLS))
    @pytest.mark.parametrize("source", ["yaml", "dockerfile"])
    def test_the_authentication_flags_are_absent(self, symbol, source,
                                                 yaml_flags,
                                                 dockerfile_flags):
        """The flags an earlier revision added and this one removed.

        Parsed tokens and not a text search, which matters more here than
        anywhere else in this file: the Dockerfile header QUOTES
        `-DWITH_AUTHENTICATION_LDAP=ON` out of the CMakeLists.txt:2075 error,
        so `assert "-DWITH_AUTHENTICATION_LDAP" not in dockerfile_text` would
        fail against a correct tree and teach the next author to delete the
        guard.
        """
        flags = yaml_flags if source == "yaml" else dockerfile_flags
        present = sorted(f for f in flags if _split(f)[0] == symbol)
        assert not present, (
            f"{source}: {symbol} is back in the list as {present}. With "
            f"libldap2-dev, libsasl2-dev and libkrb5-dev installed in the "
            f"builder it gates nothing; see FORBIDDEN_FLAG_SYMBOLS for the "
            f"argument and what would have to change to re-add it.")

    def test_the_forbidden_set_is_not_vacuously_satisfied(self, yaml_flags):
        """Non-vacuity for the absence test above.

        An empty `FORBIDDEN_FLAG_SYMBOLS`, or a parse returning nothing, would
        satisfy it over no content. The symbols are also checked to be
        well-formed, so a typo'd entry cannot pass by never matching anything.
        """
        assert FORBIDDEN_FLAG_SYMBOLS, "the forbidden set is empty"
        assert yaml_flags, "nothing parsed to check against"
        for symbol in FORBIDDEN_FLAG_SYMBOLS:
            assert symbol.startswith("-D"), symbol
            assert "=" not in symbol, (
                f"{symbol} carries a value; absence is asserted on the symbol "
                f"so that =ON and =OFF are both caught")

    def test_the_reviewers_substring_example_is_not_the_real_trap(self):
        """Recorded because it was checked, not because it was believed.

        A review that fed into this file offered `WITH_LDAP` as a substring of
        `WITH_AUTHENTICATION_LDAP`. It is not one -- after `WITH_` comes
        `AUTHENTICATION`, not `LDAP`. The shape of the warning was right and
        the instance was wrong, and leaving a false example in a docstring is
        how a correct guard gets deleted by someone who tests the example.
        """
        assert "WITH_LDAP" not in "WITH_AUTHENTICATION_LDAP"


# ---------------------------------------------------------------------------
# The prose quotes the wrong values; the parsers must not see them
# ---------------------------------------------------------------------------

class TestTheProseAroundTheInvocationIsNotMistakenForArguments:
    """The reason both parsers are bounded, asserted rather than trusted.

    `docker/percona/Dockerfile`'s apt-list comment quotes the cmake errors
    verbatim, so the file contains `-DWITH_AUTHENTICATION_LDAP=ON` and
    `-DWITH_CURL=0` -- values that broke the build -- as well as the correct
    `-DWITH_CURL=bundled`. A whole-file text search finds the wrong value as
    readily
    as the right one, so it cannot establish what cmake is actually told.
    """

    @pytest.mark.parametrize("wrong", WRONG_VALUES_QUOTED_IN_PROSE)
    def test_the_wrong_value_is_in_the_file(self, wrong, dockerfile_text):
        """If this fails the documentation stopped quoting the failure, and
        the test below stops proving anything -- fix them together."""
        assert wrong in dockerfile_text, (
            f"{wrong!r} is no longer quoted anywhere in the Dockerfile, so "
            f"the bounding test below has nothing to discriminate against")

    @pytest.mark.parametrize("wrong", WRONG_VALUES_QUOTED_IN_PROSE)
    def test_the_wrong_value_is_not_an_argument(self, wrong, dockerfile_flags):
        assert wrong not in dockerfile_flags, (
            f"{wrong!r} is being passed to cmake as an argument, not merely "
            f"quoted in the header prose. Parsed: {dockerfile_flags}")

    def test_the_prose_lives_above_the_region_the_parser_reads(
            self, dockerfile_text, cmake_region):
        """Belt and braces on the bound itself: the quoted errors are in the
        header, and the header is not in the parsed region."""
        for wrong in WRONG_VALUES_QUOTED_IN_PROSE:
            assert wrong not in cmake_region, (
                f"{wrong!r} appears inside the builder's cmake invocation "
                f"region, so the bounding is not separating prose from "
                f"arguments")
        assert len(cmake_region) < len(dockerfile_text), (
            "the 'region' is the whole file; the bounding did nothing")


def _apt_packages(stage_text: str) -> frozenset[str]:
    """Package names from every `apt-get install` in one stage.

    Truncated at the first `&&` after `install`, because the runtime stage
    chains `&& useradd ...` onto its install and those tokens are not
    packages. Option words are dropped by their leading dash.
    """
    found: set[str] = set()
    for body in re.findall(r"apt-get install((?:[^\n]*\\\n)*[^\n]*)",
                           stage_text):
        for token in body.replace("\\\n", " ").split("&&")[0].split():
            if token.startswith("-") or token in {";", "\\"}:
                continue
            found.add(token)
    return frozenset(found)


@pytest.fixture(scope="module")
def dockerfile_stages(dockerfile_text) -> dict:
    """Stage name -> that stage's text, split on `FROM ... AS <name>`.

    The split is what makes the package assertion specific. `libtirpc-dev` in
    the runtime or bench stage would satisfy a whole-file grep and still fail
    to build, because it is the builder that runs cmake.
    """
    marks = list(re.finditer(r"^FROM\s+\S+\s+AS\s+(\w+)\s*$",
                             dockerfile_text, re.MULTILINE))
    assert marks, "no named build stages found in the Dockerfile"
    stages = {}
    for index, mark in enumerate(marks):
        end = (marks[index + 1].start() if index + 1 < len(marks)
               else len(dockerfile_text))
        stages[mark.group(1)] = dockerfile_text[mark.start():end]
    return stages


@pytest.fixture(scope="module")
def builder_packages(dockerfile_stages) -> frozenset:
    assert "builder" in dockerfile_stages, sorted(dockerfile_stages)
    return _apt_packages(dockerfile_stages["builder"])


# ---------------------------------------------------------------------------
# The package the unskippable check needs, in the right stage
# ---------------------------------------------------------------------------

class TestTheBuilderStageInstallsWhatTheUnskippableCheckNeeds:
    """`libtirpc-dev`, and the stage it has to be in.

    Unlike the two flags above, this gate has no off switch:
    `CMakeLists.txt:2224-2226` runs `MYSQL_CHECK_RPC()` on every Linux build
    and `cmake/rpc.cmake:106-110` is a bare `IF(NOT RPC_INCLUDE_DIRS)` ending
    in FATAL_ERROR. So the fix is a build dependency, and the only thing a
    test can usefully hold is that the dependency is declared in the stage
    that compiles.
    """

    def test_the_stage_split_found_the_stages_this_file_has(
            self, dockerfile_stages):
        """Non-vacuity for the split: a regex that matched nothing would make
        every stage lookup below raise rather than silently pass, but a split
        that found only ONE stage would make the bounding meaningless."""
        assert {"builder", "runtime", "bench"} <= set(dockerfile_stages), (
            f"stages found: {sorted(dockerfile_stages)}")

    def test_each_stage_is_a_proper_slice_of_the_file(self, dockerfile_stages,
                                                      dockerfile_text):
        for name, text in dockerfile_stages.items():
            assert text, f"stage {name} is empty"
            assert len(text) < len(dockerfile_text), (
                f"stage {name} is the whole file; the split did nothing")

    def test_the_builder_apt_list_is_not_empty(self, builder_packages):
        """If this parsed to nothing, the membership test below would be
        asserting against an empty set and would fail loudly -- but a parse
        that returned one junk token would not, so the floor is a count."""
        assert len(builder_packages) >= 10, sorted(builder_packages)

    def test_the_builder_apt_list_is_packages_and_not_option_words(
            self, builder_packages):
        """The parse drops flags by their leading dash; this is the assertion
        that says it did, so a future `-y` creeping in as a 'package' cannot
        inflate the count above."""
        assert not [p for p in builder_packages if p.startswith("-")], (
            sorted(builder_packages))
        for expected in ("build-essential", "cmake", "python3"):
            assert expected in builder_packages, sorted(builder_packages)

    @pytest.mark.parametrize("package", sorted(BUILDER_REQUIRED_PACKAGES))
    def test_the_builder_stage_installs_it(self, package, builder_packages):
        assert package in builder_packages, (
            f"{package} is not in the BUILDER stage's apt-get install list, "
            f"so cmake's unskippable MYSQL_CHECK_RPC would still fail at "
            f"rpc.cmake:108. Builder installs: {sorted(builder_packages)}")

    @pytest.mark.parametrize("package", sorted(BUILDER_REQUIRED_PACKAGES))
    def test_a_later_stage_does_not_carry_it_instead(self, package,
                                                     dockerfile_stages):
        """The wrong-stage mutant, which a whole-file grep cannot see.

        The runtime image deliberately carries runtime libraries and no `-dev`
        packages at all -- `libssl3t64` and not `libssl-dev`, and so on -- so a
        build-time `-dev` package appearing there is always a mistake, and the
        build would fail with the package present in the file.
        """
        for name in ("runtime", "bench"):
            if name not in dockerfile_stages:
                continue
            assert package not in _apt_packages(dockerfile_stages[name]), (
                f"{package} is installed in the {name} stage. It is needed by "
                f"cmake in the BUILDER stage; here it only grows the image "
                f"and would leave configure failing.")

    @pytest.mark.parametrize("package", SASL_PACKAGES)
    def test_each_sasl_package_is_installed_under_its_own_exact_name(
            self, package, builder_packages):
        """All three, separately, as whole tokens.

        `libsasl2-dev` alone was the state in which CMakeLists.txt:2075 still
        fired: the check calls
        `FIND_LIBRARY(NAMES scram PATH_SUFFIXES sasl2)` and the dev package
        ships nothing under `sasl2/`. So "a libsasl2 package is installed" is
        not the property that matters; which ones are is.
        """
        assert package in builder_packages, (
            f"{package} is not a whole token in the builder stage's install "
            f"list. Builder installs: {sorted(builder_packages)}")

    def test_the_sasl_names_collide_by_prefix_in_both_directions(self):
        """The hazard that makes whole-token matching necessary, asserted.

        This is a property of the names themselves, so it is checked against
        the names and not the file: `libsasl2-modules` is a strict prefix of
        `libsasl2-modules-gssapi-mit`, and `libsasl2` prefixes all three. A
        guard written `any(p.startswith("libsasl2-modules") for p in pkgs)` is
        therefore satisfied by the gssapi package alone, leaving the SCRAM
        plugin missing and :2075 still firing with a green test.

        If this ever fails, the names have changed and the whole-token
        strictness above can be reconsidered deliberately.
        """
        dev, modules, gssapi = SASL_PACKAGES
        assert gssapi.startswith(modules) and gssapi != modules, (
            modules, gssapi)
        assert len({p for p in SASL_PACKAGES if p.startswith("libsasl2")}) == 3
        # ...and the dev package is NOT a prefix of either plugin, so the
        # collision is specifically between the two plugin names.
        assert not modules.startswith(dev), (dev, modules)

    def test_the_gssapi_package_alone_does_not_satisfy_the_scram_requirement(
            self, builder_packages):
        """The mutant this test exists for, stated as an assertion.

        Written so that a list carrying only `libsasl2-modules-gssapi-mit`
        fails: membership of the exact name `libsasl2-modules` is required
        independently, and the two are asserted to be different packages.
        """
        _dev, modules, gssapi = SASL_PACKAGES
        assert modules in builder_packages, (
            f"{modules} is absent. {gssapi} being present does not provide "
            f"the SCRAM plugin that FIND_LIBRARY(NAMES scram) looks for, and "
            f"a prefix check would not have noticed. Builder installs: "
            f"{sorted(builder_packages)}")
        assert modules != gssapi

    def test_the_vendor_virtual_package_name_is_not_used(self,
                                                         builder_packages):
        """`libpam0g-dev`, never the vendor script's `libpam-dev`.

        build-ps/percona-server-9.0_builder.sh:551 installs `libpam-dev`, and
        on noble that is a VIRTUAL package whose only provider is
        `libpam0g-dev` -- packages.ubuntu.com files it under "virtual". Anyone
        re-deriving this list from the vendor script will copy the virtual
        name, so the failure belongs here and not two hours into a build.
        """
        assert VENDOR_VIRTUAL_PAM_NAME not in builder_packages, (
            f"{VENDOR_VIRTUAL_PAM_NAME!r} is the vendor's spelling and is "
            f"virtual on noble; use the providing package instead. Builder "
            f"installs: {sorted(builder_packages)}")
        assert "libpam0g-dev" in builder_packages, sorted(builder_packages)

    def test_the_required_package_set_is_not_vacuous(self, builder_packages):
        """Non-vacuity for the parametrized membership test.

        An empty `BUILDER_REQUIRED_PACKAGES` would make it collect no cases at
        all and the class would look green over nothing. The count is pinned
        loosely at the vendor-derived eleven so that silently emptying the
        constant fails here.
        """
        assert len(BUILDER_REQUIRED_PACKAGES) >= 13, (
            sorted(BUILDER_REQUIRED_PACKAGES))
        assert BUILDER_REQUIRED_PACKAGES <= builder_packages, sorted(
            BUILDER_REQUIRED_PACKAGES - builder_packages)

    def test_the_stage_package_lists_actually_differ(self, dockerfile_stages):
        """Non-vacuity for the test above: if every stage parsed to the same
        set, 'not in runtime' and 'in builder' could not both hold, and a
        bounding bug would show up here rather than as a confusing failure."""
        builder = _apt_packages(dockerfile_stages["builder"])
        runtime = _apt_packages(dockerfile_stages["runtime"])
        assert builder != runtime, (
            "builder and runtime parsed to the same package set; the stage "
            "split is not separating them")
        assert runtime, "the runtime stage parsed to no packages at all"


# ---------------------------------------------------------------------------
# The mirror, in both directions
# ---------------------------------------------------------------------------

class TestTheTwoFlagListsMirrorEachOther:
    """The guard that did not exist, and the reason the lists could drift.

    `docker/percona/Dockerfile`'s own comment says "Flags mirror
    config/engines/percona.yml", and `config/engines/percona.yml` says "Both
    copies carry the same values". Two files asserting agreement with each
    other and nothing checking it is the standing shape of this repo's silent
    bugs.
    """

    def test_every_yaml_flag_is_passed_by_the_dockerfile(self, yaml_flags,
                                                         dockerfile_flags):
        missing = sorted(set(yaml_flags) - set(dockerfile_flags))
        assert not missing, (
            "declared in config/engines/percona.yml but not passed by "
            f"docker/percona/Dockerfile: {missing}. A by-hand `docker build` "
            "would compile a different server from `./run-benchmark.sh build`.")

    def test_every_dockerfile_policy_flag_is_declared_in_the_yaml(
            self, yaml_flags, dockerfile_flags):
        yaml_symbols = {s for s, _ in map(_split, yaml_flags)}
        missing = sorted(
            flag for flag in dockerfile_flags
            if _split(flag)[0] not in yaml_symbols
            and _split(flag)[0] not in DOCKERFILE_ONLY_SYMBOLS)
        assert not missing, (
            "passed by docker/percona/Dockerfile, absent from "
            f"config/engines/percona.yml, and not allowlisted: {missing}. "
            "Either declare it in the YAML or add it to "
            "DOCKERFILE_ONLY_SYMBOLS with a reason.")

    def test_the_allowlist_is_exactly_the_dockerfile_only_set(
            self, yaml_flags, dockerfile_flags):
        """Equality, not containment, in both directions at once.

        Containment would let a sixth Dockerfile-only flag through if the
        allowlist happened to be a superset, and would never notice an entry
        that went stale. This is the assertion that makes the exclusion a
        decision with a shelf life rather than a blanket.
        """
        yaml_symbols = {s for s, _ in map(_split, yaml_flags)}
        docker_symbols = {s for s, _ in map(_split, dockerfile_flags)}
        unmirrored = docker_symbols - yaml_symbols
        assert unmirrored == set(DOCKERFILE_ONLY_SYMBOLS), (
            "the Dockerfile-only flag set moved.\n"
            f"  newly unmirrored (add to the YAML, or allowlist with a "
            f"reason): {sorted(unmirrored - DOCKERFILE_ONLY_SYMBOLS)}\n"
            f"  stale allowlist entries (no longer passed, or now mirrored): "
            f"{sorted(DOCKERFILE_ONLY_SYMBOLS - unmirrored)}")

    def test_the_shared_symbols_carry_identical_values(self, yaml_flags,
                                                       dockerfile_flags):
        """Agreeing on which flags exist is half of mirroring.

        `=OFF` against `=ON`, or `=0` against `=system`, for the same symbol
        passes every membership test above on the symbol alone; and because
        the Dockerfile appends `${CMAKE_FLAGS}` last the YAML's value is the
        one that would win -- so the two files would document opposite
        policies and the build would quietly follow the YAML.
        """
        yaml_pairs = dict(map(_split, yaml_flags))
        docker_pairs = dict(map(_split, dockerfile_flags))
        shared = sorted(set(yaml_pairs) & set(docker_pairs))
        assert shared, "the two lists share no symbol at all"
        disagree = {
            symbol: {"yaml": yaml_pairs[symbol],
                     "dockerfile": docker_pairs[symbol]}
            for symbol in shared
            if yaml_pairs[symbol] != docker_pairs[symbol]
        }
        assert not disagree, f"same flag, different value: {disagree}"

    def test_the_dockerfile_still_appends_the_yaml_flags_last(
            self, cmake_region):
        """The premise the two tests above rest on.

        If `${CMAKE_FLAGS}` were dropped, `build.cmake_flags` would stop
        reaching cmake entirely and every assertion in this file would be
        checking a list that no longer influences the build -- green, and
        measuring nothing.
        """
        assert "${CMAKE_FLAGS}" in cmake_region, (
            "the builder's cmake invocation no longer expands ${CMAKE_FLAGS}, "
            "so config/engines/percona.yml build.cmake_flags is dead config")
