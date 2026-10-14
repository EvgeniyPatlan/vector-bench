"""The Percona builder's `make` invocation and the flag that makes it readable.

WHY THIS FILE EXISTS
--------------------
2026-10-14. The first `./run-benchmark.sh build --engines percona` ever run
reached 20% and then 26% of the compile and ended, both times, with this as
its final line and nothing attributable above it:

    make: *** [Makefile:166: all] Error 2

`Makefile:166` is CMake's generated top-level `all`, whose whole recipe is
`$(MAKE) -f CMakeFiles/Makefile2 all`, so that line says only "something
below me failed". The compiler diagnostic and the `[ 26%] Building CXX
object ...` line naming the file it came from are written by two different
processes, and at `-j$(nproc)` -- 80 on the reference host -- they land far
apart in the log, as does the sub-make's own `make[1]: *** Error 1`.

The fix is `make --output-sync=target`, which has GNU Make capture each
target's two streams and emit them as one block. Measured on GNU Make 4.3
with a two-level recursive build at -j4 and one failing target:

    without the flag     the `error:` text is 3 unrelated progress lines
                         below the target that produced it, with 3 more
                         after it before the top-level error
    --output-sync=target the `error:` text is the NEXT line

This file guards three things that are easy to get wrong in that one-line
edit, each of which was a live hazard when it was made:

  1. the flag is `=target`. `=recurse` groups per recursive make, and the
     generated top-level Makefile recurses once for the entire build, so
     `=recurse` buffers every line until the build ends -- hours of silence.
     `=line` keeps progress live but still splits a multi-line diagnostic
     from the target that names the file.
  2. the flag sits AFTER `-j"$JOBS"`. `make -j` is the literal string
     tests/test_e1_percona_design.py and tests/test_e4_percona_build_flags.py
     split this Dockerfile on to bound the cmake flag region. Anything
     inserted between `make` and `-j` moves that boundary and drags the
     whole flag list across it, turning both of those suites vacuous rather
     than failing them.
  3. the Dockerfile's own account of the above is checked against the
     Dockerfile, because three claims in this file's comments have already
     reached master false (PR #22, corrected in PR #24).
"""

from __future__ import annotations

import os
import re

import pytest

VB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERCONA_DOCKERFILE = os.path.join(VB_ROOT, "docker", "percona", "Dockerfile")

# The region bound that tests/test_e1_percona_design.py and
# tests/test_e4_percona_build_flags.py both rely on. Spelled once here so a
# change to it fails this file too instead of only silently widening theirs.
REGION_END_ANCHOR = "make -j"
CMAKE_ANCHOR = "cmake /src/source"


# ---------------------------------------------------------------------------
# Predicates, as functions of the text
# ---------------------------------------------------------------------------
#
# Every assertion below routes through one of these, and the mutation probes
# at the end call the same four on deliberately corrupted text. A predicate
# reachable only through the happy path cannot be shown to discriminate.

def build_command(text: str) -> str:
    """The `make` line that compiles, as written, without continuations.

    Picked by `-j`, not by position: `make install` follows it in the same
    RUN and would match a search for `make`.
    """
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("make -j"):
            return stripped.rstrip("\\").strip().rstrip(";").strip()
    return ""


def output_sync_values(text: str) -> list[str]:
    """The distinct `--output-sync=<x>` values anywhere in the file, sorted.

    Comments included on purpose: a note that quotes a value the command does
    not use is the defect this repo keeps rediscovering. Matched on letters
    only, because the live occurrence is followed by `;` and the note's by a
    backtick, and a predicate that reported those as distinct values would
    report two where there is one.
    """
    found = re.findall(r"--output-sync=([A-Za-z]+)", text)
    return sorted(set(found))


def cmake_region(text: str) -> str:
    """What the two sibling suites see as the cmake flag region."""
    if CMAKE_ANCHOR not in text:
        return ""
    after = text.split(CMAKE_ANCHOR, 1)[1]
    if REGION_END_ANCHOR not in after:
        return ""
    return after.split(REGION_END_ANCHOR, 1)[0]


def unguarded_region(text: str) -> str:
    """The region as tests/test_e1_percona_design.py:350-351 computes it.

    Transcribed deliberately, including what it omits: a bare `.split()` with
    no assertion that the separator is present. `str.split` on a missing
    separator returns the whole string, so this region runs to end of file
    rather than reporting that it could not be bounded.
    """
    if CMAKE_ANCHOR not in text:
        return ""
    return text.split(CMAKE_ANCHOR, 1)[1].split(REGION_END_ANCHOR, 1)[0]


def region_flags(text: str) -> list[str]:
    """`-D` flags as the sibling suites parse them: anchored at line start."""
    return re.findall(r"^\s*(-D\S+)", cmake_region(text), re.MULTILINE)


@pytest.fixture(scope="module")
def dockerfile_text() -> str:
    with open(PERCONA_DOCKERFILE) as handle:
        return handle.read()


# ---------------------------------------------------------------------------
# Non-vacuity first
# ---------------------------------------------------------------------------

class TestTheInputIsActuallyThere:

    def test_the_dockerfile_exists(self):
        assert os.path.isfile(PERCONA_DOCKERFILE), PERCONA_DOCKERFILE

    def test_the_dockerfile_is_not_empty(self, dockerfile_text):
        assert dockerfile_text.strip(), "the Dockerfile read back empty"

    def test_a_build_command_was_found(self, dockerfile_text):
        assert build_command(dockerfile_text), (
            "no line starting `make -j` in the Percona Dockerfile; this "
            "file's anchor is stale or the builder stage was restructured")

    def test_the_cmake_region_is_not_empty(self, dockerfile_text):
        assert cmake_region(dockerfile_text).strip(), (
            f"the region between {CMAKE_ANCHOR!r} and {REGION_END_ANCHOR!r} "
            "is blank, which makes the two sibling suites vacuous")


# ---------------------------------------------------------------------------
# The flag
# ---------------------------------------------------------------------------

class TestTheBuildIsOutputSynchronised:

    def test_the_build_command_carries_output_sync(self, dockerfile_text):
        command = build_command(dockerfile_text)
        assert "--output-sync=" in command, (
            "the compiling `make` has no --output-sync; a failure in it "
            f"cannot be attributed to a file. Command: {command!r}")

    def test_the_value_is_target(self, dockerfile_text):
        command = build_command(dockerfile_text)
        value = re.search(r"--output-sync=(\S+)", command)
        assert value is not None, command
        assert value.group(1) == "target", (
            f"--output-sync={value.group(1)} on the compiling make. `recurse` "
            "buffers the whole build (CMake's top-level Makefile recurses "
            "once for all of it) and `line` splits a multi-line diagnostic "
            "from the target that names the file. Only `target` groups a "
            "failing compile with its own progress line.")

    def test_only_one_make_invocation_is_synchronised(self, dockerfile_text):
        """`make install` does not need it and must not be where it landed."""
        commands = [
            line.strip() for line in dockerfile_text.splitlines()
            if line.strip().startswith("make ")
        ]
        carrying = [c for c in commands if "--output-sync" in c]
        assert len(carrying) == 1, (
            f"expected exactly one synchronised make, found {carrying}")
        assert carrying[0].startswith("make -j"), carrying[0]

    def test_no_bare_dash_O_was_used_instead(self, dockerfile_text):
        """`-O` is the short spelling and defaults to `target`, so it would
        pass a value check. It is refused anyway: `-O` next to `-O3` in the
        same file's CFLAGS is a reader's trap, and the long form is the one
        the comment explains."""
        command = build_command(dockerfile_text)
        assert not re.search(r"(?<!\S)-O(?!\S)", command), (
            f"bare -O on the compiling make; write --output-sync=target: "
            f"{command!r}")


class TestTheRegionAnchorSurvived:
    """The hazard specific to this edit, not to the flag.

    `make -j` is load-bearing text for two other suites. These tests fail if
    the flag was inserted on the wrong side of it, which is the mistake that
    would leave those suites green and measuring nothing.
    """

    def test_the_command_still_begins_with_the_anchor(self, dockerfile_text):
        assert build_command(dockerfile_text).startswith(REGION_END_ANCHOR), (
            f"the compiling make no longer starts with {REGION_END_ANCHOR!r}; "
            "the cmake region bound in test_e1/test_e4 has moved")

    def test_the_flag_is_outside_the_cmake_region(self, dockerfile_text):
        assert "--output-sync" not in cmake_region(dockerfile_text), (
            "--output-sync fell inside the cmake flag region, so the region "
            "now ends later than test_e1 and test_e4 believe")

    def test_the_region_still_holds_the_whole_flag_list(self, dockerfile_text):
        """A floor, not a pin. test_e4 compares the list against
        percona.yml; this only catches a region that lost most of it."""
        flags = region_flags(dockerfile_text)
        assert len(flags) >= 10, (
            f"only {len(flags)} -D flags inside the region: {flags}")
        for required in ("-DWITH_ROCKSDB=OFF", "-DWITH_COREDUMPER=OFF"):
            assert required in flags, (required, flags)


class TestTheCommentMatchesTheCommand:
    """A claim in a comment is part of the deliverable."""

    def test_the_rejected_values_are_named_and_not_used(self, dockerfile_text):
        """The note explains why not `recurse` and why not `line`. Both
        therefore appear in the file, and neither may appear in the command."""
        assert "`=recurse`" in dockerfile_text, (
            "the note no longer explains why not --output-sync=recurse")
        command = build_command(dockerfile_text)
        for rejected in ("recurse", "line", "none", "job"):
            assert f"--output-sync={rejected}" not in command, command

    def test_every_output_sync_in_the_file_is_accounted_for(self,
                                                            dockerfile_text):
        """Exactly one `--output-sync=<x>` spelling, and it is the live one.

        The note names `=recurse` and `=line` in backticks without the
        `--output-sync` prefix, and names the flag itself without a value, so
        the fully spelled form occurs exactly once in the whole file. That is
        not cosmetic. With two occurrences, a mutation driver that replaces
        the string by text cannot reach the command without also rewriting
        the prose, and four of this file's own probes silently aborted that
        way before the duplicate was removed -- each reporting a full pass
        that was the honest value read twice.
        """
        assert output_sync_values(dockerfile_text) == ["target"], (
            output_sync_values(dockerfile_text))
        # One occurrence, not merely one distinct value: see the docstring.
        assert dockerfile_text.count("--output-sync=target") == 1, (
            dockerfile_text.count("--output-sync=target"))


# ---------------------------------------------------------------------------
# Mutation probes
# ---------------------------------------------------------------------------
#
# Each corrupts the text the four predicates read, asserts the corruption took,
# and reports the honest value beside the mutant one. An identical pair is a
# dead probe and says the predicate above it proves nothing.

class TestTheProbesDiscriminate:

    def test_dropping_the_flag_changes_what_is_read(self, dockerfile_text):
        mutant = dockerfile_text.replace(" --output-sync=target", "")
        assert mutant != dockerfile_text, "mutation did not apply"
        honest = build_command(dockerfile_text)
        mutated = build_command(mutant)
        assert honest != mutated, (honest, mutated)
        assert "--output-sync=" in honest
        assert "--output-sync=" not in mutated

    def test_swapping_target_for_recurse_changes_what_is_read(
            self, dockerfile_text):
        mutant = dockerfile_text.replace("--output-sync=target",
                                         "--output-sync=recurse")
        assert mutant != dockerfile_text, "mutation did not apply"
        assert output_sync_values(dockerfile_text) == ["target"]
        assert output_sync_values(mutant) == ["recurse"]

    def test_the_wrong_side_of_the_anchor_is_detected(self, dockerfile_text):
        """The mistake the TestTheRegionAnchorSurvived class exists for, and
        the measurement that corrected this file's first account of it.

        Moving the flag ahead of `-j` leaves a Dockerfile that builds
        byte-identical output and destroys the anchor. The two sibling suites
        do NOT fail the same way, and the difference is the reason this probe
        checks both parses:

          test_e4 asserts `"make -j" in after` before splitting, so it fails
          loudly with its own message.

          test_e1:350-351 splits with no such assertion. `str.split` on a
          missing separator returns the whole string, so its region silently
          runs from the cmake invocation to end of file -- more flags, not
          fewer, every assertion still green, and the bound it believes it
          has is gone.
        """
        mutant = dockerfile_text.replace(
            'make -j"$JOBS" --output-sync=target',
            'make --output-sync=target -j"$JOBS"')
        assert mutant != dockerfile_text, "mutation did not apply"

        # Guarded parse: refuses to produce a region at all.
        assert cmake_region(dockerfile_text).strip() != ""
        assert cmake_region(mutant) == ""
        assert region_flags(dockerfile_text) != []
        assert region_flags(mutant) == []

        # Unguarded parse: produces a LONGER region, which is the silent half.
        honest = unguarded_region(dockerfile_text)
        widened = unguarded_region(mutant)
        assert len(widened) > len(honest), (len(honest), len(widened))
        assert "--output-sync" not in honest
        assert "--output-sync" in widened
        # And it now reaches text that is not the cmake invocation at all.
        assert "ARG PERCONA_TAG" not in honest
        assert "ARG PERCONA_TAG" in widened

    def test_a_region_with_no_anchor_reads_as_empty_not_as_everything(
            self, dockerfile_text):
        """The failure mode a `split()` invites: no anchor, so take the rest
        of the file and call it the region."""
        mutant = dockerfile_text.replace('make -j"$JOBS"', 'make --jobs=4')
        assert mutant != dockerfile_text, "mutation did not apply"
        assert cmake_region(dockerfile_text).strip() != ""
        assert cmake_region(mutant) == ""
        assert region_flags(mutant) == []
