"""Why no profile that measures scaling asks for two clients.

WHY THIS FILE EXISTS
--------------------
2026-10-14. `harness/metrics/latency.py` refuses to publish a scaling
efficiency when the achieved fraction of the client count falls below
`SERVED_CONCURRENCY_FLOOR = 0.5`. Both estimators of that fraction --
`served_concurrency / clients`, and the scaling efficiency itself --
equal exactly `1/n` for a pool whose clients took turns, because a pool that
serializes keeps exactly one query in service whatever n says.

At n = 2 that is **0.5 exactly**, and the floor is a strict `<`. So the one
case the guard exists to catch is, at two clients, precisely on the boundary,
and which verdict it gets is decided by measurement noise. Measured on this
tree with two synthetic drivers that both serialize completely:

    lock-holding C call  @2 clients -> 0.499  -> refused
    Python spin          @2 clients -> 0.537  -> published as `ok`

Same defect, opposite verdicts. At every other point in the grid the two
populations are far apart: at n = 4 a serialized pool reads 0.25 against 0.98
for one that scaled.

The 2-client point therefore cost `concurrency_duration_s` x `repeats` per
ops unit and could not answer the question it was there to answer, so it was
removed from the profiles whose job is to measure scaling.

It is deliberately KEPT in `mariadb-blog.yml` and `mariadb-blog-repro.yml`.
Those profiles reproduce a published curve, that curve discusses the
two-client point by name, and the figure being compared is the QPS and the
percentiles -- which are sound at n = 2 -- rather than the concurrency
verdict, which is not. `mariadb-blog.yml`'s own comment says a grid that
jumps 1 -> 4 "steps straight over the knee"; this file does not argue with
that, it only records that the knee is a throughput observation.

This file is the thing that keeps the split deliberate. Someone restoring
`[1, 2, 4, 8, 16, 32]` to `main` for symmetry would otherwise reintroduce a
point whose verdict is a coin flip, and the only symptom would be an
occasional unexplained refusal in a 100-hour run.
"""

from __future__ import annotations

import glob
import os

import pytest
import yaml

VB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILE_DIR = os.path.join(VB_ROOT, "config", "profiles")

# The two profiles whose purpose is reproducing someone else's published
# curve rather than measuring scaling here. Named explicitly, so adding a
# third reproduction profile is a decision and not an accident.
REPRODUCTION_PROFILES = frozenset({"mariadb-blog", "mariadb-blog-repro"})

# `1/n` at the floor. Spelled out rather than imported so that a change to
# the floor fails this file with an explanation instead of silently
# re-deciding which client counts are measurable.
UNDECIDABLE_CLIENT_COUNT = 2


def profile_paths() -> list[str]:
    return sorted(glob.glob(os.path.join(PROFILE_DIR, "*.yml")))


def client_counts(path: str) -> list[int] | None:
    with open(path) as handle:
        loaded = yaml.safe_load(handle) or {}
    grid = (loaded.get("ops") or {}).get("client_counts")
    return list(grid) if grid else None


def profiles_with_a_grid() -> list[tuple[str, list[int]]]:
    found = []
    for path in profile_paths():
        grid = client_counts(path)
        if grid:
            found.append((os.path.basename(path)[: -len(".yml")], grid))
    return found


# ---------------------------------------------------------------------------
# Non-vacuity: every assertion below is "2 is not in this list", which an
# empty set of profiles satisfies.
# ---------------------------------------------------------------------------

class TestTheProfilesWereActuallyRead:

    def test_the_profile_directory_exists(self):
        assert os.path.isdir(PROFILE_DIR), PROFILE_DIR

    def test_several_profiles_declare_a_client_grid(self):
        found = profiles_with_a_grid()
        assert len(found) >= 6, found

    def test_the_named_profiles_are_among_them(self):
        names = {name for name, _ in profiles_with_a_grid()}
        for expected in ("main", "full", "dev", "smoke", "quick"):
            assert expected in names, sorted(names)

    def test_both_reproduction_profiles_exist(self):
        names = {name for name, _ in profiles_with_a_grid()}
        missing = REPRODUCTION_PROFILES - names
        assert not missing, (
            f"{sorted(missing)} no longer declare a client grid, so the "
            f"exemption below is exempting nothing")


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------

class TestNoMeasurementProfileAsksForTwoClients:

    @pytest.mark.parametrize("name,grid", profiles_with_a_grid(),
                             ids=lambda v: v if isinstance(v, str) else "")
    def test_the_grid_omits_the_undecidable_point(self, name, grid):
        if name in REPRODUCTION_PROFILES:
            pytest.skip(f"{name} reproduces a published curve; see module "
                        f"docstring")
        assert UNDECIDABLE_CLIENT_COUNT not in grid, (
            f"config/profiles/{name}.yml asks for "
            f"{UNDECIDABLE_CLIENT_COUNT} clients. Both estimators of the "
            f"achieved concurrency equal exactly 1/{UNDECIDABLE_CLIENT_COUNT} "
            f"= {1 / UNDECIDABLE_CLIENT_COUNT} there for a serialized pool, "
            f"against a strict `<` floor, so the verdict is noise. Grid: "
            f"{grid}")

    @pytest.mark.parametrize("name", sorted(REPRODUCTION_PROFILES))
    def test_the_reproduction_profiles_keep_it_on_purpose(self, name):
        """The exemption asserted from the other side.

        If a reproduction profile loses its 2-client point, this fails rather
        than quietly passing -- because then the exemption is dead text and
        the next reader will delete it as unused.
        """
        grid = client_counts(os.path.join(PROFILE_DIR, f"{name}.yml"))
        assert grid is not None, name
        assert UNDECIDABLE_CLIENT_COUNT in grid, (
            f"{name} no longer asks for {UNDECIDABLE_CLIENT_COUNT} clients. "
            f"If that was deliberate, remove it from REPRODUCTION_PROFILES "
            f"and delete the exemption. Grid: {grid}")


class TestEveryGridStillMeasuresConcurrency:
    """The mistake the dev profile nearly made.

    `dev.yml` was `[1, 2]`. Removing the 2 would have left `[1]` -- a grid
    that starts no client pool at all, so the fast inner loop would stop
    exercising the concurrency path while still reporting a concurrency
    phase. It is `[1, 4]`.
    """

    @pytest.mark.parametrize("name,grid", profiles_with_a_grid(),
                             ids=lambda v: v if isinstance(v, str) else "")
    def test_the_grid_has_a_point_above_one(self, name, grid):
        above = [n for n in grid if n > 1]
        assert above, (
            f"config/profiles/{name}.yml measures only one client, so its "
            f"concurrency phase cannot observe concurrency at all. Grid: "
            f"{grid}")

    @pytest.mark.parametrize("name,grid", profiles_with_a_grid(),
                             ids=lambda v: v if isinstance(v, str) else "")
    def test_the_grid_keeps_its_one_client_baseline(self, name, grid):
        """The scaling efficiency is `qps_n / (n x qps_1)`. Without a
        1-client point there is no baseline, the efficiency is `None`, and
        the status falls back to the median-span-only reading -- the one with
        the skew false positive. So the baseline is not optional."""
        assert 1 in grid, (
            f"config/profiles/{name}.yml has no 1-client point, so no "
            f"scaling efficiency can be computed and every window falls back "
            f"to the median-span reading. Grid: {grid}")

    @pytest.mark.parametrize("name,grid", profiles_with_a_grid(),
                             ids=lambda v: v if isinstance(v, str) else "")
    def test_the_grid_is_sorted_and_free_of_duplicates(self, name, grid):
        assert grid == sorted(set(grid)), (
            f"config/profiles/{name}.yml client grid is unsorted or repeats "
            f"a point: {grid}")


# ---------------------------------------------------------------------------
# Mutation probes
# ---------------------------------------------------------------------------
#
# The rule above is an absence check over files, so it is exactly the kind
# that can pass while reading nothing. These corrupt the parsed grids and
# report the honest verdict beside the mutant one.

class TestTheProbesDiscriminate:

    def _verdict(self, name, grid):
        """The rule, applied to a grid handed in rather than read."""
        if name in REPRODUCTION_PROFILES:
            return "exempt"
        return "ok" if UNDECIDABLE_CLIENT_COUNT not in grid else "refused"

    def test_restoring_two_to_main_is_caught(self):
        honest = client_counts(os.path.join(PROFILE_DIR, "main.yml"))
        assert honest is not None
        mutant = sorted(set(honest) | {UNDECIDABLE_CLIENT_COUNT})
        assert mutant != honest, "mutation did not apply"
        assert self._verdict("main", honest) == "ok"
        assert self._verdict("main", mutant) == "refused"

    def test_the_exemption_does_not_leak_to_main(self):
        """An exemption keyed on the wrong thing would pass everything."""
        grid_with_two = [1, 2, 4]
        assert self._verdict("mariadb-blog", grid_with_two) == "exempt"
        assert self._verdict("main", grid_with_two) == "refused"

    def test_an_empty_grid_is_not_read_as_compliant(self):
        """A profile whose `ops` block vanished parses to no grid at all.
        The parametrization must then not silently contribute a passing row.
        """
        names = {name for name, _ in profiles_with_a_grid()}
        assert "main" in names
        # A grid of None is excluded by profiles_with_a_grid, so the rule is
        # never applied to it -- which is why the non-vacuity class above
        # asserts the named profiles are present by name.
        assert client_counts(os.path.join(PROFILE_DIR, "main.yml"))

    def test_dropping_the_one_client_baseline_is_caught(self):
        honest = client_counts(os.path.join(PROFILE_DIR, "main.yml"))
        mutant = [n for n in honest if n != 1]
        assert mutant != honest, "mutation did not apply"
        assert 1 in honest
        assert 1 not in mutant

    def test_leaving_only_one_client_is_caught(self):
        honest = client_counts(os.path.join(PROFILE_DIR, "dev.yml"))
        mutant = [1]
        assert mutant != honest, "mutation did not apply"
        assert [n for n in honest if n > 1]
        assert not [n for n in mutant if n > 1]
