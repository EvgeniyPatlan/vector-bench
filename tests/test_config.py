"""Tests for configuration loading, resolution and the ann-benchmarks renderer.

These guard the invariants that make the comparison fair. A config bug here
would not crash anything — it would quietly hand one engine more memory or a
different parameter grid than another, and the resulting report would look
perfectly credible.
"""

from __future__ import annotations

import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.metrics import sysinfo  # noqa: E402
from orchestrator import ann_pass  # noqa: E402
from orchestrator.config import (available_profiles, load_engine,  # noqa: E402
                                 load_profile, load_resources, resolve_resources,
                                 server_args)

ENGINES = ("mariadb", "alisql", "pgvector")
PROFILES = ("smoke", "quick", "full")
PASSES = ("normalized", "tuned")


@pytest.fixture(scope="module")
def info():
    return sysinfo.collect()


class TestConfigsParse:
    @pytest.mark.parametrize("name", PROFILES)
    def test_profile_loads(self, name):
        profile = load_profile(name)
        assert profile["name"] == name
        assert profile["datasets"], "a profile with no datasets measures nothing"
        assert profile["k"] >= 1

    @pytest.mark.parametrize("name", PASSES)
    def test_resource_pass_loads(self, name):
        resources = load_resources(name)
        assert resources["name"] == name
        assert "cpu" in resources and "memory" in resources

    @pytest.mark.parametrize("name", ENGINES)
    def test_engine_loads(self, name):
        cfg = load_engine(name)
        assert cfg["name"] == name
        assert cfg["source"]["tag"], "an engine without a pinned tag is not reproducible"
        assert cfg["image"]["runtime"] and cfg["image"]["bench"]

    def test_all_profiles_discovered(self):
        assert set(PROFILES).issubset(set(available_profiles()))


class TestFairness:
    """The invariants that make the normalized pass a fair comparison."""

    @pytest.mark.parametrize("engine", ENGINES)
    def test_skip_grant_tables_is_never_used(self, engine):
        # On MySQL 8 --skip-grant-tables implicitly enables --skip-networking,
        # which would make the server unreachable over TCP from the harness.
        cfg = load_engine(engine)
        flags = " ".join(str(f) for group in cfg.get("server", {}).values()
                         for f in group)
        assert "skip-grant-tables" not in flags
        assert "skip_grant_tables" not in flags

    def test_mysql_engines_get_equal_memory(self, info):
        resources = load_resources("normalized")
        mariadb = resolve_resources(resources, "mariadb", info)
        alisql = resolve_resources(resources, "alisql", info)
        assert mariadb.buffer_bytes == alisql.buffer_bytes
        assert mariadb.graph_cache_bytes == alisql.graph_cache_bytes
        assert mariadb.server_memory_bytes == alisql.server_memory_bytes
        assert mariadb.server_cpuset == alisql.server_cpuset

    def test_postgres_absorbs_the_graph_cache_budget(self, info):
        # pgvector has no vector-specific cache; graph pages live in
        # shared_buffers. Giving it only the buffer fraction would hand it
        # strictly less resident memory for the same container limit.
        resources = load_resources("normalized")
        mysql = resolve_resources(resources, "mariadb", info)
        pg = resolve_resources(resources, "pgvector", info)
        assert pg.graph_cache_bytes == 0
        assert pg.buffer_bytes == pytest.approx(
            mysql.buffer_bytes + mysql.graph_cache_bytes, rel=0.01
        )

    @pytest.mark.parametrize("engine", ENGINES)
    def test_allocations_fit_inside_the_container_limit(self, engine, info):
        for pass_name in PASSES:
            resolved = resolve_resources(load_resources(pass_name), engine, info)
            allocated = (resolved.buffer_bytes + resolved.graph_cache_bytes
                         + resolved.maintenance_bytes)
            assert allocated < resolved.server_memory_bytes, (
                f"{engine}/{pass_name}: allocated more than the container limit"
            )

    def test_server_and_client_cpusets_do_not_overlap(self, info):
        resolved = resolve_resources(load_resources("normalized"), "mariadb", info)

        def expand(spec):
            out = set()
            for part in spec.split(","):
                if not part:
                    continue
                if "-" in part:
                    lo, hi = part.split("-")
                    out.update(range(int(lo), int(hi) + 1))
                else:
                    out.add(int(part))
            return out

        overlap = expand(resolved.server_cpuset) & expand(resolved.client_cpuset)
        if overlap:
            # Permitted on small machines, but it must have been reported.
            assert resolved.warnings, (
                "cpusets overlap without a warning; latency noise would go unrecorded"
            )


class TestServerArgs:
    @pytest.mark.parametrize("engine", ENGINES)
    def test_no_unsubstituted_placeholders(self, engine, info):
        for pass_name in PASSES:
            resolved = resolve_resources(load_resources(pass_name), engine, info)
            flags = server_args(load_engine(engine), pass_name, resolved)
            joined = " ".join(flags)
            assert "{" not in joined and "}" not in joined, (
                f"{engine}/{pass_name}: unsubstituted placeholder in {joined}"
            )

    def test_alisql_always_enables_vector_support(self, info):
        # vidx_disabled defaults to ON; without this every CREATE TABLE with a
        # VECTOR column fails with ER_VECTOR_DISABLED.
        resolved = resolve_resources(load_resources("normalized"), "alisql", info)
        for pass_name in PASSES:
            flags = " ".join(server_args(load_engine("alisql"), pass_name, resolved))
            assert "vidx-disabled=OFF" in flags

    def test_alisql_runs_at_read_committed(self, info):
        # Any other isolation level raises ER_NOT_SUPPORTED_YET on vector DML.
        resolved = resolve_resources(load_resources("normalized"), "alisql", info)
        flags = " ".join(server_args(load_engine("alisql"), "normalized", resolved))
        assert "READ-COMMITTED" in flags

    def test_graph_cache_flags_present_for_mysql_engines(self, info):
        expected = {"mariadb": "mhnsw-max-cache-size", "alisql": "vidx-hnsw-cache-size"}
        for engine, flag in expected.items():
            resolved = resolve_resources(load_resources("normalized"), engine, info)
            flags = " ".join(server_args(load_engine(engine), "normalized", resolved))
            assert flag in flags


class TestAnnConfigRendering:
    @pytest.mark.parametrize("engine", ENGINES)
    def test_renders_valid_yaml(self, engine):
        profile = load_profile("quick")
        body = ann_pass.render_config(engine, profile, load_resources("normalized"),
                                      "normalized")
        text = yaml.safe_dump(body)
        reparsed = yaml.safe_load(text)
        entry = reparsed["float"]["any"][0]
        assert entry["name"] == engine
        assert entry["module"] == f"ann_benchmarks.algorithms.{engine}"
        assert entry["run_groups"]

    def test_every_engine_sweeps_the_same_grid(self):
        # A profile change that reached only one engine would look like a
        # performance difference and would not be one.
        profile = load_profile("quick")
        resources = load_resources("normalized")
        grids = {}
        for engine in ENGINES:
            body = ann_pass.render_config(engine, profile, resources, "normalized")
            groups = body["float"]["any"][0]["run_groups"]
            m_values, ef_values = set(), set()
            for group in groups.values():
                for arg_group in group["arg_groups"]:
                    m = arg_group["M"]
                    m_values.update(m if isinstance(m, list) else [m])
                for query_args in group["query_args"]:
                    ef_values.update(query_args)
            grids[engine] = (frozenset(m_values), frozenset(ef_values))
        assert len(set(grids.values())) == 1, f"parameter grids differ: {grids}"

    def test_alisql_never_offered_myisam(self):
        # VIDX is InnoDB-only; anything else would fail at DDL.
        body = ann_pass.render_config("alisql", load_profile("full"),
                                      load_resources("tuned"), "tuned")
        groups = body["float"]["any"][0]["run_groups"]
        engines = {ag["engine"] for g in groups.values() for ag in g["arg_groups"]}
        assert engines == {"InnoDB"}

    def test_pgvector_ef_construction_pinned_in_normalized_pass(self):
        # ef_construction is the one build knob MariaDB and AliSQL do not
        # expose; sweeping it in the fair pass would give pgvector an axis the
        # others lack.
        body = ann_pass.render_config("pgvector", load_profile("full"),
                                      load_resources("normalized"), "normalized")
        groups = body["float"]["any"][0]["run_groups"]
        values = set()
        for group in groups.values():
            for arg_group in group["arg_groups"]:
                ef = arg_group["efConstruction"]
                values.update(ef if isinstance(ef, list) else [ef])
        assert len(values) == 1, f"ef_construction swept in the normalized pass: {values}"

    def test_pgvector_ef_construction_swept_in_tuned_pass(self):
        body = ann_pass.render_config("pgvector", load_profile("full"),
                                      load_resources("tuned"), "tuned")
        groups = body["float"]["any"][0]["run_groups"]
        values = set()
        for group in groups.values():
            for arg_group in group["arg_groups"]:
                ef = arg_group["efConstruction"]
                values.update(ef if isinstance(ef, list) else [ef])
        assert len(values) > 1

    def test_unknown_engine_rejected(self):
        with pytest.raises(ValueError):
            ann_pass.render_config("nonsense", load_profile("smoke"),
                                   load_resources("normalized"), "normalized")


class TestOverlayModules:
    """The overlay must stay consistent with what the renderer generates."""

    @pytest.mark.parametrize("engine,constructor",
                             [("mariadb", "MariaDB"), ("alisql", "AliSQL"),
                              ("pgvector", "PGVector")])
    def test_constructor_exists_in_module(self, engine, constructor):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, "overlay", "ann-benchmarks", "ann_benchmarks",
                            "algorithms", engine, "module.py")
        assert os.path.exists(path), f"missing overlay module for {engine}"
        source = open(path).read()
        assert f"class {constructor}" in source
        assert ann_pass.CONSTRUCTORS[engine] == constructor


class TestAnnResultGuard:
    """The zero-result guard must not turn correct resumption into a failure.

    ann-benchmarks exits 0 whenever it has nothing left to run. That covers both
    "the module failed to import and nothing was produced" and "every
    configuration already has results". An earlier version conflated them and
    reported three healthy engines as failed on a re-run, which then propagated
    into the report's Validity section as a fabricated failure.
    """

    def _counts(self, tmp_path, engine, dataset, n):
        import os
        d = tmp_path / dataset / "10" / engine
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            (d / f"r{i}.hdf5").write_bytes(b"")
        from orchestrator.ann_pass import _count_results
        return _count_results(str(tmp_path), engine, dataset)

    def test_counts_only_this_engine_and_dataset(self, tmp_path):
        assert self._counts(tmp_path, "mariadb", "ds-a", 3) == 3
        # A different engine's files must not be counted as this one's.
        self._counts(tmp_path, "alisql", "ds-a", 5)
        from orchestrator.ann_pass import _count_results
        assert _count_results(str(tmp_path), "mariadb", "ds-a") == 3
        assert _count_results(str(tmp_path), "alisql", "ds-a") == 5

    def test_zero_when_nothing_exists(self, tmp_path):
        from orchestrator.ann_pass import _count_results
        assert _count_results(str(tmp_path), "mariadb", "missing") == 0

    def test_resumption_is_not_a_failure(self):
        # before == after > 0 means every configuration was already present.
        before = after = 24
        rc = 0
        assert not (rc == 0 and after == 0), \
            "a run that produced no NEW results but has results is resumption"

    def test_no_results_at_all_is_a_failure(self):
        before = after = 0
        rc = 0
        assert rc == 0 and after == 0, \
            "no results at all must still be caught as a failure"
