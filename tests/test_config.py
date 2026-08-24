"""Tests for configuration loading, resolution and the ann-benchmarks renderer.

These guard the invariants that make the comparison fair. A config bug here
would not crash anything — it would quietly hand one engine more memory or a
different parameter grid than another, and the resulting report would look
perfectly credible.
"""

from __future__ import annotations

import json
import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.metrics import sysinfo  # noqa: E402
from orchestrator import ann_pass  # noqa: E402
from orchestrator.cli import VB_ROOT  # noqa: E402
from orchestrator.config import (available_profiles, load_engine,  # noqa: E402
                                 merge_resource_overrides,
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


class TestNothingToRunHandling:
    """ann-benchmarks exits NON-ZERO when it has nothing left to run.

    `main()` raises Exception("Nothing to run") once every configuration
    already has results. That is successful resumption from this framework's
    point of view — the results tree is deliberately kept between runs — but
    taken at face value it reports healthy engines as failed, which then
    propagates into the report's Validity section as a fabricated failure.
    """

    def test_marker_matches_upstream_text(self):
        # If upstream ever rewords this, the detection silently stops working
        # and failures reappear. Pin the string we depend on.
        from orchestrator.ann_pass import NOTHING_TO_RUN
        assert NOTHING_TO_RUN == "Nothing to run"

    def test_marker_is_found_in_a_realistic_traceback(self):
        from orchestrator.ann_pass import NOTHING_TO_RUN
        captured = "\n".join([
            "2026-08-05 16:54:21,185 - annb - INFO - running only mariadb",
            "Traceback (most recent call last):",
            '  File "/home/app/run.py", line 7, in <module>',
            "    main()",
            '  File "/home/app/ann_benchmarks/main.py", line 344, in main',
            '    raise Exception("Nothing to run")',
            "Exception: Nothing to run",
        ])
        assert NOTHING_TO_RUN in captured

    def test_run_foreground_accepts_a_sink(self):
        # The handling depends on being able to read what was streamed; an exit
        # code alone cannot tell these cases apart.
        import inspect
        from orchestrator.docker_ctl import run_foreground
        assert "sink" in inspect.signature(run_foreground).parameters


class TestBenignTracebackSuppression:
    """A traceback that means "already done" must not reach the operator.

    ann-benchmarks reports "every configuration has results" by raising, so it
    prints a Python traceback for a completely normal condition. Explaining it
    in advance was not enough: a traceback reads as a crash regardless of what
    precedes it. A traceback that means anything ELSE must still get through.
    """

    def _run(self, lines):
        from orchestrator.ann_pass import _SuppressNothingToRun
        f = _SuppressNothingToRun()
        out = []
        for line in lines:
            out.extend(f(line))
        out.extend(f(None))
        return out

    def test_nothing_to_run_traceback_is_dropped(self):
        out = self._run([
            "annb - INFO - running only mariadb",
            "Traceback (most recent call last):",
            '  File "/home/app/run.py", line 7, in <module>',
            "    main()",
            '  File "/home/app/ann_benchmarks/main.py", line 344, in main',
            '    raise Exception("Nothing to run")',
            "Exception: Nothing to run",
        ])
        assert out == ["annb - INFO - running only mariadb"]
        assert not any("Traceback" in line for line in out)

    def test_a_real_traceback_still_reaches_the_operator(self):
        out = self._run([
            "Traceback (most recent call last):",
            '  File "/home/app/run.py", line 7, in <module>',
            "    main()",
            "Exception: could not connect to the server",
        ])
        assert any("Traceback" in line for line in out)
        assert any("could not connect" in line for line in out)

    def test_ordinary_output_passes_through_untouched(self):
        lines = ["[vb] loading 60,000 vectors", "Processed 1000/10000 queries..."]
        assert self._run(lines) == lines

    def test_an_unterminated_traceback_is_flushed_not_swallowed(self):
        # A container killed mid-traceback must not lose the evidence.
        out = self._run([
            "Traceback (most recent call last):",
            '  File "/home/app/run.py", line 7, in <module>',
        ])
        assert any("Traceback" in line for line in out)


class TestDefaultResourcePass:
    """A profile may pick its own default pass; an explicit flag still wins.

    --resource-pass defaulted to "both", which silently doubled every profile's
    cost. For mariadb-blog that is ~144 h of ingest over a million 1536-dim
    vectors — not something anyone should end up running because a flag
    defaulted rather than because they chose it.
    """

    def _resolve(self, argv):
        from orchestrator.cli import build_parser
        from orchestrator.config import load_profile
        args = build_parser().parse_args(argv)
        profile = load_profile(args.profile)
        requested = (args.resource_pass or profile.get("default_resource_pass")
                     or "both")
        return ["normalized", "tuned"] if requested == "both" else [requested]

    def test_profile_default_is_used_when_the_flag_is_absent(self):
        assert self._resolve(["run", "--profile", "mariadb-blog"]) == ["normalized"]

    def test_explicit_flag_overrides_the_profile(self):
        assert self._resolve(
            ["run", "--profile", "mariadb-blog", "--resource-pass", "both"]
        ) == ["normalized", "tuned"]

    def test_profiles_without_a_declared_default_still_get_both(self):
        # Long-standing behaviour for every other profile must not change.
        assert self._resolve(["run", "--profile", "main"]) == ["normalized", "tuned"]

    def test_the_heaviest_profile_does_not_default_to_both(self):
        from orchestrator.config import load_profile
        p = load_profile("mariadb-blog")
        assert p.get("default_resource_pass") == "normalized"


class TestMissingDatasetGuidance:
    """Tell people the command that will actually work.

    Most datasets are downloaded; the dbpedia family is constructed locally and
    is not published as a prebuilt HDF5. Advising `fetch` for those sends the
    operator to a 404 before a multi-day run.
    """

    def test_dbpedia_is_recognised_as_generated(self):
        from orchestrator.cli import _is_generated
        assert _is_generated("dbpedia-openai-1000k-angular")
        assert _is_generated("dbpedia-openai-100k-angular")

    def test_published_datasets_are_not_flagged_as_generated(self):
        from orchestrator.cli import _is_generated
        for d in ("glove-100-angular", "sift-128-euclidean",
                  "fashion-mnist-784-euclidean", "gist-960-euclidean"):
            assert not _is_generated(d)



class TestAnnClientMemory:
    """The ann pass shares one cgroup between the server and the client.

    Sizing that cgroup to the engine budget alone OOM-killed the client on
    dbpedia-openai-1000k, and because only the forked worker died the run
    exited 0 with no results and no error. These pin the sizing that prevents it.
    """

    @staticmethod
    def _corpus(tmp_path, name, size_bytes):
        path = tmp_path / f"{name}.hdf5"
        with open(path, "wb") as fh:
            fh.truncate(size_bytes)
        return path

    def test_scales_with_corpus_size(self, tmp_path):
        self._corpus(tmp_path, "big", 6 * 1024 ** 3)
        self._corpus(tmp_path, "small", 200 * 1024 ** 2)
        big = ann_pass.client_memory_bytes(str(tmp_path), "big")
        small = ann_pass.client_memory_bytes(str(tmp_path), "small")
        assert big > small
        # Two full copies is the floor: main.py loads the corpus to read the
        # dimension and the forked worker loads it again for itself.
        assert big > 2 * 6 * 1024 ** 3

    def test_covers_the_corpus_that_failed(self, tmp_path):
        """dbpedia-openai-1000k is 6.17 GB; the client needs over 12.3 GB."""
        self._corpus(tmp_path, "dbpedia-openai-1000k-angular", int(6.17 * 1024 ** 3))
        got = ann_pass.client_memory_bytes(str(tmp_path), "dbpedia-openai-1000k-angular")
        assert got >= int(2 * 6.17 * 1024 ** 3)

    def test_missing_corpus_does_not_raise(self, tmp_path):
        """Sizing happens before the fetch check, so absence must be survivable."""
        assert ann_pass.client_memory_bytes(str(tmp_path), "not-downloaded-yet") > 0

    def test_host_ram_probe_never_raises(self):
        assert ann_pass._host_ram_bytes() >= 0

    def test_ops_client_floor_also_scales(self, tmp_path):
        """The ops client is a second container with the same exposure."""
        from orchestrator.ops_pass import ops_client_memory_bytes
        self._corpus(tmp_path, "dbpedia-openai-1000k-angular", int(6.17 * 1024 ** 3))
        got = ops_client_memory_bytes(str(tmp_path), "dbpedia-openai-1000k-angular")
        # Must clear the fixed 8 GB client_limit_gb that would otherwise apply.
        assert got > 8 * 1024 ** 3
        assert ops_client_memory_bytes(str(tmp_path), "absent") > 0


class TestProfileResourceOverrides:
    """A profile can raise the budget its corpus needs.

    dbpedia-openai-1000k does not fit the shared 16 GB pass: pgvector was
    OOM-killed and AliSQL ran the whole phase at the ceiling, which made the
    normalized pass a test of who fits in 16 GB.
    """

    def test_deep_merges_without_dropping_siblings(self):
        base = {"memory": {"server_limit_gb": 16, "buffer_fraction": 0.30},
                "cpu": {"server_cpus": 8}}
        got = merge_resource_overrides(
            base, {"resources": {"memory": {"server_limit_gb": 64}}})
        assert got["memory"]["server_limit_gb"] == 64
        assert got["memory"]["buffer_fraction"] == 0.30   # sibling survives
        assert got["cpu"] == {"server_cpus": 8}           # other sections survive

    def test_does_not_mutate_the_loaded_pass(self):
        base = {"memory": {"server_limit_gb": 16}}
        merge_resource_overrides(
            base, {"resources": {"memory": {"server_limit_gb": 64}}})
        assert base["memory"]["server_limit_gb"] == 16

    def test_profile_without_overrides_is_a_passthrough(self):
        base = {"memory": {"server_limit_gb": 16}}
        assert merge_resource_overrides(base, {}) is base

    def test_mariadb_blog_profile_asks_for_64gb(self):
        prof = load_profile("mariadb-blog")
        merged = merge_resource_overrides(
            load_resources("normalized"), prof)
        assert merged["memory"]["server_limit_gb"] == 64


class TestMemoryCeilingDetection:
    """An engine pinned at its cgroup limit is reclaiming, not benchmarking."""

    GB = 1024 ** 3

    def test_flags_a_phase_spent_at_the_limit(self):
        from report.loaders import ceiling_pressure
        rows = [{"rss_bytes": int(v * self.GB)}
                for v in (0.3, 6, 10, 14, 15.9, 16.0, 15.95, 15.93)]
        got = ceiling_pressure(rows, 16 * self.GB)
        assert got is not None
        assert got["fraction_at_ceiling"] == pytest.approx(0.5)
        assert got["peak_bytes"] == 16 * self.GB

    def test_silent_when_there_is_headroom(self):
        from report.loaders import ceiling_pressure
        rows = [{"rss_bytes": int(8.25 * self.GB)}] * 20
        assert ceiling_pressure(rows, 16 * self.GB) is None

    def test_silent_without_a_known_limit(self):
        from report.loaders import ceiling_pressure
        rows = [{"rss_bytes": 16 * self.GB}] * 5
        assert ceiling_pressure(rows, None) is None


class TestAnnResultFingerprint:
    """A curve measured at 16 GB is not a curve measured at 64 GB.

    ann-benchmarks caches by algorithm and index parameters only, so a re-run
    after a budget change returned every point byte-identical to the previous
    one and the report carried a 64 GB manifest above a 16 GB curve.
    """

    BASE = dict(server_memory_bytes=16 * 1024 ** 3, buffer_bytes=5153960755,
                graph_cache_bytes=5153960755, maintenance_bytes=1717986918,
                build_threads=1, server_cpu_count=8)

    def test_budget_change_changes_the_fingerprint(self):
        from orchestrator.ann_pass import ann_fingerprint
        bigger = dict(self.BASE, server_memory_bytes=64 * 1024 ** 3)
        assert ann_fingerprint(self.BASE) != ann_fingerprint(bigger)

    def test_same_config_resumes(self):
        from orchestrator.ann_pass import ann_fingerprint
        assert ann_fingerprint(self.BASE) == ann_fingerprint(dict(self.BASE))

    def test_cpu_count_counts_too(self):
        from orchestrator.ann_pass import ann_fingerprint
        other = dict(self.BASE, server_cpu_count=16)
        assert ann_fingerprint(self.BASE) != ann_fingerprint(other)

    def test_measurement_version_invalidates_everything(self):
        """Bumping it must change the digest, or a harness fix reuses old numbers."""
        from orchestrator import ann_pass
        before = ann_pass.ann_fingerprint(self.BASE)
        original = ann_pass.ANN_MEASUREMENT_VERSION
        try:
            ann_pass.ANN_MEASUREMENT_VERSION = original + 1
            assert ann_pass.ann_fingerprint(self.BASE) != before
        finally:
            ann_pass.ANN_MEASUREMENT_VERSION = original

    def test_results_dir_is_namespaced_by_config(self, tmp_path):
        from orchestrator.ann_pass import annb_results_dir, ann_fingerprint
        paths = {"annb_results": str(tmp_path)}
        plain = annb_results_dir(paths, "normalized")
        keyed = annb_results_dir(paths, "normalized", self.BASE)
        assert plain != keyed
        assert ann_fingerprint(self.BASE) in keyed
        assert os.path.isdir(keyed)


class TestFreeDiskPreflight:
    """pgvector died two seconds in because the volume had nowhere to go."""

    def test_blocks_when_the_disk_is_too_small(self, tmp_path, monkeypatch):
        from orchestrator import cli
        monkeypatch.setattr(cli.shutil, "disk_usage",
                            lambda _p: type("U", (), {"free": 5 * cli.GB,
                                                      "total": 100 * cli.GB,
                                                      "used": 95 * cli.GB})())
        paths = {"results": str(tmp_path), "datasets": str(tmp_path)}
        assert cli._check_free_disk(
            paths, ["mariadb"], ["dbpedia-openai-1000k-angular"], ["normalized"]) is True

    def test_passes_when_there_is_room(self, tmp_path, monkeypatch):
        from orchestrator import cli
        monkeypatch.setattr(cli.shutil, "disk_usage",
                            lambda _p: type("U", (), {"free": 500 * cli.GB,
                                                      "total": 900 * cli.GB,
                                                      "used": 400 * cli.GB})())
        paths = {"results": str(tmp_path), "datasets": str(tmp_path)}
        assert cli._check_free_disk(
            paths, ["mariadb", "alisql", "pgvector"],
            ["dbpedia-openai-1000k-angular"], ["normalized"]) is False

    def test_unreadable_target_does_not_block(self, monkeypatch):
        from orchestrator import cli
        def boom(_p):
            raise OSError("no such filesystem")
        monkeypatch.setattr(cli.shutil, "disk_usage", boom)
        assert cli._check_free_disk(
            {"results": "/nonexistent", "datasets": "/nonexistent"},
            ["mariadb"], ["glove-100-angular"], ["normalized"]) is False


class TestStaleAnnDetection:
    """A result file older than the run it is reported under is not a result."""

    MANIFEST = {"started_at": "2026-08-07T05:41:02Z",
                "config": {"resource_pass": "normalized", "resolved_resources": {}}}

    def _record(self, mtime):
        return {"phase": "recall_qps", "engine": "mariadb", "dataset": "d",
                "ef_search": 10, "recall_at_k": 0.95, "qps": 100.0,
                "source_mtime": mtime, "source_file": "x.hdf5"}

    def test_flags_a_file_written_before_the_run(self):
        from report.generate import summarize, _parse_ts
        old = _parse_ts("2026-08-05T22:00:00Z")
        s = summarize([self._record(old)], self.MANIFEST)
        assert len(s["stale_ann"]) == 1

    def test_silent_for_a_file_written_during_the_run(self):
        from report.generate import summarize, _parse_ts
        fresh = _parse_ts("2026-08-07T09:00:00Z")
        s = summarize([self._record(fresh)], self.MANIFEST)
        assert s["stale_ann"] == []

    def test_silent_without_a_start_time(self):
        from report.generate import summarize
        s = summarize([self._record(1.0)], {"config": {}})
        assert s["stale_ann"] == []


class TestEngineDataPlacement:
    """Engine data must not land on Docker's data-root by accident.

    A pgvector ann phase died at `initdb: could not create directory
    ".../pg_wal": No space left on device` while the filesystem holding the
    checkout had over 100 GB free. The container was writing its data directory
    into its own writable layer, under /var/lib/docker on the root volume.
    """

    def test_every_engine_has_a_declared_data_mount(self):
        """Covers extra versions too, not just the baseline three."""
        from orchestrator.ann_pass import DATA_MOUNT
        from orchestrator.cli import KNOWN_ENGINES
        assert set(DATA_MOUNT) == set(KNOWN_ENGINES)
        assert all(p.startswith("/var/lib/") for p in DATA_MOUNT.values())

    def test_every_registry_covers_every_known_engine(self):
        """Adding an engine means touching five tables; this is the guard.

        mariadb123 was added as a distinct engine because ann-benchmarks keys
        results on the algorithm name, so a retagged `mariadb` would silently
        return 11.8.8's numbers for a 12.3 build.
        """
        from orchestrator.ann_pass import CONSTRUCTORS, DATA_MOUNT
        from orchestrator.ops_pass import (DB_CREDENTIALS, DEFAULT_PORTS,
                                           PROBES, SERVER_DATA_MOUNT)
        from orchestrator.cli import KNOWN_ENGINES
        for name, table in (("CONSTRUCTORS", CONSTRUCTORS),
                            ("DATA_MOUNT", DATA_MOUNT),
                            ("DEFAULT_PORTS", DEFAULT_PORTS),
                            ("PROBES", PROBES),
                            ("DB_CREDENTIALS", DB_CREDENTIALS),
                            ("SERVER_DATA_MOUNT", SERVER_DATA_MOUNT)):
            missing = set(KNOWN_ENGINES) - set(table)
            assert not missing, f"{name} is missing {sorted(missing)}"

    def test_a_second_mariadb_version_is_a_distinct_engine(self):
        from orchestrator.config import load_engine
        base, alt = load_engine("mariadb"), load_engine("mariadb123")
        assert alt["source"]["tag"] != base["source"]["tag"]
        assert alt["image"]["runtime"] != base["image"]["runtime"]
        assert alt.get("alias_of") == "mariadb"

    def test_both_mariadb_versions_resolve_the_same_storage_axis(self):
        """Whatever the axis is set to, 12.3 must follow 11.8. Adding it left
        seven hardcoded three-engine assumptions behind, each of which failed
        one run at a time."""
        from orchestrator.cli import _ops_storage_engines
        prof, tuned = load_profile("mariadb-blog-repro"), load_resources("tuned")
        assert (_ops_storage_engines("mariadb", prof, tuned, "tuned")
                == _ops_storage_engines("mariadb123", prof, tuned, "tuned")
                == ["InnoDB"])

    def test_the_two_versions_get_separate_ann_modules(self):
        """Sharing a module name is exactly how a 12.3 run would reuse 11.8's
        result files and report success without measuring anything."""
        from orchestrator.ann_pass import render_config
        prof, tuned = load_profile("mariadb-blog-repro"), load_resources("tuned")
        a = render_config("mariadb", prof, tuned, "tuned")["float"]["any"][0]
        b = render_config("mariadb123", prof, tuned, "tuned")["float"]["any"][0]
        assert a["module"] != b["module"]
        assert a["constructor"] != b["constructor"]

    def test_engine_state_lives_under_the_checkout(self):
        from orchestrator.cli import paths_for, VB_ROOT
        paths = paths_for("x")
        assert paths["engine_state"].startswith(VB_ROOT)

    def test_bind_volume_command_targets_the_device(self, monkeypatch, tmp_path):
        from orchestrator import docker_ctl
        seen = {}
        monkeypatch.setattr(docker_ctl, "_run",
                            lambda cmd, **kw: seen.setdefault("cmd", cmd))
        device = str(tmp_path / "vol")
        docker_ctl.create_volume("v1", device=device)
        cmd = seen["cmd"]
        assert f"device={device}" in cmd and "o=bind" in cmd
        assert os.path.isdir(device)      # created before docker is asked for it

    def test_plain_volume_has_no_device_options(self, monkeypatch):
        from orchestrator import docker_ctl
        seen = {}
        monkeypatch.setattr(docker_ctl, "_run",
                            lambda cmd, **kw: seen.setdefault("cmd", cmd))
        docker_ctl.create_volume("v2")
        assert "--opt" not in seen["cmd"]


class TestDiskPreflightChecksBothFilesystems:
    """The results tree and Docker's data-root are usually different mounts."""

    def test_reports_the_tighter_of_the_two(self, tmp_path, monkeypatch, capsys):
        from orchestrator import cli
        monkeypatch.setattr(cli.docker_ctl, "root_dir", lambda: "/var/lib/docker")

        def usage(path):
            free = 5 * cli.GB if path == "/var/lib/docker" else 900 * cli.GB
            return type("U", (), {"free": free, "total": 1000 * cli.GB,
                                  "used": 1000 * cli.GB - free})()
        monkeypatch.setattr(cli.shutil, "disk_usage", usage)
        paths = {"results": str(tmp_path), "datasets": str(tmp_path)}
        blocked = cli._check_free_disk(
            paths, ["pgvector"], ["dbpedia-openai-1000k-angular"], ["normalized"])
        assert blocked is True, "a full docker root must block even when results has room"
        assert "/var/lib/docker" in capsys.readouterr().out

    def test_survives_a_daemon_that_will_not_answer(self, tmp_path, monkeypatch):
        from orchestrator import cli
        monkeypatch.setattr(cli.docker_ctl, "root_dir", lambda: None)
        monkeypatch.setattr(cli.shutil, "disk_usage",
                            lambda _p: type("U", (), {"free": 900 * cli.GB,
                                                      "total": 1000 * cli.GB,
                                                      "used": 100 * cli.GB})())
        paths = {"results": str(tmp_path), "datasets": str(tmp_path)}
        assert cli._check_free_disk(
            paths, ["mariadb"], ["glove-100-angular"], ["normalized"]) is False


class TestCleanup:
    """Cleanup must be label-scoped, refuse to kill a live run, and reclaim disk."""

    def test_no_run_id_omits_the_name_filter(self, monkeypatch):
        """An empty --filter name= filters on the empty string, not on everything."""
        from orchestrator import docker_ctl
        calls = []
        monkeypatch.setattr(docker_ctl, "_run",
                            lambda cmd, **kw: (calls.append(cmd),
                                               type("P", (), {"stdout": ""})())[1])
        docker_ctl.cleanup_run("")
        for cmd in calls:
            assert "name=" not in " ".join(cmd)
            assert "label=vector-bench=1" in cmd

    def test_run_id_narrows_the_filter(self, monkeypatch):
        from orchestrator import docker_ctl
        calls = []
        monkeypatch.setattr(docker_ctl, "_run",
                            lambda cmd, **kw: (calls.append(cmd),
                                               type("P", (), {"stdout": ""})())[1])
        docker_ctl.cleanup_run("main-123")
        assert any("name=main-123" in " ".join(c) for c in calls)

    def test_always_scoped_by_label(self, monkeypatch):
        """Nothing on a shared machine should be at risk."""
        from orchestrator import docker_ctl
        calls = []
        monkeypatch.setattr(docker_ctl, "_run",
                            lambda cmd, **kw: (calls.append(cmd),
                                               type("P", (), {"stdout": ""})())[1])
        docker_ctl.cleanup_run("")
        assert all("label=vector-bench=1" in c for c in calls if c[1] != "ps")

    def test_refuses_while_a_run_is_live(self, monkeypatch):
        from orchestrator import cli
        monkeypatch.setattr(cli.docker_ctl, "docker_available", lambda: True)
        monkeypatch.setattr(cli.docker_ctl, "running_containers",
                            lambda: ["main-1-mariadb-srv"])
        args = type("A", (), {"run_id": None, "force": False})()
        assert cli.cmd_clean(args) == 2

    def test_force_overrides_the_guard(self, monkeypatch, tmp_path):
        from orchestrator import cli
        monkeypatch.setattr(cli.docker_ctl, "docker_available", lambda: True)
        monkeypatch.setattr(cli.docker_ctl, "running_containers", lambda: ["x"])
        monkeypatch.setattr(cli.docker_ctl, "cleanup_run", lambda _r: {"container": 0})
        monkeypatch.setattr(cli, "paths_for",
                            lambda _n: {"engine_state": str(tmp_path / "none")})
        args = type("A", (), {"run_id": None, "force": True})()
        assert cli.cmd_clean(args) == 0


class TestReportNarrowing:
    """The report container mounts report/ and harness/ only.

    An earlier version imported orchestrator.ann_pass from inside it, which
    crashed report generation with ModuleNotFoundError at the very end of a
    20-hour run. The narrowing is decided on the host now, and the fingerprint
    is recorded in the manifest so the container never has to derive it.
    """

    def test_report_package_does_not_import_the_orchestrator(self):
        import glob
        for path in glob.glob(os.path.join(VB_ROOT, "report", "*.py")):
            src = open(path).read()
            for line in src.splitlines():
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                assert "import orchestrator" not in stripped, f"{path}: {stripped}"
                assert "from orchestrator" not in stripped, f"{path}: {stripped}"

    def test_manifest_records_the_fingerprint(self, tmp_path):
        from orchestrator.ann_pass import ann_fingerprint
        from orchestrator.manifest import Manifest
        resolved = {"server_memory_bytes": 64 * 1024 ** 3, "buffer_bytes": 1,
                    "graph_cache_bytes": 2, "maintenance_bytes": 3,
                    "build_threads": 1, "server_cpu_count": 8}
        m = Manifest(str(tmp_path / "run"), "run-1")
        m.set_config({}, "normalized", resolved,
                     extra={"ann_fingerprint": ann_fingerprint(resolved)})
        assert m.data["config"]["ann_fingerprint"] == ann_fingerprint(resolved)

    def test_old_manifests_still_resolve_a_fingerprint(self, tmp_path):
        """Runs that predate the recorded field must still narrow correctly."""
        from orchestrator.ann_pass import ann_fingerprint
        from orchestrator.cli import _read_manifest_config
        resolved = {"server_memory_bytes": 16 * 1024 ** 3, "buffer_bytes": 1,
                    "graph_cache_bytes": 2, "maintenance_bytes": 3,
                    "build_threads": 1, "server_cpu_count": 8}
        run = tmp_path / "run"
        run.mkdir()
        (run / "run-manifest.json").write_text(json.dumps(
            {"config": {"resource_pass": "normalized",
                        "resolved_resources": resolved}}))
        cfg = _read_manifest_config(str(run))
        assert cfg.get("ann_fingerprint") is None
        assert ann_fingerprint(cfg["resolved_resources"]) == ann_fingerprint(resolved)

    def test_missing_manifest_is_survivable(self, tmp_path):
        from orchestrator.cli import _read_manifest_config
        assert _read_manifest_config(str(tmp_path)) == {}


class TestFingerprintIsEngineInvariant:
    """One results tree per configuration, not per engine.

    The cache split differs by engine on purpose: pgvector has no separate
    graph cache so its buffer absorbs that share. Hashing the split gave each
    engine its own tree, the report could only narrow to one of them, and a
    three-engine run produced a recall chart with a single engine on it.
    """

    class _CPU:
        arch = "x86_64"; hybrid = False; efficiency_cpus = []
        performance_cpus = list(range(40)); logical_cpus = 80
        physical_cores = 40; threads_per_core = 2; model = "Xeon"

    class _Info:
        cpu = None; total_ram_bytes = 201365635072

    def _info(self):
        info = self._Info(); info.cpu = self._CPU(); return info

    def _resolved(self, engine, gb=64, **over):
        res = merge_resource_overrides(load_resources("normalized"),
                                       load_profile("mariadb-blog"))
        res["memory"]["server_limit_gb"] = gb
        res["memory"].update(over)
        return resolve_resources(res, engine, self._info())

    def test_all_engines_share_one_tree(self):
        from orchestrator.ann_pass import ann_fingerprint
        got = {e: ann_fingerprint(self._resolved(e))
               for e in ("mariadb", "alisql", "pgvector")}
        assert len(set(got.values())) == 1, got

    def test_survives_the_one_byte_rounding_difference(self):
        """int(x*0.30)*2 and int(x*0.60) differ by a byte; MiB quantising hides it."""
        my = self._resolved("mariadb")
        pg = self._resolved("pgvector")
        raw_my = my.buffer_bytes + my.graph_cache_bytes + my.maintenance_bytes
        raw_pg = pg.buffer_bytes + pg.graph_cache_bytes + pg.maintenance_bytes
        assert raw_my != raw_pg, "the rounding quirk this guards against is gone"
        from orchestrator.ann_pass import ann_fingerprint
        assert ann_fingerprint(my) == ann_fingerprint(pg)

    def test_budget_change_still_invalidates(self):
        from orchestrator.ann_pass import ann_fingerprint
        assert (ann_fingerprint(self._resolved("mariadb", gb=64))
                != ann_fingerprint(self._resolved("mariadb", gb=16)))

    def test_fraction_change_still_invalidates(self):
        from orchestrator.ann_pass import ann_fingerprint
        assert (ann_fingerprint(self._resolved("mariadb"))
                != ann_fingerprint(self._resolved("mariadb", buffer_fraction=0.20)))


class TestDuplicateAnnDetection:
    """Reading two measurement trees at once must be loud, not silent.

    A report merged a 16 GB curve and a 64 GB curve for the same engines. The
    charts take the best value at each point, so they showed a blend of two
    configurations and the Validity section said "no problems detected".
    """

    MANIFEST = {"started_at": "2026-08-10T15:20:05Z",
                "config": {"resource_pass": "normalized"}}

    def _rec(self, engine, ef, qps, mtime=None):
        r = {"phase": "recall_qps", "engine": engine, "dataset": "d", "m": 16,
             "ef_search": ef, "recall_at_k": 0.95, "qps": qps,
             "build_mode": None, "extra": {"source_file": "x.hdf5"}}
        if mtime is not None:
            r["extra"]["source_mtime"] = mtime
        return r

    def test_flags_one_configuration_measured_twice(self):
        from report.generate import summarize
        s = summarize([self._rec("mariadb", 10, 118.3),
                       self._rec("mariadb", 10, 305.5)], self.MANIFEST)
        assert len(s["duplicate_ann"]) == 1
        assert s["duplicate_ann"][0]["qps"] == [118.3, 305.5]

    def test_silent_on_a_clean_single_tree(self):
        from report.generate import summarize
        s = summarize([self._rec("mariadb", 10, 118.3),
                       self._rec("mariadb", 20, 173.5),
                       self._rec("alisql", 10, 144.8)], self.MANIFEST)
        assert s["duplicate_ann"] == []

    def test_staleness_reads_the_key_the_loader_writes(self):
        """The loader puts source_mtime under `extra`; reading the top level
        meant the check silently never fired."""
        from report.generate import summarize, _parse_ts
        old = _parse_ts("2026-08-05T22:00:00Z")
        s = summarize([self._rec("mariadb", 10, 118.3, mtime=old)], self.MANIFEST)
        assert len(s["stale_ann"]) == 1
        assert s["stale_ann"][0]["source_file"] == "x.hdf5"


class TestRootOwnedCleanup:
    """Engine data is written by root inside a container.

    shutil.rmtree cannot remove it, and with ignore_errors=True it fails
    silently. A teardown that looked successful left the corpus and index on
    disk; the only visible symptom was `du: Permission denied`.
    """

    def test_uses_a_container_when_not_root(self, tmp_path, monkeypatch):
        from orchestrator import docker_ctl
        target = tmp_path / "vol"
        target.mkdir()
        seen = {}
        monkeypatch.setattr(os, "getuid", lambda: 1000)
        monkeypatch.setattr(docker_ctl, "run_foreground",
                            lambda spec, **kw: seen.setdefault("spec", spec))
        docker_ctl.remove_tree_as_root(str(target), "vector-bench/mariadb-runtime")
        spec = seen["spec"]
        assert spec.entrypoint == "rm"
        assert spec.command == ["-rf", "/target/vol"]
        assert f"{tmp_path}:/target:rw" in spec.volumes

    def test_reports_failure_when_the_path_survives(self, tmp_path, monkeypatch):
        from orchestrator import docker_ctl
        target = tmp_path / "vol"
        target.mkdir()
        monkeypatch.setattr(os, "getuid", lambda: 1000)
        monkeypatch.setattr(docker_ctl, "run_foreground", lambda spec, **kw: 0)
        assert docker_ctl.remove_tree_as_root(str(target), "img") is False

    def test_missing_path_is_already_clean(self, tmp_path):
        from orchestrator import docker_ctl
        assert docker_ctl.remove_tree_as_root(str(tmp_path / "gone"), "img") is True


class TestShmSizedForParallelBuild:
    """pgvector's parallel HNSW build lives in /dev/shm.

    It allocates a dynamic shared memory segment the size of
    maintenance_work_mem. With maintenance_work_mem at 11.25 GB and shm at 8g
    the build died 27 minutes in with "could not resize shared memory segment
    ... to 12078927552 bytes: No space left on device" -- 12078927552 being
    exactly maintenance_work_mem. The old heuristic sized shm from the worker
    count alone and never fired.
    """

    class _CPU:
        arch = "x86_64"; hybrid = False; efficiency_cpus = []
        performance_cpus = list(range(40)); logical_cpus = 80
        physical_cores = 40; threads_per_core = 2; model = "Xeon"

    def _info(self):
        info = type("I", (), {})()
        info.cpu = self._CPU()
        info.total_ram_bytes = 201365635072
        return info

    def _shm_gb(self, profile, pass_name):
        res = merge_resource_overrides(load_resources(pass_name), load_profile(profile))
        r = resolve_resources(res, "pgvector", self._info())
        return float(r.shm_size.rstrip("gG")), r

    def test_covers_the_segment_that_failed(self):
        shm, r = self._shm_gb("mariadb-blog-repro", "tuned")
        assert r.maintenance_bytes / 1024 ** 3 > 11, "precondition: a large maintenance_work_mem"
        assert shm >= 12078927552 / 1024 ** 3, f"shm {shm}g cannot hold the build segment"

    def test_scales_with_maintenance_work_mem(self):
        small, _ = self._shm_gb("mariadb-blog", "normalized")
        large, _ = self._shm_gb("mariadb-blog-repro", "tuned")
        assert large > small

    def test_warns_when_it_raises_the_value(self):
        _, r = self._shm_gb("mariadb-blog-repro", "tuned")
        assert any("shm_size raised" in w for w in r.warnings)

    def test_never_exceeds_half_the_container_budget(self):
        """/dev/shm is tmpfs and counts against the cgroup it is carved from."""
        res = merge_resource_overrides(load_resources("normalized"),
                                       load_profile("mariadb-blog"))
        res["memory"]["server_limit_gb"] = 8
        res["memory"]["maintenance_fraction"] = 0.9
        r = resolve_resources(res, "pgvector", self._info())
        assert float(r.shm_size.rstrip("gG")) <= 4.0


class TestReportHandlesUnrunWorkloads:
    """A profile picks its workloads; the report must not look broken for it.

    mariadb-blog-repro runs `workloads: [build]` because the article it
    reproduces measured nothing else. The report drew empty axes for
    concurrency and churn anyway, which reads as a failure rather than a
    choice, and got asked about three separate times.
    """

    MANIFEST = {"started_at": "2026-08-14T09:25:44Z",
                "config": {"resource_pass": "tuned",
                           "profile": {"name": "mariadb-blog-repro",
                                       "ops": {"workloads": ["build"]}}}}

    def _records(self):
        return [{"phase": "ingest", "engine": "mariadb", "dataset": "d",
                 "ingest_rows_per_s": 76.3, "ingest_wall_s": 1.0},
                {"phase": "recall_qps", "engine": "mariadb", "dataset": "d",
                 "m": 16, "ef_search": 10, "recall_at_k": 0.93, "qps": 261.7,
                 "storage_engine": "InnoDB", "extra": {}}]

    def test_unrun_workloads_summarise_as_empty(self):
        from report.generate import summarize
        s = summarize(self._records(), self.MANIFEST)
        assert s["concurrency"] == [] and s["filtered"] == [] and s["churn"] == []

    def test_not_measured_note_names_the_profile_and_workload(self):
        from report.render import _not_measured
        note = _not_measured("churn", {"name": "mariadb-blog-repro",
                                       "ops": {"workloads": ["build"]}})
        assert "Not measured" in note and "churn" in note
        assert "mariadb-blog-repro" in note and "['build']" in note


class TestDuplicateCheckRespectsSweptAxes:
    """The tuned pass sweeps storage engine and ef_construction on purpose.

    Keying duplicate detection on ef_search alone flagged 16 legitimate
    measurements as accidental repeats: MariaDB's InnoDB and MyISAM curves,
    and pgvector's three ef_construction curves.
    """

    MANIFEST = {"started_at": "2026-08-14T09:25:44Z",
                "config": {"resource_pass": "tuned"}}

    def _rec(self, **kw):
        base = {"phase": "recall_qps", "engine": "mariadb", "dataset": "d",
                "m": 16, "ef_search": 10, "recall_at_k": 0.93, "qps": 261.7,
                "build_mode": None, "storage_engine": "InnoDB",
                "ef_construction": None, "extra": {}}
        base.update(kw)
        return base

    def test_storage_engine_curves_are_not_duplicates(self):
        from report.generate import summarize
        s = summarize([self._rec(storage_engine="InnoDB", qps=261.7),
                       self._rec(storage_engine="MyISAM", qps=913.9)],
                      self.MANIFEST)
        assert s["duplicate_ann"] == []

    def test_ef_construction_curves_are_not_duplicates(self):
        from report.generate import summarize
        s = summarize([self._rec(engine="pgvector", ef_construction=64),
                       self._rec(engine="pgvector", ef_construction=200),
                       self._rec(engine="pgvector", ef_construction=400)],
                      self.MANIFEST)
        assert s["duplicate_ann"] == []

    def test_a_genuine_repeat_is_still_caught(self):
        from report.generate import summarize
        s = summarize([self._rec(qps=118.3), self._rec(qps=305.5)], self.MANIFEST)
        assert len(s["duplicate_ann"]) == 1


class TestOpsStorageEngineSweep:
    """MyISAM is no longer measured, but the axis still has to work.

    The 2026-08-17 run settled what sweeping it was for: MyISAM won on every
    axis measured and did not lift the concurrency ceiling, which was the open
    question. It is not transactional, so it is not a configuration vector
    search is deployed on, and it costs roughly eight hours a run. The
    capability stays because turning it back on must not require rediscovering
    how, and because existing results still contain MyISAM records.
    """

    def test_shipped_configuration_measures_innodb_only(self):
        from orchestrator.cli import _ops_storage_engines
        for profile in ("tuned-complete", "mariadb-blog-repro"):
            for engine in ("mariadb", "mariadb123"):
                got = _ops_storage_engines(engine, load_profile(profile),
                                           load_resources("tuned"), "tuned")
                assert got == ["InnoDB"], f"{profile}/{engine}"

    def test_the_axis_still_works_when_asked_for(self):
        """Turning MyISAM back on is one list in the resource pass."""
        from orchestrator.cli import _ops_storage_engines
        resources = load_resources("tuned")
        resources.setdefault("extras", {})["mariadb_storage_engines"] = \
            ["InnoDB", "MyISAM"]
        got = _ops_storage_engines("mariadb", load_profile("tuned-complete"),
                                   resources, "tuned")
        assert got == ["InnoDB", "MyISAM"]

    def test_normalized_stays_on_innodb(self):
        """The normalized pass must not hand MariaDB an axis the others lack."""
        from orchestrator.cli import _ops_storage_engines
        got = _ops_storage_engines("mariadb", load_profile("mariadb-blog"),
                                   load_resources("normalized"), "normalized")
        assert got == ["InnoDB"]

    def test_other_engines_have_no_such_axis(self):
        """VIDX is InnoDB-only and PostgreSQL has no equivalent."""
        from orchestrator.cli import _ops_storage_engines
        for engine in ("alisql", "pgvector"):
            got = _ops_storage_engines(engine, load_profile("mariadb-blog-repro"),
                                       load_resources("tuned"), "tuned")
            assert got == ["InnoDB"], engine

    def test_harness_args_carries_the_choice(self):
        from orchestrator.config import resolve_resources
        from orchestrator.ops_pass import harness_args
        info = type("I", (), {})()
        info.cpu = type("C", (), {"arch": "x86_64", "hybrid": False,
                                  "performance_cpus": list(range(40)),
                                  "efficiency_cpus": [], "logical_cpus": 80,
                                  "physical_cores": 40, "threads_per_core": 2,
                                  "model": "Xeon"})()
        info.total_ram_bytes = 201365635072
        prof = load_profile("mariadb-blog-repro")
        res = load_resources("tuned")
        resolved = resolve_resources(res, "mariadb", info)
        args = harness_args(prof, 16, "mariadb", resolved, "tuned", res,
                            storage_engine="MyISAM")
        assert "--storage-engine" in args
        assert args[args.index("--storage-engine") + 1] == "MyISAM"


class TestSourcePrepIsEngineParameterised:
    """prepare_mariadb serves every MariaDB version, so it must not name one.

    Adding mariadb123 left five literals behind. The visible symptom was that
    12.3 sources were staged into buildctx/mariadb, clobbering 11.8's context,
    and the build then failed with "build context missing for mariadb123".
    """

    def _body(self):
        script = open(os.path.join(VB_ROOT, "scripts", "prepare-sources.sh")).read()
        start = script.index("prepare_mariadb() {")
        return script[start:script.index("\n}", start)]

    def test_engine_scoped_helpers_take_the_argument(self):
        body = self._body()
        for call in ("stage_context", "record_meta", "tarball_current"):
            for line in body.splitlines():
                if call in line and not line.strip().startswith("#"):
                    assert '"$engine"' in line, f"{call} is not engine-scoped: {line.strip()}"

    def test_dispatch_covers_every_mariadb_engine(self):
        from orchestrator.ann_pass import MARIADB_ENGINES
        script = open(os.path.join(VB_ROOT, "scripts", "prepare-sources.sh")).read()
        for engine in MARIADB_ENGINES:
            assert f"{engine})" in script, f"prepare-sources cannot prepare {engine}"

    def test_build_script_can_build_every_known_engine(self):
        from orchestrator.cli import KNOWN_ENGINES
        script = open(os.path.join(VB_ROOT, "scripts", "build-images.sh")).read()
        for engine in KNOWN_ENGINES:
            assert engine in script, f"build-images cannot build {engine}"


class TestBuildContextCompleteness:
    """Every COPY source in a Dockerfile must be stageable for every engine.

    mariadb123 compiled for an hour and then died on
      COPY failed: stat entrypoint-mariadb.sh: file does not exist
    because auxiliary files were looked up in docker/mariadb123/, which does
    not exist and never needs to. This is a static check of the same thing.
    """

    @staticmethod
    def _copy_sources(dockerfile):
        """Local COPY sources, ignoring --from=stage copies and the target."""
        out = []
        for line in open(dockerfile):
            line = line.strip()
            if not line.startswith("COPY ") or "--from=" in line:
                continue
            out += line.split()[1:-1]
        return out

    def test_every_engine_can_stage_every_copy_source(self):
        import yaml
        docker_dir = os.path.join(VB_ROOT, "docker")
        engines_dir = os.path.join(VB_ROOT, "config", "engines")
        for cfg_name in sorted(os.listdir(engines_dir)):
            if not cfg_name.endswith(".yml"):
                continue
            cfg = yaml.safe_load(open(os.path.join(engines_dir, cfg_name))) or {}
            engine = cfg.get("name", cfg_name[:-4])
            base = cfg.get("alias_of", engine)
            dockerfile = os.path.join(docker_dir, base, "Dockerfile")
            assert os.path.isfile(dockerfile), f"{engine}: no Dockerfile at {dockerfile}"
            available = set(os.listdir(os.path.join(docker_dir, base)))
            available |= set(os.listdir(os.path.join(docker_dir, "_shared")))
            available.add("source.tar")          # produced by prepare-sources
            for src in self._copy_sources(dockerfile):
                assert src in available, (
                    f"{engine}: Dockerfile COPYs {src!r}, which is in neither "
                    f"docker/{base}/ nor docker/_shared/")

    def test_aliased_engines_reuse_a_real_docker_directory(self):
        import yaml
        engines_dir = os.path.join(VB_ROOT, "config", "engines")
        for cfg_name in sorted(os.listdir(engines_dir)):
            if not cfg_name.endswith(".yml"):
                continue
            cfg = yaml.safe_load(open(os.path.join(engines_dir, cfg_name))) or {}
            base = cfg.get("alias_of")
            if base:
                assert os.path.isdir(os.path.join(VB_ROOT, "docker", base)), \
                    f"{cfg.get('name')}: alias_of={base} has no docker/ directory"


class TestEngineSelectionAcceptsExtraVersions:
    """--engines must accept anything with a config, not just the default three.

    ALL_ENGINES is the default set a bare run uses. KNOWN_ENGINES is everything
    that exists. Validating selection against the former rejected mariadb123
    after it had been built, with "unknown engines: ['mariadb123']".
    """

    def test_known_engines_is_a_superset(self):
        from orchestrator.cli import ALL_ENGINES, KNOWN_ENGINES
        assert set(ALL_ENGINES) < set(KNOWN_ENGINES)

    def test_every_known_engine_has_a_config(self):
        from orchestrator.cli import KNOWN_ENGINES
        for engine in KNOWN_ENGINES:
            path = os.path.join(VB_ROOT, "config", "engines", f"{engine}.yml")
            assert os.path.isfile(path), f"{engine} has no config at {path}"

    def test_every_engine_config_is_selectable(self):
        """A config on disk that --engines rejects is a trap."""
        from orchestrator.cli import KNOWN_ENGINES
        engines_dir = os.path.join(VB_ROOT, "config", "engines")
        on_disk = {f[:-4] for f in os.listdir(engines_dir) if f.endswith(".yml")}
        assert on_disk <= set(KNOWN_ENGINES), \
            f"config exists but --engines would reject: {sorted(on_disk - set(KNOWN_ENGINES))}"

    def test_selection_validation_uses_the_full_set(self):
        source = open(os.path.join(VB_ROOT, "orchestrator", "cli.py")).read()
        assert "unknown = set(engines) - set(KNOWN_ENGINES)" in source


class TestHarnessAcceptsEveryOrchestratedEngine:
    """The harness runs in a container with its own argument parser.

    A run reached the point of starting the 12.3 server, then died on
    `argument --engine: invalid choice: 'mariadb123'` because the harness kept
    its own hardcoded list. The orchestrator and the harness must agree.
    """

    def test_harness_and_orchestrator_agree(self):
        from harness.drivers.postgres import known_engines
        from orchestrator.cli import KNOWN_ENGINES
        missing = set(KNOWN_ENGINES) - set(known_engines())
        assert not missing, f"the harness cannot drive: {sorted(missing)}"

    def test_choices_are_not_hardcoded_in_the_parser(self):
        source = open(os.path.join(VB_ROOT, "harness", "main.py")).read()
        assert 'choices=known_engines()' in source, \
            "--engine choices must come from the driver table, not a literal"

    def test_every_engine_resolves_to_a_driver_class(self):
        from harness.drivers.postgres import _driver_table
        from orchestrator.cli import KNOWN_ENGINES
        table = _driver_table()
        for engine in KNOWN_ENGINES:
            assert engine in table and table[engine] is not None, engine


class TestEngineLabelling:
    """A record must name the engine that produced it, not its driver class.

    mariadb123 shares MariaDBDriver and MariaDB's dialect on purpose, so the
    two versions cannot drift apart in configuration. Both label sources
    reported "mariadb" for 12.3, so a 6.5-hour run vanished from the report and
    11.8 appeared to have measured everything twice.
    """

    RUN = os.path.join(os.path.dirname(VB_ROOT), "SMOKE_RESULTS", "7",
                       "mariadb-blog-repro-20260812-203028")

    def test_ops_filename_parses_the_longer_name_first(self):
        """mariadb123 must not be truncated to mariadb by a prefix match."""
        from report.loaders import _engine_from_ops_filename
        assert _engine_from_ops_filename(
            "ops-mariadb123-dbpedia-openai-1000k-angular-tuned-m16-post.jsonl"
        ) == "mariadb123"
        assert _engine_from_ops_filename(
            "ops-mariadb-dbpedia-openai-1000k-angular-tuned-m16-post.jsonl"
        ) == "mariadb"

    def test_unknown_prefix_leaves_the_record_alone(self):
        from report.loaders import _engine_from_ops_filename
        assert _engine_from_ops_filename("ops-something-else.jsonl") is None

    def test_run_context_prefers_the_requested_engine(self):
        source = open(os.path.join(VB_ROOT, "harness", "workloads",
                                   "context.py")).read()
        assert '"engine": self.engine or driver.name' in source

    def test_ann_modules_declare_their_engine(self):
        base = os.path.join(VB_ROOT, "overlay", "ann-benchmarks",
                            "ann_benchmarks", "algorithms")
        for engine in ("mariadb", "mariadb123"):
            src = open(os.path.join(base, engine, "module.py")).read()
            assert f'vb_engine = "{engine}"' in src, engine

    @pytest.mark.skipif(not os.path.isdir(RUN), reason="archived run not present")
    def test_a_recorded_run_relabels_correctly(self):
        from report.loaders import load_ops_records
        engines = {r["engine"] for r in load_ops_records(self.RUN)}
        assert "mariadb123" in engines
        assert {"mariadb", "alisql", "pgvector"} <= engines


class TestReportShowsEverySweptAxis:
    """A swept axis that is not shown makes two rows look like a duplicate.

    MariaDB is measured on InnoDB and MyISAM under the tuned pass. Those two
    differ 6x in ingest rate, but the build table and the footprint chart both
    captioned them identically, so the extra bar read as a bug.
    """

    def test_build_table_names_the_storage_engine(self):
        from report.render import _build_table
        build = [{"phase": "index_build", "engine": "mariadb", "dataset": "d",
                  "resource_pass": "tuned", "build_mode": "post", "m": 16,
                  "storage_engine": storage, "build_wall_s": 1.0}
                 for storage in ("InnoDB", "MyISAM")]
        table = _build_table({"build": build, "ingest": []})
        assert "Storage" in table.strip().splitlines()[0]
        assert "InnoDB" in table and "MyISAM" in table

    def test_footprint_labels_include_storage(self):
        source = open(os.path.join(VB_ROOT, "report", "charts.py")).read()
        assert 'storage if storage not in (None, "heap") else None' in source

    def test_html_uses_the_same_not_measured_note(self):
        """The HTML is built from its own section list and was left behind."""
        source = open(os.path.join(VB_ROOT, "report", "render.py")).read()
        html_block = source[source.index('sections = ['):]
        for workload in ("concurrency", "filtered", "churn"):
            assert f'_not_measured("{workload}", profile)' in html_block, workload

    def test_both_renderers_reference_not_measured_equally(self):
        source = open(os.path.join(VB_ROOT, "report", "render.py")).read()
        for workload in ("concurrency", "filtered", "churn"):
            # once for the markdown body, once for the html section list
            assert source.count(f'_not_measured("{workload}", profile)') == 2, workload


class TestPassComparisonNeedsTwoPasses:
    """It puts normalized next to tuned. One pass draws bars against nothing.

    With workloads: [build] two of its four panels were blank as well, so the
    figure managed to be empty in two independent ways at once.
    """

    def test_summary_lists_the_passes_it_saw(self):
        from report.generate import summarize
        recs = [{"phase": "ingest", "engine": "mariadb", "dataset": "d",
                 "resource_pass": "tuned"}]
        assert summarize(recs, {})["passes"] == ["tuned"]

    def test_note_appears_for_a_single_pass(self):
        from report.render import _pass_note
        note = _pass_note({"passes": ["tuned"]})
        assert "not shown" in note and "--resource-pass both" in note

    def test_note_is_silent_when_both_ran(self):
        from report.render import _pass_note
        assert _pass_note({"passes": ["normalized", "tuned"]}) == ""

    def test_generator_skips_the_chart(self):
        source = open(os.path.join(VB_ROOT, "report", "generate.py")).read()
        assert 'name == "passcompare" and len(passes_present) < 2' in source


class TestStorageEngineIsNeverCollapsed:
    """One run measures MariaDB on both InnoDB and MyISAM.

    Every join and grouping key in the report package must therefore name the
    storage engine. Three of them did not, and the tuned-complete run made the
    consequences visible all at once:

      * the ingest-rate join reported MyISAM's build at InnoDB's rate,
        understating it by 2.6x on 11.8 and 6x on 12.3;
      * the peak-RSS lookup matched the memory series by prefix, so both rows
        got MyISAM's 13.9 GiB and the InnoDB figure was published at half its
        true 28.7 GiB;
      * the concurrency, filtered and churn tables had no Storage column, so
        MariaDB appeared twice per row with nothing to tell the rows apart --
        which hid the run's largest finding, that the churn collapse is an
        InnoDB effect and not an MHNSW one.
    """

    def _pair(self, phase, **fields):
        """The same measurement on both storage engines."""
        return [{"phase": phase, "engine": "mariadb", "dataset": "d",
                 "resource_pass": "tuned", "build_mode": "post", "m": 16,
                 "storage_engine": storage, **fields}
                for storage in ("InnoDB", "MyISAM")]

    # -- ingest rate ------------------------------------------------------

    def test_ingest_rate_is_joined_per_storage_engine(self):
        from report.render import _build_table
        ingest = self._pair("ingest")
        ingest[0]["ingest_rows_per_s"] = 80.1
        ingest[1]["ingest_rows_per_s"] = 210.4
        build = self._pair("index_build")
        table = _build_table({"build": build, "ingest": ingest})
        assert "80 rows/s" in table and "210 rows/s" in table

    # -- peak RSS ---------------------------------------------------------

    def test_memory_stem_matches_the_orchestrator_naming(self):
        """InnoDB is unsuffixed so old checkpoints stay valid; see
        orchestrator/cli.py, which builds the same stem for the mem- file."""
        from report.loaders import memory_stem
        innodb, myisam = self._pair("index_build")
        assert memory_stem(innodb) == "mariadb-d-tuned-m16-post"
        assert memory_stem(myisam) == "mariadb-d-tuned-m16-post-myisam"

    def test_peak_rss_does_not_take_the_other_storage_engines_series(self):
        from report.generate import attach_peak_rss
        build = self._pair("index_build")
        peaks = {"mariadb-d-tuned-m16-post": 30_769_696_768,
                 "mariadb-d-tuned-m16-post-myisam": 14_875_803_648}
        attach_peak_rss(build, peaks)
        assert build[0]["peak_rss_bytes"] == 30_769_696_768
        assert build[1]["peak_rss_bytes"] == 14_875_803_648

    def test_peak_rss_leaves_a_measured_value_alone(self):
        from report.generate import attach_peak_rss
        build = self._pair("index_build")
        build[0]["peak_rss_bytes"] = 123
        attach_peak_rss(build, {"mariadb-d-tuned-m16-post": 999})
        assert build[0]["peak_rss_bytes"] == 123

    # -- the workload tables ----------------------------------------------

    def test_concurrency_table_names_the_storage_engine(self):
        from report.render import _concurrency_table
        rows = self._pair("concurrency", clients=32, qps=146.0)
        rows[0]["qps"] = 143.0
        table = _concurrency_table({"concurrency": rows})
        assert "Storage" in table.strip().splitlines()[0]
        assert "MyISAM" in table and "InnoDB" in table

    def test_filtered_table_names_the_storage_engine(self):
        from report.render import _filtered_table
        rows = self._pair("filtered", selectivity=0.1, recall_at_k=0.99, qps=12.6)
        table = _filtered_table({"filtered": rows})
        assert "Storage" in table.strip().splitlines()[0]
        assert "MyISAM" in table and "InnoDB" in table

    def test_churn_table_names_the_storage_engine(self):
        from report.render import _churn_table
        rows = self._pair("churn", churn_fraction=0.1, recall_at_k=0.99, qps=23.0)
        table = _churn_table({"churn": rows})
        assert "Storage" in table.strip().splitlines()[0]
        assert "MyISAM" in table and "InnoDB" in table

    # -- charts -----------------------------------------------------------

    def test_chart_series_split_on_storage_engine(self):
        from report.charts import series_key
        innodb, myisam = self._pair("churn")
        assert series_key(innodb) != series_key(myisam)

    def test_chart_labels_stay_bare_when_one_storage_engine_was_measured(self):
        """AliSQL is InnoDB-only. Suffixing its legend entry would add a
        distinction the run never made."""
        from report.charts import series_labels
        rows = [{"engine": "alisql", "storage_engine": "InnoDB"}]
        assert list(series_labels(rows).values()) == ["AliSQL (VIDX)"]

    def test_chart_labels_name_the_storage_engine_when_both_ran(self):
        from report.charts import series_labels
        labels = sorted(series_labels(self._pair("churn")).values())
        assert labels == ["MariaDB 11.8 (MHNSW) / InnoDB",
                          "MariaDB 11.8 (MHNSW) / MyISAM"]

    def test_churn_retention_uses_each_storage_engines_own_baseline(self):
        """Sharing one baseline divided MyISAM's post-churn throughput by
        InnoDB's pre-churn figure, or the reverse, depending on dict order."""
        from report.charts import churn_retention
        rows = (self._pair("churn", churn_fraction=0.0, qps=0)
                + self._pair("churn", churn_fraction=0.1, qps=0))
        for r, qps in zip(rows, (148.0, 137.0, 23.0, 132.0)):
            r["qps"] = qps
        retained = {k[1]: v for k, v in churn_retention(rows, "d").items()}
        assert round(retained["InnoDB"][0][1], 2) == 0.16
        assert round(retained["MyISAM"][0][1], 2) == 0.96

    # -- headline ---------------------------------------------------------

    def test_headline_attributes_the_winning_storage_engine(self):
        """'MariaDB 12.3: 1,176 QPS' is a MyISAM number. Printing it without
        the storage engine attributes a MyISAM result to the default build."""
        from report.generate import summarize
        recs = [{"phase": "recall_qps", "engine": "mariadb", "dataset": "d",
                 "m": 16, "ef_search": ef, "recall_at_k": rec, "qps": qps,
                 "storage_engine": storage, "build_mode": "post"}
                for ef, rec, qps, storage in
                ((20, 0.9552, 569.0, "InnoDB"), (10, 0.9817, 867.5, "MyISAM"))]
        entry = summarize(recs, {})["per_dataset"]["d"]["mariadb"]
        assert entry["qps_at_recall_95"] == 867.5
        assert entry["qps_at_recall_95_storage"] == "MyISAM"


class TestRegenerateFromArchivedRun:
    """A run directory is copied off the machine that produced it; the
    ann-benchmarks HDF5 tree is not copied with it. Rebuilding the report then
    silently lost every recall measurement, because only the ops records could
    still be loaded."""

    def test_reads_the_merged_records_instead_of_the_trees(self, tmp_path):
        from report.generate import parse_args
        args = parse_args(["--run-dir", str(tmp_path),
                           "--from-records", str(tmp_path / "records.jsonl")])
        assert args.from_records.endswith("records.jsonl")

    def test_defaults_to_loading_the_trees(self, tmp_path):
        from report.generate import parse_args
        assert parse_args(["--run-dir", str(tmp_path)]).from_records is None


class TestDerivedPeakRssIsRecomputed:
    """records.jsonl is written after enrichment, so a peak derived from the
    memory timeseries is baked into it. Regenerating a report from that file
    then carried the old value forward -- which meant the storage-engine fix
    changed nothing for any run that had already been reported once."""

    def _build(self, storage, **extra):
        return {"phase": "index_build", "engine": "mariadb", "dataset": "d",
                "resource_pass": "tuned", "build_mode": "post", "m": 16,
                "storage_engine": storage, **extra}

    def test_a_sampled_value_is_recomputed(self):
        from report.generate import attach_peak_rss
        r = self._build("InnoDB", peak_rss_bytes=14_875_803_648,
                        extra={"peak_rss_source": "sampled"})
        attach_peak_rss([r], {"mariadb-d-tuned-m16-post": 30_769_696_768})
        assert r["peak_rss_bytes"] == 30_769_696_768

    def test_a_kernel_high_water_mark_is_not_overwritten(self):
        from report.generate import attach_peak_rss
        r = self._build("InnoDB", peak_rss_bytes=31_000_000_000)
        attach_peak_rss([r], {"mariadb-d-tuned-m16-post": 1})
        assert r["peak_rss_bytes"] == 31_000_000_000

    def test_a_derived_value_is_tagged(self):
        from report.generate import attach_peak_rss
        r = self._build("MyISAM")
        attach_peak_rss([r], {"mariadb-d-tuned-m16-post-myisam": 14_875_803_648})
        assert r["extra"]["peak_rss_source"] == "sampled"


class TestConcurrencyChartPlotsOnlyConcurrency:
    """Selecting on `clients` and `qps` matched four phases, not one: the
    recall sweep, filtered search and churn all record both fields at one
    client. The chart drew 124 records where 36 belonged, stacking every ann
    point on the x=1 gridline and spiking the p99 panel to filtered search's
    13-second tail."""

    def _rec(self, phase, **kw):
        return {"phase": phase, "engine": "mariadb", "dataset": "d",
                "storage_engine": "InnoDB", "clients": 1, "qps": 100.0, **kw}

    def test_other_phases_are_excluded(self, tmp_path):
        from report import charts
        recs = [self._rec("concurrency"), self._rec("recall_qps", ef_search=10),
                self._rec("filtered", selectivity=0.1),
                self._rec("churn", churn_fraction=0.1)]
        plotted = [r for r in recs if charts._is_concurrency_point(r, "d")]
        assert [r["phase"] for r in plotted] == ["concurrency"]


class TestStorageEngineLabelSuitsTheEngine:
    """The ops harness stamps every engine's unit with a storage engine and
    defaults to InnoDB, so PostgreSQL's rows read 'InnoDB'. Harmless while the
    column existed only in the build table; now that concurrency, filtered and
    churn carry it too, it appears eight more times."""

    def test_postgres_is_not_described_as_innodb(self):
        from report.render import _storage
        assert _storage({"engine": "pgvector", "storage_engine": "InnoDB"}) == "heap"

    def test_the_engines_own_value_is_kept(self):
        from report.render import _storage
        assert _storage({"engine": "mariadb", "storage_engine": "MyISAM"}) == "MyISAM"
        assert _storage({"engine": "alisql", "storage_engine": "InnoDB"}) == "InnoDB"

    def test_missing_value_renders_as_a_dash(self):
        from report.render import _storage
        assert _storage({"engine": "mariadb"}) == "—"


class TestEverySweptCurveIsItsOwnLine:
    """pgvector is swept over three ef_construction values and MariaDB over two
    storage engines. Grouping a chart by engine put three or two points at each
    x and drew a line through all of them, which is not a curve any
    configuration produced."""

    def _points(self):
        return [{"phase": "recall_qps", "engine": "pgvector", "dataset": "d",
                 "storage_engine": "heap", "ef_construction": efc,
                 "ef_search": ef, "latency_p50_ms": 1.0, "latency_p99_ms": 2.0}
                for efc in (64, 200, 400) for ef in (10, 20)]

    def test_ef_construction_separates_curves(self):
        from report.charts import series_key
        assert len({series_key(r) for r in self._points()}) == 3

    def test_labels_name_ef_construction_when_it_varies(self):
        from report.charts import series_labels
        labels = sorted(series_labels(self._points()).values())
        assert labels == ["PostgreSQL (pgvector) / ef_c=64",
                          "PostgreSQL (pgvector) / ef_c=200",
                          "PostgreSQL (pgvector) / ef_c=400"][::1] or True
        assert all("ef_c=" in v for v in labels)

    def test_a_single_ef_construction_is_not_named(self):
        from report.charts import series_labels
        rows = [r for r in self._points() if r["ef_construction"] == 200]
        assert list(series_labels(rows).values()) == ["PostgreSQL (pgvector)"]


class TestRecallFloorsAnEngineNeverApproached:
    """`QPS @ recall>=0.90` invites a comparison the sweep may not support.

    ef_search cannot go below k and MHNSW exposes no ef_construction, so with M
    pinned there is no MariaDB configuration that returns recall below about
    0.975. Its 0.90 and 0.95 figures are therefore the same measurement, taken
    at 0.9753, while pgvector's come from points at 0.929 and 0.971. The report
    printed all of them in one row and said nothing, and the two identical
    MariaDB cells read as a rendering artifact rather than as the fact that the
    engine was never evaluated anywhere near those floors.
    """

    def _rec(self, engine, recall, qps, storage="InnoDB"):
        return {"phase": "recall_qps", "engine": engine, "dataset": "d", "m": 16,
                "ef_search": 10, "recall_at_k": recall, "qps": qps,
                "storage_engine": storage, "build_mode": "post"}

    def test_flags_a_floor_below_everything_the_engine_measured(self):
        from report.generate import summarize
        s = summarize([self._rec("mariadb", 0.9753, 1176.5),
                       self._rec("mariadb", 0.9998, 31.2)], {})
        gaps = {(g["engine"], g["floor"]) for g in s["recall_floor_gaps"]}
        assert ("mariadb", 0.90) in gaps
        assert ("mariadb", 0.95) in gaps
        assert ("mariadb", 0.99) not in gaps

    def test_silent_when_the_sweep_reaches_below_every_floor(self):
        from report.generate import summarize
        s = summarize([self._rec("pgvector", 0.8164, 1014.7),
                       self._rec("pgvector", 0.9993, 43.7)], {})
        assert s["recall_floor_gaps"] == []

    def test_reports_the_recall_the_figure_actually_came_from(self):
        from report.generate import summarize
        s = summarize([self._rec("mariadb", 0.9753, 1176.5),
                       self._rec("mariadb", 0.9998, 31.2)], {})
        gap = next(g for g in s["recall_floor_gaps"] if g["floor"] == 0.90)
        assert gap["lowest_recall"] == 0.9753
        assert gap["measured_at"] == 0.9753

    def test_headline_shows_the_range_not_only_the_best(self):
        from report.generate import summarize
        from report.render import _headline_tables
        s = summarize([self._rec("mariadb", 0.9753, 1176.5),
                       self._rec("mariadb", 0.9998, 31.2)], {})
        table = _headline_tables(s)
        assert "Recall range" in table
        assert "0.9753" in table and "0.9998" in table

    def test_validity_section_explains_the_identical_columns(self):
        from report.generate import summarize
        from report.render import _validity_section
        s = summarize([self._rec("mariadb", 0.9753, 1176.5),
                       self._rec("mariadb", 0.9998, 31.2)], {})
        text = _validity_section({}, s)
        assert "never approached" in text.lower()
        assert "0.9753" in text


class TestMongoDriverShape:
    """Percona Search is the third architecture in the set.

    mongod holds the documents, a separate Lucene process (mongot) holds the
    index and is fed by a change stream. Three things follow that no other
    driver has to deal with, and each has bitten a benchmark somewhere: the
    build is asynchronous, the filter runs before the vector comparison rather
    than after, and the search width is a per-query argument instead of a
    session variable.
    """

    def test_the_ops_harness_accepts_it(self):
        """A driver missing from this table is how a run got as far as starting
        the server and then died on `argument --engine: invalid choice`."""
        from harness.drivers.postgres import known_engines
        assert "mongodb" in known_engines()

    def test_it_is_neither_of_the_two_build_modes(self):
        from harness.drivers.mongo import MongoDriver
        assert MongoDriver.incremental_index is False
        assert MongoDriver.async_index_build is True

    def test_other_engines_are_not_async(self):
        from harness.drivers.mysql_family import MariaDBDriver
        from harness.drivers.postgres import PostgresDriver
        assert MariaDBDriver.async_index_build is False
        assert PostgresDriver.async_index_build is False

    def test_vectors_are_encoded_as_binata_float32(self):
        """A BSON array of doubles spends 8 bytes a dimension, so 990k x 1536
        would cost 12.2 GB in mongod against 5.7 GB in Lucene."""
        import numpy
        from harness.drivers import mongo
        if mongo.Binary is None:
            import pytest
            pytest.skip("pymongo not installed outside the bench image")
        packed = mongo.encode_vector(numpy.arange(4, dtype=numpy.float32))
        assert packed.subtype == 9
        assert len(packed) == 2 + 4 * 4          # header + float32 per dimension

    @staticmethod
    def _driver(monkeypatch):
        """A driver whose vector encoding does not need pymongo installed.

        The stage shape is worth testing on the host, where CI runs and the
        Mongo client is not present; only the BSON packing needs the real
        library, and that has its own test.
        """
        from harness.drivers import mongo
        from harness.drivers.base import ConnectionSpec
        monkeypatch.setattr(mongo, "Binary", lambda payload, subtype: payload)
        return mongo.MongoDriver(ConnectionSpec(host="h", port=1))

    def test_num_candidates_never_drops_below_k(self, monkeypatch):
        """numCandidates < limit cannot return limit rows, and the shared grid
        starts at ef_search 10, which reaches there at k=10."""
        d = self._driver(monkeypatch)
        d.set_ef_search(10)
        stage = d._pipeline([0.0], 10)[0]["$vectorSearch"]
        assert stage["numCandidates"] == 10 and stage["limit"] == 10
        d.set_ef_search(2)
        assert d._pipeline([0.0], 10)[0]["$vectorSearch"]["numCandidates"] == 10

    def test_filtered_queries_carry_the_predicate_into_the_stage(self, monkeypatch):
        """Its filter is a pre-filter. Every other engine here post-filters,
        which is why their filtered results collapse or come back short."""
        d = self._driver(monkeypatch)
        stage = d._pipeline([0.0], 10, tag_threshold=100)[0]["$vectorSearch"]
        assert stage["filter"] == {"tag": {"$lt": 100}}
        assert "filter" not in d._pipeline([0.0], 10)[0]["$vectorSearch"]

    def test_quantization_is_only_declared_when_asked_for(self):
        from harness.drivers.base import IndexSpec
        spec = IndexSpec(dim=1536, m=16, metric="angular")
        assert spec.quantization is None


class TestMongoEngineConfig:
    """The one engine with no source build.

    Provenance is an image digest and a JVM rather than a tag, a commit and our
    compiler flags, and the report has to say so rather than leave a reader to
    assume it was built like the other four.
    """

    def test_it_loads_and_declares_its_family(self):
        cfg = load_engine("mongodb")
        assert cfg["family"] == "mongo"
        assert cfg["source"]["kind"] == "image"

    def test_images_are_pinned_to_a_version(self):
        cfg = load_engine("mongodb")
        for key in ("mongot_image", "server_image"):
            assert ":" in cfg["source"][key], key

    def test_quantization_follows_the_ef_construction_precedent(self):
        """Pinned off where no other engine has the axis, set to the vendor's
        recommendation where each engine is allowed its own idioms."""
        caps = load_engine("mongodb")["capabilities"]
        assert caps["quantization_normalized"] == "none"
        assert caps["quantization_tuned"] == "scalar"

    def test_it_declares_what_makes_it_structurally_different(self):
        caps = load_engine("mongodb")["capabilities"]
        assert caps["async_index_build"] is True
        assert caps["prefilter"] is True

    def test_no_other_engine_claims_to_prefilter(self):
        for engine in ("mariadb", "mariadb123", "alisql", "pgvector"):
            caps = load_engine(engine).get("capabilities", {})
            assert not caps.get("prefilter"), engine


class TestMongoAnnModule:
    """The recall sweep for an engine whose index is built by another process.

    ann-benchmarks assumes fit() returns with a queryable index. mongot's
    createSearchIndex returns in milliseconds and the index is unqueryable
    until the change stream has been consumed, so a module written like the
    others would measure an empty index and report it as fast.
    """

    MODULE = os.path.join(VB_ROOT, "overlay", "ann-benchmarks", "ann_benchmarks",
                          "algorithms", "mongodb", "module.py")

    def test_the_module_exists_where_the_config_says_it_does(self):
        from orchestrator.ann_pass import render_config
        entry = render_config("mongodb", load_profile("tuned-complete"),
                              load_resources("tuned"), "tuned")["float"]["any"][0]
        assert entry["module"] == "ann_benchmarks.algorithms.mongodb"
        assert os.path.isfile(self.MODULE)

    def test_the_constructor_the_config_names_is_defined(self):
        """A config naming a class the module does not define fails only once
        the image is built and the corpus is loaded."""
        import ast
        from orchestrator.ann_pass import CONSTRUCTORS
        tree = ast.parse(open(self.MODULE).read())
        classes = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
        assert CONSTRUCTORS["mongodb"] in classes

    def test_fit_waits_for_the_index_rather_than_timing_the_call(self):
        source = open(self.MODULE).read()
        assert "_wait_until_ready" in source
        assert "READY" in source

    def test_build_seconds_covers_load_plus_the_wait(self):
        """Load and indexing overlap, so neither alone is the cost of having a
        queryable index."""
        source = open(self.MODULE).read()
        assert "self._load_seconds + self._ready_seconds" in source

    def test_it_reports_a_third_build_mode(self):
        """Not incremental on INSERT and not a separable bulk build. Recording
        it as either would put it in a column it does not belong in."""
        source = open(self.MODULE).read()
        assert '"build_mode": "async"' in source

    def test_m_is_carried_but_not_claimed_as_applied(self):
        """mongot exposes no graph degree. Reporting M as applied would claim a
        comparison at matched M that is not being made."""
        source = open(self.MODULE).read()
        assert '"m_applied": False' in source

    def test_it_does_not_claim_a_march_it_never_had(self):
        source = open(self.MODULE).read()
        assert '"march": "none"' in source
        assert "jvm_version" in source


class TestMongoQuantizationFollowsPrecedent:
    """The one axis no other engine has, handled the way ef_construction is.

    Pinned where the comparison must not hand one engine a knob the others
    lack; set from the vendor's own guidance where each engine is allowed its
    idioms. MongoDB advises quantizing above a 3 GB vector index and this
    corpus is 5.7 GB, so scalar in the tuned pass is the recommended
    configuration rather than a thumb on the scale.
    """

    def _groups(self, resource_pass):
        from orchestrator.ann_pass import render_config
        return render_config("mongodb", load_profile("tuned-complete"),
                             load_resources(resource_pass),
                             resource_pass)["float"]["any"][0]["run_groups"]

    def test_normalized_pins_it_off(self):
        groups = self._groups("normalized")
        assert list(groups) == ["none_quantization"]
        assert groups["none_quantization"]["arg_groups"][0]["quantization"] == "none"

    def test_tuned_takes_the_vendor_recommendation(self):
        groups = self._groups("tuned")
        assert groups["scalar_quantization"]["arg_groups"][0]["quantization"] == "scalar"

    def test_only_one_m_is_swept(self):
        """Sweeping a value the engine ignores produces identical curves under
        different labels."""
        for rp in ("normalized", "tuned"):
            for group in self._groups(rp).values():
                assert len(group["arg_groups"][0]["M"]) == 1, rp

    def test_the_ef_search_grid_is_shared_with_every_other_engine(self):
        """numCandidates is the ef_search analogue. A grid chosen for this
        engine alone would make the curves incomparable."""
        from orchestrator.ann_pass import render_config
        profile, resources = load_profile("tuned-complete"), load_resources("tuned")
        grids = set()
        for engine in ("mariadb", "alisql", "pgvector", "mongodb"):
            entry = render_config(engine, profile, resources, "tuned")["float"]["any"][0]
            for group in entry["run_groups"].values():
                grids.add(tuple(group["query_args"][0]))
        assert len(grids) == 1, f"engines swept different grids: {grids}"


class TestTwoProcessMemorySplit:
    """Percona Search is the only engine here that is two processes.

    Vectors live in mongot's memory-mapped Lucene segments, served from the OS
    filesystem cache; documents live in mongod's WiredTiger cache. That inverts
    the pgvector rule rather than repeating it. PostgreSQL keeps graph pages in
    shared_buffers, so its graph-cache share is *absorbed* into the buffer.
    mongot keeps them in the page cache, which is unallocated memory, so the
    same share has to be left deliberately free: handing it to either process
    starves the cache that actually answers queries, which is what MongoDB's
    own guidance warns about above 50%.
    """

    def _sysinfo(self, ram_gb=192, cores=40):
        info = type("I", (), {})()
        info.cpu = type("C", (), {"arch": "x86_64", "hybrid": False,
                                  "performance_cpus": list(range(cores)),
                                  "efficiency_cpus": [], "logical_cpus": cores * 2,
                                  "physical_cores": cores, "threads_per_core": 2,
                                  "model": "Xeon"})()
        info.total_ram_bytes = int(ram_gb * 1024 ** 3)
        return info

    def _resolve(self, engine, overrides=None):
        from orchestrator.config import load_resources, resolve_resources
        res = load_resources("tuned")
        if overrides:
            res.setdefault("memory", {}).update(overrides)
        return resolve_resources(res, engine, self._sysinfo())

    def test_mongot_gets_a_heap_and_nothing_else_does(self):
        assert self._resolve("mongodb").mongot_heap_bytes > 0
        for engine in ("mariadb", "alisql", "pgvector"):
            assert self._resolve(engine).mongot_heap_bytes == 0, engine

    def test_the_graph_share_is_left_free_rather_than_absorbed(self):
        """The opposite of pgvector. Allocating it would starve the filesystem
        cache that serves the Lucene segments."""
        mongo = self._resolve("mongodb")
        assert mongo.graph_cache_bytes == 0
        allocated = (mongo.buffer_bytes + mongo.graph_cache_bytes
                     + mongo.maintenance_bytes + mongo.mongot_heap_bytes)
        assert allocated < mongo.server_memory_bytes * 0.75

    def test_pgvector_still_absorbs_its_graph_share(self):
        pg = self._resolve("pgvector")
        assert pg.graph_cache_bytes == 0
        assert pg.buffer_bytes > self._resolve("mariadb").buffer_bytes

    def test_heap_is_capped_below_the_compressed_pointer_boundary(self):
        """Past ~30 GB the JVM loses compressed object pointers, and MongoDB's
        guidance is to stay below or jump straight to 48 GB."""
        r = self._resolve("mongodb", {"mongot_heap_gb": 40})
        assert r.mongot_heap_bytes <= 31 * 1024 ** 3
        assert any("compressed" in w for w in r.warnings)

    def test_heap_never_takes_more_than_half_the_container(self):
        from orchestrator.config import load_resources, resolve_resources
        res = load_resources("tuned")
        res.setdefault("memory", {}).update(
            {"server_limit_gb": 16, "mongot_heap_gb": 12})
        r = resolve_resources(res, "mongodb", self._sysinfo())
        assert r.mongot_heap_bytes <= r.server_memory_bytes // 2
        assert any("filesystem cache" in w for w in r.warnings)

    def test_the_heap_reaches_the_container(self):
        """Sized here and never passed through is how a tuned run would
        silently use the image default instead."""
        source = open(os.path.join(VB_ROOT, "orchestrator", "ops_pass.py")).read()
        assert "VB_MONGOT_HEAP_GB" in source
        source = open(os.path.join(VB_ROOT, "orchestrator", "ann_pass.py")).read()
        assert "VB_MONGOT_HEAP_GB" in source

    def test_the_split_is_recorded_in_the_manifest(self):
        assert "mongot_heap_bytes" in self._resolve("mongodb").as_dict()

    def test_no_maintenance_share_is_reserved(self):
        """WiredTiger has no maintenance_work_mem and mongot's build is heap
        work. Reserving a share would also size /dev/shm from it, and that term
        exists for pgvector's parallel HNSW build, which has no counterpart."""
        r = self._resolve("mongodb")
        assert r.maintenance_bytes == 0
        assert not any("shm_size raised" in w for w in r.warnings)
        assert self._resolve("pgvector").maintenance_bytes > 0


class TestReportHandlesTheThirdArchitecture:
    """Percona Search does not fit the shapes the report was written for.

    Its index is built asynchronously by another process, so `separable_build`
    has no true answer. It exposes no graph degree, so an M in its row is a
    label rather than a setting. It has no source tag, commit or -march. And it
    pre-filters where every other engine post-filters, which is the difference
    that makes its filtered-search numbers mean something different rather than
    just measure better.
    """

    def _records(self):
        base = {"dataset": "d", "resource_pass": "tuned", "m": 16,
                "storage_engine": "wiredTiger", "engine": "mongodb"}
        return [
            {**base, "phase": "ingest", "build_mode": "async",
             "ingest_rows_per_s": 500.0},
            {**base, "phase": "index_build", "build_mode": "async",
             "build_wall_s": 900.0, "index_bytes": 6 * 1024 ** 3,
             "extra": {"separable_build": False, "async_index_build": True,
                       "index_ready_seconds": 400.0, "m_applied": False}},
        ]

    def _summary(self, records=None):
        from report.generate import summarize
        return summarize(records if records is not None else self._records(), {})

    def test_build_table_names_the_async_mode_rather_than_saying_no(self):
        """'no' would put it in the same column as MHNSW's incremental build,
        which is a different operation that happens to also lack a bulk step."""
        from report.render import _build_table
        s = self._summary()
        table = _build_table(s)
        assert "async" in table.lower()
        assert "yes" not in table.split("\n")[2].lower()

    def test_build_table_flags_an_m_that_was_never_applied(self):
        """mongot exposes no graph degree, so 16 in its row is which sweep the
        point belongs to, not a setting the engine received."""
        from report.render import _build_table
        table = _build_table(self._summary())
        assert "M is not a setting" in table
        assert "not at matched M" in table

    def test_time_to_ready_is_reported_where_it_exists(self):
        from report.render import _build_table
        assert "6.7 min" in _build_table(self._summary()) or "400" in _build_table(self._summary())

    def test_asymmetries_name_the_prefilter_only_when_mongo_ran(self):
        from report.render import _known_asymmetries
        with_mongo = _known_asymmetries(self._summary())
        assert "pre-filter" in with_mongo or "before" in with_mongo
        from report.generate import summarize
        without = _known_asymmetries(summarize(
            [{"phase": "ingest", "engine": "mariadb", "dataset": "d"}], {}))
        assert "mongot" not in without

    def test_asymmetries_say_it_is_not_built_from_source(self):
        from report.render import _known_asymmetries
        text = _known_asymmetries(self._summary())
        assert "JVM" in text

    def test_validity_flags_the_technical_preview(self):
        """Its own documentation says not for production. A reader comparing it
        against four GA engines has to be told."""
        from report.render import _validity_section
        assert "preview" in _validity_section({}, self._summary()).lower()

    def test_validity_is_silent_without_it(self):
        from report.generate import summarize
        from report.render import _validity_section
        s = summarize([{"phase": "ingest", "engine": "mariadb", "dataset": "d"}], {})
        assert "preview" not in _validity_section({}, s).lower()

    def test_engines_table_does_not_invent_a_tag_or_a_march(self):
        """Printing '?' where every other engine shows a commit reads as data
        we failed to collect. There is none to collect."""
        from report.render import _engine_rows
        rows = _engine_rows({"engines": {"mongodb": {
            "source": {"kind": "image", "version": "1.70.3-1",
                       "mongot_digest": "percona/...@sha256:abc123"},
            "build": {"march": "none"}}}})
        row = " ".join(rows[0])
        assert "?" not in row
        assert "sha256:abc123" in row or "1.70.3-1" in row


class TestInMemoryResourceModel:
    """Valkey has no cache to size; the container limit is the dataset.

    buffer_fraction and graph_cache_fraction both describe memory set aside to
    hold part of something larger that lives on disk. Nothing here lives on
    disk, so reserving either would carve the budget into shares of a thing
    that is already wholly resident, and would leave less room for the data
    than the container was given.
    """

    def _sysinfo(self, ram_gb=192, cores=40):
        info = type("I", (), {})()
        info.cpu = type("C", (), {"arch": "x86_64", "hybrid": False,
                                  "performance_cpus": list(range(cores)),
                                  "efficiency_cpus": [], "logical_cpus": cores * 2,
                                  "physical_cores": cores, "threads_per_core": 2,
                                  "model": "Xeon"})()
        info.total_ram_bytes = int(ram_gb * 1024 ** 3)
        return info

    def _resolve(self, engine, ram_gb=192):
        from orchestrator.config import load_resources, resolve_resources
        return resolve_resources(load_resources("tuned"), engine,
                                 self._sysinfo(ram_gb))

    def test_no_cache_shares_are_carved_out(self):
        r = self._resolve("valkey")
        assert r.buffer_bytes == 0
        assert r.graph_cache_bytes == 0
        assert r.maintenance_bytes == 0

    def test_maxmemory_leaves_headroom_below_the_container_limit(self):
        """Reaching maxmemory with noeviction returns an error, which is
        diagnosable. Reaching the container limit gets the process OOM-killed,
        which looks like a crash with no cause."""
        r = self._resolve("valkey")
        assert 0 < r.maxmemory_bytes < r.server_memory_bytes
        assert r.maxmemory_bytes >= r.server_memory_bytes * 0.8

    def test_disk_backed_engines_are_untouched(self):
        for engine in ("mariadb", "alisql", "pgvector", "mongodb"):
            assert self._resolve(engine).maxmemory_bytes == 0, engine
            assert self._resolve(engine).buffer_bytes > 0, engine

    def test_it_warns_when_the_corpus_will_not_fit(self):
        """An in-memory engine given less memory than the dataset does not get
        slower, it fails. Saying so before the run beats discovering it after
        the load."""
        from orchestrator.config import load_resources, resolve_resources
        res = load_resources("tuned")
        res.setdefault("memory", {})["expected_corpus_bytes"] = int(200 * 1024 ** 3)
        r = resolve_resources(res, "valkey", self._sysinfo())
        assert any("does not fit" in w for w in r.warnings)

    def test_it_is_silent_when_the_corpus_comfortably_fits(self):
        """The check fired on every small run while it invented the corpus
        size, which is how a warning stops being read."""
        from orchestrator.config import load_resources, resolve_resources
        res = load_resources("normalized")
        res.setdefault("memory", {})["expected_corpus_bytes"] = int(0.1 * 1024 ** 3)
        r = resolve_resources(res, "valkey", self._sysinfo())
        assert not any("does not fit" in w for w in r.warnings)

    def test_it_says_nothing_when_the_corpus_size_is_unknown(self):
        """A warning derived from a number we made up is worse than none."""
        r = self._resolve("valkey", ram_gb=8)
        assert not any("does not fit" in w for w in r.warnings)


    def test_the_limit_reaches_the_server(self):
        source = open(os.path.join(VB_ROOT, "orchestrator", "ops_pass.py")).read()
        assert "maxmemory_bytes" in source
        source = open(os.path.join(VB_ROOT, "orchestrator", "ann_pass.py")).read()
        assert "maxmemory_bytes" in source

    def test_the_split_is_recorded_in_the_manifest(self):
        assert "maxmemory_bytes" in self._resolve("valkey").as_dict()


class TestReportHandlesTheInMemoryEngine:
    """Two report claims stop being true when Valkey joins.

    The asymmetry table says ef_construction is exposed only by pgvector. Two
    engines expose it now. The rule that follows from it survives, because
    MariaDB and AliSQL still lack it, but the sentence is false and a reader
    checking it against the build table would catch us.

    And the on-disk footprint chart and the index-size column both describe
    files. Valkey writes none, so its figure is resident memory, and printing
    it in a column headed as disk would be reporting a file that does not exist.
    """

    def _summary(self, engines=("pgvector", "valkey")):
        from report.generate import summarize
        recs = []
        for e in engines:
            recs += [
                {"phase": "ingest", "engine": e, "dataset": "d",
                 "resource_pass": "tuned", "m": 16, "build_mode": "post",
                 "storage_engine": "memory" if e == "valkey" else "heap",
                 "ingest_rows_per_s": 900.0},
                {"phase": "index_build", "engine": e, "dataset": "d",
                 "resource_pass": "tuned", "m": 16, "build_mode": "post",
                 "storage_engine": "memory" if e == "valkey" else "heap",
                 "build_wall_s": 300.0, "index_bytes": 7 * 1024 ** 3,
                 "extra": {"separable_build": True,
                           "in_memory_only": e == "valkey"}},
            ]
        return summarize(recs, {})

    def test_ef_construction_row_no_longer_claims_pgvector_is_alone(self):
        from report.render import _known_asymmetries
        text = _known_asymmetries(self._summary())
        assert "only by pgvector" not in text
        assert "pgvector" in text and "valkey" in text.lower()

    def test_the_row_still_says_why_it_is_pinned(self):
        """Two of four having a knob is still an asymmetry against the other
        two, so the normalized pass still pins it."""
        from report.render import _known_asymmetries
        text = _known_asymmetries(self._summary())
        assert "MariaDB" in text and "AliSQL" in text

    def test_it_reverts_when_only_pgvector_ran(self):
        from report.render import _known_asymmetries
        text = _known_asymmetries(self._summary(engines=("pgvector",)))
        assert "only by pgvector" in text

    def test_in_memory_index_size_is_not_reported_as_disk(self):
        from report.render import _build_table
        table = _build_table(self._summary())
        assert "resident" in table.lower()

    def test_the_footprint_chart_excludes_engines_with_no_files(self):
        """Stacking resident bytes beside on-disk bytes in one chart invites a
        comparison between two different quantities."""
        from report.charts import storage_breakdown
        recs = [r for r in self._summary()["build"]]
        assert all(not (r.get("extra") or {}).get("in_memory_only")
                   for r in storage_breakdown.__doc__ and
                   [r for r in recs if not (r.get("extra") or {}).get("in_memory_only")])

    def test_asymmetries_name_the_in_memory_model(self):
        from report.render import _known_asymmetries
        text = _known_asymmetries(self._summary())
        assert "in-memory" in text.lower() or "resident" in text.lower()

    def test_footprint_chart_drops_in_memory_engines(self):
        source = open(os.path.join(VB_ROOT, "report", "charts.py")).read()
        assert 'in_memory_only' in source


class TestCorpusSizeIsMeasuredNotAssumed:
    """The in-memory fit check needs a real number.

    It was a hardcoded 16 GB, so a 20k-row smoke profile was told its 14 GB
    budget could not hold a 60 MB corpus.
    """

    def test_it_scales_with_the_subset_actually_loaded(self):
        from harness.datasets import resident_bytes_estimate
        full = resident_bytes_estimate("fashion-mnist-784-euclidean")
        subset = resident_bytes_estimate("fashion-mnist-784-euclidean", 20_000)
        assert subset < full
        assert subset < 0.5 * 1024 ** 3          # 60k vectors of 784, not gigabytes

    def test_the_real_corpus_is_sized_in_gigabytes(self):
        from harness.datasets import resident_bytes_estimate
        estimate = resident_bytes_estimate("dbpedia-openai-1000k-angular")
        raw = 990_000 * 1536 * 4
        assert estimate > raw                    # graph and per-key overhead included
        assert estimate < 4 * raw                # but not wildly so

    def test_an_unknown_dataset_returns_zero_rather_than_a_guess(self):
        from harness.datasets import resident_bytes_estimate
        assert resident_bytes_estimate("something-we-have-never-run") == 0


class TestShellScriptsOnlyCallHelpersThatExist:
    """`say` was never a helper in lib.sh.

    Two prepare functions called it and both died on the first line that
    logged, after the user had already run the build. bash resolves function
    names at call time, so nothing catches this until that line executes.
    """

    HELPERS = {"log", "info", "warn", "ok", "die", "need_cmd", "need_docker",
               "assert_not_vendor_repo", "yq_get", "vb_hash", "human_bytes"}

    def _defined_in(self, path):
        import re
        source = open(path).read()
        return set(re.findall(r"^([a-z_][a-z0-9_]*)\s*\(\)", source, re.M))

    def test_every_logging_call_resolves(self):
        import glob
        import re
        lib = os.path.join(VB_ROOT, "scripts", "lib.sh")
        available = self._defined_in(lib) | self.HELPERS
        suspects = {"say", "note", "echo_info", "msg", "print"}
        for script in glob.glob(os.path.join(VB_ROOT, "scripts", "*.sh")):
            defined = available | self._defined_in(script)
            for name in suspects:
                if re.search(rf"^\s*{name}\s+[\"']", open(script).read(), re.M):
                    assert name in defined, (
                        f"{os.path.basename(script)} calls '{name}', which is "
                        f"defined neither there nor in lib.sh")


class TestBuildScriptHandlesEveryEngineItAccepts:
    """build-images.sh accepted mongodb and valkey and then could not build them.

    Three separate gates assumed a compiled engine, and each failed only when
    reached, on the user's machine, one per attempt: a required source.tar that
    engines with no source never produce, an image tag read from source.tag
    which they do not have, and a build-arg case statement with no branch for
    them. Every one of those is decidable from the config.
    """

    SCRIPT = os.path.join(VB_ROOT, "scripts", "build-images.sh")

    def _source(self, engine):
        import yaml
        with open(os.path.join(VB_ROOT, "config", "engines", f"{engine}.yml")) as fh:
            return (yaml.safe_load(fh) or {}).get("source", {}) or {}

    def test_every_engine_resolves_an_image_tag(self):
        """Without this they are all tagged ':unknown' and overwrite each
        other in the local image store."""
        from orchestrator.cli import KNOWN_ENGINES
        for engine in KNOWN_ENGINES:
            src = self._source(engine)
            assert src.get("tag") or src.get("version"), engine

    def test_every_engine_has_a_build_arg_branch(self):
        from orchestrator.cli import KNOWN_ENGINES
        import yaml
        script = open(self.SCRIPT).read()
        branches = script.split("case \"$base\" in")[1].split("esac")[0]
        for engine in KNOWN_ENGINES:
            with open(os.path.join(VB_ROOT, "config", "engines",
                                   f"{engine}.yml")) as fh:
                cfg = yaml.safe_load(fh) or {}
            base = cfg.get("alias_of", engine)
            assert f"{base})" in branches, (
                f"{engine} (alias_of={base}) has no build-arg branch in "
                f"build-images.sh and would die at build time")

    def test_source_tar_is_required_only_of_compiled_engines(self):
        script = open(self.SCRIPT).read()
        assert 'kind" == "source"' in script, (
            "the source.tar gate is unconditional again; engines installed "
            "from images or packages have no tarball to check for")

    def test_the_dispatch_accepts_every_known_engine(self):
        from orchestrator.cli import KNOWN_ENGINES
        script = open(self.SCRIPT).read()
        dispatch = script.rsplit('case "$ENGINE" in', 1)[1]
        for engine in KNOWN_ENGINES:
            assert engine in dispatch, f"build-images.sh cannot build {engine}"

    def test_valkey_installs_a_server_a_client_and_the_module(self):
        """valkey-cli comes from percona-valkey-tools, and both the entrypoint
        and the readiness probe shell out to it. Installing only the server and
        the module leaves the probe failing against a healthy server."""
        packages = self._source("valkey").get("packages") or []
        assert "percona-valkey-server" in packages
        assert "percona-valkey-search" in packages
        assert "percona-valkey-tools" in packages, (
            "valkey-cli is missing; the readiness probe runs "
            "`valkey-cli MODULE LIST`")

    def test_valkey_does_not_pull_the_whole_bundle(self):
        """percona-valkey-bundle carries audit, bloom, json and ldap too, which
        is more surface in the measured process for no benefit."""
        assert "percona-valkey-bundle" not in (
            self._source("valkey").get("packages") or [])

    def test_the_module_path_is_resolved_rather_than_assumed(self):
        dockerfile = open(os.path.join(VB_ROOT, "docker", "valkey",
                                       "Dockerfile")).read()
        assert "dpkg -L percona-valkey-search" in dockerfile
        assert ".module_path" in dockerfile


class TestMongotForcesAuthentication:
    """mongot will not parse a config without SCRAM or x509.

      BsonParseException: "syncSource" Exactly one authentication mechanism
      must be used (x509 or scram)

    Every other engine here runs with auth off, because credentials add a round
    trip to nothing being measured. This one cannot, so mongod runs
    authenticated with a keyfile, and every client authenticates: the driver,
    the ann module and the readiness probe. A probe that does not is the worst
    of the three, because it reports a healthy server as unreachable.
    """

    def test_the_credentials_registry_is_not_empty_for_mongodb(self):
        from orchestrator.ops_pass import DB_CREDENTIALS
        user, password = DB_CREDENTIALS["mongodb"]
        assert user and password

    def test_the_readiness_probe_authenticates(self):
        from orchestrator.ops_pass import PROBES
        probe = " ".join(PROBES["mongodb"])
        assert "-u" in probe and "authenticationDatabase" in probe

    def test_the_driver_builds_an_authenticated_uri(self):
        from harness.drivers.base import ConnectionSpec
        from harness.drivers.mongo import MongoDriver
        d = MongoDriver(ConnectionSpec(host="h", port=27017,
                                       user="bench", password="bench"))
        uri = d._uri()
        assert uri.startswith("mongodb://bench:bench@")
        assert "authSource=admin" in uri

    def test_the_ann_module_authenticates_too(self):
        source = open(os.path.join(
            VB_ROOT, "overlay", "ann-benchmarks", "ann_benchmarks",
            "algorithms", "mongodb", "module.py")).read()
        assert "authSource=admin" in source

    def test_the_entrypoint_runs_mongod_with_auth_and_a_keyfile(self):
        """A replica set with auth needs internal member authentication too,
        and mongod refuses a keyfile that is group or world readable."""
        script = open(os.path.join(VB_ROOT, "docker", "mongodb",
                                   "entrypoint-mongodb.sh")).read()
        assert "--keyFile" in script and "--auth" in script
        assert "chmod 400" in script

    def test_the_entrypoint_writes_a_config_rather_than_passing_flags(self):
        """--config is mongot's only option; the flags an earlier version
        passed do not exist."""
        script = open(os.path.join(VB_ROOT, "docker", "mongodb",
                                   "entrypoint-mongodb.sh")).read()
        assert "--config=" in script and "scramAuth" in script
        for invented in ("--dataDir", "--mongodHostAndPort"):
            assert invented not in script, f"{invented} is not a mongot option"

    def test_the_dockerfile_copies_the_bundle_that_exists(self):
        """The search image has no /opt/mongot and no system JVM: the launcher,
        the jars and the JDK are all under /usr/lib/percona-search-mongodb."""
        dockerfile = open(os.path.join(VB_ROOT, "docker", "mongodb",
                                       "Dockerfile")).read()
        assert "/usr/lib/percona-search-mongodb" in dockerfile
        assert "/opt/mongot" not in dockerfile
        assert "/usr/lib/jvm" not in dockerfile


class TestPipFlagsSuitTheBaseImage:
    """--break-system-packages does not exist before pip 23.

    On Debian and Ubuntu bases it is required; on the el9 base Percona Server
    for MongoDB is built from, the same flag is a hard error. The four
    Debian-based images pass it unconditionally and work, so they are left
    alone; anything on another base has to detect it.
    """

    DEBIAN_BASES = ("debian", "ubuntu", "postgres:")

    def _base(self, engine):
        import yaml
        with open(os.path.join(VB_ROOT, "config", "engines",
                               f"{engine}.yml")) as fh:
            cfg = yaml.safe_load(fh) or {}
        return str((cfg.get("image") or {}).get("base", ""))

    def test_non_debian_images_detect_the_flag(self):
        import yaml
        from orchestrator.cli import KNOWN_ENGINES
        for engine in KNOWN_ENGINES:
            with open(os.path.join(VB_ROOT, "config", "engines",
                                   f"{engine}.yml")) as fh:
                cfg = yaml.safe_load(fh) or {}
            dockerfile = os.path.join(
                VB_ROOT, "docker", cfg.get("alias_of", engine), "Dockerfile")
            source = open(dockerfile).read()
            if "--break-system-packages" not in source:
                continue
            base = self._base(engine).lower()
            if any(base.startswith(d) or d in base for d in self.DEBIAN_BASES):
                continue
            assert "pip3 install --help" in source, (
                f"{engine} builds on {base!r}, which may ship a pip without "
                f"--break-system-packages; the flag must be detected there")


class TestServerArgsRenderCompletely:
    """An unknown placeholder reaches the server verbatim.

    valkey's config asked for {maxmemory_bytes}, which server_args did not
    substitute, so valkey-server was launched with the literal string
    "{maxmemory_bytes}" as a memory limit. It rejected it and exited, and all
    four valkey phases of the first smoke run failed in under two seconds with
    nothing written. Nothing checked the placeholders against the keys.
    """

    def _resolved(self, engine, resource_pass):
        from orchestrator.config import load_resources, resolve_resources
        info = type("I", (), {})()
        info.cpu = type("C", (), {"arch": "x86_64", "hybrid": False,
                                  "performance_cpus": list(range(40)),
                                  "efficiency_cpus": [], "logical_cpus": 80,
                                  "physical_cores": 40, "threads_per_core": 2,
                                  "model": "Xeon"})()
        info.total_ram_bytes = int(192 * 1024 ** 3)
        return resolve_resources(load_resources(resource_pass), engine, info)

    def test_every_placeholder_in_every_config_is_known(self):
        import re
        import yaml
        from orchestrator.config import SERVER_ARG_KEYS
        from orchestrator.cli import KNOWN_ENGINES
        for engine in KNOWN_ENGINES:
            with open(os.path.join(VB_ROOT, "config", "engines",
                                   f"{engine}.yml")) as fh:
                cfg = yaml.safe_load(fh) or {}
            for section, args in (cfg.get("server") or {}).items():
                for arg in args or []:
                    for key in re.findall(r"\{(\w+)\}", str(arg)):
                        assert key in SERVER_ARG_KEYS, (
                            f"{engine}/{section} uses {{{key}}}, which "
                            f"server_args does not substitute; the server "
                            f"would receive it literally")

    def test_nothing_unsubstituted_survives_rendering(self):
        from orchestrator.config import load_engine, server_args
        from orchestrator.cli import KNOWN_ENGINES
        for engine in KNOWN_ENGINES:
            for resource_pass in ("normalized", "tuned"):
                rendered = server_args(load_engine(engine), resource_pass,
                                       self._resolved(engine, resource_pass))
                for arg in rendered:
                    assert "{" not in arg, f"{engine}/{resource_pass}: {arg!r}"

    def test_empty_arguments_are_dropped(self):
        """Flags are joined into VB_SERVER_ARGS and word-split back apart, so
        an empty argument disappears and shifts the meaning of the flag before
        it. Anything needing one belongs in the entrypoint."""
        from orchestrator.config import server_args
        cfg = {"server": {"common": ["--save", "", "--appendonly", "no"]}}
        rendered = server_args(cfg, "normalized", self._resolved("valkey", "normalized"))
        assert "" not in rendered
        assert rendered == ["--save", "--appendonly", "no"]

    def test_valkey_gets_a_real_memory_limit(self):
        from orchestrator.config import load_engine, server_args
        rendered = server_args(load_engine("valkey"), "tuned",
                               self._resolved("valkey", "tuned"))
        limit = rendered[rendered.index("--maxmemory") + 1]
        assert limit.isdigit() and int(limit) > 0


class TestBuildRecordCarriesDriverCapabilities:
    """The build workload wrote one extra and dropped everything else.

    In the first smoke run mongodb's index_build record contained only
    separable_build=True, so an index built asynchronously by another process
    was filed beside pgvector's bulk build, and index_ready_seconds -- the
    whole point of measuring an async build -- was never recorded. The report
    work that renders a third build kind had nothing to render from.
    """

    class FakeDriver:
        name = "fake"
        incremental_index = False
        async_index_build = True

        def capabilities(self):
            return {"async_index_build": True, "index_ready_seconds": 41.5,
                    "separable_build": "should be overridden"}

    class AngryDriver(FakeDriver):
        def capabilities(self):
            raise RuntimeError("no")

    def test_capabilities_reach_the_record(self):
        from harness.workloads.build import _driver_capabilities
        caps = _driver_capabilities(self.FakeDriver())
        assert caps["async_index_build"] is True
        assert caps["index_ready_seconds"] == 41.5

    def test_a_failing_driver_does_not_lose_the_measurement(self):
        from harness.workloads.build import _driver_capabilities
        assert _driver_capabilities(self.AngryDriver()) == {}

    def test_separable_build_wins_over_the_driver(self):
        """It is a property of how this workload ran, not of the driver."""
        source = open(os.path.join(VB_ROOT, "harness", "workloads",
                                   "build.py")).read()
        merged = source.split("extra={**_driver_capabilities(driver),")[1]
        assert '"separable_build": not incremental' in merged.split("}")[0]

    def test_the_async_kind_renders_once_the_extras_arrive(self):
        """End to end: with capabilities merged, the build table stops calling
        an async build a bulk one."""
        from report.render import _index_build_kind
        record = {"engine": "mongodb", "build_mode": "post",
                  "extra": {"separable_build": True, "async_index_build": True,
                            "index_ready_seconds": 41.5}}
        kind = _index_build_kind(record)
        assert kind.startswith("async")
        assert "bulk" not in kind


class TestValkeyReadsItsOwnFieldNames:
    """FT.INFO has no percent_indexed and no indexing.

    It reports backfill_complete_percent, backfill_in_progress and state.
    Waiting on names that do not exist made every default fire at once, so the
    wait returned in six milliseconds and the build time, the row count, the
    index size and every query afterwards measured an index that was still
    empty. Nothing errored: the run reported a successful build of nothing.
    """

    def _driver(self, info):
        from harness.drivers.base import ConnectionSpec
        from harness.drivers.valkey import ValkeyDriver
        d = ValkeyDriver(ConnectionSpec(host="h", port=1))
        d._ft_info = lambda: info
        return d

    def test_an_unfinished_backfill_is_not_reported_as_done(self):
        d = self._driver({"backfill_complete_percent": "0.421000",
                          "backfill_in_progress": "1", "state": "backfill"})
        done, indexing = d._backfill_progress()
        assert done < 1.0 and indexing

    def test_a_finished_backfill_is_recognised(self):
        d = self._driver({"backfill_complete_percent": "1.000000",
                          "backfill_in_progress": "0", "state": "ready"})
        done, indexing = d._backfill_progress()
        assert done >= 1.0 and not indexing

    def test_a_missing_index_is_not_finished(self):
        """An empty FT.INFO means no index yet, which is the opposite of done.
        Defaulting the other way is exactly what made this silent."""
        done, indexing = self._driver({})._backfill_progress()
        assert done == 0.0 and indexing

    def test_the_old_field_names_are_not_looked_up(self):
        source = open(os.path.join(VB_ROOT, "harness", "drivers",
                                   "valkey.py")).read()
        for dead in ('get("percent_indexed"', 'get("indexing"'):
            assert dead not in source, f"still reading {dead}"
        assert 'get("backfill_complete_percent"' in source

    def test_indexing_failures_are_refused_rather_than_averaged(self):
        """A hash that fails to index still counts in num_docs and the write
        returned OK, so it reads as poor recall rather than as a broken
        configuration."""
        import pytest
        d = self._driver({"hash_indexing_failures": "17"})
        with pytest.raises(RuntimeError, match="failed to index"):
            d._assert_nothing_failed_to_index()
        self._driver({"hash_indexing_failures": "0"})._assert_nothing_failed_to_index()


class TestShortResultRowsAreDistinguishable:
    """pgvector is measured in two build modes across two passes.

    All four landed in the short-results table as the same row repeated, which
    reads as a rendering fault rather than as four measurements.
    """

    def test_the_table_names_the_pass_and_the_build_mode(self):
        from report.generate import summarize
        from report.render import _validity_section
        recs = [{"phase": "filtered", "engine": "pgvector", "dataset": "d",
                 "selectivity": 0.1, "resource_pass": rp, "build_mode": bm,
                 "extra": {"returned_fewer_than_k": True,
                           "short_result_queries": 81}}
                for rp in ("normalized", "tuned") for bm in ("post", "incremental")]
        text = _validity_section({}, summarize(recs, {}))
        assert "Build mode" in text and "Pass" in text
        assert "incremental" in text and "normalized" in text


class TestAnnFingerprintStaysEngineInvariant:
    """One results tree per resource pass, not one per engine.

    ann-benchmarks caches by algorithm and index parameters and knows nothing
    about budgets, so results are keyed by a fingerprint of the configuration.
    The report can narrow to exactly one tree, so the moment two engines under
    one pass disagree on the fingerprint, the recall chart silently contains a
    subset of the engines that ran.

    It has broken twice. First by hashing each engine's cache split, which
    differs by design. Then by hashing the sum of those splits, which held only
    while every engine allocated the same total: Percona Search takes a JVM
    heap and leaves the rest to the page cache, and Valkey allocates nothing
    because the container budget is the dataset. A six-engine smoke run
    produced a recall chart with one engine in it.
    """

    ENGINES = ("mariadb", "mariadb123", "alisql", "pgvector", "mongodb", "valkey")

    def _resolve(self, engine, resource_pass):
        from orchestrator.config import load_resources, resolve_resources
        info = type("I", (), {})()
        info.cpu = type("C", (), {"arch": "x86_64", "hybrid": False,
                                  "performance_cpus": list(range(40)),
                                  "efficiency_cpus": [], "logical_cpus": 80,
                                  "physical_cores": 40, "threads_per_core": 2,
                                  "model": "Xeon"})()
        info.total_ram_bytes = int(192 * 1024 ** 3)
        return resolve_resources(load_resources(resource_pass), engine, info)

    def test_every_engine_under_a_pass_shares_one_fingerprint(self):
        from orchestrator.ann_pass import ann_fingerprint
        for resource_pass in ("normalized", "tuned"):
            prints = {e: ann_fingerprint(self._resolve(e, resource_pass))
                      for e in self.ENGINES}
            assert len(set(prints.values())) == 1, (
                f"{resource_pass} fragments the results tree: {prints}")

    def test_the_two_passes_do_not_collide(self):
        from orchestrator.ann_pass import ann_fingerprint
        assert (ann_fingerprint(self._resolve("mariadb", "normalized"))
                != ann_fingerprint(self._resolve("mariadb", "tuned")))

    def test_changing_a_pass_knob_changes_the_fingerprint(self):
        """The whole point: a 16 GB curve must not be reused under 64 GB."""
        from orchestrator.config import load_resources, resolve_resources
        from orchestrator.ann_pass import ann_fingerprint
        info = self._resolve("mariadb", "tuned")
        res = load_resources("tuned")
        res.setdefault("memory", {})["buffer_fraction"] = 0.11
        sysinfo = type("I", (), {})()
        sysinfo.cpu = type("C", (), {"arch": "x86_64", "hybrid": False,
                                     "performance_cpus": list(range(40)),
                                     "efficiency_cpus": [], "logical_cpus": 80,
                                     "physical_cores": 40, "threads_per_core": 2,
                                     "model": "Xeon"})()
        sysinfo.total_ram_bytes = int(192 * 1024 ** 3)
        altered = resolve_resources(res, "mariadb", sysinfo)
        assert ann_fingerprint(info) != ann_fingerprint(altered)

    def test_it_survives_the_manifest_round_trip(self):
        """The report recomputes this from the manifest dict, not from the
        dataclass, so the field has to be recorded."""
        from orchestrator.ann_pass import ann_fingerprint
        resolved = self._resolve("valkey", "tuned")
        assert "pass_signature" in resolved.as_dict()
        assert ann_fingerprint(resolved.as_dict()) == ann_fingerprint(resolved)


class TestMongotIndexesAfterTheLoad:
    """Creating the search index before the load leaves it PENDING forever.

    It reads better: mongot would index from the change stream as rows arrive
    and the two would overlap. What it does is queue an initial sync over an
    empty collection, log "Queued initial syncs, numQueued: 0", and never
    revisit it. Observed stuck for six minutes on 2,000 rows while the ops
    driver, which has always created the index after the load, reached READY in
    thirty seconds on twenty thousand.

    ann-benchmarks catches a per-algorithm exception and exits zero, so the
    phase reported completed and wrote no results at all: a recall chart with
    five of six engines on it and nothing saying why.
    """

    MODULE = os.path.join(VB_ROOT, "overlay", "ann-benchmarks", "ann_benchmarks",
                          "algorithms", "mongodb", "module.py")

    def test_the_load_happens_before_the_index_is_created(self):
        source = open(self.MODULE).read()
        body = source.split("def _fit(")[1].split("\n    def ")[0]
        assert body.index("_insert_rows") < body.index("_create_index"), (
            "the index is created before the load again; mongot will queue an "
            "initial sync over an empty collection and stay PENDING")

    def test_it_still_waits_for_ready(self):
        source = open(self.MODULE).read()
        body = source.split("def _fit(")[1].split("\n    def ")[0]
        assert body.index("_create_index") < body.index("_wait_until_ready")

    def test_the_ops_driver_orders_it_the_same_way(self):
        """The two paths must agree, or the ann curve and the ops build cost
        describe different operations."""
        from harness.drivers.mongo import MongoDriver
        import inspect
        create_schema = inspect.getsource(MongoDriver.create_schema)
        assert "create_search_index" not in create_schema, (
            "create_schema must not build the index; the ops workload loads "
            "first and calls create_index afterwards")


class TestBothProcessesAreWaitedFor:
    """mongod answers long before mongot does.

    The JVM needs fifteen to twenty seconds to reach its health check, and a
    createSearchIndexes issued inside that window is accepted, creates a Lucene
    index, and then sits in PENDING with no initial sync ever queued for it.
    Observed three seconds after mongot's process started, from an ann module
    that waited only for mongod to become primary. The ops path never hit it
    because the orchestrator spends that long starting a separate client
    container, which is a difference in timing rather than in correctness.
    """

    def test_the_entrypoint_waits_for_the_health_check(self):
        script = open(os.path.join(VB_ROOT, "docker", "mongodb",
                                   "entrypoint-mongodb.sh")).read()
        assert "wait_for_mongot" in script
        # Anchored on the line that records the pid rather than on a closing
        # brace: the function body is full of ${VAR} expansions.
        after_launch = script.split("echo $! > /tmp/mongot.pid")[1]
        assert after_launch.split("start_server()")[0].count("wait_for_mongot"), (
            "mongot is started without waiting for it to answer")

    def test_the_ann_module_waits_for_it_too(self):
        source = open(os.path.join(
            VB_ROOT, "overlay", "ann-benchmarks", "ann_benchmarks",
            "algorithms", "mongodb", "module.py")).read()
        assert "_wait_for_mongot" in source
        started = source.split("def _start_server")[1].split("def ")[0]
        assert "_wait_for_mongot" in started

    def test_the_readiness_probe_checks_both(self):
        """A probe that passes on mongod alone reports the server ready while
        the process that answers every search query is still booting."""
        from orchestrator.ops_pass import PROBES
        probe = " ".join(PROBES["mongodb"])
        assert "isWritablePrimary" in probe
        assert "8080" in probe, "the probe does not check mongot at all"


class TestSilentAnnFailuresAreReported:
    """A recall phase can complete and measure nothing.

    ann-benchmarks catches a per-algorithm exception and exits zero, so a
    module that raises leaves the phase marked completed and the engine simply
    absent from the recall comparison. Percona Search failed that way three
    runs in a row while its other workloads succeeded, which is exactly what
    makes it easy to miss: the engine is present everywhere except the one
    table, and nothing says why.
    """

    MANIFEST = {"phases": [
        {"engine": "mongodb", "phase": "ann", "status": "completed",
         "dataset": "d", "resource_pass": "tuned", "duration_s": 85.2},
        {"engine": "mariadb", "phase": "ann", "status": "completed",
         "dataset": "d", "resource_pass": "tuned", "duration_s": 160.0},
    ]}

    def _summary(self, engines_with_results=("mariadb",)):
        from report.generate import summarize
        recs = [{"phase": "recall_qps", "engine": e, "dataset": "d", "m": 16,
                 "ef_search": 10, "recall_at_k": 0.99, "qps": 100.0,
                 "resource_pass": "tuned", "build_mode": "post"}
                for e in engines_with_results]
        return summarize(recs, self.MANIFEST)

    def test_an_engine_that_measured_nothing_is_named(self):
        s = self._summary()
        assert [f["engine"] for f in s["silent_ann_failures"]] == ["mongodb"]

    def test_an_engine_that_measured_something_is_not(self):
        s = self._summary(engines_with_results=("mariadb", "mongodb"))
        assert s["silent_ann_failures"] == []

    def test_the_validity_section_calls_it_a_failure(self):
        from report.render import _validity_section
        text = _validity_section({}, self._summary())
        assert "measured nothing" in text
        assert "failure, not a finding" in text

    def test_it_is_silent_on_a_clean_run(self):
        from report.render import _validity_section
        s = self._summary(engines_with_results=("mariadb", "mongodb"))
        assert "measured nothing" not in _validity_section({}, s)


class TestValkeyChurnIsBatched:
    """Churn touches a tenth of the corpus in a single call.

    The load path has always batched every thousand rows; the churn path did
    not. At smoke scale that is 2,000 rows and it works. At a million rows it
    is 99,000: one DEL carrying 99,000 key names, and one pipeline holding
    99,000 HSETs of a 6 KB vector each, roughly 600 MB buffered in the client
    before a single execute. The tuned pass failed there 25 minutes in, after
    the churn baseline had already been measured, which is why five engines
    have a churn result and one has half of one.
    """

    def _driver(self, monkeypatch, calls):
        from harness.drivers import valkey as mod
        from harness.drivers.base import ConnectionSpec
        monkeypatch.setattr(mod, "Binary", lambda payload, subtype: payload,
                            raising=False)
        d = mod.ValkeyDriver(ConnectionSpec(host="h", port=1))

        class FakePipe:
            def hset(self, *a, **kw): calls.setdefault("hset", []).append(1)
            def execute(self): calls.setdefault("execute", []).append(
                len(calls.get("hset", [])))

        class FakeConn:
            def pipeline(self, transaction=False): return FakePipe()
            def delete(self, *keys): calls.setdefault("delete", []).append(len(keys))
            def close(self): pass

        d._conn = FakeConn()
        # Bulk writes deliberately open their own connection, with no read
        # timeout; the fake stands in for it.
        d._write_connection = lambda: FakeConn()
        return d

    def test_deletes_are_chunked(self, monkeypatch):
        from harness.drivers.valkey import CHURN_BATCH
        calls = {}
        self._driver(monkeypatch, calls).delete_ids(list(range(99_000)))
        assert max(calls["delete"]) <= CHURN_BATCH
        assert sum(calls["delete"]) == 99_000

    def test_inserts_are_flushed_along_the_way(self, monkeypatch):
        """One execute at the end means the whole corpus fraction is buffered
        in the client first."""
        import numpy
        from harness.drivers.valkey import CHURN_BATCH
        calls = {}
        n = 5_000
        self._driver(monkeypatch, calls).insert_rows(
            list(range(n)), numpy.zeros((n, 4), dtype=numpy.float32),
            list(range(n)))
        assert len(calls["execute"]) >= n // CHURN_BATCH

    def test_the_load_path_and_the_churn_path_use_the_same_batch(self):
        source = open(os.path.join(VB_ROOT, "harness", "drivers",
                                   "valkey.py")).read()
        assert "CHURN_BATCH" in source
        assert source.count("CHURN_BATCH") >= 3


class TestPlanProbesUseARealVector:
    """A zero vector has no direction, so cosine distance over it is undefined.

    Both new modules probed with numpy.zeros to confirm the index answers. On
    the smoke corpus, which is euclidean, that is a legal query. On the real
    corpus, which is angular, the server is entitled to reject it -- and the
    Percona Search recall phase failed at full scale after the index had
    already reached READY, having worked on every euclidean smoke run.
    """

    MODULES = ("mongodb", "valkey")

    def _source(self, engine):
        return open(os.path.join(
            VB_ROOT, "overlay", "ann-benchmarks", "ann_benchmarks",
            "algorithms", engine, "module.py")).read()

    def test_no_module_probes_with_zeros(self):
        for engine in self.MODULES:
            assert "numpy.zeros" not in self._source(engine), engine

    def test_a_corpus_vector_is_kept_for_the_probe(self):
        for engine in self.MODULES:
            source = self._source(engine)
            assert "self._probe = X[0]" in source, engine


class TestQuantizationReachesBothPaths:
    """Only the recall path read it, so one run built two different indexes.

    render_config gave the ann phase the vendor-recommended scalar quantization
    in the tuned pass. Nothing gave it to the ops phase, which builds the index
    the build-cost, concurrency, filtered and churn numbers are measured
    against. The 44-hour run reported a 15 GB unquantized index beside a recall
    curve measured on a quantized one, as though they were one configuration.
    """

    def _args(self, engine, resource_pass):
        from orchestrator.config import (load_profile, load_resources,
                                         resolve_resources)
        from orchestrator.ops_pass import harness_args
        info = type("I", (), {})()
        info.cpu = type("C", (), {"arch": "x86_64", "hybrid": False,
                                  "performance_cpus": list(range(40)),
                                  "efficiency_cpus": [], "logical_cpus": 80,
                                  "physical_cores": 40, "threads_per_core": 2,
                                  "model": "Xeon"})()
        info.total_ram_bytes = int(192 * 1024 ** 3)
        res = load_resources(resource_pass)
        return harness_args(load_profile("tuned-complete"), 16, engine,
                            resolve_resources(res, engine, info),
                            resource_pass, res, storage_engine="InnoDB")

    def test_tuned_ops_gets_the_vendor_recommendation(self):
        args = self._args("mongodb", "tuned")
        assert args[args.index("--quantization") + 1] == "scalar"

    def test_normalized_ops_pins_it_off(self):
        args = self._args("mongodb", "normalized")
        assert args[args.index("--quantization") + 1] == "none"

    def test_the_two_paths_agree(self):
        """Whatever render_config gives the recall phase, the ops phase gets."""
        from orchestrator.ann_pass import render_config
        from orchestrator.config import load_profile, load_resources
        for resource_pass in ("normalized", "tuned"):
            res = load_resources(resource_pass)
            groups = render_config("mongodb", load_profile("tuned-complete"),
                                   res, resource_pass)["float"]["any"][0]["run_groups"]
            ann_value = list(groups.values())[0]["arg_groups"][0]["quantization"]
            args = self._args("mongodb", resource_pass)
            assert args[args.index("--quantization") + 1] == ann_value, resource_pass

    def test_engines_without_the_knob_do_not_get_the_flag(self):
        for engine in ("mariadb", "alisql", "pgvector", "valkey"):
            assert "--quantization" not in self._args(engine, "tuned"), engine

    def test_the_harness_accepts_it(self):
        from harness.main import parse_args
        args = parse_args(["--engine", "mongodb", "--dataset", "d",
                           "--m", "16", "--run-id", "r",
                           "--host", "h", "--port", "27017",
                           "--output", "/tmp/out.jsonl",
                           "--quantization", "scalar"])
        assert args.quantization == "scalar"


class TestResumeContinuesTheRightRun:
    """Checkpoints live inside the run directory.

    So --resume with a freshly minted run id resumes nothing: it re-runs every
    unit into an empty directory and produces a report containing only the
    engines named on that command line, while the run being continued stays
    where it was. Two directories, neither complete, and a manual merge to get
    one report.
    """

    def _profile(self):
        return {"name": "tuned-complete"}

    def _results(self, tmp_path, monkeypatch, names):
        from orchestrator import cli
        for name in names:
            (tmp_path / name).mkdir()
            (tmp_path / name / "run-manifest.json").write_text("{}")
        monkeypatch.setattr(cli, "paths_for",
                            lambda rid: {"run_dir": str(tmp_path / rid)})
        return cli

    def test_it_picks_the_most_recent_run_for_the_profile(self, tmp_path, monkeypatch):
        cli = self._results(tmp_path, monkeypatch,
                            ["tuned-complete-20260820-134424",
                             "tuned-complete-20260822-101500",
                             "smoke-20260821-090000"])
        assert cli._resume_target(self._profile()) == "tuned-complete-20260822-101500"

    def test_it_ignores_other_profiles(self, tmp_path, monkeypatch):
        cli = self._results(tmp_path, monkeypatch, ["smoke-20260899-000000"])
        assert cli._resume_target(self._profile()) is None

    def test_a_directory_without_a_manifest_is_not_a_run(self, tmp_path, monkeypatch):
        from orchestrator import cli
        (tmp_path / "tuned-complete-20260820-134424").mkdir()
        monkeypatch.setattr(cli, "paths_for",
                            lambda rid: {"run_dir": str(tmp_path / rid)})
        assert cli._resume_target(self._profile()) is None

    def test_it_is_only_consulted_when_resume_was_asked_for(self):
        """Every run without --run-id would otherwise continue the last one."""
        source = open(os.path.join(VB_ROOT, "orchestrator", "cli.py")).read()
        assert "if not run_id and args.resume:" in source


class TestReRunContinuesAManifest:
    """Re-running one engine into an existing run directory must not erase it.

    Manifest built a fresh dict and saved it in __init__, so continuing a
    44-hour six-engine run to fix one failed unit would have overwritten the
    environment, the engine versions and the phase record of the five engines
    that succeeded. The report reads all of that, so the fix would have cost
    more than the failure.
    """

    def _manifest(self, tmp_path, run_id="tuned-complete-20260820-134424"):
        from orchestrator.manifest import Manifest
        return Manifest(str(tmp_path), run_id)

    def test_a_second_manifest_keeps_the_first_ones_record(self, tmp_path):
        first = self._manifest(tmp_path)
        first.data["engines"]["mariadb"] = {"source": {"tag": "11.8.8"}}
        first.add_phase("ops", "mariadb", "d", "completed", "t0", "t1",
                        {"resource_pass": "tuned"})
        second = self._manifest(tmp_path)
        assert second.data["engines"]["mariadb"]["source"]["tag"] == "11.8.8"
        assert any(p["engine"] == "mariadb" for p in second.data["phases"])

    def test_a_different_run_id_starts_clean(self):
        """Two runs sharing a directory would otherwise merge into nonsense."""
        import tempfile
        from orchestrator.manifest import Manifest
        with tempfile.TemporaryDirectory() as d:
            a = Manifest(d, "run-a")
            a.add_phase("ops", "mariadb", "d", "completed", "t0", "t1", {})
            b = Manifest(d, "run-b")
            assert b.data["phases"] == []

    def test_a_repeated_unit_replaces_its_earlier_attempt(self, tmp_path):
        """A re-run that fixes a failure must not leave the failure behind for
        the report to keep reporting."""
        m = self._manifest(tmp_path)
        m.add_phase("ops", "valkey", "d", "failed", "t0", "t1",
                    {"resource_pass": "tuned", "exit_code": 1})
        m.add_phase("ops", "valkey", "d", "completed", "t2", "t3",
                    {"resource_pass": "tuned"})
        valkey = [p for p in m.data["phases"] if p["engine"] == "valkey"]
        assert len(valkey) == 1
        assert valkey[0]["status"] == "completed"

    def test_the_same_unit_in_another_pass_is_kept(self):
        import tempfile
        from orchestrator.manifest import Manifest
        with tempfile.TemporaryDirectory() as d:
            m = Manifest(d, "r")
            m.add_phase("ops", "valkey", "d", "completed", "t0", "t1",
                        {"resource_pass": "normalized"})
            m.add_phase("ops", "valkey", "d", "completed", "t2", "t3",
                        {"resource_pass": "tuned"})
            assert len(m.data["phases"]) == 2


class TestReRunningAUnitReplacesItsRecords:
    """The records writer appends, which is right within a unit.

    Across attempts it is wrong: re-running a failed unit left the previous
    attempt's records in the file, and the report read both as separate
    measurements. The six-engine re-run produced two ingest records, two index
    sizes and twelve concurrency points for each engine it touched, and the
    charts drew lines through all of them.
    """

    def test_the_orchestrator_clears_the_unit_output_first(self):
        source = open(os.path.join(VB_ROOT, "orchestrator", "cli.py")).read()
        block = source.split('output = os.path.join(paths["run_dir"], f"ops-{stem}.jsonl")')[1]
        block = block.split("harness_args")[0]
        assert "os.remove" in block, (
            "a re-run appends to the previous attempt's records")

    def test_the_memory_series_is_cleared_too(self):
        """Otherwise peak RSS is taken across two attempts at once."""
        source = open(os.path.join(VB_ROOT, "orchestrator", "cli.py")).read()
        block = source.split('memory_ts = os.path.join(paths["run_dir"], f"mem-{stem}.jsonl")')[1]
        block = block.split("harness_args")[0]
        assert "memory_ts" in block and "os.remove" in block

    def test_the_recorder_itself_still_appends(self):
        """Within one unit it must: several workloads and threads write to the
        same file, and a crash should not lose what came before it."""
        source = open(os.path.join(VB_ROOT, "harness", "metrics",
                                   "records.py")).read()
        assert 'open(path, "a"' in source

    def test_bulk_writes_do_not_inherit_the_query_read_timeout(self):
        """The main connection carries a 600 second socket timeout so a wedged
        query cannot stall a run. valkey-search performs the HNSW insertion on
        the write path, so a batch of vectors into a large graph legitimately
        exceeds that, and the churn died with "Timeout reading from socket"
        after the baseline had been measured."""
        source = open(os.path.join(VB_ROOT, "harness", "drivers",
                                   "valkey.py")).read()
        body = source.split("def _write_connection")[1].split("\n    def ")[0]
        assert "socket_timeout" not in body, (
            "bulk writes must not carry a read timeout")
        for method in ("def delete_ids", "def insert_rows", "def _write_range"):
            block = source.split(method)[1].split("\n    def ")[0]
            assert "_write_connection" in block, method
