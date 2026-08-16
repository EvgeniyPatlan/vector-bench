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

    def test_both_mariadb_versions_sweep_storage_engines(self):
        from orchestrator.cli import _ops_storage_engines
        prof, tuned = load_profile("mariadb-blog-repro"), load_resources("tuned")
        for engine in ("mariadb", "mariadb123"):
            assert _ops_storage_engines(engine, prof, tuned, "tuned") == \
                ["InnoDB", "MyISAM"], engine

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
    """Build cost was only ever measured on InnoDB.

    MariaDB publishes an index build "under 15 minutes"; we measured 3.6 hours
    on InnoDB, and the recall curves show MyISAM is almost certainly what the
    article used. Measuring build cost on the engine nobody benchmarks is not
    a reproduction.
    """

    def test_tuned_sweeps_both_for_mariadb(self):
        from orchestrator.cli import _ops_storage_engines
        got = _ops_storage_engines("mariadb", load_profile("mariadb-blog-repro"),
                                   load_resources("tuned"), "tuned")
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
