"""The engine registry, pinned against what the hardcoded dictionaries said.

Every value below was read out of the six per-engine dicts that used to live in
ann_pass, ops_pass, cli, charts and render, before they were moved into
config/engines/*.yml. The point of the table is that the move changed nothing:
a wrong port or a swapped data mount here would not crash, it would measure the
wrong thing or fail an hour into a run.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator import engines as engines_mod  # noqa: E402

# name -> (driver, ann_constructor, port, data_mount, server_data_mount,
#          (user, password), group, colour, marker, linestyle,
#          chart label, report label)
GOLDEN = {
    "mariadb": ("MariaDBDriver", "MariaDB", 3306, "/var/lib/vbench",
                "/server-data/data", ("bench", "bench"), "original",
                "#1f77b4", "o", "-", "MariaDB 11.8 (MHNSW)", "MariaDB 11.8 (MHNSW)"),
    "alisql": ("AliSQLDriver", "AliSQL", 3306, "/var/lib/vbench",
               "/server-data/data", ("bench", "bench"), "original",
               "#d62728", "s", "--", "AliSQL (VIDX)", "AliSQL (VIDX)"),
    "pgvector": ("PostgresDriver", "PGVector", 5432, "/var/lib/postgresql",
                 None, ("postgres", ""), "original",
                 "#2ca02c", "^", "-.", "PostgreSQL (pgvector)", "PostgreSQL (pgvector)"),
    "mariadb123": ("MariaDBDriver", "MariaDB123", 3306, "/var/lib/vbench",
                   "/server-data/data", ("bench", "bench"), "extra",
                   "#5fa8d3", "s", "--", "MariaDB 12.3 (MHNSW)", "MariaDB 12.3 (MHNSW)"),
    "mongodb": ("MongoDriver", "PerconaSearch", 27017, "/var/lib/vbench",
                "/server-data/mongot", ("bench", "bench"), "extra",
                "#9467bd", "v", "-", "Percona Search (mongot)",
                "Percona Search for MongoDB (mongot)"),
    "valkey": ("ValkeyDriver", "ValkeySearch", 6379, "/var/lib/vbench",
               None, ("", ""), "extra",
               "#e377c2", "D", "-", "Valkey (valkey-search)", "Valkey (valkey-search)"),
}

# The order a run iterates and a report presents: the original three, then the
# engines added later. Directory discovery is alphabetical and would not.
GOLDEN_ORDER = ("mariadb", "alisql", "pgvector", "mariadb123", "mongodb", "valkey")


@pytest.fixture(scope="module")
def registry():
    return engines_mod.registry(refresh=True)


class TestRegistry:
    def test_every_engine_is_found(self, registry):
        assert set(registry) == set(GOLDEN)

    def test_iteration_order_is_preserved(self):
        assert engines_mod.known_engines(refresh=True) == GOLDEN_ORDER

    @pytest.mark.parametrize("name", sorted(GOLDEN))
    def test_runtime_matches_the_old_constants(self, registry, name):
        want = GOLDEN[name]
        got = registry[name]
        assert got.driver == want[0]
        assert got.ann_constructor == want[1]
        assert got.port == want[2]
        assert got.data_mount == want[3]
        assert got.server_data_mount == want[4]
        assert got.credentials == want[5]
        assert got.group == want[6]

    @pytest.mark.parametrize("name", sorted(GOLDEN))
    def test_presentation_matches_the_old_constants(self, registry, name):
        want = GOLDEN[name]
        got = registry[name]
        assert got.color == want[7]
        assert got.marker == want[8]
        assert got.linestyle == want[9]
        assert got.chart_label == want[10]
        assert got.label == want[11]

    @pytest.mark.parametrize("name", sorted(GOLDEN))
    def test_every_engine_has_a_readiness_probe(self, registry, name):
        """A probe that is missing makes the server look permanently unready."""
        probe = registry[name].probe
        assert probe and probe[0] == "sh" and probe[1] == "-c"
        assert probe[2].strip()

    def test_style_shape_matches_charts(self, registry):
        style = registry["mariadb"].style
        assert set(style) == {"color", "marker", "linestyle", "label"}

    def test_groups(self):
        assert engines_mod.engines_in_group("original") == \
            ("mariadb", "alisql", "pgvector")
        assert engines_mod.engines_in_group("extra") == \
            ("mariadb123", "mongodb", "valkey")

    def test_get_unknown_names_the_alternatives(self):
        with pytest.raises(ValueError) as excinfo:
            engines_mod.get("nope")
        assert "mariadb" in str(excinfo.value)

    def test_as_dict_is_json_safe(self, registry):
        import json
        for name in registry:
            json.dumps(engines_mod.as_dict(name))


class TestMalformedConfigs:
    def test_missing_required_key_is_reported_not_raised(self, tmp_path, monkeypatch, capsys):
        directory = tmp_path / "engines"
        directory.mkdir()
        (directory / "broken.yml").write_text("name: broken\nruntime:\n  port: 1\n")
        (directory / "fine.yml").write_text(
            "name: fine\nruntime:\n  order: 1\n  driver: D\n  ann_constructor: C\n"
            "  port: 1\n  data_mount: /d\n")

        monkeypatch.setattr(engines_mod, "ENGINES_DIR", str(directory))
        monkeypatch.setattr(engines_mod, "load_engine",
                            lambda n: __import__("yaml").safe_load(
                                (directory / f"{n}.yml").read_text()))
        found = engines_mod.registry(refresh=True)
        assert set(found) == {"fine"}
        assert "ignoring broken" in capsys.readouterr().out

    def test_engine_without_chart_colour_gets_one(self, tmp_path, monkeypatch):
        directory = tmp_path / "engines"
        directory.mkdir()
        (directory / "newthing.yml").write_text(
            "name: newthing\nruntime:\n  driver: D\n  ann_constructor: C\n"
            "  port: 1\n  data_mount: /d\n")
        monkeypatch.setattr(engines_mod, "ENGINES_DIR", str(directory))
        monkeypatch.setattr(engines_mod, "load_engine",
                            lambda n: __import__("yaml").safe_load(
                                (directory / f"{n}.yml").read_text()))
        engine = engines_mod.registry(refresh=True)["newthing"]
        assert engine.color.startswith("#")
        assert engine.label == "newthing"


def teardown_module(_module):
    engines_mod.registry(refresh=True)
