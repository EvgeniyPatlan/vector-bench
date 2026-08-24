"""Markdown and self-contained HTML rendering for the vector-bench report.

Two rules shape the output:

* **Environment before results.** The hardware, engine versions and resource
  limits come first, so nobody reads a number without the context that scopes it.
* **Caveats are not a footer.** Anything that invalidates or narrows a result —
  a full-scan fallback, a missing SIMD path, an engine that returned fewer than
  k rows — appears in a Validity section above the charts, not buried at the end.

The HTML is fully self-contained: SVG charts are inlined, CSS is inline, and no
request leaves the page. It renders in both light and dark themes.
"""

from __future__ import annotations

import html as _html
import json
import os
from typing import Any, Dict, List, Optional

ENGINE_LABEL = {
    "mariadb": "MariaDB 11.8 (MHNSW)",
    "mariadb123": "MariaDB 12.3 (MHNSW)",
    "alisql": "AliSQL (VIDX)",
    "pgvector": "PostgreSQL (pgvector)",
    "mongodb": "Percona Search for MongoDB (mongot)",
    "valkey": "Valkey (valkey-search)",
}


def _fmt_bytes(value: Optional[float]) -> str:
    if value is None:
        return "—"
    value = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} PiB"


def _fmt(value: Any, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:,.{digits}f}{suffix}"
    return f"{value}{suffix}"


def _label(engine: str) -> str:
    return ENGINE_LABEL.get(engine, engine)


# ---------------------------------------------------------------------------
# Shared section builders (returned as Markdown; HTML converts them)
# ---------------------------------------------------------------------------

def _environment_rows(manifest: Dict[str, Any]) -> List[List[str]]:
    host = manifest.get("host", {}) or {}
    cpu = host.get("cpu", {}) or {}
    config = manifest.get("config", {}) or {}
    resolved = config.get("resolved_resources", {}) or {}

    simd = ", ".join(cpu.get("simd_flags", []) or []) or "none detected"
    rows = [
        ["CPU", f"{cpu.get('model', '?')} — {cpu.get('logical_cpus', '?')} logical / "
                f"{cpu.get('physical_cores', '?')} physical cores"],
        ["SIMD", f"{simd}  (AVX-512: {'yes' if cpu.get('has_avx512') else 'no'})"],
        ["Hybrid cores", (f"yes — {len(cpu.get('performance_cpus', []))} performance / "
                          f"{len(cpu.get('efficiency_cpus', []))} efficiency"
                          if cpu.get("hybrid") else "no")],
        ["RAM", _fmt_bytes(host.get("total_ram_bytes"))],
        ["Kernel", host.get("kernel", "?")],
        ["Docker", host.get("docker_version", "?") or "?"],
        ["cgroup", host.get("cgroup_version", "?")],
        ["Resource pass", str(config.get("resource_pass", "?"))],
        ["Server cpuset", str(resolved.get("server_cpuset", "?"))],
        ["Server memory limit", _fmt_bytes(resolved.get("server_memory_bytes"))],
        ["Buffer pool / shared_buffers", _fmt_bytes(resolved.get("buffer_bytes"))],
        ["Graph cache", _fmt_bytes(resolved.get("graph_cache_bytes"))],
        ["Build threads", str(resolved.get("build_threads", "?"))],
    ]
    return rows


def _engine_rows(manifest: Dict[str, Any]) -> List[List[str]]:
    rows = []
    for engine, info in sorted((manifest.get("engines", {}) or {}).items()):
        source = info.get("source", {}) or {}
        build = info.get("build", {}) or {}
        if source.get("kind") == "image":
            # No tag, no commit, no -march, because nothing was compiled. The
            # digest is the provenance. Printing "?" in those columns where
            # every other engine shows a commit reads as data we failed to
            # collect, and there is none to collect.
            digest = (source.get("mongot_digest") or source.get("server_digest")
                      or "")
            rows.append([
                _label(engine),
                source.get("version", "image"),
                digest.split("@")[-1][:19] if digest else "not pinned",
                "n/a (JVM)",
                "published image",
            ])
            continue
        rows.append([
            _label(engine),
            source.get("tag", "?"),
            (source.get("commit", "") or "")[:12] or "?",
            build.get("march", "?"),
            build.get("build_type", "?"),
        ])
    return rows


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    if not rows:
        return "\n_No data._\n"
    # Leading blank line: most Markdown renderers require one between a
    # paragraph and a table, and these tables always follow prose.
    out = ["", "| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join("—" if c in (None, "None", "") else str(c)
                                     for c in row) + " |")
    return "\n".join(out) + "\n"


def _fmt_age(run_started: Any, measured_at: Any) -> str:
    try:
        delta = float(run_started) - float(measured_at)
    except (TypeError, ValueError):
        return "—"
    if delta < 3600:
        return f"{delta / 60:.0f} min"
    if delta < 86400:
        return f"{delta / 3600:.1f} h"
    return f"{delta / 86400:.1f} days"


def _validity_section(manifest: Dict[str, Any], summary: Dict[str, Any]) -> str:
    parts: List[str] = []
    warnings = manifest.get("warnings", []) or []

    if summary.get("failed_phases"):
        parts.append(
            "### Phases that FAILED — these results are incomplete\n\n"
            "The run did not finish every unit it was asked to. Any engine or "
            "resource pass listed here is **missing from the comparisons below**, "
            "and its absence is a failure rather than a finding.\n"
        )
        rows = [[_label(f.get("engine", "?")), f.get("phase", "?"),
                 f.get("resource_pass", "?"), f.get("dataset", "?"),
                 str(f.get("exit_code", "?")),
                 f"{(f.get('duration_s') or 0) / 60:.1f} min"]
                for f in summary["failed_phases"]]
        parts.append(_md_table(
            ["Engine", "Phase", "Pass", "Dataset", "Exit code", "Ran for"], rows))

    if summary.get("silent_ann_failures"):
        parts.append(
            "\n### Recall phases that ran and measured nothing\n\n"
            "The engines below completed their recall phase and wrote **no "
            "measurements at all**. ann-benchmarks catches a per-algorithm "
            "exception and exits successfully, so a module that raises leaves "
            "the phase marked completed and the engine simply absent from the "
            "comparison below. That absence is a failure, not a finding, and "
            "the engine's other workloads may still have succeeded, which "
            "makes it easy to miss. The reason is in the run log for that "
            "phase, not in this report.\n"
        )
        rows = [[_label(f.get("engine", "?")), f.get("dataset") or "—",
                 f.get("resource_pass") or "—",
                 f"{(f.get('duration_s') or 0) / 60:.1f} min"]
                for f in summary["silent_ann_failures"][:20]]
        parts.append(_md_table(
            ["Engine", "Dataset", "Pass", "Ran for"], rows))

    if summary.get("plan_failures"):
        parts.append(
            "### Measurements that did not use the vector index\n\n"
            "The following measurements ran with the vector index **absent from the "
            "query plan**. A full scan returns exact results slowly, so these appear "
            "in the data as high recall at low throughput. They are not ANN "
            "measurements and must not be read as such.\n"
        )
        rows = [[_label(f.get("engine", "?")), f.get("dataset", "?"),
                 f.get("phase", "?"), str(f.get("m", "—")),
                 str(f.get("ef_search", "—")),
                 # dict.get's default does not fire on an explicit None, which
                 # is what an unfiltered measurement stores — so "None" would
                 # leak into the table where "not applicable" is meant.
                 ("—" if f.get("selectivity") is None
                  else f"{f['selectivity']:.0%}")]
                for f in summary["plan_failures"][:40]]
        parts.append(_md_table(
            ["Engine", "Dataset", "Phase", "M", "ef_search", "Selectivity"], rows))
        if len(summary["plan_failures"]) > 40:
            parts.append(f"\n_…and {len(summary['plan_failures']) - 40} more._\n")

    if "mongodb" in set(summary.get("engines") or []):
        parts.append(
            "\n### One engine is a Technical Preview\n\n"
            "**Percona Search for MongoDB is a Technical Preview and its own "
            "documentation says it is not for production use.** It is measured "
            "here beside four generally available engines. Treat a number for it "
            "as an indication of where the implementation currently stands "
            "rather than as a property of a shipped product, and expect it to "
            "move between releases in ways the others will not.\n"
        )

    if summary.get("recall_floor_gaps"):
        parts.append(
            "\n### Recall floors an engine never approached\n\n"
            "`QPS @ recall≥F` is a comparison only where both engines have "
            "measurements near F. For each engine below, **every configuration "
            "swept returned recall above the floor**, so the figure reported at "
            "that floor is the same measurement as at every lower one, taken at "
            "a materially higher accuracy than the column asks for. This is a "
            "limit of the grid rather than a fault: `ef_search` cannot go below "
            "k, and where an engine exposes no `ef_construction` there is no "
            "setting that makes its index less accurate at a fixed M.\n"
        )
        rows = [[_label(g.get("engine", "?")), g.get("dataset", "?"),
                 f"≥{g['floor']:.2f}",
                 f"{g['lowest_recall']:.4f}",
                 f"{g['measured_at']:.4f}" if g.get("measured_at") is not None else "—",
                 _fmt(g.get("qps"), 0)]
                for g in summary["recall_floor_gaps"][:30]]
        parts.append(_md_table(
            ["Engine", "Dataset", "Floor reported", "Lowest recall measured",
             "Figure taken at recall", "QPS"], rows))
        if len(summary["recall_floor_gaps"]) > 30:
            parts.append(
                f"\n_…and {len(summary['recall_floor_gaps']) - 30} more._\n")

    if summary.get("duplicate_ann"):
        parts.append(
            "\n### The same configuration was measured more than once\n\n"
            "Every row below appeared twice or more in the results this report "
            "read, which means it is reading **more than one measurement tree** "
            "— typically results from an older resource budget sitting alongside "
            "the current ones. The charts pick the best value at each point, so "
            "they are a blend of two configurations rather than a measurement of "
            "either. Point the report at a single tree with `--annb-results`, or "
            "delete the stale one.\n"
        )
        rows = [[_label(c.get("engine", "?")), c.get("dataset", "?"),
                 str(c.get("m", "—")), str(c.get("ef_search", "—")),
                 str(c.get("count")), ", ".join(str(q) for q in c.get("qps", []))]
                for c in summary["duplicate_ann"][:30]]
        parts.append(_md_table(
            ["Engine", "Dataset", "M", "ef_search", "Copies", "QPS seen"], rows))
        if len(summary["duplicate_ann"]) > 30:
            parts.append(f"\n_…and {len(summary['duplicate_ann']) - 30} more._\n")

    if summary.get("stale_ann"):
        parts.append(
            "\n### Recall results that predate this run\n\n"
            "ann-benchmarks skips any configuration that already has a result "
            "file and reports that as success, so a re-run after a configuration "
            "change returns instantly with the **previous** numbers. The recall "
            "and throughput figures below were measured before this run started "
            "and do not reflect its resource limits or harness version. Re-run "
            "the ann phase with `--force` to recompute them.\n"
        )
        seen, rows = set(), []
        for c in summary["stale_ann"]:
            key = (c.get("engine"), c.get("dataset"))
            if key in seen:
                continue
            seen.add(key)
            rows.append([_label(c.get("engine", "?")), c.get("dataset", "?"),
                         _fmt_age(c.get("run_started"), c.get("measured_at"))])
        parts.append(_md_table(["Engine", "Dataset", "Measured before this run by"], rows))

    if summary.get("memory_pressure"):
        parts.append(
            "\n### Engines that ran against their memory limit\n\n"
            "These phases spent part of their run at the container memory limit, "
            "which means the kernel was reclaiming continuously. Throughput and "
            "latency for them describe **the budget they were given rather than "
            "the implementation**, and they are not comparable with an engine that "
            "had headroom. Raise `memory.server_limit_gb` for this corpus and "
            "re-run before drawing any conclusion from these numbers.\n"
        )
        rows = [[_label(c.get("engine", "?")),
                 _fmt_bytes(c.get("peak_bytes")),
                 _fmt_bytes(c.get("limit_bytes")),
                 f"{c.get('fraction_at_ceiling', 0):.0%}",
                 f"{c.get('first_hit_fraction', 0):.0%} in"]
                for c in summary["memory_pressure"]]
        parts.append(_md_table(
            ["Engine", "Peak", "Limit", "Time at the limit", "First reached"], rows))

    if summary.get("short_result_cases"):
        parts.append(
            "\n### Filtered queries that returned fewer than k results\n\n"
            "With post-filtering, an engine can exhaust its candidate list before "
            "finding k qualifying rows. This is real engine behaviour, not a harness "
            "fault, but it means the recall figure for these points is computed over "
            "short result sets and should be read alongside the count.\n"
        )
        # Build mode and pass are what tell these apart. pgvector is measured
        # in two build modes across two passes, so four real measurements
        # rendered as the same row four times and read as a rendering fault.
        rows = [[_label(c.get("engine", "?")), c.get("dataset", "?"),
                 c.get("resource_pass") or "—", c.get("build_mode") or "—",
                 f"{(c.get('selectivity') or 0):.0%}", str(c.get("queries", "—"))]
                for c in summary["short_result_cases"][:30]]
        parts.append(_md_table(
            ["Engine", "Dataset", "Pass", "Build mode", "Selectivity",
             "Queries short of k"], rows))

    if warnings:
        parts.append("\n### Environment warnings\n\n")
        for w in warnings:
            parts.append(f"- {w}\n")

    if not parts:
        parts.append(
            "No validity problems were detected: every measurement used the vector "
            "index, no filtered query returned short, and the environment raised no "
            "warnings.\n"
        )
    return "".join(parts)


def _known_asymmetries(summary: Optional[Dict[str, Any]] = None) -> str:
    """Structural differences no configuration removes.

    Conditional on what actually ran. A row about a separate index process in a
    report of four in-engine indexes describes nothing, and a reader has to
    work out that it does not apply.
    """
    engines = set((summary or {}).get("engines") or [])
    # Which engines expose the build-quality knob is a property of the run, not
    # a constant. It was pgvector alone until valkey-search joined; the rule it
    # justifies is unchanged, but stating the old reason would be false and a
    # reader checking it against the build table would catch it.
    tunable = sorted(engines & {"pgvector", "valkey"})
    if len(tunable) > 1:
        ef_row = [
            "`ef_construction` is exposed by " + " and ".join(
                _label(e) for e in tunable),
            "Build quality can be tuned for those two and not for MHNSW or VIDX, "
            "so it is pinned to the default in the normalized pass and swept only "
            "in the tuned pass. Two engines of four having the knob is still an "
            "asymmetry against MariaDB and AliSQL, which have no equivalent.",
        ]
    else:
        ef_row = [
            "`ef_construction` is exposed only by pgvector",
            "Build quality can be tuned for pgvector but not for MHNSW or VIDX. "
            "It is pinned to the default in the normalized pass so pgvector is "
            "not given an axis the others lack.",
        ]

    rows = [
                ef_row,
                ["pgvector bulk-builds after load; MHNSW and VIDX build incrementally",
                 "Build cost is reported separately for both modes. A single "
                 "'build time' comparison would compare different operations."],
                ["AliSQL VIDX is InnoDB-only and requires READ COMMITTED",
                 "InnoDB is the headline storage engine for all engines. MariaDB's "
                 "MyISAM results, where present, are a MariaDB-only extra curve."],
                ["Client stacks differ for PostgreSQL",
                 "MariaDB and AliSQL share one client (MariaDB Connector/C) so their "
                 "numbers are mutually comparable. PostgreSQL necessarily uses "
                 "psycopg3; its client overhead differs by an unmeasured amount."],
                ["SIMD paths depend on the host CPU",
                 "Both MHNSW and VIDX document AVX-512 distance kernels. On a host "
                 "without AVX-512 both run narrower paths, and the ranking may differ "
                 "on AVX-512 hardware."],
    ]

    if "valkey" in engines:
        rows += [
            ["Valkey holds the whole dataset in memory",
             "The other engines keep a disk-backed table with a cache in front of "
             "it, so their container limit is a cache budget. Valkey's is the "
             "dataset. It has no buffer pool to size and no index on disk, and "
             "its index size below is resident memory rather than a file."],
            ["Valkey's planner chooses its filtering strategy per query",
             "It picks between pre-filtering and filtering inline during the "
             "search. The MySQL family and pgvector post-filter and Percona "
             "Search pre-filters, so the filtered section compares three "
             "strategies rather than one implemented three ways."],
            ["Valkey is installed from packages, not built from source",
             "Percona ships prebuilt packages, so it carries installed package "
             "versions instead of a tag, a commit and a `-march`, and the "
             "AVX-512 row above does not describe it."],
        ]

    if "mongodb" in engines:
        rows += [
            ["Percona Search keeps its index in a separate process",
             "mongod holds the documents and mongot holds a Lucene index fed from "
             "the change stream, where the other four keep the index inside the "
             "query engine. What is being compared includes where the index lives, "
             "not only how it is implemented."],
            ["Percona Search is not built from source",
             "It ships as packages and images only and runs on the JVM, so it "
             "carries an image digest instead of a tag, a commit and a `-march`. "
             "Its distance kernels come from the JVM's vector API, so the AVX-512 "
             "row above does not describe it."],
            ["Percona Search exposes no graph degree",
             "M is pinned at one value for every other engine and cannot be set "
             "here at all, so any row carrying an M for it is naming a sweep "
             "rather than a setting, and those comparisons are not at matched M."],
            ["Percona Search filters before the vector comparison",
             "Its `filter` is a pre-filter; the other four apply the predicate "
             "after the graph walk. That is why their filtered results either "
             "collapse in throughput or return fewer than k rows, and why a "
             "difference in the filtered section is a difference in kind."],
            ["Percona Search builds its index asynchronously",
             "createSearchIndex returns before the index exists and it is built "
             "while writes are still arriving, so its build cost is wall clock to "
             "a queryable index rather than the duration of a statement."],
        ]

    return (
        "These are structural differences between the engines that no configuration "
        "removes. They are why some comparisons below are presented in pairs rather "
        "than as a single ranking.\n\n"
        + _md_table(["Asymmetry", "Consequence for these results"], rows)
    )


def _headline_tables(summary: Dict[str, Any]) -> str:
    parts: List[str] = []
    for dataset, per_engine in sorted(summary.get("per_dataset", {}).items()):
        parts.append(f"\n#### {dataset}\n\n")
        rows = []
        for engine, entry in sorted(per_engine.items()):
            # The frontier spans every configuration the engine was swept over,
            # so on the tuned pass MariaDB's headline figures are MyISAM ones.
            # Naming the winner keeps the number from being read as the result
            # of the default build.
            multi = len(entry.get("storage_engines") or []) > 1

            def cell(floor: int) -> str:
                value = _fmt(entry.get(f"qps_at_recall_{floor}"), 0)
                storage = entry.get(f"qps_at_recall_{floor}_storage")
                return f"{value} ({storage})" if multi and storage and value != "—" else value

            # The range, not only the best. Two identical floor columns mean
            # the engine has no measurement between them, and the lowest recall
            # it reached is the number that says so.
            low, high = entry.get("min_recall"), entry.get("max_recall", 0)
            span = f"{low:.4f} to {high:.4f}" if low is not None else f"{high:.4f}"
            rows.append([
                _label(engine),
                cell(90), cell(95), cell(99),
                span,
                str(entry.get("points", 0)),
            ])
        parts.append(_md_table(
            ["Engine", "QPS @ recall≥0.90", "QPS @ recall≥0.95",
             "QPS @ recall≥0.99", "Recall range", "Points measured"],
            rows,
        ))
        parts.append(
            "\n_QPS at a recall floor is the comparison an operator makes: how fast "
            "is the engine at an accuracy I can accept. A dash means the engine did "
            "not reach that recall anywhere in the swept grid._\n"
        )
    return "".join(parts) or "_No recall/QPS data in this run._\n"


# The ops harness stamps a storage engine on every unit and defaults to
# InnoDB, which is meaningless for PostgreSQL. Its own ann records call the
# same thing `heap`; this keeps the two paths from disagreeing on the page.
_STORAGE_OVERRIDES = {"pgvector": "heap", "mongodb": "wiredTiger",
                      "valkey": "memory"}


def _storage(record: Dict[str, Any]) -> str:
    override = _STORAGE_OVERRIDES.get(record.get("engine"))
    return override or record.get("storage_engine") or "—"


def _build_table(summary: Dict[str, Any]) -> str:
    # Ingest rate lives on the `ingest` record, not on `index_build`; reading it
    # off the build record produced an em dash in every row.
    # Keyed on the storage engine as well. Without it MariaDB's InnoDB and
    # MyISAM builds collided and one rate was printed for both rows, which
    # reported MyISAM's 210 rows/s as 80 on 11.8 and its 305 as 51 on 12.3.
    ingest_by_key = {
        (i.get("engine"), i.get("dataset"), i.get("resource_pass"),
         i.get("build_mode"), i.get("storage_engine")):
            i.get("ingest_rows_per_s")
        for i in summary.get("ingest", [])
    }

    rows = []
    for r in sorted(summary.get("build", []),
                    key=lambda r: (r.get("dataset", ""), r.get("engine", ""),
                                   r.get("m") or 0, r.get("storage_engine") or "")):
        ingest_rate = ingest_by_key.get(
            (r.get("engine"), r.get("dataset"), r.get("resource_pass"),
             r.get("build_mode"), r.get("storage_engine")))
        rows.append([
            _label(r.get("engine", "?")), r.get("dataset", "?"), str(r.get("m", "—")),
            # MariaDB is measured on InnoDB and MyISAM under the tuned pass, and
            # without this column those two rows are identical on the page while
            # differing 6x in ingest rate.
            _storage(r),
            r.get("build_mode") or "—",
            _fmt(r.get("build_wall_s"), 1, " s"),
            _fmt(ingest_rate, 0, " rows/s"),
            _fmt_bytes(r.get("peak_rss_bytes")),
            _index_size(r),
            _index_build_kind(r),
        ])
    table = _md_table(
        ["Engine", "Dataset", "M", "Storage", "Build mode", "Build time",
         "Ingest rate", "Peak server RSS", "Index size", "Index build"],
        rows,
    )
    return table + _build_table_notes(summary)


def _index_size(record: Dict[str, Any]) -> str:
    """Index size, labelled when it is not a file.

    Valkey writes nothing to disk, so its figure is resident memory. Printing
    it bare in a column every other row fills with a file size invites a
    comparison between two different quantities.
    """
    size = _fmt_bytes(record.get("index_bytes"))
    if (record.get("extra") or {}).get("in_memory_only"):
        return f"{size} resident"
    return size


def _index_build_kind(record: Dict[str, Any]) -> str:
    """How the index came to exist, in three kinds rather than two.

    "no" was a fair answer while every engine either maintained its graph on
    INSERT or built it with a blocking statement afterwards. An index built by
    another process, asynchronously, while the writes are still arriving is
    neither, and calling it "no" files it beside MHNSW's incremental build,
    which is a different operation that also happens to lack a bulk step.
    """
    extra = record.get("extra") or {}
    if extra.get("async_index_build") or record.get("build_mode") == "async":
        ready = extra.get("index_ready_seconds")
        if ready:
            return f"async, ready {_fmt(ready / 60, 1)} min after load"
        return "async (separate process)"
    return "bulk after load" if extra.get("separable_build") else "incremental"


def _build_table_notes(summary: Dict[str, Any]) -> str:
    """Footnotes for values that are labels rather than settings."""
    notes: List[str] = []
    unapplied = sorted({
        _label(r.get("engine", "?")) for r in summary.get("build", [])
        if (r.get("extra") or {}).get("m_applied") is False
    })
    if unapplied:
        notes.append(
            f"**M is not a setting for {', '.join(unapplied)}.** It exposes no "
            "graph degree, so the value says which sweep the row belongs to "
            "rather than what the engine was told. Comparisons in this table "
            "are not at matched M."
        )
    if any(_index_build_kind(r).startswith("async")
           for r in summary.get("build", [])):
        notes.append(
            "**An async build overlaps the load.** Build time for those rows is "
            "the wall clock from the first write to the index reporting ready, "
            "because neither the load nor the wait alone is the cost of having "
            "a queryable index."
        )
    return ("\n" + "\n\n".join(notes) + "\n") if notes else ""


def _not_measured(workload: str, profile: Dict[str, Any]) -> str:
    """Say a workload was not enabled, rather than printing an empty table.

    A section with an empty table and a blank chart reads as a broken report.
    It usually just means `ops.workloads` did not include this one.
    """
    enabled = ((profile.get("ops") or {}).get("workloads")) or []
    return (
        f"_Not measured in this run._ The `{profile.get('name', 'profile')}` "
        f"profile ran `workloads: {list(enabled)}`, which does not include "
        f"`{workload}`. Add it to the profile and re-run to populate this "
        f"section; it reuses the corpus the build phase already loaded.\n"
    )


# The tuned pass measures MariaDB on both InnoDB and MyISAM, so an engine name
# alone does not identify a row. Without this column the two appeared as
# unexplained duplicates, and the largest result in the tuned-complete run --
# that MariaDB loses 3-10% of its throughput to churn on MyISAM against 84-86%
# on InnoDB -- was invisible on the page.
def _concurrency_table(summary: Dict[str, Any]) -> str:
    rows = []
    for r in sorted(summary.get("concurrency", []),
                    key=lambda r: (r.get("dataset", ""), r.get("engine", ""),
                                   r.get("storage_engine") or "",
                                   r.get("clients") or 0)):
        eff = (r.get("extra") or {}).get("scaling_efficiency")
        rows.append([
            _label(r.get("engine", "?")), r.get("dataset", "?"),
            _storage(r),
            str(r.get("clients", "—")),
            _fmt(r.get("qps"), 0),
            _fmt(r.get("latency_p50_ms"), 3),
            _fmt(r.get("latency_p95_ms"), 3),
            _fmt(r.get("latency_p99_ms"), 3),
            _fmt(eff, 2) if eff is not None else "—",
        ])
    return _md_table(
        ["Engine", "Dataset", "Storage", "Clients", "QPS", "p50 (ms)",
         "p95 (ms)", "p99 (ms)", "Scaling efficiency"],
        rows,
    )


def _filtered_table(summary: Dict[str, Any]) -> str:
    rows = []
    for r in sorted(summary.get("filtered", []),
                    key=lambda r: (r.get("dataset", ""), r.get("engine", ""),
                                   r.get("storage_engine") or "",
                                   r.get("selectivity") or 0)):
        extra = r.get("extra") or {}
        rows.append([
            _label(r.get("engine", "?")), r.get("dataset", "?"),
            _storage(r),
            f"{(r.get('selectivity') or 0):.0%}",
            _fmt(r.get("recall_at_k"), 4),
            _fmt(r.get("qps"), 0),
            _fmt(r.get("latency_p99_ms"), 3),
            "yes" if r.get("vector_index_used") else "no",
            str(extra.get("short_result_queries", 0)),
            extra.get("iterative_scan") or "—",
        ])
    return _md_table(
        ["Engine", "Dataset", "Storage", "Selectivity", "Recall@k", "QPS",
         "p99 (ms)", "Index used", "Short results", "Iterative scan"],
        rows,
    )


def _churn_table(summary: Dict[str, Any]) -> str:
    rows = []
    for r in sorted(summary.get("churn", []),
                    key=lambda r: (r.get("dataset", ""), r.get("engine", ""),
                                   r.get("storage_engine") or "",
                                   r.get("churn_fraction") or 0)):
        extra = r.get("extra") or {}
        rows.append([
            _label(r.get("engine", "?")), r.get("dataset", "?"),
            _storage(r),
            f"{(r.get('churn_fraction') or 0):.0%}",
            _fmt(r.get("recall_at_k"), 4),
            _fmt(extra.get("recall_drop_vs_baseline"), 4),
            _fmt(r.get("qps"), 0),
            _fmt_bytes(r.get("index_bytes")),
        ])
    return _md_table(
        ["Engine", "Dataset", "Storage", "Churn", "Recall@k",
         "Recall drop vs baseline", "QPS", "Index size"],
        rows,
    ) + _churn_notes(summary)


def _churn_notes(summary: Dict[str, Any]) -> str:
    """An engine that could not write its rows back.

    Blank cells read as a measurement that was not taken. This one was: the
    engine took the writes and did not finish them, and how far it got is the
    result. Reported here rather than left to be inferred from an empty row.
    """
    stalled = [r for r in summary.get("churn", [])
               if (r.get("extra") or {}).get("reinsert_completed") is False]
    if not stalled:
        return ""
    lines = ["\n**Re-insertion did not complete for every engine.** Recall and "
             "throughput are omitted for these rows on purpose: the corpus is "
             "missing rows, so a figure taken then describes neither the "
             "baseline nor a churned corpus. What the engine managed is the "
             "measurement.\n"]
    rows = []
    for r in stalled:
        extra = r.get("extra") or {}
        rows.append([
            _label(r.get("engine", "?")),
            f"{extra.get('rows_reinserted', 0):,} of "
            f"{extra.get('rows_expected', 0):,}",
            _fmt(extra.get("insert_seconds"), 0, " s"),
            _fmt(extra.get("reinsert_rows_per_s"), 1, " rows/s"),
            _fmt(extra.get("churn_budget_s"), 0, " s"),
        ])
    lines.append(_md_table(
        ["Engine", "Rows re-inserted", "Spent", "Rate", "Budget"], rows))
    return "".join(lines)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def render_markdown(manifest: Dict[str, Any], summary: Dict[str, Any],
                    records: List[Dict[str, Any]],
                    chart_paths: Dict[str, Dict[str, str]], title: str) -> str:
    run_id = manifest.get("run_id", "?")
    config = manifest.get("config", {}) or {}
    profile = config.get("profile", {}) or {}

    parts: List[str] = [
        f"# {title}\n\n",
        f"**MariaDB vs AliSQL vs PostgreSQL + pgvector** — HNSW vector search.\n\n",
        f"- Run: `{run_id}`\n",
        f"- Started: {manifest.get('started_at', '?')} · Finished: {manifest.get('finished_at', '?')}\n",
        f"- Profile: **{profile.get('name', '?')}** — {profile.get('description', '')}\n",
        f"- Status: {manifest.get('status', '?')}\n",
        f"- Records: {len(records):,}\n\n",
        "---\n\n## 1. Environment\n\n",
        "Every number below is scoped to this environment. Vector-search results do "
        "not transfer across CPU generations, and both MariaDB MHNSW and AliSQL VIDX "
        "select distance kernels from CPU features at runtime.\n\n",
        _md_table(["Property", "Value"], _environment_rows(manifest)),
        "\n### Engines under test\n\n",
        _md_table(["Engine", "Tag", "Commit", "-march", "Build type"],
                  _engine_rows(manifest)),
        "\n---\n\n## 2. Validity\n\n",
        _validity_section(manifest, summary),
        "\n---\n\n## 3. Known asymmetries\n\n",
        _known_asymmetries(summary),
        "\n---\n\n## 4. Recall vs throughput\n\n",
        "The headline comparison. Each engine's Pareto frontier is the best "
        "throughput it achieved at each recall level across the whole parameter "
        "sweep; the faint scatter behind it is every measured point.\n",
        _headline_tables(summary),
    ]

    for stem, paths in sorted(chart_paths.items()):
        if stem.split("-")[0] in ("pareto", "paretozoom", "qpsatrecall", "latency"):
            parts.append(f"\n![{stem}](charts/{os.path.basename(paths['svg'])})\n")

    parts += [
        "\n---\n\n## 5. Index build cost\n\n",
        "ann-benchmarks does not measure this. Note the **Separable build** column: "
        "pgvector builds its graph as one bulk operation after the load, while MHNSW "
        "and VIDX maintain theirs on every INSERT and have no separable build step. "
        "Where a run includes pgvector in `incremental` mode, that row is the "
        "like-for-like comparison.\n\n",
        _build_table(summary),
    ]
    for stem, paths in sorted(chart_paths.items()):
        if stem.split("-")[0] in ("build", "storage"):
            parts.append(f"\n![{stem}](charts/{os.path.basename(paths['svg'])})\n")

    parts += [
        "\n---\n\n## 6. Concurrency scaling\n\n",
        "Where a database should differ from an ANN library. MariaDB caches the "
        "MHNSW graph per `TABLE_SHARE`; AliSQL keeps a shared cache plus a "
        "per-transaction cache; pgvector serves graph pages from `shared_buffers`. "
        "Scaling efficiency of 1.0 is linear scaling.\n\n",
        (_concurrency_table(summary) if summary.get("concurrency")
         else _not_measured("concurrency", profile)),
    ]
    for stem, paths in sorted(chart_paths.items()):
        if stem.startswith("concurrency-"):
            parts.append(f"\n![{stem}](charts/{os.path.basename(paths['svg'])})\n")

    parts += [
        "\n---\n\n## 7. Filtered (hybrid) search\n\n",
        "Vector search with a scalar predicate, scored against ground truth "
        "recomputed exactly over the qualifying rows. **Short results** counts "
        "queries that returned fewer than k rows — a real consequence of "
        "post-filtering, not a harness fault.\n\n",
        (_filtered_table(summary) if summary.get("filtered")
         else _not_measured("filtered", profile)),
    ]
    for stem, paths in sorted(chart_paths.items()):
        if stem.startswith("filtered-"):
            parts.append(f"\n![{stem}](charts/{os.path.basename(paths['svg'])})\n")

    parts += [
        "\n---\n\n## 8. Churn\n\n",
        "Recall and throughput after deleting and re-inserting a fraction of rows. "
        "HNSW graphs degrade under deletion; how much differs by implementation.\n\n",
        (_churn_table(summary) if summary.get("churn")
         else _not_measured("churn", profile)),
    ]
    for stem, paths in sorted(chart_paths.items()):
        if stem.split("-")[0] in ("churn", "churnimpact", "passcompare") or stem == "memory-timeline":
            parts.append(f"\n![{stem}](charts/{os.path.basename(paths['svg'])})\n")

    parts += [
        "\n---\n\n## 9. Reproducing this run\n\n",
        "```bash\n"
        f"./run-benchmark.sh build --march {(config.get('resolved_resources') or {}).get('march', 'x86-64-v3')}\n"
        f"./run-benchmark.sh fetch --datasets {','.join(summary.get('datasets', []) or ['glove-100-angular'])}\n"
        f"./run-benchmark.sh run --profile {profile.get('name', 'quick')} "
        f"--resource-pass {config.get('resource_pass', 'normalized')}\n"
        "```\n\n",
        "The raw merged records are in `records.jsonl` next to this file, and the "
        "full provenance — image ids, source commits, resolved limits — is in "
        "`../run-manifest.json`.\n\n",
        "See `docs/05-methodology.md` for what is measured and how, and "
        "`docs/03-running-manually.md` to reproduce any single measurement by hand "
        "without the framework.\n",
    ]
    return "".join(parts)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

CSS = """
:root {
  --bg: #ffffff; --fg: #1a1a1a; --muted: #5a6270; --border: #e2e5ea;
  --accent: #1f77b4; --card: #f7f8fa; --warn-bg: #fff6e5; --warn-br: #e6a23c;
  --code-bg: #f2f4f7;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #14171c; --fg: #e6e8eb; --muted: #9aa4b2; --border: #2a2f38;
    --accent: #5aa9e6; --card: #1b1f26; --warn-bg: #2e2618; --warn-br: #c98a2b;
    --code-bg: #1f242c;
  }
}
:root[data-theme="light"] {
  --bg: #ffffff; --fg: #1a1a1a; --muted: #5a6270; --border: #e2e5ea;
  --accent: #1f77b4; --card: #f7f8fa; --warn-bg: #fff6e5; --warn-br: #e6a23c;
  --code-bg: #f2f4f7;
}
:root[data-theme="dark"] {
  --bg: #14171c; --fg: #e6e8eb; --muted: #9aa4b2; --border: #2a2f38;
  --accent: #5aa9e6; --card: #1b1f26; --warn-bg: #2e2618; --warn-br: #c98a2b;
  --code-bg: #1f242c;
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 2rem 1.25rem 5rem; background: var(--bg); color: var(--fg);
  font: 16px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
        "Helvetica Neue", Arial, sans-serif;
  -webkit-text-size-adjust: 100%;
}
.wrap { max-width: 1100px; margin: 0 auto; }
h1 { font-size: 1.9rem; line-height: 1.25; margin: 0 0 .35em; letter-spacing: -.01em; }
h2 { font-size: 1.35rem; margin: 2.4em 0 .6em; padding-bottom: .3em;
     border-bottom: 1px solid var(--border); }
h3 { font-size: 1.08rem; margin: 1.8em 0 .5em; }
h4 { font-size: .98rem; margin: 1.4em 0 .4em; color: var(--muted);
     text-transform: uppercase; letter-spacing: .04em; }
p, li { color: var(--fg); }
.sub { color: var(--muted); font-size: .95rem; margin-top: 0; }
a { color: var(--accent); }
code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            font-size: .875em; }
code { background: var(--code-bg); padding: .12em .38em; border-radius: 4px; }
pre { background: var(--code-bg); padding: 1rem; border-radius: 8px;
      overflow-x: auto; border: 1px solid var(--border); }
pre code { background: none; padding: 0; }
.tablewrap { overflow-x: auto; margin: 1rem 0; border: 1px solid var(--border);
             border-radius: 8px; }
table { border-collapse: collapse; width: 100%; font-size: .9rem; }
th, td { padding: .55rem .8rem; text-align: left; border-bottom: 1px solid var(--border);
         white-space: nowrap; }
th { background: var(--card); font-weight: 600; position: sticky; top: 0; }
tr:last-child td { border-bottom: none; }
td:first-child, th:first-child { white-space: normal; }
.meta { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: .8rem; margin: 1.2rem 0; }
.meta div { background: var(--card); border: 1px solid var(--border);
            border-radius: 8px; padding: .7rem .9rem; }
.meta dt { color: var(--muted); font-size: .75rem; text-transform: uppercase;
           letter-spacing: .05em; margin: 0 0 .2rem; }
.meta dd { margin: 0; font-size: .95rem; font-weight: 600; word-break: break-word; }
.warn { background: var(--warn-bg); border-left: 3px solid var(--warn-br);
        padding: .85rem 1rem; border-radius: 0 8px 8px 0; margin: 1rem 0; }
.warn p:first-child { margin-top: 0; } .warn p:last-child { margin-bottom: 0; }
figure { margin: 1.4rem 0; overflow-x: auto; }
figure svg { max-width: 100%; height: auto; display: block; }
figcaption { color: var(--muted); font-size: .85rem; margin-top: .4rem; }
.note { color: var(--muted); font-size: .9rem; font-style: italic; }
"""


def _svg_inline(path: str) -> str:
    """Inline an SVG so the page has no external requests at all."""
    try:
        with open(path, encoding="utf-8") as fh:
            svg = fh.read()
    except OSError:
        return ""
    start = svg.find("<svg")
    return svg[start:] if start >= 0 else ""


def _md_tables_to_html(markdown_fragment: str) -> str:
    """Convert the Markdown tables built above into HTML.

    A tiny purpose-built converter rather than a Markdown dependency: the only
    constructs used in these fragments are tables, list items, headings and
    emphasis, and the report should not gain a dependency for that.
    """
    lines = markdown_fragment.split("\n")
    out: List[str] = []
    table: List[List[str]] = []

    def flush_table() -> None:
        if not table:
            return
        header, *body = table
        out.append('<div class="tablewrap"><table><thead><tr>')
        # .extend, not `out +=`: augmented assignment would rebind `out` as a
        # local inside this closure and shadow the enclosing list.
        out.extend(f"<th>{_html.escape(c)}</th>" for c in header)
        out.append("</tr></thead><tbody>")
        for row in body:
            out.append("<tr>" + "".join(
                f"<td>{_inline_md(c)}</td>" for c in row) + "</tr>")
        out.append("</tbody></table></div>")
        table.clear()

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if all(set(c) <= set("-: ") and c for c in cells):
                continue  # separator row
            table.append(cells)
            continue
        flush_table()
        if not stripped:
            continue
        if stripped.startswith("#### "):
            out.append(f"<h4>{_inline_md(stripped[5:])}</h4>")
        elif stripped.startswith("### "):
            out.append(f"<h3>{_inline_md(stripped[4:])}</h3>")
        elif stripped.startswith("- "):
            out.append(f"<li>{_inline_md(stripped[2:])}</li>")
        elif stripped.startswith("_") and stripped.endswith("_"):
            out.append(f'<p class="note">{_inline_md(stripped.strip("_"))}</p>')
        else:
            out.append(f"<p>{_inline_md(stripped)}</p>")
    flush_table()

    joined = "\n".join(out)
    # Wrap runs of <li> in a <ul>.
    joined = joined.replace("<li>", "<ul><li>", 1) if "<li>" in joined else joined
    if "<li>" in joined:
        idx = joined.rfind("</li>")
        joined = joined[:idx + 5] + "</ul>" + joined[idx + 5:]
    return joined


def _inline_md(text: str) -> str:
    escaped = _html.escape(text)
    # Order matters: bold before italic, code before both.
    while "`" in escaped:
        first = escaped.find("`")
        second = escaped.find("`", first + 1)
        if second < 0:
            break
        escaped = (escaped[:first] + "<code>" + escaped[first + 1:second]
                   + "</code>" + escaped[second + 1:])
    while "**" in escaped:
        first = escaped.find("**")
        second = escaped.find("**", first + 2)
        if second < 0:
            break
        escaped = (escaped[:first] + "<strong>" + escaped[first + 2:second]
                   + "</strong>" + escaped[second + 2:])
    return escaped


def _pass_note(summary: Dict[str, Any]) -> str:
    """Explain an absent normalized-vs-tuned comparison.

    The chart hangs off section 8, so when it is skipped the section simply
    loses a figure with nothing said about it.
    """
    passes = sorted({p for p in summary.get("passes", []) if p})
    if len(passes) >= 2:
        return ""
    only = passes[0] if passes else "one"
    return (f"<p><em>The normalized-vs-tuned comparison is not shown: this run "
            f"measured the <code>{_html.escape(only)}</code> pass only. Run with "
            f"<code>--resource-pass both</code> to populate it.</em></p>")


def render_html(manifest: Dict[str, Any], summary: Dict[str, Any],
                records: List[Dict[str, Any]],
                chart_paths: Dict[str, Dict[str, str]], title: str) -> str:
    run_id = manifest.get("run_id", "?")
    host = manifest.get("host", {}) or {}
    cpu = host.get("cpu", {}) or {}
    config = manifest.get("config", {}) or {}
    profile = config.get("profile", {}) or {}
    resolved = config.get("resolved_resources", {}) or {}

    def figures(prefix: str) -> str:
        out = []
        for stem, paths in sorted(chart_paths.items()):
            if stem.startswith(prefix):
                svg = _svg_inline(paths["svg"])
                if svg:
                    out.append(f"<figure>{svg}<figcaption>{_html.escape(stem)}"
                               f"</figcaption></figure>")
        return "\n".join(out)

    cards = [
        ("Run", run_id),
        ("Profile", profile.get("name", "?")),
        ("Resource pass", config.get("resource_pass", "?")),
        ("CPU", cpu.get("model", "?")),
        ("AVX-512", "yes" if cpu.get("has_avx512") else "no"),
        ("Server cpuset", resolved.get("server_cpuset", "?")),
        ("Server memory", _fmt_bytes(resolved.get("server_memory_bytes"))),
        ("Records", f"{len(records):,}"),
    ]
    cards_html = "".join(
        f"<div><dt>{_html.escape(str(k))}</dt><dd>{_html.escape(str(v))}</dd></div>"
        for k, v in cards
    )

    warnings = manifest.get("warnings", []) or []
    warn_html = ""
    if warnings:
        items = "".join(f"<p>⚠ {_html.escape(w)}</p>" for w in warnings)
        warn_html = f'<div class="warn">{items}</div>'

    sections = [
        ("1. Environment",
         _md_tables_to_html(_md_table(["Property", "Value"], _environment_rows(manifest)))
         + "<h3>Engines under test</h3>"
         + _md_tables_to_html(_md_table(
             ["Engine", "Tag", "Commit", "-march", "Build type"], _engine_rows(manifest))),
         ""),
        ("2. Validity",
         _md_tables_to_html(_validity_section(manifest, summary)), ""),
        ("3. Known asymmetries",
         _md_tables_to_html(_known_asymmetries(summary)), ""),
        ("4. Recall vs throughput",
         _md_tables_to_html(_headline_tables(summary)),
         figures("pareto-") + figures("paretozoom-") + figures("qpsatrecall-") + figures("latency-")),
        ("5. Index build cost",
         _md_tables_to_html(_build_table(summary)), figures("build-") + figures("storage-")),
        # These three must use the same not-measured wording as the markdown.
        # The HTML is built from its own section list, so fixing only the
        # markdown left the page a reader actually opens still saying
        # "No data." for a workload the profile never ran.
        ("6. Concurrency scaling",
         _md_tables_to_html(_concurrency_table(summary) if summary.get("concurrency")
                            else _not_measured("concurrency", profile)),
         figures("concurrency-")),
        ("7. Filtered (hybrid) search",
         _md_tables_to_html(_filtered_table(summary) if summary.get("filtered")
                            else _not_measured("filtered", profile)),
         figures("filtered-")),
        ("8. Churn",
         _md_tables_to_html(_churn_table(summary) if summary.get("churn")
                            else _not_measured("churn", profile))
         + _pass_note(summary),
         figures("churn-") + figures("churnimpact-") + figures("passcompare-") + figures("memory-timeline")),
    ]

    body = "".join(
        f"<h2>{_html.escape(name)}</h2>{content}{figs}"
        for name, content, figs in sections
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_html.escape(title)} — {_html.escape(run_id)}</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
<h1>{_html.escape(title)}</h1>
<p class="sub">MariaDB vs AliSQL vs PostgreSQL + pgvector — HNSW vector search ·
{_html.escape(str(manifest.get('started_at', '')))} → {_html.escape(str(manifest.get('finished_at', '')))}</p>
<div class="meta">{cards_html}</div>
{warn_html}
{body}
<h2>9. Reproducing this run</h2>
<pre><code>./run-benchmark.sh build
./run-benchmark.sh fetch --datasets {_html.escape(','.join(summary.get('datasets', []) or ['glove-100-angular']))}
./run-benchmark.sh run --profile {_html.escape(str(profile.get('name', 'quick')))} --resource-pass {_html.escape(str(config.get('resource_pass', 'normalized')))}</code></pre>
<p class="note">Raw merged records are in <code>records.jsonl</code>; full provenance
is in <code>run-manifest.json</code>. See <code>docs/05-methodology.md</code> for
what is measured, and <code>docs/03-running-manually.md</code> to reproduce any
single measurement by hand without this framework.</p>
</div>
</body>
</html>
"""
