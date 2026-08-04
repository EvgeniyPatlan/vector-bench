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
    "mariadb": "MariaDB (MHNSW)",
    "alisql": "AliSQL (VIDX)",
    "pgvector": "PostgreSQL (pgvector)",
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


def _validity_section(manifest: Dict[str, Any], summary: Dict[str, Any]) -> str:
    parts: List[str] = []
    warnings = manifest.get("warnings", []) or []

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

    if summary.get("short_result_cases"):
        parts.append(
            "\n### Filtered queries that returned fewer than k results\n\n"
            "With post-filtering, an engine can exhaust its candidate list before "
            "finding k qualifying rows. This is real engine behaviour, not a harness "
            "fault, but it means the recall figure for these points is computed over "
            "short result sets and should be read alongside the count.\n"
        )
        rows = [[_label(c.get("engine", "?")), c.get("dataset", "?"),
                 f"{(c.get('selectivity') or 0):.0%}", str(c.get("queries", "—"))]
                for c in summary["short_result_cases"][:30]]
        parts.append(_md_table(
            ["Engine", "Dataset", "Selectivity", "Queries short of k"], rows))

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


def _known_asymmetries() -> str:
    return (
        "These are structural differences between the engines that no configuration "
        "removes. They are why some comparisons below are presented in pairs rather "
        "than as a single ranking.\n\n"
        + _md_table(
            ["Asymmetry", "Consequence for these results"],
            [
                ["`ef_construction` is exposed only by pgvector",
                 "Build quality can be tuned for pgvector but not for MHNSW or VIDX. "
                 "It is pinned to the default in the normalized pass so pgvector is "
                 "not given an axis the others lack."],
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
            ],
        )
    )


def _headline_tables(summary: Dict[str, Any]) -> str:
    parts: List[str] = []
    for dataset, per_engine in sorted(summary.get("per_dataset", {}).items()):
        parts.append(f"\n#### {dataset}\n\n")
        rows = []
        for engine, entry in sorted(per_engine.items()):
            rows.append([
                _label(engine),
                _fmt(entry.get("qps_at_recall_90"), 0),
                _fmt(entry.get("qps_at_recall_95"), 0),
                _fmt(entry.get("qps_at_recall_99"), 0),
                f"{entry.get('max_recall', 0):.4f}",
                str(entry.get("points", 0)),
            ])
        parts.append(_md_table(
            ["Engine", "QPS @ recall≥0.90", "QPS @ recall≥0.95",
             "QPS @ recall≥0.99", "Best recall", "Points measured"],
            rows,
        ))
        parts.append(
            "\n_QPS at a recall floor is the comparison an operator makes: how fast "
            "is the engine at an accuracy I can accept. A dash means the engine did "
            "not reach that recall anywhere in the swept grid._\n"
        )
    return "".join(parts) or "_No recall/QPS data in this run._\n"


def _build_table(summary: Dict[str, Any]) -> str:
    rows = []
    for r in sorted(summary.get("build", []),
                    key=lambda r: (r.get("dataset", ""), r.get("engine", ""), r.get("m") or 0)):
        rows.append([
            _label(r.get("engine", "?")), r.get("dataset", "?"), str(r.get("m", "—")),
            r.get("build_mode") or "—",
            _fmt(r.get("build_wall_s"), 1, " s"),
            _fmt(r.get("ingest_rows_per_s"), 0, " rows/s"),
            _fmt_bytes(r.get("peak_rss_bytes")),
            _fmt_bytes(r.get("index_bytes")),
            "yes" if (r.get("extra") or {}).get("separable_build") else "no",
        ])
    return _md_table(
        ["Engine", "Dataset", "M", "Build mode", "Build time", "Ingest rate",
         "Peak server RSS", "Index size", "Separable build"],
        rows,
    )


def _concurrency_table(summary: Dict[str, Any]) -> str:
    rows = []
    for r in sorted(summary.get("concurrency", []),
                    key=lambda r: (r.get("dataset", ""), r.get("engine", ""),
                                   r.get("clients") or 0)):
        eff = (r.get("extra") or {}).get("scaling_efficiency")
        rows.append([
            _label(r.get("engine", "?")), r.get("dataset", "?"),
            str(r.get("clients", "—")),
            _fmt(r.get("qps"), 0),
            _fmt(r.get("latency_p50_ms"), 3),
            _fmt(r.get("latency_p95_ms"), 3),
            _fmt(r.get("latency_p99_ms"), 3),
            _fmt(eff, 2) if eff is not None else "—",
        ])
    return _md_table(
        ["Engine", "Dataset", "Clients", "QPS", "p50 (ms)", "p95 (ms)",
         "p99 (ms)", "Scaling efficiency"],
        rows,
    )


def _filtered_table(summary: Dict[str, Any]) -> str:
    rows = []
    for r in sorted(summary.get("filtered", []),
                    key=lambda r: (r.get("dataset", ""), r.get("engine", ""),
                                   r.get("selectivity") or 0)):
        extra = r.get("extra") or {}
        rows.append([
            _label(r.get("engine", "?")), r.get("dataset", "?"),
            f"{(r.get('selectivity') or 0):.0%}",
            _fmt(r.get("recall_at_k"), 4),
            _fmt(r.get("qps"), 0),
            _fmt(r.get("latency_p99_ms"), 3),
            "yes" if r.get("vector_index_used") else "no",
            str(extra.get("short_result_queries", 0)),
            extra.get("iterative_scan") or "—",
        ])
    return _md_table(
        ["Engine", "Dataset", "Selectivity", "Recall@k", "QPS", "p99 (ms)",
         "Index used", "Short results", "Iterative scan"],
        rows,
    )


def _churn_table(summary: Dict[str, Any]) -> str:
    rows = []
    for r in sorted(summary.get("churn", []),
                    key=lambda r: (r.get("dataset", ""), r.get("engine", ""),
                                   r.get("churn_fraction") or 0)):
        extra = r.get("extra") or {}
        rows.append([
            _label(r.get("engine", "?")), r.get("dataset", "?"),
            f"{(r.get('churn_fraction') or 0):.0%}",
            _fmt(r.get("recall_at_k"), 4),
            _fmt(extra.get("recall_drop_vs_baseline"), 4),
            _fmt(r.get("qps"), 0),
            _fmt_bytes(r.get("index_bytes")),
        ])
    return _md_table(
        ["Engine", "Dataset", "Churn", "Recall@k", "Recall drop vs baseline",
         "QPS", "Index size"],
        rows,
    )


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
        _known_asymmetries(),
        "\n---\n\n## 4. Recall vs throughput\n\n",
        "The headline comparison. Each engine's Pareto frontier is the best "
        "throughput it achieved at each recall level across the whole parameter "
        "sweep; the faint scatter behind it is every measured point.\n",
        _headline_tables(summary),
    ]

    for stem, paths in sorted(chart_paths.items()):
        if stem.startswith("pareto-") or stem.startswith("paretozoom-"):
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
        if stem.startswith("build-"):
            parts.append(f"\n![{stem}](charts/{os.path.basename(paths['svg'])})\n")

    parts += [
        "\n---\n\n## 6. Concurrency scaling\n\n",
        "Where a database should differ from an ANN library. MariaDB caches the "
        "MHNSW graph per `TABLE_SHARE`; AliSQL keeps a shared cache plus a "
        "per-transaction cache; pgvector serves graph pages from `shared_buffers`. "
        "Scaling efficiency of 1.0 is linear scaling.\n\n",
        _concurrency_table(summary),
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
        _filtered_table(summary),
    ]
    for stem, paths in sorted(chart_paths.items()):
        if stem.startswith("filtered-"):
            parts.append(f"\n![{stem}](charts/{os.path.basename(paths['svg'])})\n")

    parts += [
        "\n---\n\n## 8. Churn\n\n",
        "Recall and throughput after deleting and re-inserting a fraction of rows. "
        "HNSW graphs degrade under deletion; how much differs by implementation.\n\n",
        _churn_table(summary),
    ]
    for stem, paths in sorted(chart_paths.items()):
        if stem.startswith("churn-") or stem == "memory-timeline":
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
         _md_tables_to_html(_known_asymmetries()), ""),
        ("4. Recall vs throughput",
         _md_tables_to_html(_headline_tables(summary)),
         figures("pareto-") + figures("paretozoom-")),
        ("5. Index build cost",
         _md_tables_to_html(_build_table(summary)), figures("build-")),
        ("6. Concurrency scaling",
         _md_tables_to_html(_concurrency_table(summary)), figures("concurrency-")),
        ("7. Filtered (hybrid) search",
         _md_tables_to_html(_filtered_table(summary)), figures("filtered-")),
        ("8. Churn",
         _md_tables_to_html(_churn_table(summary)),
         figures("churn-") + figures("memory-timeline")),
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
