"""Chart generation. Every chart is written as both SVG and PNG.

Design constraints, in order:

* **Readable in both light and dark.** The HTML report is viewed in whatever
  theme the reader uses, so charts use a transparent background and a palette
  with sufficient contrast against either.
* **Colour is not the only channel.** Each engine gets a distinct marker and
  line style as well as a colour, so the charts survive greyscale printing and
  the most common colour-vision deficiencies.
* **Axes state their direction.** "Higher is better" is written on the axis
  rather than assumed, because half these charts invert that (latency, build
  time, index size).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

# Engine identity is consistent across every chart in the report.
STYLE: Dict[str, Dict[str, Any]] = {
    "mariadb":  {"color": "#1f77b4", "marker": "o", "linestyle": "-",  "label": "MariaDB (MHNSW)"},
    "alisql":   {"color": "#d62728", "marker": "s", "linestyle": "--", "label": "AliSQL (VIDX)"},
    "pgvector": {"color": "#2ca02c", "marker": "^", "linestyle": "-.", "label": "PostgreSQL (pgvector)"},
}
FALLBACK = {"color": "#7f7f7f", "marker": "D", "linestyle": ":", "label": "unknown"}

GRID = {"alpha": 0.25, "linewidth": 0.6}


def style_for(engine: str) -> Dict[str, Any]:
    return STYLE.get(engine, {**FALLBACK, "label": engine})


def _new_axes(title: str, xlabel: str, ylabel: str,
              figsize: Tuple[float, float] = (9, 5.5)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, **GRID)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return fig, ax


def _save(fig, out_dir: str, stem: str) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for ext in ("svg", "png"):
        path = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(path, format=ext, bbox_inches="tight",
                    dpi=150 if ext == "png" else None, transparent=True)
        paths[ext] = path
    plt.close(fig)
    return paths


def _bytes_formatter(value: float, _pos: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024:
            return f"{value:.0f} {unit}"
        value /= 1024
    return f"{value:.0f} PiB"


# ---------------------------------------------------------------------------
# Recall / QPS
# ---------------------------------------------------------------------------

def pareto(records_by_engine: Dict[str, List[Dict[str, Any]]], dataset: str,
           k: int, out_dir: str, stem: str,
           scatter: bool = True,
           recall_floor: Optional[float] = None) -> Optional[Dict[str, str]]:
    """Recall vs QPS, the headline chart.

    Plots each engine's Pareto frontier as a line and, faintly, every measured
    point behind it. Showing the raw points matters: a frontier drawn through
    three widely-scattered measurements deserves less trust than one drawn
    through thirty tight ones, and hiding the scatter hides that.
    """
    from .loaders import pareto_frontier

    if not any(records_by_engine.values()):
        return None

    title = f"Recall vs throughput — {dataset}"
    if recall_floor is not None:
        # The decision is almost always made in the high-recall region, which a
        # full 0-1 axis compresses into a few pixels. A zoomed companion chart
        # shows that region at a usable scale without hiding the full picture.
        title += f"   (recall ≥ {recall_floor:g})"
    fig, ax = _new_axes(
        title,
        f"Recall@{k}  (higher is better →)",
        "Queries per second  (higher is better ↑)",
    )
    ax.set_yscale("log")

    plotted = False
    for engine, points in sorted(records_by_engine.items()):
        style = style_for(engine)
        usable = [p for p in points
                  if p.get("recall_at_k") is not None and p.get("qps")]
        if recall_floor is not None:
            usable = [p for p in usable if p["recall_at_k"] >= recall_floor]
        if not usable:
            continue
        plotted = True

        if scatter:
            ax.scatter(
                [p["recall_at_k"] for p in usable], [p["qps"] for p in usable],
                color=style["color"], marker=style["marker"], s=18,
                alpha=0.22, linewidths=0, zorder=2,
            )

        frontier = pareto_frontier(usable)
        if frontier:
            ax.plot(
                [p["recall_at_k"] for p in frontier], [p["qps"] for p in frontier],
                color=style["color"], marker=style["marker"],
                linestyle=style["linestyle"], linewidth=2.0, markersize=6,
                label=style["label"], zorder=3,
            )

    if not plotted:
        plt.close(fig)
        return None

    if recall_floor is not None:
        ax.set_xlim(left=recall_floor, right=1.002)
    else:
        ax.set_xlim(left=max(0.0, ax.get_xlim()[0]), right=1.005)
    ax.legend(frameon=False, fontsize=10, loc="lower left")
    return _save(fig, out_dir, stem)


# ---------------------------------------------------------------------------
# Build cost
# ---------------------------------------------------------------------------

def build_cost(records: List[Dict[str, Any]], dataset: str, out_dir: str,
               stem: str) -> Optional[Dict[str, str]]:
    """Build time, peak memory and index size against M, side by side."""
    by_engine: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        if r.get("dataset") == dataset and r.get("build_wall_s") is not None:
            by_engine.setdefault(r["engine"], []).append(r)
    if not by_engine:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    fig.patch.set_alpha(0.0)
    panels = [
        ("build_wall_s", "Index build time (s)  (lower is better ↓)", None),
        ("peak_rss_bytes", "Peak server memory  (lower is better ↓)", _bytes_formatter),
        ("index_bytes", "Index size on disk  (lower is better ↓)", _bytes_formatter),
    ]

    any_data = False
    for ax, (field, label, formatter) in zip(axes, panels):
        ax.patch.set_alpha(0.0)
        ax.set_xlabel("M (graph degree)")
        ax.set_ylabel(label, fontsize=10)
        ax.grid(True, **GRID)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        if formatter:
            ax.yaxis.set_major_formatter(FuncFormatter(formatter))

        for engine, rows in sorted(by_engine.items()):
            style = style_for(engine)
            points = sorted(
                ((r.get("m"), r.get(field)) for r in rows
                 if r.get("m") is not None and r.get(field) is not None),
                key=lambda p: p[0],
            )
            if not points:
                continue
            any_data = True
            ax.plot([p[0] for p in points], [p[1] for p in points],
                    color=style["color"], marker=style["marker"],
                    linestyle=style["linestyle"], linewidth=1.8, markersize=6,
                    label=style["label"])

    if not any_data:
        plt.close(fig)
        return None

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, fontsize=10,
                   loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 1.06))
    fig.suptitle(f"Index build cost — {dataset}", fontsize=13, y=1.14)
    fig.tight_layout()
    return _save(fig, out_dir, stem)


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------

def concurrency(records: List[Dict[str, Any]], dataset: str, out_dir: str,
                stem: str) -> Optional[Dict[str, str]]:
    """QPS, p99 latency and scaling efficiency against client count."""
    by_engine: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        if r.get("dataset") == dataset and r.get("clients") and r.get("qps"):
            by_engine.setdefault(r["engine"], []).append(r)
    if not by_engine:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    fig.patch.set_alpha(0.0)

    for ax in axes:
        ax.patch.set_alpha(0.0)
        ax.set_xlabel("Concurrent clients")
        ax.grid(True, **GRID)
        ax.set_xscale("log", base=2)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("Queries per second  (higher is better ↑)", fontsize=10)
    axes[1].set_ylabel("p99 latency (ms)  (lower is better ↓)", fontsize=10)
    axes[2].set_ylabel("Scaling efficiency  (1.0 = linear)", fontsize=10)

    for engine, rows in sorted(by_engine.items()):
        style = style_for(engine)
        rows = sorted(rows, key=lambda r: r["clients"])
        clients = [r["clients"] for r in rows]

        axes[0].plot(clients, [r["qps"] for r in rows], color=style["color"],
                     marker=style["marker"], linestyle=style["linestyle"],
                     linewidth=1.8, markersize=6, label=style["label"])

        p99 = [(c, r.get("latency_p99_ms")) for c, r in zip(clients, rows)
               if r.get("latency_p99_ms") is not None]
        if p99:
            axes[1].plot([p[0] for p in p99], [p[1] for p in p99],
                         color=style["color"], marker=style["marker"],
                         linestyle=style["linestyle"], linewidth=1.8, markersize=6)

        eff = [(c, (r.get("extra") or {}).get("scaling_efficiency"))
               for c, r in zip(clients, rows)]
        eff = [(c, v) for c, v in eff if v is not None]
        if eff:
            axes[2].plot([p[0] for p in eff], [p[1] for p in eff],
                         color=style["color"], marker=style["marker"],
                         linestyle=style["linestyle"], linewidth=1.8, markersize=6)

    # Linear-scaling reference so the efficiency panel has a meaning without
    # the reader having to remember what 1.0 represents.
    axes[2].axhline(1.0, color="#888888", linewidth=1.0, linestyle=":")
    axes[2].text(axes[2].get_xlim()[0], 1.02, " linear scaling",
                 fontsize=8, color="#888888", va="bottom")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, fontsize=10,
                   loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 1.06))
    fig.suptitle(f"Concurrency scaling — {dataset}", fontsize=13, y=1.14)
    fig.tight_layout()
    return _save(fig, out_dir, stem)


# ---------------------------------------------------------------------------
# Filtered search
# ---------------------------------------------------------------------------

def filtered(records: List[Dict[str, Any]], dataset: str, out_dir: str,
             stem: str) -> Optional[Dict[str, str]]:
    """Filtered recall and QPS against selectivity."""
    by_engine: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        if r.get("dataset") == dataset and r.get("selectivity") is not None:
            by_engine.setdefault(r["engine"], []).append(r)
    if not by_engine:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    fig.patch.set_alpha(0.0)
    for ax in axes:
        ax.patch.set_alpha(0.0)
        ax.set_xlabel("Fraction of rows passing the filter")
        ax.set_xscale("log")
        ax.grid(True, **GRID)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("Filtered recall@k  (higher is better ↑)", fontsize=10)
    axes[1].set_ylabel("Queries per second  (higher is better ↑)", fontsize=10)
    axes[1].set_yscale("log")

    for engine, rows in sorted(by_engine.items()):
        style = style_for(engine)
        rows = sorted(rows, key=lambda r: r["selectivity"])
        sel = [r["selectivity"] for r in rows]

        axes[0].plot(sel, [r.get("recall_at_k") for r in rows],
                     color=style["color"], marker=style["marker"],
                     linestyle=style["linestyle"], linewidth=1.8, markersize=6,
                     label=style["label"])
        axes[1].plot(sel, [r.get("qps") for r in rows],
                     color=style["color"], marker=style["marker"],
                     linestyle=style["linestyle"], linewidth=1.8, markersize=6)

        # An engine returning fewer than k rows is a real behaviour, not noise —
        # mark it so a high-looking recall is not read as unqualified success.
        for r in rows:
            if (r.get("extra") or {}).get("returned_fewer_than_k"):
                axes[0].scatter([r["selectivity"]], [r.get("recall_at_k")],
                                s=140, facecolors="none",
                                edgecolors=style["color"], linewidths=1.6, zorder=5)

    axes[0].set_ylim(-0.02, 1.02)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, fontsize=10,
                   loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 1.06))
    fig.suptitle(
        f"Filtered (hybrid) search — {dataset}   "
        f"○ = returned fewer than k results", fontsize=12, y=1.14
    )
    fig.tight_layout()
    return _save(fig, out_dir, stem)


# ---------------------------------------------------------------------------
# Churn
# ---------------------------------------------------------------------------

def churn(records: List[Dict[str, Any]], dataset: str, out_dir: str,
          stem: str) -> Optional[Dict[str, str]]:
    """Recall and QPS after successive delete/re-insert cycles."""
    by_engine: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        if r.get("dataset") == dataset and r.get("churn_fraction") is not None:
            by_engine.setdefault(r["engine"], []).append(r)
    if not by_engine:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    fig.patch.set_alpha(0.0)
    for ax in axes:
        ax.patch.set_alpha(0.0)
        ax.set_xlabel("Cumulative fraction of rows deleted and re-inserted")
        ax.grid(True, **GRID)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("Recall@k  (higher is better ↑)", fontsize=10)
    axes[1].set_ylabel("Queries per second  (higher is better ↑)", fontsize=10)

    for engine, rows in sorted(by_engine.items()):
        style = style_for(engine)
        rows = sorted(rows, key=lambda r: r["churn_fraction"])
        fractions = [r["churn_fraction"] for r in rows]
        axes[0].plot(fractions, [r.get("recall_at_k") for r in rows],
                     color=style["color"], marker=style["marker"],
                     linestyle=style["linestyle"], linewidth=1.8, markersize=6,
                     label=style["label"])
        axes[1].plot(fractions, [r.get("qps") for r in rows],
                     color=style["color"], marker=style["marker"],
                     linestyle=style["linestyle"], linewidth=1.8, markersize=6)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, fontsize=10,
                   loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 1.06))
    fig.suptitle(f"Index degradation under churn — {dataset}", fontsize=13, y=1.14)
    fig.tight_layout()
    return _save(fig, out_dir, stem)


def memory_timeline(series: Dict[str, List[Dict[str, Any]]], out_dir: str,
                    stem: str) -> Optional[Dict[str, str]]:
    """Server memory over the whole run.

    More informative than a single peak: a cache filling to its configured
    ceiling looks different from a build spike, which looks different again
    from a steady climb that never plateaus.
    """
    if not series:
        return None
    fig, ax = _new_axes("Server memory over the run", "Elapsed (s)",
                        "Resident memory  (lower is better ↓)", figsize=(11, 4.6))
    ax.yaxis.set_major_formatter(FuncFormatter(_bytes_formatter))

    plotted = False
    for name, rows in sorted(series.items()):
        rows = [r for r in rows if r.get("rss_bytes")]
        if not rows:
            continue
        plotted = True
        t0 = rows[0]["t"]
        engine = name.split("-")[0]
        style = style_for(engine)
        ax.plot([r["t"] - t0 for r in rows], [r["rss_bytes"] for r in rows],
                color=style["color"], linestyle=style["linestyle"],
                linewidth=1.4, label=name)

    if not plotted:
        plt.close(fig)
        return None
    ax.legend(frameon=False, fontsize=8, ncol=2)
    return _save(fig, out_dir, stem)
