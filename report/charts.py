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
    "mariadb":  {"color": "#1f77b4", "marker": "o", "linestyle": "-",  "label": "MariaDB 11.8 (MHNSW)"},
    # Same hue family as 11.8 so the two versions read as related, dashed so
    # they stay distinguishable in greyscale and for colour-blind readers.
    "mariadb123": {"color": "#5fa8d3", "marker": "s", "linestyle": "--", "label": "MariaDB 12.3 (MHNSW)"},
    "alisql":   {"color": "#d62728", "marker": "s", "linestyle": "--", "label": "AliSQL (VIDX)"},
    "pgvector": {"color": "#2ca02c", "marker": "^", "linestyle": "-.", "label": "PostgreSQL (pgvector)"},
    # Distinct hue: it is the only engine whose index lives in a separate
    # process, so it should not read as a variant of anything above it.
    "mongodb":  {"color": "#9467bd", "marker": "v", "linestyle": "-",  "label": "Percona Search (mongot)"},
    # Its own hue as well: it is the only in-memory engine, so it should not
    # read as a variant of any disk-backed one.
    "valkey":   {"color": "#e377c2", "marker": "D", "linestyle": "-",  "label": "Valkey (valkey-search)"},
}
FALLBACK = {"color": "#7f7f7f", "marker": "D", "linestyle": ":", "label": "unknown"}

GRID = {"alpha": 0.25, "linewidth": 0.6}


def style_for(engine: str) -> Dict[str, Any]:
    return STYLE.get(engine, {**FALLBACK, "label": engine})


# Axes a single engine can be swept over within one run. Two measurements of
# one engine that differ on either are different curves, not repeats of one:
# grouping a chart by engine name alone put several points at every x and drew
# a line through all of them that no configuration ever produced.
# What separates one plotted line from another, beyond the engine itself.
#
# `m` belongs here and was missing. Every engine swept a single graph degree
# until MHNSW and VIDX were given M=6 as well, so that they would have real
# measurements below recall 0.90 -- and then two configurations of the same
# engine were drawn as one line. On the latency chart, which plots against
# ef_search, that put two y-values at every x and joined them: AliSQL appeared
# to swing between 383 ms and 95 ms and back to 370 ms at adjacent search
# widths. The Pareto chart was unaffected, because it takes a frontier across
# every configuration an engine was swept over, which is what it should do.
SERIES_AXES = ("storage_engine", "ef_construction", "m")

# How each axis names itself in a legend. Empty means the value speaks for
# itself, as a storage engine does. Everything else needs saying: an M=6 curve
# labelled "ef_c=6" is worse than one labelled nothing at all.
AXIS_LABELS = {"storage_engine": "", "ef_construction": "ef_c=", "m": "M="}

# Length of a series key: the engine plus one slot per axis. churn_impact
# builds a wider key and then slices it back, and it did that with the
# literal 3 -- so adding `m` to SERIES_AXES shifted every index under it and
# the lookup stopped matching anything at all.
SERIES_KEY_WIDTH = 1 + len(SERIES_AXES)

# Colour stays with the engine so versions remain comparable at a glance; the
# storage engine is carried by the linestyle instead.
STORAGE_LINESTYLES = ("-", "--", ":", "-.")


def series_key(record: Dict[str, Any]) -> Tuple[Any, ...]:
    """Identity of one plotted line."""
    return (record.get("engine"),) + tuple(record.get(a) for a in SERIES_AXES)


def series_labels(records: List[Dict[str, Any]]) -> Dict[Tuple[Any, ...], str]:
    """Legend text per series, naming only the axes that actually vary.

    An engine swept over one value keeps its bare label: AliSQL is InnoDB-only
    and only pgvector exposes ef_construction, so naming either there would
    imply a comparison the run did not make.
    """
    varying: Dict[str, List[set]] = {}
    for r in records:
        seen = varying.setdefault(r.get("engine"), [set() for _ in SERIES_AXES])
        for i, axis in enumerate(SERIES_AXES):
            seen[i].add(r.get(axis))

    labels = {}
    for r in records:
        key = series_key(r)
        label = style_for(key[0])["label"]
        for i, axis in enumerate(SERIES_AXES):
            value = key[i + 1]
            if value is not None and len(varying[key[0]][i]) > 1:
                label += f" / {AXIS_LABELS[axis]}{value}" if AXIS_LABELS[axis] \
                    else f" / {value}"
        labels[key] = label
    return labels


def series_style(record: Dict[str, Any],
                 storages: Optional[List[Optional[str]]] = None,
                 degrees: Optional[List[Optional[int]]] = None) -> Dict[str, Any]:
    """Engine colour and marker, with the linestyle carrying storage engine."""
    style = dict(style_for(record.get("engine")))
    if storages and len(storages) > 1:
        idx = storages.index(record.get("storage_engine"))
        style["linestyle"] = STORAGE_LINESTYLES[idx % len(STORAGE_LINESTYLES)]
    # Colour is the engine and linestyle is the storage engine, so a second
    # graph degree had nothing left to carry it and two AliSQL curves came out
    # identical. Width and transparency are free: the denser graph stays solid
    # and prominent, the sparser one recedes, and which is which is legible
    # without reading the legend.
    if degrees and len(degrees) > 1 and record.get("m") is not None:
        rank = degrees.index(record.get("m"))
        style["alpha"] = 1.0 if rank == len(degrees) - 1 else 0.55
        style["linewidth"] = 2.0 if rank == len(degrees) - 1 else 1.3
    return style


def _grouped(records: List[Dict[str, Any]]
             ) -> Tuple[Dict[Tuple[Any, ...], List[Dict[str, Any]]],
                        Dict[Tuple[Any, ...], str],
                        Dict[str, List[Optional[str]]],
                        Dict[str, List[Optional[int]]]]:
    """Split records into plotted series, with their labels, storage engines
    and graph degrees -- the last two because a style has to be chosen per
    series and both need to be distinguishable within one engine's colour."""
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for r in records:
        groups.setdefault(series_key(r), []).append(r)
    per_engine: Dict[str, List[Optional[str]]] = {}
    degrees: Dict[str, List[Optional[int]]] = {}
    for key in groups:
        storages = per_engine.setdefault(key[0], [])
        if key[1] not in storages:
            storages.append(key[1])
        ms = degrees.setdefault(key[0], [])
        if key[3] not in ms:
            ms.append(key[3])
    for v in list(per_engine.values()) + list(degrees.values()):
        v.sort(key=lambda x: (x is None, x))
    return groups, series_labels(records), per_engine, degrees


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
    # Recall lives at the top of its range and the interesting differences are
    # in the last few percent, so a linear axis is the wrong instrument. On
    # 0.69-1.0 linear, MariaDB 12.3 (0.9784-0.9996) occupies the rightmost 7%
    # of the plot and AliSQL the rightmost 10%: both read as truncated rather
    # than as accurate. A logit axis gives 0.99-0.999 the same width as
    # 0.9-0.99, which is how ann-benchmarks plots recall and why.
    ax.set_xscale("logit")

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

    # A logit axis cannot reach 1.0, and the right edge has to leave room for
    # the highest point measured rather than sitting on top of it.
    highest = max(
        (p["recall_at_k"] for points in records_by_engine.values()
         for p in points if p.get("recall_at_k") is not None
         and p.get("qps")
         and (recall_floor is None or p["recall_at_k"] >= recall_floor)),
        default=0.99)
    right = min(0.9999, 1.0 - (1.0 - highest) * 0.5)
    if recall_floor is not None:
        ax.set_xlim(left=recall_floor, right=right)
    else:
        ax.set_xlim(left=max(0.5, ax.get_xlim()[0]), right=right)
    # Ticks the reader recognises, rather than logit's default 1-10^-n labels.
    ticks = [t for t in (0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999)
             if ax.get_xlim()[0] <= t <= right]
    if ticks:
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}" for t in ticks])
        ax.minorticks_off()
    ax.legend(frameon=False, fontsize=10, loc="lower left")
    return _save(fig, out_dir, stem)


# ---------------------------------------------------------------------------
# Build cost
# ---------------------------------------------------------------------------

def build_cost(records: List[Dict[str, Any]], dataset: str, out_dir: str,
               stem: str) -> Optional[Dict[str, str]]:
    """Build time, peak memory and index size against M, side by side."""
    plotted_records = [r for r in records if r.get("dataset") == dataset
                       and r.get("build_wall_s") is not None]
    if not plotted_records:
        return None
    by_series, labels, storages, degrees = _grouped(plotted_records)

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

        for key, rows in sorted(by_series.items(), key=lambda kv: str(kv[0])):
            style = series_style(rows[0], storages.get(key[0]), degrees.get(key[0]))
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
                    label=labels[key])

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

def _is_concurrency_point(record: Dict[str, Any], dataset: str) -> bool:
    """Only the concurrency sweep belongs on the concurrency chart.

    `clients` and `qps` are recorded by the recall sweep, filtered search and
    churn as well, all at one client. Selecting on those two fields alone put
    every one of them on the x=1 gridline and let filtered search's 13-second
    p99 set the scale of the latency panel.
    """
    return (record.get("phase") == "concurrency"
            and record.get("dataset") == dataset
            and record.get("clients") and record.get("qps"))


def concurrency(records: List[Dict[str, Any]], dataset: str, out_dir: str,
                stem: str) -> Optional[Dict[str, str]]:
    """QPS, p99 latency and scaling efficiency against client count."""
    plotted_records = [r for r in records if _is_concurrency_point(r, dataset)]
    if not plotted_records:
        return None
    by_series, labels, storages, degrees = _grouped(plotted_records)

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

    for key, rows in sorted(by_series.items(), key=lambda kv: str(kv[0])):
        style = series_style(rows[0], storages.get(key[0]), degrees.get(key[0]))
        rows = sorted(rows, key=lambda r: r["clients"])
        clients = [r["clients"] for r in rows]

        axes[0].plot(clients, [r["qps"] for r in rows], color=style["color"],
                     marker=style["marker"], linestyle=style["linestyle"],
                     linewidth=1.8, markersize=6, label=labels[key])

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
    plotted_records = [r for r in records if r.get("dataset") == dataset
                       and r.get("selectivity") is not None]
    if not plotted_records:
        return None
    by_series, labels, storages, degrees = _grouped(plotted_records)

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

    for key, rows in sorted(by_series.items(), key=lambda kv: str(kv[0])):
        style = series_style(rows[0], storages.get(key[0]), degrees.get(key[0]))
        rows = sorted(rows, key=lambda r: r["selectivity"])
        sel = [r["selectivity"] for r in rows]

        axes[0].plot(sel, [r.get("recall_at_k") for r in rows],
                     color=style["color"], marker=style["marker"],
                     linestyle=style["linestyle"], linewidth=1.8, markersize=6,
                     label=labels[key])
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
    plotted_records = [r for r in records if r.get("dataset") == dataset
                       and r.get("churn_fraction") is not None]
    if not plotted_records:
        return None
    by_series, labels, storages, degrees = _grouped(plotted_records)

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

    for key, rows in sorted(by_series.items(), key=lambda kv: str(kv[0])):
        style = series_style(rows[0], storages.get(key[0]), degrees.get(key[0]))
        rows = sorted(rows, key=lambda r: r["churn_fraction"])
        fractions = [r["churn_fraction"] for r in rows]
        axes[0].plot(fractions, [r.get("recall_at_k") for r in rows],
                     color=style["color"], marker=style["marker"],
                     linestyle=style["linestyle"], linewidth=1.8, markersize=6,
                     label=labels[key])
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

# ---------------------------------------------------------------------------
# Headline: throughput at a recall floor
# ---------------------------------------------------------------------------

def qps_at_recall(summary: Dict[str, Any], dataset: str, out_dir: str,
                  stem: str, floors: Sequence[float] = (0.90, 0.95, 0.99)
                  ) -> Optional[Dict[str, str]]:
    """Grouped bars: best QPS each engine reaches at each recall floor.

    This is the comparison an operator actually makes — "how fast is it at an
    accuracy I can accept" — and it is hard to read off a Pareto curve, because
    the eye compares the curves rather than their heights at one x.
    """
    per_engine = (summary.get("per_dataset", {}) or {}).get(dataset, {})
    if not per_engine:
        return None

    engines = sorted(per_engine)
    fig, ax = _new_axes(
        f"Throughput at a recall floor — {dataset}",
        "", "Queries per second  (higher is better ↑)", figsize=(9.5, 5.2),
    )
    width = 0.8 / max(len(engines), 1)
    xs = range(len(floors))
    plotted = False

    for i, engine in enumerate(engines):
        style = style_for(engine)
        vals = [per_engine[engine].get(f"qps_at_recall_{int(f * 100)}") for f in floors]
        offs = [x + i * width - 0.4 + width / 2 for x in xs]
        heights = [v if v else 0 for v in vals]
        if any(heights):
            plotted = True
        bars = ax.bar(offs, heights, width * 0.92, color=style["color"],
                      label=style["label"], edgecolor="none")
        for b, v in zip(bars, vals):
            # An engine that never reaches a floor gets an explicit marker, not
            # a zero bar that reads as "measured and very slow".
            ax.text(b.get_x() + b.get_width() / 2,
                    b.get_height() if v else 0,
                    f"{v:,.0f}" if v else "not reached",
                    ha="center", va="bottom", fontsize=7.5,
                    color=style["color"] if v else "#999999",
                    rotation=0 if v else 90)

    if not plotted:
        plt.close(fig)
        return None

    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"recall ≥ {f:.2f}" for f in floors])
    # Headroom so the value labels above the tallest bars do not collide with
    # the legend.
    ax.set_ylim(top=ax.get_ylim()[1] * 1.18)
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    return _save(fig, out_dir, stem)


# ---------------------------------------------------------------------------
# Normalized vs tuned
# ---------------------------------------------------------------------------

def pass_comparison(records: List[Dict[str, Any]], dataset: str, out_dir: str,
                    stem: str) -> Optional[Dict[str, str]]:
    """What tuning actually bought, per engine and per dimension.

    Both passes are measured but nothing previously compared them directly, so
    the central question the two-pass design exists to answer — does tuning
    change the ranking? — had to be reconstructed by hand from two tables.
    """
    dims = [
        ("recall_qps", "qps", "Query throughput", lambda r: r.get("clients") in (1, None)),
        ("index_build", "build_wall_s", "Index build time", lambda r: True),
        ("concurrency", "qps", "Throughput @ max clients", lambda r: True),
        ("filtered", "qps", "Filtered throughput", lambda r: True),
    ]

    engines, panels = set(), []
    for phase, field, label, keep in dims:
        by = {}
        for r in records:
            if (r.get("phase") != phase or r.get("dataset") != dataset
                    or not keep(r) or r.get(field) is None):
                continue
            key = (r["engine"], r.get("resource_pass"))
            # Best observed value per (engine, pass): max for throughput,
            # min for build time, since lower is better there.
            prev = by.get(key)
            val = r[field]
            by[key] = val if prev is None else (
                min(prev, val) if field.endswith("_s") else max(prev, val))
            engines.add(r["engine"])
        panels.append((label, field, by))

    engines = sorted(engines)
    if not engines or not any(p[2] for p in panels):
        return None

    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 4.8))
    fig.patch.set_alpha(0.0)
    if len(panels) == 1:
        axes = [axes]

    for ax, (label, field, by) in zip(axes, panels):
        ax.patch.set_alpha(0.0)
        ax.grid(True, axis="y", **GRID)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        lower_better = field.endswith("_s")
        ax.set_ylabel(f"{label}  ({'lower' if lower_better else 'higher'} is better"
                      f" {'↓' if lower_better else '↑'})", fontsize=9)

        for i, engine in enumerate(engines):
            style = style_for(engine)
            n = by.get((engine, "normalized"))
            u = by.get((engine, "tuned"))
            ax.bar(i - 0.19, n or 0, 0.36, color=style["color"], alpha=0.45,
                   edgecolor="none")
            ax.bar(i + 0.19, u or 0, 0.36, color=style["color"], edgecolor="none")
            if n and u:
                delta = (u - n) / n
                better = (delta < 0) if lower_better else (delta > 0)
                ax.text(i, max(n, u), f"{delta:+.0%}", ha="center", va="bottom",
                        fontsize=8,
                        color="#2ca02c" if better else "#d62728")
        ax.set_xticks(range(len(engines)))
        ax.set_xticklabels([style_for(e)["label"].split(" (")[0] for e in engines],
                           fontsize=8, rotation=15, ha="right")

    fig.suptitle(f"Normalized (pale) vs tuned (solid) — {dataset}\n"
                 f"percentage is the change tuning produced", fontsize=12, y=1.06)
    fig.tight_layout()
    return _save(fig, out_dir, stem)


# ---------------------------------------------------------------------------
# Latency distribution
# ---------------------------------------------------------------------------

def latency_percentiles(records: List[Dict[str, Any]], dataset: str,
                        out_dir: str, stem: str) -> Optional[Dict[str, str]]:
    """p50 / p95 / p99 against search width.

    Mean latency hides the tail, and the tail is what a service-level objective
    is written against. An engine with a good p50 and a bad p99 is a different
    proposition from one with both merely acceptable.
    """
    plotted_records = [r for r in records
                       if r.get("phase") == "recall_qps" and r.get("dataset") == dataset
                       and r.get("ef_search") and r.get("latency_p99_ms")]
    if not plotted_records:
        return None
    by_series, labels, storages, degrees = _grouped(plotted_records)

    fig, ax = _new_axes(
        f"Query latency distribution — {dataset}",
        "ef_search  (search width)",
        "Latency (ms)  (lower is better ↓)", figsize=(9.5, 5.4),
    )
    ax.set_xscale("log", base=2)

    for key, rows in sorted(by_series.items(), key=lambda kv: str(kv[0])):
        style = series_style(rows[0], storages.get(key[0]), degrees.get(key[0]))
        rows = sorted(rows, key=lambda r: r["ef_search"])
        ef = [r["ef_search"] for r in rows]
        p50 = [r.get("latency_p50_ms") for r in rows]
        p99 = [r.get("latency_p99_ms") for r in rows]
        # Band between p50 and p99 makes the spread visible at a glance; the
        # solid line is p50 so the medians stay comparable across engines.
        ax.fill_between(ef, p50, p99, color=style["color"], alpha=0.13, linewidth=0)
        ax.plot(ef, p50, color=style["color"], marker=style["marker"],
                linestyle=style["linestyle"], linewidth=1.9, markersize=5,
                label=f"{labels[key]} p50")
        ax.plot(ef, p99, color=style["color"], linestyle=":", linewidth=1.2,
                alpha=0.85)

    ax.legend(frameon=False, fontsize=9)
    ax.text(0.99, 0.02, "shaded band = p50 to p99;  dotted = p99",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="#888888")
    return _save(fig, out_dir, stem)


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def storage_breakdown(records: List[Dict[str, Any]], dataset: str, out_dir: str,
                      stem: str) -> Optional[Dict[str, str]]:
    """Index vs table bytes, stacked.

    Index size alone is misleading between these engines: pgvector's HNSW keeps
    the vectors inside the index, while MHNSW and VIDX keep them in the table
    and store only the graph. Only the total is comparable.
    """
    rows = [r for r in records
            if r.get("phase") == "index_build" and r.get("dataset") == dataset
            and (r.get("index_bytes") or r.get("table_bytes"))
            # An in-memory engine writes no files. Stacking its resident bytes
            # beside the others' on-disk bytes in a chart titled "on-disk
            # footprint" invites a comparison between two different quantities;
            # its memory cost is in the build table, labelled as resident.
            and not (r.get("extra") or {}).get("in_memory_only")]
    if not rows:
        return None

    labels, index_b, table_b, colors = [], [], [], []
    for r in sorted(rows, key=lambda r: (r["engine"], str(r.get("resource_pass")),
                                         str(r.get("build_mode")))):
        tag = style_for(r["engine"])["label"].split(" (")[0]
        # Storage engine belongs here for the same reason it belongs in the
        # build table: MariaDB gets an InnoDB bar and a MyISAM bar, and without
        # it they carry identical captions.
        storage = r.get("storage_engine")
        extra = [x for x in (r.get("resource_pass"),
                             storage if storage not in (None, "heap") else None,
                             r.get("build_mode")) if x]
        labels.append(f"{tag}\n{' / '.join(extra)}")
        index_b.append((r.get("index_bytes") or 0))
        table_b.append((r.get("table_bytes") or 0))
        colors.append(style_for(r["engine"])["color"])

    fig, ax = _new_axes(f"On-disk footprint — {dataset}", "",
                        "Bytes  (lower is better ↓)", figsize=(max(8, 1.5 * len(labels)), 5.2))
    ax.yaxis.set_major_formatter(FuncFormatter(_bytes_formatter))
    xs = range(len(labels))
    ax.bar(xs, index_b, 0.62, color=colors, edgecolor="none", label="index")
    ax.bar(xs, table_b, 0.62, bottom=index_b, color=colors, alpha=0.4,
           edgecolor="none", label="table")
    for x, (i, tb) in enumerate(zip(index_b, table_b)):
        ax.text(x, i + tb, _bytes_formatter(i + tb, 0), ha="center", va="bottom",
                fontsize=8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.12)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    # Below the axes, not inside them: at seven bars the in-plot version
    # overlapped the data it was explaining.
    fig.text(0.5, -0.06,
             "solid = index, pale = table.  pgvector stores vectors inside the index; "
             "MHNSW and VIDX keep them in the table — only the totals are comparable.",
             ha="center", va="top", fontsize=8, color="#888888")
    return _save(fig, out_dir, stem)


# ---------------------------------------------------------------------------
# Churn impact
# ---------------------------------------------------------------------------

def churn_retention(records: List[Dict[str, Any]], dataset: str
                    ) -> Dict[Tuple[Any, ...], List[Tuple[float, float]]]:
    """Post-churn QPS as a fraction of each series' own pre-churn baseline.

    The key names the storage engine. Without it MariaDB's InnoDB and MyISAM
    runs shared one baseline slot, so whichever was recorded second overwrote
    the first and both post-churn points were divided by it -- turning a real
    3% MyISAM loss and an 84% InnoDB loss into two numbers that were each
    wrong by the ratio between the two baselines.
    """
    baseline: Dict[Tuple[Any, ...], float] = {}
    after: Dict[Tuple[Any, ...], List[Tuple[float, float]]] = {}
    for r in records:
        if r.get("phase") != "churn" or r.get("dataset") != dataset or not r.get("qps"):
            continue
        key = series_key(r) + (r.get("resource_pass"), r.get("build_mode"))
        if (r.get("churn_fraction") or 0) == 0:
            baseline[key] = r["qps"]
        else:
            after.setdefault(key, []).append((r["churn_fraction"], r["qps"]))

    return {key: [(f, q / baseline[key]) for f, q in sorted(points)]
            for key, points in after.items() if baseline.get(key)}


def churn_impact(records: List[Dict[str, Any]], dataset: str, out_dir: str,
                 stem: str) -> Optional[Dict[str, str]]:
    """Throughput retained after churn, as a fraction of the pre-churn baseline.

    Recall usually survives churn; throughput does not, and that is invisible in
    a recall-only view. Plotted as a retention ratio so engines with different
    absolute speeds can be compared on how well they hold up.
    """
    retained = churn_retention(records, dataset)
    if not retained:
        return None

    fig, ax = _new_axes(f"Throughput retained after churn — {dataset}",
                        "Cumulative fraction of rows deleted and re-inserted",
                        "QPS as a fraction of pre-churn  (higher is better ↑)",
                        figsize=(9, 5.2))
    churn_records = [r for r in records if r.get("phase") == "churn"
                     and r.get("dataset") == dataset and r.get("qps")]
    labels = series_labels(churn_records)
    storages: Dict[str, List[Optional[str]]] = {}
    for key in {series_key(r) for r in churn_records}:
        storages.setdefault(key[0], []).append(key[1])
    for v in storages.values():
        v.sort(key=lambda x: (x is None, x))

    # Resource pass and build mode are named only when the figure actually
    # holds more than one of them. On a single-pass run they turned every
    # legend entry into "... (post) / tuned", which crowds out the storage
    # engine -- the one distinction this chart exists to show.
    passes = {key[SERIES_KEY_WIDTH] for key in retained}
    modes = {key[SERIES_KEY_WIDTH + 1] for key in retained}

    for key, points in sorted(retained.items(), key=lambda kv: str(kv[0])):
        sample = next(r for r in churn_records
                      if series_key(r) == key[:SERIES_KEY_WIDTH])
        style = series_style(sample, storages.get(key[0]))
        points = sorted(points)
        label = labels.get(key[:SERIES_KEY_WIDTH], key[0])
        if key[SERIES_KEY_WIDTH + 1] and len(modes) > 1:
            label += f" ({key[SERIES_KEY_WIDTH + 1]})"
        if key[SERIES_KEY_WIDTH] and len(passes) > 1:
            label += f" / {key[SERIES_KEY_WIDTH]}"
        ax.plot([0] + [p[0] for p in points], [1.0] + [p[1] for p in points],
                color=style["color"], marker=style["marker"],
                linestyle=style["linestyle"], linewidth=1.9, markersize=6,
                label=label)

    ax.axhline(1.0, color="#888888", linewidth=1.0, linestyle=":")
    ax.text(0.0, 1.01, " no degradation", fontsize=8, color="#888888", va="bottom")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, out_dir, stem)
