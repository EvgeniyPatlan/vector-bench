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

# matplotlib is a container dependency, not a host one. Charts are drawn inside
# a bench image, which carries it; the host is promised nothing beyond python3
# and PyYAML, and importing this module must not quietly break that promise.
# It does get imported on the host: the test suite reaches report.generate for
# its pure-logic helpers, and on a machine set up from the documented
# prerequisites that failed with ModuleNotFoundError seventy-six times.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402
    from matplotlib.ticker import FuncFormatter  # noqa: E402
except ImportError:  # pragma: no cover - every bench image has it
    matplotlib = None
    plt = None
    FuncFormatter = None


def have_matplotlib() -> bool:
    return plt is not None


def _require_matplotlib() -> None:
    """Refuse to draw rather than draw nothing.

    A report that came out with its charts silently missing would be worse than
    one that did not come out, which is the same reasoning that makes the
    generator refuse a run directory with no manifest.
    """
    if plt is None:
        raise RuntimeError(
            "matplotlib is not installed, so no chart can be drawn. Charts are "
            "generated inside a bench image which carries it: run "
            "`./run-benchmark.sh report --run-dir ...` rather than invoking "
            "report/generate.py on the host.")

# Engine identity is consistent across every chart in the report. Colour is never
# the only channel: each engine also gets a marker and a line style, so the
# charts survive greyscale printing and the common colour-vision deficiencies.
#
# Why a given engine has a given hue is recorded beside the value itself, in
# config/engines/<name>.yml.
#
# These are defaults. The authority is config/engines/*.yml, which the run copies
# into its manifest as `engines.<name>.presentation`; adopt_presentation() below
# applies it. The table stays because the report container mounts report/ and
# harness/ only -- importing the orchestrator from here crashed generation at the
# end of a twenty-hour run once -- and because a manifest written before that
# field existed still has to render.
STYLE: Dict[str, Dict[str, Any]] = {
    "mariadb":  {"color": "#1f77b4", "marker": "o", "linestyle": "-",  "label": "MariaDB 11.8 (MHNSW)"},
    "mariadb123": {"color": "#5fa8d3", "marker": "s", "linestyle": "--", "label": "MariaDB 12.3 (MHNSW)"},
    "alisql":   {"color": "#d62728", "marker": "s", "linestyle": "--", "label": "AliSQL (VIDX)"},
    "pgvector": {"color": "#2ca02c", "marker": "^", "linestyle": "-.", "label": "PostgreSQL (pgvector)"},
    "mongodb":  {"color": "#9467bd", "marker": "v", "linestyle": "-",  "label": "Percona Search (mongot)"},
    "valkey":   {"color": "#e377c2", "marker": "D", "linestyle": "-",  "label": "Valkey (valkey-search)"},
}


def adopt_presentation(manifest: Dict[str, Any]) -> None:
    """Take engine colours and labels from the run that produced the numbers.

    An engine added after this file was written has its style in its own config
    and therefore in the manifest, so it draws correctly without an edit here.
    """
    for name, info in (manifest.get("engines") or {}).items():
        presentation = (info or {}).get("presentation") or {}
        if not presentation:
            continue
        current = dict(STYLE.get(name, FALLBACK))
        for key, field in (("color", "color"), ("marker", "marker"),
                           ("linestyle", "linestyle"), ("label", "chart_label")):
            value = presentation.get(field)
            if value:
                current[key] = value
        STYLE[name] = current


FALLBACK = {"color": "#7f7f7f", "marker": "D", "linestyle": ":", "label": "unknown"}

GRID = {"alpha": 0.25, "linewidth": 0.6}


def style_for(engine: str) -> Dict[str, Any]:
    return STYLE.get(engine, {**FALLBACK, "label": engine})


# Axes a single engine can be swept over within one run. Two measurements of
# one engine that differ on either are different curves, not repeats of one:
# grouping a chart by engine name alone put several points at every x and drew
# a line through all of them that no configuration ever produced.
# Fallback axes, for records that arrive without a source file -- a report
# rebuilt with --from-records from an archive, for instance.
#
# Listing the axes by hand has now failed three times. It missed storage engine
# and ef_construction, then graph degree, and then quantization: two Percona
# Search curves were drawn as one line and came out as a sawtooth. Each time
# the fix was to add the axis that had just been added to the sweep, and each
# time the next axis was already waiting. So identity comes from the result
# filename now, which ann-benchmarks names after the build arguments and which
# therefore covers axes this code has never heard of.
SERIES_AXES = ("storage_engine", "ef_construction", "m")

# How each fallback axis names itself in a legend. Empty means the value speaks
# for itself, as a storage engine does.
AXIS_LABELS = {"storage_engine": "", "ef_construction": "ef_c=", "m": "M="}

# (engine, configuration). churn_impact widens this and slices it back.
SERIES_KEY_WIDTH = 2

# Colour stays with the engine so versions remain comparable at a glance; the
# storage engine is carried by the linestyle instead.
STORAGE_LINESTYLES = ("-", "--", ":", "-.")


def series_key(record: Dict[str, Any]) -> Tuple[Any, ...]:
    """Identity of one plotted line: the engine, and which build it came from."""
    from .loaders import build_signature
    # The whole signature, not its last element: the filename branch ends in
    # the build stem, but the column fallback ends in one field, and taking
    # [-1] blindly collapsed every fallback record into a single series.
    return (record.get("engine"), build_signature(record))


def _config_tokens(record: Dict[str, Any]) -> Optional[List[str]]:
    """The build configuration as ann-benchmarks spelled it in the filename.

    `angular_M_16_quantization_scalar` -> [angular, M, 16, quantization, scalar]
    """
    from .loaders import build_signature
    signature = build_signature(record)
    if signature[0] != "file":
        return None
    return str(signature[-1]).split("_")


def series_labels(records: List[Dict[str, Any]]) -> Dict[Tuple[Any, ...], str]:
    """Legend text per series, naming only what actually varies.

    An engine swept over one configuration keeps its bare label: naming an axis
    the run did not sweep implies a comparison it did not make.

    Where the records carry a result filename, the varying part is found by
    comparing those filenames token by token within one engine, so a new build
    knob names itself without this code being taught about it. The token before
    the one that differs is its name -- `..._quantization_scalar` against
    `..._quantization_none` gives `quantization=scalar`.
    """
    per_engine: Dict[Any, List[Dict[str, Any]]] = {}
    for r in records:
        per_engine.setdefault(r.get("engine"), []).append(r)

    labels: Dict[Tuple[Any, ...], str] = {}
    for engine, rows in per_engine.items():
        base = style_for(engine)["label"]
        tokenised = {series_key(r): _config_tokens(r) for r in rows}
        usable = [t for t in tokenised.values() if t]
        widths = {len(t) for t in usable}

        if len(usable) == len(tokenised) and len(widths) == 1 and len(usable) > 1:
            differing = [i for i in range(widths.pop())
                         if len({tuple(t)[i] for t in usable}) > 1]
            for key, tokens in tokenised.items():
                parts = [f"{tokens[i - 1]}={tokens[i]}" if i else tokens[i]
                         for i in differing]
                labels[key] = base + (" / " + " / ".join(parts) if parts else "")
            continue

        # No filenames, or filenames that cannot be compared: fall back to the
        # named axes, which is all a --from-records rebuild has.
        varying = [set() for _ in SERIES_AXES]
        for r in rows:
            for i, axis in enumerate(SERIES_AXES):
                varying[i].add(r.get(axis))
        for r in rows:
            label = base
            for i, axis in enumerate(SERIES_AXES):
                value = r.get(axis)
                if value is not None and len(varying[i]) > 1:
                    label += (f" / {AXIS_LABELS[axis]}{value}"
                              if AXIS_LABELS[axis] else f" / {value}")
            labels[series_key(r)] = label
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
    for key, rows_in in groups.items():
        sample = rows_in[0]
        storages = per_engine.setdefault(key[0], [])
        if sample.get("storage_engine") not in storages:
            storages.append(sample.get("storage_engine"))
        ms = degrees.setdefault(key[0], [])
        if sample.get("m") not in ms:
            ms.append(sample.get("m"))
    for v in list(per_engine.values()) + list(degrees.values()):
        v.sort(key=lambda x: (x is None, x))
    return groups, series_labels(records), per_engine, degrees


def _new_axes(title: str, xlabel: str, ylabel: str,
              figsize: Tuple[float, float] = (9, 5.5)):
    # Every chart starts here, so this is the one place that has to check.
    _require_matplotlib()
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

# ---------------------------------------------------------------------------
# PR 5 §5.1 item 6: "show intervals as whiskers on churn and filtered charts".
# ---------------------------------------------------------------------------

#: `extra` key holding a recall interval, written by `stats.ci_fields`.
#: Spelled here rather than imported at module scope for the reason
#: `generate.consistency_checks` gives: `report/` and `harness/` mount
#: separately and a top-level `harness` import from this package has crashed
#: report generation at the end of a long run. `recall_whiskers` imports the
#: reader function in its body; this constant only names a dict key.
RECALL_CI_EXTRA_KEY = "recall_ci95"


def recall_whiskers(rows: Sequence[Dict[str, Any]], xs: Sequence[Any],
                    ys: Sequence[Any], key: str = RECALL_CI_EXTRA_KEY):
    """The `(x, y, yerr)` an `errorbar` call needs, for the rows that have one.

    Returns `None` when no row in the series carries a usable interval, so a
    caller can skip the `errorbar` entirely rather than drawing a series of
    zero-length bars -- which would read as "measured with no uncertainty" on
    a chart where most points simply predate the field.

    `yerr` is asymmetric and clamped at zero. The published mean is rounded to
    six places (`records` stores `round(recall, 6)`) while the interval bounds
    are not, so a point can sit a few units in the twelfth decimal outside its
    own interval; matplotlib raises `ValueError: 'yerr' must not contain
    negative values` on that, which would lose the whole figure at the end of
    a run to a rounding artifact. Clamped rather than widened, so the bar
    never claims more spread than was measured.
    """
    from harness.metrics.stats import Interval

    lower, upper, px, py = [], [], [], []
    for row, x, y in zip(rows, xs, ys):
        if y is None:
            continue
        interval = Interval.from_field((row.get("extra") or {}).get(key))
        if interval is None or not interval.usable:
            continue
        px.append(x)
        py.append(y)
        lower.append(max(0.0, float(y) - interval.low))
        upper.append(max(0.0, interval.high - float(y)))
    if not px:
        return None
    return px, py, [lower, upper]


def draw_recall_whiskers(ax, rows: Sequence[Dict[str, Any]],
                         xs: Sequence[Any], ys: Sequence[Any],
                         color: str) -> bool:
    """Draw the whiskers for one series onto one axis. True if it drew any.

    `zorder=2`, between the near_pass/near_fail band the filtered chart fills
    at `zorder=1` and the markers at `zorder=5`/`6`: a bar under the band is
    invisible and a bar over the "returned fewer than k" ring would hide the
    mark it is there to draw attention to.

    No `label`, so the legend keeps one entry per series. A second handle per
    engine would double the legend's width on a six-engine run, and the
    footnote under the figure explains the bars once for all of them.
    """
    drawn = recall_whiskers(rows, xs, ys)
    if drawn is None:
        return False
    px, py, yerr = drawn
    ax.errorbar(px, py, yerr=yerr, fmt="none", ecolor=color, elinewidth=1.2,
                capsize=3, capthick=1.2, alpha=0.75, zorder=2)
    return True


#: Added to a chart title when at least one series drew whiskers. Only then:
#: on a run of archived records nothing carries an interval, and a legend
#: explaining bars that are not on the figure is worse than no explanation.
WHISKER_TITLE_NOTE = "bars = 95% bootstrap CI of the recall"


def build_cost(records: List[Dict[str, Any]], dataset: str, out_dir: str,
               stem: str) -> Optional[Dict[str, str]]:
    """Build cost against M: time to a queryable index, the bulk build step on
    its own, peak memory and index size.

    **The first two panels are a fix, and the defect was a published one.**
    This chart used to plot the raw `build_wall_s` of every engine on one axis
    under a "lower is better" arrow. That field is two different quantities:
    `workloads/build.py` writes `effective_build = build_seconds if not
    incremental else load.wall_seconds`, so for pgvector in bulk mode it is
    the `CREATE INDEX` call alone and for MariaDB MHNSW and AliSQL VIDX it is
    the ENTIRE million-row load, because they maintain the graph on every
    INSERT and there is no separable step to time.

    `build.py`'s own module docstring exists to prevent exactly this. It
    records the two figures separately "so the report can say 'pgvector's bulk
    build is N x cheaper than its own incremental build' and 'pgvector's
    incremental build costs X versus MHNSW's Y' -- two honest claims instead
    of one misleading one". The build TABLE honoured that through
    `render._index_build_kind`; this chart read `extra.separable_build`
    nowhere and selected on dataset and non-null alone.

    The gap had also just widened for a reason with nothing to do with index
    building: adding the `ctag` filter column slows the load, which raises the
    incremental engines' `build_wall_s` because it IS the load time, while
    pgvector's bulk figure is untouched -- the index build reads only the
    vector column.

    So the axes now compare like with like:

    * **Time to a queryable index** is `generate.time_to_queryable_index_s`:
      ingest + build for a bulk engine, the single figure for an incremental
      one. Every point on it is the same quantity, which is what makes both of
      the claims above answerable from it -- the first as one engine's two
      build modes, the second as two engines.
    * **Bulk index build step** is the raw `build_wall_s` restricted to
      records whose build WAS separable. An incremental engine has no point
      there, which is the honest rendering of "there is no such step", and no
      number on that panel includes a corpus load.

    A record whose `separable_build` is absent appears on neither: the two
    formulas differ by a whole corpus load and nothing in the record says
    which applies. Those engines are named in the subtitle rather than
    dropped silently -- see `generate.BUILD_KIND_UNRECORDED` for why absence
    has no honest reading here.
    """
    from report.generate import (BUILD_KIND_BULK, BUILD_KIND_UNRECORDED,
                                 build_kind, time_to_queryable_index_s)

    plotted_records = [r for r in records if r.get("dataset") == dataset
                       and r.get("build_wall_s") is not None]
    if not plotted_records:
        return None
    by_series, labels, storages, degrees = _grouped(plotted_records)

    def bulk_build_step(record):
        """`build_wall_s`, but only where it is a build STEP.

        Returning None for every other kind is what keeps a corpus load off
        this panel. Reading the field directly here instead would restore the
        defect above in one line, which is why this is a named function with
        this docstring rather than a field name in the table below.
        """
        if build_kind(record) != BUILD_KIND_BULK:
            return None
        return record.get("build_wall_s")

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.6))
    fig.patch.set_alpha(0.0)
    #: `(value of one record, axis label, tick formatter)`. A CALLABLE and not
    #: a field name: the first two panels are derivations rather than columns,
    #: and a bare field name is the form that let the misleading panel be
    #: written in the first place.
    panels = [
        (time_to_queryable_index_s,
         "Time to a queryable index (s)  (lower is better ↓)", None),
        (bulk_build_step,
         "Bulk index build step (s)  (bulk builds only)", None),
        (lambda r: r.get("peak_rss_bytes"),
         "Peak server memory  (lower is better ↓)", _bytes_formatter),
        (lambda r: r.get("index_bytes"),
         "Index size on disk  (lower is better ↓)", _bytes_formatter),
    ]

    any_data = False
    for ax, (value_of, label, formatter) in zip(axes, panels):
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
                ((r.get("m"), value_of(r)) for r in rows
                 if r.get("m") is not None and value_of(r) is not None),
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
    unrecorded = sorted({str(r.get("engine")) for r in plotted_records
                         if build_kind(r) == BUILD_KIND_UNRECORDED})
    subtitle = f"Index build cost — {dataset}"
    if unrecorded:
        subtitle += (
            f"\n{len(unrecorded)} engine(s) left off the first two panels "
            f"({', '.join(unrecorded)}): their records do not say whether the "
            f"build was separable, so the build time cannot be placed")
    fig.suptitle(subtitle, fontsize=13, y=1.14)
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


#: The two statuses' footnote marks, imported from the renderer rather than
#: declared here so the mark in this chart's legend and the mark in the
#: operating-point table's footnote cannot drift apart -- a dagger on the
#: chart explained by a footnote under a different glyph in the table is worse
#: than no mark at all. `render.py` imports no chart code, so this is the
#: direction with no cycle; it is also where `CLIENT_BOUND_MARK` already
#: lives, and that one IS spelled twice (as a literal in the suptitle below),
#: which is exactly the drift this import avoids.
#: Imported, not re-spelled, for the same reason the marks are: the chart
#: title and the operating-point table's Target column name the same number
#: and must name it the same way. `_fmt_target` is `%g` rather than fixed
#: decimals, and its docstring has the measured reason. PR 2 §2.3.3 writes the
#: title with a literal `0.95`, and `{0.95:g}` is exactly `0.95`, so the
#: default target renders the spec's own string.
from .render import (FLOOR_MARK, UNCALIBRATED_MARK,  # noqa: E402
                     UNREACHED_MARK, _fmt_target)


def _uncalibrated_engines(records: List[Dict[str, Any]], dataset: str
                          ) -> List[str]:
    """Engines plotted on this dataset's concurrency chart with no status.

    An engine qualifies when ANY of its plotted points lacks
    `extra.calibration_status`, not only when all of them do. One engine can
    be half-calibrated: the checkpoint grain is (pass, engine, dataset, phase)
    and within an ops phase (M, build_mode), so a resume after flipping
    `ops.ef_mode` can leave one M of an engine measured at the profile's
    literal ef and the next calibrated. The line drawn through both is then
    partly held to the target and partly not, and the honest treatment of a
    line like that is the same as of one held to it nowhere.
    """
    plotted = [r for r in records if _is_concurrency_point(r, dataset)]
    unheld = {r.get("engine") for r in plotted
              if not (r.get("extra") or {}).get("calibration_status")}
    return sorted(e for e in unheld if e is not None)


def _qps_panel_title(records: List[Dict[str, Any]], dataset: str) -> str:
    """What the concurrency chart's QPS panel is actually showing.

    PR 2 §2.3.3 specifies the title "QPS at recall >= 0.95". The 0.95 in it is
    `DEFAULT_TARGET_RECALL`, not a constant: `--target-recall` is a flag and
    `extra.target_recall` is stamped on every record of a calibrated run, so
    the title is built from the records being plotted and renders the spec's
    literal string whenever the run used the default. Hard-coding 0.95 would
    caption a `--target-recall 0.90` run with a floor none of its engines was
    held to -- the precise failure this title exists to prevent, inverted, and
    the reader has nothing else on the page to catch it with.

    Three cases, because the honest caption differs in each:

    * one target -- the spec's title, with that target's value.
    * none -- `ef_mode: fixed`. No calibration ran, every engine sat at the
      profile's literal ef_search, and what recall that bought differs per
      engine by up to 0.18 on these implementations. Claiming a recall floor
      here would be the chart asserting a comparison the run did not make, so
      it says the opposite.
    * more than one -- a `--from-records` rebuild over records merged from two
      runs. Naming either target would be false about the other half of the
      chart, so both are named and neither is presented as the floor.

    Then qualified once more, because a single target is not the same claim as
    a floor that held. An engine with no `calibration_status` was never put at
    the target, and the first version of this function got that right at RUN
    granularity -- no targets anywhere means no claim -- and wrong at ENGINE
    granularity: one calibrated engine beside one resumed from a fixed-ef pass
    titled the panel "QPS at recall >= 0.95" over a line that had never been
    held to 0.95, with no mark on it and no row in the operating-point table.
    See `_uncalibrated_engines` for how that state is reached without anyone
    hand-editing a file.

    `dataset` narrows the targets as well. `ops.target_recall` is one
    profile-level value (`orchestrator/ops_pass.py` passes a single
    `--target-recall`), so one run cannot vary it across datasets and this is
    unreachable from a single run directory -- but a `--from-records` rebuild
    over two merged runs can, and it put the sift run's target in the glove
    chart's title: measured, two datasets at 0.80 and 0.95 produced "QPS at
    mixed recall targets (0.8, 0.95)" on the glove panel, where every record
    plotted had been held to 0.95.
    """
    from .loaders import calibration_targets

    scoped = [r for r in records if r.get("dataset") == dataset]
    targets = calibration_targets(scoped)
    if not targets:
        return "QPS at a fixed ef_search — recall differs per engine"
    if len(targets) == 1:
        title = f"QPS at recall ≥ {_fmt_target(targets[0])}"
    else:
        title = ("QPS at mixed recall targets ("
                 + ", ".join(_fmt_target(t) for t in targets) + ")")
    if _uncalibrated_engines(records, dataset):
        title += f" — not for every engine ({UNCALIBRATED_MARK})"
    return title


def _calibration_marks(records: List[Dict[str, Any]], dataset: str
                       ) -> Dict[str, Dict[str, Any]]:
    """Per engine: which footnote marks it earned on this dataset, and why.

    Keyed on the ENGINE and not on the plotted series key, because the mark is
    a statement about the engine's floor and ceiling rather than about one
    line. `series_key` is (engine, build_signature) and `build_signature`
    omits `resource_pass`, so one engine measured in both passes is one series
    here while `operating_points` keys the pass separately -- an engine whose
    floor sits above the target in the tuned pass and not the normalized one
    therefore earns the mark, and the footnote names both so the reader is not
    left to guess which half it applies to.

    `recalls` carries the recall actually achieved, which §2.3.3 requires
    beside the `target_unreached` mark. It is the calibration sweep's recall at
    the chosen ef -- the largest in the grid for that status, which is the
    engine's best showing and what §2.1.6 says the engine is reported at. The
    concurrency record's own `recall_at_k` is a different measurement and is
    already its own column in the table.

    The dataset filter is load-bearing and is not a tidiness measure. Measured
    with it removed, over one engine daggered on glove and `ok` on sift: the
    sift chart's legend suffix went from `''` to `' †'` and its footnote from
    empty to an assertion that MariaDB cannot be slowed down to the target --
    on a chart of a dataset where that was never measured.

    `uncalibrated` is not a sweep outcome: the engine has plotted points with
    no `calibration_status` at all. It is collected here rather than beside
    the title so the legend, the footnote and the title all come from one
    derivation -- see UNCALIBRATED_MARK for how a run reaches that state.
    """
    from .loaders import operating_points

    marks: Dict[str, Dict[str, Any]] = {}

    def entry_for(engine: str) -> Dict[str, Any]:
        return marks.setdefault(engine, {"statuses": set(), "recalls": []})

    for row in operating_points(records):
        if row.get("dataset") != dataset:
            continue
        engine = row.get("engine")
        status = row.get("status")
        if engine is None or status not in ("floor_above_target",
                                            "target_unreached"):
            continue
        entry = entry_for(engine)
        entry["statuses"].add(status)
        entry["recalls"].append((status, row.get("resource_pass"),
                                 row.get("m"), row.get("recall_at_k")))

    # Only where a target was pursued at all. On `ef_mode: fixed` every engine
    # lacks a status, the title already says the chart is at a fixed ef, and
    # marking all six as exceptions to a claim nobody made would be noise.
    from .loaders import calibration_targets
    if calibration_targets([r for r in records
                            if r.get("dataset") == dataset]):
        for engine in _uncalibrated_engines(records, dataset):
            entry_for(engine)["statuses"].add("uncalibrated")
    return marks


def _mark_suffix(entry: Optional[Dict[str, Any]]) -> str:
    """The marks one legend label should carry, in footnote order."""
    if not entry:
        return ""
    # Known statuses only, by construction: `_calibration_marks` tests against
    # the two literals and adds `uncalibrated` itself. An unrecognised
    # `calibration_status` -- a future value, or a merged file from newer code
    # -- therefore reaches the table as a raw cell with no mark here and no
    # footnote, under a title that still claims the floor held. The status set
    # is closed in `calibrate.py` (`ok`, `target_unreached`,
    # `floor_above_target`) and the measurement version is what guards the
    # merged-file case, so this is not reachable today; it is the thing to
    # revisit if a fourth status is ever added.
    statuses = entry["statuses"]
    out = ""
    if "uncalibrated" in statuses:
        out += " " + UNCALIBRATED_MARK
    if "floor_above_target" in statuses:
        out += " " + FLOOR_MARK
    if "target_unreached" in statuses:
        out += " " + UNREACHED_MARK
    return out


def _agree(engines: Sequence[str]) -> Dict[str, str]:
    """Verb and pronouns for a footnote written over a LIST of engines.

    Chosen explicitly rather than written singular, because for two of the
    three footnotes multi-engine is the normal case, not the edge: an operator
    who flips `ops.ef_mode` mid-run has every engine already measured carrying
    no status, and `target_unreached` lands on whole families at low M -- the
    tuned-profile notes record all three MySQL-family engines between 0.736
    and 0.818 at M=6, every one of them below a 0.95 target.

    Written singular, the uncalibrated paragraph read, verbatim over two
    engines: "AliSQL (VIDX), PostgreSQL (pgvector) WAS not calibrated: IT
    carries no calibration_status, so IT was measured at ... does not apply to
    IT. ITS throughput ...". An explicit choice is also how
    `_refusal_message` already handles "1 record" against "2 records".
    """
    if len(engines) == 1:
        return {"was": "was", "it": "it", "its": "its", "carries": "carries",
                "is": "is"}
    return {"was": "were", "it": "they", "its": "their", "carries": "carry",
            "is": "are"}


def _calibration_footnote(marks: Dict[str, Dict[str, Any]]) -> str:
    """The footnotes §2.3.3 asks for, each printed only when it applies.

    The `floor_above_target` sentence carries §2.1.6's requirement verbatim --
    "cannot be slowed down to the target" -- because that is the one thing a
    reader of this chart must not have to infer. An engine sitting above the
    comparison recall is not being judged on the same terms as the rest, and a
    dagger with no sentence behind it reads as a minor caveat.

    A third paragraph for the engines that were never calibrated. It comes
    first, because it is the only one of the three that says the title above
    the panel does not describe that line at all.
    """
    floors = sorted(e for e, v in marks.items()
                    if "floor_above_target" in v["statuses"])
    unreached = sorted(e for e, v in marks.items()
                       if "target_unreached" in v["statuses"])
    uncalibrated = sorted(e for e, v in marks.items()
                          if "uncalibrated" in v["statuses"])
    parts: List[str] = []
    if uncalibrated:
        g = _agree(uncalibrated)
        parts.append(
            f"{UNCALIBRATED_MARK} "
            f"{', '.join(style_for(e)['label'] for e in uncalibrated)}"
            f" {g['was']} NOT calibrated: {g['carries']} no"
            f" calibration_status, so the recall floor in the title above does"
            f" not apply. Measured at the profile's literal ef_search,"
            f" {g['its']} throughput is not comparable with the rest at any"
            f" known recall")
    if floors:
        parts.append(
            f"{FLOOR_MARK} {', '.join(style_for(e)['label'] for e in floors)}"
            f" cannot be slowed down to the target: the smallest grid"
            f" ef_search is already above it, so this QPS is measured at a"
            f" HIGHER recall than the rest")
    if unreached:
        shown = []
        for engine in unreached:
            for status, _pass, _m, recall in marks[engine]["recalls"]:
                if status != "target_unreached" or recall is None:
                    continue
                shown.append(f"{style_for(engine)['label']} {recall:.4f}")
        detail = f" (best achieved: {', '.join(shown)})" if shown else ""
        g = _agree(unreached)
        parts.append(
            f"{UNREACHED_MARK} "
            f"{', '.join(style_for(e)['label'] for e in unreached)}"
            f" reached the target at no grid point and {g['is']} shown at"
            f" {g['its']} best achievable recall{detail} — a LOWER recall than"
            f" the rest")
    return "\n".join(parts)


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

    # PR 2 §2.3.3. On the QPS panel and not as the figure's title, because it
    # is true of that panel only: the latency and efficiency panels are not
    # throughput at a recall. Built from `records` rather than
    # `plotted_records` so a run whose concurrency workload was skipped for one
    # engine still reports the target the others were held to.
    axes[0].set_title(_qps_panel_title(records, dataset), fontsize=10)
    marks = _calibration_marks(records, dataset)

    for key, rows in sorted(by_series.items(), key=lambda kv: str(kv[0])):
        style = series_style(rows[0], storages.get(key[0]), degrees.get(key[0]))
        rows = sorted(rows, key=lambda r: r["clients"])
        clients = [r["clients"] for r in rows]

        axes[0].plot(clients, [r["qps"] for r in rows], color=style["color"],
                     marker=style["marker"], linestyle=style["linestyle"],
                     linewidth=1.8, markersize=6,
                     label=labels[key] + _mark_suffix(marks.get(key[0])))

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

        # A point where the CLIENT container ran out of CPU, hollowed the same
        # way the filtered chart hollows a short result set. Both panels,
        # because both carry the misreading: the throughput panel shows a
        # curve flattening and the efficiency panel shows it falling away from
        # linear, and for a client-bound point neither is evidence about the
        # engine. The clients are Python threads serializing vectors in a
        # container pinned to a few CPUs, and at 16-32 clients against a fast
        # engine that is a real ceiling of this harness rather than of the
        # database under it.
        for r in rows:
            extra = r.get("extra") or {}
            if not extra.get("client_bound"):
                continue
            axes[0].scatter([r["clients"]], [r["qps"]], s=140,
                            facecolors="none", edgecolors=style["color"],
                            linewidths=1.6, zorder=5)
            efficiency = extra.get("scaling_efficiency")
            if efficiency is not None:
                axes[2].scatter([r["clients"]], [efficiency], s=140,
                                facecolors="none", edgecolors=style["color"],
                                linewidths=1.6, zorder=5)

    # Linear-scaling reference so the efficiency panel has a meaning without
    # the reader having to remember what 1.0 represents.
    axes[2].axhline(1.0, color="#888888", linewidth=1.0, linestyle=":")
    axes[2].text(axes[2].get_xlim()[0], 1.02, " linear scaling",
                 fontsize=8, color="#888888", va="bottom")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, fontsize=10,
                   loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 1.06))
    # The calibration footnotes go below the axes, not into the suptitle that
    # already carries the hollow-marker legend: they are two full sentences
    # each and naming an engine's achieved recall there would push the title
    # wider than the figure.
    footnote = _calibration_footnote(marks)
    fig.suptitle(
        f"Concurrency scaling — {dataset}   "
        f"○ = client CPU-bound (>80% of the client container's CPUs)",
        fontsize=12, y=1.14)
    fig.tight_layout()
    if footnote:
        fig.text(0.01, -0.04, footnote, fontsize=9, color="#444444",
                 ha="left", va="top")
    return _save(fig, out_dir, stem)


# ---------------------------------------------------------------------------
# Filtered search
# ---------------------------------------------------------------------------

#: Grey, not an engine colour: a truncated point is a property of the
#: harness's budget rather than of the engine, and colouring it with the
#: engine's own palette entry would read as another measurement of it.
TIME_BOUNDED_GREY = "#8a8a8a"

#: What each filter column is, for the chart title. A reader who sees two
#: filtered charts needs to know why in the title and not in §2.4.
_COLUMN_BLURB = {
    "tag": "row_id % 100, independent of the vector space",
    "ctag": "cluster id, correlated with the vector space",
}


def filtered(records: List[Dict[str, Any]], dataset: str, out_dir: str,
             stem: str, column: Optional[str] = None
             ) -> Optional[Dict[str, str]]:
    """Filtered recall and QPS against selectivity, for one filter column.

    `column` is PR 3's "a column selector (or two charts)" done as two charts,
    which is the option that needs no JavaScript in a report that is also
    rendered to Markdown. `generate.py` calls this once per column present and
    names the files `filtered-tag-<dataset>` and `filtered-ctag-<dataset>`,
    both of which `render.py` picks up through its existing
    `figures("filtered-")` prefix.

    Drawing both columns on one pair of axes is what the previous version did
    by omission, and it is worse than it sounds: the x axis is the fraction of
    rows passing the filter, so a `tag` point at 0.0100 and a `ctag` point at
    0.0190 -- the share a nominal 1% actually delivers -- land at different x
    for the same request, and one line through both reads as an engine whose
    recall moved between two selectivities when it is two predicates.

    `column=None` plots every filtered record regardless, which is what an
    archived run needs: nothing written before PR 3 carries
    `extra.filter_column` at all.
    """
    plotted_records = [r for r in records if r.get("dataset") == dataset
                       and r.get("selectivity") is not None]
    if column is not None:
        from .generate import filter_column
        plotted_records = [r for r in plotted_records
                           if filter_column(r) == column]
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

    any_whiskers = False
    for key, rows in sorted(by_series.items(), key=lambda kv: str(kv[0])):
        style = series_style(rows[0], storages.get(key[0]), degrees.get(key[0]))
        rows = sorted(rows, key=lambda r: r["selectivity"])
        sel = [r["selectivity"] for r in rows]
        recalls = [r.get("recall_at_k") for r in rows]

        axes[0].plot(sel, recalls,
                     color=style["color"], marker=style["marker"],
                     linestyle=style["linestyle"], linewidth=1.8, markersize=6,
                     label=labels[key])
        # §5.1 item 6, on the recall panel only. The QPS panel's interval is
        # `extra.qps_ci95` and it is deliberately not drawn: that axis is
        # logarithmic and spans three decades here, so a bar a few percent
        # wide is shorter than the marker and says nothing a reader can see.
        # The number and its interval are both in the filtered table.
        if draw_recall_whiskers(axes[0], rows, sel, recalls, style["color"]):
            any_whiskers = True
        axes[1].plot(sel, [r.get("qps") for r in rows],
                     color=style["color"], marker=style["marker"],
                     linestyle=style["linestyle"], linewidth=1.8, markersize=6)

        # The near_pass / near_fail split, on the recall panel only. A query
        # whose true unfiltered nearest neighbour passes the filter has its
        # best answer inside the candidate list the engine was going to walk
        # anyway; one whose nearest neighbour fails has to keep looking, and
        # that is where post-filtering gives up. Drawn as a thin band between
        # the two rather than two more lines: the headline is still the
        # overall recall, and the gap is the finding.
        if any((r.get("extra") or {}).get("recall_near_fail") is not None
               for r in rows):
            band_sel, lo, hi = [], [], []
            for r in rows:
                extra = r.get("extra") or {}
                a, b = extra.get("recall_near_fail"), extra.get("recall_near_pass")
                if a is None or b is None:
                    continue
                band_sel.append(r["selectivity"])
                lo.append(min(a, b))
                hi.append(max(a, b))
            if band_sel:
                axes[0].fill_between(band_sel, lo, hi, color=style["color"],
                                     alpha=0.13, linewidth=0, zorder=1)

        for r in rows:
            extra = r.get("extra") or {}
            # An engine returning fewer than k rows is a real behaviour, not
            # noise — mark it so a high-looking recall is not read as
            # unqualified success.
            if extra.get("returned_fewer_than_k"):
                axes[0].scatter([r["selectivity"]], [r.get("recall_at_k")],
                                s=140, facecolors="none",
                                edgecolors=style["color"], linewidths=1.6, zorder=5)
            # A point the budget stopped early, greyed as PR 3 asks. Its
            # recall is a mean over however many queries finished, so it
            # carries variance the other points do not; `insufficient_sample`
            # (under 50 queries) gets the heavier mark of the two.
            if extra.get("time_bounded"):
                heavy = bool(extra.get("insufficient_sample"))
                axes[0].scatter([r["selectivity"]], [r.get("recall_at_k")],
                                s=190 if heavy else 150, marker="X" if heavy else "s",
                                facecolors="none", edgecolors=TIME_BOUNDED_GREY,
                                linewidths=2.0 if heavy else 1.3, zorder=6)

    axes[0].set_ylim(-0.02, 1.02)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, fontsize=10,
                   loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 1.06))
    titled = f"Filtered (hybrid) search — {dataset}"
    if column is not None:
        titled += f" — {column} ({_COLUMN_BLURB.get(column, column)})"
    fig.suptitle(
        f"{titled}\n○ = returned fewer than k results   "
        f"□ = stopped by the time budget   ✕ = under 50 queries   "
        f"band = near_pass vs near_fail recall"
        + (f"   {WHISKER_TITLE_NOTE}" if any_whiskers else ""),
        fontsize=11, y=1.18
    )
    fig.tight_layout()
    return _save(fig, out_dir, stem)


# ---------------------------------------------------------------------------
# Churn
# ---------------------------------------------------------------------------

def churn(records: List[Dict[str, Any]], dataset: str, out_dir: str,
          stem: str) -> Optional[Dict[str, str]]:
    """Recall and QPS after successive delete/re-insert cycles.

    **One series per (configuration, maintenance state), which is a fix.** PR 4
    §4.2 writes two records per churn fraction -- the index as the churn left
    it and the index after the engine's repair path ran -- and this chart
    grouped by configuration alone, so both landed at the same x in one
    series. Measured on the synthetic driver, 9 of 9 runs of one
    configuration: x = [0.0, 0.1, 0.1, 0.25, 0.25], one series, so each
    fraction was drawn as a vertical segment rather than as two curves. A line
    through both is a line no configuration ever produced.

    The pre-churn baseline record names no state, so it joins BOTH series: it
    is the x=0 point each of them is measured against, and giving it to one
    would leave the other starting at its first churn fraction.

    `churn_measurements` excludes the reader-pool records §4.3 writes. Their
    `qps` is aggregate throughput over `ops.churn_readers` clients against a
    corpus that is mid-change, and they carry no recall at all; on this chart
    they would be a third point per fraction, several times higher on the QPS
    panel and absent from the recall one.
    """
    from report.churn_state import churn_measurements, maintenance_state

    plotted_records = [r for r in churn_measurements(records)
                       if r.get("dataset") == dataset
                       and r.get("churn_fraction") is not None]
    if not plotted_records:
        return None
    _, labels, storages, degrees = _grouped(plotted_records)

    # Keyed `series_key(r) + (state,)` and sliced back with
    # `SERIES_KEY_WIDTH`, which is the idiom `churn_impact` below already uses
    # to widen this key by the resource pass and the build mode.
    states = {maintenance_state(r) for r in plotted_records} - {None}
    by_series: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for r in plotted_records:
        state = maintenance_state(r)
        for member in (sorted(states) if state is None else [state]):
            by_series.setdefault(series_key(r) + (member,), []).append(r)

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

    any_whiskers = False
    for key, rows in sorted(by_series.items(), key=lambda kv: str(kv[0])):
        style = series_style(rows[0], storages.get(key[0]), degrees.get(key[0]))
        rows = sorted(rows, key=lambda r: r["churn_fraction"])
        fractions = [r["churn_fraction"] for r in rows]
        # The state is named only when the figure actually holds more than one
        # of them, the way `churn_impact` names the pass and the build mode:
        # on a run of engines with no repair path every entry would otherwise
        # read "... / before", which is a distinction the run never made.
        label = labels[key[:SERIES_KEY_WIDTH]]
        if len(states) > 1 and key[SERIES_KEY_WIDTH]:
            label += f" / {key[SERIES_KEY_WIDTH]} repair"
        recalls = [r.get("recall_at_k") for r in rows]
        axes[0].plot(fractions, recalls,
                     color=style["color"], marker=style["marker"],
                     linestyle=style["linestyle"], linewidth=1.8, markersize=6,
                     label=label)
        axes[1].plot(fractions, [r.get("qps") for r in rows],
                     color=style["color"], marker=style["marker"],
                     linestyle=style["linestyle"], linewidth=1.8, markersize=6)
        # §5.1 item 6. The recall panel only, for the reason the filtered
        # chart gives -- and here there is a second one: the x=0 baseline
        # record joins BOTH maintenance series, so its bar is drawn twice in
        # the same place. That is honest (it IS the same measurement, and it
        # is the point each series is compared against) and it is why the bar
        # at x=0 looks heavier than the others.
        if draw_recall_whiskers(axes[0], rows, fractions, recalls,
                                style["color"]):
            any_whiskers = True

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, fontsize=10,
                   loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 1.06))
    fig.suptitle(f"Index degradation under churn — {dataset}"
                 + (f"\n{WHISKER_TITLE_NOTE}" if any_whiskers else ""),
                 fontsize=13, y=1.14)
    fig.tight_layout()
    return _save(fig, out_dir, stem)


def memory_timeline(series: Dict[str, List[Dict[str, Any]]], out_dir: str,
                    stem: str, phase: Optional[str] = None
                    ) -> Optional[Dict[str, str]]:
    """Server memory through one phase, every measurement from its own zero.

    What this chart is for is the shape: a cache filling to its ceiling looks
    different from a build spike, which looks different again from a steady
    climb that never plateaus. Shapes are compared by overlaying them, so every
    measurement starts at zero -- including two measurements that happen to
    live in one file because a phase was run twice.

    Anchoring them to the file's start instead put the second measurement 123
    hours to the right, which turned the whole chart into empty space with two
    thin bands at either end. That is the same picture as before the sessions
    were split, and it was not what splitting them was for.

    `phase` selects which measurements to draw. The ann phase loads a corpus
    and sweeps a grid; the ops phase loads it again and runs four workloads.
    They take different times and answer different questions, and twelve series
    on one axis was already crowded before any of them had two measurements.
    """
    from .loaders import split_sessions

    wanted = {}
    for name, rows in series.items():
        if phase and not name.endswith(f"-{phase}"):
            continue
        if phase is None and name.endswith("-ann"):
            continue
        rows = [r for r in rows if r.get("rss_bytes")]
        if rows:
            wanted[name] = rows
    if not wanted:
        return None

    title = "Server memory" + (f" — {phase} phase" if phase else " — ops phase")
    fig, ax = _new_axes(title, "Elapsed within the measurement (s)",
                        "Resident memory  (lower is better ↓)", figsize=(11, 4.6))
    ax.yaxis.set_major_formatter(FuncFormatter(_bytes_formatter))

    for name, rows in sorted(wanted.items()):
        engine = name.split("-")[0]
        style = style_for(engine)
        segments = split_sessions(rows)
        for index, segment in enumerate(segments):
            # Each measurement from its own zero, so the shapes overlay.
            t0 = segment[0]["t"]
            label = name if index == 0 else None
            if len(segments) > 1 and index == 0:
                label = f"{name}  ({len(segments)} measurements)"
            ax.plot([r["t"] - t0 for r in segment],
                    [r["rss_bytes"] for r in segment],
                    color=style["color"], linestyle=style["linestyle"],
                    linewidth=1.4, alpha=1.0 if index == 0 else 0.55,
                    label=label)

    ax.legend(frameon=False, fontsize=8, ncol=2)
    return _save(fig, out_dir, stem)

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
    """p50 and p99 against search width, one panel per engine.

    Mean latency hides the tail, and the tail is what a service-level objective
    is written against. An engine with a good p50 and a bad p99 is a different
    proposition from one with both merely acceptable.

    One panel each, rather than everything on one axis. Two things made the
    single-axis version unreadable and they compound: the sweep grew to a dozen
    series once graph degree and ef_construction were both being varied, and
    AliSQL's cliff put 384 ms on the same linear scale as Valkey's 0.7 ms, so
    five engines were pressed flat against the floor. Panels fix the crowding
    and a shared log scale fixes the range, while still letting a reader
    compare across panels -- which a per-engine y-scale would not.
    """
    plotted_records = [r for r in records
                       if r.get("phase") == "recall_qps" and r.get("dataset") == dataset
                       and r.get("ef_search") and r.get("latency_p99_ms")]
    if not plotted_records:
        return None
    by_series, labels, storages, degrees = _grouped(plotted_records)

    engines = sorted({key[0] for key in by_series})
    columns = min(3, len(engines))
    rows_count = (len(engines) + columns - 1) // columns
    fig, axes = plt.subplots(rows_count, columns, sharex=True, sharey=True,
                             figsize=(4.4 * columns, 3.1 * rows_count))
    fig.patch.set_alpha(0.0)
    axes = list(axes.ravel()) if hasattr(axes, "ravel") else [axes]

    values = [v for r in plotted_records
              for v in (r.get("latency_p50_ms"), r.get("latency_p99_ms"))
              if v]
    low = max(0.1, min(values) * 0.7)
    high = max(values) * 1.4

    for index, engine in enumerate(engines):
        ax = axes[index]
        ax.patch.set_alpha(0.0)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_ylim(low, high)
        ax.set_title(style_for(engine)["label"], fontsize=10.5, pad=6)
        ax.grid(True, which="major", alpha=0.25, linewidth=0.6)

        for key, series_rows in sorted(by_series.items(), key=lambda kv: str(kv[0])):
            if key[0] != engine:
                continue
            style = series_style(series_rows[0], storages.get(engine),
                                 degrees.get(engine))
            series_rows = sorted(series_rows, key=lambda r: r["ef_search"])
            ef = [r["ef_search"] for r in series_rows]
            p50 = [r.get("latency_p50_ms") for r in series_rows]
            p99 = [r.get("latency_p99_ms") for r in series_rows]
            # Band between p50 and p99 makes the spread visible at a glance;
            # the solid line is p50 so the medians stay comparable.
            ax.fill_between(ef, p50, p99, color=style["color"],
                            alpha=0.13, linewidth=0)
            # The engine is the panel, so within it the label names only what
            # varies -- the graph degree, or the build-quality setting.
            within = labels[key]
            within = within.split(" / ", 1)[1] if " / " in within else None
            ax.plot(ef, p50, color=style["color"], marker=style["marker"],
                    linestyle=style["linestyle"],
                    linewidth=style.get("linewidth", 1.9),
                    alpha=style.get("alpha", 1.0), markersize=4,
                    label=within)
            ax.plot(ef, p99, color=style["color"], linestyle=":",
                    linewidth=1.1, alpha=0.8 * style.get("alpha", 1.0))

        if any(l.get_label() and not l.get_label().startswith("_")
               for l in ax.get_lines()):
            ax.legend(frameon=False, fontsize=8, loc="upper left")

    for spare in axes[len(engines):]:
        spare.set_visible(False)

    for index, ax in enumerate(axes[:len(engines)]):
        if index >= len(engines) - columns:
            ax.set_xlabel("ef_search  (search width)", fontsize=9.5)
        if index % columns == 0:
            ax.set_ylabel("Latency (ms)  ↓", fontsize=9.5)

    fig.suptitle(f"Query latency distribution — {dataset}", fontsize=13)
    fig.text(0.99, 0.005, "solid = p50;  dotted = p99;  shaded = the span "
             "between them.  Log scale, shared across panels.",
             ha="right", va="bottom", fontsize=8, color="#888888")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
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

    **The repaired measurement is excluded, and that is the fix for a second
    instance of the same defect.** PR 4 §4.2 writes two records per churn
    fraction -- before and after the engine's repair path -- and this function
    put both in one series' point list. Since it returns `sorted(points)`, the
    pair arrived duplicated at every x AND indistinguishable by position:
    measured on the synthetic driver, x = [0.1, 0.1, 0.25, 0.25] in 9 of 9
    runs of one configuration.

    Excluded rather than split into two series, and the reason is the
    denominator rather than the plumbing. This ratio is against the PRE-CHURN
    baseline, a state before any maintenance ran, so a repaired measurement
    over it answers "what did churn plus the repair cost?" -- a different
    question from the one the axis asks, and one that cannot share an axis
    labelled "QPS as a fraction of pre-churn" without saying which of the two
    each line is. The repaired state is not lost: `charts.churn` draws it as
    its own series and `render._churn_table` gives it its own row with the
    `Repair` column naming what ran.

    This does NOT claim to fix the ratios above 1.0 that this chart can
    produce, and no mechanism is asserted for them. On the synthetic driver,
    which runs no vacuum at all, 2 of 9 runs had BOTH points at a fraction
    above 1.0 -- so the un-maintained measurement outran the baseline too, and
    "a vacuum makes it faster" is refuted as the explanation there. It is
    jitter in a 50-query QPS figure whose spread was 6,028 to 16,811 across
    those runs. Whether a real engine's post-repair throughput can beat its
    own pre-churn baseline is plausible and unmeasured.

    The reader-pool records §4.3 writes are excluded by `churn_measurements`
    for a harder reason: their `qps` is aggregate throughput over
    `ops.churn_readers` clients, so dividing it by a single-client baseline
    publishes a retention of roughly the reader count on an axis where 1.0
    means "no degradation".
    """
    from report.churn_state import churn_measurements, maintenance_state

    baseline: Dict[Tuple[Any, ...], float] = {}
    after: Dict[Tuple[Any, ...], List[Tuple[float, float]]] = {}
    for r in churn_measurements(records):
        if r.get("dataset") != dataset or not r.get("qps"):
            continue
        if maintenance_state(r) == "after":
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
    # The same record set `churn_retention` derived `retained` from, so the
    # legend text and the `next(...)` style lookup below describe the series
    # that were actually plotted. `series_labels` compares records within an
    # engine to find what varies, so feeding it a set the ratio excluded can
    # name an axis no plotted line differs on.
    from report.churn_state import churn_measurements

    churn_records = [r for r in churn_measurements(records)
                     if r.get("dataset") == dataset and r.get("qps")]
    labels = series_labels(churn_records)
    storages: Dict[str, List[Optional[str]]] = {}
    for key in {series_key(r) for r in churn_records}:
        storages.setdefault(key[0], []).append(None)
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


# ---------------------------------------------------------------------------
# ef_search calibration (appendix)
# ---------------------------------------------------------------------------

def calibration_curves(records: List[Dict[str, Any]], dataset: str,
                       out_dir: str, stem: str) -> Optional[Dict[str, str]]:
    """PR 2 §2.3.4: recall against ef_search, one line per engine.

    The audit trail for every other number in a calibrated report. The
    operating-point table states the ef each engine was given; this is the
    curve that choice was made from, so a reader can see whether the chosen
    point sits on a plateau or on a slope — and a sweep that early-stopped
    after three of twelve grid points is visibly three points long rather than
    silently short.

    It "doubles as a sanity check against the ann frontier" in the sense that a
    reader can carry a recall off this chart to §4 and find the same engine at
    the same ef_search there. The two are deliberately NOT drawn on one axis:
    `PHASE_CALIBRATION`'s own docstring is explicit that these are not Pareto
    points — they are single-client probes against the index the ops pass just
    built, measured in a different container, over a different query count, by
    a different code path's recall computation. Overlaid they would read as one
    curve, and the first thing a reader would do with a disagreement is
    average it. Kept apart, a disagreement between the two charts is what it
    actually is: a signal that one of the two measurement paths is wrong about
    this engine.

    Returns None when no sweep ran, which `generate.py` also guards ahead of
    the call: the chart is optional because the sweep is.
    """
    from .loaders import (calibration_records, operating_point_key,
                          operating_points)

    _require_matplotlib()
    points = [r for r in calibration_records(records)
              if r.get("dataset") == dataset
              and r.get("ef_search") is not None
              and r.get("recall_at_k") is not None]
    if not points:
        return None

    targets = sorted({(r.get("extra") or {}).get("target_recall")
                      for r in points
                      if (r.get("extra") or {}).get("target_recall")
                      is not None})

    fig, ax = _new_axes(
        f"ef_search calibration — {dataset}",
        "ef_search (requested)", "Recall@k  (single-client sweep)",
        figsize=(9, 5.5))
    ax.set_xscale("log", base=2)

    by_series, labels, storages, degrees = _grouped(points)
    # Which (unit, ef) pairs the run actually settled on, so the ring marks the
    # point the downstream workloads were measured at rather than this
    # function's own guess at which point the rule should have picked. Matched
    # through `operating_point_key`, via the key each row carries, so there is
    # one definition of the unit: a ring on a point no workload ran at, or no
    # ring where a sweep succeeded, is then a real disagreement between the
    # sweep and what consumed it rather than two derivations drifting.
    settled = {(row["key"], row["ef_search"])
               for row in operating_points(records)
               if row.get("dataset") == dataset
               and row.get("ef_search") is not None}

    for key, rows in sorted(by_series.items(), key=lambda kv: str(kv[0])):
        style = series_style(rows[0], storages.get(key[0]), degrees.get(key[0]))
        rows = sorted(rows, key=lambda r: r["ef_search"])
        ax.plot([r["ef_search"] for r in rows], [r["recall_at_k"] for r in rows],
                color=style["color"], marker=style["marker"],
                linestyle=style["linestyle"], linewidth=1.8, markersize=5,
                label=labels[key])
        chosen = [r for r in rows
                  if (operating_point_key(r), r["ef_search"]) in settled]
        if chosen:
            ax.scatter([r["ef_search"] for r in chosen],
                       [r["recall_at_k"] for r in chosen], s=170,
                       facecolors="none", edgecolors=style["color"],
                       linewidths=1.8, zorder=5)

    for target in targets:
        ax.axhline(target, color="#888888", linewidth=1.0, linestyle="--")
        ax.text(ax.get_xlim()[0], target + 0.002,
                f" target {_fmt_target(target)}", fontsize=8,
                color="#888888", va="bottom")
    ax.legend(frameon=False, fontsize=8)
    ax.text(0.0, -0.17,
            "◯ = the ef_search the concurrency, filtered and churn workloads "
            "were then measured at",
            transform=ax.transAxes, fontsize=8, color="#444444")
    return _save(fig, out_dir, stem)
