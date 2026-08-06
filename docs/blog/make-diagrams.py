#!/usr/bin/env python3
"""Generate the diagrams for the blog post.

Writes SVG (for the repo and the web) and PNG (for the .docx, since pandoc
cannot rasterise SVG without librsvg) into docs/blog/img/.

Everything is deterministic: fixed RandomState, no wall-clock, no sampling at
import time. Re-running produces byte-identical output, so regenerating a figure
does not create noise in the diff.

Usage:  python3 docs/blog/make-diagrams.py
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img")

INK = "#1f2328"      # text and strong strokes
MUTED = "#8b949e"    # inactive points, secondary edges
ACCENT = "#0969da"   # the thing being explained
WARN = "#cf222e"     # misses, failures
GOOD = "#1a7f37"     # hits
FILL = "#ddf4ff"     # highlighted region


def _finish(fig, name: str) -> None:
    os.makedirs(OUT, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), format=ext,
                    bbox_inches="tight", dpi=170,
                    facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  {name}.svg  {name}.png")


def _bare(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_aspect("equal")


# ---------------------------------------------------------------- recall

def recall() -> None:
    """What recall@10 counts: overlap between the true top-k and what came back."""
    rs = np.random.RandomState(7)
    pts = rs.uniform(0, 10, size=(70, 2))
    q = np.array([5.0, 5.0])

    d = np.linalg.norm(pts - q, axis=1)
    order = np.argsort(d)
    true10 = order[:10]
    # The index found 8 of the true 10 and substituted two slightly worse rows.
    returned = list(true10[:8]) + [order[15], order[21]]
    missed = [i for i in true10 if i not in returned]

    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    radius = d[true10[-1]] + 0.15
    ax.add_patch(Circle(q, radius, facecolor=FILL, edgecolor=ACCENT,
                        lw=1.2, ls="--", zorder=0))

    far = [i for i in range(len(pts)) if i not in true10 and i not in returned]
    ax.scatter(pts[far, 0], pts[far, 1], s=26, c=MUTED, zorder=2)

    hits = [i for i in returned if i in true10]
    ax.scatter(pts[hits, 0], pts[hits, 1], s=95, c=GOOD, zorder=4)
    ax.scatter(pts[missed, 0], pts[missed, 1], s=95, facecolors="none",
               edgecolors=WARN, lw=2.0, zorder=4)
    wrong = [i for i in returned if i not in true10]
    ax.scatter(pts[wrong, 0], pts[wrong, 1], s=95, c=WARN, marker="X", zorder=4)

    ax.scatter([q[0]], [q[1]], s=190, c=INK, marker="*", zorder=5)
    ax.annotate("query vector", q, textcoords="offset points", xytext=(12, 10),
                fontsize=9, color=INK)
    ax.annotate("true top 10", (q[0], q[1] + radius), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=9, color=ACCENT)

    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color=GOOD, markersize=9,
               label="returned, and correct  (8)"),
        Line2D([], [], marker="o", ls="", markerfacecolor="none",
               markeredgecolor=WARN, markeredgewidth=2, markersize=9,
               label="in the true top 10, missed  (2)"),
        Line2D([], [], marker="X", ls="", color=WARN, markersize=9,
               label="returned, but not in the top 10  (2)"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.02), frameon=False,
        fontsize=9, handletextpad=0.4)

    ax.set_title("recall@10 = 8 / 10 = 0.8", fontsize=12, color=INK, pad=12)
    _bare(ax)
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 10.5)
    _finish(fig, "recall")


# ------------------------------------------------------------------ HNSW

def hnsw() -> None:
    """The layered graph, and the greedy walk descending through it."""
    rs = np.random.RandomState(3)
    base = rs.uniform(0, 10, size=(46, 2))

    # Layers are nested, which is what lets the walk continue from where the
    # layer above left off.
    l0 = np.arange(len(base))
    l1 = np.sort(rs.choice(len(base), 18, replace=False))
    l2 = np.sort(rs.choice(l1, 6, replace=False))

    # Put the query where the sparse top layer can only get it to the right
    # region, leaving the denser layers something to refine. Picking it at
    # random tends to land next to a top-layer node, which makes the lower
    # panels look pointless.
    outer = [int(i) for i in l0 if i not in set(int(x) for x in l1)]
    far = max(outer, key=lambda i: np.linalg.norm(base[l2] - base[i], axis=1).min())
    q = base[far] + np.array([0.28, -0.22])
    layers = [l2, l1, l0]
    degree = [3, 3, 4]

    def edges_of(idx, k):
        pts = base[idx]
        out = {int(i): set() for i in idx}
        for i in range(len(idx)):
            d = np.linalg.norm(pts - pts[i], axis=1)
            for j in np.argsort(d)[1:k + 1]:
                out[int(idx[i])].add(int(idx[j]))
                out[int(idx[j])].add(int(idx[i]))
        return {a: sorted(b) for a, b in out.items()}

    def walk(links, start):
        """Greedy: hop to the closest linked node until none is closer."""
        path, cur = [start], start
        while True:
            best = min(links[cur], key=lambda j: np.linalg.norm(base[j] - q))
            if np.linalg.norm(base[best] - q) < np.linalg.norm(base[cur] - q):
                path.append(best)
                cur = best
            else:
                return path

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 4.3))
    titles = ["top layer\nfew nodes, long links",
              "middle layer\nmore nodes, shorter links",
              "bottom layer\nevery node, local links"]

    entry = int(l2[np.argmax(np.linalg.norm(base[l2] - q, axis=1))])
    start = entry
    for ax, idx, k, title in zip(axes, layers, degree, titles):
        links = edges_of(idx, k)
        for a, nbrs in links.items():
            for b in nbrs:
                ax.plot(base[[a, b], 0], base[[a, b], 1], color=MUTED,
                        lw=0.8, alpha=0.85, zorder=1)
        ax.scatter(base[idx, 0], base[idx, 1], s=30, c=MUTED, zorder=2)

        path = walk(links, start)
        for a, b in zip(path, path[1:]):
            ax.add_patch(FancyArrowPatch(base[a], base[b], arrowstyle="-|>",
                                         mutation_scale=13, color=ACCENT,
                                         lw=2.0, zorder=4,
                                         connectionstyle="arc3,rad=0.14"))
        ax.scatter(base[path, 0], base[path, 1], s=80, c=ACCENT, zorder=5)
        ax.scatter([q[0]], [q[1]], s=170, c=INK, marker="*", zorder=6)

        if idx is layers[0]:
            ax.annotate("entry point", base[entry], textcoords="offset points",
                        xytext=(10, -14), fontsize=10.5, color=ACCENT)
            # Label the query here too, otherwise the star reads as a stray mark
            # in the panel where the top layer has no nodes near it.
            ax.annotate("query vector", q, textcoords="offset points",
                        xytext=(0, 14), ha="center", fontsize=10.5, color=INK)
        if idx is layers[-1]:
            ax.annotate("query vector", q, textcoords="offset points",
                        xytext=(8, -16), fontsize=10.5, color=INK)
            ax.annotate("neighbours returned\nfrom around here", base[path[-1]],
                        textcoords="offset points", xytext=(-58, 24),
                        fontsize=10.5, color=ACCENT, ha="center")

        start = path[-1]
        ax.set_title(title, fontsize=12, color=INK, pad=8)
        _bare(ax)
        ax.set_xlim(-0.8, 10.8)
        ax.set_ylim(-0.8, 10.8)

    for a, b in ((axes[0], axes[1]), (axes[1], axes[2])):
        pa, pb = a.get_position(), b.get_position()
        fig.add_artist(FancyArrowPatch(
            (pa.x1 + 0.004, 0.5), (pb.x0 - 0.004, 0.5),
            transform=fig.transFigure, arrowstyle="-|>", mutation_scale=14,
            color=ACCENT, lw=1.8))

    fig.suptitle("HNSW: the search drops a layer when no neighbour is closer",
                 fontsize=14.5, color=INK, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _finish(fig, "hnsw")


# ------------------------------------------------------------------- IVF

def ivf() -> None:
    """Clusters built once, a couple of them searched per query."""
    rs = np.random.RandomState(11)
    centres = np.array([[2, 8], [5.5, 8.6], [8.6, 7.4], [1.8, 5], [5, 4.8],
                        [8.4, 4.2], [2.6, 1.6], [6.4, 1.4]], dtype=float)
    pts, owner = [], []
    for i, c in enumerate(centres):
        n = 13
        pts.append(c + rs.normal(0, 0.72, size=(n, 2)))
        owner += [i] * n
    pts = np.vstack(pts)
    owner = np.array(owner)

    q = np.array([6.9, 3.2])
    probed = np.argsort(np.linalg.norm(centres - q, axis=1))[:2]

    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    for i, c in enumerate(centres):
        sel = owner == i
        on = i in probed
        if on:
            r = np.linalg.norm(pts[sel] - c, axis=1).max() + 0.35
            ax.add_patch(Circle(c, r, facecolor=FILL, edgecolor=ACCENT,
                                lw=1.2, ls="--", zorder=0))
        ax.scatter(pts[sel, 0], pts[sel, 1], s=24,
                   c=(ACCENT if on else MUTED), zorder=2)
        ax.scatter(*c, s=110, marker="s", zorder=3,
                   c=(ACCENT if on else MUTED),
                   edgecolors="white", linewidths=1.2)

    ax.scatter([q[0]], [q[1]], s=190, c=INK, marker="*", zorder=5)
    ax.annotate("query vector", q, textcoords="offset points", xytext=(14, 8),
                fontsize=9, color=INK)
    ax.annotate("one representative\nvector per cluster", centres[0],
                textcoords="offset points", xytext=(-6, 38), fontsize=8.5,
                color=MUTED, ha="center",
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.0))
    ax.annotate("only these two\nare searched", centres[probed[0]],
                textcoords="offset points", xytext=(58, 46), fontsize=9,
                color=ACCENT, ha="center",
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.2))

    ax.set_title("IVF: nlist = 8 clusters, nprobe = 2 searched per query",
                 fontsize=12, color=INK, pad=10)
    _bare(ax)
    ax.set_xlim(-0.8, 11.2)
    ax.set_ylim(-1.0, 11.0)
    _finish(fig, "ivf")


# ------------------------------------------------- recall vs throughput

def tradeoff() -> None:
    """Left: the measured sweep. Right: why two curves crossing is an answer."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.3))

    # Measured: one index, ef_search varied, everything else held constant.
    ef = [10, 40, 120, 800]
    rec = [0.9593, 0.9921, 0.9969, 0.9987]
    qps = [3678, 2431, 1445, 409]

    ax1.plot(rec, qps, "-o", color=ACCENT, lw=2, markersize=7, zorder=3)
    for e, r, q in zip(ef, rec, qps):
        # The first point sits against the y axis, so its label goes right.
        dx, ha = ((12, "left") if e == ef[0] else (-8, "right"))
        ax1.annotate(f"ef_search={e}", (r, q), textcoords="offset points",
                     xytext=(dx, 12), fontsize=8.5, color=INK, ha=ha)
    ax1.set_title("one index, one setting changed", fontsize=11, color=INK)
    ax1.set_xlabel("recall@10")
    ax1.set_ylabel("queries / sec")
    ax1.set_ylim(0, 4300)
    ax1.grid(alpha=0.25, lw=0.6)

    # Illustration only: no engine is named because this is a shape, not a result.
    r = np.linspace(0.90, 0.999, 60)
    a = 3600 * (1 - r) ** 0.55 / (1 - 0.90) ** 0.55
    b = 2000 * (1 - r) ** 0.22 / (1 - 0.90) ** 0.22
    ax2.plot(r, a, color=ACCENT, lw=2, label="engine A")
    ax2.plot(r, b, color=WARN, lw=2, label="engine B")

    idx = int(np.argmin(np.abs(a - b)))
    ax2.axvline(r[idx], color=MUTED, ls=":", lw=1.2)
    ax2.annotate("below this recall A is faster,\nabove it B is",
                 (r[idx], a[idx]), textcoords="offset points",
                 xytext=(-150, 108), fontsize=9, color=INK,
                 arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.1))
    ax2.set_title("crossing curves (illustration)", fontsize=11, color=INK)
    ax2.set_xlabel("recall@10")
    ax2.set_ylabel("queries / sec")
    ax2.legend(frameon=False, fontsize=9, loc="upper right")
    ax2.grid(alpha=0.25, lw=0.6)

    for ax in (ax1, ax2):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.tight_layout()
    _finish(fig, "tradeoff")


# --------------------------------------------------------------- harness

def harness() -> None:
    """Why the server sits alone in its container."""
    fig, ax = plt.subplots(figsize=(9.2, 3.1))

    def box(x, y, w, h, title, lines, colour):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.14",
                                    linewidth=1.6, edgecolor=colour,
                                    facecolor="white", zorder=2))
        ax.text(x + w / 2, y + h - 0.34, title, ha="center", fontsize=10.5,
                color=colour, weight="bold", zorder=3)
        for i, ln in enumerate(lines):
            ax.text(x + w / 2, y + h - 0.78 - i * 0.34, ln, ha="center",
                    fontsize=8.8, color=INK, zorder=3)

    box(0.3, 0.5, 3.5, 2.2, "server container", [
        "the database, alone", "pinned cpuset", "hard memory limit",
        "peak RSS measured here",
    ], ACCENT)
    box(5.4, 0.5, 3.5, 2.2, "client container", [
        "the benchmark harness", "holds the dataset in RAM",
        "separate cores", "its memory is not counted",
    ], MUTED)

    ax.add_patch(FancyArrowPatch((3.95, 1.6), (5.25, 1.6), arrowstyle="<|-|>",
                                 mutation_scale=15, color=INK, lw=1.5))
    ax.text(4.6, 2.02, "private\nnetwork", ha="center", fontsize=9, color=INK)
    ax.text(4.6, 1.18, "queries, loads,\nEXPLAIN checks", ha="center",
            fontsize=8.4, color=MUTED)

    ax.text(4.6, 0.06,
            "Sharing one container would charge several GB of dataset arrays "
            "to the database's memory accounting.",
            ha="center", fontsize=8.8, color=MUTED, style="italic")

    ax.set_xlim(0, 9.2)
    ax.set_ylim(-0.15, 3.0)
    _bare(ax)
    ax.set_aspect("auto")
    _finish(fig, "harness")


if __name__ == "__main__":
    print("writing diagrams to", OUT)
    recall()
    hnsw()
    ivf()
    tradeoff()
    harness()
