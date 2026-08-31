"""What still has to happen before this machine can measure anything.

Status answers "what is the state". This answers "what do I do next", which is
a different question with an order to it: images, then a corpus, then the smoke
gate, then a real run. Each step reports whether it is done and what to press.

The -march check is the reason this is worth computing server-side rather than
in the page. Every engine compiles SIMD distance kernels, so building one with
a different value turns the benchmark into a comparison of compiler flags -- and
nothing about the resulting numbers looks wrong.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Any, Dict, List, Optional

#: Enough of a corpus to prove the pipeline. The smoke profile uses it.
SMOKE_DATASET = "fashion-mnist-784-euclidean"

#: A full run wants this much room; below it, a long run will fail late.
WANTED_DISK_BYTES = 80 * 1024 ** 3


def _image_build(sources_dir: str, engine: str) -> Dict[str, Any]:
    try:
        with open(os.path.join(sources_dir, f"{engine}.image.json")) as fh:
            return json.load(fh) or {}
    except (OSError, ValueError):
        return {}


def engine_state(root: str) -> List[Dict[str, Any]]:
    from orchestrator import docker_ctl
    from orchestrator import engines as engines_mod
    from orchestrator.config import load_engine

    sources_dir = os.path.join(root, "sources")
    out: List[Dict[str, Any]] = []
    for name, engine in engines_mod.registry().items():
        try:
            images = load_engine(name).get("image") or {}
        except FileNotFoundError:
            images = {}
        build = _image_build(sources_dir, name)
        out.append({
            "name": name,
            "label": engine.label,
            "color": engine.color,
            "group": engine.group,
            "tag": engine.tag,
            "runtime_built": bool(images.get("runtime"))
                             and docker_ctl.image_exists(images["runtime"]),
            "bench_built": bool(images.get("bench"))
                           and docker_ctl.image_exists(images["bench"]),
            "march": build.get("march"),
        })
    return out


def march_in_use(engines: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The -march every built image already agrees on, if they do.

    A machine part-way through building is the dangerous state: three images at
    x86-64-v3 and a fourth at native compare compiler flags, not engines.
    """
    values = sorted({e["march"] for e in engines
                     if e["bench_built"] and e.get("march")})
    return {
        "values": values,
        "agreed": values[0] if len(values) == 1 else None,
        "mixed": len(values) > 1,
    }


def _smoke_run(results_dir: str) -> Optional[Dict[str, Any]]:
    from . import runs as runs_mod
    for run in runs_mod.discover_runs(results_dir):
        if run.get("profile") == "smoke" and run.get("status") == "completed":
            return run
    return None


def plan(root: str, results_dir: str, datasets_dir: str) -> Dict[str, Any]:
    """The ordered steps, each with enough state for the page to act on."""
    engines = engine_state(root)
    march = march_in_use(engines)
    built = [e for e in engines if e["bench_built"]]
    original = [e for e in engines if e["group"] == "original"]
    original_built = [e for e in original if e["bench_built"]]

    try:
        present = sorted(f[:-len(".hdf5")] for f in os.listdir(datasets_dir)
                         if f.endswith(".hdf5"))
    except OSError:
        present = []
    # tiny-* are the synthetic corpora the dev profile makes; they prove the
    # framework works and measure nothing about an engine.
    real = [d for d in present if not d.startswith("tiny-")]

    smoke = _smoke_run(results_dir)

    try:
        usage = shutil.disk_usage(root)
        disk = {"free_bytes": usage.free, "total_bytes": usage.total,
                "enough": usage.free >= WANTED_DISK_BYTES,
                "wanted_bytes": WANTED_DISK_BYTES}
    except OSError:
        disk = {}

    steps = [
        {
            "id": "images",
            "title": "Build the engine images",
            "done": len(original_built) == len(original) and bool(original),
            "summary": f"{len(built)} of {len(engines)} engines have a bench image"
                       + (f", all built with -march={march['agreed']}"
                          if march["agreed"] else ""),
            "detail": {"engines": engines, "march": march},
        },
        {
            "id": "datasets",
            "title": "Get a corpus to measure against",
            "done": bool(real),
            "summary": (f"{len(real)} dataset(s) here: {', '.join(real)}"
                        if real else "no datasets downloaded"),
            "detail": {"present": present, "smoke_dataset": SMOKE_DATASET,
                       "smoke_dataset_present": SMOKE_DATASET in present},
        },
        {
            "id": "smoke",
            "title": "Prove the pipeline end to end",
            "done": smoke is not None,
            "summary": (f"smoke run {smoke['dir_name']} completed"
                        if smoke else "not run yet"),
            "detail": {"run": smoke},
        },
        {
            "id": "measure",
            "title": "Measure something",
            "done": False,
            "summary": "the smoke profile is a gate, not a measurement",
            "detail": {},
        },
    ]

    blocked = next((s["id"] for s in steps if not s["done"]), None)
    return {
        "steps": steps,
        "next": blocked,
        "ready": all(s["done"] for s in steps[:3]),
        "disk": disk,
        "march": march,
    }
