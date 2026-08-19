"""vector-bench orchestrator CLI.

Subcommands mirror the stages of a run so each can be repeated in isolation:

    sources   export engine sources at their pinned tags
    build     build the runtime and bench images
    fetch     download datasets
    render    regenerate the ann-benchmarks configs for a profile
    run       execute a benchmark run (the main command)
    report    generate charts and the report from an existing run
    clean     remove containers, networks and volumes left by a run

`run` is resumable: it records completed (engine, dataset, pass, phase) units in
the run directory and skips them on a re-run, so an interruption costs one unit
rather than the whole sweep.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
import time
from typing import Any, Dict, List, Optional

VB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, VB_ROOT)

from harness import datasets as datasets_mod  # noqa: E402
from harness.metrics import sysinfo as sysinfo_mod  # noqa: E402
from orchestrator import ann_pass, docker_ctl, ops_pass  # noqa: E402
from orchestrator.config import (available_profiles, load_engine,  # noqa: E402
                                 load_profile, load_resources,
                                 merge_resource_overrides, resolve_resources)
from orchestrator.manifest import Manifest, new_run_id, utcnow  # noqa: E402

GB = 1024 ** 3

# mariadb123 is opt-in via --engines: it is a second MariaDB version, not
# part of the default three-way comparison.
ALL_ENGINES = ("mariadb", "alisql", "pgvector")
# Opt-in rather than default. mariadb123 is another hour of compiling; mongodb
# is a second server process, a replica set and a JVM, and is a Technical
# Preview besides.
EXTRA_ENGINES = ("mariadb123", "mongodb", "valkey")
KNOWN_ENGINES = ALL_ENGINES + EXTRA_ENGINES


def paths_for(run_id: str) -> Dict[str, str]:
    results = os.path.join(VB_ROOT, "results")
    run_dir = os.path.join(results, run_id)
    return {
        "root": VB_ROOT,
        "sources": os.path.join(VB_ROOT, "sources"),
        "harness": os.path.join(VB_ROOT, "harness"),
        "datasets": os.path.join(VB_ROOT, "datasets"),
        "work_annb": os.path.join(VB_ROOT, "work", "ann-benchmarks"),
        "results": results,
        # Engine data directories and ops volumes. Kept under VB_ROOT so they
        # land on whatever filesystem the checkout is on, rather than on
        # Docker's data-root, which is usually the root volume.
        "engine_state": os.path.join(VB_ROOT, "state"),
        "run_dir": run_dir,
        # ann-benchmarks builds result filenames from the algorithm and its
        # parameters only — not from the resource pass. Sharing one tree across
        # passes would make the tuned pass skip every point the normalized pass
        # already produced, and the report would then present normalized numbers
        # as tuned ones. One tree per pass keeps them separate.
        "annb_results": os.path.join(results, "annb"),
        "ops_results": run_dir,
        "checkpoints": os.path.join(run_dir, "checkpoints.json"),
    }


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def _load_checkpoints(path: str) -> set:
    try:
        with open(path) as fh:
            return set(json.load(fh))
    except (OSError, json.JSONDecodeError):
        return set()


def _save_checkpoint(path: str, key: str) -> None:
    done = _load_checkpoints(path)
    done.add(key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(sorted(done), fh, indent=2)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Subcommands that shell out to the scripts
# ---------------------------------------------------------------------------

def _script(name: str, *args: str) -> int:
    path = os.path.join(VB_ROOT, "scripts", name)
    return subprocess.run([path, *args], check=False).returncode


def cmd_sources(args: argparse.Namespace) -> int:
    return _script("prepare-sources.sh", "--engine", args.engines or "all")


def cmd_build(args: argparse.Namespace) -> int:
    rc = _script("prepare-sources.sh", "--engine", args.engines or "all")
    if rc != 0:
        return rc
    build_args = ["--engine", args.engines or "all", "--target", args.target]
    if args.march:
        build_args += ["--march", args.march]
    if args.jobs:
        build_args += ["--jobs", str(args.jobs)]
    if args.no_cache:
        build_args.append("--no-cache")
    return _script("build-images.sh", *build_args)


def cmd_fetch(args: argparse.Namespace) -> int:
    fetch_args = []
    if args.datasets:
        fetch_args += ["--datasets", args.datasets]
    return _script("fetch-datasets.sh", *fetch_args)


def cmd_render(args: argparse.Namespace) -> int:
    profile = load_profile(args.profile)
    resources = merge_resource_overrides(
        load_resources(args.resource_pass), profile)
    work = paths_for("render")["work_annb"]
    if not os.path.isdir(work):
        print(f"ann-benchmarks working copy missing at {work}; "
              f"run: scripts/prepare-harness.sh", file=sys.stderr)
        return 1
    for engine in (args.engines.split(",") if args.engines else ALL_ENGINES):
        body = ann_pass.render_config(engine, profile, resources, args.resource_pass)
        path = ann_pass.write_config(work, engine, body)
        print(f"rendered {path}")
    return 0


def _dir_size(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path, onerror=lambda _e: None):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


def cmd_clean(args: argparse.Namespace) -> int:
    if not docker_ctl.docker_available():
        print("cannot talk to the Docker daemon", file=sys.stderr)
        return 1

    live = docker_ctl.running_containers()
    if live and not args.force:
        print(f"{len(live)} vector-bench container(s) are still running:",
              file=sys.stderr)
        for name in live[:10]:
            print(f"  {name}", file=sys.stderr)
        print("Cleaning now would destroy a run in progress.\n"
              "Wait for it to finish, or pass --force if you know it is dead.",
              file=sys.stderr)
        return 2

    removed = docker_ctl.cleanup_run(args.run_id or "")
    print("removed: " + ", ".join(f"{n} {kind}(s)" for kind, n in removed.items()))

    # Bind-backed volumes leave their host directory behind when the volume is
    # removed, and the ann path writes its data directory straight onto the
    # host. Neither is reclaimed by docker, so a killed run leaves a full corpus
    # and index on disk with nothing pointing at it.
    # Needs an image: the directories are root-owned and can only be removed
    # from inside a container.
    image = ""
    for engine in KNOWN_ENGINES:
        candidate = load_engine(engine).get("image", {}).get("runtime", "")
        if candidate and docker_ctl.image_exists(candidate):
            image = candidate
            break

    paths = paths_for("clean")
    state = paths["engine_state"]
    if not image and os.path.isdir(state) and os.listdir(state):
        print("no engine image available to remove the root-owned state "
              "directories; build one first, or remove them with sudo",
              file=sys.stderr)
    freed = 0
    if os.path.isdir(state):
        for sub in ("annb", "ops"):
            base = os.path.join(state, sub)
            if not os.path.isdir(base):
                continue
            for name in sorted(os.listdir(base)):
                if args.run_id and args.run_id not in name:
                    continue
                target = os.path.join(base, name)
                size = _dir_size(target)
                if docker_ctl.remove_tree_as_root(target, image):
                    freed += size
                    print(f"  removed state/{sub}/{name} ({size / GB:.1f} GB)")
                else:
                    print(f"  could NOT remove state/{sub}/{name}", file=sys.stderr)
    if freed:
        print(f"reclaimed {freed / GB:.1f} GB of engine data")
    return 0


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> int:
    if not docker_ctl.docker_available():
        print("cannot talk to the Docker daemon", file=sys.stderr)
        return 1

    profile = load_profile(args.profile)
    engines = [e.strip() for e in (args.engines or ",".join(ALL_ENGINES)).split(",") if e.strip()]
    datasets = ([d.strip() for d in args.datasets.split(",") if d.strip()]
                if args.datasets else list(profile.get("datasets", [])))
    # A profile may declare its own default pass. `mariadb-blog` does, because
    # running both passes over a 1536-dim million-vector corpus is ~144 h and
    # nobody chooses that by omission. An explicit --resource-pass always wins.
    requested = args.resource_pass or profile.get("default_resource_pass") or "both"
    passes = (["normalized", "tuned"] if requested == "both" else [requested])
    if args.resource_pass is None and requested != "both":
        print(f"note: profile '{profile.get('name')}' defaults to the "
              f"'{requested}' pass only; override with --resource-pass both")
    phases = (["ann", "ops"] if args.phases == "both" else [args.phases])

    unknown = set(engines) - set(KNOWN_ENGINES)
    if unknown:
        print(f"unknown engines: {sorted(unknown)}", file=sys.stderr)
        return 2

    run_id = args.run_id or new_run_id(profile.get("name", "run"))
    paths = paths_for(run_id)
    os.makedirs(paths["run_dir"], exist_ok=True)
    os.makedirs(paths["annb_results"], exist_ok=True)

    info = sysinfo_mod.collect()
    manifest = Manifest(paths["run_dir"], run_id)
    manifest.set_host(info)
    manifest.set_harness(paths["sources"])

    print(f"=== vector-bench run {run_id} ===")
    print(f"profile   : {profile.get('name')} — {profile.get('description', '')}")
    print(f"engines   : {', '.join(engines)}")
    print(f"datasets  : {', '.join(datasets)}")
    print(f"passes    : {', '.join(passes)}")
    print(f"phases    : {', '.join(phases)}")
    print(f"cpu       : {info.cpu.model} ({info.cpu.logical_cpus} logical, "
          f"hybrid={info.cpu.hybrid}, avx512={info.cpu.has_avx512})")
    print(f"ram       : {info.total_ram_bytes / 1024**3:.1f} GB")
    print(f"results   : {paths['run_dir']}")

    missing = [d for d in datasets
               if not os.path.exists(os.path.join(paths["datasets"], f"{d}.hdf5"))]
    if missing:
        # Not every dataset is fetchable. A few — the dbpedia family among them
        # — are built locally from a source corpus and are not published as
        # prebuilt HDF5, so telling the operator to `fetch` them sends them to a
        # 404.
        fetchable = [d for d in missing if not _is_generated(d)]
        generated = [d for d in missing if _is_generated(d)]
        print(f"\nmissing datasets: {', '.join(missing)}", file=sys.stderr)
        if fetchable:
            print(f"  download:  ./run-benchmark.sh fetch --datasets "
                  f"{','.join(fetchable)}", file=sys.stderr)
        for d in generated:
            print(f"  generate:  ./scripts/generate-dataset.sh {d}\n"
                  f"             (not published as a prebuilt file — it is built "
                  f"from a source corpus,\n"
                  f"              which means a multi-GB download and brute-force "
                  f"ground truth)", file=sys.stderr)
        return 3

    _print_load_estimate(profile, engines, datasets, passes, phases)

    if _check_free_disk(paths, engines, datasets, passes) and not args.force:
        print("  Re-run with --force to start anyway.", file=sys.stderr)
        return 4

    # Always load the file so completed sub-units are recorded against it;
    # only *honour* them when --resume was asked for.
    recorded = _load_checkpoints(paths["checkpoints"])
    checkpoints = recorded if args.resume else set()
    if checkpoints:
        print(f"resuming: {len(checkpoints)} unit(s) already complete")

    failures: List[str] = []

    # What an in-memory engine has to hold, sized from the profile's own
    # dataset rather than assumed. Without this the check fired on every small
    # run: a 20k-row smoke profile was told its 14 GB budget could not hold a
    # 60 MB corpus, which is exactly how a warning stops being read.
    expected_corpus = max(
        (datasets_mod.resident_bytes_estimate(
            d, (profile.get("ops", {}) or {}).get("subset_rows"))
         for d in profile.get("datasets", [])), default=0)

    for resource_pass in passes:
        resources = merge_resource_overrides(
            load_resources(resource_pass), profile)
        if expected_corpus:
            resources.setdefault("memory", {})["expected_corpus_bytes"] = \
                expected_corpus
        for engine in engines:
            engine_cfg = load_engine(engine)
            resolved = resolve_resources(resources, engine, info)
            # Recorded here, where orchestrator code is importable. The report
            # generator runs inside a bench image that mounts only report/ and
            # harness/, so it cannot compute this itself — an earlier version
            # imported orchestrator.ann_pass from the report and crashed the
            # whole run at the last step, after 20 hours of measurement.
            manifest.set_config(
                profile, resource_pass, resolved,
                extra={"ann_fingerprint": ann_pass.ann_fingerprint(resolved)},
            )
            manifest.set_engine(
                engine, paths["sources"],
                engine_cfg.get("image", {}).get("runtime", ""),
                engine_cfg.get("image", {}).get("bench", ""),
                {
                    "runtime": docker_ctl.image_id(
                        engine_cfg.get("image", {}).get("runtime", "")),
                    "bench": docker_ctl.image_id(
                        engine_cfg.get("image", {}).get("bench", "")),
                },
            )
            for warning in resolved.warnings:
                print(f"  ! {warning}")

            for dataset in datasets:
                for phase in phases:
                    key = f"{resource_pass}/{engine}/{dataset}/{phase}"
                    if key in checkpoints:
                        print(f"[skip] {key} (already complete)")
                        continue

                    started = utcnow()
                    print(f"\n--- {key} ---")
                    t0 = time.time()
                    try:
                        rc = _run_unit(phase, engine, dataset, profile, engine_cfg,
                                       resources, resolved, resource_pass, paths,
                                       run_id, args, checkpoints=checkpoints)
                    except Exception as exc:  # noqa: BLE001
                        rc = 1
                        # The traceback matters: these failures are usually in
                        # container plumbing, where the message alone rarely
                        # identifies which step broke.
                        print(f"[error] {key}: {exc}", file=sys.stderr)
                        traceback.print_exc()

                    elapsed = time.time() - t0
                    status = "completed" if rc == 0 else "failed"
                    manifest.add_phase(phase, engine, dataset, status, started,
                                       utcnow(), {"exit_code": rc,
                                                  "duration_s": round(elapsed, 1),
                                                  "resource_pass": resource_pass})
                    if rc == 0:
                        _save_checkpoint(paths["checkpoints"], key)
                        print(f"[ok] {key} in {elapsed / 60:.1f} min")
                    else:
                        failures.append(key)
                        print(f"[FAIL] {key} (exit {rc}) after {elapsed / 60:.1f} min",
                              file=sys.stderr)
                        if args.fail_fast:
                            manifest.finish("failed")
                            return 1

    # Result files were written by root inside the containers; hand them back.
    for engine in engines:
        image = load_engine(engine).get("image", {}).get("bench", "")
        if image and docker_ctl.image_exists(image):
            ann_pass.fix_ownership(paths["results"], image)
            break

    manifest.finish("completed" if not failures else "completed_with_failures")

    print(f"\n=== run {run_id} finished ===")
    if failures:
        print(f"{len(failures)} unit(s) failed:")
        for f in failures:
            print(f"  - {f}")
        print("re-run with --resume to retry only the failures")

    if not args.no_report:
        print("\ngenerating report…")
        if generate_report(paths, engines) != 0:
            print("report generation failed", file=sys.stderr)

    return 0 if not failures else 1


def _ops_storage_engines(engine: str, profile: Dict[str, Any],
                         resources: Dict[str, Any],
                         resource_pass: str) -> List[str]:
    """Storage engines the ops phase should measure build cost on.

    The recall path has swept this since the tuned pass existed; the ops path
    never did, so build cost was only ever measured on InnoDB. That mattered:
    MariaDB's published benchmark reports an index build under 15 minutes, we
    measured 3.6 hours on InnoDB, and the recall curves show MyISAM is almost
    certainly what the article used. Build cost on the engine nobody benchmarks
    is not a reproduction.

    VIDX is InnoDB-only and PostgreSQL has no equivalent knob, so this is a
    MariaDB-only axis.
    """
    if engine not in ann_pass.MARIADB_ENGINES:
        return ["InnoDB"]
    extras = (resources.get("extras", {}) or {}) if resource_pass == "tuned" else {}
    return list(
        extras.get("mariadb_storage_engines")
        or (profile.get("ops", {}) or {}).get("storage_engines")
        or ["InnoDB"]
    )


def _read_manifest_config(run_dir: str) -> Dict[str, Any]:
    """The run's `config` block, or an empty dict if it cannot be read."""
    try:
        with open(os.path.join(run_dir, "run-manifest.json")) as fh:
            return json.load(fh).get("config") or {}
    except (OSError, ValueError):
        return {}


def generate_report(paths: Dict[str, str], engines: List[str]) -> int:
    """Run the report generator inside a bench image.

    The generator needs numpy, h5py and matplotlib. Those already exist in every
    bench image, so running it there keeps the host's requirements at python3 and
    PyYAML — a benchmark framework should not have to install a scientific Python
    stack onto the machine it is measuring.
    """
    image = ""
    for engine in list(engines) + list(KNOWN_ENGINES):
        candidate = load_engine(engine).get("image", {}).get("bench", "")
        if candidate and docker_ctl.image_exists(candidate):
            image = candidate
            break
    if not image:
        print("no bench image available to run the report generator; "
              "build one first: ./run-benchmark.sh build", file=sys.stderr)
        return 1

    # Narrow the ann tree on the host, where the orchestrator is importable.
    # The report container mounts report/ and harness/ only, so it cannot work
    # this out for itself, and reading the whole tree would mix results from
    # every resource configuration ever measured on this machine into one
    # report. Older manifests predate the recorded fingerprint, so fall back to
    # recomputing it from the resources they did record.
    annb_mount = "/results/annb"
    cfg = _read_manifest_config(paths["run_dir"])
    pass_name = cfg.get("resource_pass")
    fingerprint = cfg.get("ann_fingerprint")
    if pass_name and not fingerprint and cfg.get("resolved_resources"):
        fingerprint = ann_pass.ann_fingerprint(cfg["resolved_resources"])
    if pass_name and fingerprint:
        candidate = os.path.join(paths["annb_results"], pass_name, fingerprint)
        if os.path.isdir(candidate):
            annb_mount = f"/results/annb/{pass_name}/{fingerprint}"
            print(f"report: ann results from {pass_name}/{fingerprint}")
        else:
            print(f"report: no ann tree at {pass_name}/{fingerprint}; "
                  f"reading everything under results/annb. Anything older than "
                  f"this run will be flagged in the report's Validity section.")

    spec = docker_ctl.ContainerSpec(
        name=f"vb-report-{os.getpid()}",
        image=image,
        network="none",
        entrypoint="python3",
        workdir="/vb",
        env={"PYTHONUNBUFFERED": "1", "PYTHONPATH": "/vb"},
        volumes=[
            f"{os.path.join(VB_ROOT, 'report')}:/vb/report:ro",
            f"{os.path.join(VB_ROOT, 'harness')}:/vb/harness:ro",
            f"{paths['results']}:/results:rw",
            f"{paths['datasets']}:/datasets:ro",
        ],
        command=[
            "/vb/report/generate.py",
            "--run-dir", f"/results/{os.path.basename(paths['run_dir'])}",
            "--annb-results", annb_mount,
            "--datasets-dir", "/datasets",
        ],
        detach=False,
    )
    rc = docker_ctl.run_foreground(spec, timeout=3600)
    if rc == 0:
        ann_pass.fix_ownership(paths["run_dir"], image)
        print(f"report: {os.path.join(paths['run_dir'], 'report', 'report.html')}")
    return rc


# Measured ingest rates, rows/s, on a Xeon Gold 6230.
#
# These were originally guessed at 400/150/5600 from a 60k-row smoke run. A
# real 1.18M-row load measured MariaDB at 147 rows/s average — 2.7x slower —
# because HNSW insert cost rises as the graph grows and rises again with M
# (glove-100: ~320 rows/s at M=8, ~70 at M=32). An estimate that is optimistic
# by 3x is worse than none, so these are the observed mid-grid numbers.
_INGEST_ROWS_PER_S = {"mariadb": 150, "alisql": 55, "pgvector": 3000}
_DATASET_ROWS = {
    "fashion-mnist-784-euclidean": 60_000,
    "glove-100-angular": 1_183_514,
    "sift-128-euclidean": 1_000_000,
    "gist-960-euclidean": 1_000_000,
    "glove-25-angular": 1_183_514,
    "deep-image-96-angular": 9_990_000,
    # MariaDB's big-vector benchmark corpus. At 1536 dimensions the per-row
    # cost is far above the others, so the row count alone understates it —
    # see _DIM_PENALTY below.
    "dbpedia-openai-1000k-angular": 1_000_000,
    "dbpedia-openai-500k-angular": 500_000,
    "dbpedia-openai-100k-angular": 100_000,
}

# Ingest rate falls with dimensionality: every graph traversal during insert
# costs distance computations proportional to the vector width. The reference
# rates above were measured on 100-dim data, so wider datasets are scaled down.
# Sublinear because SIMD amortises part of it.
_REFERENCE_DIMS = 100
# The reference rates are for a ~1M-row corpus; smaller ones ingest faster.
_REFERENCE_ROWS = 1_000_000
_DATASET_DIMS = {
    "fashion-mnist-784-euclidean": 784,
    "glove-100-angular": 100,
    "sift-128-euclidean": 128,
    "gist-960-euclidean": 960,
    "glove-25-angular": 25,
    "deep-image-96-angular": 96,
    "dbpedia-openai-1000k-angular": 1536,
    "dbpedia-openai-500k-angular": 1536,
    "dbpedia-openai-100k-angular": 1536,
}


# Rates measured directly, per (engine, dataset), on a Xeon Gold 6230. These
# beat any model and are used whenever available.
_MEASURED_ROWS_PER_S = {
    ("mariadb", "fashion-mnist-784-euclidean"): 745,
    ("alisql", "fashion-mnist-784-euclidean"): 150,
    ("pgvector", "fashion-mnist-784-euclidean"): 5655,
    ("mariadb", "glove-100-angular"): 147,
    # Measured on dbpedia-openai-1000k (990k x 1536, M=16). The formula-based
    # fallback predicted ~19 h for MariaDB and ~52 h for AliSQL against actuals
    # of 3.3 h and 6.6 h, so it was 6-8x pessimistic at this dimensionality:
    # _dim_penalty overcharges for width once the per-row cost is dominated by
    # graph traversal rather than by the distance computation itself.
    ("mariadb", "dbpedia-openai-1000k-angular"): 84,
    ("alisql", "dbpedia-openai-1000k-angular"): 41,
}


def _dim_penalty(dataset: str) -> float:
    dims = _DATASET_DIMS.get(dataset, _REFERENCE_DIMS)
    return max(1.0, (dims / _REFERENCE_DIMS) ** 0.6)


def _size_penalty(dataset: str) -> float:
    """How much slower ingest is on this corpus because the graph is bigger.

    HNSW insert cost grows with the number of nodes already indexed, and on
    these engines that dominates dimensionality outright. Measured on one
    machine with MariaDB: 60k rows at 784 dims ran at 745 rows/s while 1.18M
    rows at 100 dims ran at 147 — twenty times the rows and nearly eight times
    *fewer* dimensions, and still five times slower.

    An earlier version scaled only by dimensionality, using a rate measured at
    1.18M rows, and was therefore ~17x too pessimistic on a 60k dataset. The
    reference rates below correspond to a 1M-row corpus.
    """
    rows = _DATASET_ROWS.get(dataset, _REFERENCE_ROWS)
    return max(0.05, (rows / _REFERENCE_ROWS) ** 0.55)


def _effective_rate(engine: str, dataset: str) -> float:
    """Rows/s for this engine on this corpus: measured if known, else modelled."""
    measured = _MEASURED_ROWS_PER_S.get((engine, dataset))
    if measured:
        return float(measured)
    base = _INGEST_ROWS_PER_S.get(engine, 300)
    return base / (_dim_penalty(dataset) * _size_penalty(dataset))


# Datasets ann-benchmarks constructs locally rather than publishing as HDF5.
# `fetch` cannot retrieve these; scripts/generate-dataset.sh builds them.
_GENERATED_DATASET_PREFIXES = ("dbpedia-openai-",)


def _is_generated(dataset: str) -> bool:
    return dataset.startswith(_GENERATED_DATASET_PREFIXES)



# What one engine leaves on disk for one dataset, as a multiple of the corpus
# file. The table and the index each roughly match the raw vectors, and the
# MySQL family also writes a redo log and a doublewrite buffer. Measured on
# dbpedia-openai-1000k: a 6.2 GB corpus produced a 7.7 GB table and a 3.9 GB
# index per engine.
_DISK_PER_ENGINE = 2.2
_DISK_HEADROOM_GB = 10.0


def _check_free_disk(paths: Dict[str, Any], engines: List[str],
                     datasets: List[str], passes: List[str]) -> bool:
    """Warn before a run that cannot fit. Returns True if it looks too tight.

    A pgvector phase once died two seconds in because the volume had nowhere to
    put the cluster, and nothing in the run said so — it looked like an engine
    failure for a while. Cheap to check, expensive to discover eight hours in.
    """
    # Two filesystems matter and they are frequently different. Engine data
    # goes under VB_ROOT now, but images and any volume not created by this
    # harness still live under Docker's data-root, which on a default install
    # is the root volume. Checking only the results directory is how a run
    # passed preflight and then died at "No space left on device".
    targets = {paths.get("results") or paths.get("run_dir") or "."}
    root = docker_ctl.root_dir()
    if root:
        targets.add(root)

    usage, target = None, None
    for candidate in sorted(targets):
        try:
            u = shutil.disk_usage(candidate)
        except OSError:
            continue
        if usage is None or u.free < usage.free:
            usage, target = u, candidate
    if usage is None:
        return False

    corpus_bytes = 0
    for d in datasets:
        path = os.path.join(paths["datasets"], f"{d}.hdf5")
        try:
            corpus_bytes += os.path.getsize(path)
        except OSError:
            from harness.datasets import KNOWN_DATASETS
            corpus_bytes += int(KNOWN_DATASETS.get(d, {}).get("approx_bytes", 0) or 0)

    # Engines run one at a time and their volumes are torn down after each, so
    # the peak is one engine's footprint rather than the sum. Passes reuse the
    # same space for the same reason.
    need = corpus_bytes * _DISK_PER_ENGINE + _DISK_HEADROOM_GB * GB
    free = usage.free

    checked = " and ".join(sorted(targets))
    print(f"disk: {free / GB:.0f} GB free (tightest of {checked}), "
          f"run needs about {need / GB:.0f} GB "
          f"({len(engines)} engine(s), largest corpus footprint plus headroom)")
    if free >= need:
        return False

    print(
        f"\n! Not enough free disk. {free / GB:.0f} GB available, "
        f"about {need / GB:.0f} GB needed.\n"
        f"  An engine that cannot write its data directory fails within seconds "
        f"and looks like an engine fault, not a disk fault.\n"
        f"  Free space, or point VB_ROOT at a larger filesystem.",
        file=sys.stderr,
    )
    return True

def _print_load_estimate(profile: Dict[str, Any], engines: List[str],
                         datasets: List[str], passes: List[str],
                         phases: List[str]) -> None:
    """Estimate ingest time before the run starts.

    ann-benchmarks reloads the whole dataset for every M value, and the
    incrementally-building engines ingest at a few hundred rows per second. That
    multiplies quietly: a seven-value M grid over four datasets is days of pure
    loading. Showing the number up front is the difference between choosing that
    and discovering it six hours in.
    """
    m_count = max(1, len(profile.get("ann", {}).get("m_values", [16])))
    total_h = 0.0
    rows_per_pass: List[str] = []

    for engine in engines:
        engine_h = 0.0
        for dataset in datasets:
            rows = _DATASET_ROWS.get(dataset)
            if not rows:
                continue
            effective = _effective_rate(engine, dataset)
            if "ann" in phases:
                engine_h += (rows / effective) * m_count / 3600
            if "ops" in phases:
                ops_m = max(1, len(profile.get("ops", {}).get("m_values", [16])))
                engine_h += (rows / effective) * ops_m / 3600
        engine_h *= len(passes)
        total_h += engine_h
        rows_per_pass.append(f"{engine} ~{engine_h:.1f} h")

    if total_h < 1:
        return
    print(f"\nestimated ingest time (loading only, before any queries):")
    print("  " + "  |  ".join(rows_per_pass))
    print(f"  total ~{total_h:.1f} h across {len(passes)} pass(es), "
          f"{m_count} M value(s)")
    if total_h > 12:
        print(f"  ! This is a long run. Each M value costs a full reload of every "
              f"dataset,\n    and MHNSW/VIDX build incrementally at a few hundred "
              f"rows/s. Reduce\n    ann.m_values or the dataset list to cut it "
              f"roughly proportionally.")
    print()


def _run_unit(phase: str, engine: str, dataset: str, profile: Dict[str, Any],
              engine_cfg: Dict[str, Any], resources: Dict[str, Any],
              resolved: Any, resource_pass: str, paths: Dict[str, str],
              run_id: str, args: argparse.Namespace,
              checkpoints: Optional[set] = None) -> int:
    if phase == "ann":
        if not profile.get("ann", {}).get("enabled", True):
            print("[ann] disabled in profile; skipping")
            return 0
        body = ann_pass.render_config(engine, profile, resources, resource_pass)
        ann_pass.write_config(paths["work_annb"], engine, body)
        return ann_pass.run_engine(
            engine, dataset, profile, engine_cfg, resolved, resource_pass,
            paths, run_id, force=args.force,
        )

    if phase == "ops":
        ops_cfg = profile.get("ops", {}) or {}
        if not ops_cfg.get("enabled", True):
            print("[ops] disabled in profile; skipping")
            return 0

        m_values = list(ops_cfg.get("m_values", [16]))
        build_modes = (["post"] if engine != "pgvector"
                       else list(profile.get("ann", {}).get("pgvector_build_modes", ["post"])))
        storage_engines = _ops_storage_engines(engine, profile, resources,
                                               resource_pass)
        worst_rc = 0
        for m in m_values:
            for build_mode in build_modes:
                for storage in storage_engines:
                    # InnoDB stays unsuffixed so checkpoints written by earlier
                    # runs are still honoured; only the extra curves get a new
                    # tag.
                    tag = f"m{m}-{build_mode}"
                    if storage.lower() != "innodb":
                        tag += f"-{storage.lower()}"

                    # Checkpoint per unit, not per phase. One ops phase can be a
                    # dozen hours spread over several M values and storage
                    # engines, and each is independent work: recording only the
                    # whole phase meant an interruption threw away everything
                    # that had already completed.
                    sub_key = f"{resource_pass}/{engine}/{dataset}/ops/{tag}"
                    if checkpoints is not None and sub_key in checkpoints:
                        print(f"[skip] {sub_key} (already complete)")
                        continue

                    stem = f"{engine}-{dataset}-{resource_pass}-{tag}"
                    output = os.path.join(paths["run_dir"], f"ops-{stem}.jsonl")
                    memory_ts = os.path.join(paths["run_dir"], f"mem-{stem}.jsonl")
                    harness_args = ops_pass.harness_args(
                        profile, m, engine, resolved, resource_pass, resources,
                        build_mode=build_mode, storage_engine=storage,
                    )
                    with ops_pass.OpsRun(engine, engine_cfg, resolved,
                                         resource_pass, paths, run_id,
                                         dataset, tag) as run:
                        rc = run.run_harness(harness_args, output,
                                             memory_timeseries=memory_ts)
                    if rc == 0 and checkpoints is not None:
                        _save_checkpoint(paths["checkpoints"], sub_key)
                        checkpoints.add(sub_key)
                    worst_rc = worst_rc or rc
        return worst_rc

    raise ValueError(f"unknown phase: {phase}")


def cmd_report(args: argparse.Namespace) -> int:
    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        candidate = os.path.join(VB_ROOT, "results", args.run_dir)
        if os.path.isdir(candidate):
            run_dir = candidate
        else:
            print(f"run directory not found: {args.run_dir}", file=sys.stderr)
            return 1
    paths = paths_for(os.path.basename(run_dir))
    paths["run_dir"] = run_dir
    return generate_report(paths, list(KNOWN_ENGINES))


# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run-benchmark.sh",
        description="vector-bench — MariaDB vs AliSQL vs PostgreSQL/pgvector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"available profiles: {', '.join(available_profiles())}",
    )
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("sources", help="export engine sources at pinned tags")
    s.add_argument("--engines", default="all")
    s.set_defaults(func=cmd_sources)

    b = sub.add_parser("build", help="build engine images")
    b.add_argument("--engines", default="all")
    b.add_argument("--target", default="all", choices=("all", "runtime", "bench"))
    b.add_argument("--march", default=None,
                   help="SIMD baseline for every engine, e.g. x86-64-v3 or native")
    b.add_argument("--jobs", type=int, default=0)
    b.add_argument("--no-cache", action="store_true")
    b.set_defaults(func=cmd_build)

    f = sub.add_parser("fetch", help="download datasets")
    f.add_argument("--datasets", default=None)
    f.set_defaults(func=cmd_fetch)

    r = sub.add_parser("render", help="regenerate ann-benchmarks configs")
    r.add_argument("--profile", default="quick")
    r.add_argument("--engines", default=None)
    r.add_argument("--resource-pass", default="normalized",
                   choices=("normalized", "tuned"))
    r.set_defaults(func=cmd_render)

    run = sub.add_parser("run", help="execute a benchmark run")
    run.add_argument("--profile", default="quick",
                     help=f"one of: {', '.join(available_profiles())}")
    run.add_argument("--engines", default=None,
                     help="comma-separated subset of mariadb,alisql,pgvector")
    run.add_argument("--datasets", default=None,
                     help="override the profile's dataset list")
    # No argparse default: an unset value lets the profile choose, and falls
    # back to "both" only when the profile is silent.
    run.add_argument("--resource-pass", default=None,
                     choices=("normalized", "tuned", "both"),
                     help="normalized, tuned, or both "
                          "(default: the profile's choice, else both)")
    run.add_argument("--phases", default="both", choices=("ann", "ops", "both"))
    run.add_argument("--run-id", default=None)
    run.add_argument("--resume", action="store_true",
                     help="skip units already recorded complete in the run dir")
    run.add_argument("--force", action="store_true",
                     help="re-run ann-benchmarks points that already have results")
    run.add_argument("--fail-fast", action="store_true")
    run.add_argument("--no-report", action="store_true")
    run.set_defaults(func=cmd_run)

    rep = sub.add_parser("report", help="generate the report for an existing run")
    rep.add_argument("--run-dir", required=True)
    rep.set_defaults(func=cmd_report)

    c = sub.add_parser("clean", help="remove docker resources left by a run")
    c.add_argument("--run-id", default=None,
                   help="limit cleanup to one run (default: everything)")
    c.add_argument("--force", action="store_true",
                   help="clean even while vector-bench containers are running")
    c.set_defaults(func=cmd_clean)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
