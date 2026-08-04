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
import subprocess
import sys
import traceback
import time
from typing import Any, Dict, List, Optional

VB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, VB_ROOT)

from harness.metrics import sysinfo as sysinfo_mod  # noqa: E402
from orchestrator import ann_pass, docker_ctl, ops_pass  # noqa: E402
from orchestrator.config import (available_profiles, load_engine,  # noqa: E402
                                 load_profile, load_resources, resolve_resources)
from orchestrator.manifest import Manifest, new_run_id, utcnow  # noqa: E402

ALL_ENGINES = ("mariadb", "alisql", "pgvector")


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
    resources = load_resources(args.resource_pass)
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


def cmd_clean(args: argparse.Namespace) -> int:
    docker_ctl.cleanup_run(args.run_id or "")
    print("cleaned up docker resources")
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
    passes = (["normalized", "tuned"] if args.resource_pass == "both"
              else [args.resource_pass])
    phases = (["ann", "ops"] if args.phases == "both" else [args.phases])

    unknown = set(engines) - set(ALL_ENGINES)
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
        print(f"\nmissing datasets: {', '.join(missing)}\n"
              f"fetch them first:  ./run-benchmark.sh fetch --datasets {','.join(missing)}",
              file=sys.stderr)
        return 3

    checkpoints = _load_checkpoints(paths["checkpoints"]) if args.resume else set()
    if checkpoints:
        print(f"resuming: {len(checkpoints)} unit(s) already complete")

    failures: List[str] = []

    for resource_pass in passes:
        resources = load_resources(resource_pass)
        for engine in engines:
            engine_cfg = load_engine(engine)
            resolved = resolve_resources(resources, engine, info)
            manifest.set_config(profile, resource_pass, resolved)
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
                                       run_id, args)
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


def generate_report(paths: Dict[str, str], engines: List[str]) -> int:
    """Run the report generator inside a bench image.

    The generator needs numpy, h5py and matplotlib. Those already exist in every
    bench image, so running it there keeps the host's requirements at python3 and
    PyYAML — a benchmark framework should not have to install a scientific Python
    stack onto the machine it is measuring.
    """
    image = ""
    for engine in list(engines) + list(ALL_ENGINES):
        candidate = load_engine(engine).get("image", {}).get("bench", "")
        if candidate and docker_ctl.image_exists(candidate):
            image = candidate
            break
    if not image:
        print("no bench image available to run the report generator; "
              "build one first: ./run-benchmark.sh build", file=sys.stderr)
        return 1

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
            "--annb-results", "/results/annb",
            "--datasets-dir", "/datasets",
        ],
        detach=False,
    )
    rc = docker_ctl.run_foreground(spec, timeout=3600)
    if rc == 0:
        ann_pass.fix_ownership(paths["run_dir"], image)
        print(f"report: {os.path.join(paths['run_dir'], 'report', 'report.html')}")
    return rc


def _run_unit(phase: str, engine: str, dataset: str, profile: Dict[str, Any],
              engine_cfg: Dict[str, Any], resources: Dict[str, Any],
              resolved: Any, resource_pass: str, paths: Dict[str, str],
              run_id: str, args: argparse.Namespace) -> int:
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
        worst_rc = 0
        for m in m_values:
            for build_mode in build_modes:
                tag = f"m{m}-{build_mode}"
                output = os.path.join(paths["run_dir"],
                                      f"ops-{engine}-{dataset}-{resource_pass}-{tag}.jsonl")
                memory_ts = os.path.join(paths["run_dir"],
                                         f"mem-{engine}-{dataset}-{resource_pass}-{tag}.jsonl")
                harness_args = ops_pass.harness_args(
                    profile, m, engine, resolved, resource_pass, resources,
                    build_mode=build_mode,
                )
                with ops_pass.OpsRun(engine, engine_cfg, resolved, resource_pass,
                                     paths, run_id, dataset, tag) as run:
                    rc = run.run_harness(harness_args, output,
                                         memory_timeseries=memory_ts)
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
    return generate_report(paths, list(ALL_ENGINES))


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
    run.add_argument("--resource-pass", default="both",
                     choices=("normalized", "tuned", "both"))
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
    c.add_argument("--run-id", default=None)
    c.set_defaults(func=cmd_clean)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
