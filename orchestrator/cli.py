"""vector-bench orchestrator CLI.

Subcommands mirror the stages of a run so each can be repeated in isolation:

    sources   export engine sources at their pinned tags
    build     build the runtime and bench images
    fetch     download datasets
    render    regenerate the ann-benchmarks configs for a profile
    run       execute a benchmark run (the main command)
    report    generate charts and the report from an existing run
    clean     remove containers, networks and volumes left by a run
    lab       run one script against one engine, recording nothing

`run` is resumable: it records completed (engine, dataset, pass, phase) units in
the run directory and skips them on a re-run, so an interruption costs one unit
rather than the whole sweep.
"""

from __future__ import annotations

import argparse
import ast
import getpass
import json
import os
import re
import shutil
import socket
import subprocess
import threading
import sys
import traceback
import time
from typing import Any, Dict, List, Optional

VB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, VB_ROOT)

from harness import datasets as datasets_mod  # noqa: E402
from harness.metrics import sysinfo as sysinfo_mod  # noqa: E402
from orchestrator import ann_pass, docker_ctl, ops_pass  # noqa: E402
from orchestrator import engines as engines_mod  # noqa: E402
from orchestrator.config import (available_profiles, load_engine,  # noqa: E402
                                 load_profile, load_resources,
                                 merge_resource_overrides, resolve_resources)
from orchestrator.manifest import Manifest, new_run_id, utcnow  # noqa: E402
# The cost estimate lives in its own module; these are re-exported because
# `webui/control.py` and `tests/test_config.py` import them from here and the
# split must not be visible to either. Import them from orchestrator.estimate
# in new code. `_INGEST_ROWS_PER_S` keeps its underscore despite being read
# across modules because tests/test_config.py imports it by that name.
from orchestrator.estimate import (_INGEST_ROWS_PER_S,  # noqa: E402,F401
                                   estimate_load_hours, print_load_estimate)

GB = 1024 ** 3

# The original three-way comparison, and the engines that arrived after it.
# Both are read from config/engines/*.yml rather than listed here: a name in one
# place and a config in another is how an engine ends up half-registered.
ALL_ENGINES = engines_mod.engines_in_group("original")
EXTRA_ENGINES = engines_mod.engines_in_group("extra")
# What `run` measures when nobody says otherwise. This once defaulted to the
# original three long after the other three had joined the study, so
# `run --profile smoke` proved three engines and a forty-hour run would have
# measured three -- both reporting success, because an engine nobody asked for
# cannot fail.
KNOWN_ENGINES = engines_mod.known_engines()


# Docker's own rule for container and volume names. The run id becomes both, so
# anything outside this set fails at `docker run`, not here.
RUN_ID_ALLOWED = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")


def run_id_problem(run_id: str) -> Optional[str]:
    """Reject a run id Docker will not accept, before anything is created.

    A shell variable picked up the escape sequence an arrow key sends, so
    $BIG held "tuned-complete-20260820-134424\x1b[A". Everything derived from
    it inherited the escape: the results directory, the manifest, the ann
    results tree. Two phases ran and wrote into that directory before Docker
    refused the third with

        Invalid container name (...134424[A-annb-mongodb-dbpedia-openai)

    which named the symptom four hundred lines after the cause. The id is the
    first thing the run has and the last thing anyone inspects, so it is
    checked before a single directory is made.
    """
    if RUN_ID_ALLOWED.match(run_id):
        return None
    printable = "".join(c if c.isprintable() else repr(c)[1:-1] for c in run_id)
    lines = [
        f"invalid --run-id: {printable}",
        f"  as characters: {run_id!r}",
        "  Docker allows [a-zA-Z0-9][a-zA-Z0-9_.-] in container and volume "
        "names, and this id becomes both.",
    ]
    if any(not c.isprintable() for c in run_id):
        lines.append(
            "  It contains a non-printing character. A shell variable that was "
            "edited with the arrow keys is the usual source; set it again by "
            "typing or pasting the id rather than recalling it.")
    return "\n".join(lines)


def paths_for(run_id: str) -> Dict[str, str]:
    results = os.path.join(VB_ROOT, "results")
    run_dir = os.path.join(results, run_id)
    return {
        "root": VB_ROOT,
        "sources": os.path.join(VB_ROOT, "sources"),
        "harness": os.path.join(VB_ROOT, "harness"),
        "scripts": os.path.join(VB_ROOT, "scripts"),
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


def cmd_generate(args: argparse.Namespace) -> int:
    """Build a dataset ann-benchmarks constructs rather than publishes.

    `fetch` cannot retrieve these: they are assembled from a source corpus and
    then have their ground truth computed by brute force, which is hours of
    work and tens of GB rather than a download.
    """
    if args.list:
        return _script("generate-dataset.sh", "--list")
    if not args.dataset:
        print("which dataset? try: ./run-benchmark.sh generate --list",
              file=sys.stderr)
        return 2
    return _script("generate-dataset.sh", args.dataset)


def cmd_render(args: argparse.Namespace) -> int:
    profile = load_profile(args.profile)
    resources = merge_resource_overrides(
        load_resources(args.resource_pass), profile)
    work = paths_for("render")["work_annb"]
    if not os.path.isdir(work):
        print(f"ann-benchmarks working copy missing at {work}; "
              f"run: scripts/prepare-harness.sh", file=sys.stderr)
        return 1
    for engine in (args.engines.split(",") if args.engines else KNOWN_ENGINES):
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


def cmd_export(args: argparse.Namespace) -> int:
    """Package a run for someone who does not have this checkout."""
    from orchestrator.export import bundle_filename, write_bundle

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        candidate = os.path.join(VB_ROOT, "results", args.run_dir)
        if not os.path.isdir(candidate):
            print(f"run directory not found: {args.run_dir}", file=sys.stderr)
            return 1
        run_dir = candidate

    run_id = os.path.basename(run_dir.rstrip(os.sep))
    out = args.output or os.path.join(os.getcwd(), bundle_filename(run_id))
    ok, detail = write_bundle(run_dir, out)
    if not ok:
        print(detail, file=sys.stderr)
        return 1

    size = os.path.getsize(detail)
    print(f"{detail}  ({size / 1024 / 1024:.1f} MB)")
    print("  contains the report, the raw records and a README explaining what "
          "it is.\n  The recipient needs nothing installed to read "
          "report/report.html.")
    return 0


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

def _resume_target(profile: Dict[str, Any]) -> Optional[str]:
    """The run --resume should continue, when no --run-id was given.

    Checkpoints live inside the run directory, so --resume with a fresh
    timestamped run id resumes nothing: it silently re-runs every unit into an
    empty directory and produces a report containing only the engines named on
    that command line. The results of the run being continued stay where they
    were, and the two have to be merged by hand afterwards.

    Picking the most recent directory for the same profile is what --resume
    plainly means. It is announced rather than assumed, because continuing the
    wrong run is worse than starting a new one.
    """
    name = profile.get("name", "run")
    results = paths_for("x")["run_dir"].rsplit(os.sep, 1)[0]
    try:
        candidates = sorted(
            d for d in os.listdir(results)
            if d.startswith(f"{name}-")
            and os.path.isfile(os.path.join(results, d, "run-manifest.json"))
        )
    except OSError:
        return None
    if not candidates:
        print("--resume: no previous run found for profile "
              f"{name!r}; starting a new one")
        return None
    latest = candidates[-1]
    print(f"--resume: continuing {latest} "
          f"(pass --run-id to continue a different one)")
    return latest


def cmd_run(args: argparse.Namespace) -> int:
    if not docker_ctl.docker_available():
        print("cannot talk to the Docker daemon", file=sys.stderr)
        return 1

    profile = load_profile(args.profile)
    engines = [e.strip() for e in
               (args.engines or ",".join(KNOWN_ENGINES)).split(",") if e.strip()]
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

    run_id = args.run_id
    if not run_id and args.resume:
        run_id = _resume_target(profile)
    run_id = run_id or new_run_id(profile.get("name", "run"))
    problem = run_id_problem(run_id)
    if problem:
        print(problem, file=sys.stderr)
        return 2
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

    print_load_estimate(profile, engines, datasets, passes, phases)

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
                presentation=engines_mod.presentation(engine),
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


def _yaml_mode_name(value: Any) -> str:
    """The name a YAML scalar was written as, for keywords YAML 1.1 steals.

    `iterative_scan: [off, relaxed_order]` does not parse to the strings it
    looks like: YAML 1.1 reads a bare `off` as boolean False, so the list comes
    back as `[False, "relaxed_order"]`. `str(False)` is "False", which is not a
    pgvector mode and would reach the server as
    `SET hnsw.iterative_scan = False`.
    """
    if value is True:
        return "on"
    if value is False:
        return "off"
    return str(value)


def _ops_iterative_scan(engine: str, profile: Dict[str, Any],
                        resources: Dict[str, Any],
                        resource_pass: str) -> Optional[str]:
    """The pgvector iterative-scan mode the ops phase should pin, or None.

    `ops.pgvector_iterative_scan` has been declared by `main`, `full` and the
    tuned extras for as long as it has existed, the whole plumbing chain to
    `SET hnsw.iterative_scan` is built, and the report renders a column for
    it -- but nothing ever passed it. Across the archived results, 58 pgvector
    ops files and 69 filtered records, every one reads `iterative_scan: null`.
    So three config files declared two modes, the report printed a column, and
    the dimension never ran.

    Pinning the FIRST declared value rather than looping is deliberate on both
    counts. Explicit, because pgvector's own default is `off`: pinning it
    changes no measured number today while making the record say `off` instead
    of null, and the repo's standard is that a vendor default left in place
    judges the vendor on a value it plainly intended you to set. First rather
    than all, because looping here would multiply ops *units* -- `_run_unit`
    reloads the corpus per unit, so varying a filtered-query setting would
    re-run build, concurrency and churn and double pgvector's ingest to do it.
    The loop belongs inside `harness/workloads/filtered.py`, which sets the
    mode per selectivity against one loaded index; that is T3c's, not this
    call site's.

    A profile that declares nothing still passes nothing, so the engine keeps
    its own default and the record keeps saying so.

    One new way for a run to die, named so nobody is surprised by it: a `SET`
    is now issued where none ever was, and nothing catches around it, so a mode
    this server rejects costs the whole ops unit -- `harness/main.py` wraps
    concurrency, filtered and churn in one `except`, so the two workloads
    either side of filtered are lost with it. The unit is checkpointed, so
    `--resume` retries exactly that one. The validation below is what keeps the
    risk to "pgvector removed a mode its own config still declares" rather than
    "someone made a typo". Making that call fail soft belongs to whoever owns
    `harness/`.
    """
    if engine != "pgvector":
        return None
    extras = (resources.get("extras", {}) or {}) if resource_pass == "tuned" else {}
    declared = (extras.get("pgvector_iterative_scan")
                or (profile.get("ops", {}) or {}).get("pgvector_iterative_scan"))
    if not declared:
        return None
    mode = _yaml_mode_name(list(declared)[0])
    # Validated against the engine's own declaration rather than a list here:
    # the driver interpolates this into a SET statement, so an unrecognised
    # value would reach the server verbatim.
    valid = {_yaml_mode_name(m) for m in
             (load_engine(engine).get("capabilities", {}) or {})
             .get("iterative_scan_modes", [])}
    if valid and mode not in valid:
        print(f"[ops] ignoring unknown pgvector iterative_scan mode {mode!r}; "
              f"known modes are {', '.join(sorted(valid))}", file=sys.stderr)
        return None
    return mode


def _read_manifest_config(run_dir: str) -> Dict[str, Any]:
    """The run's `config` block, or an empty dict if it cannot be read."""
    try:
        with open(os.path.join(run_dir, "run-manifest.json")) as fh:
            return json.load(fh).get("config") or {}
    except (OSError, ValueError):
        return {}


def generate_report(paths: Dict[str, str], engines: List[str],
                    datasets: Optional[str] = None) -> int:
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
            *(["--datasets", datasets] if datasets else []),
        ],
        detach=False,
    )
    rc = docker_ctl.run_foreground(spec, timeout=3600)
    if rc == 0:
        ann_pass.fix_ownership(paths["run_dir"], image)
        print(f"report: {os.path.join(paths['run_dir'], 'report', 'report.html')}")
    return rc


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


# **THE TWO KEYS LIVE IN DIFFERENT BLOCKS, AND THAT IS THE FIX FOR A
# DEFECT RATHER THAN AN INCONSISTENCY.** `pgvector_build_modes` drives
# BOTH passes -- `ann_pass.render_config` emits a run group per mode
# and this loop runs an ops unit per mode -- so it belongs with the
# ann configuration and is read from there by both.
#
# Percona's drives the OPS pass only, because the ann overlay module
# reads no `build_mode` (see `ann_pass.render_config`'s percona
# branch, which REFUSES the ann key rather than emitting two identical
# curves). Putting it under `ann:` as `percona_build_modes` was tried
# and was wrong twice over: `render_config` is handed the whole
# profile and polices that key, so `run --profile percona` failed its
# ann phase and `render --profile percona` raised uncaught from
# `cmd_render` -- which loops over KNOWN_ENGINES, so the traceback
# arrived for a profile that names one engine. Under `ops:` the ann
# renderer structurally cannot see it, and the one profile that sets
# it gets both an ef_construction sweep on the ann side and both build
# modes on the ops side, which is what it exists for.
#
# Checked for every shipped profile by
# `tests/test_e2_percona_wiring.py::TestEveryShippedProfileRenders`.
#
# **The default WAS the literal `["post"]` for every engine including
# Percona, and for Percona that default could not complete: see
# `_ops_build_modes` below, which now takes it from the engine's own
# `capabilities.bulk_vector_index_build`. The asymmetry the rest of this
# paragraph describes is real and unchanged for every engine whose bulk
# build works.** For
# MariaDB and AliSQL `post` is a no-op -- `create_index` does nothing
# and `capabilities()["incremental_index"]` is True, so the measured
# build is the incremental one either way. For Percona `post` is a
# genuine bulk build, so a profile that names no mode measures
# Percona's bulk build against the other two's incremental one. That
# is the honest default (it is each engine's idiomatic build, which is
# what `docs/05-methodology.md` says these comparisons report), but it
# is not like-for-like, and `build_mode` is on every record so the
# report can say which. `config/profiles/percona.yml` ran both until the
# bulk build was found to kill the server at the pinned version; it now
# asks for `[incremental]` alone.
_OPS_BUILD_MODE_KEYS = {
    "pgvector": ("ann", "pgvector_build_modes"),
    "percona": ("ops", "percona_build_modes"),
}


def _bulk_vector_index_build(engine_cfg: Dict[str, Any]) -> bool:
    """Whether this engine's bulk vector index build works at its pinned source.

    ONE reading of one fact, because there are now two consumers of it and
    they must not drift: `_ops_build_modes` below picks the ops phase's
    default build mode from it, and `ops_pass.harness_args` tells the ops
    harness about it so a driver whose repair path REBUILDS the index does not
    time a statement that cannot work. `maintain()`'s `ALTER TABLE … DROP KEY
    vi, ADD VECTOR KEY …, ALGORITHM=INPLACE` is the same statement
    `create_index` issues for `build_mode: post`, which is why both answers
    come off the same key. It was already read twice inside
    `_ops_build_modes` alone before this existed.

    **Absence reads as True**, and that default is the whole compatibility
    story: the key is declared by exactly one engine config
    (`config/engines/percona.yml`, false at its pin), which
    `tests/test_e9_percona_bulk_build_crash.py` pins. mariadb, mariadb123,
    alisql, pgvector, mongodb and valkey declare nothing and are unaffected in
    both consumers.
    """
    caps = engine_cfg.get("capabilities", {}) or {}
    return bool(caps.get("bulk_vector_index_build", True))


def _ops_build_modes(engine: str, profile: Dict[str, Any],
                     engine_cfg: Dict[str, Any]) -> List[str]:
    """Which build modes the ops phase runs for one engine.

    A function rather than a block inside `_run_unit` so the decision can be
    exercised without starting a server: every arm below is reachable from a
    profile dict and an engine config, and `tests/test_e9_percona_bulk_build
    _crash.py` reports the honest and mutant value of each.
    """
    mode_ref = _OPS_BUILD_MODE_KEYS.get(engine)
    # Derived once and used twice below. It used to be two `caps.get` calls
    # with the same default spelled out in both, which is the shape a drift
    # between the two readings hides in; it is now
    # `_bulk_vector_index_build`, shared with the ops harness's flag.
    bulk_build = _bulk_vector_index_build(engine_cfg)
    # **THE DEFAULT COMES FROM THE ENGINE, NOT FROM A LITERAL HERE, AND THAT
    # CHANGE FIXED A CRASH IN EVERY PROFILE THAT NAMES NO MODE.** The
    # argument at `_OPS_BUILD_MODE_KEYS` is that `post` is the honest default
    # because it is each engine's idiomatic build. That holds for every engine
    # whose bulk build WORKS. Percona's does not at its pinned version:
    # `ALTER TABLE ... ADD VECTOR KEY ... ALGORITHM=INPLACE` kills the server,
    # so a literal `["post"]` default meant `main`, `smoke`, `dev` and every
    # other profile that sets no `percona_build_modes` ran Percona in the one
    # mode that cannot complete -- after paying for the whole load first. The
    # evidence is at `capabilities.bulk_vector_index_build` in
    # `config/engines/percona.yml`.
    #
    # An engine that declares nothing keeps `post`, so this widens where the
    # default comes from rather than changing the default for anybody else.
    default_modes = ["post"] if bulk_build else ["incremental"]
    if mode_ref:
        block, key = mode_ref
        modes = list((profile.get(block) or {}).get(key, default_modes))
    else:
        modes = list(default_modes)
    # WARNED AND HONOURED, not refused. Whoever tests a newer Percona for the
    # fix has to be able to ask for the broken mode on purpose, and a refusal
    # would mean editing the engine config to do it. The warning names the
    # symptom so the resulting lost connection is not read as a harness fault.
    if "post" in modes and not bulk_build:
        print(f"[ops] WARNING: {engine} declares "
              f"capabilities.bulk_vector_index_build: false and this profile "
              f"asks for build_mode 'post' anyway. At the pinned version that "
              f"ALTER kills the server after the load has already been paid "
              f"for, and the unit will fail with a lost connection. "
              f"Honouring the explicit request. Note that the SAME "
              f"declaration also withdraws the churn repair path in EVERY "
              f"build mode -- `maintain()` issues that same ALTER, so the "
              f"harness is told to declare repair_path 'none' and writes no "
              f"post-maintenance record. Flipping the capability is what "
              f"restores both.", file=sys.stderr)
    return modes


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
        # Which build modes the OPS phase runs for this engine. Keyed per
        # engine because "both modes" is only meaningful for an engine that
        # has both: MariaDB and AliSQL maintain the graph inside every INSERT
        # and have no separable build, so a second mode for them would be the
        # same measurement twice.
        #
        # Percona is the second engine here that genuinely has both -- it
        # builds incrementally on INSERT and also has a real bulk path
        # (`ALTER TABLE … ADD VECTOR KEY` from one clustered scan), which
        # `PerconaDriver.create_schema`/`create_index` split on. It is added
        # beside pgvector rather than instead of a general lookup because the
        # list of engines that can answer the question is the thing being
        # declared, and an engine that cannot would silently produce a
        # duplicate curve.
        #
        build_modes = _ops_build_modes(engine, profile, engine_cfg)
        # Read here, once per phase, rather than inside the unit loop: it is a
        # property of the engine's pinned source and cannot vary with M, the
        # build mode or the storage engine. The same value `build_modes` above
        # was derived from, which is the point of there being one reader.
        bulk_build = _bulk_vector_index_build(engine_cfg)
        storage_engines = _ops_storage_engines(engine, profile, resources,
                                               resource_pass)
        worst_rc = 0
        # **A FAILED UNIT USED TO BE SILENT, AND THE PHASE'S EXIT CODE WAS THE
        # ONLY TRACE OF IT.** One ops phase is M values x build modes x
        # storage engines, each an independent measurement, and `worst_rc`
        # folds all of them into one number -- so a phase in which one unit
        # crashed and another completed perfectly printed a bare
        # `[FAIL] ... (exit 1)` with nothing saying which half was which. It
        # cost a real diagnosis: a crashed `post` unit was read as a failure
        # of the `incremental` unit that had just succeeded.
        #
        # `--fail-fast` deliberately stays PHASE-level and is not applied
        # here. The units are independent, and stopping the loop on the first
        # one would have thrown away the successful measurement that proved
        # which half was broken. Visibility is the fix; aborting is not.
        unit_count = (len(m_values) * len(build_modes)
                      * len(storage_engines))
        failed_units: List[str] = []
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
                    # The recorder appends, which is right within a unit and
                    # wrong across attempts: re-running a failed unit left the
                    # previous attempt's records in the file and the report
                    # read both as separate measurements. A six-engine re-run
                    # produced twelve concurrency points per engine and two
                    # index sizes. A unit is atomic, so its output starts empty.
                    for stale in (output, memory_ts):
                        if os.path.exists(stale):
                            os.remove(stale)
                    harness_args = ops_pass.harness_args(
                        profile, m, engine, resolved, resource_pass, resources,
                        build_mode=build_mode, storage_engine=storage,
                        iterative_scan=_ops_iterative_scan(
                            engine, profile, resources, resource_pass),
                        bulk_vector_index_build=bulk_build,
                    )
                    with ops_pass.OpsRun(engine, engine_cfg, resolved,
                                         resource_pass, paths, run_id,
                                         dataset, tag) as run:
                        rc = run.run_harness(harness_args, output,
                                             memory_timeseries=memory_ts)
                    if rc == 0:
                        if checkpoints is not None:
                            _save_checkpoint(paths["checkpoints"], sub_key)
                            checkpoints.add(sub_key)
                    else:
                        failed_units.append(tag)
                        print(f"[ops] UNIT FAILED: {tag} (exit {rc})",
                              file=sys.stderr)
                    worst_rc = worst_rc or rc
        if failed_units:
            print(f"[ops] {len(failed_units)} of {unit_count} units failed: "
                  f"{', '.join(failed_units)}", file=sys.stderr)
        return worst_rc

    raise ValueError(f"unknown phase: {phase}")


WEBUI_IMAGE = "vector-bench/webui:latest"


def webui_container_name(port: int) -> str:
    return f"vb-webui-{int(port)}"


def _webui_loopback(host: str) -> bool:
    return host in ("127.0.0.1", "::1", "localhost", "")


def _webui_reachable_host(host: str) -> str:
    """The address to talk to a server published on `host`."""
    return "127.0.0.1" if host in ("0.0.0.0", "", "::") else host


def _socket_group(socket_path: str) -> Optional[str]:
    """The name of the group owning the Docker socket.

    Read rather than assumed: it is `docker` on most installs and is not on all
    of them, and telling someone to add themselves to a group that does not
    exist wastes the one instruction they were given.
    """
    try:
        import grp
        return grp.getgrgid(os.stat(socket_path).st_gid).gr_name
    except (OSError, KeyError, ImportError):
        return None


def _no_docker_message() -> str:
    user = getpass.getuser()
    socket_path = os.environ.get("DOCKER_HOST", "").replace("unix://", "") \
        or "/var/run/docker.sock"
    lines = [f"cannot reach the Docker daemon as {user!r}.",
             "  Check with:  docker info"]

    if os.path.exists(socket_path):
        group = _socket_group(socket_path)
        gid = os.stat(socket_path).st_gid
        named = group or f"the group with gid {gid}"
        lines += [
            f"  {socket_path} is owned by {named}"
            + (f" (gid {gid})" if group else "") + ".",
            f"  A service user gets the groups it holds in /etc/group, so if "
            f"{user!r} is not in it, nothing this runs can speak to Docker:",
            f"      id {user}",
        ]
        if group:
            lines.append(f"      sudo usermod -aG {group} {user}")
            lines.append(f"      sudo systemctl restart vector-bench-web")
    else:
        lines.append(f"  {socket_path} does not exist. Is Docker installed and "
                     f"running?  systemctl status docker")
    return "\n".join(lines)


def _outbound_address() -> Optional[str]:
    """This machine's address on the network it routes through.

    Opening a UDP socket to a documentation address asks the routing table
    which interface would be used; nothing is sent and nothing listens there.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.settimeout(0.2)
            probe.connect(("192.0.2.1", 1))  # TEST-NET-1, RFC 5737
            return probe.getsockname()[0]
    except OSError:
        return None


def _port_holder(host: str, port: int) -> Optional[str]:
    """A description of what holds this port, or None if it is free.

    Docker's own message for a taken port names an endpoint id and a network
    driver, which tells an operator nothing about what to do next.
    """
    probe = _webui_reachable_host(host)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        if sock.connect_ex((probe, port)) != 0:
            return None

    for container in docker_ctl.containers_publishing(port):
        return f" by container {container}"
    return ""


def _webui_responds(host: str, port: int) -> bool:
    import urllib.error
    import urllib.request
    url = f"http://{_webui_reachable_host(host)}:{port}/api/health"
    try:
        with urllib.request.urlopen(url, timeout=1) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError):
        return False


def cmd_web(args: argparse.Namespace) -> int:
    """Serve the web UI.

    In a container by default, following the report generator: the host keeps
    its python3-and-pyyaml guarantee. The repo is mounted at its own absolute
    path because the orchestrator inside the container hands host paths to the
    Docker daemon -- a container-only path would resolve to nothing on the host
    and silently mount an empty directory.
    """
    auth_enabled = args.auth
    if not _webui_loopback(args.host) and not args.no_auth:
        auth_enabled = True
    if args.auth and args.no_auth:
        print("--auth and --no-auth are contradictory", file=sys.stderr)
        return 2

    if args.no_container:
        from webui.server import serve
        return serve(VB_ROOT, args.host, args.port, args.allow_control,
                     auth_enabled=auth_enabled,
                     password=os.environ.get("VB_WEB_PASSWORD"),
                     behind_proxy=args.behind_proxy)

    # Ask whether Docker answers at all before asking what images it has.
    # image_exists() cannot tell "no such image" from "cannot reach the daemon",
    # so a permission problem on the socket used to be reported as a missing
    # image -- which sends you off to rebuild something you already have.
    if not docker_ctl.docker_available():
        print(_no_docker_message(), file=sys.stderr)
        return 1

    if not docker_ctl.image_exists(WEBUI_IMAGE):
        print(f"{WEBUI_IMAGE} not found; build it with:\n"
              f"  ./scripts/build-images.sh --engine webui", file=sys.stderr)
        return 1

    # Run as the invoking user, not root. The repo is a bind mount owned by that
    # user: as root, git refuses the working copy as "dubious ownership" and
    # every file the server writes lands root-owned on the host.
    # Named for the port, not the pid. A stable name is what lets a service
    # unit clear a leftover container before starting and after stopping; a
    # pid-derived one is different every time and so cannot be cleaned up by
    # anything but luck. The port keeps two instances from colliding.
    container = webui_container_name(args.port)
    command = [
        "docker", "run", "--rm", "--name", container,
        "--publish", f"{args.host}:{args.port}:8080",
        "--volume", f"{VB_ROOT}:{VB_ROOT}",
        "--workdir", VB_ROOT,
        "--user", f"{os.getuid()}:{os.getgid()}",
        "--env", "PYTHONUNBUFFERED=1",
        # No home directory exists for that uid inside the image.
        "--env", "HOME=/tmp",
    ]
    if args.allow_control:
        # Launching runs means driving the host's Docker daemon. A non-root user
        # reaches the socket only through its group, which is read from the
        # socket itself rather than assumed to be called "docker".
        socket_path = "/var/run/docker.sock"
        command += ["--volume", f"{socket_path}:{socket_path}"]
        try:
            command += ["--group-add", str(os.stat(socket_path).st_gid)]
        except OSError:
            print(f"warning: cannot stat {socket_path}; launching runs from the "
                  f"UI may fail with a permission error", file=sys.stderr)

    if os.environ.get("VB_WEB_PASSWORD"):
        command += ["--env", "VB_WEB_PASSWORD"]

    command += [WEBUI_IMAGE, "--root", VB_ROOT, "--host", "0.0.0.0", "--port", "8080",
                # The container binds every interface; only the publish address
                # decides who can reach it, and only this side knows it.
                "--published-host", args.host]
    if args.allow_control:
        command.append("--allow-control")
    # The container always binds 0.0.0.0 -- publishing decides who reaches it --
    # so the auth decision is made here from the published address, not there.
    command.append("--auth" if auth_enabled else "--no-auth")
    if args.behind_proxy:
        command.append("--behind-proxy")

    # An unclean stop leaves the previous container running and holding the
    # port. Match any of ours on it, not just the name this invocation would
    # choose: containers made before the naming changed are called after the
    # pid that started them, so an exact-name check walked straight past them
    # and every restart then failed with "port is already in use".
    stale = [name for name in docker_ctl.containers_publishing(args.port)
             if name.startswith("vb-webui-")]
    for name in stale:
        print(f"removing a leftover {name} from an unclean stop")
        docker_ctl.remove(name)
    if stale:
        time.sleep(1)

    busy = _port_holder(args.host, args.port)
    if busy:
        print(f"port {args.port} is already in use{busy}.\n"
              f"  Use a different one:  ./run-benchmark.sh web --port {args.port + 1}",
              file=sys.stderr)
        return 1

    mode = "control enabled" if args.allow_control else "read-only"
    scheme = "https" if args.behind_proxy else "http"
    url = f"{scheme}://{args.host}:{args.port}"
    print(f"starting the web UI ({mode}, auth {'on' if auth_enabled else 'off'}) …",
          flush=True)

    # The URL is announced by a watcher once the server actually answers, not
    # before the container is started. Printing it up front meant a failure --
    # a port clash, a missing mount -- arrived underneath a line claiming
    # success, which is the opposite of what an error message is for.
    ready = threading.Event()

    def announce() -> None:
        deadline = time.time() + 30
        while time.time() < deadline and not ready.is_set():
            if _webui_responds(args.host, args.port):
                # "http://0.0.0.0:8085" is not an address anyone can open.
                # Bound to everything means the useful answer is the address
                # other machines would actually type.
                if args.host in ("0.0.0.0", "", "::"):
                    outbound = _outbound_address()
                    if outbound:
                        print(f"\n  {scheme}://{outbound}:{args.port}"
                              f"   (share this one)")
                    print(f"  {scheme}://127.0.0.1:{args.port}"
                          f"   (on this machine)")
                else:
                    print(f"\n  {url}")
                if not args.allow_control:
                    print("  read-only; add --allow-control to edit profiles "
                          "and launch runs")
                print("  Ctrl-C to stop\n", flush=True)
                return
            time.sleep(0.4)

    watcher = threading.Thread(target=announce, daemon=True)
    watcher.start()
    try:
        code = subprocess.call(command)
    except KeyboardInterrupt:
        return 0
    finally:
        ready.set()

    if code != 0:
        print(f"\nthe web UI container exited with {code}; see the error above.",
              file=sys.stderr)
    return code


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
    return generate_report(paths, list(KNOWN_ENGINES), args.datasets)


# ---------------------------------------------------------------------------
# lab
# ---------------------------------------------------------------------------

#: Where a lab session's server log and volume live. Under state/, never under
#: results/: results/ is the namespace a report reads, and a lab session's
#: numbers come from vectors it made up.
LAB_DIRNAME = "lab"


def lab_paths(session: str) -> Dict[str, str]:
    """Paths for a lab session — the same shape OpsRun wants, minus results.

    `ops_results` is the run directory in a real run, which is what makes the
    harness's records land somewhere permanent. Here both it and `run_dir`
    point into state/, so the only thing a session leaves behind is the
    server's log and its data volume, and the latter goes at teardown.
    """
    base = os.path.join(VB_ROOT, "state", LAB_DIRNAME, session)
    paths = paths_for(session)
    paths.update({"run_dir": base, "ops_results": base})
    return paths


def lab_scripts() -> List[tuple]:
    """The scripts `lab` can run, with the first line of each docstring."""
    directory = os.path.join(VB_ROOT, "scripts")
    found = []
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".py"):
            continue
        summary = ""
        try:
            with open(os.path.join(directory, name)) as fh:
                tree = ast.parse(fh.read())
            summary = (ast.get_docstring(tree) or "").split("\n")[0]
        except (OSError, SyntaxError):
            pass
        found.append((name, summary))
    return found


def _lab_script_problem(script: str) -> Optional[str]:
    """Reject anything that is not a script in scripts/, before Docker sees it.

    A path separator here would mount-escape via the container's own view of
    /opt/scripts, and a missing file would only be discovered after the engine
    had been started -- which for the MySQL family is several minutes of an
    operator's time to learn they made a typo.
    """
    if os.path.basename(script) != script:
        return (f"'{script}': name a script in scripts/, not a path. "
                f"The directory is mounted for you.")
    if not script.endswith(".py"):
        return f"'{script}': lab scripts are Python; this is not a .py file."
    directory = os.path.join(VB_ROOT, "scripts")
    path = os.path.join(directory, script)
    if not os.path.isfile(path):
        names = ", ".join(name for name, _ in lab_scripts()) or "(none yet)"
        return f"no such script: scripts/{script}\navailable: {names}"
    # isfile() follows symlinks, so a link in scripts/ would be accepted and
    # run while looking like a file in the repo. Nothing puts one there today;
    # the check costs a line and means the name an operator reads is the file
    # that runs.
    if os.path.realpath(path) != os.path.join(os.path.realpath(directory),
                                              script):
        return (f"scripts/{script} is a link to "
                f"{os.path.realpath(path)}; lab runs files in scripts/, so "
                f"that the name in the command is the code that runs.")
    return None


#: lab's own options. Listed rather than read off the parser at call time,
#: which would mean either module-level mutable state or reaching into
#: argparse's privates; a test asserts this matches the subparser exactly, so a
#: flag added and forgotten here fails rather than quietly going unmentioned.
_LAB_OPTIONS = frozenset({
    "-h", "--help", "--engine", "--resource-pass", "--shell", "--list",
    "--client-memory-gb", "--force", "--timeout",
})


def measurement_in_progress() -> List[str]:
    """Running containers that belong to a measurement rather than a utility.

    The web UI, the pruner, the chown and rm helpers are all vector-bench
    containers a lab session may safely sit beside, and they are exactly the
    ones named `vb-*`. Everything else carries a run id in its name, which is
    what makes it part of a measurement -- see ann_pass's
    `{run_id}-annb-{engine}-{dataset}` and OpsRun's `-srv`/`-cli` pair.
    """
    return sorted(name for name in docker_ctl.running_containers()
                  if not name.startswith("vb-"))


def cmd_lab(args: argparse.Namespace) -> int:
    """Start one engine and run one script against it. Nothing is recorded.

    This exists because the questions a run raises cannot be answered by
    another run. Valkey's filtered search came back six times slower at 1%
    selectivity than at 10%, and settling whether that was the engine or our
    query took four query shapes against a populated index -- five minutes of
    work reachable, before this, only by hand-starting containers or by paying
    for a forty-hour profile.
    """
    if args.list:
        for name, summary in lab_scripts():
            print(f"  {name:<28} {summary}")
        return 0

    if not args.script and not args.shell:
        print("name a script (see --list), or pass --shell", file=sys.stderr)
        return 2

    if not args.engine:
        print(f"--engine is required; one of: {', '.join(KNOWN_ENGINES)}",
              file=sys.stderr)
        return 2
    if args.engine not in KNOWN_ENGINES:
        print(f"unknown engine: {args.engine}\nknown: {', '.join(KNOWN_ENGINES)}",
              file=sys.stderr)
        return 2

    # Not when --shell: the script is ignored below, and refusing an
    # invocation over an argument nothing reads is its own small confusion.
    if args.script and not args.shell:
        problem = _lab_script_problem(args.script)
        if problem:
            print(problem, file=sys.stderr)
            return 2

    # argparse.REMAINDER takes everything after the script name, which is what
    # lets a script have any flags it likes -- and also means a lab flag typed
    # after the script name silently becomes the script's problem. Saying so
    # costs a line and saves reading a stranger's argparse error.
    misplaced = sorted(set(args.script_args) & _LAB_OPTIONS)
    if misplaced:
        print(f"note: {', '.join(misplaced)} came after the script name, so "
              f"it is being passed to\n      {args.script} rather than to "
              f"lab. Put lab's own flags before the script name.",
              file=sys.stderr)

    if not docker_ctl.docker_available():
        print("cannot talk to the Docker daemon", file=sys.stderr)
        return 1

    # A lab session takes the same cores and the same memory the resource pass
    # hands a run, because that is the whole point of it. Started beside a live
    # run, it therefore contends with one -- and the damage lands on the run,
    # whose numbers are the ones being kept. This is the command most likely to
    # be typed into a second terminal while the first is measuring, since it
    # exists to answer questions a run raises, so the invariant is checked here
    # rather than trusted.
    busy = measurement_in_progress()
    if busy and not args.force:
        # The flag is spelled out in position because everything after the
        # script name goes to the script: `lab ... probe.py --force` hands
        # --force to the probe, which dies on it, and the guard stays up.
        print(f"a measurement is running: {', '.join(busy)}\n"
              f"lab would take the same cores and memory and spoil it. Wait "
              f"for it to finish, or,\nif those containers are not a "
              f"measurement you care about:\n\n"
              f"  ./run-benchmark.sh lab --engine {args.engine} --force "
              f"{args.script or '--shell'}\n", file=sys.stderr)
        return 5

    session = f"lab-{args.engine}-{time.strftime('%Y%m%d-%H%M%S')}"
    paths = lab_paths(session)
    os.makedirs(paths["run_dir"], exist_ok=True)

    info = sysinfo_mod.collect()
    engine_cfg = load_engine(args.engine)
    resources = load_resources(args.resource_pass)
    resolved = resolve_resources(resources, args.engine, info)

    print(f"=== vector-bench lab: {args.engine} ===")
    print(f"session   : {session}")
    print(f"pass      : {args.resource_pass} (same server flags a run of this "
          f"pass would use)")
    print(f"what      : {'interactive shell' if args.shell else args.script}")
    print(f"logs      : {os.path.relpath(paths['run_dir'], VB_ROOT)}")
    print("note      : nothing here is recorded. No records are written and "
          "no report can read this.\n")

    client_memory = (int(args.client_memory_gb * GB)
                     if args.client_memory_gb else None)

    try:
        with ops_pass.OpsRun(args.engine, engine_cfg, resolved,
                             args.resource_pass, paths, session,
                             dataset="lab", tag="lab") as run:
            if args.shell:
                print(f"[lab] the server is reachable as host "
                      f"'{run.server_name}' on port {run.port}")
                print(f"[lab] harness importable; scripts in /opt/scripts\n")
                return run.run_script("", [], interactive=True,
                                      client_memory_bytes=client_memory)
            return run.run_script(args.script, args.script_args,
                                  client_memory_bytes=client_memory,
                                  timeout_s=args.timeout)
    except KeyboardInterrupt:
        # OpsRun.__exit__ has already torn the pair down by the time this is
        # caught; saying so is the difference between a clean stop and an
        # operator hunting for a container that is not there.
        print("\n[lab] interrupted; containers removed", file=sys.stderr)
        return 130
    except docker_ctl.DockerError as exc:
        print(f"[lab] {exc}", file=sys.stderr)
        return 1


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

    g = sub.add_parser("generate",
                       help="build a dataset that is not published for download")
    g.add_argument("dataset", nargs="?", default=None)
    g.add_argument("--list", action="store_true",
                   help="datasets that must be generated rather than fetched")
    g.set_defaults(func=cmd_generate)

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
    rep.add_argument("--datasets", default=None,
                     help="comma-separated corpora to report on. The ann "
                          "results tree is keyed by resource configuration, "
                          "not by corpus, so a machine that measured two "
                          "corpora under one configuration reports both.")
    rep.set_defaults(func=cmd_report)

    w = sub.add_parser("web", help="serve the configuration and report web UI")
    w.add_argument("--port", type=int, default=8080)
    w.add_argument("--host", default="127.0.0.1",
                   help="host interface to publish on; loopback by default, "
                        "reach a remote rig over an SSH port-forward")
    w.add_argument("--allow-control", action="store_true",
                   help="enable profile editing and run launching "
                        "(mounts the Docker socket)")
    w.add_argument("--no-container", action="store_true",
                   help="run the server directly on the host instead")
    w.add_argument("--auth", action="store_true",
                   help="require a password (implied by a non-loopback --host)")
    w.add_argument("--no-auth", action="store_true",
                   help="publish on a non-loopback address with no password. "
                        "Only for a network you already trust")
    w.add_argument("--behind-proxy", action="store_true",
                   help="TLS is terminated in front; marks cookies Secure")
    w.set_defaults(func=cmd_web)

    e = sub.add_parser("export", help="package a run to send to someone")
    e.add_argument("--run-dir", required=True,
                   help="results/<run-id>, or just the run id")
    e.add_argument("--output", default=None,
                   help="path for the .tar.gz (default: ./vector-bench-<run-id>.tar.gz)")
    e.set_defaults(func=cmd_export)

    lab = sub.add_parser(
        "lab", help="run one diagnostic script against one engine, recording nothing",
        description=(
            "Start one engine exactly as a run of the given resource pass would "
            "start it, run one script from scripts/ against it, and tear "
            "everything down -- including on Ctrl-C. Nothing is recorded: no "
            "records are written and no report can read a lab session. It is "
            "for answering a question a run raised, not for producing a "
            "measurement."),
        epilog=("examples:\n"
                "  ./run-benchmark.sh lab --list\n"
                "  ./run-benchmark.sh lab --engine valkey probe-valkey-filter.py --rows 200000\n"
                "  ./run-benchmark.sh lab --engine valkey --shell"),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    lab.add_argument("script", nargs="?", default=None,
                     help="a .py file in scripts/; it must accept --host and --port")
    lab.add_argument("script_args", nargs=argparse.REMAINDER,
                     help="everything after the script name is passed to it")
    lab.add_argument("--engine", default=None,
                     help=f"one of: {', '.join(KNOWN_ENGINES)}")
    # tuned, not normalized: the long profiles that raise these questions run
    # tuned, and a probe started under a different budget than the run it is
    # investigating can disprove something the run never did.
    lab.add_argument("--resource-pass", default="tuned",
                     choices=("normalized", "tuned"),
                     help="server flags and limits to start the engine with "
                          "(default: tuned)")
    lab.add_argument("--shell", action="store_true",
                     help="a prompt in the bench container instead of a script")
    lab.add_argument("--list", action="store_true",
                     help="list the scripts lab can run")
    lab.add_argument("--client-memory-gb", type=float, default=None,
                     help="override the client container limit; raise it for "
                          "scripts that hold a million vectors")
    lab.add_argument("--force", action="store_true",
                     help="start even though a measurement is running")
    lab.add_argument("--timeout", type=int, default=4 * 3600,
                     help="seconds before the script's container is killed")
    lab.set_defaults(func=cmd_lab)

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
