"""Run manifest: the provenance that makes a number interpretable.

A result without its manifest is not a result. The report generator refuses to
emit a report for a run directory that has no manifest, because every one of the
conclusions this framework can produce depends on facts recorded here — which
CPU, which SIMD flags, which source commit, which resource limits, which image.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import socket
import subprocess
import sys
from typing import Any, Dict, List, Optional

# Imported, never re-declared. The constant lives in the harness because ops
# records are written from inside the harness container, which mounts harness/
# only and must not import orchestrator; the orchestrator has the whole repo and
# can read across. A second copy here would drift from the one that does the
# stamping, and the manifest would then disagree with its own records.
from harness.metrics.records import OPS_MEASUREMENT_VERSION

MANIFEST_NAME = "run-manifest.json"


def utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}-{_dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def _pass_entry(resource_pass: str, resolved: Any,
                fingerprint: Optional[str]) -> Dict[str, Any]:
    """One pass's record: its name, its limits and its results-tree key.

    The three keys are spelled exactly as the single-valued keys beside them
    in the config block, so a reader who knows that block knows an entry.

    Absent stays absent. A missing `ann_fingerprint` means nobody computed one
    for this pass, and the generator's stdout warning says so -- in those
    words, "cannot narrow to that pass's results tree". It deliberately does
    not say the recall is missing, and §2 Validity may say nothing at all:
    see the retraction in `_accumulate_passes` below for the branch where the
    whole-tree fallback publishes the pass anyway. Writing `None` here would
    make the same statement harder to tell apart from a tree keyed on the
    string "None".
    """
    entry: Dict[str, Any] = {
        "resource_pass": str(resource_pass),
        "resolved_resources": resolved,
    }
    if fingerprint is not None:
        entry["ann_fingerprint"] = fingerprint
    return entry


def _measured_on_this_host(recorded_host: Any) -> bool:
    """Whether the manifest's recorded host is the one running now.

    Empty counts as "yes": `cli.py` calls `set_host` before `set_config`, so a
    fresh run always has a hostname by the time a pass is recorded, and the
    only way to reach this with nothing recorded is a manifest written by
    something that never called `set_host` -- where refusing to record a
    fingerprint would block the recovery for no stated reason.

    Both spellings are accepted because the writer and the existing reader
    disagree: `harness/metrics/sysinfo.py` stores `platform.node()`, and
    `webui/runs.py` already compares the same field against
    `socket.gethostname()`. On Linux they are the same string; matching either
    means a machine is never called foreign because of which API asked.
    """
    if not recorded_host:
        return True
    return str(recorded_host) in (platform.node(), socket.gethostname())


def _accumulate_passes(prior: Dict[str, Any], resource_pass: str,
                       resolved: Any, fingerprint: Optional[str],
                       recorded_host: Any = None) -> List[Dict[str, Any]]:
    """`prior["passes"]` extended to cover `resource_pass`, as a new list.

    Why the list exists. `set_config` is called once per (pass, engine) and
    rebuilds the whole `config` block every time, so `config.resource_pass`,
    `config.resolved_resources` and `config.ann_fingerprint` end up describing
    whichever pass was configured last. A run of `--resource-pass both`
    therefore left behind a manifest that described a run of one pass, and
    every reader believed it. `full-20261015-215006` ran `normalized` then
    `tuned` -- 16 phases, 8 per pass, 41.9 h in total -- and its report
    narrowed the ann results tree to `tuned/a9349676` alone. The three
    completed normalized recall phases, 0.50 h + 4.80 h + 7.78 h = 13.1 hours
    of measurement, were simply absent from it: of 397 records in
    `report/records.jsonl`, none carried `phase: recall_qps` with
    `resource_pass: normalized`. Section 1 then printed the tuned machine --
    cpuset `0-75`, 112.5 GiB -- above tables holding rows measured with 8 CPUs
    and 16 GB. Nothing crashed and nothing looked wrong.

    `ann_fingerprint` is what makes this more than a labelling problem. It
    keys `results/annb/<pass>/<fingerprint>/` by the pass's *declared* knobs,
    so the normalized tree sits under a different string, and the report
    cannot derive it: that container mounts `report/` and `harness/` only, and
    importing the orchestrator from there once crashed generation at the end of
    a twenty-hour run. The string has to be recorded here, on the host, by the
    very function that keys the tree -- exactly as `presentation` is.

    Two rules, and both exist because `cli.py` calls `set_config` *before* the
    checkpoint skip, so `run --resume --run-id <old-run>` reaches it even when
    every unit is skipped and nothing is measured:

    **A pass already in the list is never rewritten.** Overwriting it on a
    resume would write today's resolved limits and today's fingerprint over a
    directory measured under other ones, which is the provenance bug the long
    comment in `set_config` describes for `ops_measurement_version`. Appending
    a pass that has no entry is additive; rewriting one is not.

    **The pass the block already describes is adopted before this call's.**
    The single-valued keys are already a complete entry for one pass, and on
    an archived run they are the only honest record of it -- they were written
    when it was measured. Adopting them first means a resume of
    `full-20261015-215006` recovers `tuned/a9349676` as measured, rather than
    re-deriving it from whatever machine the resume happens to run on, and
    then adds `normalized` beside it. That resume records both passes and
    measures nothing, which is how a run that has already cost 42 hours gets a
    complete report without being repeated.

    **A pass added on a machine that did not measure the run gets no
    fingerprint.** The entry's limits and fingerprint for a newly appended
    pass are this machine's derivation, which is right when the resume runs
    where the run ran and wrong otherwise. Wrong-and-absent is safe: the tree
    is not on disk, and the report names the pass and says its recall cannot
    be located. Wrong-and-*present* is not -- if this machine has itself
    measured that pass under its own knobs, `annb/<pass>/<its fingerprint>/`
    exists, and the report would merge another machine's recall curve as this
    run's under a §1 block carrying that machine's limits. The staleness check
    does not save it: that catches a tree measured BEFORE the run, not one
    measured after. So when the recorded host is not this one, the pass is
    recorded without a fingerprint. Same-machine recovery, the case this is
    for, is untouched. `webui/runs.py` compares the same field for the same
    reason.

    **That closes one of the two branches, not both.** This comment used to
    end "and the report says it cannot be located -- which is the truth", and
    that is only true where this host has never measured the pass: no
    directory, nothing read, and §2 names it. Where this host HAS measured it,
    withholding the fingerprint sends the report into its whole-tree fallback,
    which reads that very directory; §4 labels those records with the pass
    name taken from the directory, `summarize`'s intersection then sees recall
    for the pass and suppresses §2's notice, and nothing in the document says
    the tree was not this run's. §4's label is correct and no false sentence is
    printed, so this is an omitted warning about a hazard that predates the
    per-pass list rather than a new wrong number -- but it is not "the truth"
    that the report says anything at all. Telling a narrowed read from a
    fallback one is a backlog item; the records already carry the
    discriminator in `extra.source_file`.
    """
    out: List[Dict[str, Any]] = [dict(e) for e in (prior.get("passes") or [])
                                 if isinstance(e, dict) and e.get("resource_pass")]
    named = {e["resource_pass"] for e in out}
    carried = prior.get("resource_pass")
    if carried and str(carried) not in named:
        out.append(_pass_entry(str(carried), prior.get("resolved_resources"),
                               prior.get("ann_fingerprint")))
        named.add(str(carried))
    if str(resource_pass) not in named:
        # The adoption above keeps the archive's own fingerprint whatever host
        # this is; only a NEWLY appended pass is this machine's derivation, so
        # only it is withheld.
        out.append(_pass_entry(
            resource_pass, resolved,
            fingerprint if _measured_on_this_host(recorded_host) else None))
    return out


class Manifest:
    def __init__(self, run_dir: str, run_id: str):
        self.run_dir = run_dir
        self.path = os.path.join(run_dir, MANIFEST_NAME)
        os.makedirs(run_dir, exist_ok=True)
        # A re-run into an existing run directory continues that run rather
        # than replacing it. Building a fresh manifest here overwrote the
        # environment, the engine versions and every other engine's phase
        # record, so resuming one failed unit of a six-engine run destroyed the
        # provenance of the five that had succeeded.
        existing = self._load_existing(run_id)
        if existing is not None:
            self.data = existing
            self.data["status"] = "running"
            self.data["finished_at"] = None
            self.save()
            return

        self.data: Dict[str, Any] = {
            "run_id": run_id,
            "started_at": utcnow(),
            "finished_at": None,
            "status": "running",
            "framework": {
                "name": "vector-bench",
                "commit": _git_commit(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__)))),
            },
            "host": {},
            "engines": {},
            "phases": [],
            # Stamped at creation and never again -- see set_config. A run was
            # measured under the version it was measured under, and this is the
            # only moment at which "the code running now" and "the code that
            # produced the records in this directory" are the same thing.
            "config": {"ops_measurement_version": OPS_MEASUREMENT_VERSION},
            "warnings": [],
        }
        self.save()

    # -- population -----------------------------------------------------

    def set_host(self, sysinfo: Any) -> None:
        self.data["host"] = sysinfo.to_dict()
        self.data["host"]["platform"] = platform.platform()
        cpu = sysinfo.cpu
        if not cpu.has_avx512:
            # Not a defect — but both MHNSW and VIDX document AVX-512 distance
            # kernels, so a run without it is measuring their fallback paths and
            # any conclusion is scoped to hardware like this.
            self.add_warning(
                "This CPU has no AVX-512. MariaDB MHNSW and AliSQL VIDX both "
                "document AVX-512 distance kernels, so both are running narrower "
                "SIMD paths here. Results are valid for this class of hardware "
                "and should not be extrapolated to AVX-512 machines."
            )
        if cpu.hybrid:
            self.add_warning(
                f"Hybrid CPU detected ({len(cpu.performance_cpus)} performance / "
                f"{len(cpu.efficiency_cpus)} efficiency logical CPUs). Runs are "
                f"pinned to one core class; unpinned numbers from this machine "
                f"would carry scheduling noise larger than the effects measured."
            )

    def set_config(self, profile: Dict[str, Any], resource_pass: str,
                   resolved: Any, extra: Optional[Dict[str, Any]] = None) -> None:
        # Carried forward from whatever this directory already said, never
        # re-stamped, and never invented for a manifest that does not have it.
        #
        # This method is called unconditionally per (pass, engine) in
        # `cli.py`, *before* the checkpoint skip -- so `run --resume --run-id
        # <old-run>` reaches it even when every unit is skipped and nothing is
        # measured. Writing the current constant here therefore rewrote an
        # archived run's manifest to a version its records were never measured
        # under, and the report then refused that run forever. Re-stamping is
        # the provenance bug; the refusal was right.
        #
        # Absent stays absent for the same reason. A run recorded before this
        # key existed was measured under some version this code cannot know,
        # and `report/loaders.py` reads a missing key as "nothing is declared,
        # check nothing" -- which is correct, where inventing today's constant
        # would be a claim about a measurement nobody here witnessed.
        #
        # A resume that *does* bump the version and then measures something new
        # is still caught, by the other half of the check: the directory ends
        # up holding records at two versions.
        prior = self.data.get("config") or {}
        declared = prior.get("ops_measurement_version")
        self._warn_if_resuming_across_a_bump(declared)
        config: Dict[str, Any] = {
            "profile": profile,
            "resource_pass": resource_pass,
            "resolved_resources": resolved.as_dict() if hasattr(resolved, "as_dict") else resolved,
            **(extra or {}),
        }
        # After the splat either way, so a caller's `extra` cannot smuggle a
        # version in. It describes the code that produced the numbers, not a
        # caller's opinion of it.
        config.pop("ops_measurement_version", None)
        if declared is not None:
            config["ops_measurement_version"] = declared
        # No pop needed above this, unlike the version one: that assignment
        # is conditional, so a smuggled value survives when the manifest
        # declares nothing, which is every archived run. This one is
        # unconditional and `_accumulate_passes` reads `prior` rather than
        # `config`, so a `passes` key in a caller's `extra` is overwritten
        # whatever it holds. A pop here looked prudent and was dead code:
        # deleting it changed no test.
        config["passes"] = _accumulate_passes(
            prior, resource_pass, config["resolved_resources"],
            config.get("ann_fingerprint"),
            recorded_host=(self.data.get("host") or {}).get("hostname"))
        self.data["config"] = config
        # Persisted here rather than left to the next populator's save(). Every
        # other method on this class saves, and set_config did not only because
        # set_engine always happened to follow it — so a run that died between
        # the two left a manifest with no config block at all, and the report
        # refuses a manifest it cannot read a resource pass out of.
        self.save()
        for warning in getattr(resolved, "warnings", []) or []:
            self.add_warning(warning)

    def _warn_if_resuming_across_a_bump(self, declared: Any) -> None:
        """Say so now, not at report time, when a resume straddles a bump.

        The refusal downstream is correct and must stay: records measured
        under two ops measurement versions are two different measurements and
        cannot share a chart. But the operator learns about it from
        `report/loaders.py`, which runs *after* the measuring — and the
        measuring is the expensive half. Resuming a `main` run after a bump
        means up to days of ingest before anything says the directory it is
        filling cannot be reported as one. This is the only moment that knows
        both numbers and still precedes the work: `cli.py` calls `set_config`
        per (pass, engine) above the `(already complete)` checkpoint skip, so
        a resume reaches it even when every unit is skipped and nothing is
        measured. That ordering is pinned by
        `tests/test_config.py::TestResumingDoesNotRewriteTheVersion::test_cli_still_calls_set_config_before_the_checkpoint_skip`
        rather than cited by line number here: the pair this comment carried,
        :461 and :484, had drifted to :468 and :491 by the time anyone
        checked. The class is part of the node id -- the first version of this
        citation omitted it and collected nothing, which is the same lesson
        again: a reference being drift-RESISTANT in form says nothing about it
        being right, and the only way to know is to run it. If that call ever moves below the unit loop
        this notice becomes useless, and no test will say so -- "checkpoint"
        appears a dozen times either side of the call, so the obvious grep
        for the ordering passes whatever the order is.

        Deliberately not a failure. Finishing the run and generating one
        report per version is a legitimate, and often the right, response --
        the older records are not wrong, they are just not comparable. Making
        this fatal would turn a directory that needs splitting into a run that
        cannot continue.

        Once per process, not once per (pass, engine): six engines across two
        passes would print this twelve times, and a notice repeated twelve
        times reads as a loop rather than as a warning.
        """
        if declared is None or declared == OPS_MEASUREMENT_VERSION:
            return
        if getattr(self, "_bump_warned", False):
            return
        self._bump_warned = True
        print(
            f"\n[manifest] WARNING: this run directory was measured under ops "
            f"measurement version {declared!r}, and this code measures version "
            f"{OPS_MEASUREMENT_VERSION!r}.\n"
            f"[manifest] What that means: anything measured from here on is "
            f"not comparable with what is already in {self.run_dir} — the two "
            f"versions are different measurements, not two samples of one.\n"
            f"[manifest] The run will proceed and the manifest keeps saying "
            f"{declared!r}, which is the honest record of what it holds. But "
            f"`report` will refuse the directory as a whole once it contains "
            f"both, so plan on one report per version: generate the "
            f"version-{declared!r} report before this run appends to it, or "
            f"split the ops-*.jsonl files by extra.ops_measurement_version "
            f"afterwards.\n"
            f"[manifest] To re-measure everything under version "
            f"{OPS_MEASUREMENT_VERSION!r} instead, start a fresh --run-id "
            f"rather than resuming this one.\n",
            file=sys.stderr, flush=True)

    def set_engine(self, engine: str, sources_dir: str,
                   image_runtime: str, image_bench: str,
                   image_ids: Dict[str, str],
                   presentation: Optional[Dict[str, Any]] = None) -> None:
        source = _read_json(os.path.join(sources_dir, f"{engine}.source.json")) or {}
        build = _read_json(os.path.join(sources_dir, f"{engine}.image.json")) or {}
        self.data["engines"][engine] = {
            "source": source,
            "build": build,
            "images": {
                "runtime": {"ref": image_runtime, "id": image_ids.get("runtime", "unknown")},
                "bench": {"ref": image_bench, "id": image_ids.get("bench", "unknown")},
            },
            # Label, colour, marker and line style, so a report can name and draw
            # an engine the way its config defined it without reading that
            # config. The report container mounts report/ and harness/ only --
            # importing the orchestrator from there once crashed generation at
            # the end of a twenty-hour run -- so what it needs travels here,
            # exactly as ann_fingerprint does.
            "presentation": presentation or {},
        }
        self.save()

    def set_harness(self, sources_dir: str) -> None:
        annb = _read_json(os.path.join(sources_dir, "annbench.source.json"))
        if annb:
            self.data["ann_benchmarks"] = annb

    def _load_existing(self, run_id: str) -> Optional[Dict[str, Any]]:
        """The manifest already in this directory, if it is the same run."""
        try:
            with open(self.path) as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return None
        return data if data.get("run_id") == run_id else None

    def add_phase(self, name: str, engine: str, dataset: str,
                  status: str, started_at: str, finished_at: str,
                  detail: Optional[Dict[str, Any]] = None) -> None:
        # A repeated unit replaces its earlier attempt rather than sitting
        # beside it. Otherwise a re-run that fixes a failure leaves the failure
        # in the manifest, and the report keeps reporting it.
        self.data["phases"] = [
            p for p in self.data["phases"]
            if not (p.get("phase") == name and p.get("engine") == engine
                    and p.get("dataset") == dataset
                    and p.get("resource_pass") == (detail or {}).get("resource_pass"))
        ]
        self.data["phases"].append({
            "phase": name,
            "engine": engine,
            "dataset": dataset,
            "status": status,
            "started_at": started_at,
            "finished_at": finished_at,
            **(detail or {}),
        })
        self.save()

    def add_warning(self, message: str) -> None:
        if message not in self.data["warnings"]:
            self.data["warnings"].append(message)
            self.save()

    def finish(self, status: str = "completed") -> None:
        self.data["finished_at"] = utcnow()
        self.data["status"] = status
        self.save()

    # -- persistence ----------------------------------------------------

    def save(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self.data, fh, indent=2, sort_keys=True, default=str)
            fh.write("\n")
        os.replace(tmp, self.path)

    @staticmethod
    def load(run_dir: str) -> Dict[str, Any]:
        path = os.path.join(run_dir, MANIFEST_NAME)
        data = _read_json(path)
        if data is None:
            raise FileNotFoundError(
                f"no manifest at {path}. Results without a manifest cannot be "
                f"reported: the environment they were produced in is unknown."
            )
        return data


def _git_commit(path: str) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", path, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        return proc.stdout.strip() or "uncommitted"
    except Exception:
        return "unknown"
