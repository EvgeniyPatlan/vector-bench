"""Package a run for someone who does not have this checkout.

Two audiences, one archive. Someone who just wants to read the result opens
report/report.html, which is self-contained -- charts inlined, no network.
Someone who has vector-bench extracts the whole thing into results/ and gets
the run in their own UI, with its own machine's manifest attached.

The README is written into the archive because a tarball that arrives without
one is a tarball nobody opens.
"""

from __future__ import annotations

import fnmatch
import json
import os
import tarfile
import tempfile
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

README_NAME = "README.txt"

# What a core dump is called, as a filename glob, and the reason this bundle
# knows the word at all: the crash-capture path copies
# `find -name 'core*' -type f` out of the engine's data volume into
# `<run_dir>/cores/<engine>-<dataset>-<phase>-<pass>/`, keeping each file's own
# name. So the set of core files that can be sitting in a run directory is
# exactly the set this glob describes, and nothing else here has to guess.
#
# **MUST STAY IN STEP WITH `ops_pass.CORE_FILE_GLOB`**, which is the constant
# that capture matches on. Duplicated rather than imported, deliberately and
# for two reasons. The first is timing: the capture half is not on `master`
# yet, and a branch that imports a constant which does not exist is red until
# the other one merges, for no benefit. The second outlives the first --
# `export.py` is a leaf that imports the standard library and nothing else,
# which is what lets `webui/server.py` import it inside a request handler;
# `ops_pass` drags in `docker_ctl`, `engines`, `config` and `manifest`, none of
# which a web request that tars a directory has any use for. Collapsing the
# duplication therefore means moving the string somewhere both halves can read
# without `export` acquiring that dependency -- not importing `ops_pass` here.
CORE_FILE_GLOB = "core*"


def bundle_filename(run_id: str) -> str:
    return f"vector-bench-{run_id}.tar.gz"


def _fmt_bytes(value: Optional[int]) -> str:
    if not value:
        return "unknown"
    step = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if step < 1024 or unit == "TiB":
            return f"{step:.0f} {unit}" if unit == "B" else f"{step:.1f} {unit}"
        step /= 1024
    return f"{value} B"


def _logged_datasets(manifest: Dict[str, Any]) -> List[str]:
    """Corpora the phase log names, in the order the run reached them.

    Deduplicated rather than counted: one corpus appears once per
    (engine, phase, pass) that touched it.

    **First-seen order, deliberately, and pinned by a test.** This diverges
    from `report/scope.py::_ordered`, which sorts: its axes are compared
    against each other and against fixtures, where sorting is what makes two
    reads equal, and this is prose for a human reading one line. Run order
    tells that reader which corpus the run got to before it stopped, which
    alphabetical order destroys. Changing `return out` to `return
    sorted(out)` is a one-word mutant that passed the whole suite until
    `test_the_log_order_is_the_run_order_and_not_the_alphabet` existed, and
    it would have taken `full-20261015-215006`'s README line off
    byte-identical with the version this replaced -- `full` declares
    fashion, glove, sift, gist, which is not sorted. That byte-identical
    result on all 12 archived bundles is the measurement this whole term
    order was justified by, so the property it rests on gets a test.

    `str()` for the same reason `_accumulate_datasets` uses it: a
    hand-edited manifest with a non-string corpus name would otherwise reach
    `", ".join(...)` and raise, and `readme_text` is called from the export
    path and from `webui/server.py`, where a traceback is user-visible.
    """
    out: List[str] = []
    for phase in (manifest.get("phases") or []):
        name = phase.get("dataset") if isinstance(phase, dict) else None
        if name and str(name) not in out:
            out.append(str(name))
    return out


def _without_core_dumps(
        run_id: str,
        dropped: List[Tuple[str, int]],
) -> Callable[[tarfile.TarInfo], Optional[tarfile.TarInfo]]:
    """A `tarfile.add(filter=...)` callable that leaves core dumps out.

    Returning `None` for a member is how `filter` drops one, and `add` applies
    the filter to every member it walks, so this sees a core at any depth. It
    needs to: a core reaches a run directory in two shapes, the captured
    `cores/<engine>-<dataset>-<phase>-<pass>/core.<pid>` and -- for any process
    whose working directory is the mounted run directory itself -- a bare
    `core.<pid>` in the root of it.

    Each dropped member is appended to `dropped` as (path relative to the run
    directory, size in bytes), which is what the README then discloses. A
    mutable accumulator because `filter` has nowhere else to put it; the caller
    freezes the list before passing it on.

    WHAT IT WOULD DROP THAT IS NOT A CORE, said here rather than discovered
    later: any regular file whose basename matches the glob, so `cores.jsonl`
    or `core-notes.txt` would go too. Nothing writes such a name. The twelve
    archived run directories under `ALL_RESULTS/results/` hold
    `checkpoints.json`, `run-manifest.json`, `mem-*.jsonl`, `ops-*.jsonl`,
    `logs/*.log`, `.cache/*.npy` and `report/`, plus `vb-label.json` from a web
    UI import, and not one entry in any of them begins with those four
    letters. Narrowing the predicate below the capture side's own glob would be
    the worse failure anyway: it would keep a file that half had collected
    precisely because it is a core.

    WHAT IT DOES NOT DROP IS A DIRECTORY, whatever the directory is called.
    Two consequences, both wanted. `filter` returning `None` for a directory
    makes `add` skip the entire subtree without ever walking into it, so
    excluding `cores/` as a unit would be cheaper and would leave nothing to
    name or size in the README -- and the disclosure is the half of this that
    matters. And a directory named `core-something` keeps its contents, each
    file judged on its own basename. The emptied `cores/<quad>/` directory
    travels in the bundle, which is a second signal to the recipient that
    something was taken out of it.

    Matched case-sensitively because `find -name` on the capture side is, and
    because the kernel writes the name in lower case: on a case-insensitive
    filesystem `fnmatch.fnmatch` would additionally drop a `CORE.DAT` that
    capture had never looked at.

    A SYMLINK OR HARD LINK whose own name matches is also not a regular file
    and is also kept, and the two differ. A symlink costs nothing -- tar
    stores the target string, not the bytes. A hard link would be stored as a
    link to the member its inode was first seen under, and `tarfile` registers
    that inode in `gettarinfo`, BEFORE the filter runs, so the member it names
    may be one this filter has just dropped and extracting it would fail. That
    is recorded rather than handled: nothing here makes either one -- capture
    is `cp -p`, which writes a new file -- and the failure is loud at
    extraction rather than a silent omission, which is the direction this
    repository prefers to fail in.
    """
    def keep(info: tarfile.TarInfo) -> Optional[tarfile.TarInfo]:
        if info.isfile() and fnmatch.fnmatchcase(os.path.basename(info.name),
                                                 CORE_FILE_GLOB):
            dropped.append((os.path.relpath(info.name, run_id), info.size))
            return None
        return info
    return keep


def _excluded_block(excluded: Sequence[Tuple[str, int]]) -> str:
    """The README's account of what was left out, or nothing at all.

    The empty string when nothing was excluded, rather than a heading over an
    empty list: every bundle written before core capture existed, and every
    bundle of a run that did not crash, has nothing to disclose, and a section
    reading "(none)" would make a reader of the ordinary case go looking for
    the thing that was taken. `readme_text` interpolates this on a line of its
    own, so empty leaves the README byte-identical to the one those bundles
    carry.

    A size of 0 is printed as `0 B` and not through `_fmt_bytes`, which
    answers a falsy byte count with "unknown". The size of a truncated core is
    known -- it is zero -- and this section exists so the recipient can tell
    what they did not get. Saying "unknown" about a number we hold is the
    small version of the thing this whole file is careful about.
    """
    if not excluded:
        return ""
    # Size first, right-aligned, path second and free to run off to the right.
    # The obvious way round -- pad the path to a column and put the size after
    # it -- collides the two the moment a path is longer than the column, and
    # `cores/percona-gist-960-euclidean-ann-tuned/core.4211` is 51 characters,
    # so it collides in the ordinary case: `core.421154.7 GiB`. A size welded
    # to the end of a filename is a line a reader can misread rather than one
    # they can skip.
    lines = "\n".join(
        f"  {_fmt_bytes(size) if size else '0 B':>10}  {name}"
        for name, size in excluded)
    return f"""
NOT IN THIS BUNDLE
  Everything in the run directory travelled except the files below, which were
  left out on purpose:

{lines}

  Each is a server core dump: a full memory image of the database process at
  the moment it died, which on these workloads runs to tens of gigabytes and
  can only be read against the exact build and machine that wrote it. Sending
  one would be sending something unusable, slowly. Each is still in the run
  directory on the machine that measured, and what travels from a core is the
  BACKTRACE taken from it there -- `scripts/gdb-backtrace.py` in this repo
  produces one, and it is a few kilobytes. Ask for that if a crash is what
  brought you here.
"""


def readme_text(run_id: str, manifest: Dict[str, Any],
                excluded: Sequence[Tuple[str, int]] = ()) -> str:
    """What the recipient is looking at, and what they can trust it for.

    `excluded` is every (path relative to the run directory, size in bytes)
    that `write_bundle` kept out of the archive, and it is named and sized in
    the text because a recipient who is not told cannot tell. They have the
    bundle and no way to compare it against the run directory, so an
    undisclosed omission leaves them holding a credible-looking copy of a run
    that is missing something -- which is the failure mode this repository is
    most careful about. It defaults to empty for the callers that render the
    README without writing an archive.
    """
    host = manifest.get("host") or {}
    cpu = host.get("cpu") or {}
    config = manifest.get("config") or {}
    profile = config.get("profile") or {}
    engines = manifest.get("engines") or {}

    # Three terms in decreasing order of evidence, under a heading that says
    # WHAT WAS MEASURED, because that is what the line has to be true of.
    #
    # The phase log first. `add_phase` is appended after a unit returns -- a
    # failed one included -- so it is the only term that records measurement
    # rather than intent. `config.resolved_datasets` is written by
    # `set_config`, which `cmd_run` calls *above* the dataset loop, so a run
    # that dies in its first unit resolved four corpora and measured one, and
    # this line would otherwise name four under that heading.
    # `report/scope.py` and `webui/runs.py` prefer the log for the same
    # reason, and `scope.py` states the cost of doing so: a corpus whose unit
    # never returned is excluded, and the report names it in its Validity
    # census rather than publishing it.
    #
    # Then the resolved list, which is what a run killed before its first
    # unit returned has instead of a log -- and which, unlike the
    # declaration, is narrowed by `--datasets`. Then the declaration, which
    # is all a bundle archived before `resolved_datasets` existed has.
    #
    # Measured on all 12 archived run dirs under `ALL_RESULTS/results/`: the
    # line is byte-identical under this order and under the declaration
    # alone, including `percona-20261015-213817`, whose log is empty, and
    # `percona-20261015-183321`, whose one unit failed and is logged.
    datasets = (_logged_datasets(manifest)
                or list(config.get("resolved_datasets") or ())
                or list(profile.get("datasets") or ()))

    engine_lines = []
    for name, info in sorted(engines.items()):
        build = info.get("build") or {}
        engine_lines.append(
            f"  {name:<12} {build.get('tag', '?'):<22} -march={build.get('march', '?')}")

    simd = ", ".join(cpu.get("simd_flags") or []) or "unknown"
    return f"""vector-bench run: {run_id}

WHAT THIS IS
  A benchmark of built-in vector search across relational databases, all using
  an HNSW index, so the comparison is of implementations rather than of
  algorithms.

TO READ IT
  Open report/report.html in any browser. It is self-contained: the charts are
  inlined and nothing is fetched from the network, so it works offline and
  looks the same everywhere.

  Read its Validity section before the charts. It lists measurements that did
  not use the vector index, filtered queries that returned short, and anything
  about the environment that narrows what the numbers mean.

THE NUMBERS ARE SCOPED TO THIS MACHINE
  host            {host.get('hostname', 'unknown')}
  CPU             {cpu.get('model', 'unknown')}
  cores           {cpu.get('physical_cores', '?')} physical / {cpu.get('logical_cpus', '?')} logical{' (hybrid)' if cpu.get('hybrid') else ''}
  SIMD            {simd}
  AVX-512         {'yes' if cpu.get('has_avx512') else 'no'}
  RAM             {_fmt_bytes(host.get('total_ram_bytes'))}
  kernel          {host.get('kernel', 'unknown')}

  MariaDB MHNSW and AliSQL VIDX both document AVX-512 distance kernels. Results
  from a host without it do not transfer to one with it.

WHAT WAS MEASURED
  profile         {profile.get('name', '?')} — {profile.get('description', '')}
  resource pass   {config.get('resource_pass', '?')}
  datasets        {', '.join(datasets) or '?'}
  started         {manifest.get('started_at', '?')}
  status          {manifest.get('status', '?')}

  engines
{chr(10).join(engine_lines) if engine_lines else '    (none recorded)'}

FILES
  report/report.html    the report, self-contained
  report/report.md      the same in Markdown
  report/records.jsonl  every measurement, one flat JSON object per line
  report/charts/        the charts as SVG and PNG
  run-manifest.json     hardware, versions, resolved limits — the provenance
  ops-*.jsonl           raw ops-harness records
  mem-*.jsonl           server memory timeseries
{_excluded_block(excluded)}
IF YOU HAVE VECTOR-BENCH
  Extract this directory into results/ and it appears in the web UI with no
  further step:

      tar xzf {bundle_filename(run_id)} -C /path/to/vector-bench/results/
      ./run-benchmark.sh web

  Viewing works. REGENERATING the report on your machine does not produce the
  same thing: the recall measurements live in results/annb/, a sibling of this
  directory rather than part of it, and scoring them needs the dataset file.
  Neither is in here. Without them a regenerated report loses its recall
  section; with unrelated ann results present it reads those instead. The UI
  warns before it lets you.

  The report in here is the one that was generated where it was measured.
"""


def write_bundle(run_dir: str, out_path: str,
                 manifest: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
    """Tar the run directory with a README explaining it. Returns (ok, detail).

    Core dumps are the one thing in the run directory that does not travel --
    `_without_core_dumps` has the predicate and why it is shaped that way --
    and the README names each one with its size, which `_excluded_block`
    writes. The exclusion is the cheap half: an archive that quietly lacks a
    file its recipient believes is in there is worse than an archive too large
    to send, because the first one looks right.

    `detail` on success is `out_path` alone, and a size does not belong in it:
    `cli.cmd_export` passes `detail` to `os.path.getsize` and prints the size
    itself, so any decoration here raises `FileNotFoundError` in a caller that
    has already written a correct bundle.
    """
    run_dir = os.path.abspath(run_dir)
    if not os.path.isdir(run_dir):
        return False, f"no such run directory: {run_dir}"

    run_id = os.path.basename(run_dir.rstrip(os.sep))
    if manifest is None:
        try:
            with open(os.path.join(run_dir, "run-manifest.json")) as fh:
                manifest = json.load(fh)
        except (OSError, ValueError):
            return False, ("no readable run-manifest.json: a run without its "
                           "provenance is not worth sending")

    dropped: List[Tuple[str, int]] = []
    try:
        with tarfile.open(out_path, "w:gz") as archive:
            archive.add(run_dir, arcname=run_id,
                        filter=_without_core_dumps(run_id, dropped))
            # The README is rendered here, after the walk, because the walk is
            # what discovers the exclusions it has to disclose -- and added
            # last, which is where it already was.
            readme = readme_text(run_id, manifest, tuple(dropped))
            with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
                tmp.write(readme)
                tmp_path = tmp.name
            try:
                archive.add(tmp_path, arcname=os.path.join(run_id, README_NAME))
            finally:
                os.unlink(tmp_path)
    except (OSError, tarfile.TarError) as exc:
        return False, f"could not write {out_path}: {exc}"

    return True, out_path
