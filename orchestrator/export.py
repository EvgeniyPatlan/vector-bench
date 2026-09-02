"""Package a run for someone who does not have this checkout.

Two audiences, one archive. Someone who just wants to read the result opens
report/report.html, which is self-contained -- charts inlined, no network.
Someone who has vector-bench extracts the whole thing into results/ and gets
the run in their own UI, with its own machine's manifest attached.

The README is written into the archive because a tarball that arrives without
one is a tarball nobody opens.
"""

from __future__ import annotations

import json
import os
import tarfile
import tempfile
from typing import Any, Dict, Optional, Tuple

README_NAME = "README.txt"


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


def readme_text(run_id: str, manifest: Dict[str, Any]) -> str:
    """What the recipient is looking at, and what they can trust it for."""
    host = manifest.get("host") or {}
    cpu = host.get("cpu") or {}
    config = manifest.get("config") or {}
    profile = config.get("profile") or {}
    engines = manifest.get("engines") or {}

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
  datasets        {', '.join(profile.get('datasets') or []) or '?'}
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
    """Tar the run directory with a README explaining it. Returns (ok, detail)."""
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

    readme = readme_text(run_id, manifest)
    try:
        with tarfile.open(out_path, "w:gz") as archive:
            archive.add(run_dir, arcname=run_id)
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
