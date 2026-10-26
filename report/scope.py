"""Which engines and corpora a run measured, and what the shared tree holds
that it did not.

THE DEFECT THIS EXISTS FOR
--------------------------
`results/annb/<pass>/<fingerprint>/` is shared across runs **on purpose**.
`ann_fingerprint` keys it on the resource pass's declared knobs and is
deliberately engine-invariant, so a seven-engine run writes one tree and
produces one seven-engine recall chart. Making it engine-specific fragments
the tree, and twice already that turned a six-engine smoke run into a recall
chart with one engine in it. Nothing here touches that layout, and nothing
here may: the narrowing belongs in the report's *record selection*.

The price of the shared tree is that it is not a record of any one run, and
the report read all of it. Measured on `percona-20261021-134142`, a run whose
manifest holds one key under `engines`, whose two `phases` entries are both
`percona` and both `completed`, and which finished in 112 seconds:
`report/records.jsonl` carried 396 records across seven engines and three
corpora -- alisql 18, mariadb 18, mariadb123 18, mongodb 26, percona 91,
pgvector 150, valkey 75. 388 of them came out of HDF5 files written on
2026-10-14, -16, -17, -19 and -20, and every one was stamped with this run's
`run_id`. §1 "Engines under test" named Percona Server alone -- correctly,
from the manifest -- directly above a §4 comparing it with six engines it was
never run beside, and beside `pareto-gist-960-euclidean.png` for a profile
whose `datasets:` names only fashion-mnist.

WHY THE MANIFEST IS THE ONLY SOURCE
-----------------------------------
`Manifest.set_engine` writes one key per engine in the resolved engine list,
once, before that engine's first unit runs -- so the `engines` block already
IS the resolved list, with any `--engines` narrowing applied. The report
container mounts report/ and harness/ only and must not import the
orchestrator, and a standalone `report --run-dir <archived bundle>` has no
flag to be given, so the manifest has to serve the archived case regardless.
Reading it in both cases means one reader rather than two, which is the same
reason `recorded_passes` is shared between the tree narrowing and the sections
that describe the run.

That is also the stronger of the two sources where they can disagree. A
`run --resume --engines percona` into a directory that previously measured
seven narrows the flag and not the block -- and the block is right: that
directory holds seven engines' ops records and this report describes all of
them. A flag would have cut six engines' recall curves out from under their
own ops rows.

WHAT IS NEVER NARROWED
----------------------
Path B. Every `ops-*.jsonl` is inside the run directory, written by this run,
so an ops record is this run's own measurement whatever the manifest's engine
block says -- and nothing else holds a copy of it. Only `recall_qps` records
are candidates here, because the shared tree is the only input this report has
that belongs to more than one run.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

#: The phase read out of the shared tree. Spelled rather than imported from
#: `harness.metrics.records`, in the voice `generate.summarize` uses for the
#: same literal: this module is pure logic over records and stays importable
#: on a host with nothing installed.
PHASE_RECALL = "recall_qps"

#: Why a record was left out, as a code the renderer turns into prose. Three
#: values rather than two, because an out-of-scope engine measured on an
#: out-of-scope corpus is not evidence of either on its own -- percona's own
#: gist and glove curves in the reference bundle are the in-scope engine on an
#: out-of-scope corpus, and naming that row "engine" would send a reader
#: looking for an engine nobody ran.
OUT_OF_SCOPE_ENGINE = "engine"
OUT_OF_SCOPE_DATASET = "dataset"
OUT_OF_SCOPE_BOTH = "both"


def _phase_pairs(manifest: Dict[str, Any]) -> List[Tuple[Any, Any]]:
    """(engine, dataset) for every unit the phase log records.

    `add_phase` is appended once a unit returns, whatever its exit code, so a
    FAILED unit is here and an unfinished one is not. Both halves matter and
    they pull in opposite directions; see `measured_datasets`.
    """
    return [(phase.get("engine"), phase.get("dataset"))
            for phase in (manifest.get("phases") or [])
            if isinstance(phase, dict)]


def _ordered(values: Iterable[Any]) -> List[str]:
    """Distinct, non-empty, sorted as text.

    Sorted rather than first-seen, because these sets are printed in the
    report and compared in tests, and `engines` is a JSON object whose key
    order is whatever the file happened to be written in.
    """
    return sorted({str(v) for v in values if v})


def measured_engines(manifest: Dict[str, Any]) -> List[str]:
    """Every engine this run configured or ran, or [] if it cannot be told.

    The UNION of the `engines` block and the phase log, and both terms do
    work.

    The block is the resolved engine list: `Manifest.set_engine` writes a key
    per engine before its first unit, so it carries an `--engines` narrowing
    and it survives a run that died before any unit finished.

    The phase log catches the other direction. The block is written by the
    orchestrator at configure time and a hand-edited, truncated or
    partially-copied manifest can lose a key whose unit the log still records;
    an engine with a phase in the log unquestionably ran, and dropping its
    measurements would be the over-narrowing this whole change must not cause.

    **A FAILED phase counts, and that is a decision and not an oversight.**
    `add_phase` records a unit whatever its exit code, and an engine whose
    every unit failed must stay in scope so that its absence from the charts
    surfaces as §2's "Phases that FAILED" row rather than as an engine that
    quietly is not there. `full-20261015-215006` is that run: 16 percona
    phases, 7 completed and 9 failed, and nothing else started. Filtering on
    `status == "completed"` would have scoped out whichever engine the reader
    most needs told about.

    **`config.profile.engines` is deliberately NOT a term**, and it is the
    wrong concept rather than merely a weaker one. It is what the profile
    declares, not what ran, and all four archived bundles say so:

    * `percona-20261020-140151` declares `engines: [percona]` and genuinely
      started and completed all seven, because it predates the fix that made
      the profile key restrict a run. Its `engines` block and its phase log
      both hold seven. A scope taken from the declaration would delete six
      engines of real measurement from the one report whose breadth is
      correct.
    * both `full-20261015-215006` bundles declare **nothing** -- the `full`
      profile has no `engines:` key, so the value is `None`. Reading "declared
      nothing" as "every known engine" would re-admit the whole contamination
      for precisely the profiles that cost hundreds of hours to run, and one
      of those two bundles started percona alone while the other started all
      seven. Only the block and the log tell them apart.
    * and `--engines` narrows a run without touching the profile, so the
      declaration cannot narrow what it does not know about.

    An empty return means "this manifest does not say", not "this run
    measured nothing"; `narrow_to_measured_scope` leaves that axis alone.
    """
    return _ordered(list(manifest.get("engines") or {})
                    + [engine for engine, _dataset in _phase_pairs(manifest)])


def measured_datasets(manifest: Dict[str, Any]) -> List[str]:
    """Every corpus this run ran, or [] if it cannot be told.

    **The phase log, for the same reason the engines axis uses it, and
    `config.profile.datasets` is NOT a union term.** It is the corpus-side
    twin of `config.profile.engines`: a declaration the orchestrator is free
    to override, not a record of this run. `--datasets` on the `run` command
    narrows the run and leaves the declaration alone, because `cli.py` builds
    a local list from the flag and hands `set_config` the profile dict
    unchanged -- so `run --profile full --datasets fashion-mnist-784-euclidean`
    leaves a manifest still declaring four corpora, and a union would publish
    the other three out of the shared tree. That is the defect this module
    exists for, surviving its own fix. The phase log records what each unit
    actually measured and so carries the flag for free.

    The declared list is a FALLBACK and only when the phase log names no
    corpus at all -- a run killed before its first unit returned. There is
    then no per-run record to prefer, and narrowing to what the profile asked
    for is better than not narrowing at all. It is not a second opinion
    alongside the log; it is what is left when there is no log.

    **What this costs, deliberately.** `add_phase` is appended after a unit
    returns, so a run killed inside its SECOND corpus has no entry for that
    corpus while ann-benchmarks may already have written result files for it.
    Those files are excluded -- and named in §2, with the count and the
    reason, which is the whole point of the census. The alternative is to
    admit them, and nothing in the tree distinguishes a half-finished file
    this run wrote from one another run wrote weeks ago: the shared tree has
    no run id in it. Publishing another run's curve is worse than printing a
    row that says four result files for a corpus were left out, so the
    conservative reading wins. A FAILED unit is a different case and is kept,
    because `add_phase` records it: see `measured_engines`.

    Measured on all four archived bundles, the two readings are
    indistinguishable -- `config.profile.datasets` and the set of corpora in
    `phases[]` are equal in every one of them, including both `full` runs,
    whose profile declares four corpora and whose phase log names the same
    four. So this choice costs nothing on the archives and closes the
    `--datasets` hole on the next run.
    """
    from_phases = _ordered(dataset
                           for _engine, dataset in _phase_pairs(manifest))
    if from_phases:
        return from_phases
    profile = ((manifest.get("config") or {}).get("profile") or {})
    return _ordered(profile.get("datasets") or ())


def _reason(engine_ok: bool, dataset_ok: bool) -> str:
    if engine_ok:
        return OUT_OF_SCOPE_DATASET
    if dataset_ok:
        return OUT_OF_SCOPE_ENGINE
    return OUT_OF_SCOPE_BOTH


def narrow_to_measured_scope(
        records: Sequence[Dict[str, Any]],
        manifest: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """`(kept, excluded)` -- the records this run measured, and a census.

    Nothing is mutated: both lists hold the record objects as handed in, and
    the census is built fresh.

    A `recall_qps` record is dropped when its engine or its corpus is outside
    what the manifest says this run measured. Every other phase is kept
    whatever it names, because every other phase comes out of the run
    directory. A record naming no engine or no corpus is kept too: it cannot
    be attributed, and this function's companion requirement is that nothing
    leaves the report unnamed.

    An axis the manifest cannot describe is not narrowed. Reading too much is
    the defect; reading nothing would delete the whole recall half of a report
    for every archived manifest written before the `engines` block carried the
    resolved list, which is worse. `generate.main` says so on stdout when it
    happens, and §2's staleness notice is the guard that still fires there.

    **Two independent axes, not the set of (engine, corpus) pairs the phase
    log holds.** Pairs would be tighter -- a run that measured A on X and B on
    Y does not vouch for A on Y -- and they are not available: the declared
    `datasets:` list is per run and not per engine, so half the evidence
    cannot form a pair at all, and a pair taken from the phase log alone
    reintroduces exactly the interruption hole `measured_datasets` exists to
    close, once per engine. The product is the widest reading both sources can
    express, and it is still narrow enough to empty the shared tree of every
    other run: on the two bundles behind this change it drops 353 of 396
    records and 48 of 446 respectively, and leaves both reports describing
    only what their own run did.

    The census is aggregated per (engine, corpus) and counts FILES as well as
    records: `extra.source_file` is the result file's path inside the tree, so
    the file count is the figure an operator can check against the directory.
    A dropped record with no `source_file` -- a hand-assembled or third-party
    records file -- counts towards the records and not the files, rather than
    being counted as a file that does not exist.

    The file count is of distinct `(resource pass, source file)` pairs and not
    of records carrying a path, which is one file on disk either way. `source
    file` is relative to the tree it was read from and the trees are per pass,
    so one build configuration measured in both passes yields the SAME
    relative path twice -- the collision `build_signature` is documented for.
    Counting paths alone would report two files as one; counting records would
    report two records of one file as two.

    It is derived from records the loader actually read and the report would
    otherwise have published, not from a directory listing. That is the same
    distinction `summarize` draws for the missing-pass notice, and for the
    same reason: a claim about what was excluded has to be a claim about
    measurements, and a file the loader could not read for want of ground
    truth is a different fact with a different remedy.
    """
    engines = set(measured_engines(manifest or {}))
    datasets = set(measured_datasets(manifest or {}))
    kept: List[Dict[str, Any]] = []
    # Keyed by the pair, so one row per (engine, corpus) rather than one per
    # result file: the reference bundle would otherwise print 353 rows.
    census: Dict[Tuple[str, str], Dict[str, Any]] = {}
    # Outside the rows, so no row carries a set into the summary the renderer
    # reads.
    seen_files: Dict[Tuple[str, str], set] = {}
    for record in records:
        engine = record.get("engine")
        dataset = record.get("dataset")
        engine_ok = not engines or not engine or str(engine) in engines
        dataset_ok = not datasets or not dataset or str(dataset) in datasets
        if (record.get("phase") != PHASE_RECALL
                or (engine_ok and dataset_ok)):
            kept.append(record)
            continue
        key = (str(engine), str(dataset))
        row = census.setdefault(key, {
            "engine": str(engine), "dataset": str(dataset),
            "records": 0, "files": 0,
            "reason": _reason(engine_ok, dataset_ok),
        })
        row["records"] += 1
        source = (record.get("extra") or {}).get("source_file")
        if source:
            seen_files.setdefault(key, set()).add(
                (str(record.get("resource_pass") or ""), str(source)))
    for key, row in census.items():
        row["files"] = len(seen_files.get(key, ()))
    excluded = sorted(census.values(),
                      key=lambda row: (-row["records"], row["engine"],
                                       row["dataset"]))
    return kept, excluded


def excluded_file_count(excluded: Sequence[Dict[str, Any]]) -> int:
    """Result files in the shared tree that belonged to other runs."""
    return sum(row.get("files") or 0 for row in excluded)


def excluded_record_count(excluded: Sequence[Dict[str, Any]]) -> int:
    return sum(row.get("records") or 0 for row in excluded)
