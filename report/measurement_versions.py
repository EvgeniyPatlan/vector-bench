"""Refusing to put two ops measurement versions on one chart.

Split out of `report/loaders.py`, which had grown to 793 lines against this
repo's 800-line ceiling: the next addition would have breached it, and the
file belongs to another change in this series, which would then have inherited
a refactor in the middle of unrelated work.

The job here is one thing. `extra.ops_measurement_version` says which version
of the ops measurement path produced a record, and `run-manifest.json` says
which version the run declares. Two versions in one report is not a chart with
a caveat, it is two different measurements drawn as one series -- so the report
refuses, and this module decides when and says why.

It is the ops counterpart of `ann_fingerprint`, which keeps a 16 GB recall
curve out of a report whose manifest says 64 GB by narrowing the shared ann
results tree. Ops records have no tree to narrow, so the equivalent is a
refusal at load time, before a chart, a dataset or the ann tree is touched.

Imported by `report/loaders.py` (which re-exports the public names, so older
`from report.loaders import ...` call sites keep working), `report/generate.py`
and `webui/records.py` -- the three readers that can put records in front of a
person. Like the rest of `report/`, it must not import `orchestrator`: this
package runs inside a bench image that mounts report/ and harness/ only.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

class MixedMeasurementVersions(Exception):
    """Ops records from two measurement versions were asked to share a report.

    What keeps this escaping `load_ops_records` is its POSITION -- it is raised
    after the per-file loop, not inside it. The handler in that loop is
    `except Exception`, so it would swallow this refusal whatever the class
    inherited from, and an earlier version of this docstring credited the base
    class with protection it does not provide.

    Deliberately NOT a ValueError, for a different and narrower reason.
    `read_records` raises ValueError for a malformed JSONL file, so the two
    would be indistinguishable to a caller that wraps these loads in
    `except (OSError, ValueError)` -- which `webui/setup.py`,
    `webui/importing.py` and `orchestrator/export.py` all already do elsewhere.
    No caller on this path does today; the point is that the next one cannot
    silently turn the guard off by being careful about file errors. Nothing is
    lost by not subclassing it: `webui/server.py` catches bare `Exception` and
    turns it into a 500 that names the fault.
    """


def ops_measurement_version(record: Dict[str, Any]) -> Any:
    """Which ops measurement version produced this record.

    An absent value is version 1, not "unknown": every ops record written
    before `OPS_MEASUREMENT_VERSION` existed was produced by the behaviour
    that constant calls 1, and treating them as unknown would make them
    comparable with everything.

    A value that is present but not an integer is returned as-is rather than
    coerced or dropped, so a hand-edited or truncated file is refused instead
    of quietly comparing equal to nothing. `"2"` is not `2`.
    """
    from harness.metrics.records import (IMPLICIT_OPS_MEASUREMENT_VERSION,
                                         OPS_MEASUREMENT_VERSION_KEY)

    extra = record.get("extra") or {}
    if OPS_MEASUREMENT_VERSION_KEY not in extra:
        return IMPLICIT_OPS_MEASUREMENT_VERSION
    return extra[OPS_MEASUREMENT_VERSION_KEY]


def _is_ops_record(record: Dict[str, Any]) -> bool:
    """Recall/QPS records come from the other measurement path.

    They are versioned by ANN_MEASUREMENT_VERSION, which the ann results tree
    already keys on, and they never carry an ops version. Counting them as
    version 1 would refuse every report that has both paths in it.
    """
    from harness.metrics.records import PHASE_RECALL

    return record.get("phase") != PHASE_RECALL


def ops_measurement_versions(
        records: List[Dict[str, Any]]) -> Dict[Any, List[Dict[str, Any]]]:
    """Ops records grouped by the measurement version that produced them."""
    grouped: Dict[Any, List[Dict[str, Any]]] = {}
    for record in records:
        if not _is_ops_record(record):
            continue
        grouped.setdefault(ops_measurement_version(record), []).append(record)
    return grouped


def _describe(record: Dict[str, Any]) -> str:
    """Enough of a record to find it in the directory it came from."""
    return (f"{record.get('engine')}/{record.get('dataset')}"
            f"/{record.get('phase')}"
            f" m={record.get('m')} {record.get('build_mode')}"
            f" pass={record.get('resource_pass')}"
            f" @{record.get('timestamp')}")


def refuse_mixed_measurement_versions(records: List[Dict[str, Any]], *,
                                      source: str,
                                      declared: Optional[Any] = None) -> None:
    """Refuse a record set that cannot honestly go on one chart.

    This is the ops-side counterpart of `ann_fingerprint`. There the report
    narrows the shared ann results tree to the configuration the manifest
    declares, so a 16 GB recall curve cannot appear under a 64 GB manifest.
    Ops records have no tree to narrow — they all land in the run directory —
    so the equivalent is a refusal, and it happens at load time, before a
    chart, a dataset or the ann tree is touched. A refusal at the end of
    report generation would be barely better than none.

    Refusing rather than labelling is a deliberate choice between the two the
    convention allows. A labelled chart still invites the comparison it is
    labelled against, and a reader who wants to compare two versions can
    generate two reports; nothing is lost that was trustworthy to begin with.
    The one outcome worse than either is a silent group-by that keeps the
    newer half and drops the older, which is what happens if nobody checks.

    Two conditions are refused:

    * more than one version among the records themselves — a resumed run that
      appended to a file written by older code, or an archived file copied in;
    * a single version that disagrees with the version the manifest declares —
      records copied in from elsewhere, or a manifest edited by hand. Nothing
      in the records contradicts itself in that case, which is the shape the
      ann incident took: the provenance above the numbers said 64 GB and the
      numbers were measured at 16. (A *resume* no longer produces this:
      `Manifest.set_config` carries an existing version forward instead of
      re-stamping it, because it is reached before the checkpoint skip and so
      ran even when a resumed run measured nothing at all.)
    """
    groups = ops_measurement_versions(records)
    if not groups:
        return

    # `_holds` rather than `in`, because `in` hashes `declared` and a
    # hand-edited manifest can hold a list or a dict there -- TypeError:
    # unhashable type, from a guard whose whole job is to survive malformed
    # provenance.
    unmatched = declared is not None and not _holds(groups, declared)
    if not unmatched and len(groups) < 2:
        return

    sides: List[Any] = sorted(groups.items(), key=lambda kv: _sort_key(kv[0]))
    if unmatched:
        # Appended as an empty side rather than merged into `groups`: there are
        # no records at the declared version, and the message must say that
        # rather than count "0 records" as one half of a conflict.
        sides.append((declared, []))
    # `declared` is handed on, but which branch describes the result is decided
    # from `sides` -- see _refusal_message. An unmatched manifest version used
    # to route to the single-version wording on its own, which was true when
    # the records agreed and four separate lies when they did not.
    raise MixedMeasurementVersions(
        _refusal_message(source, sides, declared if unmatched else None))


def _holds(groups: Dict[Any, Any], version: Any) -> bool:
    """`version in groups`, for a version that may not be hashable."""
    try:
        return version in groups
    except TypeError:
        return False


def _sort_key(version: Any) -> Any:
    """Integers in numeric order first, then anything malformed, by text."""
    return (0, version) if isinstance(version, int) else (1, repr(version))


#: Shared by both branches when every version involved denotes the same
#: measurement. Without it the refusal told operators to re-measure data
#: `docs/05-methodology.md` calls directly comparable -- and on versions 1 and
#: 2, the only pair that can occur today, that is every operator who resumes an
#: archived run and measures one more unit.
#:
#: The consequence clause is per-branch: "a report per version" is the remedy
#: when the records themselves differ, and is nonsense when they agree and only
#: the manifest above them disagrees.
_SAME_MEASUREMENT_FACT = (
    "These versions denote the same measurement: version 2 changed nothing "
    "measured, it only started recording the version. So none of these numbers "
    "need re-measuring and none of them is wrong"
)
_SAME_MEASUREMENT_MIXED = (
    _SAME_MEASUREMENT_FACT
    + " -- but the report prints one version above the numbers and cannot "
      "print two, so each version needs a report of its own."
)
_SAME_MEASUREMENT_MISMATCH = (
    _SAME_MEASUREMENT_FACT
    + " -- what is wrong is the label above them, not the numbers."
)


def _equivalent(versions: List[Any]) -> bool:
    from harness.metrics.records import ops_versions_measure_the_same_thing

    return ops_versions_measure_the_same_thing(versions)


def _mismatch_text(source: str, measured: Any, declared: Any,
                   versions: Optional[List[Any]] = None) -> Any:
    """Head and tail for records that agree with each other but not the manifest.

    Every sentence here is true only of a directory whose records really do
    agree, so `_refusal_message` must not route a multi-version set to it. See
    its docstring for what that produced.

    The remediation deliberately does not mention moving or dropping records.
    There is one version of them; removing it leaves the directory empty and
    the report exits with "no records found", which is a worse outcome than the
    refusal and was what the shared tail used to advise.

    `versions` is every version involved and defaults to the pair this branch
    describes. The equivalence claim is made over that whole set, never over
    one side of it.
    """
    if versions is None:
        versions = [measured, declared]
    head = [
        f"{source}: every ops record here was measured under version "
        f"{measured!r}, but the run manifest declares version {declared!r}.",
        "",
        "The records agree with each other. What disagrees is "
        "config.ops_measurement_version in run-manifest.json, and that is the "
        "provenance the report prints above the numbers -- the same failure as "
        "a manifest saying 64 GB above a curve measured at 16.",
    ]
    tail = ["Dropping records cannot fix this: there is only one version of "
            "them, and removing it leaves nothing to report.", ""]
    if _equivalent(versions):
        # No re-measurement clause at all: there is nothing to re-measure.
        tail += [
            _SAME_MEASUREMENT_MISMATCH,
            "",
            f"Set config.ops_measurement_version in run-manifest.json to "
            f"{measured!r}, which is what these records were written under, or "
            f"report them against the manifest that was written alongside "
            f"them.",
        ]
    else:
        tail += [
            f"If these numbers really were measured under version {measured!r},"
            f" the manifest is what needs correcting: set "
            f"config.ops_measurement_version in run-manifest.json to "
            f"{measured!r}, or report them against the manifest that was "
            f"written alongside them. If they genuinely predate version "
            f"{declared!r} and it changed what is measured, they have to be "
            f"re-measured before they can be compared with it.",
        ]
    return head, tail


def _mixed_records_text(source: str, versions: List[Any],
                        declared: Optional[Any] = None) -> Any:
    """Head and tail for a directory whose records disagree among themselves.

    `declared` is the manifest's version when no record carries it. The records
    disagreeing with each other is the headline problem, but the manifest is
    wrong too and saying so costs one sentence -- an operator who splits the
    records and regenerates would otherwise be refused a second time by a check
    the first message never mentioned.

    `versions` is every version involved, the declared one included, because
    that is what the equivalence claim is made over. Passing only the record
    versions evaluated {1, 2, 3} as {1, 2} and came back "equivalent", which
    asserted comparability across a real bump -- the one thing this wording
    must never claim by accident.
    """
    if _equivalent(versions):
        why = _SAME_MEASUREMENT_MIXED
    else:
        why = ("A measurement version changes what a number means. The same "
               "engine, dataset, M and ef_search measured under two of them "
               "are two different measurements, and plotting them as one "
               "series compares this harness against itself. This is the ops "
               "counterpart of ann_fingerprint keeping a 16 GB recall curve "
               "out of a 64 GB report.")
    head = [
        f"{source}: ops records from {len(versions)} different measurement "
        f"versions cannot go on one chart.",
        "",
        why,
    ]
    tail = [
        "Report each version on its own. The records above come from the "
        "ops-*.jsonl files in this run directory, or from the file "
        "--from-records named -- and one file can hold more than one version, "
        "because a resumed run appends to the file it already wrote -- so "
        "move or split them until one version is left, and generate one report "
        "per version.",
    ]
    if declared is not None:
        head.append("")
        head.append(
            f"The run manifest declares version {declared!r}, which none of "
            f"these records were measured under, so "
            f"config.ops_measurement_version in run-manifest.json is wrong as "
            f"well.")
        tail.append("")
        tail.append(
            f"Then set config.ops_measurement_version in run-manifest.json to "
            f"the version of the half you are reporting, because it currently "
            f"says {declared!r} and will refuse that half too.")
    return head, tail


def _refusal_message(source: str, sides: List[Any],
                     declared: Optional[Any] = None) -> str:
    """The text of a refusal: what is wrong, what is in the directory, what to do.

    `declared` is the manifest's version when no record carries it, and None
    when the manifest agrees or says nothing. Which branch describes the result
    is decided from `sides`, NOT from `declared` alone, and that distinction is
    the whole point of this function:

    * one record version plus a disagreeing manifest -> _mismatch_text. The
      records agree with each other; only the label above them is wrong.
    * two or more record versions -> _mixed_records_text, whether or not the
      manifest also disagrees. "A report per version" is the remedy here, and
      dropping one half is part of the fix rather than the end of the data.

    Routing on `declared` alone sent the second case to the first branch, where
    every sentence was written for a directory whose records agree. Over v1 and
    v3 records under a manifest declaring v2 it claimed every record was
    version 1, that the records agreed, that there was only one version of them
    and that removing it would leave nothing to report -- with version 3 listed
    three lines below -- and, because only sides[0][0] reached the equivalence
    predicate, asserted that none of the numbers needed re-measuring. The
    remedy it then gave refused again, so it looped.
    """
    measured = [side for side in sides if side[1]]
    all_versions = [v for v, _r in sides]
    head, tail = (
        _mismatch_text(source, measured[0][0], declared, all_versions)
        if declared is not None and len(measured) == 1
        else _mixed_records_text(source, all_versions, declared))

    lines = head + [""]
    for version, found in sides:
        if not found:
            lines.append(f"  version {version!r}: declared by the run "
                         f"manifest, no records measured under it")
            continue
        plural = "record" if len(found) == 1 else "records"
        lines.append(f"  version {version!r}: {len(found)} {plural}")
        for record in found[:3]:
            lines.append(f"      {_describe(record)}")
        if len(found) > 3:
            lines.append(f"      ... and {len(found) - 3} more")
    lines += [
        "",
        "Records with no extra.ops_measurement_version were written before the "
        "version existed and count as version 1.",
        "",
    ]
    return "\n".join(lines + tail)


def declared_ops_version(run_dir: str) -> Optional[Any]:
    """The ops measurement version this run's manifest claims, if any.

    Public because three readers need it, not one. `load_ops_records` passes it
    below, and `report/generate.py`'s --from-records branch and
    `webui/records.py`'s merged-file branch each pass it too -- they were
    calling the guard without it, so the manifest half of the check reached
    only the path that happens to live in this module. A report rebuilt with
    --from-records then printed version-2 provenance above version-1 numbers,
    which is the whole failure shape the check exists for, and --from-records
    is the first thing an operator reaches for after being refused.

    Read straight out of the JSON rather than through `orchestrator.manifest`:
    this package runs inside a bench image that mounts report/ and harness/
    only. A missing key means a run that predates the field — every archived
    run in this project — and is not a disagreement, so nothing is checked
    against it.
    """
    path = os.path.join(run_dir, "run-manifest.json")
    try:
        with open(path) as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    config = data.get("config") or {}
    if not isinstance(config, dict):
        return None
    return config.get("ops_measurement_version")
