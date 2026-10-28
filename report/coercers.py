"""Turning an attribute of unknown type into a record field, or into nothing.

Split out of `report/loaders.py` for the reason
`report/measurement_versions.py` was split out of it before: that file sits
against this repo's 800-line ceiling, the extraction had already been taken
once at 793 lines so that "the next addition would not breach it", and the
next addition arrived. Four functions that convert and never interpret are the
part of that file with no connection to either measurement path, so they are
the part that leaves.

**Why a coercer at all, rather than reading the attribute straight.** The ann
side of `loaders.py` reads HDF5 attributes through h5py, which hands back
numpy scalars rather than Python ones, and the record those land in is written
to `report/records.jsonl` by `report/generate.py` with
`json.dumps(..., default=str)`. That `default=str` is the hazard, and it is
not the one it looks like:

* `numpy.float64` is a subclass of `float`, so it serialises as a JSON number
  and nothing is wrong.
* `numpy.float32` is **not** a subclass of `float`. `json.dumps` cannot
  serialise it, so `default=str` catches it and writes the JSON **string**
  `"283.27"` -- at full width, silently, with no error anywhere. A later
  `report --from-records` reads that back as a `str`, and a measurement whose
  type depends on which numpy width a result file happened to use is a
  measurement a chart will sort as text or drop on a comparison.

So nothing in this file exists to prevent a crash. It exists because the
writer downstream cannot crash, which is worse: `float()` and `int()` here fix
the type at the boundary, where the value is still next to the file it came
from. Whether h5py returns a 32- or 64-bit scalar is not something a record
field should carry.

Returning `None` rather than raising is the other half. `load_annb_results`
wraps one result FILE in `except Exception`, so a coercer that raised on one
unreadable attribute would cost the recall point beside it. A value this
cannot convert is a value the file does not usefully hold, and the caller
decides whether that means an absent key or a `None` column.

Like the rest of `report/`, this must not import `orchestrator`: the package
runs inside a bench image that mounts `report/` and `harness/` only.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy


def maybe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def maybe_float(value: Any) -> Optional[float]:
    """`maybe_int`'s float sibling, down to the two exceptions it swallows.

    A plain `float`, whatever width of numpy scalar arrives -- see the module
    docstring for what `default=str` does to the one that is not a `float`
    subclass. NaN is NOT filtered here: it converts cleanly and is a value the
    file genuinely holds, so whether it belongs in a record is the caller's
    decision and `load_annb_results` makes it explicitly.
    """
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def maybe_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def maybe_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, (bool, numpy.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("true", "1"):
        return True
    if text in ("false", "0"):
        return False
    return None
