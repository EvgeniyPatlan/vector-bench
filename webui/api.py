"""JSON API handlers.

Handlers are plain functions over an Api context and return
(status, payload) so they can be tested without a socket.
"""

from __future__ import annotations

import os
import re
from typing import Any, Callable, Dict, List, Optional, Pattern, Tuple

from . import records as records_mod
from . import runs as runs_mod

#: Cap on records returned in one response; the UI aggregates server-side for
#: anything larger.
MAX_RECORDS = 5000

Response = Tuple[int, Any]


class Api:
    def __init__(self, root: str, allow_control: bool = False):
        self.root = root
        self.results_dir = os.path.join(root, "results")
        self.config_dir = os.path.join(root, "config")
        self.datasets_dir = os.path.join(root, "datasets")
        self.allow_control = allow_control
        self._jobs = None

    @property
    def jobs(self):
        if self._jobs is None:
            from orchestrator.cli import KNOWN_ENGINES
            from .jobs import JobStore
            self._jobs = JobStore(self.root, KNOWN_ENGINES)
        return self._jobs


def _run_dir_or_error(api: Api, run_id: str) -> Tuple[Optional[str], Optional[Response]]:
    run_dir = runs_mod.resolve_run_dir(api.results_dir, run_id)
    if run_dir is None:
        return None, (404, {"error": f"no such run: {run_id}"})
    return run_dir, None


def health(api: Api, _m, _q, _b=None) -> Response:
    return 200, {
        "ok": True,
        "root": api.root,
        "control_enabled": api.allow_control,
    }


def list_runs(api: Api, _m, _q, _b=None) -> Response:
    return 200, {"runs": runs_mod.discover_runs(api.results_dir)}


def get_run(api: Api, match, _q, _b=None) -> Response:
    run_id = match.group("run_id")
    run_dir, error = _run_dir_or_error(api, run_id)
    if error:
        return error
    manifest = runs_mod.load_manifest(run_dir)
    if manifest is None:
        return 404, {"error": f"unreadable manifest for {run_id}"}
    return 200, {
        "summary": runs_mod.summarize(run_id, run_dir, manifest),
        "manifest": manifest,
    }


def get_facets(api: Api, match, _q, _b=None) -> Response:
    run_dir, error = _run_dir_or_error(api, match.group("run_id"))
    if error:
        return error
    recs, source = records_mod.load_records(run_dir)
    return 200, {
        "source": source,
        "total": len(recs),
        "facets": records_mod.facets(recs),
        "measures": records_mod.available_measures(recs),
    }


def get_records(api: Api, match, query, _b=None) -> Response:
    run_dir, error = _run_dir_or_error(api, match.group("run_id"))
    if error:
        return error
    recs, source = records_mod.load_records(run_dir)
    selected = records_mod.filter_records(recs, query)
    return 200, {
        "source": source,
        "total": len(recs),
        "matched": len(selected),
        "truncated": len(selected) > MAX_RECORDS,
        "records": selected[:MAX_RECORDS],
    }


def get_series(api: Api, match, query, _b=None) -> Response:
    run_dir, error = _run_dir_or_error(api, match.group("run_id"))
    if error:
        return error

    x = (query.get("x") or ["recall_at_k"])[0]
    y = (query.get("y") or ["qps"])[0]
    group_by = [g for g in (query.get("group_by") or ["engine"])[0].split(",") if g]
    if not group_by:
        group_by = ["engine"]

    filters = {k: v for k, v in query.items()
               if k not in ("x", "y", "group_by")}
    recs, _source = records_mod.load_records(run_dir)
    selected = records_mod.filter_records(recs, filters)
    return 200, {
        "x": x,
        "y": y,
        "group_by": group_by,
        "matched": len(selected),
        "series": records_mod.series(selected, x, y, group_by),
    }


ROUTES: List[Tuple[str, Pattern, Callable[..., Response]]] = [
    ("GET", re.compile(r"^/api/health$"), health),
    ("GET", re.compile(r"^/api/runs$"), list_runs),
    ("GET", re.compile(r"^/api/runs/(?P<run_id>[^/]+)$"), get_run),
    ("GET", re.compile(r"^/api/runs/(?P<run_id>[^/]+)/facets$"), get_facets),
    ("GET", re.compile(r"^/api/runs/(?P<run_id>[^/]+)/records$"), get_records),
    ("GET", re.compile(r"^/api/runs/(?P<run_id>[^/]+)/series$"), get_series),
]


def extend(routes) -> None:
    """Append additional routes (the control surface adds its own)."""
    ROUTES.extend(routes)


def dispatch(api: Api, method: str, path: str,
             query: Dict[str, List[str]],
             body: Optional[Dict[str, Any]] = None) -> Optional[Response]:
    """Resolve a request, or None when no route matches."""
    matched_path = False
    for route_method, pattern, handler in ROUTES:
        match = pattern.match(path)
        if not match:
            continue
        matched_path = True
        if route_method == method:
            return handler(api, match, query, body)
    if matched_path:
        return 405, {"error": f"method not allowed: {method}"}
    return None
