"""What a run will cost, before it starts.

Two estimates, accumulated separately and printed together:

* **Ingest** -- rows / measured rate, per engine, per dataset, multiplied by
  the M grid and the resource passes. This is the original estimate and its
  numbers are published: `docs/07-planning-a-run.md` quotes them, and the web
  UI renders `total_hours` as "estimated ingest". It does not move.
* **Query time** -- a registry of named terms (`_QUERY_TERMS`), each derived
  from keys the profiles already carry, with a typical figure and a ceiling
  carried separately wherever a budget caps the work.

Lifted out of `orchestrator/cli.py` once the second estimate arrived: `cli.py`
was 1,604 lines before this work and 2,056 with query time added inline,
against this repo's 800-line guideline. Those two are git facts and cannot
rot; a figure for what `cli.py` measures today would, and the first version of
this sentence went stale inside the same commit that wrote it. Most of what is
here is new rather than moved -- the ingest tables and `estimate_load_hours`
came across, the query half was written for it. `cli.py` re-exports everything
that was importable from it before, so `webui/control.py` and
`tests/test_config.py` resolve unchanged.

Nothing here starts a container, reads a run directory or touches the network.
It is arithmetic over a profile, which is what makes it testable
(`tests/test_estimate.py`) and cheap enough to call from the web UI's request
path on every form change.
"""

from __future__ import annotations

from typing import (Any, Callable, Dict, List, NamedTuple, Optional, Sequence,
                    Tuple)

# Imported for `MARIADB_ENGINES`, a two-element tuple, and nothing else. It is
# declared in ann_pass.py and belongs in engines.py with the rest of the
# registry; moving it needs both of those files, which this task does not own.
# It costs this module nothing in practice: the web UI's estimate handler
# already imports orchestrator.cli for KNOWN_ENGINES, so ann_pass and
# docker_ctl are on that path either way.
from . import ann_pass

# Measured ingest rates, rows/s, on a Xeon Gold 6230.
#
# These were originally guessed at 400/150/5600 from a 60k-row smoke run. A
# real 1.18M-row load measured MariaDB at 147 rows/s average — 2.7x slower —
# because HNSW insert cost rises as the graph grows and rises again with M
# (glove-100: ~320 rows/s at M=8, ~70 at M=32). An estimate that is optimistic
# by 3x is worse than none, so these are the observed mid-grid numbers.
_INGEST_ROWS_PER_S = {"mariadb": 150, "mariadb123": 110, "alisql": 55,
                      "pgvector": 3000, "valkey": 40000, "mongodb": 60000}
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
    ("mariadb", "dbpedia-openai-1000k-angular"): 74,
    ("mariadb123", "dbpedia-openai-1000k-angular"): 51,
    ("alisql", "dbpedia-openai-1000k-angular"): 43,
    # The three that do not maintain the graph on the write path. Their rates
    # are the load alone; the index build is separate and is not what this
    # estimate is for. Two to three orders of magnitude above the InnoDB
    # engines, which is the point.
    ("pgvector", "dbpedia-openai-1000k-angular"): 617,
    ("valkey", "dbpedia-openai-1000k-angular"): 30122,
    ("mongodb", "dbpedia-openai-1000k-angular"): 71182,
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


# --- query time -------------------------------------------------------------
#
# The ingest estimate is ingest only, and stays that way: it is what its label
# says, docs/07-planning-a-run.md quotes its numbers, and two web-UI renderers
# publish `total_hours` as "estimated ingest". What follows is a *second*,
# separately accumulated estimate of time spent querying, so the printed output
# can show ingest, query time and a total -- and so "a run longer than its
# printed estimate is a bug" can be a true statement about something. Against
# the ingest figure alone it never was: that figure explicitly excludes every
# query, so every run that ever ran exceeded it.
#
# Two rules hold this together, and both exist because an over-confident
# estimate is worse than none:
#
#   * A cost that is a *cap* is reported as a cap. A churn budget of 1800 s is
#     a ceiling the workload may not reach, so the typical figure and the
#     ceiling are carried separately rather than collapsed into one number.
#   * A cost this repo has no measurement for is *named*, not guessed and not
#     scored zero -- the same treatment `datasets_without_estimate` already
#     gives a dataset with no row count.

# Single-client query rate per (engine, dataset), for the workloads whose cost
# is a number of queries rather than a span of time.
#
# Deliberately empty. No run in this repo has recorded a per-(engine, dataset)
# QPS anywhere this estimator can read, and the ingest table has already paid
# for the alternative: rates guessed off a 60,000-row smoke run were optimistic
# by 2.7x against the first real 1.18M-row load, which docs/07 calls worse than
# having no estimate at all. So every engine missing from here is reported in
# `engines_without_query_rate`, and the query-count costs are reported as not
# modelled. Add a measured figure and that engine drops out of the gap list;
# a term that needs a rate reads it from `plan.query_qps`.
_MEASURED_QUERY_QPS: Dict[Tuple[str, str], float] = {}

# What costs a number of queries rather than a span of time, keyed by the ops
# workload that pays it. Each is named in the output while no rate exists.
#
# Filtered search is deliberately NOT here: it has a term of its own, which
# carries both the shape of the cost and the fact that nothing has measured it.
# Naming it in both places would have the printed output charge for it and
# disclaim it in the same breath.
_PER_QUERY_OPS_COSTS: Dict[str, Tuple[str, ...]] = {
    "churn": (
        "churn re-measurement: ops.max_queries before churn and again after "
        "each churn fraction",
        # churn_budget_s bounds _reinsert and nothing else: churn.py wraps
        # driver.delete_ids in a Heartbeat and a Timer with no budget, so a
        # quarter of a million tombstones is an open-ended cost. The term above
        # is named "churn re-insert" for exactly this reason, and saying so
        # here stops the printed caveat reading as "churn is fully bounded".
        "churn delete: ops.churn_fractions of the corpus removed per step, "
        "with no budget bounding the delete itself",
    ),
    "concurrency": (
        "concurrency warm-up: 50 queries per client before each measured "
        "window",
    ),
}
_ANN_QUERY_COST = ("recall/QPS sweep: every ann.ef_search point, `runs` times, "
                   "over each dataset's query set")


def _query_qps(engine: str, dataset: str) -> Optional[float]:
    """Measured single-client QPS, or None -- never a guess."""
    return _MEASURED_QUERY_QPS.get((engine, dataset))


def _ops_rows(ops: Dict[str, Any], dataset: str) -> int:
    """Rows the ops phase actually loads, which is what churn works on.

    `ops.subset_rows` caps it (smoke loads 20,000 of fashion-mnist's 60,000).
    A dataset with no row count and a subset is bounded by the subset, which is
    an upper bound rather than a guess; one with neither is 0, and is already
    reported in `datasets_without_estimate`.

    Note the asymmetry with the ingest estimate, which does **not** read
    `subset_rows` and charges smoke the full 60,000 rows for both phases. That
    is pre-existing and deliberately frozen -- the ingest numbers were not
    allowed to move when query time was added -- so smoke's ingest figure is
    about three times its churn figure for the same corpus, and the difference
    is this key, not an error in either.
    """
    rows = int(_DATASET_ROWS.get(dataset, 0))
    subset = ops.get("subset_rows")
    if subset:
        return min(rows, int(subset)) if rows else int(subset)
    return rows


def _ops_units(profile: Dict[str, Any], engine: str) -> int:
    """How many times the ops phase runs for one (engine, dataset, pass).

    `_run_unit` loops ops M values x build modes x storage engines, and every
    combination pays the query workloads again. pgvector is the one that bites
    today: `main` measures both `pgvector_build_modes`, so pgvector runs every
    ops workload twice per dataset.

    The storage-engine axis is read from the profile only. The tuned pass can
    widen it through `resources.extras.mariadb_storage_engines`, which this
    function is not given -- `_ops_storage_engines` takes `resources` and the
    estimate does not. Today every configured value is `[InnoDB]`, so nothing
    is lost; a profile that widens it would under-report here.
    """
    ops = profile.get("ops", {}) or {}
    ann = profile.get("ann", {}) or {}
    m_values = max(1, len(ops.get("m_values", [16])))
    if engine == "pgvector":
        build_modes = max(1, len(ann.get("pgvector_build_modes", ["post"])))
    else:
        build_modes = 1
    if engine in ann_pass.MARIADB_ENGINES:
        storage = max(1, len(ops.get("storage_engines") or ["InnoDB"]))
    else:
        storage = 1
    return m_values * build_modes * storage


class _QueryPlan(NamedTuple):
    """Everything a query-time term is given, for one (engine, dataset).

    `units` is `_ops_units` for this engine -- ops M values x build modes x
    storage engines, i.e. how many times the ops workloads run per (dataset,
    pass). `passes` is the number of resource passes. A term with
    scope="unit" is multiplied by both; see `_QueryTerm.scope`.
    """
    profile: Dict[str, Any]
    ops: Dict[str, Any]
    ann: Dict[str, Any]
    workloads: List[str]
    phases: List[str]
    k: int
    engine: str
    dataset: str
    rows: int
    write_rows_per_s: float
    query_qps: Optional[float]
    units: int
    passes: int


class _QueryTerm(NamedTuple):
    """One named contribution to query time.

    `typical_s` is what the configuration should cost; `worst_s` is what it
    cannot exceed. They differ whenever a budget caps the work, and the gap is
    the honest part of the estimate -- see `_churn_reinsert` for why collapsing
    them would have hidden two lost runs.

    `bounded` says whether `worst_s` is a real ceiling. A term with nothing
    capping it must say so, because the printed worst case is the number the
    "a run longer than its estimate is a bug" criterion refers to.

    `scope` says what the seconds are per. A term is always asked once per
    (engine, dataset) -- that is what a `_QueryPlan` describes -- so the scope
    only says how many times the cost recurs *within* that:

        "unit"  (default) once per ops unit, per pass: x units x passes.
                Everything the ops harness runs is per unit, because
                `_run_unit` reloads and re-measures for every combination of
                ops M value, build mode and storage engine.
        "pass"  once per resource pass: x passes. For work done once against
                an (engine, dataset) however many ops units follow.

    There is deliberately no whole-run scope. The accumulator cannot offer one
    honestly -- it would have to decide which (engine, dataset) iteration owned
    it -- and a cost that is genuinely independent of engine and dataset does
    not belong in a per-engine estimate.
    """
    name: str
    typical_s: float
    worst_s: float
    basis: str
    bounded: bool
    scope: str = "unit"


def _scope_factor(plan: _QueryPlan, scope: str) -> int:
    if scope == "unit":
        return plan.units * plan.passes
    if scope == "pass":
        return plan.passes
    raise ValueError(
        f"unknown query-term scope: {scope!r}. A term is asked once per "
        f"(engine, dataset); scope says how often the cost recurs inside "
        f"that -- 'unit' (x ops units x passes) or 'pass' (x passes).")


def _concurrency_windows(plan: _QueryPlan) -> Optional[_QueryTerm]:
    """The measured windows of the concurrency workload.

    Exact, and the only term here that is: each point runs for a fixed span
    whatever the engine manages, which is why the workload is written that way
    ("a slow configuration cannot stretch the run indefinitely and every point
    is measured over an identical window").

    `ops.concurrency_repeats` is already honoured by `ops_pass.harness_args`
    and defaults to 1, so a profile that sets it today gets that many windows
    per point and this has to count them.

    The per-client warm-up is not here: it is 50 queries, and a query count
    needs a rate this repo does not have. See `_PER_QUERY_OPS_COSTS`.
    """
    if "concurrency" not in plan.workloads:
        return None
    counts = [int(c) for c in plan.ops.get("client_counts", [1])]
    # The workload skips any point with more clients than it has queries.
    max_queries = int(plan.ops.get("max_queries", 1000))
    points = [c for c in counts if c <= max_queries]
    duration = float(plan.ops.get("concurrency_duration_s", 20))
    repeats = max(1, int(plan.ops.get("concurrency_repeats", 1)))
    if not points or duration <= 0:
        return None
    seconds = len(points) * duration * repeats
    return _QueryTerm(
        name="concurrency windows",
        typical_s=seconds,
        worst_s=seconds,
        basis=(f"{len(points)} client count(s) x {duration:.0f}s"
               + (f" x {repeats} repeat(s)" if repeats > 1 else "")
               + " per ops unit, fixed by the profile not by the engine"),
        bounded=True,
    )


def _filtered_search(plan: _QueryPlan) -> Optional[_QueryTerm]:
    """Filtered search: a fixed number of queries per selectivity, uncapped.

    The shape is exact and the duration is unknown, and both facts are printed.
    What runs is `len(ops.selectivities)` points x one iterative-scan mode x
    `ops.max_queries` queries per ops unit. What it costs depends on a rate
    nothing here has measured, and no budget caps it, so this term reports
    `bounded=False` and the run total says it is not yet a ceiling. That is the
    honest state: on MariaDB at 1% selectivity the measured rate was 0.9 QPS
    against 326 unfiltered, so 2,000 queries cost 37 minutes for one point --
    a 360x spread that no default could stand in for.

    It is a term rather than a line in `_PER_QUERY_OPS_COSTS` because it is
    where the two things that would quantify it must land. A figure in
    `_MEASURED_QUERY_QPS` turns `typical_s` into a duration, below, and needs
    nothing else: a rate is an input to the estimate, not a behaviour of the
    run.

    A ceiling is different, and this function deliberately does **not** read an
    `ops.filtered_budget_s`. Such a key exists nowhere else -- not in a profile,
    not in `ops_pass.harness_args`, not in `harness/main.py`, not in
    `filtered.py` -- so honouring it here would report `bounded=True`, print a
    concrete ceiling and drop the "not yet a ceiling" caveat while nothing
    capped anything. Contrast `ops.churn_budget_s`, which `_churn_reinsert` is
    right to trust: `ops_pass.py` passes it and `churn.py` enforces it. T3c
    adds the flag, the plumbing and the enforcement, and then adds the branch
    here; `tests/test_estimate.py` cross-checks the two halves against each
    other so whichever moves first is told about the other.

    One iterative-scan mode is counted, and it is not a factor in the
    arithmetic: `cli.py` pins the first declared
    `ops.pgvector_iterative_scan` value and the workload runs that one mode, so
    a `* modes` here could only ever be `* 1`. It was written that way and no
    fixture could set it to anything else, which made the promise it carried
    untestable. When T3c loops the modes inside
    `harness/workloads/filtered.py`, multiply `points` below by the mode count
    and name it in `shape`; the factor arrives with the test that can move it.
    """
    if "filtered" not in plan.workloads:
        return None
    # Defaulted exactly as ops_pass.harness_args defaults it. A profile that
    # omits the key still runs one selectivity, so charging nothing here would
    # drop a real cost out of the terms list *and* out of the disclosure list.
    selectivities = list(plan.ops.get("selectivities") or [0.1])
    max_queries = int(plan.ops.get("max_queries", 1000))
    points = len(selectivities)
    queries = points * max_queries
    shape = (f"{points} "
             f"{'selectivity' if points == 1 else 'selectivities'}"
             f" x {max_queries:,} queries = {queries:,} queries per ops unit, "
             f"one iterative-scan mode")

    if plan.query_qps:
        seconds = queries / plan.query_qps
        return _QueryTerm(
            name="filtered search", typical_s=seconds, worst_s=seconds,
            basis=(f"{shape} at a measured {plan.query_qps:,.1f} QPS; nothing "
                   f"caps it, so this is a figure and not a ceiling"),
            bounded=False)
    return _QueryTerm(
        name="filtered search", typical_s=0.0, worst_s=0.0,
        basis=(f"{shape}, at a rate nothing here has measured and with no "
               f"budget anywhere in the run to cap it"),
        bounded=False)


def _churn_reinsert(plan: _QueryPlan) -> Optional[_QueryTerm]:
    """Writing the churned rows back, which is the churn workload's long pole.

    `ops.churn_fractions` are cumulative: [0.10, 0.25] deletes and re-inserts
    10% and then a further 15%, for 25% in total rather than 35%.

    The typical figure uses the engine's *load* rate, the only write rate this
    repo has measured. The worst case is `ops.churn_budget_s` per step, and the
    distance between the two is the whole reason the budget exists: Valkey
    loaded this corpus at 30,195 rows/s and then wrote the same rows back at
    86, because pipelined writes into a populated index do not return. Two runs
    were lost waiting on it. An estimate that printed only the modelled figure
    would have been wrong by the entire budget, which is exactly the failure
    the run-time convention is about.

    The re-insert only. `churn_budget_s` bounds `_reinsert` and nothing else,
    so the delete that precedes each step is open-ended and is disclosed in
    `_PER_QUERY_OPS_COSTS` rather than counted here.
    """
    if "churn" not in plan.workloads or plan.rows <= 0:
        return None
    fractions = [float(f) for f in plan.ops.get("churn_fractions", [0.1])]
    budget = float(plan.ops.get("churn_budget_s", 1800))
    rate = plan.write_rows_per_s
    typical = 0.0
    worst = 0.0
    rows = 0
    churned_so_far = 0.0
    for fraction in fractions:
        step = max(0.0, fraction - churned_so_far)
        if step <= 0:
            continue
        count = int(round(step * plan.rows))
        if count == 0:
            continue
        churned_so_far = fraction
        rows += count
        typical += min(count / rate, budget) if rate > 0 else budget
        worst += budget
    if worst <= 0:
        return None
    return _QueryTerm(
        name="churn re-insert",
        typical_s=typical,
        worst_s=worst,
        basis=(f"{rows:,} rows written back at ~{rate:,.0f} rows/s (the load "
               f"rate; re-insert can be far slower), {budget:.0f}s cap/step"),
        bounded=True,
    )


# The extension point. Each entry takes a `_QueryPlan` and returns a
# `_QueryTerm` or None ("this profile does not run it"). Everything else --
# both totals, the per-engine split, the printed block and /api/estimate --
# follows from the registry, so adding a cost is one function and one line
# here, and nothing reshapes.
#
#   T1d  concurrency repeats  -- already multiplied inside _concurrency_windows
#   T2a  ef-calibration sweep -- gate on plan.phases/plan.ann, scope "unit"
#   T3c  filtered budget      -- no new term needed; _filtered_search's own
#                                docstring has the instruction, and is the
#                                only place that carries it
#   T4a  churn maintenance    -- gate on "churn" in plan.workloads
#
# Ordered as the profiles order `ops.workloads`, so the printed block reads in
# the order the run pays the costs.
_QUERY_TERMS: Tuple[Callable[[_QueryPlan], Optional[_QueryTerm]], ...] = (
    _concurrency_windows,
    _filtered_search,
    _churn_reinsert,
)


class _TermTally:
    """Sums query-time terms by name, keeping the order the registry produced.

    A term is asked once per (engine, dataset), so the same name arrives many
    times with a different basis each time (different write rates, different
    row counts). The distinct bases are kept so the printed line can quote one
    as an example rather than implying it covers every engine.
    """

    def __init__(self) -> None:
        self._order: List[str] = []
        self._typical: Dict[str, float] = {}
        self._worst: Dict[str, float] = {}
        self._bases: Dict[str, List[str]] = {}
        self._bounded: Dict[str, bool] = {}

    def add(self, term: _QueryTerm, factor: int) -> None:
        name = term.name
        if name not in self._typical:
            self._order.append(name)
            self._typical[name] = 0.0
            self._worst[name] = 0.0
            self._bases[name] = []
            self._bounded[name] = True
        self._typical[name] += term.typical_s * factor
        self._worst[name] += term.worst_s * factor
        # One uncapped contribution makes the whole term uncapped.
        self._bounded[name] = self._bounded[name] and term.bounded
        if term.basis not in self._bases[name]:
            self._bases[name].append(term.basis)

    @property
    def typical_s(self) -> float:
        return sum(self._typical.values())

    @property
    def worst_s(self) -> float:
        return sum(self._worst.values())

    def rows(self) -> List[Dict[str, Any]]:
        return [{
            "name": name,
            "typical_seconds": round(self._typical[name], 3),
            "worst_seconds": round(self._worst[name], 3),
            "typical_hours": round(self._typical[name] / 3600, 2),
            "worst_hours": round(self._worst[name] / 3600, 2),
            "basis": (self._bases[name][0] if len(self._bases[name]) == 1
                      else f"e.g. {self._bases[name][0]}"),
            "bounded": self._bounded[name],
            # A term that knows its shape but not its duration contributes
            # nothing to either figure. Printing "~0.0 h" for it would read as
            # "free" rather than "unmeasured", so it is flagged instead.
            "quantified": self._worst[name] > 0,
        } for name in self._order]


def _query_costs_not_modelled(profile: Dict[str, Any], workloads: Sequence[str],
                              phases: Sequence[str]) -> List[str]:
    """The query costs that exist, are paid, and cannot be put a number on."""
    ann = profile.get("ann", {}) or {}
    costs: List[str] = []
    if "ann" in phases and ann.get("enabled", True):
        costs.append(_ANN_QUERY_COST)
    for workload, texts in _PER_QUERY_OPS_COSTS.items():
        if workload in workloads:
            costs += list(texts)
    return costs


def _estimate_query_hours(profile: Dict[str, Any], engines: List[str],
                          datasets: List[str], passes: List[str],
                          phases: List[str]) -> Dict[str, Any]:
    """Accumulate every registered query-time term over the plan."""
    ops = profile.get("ops", {}) or {}
    ann = profile.get("ann", {}) or {}
    ops_runs = "ops" in phases and ops.get("enabled", True)
    workloads = ([str(w) for w in (ops.get("workloads") or [])]
                 if ops_runs else [])

    tally = _TermTally()
    per_engine_s: Dict[str, float] = {e: 0.0 for e in engines}

    for engine in engines:
        for dataset in datasets:
            plan = _QueryPlan(
                profile=profile, ops=ops, ann=ann, workloads=list(workloads),
                phases=list(phases), k=int(profile.get("k", 10)),
                engine=engine, dataset=dataset,
                rows=_ops_rows(ops, dataset),
                write_rows_per_s=_effective_rate(engine, dataset),
                query_qps=_query_qps(engine, dataset),
                units=_ops_units(profile, engine), passes=len(passes),
            )
            for make_term in _QUERY_TERMS:
                term = make_term(plan)
                if term is None:
                    continue
                factor = _scope_factor(plan, term.scope)
                tally.add(term, factor)
                per_engine_s[engine] += term.typical_s * factor

    terms = tally.rows()
    not_modelled = _query_costs_not_modelled(profile, workloads, phases)
    # Only an engine whose rate is missing *and* which is about to be asked to
    # run queries is a gap worth printing.
    no_rate = ([e for e in engines
                if any(_query_qps(e, d) is None for d in datasets)]
               if not_modelled and datasets else [])

    return {
        "query_hours": round(tally.typical_s / 3600, 2),
        "query_hours_worst": round(tally.worst_s / 3600, 2),
        "query_per_engine_hours": {e: round(s / 3600, 2)
                                   for e, s in per_engine_s.items()},
        "query_terms": terms,
        "query_worst_case_is_a_ceiling": all(t["bounded"] for t in terms),
        "engines_without_query_rate": no_rate,
        "query_costs_not_modelled": not_modelled,
    }


def estimate_load_hours(profile: Dict[str, Any], engines: List[str],
                        datasets: List[str], passes: List[str],
                        phases: List[str]) -> Dict[str, Any]:
    """Estimate ingest time before the run starts, and query time beside it.

    ann-benchmarks reloads the whole dataset for every M value, and the
    incrementally-building engines ingest at a few hundred rows per second. That
    multiplies quietly: a seven-value M grid over four datasets is days of pure
    loading. Showing the number up front is the difference between choosing that
    and discovering it six hours in.

    Returns the breakdown rather than printing it, so the CLI and the web UI
    cannot report different numbers for the same plan.

    `total_hours` still means *ingest*, and must keep meaning it:
    `webui/static/control.js` and `webui/static/setup.js` both render it as
    "estimated ingest ~N h", and docs/07-planning-a-run.md quotes its figures.
    Redefining it as the run total would have changed what both pages have been
    telling people. `ingest_hours` is the same number under a name that cannot
    be misread, and `worst_case_total_hours` is the figure to plan against --
    the one the "a run longer than its printed estimate is a bug" criterion
    refers to. `long_run` stays keyed on ingest because the warning it triggers
    is about M values reloading the corpus.

    Query time is accumulated separately by `_estimate_query_hours`; see the
    comment block above it for what is modelled, what is only a ceiling, and
    what is named as unknown instead of being guessed at.
    """
    m_count = max(1, len(profile.get("ann", {}).get("m_values", [16])))
    ops_m = max(1, len(profile.get("ops", {}).get("m_values", [16])))
    per_engine: Dict[str, float] = {}
    unknown: List[str] = [d for d in datasets if d not in _DATASET_ROWS]

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
                engine_h += (rows / effective) * ops_m / 3600
        per_engine[engine] = round(engine_h * len(passes), 2)

    total_h = round(sum(per_engine.values()), 2)
    query = _estimate_query_hours(profile, engines, datasets, passes, phases)
    # Summed from the rounded parts so the printed lines add up for a reader
    # checking the arithmetic by eye.
    return {
        "total_hours": total_h,
        "ingest_hours": total_h,
        "per_engine_hours": per_engine,
        "m_values": m_count,
        "ops_m_values": ops_m,
        "passes": len(passes),
        "phases": list(phases),
        "datasets_without_estimate": unknown,
        "long_run": total_h > 12,
        **query,
        "typical_total_hours": round(total_h + query["query_hours"], 2),
        "worst_case_total_hours": round(
            total_h + query["query_hours_worst"], 2),
    }


def print_load_estimate(profile: Dict[str, Any], engines: List[str],
                         datasets: List[str], passes: List[str],
                         phases: List[str]) -> None:
    estimate = estimate_load_hours(profile, engines, datasets, passes, phases)
    # The gate is the whole run, not ingest alone. It used to be ingest, which
    # meant `smoke` -- 0.5 h of loading against a 3.5 h churn ceiling across
    # six engines -- printed nothing at all.
    if estimate["worst_case_total_hours"] < 1:
        return
    total_h = estimate["ingest_hours"]
    rows_per_pass = [f"{engine} ~{hours:.1f} h"
                     for engine, hours in estimate["per_engine_hours"].items()]
    print(f"\nestimated ingest time (loading only, before any queries):")
    print("  " + "  |  ".join(rows_per_pass))
    print(f"  total ~{total_h:.1f} h across {estimate['passes']} pass(es), "
          f"{estimate['m_values']} M value(s)")
    if estimate["long_run"]:
        print(f"  ! This is a long run. Each M value costs a full reload of every "
              f"dataset,\n    and MHNSW/VIDX build incrementally at a few hundred "
              f"rows/s. Reduce\n    ann.m_values or the dataset list to cut it "
              f"roughly proportionally.")
    _print_query_estimate(estimate)
    print()


def _print_query_estimate(estimate: Dict[str, Any]) -> None:
    """The second half: query time, its ceiling, the total, and the gaps.

    Printed as two figures per term wherever they differ. One confident number
    would be the wrong shape for these costs -- a budget is a ceiling, not a
    duration -- and the criterion that a run must not outlast its estimate is
    about the ceiling, so the ceiling is what is labelled as the number to plan
    against.
    """
    terms = estimate["query_terms"]
    if terms:
        print("\nestimated query time (the parts the profile puts a clock on):")
        width = max(len(t["name"]) for t in terms)
        for t in terms:
            if not t["quantified"]:
                figure = "cost not quantified"
            elif abs(t["worst_hours"] - t["typical_hours"]) < 0.05:
                figure = f"~{t['typical_hours']:.1f} h"
            else:
                figure = (f"~{t['typical_hours']:.1f} h typical, "
                          f"~{t['worst_hours']:.1f} h worst case")
            print(f"  {t['name']:<{width}}  {figure}")
            print(f"  {'':<{width}}    {t['basis']}")
        print(f"  subtotal ~{estimate['query_hours']:.1f} h typical, "
              f"~{estimate['query_hours_worst']:.1f} h worst case")

    print(f"\nestimated run time ~{estimate['typical_total_hours']:.1f} h "
          f"typical, ~{estimate['worst_case_total_hours']:.1f} h worst case")
    # The caveat comes before the claim, and the claim is conditional on it.
    # Printed the other way round, two adjacent lines said the worst case was a
    # number a run cannot exceed and then that a run can exceed it.
    ceiling = estimate["query_worst_case_is_a_ceiling"]
    if terms and not ceiling:
        loose = ", ".join(t["name"] for t in terms if not t["bounded"])
        print(f"  ! Not yet a ceiling: nothing caps {loose}, so a run can "
              f"exceed the worst case until that is fixed.")
        print("  Plan against it all the same: it is the largest figure this "
              "configuration justifies.")
    else:
        print("  Plan against the worst case: a run longer than it is a bug.")

    gaps = estimate["query_costs_not_modelled"]
    if gaps:
        who = ", ".join(estimate["engines_without_query_rate"])
        print(f"  ! Neither figure includes the costs below. They are a number "
              f"of queries, and\n    there is no measured query rate for "
              f"{who or 'these engines'} to multiply them by:")
        for gap in gaps:
            print(f"      {gap}")
