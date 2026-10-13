"""An in-process fake engine, so the ops workloads are testable without a server.

Every workload in `harness/workloads/` could previously only be exercised by
starting a database, loading a corpus and waiting: the cheapest honest check
that `churn.run` still records what it claims to record was a multi-hour run on
real hardware. That cost is why the ops path has had the least test coverage of
anything in this repo, and why the bugs it shipped were the quiet kind — a
record written under the wrong phase, a recall number scored against the wrong
ground truth — rather than crashes.

`FakeDriver` is an `EngineDriver` whose storage is a handful of NumPy arrays --
vectors, ids, one per filter column, and a tombstone flag -- and whose ANN
behaviour is a deliberately crude model with the knobs turned outward. It is
not a simulator of any real engine and must never be used to produce a number
that appears in a report. Its only job is to let a test assert *the harness's*
behaviour in milliseconds:

* recall that genuinely falls as `ef_search` drops (`ef_half`), so a recall
  assertion in a later test cannot pass vacuously,
* a settable per-query latency (`latency_s`, `latency_jitter`), so percentile
  and QPS plumbing can be checked against known inputs, and a settable ingest
  cost (`load_latency_s`) that the driver reports as its own, which is the
  only way to give the build workload's incremental branch a measurable
  number — see `load()`,
* tombstones that cost recall until `maintain()` runs (`tombstone_penalty`), so
  the churn workload has something to measure degrading,
* post-filtering over the top-`ef` candidate list only, so short result sets
  and the correlated-filter failure mode arise from the model rather than
  being faked per test,
* a settable `explain_uses_vector_index()` flag, because "the planner ignored
  the index" is a reportable state the harness has to carry through.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence

import numpy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import datasets as ds  # noqa: E402
from harness.datasets import (ANGULAR, EUCLIDEAN, FILTER_COLUMNS,  # noqa: E402
                              FILTER_CTAG, FILTER_TAG, Dataset)
from harness.drivers.base import (ConnectionSpec, EngineDriver,  # noqa: E402
                                  IndexSpec, LoadResult, RowAttrs)


@dataclass(frozen=True)
class _Rows:
    """One consistent snapshot of everything the fake engine holds.

    Frozen, and replaced wholesale on every write, for two reasons. It is the
    repo's immutability convention, and it buys thread safety for nothing: the
    concurrency workload points several driver clones at one store, and a
    reader that took the snapshot before a write sees a complete older state
    rather than new vectors beside old ids.
    """

    vectors: numpy.ndarray      # (n, dim) float32
    ids: numpy.ndarray          # (n,) int64 — row ids as the engine knows them
    # One array per filter column, held separately rather than as a dict keyed
    # by column name. The real engines store two columns and a test that wants
    # to show a predicate reading the wrong one needs them to be able to hold
    # different values; a single array shared under two names could not fail
    # that way, so the test would pass whichever column the code read.
    tags: numpy.ndarray         # (n,) int64 — `tag`, uncorrelated with vectors
    ctags: numpy.ndarray        # (n,) int64 — `ctag`, a cluster id
    tombstoned: numpy.ndarray   # (n,) bool  — deleted but not yet reclaimed


class _Store:
    """The rows, shared between a driver and its clones."""

    def __init__(self, dim: int):
        self.dim = int(dim)
        self.rows = _empty_rows(dim)
        self.lock = threading.Lock()
        self._clones = 0

    def next_clone_id(self) -> int:
        with self.lock:
            self._clones += 1
            return self._clones


def _empty_rows(dim: int) -> _Rows:
    return _Rows(
        vectors=numpy.empty((0, max(int(dim), 0)), dtype=numpy.float32),
        ids=numpy.empty(0, dtype=numpy.int64),
        tags=numpy.empty(0, dtype=numpy.int64),
        ctags=numpy.empty(0, dtype=numpy.int64),
        tombstoned=numpy.empty(0, dtype=bool),
    )


class FakeDriver(EngineDriver):
    """An EngineDriver backed by NumPy arrays and a tunable error model."""

    name = "fake"

    def __init__(self, spec: Optional[ConnectionSpec] = None, *,
                 dim: int = 0, metric: str = EUCLIDEAN,
                 ef_search: int = 64, ef_half: float = 64.0,
                 latency_s: float = 0.0, latency_jitter: float = 0.0,
                 load_latency_s: float = 0.0,
                 tombstone_penalty: float = 1.0,
                 uses_vector_index: bool = True,
                 capabilities: Optional[Dict[str, Any]] = None,
                 version: str = "fake-engine 0.1",
                 tag_columns: Sequence[str] = FILTER_COLUMNS,
                 seed: int = 20261012,
                 _store: Optional[_Store] = None):
        super().__init__(spec or ConnectionSpec(host="fake", port=0))
        self.ef_half = float(ef_half)
        self.latency_s = float(latency_s)
        self.latency_jitter = float(latency_jitter)
        self.load_latency_s = float(load_latency_s)
        self.tombstone_penalty = float(tombstone_penalty)
        self.uses_vector_index = bool(uses_vector_index)
        # Deliberately one key, not the three the base class defaults to: a
        # test should see exactly which capability the workload under test
        # consulted, and `build.run` reads only this one.
        self._capabilities = dict(capabilities if capabilities is not None
                                  else {"incremental_index": True})
        self.incremental_index = bool(
            self._capabilities.get("incremental_index", True))
        self._version = version
        self._tag_columns = tuple(tag_columns)
        self._default_metric = metric
        self._default_dim = int(dim)
        self._seed = int(seed)
        self._rng = numpy.random.default_rng(seed)
        # Latency draws come from a second, independent stream. Sharing one
        # Generator would make turning `latency_jitter` on shift every
        # subsequent recall draw: a later percentile test would watch its
        # recall numbers move for no modelled reason, with nothing in the
        # model to explain it.
        self._latency_rng = numpy.random.default_rng(
            numpy.random.SeedSequence([int(seed), 1]))
        self._ef_search = int(ef_search)
        self._index: Optional[IndexSpec] = None
        self._connected = False
        self._store = _store if _store is not None else _Store(dim)
        # Every iterative-scan mode `set_iterative_scan` was asked for, in
        # order. A list rather than a single value because the thing worth
        # asserting is the SEQUENCE: PR 3 §3.4 item 4 has the filtered
        # workload measure its grid once per declared mode, and "the last mode
        # won" cannot tell a two-mode loop from a one-mode loop that set the
        # mode twice.
        self.iterative_scan_calls: List[str] = []
        self.iterative_scan: Optional[str] = None
        # Call counters, so a test can prove which schema calls a workload made
        # rather than inferring it from a timing.
        self.create_schema_calls = 0
        self.create_index_calls = 0
        self.drop_schema_calls = 0

    # -- lifecycle ------------------------------------------------------

    def connect(self) -> None:
        self._connected = True

    def close(self) -> None:
        self._connected = False

    def clone(self) -> "FakeDriver":
        """A sibling driver sharing these rows but with its own session state.

        This is what the concurrency workload's `driver_factory` returns. Each
        client needs its own `ef_search`, its own connection flag and above all
        its own RNG — one Generator driven from several threads is not safe and
        would make the error model's draws depend on interleaving.
        """
        sibling = FakeDriver(
            self.spec, dim=self._default_dim, metric=self._default_metric,
            ef_search=self._ef_search, ef_half=self.ef_half,
            latency_s=self.latency_s, latency_jitter=self.latency_jitter,
            load_latency_s=self.load_latency_s,
            tombstone_penalty=self.tombstone_penalty,
            uses_vector_index=self.uses_vector_index,
            capabilities=self._capabilities, version=self._version,
            tag_columns=self._tag_columns,
            seed=self._seed + self._store.next_clone_id(),
            _store=self._store,
        )
        sibling._index = self._index
        return sibling

    # -- schema ---------------------------------------------------------

    def drop_schema(self) -> None:
        self.drop_schema_calls += 1
        # The store object itself survives, so clones handed out earlier stay
        # attached to the driver they were cloned from.
        with self._store.lock:
            self._store.rows = _empty_rows(self._store.dim)

    def create_schema(self, index: IndexSpec) -> None:
        self.create_schema_calls += 1
        self._index = index
        self._store.dim = int(index.dim)

    def create_index(self, index: IndexSpec) -> None:
        self.create_index_calls += 1
        self._index = index

    # -- data -----------------------------------------------------------

    def load(self, vectors: numpy.ndarray, attrs,
             threads: int = 1, start_id: int = 0) -> LoadResult:
        """Ingest rows, reporting how long it took.

        `load_latency_s` buys a measurable ingest cost, and it has to be spent
        in here rather than around the call. `build.py` times the bulk build
        with a timer of its own but takes the load cost from this
        `LoadResult.wall_seconds` — the driver's own word for it — so a sleep
        wrapped around `load()` would never reach the record. One fixed cost
        per call, not per row: enough to lift the figure clear of the
        millisecond rounding in the record, which is all a test needs.
        """
        started = time.perf_counter()
        ids = numpy.arange(start_id, start_id + len(vectors), dtype=numpy.int64)
        self._append(ids, vectors, attrs)
        if self.load_latency_s > 0.0:
            time.sleep(self.load_latency_s)
        return LoadResult(rows=len(vectors),
                          wall_seconds=time.perf_counter() - started,
                          threads=max(1, int(threads)))

    def insert_rows(self, ids: Sequence[int], vectors: numpy.ndarray,
                    attrs) -> None:
        self._append(numpy.asarray([int(i) for i in ids], dtype=numpy.int64),
                     vectors, attrs)

    def delete_ids(self, ids: Sequence[int]) -> None:
        """Mark rows deleted without reclaiming them.

        A tombstoned row leaves the result set immediately — returning a
        deleted id would be an engine correctness bug, and would make the churn
        workload's recall arithmetic meaningless — but it stays in the
        candidate pool and costs recall through `tombstone_penalty` until
        `maintain()` reclaims it. That is the whole point of the churn
        measurement: an HNSW graph full of tombstones still has to be
        traversed.
        """
        wanted = numpy.asarray([int(i) for i in ids], dtype=numpy.int64)
        if len(wanted) == 0:
            return
        with self._store.lock:
            rows = self._store.rows
            victims = numpy.isin(rows.ids, wanted)
            self._store.rows = replace(
                rows, tombstoned=(rows.tombstoned | victims))

    def maintain(self) -> int:
        """Reclaim tombstoned rows, as an index maintenance pass would.

        Returns the number of rows purged. Not part of `EngineDriver`: it is
        the lever a test needs to show that the recall a churned index lost is
        the tombstones' fault and comes back when they go.
        """
        with self._store.lock:
            rows = self._store.rows
            purged = int(rows.tombstoned.sum())
            keep = ~rows.tombstoned
            self._store.rows = _Rows(
                vectors=rows.vectors[keep], ids=rows.ids[keep],
                tags=rows.tags[keep], ctags=rows.ctags[keep],
                tombstoned=rows.tombstoned[keep],
            )
            return purged

    def count_rows(self) -> int:
        """Live rows only — a tombstone is invisible to `SELECT count(*)`."""
        rows = self._store.rows
        return int((~rows.tombstoned).sum())

    def _append(self, ids: numpy.ndarray, vectors, attrs) -> None:
        """Append rows, taking a `RowAttrs` or a bare array of tag values.

        Coerced through `RowAttrs.coerce`, exactly as the four real drivers
        do, so this fake keeps the interface they implement rather than a
        laxer one. The bare-array form mirrors `tag` into `ctag`, which is
        what the helpers at the bottom of this file and most existing tests
        get; a test that needs the two columns to differ passes a `RowAttrs`
        and this is the only place that matters.
        """
        attrs = RowAttrs.coerce(attrs)
        new_vectors = numpy.asarray(vectors, dtype=numpy.float32)
        new_tags = numpy.asarray([int(t) for t in attrs.tag], dtype=numpy.int64)
        new_ctags = numpy.asarray([int(t) for t in attrs.ctag],
                                  dtype=numpy.int64)
        with self._store.lock:
            rows = self._store.rows
            if len(rows.ids) == 0:
                # The first write establishes the width; an empty store may
                # have been built before any IndexSpec named a dimension.
                self._store.dim = int(new_vectors.shape[1])
                stacked = new_vectors
            else:
                stacked = numpy.concatenate([rows.vectors, new_vectors])
            self._store.rows = _Rows(
                vectors=stacked,
                ids=numpy.concatenate([rows.ids, ids]),
                tags=numpy.concatenate([rows.tags, new_tags]),
                ctags=numpy.concatenate([rows.ctags, new_ctags]),
                tombstoned=numpy.concatenate(
                    [rows.tombstoned, numpy.zeros(len(ids), dtype=bool)]),
            )

    # -- query ----------------------------------------------------------

    def set_ef_search(self, ef_search: int) -> None:
        self._ef_search = int(ef_search)

    def set_iterative_scan(self, mode: str) -> None:
        """Record a pgvector iterative-scan mode; change no behaviour.

        This exists because `harness/workloads/filtered.py` gates on
        `hasattr(driver, "set_iterative_scan")` and this fake had no such
        method, so the branch that applies the mode -- and the loop that now
        measures the grid once per mode -- could not be reached in a test at
        all. `ops.pgvector_iterative_scan` was declared in three config files
        and executed in none of 69 archived records, and an untestable branch
        is how that lasted the life of the project.

        Deliberately NOT modelled. The real knob changes how far pgvector
        walks its graph when a filter rejects candidates, which would change
        recall; faking a recall effect here would invite a test to asserting
        a number this fake invented, and the fake "must never be used to
        produce a number that appears in a report". What a test may assert is
        that the workload asked for the right modes in the right order, which
        is what `iterative_scan_calls` is for.

        The mode is not validated. `PostgresDriver.set_iterative_scan`
        interpolates it into a `SET` and the whitelist that protects that
        lives in the orchestrator (`cli.py:_ops_iterative_scan`, against the
        engine config's own `iterative_scan_modes`), so a fake that refused an
        unknown mode would be testing a guard the driver does not have.
        """
        self.iterative_scan = str(mode)
        self.iterative_scan_calls.append(str(mode))

    def p_ef(self, ef: Optional[int] = None) -> float:
        """p(ef) = min(1, ef / (ef + ef_half)).

        The probability that any one true neighbour survives the search. With
        `ef_half = 0` the fake is exact, which is what a test wants when it is
        measuring something other than recall.
        """
        ef = self._ef_search if ef is None else ef
        denominator = ef + self.ef_half
        if denominator <= 0:
            return 1.0
        # Clamped at both ends. A negative ef_search would otherwise produce a
        # negative probability, and a negative probability does not announce
        # itself: multiplied by the negative factor a tombstone_penalty above
        # 1.0 produces in keep_probability(), two sign errors cancel into a
        # plausible-looking keep rate.
        return max(0.0, min(1.0, ef / denominator))

    def tombstone_fraction(self) -> float:
        rows = self._store.rows
        if len(rows.ids) == 0:
            return 0.0
        return float(rows.tombstoned.sum()) / float(len(rows.ids))

    def keep_probability(self) -> float:
        """p(ef), less what the tombstones in the graph cost.

        `tombstone_penalty` is normalised against the tombstoned *fraction* of
        the whole array, so a penalty of 1.0 means "an index that is entirely
        tombstones finds nothing" and the recall curve against churn is linear.

        Clamped to [0, 1]: a penalty above 1.0 drives the factor negative, and
        a probability outside the unit interval silently stops behaving like
        one — `rng.random() < p` simply always or never fires, which reads as a
        working model with a surprising recall rather than as a bad argument.
        """
        penalty = self.tombstone_penalty * self.tombstone_fraction()
        return max(0.0, min(1.0, self.p_ef() * (1.0 - penalty)))

    def query(self, vector, k: int) -> List[int]:
        self._require_connection()
        self._sleep()
        rows = self._store.rows
        return [int(i) for i in rows.ids[self._candidates(rows, vector, k)]]

    def query_filtered(self, vector, k: int, threshold: int,
                       column: str = FILTER_TAG) -> List[int]:
        """Post-filter the ANN candidate list — the top `ef` only.

        Post-filtering rather than pre-filtering is what makes this fake useful
        for the filtered workload: with a selective predicate the top-`ef`
        candidates rarely contain `k` qualifying rows, so short result sets and
        the correlated-filter collapse appear on their own instead of being
        staged test by test.

        `column` names which scalar column the predicate reads, and it reads
        that column's own values: `_Rows` holds `tag` and `ctag` separately,
        so a driver or workload that asked for one and got the other produces
        a different result set here rather than the same one. It used to be
        one array under two names, which made "the right column was read"
        untestable -- every answer was the same answer.

        The whitelist is `tag_columns`, which defaults to the dataset
        module's `FILTER_COLUMNS` rather than to a literal of its own, so a
        column this fake accepts is a column the real drivers accept.
        """
        self._require_connection()
        values = self._column_values(column)
        self._sleep()
        rows = self._store.rows
        candidates = self._candidates(rows, vector, self._ef_search)
        qualifying = candidates[values[candidates] < int(threshold)]
        return [int(i) for i in rows.ids[qualifying[:k]]]

    def _column_values(self, column: str) -> numpy.ndarray:
        """One filter column's values, or `ValueError` for anything else.

        Raises before `_sleep()` and before any candidate is computed, so a
        rejected column costs nothing and cannot half-run a query.
        """
        if column not in self._tag_columns:
            raise ValueError(
                f"no such column: {column!r} (have {list(self._tag_columns)})")
        rows = self._store.rows
        return rows.ctags if column == FILTER_CTAG else rows.tags

    def explain_uses_vector_index(self, vector, k: int,
                                  tag_threshold: Optional[int] = None,
                                  column: str = FILTER_TAG) -> bool:
        """The configured flag, but only for a column that exists.

        The flag is what tests set; the column check is here because this is
        the plan check every workload runs before it measures, and a workload
        that asked about a column the engine has no index on must not be told
        the index is fine. `tag_threshold` keeps its name: `test_fake_driver`
        passes it by keyword.
        """
        self._column_values(column)
        return self.uses_vector_index

    # -- the error model ------------------------------------------------

    def _candidates(self, rows: _Rows, vector, count: int) -> numpy.ndarray:
        """The candidate list: true top-`count`, with ANN error applied.

        Each true neighbour is kept independently with probability
        `keep_probability()` — independently, per neighbour, because a single
        draw for the whole list would give a recall of either 1 or 0 per query
        and no usable curve at all. A neighbour that is missed is replaced, in
        its own position so the list stays distance-ordered, by a random live
        row drawn from outside the true list.

        Two exclusions matter, and both exist so a recall assertion means what
        it says:

        * The filler pool excludes every id in the true top-`count`, not just
          the ones still in the result. A filler that happened to be the very
          neighbour just dropped would hand recall back and push measured
          recall above p(ef), which would make the error model untestable.
        * The filler pool excludes tombstoned rows, which are never returned.

        Returns positions into `rows`, not ids, so the caller can read tags
        from the same snapshot without a second lookup.
        """
        truth = self._nearest_live(rows, vector, count)
        if len(truth) == 0:
            return truth
        kept = self._rng.random(len(truth)) < self.keep_probability()
        holes = [int(i) for i in numpy.nonzero(~kept)[0]]
        fillers = self._draw_live(rows, len(holes), exclude=truth)
        out: List[Optional[int]] = [int(p) for p in truth]
        for slot, filler in zip(holes, fillers):
            out[slot] = int(filler)
        # A hole with no filler left — the live pool is smaller than the
        # candidate list — is dropped. Padding it with a duplicate would
        # silently inflate recall.
        for slot in holes[len(fillers):]:
            out[slot] = None
        return numpy.array([p for p in out if p is not None], dtype=numpy.int64)

    def _nearest_live(self, rows: _Rows, vector, count: int) -> numpy.ndarray:
        """Positions of the `count` nearest live rows, nearest first."""
        live = numpy.nonzero(~rows.tombstoned)[0]
        count = min(int(count), len(live))
        if count <= 0:
            return numpy.empty(0, dtype=numpy.int64)
        scores = self._scores(rows, vector)[live]
        nearest = numpy.argpartition(scores, kth=count - 1)[:count]
        return live[nearest[numpy.argsort(scores[nearest])]]

    def _draw_live(self, rows: _Rows, n: int,
                   exclude: numpy.ndarray) -> numpy.ndarray:
        """`n` distinct live positions, excluding `exclude`."""
        if n <= 0:
            return numpy.empty(0, dtype=numpy.int64)
        # `~` already returns a fresh array, so masking it mutates nothing the
        # store still holds.
        eligible = ~rows.tombstoned
        eligible[exclude] = False
        pool = numpy.nonzero(eligible)[0]
        if len(pool) == 0:
            return numpy.empty(0, dtype=numpy.int64)
        return self._rng.choice(pool, size=min(n, len(pool)), replace=False)

    def _scores(self, rows: _Rows, vector) -> numpy.ndarray:
        """Distance-ordering scores for every row; lower is nearer.

        Uses the same two identities `harness.datasets` uses for ground truth,
        so the fake's "true" top-k is exactly the ground truth the workloads
        score it against. Anything else and a perfect fake would measure as
        imperfect.
        """
        query = numpy.asarray(vector, dtype=numpy.float32)
        if self._metric() == ANGULAR:
            norms = numpy.linalg.norm(rows.vectors, axis=1)
            norms = numpy.where(norms == 0, 1.0, norms)
            query_norm = float(numpy.linalg.norm(query)) or 1.0
            return -(rows.vectors @ query) / (norms * query_norm)
        return ((rows.vectors - query) ** 2).sum(axis=1)

    def _metric(self) -> str:
        if self._index is not None and self._index.metric:
            return self._index.metric
        return self._default_metric

    def _sleep(self) -> None:
        """Per-query latency. Skipped entirely when both knobs are off.

        `time.sleep()` costs tens of microseconds even for a zero argument, and
        the workload tests run tens of thousands of queries; the guard is what
        keeps them inside their seconds-long budget.
        """
        if self.latency_s <= 0.0 and self.latency_jitter <= 0.0:
            return
        # Jitter is added, never subtracted, so latency_s stays the floor and a
        # percentile test gets a tail it can assert on. Drawn from the latency
        # stream, never the error model's.
        time.sleep(self.latency_s
                   + self.latency_jitter * float(self._latency_rng.random()))

    def _require_connection(self) -> None:
        if not self._connected:
            raise RuntimeError("query on a FakeDriver that is not connected")

    # -- sizing / metadata ----------------------------------------------

    def index_bytes(self) -> int:
        """Graph size, counting tombstones.

        Deleting rows does not shrink the index until `maintain()` runs, which
        is the behaviour the churn workload records alongside recall.
        """
        rows = self._store.rows
        m = self._index.m if self._index is not None else 16
        return int(len(rows.ids) * max(m, 1) * 8)

    def table_bytes(self) -> int:
        dim = self._store.dim or self._default_dim
        return int(self.count_rows() * (dim * 4 + 16))

    def server_version(self) -> str:
        return self._version

    def capabilities(self) -> Dict[str, Any]:
        return dict(self._capabilities)


# ---------------------------------------------------------------------------
# Shared test corpus and helpers
# ---------------------------------------------------------------------------
# Both test modules that drive this fake need the same corpus, the same index
# spec and the same recall measurement, so they live beside the fake rather
# than in either test file. This follows `tests/synthetic.py`, which is how
# this repo already shares test scaffolding: a plain module in `tests/`, not a
# conftest. The repo has no conftest.py anywhere, and introducing its first one
# to share a handful of functions would be a larger structural change than the
# problem warrants.

# Small enough that exact ground truth is one argsort and the whole corpus is a
# 128 KB array; large enough that a 1% filter still selects 20 rows and a
# recall figure over 50 queries x k=10 has 500 samples behind it.
N_ROWS = 2_000
DIM = 16
N_QUERIES = 50
K = 10


def tiny_dataset(seed: int = 20261012, metric: str = ds.EUCLIDEAN) -> Dataset:
    """A synthetic Dataset with exact ground truth, built in-process.

    Exact, not approximate: these tests assert that a fake with p(ef) = 1 has
    recall 1, so the "truth" the workloads score against has to be the real
    top-k rather than something close to it.

    Available in both metric spaces over the same corpus. Euclidean alone
    would leave the angular branch of the fake's distance ordering untested,
    and angular is what this framework's headline dataset
    (glove-100-angular) and three of its four dbpedia corpora use.
    """
    rng = numpy.random.default_rng(seed)
    train = rng.random((N_ROWS, DIM), dtype=numpy.float32)
    test = rng.random((N_QUERIES, DIM), dtype=numpy.float32)
    if metric == ds.ANGULAR:
        # Cosine distance, 1 - cos(theta); the ordering is the reverse of the
        # normalized dot product, which is the identity harness.datasets uses.
        scores = 1.0 - (_unit(test) @ _unit(train).T)
    else:
        scores = ((test ** 2).sum(1)[:, None]
                  - 2.0 * (test @ train.T)
                  + (train ** 2).sum(1)[None, :])
    # 100 columns, the width ann-benchmarks' HDF5 files ship.
    neighbors = numpy.argsort(scores, axis=1)[:, :100].astype(numpy.int64)
    ordered = numpy.take_along_axis(scores, neighbors, axis=1)
    distances = (ordered if metric == ds.ANGULAR
                 else numpy.sqrt(numpy.maximum(ordered, 0.0)))
    return Dataset(name=f"tiny-{DIM}-{metric}", train=train, test=test,
                   neighbors=neighbors, distances=distances, metric=metric)


def _unit(a: numpy.ndarray) -> numpy.ndarray:
    return a / numpy.linalg.norm(a, axis=1, keepdims=True)


def index_spec(dataset: Dataset, m: int = 16,
               build_mode: str = "post") -> IndexSpec:
    return IndexSpec(dim=dataset.dim, m=m, metric=dataset.metric,
                     build_mode=build_mode)


def loaded_driver(dataset: Dataset, attrs=None, **kwargs) -> FakeDriver:
    """A connected FakeDriver holding the whole tiny corpus.

    `attrs` defaults to `assign_tags(N_ROWS)`, a bare array, which
    `RowAttrs.coerce` mirrors into `ctag`. Mirrored is the right default for
    the tests that only care about `tag`, and it is the wrong thing to build a
    `ctag` test on -- a predicate reading either column would then give the
    same answer, which is the vacuity this parameter exists to let a caller
    avoid. Pass a `RowAttrs` whose two columns differ.
    """
    driver = FakeDriver(dim=dataset.dim, metric=dataset.metric, **kwargs)
    driver.connect()
    driver.create_schema(index_spec(dataset))
    driver.load(dataset.train,
                ds.assign_tags(N_ROWS) if attrs is None else attrs)
    return driver


def measure_recall(driver: FakeDriver, dataset: Dataset, k: int = K,
                   id_map=None, repeats: int = 1) -> float:
    """Recall over the whole query set, against the dataset's own neighbours.

    `id_map` translates the expected ids the way the churn workload does: a row
    re-inserted under a fresh id is still the correct answer, and scoring the
    old id would report a recall drop that is bookkeeping rather than graph
    degradation.

    `repeats` runs the query set more than once. Recall here is the mean of
    `repeats * n_test * k` Bernoulli draws, so repeating shrinks the standard
    error and lets a test pin recall to the model's own formula tightly enough
    to notice a few percent of drift, instead of only a gross failure.
    """
    truth = dataset.neighbors[:, :k]
    if id_map:
        truth = numpy.array(
            [[id_map.get(int(t), int(t)) for t in row] for row in truth],
            dtype=numpy.int64)
    returned = []
    expected = []
    for _ in range(max(1, repeats)):
        for query, want in zip(dataset.test, truth):
            returned.append(driver.query(query, k))
            expected.append(want)
    return ds.recall_at_k(returned, numpy.array(expected), k)


def rotate_ids(driver: FakeDriver, dataset: Dataset, count: int) -> dict:
    """Delete `count` rows and re-insert their vectors under fresh ids.

    One churn step, done by hand: the data distribution — and so the correct
    answers — is unchanged, and the tombstones left behind are the only thing
    that can move recall. Returns the id map to score against.
    """
    victims = list(range(N_ROWS - count, N_ROWS))
    new_ids = list(range(N_ROWS, N_ROWS + count))
    driver.delete_ids(victims)
    driver.insert_rows(new_ids, dataset.train[victims],
                       ds.assign_tags(N_ROWS)[victims])
    return dict(zip(victims, new_ids))
