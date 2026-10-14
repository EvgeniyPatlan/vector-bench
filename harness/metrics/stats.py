"""Uncertainty intervals for the numbers that end up on a chart.

Every figure this framework publishes is one sample of a noisy process: a QPS
number taken again five minutes later is a different QPS number. Without an
interval attached, a 3% gap between two engines reads as a result, and the
report has no way to say "these two measurements cannot be told apart". That is
the most expensive kind of wrong this instrument can be, because it does not
crash and the chart looks fine.

The bootstrap is used rather than a closed-form interval because almost nothing
measured here is normally distributed: latencies are right-skewed with a long
tail, per-query recall is a bounded fraction that piles up at 1.0, and the
samples are small (one point per configuration). Resampling makes no
distributional assumption, and `numpy.percentile` over the resampled statistic
is the whole implementation.

NumPy only — the host is promised nothing beyond python3, PyYAML, docker, git
and curl, so scipy.stats is not available here and must not become a
dependency (see CLAUDE.md). NumPy itself lives in the images this module runs
inside, and on the machine that runs the test suite.
"""

from __future__ import annotations

import math
from typing import (Any, Callable, Dict, NamedTuple, Optional, Sequence, Tuple,
                    Union)

import numpy

# The bootstrap draws its resamples from a PRNG, so the interval it reports is
# itself a random variable: the same records resampled with fresh entropy give
# error bars that wobble by a fraction of a percent on every regeneration.
# Readers compare charts across report runs, and a wobble with no cause is
# indistinguishable from a real change in the measurement — so the seed has a
# fixed default rather than None, and a report regenerated from the same
# records is byte-identical. The particular value is arbitrary; that it never
# changes is not.
DEFAULT_SEED = 42

# Resample count. 1000 is the conventional floor for a percentile interval:
# each tail is then estimated from ~25 order statistics at alpha=0.05, which is
# enough that the bound moves in the third decimal, not the second. The cost is
# an (n, len(values)) float64 matrix — 1000 x 10,000 is 80 MB — which is why it
# stays a parameter instead of being raised "to be safe".
DEFAULT_RESAMPLES = 1000


def bootstrap_ci(values: Union[Sequence[float], numpy.ndarray],
                 stat: Callable[..., object] = numpy.mean,
                 n: int = DEFAULT_RESAMPLES,
                 alpha: float = 0.05,
                 seed: int = DEFAULT_SEED) -> Tuple[float, float]:
    """Percentile-bootstrap confidence interval for `stat(values)`.

    Convention: **percentile bootstrap on the resampled statistic**. `n`
    resamples of `len(values)` values are drawn with replacement, `stat` is
    applied to each, and the bounds are the `alpha/2` and `1 - alpha/2`
    quantiles of those `n` statistics. So `alpha=0.05` returns the 2.5th and
    97.5th percentiles — a 95% interval. No bias correction or acceleration
    (BCa): the extra machinery matters for strongly skewed statistics of tiny
    samples, and an interval a reader cannot reconstruct from this docstring is
    worth less here than a slightly conservative one.

    Degenerate input, all three of which real runs produce, and none of which
    raises — this is called while writing the report at the end of a run that
    may have taken twenty hours, and an exception escaping here would lose the
    entire report rather than one error bar:

    - **empty** `values` -> `(nan, nan)`, matching `latency.percentile` and
      `datasets.recall_at_k`. Returning zeros would publish a real-looking
      interval around a measurement that never happened.
    - **one value** -> `(stat([v]), stat([v]))`, a zero-width interval. Every
      resample of a one-element vector is that element; resampling cannot
      manufacture spread the data does not contain.
    - **all identical values** -> likewise zero width. For a phase that pinned
      the same recall in every repeat, "no spread measurable" is the correct
      answer, and the report should render it as a point rather than a bar.
    - a `nan` anywhere in `values` propagates to both bounds. Dropping it would
      quietly average over fewer configurations than the chart claims.

    `alpha`, `n` and the shape of `values` are validated and raise ValueError.
    Unlike the data cases above, those are caller mistakes that cannot first
    appear after a long run, so failing fast is safe and far clearer than
    inventing an interval.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n!r}")

    sample = numpy.asarray(values, dtype=float)
    if sample.ndim != 1:
        raise ValueError(
            f"values must be a one-dimensional sequence, got shape {sample.shape}")
    if sample.size == 0:
        return (float("nan"), float("nan"))

    rng = numpy.random.default_rng(seed)
    # One (n, len(sample)) index draw, not n draws of len(sample) inside a
    # Python loop: the report calls this once per record, and at a few thousand
    # records the loop version is seconds of report generation spent on error
    # bars.
    draws = rng.integers(0, sample.size, size=(n, sample.size))
    resampled = sample[draws]

    try:
        statistics = stat(resampled, axis=1)
    except TypeError:
        # Fallback for a statistic written as a plain function of one vector
        # (no `axis` keyword). This loops internally and is therefore the slow
        # path; the axis-aware path above is the one that matters. Only a
        # TypeError is caught, so a statistic that itself raises TypeError is
        # retried once and then reported from inside apply_along_axis rather
        # than being swallowed.
        statistics = numpy.apply_along_axis(stat, 1, resampled)

    low, high = numpy.percentile(
        statistics, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    # Plain floats, not numpy scalars: these go into a record's `extra` dict and
    # from there through json.dumps.
    return (float(low), float(high))


# ---------------------------------------------------------------------------
# PR 5 §5.1 items 4 and 6: an interval a record can carry and a report can
# compare without inventing a claim.
#
# `bootstrap_ci` returns two bounds and nothing else, which is everything an
# error bar needs and one fact short of what a TIE needs. Measured on master,
# both of these come back exactly `width == 0.0`:
#
#     bootstrap_ci([1.0] * 50)  -> (1.0, 1.0)      500 queries, every one
#                                                  found the whole top-k
#     bootstrap_ci([0.97])      -> (0.97, 0.97)    one query
#
# The first is a measurement -- "no spread is present in these 50 samples" is
# the honest reading and the docstring above says so. The second is the
# absence of one. From the bounds alone they are the same two numbers, so a
# report that marks two engines tied when their intervals overlap would
# publish "these engines performed the same" off a single query per engine.
# The sample count is the only thing that separates them, so it travels with
# the bounds rather than being recoverable from them.
# ---------------------------------------------------------------------------

#: Samples below which "no spread" is not a measurement. Two is the smallest
#: number that can disagree with itself; at one, every resample is the same
#: element and the zero width is arithmetic rather than evidence.
MIN_SPREAD_SAMPLES = 2


class Interval(NamedTuple):
    """A bootstrap interval and the sample count behind it.

    `low`/`high` are `bootstrap_ci`'s own bounds, unmodified -- this type adds
    a field, it does not re-derive a number. `n` is `len(values)`, which is
    what `usable` below needs and what a reader of `extra.recall_ci95` in a
    stored record needs just as much: an interval whose provenance is gone is
    an error bar nobody can audit.
    """

    low: float
    high: float
    n: int

    @property
    def usable(self) -> bool:
        """Whether this interval supports any claim at all.

        Non-finite bounds mean the measurement did not happen (empty input, or
        a `nan` among the values, both of which `bootstrap_ci` propagates by
        design). `n < MIN_SPREAD_SAMPLES` means the width is not evidence.
        Both are reported as "unknown" rather than as a width of zero, because
        a width of zero is a strong claim.
        """
        return (self.n >= MIN_SPREAD_SAMPLES
                and math.isfinite(self.low) and math.isfinite(self.high))

    def as_field(self) -> Optional[Dict[str, Any]]:
        """The `extra.*_ci95` value, or None where there is nothing to store.

        A dict rather than a two-element list. The list is the conventional
        shape and it is the one that loses `n`, and `n` is the field that
        stops a one-sample interval being published as a tie. `None` for a
        non-finite interval rather than `NaN`: `json.dumps` emits a bare
        `NaN`, which is not JSON any strict parser will read back, and a null
        is unambiguously "not measured" where a number is never that.
        """
        if not math.isfinite(self.low) or not math.isfinite(self.high):
            return None
        return {"low": float(self.low), "high": float(self.high),
                "n": int(self.n)}

    @classmethod
    def from_field(cls, value: Any) -> Optional["Interval"]:
        """Read one back out of a record, or None if it is not one.

        Tolerant of the two-element list as well, because a hand-written
        fixture or an archived record may carry that shape; such an interval
        has no sample count, so it is read with `n = 0` and is therefore
        `usable == False`. That is the conservative direction: an interval
        whose provenance the record does not state cannot support a tie.
        """
        if isinstance(value, dict):
            low, high = value.get("low"), value.get("high")
            if low is None or high is None:
                return None
            return cls(float(low), float(high), int(value.get("n") or 0))
        if isinstance(value, (list, tuple)) and len(value) == 2:
            if value[0] is None or value[1] is None:
                return None
            return cls(float(value[0]), float(value[1]), 0)
        return None


def mean_interval(values: Union[Sequence[float], numpy.ndarray],
                  **kwargs: Any) -> Interval:
    """`bootstrap_ci` of the mean, with the sample count attached.

    `numpy.mean` and not a wrapper around it, deliberately. `bootstrap_ci`
    calls `stat(resampled, axis=1)` on an `(n, len(values))` matrix and falls
    back to `apply_along_axis` only on TypeError, so a statistic whose
    signature ACCEPTS `axis` and whose body ignores it reduces the whole
    matrix to one number and the interval comes back zero-width -- measured on
    ten varied latencies, `(0.01029375, 0.01029375)` against the honest
    `(0.00965, 0.01095)`, and not even centred on the real mean of 0.01030.
    `numpy.mean` is axis-aware, so that collapse is unreachable here; it is
    reachable the moment someone passes a hand-written statistic, which is why
    this function exists rather than each caller choosing its own.
    """
    sample = numpy.asarray(values, dtype=float)
    low, high = bootstrap_ci(sample, numpy.mean, **kwargs)
    return Interval(low, high, int(sample.size))


def qps_interval_from_latencies(
        latencies_s: Union[Sequence[float], numpy.ndarray],
        **kwargs: Any) -> Interval:
    """PR 5 §5.1 item 3, done as the item describes it.

    Single-client QPS in this framework is `len(samples) / sum(samples)`
    (`metrics/latency.py:LatencyCollector.qps`), which is exactly
    `1 / mean(latency)`. So the quantity with a distribution worth resampling
    is the mean latency, and the QPS bounds are its bounds inverted -- which
    also SWAPS them, because inversion is decreasing: the slow end of the
    latency interval is the low end of the throughput one. Bootstrapping a
    ratio of sums directly would resample the same information twice and
    report a narrower interval for it.

    A non-positive or non-finite mean-latency bound yields a non-finite QPS
    bound rather than `latency.qps`'s `max(..., 1e-12)` clamp. The clamp is
    right where a single number must be produced for a record; here, a
    throughput of 1e12 would be a published error bar reaching to a trillion
    queries per second.
    """
    latency = mean_interval(latencies_s, **kwargs)

    def invert(bound: float) -> float:
        if not math.isfinite(bound) or bound <= 0.0:
            return float("nan")
        return 1.0 / bound

    return Interval(invert(latency.high), invert(latency.low), latency.n)


#: Phases whose samples are not independent inside one measurement, and which
#: PR 5 §5.1 item 5 therefore forbids bootstrapping: "Concurrency keeps its
#: repeat-based spread (PR 1.5); do not bootstrap within one window, because
#: samples inside a window are not independent." Sixteen clients queuing on
#: one server produce latencies that rise and fall together, so resampling
#: them estimates the spread of a quantity nobody measured and reports it
#: narrower than the window-to-window spread that is already recorded in
#: `extra.qps_samples`.
#:
#: Spelled here rather than imported from `harness.metrics.records`, following
#: `workloads/churn_readers.py:OPS_UNIT_KEYS`: this module promises NumPy and
#: nothing else, and `tests/test_t5b_stats.py` pins the string to
#: `records.PHASE_CONCURRENCY` so the two declarations cannot drift.
#:
#: It is a refusal and not a comment because a comment lasts exactly as long
#: as the next person who reads it. `ci_fields` is the only way a workload
#: writes these keys, and it raises here.
NON_INDEPENDENT_PHASES = frozenset({"concurrency"})

#: Keys `ci_fields` writes. Named so a test can assert the set rather than
#: spelling two string literals that a typo would silently split in two.
RECALL_CI_KEY = "recall_ci95"
QPS_CI_KEY = "qps_ci95"


def ci_fields(phase: str,
              recall_per_query: Optional[Sequence[float]] = None,
              latencies_s: Optional[Sequence[float]] = None,
              **kwargs: Any) -> Dict[str, Any]:
    """The `extra` keys PR 5 §5.1 item 4 asks for, for one record.

    `phase` is not decoration: it is the enforcement point for item 5. A
    future reader who adds error bars to the concurrency sweep will reach for
    this function, pass `PHASE_CONCURRENCY`, and get a ValueError naming the
    item and the reason rather than a narrower-than-real interval on a chart.

    Either argument may be omitted -- a filtered point whose budget expired
    before any query finished has neither -- and the corresponding key is then
    absent rather than null, because `extra` is additive and an absent key is
    the one shape every archived record already has.
    """
    if phase in NON_INDEPENDENT_PHASES:
        raise ValueError(
            f"PR 5 §5.1 item 5 forbids a bootstrap interval on {phase!r}: "
            f"samples inside one window are not independent, so resampling "
            f"them understates the spread. The window-to-window spread from "
            f"`ops.concurrency_repeats` is the measurement of uncertainty for "
            f"that phase and is already recorded in `extra.qps_samples`.")

    out: Dict[str, Any] = {}
    if recall_per_query is not None:
        field = mean_interval(recall_per_query, **kwargs).as_field()
        if field is not None:
            out[RECALL_CI_KEY] = field
    if latencies_s is not None:
        field = qps_interval_from_latencies(latencies_s, **kwargs).as_field()
        if field is not None:
            out[QPS_CI_KEY] = field
    return out


#: The three answers to "are these two engines distinguishable here?".
#:
#: Three and not two. A boolean overlap test has to put the unmeasured cases
#: somewhere, and both placements publish something false: measured, a
#: `(nan, nan)` interval against a real one gives False under `a.low <=
#: b.high and b.low <= a.high` AND under the strict form, so "no measurement"
#: arrives at the reader as "these two engines are ordered". The opposite
#: default is no better -- it would call every failed measurement a tie.
TIE_TIED = "tied"
TIE_SEPARATED = "separated"
TIE_UNKNOWN = "unknown"


def tie_verdict(a: Optional[Interval], b: Optional[Interval]) -> str:
    """Whether two intervals support a claim that the two differ.

    **Closed overlap**, `a.low <= b.high and b.low <= a.high`, after the
    unusable intervals are removed. The direction matters and is not a detail:
    with the degenerate cases left in, `(1.0, 1.0)` against `(0.994, 1.0)`
    reads TIED under the closed form and SEPARATED under the strict one, from
    the same two measurements. Closed is the conservative choice of the two --
    it declines to separate engines whose intervals touch -- and the whole
    point of item 6 is to stop the report ordering engines it cannot tell
    apart.

    `TIE_UNKNOWN` whenever either side is missing or unusable, so a reader
    never has to infer which of the three answers an absent mark meant.
    """
    if a is None or b is None or not a.usable or not b.usable:
        return TIE_UNKNOWN
    return TIE_TIED if (a.low <= b.high and b.low <= a.high) else TIE_SEPARATED
