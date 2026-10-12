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

from typing import Callable, Sequence, Tuple, Union

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
