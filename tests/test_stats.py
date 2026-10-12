"""Unit tests for the bootstrap confidence interval.

A confidence interval is the one number in the report nobody can sanity-check
by eye: a too-narrow interval makes two engines look separated when the
measurement cannot tell them apart, and a too-wide one hides a real difference.
Both failures publish cleanly. So the properties pinned here are coverage (does
a 95% interval actually contain the truth about 95% of the time), determinism
(does regenerating the report move the error bars), the percentile convention,
and the three degenerate inputs the report will certainly hand it — an empty
vector, one value, and a column of identical values.

Run: python3 -m pytest tests/ -v      (from the vector-bench directory)
"""

from __future__ import annotations

import os
import sys

import numpy
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.metrics import stats  # noqa: E402


# ---------------------------------------------------------------------------
# Coverage — the property the interval exists for
# ---------------------------------------------------------------------------

class TestCoverage:
    def test_covers_the_true_mean_in_roughly_95_percent_of_trials(self):
        """A seeded 200-trial coverage experiment on a known normal sample.

        The acceptance band is deliberately wide (90–99%): the percentile
        bootstrap is known to undercover slightly at small sample sizes, and
        200 trials have a sampling error of about 1.5 points on their own. Both
        the trial data and every interval are seeded, so this test either
        always passes or always fails — a coverage test that fails once a
        fortnight gets disabled, which is worse than having no coverage test.
        """
        true_mean = 5.0
        trials = 200
        rng = numpy.random.default_rng(20240101)
        covered = 0
        for trial in range(trials):
            sample = rng.normal(loc=true_mean, scale=2.0, size=100)
            low, high = stats.bootstrap_ci(sample, seed=1000 + trial)
            if low <= true_mean <= high:
                covered += 1
        fraction = covered / trials
        assert 0.90 <= fraction <= 0.99, f"coverage was {fraction:.3f}"


# ---------------------------------------------------------------------------
# Determinism — reruns of the report must not move the error bars
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_arguments_give_identical_bounds(self):
        sample = numpy.random.default_rng(4).normal(size=50)
        first = stats.bootstrap_ci(sample, seed=11)
        second = stats.bootstrap_ci(sample, seed=11)
        assert first == second
        assert repr(first) == repr(second), "bounds differ below repr precision"

    def test_the_default_seed_is_a_fixed_value_not_fresh_entropy(self):
        # This is the actual requirement: two runs of report generation over the
        # same records must print the same interval. A seed defaulting to None
        # would reseed from OS entropy and give every regeneration slightly
        # different error bars, which readers would compare as if they meant
        # something.
        sample = [0.91, 0.93, 0.88, 0.95, 0.90, 0.92, 0.89, 0.94]
        assert stats.bootstrap_ci(sample) == stats.bootstrap_ci(sample)
        assert stats.bootstrap_ci(sample) == stats.bootstrap_ci(
            sample, seed=stats.DEFAULT_SEED)

    def test_a_different_seed_is_allowed_to_move_the_bounds(self):
        # Guards against the opposite bug: a seed argument that is ignored,
        # which would make the determinism tests above pass trivially.
        sample = numpy.random.default_rng(5).normal(size=40)
        assert stats.bootstrap_ci(sample, seed=1) != stats.bootstrap_ci(sample, seed=2)


# ---------------------------------------------------------------------------
# Convention — percentile bootstrap, alpha=0.05 means 95%
# ---------------------------------------------------------------------------

class TestConvention:
    def test_alpha_is_split_into_two_tails_of_the_resample_distribution(self):
        # Recomputed here from first principles so the convention is readable
        # rather than inferred from the arithmetic in stats.py: the bounds are
        # the 2.5th and 97.5th percentiles of the resampled statistic.
        sample = numpy.random.default_rng(6).normal(size=30)
        rng = numpy.random.default_rng(99)
        draws = rng.integers(0, sample.size, size=(500, sample.size))
        expected = numpy.percentile(sample[draws].mean(axis=1), [2.5, 97.5])
        low, high = stats.bootstrap_ci(sample, n=500, alpha=0.05, seed=99)
        assert low == pytest.approx(expected[0])
        assert high == pytest.approx(expected[1])

    def test_a_smaller_alpha_gives_a_wider_interval(self):
        sample = numpy.random.default_rng(7).normal(size=60)
        wide = stats.bootstrap_ci(sample, alpha=0.01, seed=3)
        narrow = stats.bootstrap_ci(sample, alpha=0.50, seed=3)
        assert wide[0] < narrow[0] <= narrow[1] < wide[1]

    def test_the_interval_brackets_the_point_estimate(self):
        sample = numpy.random.default_rng(8).normal(loc=3.0, size=80)
        low, high = stats.bootstrap_ci(sample, seed=3)
        assert low <= sample.mean() <= high

    def test_an_alternative_statistic_is_honoured(self):
        # The report will want intervals on more than the mean; a median
        # interval must come from medians, not silently from means.
        sample = numpy.concatenate([numpy.zeros(50), numpy.full(50, 100.0)])
        mean_low, mean_high = stats.bootstrap_ci(sample, seed=3)
        med_low, med_high = stats.bootstrap_ci(sample, stat=numpy.median, seed=3)
        assert (med_low, med_high) != (mean_low, mean_high)
        assert med_low == 0.0 and med_high == 100.0

    def test_a_statistic_that_does_not_take_an_axis_still_works(self):
        # A hand-written statistic is a plain function of one vector; it must
        # not fail with "unexpected keyword argument 'axis'" deep inside report
        # generation.
        def spread(vector):
            return float(vector.max() - vector.min())

        low, high = stats.bootstrap_ci([1.0, 2.0, 3.0, 9.0], stat=spread, seed=3)
        assert 0.0 <= low <= high <= 8.0


# ---------------------------------------------------------------------------
# Degenerate input — all three of these arrive from real runs
# ---------------------------------------------------------------------------

class TestDegenerateInput:
    def test_empty_input_is_nan_not_zero_and_does_not_raise(self):
        # Matches latency.percentile and recall_at_k: zero would silently look
        # like a real, terrible result, and an exception escaping here would
        # lose the report at the end of a twenty-hour run.
        low, high = stats.bootstrap_ci([])
        assert numpy.isnan(low) and numpy.isnan(high)

    def test_a_single_value_collapses_to_that_value(self):
        # Every resample of a one-element vector is that element, so the
        # honest interval has zero width. The report must render it as "no
        # spread measurable", not as an error.
        assert stats.bootstrap_ci([3.5]) == (3.5, 3.5)

    def test_identical_values_give_a_zero_width_interval(self):
        assert stats.bootstrap_ci([2.0] * 25) == (2.0, 2.0)

    def test_a_nan_in_the_sample_propagates(self):
        # recall_at_k returns nan for an empty result set, so a nan can reach
        # here. Dropping it would quietly average over fewer configurations
        # than the chart claims; propagating it makes the gap visible.
        low, high = stats.bootstrap_ci([0.9, float("nan"), 0.95])
        assert numpy.isnan(low) and numpy.isnan(high)

    def test_two_values_still_produce_an_ordered_interval(self):
        low, high = stats.bootstrap_ci([1.0, 3.0])
        assert low <= high
        assert 1.0 <= low and high <= 3.0


# ---------------------------------------------------------------------------
# Shape of the implementation and of its output
# ---------------------------------------------------------------------------

class TestMechanics:
    def test_resampling_is_one_vectorized_index_draw(self):
        # The report calls this per record, so n resamples must cost one
        # (n, len(values)) index draw rather than a Python loop over n.
        drawn = []
        real_default_rng = numpy.random.default_rng

        class CountingGenerator:
            def __init__(self, inner):
                self._inner = inner

            def integers(self, *args, **kwargs):
                drawn.append(kwargs.get("size"))
                return self._inner.integers(*args, **kwargs)

            def __getattr__(self, name):
                return getattr(self._inner, name)

        numpy.random.default_rng = lambda seed=None: CountingGenerator(
            real_default_rng(seed))
        try:
            stats.bootstrap_ci([1.0, 2.0, 3.0, 4.0], n=50)
        finally:
            numpy.random.default_rng = real_default_rng
        assert drawn == [(50, 4)], f"expected one (50, 4) draw, got {drawn}"

    def test_bounds_are_plain_floats(self):
        # These end up in a Record's extra dict and then in JSON; keeping them
        # plain floats means no numpy scalar ever reaches json.dumps.
        low, high = stats.bootstrap_ci([1.0, 2.0, 3.0])
        assert type(low) is float and type(high) is float

    def test_plain_python_sequences_are_accepted(self):
        assert stats.bootstrap_ci((1.0, 2.0, 3.0, 4.0), seed=3) == \
            stats.bootstrap_ci([1.0, 2.0, 3.0, 4.0], seed=3)

    def test_caller_mistakes_fail_loudly(self):
        # Unlike degenerate data, these are programming errors: they cannot
        # appear only after a long run, so failing fast is safe and clearer
        # than inventing an interval.
        with pytest.raises(ValueError):
            stats.bootstrap_ci([1.0, 2.0], alpha=0.0)
        with pytest.raises(ValueError):
            stats.bootstrap_ci([1.0, 2.0], alpha=1.0)
        with pytest.raises(ValueError):
            stats.bootstrap_ci([1.0, 2.0], n=0)
        with pytest.raises(ValueError):
            stats.bootstrap_ci([[1.0, 2.0], [3.0, 4.0]])
