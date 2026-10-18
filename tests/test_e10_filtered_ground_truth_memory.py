"""`masked_ground_truth` must get cheaper without answering differently.

Run `full-20261015-215006`, `gist-960-euclidean / tuned / M=32`, ingested
1,000,000 x 960 vectors in 5 h 46 m and then lost the filtered 50% point and
the whole churn step: `percona-ops-m32-incremental-tuned.log` ends mid-sentence
at "computing exact ground truth for tag at a nominal 50% of rows…" with no
traceback, while the server container's memory samples run to the end of the
phase. No exception means a kill, and the client container's limit is 8 GiB.
`masked_ground_truth` was holding a 1.92 GB gather, rebuilding a second
1.92 GB copy of it on every query batch, and asking `argpartition` for a
1.02 GB int64 array it then sliced down to ten columns — on top of the 3.84 GB
`dataset.train` the caller still holds in the same process.

**The bar for fixing that is bit-identity, not closeness.** This function is
the yardstick every filtered recall number in every report is scored against,
and the archived reports were scored against the old array. A version that is
leaner but breaks a tie differently silently rescores every engine, and would
do it invisibly: the output is full-shaped and entirely plausible either way.
So `_master_masked_ground_truth` below is the pre-fix algorithm copied
verbatim, and the tests compare with `numpy.array_equal` across both metrics,
k=1 and k>1, mask densities from `k` qualifying rows to all of them, and block
sizes that do and do not divide the qualifying count.

The last class is the other half of the answer: it measures the two reasons the
corpus rows are gathered in blocks but still **scored whole**, instead of being
scored in blocks and merged into a running top-k, which is what would make the
peak independent of the selectivity. Those tests assert that a blocked merge
*disagrees* with the single pass. If one of them ever fails it is not a defect
— it means the obstacle has gone and the merge has become feasible.

Run: python3 -m pytest tests/test_e10_filtered_ground_truth_memory.py -v
"""

from __future__ import annotations

import os
import sys
import tracemalloc

import numpy
import pytest

VB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, VB_ROOT)

from harness import datasets as ds  # noqa: E402


# ---------------------------------------------------------------------------
# The reference. Every computational statement below is `masked_ground_truth`'s
# as it stood before the memory fix, character for character. Only the argument
# validation is dropped: the rewrite left the pre-existing checks alone and
# appended a `row_block` one, which has its own cases below, so re-stating them
# here would test them twice and prove nothing about the arithmetic. This is
# what the published numbers were produced with, so it is what the rewrite has
# to agree with.
# ---------------------------------------------------------------------------

def _master_masked_ground_truth(train, test, metric, k, mask, batch_size=256):
    qualifying = numpy.nonzero(numpy.asarray(mask))[0]
    if len(qualifying) == 0:
        raise ValueError("no rows qualify for this filter")

    subset = train[qualifying]
    if metric == ds.ANGULAR:
        subset = ds._normalize(subset)
        queries = ds._normalize(test)
    else:
        queries = test

    k = min(k, len(qualifying))
    out = numpy.empty((len(queries), k), dtype=numpy.int64)

    for start in range(0, len(queries), batch_size):
        batch = queries[start:start + batch_size]
        if metric == ds.ANGULAR:
            scores = -(batch @ subset.T)
        else:
            scores = (
                (batch ** 2).sum(axis=1, keepdims=True)
                - 2.0 * (batch @ subset.T)
                + (subset ** 2).sum(axis=1)[None, :]
            )
        idx = numpy.argpartition(scores, kth=k - 1, axis=1)[:, :k]
        ordered = numpy.take_along_axis(
            idx,
            numpy.argsort(numpy.take_along_axis(scores, idx, axis=1), axis=1),
            axis=1,
        )
        out[start:start + len(batch)] = qualifying[ordered]

    return out


# ---------------------------------------------------------------------------
# Corpora. Three kinds, because they fail in different places: random vectors
# exercise the ordinary path, exact duplicates put a tie inside and across the
# k-th boundary, and the one-ULP corpus separates neighbours by less than the
# error a reshaped matrix multiply introduces.
# ---------------------------------------------------------------------------

def _random_corpus(n, dim, seed):
    rng = numpy.random.default_rng(seed)
    return rng.standard_normal((n, dim), dtype=numpy.float32)


def _duplicate_corpus(n, dim, copies, seed):
    """`copies` identical rows per distinct vector: exactly equal distances.

    Exact, not near: every copy of a row is the same bits, so its distance to a
    query is the same bits too, whatever the BLAS does. Which copies come back
    and in what order is then purely a tie-break.
    """
    rng = numpy.random.default_rng(seed)
    base = rng.standard_normal((max(1, n // copies), dim), dtype=numpy.float32)
    return numpy.repeat(base, copies, axis=0)[:n].copy()


def _near_tie_corpus(n, dim, seed):
    """Every odd row is its even neighbour scaled by 1 + 1e-7."""
    rng = numpy.random.default_rng(seed)
    corpus = rng.standard_normal((n, dim), dtype=numpy.float32)
    corpus[1::2] = corpus[0::2][:len(corpus[1::2])] * numpy.float32(1.0000001)
    return corpus


def _mask_of_exactly(n, m, seed):
    """A mask with exactly `m` scattered True entries.

    Scattered rather than a leading slice, so the blocked gather has to follow
    a real index array; exact, so a test can name block sizes that divide the
    qualifying count and block sizes that do not.
    """
    rng = numpy.random.default_rng(seed)
    mask = numpy.zeros(n, dtype=bool)
    mask[rng.permutation(n)[:m]] = True
    return mask


#: (rows, dim, queries). 960 is gist's width and the shape the kill happened
#: at; the narrow ones run the same code with a different BLAS kernel.
SHAPES = [(1200, 960, 24), (2000, 64, 40), (600, 8, 17)]

#: Qualifying counts, from "barely more than k" to "every row". 37 and 611 are
#: prime, so most block sizes cannot divide them.
QUALIFYING = [11, 37, 611, None]

#: Block sizes. 1 and 611 bracket the extremes (one row per block; one block
#: for the whole subset), 7 divides nothing here, 100 divides 600 and 1200.
ROW_BLOCKS = [1, 7, 100, 611, ds.GROUND_TRUTH_ROW_BLOCK]


class TestTheRewriteReturnsTheSameArray:
    """Bit-identity against the pre-fix algorithm, across the whole grid."""

    @pytest.mark.parametrize("metric", [ds.EUCLIDEAN, ds.ANGULAR])
    @pytest.mark.parametrize("k", [1, 10])
    @pytest.mark.parametrize("n_qualify", QUALIFYING)
    @pytest.mark.parametrize("row_block", ROW_BLOCKS)
    def test_random_corpus(self, metric, k, n_qualify, row_block):
        n, dim, nq = SHAPES[0]
        train = _random_corpus(n, dim, seed=101)
        test = _random_corpus(nq, dim, seed=102)
        mask = (numpy.ones(n, dtype=bool) if n_qualify is None
                else _mask_of_exactly(n, n_qualify, seed=103))
        expected = _master_masked_ground_truth(train, test, metric, k, mask)
        got = ds.masked_ground_truth(train, test, metric, k, mask,
                                     row_block=row_block)
        assert numpy.array_equal(got, expected)

    @pytest.mark.parametrize("metric", [ds.EUCLIDEAN, ds.ANGULAR])
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("row_block", [1, 7, 100, 10 ** 7])
    def test_every_shape_at_both_metrics(self, metric, shape, row_block):
        n, dim, nq = shape
        train = _random_corpus(n, dim, seed=201)
        test = _random_corpus(nq, dim, seed=202)
        mask = _mask_of_exactly(n, n // 2, seed=203)
        expected = _master_masked_ground_truth(train, test, metric, 10, mask)
        got = ds.masked_ground_truth(train, test, metric, 10, mask,
                                     row_block=row_block)
        assert numpy.array_equal(got, expected)

    @pytest.mark.parametrize("metric", [ds.EUCLIDEAN, ds.ANGULAR])
    @pytest.mark.parametrize("k", [1, 3, 10, 25])
    @pytest.mark.parametrize("row_block", [1, 7, 64, 10 ** 7])
    def test_tied_distances_from_duplicate_vectors(self, metric, k, row_block):
        """The case a top-k merge across blocks would get wrong.

        With eight identical copies of every vector, the k-th boundary falls
        inside a group of exactly equal distances for most queries, so *which*
        rows come back and in what order is decided entirely by the tie-break.
        `argpartition` does not resolve such a tie by row id — the last class
        in this file measures that — so this is where a cheaper implementation
        stops being the same yardstick.
        """
        n, dim, nq = 2000, 16, 60
        train = _duplicate_corpus(n, dim, copies=8, seed=301)
        test = _random_corpus(nq, dim, seed=302)
        mask = _mask_of_exactly(n, 611, seed=303)
        expected = _master_masked_ground_truth(train, test, metric, k, mask)
        got = ds.masked_ground_truth(train, test, metric, k, mask,
                                     row_block=row_block)
        assert numpy.array_equal(got, expected)

    @pytest.mark.parametrize("metric", [ds.EUCLIDEAN, ds.ANGULAR])
    @pytest.mark.parametrize("row_block", [1, 7, 100, 10 ** 7])
    def test_a_corpus_whose_neighbours_are_one_ulp_apart(self, metric,
                                                         row_block):
        """Distances too close for a reshaped matrix multiply to preserve.

        This corpus is what makes the blocked-merge measurement in the last
        class bite; the rewrite has to come through it unchanged, which it does
        because it never reshapes the matrix multiply.
        """
        n, dim, nq = 4000, 32, 40
        train = _near_tie_corpus(n, dim, seed=401)
        test = _random_corpus(nq, dim, seed=402)
        mask = _mask_of_exactly(n, 2000, seed=403)
        expected = _master_masked_ground_truth(train, test, metric, 10, mask)
        got = ds.masked_ground_truth(train, test, metric, 10, mask,
                                     row_block=row_block)
        assert numpy.array_equal(got, expected)

    @pytest.mark.parametrize("metric", [ds.EUCLIDEAN, ds.ANGULAR])
    @pytest.mark.parametrize("row_block", [7, 100, 10 ** 7])
    def test_a_final_query_batch_that_is_short_but_not_one(self, metric,
                                                           row_block):
        """300 queries at the 256 default, so the last batch holds 44.

        The score matrix is allocated once and used through a `[:len(batch)]`
        view, and operating on the whole buffer instead of that view is wrong
        *only* when the final batch is shorter than the buffer by more than one
        row. Every other grid in this file either fits in a single batch or
        leaves a remainder of exactly 1, and a remainder of 1 cannot tell the
        two apart. Without this case the mistake would be caught only by the
        memory probes, and only because they happen to raise.
        """
        n, dim, nq = 1500, 32, 300
        assert nq % 256 == 44, "the short final batch this test exists for"
        train = _random_corpus(n, dim, seed=911)
        test = _random_corpus(nq, dim, seed=912)
        mask = _mask_of_exactly(n, 611, seed=913)
        expected = _master_masked_ground_truth(train, test, metric, 10, mask)
        got = ds.masked_ground_truth(train, test, metric, 10, mask,
                                     row_block=row_block)
        assert numpy.array_equal(got, expected)

    def test_queries_drawn_from_the_corpus_itself(self):
        """Distance exactly zero for one row, and ties behind it."""
        n, dim = 1500, 24
        train = _duplicate_corpus(n, dim, copies=4, seed=501)
        test = train[:30].copy()
        mask = numpy.ones(n, dtype=bool)
        for metric in (ds.EUCLIDEAN, ds.ANGULAR):
            expected = _master_masked_ground_truth(train, test, metric, 10,
                                                   mask)
            for row_block in (1, 7, 100, 10 ** 7):
                got = ds.masked_ground_truth(train, test, metric, 10, mask,
                                             row_block=row_block)
                assert numpy.array_equal(got, expected), (metric, row_block)

    @pytest.mark.parametrize("batch_size", [1, 3, 256, 10 ** 6])
    def test_the_two_blocking_parameters_are_independent(self, batch_size):
        """`batch_size` batches queries; `row_block` blocks corpus rows.

        They are separate knobs over separate axes, and the rewrite must not
        have quietly tied one to the other.
        """
        train = _random_corpus(900, 48, seed=601)
        test = _random_corpus(37, 48, seed=602)
        mask = _mask_of_exactly(900, 611, seed=603)
        expected = _master_masked_ground_truth(train, test, ds.EUCLIDEAN, 10,
                                               mask, batch_size=batch_size)
        for row_block in (1, 7, 611, 10 ** 7):
            got = ds.masked_ground_truth(train, test, ds.EUCLIDEAN, 10, mask,
                                         batch_size=batch_size,
                                         row_block=row_block)
            assert numpy.array_equal(got, expected), row_block

    def test_the_grid_covers_aligned_and_unaligned_block_sizes(self):
        """The coverage claim, checked rather than asserted in a comment.

        A sweep where every block size happened to divide every qualifying
        count would never exercise a short final block — the place an
        off-by-one in the blocked gather would land — and would still be green.
        """
        counts = [c for c in QUALIFYING if c is not None] + [600, 1000]
        aligned = [(c, b) for c in counts for b in ROW_BLOCKS
                   if b < c and c % b == 0]
        unaligned = [(c, b) for c in counts for b in ROW_BLOCKS
                     if b < c and c % b != 0]
        assert aligned, "no tested block size divides its qualifying count"
        assert unaligned, "every tested block size divides its count exactly"

    def test_the_tag_wrapper_forwards_both_blocking_parameters(self,
                                                               monkeypatch):
        """Asserts the *call*, because the output cannot see this mutation.

        `row_block` provably cannot change the answer — that is what the whole
        class above establishes — so a wrapper that dropped the keyword would
        return exactly the right array and no comparison of outputs could ever
        notice. An earlier version of this test compared outputs and stayed
        green with `row_block=row_block` deleted from `filtered_ground_truth`,
        which is the definition of a dead probe. The sentinel defaults make a
        dropped argument visible, and recording named parameters rather than
        `**kwargs` means forwarding them positionally still passes.

        `tests/test_datasets.py::TestMaskedGroundTruth` already checks that the
        wrapper's array agrees with the mask form, so that is not repeated.
        """
        seen = {}

        def recorder(train, test, metric, k, mask,
                     batch_size=-1, row_block=-1):
            seen["batch_size"] = batch_size
            seen["row_block"] = row_block
            return numpy.zeros((len(test), k), dtype=numpy.int64)

        monkeypatch.setattr(ds, "masked_ground_truth", recorder)
        train = _random_corpus(800, 32, seed=701)
        test = _random_corpus(20, 32, seed=702)

        ds.filtered_ground_truth(train, test, ds.EUCLIDEAN, 5,
                                 tag_threshold=7, buckets=20,
                                 batch_size=64, row_block=7)
        assert seen == {"batch_size": 64, "row_block": 7}

        # And the wrapper must forward its own defaults, not a literal of its
        # own choosing: a hardcoded 8192 would pass the check above.
        seen.clear()
        ds.filtered_ground_truth(train, test, ds.EUCLIDEAN, 5,
                                 tag_threshold=7, buckets=20)
        assert seen == {"batch_size": 256,
                        "row_block": ds.GROUND_TRUTH_ROW_BLOCK}

    @pytest.mark.parametrize("row_block", [0, -1])
    def test_a_non_positive_row_block_raises(self, row_block):
        train = _random_corpus(40, 4, seed=801)
        test = _random_corpus(2, 4, seed=802)
        with pytest.raises(ValueError, match="row_block"):
            ds.masked_ground_truth(train, test, ds.EUCLIDEAN, 3,
                                   numpy.ones(40, dtype=bool),
                                   row_block=row_block)


class TestPeakAllocationIsBounded:
    """What the fix was for, measured rather than argued.

    `tracemalloc` is used because it accounts numpy's array data as well as
    Python objects, and because peak *allocation* is unaffected by whatever
    else is running on the box — unlike a wall-clock number, which is why
    there is no timing assertion anywhere in this file.
    """

    # 30,000 x 512 float32: a 61.44 MB subset against a 30.72 MB score matrix,
    # so the arrays whose size the fix changed dominate the reading, and four
    # `row_block` blocks cover the subset — enough that a block temporary is a
    # fraction of it rather than comparable to it, which is the regime the real
    # shapes are in (500,000 gist rows is 62 blocks — 61 full and a 288-row
    # remainder). Runs in ~0.2 s.
    ROWS, DIM, QUERIES, K = 30_000, 512, 300, 10

    def _probe(self, fn):
        train = _random_corpus(self.ROWS, self.DIM, seed=901)
        test = _random_corpus(self.QUERIES, self.DIM, seed=902)
        mask = numpy.ones(self.ROWS, dtype=bool)
        tracemalloc.start()
        try:
            before = tracemalloc.get_traced_memory()[1]
            out = fn(train, test, mask)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        assert out.shape == (self.QUERIES, self.K)
        return peak - before

    def _unavoidable_bytes(self):
        """The two arrays a single-pass exact brute force has to hold.

        The gathered subset, and one score matrix per query batch. Everything
        else the old implementation allocated was a temporary that did not need
        to exist.
        """
        subset = self.ROWS * self.DIM * 4
        scores = min(256, self.QUERIES) * self.ROWS * 4
        return subset + scores

    def test_the_peak_is_close_to_what_the_algorithm_must_hold(self):
        peak = self._probe(
            lambda tr, te, m: ds.masked_ground_truth(
                tr, te, ds.EUCLIDEAN, self.K, m))
        budget = int(self._unavoidable_bytes() * 1.25)
        assert peak < budget, (
            f"peak {peak / 1e6:.1f} MB exceeds {budget / 1e6:.1f} MB; the "
            f"subset and one score matrix are {self._unavoidable_bytes() / 1e6:.1f} MB"
        )

    def test_the_probe_can_tell_the_two_implementations_apart(self):
        """Without this the bound above could be passing on a dead probe.

        An identical pair of numbers would mean `tracemalloc` is not seeing the
        allocations the fix removed, and the bound would be measuring nothing.
        """
        fixed = self._probe(
            lambda tr, te, m: ds.masked_ground_truth(
                tr, te, ds.EUCLIDEAN, self.K, m))
        master = self._probe(
            lambda tr, te, m: _master_masked_ground_truth(
                tr, te, ds.EUCLIDEAN, self.K, m))
        assert master > fixed * 1.3, (
            f"pre-fix peak {master / 1e6:.1f} MB vs fixed {fixed / 1e6:.1f} MB "
            f"— too close for the bound above to be measuring anything"
        )

    def test_the_angular_path_no_longer_holds_two_subsets(self):
        """The normalized copy used to be live beside the gather it replaced."""
        fixed = self._probe(
            lambda tr, te, m: ds.masked_ground_truth(
                tr, te, ds.ANGULAR, self.K, m))
        subset = self.ROWS * self.DIM * 4
        scores = min(256, self.QUERIES) * self.ROWS * 4
        assert fixed < subset * 2 + scores, (
            f"angular peak {fixed / 1e6:.1f} MB leaves room for a second full "
            f"subset of {subset / 1e6:.1f} MB"
        )


class TestWhyTheRowsAreNotScoredInBlocks:
    """The measurements behind `masked_ground_truth`'s "scored whole".

    Scoring the corpus in blocks and merging a running top-k is the only way to
    make the peak independent of the selectivity, and it is not done because it
    answers differently. Each test here asserts that a blocked merge disagrees
    with the single pass.

    **A failure in this class is news, not a defect.** It means the obstacle it
    measures has gone — a BLAS that is shape-invariant, or an `argpartition`
    that resolves ties by position — and that the merge is worth revisiting.
    """

    @staticmethod
    def _canonical_merge(train, test, metric, k, mask, row_block,
                         batch_size=256):
        """Row-blocked scoring with a running top-k, ties broken by row id.

        The best a blocked merge can do: a *defined* tie-break, since it never
        has the whole score row in front of it and so cannot reproduce an
        arbitrary one.
        """
        qualifying = numpy.nonzero(numpy.asarray(mask))[0]
        n = len(qualifying)
        queries = ds._normalize(test) if metric == ds.ANGULAR else test
        k = min(k, n)
        out = numpy.empty((len(queries), k), dtype=numpy.int64)
        for start in range(0, len(queries), batch_size):
            batch = queries[start:start + batch_size]
            m = len(batch)
            best_s = numpy.full((m, 0), numpy.inf, dtype=numpy.float32)
            best_i = numpy.zeros((m, 0), dtype=numpy.int64)
            for rs in range(0, n, row_block):
                rows = qualifying[rs:rs + row_block]
                block = train[rows]
                if metric == ds.ANGULAR:
                    block = ds._normalize(block)
                    sc = -(batch @ block.T)
                else:
                    sc = (
                        (batch ** 2).sum(axis=1, keepdims=True)
                        - 2.0 * (batch @ block.T)
                        + (block ** 2).sum(axis=1)[None, :]
                    )
                cand_s = numpy.concatenate([best_s, sc], axis=1)
                cand_i = numpy.concatenate(
                    [best_i, numpy.broadcast_to(rows, (m, len(rows)))], axis=1)
                keep = min(k, cand_s.shape[1])
                order = numpy.lexsort((cand_i, cand_s), axis=1)[:, :keep]
                best_s = numpy.take_along_axis(cand_s, order, axis=1)
                best_i = numpy.take_along_axis(cand_i, order, axis=1)
            out[start:start + m] = best_i
        return out

    #: The order the ten tied rows come back in, measured, both metrics. Quoted
    #: in `masked_ground_truth`'s docstring as the reason a blocked merge
    #: cannot reproduce the tie-break, so it is pinned here rather than
    #: described.
    TIED_ORDER = [6, 9, 0, 8, 7, 1, 5, 2, 4, 3]

    @pytest.mark.parametrize("metric", [ds.EUCLIDEAN, ds.ANGULAR])
    def test_argpartition_does_not_resolve_a_tie_by_row_id(self, metric):
        """Ten rows at distance zero inside a hundred: which order?

        Exact arithmetic — every copy is the same bits, so every distance is
        the same bits — and the order that comes back is still not `[0..9]`. It
        is whatever `argpartition`'s introselect left behind, and a merge that
        never sees the whole score row cannot reproduce it.

        The tied group has to sit inside a larger corpus for this to show
        anything, which is the correction this test carries: ten tied rows
        *and nothing else* come back in row order, because introselect leaves
        an array that short alone (`numpy.argpartition(numpy.zeros(10), kth=9)`
        is the identity permutation). A reader who checked the claim against
        that corpus would conclude ties are stable and that the blocked merge
        is available. The second assertion below is there so nobody does.
        """
        dim, copies = 4, 10
        base = _random_corpus(copies, dim, seed=1001)
        train = numpy.repeat(base, copies, axis=0).copy()
        test = base[:1].copy()
        got = ds.masked_ground_truth(train, test, metric, copies,
                                     numpy.ones(len(train), dtype=bool))
        tied = numpy.arange(copies)

        assert set(got[0].tolist()) == set(tied.tolist()), (
            "the tied group itself should come back")
        # The load-bearing claim: an order, but not a defined one.
        assert not numpy.array_equal(got[0], tied), (
            f"argpartition returned the tied group in row order {got[0]!r}; if "
            f"ties are now resolved by position, a blocked top-k merge can "
            f"reproduce them and masked_ground_truth's note should be revisited"
        )
        # The exact permutation, because the docstring quotes it.
        assert got[0].tolist() == self.TIED_ORDER, (
            f"got {got[0].tolist()}, expected {self.TIED_ORDER}; introselect's "
            f"arbitrary order has changed. The refusal above still stands on "
            f"the weaker assertion — update this list and the permutation "
            f"quoted in masked_ground_truth's docstring together"
        )

    def test_a_tied_group_with_nothing_else_is_returned_in_row_order(self):
        """The counter-case, pinned so the docstring's caveat stays true.

        This is the construction the docstring used to describe, and it is
        stable. Keeping it executable is what stops the caveat decaying back
        into the false claim it replaced.
        """
        base = _random_corpus(1, 4, seed=1001)
        train = numpy.repeat(base, 10, axis=0).copy()
        got = ds.masked_ground_truth(train, base[:1].copy(), ds.EUCLIDEAN, 10,
                                     numpy.ones(10, dtype=bool))
        assert got[0].tolist() == list(range(10))

    @pytest.mark.parametrize("metric", [ds.EUCLIDEAN, ds.ANGULAR])
    def test_a_blocked_merge_disagrees_on_tied_distances(self, metric):
        n, dim, nq = 2000, 16, 50
        train = _duplicate_corpus(n, dim, copies=8, seed=1101)
        test = _random_corpus(nq, dim, seed=1102)
        mask = numpy.ones(n, dtype=bool)
        single = ds.masked_ground_truth(train, test, metric, 10, mask)
        merged = self._canonical_merge(train, test, metric, 10, mask,
                                       row_block=512)
        assert not numpy.array_equal(single, merged), (
            "a blocked merge now agrees on tied distances — see the class "
            "docstring before acting on it"
        )

    @pytest.mark.parametrize("metric", [ds.EUCLIDEAN, ds.ANGULAR])
    def test_a_blocked_merge_disagrees_when_neighbours_are_one_ulp_apart(
            self, metric):
        """Not a tie-break problem: the distances themselves come out different.

        Splitting the matrix multiply hands the BLAS a different shape, it
        picks different blocking, and float32 accumulation is not associative.
        The error is tiny — a few parts in a million — and on a corpus this
        tightly packed it is enough to reorder the top ten.
        """
        n, dim, nq = 20_000, 128, 32
        train = _near_tie_corpus(n, dim, seed=1201)
        test = _random_corpus(nq, dim, seed=1202)
        mask = numpy.ones(n, dtype=bool)
        single = ds.masked_ground_truth(train, test, metric, 10, mask)
        merged = self._canonical_merge(train, test, metric, 10, mask,
                                       row_block=4096)
        assert not numpy.array_equal(single, merged), (
            "a blocked merge now agrees on a one-ULP corpus — see the class "
            "docstring before acting on it"
        )

    def test_row_blocked_squared_norms_are_bit_identical(self):
        """The property the hoisted term actually depends on.

        `(x ** 2).sum(axis=1)` reduces along the contiguous axis one row at a
        time, so a row block reads nothing outside itself and the result cannot
        depend on how the rows were grouped. This is why the hoist is safe
        where the matrix multiply is not, and if it ever stops holding the
        hoist is unsound.
        """
        for (n, dim) in [(1000, 960), (997, 128), (5000, 64)]:
            x = _random_corpus(n, dim, seed=1300 + dim)
            whole = (x ** 2).sum(axis=1)
            for blk in (1, 7, 64, 997, 8192):
                got = numpy.empty(n, dtype=whole.dtype)
                for s in range(0, n, blk):
                    got[s:s + blk] = (x[s:s + blk] ** 2).sum(axis=1)
                assert numpy.array_equal(got, whole), (n, dim, blk)

    def test_einsum_is_not_a_drop_in_for_the_hoisted_term(self):
        """The obvious one-liner, and why the hoist does not use it.

        `numpy.einsum("ij,ij->i", x, x)` allocates only the n-element result,
        which is exactly what is wanted, but it accumulates straight through
        where `.sum(axis=1)` sums pairwise. The term is per *column* of the
        score matrix, so it decides the ordering, and substituting it would
        rescore every archived filtered recall number.
        """
        x = _random_corpus(2000, 960, seed=1400)
        assert not numpy.array_equal(numpy.einsum("ij,ij->i", x, x),
                                     (x ** 2).sum(axis=1)), (
            "einsum now agrees with .sum(axis=1) bit for bit; the hoist may "
            "then be written either way, and this is not a defect"
        )
