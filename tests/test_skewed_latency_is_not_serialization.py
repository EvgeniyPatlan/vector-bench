"""`served / clients` below a half is latency SKEW, not a serialized pool.

`metrics/latency.scaling_status` refused a scaling efficiency whenever
`served_concurrency` fell below half the client count, and the harness
publishes that refusal as "the clients serialized on the interpreter lock, or
the engine answers one query at a time". For a pool with no think time --
which is every concurrency point, since the client loop has no pause between
queries -- that condition is neither of those things. It is `p50 < mean / 2`,
and a two-mode service time puts it there while the pool scales perfectly.

**The arithmetic, which `TestTheConditionIsAStatementAboutLatencySkew` runs
rather than quotes.** `served` and `outstanding` are sums over the same kept
set, so `served / outstanding == p50 / mean` identically, and
`outstanding / clients` is 0.84-1.00 on every regime this repository has ever
measured. So `served / clients` is `p50 / mean` to within that factor. For a
service time that is `f` with probability `1-p` and `s` otherwise the median
is `f` and the mean is `(1-p) f + p s`, so the refusal fires exactly when
`s/f > (1+p)/p` -- 4.3x at p=0.3, 11x at p=0.1, 21x at p=0.05. A graph cache
smaller than the graph is that distribution, and `CLAUDE.md` names cache
sizing as a first-class fairness concern for MHNSW and VIDX. The next run is
1M rows; the 60k-row smoke run that motivated the guard fits any cache, which
is why this did not show.

**MEASURED HERE, not supplied.** Through the real `ClientPool` in process
mode, 4 clients, synthetic drivers that sleep -- releasing the lock, no shared
queue, so the server is explicitly not the bottleneck:

    regime                        efficiency   served/clients   p50/mean
    narrow latency (control)           0.998            0.973      0.988
    30% of queries at 20x cost         0.984            0.163      0.164
    10% at 50x                         0.959            0.191      0.193
     5% at 50x                         1.000            0.327      0.331
    saturated server, narrow           0.277            0.989      0.994
    saturated AND 30% at 20x           0.233            0.155      0.156
    clients took turns (threads)       0.248            0.246      0.252

Rows two to four scaled LINEARLY and were all flagged
`concurrency_not_achieved`. Those four rows were taken at 2 s windows on a box
whose twenty cores were otherwise busy; the fixture below rebuilds the shape
at 0.5 s and asserts verdicts and ratios rather than decimals, for the reason
`tests/test_g3_process_clients.py` gives.

**The repair is a second estimator of one quantity, not a second threshold.**
`scaling_efficiency` is `qps_n / (n x qps_1)`, and at one client a span holds
no queue and no lock, so `qps_1` is the reciprocal of the per-query service
cost: Little's law makes `efficiency` the share of the clients in service,
computed from the one-client mean instead of the n-client median. The two are
biased in disjoint regimes -- skew drags the median one down, a server-side
queue drags the throughput one down -- and both only downwards, so the larger
is the better bound and the SAME floor of 0.5 judges it. Measured, the
efficiency column is 0.96-1.00 where the pool scaled and 0.22-0.28 where it
did not, so 0.5 is again the midpoint of a measured gap -- a factor of 1.9
below the lowest scaling row and 1.8 above the highest non-scaling one.

**WHAT THIS DOES NOT FIX, asserted rather than admitted in prose.**
`TestTheResidueHoldsBothCausesAndCannotSeparateThem` measures a saturated
server whose service time is ALSO two-moded, and a server whose cache misses
queue on one shared resource. Both land at low efficiency AND low
`served / clients` -- the same quadrant as clients genuinely taking turns, and
within its measured range on all three quantities. No quantity in one window
separates them, which is what `scaling_status`'s docstring has said since it
was written: the status names the observation and both its causes are refused.
That is why no third status was added. The term earns its place on the case
where NEITHER reading is true -- a pool that scaled and is merely skewed.

**Mutation discipline (R4).** Every claim below is checked by injecting the
regression at run time -- a monkeypatch, or a verbatim copy of the pre-change
rule run over the same window -- never by editing a file. Each probe asserts
that its mutation actually landed before reading it, and each docstring
carries the honest value and the mutant value. An identical pair is called out
as a dead probe rather than reported as a guard that holds.

**No engine is started here and no number in this file comes from one.**
"""

from __future__ import annotations

import ctypes
import ctypes.util
import multiprocessing
import os
import random
import sys
import time
from typing import Callable, Dict, List, Optional

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.drivers.base import IndexSpec                            # noqa: E402
from harness.metrics import latency as latency_mod                    # noqa: E402
from harness.metrics.latency import (LatencyCollector,                # noqa: E402
                                     SCALING_CONCURRENCY_NOT_ACHIEVED,
                                     SCALING_OK, SERVED_CONCURRENCY_FLOOR,
                                     achieved_fraction, merge_collectors,
                                     scaling_status, served_fraction,
                                     throughput_fraction, window_concurrency)
from harness.workloads import clients as cl                           # noqa: E402
from tests.fake_driver import (FakeDriver, K, N_QUERIES, N_ROWS,      # noqa: E402
                               index_spec, tiny_dataset)
# The pool G3 retired. The only thing in the tree that can still make clients
# genuinely take turns, which is the true positive this file must not drop.
from tests.retired_thread_pool import thread_window                   # noqa: E402

MP = multiprocessing.get_context("fork")

#: The fast mode's wall cost, identical across every regime here. 0.5 ms for
#: the reason `tests/test_g1_client_concurrency.py:QUERY_S` records: well
#: below the interpreter's 5 ms switch interval, so a lock-holding query's
#: wait lands in the tail rather than in the median, and well above the pool's
#: own per-query bookkeeping.
QUERY_S = 0.0005

#: One measured window. Short, because the fixture sweeps five regimes at
#: three client counts twice; long enough that a one-client window of the
#: WIDEST regime completes over a hundred queries, so its median is the
#: driver's and not the box's.
WINDOW_S = 0.5

#: Windows per point, reduced by MAX qps. Foreign load can only depress a
#: window, and the question asked of each point is what the pool can achieve.
REPEATS = 2

#: Client counts the fixture sweeps. Two and four, because the defect and the
#: repair both have to hold at a count other than the one they were found at,
#: and because `config/profiles/main.yml` sweeps `[1, 2, 4, 8, 16, 32]`.
CLIENT_COUNTS = (1, 2, 4)

#: What every synthetic query returns. Fixed ids rather than the fake's own
#: neighbours, and load-bearing for the reason `test_g1`'s `SYNTHETIC_IDS`
#: gives: `FakeDriver.query` runs a NumPy scan and NumPy RELEASES the
#: interpreter lock inside it, which would land inside the span the instrument
#: sums and stop the lock-holding regime from holding the lock.
SYNTHETIC_IDS = list(range(K))

#: `usleep` through `ctypes.PyDLL`, which does NOT release the interpreter
#: lock around the call -- so four THREADS take turns while the call itself
#: costs almost no CPU. `tests/test_g3_process_clients.py:_HELD_USLEEP` has
#: the measurement that says why a CPU-free wait and not a spin: the spin's
#: verdict is about how many cores the box has spare, and this one's is about
#: the lock.
_LIBC_NAME = ctypes.util.find_library("c") or "libc.so.6"
_HELD_USLEEP = ctypes.PyDLL(_LIBC_NAME).usleep
_HELD_USLEEP.argtypes = [ctypes.c_uint]
_USLEEP_US = int(QUERY_S * 1e6)

_rngs: Dict[int, random.Random] = {}


def _rng() -> random.Random:
    """A generator per PROCESS, keyed by pid.

    The pool forks its clients, so every child inherits one `random` state
    and would draw the identical miss sequence. Not a correctness problem for
    an independent-clients model, but it would make the regimes' slow queries
    land in lockstep, which is a different experiment from the one described.
    """
    pid = os.getpid()
    generator = _rngs.get(pid)
    if generator is None:
        generator = random.Random(0x5EED ^ pid)
        _rngs[pid] = generator
    return generator


# ---------------------------------------------------------------------------
# The regimes
# ---------------------------------------------------------------------------

def regime_narrow() -> Callable[[], None]:
    """The control: one service time, the lock released for all of it."""
    return lambda: time.sleep(QUERY_S)


def regime_skew(p: float, penalty: float) -> Callable[[], None]:
    """A two-mode service time, the lock released for all of it.

    The cache-miss shape: `penalty` x the fast cost on a share `p` of
    queries. Nothing is shared between clients, so the pool MUST scale and
    any refusal over it is a false positive.
    """
    def spend() -> None:
        time.sleep(QUERY_S * (penalty if _rng().random() < p else 1.0))
    return spend


def regime_saturated() -> Callable[[], None]:
    """A server that IS the bottleneck, narrow service time.

    A shared FIFO: a query waits as long as there are queries in flight, and
    the client is blocked on it for all of it. With n clients looping each
    sees depth n, so throughput is flat and efficiency about 1/n -- the shape
    a reader calls an engine ceiling, and it is one. **This is the case the
    guard must not fire on**, and the case the repair's `max` exists for.

    The depth counter is a `multiprocessing.Value` created BEFORE any fork
    and not a `threading.Lock`: under fork each client would otherwise see
    its own queue at depth one, every query would cost the base latency, the
    pool would scale and the row would protect nothing. The same trap
    `tests/test_g1_client_concurrency.py:saturated` records being caught by.
    """
    depth = MP.Value("i", 0)

    def spend() -> None:
        with depth.get_lock():
            depth.value += 1
            here = depth.value
        try:
            time.sleep(QUERY_S * here)
        finally:
            with depth.get_lock():
                depth.value -= 1
    return spend


def regime_saturated_skewed(p: float, penalty: float) -> Callable[[], None]:
    """The combination: a shared FIFO whose SERVICE time is two-moded.

    The case the repair cannot separate from a serialized pool, measured
    rather than argued. Same shared counter as `regime_saturated`, with this
    query's own draw as its cost, so the wait is the depth times a service
    time that has a tail.
    """
    depth = MP.Value("i", 0)

    def spend() -> None:
        with depth.get_lock():
            depth.value += 1
            here = depth.value
        cost = QUERY_S * (penalty if _rng().random() < p else 1.0)
        try:
            time.sleep(cost * here)
        finally:
            with depth.get_lock():
                depth.value -= 1
    return spend


def regime_held() -> Callable[[], None]:
    """Clients genuinely taking turns: a C call that holds the lock.

    The TRUE POSITIVE. Only serializes the retired THREAD pool -- separate
    interpreters do not share one lock -- so every reading of it below comes
    from `tests/retired_thread_pool.py`, which exists for exactly this.
    """
    return lambda: _HELD_USLEEP(_USLEEP_US)


#: The regimes, and which arm each is measured in. The lock-holding one has to
#: be the thread arm or it does not serialize at all; the rest are the real
#: pool, because what they claim is about the pool that ships.
REGIMES: Dict[str, Callable[[], Callable[[], None]]] = {
    "narrow": regime_narrow,
    "skew30x20": lambda: regime_skew(0.30, 20.0),
    "skew10x50": lambda: regime_skew(0.10, 50.0),
    "saturated": regime_saturated,
    "sat_skew30x20": lambda: regime_saturated_skewed(0.30, 20.0),
    "held": regime_held,
}

#: Measured in the retired thread pool. One regime, and that is the point:
#: the live pool cannot be made to serialize through Python's own lock.
THREAD_ARM = ("held",)

#: Scaled linearly and has a skewed latency, so a refusal over it is the
#: false positive this file is about.
SCALED_BUT_SKEWED = ("skew30x20", "skew10x50")

#: The one of them the strict VERDICT is keyed to, and the choice is about
#: variance rather than about preference. Both scale; the efficiency of the
#: 10%-at-50x regime is estimated from a window in which only a tenth of the
#: queries are slow, so its one-client baseline is the noisiest number in the
#: table -- measured 0.959 over 2 s windows, 1.237 over 1 s, 0.789 over 0.6 s
#: and 0.747 in this fixture's own 0.5 s. The 30%-at-20x regime measured
#: 0.973-1.051 over every window length tried. The other row is
#: reported and its DEFECT SHAPE is asserted, which is the part that does not
#: depend on the box.
LOW_VARIANCE_SKEW = ("skew30x20",)

#: Did not scale, and the reason is the engine. Must not be refused.
ENGINE_CEILING = ("saturated",)

#: Did not scale and is skewed too: the residue, which holds both causes.
AMBIGUOUS = ("sat_skew30x20",)

#: The counts at which a regime that fails to scale is UNAMBIGUOUSLY below
#: the floor on the throughput estimator.
#:
#: **Two clients is not one of them, and that is a measurement rather than a
#: convenience.** A pool that serializes completely delivers one client's
#: throughput, so its efficiency is `1/n` -- which is 0.5 at two clients,
#: exactly the floor, for both the true positive and the engine ceiling.
#: Measured: the lock-holding thread arm read efficiency 0.497 and
#: `served / clients` 0.496 at two clients, a hair under; the saturated server
#: read 0.538, a hair over. `TestTwoClientsIsAKnifeEdgeForBothEstimators`
#: reports that pair instead of asserting a verdict nobody can calibrate, and
#: it is a property of the PRE-EXISTING floor and not of this change --
#: `config/profiles/main.yml` sweeps two clients.
UNAMBIGUOUS_COUNTS = tuple(c for c in CLIENT_COUNTS if c >= 4)


# ---------------------------------------------------------------------------
# Fixtures: one table, measured once
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dataset():
    return tiny_dataset()


@pytest.fixture(scope="module")
def queries(dataset):
    return dataset.test[:N_QUERIES]


@pytest.fixture(scope="module")
def base(dataset):
    driver = FakeDriver(dim=dataset.dim, metric=dataset.metric, ef_search=32,
                        ef_half=64.0)
    driver.connect()
    driver.create_schema(index_spec(dataset))
    driver.load(dataset.train[:N_ROWS], list(range(N_ROWS)))
    return driver


def factory_for(base: FakeDriver, spend: Callable[[], None]):
    """A driver factory whose clones spend `spend()` and return fixed ids.

    Bound on the INSTANCE and not on a subclass: `FakeDriver.clone` returns a
    plain `FakeDriver`, so an override would not reach the clients the pool
    builds. Inside the timed region deliberately -- the client brackets
    `driver.query` alone and the spans are the whole input.
    """
    def clone():
        client = base.clone()

        def query(vector, k):
            spend()
            return SYNTHETIC_IDS[:k]

        client.query = query
        return client
    return clone


def _process_point(factory, queries, clients: int) -> Dict[str, float]:
    """One window of the real pool, reduced to the figures the rule reads."""
    pool = cl.ClientPool(factory, queries, clients, K, 32,
                         IndexSpec(dim=queries.shape[1], m=16,
                                   metric="euclidean"), warmup=2)
    pool.start(f"skew-probe {clients}-client warm-up", prefix="skew")
    time.sleep(WINDOW_S)
    window = pool.stop()
    return {"qps": window.qps, "served": window.served_concurrency,
            "outstanding": window.outstanding_concurrency,
            "p50": window.merged.get("latency_p50_ms"),
            "mean": window.merged.get("latency_mean_ms"),
            "clients": clients}


def _thread_point(factory, queries, clients: int) -> Dict[str, float]:
    """One window of the RETIRED thread pool, reduced the same way.

    Through the shared helper, so the only difference from the live pool is
    where the queries ran. The percentiles come from the real
    `merge_collectors` over the very spans the helper's `served` summed.
    """
    raw = thread_window(factory, queries, clients, K, WINDOW_S)
    merged = merge_collectors(raw["collectors"], raw["wall"],
                              cutoff=raw["cutoff"],
                              window_start=raw["window_start"])
    return {"qps": raw["qps"], "served": raw["served"],
            "outstanding": raw["outstanding"],
            "p50": merged.get("latency_p50_ms"),
            "mean": merged.get("latency_mean_ms"),
            "clients": clients}


def _best(arm, factory, queries, clients: int) -> Dict[str, float]:
    return max((arm(factory, queries, clients) for _ in range(REPEATS)),
               key=lambda point: point["qps"])


@pytest.fixture(scope="module")
def table(base, queries) -> Dict[str, Dict[int, Dict[str, float]]]:
    """Every regime at every client count, measured once for the module.

    One fixture because the whole thing is one experiment: every regime pays
    the same fast-mode cost, and measuring the rows at different times under
    different foreign load is how two columns of one table come to disagree
    for a reason that is not the regime.
    """
    out: Dict[str, Dict[int, Dict[str, float]]] = {}
    for name, build in REGIMES.items():
        arm = _thread_point if name in THREAD_ARM else _process_point
        factory = factory_for(base, build())
        one = _best(arm, factory, queries, 1)
        row = {1: dict(one, efficiency=1.0)}
        for clients in CLIENT_COUNTS:
            if clients == 1:
                continue
            point = _best(arm, factory, queries, clients)
            point["efficiency"] = (point["qps"] / (clients * one["qps"])
                                   if one["qps"] else float("nan"))
            row[clients] = point
        out[name] = row
    return out


def fractions(point: Dict[str, float]) -> Dict[str, float]:
    """The three shares the rule and its evidence are expressed in."""
    clients = point["clients"]
    return {"served": point["served"] / clients,
            "outstanding": point["outstanding"] / clients,
            "efficiency": point["efficiency"],
            "p50_over_mean": point["p50"] / point["mean"]}


def report(table) -> str:
    """The table as a reader sees it, for a failure message."""
    lines = [f"{'regime':15s} {'n':>2s} {'eff':>7s} {'srv/n':>7s} "
             f"{'out/n':>7s} {'p50/mean':>9s}"]
    for name, row in table.items():
        for clients, point in sorted(row.items()):
            share = fractions(point)
            lines.append(
                f"{name:15s} {clients:2d} {share['efficiency']:7.3f} "
                f"{share['served']:7.3f} {share['outstanding']:7.3f} "
                f"{share['p50_over_mean']:9.3f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The pre-change rule, kept verbatim so every verdict can be compared
# ---------------------------------------------------------------------------

def pre_change_scaling_status(served: Optional[float],
                              clients: int) -> Optional[str]:
    """`scaling_status` as it stood at fbb48cd, copied body for body.

    R2: a repair has to be shown against the form it replaced, run over the
    same input. Calling the live function with no `efficiency` reaches the
    same arithmetic today, and that is checked below rather than assumed --
    but it is a property of the current default and not of the old file, so
    the old body is kept rather than relied upon.
    """
    fraction = served_fraction(served, clients)
    if fraction is None:
        return None
    if fraction < latency_mod.SERVED_CONCURRENCY_FLOOR:
        return SCALING_CONCURRENCY_NOT_ACHIEVED
    return SCALING_OK


# ---------------------------------------------------------------------------
# The condition, as arithmetic
# ---------------------------------------------------------------------------

class TestTheConditionIsAStatementAboutLatencySkew:
    """`served / clients < 0.5` is `p50 < mean / 2`, over hand-built spans.

    No pool and no driver: which side of a floor a ratio falls on is a
    property of the arithmetic, and the end-to-end classes below measure the
    real thing.
    """

    @staticmethod
    def _collector(durations):
        """A collector holding spans of exactly `durations`, laid end to end
        so every one of them is inside the window."""
        collector = LatencyCollector(warmup=0)
        at = 0.0
        for seconds in durations:
            collector.add_span(at, at + seconds)
            at += seconds
        return collector

    @staticmethod
    def _two_mode(p, penalty, count=1000, fast=0.001):
        """`count` service times, a share `p` of them `penalty` x `fast`.

        Deterministic rather than drawn: the closed form below is about the
        proportion, and a sampled proportion would make the assertion about
        the seed.
        """
        slow = int(round(count * p))
        return [fast * penalty] * slow + [fast] * (count - slow)

    def test_served_over_outstanding_is_exactly_p50_over_mean(self):
        """The identity the whole defect rests on, read off one window.

        Both figures are sums over the SAME kept set, so the ratio is the
        median over the mean with nothing left over. Honest: 0.1644 for a
        30%-at-20x set, equal to `p50 / mean` to 1e-12. Mutant (`percentile`
        rebound on `harness.metrics.latency` to return the mean, which is the
        shape of a mean-latency implementation of `served`): 1.0 over the very
        same window -- so the two differ by a factor of six and the probe is
        not vacuous.

        **Stated as `served / outstanding` and not as `served / clients`,
        because only the first is exact.** `served / clients` is this ratio
        times `outstanding / clients`, which is 0.84-1.00 measured but is not
        1 by construction. The end-to-end classes carry that factor.

        Over whatever set the window KEEPS, which is the invariant that makes
        it exact: both figures are sums over one `span_disposition` call, so a
        span the cutoff trims leaves both of them. Measured unbounded here so
        the closed form below is the only thing the input has to satisfy; the
        trimming rule itself is `test_g1_client_concurrency.py`'s.

        Measured: `served / outstanding` 0.14925373134332912 against a
        `p50 / mean` of 0.14925373134332912 -- a difference of exactly 0.0 --
        and the PUBLISHED columns, which `as_dict` rounds to four decimals,
        give 0.14925373134328357, which is the closed form
        `1 / (1 - p + p s/f)` to the last digit.
        """
        durations = self._two_mode(0.30, 20.0)
        collector = self._collector(durations)
        wall = sum(durations)
        got = window_concurrency([collector], wall)
        merged = merge_collectors([collector], wall)
        ratio = got.served / got.outstanding
        # The unrounded quantities, off the very spans the collector holds.
        kept = sorted(sample.seconds for sample in collector._samples)
        exact = (latency_mod.percentile(kept, 0.50)
                 / (sum(kept) / len(kept)))
        assert ratio == exact
        published = merged["latency_p50_ms"] / merged["latency_mean_ms"]
        assert ratio == pytest.approx(published, abs=1e-12)
        assert published == pytest.approx(1.0 / (0.7 + 0.3 * 20.0), abs=1e-15)

        original = latency_mod.percentile
        try:
            latency_mod.percentile = (
                lambda samples, q: sum(samples) / len(samples))
            # The mutation landed. Four probes in the PR under review aborted
            # silently because what they replaced was ambiguous, so this is
            # asserted and not assumed.
            assert latency_mod.percentile is not original
            mutant = window_concurrency([collector], wall, cutoff=wall,
                                        window_start=0.0)
        finally:
            latency_mod.percentile = original
        assert latency_mod.percentile is original
        mutant_ratio = mutant.served / mutant.outstanding
        assert mutant_ratio == pytest.approx(1.0, rel=1e-9)
        assert mutant_ratio != pytest.approx(ratio, rel=1e-3)

    @pytest.mark.parametrize("p", [0.30, 0.10, 0.05])
    def test_the_old_rule_trips_at_the_penalty_the_closed_form_predicts(
            self, p):
        """`s/f > (1+p)/p`, with the verdict read on both sides of it.

        4.333x at p=0.3, 11x at p=0.1, 21x at p=0.05. The set is built with
        `outstanding == clients` exactly -- the spans tile the window and the
        client count is the number laid down -- so the factor the previous
        test warns about is 1 here and the condition is exactly `p50/mean`.

        Honest and mutant are the two SIDES: a penalty 10% under the
        predicted one reads `ok`, 10% over reads `concurrency_not_achieved`.
        A differing pair, so the boundary is where the closed form says and
        not somewhere the test would have accepted either way.
        """
        boundary = (1.0 + p) / p
        verdicts = {}
        for label, penalty in (("under", boundary * 0.9),
                               ("over", boundary * 1.1)):
            durations = self._two_mode(p, penalty)
            # One collector per client, each holding the same distribution,
            # so the window is `clients` spans deep throughout.
            clients = 4
            collectors = [self._collector(durations) for _ in range(clients)]
            wall = sum(durations)
            got = window_concurrency(collectors, wall, cutoff=wall,
                                     window_start=0.0)
            assert got.outstanding == pytest.approx(clients, rel=1e-9)
            verdicts[label] = pre_change_scaling_status(got.served, clients)
        assert verdicts["under"] == SCALING_OK, (p, verdicts)
        assert verdicts["over"] == SCALING_CONCURRENCY_NOT_ACHIEVED, (p,
                                                                     verdicts)
        assert verdicts["under"] != verdicts["over"]

    def test_the_new_rule_publishes_both_sides_when_the_pool_scaled(self):
        """The repair, over the same hand-built input.

        A 30%-at-20x set at four clients: honest (`efficiency` 1.0 supplied,
        which is what a linearly scaling pool measures) reads `ok`; mutant
        (the pre-change rule, same window) reads
        `concurrency_not_achieved`. Those are the two verdicts the defect is
        the difference between.
        """
        durations = self._two_mode(0.30, 20.0)
        clients = 4
        collectors = [self._collector(durations) for _ in range(clients)]
        wall = sum(durations)
        got = window_concurrency(collectors, wall, cutoff=wall,
                                 window_start=0.0)
        honest = scaling_status(got.served, clients, 1.0)
        mutant = pre_change_scaling_status(got.served, clients)
        assert honest == SCALING_OK
        assert mutant == SCALING_CONCURRENCY_NOT_ACHIEVED
        assert honest != mutant


# ---------------------------------------------------------------------------
# The repair's shape, with no measurement in it
# ---------------------------------------------------------------------------

class TestTheRepairIsOneFloorOverTwoEstimators:
    """`achieved_fraction` and what the rule does with it.

    Hand-built: these are properties of the comparison. The threshold is the
    one that was already there, and that is the claim -- a second constant
    would be a second thing to justify.
    """

    def test_the_fraction_is_the_larger_of_the_two_estimators(self):
        """`max`, and the two orderings are both exercised.

        Honest: 0.9 whichever estimator holds it. Mutant (`min`, the shape of
        a conjunction written as an intersection of refusals): 0.2 and 0.3 on
        the same two inputs, which is below the floor in both -- so the
        mutation moves the verdict and is not a dead probe.
        """
        skewed = achieved_fraction(0.8, 4, 0.9)       # served/n 0.2, eff 0.9
        queued = achieved_fraction(3.6, 4, 0.3)       # served/n 0.9, eff 0.3
        assert skewed == pytest.approx(0.9)
        assert queued == pytest.approx(0.9)
        assert min(0.2, 0.9) == pytest.approx(0.2)
        assert min(0.9, 0.3) == pytest.approx(0.3)
        assert scaling_status(0.8, 4, 0.9) == SCALING_OK
        assert scaling_status(3.6, 4, 0.3) == SCALING_OK

    def test_a_min_rule_would_refuse_both_and_is_run_to_show_it(self):
        """The mutation inside `scaling_status`, not beside it.

        `scaling_status` resolves `achieved_fraction` through this module's
        globals on every call, so rebinding it reaches the rule. Honest: `ok`
        for a skewed-but-scaling window and `ok` for a queued one. Mutant
        (`achieved_fraction` rebound to the `min` form):
        `concurrency_not_achieved` for both.
        """
        original = latency_mod.achieved_fraction
        try:
            latency_mod.achieved_fraction = (
                lambda served, clients, efficiency=None: min(
                    served_fraction(served, clients), efficiency))
            assert latency_mod.achieved_fraction is not original
            mutant_skewed = scaling_status(0.8, 4, 0.9)
            mutant_queued = scaling_status(3.6, 4, 0.3)
        finally:
            latency_mod.achieved_fraction = original
        assert latency_mod.achieved_fraction is original
        assert scaling_status(0.8, 4, 0.9) == SCALING_OK
        assert scaling_status(3.6, 4, 0.3) == SCALING_OK
        assert mutant_skewed == SCALING_CONCURRENCY_NOT_ACHIEVED
        assert mutant_queued == SCALING_CONCURRENCY_NOT_ACHIEVED

    def test_the_floor_is_the_one_that_was_already_there(self):
        """No second constant, and the rule reads the live one.

        Honest (floor 0.5): a window at `served / clients` 0.2 and efficiency
        0.45 is refused. Mutant (floor rebound to 0.4 **on
        `harness.metrics.latency`**): the same window reads `ok`. The rule
        looks the floor up in its own module globals on every call, so the
        rebinding lands; a floor captured in a default argument would make
        this probe dead.
        """
        original = latency_mod.SERVED_CONCURRENCY_FLOOR
        honest = scaling_status(0.8, 4, 0.45)
        try:
            latency_mod.SERVED_CONCURRENCY_FLOOR = 0.4
            assert latency_mod.SERVED_CONCURRENCY_FLOOR != original
            mutant = scaling_status(0.8, 4, 0.45)
        finally:
            latency_mod.SERVED_CONCURRENCY_FLOOR = original
        assert latency_mod.SERVED_CONCURRENCY_FLOOR == original
        assert honest == SCALING_CONCURRENCY_NOT_ACHIEVED
        assert mutant == SCALING_OK
        assert honest != mutant

    def test_an_unmeasured_window_stays_none_even_with_an_efficiency(self):
        """None is "nobody could look", which an efficiency cannot repair.

        A window that counted no query has nothing in it to have been in
        service, and an efficiency over it divides a throughput that was never
        measured. Honest: None. Mutant (`achieved_fraction` rebound to fall
        back to the efficiency, which is the obvious reading of "take
        whichever you have"): `ok` -- a window nobody measured published as a
        pass, which is the fault `PoolWindow.client_bound` draws the same
        distinction to avoid.
        """
        honest = scaling_status(None, 4, 1.0)
        original = latency_mod.achieved_fraction
        try:
            latency_mod.achieved_fraction = (
                lambda served, clients, efficiency=None: (
                    served_fraction(served, clients)
                    if served is not None else efficiency))
            assert latency_mod.achieved_fraction is not original
            mutant = scaling_status(None, 4, 1.0)
        finally:
            latency_mod.achieved_fraction = original
        assert honest is None
        assert mutant == SCALING_OK
        assert honest != mutant
        assert achieved_fraction(None, 4, 1.0) is None
        assert achieved_fraction(1.0, 0, 1.0) is None

    def test_omitting_the_efficiency_is_byte_for_byte_the_old_rule(self):
        """The migration state, asserted so it cannot drift.

        Two callers in `harness/workloads/` still compute this status without
        an efficiency, so the two-argument form has to keep its old verdicts
        exactly -- otherwise this change moves published numbers in a path
        nobody routed. Checked across the floor rather than at one point.

        Honest and mutant here are the two FORMS over one input set, and they
        are deliberately IDENTICAL: that is the claim. The differing pair for
        the same inputs is in the test above, where an efficiency is supplied.
        """
        for served in (0.0, 0.5, 1.9, 2.0, 2.1, 3.9, 4.0, 8.0):
            assert (scaling_status(served, 4)
                    == pre_change_scaling_status(served, 4)), served
        assert scaling_status(None, 4) is pre_change_scaling_status(None, 4)
        # And the new argument is what makes the difference, on an input the
        # two-argument form refuses.
        assert scaling_status(0.8, 4) == SCALING_CONCURRENCY_NOT_ACHIEVED
        assert scaling_status(0.8, 4, 0.99) == SCALING_OK

    def test_throughput_fraction_is_the_efficiency_itself(self):
        """Named, not computed, and the name is the argument.

        `scaling_efficiency` IS a concurrency fraction -- `qps_n / qps_1` is
        Little's law with the one-client service cost, divided by n. A
        function that rescaled it would be claiming otherwise.
        """
        assert throughput_fraction(0.25) == 0.25
        assert throughput_fraction(None) is None
        assert achieved_fraction(0.8, 4, 0.9) == pytest.approx(
            max(served_fraction(0.8, 4), throughput_fraction(0.9)))


# ---------------------------------------------------------------------------
# The false positive, through the real pool
# ---------------------------------------------------------------------------

class TestAPoolThatScaledIsNotRefusedForBeingSkewed:
    """The defect and the repair, measured end to end at 2 and 4 clients.

    Verdicts and ratios against the floor rather than decimals, because this
    box is shared -- the same discipline `tests/test_g3_process_clients.py`
    adopts and for the same reason.
    """

    @pytest.mark.parametrize("regime", LOW_VARIANCE_SKEW)
    @pytest.mark.parametrize("clients", [c for c in CLIENT_COUNTS if c > 1])
    def test_the_old_rule_refused_it_and_the_new_one_does_not(
            self, table, regime, clients):
        """Honest (new rule) against mutant (the pre-change body), one window.

        Measured at 4 clients and 2 s windows: 30%-at-20x reads efficiency
        0.984 with `served / clients` 0.163, and 10%-at-50x 0.959 with 0.191.
        Both scaled -- nothing is shared between these clients -- and both
        were refused on the median alone.

        The two verdicts must differ, which is what makes this a repair
        rather than a restatement.
        """
        point = table[regime][clients]
        share = fractions(point)
        assert share["served"] < SERVED_CONCURRENCY_FLOOR, (
            f"{regime} at {clients}: the median-span estimator no longer "
            f"falls under the floor, so this probe is not measuring the "
            f"defect it claims\n{report(table)}")
        assert share["efficiency"] > SERVED_CONCURRENCY_FLOOR, (
            f"{regime} at {clients}: the pool did not scale, so it is not "
            f"the false-positive regime\n{report(table)}")
        honest = scaling_status(point["served"], clients, share["efficiency"])
        mutant = pre_change_scaling_status(point["served"], clients)
        assert honest == SCALING_OK, report(table)
        assert mutant == SCALING_CONCURRENCY_NOT_ACHIEVED, report(table)
        assert honest != mutant

    @pytest.mark.parametrize("regime", SCALED_BUT_SKEWED)
    def test_the_throughput_really_rose_with_the_client_count(self, table,
                                                              regime):
        """Non-vacuity: the pool scaled, and not only its efficiency column.

        A verdict that moved because the efficiency was mismeasured would be
        the instrument loosened rather than the defect removed. Honest: qps
        at four clients is more than twice qps at one. Mutant: the same
        comparison at one client against itself, which is 1.0 by construction
        and is the reading a broken baseline would give.
        """
        one = table[regime][1]
        four = table[regime][4]
        honest = four["qps"] / one["qps"]
        mutant = one["qps"] / one["qps"]
        assert honest > 2.0, (regime, honest, report(table))
        assert mutant == pytest.approx(1.0)
        assert honest != pytest.approx(mutant, rel=0.1)

    @pytest.mark.parametrize("regime", SCALED_BUT_SKEWED)
    def test_every_skewed_regime_has_the_defect_shape(self, table, regime):
        """The part of the claim that does not depend on the box.

        For BOTH skewed regimes, including the noisier one the verdict is not
        keyed to: the pool's throughput rose with the client count AND its
        `served / clients` is under the floor. That conjunction is the defect
        -- a pool that scaled, refused -- and it holds whatever the efficiency
        estimator's variance is, because the throughput comparison is a ratio
        of counts.

        Honest: the pair above. Mutant (the narrow control over the same two
        quantities): it scaled too, and its `served / clients` is ABOVE the
        floor, so the refusal is attributable to the skew and to nothing else
        about these drivers.
        """
        point = table[regime][4]
        share = fractions(point)
        scaled = point["qps"] / table[regime][1]["qps"]
        assert scaled > 2.0, (regime, scaled, report(table))
        assert share["served"] < SERVED_CONCURRENCY_FLOOR, report(table)
        control = fractions(table["narrow"][4])
        control_scaled = table["narrow"][4]["qps"] / table["narrow"][1]["qps"]
        assert control_scaled > 2.0, report(table)
        assert control["served"] > SERVED_CONCURRENCY_FLOOR, report(table)
        assert pre_change_scaling_status(point["served"], 4) != \
            pre_change_scaling_status(table["narrow"][4]["served"], 4)

    def test_the_control_is_not_refused_by_either_rule(self, table):
        """The narrow-latency regime, so the probe above is about the skew.

        Honest and mutant are the two RULES, and they are deliberately the
        same here: a control that moved would mean the repair had changed
        something other than the skewed case. Called out as such, because an
        identical pair is otherwise the signature of a dead probe.
        """
        for clients in (c for c in CLIENT_COUNTS if c > 1):
            point = table["narrow"][clients]
            share = fractions(point)
            assert share["served"] > SERVED_CONCURRENCY_FLOOR, report(table)
            assert share["efficiency"] > SERVED_CONCURRENCY_FLOOR, (
                report(table))
            new = scaling_status(point["served"], clients,
                                 share["efficiency"])
            old = pre_change_scaling_status(point["served"], clients)
            assert new == old == SCALING_OK, report(table)


# ---------------------------------------------------------------------------
# The trap: a low efficiency is not on its own a refusal
# ---------------------------------------------------------------------------

class TestAnEngineCeilingIsStillPublished:
    """The saturated server, which has a LOW efficiency and must not be
    refused.

    The whole reason the repair is a `max` over two estimators and not a
    second threshold on the efficiency: that column is 0.277 at four clients
    for a server that genuinely is the bottleneck -- indistinguishable from
    the true positive's 0.248 -- and what saves it is `served / clients`
    staying at its client count, because a real queue raises the median along
    with everything else.
    """

    @pytest.mark.parametrize("regime", ENGINE_CEILING)
    @pytest.mark.parametrize("clients", UNAMBIGUOUS_COUNTS)
    def test_the_queue_keeps_its_median_at_the_client_count(
            self, table, regime, clients):
        """Honest (the rule) against mutant (the efficiency term alone).

        Measured at four clients: efficiency 0.277, `served / clients` 0.989,
        `p50 / mean` 0.994, and the median span rose from 0.576 ms at one
        client to 2.075 ms at four -- about the client count, which is what a
        queue does and a stalled client does not.

        Honest: `ok`. Mutant (the efficiency fed through the served slot, so
        the rule judges it alone): `concurrency_not_achieved` -- a real engine
        ceiling refused, which the floor's docstring names as worse than the
        defect.
        """
        point = table[regime][clients]
        share = fractions(point)
        assert share["efficiency"] < SERVED_CONCURRENCY_FLOOR, (
            f"{regime} at {clients}: the saturated regime scaled, so it is "
            f"not modelling an engine ceiling\n{report(table)}")
        assert share["served"] > SERVED_CONCURRENCY_FLOOR, (
            f"{regime} at {clients}: the queue did not raise the median, so "
            f"this row is not the one the floor protects\n{report(table)}")
        honest = scaling_status(point["served"], clients, share["efficiency"])
        mutant = scaling_status(share["efficiency"] * clients, clients)
        assert honest == SCALING_OK, report(table)
        assert mutant == SCALING_CONCURRENCY_NOT_ACHIEVED, report(table)
        assert honest != mutant

    def test_the_median_grew_with_the_client_count(self, table):
        """The mechanism behind the row above, read off the same windows.

        Honest: the saturated regime's median span at four clients is more
        than twice its median at one. Mutant: the same ratio for the
        lock-holding regime, where the median did not move -- measured 0.558
        to 0.559 ms. The pair is what "a real queue raises the whole
        distribution" means, and it is the only quantity measured here that
        tells the two apart.
        """
        honest = (table["saturated"][4]["p50"] / table["saturated"][1]["p50"])
        mutant = table["held"][4]["p50"] / table["held"][1]["p50"]
        assert honest > 2.0, (honest, report(table))
        assert mutant < 1.5, (mutant, report(table))
        assert honest != pytest.approx(mutant, rel=0.2)


# ---------------------------------------------------------------------------
# The true positive, which must survive the repair
# ---------------------------------------------------------------------------

class TestClientsTakingTurnsAreStillRefused:
    """The tripwire, through the retired thread pool.

    `concurrency_not_achieved` is close to unreachable since the clients
    became OS processes, so the guard is a tripwire rather than an expected
    finding -- and a tripwire nothing can trip is indistinguishable from a
    broken one. `tests/retired_thread_pool.py` is the one thing in the tree
    that can still trip it, and the repair has to leave it tripping.
    """

    @pytest.mark.parametrize("clients", [c for c in CLIENT_COUNTS if c > 1])
    def test_both_estimators_are_low_so_the_refusal_stands(self, table,
                                                           clients):
        """Honest (new rule) and mutant (pre-change rule), deliberately EQUAL.

        Measured in the thread arm: efficiency and `served / clients` are the
        same number to three decimals at every client count -- 0.499/0.494 at
        two, 0.248/0.246 at four, 0.124/0.123 at eight (the eight-client
        reading is measured but not swept here) -- because flat throughput and
        a flat median make `served` a constant and `1/n` both figures. So the
        larger of the two is still under the floor and the verdict does not
        move.

        An identical pair is normally a dead probe. It is the deliverable
        here, and the differing pair that proves the input is not inert is in
        the next test.
        """
        point = table["held"][clients]
        share = fractions(point)
        assert share["served"] < SERVED_CONCURRENCY_FLOOR, (
            f"the thread arm no longer serializes at {clients} clients, so "
            f"this file is not measuring the positive it claims"
            f"\n{report(table)}")
        assert share["efficiency"] < SERVED_CONCURRENCY_FLOOR, (
            f"the thread arm's throughput scaled at {clients} clients"
            f"\n{report(table)}")
        honest = scaling_status(point["served"], clients, share["efficiency"])
        mutant = pre_change_scaling_status(point["served"], clients)
        assert honest == SCALING_CONCURRENCY_NOT_ACHIEVED, report(table)
        assert mutant == SCALING_CONCURRENCY_NOT_ACHIEVED, report(table)
        assert honest == mutant

    def test_a_mismeasured_efficiency_is_what_would_drop_it(self, table):
        """What the new term costs if the baseline it divides by is wrong.

        The efficiency is a ratio across TWO windows, and this repository's
        `window_ratio` exists because ratios of separately measured
        quantities have published two wrong results here. Honest (the measured
        efficiency, about a quarter at four clients):
        `concurrency_not_achieved`. Mutant (efficiency 1.0, which is what a
        one-client baseline depressed by foreign load would produce): `ok` --
        the true positive dropped. Run so the failure mode is recorded rather
        than described, and so the report can name the mitigation.
        """
        point = table["held"][4]
        share = fractions(point)
        honest = scaling_status(point["served"], 4, share["efficiency"])
        mutant = scaling_status(point["served"], 4, 1.0)
        assert honest == SCALING_CONCURRENCY_NOT_ACHIEVED, report(table)
        assert mutant == SCALING_OK
        assert honest != mutant

    def test_the_pool_delivered_one_clients_throughput_and_not_more(
            self, table):
        """Non-vacuity for the row above: the throughput really is flat.

        Honest: four lock-holding threads deliver less than 1.5x one thread.
        Mutant: the control regime over the same arm is not available in this
        table, so the comparison is against the same regime at one client,
        which is 1.0 -- and the honest figure has to sit near it rather than
        near four. Both read off the fixture.
        """
        honest = table["held"][4]["qps"] / table["held"][1]["qps"]
        assert honest < 1.5, (honest, report(table))
        assert honest > 0.5, (honest, report(table))


# ---------------------------------------------------------------------------
# What the repair does NOT fix
# ---------------------------------------------------------------------------

class TestTwoClientsIsAKnifeEdgeForBothEstimators:
    """Both estimators are `1/n` at two clients, so both land on the floor.

    **A property of the floor as it already stood, surfaced by this work
    rather than created by it, and `config/profiles/main.yml` sweeps two
    clients.** A pool that serializes completely delivers one client's
    throughput however many clients it has, so its efficiency is `1/n` --
    0.5 at two -- and `served / clients` is the same 0.5 for the same reason.
    An engine ceiling at two clients is also at about a half. The two
    populations the floor was chosen as the midpoint of therefore MEET at two
    clients, and nothing in this change moves that.

    Reported rather than asserted into a verdict, because a verdict here would
    be a verdict about which side of 0.5 the box's noise fell on.
    """

    def test_the_two_regimes_are_not_separated_at_two_clients(self, table):
        """The pair, read off the fixture and asserted only as an overlap.

        Measured: the lock-holding thread arm at two clients read efficiency
        0.497 and `served / clients` 0.496; the saturated server read 0.538
        and 0.985. So the THROUGHPUT estimator does not separate them at two
        clients -- the two readings are within 10% of each other and straddle
        the floor -- while the MEDIAN one still does, by a factor of two.

        Honest: that overlap on the efficiency column. Mutant: the same two
        regimes at four clients, where the efficiency readings are 0.248 and
        0.278 -- still overlapping, which is the point the `max` exists for --
        and the median column separates them 0.244 against 0.989. Both pairs
        read back, so the probe reports the structure rather than a verdict.
        """
        turns_two = fractions(table["held"][2])
        ceiling_two = fractions(table["saturated"][2])
        gap = ceiling_two["efficiency"] / turns_two["efficiency"]
        # Measured 1.082 (0.538 against 0.497). The bound is wide because
        # both halves are near 0.5 by construction and a short window on a
        # busy box moves either of them; what is being asserted is that they
        # do not separate, and a factor of 1.6 either way is still no
        # separation.
        assert 0.625 < gap < 1.6, (gap, report(table))
        assert (ceiling_two["served"] / turns_two["served"]) > 1.6, (
            report(table))
        turns_four = fractions(table["held"][4])
        ceiling_four = fractions(table["saturated"][4])
        assert (ceiling_four["efficiency"] / turns_four["efficiency"]) < 2.0, (
            report(table))
        assert (ceiling_four["served"] / turns_four["served"]) > 2.0, (
            report(table))

    def test_a_completely_serialized_pool_sits_on_the_floor_at_two_clients(
            self):
        """The arithmetic behind it, with no measurement in it.

        Honest (`1/n` at two clients, the exact serialization prediction, on
        both estimators): `ok` -- 0.5 is not BELOW the floor, and the floor is
        documented as a `<`. Mutant (a hair under, which is what the thread
        arm actually measured): `concurrency_not_achieved`. So which verdict a
        perfectly serialized two-client pool gets is decided by noise, under
        the old rule and the new one alike.
        """
        honest = scaling_status(1.0, 2, 0.5)
        mutant = scaling_status(0.999, 2, 0.4995)
        assert honest == SCALING_OK
        assert mutant == SCALING_CONCURRENCY_NOT_ACHIEVED
        assert honest != mutant
        # And the old rule agrees with both, so this is not something the
        # efficiency term introduced.
        assert pre_change_scaling_status(1.0, 2) == SCALING_OK
        assert (pre_change_scaling_status(0.999, 2)
                == SCALING_CONCURRENCY_NOT_ACHIEVED)


class TestTheResidueHoldsBothCausesAndCannotSeparateThem:
    """A saturated server whose service time is also two-moded.

    **This is the reject inside the repair, and it is measured rather than
    conceded.** Such a server has a low efficiency -- it IS the bottleneck --
    and a low `served / clients` -- its service time has a tail -- so it lands
    in the same quadrant as clients genuinely taking turns and within that
    regime's measured range on every quantity here. The `max` does not rescue
    it and nothing available in one window would.

    That is not a mislabel: `concurrency_not_achieved` names the OBSERVATION
    and both its causes are refused, and "the engine answers one query at a
    time" is true of this regime. It is why no third status was added -- a
    value meaning "withheld, cannot attribute" would be a synonym for the one
    that is already there, and would have to be plumbed through `report/` to
    say nothing new.
    """

    @pytest.mark.parametrize("regime", AMBIGUOUS)
    @pytest.mark.parametrize("clients", UNAMBIGUOUS_COUNTS)
    def test_the_combination_is_refused_by_both_rules(self, table, regime,
                                                      clients):
        """Honest and mutant are the two RULES and are deliberately EQUAL.

        Measured at four clients in this fixture: efficiency 0.260 and
        `served / clients` 0.166, against the lock-holding arm's 0.248 and
        0.244; at 2 s windows, 0.233 and 0.155 against 0.248 and 0.246. The
        repair
        leaves this verdict exactly where it was, which is the honest outcome
        for a case one window cannot attribute -- and it means the repair
        never flags anything the old rule did not.
        """
        point = table[regime][clients]
        share = fractions(point)
        assert share["served"] < SERVED_CONCURRENCY_FLOOR, report(table)
        assert share["efficiency"] < SERVED_CONCURRENCY_FLOOR, report(table)
        honest = scaling_status(point["served"], clients, share["efficiency"])
        mutant = pre_change_scaling_status(point["served"], clients)
        assert honest == SCALING_CONCURRENCY_NOT_ACHIEVED, report(table)
        assert honest == mutant

    def test_it_is_inside_the_true_positives_range_on_every_quantity(
            self, table):
        """Why no quantity here separates them, as a measurement.

        The three shares the instrument has -- efficiency, `served / clients`
        and `outstanding / clients` -- are all within a factor of two across
        the two regimes at four clients. Honest: that overlap. Mutant: the
        saturated NARROW regime over the same three, where `served / clients`
        is a factor of four away from the true positive's and the separation
        is real. So the pair shows the overlap is a property of the skew and
        not of the measurement.
        """
        turns = fractions(table["held"][4])
        both = fractions(table["sat_skew30x20"][4])
        ceiling = fractions(table["saturated"][4])
        # Measured at four clients: efficiency 0.260/0.248 = 1.048,
        # `served / clients` 0.166/0.244 = 0.680, `outstanding / clients`
        # 0.946/0.939 = 1.007. Over the probe runs behind this file the
        # served ratio spanned 0.61-0.68, so the bound is set wide enough
        # that it asserts the OVERLAP rather than a decimal.
        for key in ("efficiency", "served", "outstanding"):
            ratio = both[key] / turns[key]
            assert 0.4 < ratio < 2.5, (key, ratio, report(table))
        separated = ceiling["served"] / turns["served"]
        assert separated > 2.0, (separated, report(table))

    def test_the_repair_never_refuses_what_the_old_rule_published(self,
                                                                  table):
        """The whole table, as one claim: the change is one-directional.

        A refusal the old rule did not make would be a new false positive,
        and this is the only probe that would catch one in a regime nobody
        thought to classify. Honest: every point either keeps its verdict or
        moves from refused to `ok`. Mutant (`achieved_fraction` rebound to the
        `min` form): points move the other way, and the count of them is
        read back so the probe cannot pass by finding nothing to compare.
        """
        def verdicts(rule):
            out = {}
            for name, row in table.items():
                for clients, point in row.items():
                    share = fractions(point)
                    out[(name, clients)] = (
                        rule(point["served"], clients, share["efficiency"]),
                        pre_change_scaling_status(point["served"], clients))
            return out

        honest = verdicts(scaling_status)
        loosened = [key for key, (new, old) in honest.items()
                    if new != old]
        tightened = [key for key, (new, old) in honest.items()
                     if new == SCALING_CONCURRENCY_NOT_ACHIEVED
                     and old == SCALING_OK]
        assert tightened == [], (tightened, report(table))
        assert loosened, (
            f"no verdict moved at all, so this probe compared nothing"
            f"\n{report(table)}")

        original = latency_mod.achieved_fraction
        try:
            latency_mod.achieved_fraction = (
                lambda served, clients, efficiency=None: min(
                    served_fraction(served, clients), efficiency)
                if served is not None else None)
            assert latency_mod.achieved_fraction is not original
            mutant = verdicts(scaling_status)
        finally:
            latency_mod.achieved_fraction = original
        assert latency_mod.achieved_fraction is original
        mutant_tightened = [key for key, (new, old) in mutant.items()
                            if new == SCALING_CONCURRENCY_NOT_ACHIEVED
                            and old == SCALING_OK]
        assert mutant_tightened, (
            f"the min form refused nothing the old rule published, so this "
            f"mutation measured nothing\n{report(table)}")
