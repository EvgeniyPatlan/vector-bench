"""E10 -- three tables in `report/render.py` printed a number from a different
ops unit, because their join key and their header named fewer axes than the
ops phase loops.

WHAT WAS OBSERVED, and the whole reason this file exists: the run bundle
`full-20261015-215006` -- profile `full`, engine `percona`, 4 datasets x
M 8/16/32 x both resource passes. None of these three defects can show on a
narrower grid than that, which is why they survived to be published: every
one of them needs two values of one axis inside a single ops unit. Its
`report/report.md` contained, in section 5 -- four of its eleven rows, with
the engine, dataset and storage columns elided:

    | ... | M | ... | Build time  | Ingest rate |
    | ... | 16 | ... |     167.1 s |   598 rows/s |   <- 60,000 rows / 167.1 s = 359
    | ... | 32 | ... |     370.3 s |   545 rows/s |   <- 60,000 rows / 370.3 s = 162
    | ... | 32 | ... |     356.9 s |   598 rows/s |   <- 60,000 rows / 356.9 s = 168
    | ... | 16 | ... |   2,344.5 s |   206 rows/s |   <- 1,000,000 / 2,344.5 = 427

Four of eleven rows. Each one is arithmetically self-refuting: for an
incremental build the build wall clock IS the corpus load, so the row's own
two numbers imply a third rate that neither of them is. The honest rates are
in `report/records.jsonl` and are 359.12, 162.01, 168.13 and 426.53 -- the
rates those rows' own arithmetic implies, to the digit the table prints.

THE CAUSE was `_build_table`'s `ingest_by_key`, keyed on
(engine, dataset, resource_pass, build_mode, storage_engine) with `m` absent.
Three ingest records differing only in M collapsed into one dict entry and
the last one written won. The comment above that dict already recorded
adding `storage_engine` for the identical reason one axis over -- MariaDB's
InnoDB and MyISAM builds collided and one rate was printed for both rows --
so this is the same defect, found again because a run finally varied the axis.

THE SECOND HALF of the same defect is a header rather than a key. Section 5
had no `Pass` column, section 7 (concurrency) had neither `Pass` nor `M`, and
section 9's first churn table had neither, while section 8 (filtered) and the
churn-under-writes table below section 9 already carried both. In the same
run that cost:

  * section 5: fashion-mnist M=8 printed twice, 110.1 s at 6.5 GiB peak RSS
    and 100.3 s at 44.4 GiB, with nothing saying which machine either ran on;
  * section 7: five fashion-mnist rows at one client reading 1,677 / 2,037 /
    1,952 / 1,673 / 1,789 QPS -- normalized M=32, normalized M=8, tuned M=16,
    tuned M=32, tuned M=8, in that order -- rendered as five identical
    identities;
  * section 9: every (dataset, fraction, repair state) five times over, e.g.
    at 10% before repair recalls 0.9865 / 0.9594 / 0.9672 / 0.9860 / 0.9598
    beside QPS 1,503 / 1,898 / 1,561 / 1,331 / 1,829, which reads as one
    noisy measurement rather than five readings of five indexes.

The pass is the damaging half of that. `config/resources/normalized.yml`
grants the server 8 CPUs and a 16 GB limit; the tuned pass in that run gave
it 76 cores and 112.5 GiB, and section 1 names a single pass for the whole
document -- so an 8-core row read as a 76-core one.

PROBE STYLE, per this suite's convention. Every decision test names its
HONEST and its MUTANT value in its docstring, and these are repairs rather
than additions, so each one also exercises the PRE-CHANGE form over the very
same records. There are two of those, because the defect has two halves:

  * the pre-change JOIN is copied verbatim into `_old_ingest_by_key` and run
    against the same ingest records, so "the old key published 598.41 three
    times" is a value this file measures and not a claim it makes;
  * the pre-change HEADERS are column-name lists (`OLD_IDENTITY`) read back
    out of the table the FIXED code renders, so "those columns carry no
    information that tells these rows apart" is measured on the page rather
    than on a tuple hand-copied out of the old source, which could drift.

A repair whose old form also passes is not a repair, and an identical pair is
a dead probe. Two tests here are deliberately non-regression guards rather
than probes and say so in their own docstrings; the rest fail against the
pre-change renderer. Nothing on disk is patched.

Pure logic. No engine, no container, no benchmark run; the records below are
dicts carrying the figures the bundle recorded.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from report import render as render_mod                               # noqa: E402

FASHION = "fashion-mnist-784-euclidean"
SIFT = "sift-128-euclidean"

#: Rows in the two datasets the four wrong rows came from. The divisor in
#: every "rate implied by the same row" figure in this file's header.
ROWS = {FASHION: 60_000, SIFT: 1_000_000}


def _cells(table: str) -> List[List[str]]:
    """A rendered Markdown table as rows of cells, separator row dropped.

    Cells and not a substring search, for the reason `tests/test_config.py`'s
    copy of this helper gives: "598" turns up in several columns of these
    tables and in the raw records, so a membership test against the whole
    fragment is satisfied by the number appearing somewhere else entirely --
    which is exactly the failure mode this file is about.
    """
    rows = []
    for line in table.strip().splitlines():
        line = line.strip()
        if not (line.startswith("|") and line.endswith("|")):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(c and set(c) <= set("-: ") for c in cells):
            continue
        rows.append(cells)
    return rows


def _column(table: str, name: str) -> List[str]:
    """One column's body cells, by header name.

    By name rather than by index so these guards survive another column being
    inserted, and so a table that does not have the column fails loudly here
    rather than silently reading its neighbour.
    """
    rows = _cells(table)
    header, body = rows[0], rows[1:]
    assert name in header, (name, header)
    return [row[header.index(name)] for row in body]


def _identities(table: str, names: List[str]) -> List[tuple]:
    """Each body row reduced to the named columns.

    Used to read the PRE-CHANGE header's identity columns out of the table
    the fixed code renders. Reading them from the rendered text rather than
    from a tuple hand-copied out of the old source is the stronger form: a
    hand-copied tuple can drift from what the renderer actually did, and the
    claim being made here is about the page.
    """
    rows = _cells(table)
    header, body = rows[0], rows[1:]
    for name in names:
        assert name in header, (name, header)
    return [tuple(row[header.index(name)] for name in names) for row in body]


# ---------------------------------------------------------------------------
# The run's own records, for the units that produced the four wrong rows.
# ---------------------------------------------------------------------------

def _ingest(dataset: str, pass_: str, m: int, rate: float) -> Dict[str, Any]:
    return {"phase": "ingest", "engine": "percona", "dataset": dataset,
            "resource_pass": pass_, "m": m, "build_mode": "incremental",
            "storage_engine": "InnoDB", "rows": ROWS[dataset],
            "ingest_rows_per_s": rate}


def _build(dataset: str, pass_: str, m: int, wall: float,
           rss: int) -> Dict[str, Any]:
    return {"phase": "index_build", "engine": "percona", "dataset": dataset,
            "resource_pass": pass_, "m": m, "build_mode": "incremental",
            "storage_engine": "InnoDB", "rows": ROWS[dataset],
            "build_wall_s": wall, "peak_rss_bytes": rss,
            "extra": {"separable_build": False}}


#: The tuned fashion-mnist sweep, verbatim from
#: `full-20261015-215006/report/records.jsonl`. Three M values, one pass, one
#: storage engine, one build mode: the grid that collapsed.
#:
#: **In the file's own order, which is M 16, 32, 8 and not 8, 16, 32.** A
#: dict comprehension keeps the LAST entry for a colliding key, so the order
#: these were written in is what decided which rate the run published three
#: times -- M=8's 598.41. Sorting them here would change the mutant value and
#: stop this file reproducing the published defect.
TUNED_FASHION_INGEST = [
    _ingest(FASHION, "tuned", 16, 359.12),
    _ingest(FASHION, "tuned", 32, 168.13),
    _ingest(FASHION, "tuned", 8, 598.41),
]
TUNED_FASHION_BUILD = [
    _build(FASHION, "tuned", 16, 167.074, 47_670_747_136),
    _build(FASHION, "tuned", 32, 356.870, 47_703_887_872),
    _build(FASHION, "tuned", 8, 100.265, 47_674_806_272),
]


def _old_ingest_by_key(ingest: List[Dict[str, Any]]) -> Dict[Any, Any]:
    """`_build_table`'s pre-change join, verbatim in shape: no `m`.

    Copied rather than described, so "the old form published the wrong rate"
    is a value this file measures and not a claim it makes.
    """
    return {
        (i.get("engine"), i.get("dataset"), i.get("resource_pass"),
         i.get("build_mode"), i.get("storage_engine")):
            i.get("ingest_rows_per_s")
        for i in ingest
    }


class TestTheIngestRateBelongsToTheRowsOwnM:
    """D1. One rate per (unit, M), not one rate per unit."""

    def _table(self) -> str:
        return render_mod._build_table({"build": TUNED_FASHION_BUILD,
                                        "ingest": TUNED_FASHION_INGEST})

    def test_each_m_row_carries_its_own_measured_rate(self):
        """Honest: 598 / 359 / 168 rows/s for M 8 / 16 / 32.

        Mutant (the key with `m` removed again): 598 / 598 / 598 -- the last
        ingest record written, three times. Two of the three cells move, and
        they are two of the four rows the run published wrongly.
        """
        table = self._table()
        assert _column(table, "M") == ["8", "16", "32"]
        assert _column(table, "Ingest rate") == ["598 rows/s", "359 rows/s",
                                                 "168 rows/s"]

    def test_the_pre_change_key_published_one_rate_for_all_three(self):
        """R2: the old form, over these very records, is wrong where the new
        form is right.

        Old form: the five-tuple key collapses three ingest records into ONE
        entry, so all three builds read 598.41. New form: 598.41, 359.12,
        168.13. A repair whose old form also distinguishes them is not a
        repair, and this is the line that says which it is.
        """
        old = _old_ingest_by_key(TUNED_FASHION_INGEST)
        assert len(old) == 1, old
        old_rates = [
            old[(b["engine"], b["dataset"], b["resource_pass"],
                 b["build_mode"], b["storage_engine"])]
            for b in TUNED_FASHION_BUILD]
        assert old_rates == [598.41, 598.41, 598.41], old_rates
        assert len(set(old_rates)) == 1, "the old rates must collide"

        new = _column(self._table(), "Ingest rate")
        assert len(set(new)) == 3, new
        assert [f"{r:,.0f} rows/s" for r in old_rates] != new

    def test_the_rate_and_the_build_time_in_one_row_no_longer_disagree(self):
        """The arithmetic that exposed the defect, run as a guard.

        For an incremental build the build wall clock is the corpus load, so
        `rows / build_wall_s` must equal the rate printed beside it. Honest:
        implied 598.4 / 359.1 / 168.1 against printed 598 / 359 / 168 -- every
        row inside one row/s, and the M=8 figure lands on the recorded 598.41
        to the second decimal, which is the evidence that the two quantities
        really are one measurement. Mutant: the M=16 row implies 359.1 and
        prints 598, the M=32 row implies 168.1 and prints 598 -- the
        published contradiction, in a unit test.
        """
        rows = _cells(self._table())
        header, body = rows[0], rows[1:]
        # Keyed on the M cell rather than zipped against the input list: the
        # table sorts and the fixture is in the records' own order, and a
        # positional pairing of two differently ordered sequences is the
        # defect this file is about wearing a test's clothes.
        printed = {row[header.index("M")]: row[header.index("Ingest rate")]
                   for row in body}
        assert len(printed) == len(TUNED_FASHION_BUILD)
        for record in TUNED_FASHION_BUILD:
            cell = printed[str(record["m"])]
            implied = record["rows"] / record["build_wall_s"]
            shown = float(cell.replace(" rows/s", "").replace(",", ""))
            assert abs(implied - shown) <= 1.0, (record["m"], implied, cell)

    def test_the_sift_row_the_run_published_as_206_reads_427(self):
        """The fourth wrong row, which came from a different dataset and a
        different collision order -- two ingest records, M=16 then M=32, so
        the loser was the FIRST one rather than the middle one.

        Honest: 427 rows/s beside 2,344.5 s over 1,000,000 rows (implied
        426.5). Mutant: 206 rows/s, which is M=32's rate, beside M=16's build.
        """
        ingest = [_ingest(SIFT, "tuned", 16, 426.53),
                  _ingest(SIFT, "tuned", 32, 206.15)]
        build = [_build(SIFT, "tuned", 16, 2_344.488, 48_482_803_712),
                 _build(SIFT, "tuned", 32, 4_850.863, 48_808_448_000)]
        table = render_mod._build_table({"build": build, "ingest": ingest})
        assert _column(table, "Ingest rate") == ["427 rows/s", "206 rows/s"]

        old = _old_ingest_by_key(ingest)
        assert old[("percona", SIFT, "tuned", "incremental", "InnoDB")] \
            == 206.15, "the old form must publish M=32's rate for M=16"

    def test_a_rate_measured_under_the_other_pass_is_not_borrowed(self):
        """The pass stayed in the key when `m` joined it.

        A non-regression guard rather than a probe of the published defect:
        `resource_pass` was in the key before this change and the pre-change
        code passes this test too. Its mutant arm is therefore executed here
        rather than left to a revert -- a key without the pass, over these
        same records, returns 598.41 and would print a rate measured on a
        76-core server against a build that ran on 8 cores.

        Honest: an em dash, because this run measured no normalized M=8 rate.
        Mutant: 598 rows/s.
        """
        build = _build(FASHION, "normalized", 8, 110.140, 6_948_192_256)
        ingest = [_ingest(FASHION, "tuned", 8, 598.41)]
        table = render_mod._build_table({"build": [build], "ingest": ingest})
        assert _column(table, "Ingest rate") == ["—"]

        passless = {(i["engine"], i["dataset"], i["m"], i["build_mode"],
                     i["storage_engine"]): i["ingest_rows_per_s"]
                    for i in ingest}
        mutant = passless[(build["engine"], build["dataset"], build["m"],
                           build["build_mode"], build["storage_engine"])]
        assert mutant == 598.41
        assert render_mod._fmt(mutant, 0, " rows/s") == "598 rows/s"


class TestTheBuildTableNamesItsResourcePass:
    """D1, second half: section 5 had no `Pass` column.

    The run printed fashion-mnist M=8 twice, 110.1 s at 6.5 GiB peak RSS and
    100.3 s at 44.4 GiB. Those are the normalized and the tuned pass -- 8 CPUs
    and 16 GB against 76 and 112.5 GiB -- and nothing on the page said so.
    """

    #: The two rows as the run wrote them. Identical but for the pass.
    PAIR = [_build(FASHION, "normalized", 8, 110.140, 6_948_192_256),
            _build(FASHION, "tuned", 8, 100.265, 47_674_806_272)]

    #: The identity columns section 5 carried before this change.
    OLD_IDENTITY = ["Engine", "Dataset", "M", "Storage", "Build mode"]

    def test_the_two_passes_are_distinguishable_rows(self):
        """Honest: the `Pass` column reads `normalized` and `tuned`.

        Old form, read back out of the rendered table: both rows carry the
        identity ('percona', 'fashion-mnist-784-euclidean', '8', 'InnoDB',
        'incremental'), byte-identical, with 6.5 GiB and 44.4 GiB of peak RSS
        beside them. Distinguishable: False.
        """
        table = render_mod._build_table({"build": self.PAIR, "ingest": []})
        old = _identities(table, self.OLD_IDENTITY)
        assert len(old) == 2 and old[0] == old[1], old

        assert _column(table, "Pass") == ["normalized", "tuned"]
        body = _cells(table)[1:]
        assert body[0] != body[1]

    def test_the_peak_rss_stays_with_its_own_pass(self):
        """The figure the missing column made unreadable, now beside the pass
        that produced it. Honest: 6.5 GiB on normalized, 44.4 GiB on tuned.
        """
        table = render_mod._build_table({"build": self.PAIR, "ingest": []})
        rows = _cells(table)
        header, body = rows[0], rows[1:]
        pairs = {row[header.index("Pass")]: row[header.index("Peak server RSS")]
                 for row in body}
        assert pairs == {"normalized": "6.5 GiB", "tuned": "44.4 GiB"}

    def test_a_record_with_no_pass_renders_the_em_dash(self):
        """Archived records predate the field. The dash is this file's
        convention for "nobody recorded this" everywhere else, and a blank
        would read as a column that failed to render.
        """
        bare = dict(_build(FASHION, "tuned", 8, 100.265, 1))
        bare.pop("resource_pass")
        table = render_mod._build_table({"build": [bare], "ingest": []})
        assert _column(table, "Pass") == ["—"]


def _concurrency(pass_: str, m: int, qps: float, recall: float,
                 spread: float, clients: int = 1) -> Dict[str, Any]:
    """One concurrency record, as `full-20261015-215006` wrote it."""
    return {"phase": "concurrency", "engine": "percona", "dataset": FASHION,
            "resource_pass": pass_, "m": m, "build_mode": "incremental",
            "storage_engine": "InnoDB", "clients": clients, "qps": qps,
            "recall_at_k": recall, "latency_p50_ms": 0.5,
            "latency_p95_ms": 0.6, "latency_p99_ms": 0.7,
            "extra": {"repeats": 3, "qps_spread_pct": spread,
                      "query_pool": 10_000, "query_reuse": 5.0,
                      "scaling_efficiency": 1.0, "served_concurrency": 0.98,
                      "client_cpu_util": 0.03, "client_cpus": 4}}


#: The five one-client fashion-mnist points the run published as five
#: indistinguishable rows, in the order section 7 printed them.
FIVE_UNITS = [
    _concurrency("normalized", 32, 1_677.0, 0.9871, 0.6),
    _concurrency("normalized", 8, 2_037.0, 0.9629, 0.1),
    _concurrency("tuned", 16, 1_952.0, 0.97138, 0.9),
    _concurrency("tuned", 32, 1_673.0, 0.98813, 6.4),
    _concurrency("tuned", 8, 1_789.0, 0.96223, 0.9),
]


class TestTheConcurrencyTableNamesItsOpsUnit:
    """D2. Section 7 carried `Storage` and neither `Pass` nor `M`.

    `_run_unit` loops `m_values x build_modes x storage_engines` inside each
    resource pass and every pass through it rebuilds the index, so a
    concurrency row is a statement about one of those units and not about the
    engine. Section 8's filtered table has carried Pass, M and Build mode
    since PR 3 for exactly this reason; this is that fix, one section over.
    """

    #: The identity columns section 7 carried before this change.
    OLD_IDENTITY = ["Engine", "Dataset", "Storage", "Clients"]

    def _table(self) -> str:
        return render_mod._concurrency_table({"concurrency": FIVE_UNITS})

    def test_five_units_render_as_five_distinguishable_rows(self):
        """Honest: five distinct identities, because `Pass` and `M` are in
        them -- (normalized, 8), (normalized, 32), (tuned, 8), (tuned, 16),
        (tuned, 32).

        Old form, read back out of the rendered table: one identity,
        ('percona', 'fashion-mnist-784-euclidean', 'InnoDB', '1'), five times
        over, with 2,037 and 1,673 QPS beside it -- a 22% spread presented as
        five readings of one configuration. Distinct identities: 1 of 5.
        """
        old = _identities(self._table(), self.OLD_IDENTITY)
        assert len(old) == 5 and len(set(old)) == 1, old

        rows = _cells(self._table())
        header, body = rows[0], rows[1:]
        keys = {(row[header.index("Pass")], row[header.index("M")])
                for row in body}
        assert keys == {("normalized", "8"), ("normalized", "32"),
                        ("tuned", "8"), ("tuned", "16"), ("tuned", "32")}
        assert len({tuple(row) for row in body}) == 5

    def test_each_qps_is_rendered_beside_the_machine_that_produced_it(self):
        """The pass is the damaging half: normalized gives the server 8 CPUs
        and 16 GB, tuned gave it 76 and 112.5 GiB in this run, and section 1
        names one pass for the whole document. Honest: 2,037 is labelled
        `normalized`. Mutant: it is labelled nothing, and reads as tuned.
        """
        rows = _cells(self._table())
        header, body = rows[0], rows[1:]
        by_qps = {row[header.index("QPS")]:
                  (row[header.index("Pass")], row[header.index("M")])
                  for row in body}
        assert by_qps["2,037"] == ("normalized", "8")
        assert by_qps["1,677"] == ("normalized", "32")
        assert by_qps["1,952"] == ("tuned", "16")
        assert by_qps["1,673"] == ("tuned", "32")
        assert by_qps["1,789"] == ("tuned", "8")

    def test_the_m_column_sorts_as_a_number(self):
        """`full` sweeps 8/16/32, which as text sorts 16, 32, 8. The filtered
        table's comment records the same trap; this is the same `m or 0`.
        """
        rows = _cells(render_mod._concurrency_table(
            {"concurrency": [r for r in FIVE_UNITS
                             if r["resource_pass"] == "tuned"]}))
        header, body = rows[0], rows[1:]
        assert [row[header.index("M")] for row in body] == ["8", "16", "32"]

    def test_a_record_with_no_unit_axes_renders_dashes(self):
        """A point written before the fields existed gets the em dash, not a
        zero: `_fmt(None, 0)` is "nobody measured this" and `0` would be a
        graph degree no engine was ever given.
        """
        bare = {"phase": "concurrency", "engine": "valkey", "dataset": "d",
                "clients": 32, "qps": 5_820.0, "extra": {}}
        table = render_mod._concurrency_table({"concurrency": [bare]})
        assert _column(table, "Pass") == ["—"]
        assert _column(table, "M") == ["—"]


def _churn(pass_: str, m: int, fraction: float, qps: float, recall: float,
           state: str | None, index_bytes: int) -> Dict[str, Any]:
    """One churn measurement, as `full-20261015-215006` wrote it."""
    extra: Dict[str, Any] = {}
    if state is not None:
        extra["maintenance_state"] = state
        if state == "after":
            extra["maintenance"] = "ALTER TABLE (index replace)"
            extra["maintenance_s"] = 0.0
    return {"phase": "churn", "engine": "percona", "dataset": FASHION,
            "resource_pass": pass_, "m": m, "build_mode": "incremental",
            "storage_engine": "InnoDB", "churn_fraction": fraction,
            "qps": qps, "recall_at_k": recall, "index_bytes": index_bytes,
            "extra": extra}


#: The five 10%-before-repair rows the run published with byte-identical
#: identity cells, in the order section 9 printed them.
FIVE_CHURN_UNITS = [
    _churn("normalized", 32, 0.10, 1_503.0, 0.9865, "before", 285_212_672),
    _churn("normalized", 8, 0.10, 1_898.0, 0.9594, "before", 281_018_368),
    _churn("tuned", 16, 0.10, 1_561.0, 0.9672, "before", 281_018_368),
    _churn("tuned", 32, 0.10, 1_331.0, 0.9860, "before", 281_018_368),
    _churn("tuned", 8, 0.10, 1_829.0, 0.9598, "before", 281_018_368),
]


class TestTheChurnTableNamesItsOpsUnit:
    """D3. Section 9's first churn table carried neither `Pass` nor `M`.

    `churn_group` in the same function has always held both constant when
    deciding what a row is tied with, so the renderer knew they were part of
    a row's identity while the header named neither -- and the
    churn-under-writes table immediately below prints both. This is the same
    shape as the `Repair` column, which was also a fix: two real measurements
    rendered as one row twice, read as a noisy measurement.
    """

    #: The identity columns section 9's first table carried before this
    #: change. `Repair` is in the list because it is itself a fix of this
    #: shape and the five units collide through it too.
    OLD_IDENTITY = ["Engine", "Dataset", "Storage", "Churn", "Repair"]

    def test_five_units_render_as_five_distinguishable_rows(self):
        """Honest: five distinct identities.

        Old form, read back out of the rendered table: one identity,
        ('percona', 'fashion-mnist-784-euclidean', 'InnoDB', '10%',
        'before'), five times, with recalls 0.9865 / 0.9594 / 0.9672 /
        0.9860 / 0.9598 and QPS 1,503 / 1,898 / 1,561 / 1,331 / 1,829 beside
        it. Distinct identities: 1 of 5.
        """
        table = render_mod._churn_table({"churn": FIVE_CHURN_UNITS})
        old = _identities(table, self.OLD_IDENTITY)
        assert len(old) == 5 and len(set(old)) == 1, old

        rows = _cells(table)
        header, body = rows[0], rows[1:]
        keys = {(row[header.index("Pass")], row[header.index("M")])
                for row in body}
        assert keys == {("normalized", "8"), ("normalized", "32"),
                        ("tuned", "8"), ("tuned", "16"), ("tuned", "32")}
        assert len({tuple(row) for row in body}) == 5

    def test_the_repair_pair_within_one_unit_is_still_two_rows(self):
        """The columns added here must not cost the distinction the `Repair`
        column exists to make.

        A non-regression guard rather than a probe: `Repair` predates this
        change and the pre-change code passes this test too. It is here
        because adding two columns ahead of `Repair` and putting both in the
        sort key is exactly the edit that could reorder a before/after pair,
        and `_CHURN_STATE_ORDER` exists because a plain string sort already
        did once -- "after" sorts before "before".

        Honest: one unit's pair renders as two rows, `before` then `after
        ALTER TABLE (index replace)`, in the order they were measured.
        """
        pair = [_churn("tuned", 16, 0.10, 1_561.0, 0.9672, "before",
                       281_018_368),
                _churn("tuned", 16, 0.10, 1_561.0, 0.9672, "after",
                       281_018_368)]
        rows = _cells(render_mod._churn_table({"churn": pair}))
        header, body = rows[0], rows[1:]
        states = [row[header.index("Repair")] for row in body]
        assert states[0] == "before"
        assert states[1].startswith("after ALTER TABLE (index replace)")
        assert body[0] != body[1]

    def test_the_m_column_sorts_as_a_number(self):
        rows = _cells(render_mod._churn_table(
            {"churn": [r for r in FIVE_CHURN_UNITS
                       if r["resource_pass"] == "tuned"]}))
        header, body = rows[0], rows[1:]
        assert [row[header.index("M")] for row in body] == ["8", "16", "32"]

    def test_a_record_with_no_unit_axes_renders_dashes(self):
        bare = {"phase": "churn", "engine": "mariadb", "dataset": "d",
                "storage_engine": "InnoDB", "churn_fraction": 0.25,
                "qps": 23.0, "recall_at_k": 0.99, "extra": {}}
        table = render_mod._churn_table({"churn": [bare]})
        assert _column(table, "Pass") == ["—"]
        assert _column(table, "M") == ["—"]


class TestEveryOpsUnitTableSpellsTheTwoColumnsTheSameWay:
    """One reader convention across the five tables that render an ops unit.

    A column named `Pass` in one section and `Resource pass` in the next is a
    second thing for a reader to notice, and this report is read by scanning
    down a column. The two already-correct tables are in the list as the
    reference and as the non-vacuity check: if the header text ever changes
    there, this test says so rather than silently passing on three tables.
    """

    def test_all_five_tables_carry_pass_and_m(self):
        build = render_mod._build_table(
            {"build": TUNED_FASHION_BUILD, "ingest": TUNED_FASHION_INGEST})
        concurrency = render_mod._concurrency_table(
            {"concurrency": FIVE_UNITS})
        churn = render_mod._churn_table({"churn": FIVE_CHURN_UNITS})
        filtered = render_mod._filtered_table({"filtered": [
            {"phase": "filtered", "engine": "percona", "dataset": FASHION,
             "resource_pass": "tuned", "m": 16, "build_mode": "incremental",
             "storage_engine": "InnoDB", "selectivity": 0.01,
             "recall_at_k": 0.99, "qps": 100.0, "extra": {}}]})
        during = render_mod._churn_during_writes_table(
            {"churn_during_writes": [
                {"engine": "percona", "dataset": FASHION,
                 "resource_pass": "tuned", "m": 16,
                 "build_mode": "incremental", "storage_engine": "InnoDB",
                 "churn_fraction": 0.10, "readers": 4,
                 "baseline_p99_ms": 10.0, "under_writes_p99_ms": 20.0,
                 "p99_ratio": 2.0, "baseline_qps": 100.0,
                 "under_writes_qps": 50.0, "writer_rows_per_s": 10.0}]})
        for name, table in (("build", build), ("concurrency", concurrency),
                            ("churn", churn), ("filtered", filtered),
                            ("churn_during_writes", during)):
            header = _cells(table)[0]
            assert "Pass" in header, (name, header)
            assert "M" in header, (name, header)
