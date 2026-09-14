#!/usr/bin/env python3
"""Find out why Valkey's filtered search is slow, and whether we caused it.

The tuned run measured Valkey at 8.2 queries a second on a 1%-selective
filter, against 53.5 at 10% and 479 unfiltered. Getting *slower* as the
predicate narrows is what applying the predicate during a graph walk costs:
with one node in a hundred qualifying, the walk has to travel much further to
find ten of them.

That is not what valkey-search says it does. Its planner is meant to choose
between inline filtering and pre-filtering -- filter first, then search the
survivors exactly -- and Google's Memorystore documentation, which runs this
module, says pre-filtering is picked "when the filtered search space is much
smaller than the original". 9,900 vectors out of 990,000 is much smaller. At
that size an exact scan is a couple of milliseconds. We measured 120.

There is no documented parameter to force the choice, so if the planner can be
reached at all it is through the shape of the query. Two things in ours could
plausibly stop it:

  EF_RUNTIME is an HNSW parameter. Pre-filtering does an exact search and has
  no ef, so naming one may commit the planner to the graph path before it
  considers anything else.

  `-inf` as the lower bound may defeat cardinality estimation. A planner that
  cannot put a number on how many rows qualify cannot know the filtered space
  is small, and inline filtering is the safe default when you don't know.

So this runs the same query four ways and reports latency and recall for each.
If one form is fast, the fix is a one-line change in the driver and the run's
Valkey filtered numbers are ours, not the engine's. If all four are slow, 8.2
QPS is what valkey-search does here and the result stands as measured.

Run it with the lab, which starts Valkey with the flags a tuned run gives it
and tears everything down afterwards:

    ./run-benchmark.sh lab --engine valkey probe-valkey-filter.py --rows 200000

200,000 rows is enough for the shapes to separate and loads in seconds. Add
--rows 990000 --client-memory-gb 16 to confirm at the size the run used.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import numpy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.drivers.base import ConnectionSpec, IndexSpec  # noqa: E402
from harness.drivers.valkey import (INDEX, PREFIX, TAG_FIELD,  # noqa: E402
                                    VECTOR_FIELD, ValkeyDriver, encode_vector)

# The four query shapes. Each is (label, filter expression, ef clause).
#
# The bounded form asks for exactly the same rows as the open one: tags are
# assigned 0..99, so `tag < 1` and `tag == 0` select the same set. Only the
# expression differs, which is the point.
SHAPES = [
    ("as the harness sends it", "@{tag}:[-inf ({t}]", "EF_RUNTIME {ef}"),
    ("no EF_RUNTIME", "@{tag}:[-inf ({t}]", ""),
    ("bounded range", "@{tag}:[0 {tmax}]", "EF_RUNTIME {ef}"),
    ("bounded, no EF_RUNTIME", "@{tag}:[0 {tmax}]", ""),
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=6379)
    p.add_argument("--rows", type=int, default=200_000)
    p.add_argument("--dim", type=int, default=1536)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--ef", type=int, default=100,
                   help="EF_RUNTIME, matching the ops profile's ef_search")
    p.add_argument("--queries", type=int, default=50,
                   help="queries per shape; the slow shapes take ~120 ms each")
    p.add_argument("--load-threads", type=int, default=8)
    return p.parse_args(argv)


def exact_neighbours(train, tags, query, k, threshold):
    """Ground truth over the qualifying rows only, by brute force."""
    qualifying = numpy.flatnonzero(tags < threshold)
    block = train[qualifying]
    # Cosine distance on unit-normalised vectors is a dot product.
    scores = block @ query
    best = numpy.argpartition(-scores, min(k, len(scores) - 1))[:k]
    return set(qualifying[best].tolist())


def run_shape(driver, label, filter_tpl, ef_tpl, queries, tags_below,
              tag_max, k, ef, truth):
    """One query shape, timed per query, scored against exact ground truth."""
    expr = filter_tpl.format(tag=TAG_FIELD, t=tags_below, tmax=tag_max)
    ef_clause = (" " + ef_tpl.format(ef=max(ef, k))) if ef_tpl else ""
    latencies, recalls = [], []

    for vector, expected in truth:
        query = (f"{expr}=>[KNN {k} @{VECTOR_FIELD} $vec{ef_clause}]")
        started = time.perf_counter()
        try:
            raw = driver._conn.execute_command(
                "FT.SEARCH", INDEX, query,
                "PARAMS", "2", "vec", encode_vector(vector),
                "LIMIT", "0", str(k), "NOCONTENT", "DIALECT", "2")
        except Exception as exc:
            return None, None, f"{type(exc).__name__}: {exc}"
        latencies.append((time.perf_counter() - started) * 1000.0)
        got = set()
        for item in raw[1:]:
            key = item.decode() if isinstance(item, bytes) else str(item)
            if key.startswith(PREFIX):
                got.add(int(key[len(PREFIX):]))
        recalls.append(len(got & expected) / max(len(expected), 1))

    return statistics.median(latencies), sum(recalls) / len(recalls), None


def main(argv=None) -> int:
    args = parse_args(argv)
    driver = ValkeyDriver(ConnectionSpec(host=args.host, port=args.port))
    driver.connect()

    print(f"server {args.host}:{args.port}, {args.rows:,} x {args.dim}")
    driver.drop_schema()
    index = IndexSpec(dim=args.dim, m=16, metric="angular",
                      ef_construction=200, build_mode="post")
    driver.create_schema(index)

    # Unit-normalised, so a dot product is the cosine score and the ground
    # truth below needs no division.
    train = numpy.random.rand(args.rows, args.dim).astype(numpy.float32)
    train /= numpy.linalg.norm(train, axis=1, keepdims=True)
    tags = (numpy.arange(args.rows) % 100).astype(numpy.int32)

    load = driver.load(train, tags, threads=args.load_threads)
    print(f"load {load.wall_seconds:.1f}s, building index…", flush=True)
    started = time.time()
    driver.create_index(index)
    print(f"index built in {time.time() - started:.1f}s, "
          f"{driver.count_rows():,} docs\n")

    queries = train[numpy.random.choice(args.rows, args.queries, replace=False)]

    for selectivity, threshold in ((0.10, 10), (0.01, 1)):
        qualifying = int((tags < threshold).sum())
        print(f"=== {selectivity:.0%} selectivity — {qualifying:,} rows qualify")
        truth = [(q, exact_neighbours(train, tags, q, args.k, threshold))
                 for q in queries]

        baseline = None
        for label, filter_tpl, ef_tpl in SHAPES:
            p50, recall, error = run_shape(
                driver, label, filter_tpl, ef_tpl, args.queries, threshold,
                threshold - 1, args.k, args.ef, truth)
            if error:
                print(f"  {label:<26} {error}")
                continue
            if baseline is None:
                baseline = p50
            speedup = baseline / p50 if p50 else 0
            flag = "  <-- faster" if speedup > 1.5 else ""
            print(f"  {label:<26} {p50:8.2f} ms   recall {recall:.4f}"
                  f"   {speedup:5.2f}x{flag}")
        print()

    print("Reading this: a form that is much faster at 1% than at 10% reached "
          "the pre-filter.\nOne that is slower at 1% is walking the graph. If "
          "every form is slower at 1%,\nthe planner is not reachable from a "
          "plain FT.SEARCH and 8 QPS is the engine.")
    driver.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
