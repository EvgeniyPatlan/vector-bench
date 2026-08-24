#!/usr/bin/env python3
"""Check that Valkey can re-insert into a populated index, before a long run.

The tuned churn spent two hours re-inserting 803 of 99,000 rows and then made
no further progress. The server was healthy throughout: index caught up with
the keyspace, empty mutation queue, 6 GB resident against a 101 GB limit, and
12% of the container's memory in use. What it was doing was 30 GB of block I/O,
which an in-memory store with persistence disabled has no reason to do.

This reproduces that path directly, at whatever scale is asked for, and reports
the re-insert rate against the bulk load rate. Those two numbers are the whole
question: the load writes into a collection with no index yet and has always
been fast, while churn writes into a populated HNSW graph.

Usage, inside the bench image against a running server container:

    python3 verify-valkey-churn.py --host valkey-srv --rows 100000

Start with a tenth of the corpus. If the rates are within an order of magnitude
of each other the path is healthy and the full run is worth starting; if the
re-insert rate collapses again, kill it and say so rather than waiting.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.drivers.base import ConnectionSpec, IndexSpec  # noqa: E402
from harness.drivers.valkey import ValkeyDriver  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=6379)
    p.add_argument("--rows", type=int, default=100_000,
                   help="corpus size; 990000 is the real one")
    p.add_argument("--dim", type=int, default=1536)
    p.add_argument("--churn", type=float, default=0.10)
    p.add_argument("--load-threads", type=int, default=8)
    return p.parse_args(argv)


def check_persistence(driver) -> bool:
    """The thing most likely to be wrong, checked before anything slow runs.

    An in-memory store with default save points forks and rewrites the whole
    dataset every sixty seconds once writes are flowing. That is both the disk
    traffic and the stalled main thread.
    """
    ok = True
    for key, expected in (("save", ""), ("appendonly", "no")):
        value = driver._config_get(key)
        state = "OK" if str(value or "") == expected else "NOT DISABLED"
        if state != "OK":
            ok = False
        print(f"  {key:12} = {value!r:24} {state}")
    if not ok:
        print("\n  Persistence is on. The image is older than the entrypoint "
              "that disables it:\n    ./run-benchmark.sh build --engine valkey\n")
    return ok


def main(argv=None) -> int:
    args = parse_args(argv)
    spec = ConnectionSpec(host=args.host, port=args.port)
    driver = ValkeyDriver(spec)
    driver.connect()

    print(f"server {args.host}:{args.port}")
    persistence_ok = check_persistence(driver)

    driver.drop_schema()
    index = IndexSpec(dim=args.dim, m=16, metric="angular",
                      ef_construction=200, build_mode="post")
    driver.create_schema(index)

    print(f"\ngenerating {args.rows:,} x {args.dim} vectors "
          f"({args.rows * args.dim * 4 / 1024 ** 3:.1f} GB)")
    vectors = numpy.random.rand(args.rows, args.dim).astype(numpy.float32)
    tags = numpy.arange(args.rows) % 100

    load = driver.load(vectors, tags, threads=args.load_threads)
    print(f"load        {load.wall_seconds:8.1f}s  "
          f"{load.rows_per_second:10,.0f} rows/s   (no index yet)")

    started = time.time()
    driver.create_index(index)
    print(f"index build {time.time() - started:8.1f}s  "
          f"{driver.count_rows():,} docs indexed")

    count = int(args.rows * args.churn)
    print(f"\nchurning {count:,} rows into the populated index")

    started = time.time()
    driver.delete_ids(list(range(count)))
    delete_s = time.time() - started
    print(f"delete      {delete_s:8.1f}s  {count / max(delete_s, 1e-9):10,.0f} rows/s")

    # Chunked so progress is visible: the whole point is telling a slow path
    # from a stalled one, and one blocking call tells you neither.
    chunk = max(1, count // 20)
    started = time.time()
    done = 0
    for begin in range(0, count, chunk):
        end = min(begin + chunk, count)
        ids = list(range(args.rows + begin, args.rows + end))
        driver.insert_rows(ids, vectors[begin:end], tags[begin:end])
        done = end
        rate = done / (time.time() - started)
        print(f"  re-inserted {done:>8,} / {count:,}  "
              f"{rate:8,.0f} rows/s  eta {(count - done) / max(rate, 1e-9):6.0f}s",
              flush=True)
    insert_s = time.time() - started
    rate = count / max(insert_s, 1e-9)

    print(f"\nre-insert   {insert_s:8.1f}s  {rate:10,.0f} rows/s")
    print(f"load was    {load.rows_per_second:10,.0f} rows/s  "
          f"-> re-insert is {load.rows_per_second / max(rate, 1e-9):,.0f}x slower")
    print(f"rows now    {driver.count_rows():,}")

    # A tenth of a million rows at the observed rate is what the real churn
    # costs, and it is the number that decides whether to start the run.
    projected = 99_000 / max(rate, 1e-9)
    print(f"\n990k corpus, 10% churn would take {projected / 60:,.1f} min "
          f"at this rate")
    if not persistence_ok:
        print("...and persistence is still on, so this number means nothing "
              "until the image is rebuilt")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
