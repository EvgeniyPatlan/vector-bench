#!/usr/bin/env python3
"""Find out why Valkey will not take a write into a populated index.

The tuned churn has now failed the same way five times: a pipeline of a
thousand HSETs into the populated index does not come back, while a single
HSET into that same index, on a connection opened at the moment of the stall,
returns in one millisecond. The index is ready, the mutation queue is empty,
memory is 6 GB against a 101 GB cap, and the server sits at exactly the idle
CPU it had before the corpus was loaded.

Every explanation offered for that was reasoned from the outside and five were
wrong. This runs the experiment instead, and it is small enough to run while
something else is going on:

  1. write a batch into the populated index, having deleted nothing
  2. delete a batch, then write one
  3. halve the batch until a write returns

(1) against (2) says whether the delete is involved at all -- which is the
open question, and the one the benchmark cannot answer because it always
deletes first. (3) says whether it is about size, and where.

    python3 verify-valkey-churn.py --host valkey-srv --rows 100000

Start at a tenth of the corpus. The failure is index-size dependent -- it does
not happen at 20,000 rows and does at 990,000 -- so if a tenth comes back
clean, say so and raise it rather than concluding the path is healthy.

Add --full to run a whole churn cycle at the end and report the rate.
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
    p.add_argument("--batch", type=int, default=1000,
                   help="pipeline size to start the bisect from; the load and "
                        "the churn both use 1000")
    p.add_argument("--full", action="store_true",
                   help="after the probes, run a whole churn cycle and report "
                        "the re-insert rate against the load rate")
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


def _write(driver, first_id: int, count: int, vectors, tags, timeout_s: float):
    """One pipeline of `count` HSETs on a fresh connection. Never raises."""
    from harness.drivers.valkey import PREFIX, TAG_FIELD, VECTOR_FIELD, encode_vector
    conn = driver._write_connection(timeout_s=timeout_s)
    try:
        pipe = conn.pipeline(transaction=False)
        for i in range(count):
            pipe.hset(f"{PREFIX}{first_id + i}", mapping={
                TAG_FIELD: int(tags[i]),
                VECTOR_FIELD: encode_vector(vectors[i]),
            })
        started = time.time()
        pipe.execute()
        return True, time.time() - started
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        try:
            conn.close()
        except Exception:
            pass


def probe_without_deleting(driver, args, vectors, tags) -> bool:
    """Does a batched write into the populated index work on its own?

    The benchmark always deletes first, so it cannot separate 'writing into a
    live HNSW graph is what stalls' from 'writing into one that has just lost
    a tenth of its nodes is what stalls'. This is the half it never runs.
    """
    print(f"\n1. write {args.batch:,} new rows, nothing deleted first")
    base = args.rows * 10
    ok, detail = _write(driver, base, args.batch, vectors, tags, 60.0)
    if ok:
        print(f"   OK in {detail:.2f}s "
              f"({args.batch / max(detail, 1e-9):,.0f} rows/s)")
        print("   -> a batched write into the populated index is fine by "
              "itself. Whatever stalls the churn involves the delete.")
    else:
        print(f"   {detail}")
        print("   -> a batched write into the populated index stalls with no "
              "delete anywhere near it. The delete is not involved.")
    return ok


def probe_after_deleting(driver, args, vectors, tags) -> bool:
    """The churn's own sequence, at one batch instead of ninety-nine."""
    from harness.drivers.valkey import PREFIX
    print(f"\n2. delete {args.batch:,} rows, then write {args.batch:,} new ones")
    conn = driver._write_connection(timeout_s=60.0)
    try:
        started = time.time()
        conn.delete(*[f"{PREFIX}{i}" for i in range(args.batch)])
        print(f"   delete returned in {time.time() - started:.2f}s")
    finally:
        conn.close()
    driver._wait_for_mutations(timeout_s=120)

    base = args.rows * 20
    ok, detail = _write(driver, base, args.batch, vectors, tags, 60.0)
    if ok:
        print(f"   OK in {detail:.2f}s "
              f"({args.batch / max(detail, 1e-9):,.0f} rows/s)")
    else:
        print(f"   {detail}")
    return ok


def bisect_batch(driver, args, vectors, tags) -> int:
    """Halve until a write returns. Returns the size that worked, or 0."""
    print("\n3. halving the batch until a write returns")
    size = args.batch
    base = args.rows * 30
    while size >= 1:
        ok, detail = _write(driver, base, size, vectors, tags, 60.0)
        base += size
        if ok:
            print(f"   {size:>6,} rows: OK in {detail:.2f}s "
                  f"({size / max(detail, 1e-9):,.0f} rows/s)")
            if size == args.batch:
                print("   -> the configured batch works here. The stall needs "
                      "a bigger index than this run built.")
            else:
                print(f"   -> {size:,} is the largest that comes back. "
                      f"99,000 rows at this rate is "
                      f"{99_000 / max(size / max(detail, 1e-9), 1e-9) / 60:.1f} min.")
            return size
        print(f"   {size:>6,} rows: {detail}")
        size //= 2
    print("   -> not even one row at a time. This is not about batching.")
    return 0


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

    probe_vectors = numpy.random.rand(args.batch, args.dim).astype(numpy.float32)
    probe_tags = numpy.arange(args.batch) % 100

    clean = probe_without_deleting(driver, args, probe_vectors, probe_tags)
    after_delete = probe_after_deleting(driver, args, probe_vectors, probe_tags)
    working = args.batch if (clean and after_delete) else bisect_batch(
        driver, args, probe_vectors, probe_tags)

    print("\nsummary")
    print(f"  index holds       {driver.count_rows():,} docs")
    print(f"  write, no delete  {'OK' if clean else 'STALLED'}")
    print(f"  write after delete {'OK' if after_delete else 'STALLED'}")
    print(f"  largest batch     {working:,}" if working else
          "  largest batch     none")

    if not persistence_ok:
        print("\n  ...and persistence is still on, so none of this means "
              "anything until the image is rebuilt")
        return 2

    if not args.full:
        return 0 if working else 1

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
    print(f"\n990k corpus, 10% churn would take "
          f"{99_000 / max(rate, 1e-9) / 60:,.1f} min at this rate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
