"""Ops-harness entrypoint. Runs inside the client container.

The orchestrator has already started the engine's server container with the
right resource limits and server flags; this process only connects to it, runs
the requested workloads, and appends records.

Deliberately does NOT start or configure any server: keeping that in the
orchestrator is what guarantees every engine gets identical treatment and that
the treatment is recorded in the manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from typing import List, Optional

# Allow running as `python3 /opt/vb-harness/main.py` as well as `-m harness.main`.
if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "harness"

from .datasets import load as load_dataset  # noqa: E402
from .drivers.base import ConnectionSpec, IndexSpec  # noqa: E402
from .drivers.postgres import get_driver, known_engines  # noqa: E402
from .metrics.records import Recorder  # noqa: E402
from .workloads import build as build_workload  # noqa: E402
from .workloads import churn as churn_workload  # noqa: E402
from .workloads import concurrency as concurrency_workload  # noqa: E402
from .workloads import filtered as filtered_workload  # noqa: E402
from .workloads.context import RunContext  # noqa: E402

ALL_WORKLOADS = ("build", "concurrency", "filtered", "churn")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="vb-harness",
        description="vector-bench ops harness (build cost, concurrency, filtered, churn)",
    )
    p.add_argument("--engine", required=True, choices=known_engines())
    p.add_argument("--host", required=True, help="server container hostname")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--user", default=os.environ.get("VB_DB_USER", "bench"))
    p.add_argument("--password", default=os.environ.get("VB_DB_PASSWORD", "bench"))
    p.add_argument("--database", default="ann")
    p.add_argument(
        "--server-data-dir", default=None,
        help="path where this process can read the server's data directory "
             "read-only; enables exact on-disk index sizing",
    )

    p.add_argument("--dataset", required=True)
    p.add_argument("--datasets-dir", default="/datasets")
    p.add_argument("--subset-rows", type=int, default=None,
                   help="load only the first N training vectors (smoke profiles)")
    p.add_argument("--max-queries", type=int, default=1000)

    p.add_argument("--m", type=int, required=True)
    p.add_argument("--ef-construction", type=int, default=None)
    p.add_argument("--ef-search", type=int, default=100)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--storage-engine", default="InnoDB")
    p.add_argument("--build-mode", default="post", choices=("post", "incremental"))
    p.add_argument("--load-threads", type=int, default=1)

    p.add_argument("--workloads", default=",".join(ALL_WORKLOADS),
                   help=f"comma-separated subset of: {','.join(ALL_WORKLOADS)}")
    p.add_argument("--client-counts", default="1,2,4,8,16,32")
    p.add_argument("--concurrency-duration", type=float, default=20.0)
    p.add_argument("--selectivities", default="0.01,0.10,0.50")
    p.add_argument("--churn-fractions", default="0.10,0.25")
    p.add_argument("--iterative-scan", default=None,
                   help="pgvector only: off | relaxed_order | strict_order")

    p.add_argument("--run-id", required=True)
    p.add_argument("--resource-pass", default="normalized",
                   choices=("normalized", "tuned"))
    p.add_argument("--engine-tag", default=os.environ.get("VB_ENGINE_TAG"))
    p.add_argument("--march", default=None)
    p.add_argument("--output", required=True, help="JSONL file to append records to")
    p.add_argument("--cache-dir", default="/results/.cache")
    return p.parse_args(argv)


def _floats(csv: str) -> List[float]:
    return [float(x) for x in csv.split(",") if x.strip()]


def _ints(csv: str) -> List[int]:
    return [int(x) for x in csv.split(",") if x.strip()]


def _detect_march() -> str:
    for path in ("/opt/mariadb/.march", "/opt/alisql/.march",
                 "/opt/pgvector-artifacts/.march"):
        try:
            with open(path) as fh:
                return fh.read().strip()
        except OSError:
            continue
    return "unknown"


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    workloads = [w.strip() for w in args.workloads.split(",") if w.strip()]
    unknown = set(workloads) - set(ALL_WORKLOADS)
    if unknown:
        print(f"unknown workloads: {sorted(unknown)}", file=sys.stderr)
        return 2

    spec = ConnectionSpec(
        host=args.host, port=args.port, user=args.user, password=args.password,
        database=args.database, data_dir=args.server_data_dir,
    )
    index = IndexSpec(
        dim=0,  # filled from the dataset below
        m=args.m,
        metric="",  # filled from the dataset below
        ef_construction=args.ef_construction,
        storage_engine=args.storage_engine,
        build_mode=args.build_mode,
    )

    print(f"[vb-harness] loading dataset {args.dataset} from {args.datasets_dir}")
    dataset = load_dataset(args.datasets_dir, args.dataset)
    index.dim = dataset.dim
    index.metric = dataset.metric
    print(
        f"[vb-harness] {dataset.name}: {dataset.n_train:,} train x {dataset.dim} dims, "
        f"{dataset.n_test:,} test, metric={dataset.metric}"
    )

    def driver_factory():
        return get_driver(args.engine, spec)

    recorder = Recorder(args.output)
    ctx = RunContext(
        run_id=args.run_id,
        resource_pass=args.resource_pass,
        recorder=recorder,
        engine_tag=args.engine_tag,
        march=args.march or _detect_march(),
        k=args.k,
        cache_dir=args.cache_dir,
    )

    driver = driver_factory()
    failures: List[str] = []
    try:
        driver.connect()
        print(f"[vb-harness] connected to {args.engine}: {driver.server_version()}")

        if "build" in workloads:
            build_workload.run(
                ctx, driver, dataset, index,
                load_threads=args.load_threads, subset_rows=args.subset_rows,
            )
            # The authoritative row count comes from the engine, not from the
            # profile: ground truth must describe what was actually indexed.
            # A profile that loads a subset (smoke uses 20,000 of 60,000) would
            # otherwise be scored against rows the engine never received.
            indexed_rows = driver.count_rows()
        else:
            # Later workloads need a populated table; without the build workload
            # the caller must have populated it in an earlier invocation.
            rows = driver.count_rows() if _table_exists(driver) else 0
            if rows == 0:
                print("[vb-harness] ERROR: no data present and 'build' was not "
                      "requested; nothing to measure", file=sys.stderr)
                return 3
            driver._index = index
            indexed_rows = rows

        # A plan check before the measurement workloads. Every one of these
        # engines can silently fall back to a full scan, which yields exact
        # results slowly and is indistinguishable from "accurate but slow"
        # unless it is checked.
        driver.set_ef_search(args.ef_search)
        if not driver.explain_uses_vector_index(dataset.test[0], args.k):
            print(
                "[vb-harness] WARNING: the vector index is not in the query plan. "
                "Results below measure a full scan, not ANN search.",
                file=sys.stderr,
            )

        if "concurrency" in workloads:
            concurrency_workload.run(
                ctx, driver_factory, dataset, index,
                ef_search=args.ef_search,
                client_counts=_ints(args.client_counts),
                duration_s=args.concurrency_duration,
                max_queries=args.max_queries,
            )

        if "filtered" in workloads:
            filtered_workload.run(
                ctx, driver, dataset, index,
                ef_search=args.ef_search,
                selectivities=_floats(args.selectivities),
                max_queries=args.max_queries,
                iterative_scan=args.iterative_scan,
                indexed_rows=indexed_rows,
            )

        # Churn mutates the table, so it runs last: everything before it sees a
        # pristine index, and nothing after it inherits a degraded one.
        if "churn" in workloads:
            churn_workload.run(
                ctx, driver, dataset, index,
                ef_search=args.ef_search,
                fractions=_floats(args.churn_fractions),
                max_queries=args.max_queries,
                indexed_rows=indexed_rows,
            )

    except Exception:
        failures.append(traceback.format_exc())
        print("[vb-harness] FAILED:\n" + failures[-1], file=sys.stderr)
    finally:
        try:
            driver.close()
        except Exception:
            pass
        recorder.close()

    if failures:
        return 1
    print(f"[vb-harness] done; records appended to {args.output}")
    return 0


def _table_exists(driver) -> bool:
    try:
        driver.count_rows()
        return True
    except Exception:
        return False


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
