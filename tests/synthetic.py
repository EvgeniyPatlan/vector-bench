"""Generate a synthetic run directory so the report pipeline can be exercised
without spending hours on a real benchmark.

Used by tests/test_report.py and by `tests/make-demo-report.sh`, which produces a
sample report for reviewing layout and wording before any real data exists.

The numbers are fabricated and shaped to exercise every branch of the report:
one engine that wins on throughput, one that wins on recall, one plan failure,
one short-result filtered case, and a churn series that degrades.
"""

from __future__ import annotations

import json
import math
import os
import random
from typing import Any, Dict, List

ENGINES = ("mariadb", "alisql", "pgvector")
DATASET = "fashion-mnist-784-euclidean"


def _timestamp(i: int) -> str:
    return f"2026-08-03T12:{i % 60:02d}:00Z"


def make_manifest(run_id: str) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "started_at": "2026-08-03T12:00:00Z",
        "finished_at": "2026-08-03T14:30:00Z",
        "status": "completed",
        "framework": {"name": "vector-bench", "commit": "synthetic"},
        "host": {
            "cpu": {
                "model": "Synthetic CPU @ 3.5GHz",
                "arch": "x86_64",
                "logical_cpus": 20,
                "physical_cores": 14,
                "sockets": 1,
                "numa_nodes": 1,
                "threads_per_core": 2,
                "hybrid": True,
                "performance_cpus": list(range(12)),
                "efficiency_cpus": list(range(12, 20)),
                "simd_flags": ["avx", "avx2", "fma", "f16c"],
                "has_avx512": False,
            },
            "total_ram_bytes": 64 * 1024 ** 3,
            "available_ram_bytes": 40 * 1024 ** 3,
            "kernel": "6.8.0-synthetic",
            "docker_version": "29.1.3",
            "cgroup_version": "v2",
            "hostname": "synthetic",
            "python_version": "3.12.3",
        },
        "engines": {
            "mariadb": {
                "source": {"tag": "mariadb-11.8.8", "commit": "a" * 40},
                "build": {"march": "x86-64-v3", "build_type": "RelWithDebInfo"},
                "images": {"runtime": {"ref": "vector-bench/mariadb-runtime", "id": "sha256:1"},
                           "bench": {"ref": "vector-bench/mariadb-bench", "id": "sha256:2"}},
            },
            "alisql": {
                "source": {"tag": "AliSQL-8.0.44-2", "commit": "b" * 40},
                "build": {"march": "x86-64-v3", "build_type": "RelWithDebInfo"},
                "images": {"runtime": {"ref": "vector-bench/alisql-runtime", "id": "sha256:3"},
                           "bench": {"ref": "vector-bench/alisql-bench", "id": "sha256:4"}},
            },
            "pgvector": {
                "source": {"tag": "v0.8.6", "commit": "c" * 40},
                "build": {"march": "x86-64-v3", "build_type": "Release"},
                "images": {"runtime": {"ref": "vector-bench/pgvector-runtime", "id": "sha256:5"},
                           "bench": {"ref": "vector-bench/pgvector-bench", "id": "sha256:6"}},
            },
        },
        "config": {
            "profile": {"name": "synthetic",
                        "description": "fabricated data for pipeline testing"},
            "resource_pass": "normalized",
            "resolved_resources": {
                "name": "normalized",
                "server_cpuset": "0,2,4,6,8,10",
                "client_cpuset": "12,13",
                "server_cpu_count": 6,
                "server_memory_bytes": 16 * 1024 ** 3,
                "buffer_bytes": int(16 * 1024 ** 3 * 0.35),
                "graph_cache_bytes": int(16 * 1024 ** 3 * 0.35),
                "maintenance_bytes": int(16 * 1024 ** 3 * 0.15),
                "build_threads": 1,
                "shm_size": "2g",
                "hybrid_cpu": True,
                "core_class_used": "performance",
                "warnings": [],
            },
        },
        "phases": [],
        "warnings": [
            "This CPU has no AVX-512. MariaDB MHNSW and AliSQL VIDX both document "
            "AVX-512 distance kernels, so both are running narrower SIMD paths here.",
            "Hybrid CPU detected (12 performance / 8 efficiency logical CPUs). Runs "
            "are pinned to one core class.",
        ],
    }


def make_records(run_id: str) -> List[Dict[str, Any]]:
    rng = random.Random(20260803)
    records: List[Dict[str, Any]] = []

    # Distinct shapes so the charts are visibly different per engine.
    shape = {
        "mariadb": {"speed": 9000.0, "ceiling": 0.995},
        "alisql": {"speed": 7200.0, "ceiling": 0.998},
        "pgvector": {"speed": 12000.0, "ceiling": 0.988},
    }

    base = {
        "run_id": run_id, "dataset": DATASET, "metric_space": "euclidean",
        "resource_pass": "normalized", "march": "x86-64-v3", "k": 10,
    }

    # --- recall / QPS ------------------------------------------------
    n = 0
    for engine in ENGINES:
        for m in (8, 16, 32):
            for ef in (10, 20, 40, 80, 160, 320):
                n += 1
                cfg = shape[engine]
                recall = cfg["ceiling"] * (1 - math.exp(-ef / (14.0 + 40.0 / m)))
                recall = min(cfg["ceiling"], recall * (0.94 + 0.02 * math.log2(m)))
                qps = cfg["speed"] / (1 + ef / 22.0) / (1 + m / 48.0)
                records.append({
                    **base, "phase": "recall_qps", "engine": engine,
                    "engine_version": f"{engine}-synthetic",
                    "storage_engine": "heap" if engine == "pgvector" else "InnoDB",
                    "m": m, "ef_search": ef,
                    "recall_at_k": round(min(1.0, recall + rng.uniform(-0.004, 0.004)), 6),
                    "qps": round(qps * rng.uniform(0.97, 1.03), 2),
                    "clients": 1,
                    "latency_p50_ms": round(1000.0 / qps, 4),
                    "latency_p95_ms": round(1600.0 / qps, 4),
                    "latency_p99_ms": round(2400.0 / qps, 4),
                    "index_bytes": int(60e6 + m * 3.1e6),
                    "vector_index_used": True,
                    "timestamp": _timestamp(n),
                })

    # One plan failure, so the Validity section has something to report.
    records.append({
        **base, "phase": "recall_qps", "engine": "alisql",
        "engine_version": "alisql-synthetic", "storage_engine": "InnoDB",
        "m": 8, "ef_search": 800, "recall_at_k": 1.0, "qps": 41.3, "clients": 1,
        "vector_index_used": False, "timestamp": _timestamp(99),
        "notes": "optimizer chose a full scan at this LIMIT",
    })

    # --- build cost ---------------------------------------------------
    for engine in ENGINES:
        incremental = engine != "pgvector"
        for m in (8, 16, 32):
            ingest = 5200.0 if not incremental else 2300.0 - m * 18
            wall = 60000 / ingest
            records.append({
                **base, "phase": "ingest", "engine": engine,
                "engine_version": f"{engine}-synthetic", "m": m,
                "storage_engine": "heap" if engine == "pgvector" else "InnoDB",
                "rows": 60000, "ingest_wall_s": round(wall, 2),
                "ingest_rows_per_s": round(ingest, 1), "ingest_threads": 1,
                "timestamp": _timestamp(m),
            })
            build_wall = wall if incremental else wall + m * 1.6
            records.append({
                **base, "phase": "index_build", "engine": engine,
                "engine_version": f"{engine}-synthetic", "m": m,
                "storage_engine": "heap" if engine == "pgvector" else "InnoDB",
                "build_mode": "incremental" if incremental else "post",
                "rows": 60000,
                "build_wall_s": round(build_wall, 2),
                "ingest_wall_s": round(wall, 2), "ingest_threads": 1,
                "peak_rss_bytes": int(1.1e9 + m * 4.2e7),
                "index_bytes": int(60e6 + m * 3.1e6),
                "table_bytes": int(1.9e8),
                "extra": {"separable_build": not incremental},
                "timestamp": _timestamp(m),
            })

    # --- concurrency ---------------------------------------------------
    for engine in ENGINES:
        # Different saturation points, which is the whole point of the chart.
        knee = {"mariadb": 8, "alisql": 16, "pgvector": 4}[engine]
        base_qps = shape[engine]["speed"] / 6.0
        for clients in (1, 2, 4, 8, 16, 32):
            scale = clients / (1 + (clients / knee) ** 1.6)
            qps = base_qps * scale
            records.append({
                **base, "phase": "concurrency", "engine": engine,
                "engine_version": f"{engine}-synthetic", "m": 16, "ef_search": 100,
                "storage_engine": "heap" if engine == "pgvector" else "InnoDB",
                "clients": clients, "qps": round(qps, 2),
                "latency_mean_ms": round(clients * 1000.0 / qps, 4),
                "latency_p50_ms": round(clients * 900.0 / qps, 4),
                "latency_p95_ms": round(clients * 1700.0 / qps, 4),
                "latency_p99_ms": round(clients * 2900.0 / qps, 4),
                "queries_executed": int(qps * 20),
                "extra": {"scaling_efficiency": round(scale / clients, 4),
                          "duration_s": 20},
                "timestamp": _timestamp(clients),
            })

    # --- filtered -------------------------------------------------------
    for engine in ENGINES:
        for selectivity in (0.01, 0.10, 0.50):
            degradation = {"mariadb": 0.55, "alisql": 0.40, "pgvector": 0.80}[engine]
            recall = min(0.99, 0.35 + degradation * selectivity + 0.45 * selectivity)
            short = int(180 * (1 - selectivity)) if engine == "alisql" else 0
            records.append({
                **base, "phase": "filtered", "engine": engine,
                "engine_version": f"{engine}-synthetic", "m": 16, "ef_search": 100,
                "storage_engine": "heap" if engine == "pgvector" else "InnoDB",
                "selectivity": selectivity,
                "recall_at_k": round(recall, 6),
                "qps": round(shape[engine]["speed"] / 8 * (0.4 + selectivity), 2),
                "clients": 1,
                "latency_p50_ms": 0.8, "latency_p95_ms": 1.6, "latency_p99_ms": 2.9,
                "vector_index_used": True,
                "extra": {"tag_threshold": int(selectivity * 100),
                          "short_result_queries": short,
                          "returned_fewer_than_k": short > 0,
                          "iterative_scan": "relaxed_order" if engine == "pgvector" else None},
                "timestamp": _timestamp(int(selectivity * 100)),
            })

    # --- churn -----------------------------------------------------------
    for engine in ENGINES:
        decay = {"mariadb": 0.030, "alisql": 0.018, "pgvector": 0.055}[engine]
        baseline = 0.972
        for fraction in (0.0, 0.10, 0.25):
            recall = baseline - decay * (fraction / 0.10)
            records.append({
                **base, "phase": "churn", "engine": engine,
                "engine_version": f"{engine}-synthetic", "m": 16, "ef_search": 100,
                "storage_engine": "heap" if engine == "pgvector" else "InnoDB",
                "churn_fraction": fraction, "clients": 1,
                "recall_at_k": round(recall, 6),
                "qps": round(shape[engine]["speed"] / 7 * (1 - fraction * 0.3), 2),
                "rows": 60000, "index_bytes": int(1.1e8 * (1 + fraction * 0.2)),
                "extra": {"recall_drop_vs_baseline": round(baseline - recall, 6)},
                "timestamp": _timestamp(int(fraction * 100)),
            })

    return records


def write_run(run_dir: str, run_id: str = "synthetic-run") -> str:
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "run-manifest.json"), "w") as fh:
        json.dump(make_manifest(run_id), fh, indent=2, sort_keys=True)
    path = os.path.join(run_dir, f"ops-synthetic-{DATASET}-normalized-m16-post.jsonl")
    with open(path, "w") as fh:
        for record in make_records(run_id):
            fh.write(json.dumps(record, sort_keys=True) + "\n")

    # A short memory timeseries so the timeline chart has input.
    mem_path = os.path.join(run_dir, "mem-mariadb-{}-normalized-m16-post.jsonl".format(DATASET))
    with open(mem_path, "w") as fh:
        for i in range(120):
            fh.write(json.dumps({
                "t": 1000.0 + i * 0.25,
                "container": "synthetic-srv",
                "rss_bytes": int(4e8 + 6e8 * (1 - math.exp(-i / 25.0))),
                "peak_bytes": int(1.1e9),
                "cpu_seconds": i * 0.2,
            }) + "\n")
    return run_dir
