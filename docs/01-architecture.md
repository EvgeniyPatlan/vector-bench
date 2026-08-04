# Architecture

## 1. System overview

`vector-bench` is a self-contained directory. It never writes to the three vendor
repositories — it reads them, clones from them into its own `sources/`, and does
all work under its own tree.

```mermaid
flowchart TB
    subgraph host["Host — VECTOR_RESEARCH/"]
        direction TB
        subgraph ro["Read-only inputs (never modified)"]
            MDBSRC["server/<br/><i>MariaDB git repo</i>"]
            ALISRC["AliSQL/<br/><i>AliSQL git repo</i>"]
            ANNSRC["ann-benchmarks/<br/><i>MariaDB fork</i>"]
        end

        subgraph vb["vector-bench/ — all our work"]
            direction TB
            ENTRY["run-benchmark.sh<br/><b>single entrypoint</b>"]
            CFG["config/<br/>profiles · engines · resources"]
            SRC["sources/<br/><i>pinned clones @ tag</i>"]
            WORK["work/ann-benchmarks/<br/><i>clone + overlay applied</i>"]
            OVL["overlay/<br/>alisql · mariadb · pgvector modules"]
            HARN["harness/<br/><i>ops workloads</i>"]
            REP["report/generate.py"]
            RES["results/&lt;run-id&gt;/"]
        end
    end

    subgraph dkr["Docker images"]
        direction TB
        RT1["mariadb-runtime"]
        RT2["alisql-runtime"]
        RT3["pgvector-runtime"]
        BN1["mariadb-bench"]
        BN2["alisql-bench"]
        BN3["pgvector-bench"]
        RT1 --> BN1
        RT2 --> BN2
        RT3 --> BN3
    end

    MDBSRC -.->|git clone --branch tag| SRC
    ALISRC -.->|git clone --branch tag| SRC
    ANNSRC -.->|git clone| WORK
    OVL -->|copied over| WORK
    CFG --> ENTRY
    SRC -->|git archive → build ctx| dkr
    ENTRY --> dkr
    WORK --> BN1 & BN2 & BN3
    HARN --> RT1 & RT2 & RT3
    BN1 & BN2 & BN3 --> RES
    RT1 & RT2 & RT3 --> RES
    RES --> REP
    REP --> OUT["report.md · report.html · charts/"]

    classDef readonly fill:#f5f5f5,stroke:#999,stroke-dasharray:4 3,color:#333
    class MDBSRC,ALISRC,ANNSRC readonly
```

The key structural point: **two independent measurement paths write into one
results directory.**

- The `*-bench` images run under ann-benchmarks and produce recall/QPS.
- The `*-runtime` images are driven directly by our ops harness and produce build
  cost, concurrency, filtered-search and churn numbers.

Both emit the same JSON-lines record schema, so the report generator does not
care which path produced a given metric.

## 2. Container build model

Each engine has one Dockerfile with two published targets.

```mermaid
flowchart LR
    subgraph build["Stage: builder"]
        direction TB
        B1["distro + toolchain<br/>+ build deps"]
        B2["COPY source.tar<br/><i>git archive of pinned tag</i>"]
        B3["cmake · make -j · make install<br/>→ /opt/&lt;engine&gt;"]
        B1 --> B2 --> B3
    end

    subgraph rt["Target: &lt;engine&gt;-runtime"]
        direction TB
        R1["distro + runtime libs only<br/><i>no compiler, no source</i>"]
        R2["COPY --from=builder /opt/&lt;engine&gt;"]
        R3["entrypoint: init datadir + start server"]
        R1 --> R2 --> R3
    end

    subgraph bn["Target: &lt;engine&gt;-bench"]
        direction TB
        N1["FROM &lt;engine&gt;-runtime"]
        N2["+ python3, numpy, h5py<br/>+ DB client driver"]
        N3["+ ann-benchmarks run_algorithm.py"]
        N1 --> N2 --> N3
    end

    B3 ==>|artifact| R2
    R3 ==> N1

    rt -.->|"docker run — manual testing,<br/>see docs/03-running-manually.md"| USE1["you, by hand"]
    bn -.->|"driven by ann-benchmarks"| USE2["recall / QPS sweep"]
```

`SOURCE_MODE` build-arg selects where the source tarball comes from:
`local` (default — `git archive` out of `sources/`) or `upstream` (clone from
GitHub inside the builder, for publishing a recipe others can rebuild).

## 3. Vector-search internals, side by side

The three engines solve the same problem with materially different plumbing.
This is what the benchmark is actually measuring.

```mermaid
flowchart TB
    subgraph mdb["MariaDB 11.8 — MHNSW"]
        direction TB
        M1["SQL layer<br/><code>ORDER BY vec_distance_cosine(v,q) LIMIT k</code>"]
        M2["Optimizer picks VECTOR INDEX"]
        M3["MHNSW graph cache<br/><i>per TABLE_SHARE</i><br/><code>mhnsw_max_cache_size</code>"]
        M4["Graph stored in a hidden<br/>companion table (#i# files)"]
        M5["Storage engine:<br/>InnoDB or MyISAM"]
        M1 --> M2 --> M3 --> M4 --> M5
    end

    subgraph ali["AliSQL 8.0.44-2 — VIDX"]
        direction TB
        A1["SQL layer<br/><code>ORDER BY VEC_DISTANCE(v,q) LIMIT k</code>"]
        A2["Optimizer by cost, or FORCE INDEX"]
        A3["Two caches:<br/><b>MHNSW Share</b> (TABLE_SHARE, read-only trx)<br/><b>MHNSW Trx</b> (per-session, RW trx)<br/><code>vidx_hnsw_cache_size</code>"]
        A4["Graph in an InnoDB<br/>auxiliary table, 1 row per node"]
        A5["Storage engine:<br/><b>InnoDB only</b><br/><i>requires READ-COMMITTED</i>"]
        A1 --> A2 --> A3 --> A4 --> A5
    end

    subgraph pg["PostgreSQL 17 — pgvector HNSW"]
        direction TB
        P1["SQL layer<br/><code>ORDER BY embedding &lt;=&gt; q LIMIT k</code>"]
        P2["Planner picks hnsw index scan"]
        P3["No dedicated graph cache —<br/>pages served from <code>shared_buffers</code>"]
        P4["Graph in a normal index relation<br/>(index AM pages)"]
        P5["Heap + index, standard buffer manager"]
        P1 --> P2 --> P3 --> P4 --> P5
    end
```

Consequences that show up in the results:

- **Cache design drives concurrency behaviour.** MariaDB's single per-share cache
  and AliSQL's share+transaction split respond differently as client count rises.
  pgvector inherits PostgreSQL's buffer manager, which is mature but not
  vector-aware.
- **Graph-in-a-table (MariaDB, AliSQL) vs graph-in-an-index-AM (pgvector)**
  changes what "index size on disk" means and how build cost scales.
- **`ef_construction` is only tunable in pgvector**, so build-quality tradeoffs
  are not symmetric.

## 4. Benchmark data flow

```mermaid
sequenceDiagram
    autonumber
    participant U as run-benchmark.sh
    participant D as datasets/
    participant I as Docker image
    participant E as Engine (in container)
    participant R as results/&lt;run-id&gt;/

    U->>U: resolve profile + engines + datasets<br/>collect sysinfo, write run-manifest.json
    U->>D: fetch HDF5 dataset (train, test, ground-truth)
    U->>I: build if image digest missing

    rect rgb(240, 246, 255)
    note over U,R: Path A — recall / QPS (ann-benchmarks)
    U->>I: run.py --algorithm &lt;engine&gt; --dataset … --runs N
    I->>E: start server, CREATE TABLE + VECTOR INDEX
    E->>E: ingest train vectors, build HNSW
    loop each ef_search in grid
        I->>E: query all test vectors, k=10
        E-->>I: neighbour ids + per-query latency
    end
    I->>R: results/&lt;dataset&gt;/&lt;k&gt;/&lt;engine&gt;/*.hdf5
    end

    rect rgb(245, 255, 245)
    note over U,R: Path B — ops harness
    U->>E: docker run &lt;engine&gt;-runtime (cpuset + mem limit)
    U->>E: ingest, timed; build index, timed
    E-->>U: build wall/CPU time, peak RSS (cgroup), index bytes
    U->>E: concurrency sweep 1→32 clients
    E-->>U: QPS + p50/p95/p99
    U->>E: filtered search @ 1% / 10% / 50% selectivity
    E-->>U: recall vs recomputed filtered ground truth
    U->>E: churn 10% / 25%, re-measure
    U->>R: results/ops/*.jsonl
    end

    U->>R: finalize manifest (durations, image digests)
    U->>U: report/generate.py
    U->>R: report.md · report.html · charts/*.svg
```

## 5. Results pipeline

```mermaid
flowchart LR
    A1["ann-benchmarks<br/>HDF5 results"] --> N["normalize.py<br/><i>one record schema</i>"]
    A2["ops harness<br/>JSONL"] --> N
    A3["run-manifest.json<br/><i>env + versions</i>"] --> N
    N --> DF["records.parquet /<br/>records.jsonl"]
    DF --> C1["Pareto: recall@10 vs QPS<br/><i>per dataset</i>"]
    DF --> C2["Build cost:<br/>time · RAM · index bytes vs M"]
    DF --> C3["Concurrency:<br/>QPS + p99 vs clients"]
    DF --> C4["Filtered: recall/QPS<br/>vs selectivity"]
    DF --> C5["Churn: recall drift"]
    C1 & C2 & C3 & C4 & C5 --> MD["report.md"]
    C1 & C2 & C3 & C4 & C5 --> HTML["report.html<br/><i>self-contained</i>"]
```

A record is one row of:

```
run_id, engine, engine_version, dataset, metric_space, pass (normalized|tuned),
storage_engine, M, ef_construction, ef_search, k, clients, phase,
recall_at_k, qps, latency_p50_ms, latency_p95_ms, latency_p99_ms,
build_wall_s, build_cpu_s, peak_rss_bytes, index_bytes, ingest_rows_per_s,
selectivity, churn_fraction, timestamp
```

Unused fields are null. One flat schema keeps the report generator simple and
makes the raw data trivially queryable with any tool.
