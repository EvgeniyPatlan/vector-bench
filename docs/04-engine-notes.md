# Engine notes

What each implementation actually does, and the traps in each. Everything here
was read out of the source trees or the engines' own test suites, not out of
marketing material.

## Quick reference

| | MariaDB 11.8 MHNSW | AliSQL 8.0.44-2 VIDX | pgvector 0.8 |
| --- | --- | --- | --- |
| Enable | on by default | `SET GLOBAL vidx_disabled=OFF` **and** session `READ-COMMITTED` | `CREATE EXTENSION vector` |
| Column type | `VECTOR(N)` | `VECTOR(N)`, N ≤ 16383 | `vector(N)` |
| Binary literal | raw LE float32 | raw LE float32 | `'[…]'::vector` |
| Text helper | `VEC_FromText` / `VEC_ToText` | `VEC_FROMTEXT` / `VEC_TOTEXT` | cast from text |
| Index DDL | `VECTOR INDEX vi (v) M=n DISTANCE=cosine` | `VECTOR INDEX vi (v) M=n DISTANCE=COSINE` | `USING hnsw (v vector_cosine_ops) WITH (m, ef_construction)` |
| **M** | index option (or `mhnsw_default_m`) | index option (or `vidx_hnsw_default_m`) | `WITH (m = …)` |
| **ef_construction** | **not exposed** | **not exposed** | `WITH (ef_construction = …)` |
| Search width | `mhnsw_ef_search` | `vidx_hnsw_ef_search` (default 20) | `hnsw.ef_search` |
| Graph cache | `mhnsw_max_cache_size` (**default 16 MiB**) | `vidx_hnsw_cache_size` (**default 16 MiB**) | none — `shared_buffers` |
| Distance fn | `VEC_DISTANCE_COSINE` / `_EUCLIDEAN` | same names | `<=>` / `<->` |
| Storage engine | InnoDB **or MyISAM** | **InnoDB only** | heap |
| Isolation | any | **READ COMMITTED required** | any |
| Graph location | companion table `<t>#i#<nn>` | InnoDB aux table `vidx_%016lx_%02x` | index relation |
| Build timing | incremental, on INSERT | incremental, on INSERT | **bulk, after load** |

---

## MariaDB — MHNSW

Source: `sql/vector_mhnsw.cc`, `sql/vector_mhnsw.h`. GA in the 11.7/11.8 series.

**Configuration surface.** Four plugin variables, declared at the bottom of
`vector_mhnsw.cc`:

```c
MYSQL_SYSVAR_ULONGLONG(max_cache_size, …)   // global
MYSQL_THDVAR_UINT(ef_search, …)             // session
MYSQL_THDVAR_UINT(default_m, …)             // session
MYSQL_THDVAR_ENUM(default_distance, …)      // session
```

plus two index options that default from the session variables:

```c
HA_IOPTION_SYSVAR("m",        M,      default_m),
HA_IOPTION_SYSVAR("distance", metric, default_distance),
```

So `M` can be given per index in DDL, which is what this framework does — it
matches how AliSQL takes `M` and keeps the two engines configured alike.

**Graph storage.** The HNSW graph lives in a hidden "high-level index"
companion table named `<table>#i#<nn>`, one row per node. Consequences: index
size is a filesystem measurement rather than a catalog lookup, and the graph
inherits the storage engine's page cache behaviour.

**Caching.** One cache per `TABLE_SHARE`, bounded by `mhnsw_max_cache_size`,
shared across sessions. Under concurrency all clients contend for the same
structure, which is what the concurrency workload is designed to expose.

**MyISAM.** MariaDB supports MyISAM for vector tables, and MariaDB's own
published benchmark uses it. AliSQL cannot, so this framework's headline
comparison is InnoDB for both, with MyISAM reported as a MariaDB-only extra
curve in the tuned pass.

**Not exposed:** `ef_construction`. Build quality is tunable only through `M`.
Verified on 11.8.8 — the only `mhnsw*` variables the server has are
`mhnsw_default_distance`, `mhnsw_default_m`, `mhnsw_ef_search` and
`mhnsw_max_cache_size`, and supplying it as an index option fails outright:

```
CREATE TABLE t (..., VECTOR INDEX vi (v) M=6 EF_CONSTRUCTION=200);
ERROR 1911 (HY000): Unknown option 'EF_CONSTRUCTION'
```

This is the most load-bearing claim in the methodology — it is why the
normalized pass pins pgvector's `ef_construction` — so it is stated with
evidence rather than asserted.

---

## AliSQL — VIDX

Source: `sql/vidx/vidx_hnsw.cc`, `sql/vidx/vidx_index.cc`, `sql/vidx/vidx_field.cc`,
headers under `include/vidx/`. Shipped in AliSQL-8.0.44-2, based on MySQL 8.0.44.

### Trap 1 — it is off by default

`vidx_disabled` defaults to `ON`. Any `VECTOR` column or `VECTOR INDEX` fails
with `ER_VECTOR_DISABLED` until you set it `OFF`. The runtime image passes
`--vidx-disabled=OFF` at startup.

### Trap 2 — READ COMMITTED is mandatory

Every vector operation raises `ER_NOT_SUPPORTED_YET` at any other isolation
level. This is a session setting and must be applied on every connection,
including every connection in a connection pool.

### Trap 3 — the optimizer will silently choose a full scan

AliSQL costs the vector index against a table scan and picks the scan when the
`LIMIT` is large relative to the row count. A full scan returns **exact**
results, so the symptom is recall ≈ 1.0 with collapsed throughput —
indistinguishable from "very accurate but slow" unless you check the plan.

**The threshold is a fraction of the table, not a fixed LIMIT**, and it does not
depend on `ef_search`. Measured on AliSQL-8.0.44-2 (`tests/verify-alisql-traps.sh`):

| Table size | index used | falls back to a scan |
| --- | --- | --- |
| 100 rows | `LIMIT` ≤ 25 | `LIMIT` ≥ 40 |
| 1000 rows | `LIMIT` ≤ 250 | `LIMIT` ≥ 290 |

Identical results at `vidx_hnsw_ef_search` = 20, 40 and 80 — the search width
plays no part in the decision. The crossover sits around **25–28% of rows**.

AliSQL's own `mysql-test/suite/rds/t/vidx_dml.test` appears to show a fixed
`LIMIT 16` / `LIMIT 17` boundary, and an earlier version of this document
repeated that reading. It is the same rule: that test deletes a third of its 100
rows first, so 17 of the ~67 remaining is ~25%.

**What this means in practice.** At benchmark scale the fallback does not
trigger — `k=10` against 60,000 rows is 0.017% of the table, three orders of
magnitude below the crossover. The plan guard is therefore cheap insurance
rather than a live hazard for these workloads. It stays on because the failure
is silent and would invalidate a whole run: every driver runs `EXPLAIN` per
configuration and records `vector_index_used`, and the report lists any
measurement where it was false in its Validity section, above the charts.

Mitigation when you need the index regardless of cost: `FORCE INDEX (vi)`.

### Trap 4 — the default graph cache is 16 MiB (and not only in AliSQL)

`vidx_hnsw_cache_size` defaults to 16777216 bytes — far below the graph size of
any realistic corpus, so an out-of-the-box AliSQL will thrash.

This is **not** an AliSQL-specific failing, and an earlier draft of this document
was wrong to imply it was. MariaDB ships the identical default: `SHOW VARIABLES
LIKE 'mhnsw%'` on 11.8.8 reports `mhnsw_max_cache_size = 16777216`. Both engines
degrade badly on their defaults. The framework sets both from the resource
profile so neither is judged on a value its vendor plainly intended you to
change.

### Values

`Field_vector::store(const char*, size_t, cs)` accepts a **binary string of
exactly 4 × dim bytes** — it validates the length and rejects NaN and Inf. So
binary binding works, and `VEC_FROMTEXT()` text parsing is avoidable. This
matters for fairness: charging AliSQL for text parsing that MariaDB does not
pay would be a client-side artefact, not an engine difference.

`VECTOR(N)` is stored as `varbinary(4N)` and presented that way in
`INFORMATION_SCHEMA.COLUMNS` — the data dictionary rewrites the type into a
versioned comment so tools that do not know `VECTOR` still see something valid.

### Caching

Two caches with different lifetimes (`vidx_hnsw.cc`):

- **MHNSW Share** — attached to the auxiliary table's `TABLE_SHARE`, shared by
  read-only transactions.
- **MHNSW Trx** — attached to the session via `thd_set_ha_data`; a read-write
  transaction keeps accessed and modified nodes in its own cache and updates
  the shared cache at commit.

This split is the most interesting structural difference from MariaDB's single
shared cache, and it should show up under mixed read/write concurrency.

### Other constraints

- InnoDB only.
- Vector index DDL cannot use `ALGORITHM=INPLACE`.
- Vector indexes cannot be `INVISIBLE`.
- Rows with `NULL` vectors are excluded from the index and sort last.
- HNSW layer assignment is randomised: replicas built from identical rows are
  not guaranteed to have identical graph topology.
- `M` range is 3–200; `ef_search` range is 1–10000.
- **Not exposed:** `ef_construction`.

---

## PostgreSQL + pgvector

Source: the pgvector extension, built against PostgreSQL 17.

**Bulk build.** The defining difference: pgvector builds its HNSW graph as one
operation after the data is loaded, parallelisable across
`max_parallel_maintenance_workers` and sized by `maintenance_work_mem`. MHNSW
and VIDX maintain their graphs on every INSERT and have no equivalent.

Comparing pgvector's bulk build against the others' incremental build compares
two different operations. This framework measures both, via `build_mode`:

- `post` — load, then `CREATE INDEX`. pgvector's idiomatic path.
- `incremental` — `CREATE INDEX` on the empty table, then load. Mirrors what
  MHNSW and VIDX are forced to do.

**No vector-specific cache.** Graph pages come through `shared_buffers` like
any other index pages. Mature and well-understood, but not vector-aware — there
is no equivalent of pinning the graph in memory. In the normalized pass
pgvector's buffer budget therefore absorbs the graph-cache fraction the other
two get separately; otherwise it would be handed strictly less resident memory
for the same container limit.

**`ef_construction`.** The one build-quality knob only pgvector has. Pinned to
its default of 200 in the normalized pass so pgvector is not given a tuning axis
the others lack; swept in the tuned pass and reported separately.

**Iterative index scan (0.8+).** `hnsw.iterative_scan` = `off` /
`relaxed_order` / `strict_order`. Without it, a selective `WHERE` can exhaust
the `ef_search` candidate list before k qualifying rows are found, and the query
returns fewer than k results. That is legitimate behaviour, but it makes
filtered recall ambiguous unless the mode is stated — so the framework sets it
explicitly and records it with every filtered measurement.

**Operator/opclass must match.** `<=>` requires `vector_cosine_ops`; `<->`
requires `vector_l2_ops`. A mismatch produces a sequential scan **with no error
and no warning** — pgvector's equivalent of AliSQL's silent fallback, and just
as capable of turning a benchmark into a brute-force measurement. Verified on
pgvector 0.8.6 against a cosine index over 5,000 rows:

```
-- matching operator: index used
EXPLAIN SELECT id FROM m ORDER BY embedding <=> '[…]' LIMIT 10;
  Limit
    ->  Index Scan using m_cos on m
          Order By: (embedding <=> '[…]'::vector)

-- mismatched operator against the same index: silently brute force
EXPLAIN SELECT id FROM m ORDER BY embedding <-> '[…]' LIMIT 10;
  Limit
    ->  Sort
          Sort Key: ((embedding <-> '[…]'::vector))
          ->  Seq Scan on m
```

The driver's plan check catches this: it requires the index name to appear in
the plan, and records `vector_index_used=false` otherwise.

**Storage.** `SET STORAGE PLAIN` on the vector column keeps vectors inline;
TOASTed vectors add a detoast to every distance comparison.

---

## Things that are true of all of them

- **The optimizer can always decline the index.** Every one of them will fall
  back to exact brute force under some conditions, and every one makes that
  look like high recall at low throughput. Check the plan, every time.
- **Metric is fixed at index creation.** You cannot query a cosine index with a
  Euclidean distance function and get index acceleration.
- **Recall is not deterministic across rebuilds.** HNSW construction involves
  randomised layer assignment; two indexes built from identical data will not
  have identical graphs, and recall will differ slightly.
- **Warm-up matters.** The first queries after startup pay for cold graph
  caches. Measuring them as steady state understates every engine, and
  understates the ones with larger caches most.
