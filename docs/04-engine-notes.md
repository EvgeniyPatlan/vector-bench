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

## Valkey — valkey-search

### Filtered search gets slower as the predicate narrows

valkey-search describes two strategies for a hybrid query: **pre-filtering**
(resolve the filter from the secondary index, then brute-force the survivors by
distance) and **inline filtering** (walk the HNSW graph, testing the predicate
as you go). Its documentation says the planner picks between them, and Google's
Memorystore documentation — which runs this module — says pre-filtering is
chosen "when the filtered search space is much smaller than the original".

Measured, it does not. Latency rises as the predicate tightens, which is the
signature of the graph walk: with one node in a hundred qualifying, the walk
travels much further to find ten of them.

| Selectivity | p50 | QPS |
| --- | --- | --- |
| unfiltered | — | 479 |
| 10% | 18.16 ms | 53.5 |
| 1% | 120.56 ms | 8.2 |

9,900 vectors out of 990,000 is a search space where an exact scan takes a
couple of milliseconds. It took 120.

**There is no way to ask for the other strategy.** RediSearch exposes exactly
this choice as `HYBRID_POLICY ADHOC_BF | BATCHES`; valkey-search's parser
rejects the keyword outright:

```
Error parsing vector similarity parameters:
`[KNN 10 @embedding $vec EF_RUNTIME 100 HYBRID_POLICY ADHOC_BF]`.
Unexpected argument `HYBRID_POLICY`
```

Two things about our own query were ruled out before concluding that. Dropping
`EF_RUNTIME` does not reach the pre-filter — it lowers search effort, and recall
falls with it. Replacing the open `[-inf (threshold]` bound with a bounded
`[0 max]` range, in case an unbounded side defeated cardinality estimation,
changes nothing: 136.72 ms against 136.72 ms.

Reproduce all of it in about five minutes:

```bash
./run-benchmark.sh lab --engine valkey probe-valkey-filter.py --rows 200000
```

The probe generates random vectors, so its *recall* is not comparable to a run's
— random high-dimensional vectors are nearly equidistant and any HNSW graph over
them is poor. Only the comparison between shapes, and the direction across
selectivities, carries over. Both reproduce the run: 5.2x slower at 1% than at
10% in the probe, 6.6x in the run.

---

## MyVector — hnswlib as a MySQL plugin

Source: `askdba/myvector` release v1.26.9, commit `384d9df7`, built in-tree
into MySQL `mysql-9.7.2` (the server line Percona Server's `9.7` series is
based on). Symbols below are in that tree's `src/myvector.cc`,
`src/myvector_binlog.cc`, `src/myvector_plugin.cc`, `include/hnswdisk.h` and
`sql/myvectorplugin.sql`. Driver: `harness/drivers/myvector.py`.

The only engine here whose index is not part of the server's storage layer.
The graph is an hnswlib graph held in the plugin's memory and saved to files
in `myvector_index_dir`; the table is an ordinary InnoDB table with a native
`VECTOR(n)` column. How each piece works, and the traps in it:

| | MyVector v1.26.9 |
| --- | --- |
| Declared by | column `COMMENT 'MYVECTOR Column \|type=HNSW,dim=…,size=…,M=…,ef=…,dist=…,track=updatets'` |
| Built by | `CALL mysql.myvector_index_build('db.t.col','pk')`, after the load |
| Searched by | `WHERE MYVECTOR_IS_ANN('db.t.col','pk', vec, 'nn=k,ef_search=e')`, rewritten before parsing |
| **M** / **ef_construction** | `M=` / `ef=` column options (exact case) |
| Search width | `ef_search=` inside each call; hnswlib searches at `max(ef, k)` |
| Cosine | `dist=Cosine` → `AngularDistanceSpace`, a **scalar** loop (`computeCosineDistance`); L2 uses hnswlib's SIMD `L2Space` |
| Capacity | `size=` is hnswlib's fixed `max_elements`; `addPoint` throws past it, inside mysqld |
| Writes after the build | `myvector_index_refresh` over a `track=` timestamp — **adds only** (below) |
| Online (binlog) maintenance | documented; **exits after its first idle second** on this server (below) |
| Filtered search | **none** |

### Trap 1 — options are case-sensitive and unknown spellings are silent

`HNSWMemoryIndex::HNSWMemoryIndex` reads `getIntOption("M", 16)` and compares
`getOption("dist") == "Cosine"`. `m=32` builds M=16; `dist=cosine` builds L2.
Nothing warns. `MYVECTOR_INDEX_STATUS` prints `M` as the effective value but
`Distance` **as written**, and does not print `ef` at all. The driver writes
the exact spellings, then reads status back and refuses a build whose `Type`,
`M`, `Dimension`, `Distance`, capacity or row count is not what it declared.

### Trap 2 — a failed build is a result row, not an error

`myvector_index_build` opens its own connection back to the server using
`myvector.cnf` (`BuildMyVectorIndexSQL`); the file must be mode 0600/0400 or it
is refused. Every failure — no file, refused connection, failed save — comes
back as text in the procedure's result set and the `CALL` succeeds. The driver
requires the text to start `SUCCESS` and to report the loaded row count.

### Trap 3 — the binlog listener does not survive idleness, and never deletes

`docs/ONLINE_INDEX_UPDATES.md` describes `online=Y` as real-time maintenance
from the binary log, applying inserts, updates and deletes. Two things are
true instead, the first observed in this image and the second read from the
source:

- **It dies.** `myvector_binlog_loop` opens the stream with a 1-second
  `MYSQL_OPT_READ_TIMEOUT` and treats any fetch error whose text lacks
  `timed out` as fatal. Against MySQL 9.7.2 the idle-timeout error text is
  empty, so the server log shows `Binlog fetch failed: ` and `Exiting binlog
  func` about a second after startup, no `Binlog Dump` thread remains, and a
  row inserted after that never reaches the graph. Nothing restarts it.
- **It only inserts.** The loop dispatches `TABLE_MAP_EVENT` and
  `WRITE_ROWS_EVENT` and nothing else, and `myvector_table_op` only ever calls
  `insertVector`. (Its row parser also reads the key from a `MYSQL_TYPE_LONG`
  column only, so a BIGINT key would be indexed as 0.)

So the framework does not use `online=Y` and the engine gets no binlog
requirement: its server flags log exactly as Percona Server's do. Writes after
the build reach the index through the release's other documented incremental
path, a `track=` timestamp column and `myvector_index_refresh`, which the
churn workload's `wait_index_synced` hook runs after each re-insert and times
(`index_sync_s`). The refresh selects rows changed since the last build or
refresh at one-second resolution, so the driver waits out the watermark
second after each build and refresh — at most one second, inside the timed
maintenance — or a row written in that second would never be selected.

**`index_sync_s` is therefore not an incremental sync, and is not comparable
to another engine's.** A refresh is `myvector_open_index_impl` (action
`refresh`) → `BuildMyVectorIndexSQL`, which inserts the changed rows and then
calls `saveIndex(myvector_index_dir, "build")` — a full rewrite of the graph
file — after which `myvector_open_index_impl` calls `saveIndex(…, "refresh")`,
an incremental checkpoint, on top. So the figure is the re-inserted rows plus
one whole-graph write plus a checkpoint, and it grows with the corpus rather
than with the churn. It is one procedure call, so the parts cannot be timed
apart from SQL. The driver stamps this on every record
(`extra.index_sync_semantics`) and the report prints it under §9 Churn and in
Validity.

**Neither path removes a deleted row.** It stays in the graph; the search
still returns its id; the rewritten `id IN (…)` drops it, and the query
returns fewer than k rows. That is the degradation churn measures. Every
re-insert also consumes a graph slot for good, so the driver sizes the graph
at 1.5× the loaded rows and refuses a re-insert that would overflow it rather
than let hnswlib throw inside mysqld. The repair is a full
`myvector_index_build`, which takes `LOCK TABLES … READ` for its duration.

### Trap 4 — the rewrite is textual

The audit-plugin rewrite (`myvector_query_rewrite`) fires only on a statement
whose first character is `S`/`E`/`C`/`A` and that matches `^SELECT\s+` or
`^EXPLAIN\s+` — a leading comment or space disables it. It finds the end of
`MYVECTOR_IS_ANN(…)` by counting parentheses over the raw text, quotes
included, so an escaped binary vector can derail it with the bytes 0x28/0x29.
Worse, **a long statement kills the server**: with the query vector inline as a
hex literal, a 100-dim search (a 0.9 KB statement) works and a 784-dim one
(6.4 KB) makes mysqld exit with nothing in its error log — reproduced on this
image, and the cause of the first smoke run's failure. The same search with
the vector in a user variable works. So every query is two statements, `SET
@vb_q = X'…'` (no `MYVECTOR` in it, so the rewrite returns at its first
`strstr`) and the search over `@vb_q` — a second round trip per query that no
other engine here pays, charged by the plugin's design. `ef_search` mutates the shared index
(`setSearchEffort`), so it is sent on every query; omitting it searches at
whatever another session last asked for.

Both query-path asymmetries — the second round trip, and Cosine's scalar
distance loop (`AngularDistanceSpace` → `computeCosineDistance`; L2 runs
hnswlib's SIMD `L2Space`) — are recorded on every record
(`extra.round_trips_per_query`, `extra.distance_impl`, set by
`MyVectorDriver.query_path_facts` on the ops path and by the overlay module's
`get_additional` on the ann path) and footnoted by the report under §4, §7 and
in Validity.

### Trap 5 — dropping the table does not drop the index

`DROP DATABASE` leaves the plugin's in-memory index object alive under the
same `db.table.col` name, and the next build reuses that object — with the
dimension, M and ef it was constructed with. The driver calls
`myvector_index_drop` first, and the status check above would catch a stale
object anyway.

### No filtered search

`myvector_ann_set_init` accepts three or four arguments and the options string
carries `nn` and `ef_search` only. A predicate beside `MYVECTOR_IS_ANN`
filters the k results after the search (the release's own
`docs/LIMITATIONS.md` says so). The driver declares the `filtered` workload
declined (`EngineDriver.declined_workloads`); the harness skips it, prints the
reason and stamps `extra.declined_workloads` on every record of the run.

### Build mode

`post` only: load, then one `myvector_index_build`. There is no incremental
mode to compare it with, because the only write-path maintenance is the dead
binlog listener; the driver refuses `build_mode: incremental` with that
reason. `myvector_index_bg_threads` is the build's parallelism — 1 in the
normalized pass, `{build_threads}` in the tuned one; the plugin only builds in
parallel at 2 or more (`HNSWMemoryIndex::startParallelBuild`).

### Memory: the graph is outside the buffer pool, and the build holds the table twice

The graph is hnswlib's, in mysqld's own heap: it is not InnoDB pages, it is not
bounded by `innodb_buffer_pool_size` or by any MyVector variable, and it cannot
be paged out — the whole graph is resident or the search does not run. It IS
charged the same way as every other engine's memory: it is mysqld's address
space, inside the server container, so the container memory limit bounds it
and the run's `mem-*.jsonl` sampler and peak-RSS figures include it. What
differs is how the same budget is split. The normalized pass gives every
MySQL-family server a buffer pool of 30% of 16 GiB (4.8 GiB) and MariaDB/AliSQL
a graph cache of another 30%; MyVector's buffer pool serves only the table and
the k primary-key lookups per query, and its graph takes whatever is left.

Size, per element, is `MYVECTOR_INDEX_STATUS`'s `Element Data Size` — the
vector, the level-0 links (`2·M` 4-byte ids plus a count) and the 8-byte label;
3,276 B at 784 dims and M=16, measured — plus about 100 B of per-element arrays
and the label map. The build additionally keeps the WHOLE `(id, vector)` result
set client-side inside mysqld while it inserts (`BuildMyVectorIndexSQL`:
`mysql_store_result`), so its peak is roughly buffer pool + graph + one more
copy of the vectors. At M=16, 1M rows, estimated from that formula:

| Corpus | Graph | Build transient | Buffer pool (normalized) | Build peak | Normalized limit |
| --- | --- | --- | --- | --- | --- |
| gist-960 | ~3.8 GiB | ~3.6 GiB | 4.8 GiB | **~12.5 GiB** | 16 GiB — fits, ~3 GiB spare |
| dbpedia-openai-1000k (1536-d) | ~6.0 GiB | ~5.7 GiB | 4.8 GiB | **~17 GiB** | 16 GiB — does **not** fit |

No shipped profile runs dbpedia under the 16 GiB normalized limit:
`mariadb-blog` sets 64 GiB and `tuned-complete` uses the tuned pass's host
fraction. A profile that did would see MyVector killed by the container's OOM
killer during the build, while the cache-based engines merely slowed down — the
same cache-versus-wall distinction CLAUDE.md draws for Percona's
`innodb_hnsw_max_memory`. The build record carries
`extra.myvector_graph_bytes_floor` (level-0 bytes) and every record carries
`extra.memory_note`; the report states both under §5 and in Validity. Not done,
because it changes a budget and that is the user's call: (a) give MyVector's
normalized buffer pool the graph-cache share instead of its own 30%, so its
graph has the headroom the MHNSW/VIDX caches are given; or (b) refuse a
MyVector unit up front when the estimated build peak exceeds the server limit,
the way the free-disk check refuses a run, instead of letting it OOM.

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
