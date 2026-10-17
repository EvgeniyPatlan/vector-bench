# Adding Percona Server as a seventh engine

Percona Server's `vector-mvp` branch carries a native InnoDB HNSW vector index.
This document is the investigation that has to happen before any of it is wired
in: what the dialect actually is, what changes in which file, and — kept
separate, in [§7](#7-what-only-a-live-server-can-settle) — what no amount of
reading can answer.

**Nothing in this repository drives Percona yet.** No driver is registered, no
`config/engines/percona.yml` exists, and no profile names it. The two long-lead
artefacts are written and unbuilt: `docker/percona/` and
`overlay/ann-benchmarks/ann_benchmarks/algorithms/percona/`. They exist now
because the image compiles a server from source, and that is the item with the
longest lead time in the whole addition.

## 0. Provenance of every claim here

| Kind of claim | How it is sourced | Example |
|---|---|---|
| Percona behaviour | `file:line` on the pinned SHA | `sql/field.cc:7097` |
| This repo's behaviour | `file:line` in this checkout, value quoted | `vb_mysql.py:96` |
| A number | its inputs listed, each marked measured or supplied | [§3](#3-innodb_hnsw_max_memory-is-not-a-cache) |

MTR `.result` files are preferred over `Docs/percona-hnsw.md` wherever both say
the same thing: a `.result` file is the server's own recorded output, a design
document is an intention. Where only the design document says something, it is
cited as such and marked.

### The pin

| | |
|---|---|
| Repository | `https://github.com/percona/percona-server` |
| Branch | `vector-mvp` |
| **Pinned SHA** | **`02a8b80c70b0ccf6c9035d8a2307e7fa988cc304`** |
| Tip **author** date | 2026-10-08T12:52:19Z |
| Tip **committer** date | 2026-10-13T13:05:40Z |
| Tip subject | `PS-11626: IGNORE INDEX ... FOR ORDER BY (vk)` is ignored for vector (ANN) queries |
| Server version | MySQL 9.7.1 LTS base (`MYSQL_VERSION`: `MAJOR=9 MINOR=7 PATCH=1 MATURITY="LTS"`) |
| MTR suite | `mysql-test/suite/percona_vector/` — **91** `.test` files and 91 `.result` files, plus two in `mysql-test/suite/sys_vars/` |

That SHA was the branch tip when this was written (`gh api
'repos/percona/percona-server/commits?sha=vector-mvp&per_page=1'`).

**Dates on this branch need their basis stated, and the table above gives both.**
Under **author** dates the tip is *not* the newest commit in its own history —
ancestors are dated 2026-10-12 and 2026-10-13, which is the fingerprint of a
rebase. Under **committer** dates the tip is the newest, as it must be. An
earlier revision of this table printed one date without saying which, and then
built the warning on it; the warning's conclusion survives either way, for a
simpler reason than the one it gave: **a date never tells you whether you have
the tip, so compare SHAs.** Every date in the table below is an author date,
because that is what `git log` shows by default.

### What the commit log says about stability

Of the twelve newest commits, **four** are crash or assertion fixes and two
more are the features that brought the vector work to its current state. All
six carry author dates between 2026-10-02 and 2026-10-13 — note that two are
*later* than the tip's own author date, which is the rebase above, not an
error:

| SHA | Date | What it fixed |
|---|---|---|
| `1d62ab22bc75` | 2026-10-13 | PS-11612: store resolved HNSW parameters in the DD |
| `5c30ac32877a` | 2026-10-12 | PS-11673: debug assertion in `Dictionary_client::acquire` on cross-schema `RENAME TABLE` after restart |
| `a86503214416` | 2026-10-12 | PS-11615: assertion `!other->is_vector()` in `vec_index_of()` when a vector search runs during an in-place index replace |
| `af72e2c0fa40` | 2026-10-12 | PS-11591: assertion `inserted.second` in `HNSW::insert()` — crash on INSERT after an **unclean restart** |
| `eebc427aab7b` | 2026-10-02 | PS-11623: assertion in `ddl0loader.cc:273` when an index build aborts on `innodb_hnsw_max_memory` exhaustion |
| `d436bb1c77d7` | 2026-10-09 | PS-11618: COSINE distance support for vector indexes |

Read as a group that is a branch under daily development whose *recent* failures
cluster in three places: restart recovery, concurrent DDL against a live search,
and the memory ceiling. Two of those three are things a four-day benchmark run
does — it restarts servers between phases and it loads until something stops it
— which is the argument for the smoke run in [§7](#7-what-only-a-live-server-can-settle)
rather than an argument against the engine.

Three qualifications on reading that table as a risk assessment, all of which
cut against over-reading it:

- Each row is a **fix**, so the pin contains it. The list says where the bugs
  were, not where they are.
- Two are explicitly debug-build assertions (`a86503214416`, `5c30ac32877a` say
  "debug"). `config/engines/percona.yml` will build `RelWithDebInfo` with
  `NDEBUG`, as every other engine here does, so those two cannot fire in a
  benchmark build. They still matter, because an assertion marks a state the
  author did not expect, and in a release build that state is reached silently.
- Nothing in a commit log shows what is still open. This is a survey of fixed
  things and must not be reported as a clean bill of health.

## 1. The question that had to be answered first: is there an `ef_search`?

**Yes.** `innodb_hnsw_ef_search` is a real, documented, tested system variable
with both SESSION and GLOBAL scope, and it is read per query.

The primary source is the declaration itself, `ha_innodb.cc:1205-1209`, which
carries the scope, the default and both bounds in five lines:

```c
static MYSQL_THDVAR_ULONG(
    hnsw_ef_search, PLUGIN_VAR_RQCMDARG,
    "Minimum candidate-list width for HNSW vector index searches (ANN"
    " recall/latency knob; the effective width is max(ef, LIMIT))",
    nullptr, nullptr, 40, 1, 100000, 0);
```

`MYSQL_THDVAR_*` is the session-variable macro, and the trailing arguments are
default `40`, minimum `1`, maximum `100000`. It is registered in
`innobase_system_variables[]` at `ha_innodb.cc:25112`. The engine reads it
through `THDVAR` on the query path — `ha_innodb.cc:12297-12300`:

```c
  const dberr_t err = vec_ann_open(
      vindex, reinterpret_cast<const float *>(m_vec_query.data()),
      std::max<size_t>(limit, 1), THDVAR(m_user_thd, hnsw_ef_search),
      m_user_thd, &m_vec_search);
```

`THDVAR` is a per-session read taken inside the scan's open, which is what
"per query" means mechanically rather than by assertion. Note also that the
help text states the `max(ef, LIMIT)` clamp as part of the variable's own
contract — see the riders below.

The behaviour at the edges is in the server's own recorded output,
`mysql-test/suite/sys_vars/r/innodb_hnsw_ef_search_basic.result`:

```
 1  SET @start_global_value = @@global.innodb_hnsw_ef_search;
 2  # Default: 40, in both scopes.
 3  SELECT @@global.innodb_hnsw_ef_search, @@session.innodb_hnsw_ef_search;
 4  @@global.innodb_hnsw_ef_search  @@session.innodb_hnsw_ef_search
 5  40  40
...
20  # A session value does not change the global one.
21  SET SESSION innodb_hnsw_ef_search = 200;
22  SELECT @@global.innodb_hnsw_ef_search, @@session.innodb_hnsw_ef_search;
23  @@global.innodb_hnsw_ef_search  @@session.innodb_hnsw_ef_search
24  40  200
...
30  # The bounds, and values outside them clamped with a warning.
31  SET SESSION innodb_hnsw_ef_search = 1;
...
35  SET SESSION innodb_hnsw_ef_search = 100000;
...
39  SET SESSION innodb_hnsw_ef_search = 0;
40  Warnings:
41  Warning 1292  Truncated incorrect innodb_hnsw_ef_search value: '0'
```

So: **default 40, range 1..100000, clamped with a warning outside it, integers
only** (`1.1`, `1e1` and `'AUTO'` are all `ER_WRONG_TYPE_FOR_VAR`, lines 52-57).

It is genuinely **per query and not per index**. Two independent sources:

- `mysql-test/suite/percona_vector/t/search.test:127-130`, inside a test of the
  search path itself:
  ```
  127  --echo # ef_search is a session knob.
  128  SET SESSION innodb_hnsw_ef_search = 200;
  129  SELECT id FROM t1 ORDER BY DISTANCE(v, STRING_TO_VECTOR('[0,0]'), 'EUCLIDEAN') LIMIT 3;
  130  SET SESSION innodb_hnsw_ef_search = default;
  ```
- `Docs/percona-hnsw.md:656-658`: "`ef_search` comes from the session variable
  `innodb_hnsw_ef_search` (default 40); the effective width is
  `max(ef_search, batch)`. It is per-query rather than per-index because the
  same index serves a dashboard wanting speed and a batch job wanting recall."

**Consequences for PR 2's calibration phase.** It is a normal engine: it has a
recall/QPS frontier, the calibration sweep has something to do, and
`set_ef_search` is one `SET SESSION`. Two details change what the sweep means:

- **The clamp is `max(ef, LIMIT)`**, stated in the variable's own help text
  (`ha_innodb.cc:1208`) and visible in the call at `:12299`, where `ef_search`
  is passed beside `std::max<size_t>(limit, 1)`; `Docs/percona-hnsw.md:656`
  calls the same thing `max(ef_search, batch)`. So the effective width is never
  below `LIMIT k`. At `k = 10` every grid
  value here is above 10, so the grid is taken verbatim and
  `effective_ef()`'s default pass-through is correct. **But** the harness
  already has a hook for exactly this shape — `EngineDriver.effective_ef(ef, k)`
  at `harness/drivers/base.py:374`, which exists because MongoDB clamps
  `numCandidates` up to `limit`. If a profile ever sweeps `ef_search` below `k`,
  Percona clamps the same way MongoDB does and the driver must say so. The
  follow-on task should implement `effective_ef` as `max(ef, k)` and not leave
  the inherited identity, because the inherited one would be wrong at the one
  grid point where it mattered and silently right everywhere else.
- `ef_construction` is *also* exposed, as an index build option — see the table
  in §2. So Percona has **both** a query-time and a build-time effort knob,
  which no other MySQL-family engine here has. That is a finding for the report:
  `CLAUDE.md`'s "Structural asymmetries no config removes" bullet currently
  reads "`ef_construction` is exposed only by pgvector and Valkey", and adding
  Percona makes that sentence false.

## 2. The dialect surface

Throughout: **PS** = Percona Server at the pinned SHA; **MDB** = MariaDB MHNSW
as this repo already drives it.

### 2.1 Column type and value binding

| | Percona Server | MariaDB MHNSW (for comparison) |
|---|---|---|
| Column | `v VECTOR(n) NOT NULL` | `v VECTOR(n) NOT NULL` |
| Max `n` | 16383 — `vector-common/vector_constants.h:38`, `constexpr unsigned int max_dimensions = 16383` | n/a here |
| Text constructor | `STRING_TO_VECTOR('[1,2]')`, alias `TO_VECTOR()` | `VEC_FromText()` |
| Read-back | `VECTOR_TO_STRING(v)`; a bare `SELECT v` returns raw bytes | `VEC_ToText()` |
| Width helper | `VECTOR_DIM(v)` | n/a |
| **Raw float32 bytes accepted?** | **Yes, if the parameter carries the binary charset** | Yes |

`STRING_TO_VECTOR` and `TO_VECTOR` are the *same function* —
`sql/item_create.cc:1656-1657` registers both names against
`Item_func_to_vector`. `VECTOR_TO_STRING` is `:1659`, `VECTOR_DIM` is `:1660`.

The binding question is the one that decides whether this repo's existing batched
loader works unchanged, and it is settled by
`Field_vector::store(const char *from, size_t length, const CHARSET_INFO *cs)`
at `sql/field.cc:7094`:

```c
7096  uint32 dimensions = get_dimensions(length, Field_vector::precision);
7097  if (dimensions == UINT32_MAX || cs != &my_charset_bin) {
7098    my_error(ER_DATA_INCOMPATIBLE_WITH_VECTOR, MYF(0), "string", length);
7099    return TYPE_ERR_BAD_VALUE;
7100  }
...
7107  /* Check for NAN or INF value in the vector. */
...
7130  return Field_blob::store(from, length, cs);
```

with `Field_vector::precision = sizeof(float)` (`sql/field.h:3819`) and
`get_dimensions` returning `UINT32_MAX` unless the length is a non-zero multiple
of 4 (`vector-common/vector_constants.h:96-102`). On a little-endian host the
bytes are then stored verbatim; the `WORDS_BIGENDIAN` branch at `:7117-7128`
byte-swaps, so the wire format is **little-endian float32** either way.

So **`harness/drivers/mysql_family.py:58`'s `to_binary_f32()` produces exactly
the right bytes**, and `overlay/.../vb_mysql.py:51`'s copy does too. Three
caveats, in descending order of how likely they are to bite:

1. **`cs != &my_charset_bin` is a hard refusal, not a conversion.** The
   parameter must reach the server as binary, not as utf8mb4. The repo passes
   Python `bytes`, which MariaDB Connector/C sends as `MYSQL_TYPE_BLOB`, and
   that is what makes this work today for AliSQL's `Field_vector`. It is
   *expected* to work identically here, and it is on the §7 list because
   "expected" is not "measured" and the failure mode is every INSERT erroring.
2. NaN and Inf are refused per element (`:7107-7115`). The datasets are real
   embeddings and contain neither, but a synthetic corpus could.
3. A length that is not `4 × n` for the declared `n` is refused. The repo always
   sends `4 × dim`.

**Percona does not need the text path, so it must not use it.** That is the same
argument `overlay/.../algorithms/alisql/module.py:22-24` makes for AliSQL:
routing one engine through `STRING_TO_VECTOR()` while the others take binary
would charge Percona for text parsing that belongs to the client, and would show
up as a Percona ingest penalty. Everything in this repo binds binary; Percona
keeps doing so.

### 2.2 Index DDL

| | Percona Server | MariaDB MHNSW |
|---|---|---|
| Inline in `CREATE TABLE` | **Yes** | Yes |
| By `ALTER TABLE` | **Yes** — `ALTER TABLE t ADD VECTOR KEY vk (v) TYPE hnsw (…)` | n/a (inline only in this repo's usage) |
| By `CREATE INDEX` | **Yes** — `CREATE VECTOR INDEX vk ON t (v) TYPE hnsw (…)` | no |
| Spelling | `VECTOR KEY` / `VECTOR INDEX`, then `TYPE hnsw` or `USING hnsw`, then `(opts)` | `VECTOR INDEX (v) M=n DISTANCE=m` |
| Canonical form | `SHOW CREATE TABLE` emits `USING \`hnsw\`` | — |
| Indexes per table | **exactly one** | more than one allowed |
| **Primary key** | **`BIGINT UNSIGNED`, single column** — explicit, or one MySQL promotes from a `UNIQUE NOT NULL` key | any |

All three creation paths are exercised in
`mysql-test/suite/percona_vector/t/cosine.test` — inline at `:9`, `CREATE VECTOR
INDEX` at `:19`, `ALTER TABLE … ADD VECTOR KEY` at `:33` and `ADD VECTOR INDEX`
at `:42`. `USING hnsw` is accepted as a synonym for `TYPE hnsw`
(`…/t/index_syntax.test:173`). A second vector index on one table is
`ER_NOT_SUPPORTED_YET` (`…/t/index_syntax.test:210-217`), and so is a second one
on the same column (`:219-226`).

**The primary-key requirement is the single biggest difference, because it
changes the schema rather than the index clause.** `index_syntax.result` records
four refusals with one message:

```
186  CREATE TABLE t1 ( id BIGINT PRIMARY KEY, …              -> refused (:190)
192  CREATE TABLE t1 ( id INT UNSIGNED PRIMARY KEY, …        -> refused (:196)
198  CREATE TABLE t1 ( id1 …, id2 …, PRIMARY KEY (id1,id2)…  -> refused (:204)
284  ALTER TABLE t1 CHANGE id id INT UNSIGNED;               -> refused (:285)

ERROR 42000: This version of MySQL doesn't yet support 'vector indexes on
tables lacking a BIGINT UNSIGNED single column primary key'
```

and `…/r/pk_combinations.result:16` records the same message for a table with no
primary key at all, with `:4-5` calling it a "current temporary restriction".

**Both of this repo's table definitions declared `id INT PRIMARY KEY`** —
`overlay/.../vb_mysql.py:96`, which still does, and the ops driver, which no
longer does. Percona refuses that column outright, so the Percona DDL must say
`id BIGINT UNSIGNED PRIMARY KEY`, and §4.1's answer was to make the type a
`Dialect` field: `harness/drivers/mysql_family.py:234` now reads
`id {self.dialect.pk_type} PRIMARY KEY` with `pk_type` defaulting to `"INT"`,
so MariaDB and AliSQL emit exactly what they emitted before. The ann overlay
needs no such field because `percona/module.py` overrides `create_table`
outright.

That is a real, if small, asymmetry the report should state rather than bury:
Percona's clustered index and its aux table's `base_pk` are both 4 bytes per row
wider than MariaDB's and AliSQL's, by the server's requirement and not by our
choice. At 1.18M rows that is about 4.7 MB in the base table — far below the
resolution of any figure here, but `index_bytes` and `table_bytes` are reported
and someone will notice.

### 2.3 Index options

| Option | Range | Default | Evidence |
|---|---|---|---|
| `M` | 2..200 | **25** | `storage/innobase/include/vec0vec.h:55`, `:64-65` |
| `ef_construction` | 1..4096 | 200 | `vec0vec.h:56`, `:71-72` |
| `metric` | `euclidean` \| `cosine` | `euclidean` | `vec0vec.h:57-58`; refusals at `index_syntax.result:789-808` |

"`M`, `metric` and `ef_construction` are settable, and they are all the
parameters there are" — `Docs/percona-hnsw.md:1293`.

Syntax rules, each with a recorded refusal:

- An option value is **a bare identifier or an integer, never a quoted string**.
  `metric = cosine` is right; `metric = 'cosine'` is a parse error
  (`Docs/percona-hnsw.md:1305-1307`), and so are `M = 6.0`
  (`…/t/index_syntax.test:356`) and `M = "6"` (`:363`).
- An unknown option name is `ER_ILLEGAL_INDEX_CONSTRUCTION_PARAMETER`
  (`…/t/index_syntax.test:377`); a bad value is
  `ER_ILLEGAL_INDEX_CONSTRUCTION_PARAMETER_VALUE` (`:384`, `:391`), which prints
  only the value and not which parameter it belonged to —
  `dd_params_resolved.result:77` is `ERROR HY000: Illegal index construction
  parameter value: 0.` That is why the overlay module range-checks `M` and
  `efConstruction` itself before the server sees them.
- **Defaults are resolved at create time and stored.** An index created with no
  option list comes back from `SHOW CREATE TABLE` with the full canonical set:
  `dd_params_resolved.result:8-19` creates `VECTOR KEY vk (v) TYPE hnsw` and
  gets back `` VECTOR KEY `vk` (`v`) USING `hnsw` (`M`=25,
  `ef_construction`=200, `metric`=`euclidean`) ``. So the manifest can read the
  parameters a graph was actually built with out of `SHOW CREATE TABLE`, which
  is better than trusting the DDL we sent.

### 2.4 The distance expression

| | Percona Server | MariaDB MHNSW |
|---|---|---|
| Form | `DISTANCE(v, q, 'METRIC')` — **arity exactly 3** | `vec_distance_<metric>(v, q)` — one function per metric |
| L2 | `DISTANCE(v, q, 'EUCLIDEAN')` | `vec_distance_euclidean(v, q)` |
| L2 squared | `DISTANCE(v, q, 'EUCLIDEAN_SQUARED')` — also served by an L2 index | — |
| Cosine | `DISTANCE(v, q, 'COSINE')` | `vec_distance_cosine(v, q)` |
| Inner product | `DISTANCE(v, q, 'DOT')` — **exact scan only, cannot be indexed** | — |
| Manhattan | `DISTANCE(v, q, 'MANHATTAN')` — exact scan only | — |

`sql/item_create.cc:1403` is `{"DISTANCE", SQL_FN(Item_func_vector_distance, 3)}`
— the metric argument is mandatory. The five names are the table at
`vector-common/vector_constants.h:63-69`, resolved case-insensitively through
`my_strnncoll` on `my_charset_latin1` (`:73-82`), which is why the suite uses
`'EUCLIDEAN'` (`…/t/search.test:24`) and `'euclidean'`
(`…/t/index_build.test:79`) interchangeably.

**A correction to the brief this task was given.** The brief said "L2 and
inner-product kernels with AVX-512 paths", which is true of the *kernels* and
not of the *index*. An HNSW index can only be built on `euclidean` or `cosine`:

```
index_syntax.result:797-802
  CREATE TABLE t ( … VECTOR KEY ( v1 ) TYPE hnsw ( metric = dot ) );
  ERROR 42000: This version of MySQL doesn't yet support 'HNSW indexes on
  anything but the EUCLIDEAN or COSINE metric'
```

with the same message for `euclidean_squared` (`:789-791`) and `manhattan`
(`:803-808`). `DISTANCE(…, 'DOT')` still works, as an exact full scan.

This costs this benchmark nothing — the harness needs `angular` → `cosine` and
`euclidean` → `euclidean`, both supported — but **the published report cannot
show a Percona inner-product frontier**, and if an inner-product dataset is ever
added, Percona has no index for it while pgvector and Valkey do.

Two more rules the index enforces, both relevant to `query_filtered`:

- **A cosine index serves only `'COSINE'`; an L2 index serves only
  `'EUCLIDEAN'` and `'EUCLIDEAN_SQUARED'` — never the cross product.**
  `…/t/cosine.test:273-296` asserts both directions after a restart, and
  `Docs/percona-hnsw.md:1312-1315` explains it: the optimizer reads
  `KEY::vector_index_metric` from the stored option. So a dialect that built the
  index with one metric and queried with another gets a silent full scan, not an
  error.
- `ORDER BY` must be **ascending** and there must be a `LIMIT`. Both missing
  cases, plus `DESC`, plus `DISTANCE()` only in `WHERE`, plus both arguments
  non-constant, are listed as "shapes that must NOT use the index" at
  `…/t/search.test:44-68`.

### 2.5 How the index is maintained: both ways, which is new here

| | Percona Server | MariaDB MHNSW | pgvector |
|---|---|---|---|
| On `INSERT` | **incremental** | incremental | no |
| Separable bulk build | **yes** | no | yes |

This is the second place Percona breaks a sentence in `CLAUDE.md`. It maintains
the graph on every `INSERT` — `…/t/insert.test` loads rows and
`…/t/insert_graph.test` checks the graph grew — *and* it has a real bulk path:
`ALTER TABLE … ADD VECTOR KEY` builds the graph from one clustered scan.
`…/t/index_build.test:20-23` names it:

```
20  --echo # First ADD, explicit INPLACE: rebuilds the table with the column and
21  --echo # builds a node per row. An in-place rebuild that wrote the column and
22  --echo # no graph is the case that used to give nodes=0.
23  ALTER TABLE t1 ADD VECTOR KEY k1 (v) TYPE hnsw (M = 4), ALGORITHM=INPLACE;
```

and `Docs/percona-hnsw.md` contrasts the two build algorithms: "`INPLACE` builds
the graph in memory and bulk-loads the aux (§15), so a COPY-built aux is correct
but slower to build and larger."

So Percona is the **second** engine in this benchmark that can run both
`build_mode`s, and the **first in the MySQL family**. An earlier revision of
this document called it the first outright and the only one where "incremental
versus bulk" is a within-engine comparison; both were wrong. **pgvector already
does it**: `pgvector_build_modes` appears in eight profiles including `smoke`,
`main`, `full` and `quick`, and `harness/workloads/build.py:12-16` was written
for exactly that comparison —

> …so the report can say "pgvector's bulk build is N× cheaper than its own
> incremental build" and "pgvector's incremental build costs X versus MHNSW's
> Y" — two honest claims instead of one misleading one.

What Percona adds is still worth having, but it is narrower than "a new
capability": it **decouples build mode from MySQL-versus-Postgres**. Today the
only within-engine build-mode curve belongs to the one engine that is also the
only Postgres, so "bulk is cheaper than incremental" and "Postgres differs from
MySQL here" cannot be separated. A second such curve, on a MySQL-family server,
separates them. That is a
genuinely valuable result for the report and it is worth a profile.

**There is a defect in `harness/workloads/build.py` directly in the way.**
Lines 93-95 are:

```python
    incremental = driver.capabilities().get("incremental_index", True)
    if index.build_mode == "incremental":
        incremental = True
```

`build_mode == "incremental"` forces the flag true; **`build_mode == "post"` has
no counterpart that forces it false.** The flag then decides everything
downstream: whether `driver.create_index(index)` is called at all (`:109`),
and what `build_wall_s` means (`:120`,
`effective_build = build_seconds if not incremental else load.wall_seconds`).

For every engine that exists today this is harmless, because `incremental_index`
is a fixed property of the engine and the two never disagree. For Percona they
do: a class-level `incremental_index = True` would make a `build_mode: post`
Percona run load an already-indexed table, never call `create_index`, and record
`build_wall_s` as the load time — a number labelled "bulk build after load" in
neither the notes nor the `extra` dict, since both branch off the same flag. The
record would read as incremental and nothing would say the profile asked for
something else.

The fix belongs to whoever owns the Percona driver, and the smaller of the two
options is the right one: **make the driver's `incremental_index` and
`capabilities()` depend on `IndexSpec.build_mode`**, so the existing
`build.py` logic reaches the right branch without `build.py` changing. The
driver already receives the spec in `create_schema`. The alternative — adding
`elif index.build_mode == "post": incremental = False` to `build.py` — is one
line and arguably more honest, but it changes a file that governs every engine's
build accounting, and it is not this task's to change.

### 2.6 Server variables

| Variable | Scope | Default | Harness must | Evidence |
|---|---|---|---|---|
| `innodb_hnsw_ef_search` | SESSION + GLOBAL | 40 | **set** per session (calibration) | `sys_vars/r/innodb_hnsw_ef_search_basic.result` |
| `innodb_hnsw_max_memory` | **GLOBAL only** | 1073741824 (1 GiB), `0` = unlimited | **set** from the resource pass — see §3 | `sys_vars/r/innodb_hnsw_max_memory_basic.result:3-5, :13-17, :23-26` |
| `Innodb_hnsw_memory_used` | GLOBAL status | — | **record** | `percona_vector/r/hnsw_memory_used.result:13-15` |
| `optimizer_switch` → `hypergraph_optimizer` | SESSION + GLOBAL | **unreachable in this build** | record only | `sys_vars.cc:232-234`, `:3435-3438`, `:3441-3452` |

**That table is complete, and it is now established from the registration array
rather than from filenames.** InnoDB exposes a variable by listing it in
`innobase_system_variables[]`, so that array is the authoritative enumeration.
At the pin it contains exactly two HNSW entries —

```
ha_innodb.cc:25112      MYSQL_SYSVAR(hnsw_ef_search),
ha_innodb.cc:25174      MYSQL_SYSVAR(hnsw_max_memory),
```

declared at `:1205-1209` and `:23843-23844` respectively — and the status-variable
array contains exactly one:

```
ha_innodb.cc:1335       {"hnsw_memory_used", (char *)&export_vars.innodb_hnsw_memory_used,
```

A grep of the whole file for `hnsw` returns no other variable of either kind.
**There is no third.**

An earlier revision of this section reached the same conclusion by a method that
could not support it: a recursive tree listing matching `innodb_hnsw`, which
returned the four `.test`/`.result` paths of the two `sys_vars` tests. That was
wrong twice over — it searched **path names, not file contents**, so a variable
with no `sys_vars` test would not have appeared; and the GitHub recursive tree
API returns `truncated: true` at this SHA, so the listing was not even complete
on its own terms. The same truncation is why that revision said the
`percona_vector` suite had 65 tests when it has 91. A truncated listing is not
an enumeration, and it does not announce itself in the part you read.

`Innodb_hnsw_memory_used` is a status variable worth recording for its own sake —
it is the only engine here that reports its graph's live memory, and
`hnsw_memory_used.result:13-15` shows it reading `0` after a restart (the graph
is loaded lazily) and rising on the first search.

The **hypergraph optimizer** row is the one trap in this table that is a
behaviour and not a knob. `Docs/percona-hnsw.md:1209-1210`: "The read path is the
old optimizer's, with no cost model. A session using the hypergraph optimizer
plans scan-plus-sort, with correct results and no index." A session with
`hypergraph_optimizer=ON` therefore gets **exact results at full-scan speed**,
which is this repo's canonical silent failure — "indistinguishable in output from
'very accurate but slow'". `explain_uses_vector_index()` catches it, which is
why that check must stay on for Percona even though, unlike AliSQL, Percona
never declines the index for *cost* reasons (vector index access is priced at
`prohibitive_cost = 1e100` specifically "so that it always loses against any
other access method, since a vector index is only ever activated for ORDER BY
... LIMIT" — `vector-common/vector_constants.h:40-47`).

**In this build the hypergraph risk cannot occur, and that is now established
rather than assumed.** Three gates in `sql/sys_vars.cc`, all keyed on CMake
symbols `docker/percona/Dockerfile` does not define:

- `:232-234` — the hypergraph bit is included in the default `optimizer_switch`
  only `#if defined(ENABLE_HYPERGRAPH_OPTIMIZER)`, so it is **off by default**.
- `:3435-3438` — `update_optimizer_switch()` masks the bit out of the global
  value `#ifndef WITH_HYPERGRAPH_OPTIMIZER`, so it cannot survive startup.
- `:3441-3452` — `check_optimizer_switch()` rejects an attempt to turn it on
  with `ER_HYPERGRAPH_NOT_SUPPORTED_YET`, naming "this build configuration; to
  enable it, build with CMake option `WITH_HYPERGRAPH_OPTIMIZER=ON`".

So a session cannot reach the hypergraph planner on this image even
deliberately. The table's "record, and verify OFF" is therefore cheap insurance
against a future build flag rather than a live hazard, and the
`explain_uses_vector_index()` check still earns its place on the *shape*
failures in §2.4 — no `LIMIT`, `DESC`, a mismatched metric — which are reachable
and silent.

### 2.7 Maintenance, and which churn hook applies

| Operation | Percona Server | Evidence |
|---|---|---|
| `OPTIMIZE TABLE` | accepted, and **does compact** — it rebuilds the table in place and recreates the vector index from the copied rows | `handler0alter.cc:1393-1404`; `…/t/insert.test:57-60` |
| Index replace in one `ALTER` | also compacts, **without rewriting the base table** | `…/t/index_build.test:52-59` |
| `ALGORITHM=INSTANT`, `LOCK=NONE` | refused while a vector index exists | `Docs/percona-hnsw.md:181`, `:183`, `:1232-1233` |
| Lock level of **any** ALTER leaving a vector index | **`LOCK=SHARED`** — never ONLINE, so **readers are not blocked, writers are** | `handler0alter.cc:1496-1500`, `:1509-1513`, `:1523-1524` |
| `DELETE` | drops the row from results immediately; leaves a graph tombstone | `…/t/search.test:119-121`; `…/t/index_build.test:52-54` |
| Vacuum-like background process | **none** | no such test or variable on the tree |

**`OPTIMIZE TABLE` does compact, and an earlier revision of this document said
it did not.** The source is explicit — `handler0alter.cc:1393-1404`:

```
1393  /* A table with a vector index can be rebuilt in place. The rebuild
1394  recreates every index on the new table, the vector index among them, so
1395  the graph and a new aux are built from the copied rows, with their labels
1396  intact and base_pk following the new primary key.
1397
1398  Such a rebuild is not ONLINE - the branch below clears `online` for it,
1399  because the row log cannot maintain a graph while DML runs against it.
1400  LOCK=SHARED it is.
1401
1402  percona_vector.alter_rebuild covers the shapes: FORCE, OPTIMIZE,
1403  ENGINE=InnoDB, ROW_FORMAT, a primary key swap, ADD and DROP COLUMN, and
1404  the AUTO_INCREMENT ones that need the label counter to survive. */
```

`OPTIMIZE` is named at `:1402`, the graph is rebuilt "from the copied rows"
(`:1395`), and a DELETEd row is not among the copied rows — so the tombstones
go.

The error is worth recording because of how it happened, not just that it did.
`…/t/index_build.test:52-54` writes `the first working "OPTIMIZE" for a vector
index` **with scare quotes**, hedging to mean "the first thing that does what
you would want of OPTIMIZE, at index level". The block quote below preserves
them faithfully; the prose then re-rendered it as `` `OPTIMIZE` `` and read a
hedge as a claim about the statement. A quoted hedge is not evidence for the
unhedged version of itself.

```
52  --echo # A DELETEd row is not collected by the build, so replacing the index
53  --echo # compacts tombstones — the first working "OPTIMIZE" for a vector
54  --echo # index. The labels are kept, so row 2's leaves a gap.
55  DELETE FROM t1 WHERE id = 2;
56  ALTER TABLE t1 DROP KEY k2, ADD VECTOR KEY k3 (v) TYPE hnsw (M = 4),
57    ALGORITHM=INPLACE;
```

**`maintain()` is still the index replace, for a different reason.** Both
statements compact; they differ in what else they do. `OPTIMIZE TABLE` rewrites
the whole base table — every row, every secondary index — to reclaim graph
tombstones, so its cost is dominated by work that has nothing to do with the
vector index, and the churn phase would be measuring a table rebuild. The
index replace touches only the aux and reuses each row's existing label, which
`…/t/index_build.test:37-44` proves by asserting `TABLE_ID` is unchanged across
it. That is the narrower repair and the one whose cost means something.

Anyone sizing the churn phase should know `OPTIMIZE TABLE` is available, which
the previous wording would have wrongly ruled out.

**The index replace takes a SHARED lock, so it blocks writers and not readers.**
This was an open question in a previous revision's §7, on the reasoning that
`handler0alter.cc:1398-1400`'s "LOCK=SHARED it is" was scoped to a table
*rebuild* and the index replace is not one (`TABLE_ID` is unchanged,
`…/t/index_build.test:37-44`). That scoping was right but the question was
answerable anyway, about a hundred lines further on, from a rule that is not
about rebuilds at all:

```
1496  /* No ALTER that leaves a vector index on the table runs with LOCK=NONE,
1497  whatever it changes. A rebuild applies concurrent DML through the row
1498  log, and row_log_apply maintains neither the HNSW graph nor the aux
1499  table, so a row written during the rebuild would reach the new table and
1500  not the index. Same reason ADD VECTOR INDEX is offline above.
...
1509  if (online && innobase_vector_exist(altered_table)) {
1510    ha_alter_info->unsupported_reason = innobase_get_err_msg(
1511        ER_ALTER_OPERATION_NOT_SUPPORTED_REASON_VECTOR_NOLOCK);
1512    online = false;
1513  }
...
1523  return online ? HA_ALTER_INPLACE_NO_LOCK_AFTER_PREPARE
1524                : HA_ALTER_INPLACE_SHARED_LOCK_AFTER_PREPARE;
```

`DROP KEY vi, ADD VECTOR KEY vi` leaves a vector index, so
`innobase_vector_exist(altered_table)` holds, `online` is forced false, and the
handler returns `HA_ALTER_INPLACE_SHARED_LOCK_AFTER_PREPARE`. A shared metadata
lock **permits concurrent reads and blocks writes** for the duration.

Two consequences, and they pull in opposite directions:

- **§5's search-during-DDL test stays meaningful, and is meaningful *because*
  of this.** `a86503214416` fixed an assertion when a vector search runs during
  an in-place index replace; if the replace excluded readers there would be
  nothing to test. It does not, so a reader can be mid-scan while the graph is
  replaced under it, which is exactly the state that assertion marked.
- **The churn phase's writers stop for the whole rebuild.** On a 1M-row corpus
  that is on the order of a bulk build (§2.5), and `churn.py:525-526` times the
  call, so the cost is recorded — but any concurrent write workload is paused,
  not slowed, and that belongs in `docs/07-planning-a-run.md` before anyone
  schedules it.

`:1502-1504` notes one exception that does not apply here: dropping the **last**
vector index may run online, because the new table has no graph to maintain.
The replace keeps one.

So, hook by hook (`harness/drivers/base.py:325-371`):

| Hook | Percona | Why |
|---|---|---|
| `prepare_churn()` | **no-op — inherit** | There is no autovacuum-equivalent to pin down. This hook exists for pgvector, whose autovacuum decides on its own schedule whether tombstones are still there when recall is re-measured; Percona has no background graph maintenance at all, so there is no knob whose state could differ between the baseline and the post-churn measurement. |
| `wait_index_synced()` | **no-op — inherit `0.0`** | The graph is maintained on the write path, inside the `INSERT`. There is nothing to wait for and the honest value is zero, which is what the inherited method returns. Returning a small positive number "to be safe" would be a fabricated measurement in a field documented as a measurement. |
| `maintain()` | **implement** — `ALTER TABLE t1 DROP KEY vi, ADD VECTOR KEY vi (v) TYPE hnsw (…), ALGORITHM=INPLACE`, returning `"ALTER TABLE (index replace)"` | This is the one hook Percona changes relative to MariaDB and AliSQL, both of which inherit `None`. Percona has a repair path, it is the only MySQL-family engine here that does, and `maintain()`'s contract is to run it and name it. Returning `None` would make the churn workload write one record per fraction instead of a before/after pair, and would lose the measurement this engine is uniquely able to supply. |
| `repair_path` | **`"index-replace"`** | `base.py:239-258` is explicit that this is declared and not inferred, because the churn workload stamps it onto the record *before* maintenance runs. Note that `base.py:255-256` currently says "MariaDB, AliSQL, Valkey and Percona Search inherit `"none"`" — "Percona Search" there is the MongoDB engine, not this one, and the sentence stays true. |

Two warnings about that `maintain()`, both of which make it a *measurement* and
not a tidy-up:

- The replace must re-state the full option list. The DDL that rebuilds the
  index decides the new graph's `M`, `ef_construction` and `metric`, and
  omitting them silently rebuilds at Percona's defaults — `M=25` instead of the
  profile's. The post-maintenance recall would then be a different index's
  recall. Read the options back out of `SHOW CREATE TABLE` (§2.3) and re-send
  them rather than re-deriving them from the profile.
- It rebuilds the whole graph, so on a 1M-row corpus it costs on the order of a
  whole bulk build. `churn.py:525-526` times the call, so the cost is recorded
  rather than hidden, but a churn phase that previously ran in minutes for the
  MySQL family will not for Percona. That belongs in `docs/07-planning-a-run.md`
  before anyone schedules it.

## 3. `innodb_hnsw_max_memory` is not a cache, and that is a fairness problem

This is the finding most likely to change what gets run, so it has a section.

`CLAUDE.md` already treats graph memory as a fairness parameter: "**Both
MySQL-family engines ship a 16 MiB graph cache by default**
(`mhnsw_max_cache_size`, `vidx_hnsw_cache_size`). The framework sets both from
the resource pass; leaving the default judges a vendor on a value it plainly
intended you to change." The obvious move is to add `innodb_hnsw_max_memory` as
the third member of that family, driven by `graph_cache_bytes`.

**That would be wrong, and wrong in a way that produces a plausible number.**
The two things are not the same kind of limit.

- MariaDB's and AliSQL's are **caches**. A graph larger than the cache still
  works; it is read from the companion table and performance degrades.
- Percona's is a **hard allocation ceiling**. `Docs/percona-hnsw.md:1537-1540`:
  "`innodb_hnsw_max_memory` bounds, in bytes, the memory held by HNSW graphs
  **across all tables and all indexes**." Exceeding it does not evict anything —
  it **fails the statement**. `:1555` and `hnsw_memory_used.result:69-75`:

```
69  # The limit is compared with this number: at it, an insert is refused,
70  # one byte above it, the insert goes through.
71  SET GLOBAL innodb_hnsw_max_memory = @t1_loaded;
72  INSERT INTO t1 VALUES (4, STRING_TO_VECTOR('[0,0,0,1]'));
73  ERROR HY000: Memory capacity of N bytes for 'innodb_hnsw_max_memory' exceeded.
    The vector index graph was left unchanged. Raise innodb_hnsw_max_memory and retry.
```

There is **no per-node eviction**. The only way the counter goes down is a whole
graph being freed: `:1562-1564` says "down by exactly a graph's bytes when that
graph is freed (DROP, a rebuild, eviction of its table)", and the only eviction
discussed anywhere on the tree is dictionary-cache eviction of a whole
`dict_table_t` (§22, `:1453-1461`). So **the entire graph must fit** in
`innodb_hnsw_max_memory`.

**And "the load stops partway through" is not the only failure — it is not even
the worst one.** `Docs/percona-hnsw.md:1555-1556`: "The limit is checked before
an insert or a load starts **and again as a cold search faults nodes in**." A
*query* can therefore fail with `ER_CAPACITY_EXCEEDED`, not only an INSERT.

Combine that with the lazy load §2.6 establishes — `Innodb_hnsw_memory_used`
reads `0` after a restart and the graph is faulted in node by node
(`hnsw_memory_used.result:11-15`, `Docs/percona-hnsw.md` §11) — and the
exposure is not a one-off at ingest. It **recurs at the start of every
post-restart query phase**, because each one re-faults the graph from nothing
and re-crosses the limit on the way up. A run that survived ingest can still
die in the first minute of querying, after the expensive part is already paid
for, and the orchestrator restarts servers between phases.

That shape also makes the failure harder to read than an ingest failure: an
ingest that stops has an obvious cause, whereas a query phase that errors
partway through a recall sweep looks like an engine defect.

Setting it equal to the other two engines' `graph_cache_bytes` would therefore
compare a cache budget against a wall. Under `normalized.yml`'s
`graph_cache_fraction: 0.30` the three engines would get the same number of
bytes and one of them would be unable to finish the load.

### What the default costs, with the inputs named

**These figures are projections, not measurements.** Nothing here was run
against a server. The chain is:

| Input | Status |
|---|---|
| per-row floor formula `dims × 4 + 2 × M × 8` | **measured on Percona**, in the sense that it is the floor an MTR test asserts the server stays above — `hnsw_memory_used.result:63`, `SET @floor = 1000 * (64 * 4 + 2 * 16 * 8);` with `:64-67` asserting `actual >= floor` and `actual <= 2 × floor + 2 × 65536` |
| row counts | **supplied** — from this repo (`harness/datasets.py:492` quotes "glove's 1,183,514 training rows"; `docs/07-planning-a-run.md:505` calls fashion-mnist "a 60,000-row smoke run"; `config/profiles/smoke.yml` sets `subset_rows: 20000`) |
| dimensionality | **supplied** — from the dataset names |
| `M` | **supplied** — from the profiles |
| 1 GiB default | **measured** — `innodb_hnsw_max_memory_basic.result:3-5` |

Because row counts, dims and `M` are all supplied, **none of the numbers below
is a measurement.** They are what the server's own asserted floor implies for
these configurations, and the regime is "graph fully resident, one vector index,
nothing else on the server" — which is the regime a benchmark run is in, and is
also the most favourable one, since the bound is server-wide.

| Dataset / phase | rows | dims | M | floor | floor ÷ 1 GiB | over the default? |
|---|---|---|---|---|---|---|
| fashion-mnist-784, ops smoke subset | 20,000 | 784 | 16 | 67,840,000 B | 0.063 | no |
| fashion-mnist-784, ann training set | 60,000 | 784 | 16 | 203,520,000 B | 0.190 | no |
| glove-100-angular (`main`) | 1,183,514 | 100 | 16 | 776,385,184 B | 0.723 | floor no, **ceiling yes (1.446)** |
| glove-100-angular (`main`) | 1,183,514 | 100 | 32 | 1,079,364,768 B | **1.005** | **yes** |
| gist-960-euclidean (`full`) | per 1M | 960 | 16 | 4,096,000,000 B | **3.815** | **yes, ~4×** |

Read three ways:

- **The smoke profile fits the default comfortably** (0.19 GiB at worst). So the
  smoke run will *not* surface this, and must not be taken as evidence that the
  default is fine.
- **The `main` profile does not.** At `M=32` on glove-100 the *floor* alone
  exceeds 1 GiB, so the load is guaranteed to stop. At `M=16` the floor fits but
  the MTR's own ceiling (2× floor) does not, so whether it completes is a
  coin-flip on the layer draws — the worst possible failure shape for a two-day
  run.
- Expressed as a row limit at the default, same basis: 316,551 rows at 784 dims
  and `M=16`; 1,177,348 rows at 100 dims and `M=32`; 262,144 rows at 960 dims
  and `M=16`.

### The policy: `innodb_hnsw_max_memory = 0`

**This is decided, not proposed.** The choice was the user's to make, because it
is a fairness judgement rather than a fact, and it has been made:
`config/engines/percona.yml` sets `--innodb-hnsw-max-memory=0` (unlimited),
with `Innodb_hnsw_memory_used` recorded per phase. The two policies below are
kept as the justification for it, not as an open question — a reader asking
"why is this engine's graph budget unlimited when the other two have a cache
size" needs the whole argument, and a decision whose reasoning has been deleted
gets relitigated.

What the policy does **not** change: `--innodb-hnsw-max-memory` must not be fed
from `graph_cache_bytes`, and the report must state the asymmetry rather than
paper over it.

1. **`0` (unlimited), with `Innodb_hnsw_memory_used` recorded per phase —
   CHOSEN.** The graph is then bounded only by the container's memory limit,
   which the orchestrator already sets and already records — so Percona is held
   to the same *total* memory budget as every other engine, measured where the
   other engines' budgets are measured. This is the honest like-for-like, and
   the recorded status variable is what makes it auditable — **with one caveat
   that has to travel with every figure taken from it**, below.
2. **`graph_cache_bytes`, accepting that some configurations will fail.** This
   is "same number of bytes for the same purpose", which is defensible on its
   face, but it produces an engine that cannot complete the load at `M=32` while
   the other two complete it slowly. A benchmark in which one engine errors and
   two degrade is reporting on a difference in limit semantics, not on index
   quality, and the chart would not say so. And under this policy the error is
   not confined to ingest: per the note above, a post-restart query phase can
   raise `ER_CAPACITY_EXCEEDED` as it faults the graph back in.

**`Innodb_hnsw_memory_used` understates the graph, and policy (1) leans on it.**
`Docs/percona-hnsw.md:1567-1568`: "It does not count the node map
(`HNSW::m_nodes`), which uses the normal heap, roughly 30 to 50 bytes per node,
or search scratch space."

So the gauge is a floor on true graph memory, not a measurement of it. At
glove-100's 1,183,514 rows that is **35.5 to 59.2 MB unreported** — inputs: the
row count is **supplied** (`harness/datasets.py:492`) and the 30-to-50-bytes
range is **Percona's own stated estimate**, so the figure is a projection from a
range, not a measurement, and it is one node map per vector index. Small against
a 16 GB container and not small against a 1 GiB ceiling: under policy (2) it is
about 3 to 6 per cent of the entire budget, invisible to the variable that is
supposed to be auditing it. A reader reconciling `Innodb_hnsw_memory_used`
against the container's RSS will find the difference and should not have to
rediscover why.

**So: any figure quoted from `Innodb_hnsw_memory_used` is a lower bound on the
graph's memory and must be labelled as one.** That applies to the chosen
policy's audit trail specifically — the whole point of recording the variable
under policy (1) is to show Percona stayed inside the container budget, and the
number that shows it is short by one node map per vector index.

The value is a declared knob either way, so it lands in `run-manifest.json` and
in `ann_fingerprint()`'s keying, which is what stops a 1 GiB curve being
silently reused under an unlimited manifest. Under the chosen policy that
keying is what keeps any curve measured during the smoke gate — where the
default was never reached, because the gate's worst case is 0.19 GiB and the
ceiling is 1 GiB, so **the gate cannot surface this at all** — from being
reused under a manifest that set the knob.

## 4. File-by-file change list for the follow-on task

Nothing in this list is done. Items marked **done** are the two long-lead
artefacts this task wrote.

### 4.1 `harness/drivers/mysql_family.py` — the `Dialect`

`MySQLDialect` (`:57-68`) is a flat dataclass of values, and three of Percona's
differences are not values today because `MySQLFamilyDriver` spells the
statements out. Add these fields, each with its default set to what MariaDB and
AliSQL already do so neither engine changes:

| New field | Default | Percona |
|---|---|---|
| `pk_type: str` | `"INT"` | `"BIGINT UNSIGNED"` |
| `index_clause: str` | `"VECTOR INDEX {name} (v) M={m} DISTANCE={metric}"` | `"VECTOR KEY {name} (v) TYPE hnsw (M = {m}, ef_construction = {ef_construction}, metric = {metric})"` |
| `distance_expr: str` | `"vec_distance_{metric}(v, %s)"` | `"DISTANCE(v, %s, '{metric_upper}')"` |
| `plan_marker: str` | `""` (meaning "use `index_name`") | `"vector_ann"` |
| `ef_construction: Optional[int]` | `None` | from `IndexSpec` |

Then:

- `create_schema` (`:133-156`) interpolates `pk_type` and `index_clause` instead
  of hard-coding `id INT PRIMARY KEY` and the MariaDB index clause.
- `_select` (`:287-323`) interpolates `distance_expr` instead of hard-coding
  `vec_distance_{...}(v, %s)`. Note the **parameter order changes**: MariaDB's
  filtered form binds `(threshold, vector)` because the predicate precedes the
  `ORDER BY`; Percona's is the same order, because the metric is a literal and
  not a parameter. Keep it that way — making the metric a bound parameter would
  add a third placeholder and reorder every call site.
- `explain_uses_vector_index` (`:337-373`) must test `plan_marker` when set.
  **This is not cosmetic, and it is not a hypothetical.** `:362` is
  `used = self.dialect.index_name in plan`, a substring test against the joined
  EXPLAIN text. Percona's own recorded output contains a plan that defeats it —
  `…/r/optimizer.result:491-492`:

  ```
  id select_type table partitions type possible_keys key  key_len ref  rows filtered Extra
  1  SIMPLE      t1    NULL       ALL  v1            NULL NULL    NULL X    100.00   Using filesort
  ```

  The only index on that table is a `VECTOR KEY`. The access type is `ALL`,
  `key` is `NULL`, `Extra` is `Using filesort` — **a full table scan** — and the
  index name `v1` is right there in `possible_keys`, so
  `index_name in plan` returns True. That is the brute-force scan this repo
  guards against, published as an ANN search, demonstrated rather than argued.
  `:610` is the second shape (`type=eq_ref, possible_keys=u,v, key=u` — the
  vector index offered, a different one taken), and the optimizer trace at
  `:471` gives the mechanism:
  `{"cause": "vector", "index": "v", "usable": false}` in
  `potential_range_indexes`. A filtered query — `WHERE tag < ? ORDER BY
  DISTANCE(…) LIMIT k`, which is what the ops harness runs — is exactly `:610`'s
  shape.

  `vector_ann` is the access type and appears only when the vector path was
  taken (`:95-96`). This is the same vacuity shape this repo
  has already been bitten by ten times — a substring satisfied by a longer
  string — and it is worth noticing that `CLAUDE.md`'s own history records
  `index_plan_marker: "v"` having been wrong in `mariadb.yml` for exactly this
  reason.
- **Issue `EXPLAIN FORMAT=TRADITIONAL`, not bare `EXPLAIN`** (`:356`). One free
  hardening while that method is open. `vector_ann` is a value in the `type`
  column, which only the tabular format has. `explain_format` defaults to
  TRADITIONAL so a bare `EXPLAIN` works today — but the variable is settable and
  reachable through `VB_SERVER_ARGS`, and under `TREE` or `JSON` the marker
  vanishes and the check records a **false negative about a server that did use
  the index**. That lands in the report's Validity section and sends someone
  hunting an engine bug that is not there. No MTR test could catch this, because
  MTR output is always TRADITIONAL; the suite spells the keyword out anyway at
  `…/r/optimizer.result:488`. `percona/module.py` already does this.

  **The risk this creates for the OTHER two engines is settled, and it was a
  real one rather than a theoretical one.** `explain_uses_vector_index` is
  shared: hardening it sends `EXPLAIN FORMAT=TRADITIONAL` to MariaDB and AliSQL
  as well, and a server that rejected the keyword would have turned the ops
  plan check into a **permanent false negative on three engines to protect
  one** — every configuration reported as not using its index, the whole
  Validity section full, and the one engine the change was for the only one
  working. Established by reading the three parsers rather than by assuming
  MySQL-family compatibility: MariaDB 11.8.8 `sql/sql_yacc.yy:14865-14874`,
  MariaDB 12.3.2 `:15297-15308` and AliSQL 8.0.44 `:14163-14171` all accept
  `FORMAT = TRADITIONAL` in the `describe` rule. Recorded here because "it is
  the default, so the keyword must be accepted" is the kind of inference this
  document is not allowed to make.

  The ann overlay's shared `_verify_plan` is deliberately NOT hardened, and
  that is a different judgement rather than an inconsistency: its check is
  name-only, and an index name appears in `TREE` and `JSON` output too, so the
  keyword would buy it nothing. Only a check that reads the `type` column
  needs the format pinned.
- `capabilities()` (`:438-439`) currently returns a constant
  `{"incremental_index": True, "ef_construction_tunable": False}`. For Percona
  both values change, and `incremental_index` must depend on
  `IndexSpec.build_mode` — see §2.5 for why, and for what `build.py` does if it
  does not.
- `effective_ef(ef, k)` should become `max(ef, k)` for Percona — see §1.
- `maintain()` and `repair_path` — see §2.7.
- A `PerconaDriver(MySQLFamilyDriver)` class and a `PERCONA_DIALECT` beside
  `MARIADB_DIALECT` and `ALISQL_DIALECT` (`:446-484`).

### 4.2 `harness/drivers/postgres.py` — `_driver_classes()`

`_driver_classes()` at **`postgres.py:689`** is the driver-class-name registry.
Add one entry, `"PerconaDriver": PerconaDriver`, to the dict at `:701-713` and
import it from `.mysql_family` at `:698`. Its docstring is explicit that this is
"one entry per architecture, not per engine" — Percona is a new architecture
only because of the dialect gaps in §4.1; once those are `Dialect` fields it is
arguably a variant, but it needs its own class for `capabilities()`,
`maintain()`, `repair_path` and `effective_ef`.

`_driver_table()` at `:710-730` maps engine name to class for "callers that
still think in engines". Add `"percona"` there too, for symmetry — the
orchestrator passes `--driver` by name so it is not load-bearing.

### 4.3 `config/engines/percona.yml`

**Do not write an `sql:` block.** At the time of writing, task C1 is in flight
(uncommitted in the main checkout, with `tests/test_c1_engine_sql.py` present
and `config/engines/mariadb.yml` already stripped) and removes the `sql:` block
from every engine config, because its contents had drifted — `index_plan_marker:
"v"` against a driver that names the index `vi`, a `create_table` with no `tag`
column beside its own `tag` filter, and no mention of `ctag`. A new config
carrying one would fail that test and reintroduce exactly what C1 removes. The
dialect lives in the driver; this file carries build, server flags and `runtime:`.

Required keys, modelled on `alisql.yml`:

```yaml
name: percona
display_name: Percona Server
family: mysql
source:
  # The key is `tag` even though the value is a SHA, and that is not a
  # cosmetic choice. `orchestrator/engines.py:84-86` is
  # `return str(source.get("tag") or source.get("version") or "")` — there is
  # no `sha` key, so a `source: sha:` block would make `_source_tag()` return
  # the empty string and `scripts/prepare-sources.sh` fall back to its
  # hard-coded default for that engine. `ops_pass.py:254` reads the same key
  # into `VB_ENGINE_TAG`, which is what reaches the manifest. `git archive`
  # takes a SHA wherever it takes a tag, so the mechanism is unchanged.
  tag: 02a8b80c70b0ccf6c9035d8a2307e7fa988cc304   # branch vector-mvp
  upstream: https://github.com/percona/percona-server.git
  # No `submodules:` key, deliberately: all four of the pin's submodules are
  # switched off in the build instead, which takes four cmake flags, one of
  # them non-obvious. See docker/percona/Dockerfile note 2.
image:
  runtime: vector-bench/percona-runtime
  bench: vector-bench/percona-bench
build:
  march: x86-64-v3        # MUST equal every other engine's — see CLAUDE.md
  cmake_flags: [...]      # as in docker/percona/Dockerfile
server:
  common: [--skip-name-resolve]
  normalized:
    - --innodb-buffer-pool-size={buffer_bytes}
    - --innodb-hnsw-max-memory=0          # §3 POLICY, decided; NOT {graph_cache_bytes}
    - --innodb-flush-log-at-trx-commit=2
    - --innodb-doublewrite=0
capabilities:
  ef_construction_tunable: true           # unlike mariadb and alisql
  m_per_index: true
  requires_read_committed: false          # unlike alisql
  filtered_search: true
runtime:
  order: 70
  driver: PerconaDriver
  ann_constructor: PerconaServer          # must be unique — see below
  port: 3306
  data_mount: /var/lib/vbench
  server_data_mount: /server-data/data
  credentials: {user: bench, password: bench}
  group: extra
  probe: ["sh", "-c", "/opt/percona/bin/mysql -ubench -pbench --socket=/var/run/vbench/percona.sock -e 'SELECT 1' >/dev/null 2>&1"]
  chart: {color: "#ff7f0e", marker: "P", linestyle: "-", label: "Percona Server (InnoDB HNSW)"}
  label: "Percona Server (InnoDB HNSW)"
```

Two collisions the validator refuses, both named in `CLAUDE.md`: a duplicate
`ann_constructor` and a duplicate image tag. `PerconaServer` collides with
nothing — the six existing constructors are `MariaDB`, `AliSQL`, `PGVector`,
`MariaDB123`, `PerconaSearch`, `ValkeySearch`. Note `PerconaSearch` is the
MongoDB engine; the two names are close enough to swap by accident, and swapping
them means one engine reports the other's recall, which is the defect
`mariadb123` exists to prevent.

The `#ff7f0e` colour is unused by the six in `GOLDEN` (`#1f77b4`, `#d62728`,
`#2ca02c`, `#5fa8d3`, `#9467bd`, `#e377c2`) and the `P` marker is unused
(`o`, `s`, `^`, `s`, `v`, `D`).

**The guard that currently blocks this file.** A hard-coded engine-family list
asserts the known set, so adding a seventh config fails it until that assertion
is relaxed. That is another task's; this one does not write the file.

### 4.4 `tests/test_engine_registry.py` — `GOLDEN` and `GOLDEN_ORDER`

`GOLDEN` (`:24-46`) is a name → 12-tuple of `(driver, ann_constructor, port,
data_mount, server_data_mount, (user, password), group, colour, marker,
linestyle, chart label, report label)`, read by four parametrized tests
(`:63`, `:75`, `:85`) plus `test_every_engine_is_found` (`:58`). Add a
`"percona"` row matching the `runtime:` block above exactly.

`GOLDEN_ORDER` (`:48`) is `("mariadb", "alisql", "pgvector", "mariadb123",
"mongodb", "valkey")` — "the original three, then the engines added later".
Percona is appended: `(..., "valkey", "percona")`, with `order: 70` in the
config to put it there.

`test_groups` (`:96-100`) asserts the group membership tuples literally:
`engines_in_group("extra") == ("mariadb123", "mongodb", "valkey")`. With
`group: extra` that assertion must gain `"percona"`. It is a value assertion and
not a membership one, so it will fail rather than silently pass — which is the
right behaviour and worth not "fixing" by loosening it.

### 4.5 `webui/static/fields.js` — colour and label

Two literal maps at the top of the file, each listing every engine:

- `ENGINE_COLOR` gained `percona: "#ff7f0e",`.
- `ENGINE_LABEL` gained `percona: "Percona Server (InnoDB HNSW)",`.

(Written without line numbers on purpose. These two maps grow by a line per
engine, so a citation here rots on every addition while pointing at nothing a
reader could not find by name in a twelve-line file.)

Both must match `config/engines/percona.yml`'s `chart.color` and `label`
character for character. They are a separate copy of the same facts — the
front end is static and cannot read the engine registry — so this is the one
place in the addition where two files have to be kept in step by hand.

The index-build chart's hint read "What the index costs to make. pgvector
bulk-builds; MHNSW and VIDX build on every INSERT." That sentence is
incomplete with Percona in the chart (§2.5) and now names the third
behaviour. Scoped to the ops path, which is the only path where Percona's
build mode is a real axis — see §4.10.

### 4.6 Profiles

- `config/profiles/smoke.yml` — the gate before every long run. It names no
  engines, so Percona joins automatically once the registry knows it; the
  `description` said "all six engines" and now says seven. Its
  20,000-row ops subset and 60,000-row ann set both fit the default
  `innodb_hnsw_max_memory` (§3), so **the smoke run will not exercise the memory
  ceiling** — that has to be checked deliberately.
- `config/profiles/main.yml` and `full.yml` — no engine list to edit, but §3
  says the memory setting must be right before either is started, and §2.7 says
  the churn phase now costs a full graph rebuild per fraction for this engine.
- `config/profiles/percona.yml` is the new Percona-only profile and is the home
  for the two things only this engine can show: an `ef_construction` sweep, and
  `build_mode: [post, incremental]` as a within-engine comparison. Neither
  belongs in `main`, because neither has a counterpart on MariaDB or AliSQL.
  **The build-mode half runs on the ops path only**, and the ann half could not
  be delivered — see §4.10.
  **The build-mode comparison is unmeasurable at the pinned version** and the
  profile asks for `[incremental]` alone: `ALTER TABLE … ADD VECTOR KEY …
  ALGORITHM=INPLACE` kills the server, so `post` cannot complete a unit. The
  reproduction is recorded at `capabilities.bulk_vector_index_build` in
  `config/engines/percona.yml`, which is also what the ops loop reads to keep
  the default off that mode for every profile that names none. Restore the
  pair when the server is fixed.
- `docs/07-planning-a-run.md` carries the cost model (`_INGEST_ROWS_PER_S` in
  `orchestrator/estimate.py`). Percona's ingest rate is unmeasured, so it has no
  entry, and an estimate for it would be a guess presented as a plan. It stays
  out until the smoke run supplies one.

  **Leaving it out was not enough on its own, and that was a defect rather than
  a gap.** `_effective_rate` ended in `_INGEST_ROWS_PER_S.get(engine, 300)`, so
  an engine with no entry was silently modelled at 300 rows/s — 2x the measured
  MariaDB figure and 5.5x the measured AliSQL one, for an engine that builds its
  graph inside every INSERT. On `main` that put 9.10 h of invented ingest into a
  published total (102.16 h against the honest 93.06 h) with nothing printed to
  say the number was a guess, and it contradicted the comment four lines above
  the table it read from: "an estimate that is optimistic by 3x is worse than
  none". The fallback now returns `None`, and `estimate_load_hours` reports
  `engines_without_ingest_rate` on the same terms as the query half's
  `engines_without_query_rate` — named, excluded from the total, and said so in
  the printed block.

### 4.7 Already written by this task

| Path | State |
|---|---|
| `docker/percona/Dockerfile` | **done**, never built |
| `docker/percona/entrypoint-percona.sh` | **done**, never run |
| `docker/percona/init.sql` | **done**, never run |
| `overlay/.../algorithms/percona/module.py` | **done**, never run |
| `overlay/.../algorithms/percona/config.yml` | **done**, never rendered |
| `scripts/build-images.sh`, `scripts/prepare-sources.sh`, `scripts/lib.sh` | **not done** — see below |

`scripts/lib.sh` needs one line too, and it is a safety line rather than a
functional one. `:24-26` export `VB_REPO_MARIADB`, `VB_REPO_ALISQL` and
`VB_REPO_ANNB`, and `assert_not_vendor_repo()` at `:65-76` loops over exactly
those three to refuse writing inside a read-only vendor checkout (`:73`,
`die "refusing to write inside read-only vendor repo: $canon"`). A Percona
vendor checkout gets **no protection at all** until `VB_REPO_PERCONA` is
exported beside them and added to the loop. `CLAUDE.md` names the vendor
repositories as read-only inputs that must never be checked out, fetched into or
modified; right now that rule is enforced for three of them and would not be for
the fourth.

Both build scripts need a `percona` case, and `build-images.sh` will say so loudly:
its `case` ends at `:141` with `*) die "engine '$engine' has no known
build-arg (alias_of=$base)"`, so a missing entry is an immediate, clear failure
rather than a silent one. The entry follows the convention at `:129-131` —
`percona) bargs+=(--build-arg "PERCONA_TAG=${tag}") ;;` — and
`docker/percona/Dockerfile:725` declares `ARG PERCONA_TAG` to receive it.
`prepare-sources.sh` must ship a `git archive` of the pinned SHA (`:98` is the
pattern) and must *not* try to carry submodules.

### 4.8 The `march` field goes blank in **two** places, not one

There are two independent `-march` readers, in two different containers, and
only one of them is reachable from the overlay module.

**The ann path.** `VBMySQLBase._image_march()` (`overlay/.../vb_mysql.py:538-564`)
read a fixed pair of paths, `/opt/mariadb/.march` and `/opt/alisql/.march`, and
returned `"unknown"` otherwise. `percona/module.py` overrode it to try
`/opt/percona/.march` first and then fall back, so **the ann path was already
correct** in the file the design task wrote.

**The ops path, which the override cannot reach.** `harness/main.py:209-227` has
its own:

```python
227      for path in ("/opt/mariadb/.march", "/opt/alisql/.march",
228                   "/opt/pgvector-artifacts/.march",
229                   "/opt/percona/.march"):
```

consumed at `:283` as `march=args.march or _detect_march()`. That runs in the
**harness container**, which mounts `harness/` and not `overlay/`, so no
override in an ann-benchmarks module can affect it. `orchestrator/ops_pass.py`
contains the string `march` zero times, so nothing passes `--march` in and the
`or` falls through to the detector. **Percona's ops records would therefore
have carried `march: "unknown"` too**, with its ann records looking fine —
half a run measured and nothing saying which half.

**Both tuples now carry the path and the override is gone**, which is what the
last paragraph of this section asks for. `percona/module.py` keeps a comment
where the override was, naming the second reader, so that the absence reads as
a decision rather than as an omission someone should repair.

Either blank field is the failure shape this project has already paid for once:
a probe that hard-codes a fixed set and whose figure reaches a published
artefact looking measured. It matters more than it looks, because "never mix
`-march` between engines" is a stated invariant — all seven engines compile SIMD
distance kernels — and `march` is the field that would catch a mixed build. Two
readers means two chances to miss it, and the one the module can fix is not the
one a long run depends on most.

The follow-on task should add `/opt/percona/.march` to **both** tuples and then
drop the override, since the next engine added will hit the same thing twice.

### 4.9 A family-wide latent defect, recorded and deliberately not fixed

This is not Percona's and not this task's, but it was found while matching the
family's entrypoint and it should not be lost.

`docker/percona/entrypoint-percona.sh` copies a two-line pattern from its
siblings. The **call site** is byte-identical in all three —
`docker/alisql/entrypoint-alisql.sh:94`,
`docker/mariadb/entrypoint-mariadb.sh:95`, `entrypoint-percona.sh` — and
`user_args()` behaves identically in all three though it is not byte-identical:
alisql `:40` and percona `:99` are one-liners, mariadb `:40-44` spreads the same
test over four lines with a comment. The behaviour that matters is the last
command in the body, which is the same everywhere:

```bash
user_args() { [[ "$(id -u)" -eq 0 ]] && printf '%s' "--user=root"; }
...
  local u; u="$(user_args)"; [[ -n "$u" ]] && args+=( "$u" )
```

Under `set -euo pipefail` (percona `:77`, alisql `:18`, mariadb `:16`), when
the container does **not** run as root, `user_args` returns 1 — the status of
its failed `[[ ]]` — and because the assignment is written as the separate
command `u="$(user_args)"` rather than `local u="$(user_args)"`, that non-zero
status is the command's status and `set -e` aborts the script. The server is
never exec'd and nothing is logged.

Reproduced with the pattern extracted verbatim and `id -u` stubbed:

```
FAKE_UID=0     -> REACHED_EXEC args=--datadir=/x --user=root   rc=0
FAKE_UID=1000  -> (no output)                                  rc=1
```

It is **latent**: no image in this repo sets `USER`, so every container runs as
root and takes the `rc=0` path. It would fire the moment one does, or under a
runtime that drops to a non-root user — and it would present as a server that
never starts and never says why, which is the expensive kind.

Matching the family was the right call for a new engine; diverging would have
left one of three entrypoints behaving differently for a reason unrelated to
Percona. The fix, when someone takes it, is one character per file (`local
u="$(user_args)"`, which returns `local`'s status) applied to all three
together.

### 4.10 `orchestrator/ann_pass.py` — Percona hits its `raise`

This file is the one that *renders* the overlay `config.yml` this task wrote,
and the previous revision of this document never mentioned it.

`render_config()` (from `:125`) dispatches on engine name and ends:

```python
301        raise ValueError(f"unknown engine: {engine}")
```

The branches are `:150` (`if engine in MARIADB_ENGINES + ("alisql",)`), `:167`
(pgvector), `:186` (valkey) and `:278` (mongodb). **Percona matched none of
them and landed on the `raise`.** Either add it to the MySQL-family condition at
`:150` or give it a branch of its own, and a branch of its own is what the two
things §4.6 proposes actually need:

The branch this section asked for is now `:206` (percona), and it was inserted
at the position mongodb's branch used to occupy — which is why this list said
`(mongodb)` against that line for two revisions, pointing at Percona's own
branch while the sentence above it said Percona had none. Every `` `:N` ``
below is a line of `orchestrator/ann_pass.py`; the labelled ones are checked
against the branch they claim by
`tests/test_e10_failures_explain_themselves.py`, which is what caught that.

- **`efConstruction`.** No existing MySQL-family branch emits it. The two that
  do are `:180` (pgvector) and `:199` (valkey), and both sit inside a
  build-mode loop. The MySQL-family branch at `:150` has no equivalent, so a
  Percona `arg_groups` carrying `efConstruction` has nowhere to come from.
- **`build_modes`.** `:176` reads `ann.get("pgvector_build_modes", ["post"])`
  and `:195` reads `ann.get("valkey_build_modes", ["post"])`. A
  `percona_build_modes` key needs the same treatment, and it is the key that
  makes §2.5's within-engine build comparison runnable.

So §4.6's Percona-only profile is not a profile edit alone; it is a profile
edit plus a `render_config` branch. The failure without it is loud — a
`ValueError` at render time, before anything is built — which is the good kind.

#### The `build_modes` half of that could not be delivered, and this is why

**`ann.percona_build_modes` is REFUSED by the new branch, not honoured.** This
is the one item in §4 that the follow-on task stopped on rather than
implemented, and the reason is a file §4.7 marks **done**.

`pgvector/module.py` and `valkey/module.py` each read `build_mode` out of
`method_param`, validate it against `("post", "incremental")`, branch on it in
`fit`, and report it back in `get_additional`. **`algorithms/percona/module.py`
contains the string zero times, and so does the `VBMySQLBase` it inherits
from.** Emitting the key anyway would produce two run groups, `post_build` and
`incremental_build`, differing only in a `method_param` nothing reads — two
*identical* recall curves under two labels, with no field in the record to say
they were the same measurement. That is the shape this repo guards against
everywhere else, so `render_config` raises with a message naming the missing
support instead.

Three consequences worth carrying:

- **The ops half is real and is where the comparison comes from.**
  `PerconaDriver.capabilities()` derives `incremental_index` from
  `IndexSpec.build_mode`, and `create_schema`/`create_index` split the index
  out of the table definition in `post` mode, so `build_wall_s` times a genuine
  bulk build. `config/profiles/percona.yml` runs both modes through the ops
  path. Since `build_wall_s` and `index_bytes` are ops-only fields anyway, the
  measurement §2.5 argues for is available — it is the *recall* curve per build
  mode that is not.
- **§4 never mentions `orchestrator/cli.py`, and it had to change.** The ops
  path hard-coded `build_modes = ["post"] if engine != "pgvector" else …`, so
  `PerconaDriver`'s build-mode support was unreachable from a profile until
  that lookup became per-engine. `estimate._ops_units` carries the same list
  and had to match, or the printed estimate is short by a whole ops unit of
  every query term. Both now read a shared two-engine mapping.
- **It makes Percona a fourth engine that cannot supply the build-mode
  cross-check.** `report/crosscheck.py:build_mode_axis_engines` drops the axis
  for any engine whose recall path is silent about it, which was MariaDB,
  MariaDB 12.3 and AliSQL — engines for which a build mode is *meaningless*.
  Percona joins them for a different reason: the axis is real for it and the
  recall path just does not say so.

Giving the overlay module a `build_mode` the way pgvector's and valkey's have
one is a small, well-precedented change that would close all three at once. It
is not this task's file.

## 5. The smoke run, and what it has to survive

The user's decision is to pin a SHA and prove it on a smoke run before
committing to a four-day run. Given §0's commit log, that run has to survive
more than "it produced numbers". In rough order of what the log says is most
likely to break:

1. **A restart with a live vector-indexed table.** `af72e2c0fa40` is an
   assertion in `HNSW::insert()` after an *unclean* restart, and
   `5c30ac32877a` is a dictionary assertion after a restart. The orchestrator
   restarts servers between phases. The smoke run must include at least one
   restart followed by an `INSERT` and a search, and one of them should be
   unclean (`docker kill`, not `SHUTDOWN`) because that is the case the fix was
   for.
2. **An index build that hits `innodb_hnsw_max_memory`.** `eebc427aab7b` is an
   assertion when a build aborts on capacity exhaustion. §3 shows the smoke
   profile does not reach the ceiling on its own, so this has to be provoked:
   set `innodb_hnsw_max_memory` low deliberately on a throwaway run and check
   the failure is a clean `ER_CAPACITY_EXCEEDED` that leaves a usable server.
   Doing this *before* the long run is the difference between a known limit and
   a two-day run that dies at hour thirty.
3. **A search concurrent with DDL.** `a86503214416` is an assertion when a
   vector search runs during an in-place index replace. The churn phase's
   `maintain()` (§2.7) does exactly that if any reader is still running, so the
   smoke run's churn phase is the test.
4. Every phase writing records, the plan check passing on both filter columns,
   and the report rendering with seven series.

## 6. Executable guards

`tests/test_e1_percona_design.py` holds the checks that keep this document and
the new artefacts from drifting apart. It asserts, against the files rather
than against prose: that the dialect claims in §2 match what
`percona/module.py` actually emits; that `docker/percona/` has the same shape as
the other engines' directories; that every one of the pin's four submodules has
a cmake flag and that each flag is really passed; that `init.sql` neither copies
AliSQL's `mysql_native_password` line nor sets a measurement parameter the
orchestrator owns; that the graph-memory fallback is conditional; that the
module's plan marker is not satisfiable by the index name alone and pins
`FORMAT=TRADITIONAL`; that §1's stated default, maximum and scope match the
declaration §1 itself quotes; that **every line number this document cites into
this repo still points at what the document says it does**; that the line
numbers this document quotes for
`_driver_classes()`, `build.py`'s build-mode branch and `ann_pass`'s `raise`
still point at those things; and that the §3 arithmetic reproduces from the MTR
formula. It reads this document as a file where a claim here is the thing under
test, so a figure edited in one place and not the other fails.

What those guards deliberately do **not** do is assert anything about Percona's
behaviour. Nothing in a test file in this repository can establish what a server
does, and a test that asserted "Percona requires a BIGINT UNSIGNED primary key"
would be asserting that this document contains a sentence.

### Line numbers: the document reports its own rot

This document makes about sixty `file:line` citations. Twenty-six point into
this repo and the rest into the pinned Percona tree. For most of this
document's life **two** of the twenty-six were checked, and the other
twenty-four rotted silently.

That is not a theoretical risk. Merging three PRs under this work invalidated
several in a day:

| file | was → is | cause |
|---|---|---|
| `harness/drivers/postgres.py` | 481 → 689 | C4 (#13) added 230 lines; the old line became a sentence of prose |
| `harness/main.py` | 197-199 → 209-211 | T4b (#12), uniform +12 |
| `harness/main.py` | 252 → 264 | same |
| `harness/workloads/churn.py` | 310-311 → 484-485 | T4b moved `maintain()`'s timing |
| `docs/07-planning-a-run.md` | 478 → 481 | unrelated edit |

(The superseded numbers above are written as plain digits rather than in
`file:line` form on purpose. A dead pointer typeset like a live one invites a
reader to follow it — and the citation guard, correctly, counted them as live
claims and failed until they were demoted. It caught its own document's prose
within a minute of being written, which is the behaviour being paid for.)

The first was caught by the one guard that existed — which is the whole
argument for having the rest. **The last two were caught by nothing**, and were
found only by walking every citation in the document against the files.

So every repo-side citation is now a row in a table in
`tests/test_e1_percona_design.py`, keyed by the citation *as this document
spells it* and carrying the string the document is pointing at. Both directions
fail loudly: a number edited here without the table, and a number that no
longer matches the file. The parser that finds them is itself asserted to find
something, so an empty match cannot pass as full coverage.

Two things it deliberately does not do:

- **Percona citations are skipped, by name.** They are pinned by SHA and are
  not in this checkout. Classification is a deny-list of filenames rather than
  "whatever does not resolve locally", because resolving by existence would
  silently check the document against a repo file that happened to share a
  name with a Percona source — worse than not checking. The cost is that a
  newly cited Percona file must be classified, and the completeness check is
  what makes that cost visible: it is how `ddl0loader.cc:273`, cited in §0's
  commit table, was caught unclassified the first time the guard ran.
- **Bare `:N` continuations are covered but not discovered.** Where the
  document writes "`postgres.py:689` … the dict at `:701-713`", the second
  citation has no path inside its backticks, so the parser cannot see it. The
  four such citations that point into this repo are checked explicitly. A
  *new* one could be added without a row and nothing would notice. That is the
  honest limit of the completeness claim.

### What a prose guard can and cannot do

§1 was attacked four times, each attack conceding the previous defence and
moving one layer down. The progression is the clearest statement of the limit
this series has produced, and it is worth keeping because the temptation at
every step was to add one more substring:

| | attack | status |
|---|---|---|
| **A** | delete the content, keep three tokens | caught — polarity and placement |
| **B** | keep every token, invert the **conclusions** | caught — conclusions pinned to their values |
| **C** | keep tokens **and** correct conclusions, invert the **facts** | caught — prose cross-checked against the quoted declaration |
| **D** | prose that contradicts no quote it carries | **not substring-checkable** |

C is the last layer that is cheaply checkable, and only because **the section
already quotes the primary source for every number in it**. The attack that
got through all 104 guards stated the variable was GLOBAL-only and established
once per connection, and that the default was 100000 with a floor of 40 — three
lines below a verbatim quote of `nullptr, nullptr, 40, 1, 100000, 0)` that says
otherwise. The guards required that quote; nothing compared the prose to it.
Now five checks do, value against role: which number is the default, which is
the maximum, whether the default is miscalled a floor, whether the scope
contradicts `MYSQL_THDVAR_*`, and whether the per-query read is cited.

**Pinning the numbers to the quote is the right place to stop.** What remains
after C is prose that contradicts no quote it carries, and no amount of string
matching reaches that. Reviewing the claim against the source is the only
defence left, and a fifth layer of substrings would buy nothing while adding
more ways to fire on correct writing — which is the other failure mode, below.

**Three of these guards exist because the first versions of them were vacuous,
and the review found all three.** They are worth naming, because the shapes
recur:

- A guard on `_image_march`'s path sliced the function's *text*, and the
  method's **docstring** names the path — so deleting it from the real tuple
  left the guard green. It now reads the tuple out of the AST. Measured: with
  the path removed, the text assertion returned True and the AST assertion
  False.
- The §1 guard required only that the section contain `innodb_hnsw_ef_search`,
  `**Yes.**` and `SESSION`. Replacing the whole section with hedged prose that
  kept those three tokens — no citations, no bounds, neither rider — passed.
  It now asserts **polarity and placement**: the affirmative answer must
  precede the first citation, the negating phrasings must be absent, and the
  section must carry at least four `file:line` citations and the quoted sysvar
  declaration. *A guard that asserts prose contains certain strings does not
  guard the prose.*
- The submodule-count guard compared the mapping's length against the **set** of
  number words anywhere in the Dockerfile note. The note's prose contains two,
  three, four and one, so `len(mapping) in {1,2,3,4}` held for any plausible
  mapping; with libkmip deleted the mapping was 3 and the guard passed. It now
  parses one deterministic phrase, `TAKES <n> FLAGS`, from the note's heading.
  That is membership-not-value, which is already on this repo's list of
  measured vacuity shapes.

**And one guard failed the opposite way, which is the hazard a tightening pass
creates.** The guard on §1's first rider demanded the literal `should
implement`, and it fired on rewordings that preserve every fact and both
conclusions — `should **override**` (if anything the more precise verb, since
`effective_ef` is an inherited concrete method with a default at
`base.py:374`, so overriding is literally what happens) and `**must**
implement` (a strictly stronger instruction). A second ban on `needs no edit`
was subject-blind: it fired on a true sentence about a different subject, and
that phrase recurs in a change-list document by nature.

A guard that fires on correct prose teaches the next author to weaken or delete
it, which is how the vacuity shapes reached this repo in the first place. **A
false positive on a correct document is a slower-acting version of the same
failure.** Both were loosened to the shape that survives rewording: require
the instruction *paired with its value* rather than one verb in one modal
voice, and scope a ban to the sentence carrying its subject. Both fixes were
measured against the rewordings they used to reject and against the attack
they still have to catch.

## 7. What only a live server can settle

Everything above is reading. These are the open questions, in the order the
smoke run will hit them. None is a reason not to proceed; all of them are
reasons the smoke run is not optional.

**Two questions that were on this list have been struck**, because they turned
out to be answerable by reading after all — see the note at the end of this
section. Keeping a settled question on an open list is its own kind of error:
it spends a smoke run's attention on something already known and makes the real
unknowns look less urgent.

1. **Does the image build?** Nothing here has been compiled, but this question
   was **not entirely open** and treating it as one cost a real bug. The
   submodule half was readable, and the answer was **no**: the first revision
   of `docker/percona/Dockerfile` enumerated all four submodules, miscounted
   them as two, and left `extra/libkmip` unmitigated. `CMakeLists.txt:2295-2297`
   descends into it when **neither** `WITH_COMPONENT_KEYRING_KMIP` nor
   `WITHOUT_COMPONENT_KEYRING_KMIP` is defined, and `prepare-sources.sh` ships
   a `git archive`, which omits submodule paths — so cmake configure would have
   failed on a directory that is not in the tarball. Fixed with
   `-DWITHOUT_COMPONENT_KEYRING_KMIP=1`; the Dockerfile now carries one flag per
   submodule and explains why `-DWITH_...=OFF` would not have worked.

   What genuinely remains: whether a 9.7 tree builds clean with Ubuntu 24.04's
   GCC 13 (the branch recently made clang-23 its CI default — `ccb8bbd9f156`),
   and whether the apt list is complete. Out-of-source is required, not
   optional: `CMakeLists.txt:583` defaults `FORCE_INSOURCE_BUILD` OFF.
2. **Can MariaDB Connector/C authenticate over TCP?** This base has no
   `mysql_native_password` — `mysql-test/r/mysql_native_password_removed.result:4-5`
   records `CREATE USER … IDENTIFIED WITH mysql_native_password BY 'abcd'`
   returning `ERROR HY000: Plugin 'mysql_native_password' is not loaded` — so
   copying `docker/alisql/init.sql`'s line would make the container fail to
   boot, and `docker/percona/init.sql` uses the server default
   `caching_sha2_password` instead. The **ann pass** connects over a unix socket
   (`vb_mysql.py:237`, `unix_socket=`), where the cleartext exchange is allowed.
   The **ops harness** connects over TCP (`mysql_family.py:162-172`, `host=`/`port=`),
   where `caching_sha2_password` needs TLS or the server's RSA public key.
   `caching_sha2_password_auto_generate_rsa_keys` defaults ON so the key exists;
   whether Connector/C 3.3.x asks for it without an explicit option is untested.
   **This is the first thing to check, because it blocks every ops measurement.**
3. **Does a `bytes` parameter arrive with the binary charset?** §2.1: `cs !=
   &my_charset_bin` is a hard refusal at `sql/field.cc:7097`. Expected to work,
   since AliSQL's `Field_vector` takes the same path today, but the failure mode
   is every INSERT erroring and it costs one statement to check.
4. **Does `information_schema.COLUMNS.CHARACTER_OCTET_LENGTH` populate for a
   `VECTOR` column?** `vb_mysql.py:428-436` divides it by 4 and returns 1 if it
   is NULL, which would make `_warm_cache` warm nothing and `_verify_plan`
   report a false negative. `percona/module.py` prefers `VECTOR_DIM()`
   (`sql/item_create.cc:1660`) and falls back, so it is covered either way — but
   the shared path is still wrong for the next engine.
5. **Actual graph memory against the §3 floor.** The projections use an MTR
   assertion's floor and its 2× ceiling, and §3 now also records that
   `Innodb_hnsw_memory_used` omits the node map, so the gauge itself is a floor.
   One reading after a 20,000-row load replaces the whole table with a
   measurement, and the `main` profile should not start until it has.
6. **Ingest rate.** Unknown. MHNSW and VIDX manage ~150 and ~55 rows/s on 1M
   rows (`CLAUDE.md`), and Percona's figure decides whether a Percona column in
   the `main` profile is affordable at all. It is also the number
   `orchestrator/estimate.py` needs and must not be guessed.
7. **Is `FORCE INDEX (vi)` accepted on a `VECTOR KEY`?** `Dialect.force_index_hint`
   is set for all three MySQL-family engines and is dead repo-wide today —
   `vb_mysql.py:400` only uses it when `force_index=True` and no call site
   passes that — so this costs nothing right now. It is listed because Percona
   is the engine where it might not *parse*: the pin's own tip commit
   (`02a8b80c`, PS-11626) is about index hints and vector ANN queries, having
   just fixed `IGNORE INDEX … FOR ORDER BY (vk)` being ignored for them. If a
   future caller switches the hint on, a `FORCE INDEX` the parser rejects turns
   every plan check into an exception instead of a plan.
8. **The ops path's `march` field.** §4.8: `harness/main.py:209-227` is a second
   `-march` reader in a different container, which no overlay override can
   reach. Both tuples now carry `/opt/percona/.march`, so this should read the
   image's real value — but *nothing in this repository can prove that*, since
   proving it means a built image with a `/opt/percona/.march` in it. It stays
   on this list for that reason, and because it is a thing to **look at in the
   first smoke run's records**: it will not fail anything, it will just
   quietly say `unknown` if the file is not where the Dockerfile puts it.
### Three items struck from this list, and why

Each was on it in a previous revision. All three are now settled from the
pinned tree, and all three are **reassuring** — which is exactly why they
should not sit on an open-questions list implying doubt. Keeping a settled
question open is not a conservative choice; it spends a smoke run's attention
on something already known and makes the real unknowns look less urgent.

- **`hypergraph_optimizer` cannot be enabled in this build.** Three gates in
  `sql/sys_vars.cc` (`:232-234`, `:3435-3438`, `:3441-3452`) all key on CMake
  symbols the Dockerfile does not define, and the last raises
  `ER_HYPERGRAPH_NOT_SUPPORTED_YET` on an attempt. The silent-full-scan risk
  §2.6 described cannot occur here. Moved into §2.6.
- **There is no third `innodb_hnsw_*` variable.** `innobase_system_variables[]`
  registers exactly two (`ha_innodb.cc:25112`, `:25174`) and the status array
  exactly one (`:1335`). §2.6's table is complete, established from the
  registration arrays rather than from the truncated tree listing the previous
  revision used. Moved into §2.6.
- **The index replace does not block readers.** It takes a SHARED lock, which
  permits concurrent reads and blocks writes: any ALTER leaving a vector index
  is forced offline (`handler0alter.cc:1496-1500`, `:1509-1513`) and the
  handler returns `HA_ALTER_INPLACE_SHARED_LOCK_AFTER_PREPARE` (`:1523-1524`).
  The previous revision kept this open because the "LOCK=SHARED it is" comment
  it had found was scoped to a table rebuild and the replace is not one — a
  correct refusal to generalise, but the answer was a hundred lines further on
  under a rule that is not about rebuilds. Moved into **§2.7**, not §2.6: it is
  a lock level, and the maintenance table is where the other lock facts live.
