# Benchmarking vector indexes

vector-bench runs performance tests for vector search in databases. You name the
engines, it builds them from pinned versions, puts each one in the same
container on the same cores with the same data, runs the same measurements, and
writes a report.

This post is the method. Results are separate.

## The problem being indexed

Embeddings are fixed-length float arrays produced by a model, typically 100 to
1536 dimensions, with the property that semantically similar inputs land close
together under some distance metric. Cosine and L2 are the usual ones, and which
applies is a property of the model, not a choice you get to make at query time.

Similarity search is k-nearest-neighbour search over those arrays:

```sql
SELECT id FROM documents ORDER BY distance(embedding, ?) LIMIT 10;
```

**k** is the number of rows requested, 10 here. Exact kNN means computing the
distance from the query vector to every row and sorting. No B-tree or hash
orders by proximity in 1536 dimensions, so exact search is a full scan with
arithmetic proportional to rows times dimensions.

Vector indexes trade exactness for speed. They examine a few thousand candidates
instead of every row, which is **approximate nearest neighbour** search, ANN.
The results are usually correct. Quantifying "usually" is what most of this
benchmark is about.

## HNSW and IVF

**HNSW** (Hierarchical Navigable Small World) is a layered proximity graph. Each
vector is a node linked to some number of its nearest neighbours. Upper layers
are sparse with long-range edges, lower layers dense and local. A search enters
at the top, descends greedily toward the query vector, and finishes in the
bottom layer.

Two parameters:

- **M**, the number of links per node. Fixed at build time. Higher M means a
  better-connected graph and better recall at a given search width, plus a
  slower build and a larger index.
- **ef_search**, the size of the candidate list the search maintains. A session
  variable. Higher visits more nodes, finds better answers, runs slower.

**ef_construction** is the same idea applied during the build. Not every engine
exposes it.

**IVF** (Inverted File) partitions rather than links. Build time clusters the
vectors into `nlist` cells with centroids; query time compares against the
centroids and scans the nearest `nprobe` cells. Cheaper to build and lighter in
memory than HNSW, generally worse recall at the same latency, and it misses when
the true neighbour sits just over a cell boundary.

We only test engines running HNSW, which is what most databases have shipped.
Scoring an IVF engine on the same chart would mostly measure the gap between two
algorithms rather than implementation quality, so IVF-only engines get their own
bucket.

## Recall

**Recall@k** is the fraction of the true top-k that came back. Nine of ten
correct rows scores 0.9 for that query, averaged over the query set.

Scoring it needs the exact answer, the **ground truth**, computed once by brute
force with no index involved. Public ANN datasets ship theirs.

The thing that makes this unlike other database benchmarking: recall is a
setting, not a property. Same index, same data, same hardware, same queries,
`ef_search` the only variable:

```
ef_search=10    3,678 queries/sec    recall 0.9593
ef_search=800     409 queries/sec    recall 0.9987
```

Nine times the throughput and both rows are valid measurements. A vector QPS
figure without its recall can't be interpreted, and a recall figure without
throughput is worth just as little, since recall 1.0 is always available by
scanning. Every measurement here is a pair.

## What the harness puts on each engine

One table per engine: id, an integer `tag` for filter tests, the vector, and an
HNSW index on it at a configured M.

```sql
CREATE TABLE t1 (
  id   INTEGER PRIMARY KEY,
  tag  INTEGER NOT NULL,
  v    VECTOR(1536)
);
```

Two queries, unfiltered and filtered:

```sql
SELECT id FROM t1                 ORDER BY distance(v, ?) LIMIT 10;
SELECT id FROM t1 WHERE tag < ?   ORDER BY distance(v, ?) LIMIT 10;
```

`tag` holds 0-99 uniformly, so `tag < 10` passes ~10% of rows and `tag < 1`
passes ~1%. That's the selectivity control.

Syntax differs everywhere: index declared in CREATE TABLE or as a separate
statement, distance functions under different names. That's the driver's
problem and it's the only engine-specific code in the harness.

Each engine also has at least one setup detail that silently invalidates the
numbers. PostgreSQL, for instance, TOASTs a 1536-dimension vector out of line
unless the column is set `STORAGE PLAIN`, putting a detoast on every distance
comparison. One line of DDL, and without it you publish an engine looking slow
for reasons unrelated to its vector search.

## Measurements

**Recall against throughput.** Sweep `ef_search` against a fixed index,
recording recall and QPS at each point, repeated at a few values of M. k=10
throughout. Query vectors come from the dataset's held-out set, never from
loaded rows, since searching for vectors already in the index is a much easier
problem.

The two parameters behave differently and it shapes the run: `ef_search` is a
session variable, so extra points reuse the index and cost almost nothing. M is
structural, so each value means dropping the table and reloading the corpus,
which is hours. Profiles carry many `ef_search` points and few M values.

**Build cost.** Wall time, rows/sec, on-disk index size, peak RSS.

Engines don't build alike. Some maintain the graph on every INSERT, so the index
is complete when the load finishes and there's no separate build. Others load
first and build in one pass afterwards, which is far faster in total but leaves
the table unusable for vector search until it completes. On an engine supporting
both we measured 18x between its own two paths. A bulk-build number against
another engine's incremental number is not a comparison of engines, so both
paths get measured and labelled wherever they exist.

Peak RSS comes from the server container's cgroup with the server alone in it.
The client runs in a second container over a private network, otherwise several
GB of Python arrays holding the dataset land in the database's accounting.

**Concurrency.** QPS and latency percentiles from 1 to 32 clients. Graph caching
strategies differ substantially between engines and only show up under
contention. Scaling efficiency is reported next to raw QPS, because an engine
flat from 2 clients onward with p99 degrading 15x is doing something different
from one that scales.

**Filtered search**, at selectivities down to 1%. Filtering changes the correct
answer: the true top 10 among `tag < 10` is not the true top 10 overall. Ground
truth is recomputed by brute force over the passing subset for every
selectivity. Scoring filtered results against the shipped unfiltered ground
truth returns near-zero recall for everyone, which we did for a while.

Queries returning fewer than k rows are counted separately. An engine can
exhaust its candidate list before finding 10 rows that satisfy the predicate; in
one run that hit 81 of 200 queries. Recall over short result sets is real but
not comparable to recall over full ones, so the count travels with it.

**Churn.** Recall and throughput before and after deleting and reinserting part
of the corpus, since deletions leave graph edges pointing at tombstones. Whether
a rebuild recovers the loss is untested so far.

## Controls

Everything runs twice. The **normalized** pass gives every engine identical CPU,
memory and cache budgets. The **tuned** pass gives each one what its
documentation recommends. Tuned is more realistic and less controlled, so it
doesn't replace normalized. A result holding across both is about the engine.

Cores are pinned explicitly, one logical CPU per physical core (SMT siblings
share execution units and don't behave like two cores), and never mixed across
P-cores and E-cores on hybrid parts, where migration between core types produces
more variance than several of the effects being measured. Durability is relaxed
identically everywhere, otherwise the comparison is between default fsync
policies.

Some asymmetries can't be removed. A knob only one engine exposes stays unused
in the normalized pass rather than handing that engine a tuning axis the others
lack. An engine requiring a specific isolation level sets it for everybody.
Placeholder defaults get sized from a shared budget, since judging an engine on
a 16 MiB graph cache measures nothing. These go in a "known asymmetries" section
above the results.

Several of these implementations ship hand-written AVX-512 distance kernels. A
512-bit FMA covers 16 floats per instruction, so the same index on a CPU without
AVX-512 is a materially different benchmark, and the penalty isn't uniform
across engines. CPU model and ISA flags are in every manifest, along with engine
versions and commits, image IDs, and resource limits as resolved rather than as
requested. The report generator won't run without a manifest.

## Reading the results

Validity section first. Reports go environment, validity, known asymmetries,
results, so failed phases, short result sets and missing ISA extensions arrive
before any chart.

The failure mode to watch for is the silent full scan. An engine that declines
the vector index and scans returns exact results slowly, which presents as high
recall and low throughput and is indistinguishable from a conservatively tuned
index without reading the plan.

It happens for mundane reasons. One optimizer costs the vector index against a
scan and takes the scan once LIMIT exceeds roughly a quarter of the table, and
we haven't found a setting that moves it. Another falls back with no error and
no warning when the query's distance operator doesn't match the operator class
the index was built with, so a one-character difference yields a sequential scan
and a sort.

Every driver therefore runs EXPLAIN per configuration and checks the index name
appears in the plan.

```
WARNING: vector index NOT used (k=10, filtered=True). Plan: ...Seq Scan...
```

Anything that scanned lands in validity. This is the easiest way to produce
impressive vector numbers by accident.

Recall/throughput results are a curve, not a number: sweep `ef_search`, plot
recall against QPS, take the upper-left envelope. One engine beats another only
where its curve sits above at equal recall, and crossing curves mean the answer
depends on the accuracy you need. Since curves invite comparing shapes rather
than heights at one x, there are also bars for QPS at recall floors of 0.90,
0.95 and 0.99.

## What went wrong building it

First ingest numbers were garbage: one INSERT per round trip with autocommit on,
88 rows/sec. Batching 500 per transaction took the same engine to 373.

Filtered search and churn were scored against full-corpus ground truth on runs
that used a subset of rows, so every engine looked bad and the bug was ours.
Ground truth is now keyed on dataset, k, row count and selectivity.

Both resource passes shared a results directory, and the ANN runner skips
configurations that already have results. The tuned pass was skipping everything
normalized had computed, so tuned numbers were mostly normalized numbers.
Separate directories per pass.

Readiness probes lie. One engine's standard connection check succeeds before the
database it creates exists, so the probe passed and the first query failed. The
probes run a real query against the real database now.

Most recently, on a 1536-dimension corpus: the ANN runner holds the dataset in
memory twice, once in the parent and again in a forked worker, unshared. That's
~12 GB for a million embeddings on top of the server, in a container sized for
the server alone. The kernel killed the worker. The runner doesn't check worker
exit codes, so it logged "Terminating 1 workers", exited 0 and wrote nothing,
which is identical to a run with nothing left to do. Container memory is sized
from the dataset now.

## Out of scope

Hardware, per the AVX-512 note above.

Scale past ~1M vectors. Datasets run 60k to 1.2M and stay largely cache
resident. At 10M the graphs go to disk and the ordering may not survive.

Anything else on the box. A forgotten build moved one engine's numbers 2x during
development.

Your query distribution. Public datasets are comparable, not representative.

## Adding an engine

Four artifacts: a Dockerfile producing runtime and test images from a pinned
version, a config mapping ports, credentials and server settings onto the
normalized budget, a module for the recall/throughput path, and a driver
implementing create index, load, query, filtered query, index size and the
EXPLAIN check. The driver runs about 200 lines. Nothing else changes, and
adding an engine doesn't alter what existing numbers mean.

## Next

Results, published with manifests and raw per-configuration records.

[github.com/EvgeniyPatlan/vector-bench](https://github.com/EvgeniyPatlan/vector-bench).
If we're measuring something wrong, tell us.
