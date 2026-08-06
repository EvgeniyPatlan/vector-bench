# Benchmarking vector indexes

vector-bench runs performance tests for vector search in databases. You name the
engines you want. It builds them from pinned versions, puts each one in the same
container on the same cores with the same data, runs the same measurements, and
writes a report.

This post is about the method. The results are a separate post.

## What's being indexed

An embedding is a fixed-length array of floats produced by a model. Typical
sizes run from 100 to 1536 dimensions. The useful property is that semantically
similar inputs come out close together under some distance metric, usually
cosine or L2. Which metric applies is a property of the model that produced the
embeddings, not something you choose at query time.

Similarity search is k-nearest-neighbour search over those arrays:

```sql
SELECT id FROM documents ORDER BY distance(embedding, ?) LIMIT 10;
```

**k** is the number of rows you ask for, 10 in that query.

Getting that answer exactly right means computing the distance from the query
vector to every row, then sorting. No B-tree or hash index helps, because
neither one orders by proximity in 1536 dimensions. So exact vector search is a
full scan, with arithmetic proportional to rows times dimensions on top.

A vector index trades exactness for speed. Instead of every row it examines a
few thousand promising candidates and returns the best it found. That's
**approximate nearest neighbour** search, or ANN. The answers are usually
correct. Putting a number on "usually" is what most of this benchmark does.

## How the indexes work

**HNSW** stands for Hierarchical Navigable Small World. If you know how a skip
list works, the shape is familiar.

It's a graph of vectors, built in layers. Every vector is a node, linked to some
number of its nearest neighbours. The top layer has few nodes and its links jump
long distances across the data. Each layer below has more nodes and shorter
links. A search starts at the top and keeps hopping to whichever neighbour is
closest to the query vector. When it can't get closer it drops a layer and
carries on, until it runs out of layers.

Two parameters matter:

- **M** is the number of links per node. It's fixed when the index is built.
  Higher M gives a better-connected graph, so better recall at a given search
  width, at the cost of a slower build and a larger index.
- **ef_search** is how big a candidate list the search keeps as it walks. It's a
  session variable, so you can change it per query. Higher means the search
  visits more nodes, finds better answers and runs slower.

![Three panels showing an HNSW search descending from a sparse top layer to a
dense bottom layer, getting closer to the query vector at each
step](img/hnsw.svg)

There's also **ef_construction**, which is the same idea applied while the index
is being built. Not every engine exposes it.

**IVF** stands for Inverted File. It partitions the data instead of linking it,
not unlike list partitioning on a table.

At build time it groups the vectors into `nlist` clusters, and each cluster gets
one representative vector at its centre. At query time it compares the query
against those representatives, picks the closest `nprobe` clusters, and searches
only inside those. It builds much faster than HNSW and uses less memory, but
usually gives worse recall at the same speed. It misses when the true neighbour
happens to sit just outside the clusters it looked in.

![Vectors grouped into eight clusters, each with a representative vector, with
the two clusters nearest the query highlighted as the only ones
searched](img/ivf.svg)

We only test engines running HNSW, which is what most databases have shipped.
Scoring an IVF engine on the same chart would mostly measure the difference
between two algorithms rather than how well each database implemented one, so
IVF-only engines go in their own bucket.

## Recall, and why one number is never enough

**Recall@k** is the fraction of the true top-k that actually came back. If nine
of the ten rows returned belong in the real top 10, that query scored 0.9.
Average over the query set and you have recall@10. A reported recall of 0.9593
means that on average about 9.6 of every 10 rows returned were correct.

![A query vector with the true ten nearest neighbours circled: eight were
returned correctly, two were missed, and two rows from outside the circle were
returned instead](img/recall.svg)

Scoring it requires knowing the right answer. That's the **ground truth**,
computed once by brute force with no index involved. The public ANN datasets
ship theirs alongside the vectors.

Here's what makes this different from most database benchmarking. Recall isn't a
property of the engine. It's a setting. Below is one index, same data, same
hardware, same query set, with `ef_search` as the only variable:

```
ef_search=10    3,678 queries/sec    recall 0.9593
ef_search=800     409 queries/sec    recall 0.9987
```

Nine times the throughput separates those rows, and both are valid measurements
of the same index.

So a vector search QPS number with no recall next to it can't be interpreted.
You don't know how often it was returning wrong answers. A recall number with no
throughput is just as useless, because recall 1.0 is always available if you
turn the index off and scan.

Every measurement in this benchmark is a pair.

## What the harness puts on each engine

One table per engine. An id, an integer `tag` column used only by the filter
tests, the vector, and an HNSW index on the vector at a configured M.

```sql
CREATE TABLE t1 (
  id   INTEGER PRIMARY KEY,
  tag  INTEGER NOT NULL,
  v    VECTOR(1536)
);
```

Then two queries. The plain top-k search, and the same search restricted to a
subset of rows:

```sql
SELECT id FROM t1                ORDER BY distance(v, ?) LIMIT 10;
SELECT id FROM t1 WHERE tag < ?  ORDER BY distance(v, ?) LIMIT 10;
```

`tag` holds values 0 to 99, spread evenly. So `tag < 10` passes about 10% of the
rows and `tag < 1` passes about 1%. That's how filter selectivity is controlled.

Every engine writes all of this differently. Some declare the index inside
CREATE TABLE, others need a separate CREATE INDEX. The distance functions have
different names. Translating that is the driver's job, and drivers are the only
engine-specific code in the harness.

Each engine also has at least one setup detail that will quietly ruin the
numbers. PostgreSQL, for example, stores oversized values out of line via TOAST,
and a 1536-dimension vector counts as oversized. Unless the column is set to
`STORAGE PLAIN`, every distance comparison pays for an extra fetch. That's one
line of DDL. Miss it and you publish an engine looking slow for a reason that
has nothing to do with its vector search.

## What we measure

**Recall against throughput.** Sweep `ef_search` against a fixed index and
record recall and QPS at each point. Repeat at a few values of M. k=10
throughout. The query vectors are the ones the dataset ships for this purpose.
They are not rows we loaded, because searching for a vector that's already in
the index is a much easier problem and would flatter everyone.

The two settings behave very differently, and that decides how long a run
takes. `ef_search` is just a session variable, so sweeping it reuses the index
that's already built and each extra point is nearly free. M is baked into the
index. Every value of M means dropping the table and loading the whole dataset
again, which takes hours. So we test many values of `ef_search` and only a few
values of M.

**Build cost.** Wall time, rows per second, index size on disk, and peak memory.

Engines don't build the same way, which makes this the easiest place to publish
nonsense. Some maintain the graph on every INSERT, so the index is finished the
moment the load is done and there's no separate build step. Others load the rows
first and build the whole graph afterwards in a single pass. The second approach
is much faster overall, but the table can't serve vector search until it
finishes.

On one engine that supports both, we measured an 18x difference between its own
two paths. So putting a bulk-build number next to another engine's incremental
number isn't a comparison of engines. We measure both paths wherever an engine
has both, and the report labels which is which.

Peak memory comes from the server container's own accounting, and the server is
the only thing in that container. The test client runs in a second container over a
private network. Otherwise the several GB of Python arrays holding the dataset
would be charged to the database.

![The server container holding only the database with a pinned cpuset and a hard
memory limit, connected over a private network to a separate client container
running the harness](img/harness.svg)

**Concurrency.** QPS and latency percentiles from 1 to 32 clients. Engines cache
their graphs in quite different ways, and that only becomes visible when clients
compete for the same cache. Next to the raw QPS we report how much of the
ideal speedup each engine actually got. An engine that stops gaining throughput
at 2 clients while its p99 latency gets 15 times worse is behaving very
differently from one that keeps scaling.

**Filtered search**, at selectivities down to 1% of rows passing.

Filtering changes what the correct answer is. The true top 10 among rows with
`tag < 10` isn't the true top 10 overall. So for every selectivity we recompute
the ground truth by brute force over just the rows that pass. Scoring filtered
results against the unfiltered ground truth that shipped with the dataset gives
every engine a recall near zero. We did exactly that for a while.

We also count queries that returned fewer than 10 rows. An engine can run out of
candidates before it finds 10 that satisfy the predicate. In one run that
happened on 81 of 200 queries. Those recall numbers are real, but they aren't
comparable to recall over full result sets, so the count is reported next to
them.

**Churn.** Recall and throughput before and after deleting and reinserting part
of the corpus. Deletions leave graph edges pointing at rows that are gone, so
HNSW is expected to degrade. Whether rebuilding the index recovers the loss is
something we haven't tested yet.

## Keeping the comparison fair

Everything runs twice.

The **normalized** pass gives every engine identical CPU, memory and cache
budgets. A difference in those results is about the implementation, not about
who got more RAM. The **tuned** pass lets each engine use what its own
documentation recommends. Tuned is more realistic and less controlled, so it
doesn't replace normalized. A result that holds across both passes is about the
engine.

Cores are pinned explicitly. We use one logical CPU per physical core, because
SMT siblings share execution units and two threads on one core don't perform
like two cores. On hybrid CPUs we never mix P-cores and E-cores, since migration
between core types adds more variance than several of the effects we're trying
to measure. Durability settings are relaxed identically everywhere, otherwise
the comparison would be between default fsync policies.

Some differences can't be equalised, so we record them instead. A knob that only
one engine exposes goes unused in the normalized pass, because using it would
hand that engine a tuning axis the others don't have. An engine that requires a
particular isolation level gets that level set for everyone. Defaults that are
clearly placeholders get sized from a shared budget, since judging an engine on
a 16 MiB graph cache doesn't measure anything. All of these appear in a "known
asymmetries" section above the results.

Several of these implementations ship hand-written AVX-512 code for the distance
calculations. One AVX-512 instruction does the arithmetic for 16 floats at once,
so distance maths runs several times faster on a CPU that supports it. The same
index on a machine without AVX-512 is effectively a different benchmark, and the
slowdown isn't the same for every engine.

So the CPU model and its feature flags go into every run's manifest, along with
engine versions and commits, image IDs, and the resource limits as they actually
resolved rather than as requested. The report generator
won't run without a manifest.

## Reading the results

Read the validity section before you look at a chart. The reports go
environment, then validity, then known asymmetries, then results. Failed phases,
short result sets and a missing instruction set all show up before any graph.

The failure mode to watch for is the silent full scan. An engine that decides
not to use the vector index and scans the table instead returns exact results,
slowly. In the output that looks like high recall and low throughput, which is
indistinguishable from a conservatively tuned index unless you read the plan.

It happens for ordinary reasons. One engine's optimizer costs the vector index
against a table scan and picks the scan once the LIMIT is above roughly a
quarter of the table, and we haven't found a setting that moves it. Another
falls back with no error and no warning if the query asks for a different
distance than the index was built for. Build the index for cosine, write the
query with the L2 operator, and you get a sequential scan and a sort with
nothing to tell you about it.

So every driver runs EXPLAIN for each configuration and checks that the index
name appears in the plan.

```
WARNING: vector index NOT used (k=10, filtered=True). Plan: ...Seq Scan...
```

Anything that scanned goes into validity. This is the easiest way to produce
impressive vector benchmark numbers by accident.

For the recall and throughput results themselves, the useful presentation is a
curve rather than a number. Sweep `ef_search`, plot recall on one axis and QPS
on the other, and keep the best points: for each level of accuracy, the highest
throughput anything reached at it. One engine beats another only where its curve
sits above the other's at the same recall. If the curves cross, the answer
depends on how accurate you need to be.

![Left: the measured sweep, throughput falling from 3,678 to 409 queries per
second as recall rises from 0.959 to 0.999. Right: two curves crossing, so which
engine is faster depends on the recall you need](img/tradeoff.svg)

Curves invite comparing shapes instead of heights at one point, so there are
also bar charts of QPS at recall floors of 0.90, 0.95 and 0.99. Pick the
accuracy you'd accept and read across.

## What went wrong while building it

Our first ingest numbers were garbage. The load path was doing one INSERT per
network round trip with autocommit on, and we measured 88 rows per second.
Batching 500 rows per transaction took the same engine to 373.

Filtered search and churn were scored against the full-corpus ground truth even
on runs that used a subset of rows. Every engine looked bad, and the bug was
ours. Ground truth is now keyed on dataset, k, row count and selectivity.

Both resource passes shared one results directory, and the ANN runner skips
configurations that already have results. So the tuned pass was skipping
everything the normalized pass had computed, and the tuned numbers were mostly
normalized numbers.

Readiness probes lie. One engine's standard connection check returns success
before the database it's supposed to create actually exists. Our probe passed,
the first query failed, and it looked like an engine problem for longer than it
should have. Probes now run a real query against the real database.

The most recent one, on a 1536-dimension dataset. The ANN runner holds the whole
dataset in memory twice, once in the parent process and again in a forked
worker, and the copies aren't shared. That's about 12 GB for a million
embeddings, on top of whatever the server is using, in a container we had sized
for the server alone. The kernel killed the worker. The runner doesn't check
worker exit codes, so it logged "Terminating 1 workers", exited successfully and
wrote nothing. That's identical to what a run with nothing left to do looks
like. Container memory is sized from the dataset now.

## What this doesn't measure

Your hardware, for the AVX-512 reason above.

Scale past about a million vectors. The datasets run from 60,000 to 1.2 million
rows and stay largely in cache. At 10 million the graphs go to disk and the
ordering may not survive.

Anything else running on the machine. A build we'd forgotten about moved one
engine's numbers by 2x during development.

Your query distribution. Public datasets are comparable, not representative.

## Adding an engine

Four things. A Dockerfile producing a runtime image and a test image from a
pinned version. A config mapping ports, credentials and server settings onto the
normalized budget. A module for the recall and throughput path. And a driver
implementing create index, load, query, filtered query, index size and the
EXPLAIN check.

The driver runs about 200 lines. Nothing else in the harness changes, and adding
an engine doesn't alter what any existing number means.

## What's next

Results, published with the manifests and the raw per-configuration records so
they can be checked.

Everything is at
[github.com/EvgeniyPatlan/vector-bench](https://github.com/EvgeniyPatlan/vector-bench).
If we're measuring something wrong, tell us.
