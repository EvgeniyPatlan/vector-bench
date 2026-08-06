# A plain introduction to benchmarking vector indexes

vector-bench runs performance tests for vector search in databases. You tell it
which engines you want, it builds them from pinned versions, puts each one in
the same container on the same cores with the same data, runs the same
measurements, and writes a report.

This post is about the method. The results are a separate post.

If you already work with databases but haven't done anything with vectors, the
first half of this is for you. Vector search has its own vocabulary and most of
the benchmark numbers floating around are hard to interpret without it.

## What a vector index is

Something turns your data into a list of numbers. A sentence, an image, a
product description goes into a model and comes out as an array of floats, maybe
100 of them, maybe 1536. That array is called an embedding, or just a vector.
The useful property is that similar things produce vectors that sit close
together, so "find me similar documents" becomes "find me the closest vectors".

Closeness is a distance. Two common ones are Euclidean distance (straight-line
distance) and cosine distance (the angle between the vectors, which ignores how
long they are). Which one you use depends on the model that produced the
embeddings.

So the query you want is: given this vector, return the 10 rows whose vectors
are nearest to it. In SQL that's

```sql
SELECT id FROM documents ORDER BY distance(embedding, ?) LIMIT 10;
```

The number of rows you ask for is conventionally called **k**. Here k=10. This
is a **top-k** query, and the rows it returns are the **nearest neighbours** of
your query vector.

Now the problem. To get that answer exactly right, the database has to compute
the distance from your query vector to every single row and then sort. On a
million rows of 1536 floats that's a lot of arithmetic per query, and no
ordinary index helps, because a B-tree can't order by "closeness in 1536
dimensions". Exact vector search is a full scan with heavy math attached.

A **vector index** is the fix, and the trade it makes is that it stops being
exact. Instead of looking at every row it looks at a few thousand promising
ones and returns the best it found. This is called **approximate nearest
neighbour** search, or **ANN**. It's usually right and it's dramatically faster.
Usually right is the part that matters for benchmarking.

## The two index types you'll run into

**HNSW** stands for Hierarchical Navigable Small World. It's a graph. Every
vector is a node, and each node keeps links to some of its nearest neighbours.
The links are arranged in layers: the top layer is sparse and its links jump
long distances, and each layer down gets denser and more local. A search enters
at the top, walks greedily toward the query vector, drops down a layer, and
repeats until it's in the neighbourhood of the answer.

HNSW has two settings worth knowing:

- **M** is how many links each node keeps. It's fixed when the index is built.
  Higher M gives a better-connected graph, so better accuracy, but the index
  takes longer to build and takes more space.
- **ef_search** is how many candidate nodes the search is allowed to visit per
  query. It's a session setting you can change at any time. Higher means the
  search looks harder, finds better answers, and runs slower.

(There's also **ef_construction**, the same idea as ef_search but applied while
building the index. Not every engine exposes it.)

**IVF** stands for Inverted File. It partitions instead of building a graph.
At build time it clusters the vectors into some number of lists, each with a
centroid. At query time it compares the query against the centroids, picks the
few closest lists, and searches only those. Its knobs are **nlist** (how many
clusters) and **nprobe** (how many of them to actually search per query).

IVF builds much faster and uses less memory. HNSW usually gives better accuracy
at the same speed. Most databases adding vector search have picked HNSW.

We only test engines running HNSW. Putting an IVF engine on the same chart would
mostly measure the difference between two algorithms rather than how well each
database implemented one. Anything IVF-only gets scored separately.

## Recall, which is the number everyone leaves out

The index is approximate, so sometimes the 10 rows it returns aren't the real 10
nearest. **Recall@k** is how much of the correct answer you actually got.

Ask for 10 neighbours. If 9 of the rows you got back belong in the true top 10,
that query scored 0.9. Average that over a few thousand queries and you have
recall@10, usually written as a number between 0 and 1.

To score it you need to know the true answer, which is called the **ground
truth**. That's computed once by brute force, comparing every query against
every vector in the dataset with no index involved. The standard public
datasets ship with theirs already computed.

Here's the part that makes vector benchmarking different from anything else you
measure in a database. Recall is not a property of the engine. It's a setting.
Turn `ef_search` up and the index visits more candidates, so recall goes up and
throughput goes down. Same index, same data, same hardware, same query set. Only
`ef_search` changed:

```
ef_search=10    3,678 queries/sec    recall 0.9593
ef_search=800     409 queries/sec    recall 0.9987
```

Nine times the throughput between those two rows and both of them are true. If
someone tells you their database does 3,678 vector queries a second, they've
told you nothing, because you don't know what they set `ef_search` to and
therefore don't know how often the answers were wrong.

The reverse is just as bad. A recall number with no throughput next to it is
also meaningless, because recall 1.0 is always available if you're willing to
turn the index off and scan the table.

So every measurement in this benchmark is a pair. Recall and speed, together,
never one without the other.

## What the harness actually does to each engine

Every engine gets the same table: an id, a vector column, and one extra integer
column we use for filtering. An HNSW index on the vector column, built with a
given M.

```sql
CREATE TABLE t1 (
  id    INTEGER PRIMARY KEY,
  tag   INTEGER NOT NULL,     -- only used for the filtered tests
  v     VECTOR(1536)          -- the embedding
);
-- plus an HNSW index on v, with M as configured
```

Then two queries. The plain top-k search:

```sql
SELECT id FROM t1 ORDER BY distance(v, ?) LIMIT 10;
```

and the same search restricted to a subset of rows, which is the case people
actually care about in a database:

```sql
SELECT id FROM t1 WHERE tag < ? ORDER BY distance(v, ?) LIMIT 10;
```

The `tag` column is filled with values 0 to 99 spread evenly, so `tag < 10`
selects roughly 10% of the table and `tag < 1` selects roughly 1%. That's how we
control filter selectivity.

Every engine writes this differently. Some declare the index inside CREATE
TABLE, others need a separate CREATE INDEX. The distance function has a
different name everywhere. That's the driver's job to translate, and it's the
only engine-specific code in the harness.

Each engine also has at least one setup detail that will quietly wreck your
numbers if you miss it. One example: PostgreSQL stores oversized column values
out of line in a separate table (this is called TOAST), and a 1536-dimension
vector qualifies as oversized, so unless you tell it otherwise every single
distance comparison pays for an extra fetch and decompress. One line of DDL
fixes it. Miss it and you publish an engine looking slow for a reason that has
nothing to do with its vector search. Every driver carries a few of these.

## What we measure

**Recall against throughput.** Sweep `ef_search` across its useful range against
a fixed index and record the recall and queries per second at each point. Repeat
at a few values of M. k=10 everywhere. The query vectors come from the dataset's
own held-out query set, never from the rows we loaded, because searching for
vectors that are already in the index is a much easier problem and would
flatter everyone.

Worth knowing why those two settings are treated differently. `ef_search` is a
session variable, so sweeping it reuses the same index and extra points are
nearly free. M is baked into the index, so every M value means dropping the
table and loading the entire dataset again, which is hours. That's why the
profiles have many `ef_search` points and only a few M values.

**Build cost.** Wall time, rows per second, index size on disk, peak memory.

This is the easiest place to publish nonsense, because engines don't all build
the same way. Some maintain the graph on every INSERT, so the index is finished
the moment the data is loaded and there's no separate build step. Others load
the rows first and build the whole graph afterwards in one pass, which is much
faster in total but means the table is unusable for vector search until it
finishes. On one engine that supports both we measured an 18x difference between
its own two paths. Putting one engine's bulk-build number next to another
engine's incremental number isn't a comparison of engines. So we measure both
paths wherever an engine has both, and the report labels which is which.

Peak memory is read from the server container's cgroup, and the server is the
only thing in that container. The test client runs in a second container over a
private network. Otherwise the several GB of Python arrays holding the dataset
would be counted as database memory.

**Concurrency.** Queries per second and latency percentiles from 1 up to 32
concurrent clients. Single-client throughput tells you very little, and engines
cache their graphs in quite different ways, which only shows up when clients
compete for the same cache. We report scaling efficiency next to raw QPS,
because an engine that stops gaining throughput at 2 clients while its p99
latency gets 15 times worse is behaving very differently from one that scales.

**Filtered search.** The `WHERE tag < ?` query above, at several selectivities
down to 1% of rows passing.

Filtering changes what the correct answer is. The true top 10 among rows with
`tag < 10` is not the true top 10 overall, so for every selectivity we recompute
the ground truth by brute force over only the rows that pass the filter.
Scoring filtered results against the unfiltered ground truth that shipped with
the dataset gives every engine a recall near zero and means nothing. We made
exactly that mistake for a while, see below.

We also count how many queries came back with fewer than 10 rows. An engine can
run out of candidates before it finds 10 that satisfy the filter, and in one run
that happened on 81 of 200 queries. Those recall numbers are real but they're
not comparable to recall over full result sets, so the count is reported next to
them.

**Churn.** Recall and throughput before and after deleting and reinserting part
of the corpus. HNSW graphs are expected to degrade when rows are deleted,
because deletions leave the graph's links pointing at things that aren't there.
We haven't yet tested whether rebuilding the index recovers what's lost.

## How we keep it fair

Everything runs twice.

The **normalized** pass gives every engine the same CPU, memory and cache
budget, so a difference in the results is about the implementation rather than
about who was handed more RAM. The **tuned** pass lets each engine use the
settings its own documentation recommends. Tuned is more realistic and less
controlled, so it doesn't replace normalized. A result that holds in both passes
is about the engine.

Hardware is pinned. Each container gets an explicit set of CPU cores, and we
avoid two things that add noise:

- **SMT siblings.** SMT (Intel calls it Hyper-Threading) presents one physical
  core as two logical CPUs that share execution resources. Two threads on one
  physical core don't perform like two cores, so we only ever use one logical
  CPU per physical core.
- **Mixed core types.** Newer Intel desktop and laptop CPUs are hybrid: fast
  P-cores (performance) and slower E-cores (efficiency) in the same package. If
  the scheduler moves your benchmark between them, the variance swamps whatever
  you were trying to measure. We pin to one type.

Durability settings are relaxed identically everywhere, otherwise we'd be
comparing default fsync policies instead of vector search.

Some differences can't be equalised, so we write them down instead of pretending
they aren't there. A knob only one engine exposes doesn't get used in the
normalized pass, because that would hand one engine a tuning axis the others
don't have. An engine that requires a particular transaction isolation level
gets that level set for everybody. Defaults that are obviously placeholders (one
family of engines ships a 16 MiB graph cache, which is nothing) get sized from a
shared budget rather than left alone. All of these go in a "known asymmetries"
section of the report, above the results.

Every run writes a manifest recording the CPU model, its instruction set
extensions, engine versions and commits, image IDs, and the resource limits as
they actually resolved rather than as requested. The report generator refuses to
run without one.

## Reading the output

Read the validity section before you look at any chart. Our reports go
environment, then validity, then known asymmetries, then results. If a phase
failed, or an engine returned short result sets, or the CPU turned out to lack
the instruction set the engines wanted, that's in front of you before you've
formed an opinion.

The specific thing to watch for is the **silent full scan**. Engines will
quietly decide not to use the vector index and scan the table instead. A scan
returns exact results, slowly, so in the output it looks like high recall and
low throughput, which is indistinguishable from a carefully tuned index unless
you check the query plan.

This happens for ordinary reasons. One engine's optimizer costs the vector index
against a table scan and picks the scan once the LIMIT is above roughly a
quarter of the table, and we haven't found a setting that changes that. Another
falls back with no error and no warning if the distance operator in the query
doesn't match the one the index was built for, so a one-character difference
gets you a sequential scan and a sort.

So every driver runs EXPLAIN for each configuration and checks that the index
name appears in the plan.

```
WARNING: vector index NOT used (k=10, filtered=True). Plan: ...Seq Scan...
```

Anything that scanned goes into validity. This is the easiest way to produce
impressive vector benchmark numbers by accident.

For the recall and throughput results, the honest presentation is a curve rather
than a number. Sweep `ef_search`, plot recall on one axis and QPS on the other,
and take the upper-left edge of the points: the best throughput anyone achieved
at each level of accuracy. One engine beats another only if its curve sits above
the other's at the same recall. If the curves cross, the answer depends on how
accurate you need to be, and that's a real answer rather than a dodge.

Curves are awkward to read, because the eye compares shapes instead of heights
at one point. So we also plot the thing you probably want, which is throughput
at an accuracy you'd accept: bar charts at recall floors of 0.90, 0.95 and 0.99.

## About AVX-512

One hardware detail matters enough to call out. AVX-512 is a set of CPU
instructions that operate on 512 bits at a time, so a single instruction can
compute across 16 floats instead of one. Distance calculations are exactly that
kind of work, and several of these vector index implementations have hand-written
AVX-512 code paths.

The practical consequence is that vector search results from a machine without
AVX-512 do not transfer to a machine with it, and the gap isn't small or uniform
across engines. The CPU model and its instruction set flags are in every
manifest for that reason.

## Things that went wrong while we built this

Our first ingest numbers were garbage. The load path was doing one INSERT per
network round trip with autocommit on, and we measured 88 rows per second.
Batching 500 rows per transaction took the same engine to 373. None of that was
the engine's fault.

Filtered search and churn were being scored against the full-corpus ground truth
even when the run used a subset of rows, so every engine looked bad and it was
our bug. Ground truth is now cached per dataset, k, row count and selectivity,
and recomputed when any of those change.

Both resource passes shared one results directory, and the ANN runner skips
configurations that already have results. So the tuned pass was skipping
everything the normalized pass had already computed, and the tuned numbers were
mostly normalized numbers wearing a different label. Separate directories per
pass now.

Readiness probes lie. One engine's standard "are you accepting connections"
check returns success before the database it's supposed to create actually
exists, so our probe passed, the first query failed, and it looked like an
engine problem for longer than it should have. Probes now run a real query
against the real database.

The most recent one, on a 1536-dimension dataset. The ANN runner holds the whole
dataset in memory twice, once in the parent process and once again in a forked
worker, and the copies aren't shared. That's about 12 GB for a million
embeddings, on top of whatever the server is using, in a container we had sized
for the server alone. The kernel killed the worker. The runner doesn't check
worker exit codes, so it logged "Terminating 1 workers", exited successfully and
wrote no results, which looks exactly like a run that had nothing left to do. We
size that container from the dataset now.

## What we don't measure

Your hardware, as above.

Anything much past a million vectors. The datasets we use run from 60,000 to
1.2 million rows and mostly stay in cache. At 10 million the graphs stop fitting
in memory and the ordering could change completely.

Anything else running on the machine. A build we'd left running moved one
engine's numbers by a factor of two during development.

Your queries. Public datasets are standard, which is what makes them useful for
comparison and also what makes them not your workload.

## Adding a database

This is the part we cared about most, because the whole point was to avoid
rebuilding the apparatus for every new engine. Each one needs:

- a Dockerfile producing a runtime image and a test image from a pinned version
- a config declaring ports, credentials, and which server settings map to the
  normalized CPU and memory budget
- a module for the recall and throughput side
- a driver: create index, load, query, filtered query, index size, and the
  EXPLAIN check

The driver is about 200 lines. Nothing else in the harness changes, and adding
an engine doesn't change what any existing number means, so results can be
published before the list is finished.

If there's a database you want measured, that's the shape of the work.

## What's next

Results, published with the manifests and the raw per-configuration records so
they can be checked rather than taken on trust.

Everything is at
[github.com/EvgeniyPatlan/vector-bench](https://github.com/EvgeniyPatlan/vector-bench).
If we're measuring something wrong, tell us.
