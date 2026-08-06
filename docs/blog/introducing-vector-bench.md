# How we benchmark vector search in databases

vector-bench runs vector search performance tests against databases. You tell it
which engines you want, it builds them from pinned tags, puts each one in the
same container on the same cores with the same datasets, runs the same
measurements, and writes a report.

This post is the method. Which engines we're running it against, and how they
did, is the next one.

Here's why the method needs writing down at all. One HNSW index, same data, one
setting changed:

```
ef_search=10    3,678 queries/sec    95.93% of the correct answers
ef_search=800     409 queries/sec    99.87%
```

Nine times the throughput between those two rows and both are true. A vector
search QPS number on its own doesn't mean anything.

## Why we built it

Every database is shipping vector search now and all of them publish numbers.
Checking those numbers means building the engine, pinning the version, getting
datasets and ground truth, writing a client, keeping the hardware identical, and
then repeating all of it for the next engine. Most of that work is the same
every time.

We wanted to do it once. Adding an engine should be a config file and a driver.
Running the tests should be:

```bash
./run-benchmark.sh build --march native   # engines, from pinned tags
./run-benchmark.sh fetch                  # datasets and ground truth
./run-benchmark.sh run --profile main     # measure, then report
```

ann-benchmarks already exists and it's good. We use it for half of this. But it
was written to compare vector libraries and dedicated vector stores, and our
questions were about databases: how long does it take to load a million vectors
into a running server, what happens when you add a WHERE clause, what does the
index look like after a few million deletes and inserts.

One rule about what goes in. Everything we test has to be running HNSW, so that
differences come from the implementation rather than from one engine picking IVF
and another picking HNSW. Anything that only does IVF gets its own bucket.

## What we took from ann-benchmarks

The recall and throughput half is
[ann-benchmarks](https://github.com/erikbern/ann-benchmarks), Erik Bernhardsson's
suite, MIT licensed.

We add one algorithm module per engine and a generated config. We don't touch
the runner, the metrics code or the dataset handling. So recall is computed by
the same code behind the published ANN numbers, not by something we wrote to
measure ourselves. The commit goes in the manifest. The vendor checkout is
read-only and each run clones it into a throwaway copy before dropping our
modules in.

Build cost, concurrency, filtered search and churn are ours. ann-benchmarks
isn't built for those.

## What recall is

HNSW is approximate on purpose. Getting the true 10 nearest vectors out of a
million means computing a million distances every query. The index walks a graph
instead, looks at a few thousand candidates, returns the best it saw. Usually
that's right.

Recall@10 is how much of the correct top 10 you got back. Nine of ten rows
right, the query scores 0.9. Average over the query set and that's the number we
report.

The correct answer comes from brute force, computed once and shipped with the
dataset.

Recall is something you configure. Every HNSW implementation has a knob for how
wide to search, whatever it happens to call it. Turn it up, the engine visits
more candidates, recall goes up, throughput goes down. That's the 3,678 against
409 at the top. So a QPS number with no recall next to it can't be checked, and
neither can a recall number with no QPS, because recall 1.0 is always available
if you scan the table.

## What the harness runs

Same shape everywhere. A table with an id, a tag column to filter on, and the
vector. An HNSW index on the vector with a given M. A top-k query ordered by
distance, and the same query with a WHERE clause on the tag.

```sql
CREATE TABLE t1 (id, tag, v VECTOR(dim));
-- HNSW index on v, M as configured

SELECT id FROM t1 ORDER BY <distance>(v, ?) LIMIT 10;
SELECT id FROM t1 WHERE tag < ? ORDER BY <distance>(v, ?) LIMIT 10;
```

The dialect differs per engine and that's the driver's problem, not the
harness's. Some engines declare the index inside CREATE TABLE, some need a
separate CREATE INDEX, and that difference turns out to matter for build cost.

Each engine also has at least one setup detail that will quietly ruin your
numbers if you miss it. One example, since it's the kind of thing that's easy to
overlook: PostgreSQL will TOAST a large vector out of line unless the column is
set to `STORAGE PLAIN`, and then every distance comparison pays a detoast. The
docs warn about it. Miss it and you publish an engine looking slow for a reason
that has nothing to do with its vector search. Every driver carries a few of
these and they belong in the driver, not in the results.

## What we measure

Recall against throughput, swept across the search-width knob at several values
of M. k=10 throughout. Queries come from the dataset's held-out set, never from
the corpus.

Build cost: wall time, ingest rate, index size on disk, peak RSS. This is the
easiest place to publish nonsense, because engines don't all build the same way.
Some maintain the graph on every INSERT and have no bulk mode. Others load the
rows first and build the graph afterwards in one pass. Those are different
operations, and on one engine that supports both we measured an 18x difference
between its own two paths. Putting one engine's bulk number next to another's
incremental number isn't a comparison. So we measure both paths wherever an
engine has both, and the report labels which is which.

Peak memory comes from the server container's cgroup, and the server is alone in
that container. The harness runs in a second container over a private network.
Otherwise the several GB of NumPy holding the dataset gets charged to the
database.

Concurrency from 1 to 32 clients, QPS and latency percentiles. Engines cache
their graphs in very different ways, and none of that is visible until clients
start competing for the same cache. We report scaling efficiency next to raw
QPS. An engine that stops gaining throughput at 2 clients while p99 goes up 15x
is not the same as one that scales.

Filtered search at several selectivities, down to 1% of rows passing. Filtering
changes what counts as correct, because the true top 10 among rows with
`tag < 10` isn't the true top 10 overall. For every selectivity we recompute
exact ground truth by brute force over the rows that pass. Scoring against the
shipped unfiltered ground truth gives every engine a recall near zero. We did
exactly that for a while, see below.

We count queries that came back with fewer than 10 rows too. An engine can run
out of candidates before finding 10 that pass the filter, and in one run that
happened on 81 of 200 queries. The recall number is still real but it isn't
comparable to one over full result sets, so the count sits next to it. Some
engines have a setting for this behaviour and we record which mode was used.

Churn: recall and throughput before and after deleting and reinserting part of
the corpus. HNSW is supposed to degrade under deletes. We haven't tested whether
a rebuild recovers what's lost, which is the obvious next question.

## How we keep it fair

Everything runs twice.

The normalized pass gives every engine the same CPU, memory and cache budget.
The tuned pass lets each one use what its documentation recommends. The tuned
pass is more realistic and less controlled, so it doesn't replace the first. A
result that holds in both is about the engine.

CPUs are pinned, SMT siblings excluded, P-cores and E-cores never mixed, because
that scheduling variance is bigger than some of what we're measuring. Durability
is relaxed identically everywhere.

Some things can't be equalised, so we write them down instead of pretending. A
knob that only one engine exposes doesn't get used in the normalized pass, since
that would hand one engine a tuning axis the others don't have. An engine that
requires a particular isolation level sets that level for everybody. Defaults
that are obviously placeholders get sized from one budget rather than left
alone, because judging an engine on a 16 MiB cache its vendor expects you to
change isn't measuring anything. Each of these goes in a "known asymmetries"
section of the report, above the results.

Every run writes a manifest: CPU model and SIMD flags, engine source tags and
commits, image IDs, and the resource limits as they resolved rather than as
requested. The report generator won't run without one.

## Reading the output

Read the validity section before the charts. The reports go environment,
validity, known asymmetries, then results, so a failed phase or a missing
AVX-512 or an engine returning short result sets is in front of you early.

The thing to watch for is the silent full scan. Engines will quietly stop using
the vector index and scan the table instead, which returns exact results slowly.
That shows up as high recall and low throughput, and it looks identical to a
conservatively tuned index.

It happens in practice and for different reasons. One engine costs the index
against a scan and takes the scan once the LIMIT is above a quarter or so of the
table, and we haven't found a setting that moves it. Another falls back with no
error and no warning if the operator in the query doesn't match the operator
class the index was built with, so a one-character mistake gets you a sequential
scan and a sort.

The drivers run EXPLAIN for every configuration and check the index name is in
the plan.

```
WARNING: vector index NOT used (k=10, filtered=True). Plan: ...Seq Scan...
```

Anything that scanned goes in validity. This is the easiest way to produce
impressive vector benchmark numbers by accident.

For recall against throughput, the useful presentation is a curve: sweep the
search width, plot recall against QPS, take the upper-left edge. One engine
beats another only if its curve sits above the other's at the same recall, and
crossing curves mean it depends on how accurate you need to be. Curves are also
awkward, because the eye compares shapes instead of heights at one x, so we plot
QPS at recall floors of 0.90, 0.95 and 0.99 as well.

## Things that went wrong while we built this

Our first ingest numbers were garbage. The load path was doing one INSERT per
round trip with autocommit on, and we measured 88 rows/s. Batching 500 rows with
explicit commits took the same engine to 373. Nothing about that was the
engine's fault.

Filtered search and churn were scored against the full-corpus ground truth even
when the run used a subset of rows, so every engine looked bad and it was our
bug. Ground truth is cached per dataset, k, row count and selectivity now.

Both resource passes shared one ann-benchmarks results directory, and
ann-benchmarks skips configurations that already have results. The tuned pass
was skipping everything the normalized pass had computed, so the tuned numbers
were mostly normalized numbers. Separate directories per pass.

Readiness probes lie. One engine's standard "are you up" check returns success
before the database it's supposed to create exists, so our probe passed and the
first query failed, and it looked like an engine problem for longer than it
should have. Probes run a real query against the real database now.

Most recent one, on a 1536-dimension corpus. ann-benchmarks holds the whole
dataset in memory twice, once in the parent and once in the forked worker, and
the copies aren't shared. That's about 12 GB for a million embeddings, on top of
the server, in a container we'd sized for the server alone. The kernel killed
the worker. ann-benchmarks doesn't check worker exit codes, so it logged
"Terminating 1 workers", exited 0, and wrote no results, which looks the same as
a run that had nothing to do. We size that container from the dataset now.

## What we don't measure

Your hardware. AVX-512 distance kernels are common in these implementations, so
numbers from a machine without AVX-512 don't carry to one with it. The CPU and
its flags are in every manifest.

Anything much past a million vectors. The datasets run 60k to 1.2M and mostly
stay in cache. At 10M the graphs go to disk and the ordering could change.

Anything else running on the box. A build we'd left going moved one engine's
numbers by 2x during development.

Your queries. Standard datasets are standard.

## Adding a database

This is the part we cared about most, since the whole point was to not rebuild
the apparatus for each engine. Each one needs:

- a Dockerfile producing a runtime image and a bench image from a pinned tag
- a config declaring ports, credentials, and which server flags map to the
  normalized CPU and memory budget
- an ann-benchmarks module for the recall/QPS side
- a driver: create index, load, query, filtered query, index size, and the
  EXPLAIN check

The driver is about 200 lines. Nothing else in the harness changes, and adding
an engine doesn't change what any existing number means, so results can go out
before the list is finished.

If there's a database you want measured, that's the shape of the work.

## What's next

Results, on the engines we've been running against, published with the manifests
and the raw per-configuration records so they can be checked.

Everything is at
[github.com/EvgeniyPatlan/vector-bench](https://github.com/EvgeniyPatlan/vector-bench):
harness, drivers, Dockerfiles, docs. If we're measuring something wrong, tell us.
