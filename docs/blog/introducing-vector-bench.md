# How we benchmark vector search in databases

vector-bench is a tool for running vector search performance tests against
databases, without having to build the whole apparatus yourself every time. You
point it at the engines you care about and it builds them from pinned tags,
gives each one the same containers and the same hardware and the same datasets,
runs the same measurements against all of them and writes a report.

We're using it on MariaDB (MHNSW), AliSQL (VIDX) and PostgreSQL with pgvector,
because those are the ones we wanted to check first. The list isn't the point
and it will grow.

The results go in a separate post. This one is about the method, because a
vector benchmark is unusually easy to get wrong and most of the wrong answers
look fine in a chart.

Here's the short version of why. Take one MariaDB index, leave the data alone,
change one setting:

```
mhnsw_ef_search=10    3,678 queries/sec    95.93% of the correct answers
mhnsw_ef_search=800     409 queries/sec    99.87%
```

Nine times the throughput, same index, same machine, same query set. Both rows
are true. If someone tells you their database does 3,678 vector queries a
second, they have told you nothing, and they may not know it.

## Why we built it

Every database is adding vector search, and every one of them publishes numbers.
Checking those numbers yourself means building each engine, pinning versions,
setting up datasets and ground truth, writing a client per engine, keeping the
hardware identical, and then doing all of it again for the next engine you want
to look at. That's a week of work before you measure anything, and most of it is
the same work every time.

So the goal was to do that once. Adding an engine should be a config and a
driver, not a project. Running the tests should be one command:

```bash
./run-benchmark.sh build --march native   # engines, from pinned tags
./run-benchmark.sh fetch                  # datasets and ground truth
./run-benchmark.sh run --profile main     # measure, then report
```

ann-benchmarks already exists and it's good, and we use it for half of this, but
it was built to compare vector libraries and dedicated vector stores. The
questions we had were about databases. How long does it take to load a million
vectors into a running server? What happens when you add a WHERE clause? What
does the index look like after a few million deletes and inserts? Those only
matter if the vectors live in your database next to everything else, which is
the whole reason anyone uses these features.

One rule about what goes in: every engine we test has to be running HNSW. That
keeps the comparison about implementations rather than about one engine picking
IVF and another picking HNSW. If we add something that only does IVF it goes in
its own bucket.

## What we took from ann-benchmarks

The recall and throughput half of this is
[ann-benchmarks](https://github.com/erikbern/ann-benchmarks), Erik Bernhardsson's
suite, MIT licensed. We didn't write our own recall measurement.

What we add is three algorithm modules, one per engine, and a generated config
for each. We don't touch the runner, the metrics code or the dataset handling.
For a benchmark that distinction is the whole point: the recall numbers come out
of the same code that produced the published ANN results everyone already
compares against, and not out of something we wrote ourselves to measure
ourselves. The commit we ran against goes into every manifest.

The vendor checkout stays read-only. Each run clones it into a throwaway working
copy and drops our modules in there, so nothing we do can quietly change the
thing doing the measuring.

MariaDB's own big vector search benchmark ran on a fork of ann-benchmarks as
well, which is part of why we picked it. Same tool, so the results are
comparable in kind.

The other half, build cost and concurrency and filtered search and churn, is
ours. ann-benchmarks isn't built to ask those questions and there's no reason it
should be.

## What recall is

HNSW is approximate on purpose. Getting the true 10 nearest vectors out of a
million means computing a million distances for every query. The index walks a
graph instead, looks at a few thousand candidates and returns the best it saw.
Usually that's the right answer. Not always.

Recall@10 is how much of the correct top 10 you actually got back. Nine of the
ten rows right, that query scores 0.9. Average over the query set and you have
the number we report.

The correct answer comes from brute force, computed once and shipped with the
dataset. That's the only reason scoring an approximate index is possible at all.

Recall is something you configure. Every HNSW implementation has a knob for how
wide to search: `mhnsw_ef_search`,
`vidx_hnsw_ef_search`, `hnsw.ef_search`. Turn it up, the engine visits more
candidates, recall goes up and throughput goes down. That's the 3,678 against
409 above. Which means a QPS number without recall can't be checked, and a
recall number without QPS can't either, since you can always get recall 1.0 by
scanning the table.

## What the harness actually runs

The schema is the same shape everywhere: an id, a tag column we filter on, and
the vector. For MariaDB and AliSQL the index is part of the table:

```sql
CREATE TABLE t1 (
  id INT PRIMARY KEY,
  tag INT NOT NULL,
  v VECTOR(1536) NOT NULL,
  KEY tag_idx (tag),
  VECTOR INDEX v_idx (v) M=16 DISTANCE=cosine
) ENGINE=InnoDB
```

and the query is

```sql
SELECT id FROM t1 ORDER BY vec_distance_cosine(v, ?) LIMIT 10
```

For pgvector the index is a separate statement, which turns out to matter a lot
(see build cost below):

```sql
CREATE TABLE t1 (id int PRIMARY KEY, tag int NOT NULL, embedding vector(1536));
ALTER TABLE t1 ALTER COLUMN embedding SET STORAGE PLAIN;
CREATE INDEX t1_hnsw ON t1 USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 200);

SELECT id FROM t1 ORDER BY embedding <=> $1 LIMIT 10;
```

That `SET STORAGE PLAIN` is not cosmetic. Left alone, PostgreSQL will TOAST a
1536-dimension vector out of line and then every distance comparison pays a
detoast. pgvector's own docs warn about it. Forgetting it would have made
pgvector look slow for a reason that has nothing to do with pgvector.

## What we measure

**Recall against throughput**, swept across the search-width knob at several
values of M, one point per configuration. This part runs through ann-benchmarks,
as described above. k=10 throughout, and the queries are the dataset's own
held-out set, never rows taken from the corpus.

**Build cost**: wall time, ingest rate, index size on disk, peak RSS.

This is the one where it's easiest to publish nonsense. MHNSW and VIDX build the
graph on every INSERT and have no bulk mode. pgvector loads the rows first and
builds the graph afterwards in one pass. Those are not the same operation. In
one of our runs pgvector's bulk path did 5,692 rows/s and pgvector's own
incremental path did 312 on the identical data, an 18x spread that is entirely
about when the graph gets built and nothing to do with the other engines. So we
run pgvector both ways and label which is which, and if you only quote the bulk
number next to MariaDB's incremental one you get a comparison that means
nothing.

Peak memory comes from the server container's cgroup, and the server is alone in
that container. The harness runs in a second container over a private network.
If they shared, the several GB of NumPy holding the dataset would be charged to
the database.

**Concurrency**, 1 to 32 clients, QPS and latency percentiles. The three engines
cache their graphs in completely different ways (MariaDB one cache per table
object, AliSQL a shared cache plus a per-transaction one, pgvector no vector
cache at all with graph pages coming out of shared_buffers), and none of that is
visible until clients start competing for it. We report scaling efficiency next
to raw QPS, because an engine that stops gaining throughput at 2 clients while
p99 latency gets 15x worse is worth knowing about.

**Filtered search**, which is the case that justifies keeping vectors in a
database instead of somewhere else. We run several selectivities down to 1% of
rows passing.

Filtering changes what counts as correct. The true top 10 among rows with
`tag < 10` is not the true top 10 overall, so for every selectivity we recompute
exact ground truth by brute force over just the rows that pass. Scoring filtered
results against the shipped unfiltered ground truth gives every engine a recall
near zero and tells you nothing at all. We got this wrong ourselves for a while,
which is covered below.

We also count queries that came back with fewer than 10 rows, because an engine
can run out of candidates before finding 10 that pass the filter. In one run that
happened on 81 of 200 queries. That recall number isn't invalid, but it is not
comparable to one computed over full result sets, so the count sits next to it.
pgvector 0.8 has `hnsw.iterative_scan` for exactly this, and we record which mode
produced each measurement.

**Churn**: recall and throughput before and after deleting and reinserting part
of the corpus. HNSW is expected to degrade under deletes and we wanted to see
whether it does. If you're writing to this data continuously this matters more
than any of the static numbers.

## How we keep it fair

Everything runs twice.

The normalized pass gives every engine the same CPU, memory and cache budget, so
that a difference is about the implementation and not about who got more RAM.
The tuned pass lets each engine use what its own documentation recommends. That
one is more realistic and less controlled, which is why it doesn't replace the
first. A result that survives both passes is about the engine. One that flips
between them is interesting for a different reason.

CPUs are pinned explicitly, SMT siblings are excluded, and P-cores and E-cores
are never mixed, because that scheduling variance is larger than some of the
effects we're trying to measure. Durability is relaxed identically everywhere,
otherwise we'd be comparing default fsync policies.

Some things can't be equalised and we write those down instead of hiding them.
`ef_construction` only exists in pgvector; MariaDB rejects it outright with
`ERROR 1911 (HY000): Unknown option 'EF_CONSTRUCTION'`, so in the normalized
pass pgvector stays on its default rather than getting a tuning axis nobody else
has. AliSQL's VIDX is InnoDB-only and needs READ COMMITTED, so everything runs
READ COMMITTED. Both MySQL-family engines ship a 16 MiB default graph cache,
which is far too small for a real corpus, so we size both from one budget rather
than judge either on a value its vendor obviously expects you to change.

Each run writes a manifest with the CPU model and SIMD flags, engine source tags
and commits, image IDs, and the resource limits as they were actually resolved
rather than as requested. The report generator refuses to run without one.

## Reading the output

Look at the validity section before the charts. Our reports go environment,
validity, known asymmetries, then results, in that order, so a failed phase or a
missing AVX-512 or an engine that returned short result sets is in front of you
before you've formed an opinion.

The thing to watch for is the silent full scan. All three engines will quietly
decide not to use the vector index and scan the table instead, which returns
exact results slowly. In the output that looks like high recall and low
throughput, and you cannot tell it apart from a carefully tuned index without
looking at the plan.

This happens in practice. AliSQL costs the index against a scan and picks the
scan once LIMIT gets above roughly 25-28% of the table (on a 100-row table the
switch is somewhere between LIMIT 25 and LIMIT 40, and `ef_search` makes no
difference to it). pgvector falls back with no error and no warning when the
operator doesn't match the index's operator class, so `<->` against a
`vector_cosine_ops` index gives you a Seq Scan and a Sort and no indication that
anything is wrong.

So the drivers run EXPLAIN for every configuration and check the index name
appears in the plan:

```
[pgvector] WARNING: HNSW index NOT used (k=10, filtered=True). Plan: ...Seq Scan...
```

Anything that scanned goes into validity. This is the single easiest way to
produce impressive vector benchmark numbers, and a reason to be suspicious of
any benchmark that doesn't mention checking.

For the recall/throughput results themselves, the honest presentation is a curve
rather than a number: sweep the search width, plot recall against QPS, take the
upper-left edge. One engine beats another only if its curve is above the other's
at the same recall, and if they cross then the answer depends on how accurate
you need to be. Curves are also awkward to read, since the eye compares shapes
instead of heights at one point, so we also plot QPS at recall floors of 0.90,
0.95 and 0.99. Pick the accuracy you'd actually accept and read across.

## Things that went wrong while we built this

Worth listing, partly because they're the reason to trust the rest and partly
because anyone building something similar will hit them.

Our first ingest numbers were garbage. The ann-benchmarks path was inserting one
row per round trip with autocommit on, and we measured 88 rows/s on glove-100.
Batching 500 rows with explicit commits took the same engine to 373. If we'd
published the first number we'd have been benchmarking our own driver.

Filtered search and churn were being scored against the full-corpus ground truth
even when the run used a subset of rows. Every engine looked bad in a way that
was our fault. Ground truth is now cached per (dataset, k, row count,
selectivity) and recomputed when any of those change.

The two resource passes shared one ann-benchmarks results directory, and
ann-benchmarks skips configurations that already have results. So the tuned pass
was quietly skipping every point the normalized pass had already computed and
the tuned numbers were mostly normalized numbers. Separate directories per pass
now.

`pg_isready` returns success before the database in `POSTGRES_DB` exists. Our
readiness probe passed, the first query failed with "database ann does not
exist", and it looked like a pgvector problem. The probe now runs an actual
`SELECT 1` against the real database.

The most recent one, on a 1536-dimension corpus: ann-benchmarks holds the whole
dataset in memory twice, once in the parent process and once again in the forked
worker, and neither copy is shared. That's about 12 GB for a million OpenAI
embeddings, on top of whatever the server is using, in a container we had sized
for the server alone. The kernel killed the worker. ann-benchmarks doesn't check
worker exit codes, so it logged "Terminating 1 workers", exited 0 and wrote no
results, which looks exactly like a clean run that had nothing to do. We size
that container from the dataset now, and the harness says so when a phase
produces nothing.

## What we don't measure

Your hardware. MHNSW and VIDX both have AVX-512 distance kernels, so numbers
from a machine without AVX-512 don't carry over to one with it. The CPU and its
flags are in every manifest for that reason.

Anything much past a million vectors. Our datasets run 60k to 1M and mostly stay
in cache. At 10M the graphs go to disk and the ordering could change completely.

Anything else running on the machine. A build we'd left going moved one engine's
numbers by 2x during development. We check CPU, SIMD and cpuset but we can't see
your other workload.

Your query distribution. Standard datasets are standard, not representative of
what you're doing.

## Adding a database

This is the part we cared most about getting right, because the whole point was
to not rebuild the apparatus for every new engine. Each one needs four things:

- a Dockerfile producing a runtime image and a bench image from a pinned tag
- a config declaring ports, credentials, and which server flags map to the
  normalized memory and CPU budget
- an ann-benchmarks module for the recall/QPS side
- a driver implementing create index, load, query, filtered query, index size,
  and the EXPLAIN check that says whether the index was used

The driver is about 200 lines and the rest is configuration. Nothing else in the
harness changes, and adding an engine doesn't change what any existing number
means, so results can be published before the list is finished.

If you want a database measured that isn't in there yet, that's the shape of the
work, and we'd rather have the driver than the request.

## What's next

Million-vector runs are going now. Those results get published with the
manifests and the raw per-configuration records, so you can check them instead
of trusting them.

Everything is at
[github.com/EvgeniyPatlan/vector-bench](https://github.com/EvgeniyPatlan/vector-bench):
harness, drivers, Dockerfiles, docs. If we're measuring something wrong, or
being unfair to an engine you know better than we do, tell us. Running it
yourself and sending us the output is even better.
