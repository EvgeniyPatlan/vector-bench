# How we benchmark vector search in databases

vector-bench runs vector search performance tests against databases. You tell it
which engines you want, it builds them from pinned tags, puts each one in the
same container on the same cores with the same datasets, runs the same
measurements, and writes a report.

We're running it against MariaDB 11.8.8, AliSQL 8.0.44-2, and PostgreSQL 17 with
pgvector 0.8.6. Those are the ones we wanted to look at first.

Results are a separate post. This one is the method.

Here is why the method needs writing down. One MariaDB index, same data, one
setting changed:

```
mhnsw_ef_search=10    3,678 queries/sec    95.93% of the correct answers
mhnsw_ef_search=800     409 queries/sec    99.87%
```

Nine times the throughput between those two rows, and both are true. A vector
search QPS number on its own doesn't mean anything.

## Why we built it

Every database is shipping vector search now and all of them publish numbers.
Checking those numbers means building the engine, pinning the version, getting
datasets and ground truth, writing a client, keeping the hardware identical, and
then repeating it for the next engine. Most of that work is the same every time.

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

Everything we test has to be running HNSW. That's a rule about what goes in, not
an observation about the three engines that are in there now. If we add
something that only does IVF it gets its own bucket.

## What we took from ann-benchmarks

The recall and throughput half is
[ann-benchmarks](https://github.com/erikbern/ann-benchmarks), Erik Bernhardsson's
suite, MIT licensed.

We add three algorithm modules and a generated config for each. We don't touch
the runner, the metrics code or the dataset handling. So recall is computed by
the same code behind the published ANN numbers, not by something we wrote to
measure ourselves. The commit goes in the manifest. The vendor checkout is
read-only and each run clones it into a throwaway copy before dropping our
modules in.

MariaDB's own big vector search benchmark ran on a fork of ann-benchmarks too,
which is part of why we picked it.

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
wide to search: `mhnsw_ef_search`, `vidx_hnsw_ef_search`, `hnsw.ef_search`. Turn
it up, the engine visits more candidates, recall goes up, throughput goes down.
That's the 3,678 against 409 at the top. So a QPS number with no recall next to
it can't be checked, and neither can a recall number with no QPS, because recall
1.0 is always available if you scan the table.

## What the harness runs

Same schema everywhere: an id, a tag column to filter on, the vector. For
MariaDB and AliSQL the index is declared in the table.

```sql
CREATE TABLE t1 (
  id INT PRIMARY KEY,
  tag INT NOT NULL,
  v VECTOR(1536) NOT NULL,
  KEY tag_idx (tag),
  VECTOR INDEX v_idx (v) M=16 DISTANCE=cosine
) ENGINE=InnoDB
```

```sql
SELECT id FROM t1 ORDER BY vec_distance_cosine(v, ?) LIMIT 10
```

For pgvector the index is a separate statement, which matters for build cost
later.

```sql
CREATE TABLE t1 (id int PRIMARY KEY, tag int NOT NULL, embedding vector(1536));
ALTER TABLE t1 ALTER COLUMN embedding SET STORAGE PLAIN;
CREATE INDEX t1_hnsw ON t1 USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 200);

SELECT id FROM t1 ORDER BY embedding <=> $1 LIMIT 10;
```

`SET STORAGE PLAIN` is not cosmetic. Without it PostgreSQL TOASTs a
1536-dimension vector out of line and every distance comparison pays a detoast.
pgvector's docs warn about this. We would have published pgvector looking slow
for a reason that has nothing to do with pgvector.

## What we measure

Recall against throughput, swept across the search-width knob at several values
of M. k=10 throughout. Queries come from the dataset's held-out set, never from
the corpus.

Build cost: wall time, ingest rate, index size on disk, peak RSS. This is the
easiest place to publish nonsense. MHNSW and VIDX build the graph on every
INSERT and have no bulk mode. pgvector loads the rows and builds the graph
afterwards in one pass. In one run pgvector's bulk path did 5,692 rows/s and its
own incremental path did 312 on the same data. That 18x is about when the graph
gets built. We run pgvector both ways and label them, because putting the bulk
number next to MariaDB's incremental one isn't a comparison.

Peak memory comes from the server container's cgroup, and the server is alone in
that container. The harness runs in a second container over a private network.
Otherwise the several GB of NumPy holding the dataset gets charged to the
database.

Concurrency from 1 to 32 clients, QPS and latency percentiles. The three engines
cache their graphs differently: MariaDB one cache per table object, AliSQL a
shared cache plus a per-transaction one, pgvector nothing vector-specific with
graph pages coming out of shared_buffers. We report scaling efficiency next to
raw QPS. An engine that stops gaining throughput at 2 clients while p99 goes up
15x is not the same as one that scales.

Filtered search at several selectivities, down to 1% of rows passing. Filtering
changes what counts as correct, because the true top 10 among rows with
`tag < 10` isn't the true top 10 overall. For every selectivity we recompute
exact ground truth by brute force over the rows that pass. Scoring against the
shipped unfiltered ground truth gives every engine a recall near zero. We did
exactly that for a while, see below.

We count queries that came back with fewer than 10 rows too. An engine can run
out of candidates before finding 10 that pass the filter, and in one run that
happened on 81 of 200 queries. The recall number is still real but it isn't
comparable to one over full result sets, so the count sits next to it. pgvector
0.8 has `hnsw.iterative_scan` for this and we record which mode was used.

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

Some things can't be equalised, so we write them down. `ef_construction` only
exists in pgvector; MariaDB rejects it with `ERROR 1911 (HY000): Unknown option
'EF_CONSTRUCTION'`, so in the normalized pass pgvector stays on its default
instead of getting a tuning axis nobody else has. AliSQL's VIDX is InnoDB-only
and needs READ COMMITTED, so everything runs READ COMMITTED. Both MySQL-family
engines ship a 16 MiB default graph cache, which is far too small for a real
corpus, so we set both from one budget.

Every run writes a manifest: CPU model and SIMD flags, engine source tags and
commits, image IDs, and the resource limits as they resolved rather than as
requested. The report generator won't run without one.

Our own box is a dual-socket Xeon Gold 6230, 40 physical cores, 187 GB RAM, with
AVX-512, and the engines are built `--march=native` on it.

## Reading the output

Read the validity section before the charts. The reports go environment,
validity, known asymmetries, then results, so a failed phase or a missing
AVX-512 or an engine returning short result sets is in front of you early.

The thing to watch for is the silent full scan. All three engines will quietly
stop using the vector index and scan instead, which returns exact results
slowly. That shows up as high recall and low throughput, and it looks identical
to a conservatively tuned index.

It happens. AliSQL costs the index against a scan and takes the scan once LIMIT
is above roughly 25-28% of the table. On a 100-row table the switch is somewhere
between LIMIT 25 and LIMIT 40. `ef_search` makes no difference to it, and we
haven't found a setting that moves it. pgvector falls back with no error and no
warning when the operator doesn't match the index's operator class, so `<->`
against a `vector_cosine_ops` index gets you a Seq Scan and a Sort.

The drivers run EXPLAIN for every configuration and check the index name is in
the plan.

```
[pgvector] WARNING: HNSW index NOT used (k=10, filtered=True). Plan: ...Seq Scan...
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

Our first ingest numbers were garbage. The ann-benchmarks path was doing one
INSERT per round trip with autocommit on, and we measured 88 rows/s on
glove-100. Batching 500 rows with explicit commits took the same engine to 373.

Filtered search and churn were scored against the full-corpus ground truth even
when the run used a subset of rows, so every engine looked bad and it was our
bug. Ground truth is cached per dataset, k, row count and selectivity now.

Both resource passes shared one ann-benchmarks results directory, and
ann-benchmarks skips configurations that already have results. The tuned pass
was skipping everything the normalized pass had computed, so the tuned numbers
were mostly normalized numbers. Separate directories per pass.

`pg_isready` returns success before the database named in `POSTGRES_DB` exists.
The readiness probe passed, the first query failed with "database ann does not
exist", and it looked like a pgvector problem for longer than it should have.
The probe runs a real `SELECT 1` now.

Most recent one, on the 1536-dimension corpus. ann-benchmarks holds the whole
dataset in memory twice, once in the parent and once in the forked worker, and
the copies aren't shared. That's about 12 GB for a million OpenAI embeddings, on
top of the server, in a container we'd sized for the server alone. The kernel
killed the worker. ann-benchmarks doesn't check worker exit codes, so it logged
"Terminating 1 workers", exited 0, and wrote no results, which looks the same as
a run that had nothing to do. We size that container from the dataset now.

## What we don't measure

Your hardware. MHNSW and VIDX both have AVX-512 distance kernels, so numbers
from a machine without AVX-512 don't carry to one with it.

Anything much past a million vectors. The datasets we use run 60k to 1.2M
(fashion-mnist-784, glove-100, sift-128, gist-960, dbpedia-openai-1000k) and
mostly stay in cache. At 10M the graphs go to disk and the ordering could change.

Anything else running on the box. A build we'd left going moved one engine's
numbers by 2x during development.

Your queries. Standard datasets are standard.

There are also things we can measure but can't yet explain. On glove-100
MariaDB loads at about 150 rows/s and AliSQL at about 55, same schema, same
client, same M, so 2.2 hours against 6. We haven't profiled either one to find
out why.

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

Million-vector runs are going now. Those results get published with the
manifests and the raw per-configuration records so they can be checked.

Everything is at
[github.com/EvgeniyPatlan/vector-bench](https://github.com/EvgeniyPatlan/vector-bench):
harness, drivers, Dockerfiles, docs. If we're measuring something wrong, or
being unfair to an engine you know better than we do, tell us.
