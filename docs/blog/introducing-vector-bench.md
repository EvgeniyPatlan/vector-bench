# How we benchmark vector search in databases

vector-bench is a harness that compares vector search built into general-purpose
databases. Right now that's MariaDB (MHNSW), AliSQL (VIDX), and PostgreSQL with
pgvector. Same hardware, same containers, same datasets, same queries, and the
engines built from pinned tags with the same compiler flags.

We're publishing the method before the numbers. Results come in a later post,
and we'll be adding more databases as we go, so it's worth writing down what
we measure and how before anyone has to trust a chart.

Here's why that matters. Take one MariaDB index, don't touch the data, change
one setting. At `mhnsw_ef_search=10` it does 3,678 queries a second and finds
95.9% of the correct answers. At 800 it does 409 and finds 99.9%. Same index,
same machine, nine times the throughput between them, and both numbers are
true. Quote either one by itself and you've said nothing at all.

That's the whole problem. Vector search has a speed/accuracy dial on it, and
anyone can pick the setting that flatters them.

## Why we built it

We wanted to know how MariaDB's vector search compares to AliSQL's, and later
to pgvector. There are good ANN benchmarks already (we use ann-benchmarks for
part of this), but they're built to compare vector libraries and dedicated
vector stores. We had different questions. What does it cost to load a million
vectors into a running database? What happens when you put a WHERE clause on
the query? What happens after a month of deletes and inserts? Those are
database questions, and they're the reason you'd keep vectors in MySQL or
Postgres instead of running a separate system.

All three engines implement HNSW, which we did on purpose. With the algorithm
held constant, whatever we see is down to the implementation and not to
somebody picking a different index type.

## What recall is

Approximate nearest neighbour search is approximate on purpose. Finding the
actual nearest 10 vectors out of a million means computing a million distances,
every time. HNSW walks a graph instead, looks at a few thousand candidates and
returns what it found. Usually that's right. Not always.

Recall@10 is the share of the correct top 10 that the engine actually gave
back. If 9 of the 10 rows it returned belong in the true top 10, that query
scored 0.9. Average it over the query set and that's your recall.

The correct answer comes from brute force: every query compared against every
vector, computed once. The standard datasets ship with it, which is the only
reason you can score an approximate index at all.

Two things follow from this, and they shape everything else we do.

Recall isn't a property of an engine, it's a setting. Every HNSW
implementation has a knob for how wide to search (`mhnsw_ef_search`,
`vidx_hnsw_ef_search`, `hnsw.ef_search`). Turn it up and the engine looks at
more candidates, recall goes up, throughput goes down. That's the 3,678 against
409 from the top of this post.

So a single number can't be checked. QPS without recall is meaningless, and so
is recall without QPS, because perfect recall is free if you're willing to scan
the whole table. We never report one without the other.

## What we measure

Recall against throughput is the standard ANN measurement and we do it, but
it's the least interesting part. The rest is where databases differ from
libraries.

**Recall vs throughput.** Sweep the search-width knob against a fixed index, at
several values of the graph parameter M. Every point is one recall/QPS pair. We
run this through ann-benchmarks, so the numbers are comparable in kind to
published ANN results. k=10 everywhere, and the queries are the dataset's own
held-out query set, never rows sampled from the corpus.

**Build cost.** How long it takes to get the data indexed, the ingest rate, the
size of the index on disk, and peak memory.

This one needs care, because the engines don't do the same thing. MHNSW and
VIDX build the graph on every INSERT, there's no bulk mode. pgvector builds its
graph in one pass after the data is already loaded. Those are different
operations and comparing them gives you a number that means nothing: in one of
our runs pgvector's bulk path did 5,692 rows/s and its own incremental path did
312, an 18x gap that's entirely about when the graph gets built. So we measure
both paths wherever an engine has both, and the report says which is which.

Peak memory comes from the container's cgroup, with the server alone in that
container. The harness runs somewhere else. If we shared, the NumPy arrays
holding the dataset would get charged to the database and every memory number
would be wrong.

**Concurrency.** QPS and latency percentiles from 1 to 32 clients. Single-client
throughput tells you nothing about a server under load, and these three cache
their graphs in completely different ways (one cache per table object, a shared
cache plus a per-transaction one, or no vector cache at all with the graph
served out of the normal buffer pool). You only see that when clients start
competing. We report scaling efficiency next to raw QPS, because an engine that
stops scaling at 2 clients while p99 latency gets 15x worse is telling you
something the throughput column won't.

**Filtered search.** Vector search with a WHERE clause, which is supposedly the
reason to keep vectors in your database at all. We run it at several
selectivities, down to 1% of rows passing the filter.

Filtering changes what the right answer is. The true top 10 among rows where
`tag < 10` isn't the true top 10 overall, so we recompute exact ground truth by
brute force over the rows that pass, for every selectivity. Scoring filtered
results against unfiltered truth would give every engine near-zero recall and
tell you nothing.

We also count how many queries came back with fewer than 10 rows. An engine can
run out of candidates before it finds 10 that pass the filter, and in one run
that happened on 81 queries out of 200. Recall over short result sets isn't
wrong exactly, but it isn't comparable to recall over full ones unless the count
is sitting right next to it.

**Churn.** Recall and throughput before and after deleting and re-inserting a
chunk of the corpus. HNSW graphs are supposed to degrade under deletes, so we
check. If you're writing to this data continuously, this matters more than any
static number above it.

## How we keep it fair

We run everything twice.

The **normalized** pass gives every engine identical CPU, memory and cache
budgets. If something differs here, it's the implementation and not the
resources it was handed.

The **tuned** pass lets each engine use what its own documentation recommends.
More realistic, less controlled, which is why it doesn't replace the first one.
If a result holds in both passes it's about the engine. If it flips, that's
interesting on its own.

Server and client are in separate containers on a private network. CPUs are
pinned explicitly, SMT siblings excluded, P-cores and E-cores never mixed
(that variance is bigger than a lot of what we're trying to measure).
Durability is relaxed the same way everywhere, because leaving each engine on
its own defaults would compare fsync policies instead of vector search.

Some differences can't be normalized away, so we write them down instead of
pretending:

- `ef_construction` only exists in pgvector. MariaDB refuses it outright with
  `ERROR 1911 (HY000): Unknown option 'EF_CONSTRUCTION'`. In the normalized pass
  we pin pgvector to its default rather than give it a tuning knob the others
  don't have.
- AliSQL's VIDX only works on InnoDB and needs READ COMMITTED, so we put
  everything on READ COMMITTED and take isolation level out of the picture.
- Both MySQL-family engines ship a 16 MiB default graph cache, which is far too
  small for a real corpus. We set both from one budget instead of judging either
  on a number its vendor clearly expects you to change.

Every run writes a manifest with the CPU model and its SIMD flags, engine source
tags and commits, image IDs, and the resource limits as they actually got
resolved (not as we asked for them). The report generator won't produce a report
without one, and every report ends with the commands to reproduce that run.

## How to read the results

**Check the validity section first.** Our reports go environment, validity,
known asymmetries, and only then results. If a phase failed, if an engine
returned short result sets, if the CPU turned out to have no AVX-512, it's in
front of the charts and not in a footnote at the bottom.

**Watch for the full scan.** All three engines will quietly stop using the
vector index and scan the table instead. A scan gives exact results, slowly, so
in the output it looks like high recall and low throughput. You can't tell that
apart from a carefully tuned index unless you look at the query plan.

This is not hypothetical. AliSQL costs the index against a scan and picks the
scan once the LIMIT is somewhere above 25-28% of the table (on a 100-row table
the switch happens between LIMIT 25 and LIMIT 40, and `ef_search` doesn't affect
it at all). pgvector falls back silently, no error and no warning, if the
operator in your query doesn't match the index's operator class: use `<->`
against a `vector_cosine_ops` index and you get a Seq Scan and a Sort.

So every driver runs EXPLAIN for every configuration and records whether the
index was used. Anything that scanned shows up in validity. This is the easiest
way to produce impressive nonsense in a vector benchmark, and a good reason to
be suspicious of any that doesn't mention it.

**Read the curve, not the point.** Because recall trades against speed, the
honest way to show this is a curve: sweep the search width, plot recall against
QPS, take the upper-left edge. One engine is faster than another only if its
curve sits above the other's at the same recall. If the curves cross, the answer
depends on how accurate you need to be, and that's a real answer.

Curves are also annoying to read, because your eye compares shapes instead of
heights at one point. So we also plot the thing you actually want to know, which
is how fast it goes at an accuracy you'd accept: bars at recall floors of 0.90,
0.95 and 0.99. Pick your floor, read across.

## What we don't measure

Your hardware. MHNSW and VIDX both have AVX-512 distance kernels, so results
from a machine without AVX-512 don't transfer to one with it. That's why the CPU
and its SIMD flags are in every manifest.

Anything past about a million vectors. Our datasets are 60k to 1M and mostly fit
in cache. At 10M the graphs go to disk and the ordering could change completely.

Anything else running on the box. During development a build we'd left running
moved one engine's numbers by 2x. We check CPU, SIMD and cpuset, but we can't
see your other workload. Run this on a quiet machine.

Your queries. Standard datasets are standard, not representative of whatever
you're doing.

## Adding another database

The list will grow, so we made adding one cheap. Each engine needs four things:
a Dockerfile that builds a runtime image and a bench image from a pinned tag, a
config declaring ports and credentials and which server flags map to the
normalized budget, an ann-benchmarks module for the recall/QPS side, and a
driver for the ops side (create index, load, query, filtered query, index size,
and a check that answers whether the index actually got used).

Nothing else moves. Adding an engine doesn't change what any existing number
means, which is what lets us publish results before the list is finished.

## What's next

Results. The harness is running end to end on all three engines and we have
million-vector runs going now. Those get published with the manifests and the
raw per-configuration records, so you can check them rather than take our word
for it.

Everything is on GitHub at
[github.com/EvgeniyPatlan/vector-bench](https://github.com/EvgeniyPatlan/vector-bench):
the harness, the drivers, the Dockerfiles and the docs. If you think we're
measuring something wrong, or being unfair to an engine you know well, tell us.
Better yet, run it yourself and show us the output.
