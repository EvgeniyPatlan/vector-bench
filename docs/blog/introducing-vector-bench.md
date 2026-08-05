# Your vector index is fast. At what recall?

Here are three databases answering the same question — find the 10 nearest
vectors — on the same machine, same data, same index type, same compiler flags:

| Engine | recall@10 | queries/sec |
| --- | ---: | ---: |
| MariaDB 11.8.8 | 0.9971 | 2,211 |
| AliSQL 8.0.44-2 | 0.9978 | 1,384 |
| PostgreSQL 17 + pgvector 0.8.6 | 0.9720 | 2,271 |

Read the QPS column alone and pgvector wins. Read both columns and it doesn't:
it is returning a measurably worse answer set. That single pairing is most of
what this post is about.

The numbers come from `vector-bench`, a harness that compares built-in vector
search across MariaDB (MHNSW), AliSQL (VIDX) and PostgreSQL (pgvector). All
three implement HNSW, so the comparison isolates *implementation quality*
rather than algorithm choice. Everything below was measured on a dual-socket
Xeon Gold 6230 with AVX-512, all three engines compiled `--march=native`.

---

## What recall actually is

Approximate nearest neighbour search is approximate on purpose. Finding the
true 10 nearest vectors out of a million means computing a million distances.
An HNSW index instead walks a graph, visits a few thousand candidates, and
returns what it found. It is usually right. It is not always right.

**Recall@k is the fraction of the true top-k that the engine actually
returned.** Ask for 10 neighbours, get 9 that belong in the true top 10, and
that query scored 0.9. Average over the whole query set and you have
recall@10.

The "true" answer comes from brute force — every query compared against every
vector, computed once and shipped with the dataset. That is why a benchmark can
score an approximate index at all.

Two consequences worth internalising:

**Recall is a dial, not a property.** Every HNSW implementation exposes a
search-width knob — `mhnsw_ef_search`, `vidx_hnsw_ef_search`, `hnsw.ef_search`.
Turn it up, the engine visits more candidates, recall rises, throughput falls.
The same MariaDB index measured across that dial:

| ef_search | recall@10 | QPS |
| ---: | ---: | ---: |
| 10 | 0.9593 | 3,678 |
| 40 | 0.9921 | 2,431 |
| 120 | 0.9969 | 1,445 |
| 800 | 0.9987 | 409 |

Nine times the throughput at the top of that table, and four percentage points
of recall between the ends. "MariaDB does 3,678 QPS" and "MariaDB does 409 QPS"
are both true statements about the same index.

**A single number is therefore meaningless.** Any vector search QPS figure
without its recall is unfalsifiable. So is any recall figure without its
throughput — perfect recall is free if you're willing to scan the table.

---

## How to read the results

### The Pareto frontier

Because recall trades against speed, the honest presentation is a curve, not a
bar. Sweep the search width, plot recall on x and QPS on y, and take the
upper-left envelope: for each recall level, the best throughput anyone achieved
at it.

An engine is faster than another **only if its frontier sits above the other's
at the same recall.** Curves that cross mean the answer depends on your
accuracy target, and that is a real result rather than a cop-out.

The reports also include the comparison an operator actually makes — *how fast
is it at an accuracy I can accept* — as a bar chart at recall floors of 0.90,
0.95 and 0.99. That is hard to read off a curve, because the eye compares
curve shapes rather than their heights at one x.

### The trap that invalidates everything

All three engines can silently decline to use the vector index and fall back to
a full table scan. A scan returns **exact** results, slowly. In the output that
looks like *high recall, low throughput* — indistinguishable from a
conservatively-tuned index unless you check the query plan.

It is not theoretical. Measured behaviours, all reproducible:

- **AliSQL** costs the vector index against a table scan and picks the scan
  when the `LIMIT` exceeds roughly 25–28% of rows. On a 100-row table the
  crossover sits between `LIMIT 25` and `LIMIT 40`; on 1,000 rows, between 250
  and 290. It does not depend on `ef_search` at all.
- **pgvector** falls back with no error and no warning if the query operator
  does not match the index's operator class — `<->` against a
  `vector_cosine_ops` index gives you `Seq Scan` and a `Sort`.

So every driver in the harness runs `EXPLAIN` for each configuration and
records whether the index was used. Any measurement where it was not appears in
the report's **Validity** section, above the charts, not in a footnote.

### Read Validity first

The report's section order is deliberate: environment, then validity, then
known asymmetries, then results. If a run lost an engine to a failure, or an
engine returned fewer than k results, or the CPU lacks AVX-512, that appears
before any chart. A number whose caveats arrive after it has already been read
is a number that has already misled someone.

---

## What the benchmark measures

Recall versus throughput is the standard ANN metric and it is table stakes. The
more interesting dimensions are the ones that separate *a database* from *an
ANN library*.

### Index build cost

The largest and most consistent gap in everything measured so far.

| Engine | mode | rows/s | build | index size | peak RSS |
| --- | --- | ---: | ---: | ---: | ---: |
| pgvector | bulk build after load | **5,692** | 18.0 s | 78 MB | 0.75 GB |
| MariaDB | incremental on INSERT | 747 | 26.8 s | 48 MB | **0.54 GB** |
| pgvector | incremental | 312 | 64.1 s | 78 MB | 0.75 GB |
| AliSQL | incremental on INSERT | 146 | 136.9 s | 48 MB | 1.34 GB |

*60,000 vectors × 784 dimensions, M=16.*

Two findings and one methodological trap.

**MariaDB ingests about 5× faster than AliSQL** on identical work — same
schema, same client library, same M, same storage engine — using 2.5× less
memory. That gap has reproduced in every run.

**pgvector's bulk build is 18× its own incremental build.** Which brings the
trap: pgvector constructs its HNSW graph as one operation *after* the data
loads, while MHNSW and VIDX maintain theirs on every INSERT. Comparing
pgvector's 5,692 rows/s against MariaDB's 747 compares two different
operations. The harness therefore measures pgvector both ways, because
"pgvector builds faster" and "pgvector builds faster when allowed to bulk
build" are different claims and only one of them is like-for-like.

This is not an academic distinction. On a million-vector corpus MariaDB takes
about 2.2 hours to load and AliSQL about 6; pgvector takes 7 minutes. If you
reindex regularly, that difference dominates everything on the query side.

### Concurrency

Single-client QPS says nothing about behaviour under load, and the three
engines cache their graphs very differently: MariaDB keeps one cache per
`TABLE_SHARE`, AliSQL keeps a shared cache plus a per-transaction one,
pgvector has no vector-specific cache at all and serves graph pages through
`shared_buffers`.

On a million-vector corpus, 40 physical cores:

| clients | QPS | p99 |
| ---: | ---: | ---: |
| 1 | 184 | 52 ms |
| 2 | 326 | 62 ms |
| 4 | 324 | 58 ms |
| 8 | 324 | 158 ms |
| 16 | 324 | 371 ms |
| 32 | 325 | **785 ms** |

MariaDB saturates at two clients. Four, eight, sixteen and thirty-two all land
within 1% of each other, while p99 degrades 15×. Additional concurrency buys
nothing and costs latency. pgvector, on the same dimension, scales at 0.95
efficiency to four clients.

You cannot see any of that in a recall/QPS curve.

### Filtered search

Vector search with a `WHERE` clause — the case that supposedly justifies
keeping vectors in your database rather than a dedicated store.

| Engine | recall@10 | QPS | queries returning < k |
| --- | ---: | ---: | ---: |
| MariaDB | 0.9990 | 183 | 0 |
| AliSQL | 0.9990 | 172 | 0 |
| pgvector | 0.8875 | 858 | **81 / 200** |

*10% of rows passing the filter.*

pgvector looks 4.7× faster, and part of that is work it didn't do: on 81 of 200
queries it returned **fewer than 10 results**, having exhausted its candidate
list before finding 10 rows that passed the filter. That is legitimate engine
behaviour — pgvector 0.8's `iterative_scan` exists to address it and was
deliberately left at its default here — but a recall number computed over short
result sets needs the count beside it, which is why the harness reports both.

Filtering also changes what "correct" means. The true top-10 among rows where
`tag < 10` is not the true top-10 overall, so the harness recomputes exact
ground truth by brute force over the qualifying subset. Scoring filtered
results against unfiltered truth would report near-zero recall for every engine
and teach you nothing.

At 1% selectivity on a million rows, MariaDB dropped to **0.9 QPS** with a p99
of 4.3 seconds — against 326 QPS unfiltered. A 360× penalty, with the index
confirmed in use.

### Churn

Recall and speed after deleting and re-inserting rows. HNSW graphs are widely
expected to degrade under deletion, so this measures whether they do.

Recall did not move — 1.0000 before and after 10% churn on all three engines.
Throughput did:

| Engine | before | after 10% churn |
| --- | ---: | ---: |
| MariaDB | 1,411 | 806 (−43%) |
| AliSQL | 1,019 | 661 (−35%) |
| pgvector | 930 | 933 (−0%) |

For a workload with ongoing writes that matters more than any static number
above it. On the million-vector corpus, recall actually *rose* slightly after
churn (0.8385 → 0.8630) — re-inserting rows rebuilt parts of the graph better
than the original incremental construction had.

---

## What it does not measure, and what would change the answers

**Hardware.** MHNSW and VIDX both document AVX-512 distance kernels. Results
from a machine without AVX-512 do not transfer to one with it, and every report
records the CPU model and its SIMD flags for that reason.

**Scale.** These corpora are 60k–1M vectors and largely cache-resident. At 10M
the graphs go to disk and the ranking may invert entirely.

**Configuration asymmetries no setting removes.** `ef_construction` is exposed
only by pgvector — MariaDB rejects it outright with `ERROR 1911 (HY000):
Unknown option 'EF_CONSTRUCTION'` — so the fair pass pins pgvector to its
default rather than handing it a tuning axis the others lack. AliSQL's VIDX is
InnoDB-only and requires READ COMMITTED. Both MySQL-family engines ship a
16 MiB default graph cache, far too small for any real corpus, so the harness
sets both from one budget rather than judging either on a value its vendor
plainly intended you to change.

**Anything running alongside it.** A competing build distorted one engine's
numbers by 2× during development. The harness reports CPU, SIMD and cpuset
problems; a concurrent workload is invisible to it.

---

## The short version

If you take four things from this:

1. **A QPS number without a recall number is not a result.** Nor the reverse.
2. **Check the query plan.** All three engines can silently fall back to a full
   scan, and it looks like accuracy, not failure.
3. **Build cost and query cost are separate questions** and can point in
   opposite directions. pgvector builds 18× faster than it does incrementally;
   MariaDB queries fastest single-client and stops scaling at two.
4. **Match the operation before comparing the numbers.** Bulk build against
   incremental build is not a comparison, however tempting the ratio looks.

The harness, the raw records and the methodology are at
[github.com/EvgeniyPatlan/vector-bench](https://github.com/EvgeniyPatlan/vector-bench).
Every run writes a manifest with CPU flags, engine commits, image digests and
resolved resource limits; the report generator refuses to produce a report
without one. A result without its environment is not a result.
