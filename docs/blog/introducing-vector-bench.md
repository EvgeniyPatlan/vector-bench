# What we measure, and why one number is never the answer

Same index. Same machine. Same data. Same queries. One configuration knob
changed:

| `mhnsw_ef_search` | recall@10 | queries/sec |
| ---: | ---: | ---: |
| 10 | 0.9593 | 3,678 |
| 800 | 0.9987 | 409 |

Both rows are true statements about the same index. Nine times the throughput
separates them. Anyone quoting either number on its own is telling you nothing —
and both are available to whoever is doing the quoting.

That is the problem `vector-bench` exists to solve. This post is about the
method: what gets measured, how, and how to read the output. The results come
later, and they will cover more databases than they do today.

---

## What this is

A harness that compares vector search built into general-purpose databases. Not
a comparison of dedicated vector stores, and not a comparison of ANN libraries —
the question is what you get when the vectors live next to the rest of your data.

Three engines are wired up today: MariaDB (MHNSW), AliSQL (VIDX) and PostgreSQL
with pgvector. All three implement HNSW, which is deliberate: with the algorithm
held constant, differences belong to the implementations rather than to the
choice of index. More engines will be added, and the method is built to absorb
them without changing what the existing numbers mean.

Everything runs in containers built from pinned upstream tags, so a run is
reproducible on hardware other than ours and the binaries are ours rather than a
vendor's convenience build.

---

## What recall actually is

Approximate nearest neighbour search is approximate on purpose. Finding the true
10 nearest vectors in a million means computing a million distances. An HNSW
index instead walks a graph, visits a few thousand candidates, and returns what
it found. It is usually right. It is not always right.

**Recall@k is the fraction of the true top-k that the engine actually
returned.** Ask for 10 neighbours, get 9 that belong in the true top 10, and
that query scored 0.9. Average over the query set and you have recall@10, which
is what we report throughout.

The "true" answer comes from brute force — every query compared against every
vector, computed once. Standard datasets ship with it. When a workload changes
what "correct" means, we recompute it rather than reuse it; more on that below.

Two consequences drive the whole design:

**Recall is a dial, not a property.** Every HNSW implementation exposes a
search-width knob — `mhnsw_ef_search`, `vidx_hnsw_ef_search`, `hnsw.ef_search`.
Turn it up, the engine visits more candidates, recall rises, throughput falls.
That is the table at the top of this post. An engine does not *have* a QPS
figure; it has a curve, and where you sit on it is your choice.

**So a single number is unfalsifiable.** A QPS figure without its recall cannot
be checked. Neither can a recall figure without its throughput — perfect recall
is free if you are willing to scan the table. We never report one without the
other, and neither should anyone else.

---

## What exactly gets measured

Recall against throughput is the standard ANN metric and it is table stakes. The
dimensions that follow are the ones that separate *a database* from *an ANN
library*, and they are where most of the design effort went.

### 1. Recall vs throughput

Sweep the search-width knob across its useful range against a fixed index, at
several values of the graph parameter `M`. Each point is one (recall, QPS) pair.
This runs through ann-benchmarks — the same tooling the field already uses — so
the numbers are comparable in kind to published ANN results.

`k=10` throughout. Query sets are the datasets' own held-out queries, never
sampled from the corpus.

### 2. Index build cost

Time, resulting index size, ingest rate and peak RSS to get the corpus indexed.

This is where the most care is needed, because the engines do not do the same
thing. MHNSW and VIDX maintain the HNSW graph on **every INSERT**; there is no
bulk build. pgvector constructs its graph in one pass **after** the data lands.
Comparing an incremental rate against a bulk rate is not a comparison, however
tempting the ratio looks — in one run pgvector's bulk path measured 5,692 rows/s
against 312 rows/s for its own incremental path, an 18× spread that belongs
entirely to build strategy.

So both paths are measured wherever an engine supports both, and the report
labels which is which. "Engine X builds faster" and "engine X builds faster when
allowed to bulk build" are different claims.

Peak memory comes from the container's own cgroup (`memory.peak`), with the
server running alone in that container. If the harness shared it, the NumPy
arrays holding the dataset would be charged to the engine and every memory
number would be wrong.

### 3. Concurrency

QPS and latency percentiles from 1 to 32 concurrent clients.

Single-client throughput says nothing about behaviour under load, and these
engines cache their graphs very differently — one cache per table object, a
shared cache plus a per-transaction one, or no vector-specific cache at all with
graph pages served from the general buffer pool. Those choices only become
visible when clients contend.

We report scaling efficiency alongside raw QPS, because an engine that stops
scaling at two clients while p99 latency degrades 15× is telling you something a
throughput column alone will not.

### 4. Filtered search

Vector search with a `WHERE` clause — the case that supposedly justifies keeping
vectors in your database rather than in a dedicated store. Measured at several
selectivities, down to 1% of rows passing the filter.

Filtering changes what "correct" means. The true top-10 among rows where
`tag < 10` is not the true top-10 overall, so the harness **recomputes exact
ground truth by brute force over the qualifying subset** for every selectivity.
Scoring filtered results against unfiltered truth would report near-zero recall
for every engine and teach you nothing.

It also reports **how many queries returned fewer than k results**. An engine
can exhaust its candidate list before finding 10 rows that pass the filter; in
one run that happened on 81 of 200 queries. Recall computed over short result
sets is not wrong, but it is not comparable to recall over full ones unless the
count is sitting next to it.

### 5. Churn

Recall and throughput before and after deleting and re-inserting a share of the
corpus. HNSW graphs are widely expected to degrade under deletion, and for any
workload with ongoing writes this matters more than any static number above it.

---

## How the comparison is kept fair

### Two passes, on purpose

**Normalized** hands every engine identical CPU, memory and cache budgets. Any
difference in the results belongs to the implementations rather than to how much
machine each one was given.

**Tuned** lets each engine use settings its own documentation recommends. This
is the more realistic comparison and the less controlled one, which is why it
does not replace the normalized pass.

Running both is the point. If a ranking survives both, it is about the engines.
If it flips, that is a finding in itself.

### Identical containment

Server and client run in separate containers on a private network, with explicit
cpuset pinning, SMT siblings excluded, and P-cores and E-cores never mixed. Same
compiler flags for every engine. Durability relaxed identically across all of
them — leaving each at its own default would compare fsync policies rather than
vector search.

### Asymmetries we cannot remove, and do not hide

Some differences cannot be normalized away, so the report states them rather
than papering over them:

- `ef_construction` is exposed only by pgvector. MariaDB rejects it outright
  (`ERROR 1911 (HY000): Unknown option 'EF_CONSTRUCTION'`). The normalized pass
  pins pgvector to its default rather than handing it a tuning axis the others
  lack.
- AliSQL's VIDX is InnoDB-only and requires READ COMMITTED, so every engine is
  set to READ COMMITTED to remove isolation level as a variable.
- Both MySQL-family engines ship a 16 MiB default graph cache, far too small for
  any real corpus. Both are set from one budget rather than judged on a value
  their vendors plainly intended you to change.

Every run writes a manifest: CPU model and SIMD flags, engine source tags and
commits, image IDs, and the resource limits as actually resolved rather than as
requested. The report generator refuses to produce a report without one, and
every report ends with a section on reproducing that exact run. A result without
its environment is not a result.

---

## How to read the output

### Validity first

The report's section order is deliberate: environment, then validity, then known
asymmetries, then results. If a run lost an engine to a failure, if an engine
returned short result sets, if the CPU lacks AVX-512 — that appears **before**
any chart. A caveat that arrives after the number has been read is a caveat that
has already failed.

### The trap that invalidates everything

All three engines can silently decline to use the vector index and fall back to
a full table scan. A scan returns **exact** results, slowly. In the output that
looks like *high recall, low throughput* — indistinguishable from a
conservatively-tuned index unless you check the query plan.

It is not theoretical. Both of these are reproducible today:

- AliSQL costs the vector index against a table scan and picks the scan when the
  `LIMIT` exceeds roughly 25–28% of rows — on a 100-row table the crossover sits
  between `LIMIT 25` and `LIMIT 40`. It does not depend on `ef_search` at all.
- pgvector falls back with no error and no warning when the query operator does
  not match the index's operator class: `<->` against a `vector_cosine_ops`
  index gives you a `Seq Scan` and a `Sort`.

So every driver runs `EXPLAIN` for each configuration and records whether the
index was used. Any measurement where it was not is flagged in Validity, not
buried in a footnote. This is the single easiest way to produce impressive
nonsense in a vector benchmark, and the reason to distrust any that does not
mention it.

### The frontier, and the question you actually have

Because recall trades against speed, the honest presentation is a curve. Sweep
the search width, plot recall against QPS, take the upper-left envelope: for
each recall level, the best throughput achieved at it.

An engine is faster than another **only if its frontier sits above the other's
at the same recall.** Curves that cross mean the answer depends on your accuracy
target — a real result, not a hedge.

Curves are also awkward to read, because the eye compares shapes rather than
heights at one x. So the report also gives the comparison an operator actually
makes — *how fast is it at an accuracy I can accept* — as bars at recall floors
of 0.90, 0.95 and 0.99. Pick your floor, read across.

---

## What this does not measure

**Hardware.** MHNSW and VIDX both document AVX-512 distance kernels. Results
from a machine without AVX-512 do not transfer to one with it, which is why the
CPU model and its SIMD flags are recorded in every manifest.

**Scale beyond ~1M vectors.** Current corpora are 60k–1M and largely
cache-resident. At 10M the graphs go to disk and rankings may invert entirely.

**Anything sharing the machine.** During development, a competing build
distorted one engine's numbers by 2×. The harness reports CPU, SIMD and cpuset
problems; a concurrent workload is invisible to it. Run on a quiet box.

**Your query distribution.** Standard datasets are standard, not representative
of you.

---

## Adding an engine

The engine list will grow, so the cost of growing it is part of the design. Each
engine contributes four things:

1. A multi-stage Dockerfile producing a runtime image and a bench image from a
   pinned upstream tag.
2. A config declaring its ports, credentials, server flags and which knobs map
   to the normalized budget.
3. An ann-benchmarks module for the recall/QPS path.
4. A driver for the ops path implementing a small interface — create index,
   load, query, filtered query, index size, and a plan check that answers *was
   the index used*.

Nothing else changes. Adding an engine does not alter what any existing number
means, which is what makes it safe to publish results before the list is
complete.

---

## What comes next

Results. The harness is validated end to end on all three engines across both
measurement paths, and full runs at million-vector scale are underway. Those
will be published with the manifests, the raw per-configuration records, and the
methodology above to argue with.

The harness, the drivers and the documentation are at
[github.com/EvgeniyPatlan/vector-bench](https://github.com/EvgeniyPatlan/vector-bench).
If you think a measurement here is wrong or unfair to an engine, the useful form
of that argument is a run.
