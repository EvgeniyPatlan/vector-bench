"""Ops-harness driver for Valkey + valkey-search.

The only in-memory engine in the set. The MySQL family and pgvector keep a
disk-backed table with a cache in front of it, and Percona Search keeps its
index in a separate process; Valkey keeps everything resident, always. Four
consequences run through this file.

  There is no on-disk index to size. `index_bytes` is measured as the resident
  memory the index costs, taken as the difference in used_memory across index
  creation, and `table_bytes` is what the hashes themselves occupy. Reporting
  zero would read as a missing measurement, and reporting a file size would be
  reporting a file that does not exist.

  Eviction is the trap. If Valkey reaches maxmemory under any policy except
  noeviction it drops keys, and vectors vanishing mid-run looks exactly like a
  bad index: recall falls, nothing errors. The policy is pinned in the server
  config and checked here after the load, because a benchmark that silently
  measures a partial dataset is worse than one that fails.

  EF_RUNTIME is a per-query modifier inside the query string, not a session
  variable, so `set_ef_search` records rather than sends.

  The index can be created before or after the load, and both are legitimate.
  Creating it first indexes on write, which is what MHNSW and VIDX are forced
  into; creating it after triggers a backfill, which is pgvector's bulk build.
  `build_mode` selects which comparison is being made, exactly as it does for
  pgvector, and the backfill is waited out through FT.INFO's own
  backfill_complete_percent rather than assumed complete.
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Any, Dict, List, Optional, Sequence

import numpy

from .base import DATABASE, TABLE, ConnectionSpec, EngineDriver, IndexSpec, LoadResult

try:
    import valkey as valkey_client
except ImportError:  # pragma: no cover - only the bench image has this
    try:
        import redis as valkey_client        # protocol-compatible fallback
    except ImportError:
        valkey_client = None

PROGRESS_INTERVAL_S = float(os.environ.get("VB_PROGRESS_INTERVAL", "20"))

INDEX = "idx"
PREFIX = "v:"
VECTOR_FIELD = "embedding"
TAG_FIELD = "tag"
DIALECT = 2

BACKFILL_POLL_S = float(os.environ.get("VB_VALKEY_POLL_INTERVAL", "2"))
BACKFILL_TIMEOUT_S = float(os.environ.get("VB_VALKEY_BACKFILL_TIMEOUT", "43200"))

METRIC = {"angular": "COSINE", "euclidean": "L2"}

# Churn touches a fraction of the whole corpus in one call, so the same
# batching the load path uses applies here. Unbatched, a 10% churn of a
# million rows is one DEL of 99,000 keys and one pipeline holding 600 MB.
CHURN_BATCH = int(os.environ.get("VB_VALKEY_CHURN_BATCH", "1000"))

# Long enough that no honest write hits it, short enough that a reply which is
# never coming does not hold the machine until someone notices. Never None:
# that is what turned a stalled churn into a run that hung overnight.
WRITE_TIMEOUT_S = float(os.environ.get("VB_VALKEY_WRITE_TIMEOUT", "300"))

# The stall diagnosis must not become a second stall. A healthy write of one
# hash is a millisecond; thirty seconds is the difference between "slow" and
# "not coming".
CANARY_TIMEOUT_S = float(os.environ.get("VB_VALKEY_CANARY_TIMEOUT", "30"))

# The churn write gets a much shorter leash than the load. A healthy batch
# of a thousand took a quarter of a second during the load, so a minute is
# already two orders of magnitude of headroom -- and the batch size is
# halved on each stall, so this is paid once per halving rather than once.
CHURN_WRITE_TIMEOUT_S = float(
    os.environ.get("VB_VALKEY_CHURN_WRITE_TIMEOUT", "60"))

# Where the diagnosis looks for the turnover. One row is known to work and
# a thousand is known not to; the point of the ladder is to say where
# between them the write stops coming back, because that is the difference
# between a batching limit and something that is not about size at all.
_PROBE_SIZES = (1, 10, 100, 1000)

# How long a write may be outstanding before the server is asked about it,
# rather than after. A healthy batch of a thousand took a quarter of a
# second during the load, so anything still running at twenty is already
# two orders of magnitude out and worth a look.
STALL_OBSERVE_S = float(os.environ.get("VB_VALKEY_STALL_OBSERVE", "20"))


class _WriteStalled(RuntimeError):
    """A batch that did not come back. Carries no diagnosis of its own:
    the caller decides whether this is one to retry smaller or the last
    one, and only the last one is worth a full report."""



def _optional_int(value) -> Optional[int]:
    """int(value), or None for a field FT.INFO did not report.

    The distinction is the whole point: a field that is absent means the
    build does not report it, and treating that as zero is how a wait that
    was supposed to guard the measurement became a no-op.
    """
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def encode_vector(vector) -> bytes:
    """FLOAT32 little-endian, which is what the module expects in PARAMS."""
    return numpy.asarray(vector, dtype="<f4").tobytes()


class ValkeyDriver(EngineDriver):
    name = "valkey"
    incremental_index = False

    def __init__(self, spec: ConnectionSpec):
        super().__init__(spec)
        self._index: Optional[IndexSpec] = None
        self._ef_runtime: int = 100
        self._bytes_before_index = 0
        self._bytes_after_index = 0
        self._backfill_seconds = 0.0
        # Not a constant: see insert_rows. Starts where the load writes
        # and shrinks only if the engine will not take that size.
        self._write_batch = CHURN_BATCH

    # -- lifecycle ------------------------------------------------------

    def connect(self) -> None:
        if valkey_client is None:
            raise RuntimeError("valkey (or redis) client is not installed in this image")
        self._conn = valkey_client.Valkey(
            host=self.spec.host, port=self.spec.port,
            socket_timeout=600, socket_connect_timeout=30,
        ) if hasattr(valkey_client, "Valkey") else valkey_client.Redis(
            host=self.spec.host, port=self.spec.port,
            socket_timeout=600, socket_connect_timeout=30,
        )
        self._conn.ping()
        self._assert_no_eviction()

    def _assert_no_eviction(self) -> None:
        """Refuse to measure a configuration that can silently drop vectors.

        Any policy except noeviction turns a full memory condition into missing
        keys rather than an error, and the result is a recall figure computed
        over a corpus that is no longer the corpus.
        """
        policy = self._config_get("maxmemory-policy")
        if policy and policy != "noeviction":
            raise RuntimeError(
                f"maxmemory-policy is {policy!r}, not 'noeviction'. Under any "
                f"other policy Valkey drops keys when it reaches maxmemory, and "
                f"a vanished vector is indistinguishable from a bad index in the "
                f"results. Fix the server configuration rather than this check."
            )

    def _config_get(self, key: str) -> Optional[str]:
        try:
            value = self._conn.config_get(key)
            raw = value.get(key) if isinstance(value, dict) else None
            return raw.decode() if isinstance(raw, bytes) else raw
        except Exception:
            return None

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # -- schema ---------------------------------------------------------

    def drop_schema(self) -> None:
        try:
            self._conn.execute_command("FT.DROPINDEX", INDEX)
        except Exception:
            pass
        try:
            self._conn.flushall()
        except Exception:
            pass

    def create_schema(self, index: IndexSpec) -> None:
        """No schema to declare.

        Valkey has no table: the hashes are the data and the index is declared
        over a key prefix. In incremental mode the index is created here so
        writes are indexed as they arrive; in post mode create_index does it
        afterwards and the module backfills.
        """
        self._index = index
        if index.build_mode == "incremental":
            self.create_index(index)

    def create_index(self, index: IndexSpec) -> None:
        self._index = index
        if self._index_exists():
            return
        self._bytes_before_index = self._used_memory()
        args = [
            "FT.CREATE", INDEX, "ON", "HASH", "PREFIX", "1", PREFIX,
            "SCHEMA",
            TAG_FIELD, "NUMERIC",
            VECTOR_FIELD, "VECTOR", "HNSW", "8",
            "TYPE", "FLOAT32",
            "DIM", str(index.dim),
            "DISTANCE_METRIC", METRIC[index.metric],
            "M", str(index.m),
        ]
        if index.ef_construction:
            args += ["EF_CONSTRUCTION", str(index.ef_construction)]
            # attr_count counts the key/value pairs after the algorithm name.
            args[args.index("8")] = "10"
        self._conn.execute_command(*args)
        self._backfill_seconds = self._wait_for_backfill()
        self._bytes_after_index = self._used_memory()

    def _index_exists(self) -> bool:
        try:
            self._conn.execute_command("FT.INFO", INDEX)
            return True
        except Exception:
            return False

    def _wait_for_backfill(self) -> float:
        """Wait until the module has finished indexing what already exists.

        FT.CREATE returns immediately and indexes in the background, so a query
        issued straight afterwards searches a partial graph and reports a recall
        that has nothing to do with the index parameters.
        """
        started = time.time()
        last_report = started
        while True:
            done, indexing = self._backfill_progress()
            if done >= 1.0 and not indexing:
                return time.time() - started
            waited = time.time() - started
            if waited > BACKFILL_TIMEOUT_S:
                raise TimeoutError(
                    f"valkey-search backfill still at {done:.1%} after "
                    f"{waited / 3600:.1f} h")
            if time.time() - last_report >= PROGRESS_INTERVAL_S:
                print(f"[valkey] backfill {done:.1%}, {waited / 60:.1f} min elapsed",
                      flush=True)
                last_report = time.time()
            time.sleep(BACKFILL_POLL_S)

    def _backfill_progress(self):
        """Read valkey-search's own field names.

        Guessing these cost a whole smoke run. FT.INFO reports
        `backfill_complete_percent`, `backfill_in_progress` and `state`; there
        is no `percent_indexed` and no `indexing`. Looking for the names that
        do not exist made every default fire at once, so the wait returned
        immediately and the build, the row count, the index size and every
        query afterwards measured an index that was still empty.
        """
        info = self._ft_info()
        if not info:
            # No index yet, or FT.INFO failed. Not "finished".
            return 0.0, True
        try:
            done = float(info.get("backfill_complete_percent", 0.0))
        except (TypeError, ValueError):
            done = 0.0
        in_progress = str(info.get("backfill_in_progress", "1")) not in ("0", "false", "False")
        ready = str(info.get("state", "")).lower() in ("", "ready")
        return done, (in_progress or not ready)

    def _ft_info(self) -> Dict[str, Any]:
        try:
            raw = self._conn.execute_command("FT.INFO", INDEX)
        except Exception:
            return {}
        out: Dict[str, Any] = {}
        for i in range(0, len(raw) - 1, 2):
            key = raw[i].decode() if isinstance(raw[i], bytes) else str(raw[i])
            value = raw[i + 1]
            out[key] = value.decode() if isinstance(value, bytes) else value
        return out

    def _assert_nothing_failed_to_index(self) -> None:
        """A hash that fails to index is not an error anywhere else.

        The document still counts toward num_docs, the write returned OK and
        the query returns fewer neighbours than it should. It reads as poor
        recall rather than as a broken configuration, which is the failure this
        framework exists to catch rather than average into a curve.
        """
        info = self._ft_info()
        try:
            failures = int(info.get("hash_indexing_failures", 0))
        except (TypeError, ValueError):
            return
        if failures:
            raise RuntimeError(
                f"valkey-search failed to index {failures:,} hashes. The most "
                f"likely cause is a vector whose byte length does not match "
                f"DIM x 4. Recall computed against this index would be wrong "
                f"rather than merely low."
            )

    # -- data -----------------------------------------------------------

    def load(self, vectors: numpy.ndarray, tags: numpy.ndarray,
             threads: int = 1, start_id: int = 0) -> LoadResult:
        rows = len(vectors)
        started = time.time()
        if threads <= 1:
            self._write_range(vectors, tags, start_id, 0, rows)
        else:
            chunk = (rows + threads - 1) // threads
            with ThreadPoolExecutor(max_workers=threads) as pool:
                futures = [
                    pool.submit(self._write_range, vectors, tags, start_id,
                                begin, min(begin + chunk, rows))
                    for begin in range(0, rows, chunk)
                ]
                for f in futures:
                    f.result()
        elapsed = time.time() - started
        self._assert_nothing_evicted()
        return LoadResult(rows=rows, wall_seconds=elapsed, threads=threads)

    def _write_range(self, vectors, tags, start_id: int,
                     begin: int, end: int, batch: int = 1000) -> None:
        # One connection per worker: a shared client would make the thread
        # count a property of the connection pool rather than a measure of
        # write concurrency.
        conn = self._write_connection()
        try:
            pipe = conn.pipeline(transaction=False)
            for i in range(begin, end):
                pipe.hset(f"{PREFIX}{start_id + i}", mapping={
                    TAG_FIELD: int(tags[i]),
                    VECTOR_FIELD: encode_vector(vectors[i]),
                })
                if (i - begin + 1) % batch == 0:
                    pipe.execute()
            pipe.execute()
        finally:
            conn.close()

    def _assert_nothing_evicted(self) -> None:
        try:
            evicted = int(self._conn.info("stats").get("evicted_keys", 0))
        except Exception:
            return
        if evicted:
            raise RuntimeError(
                f"Valkey evicted {evicted:,} keys during the load. The corpus "
                f"is no longer the corpus and every recall figure from this run "
                f"would be computed against a dataset that is missing rows. "
                f"Raise the container memory limit or lower maxmemory."
            )

    def _write_connection(self, timeout_s: Optional[float] = None):
        """A connection for bulk writes: generous timeout, keepalive, not none.

        This has been wrong in both directions. It first carried the driver's
        600 second query timeout, which a slow write legitimately exceeds. It
        then carried no timeout at all, which is worse: a reply that never
        arrives blocks forever, and that is exactly what happened. The client
        sat at 0% CPU, 6.4 GB of a 10.6 GB limit, every cgroup memory counter
        at zero, waiting on a socket, while the server answered a fresh HSET
        into the same indexed prefix in 104 milliseconds.

        A timeout that a real write cannot hit, plus keepalive so a dead peer
        is noticed rather than waited on, plus a health check so a connection
        that has gone stale is replaced instead of trusted.
        """
        factory = getattr(valkey_client, "Valkey", None) or valkey_client.Redis
        return factory(
            host=self.spec.host, port=self.spec.port,
            socket_timeout=WRITE_TIMEOUT_S if timeout_s is None else timeout_s,
            socket_connect_timeout=30,
            socket_keepalive=True,
            health_check_interval=30,
        )

    def delete_ids(self, ids: Sequence[int]) -> None:
        """Batched, because a churn deletes a tenth of the corpus at once.

        One DEL carrying 99,000 key names is a single multi-bulk command of a
        megabyte or so. The load path has always batched; this did not, and it
        worked at smoke scale and failed at a million rows.
        """
        keys = [f"{PREFIX}{int(i)}" for i in ids]
        before = self.count_rows()
        conn = self._write_connection()
        try:
            for start in range(0, len(keys), CHURN_BATCH):
                conn.delete(*keys[start:start + CHURN_BATCH])
        finally:
            conn.close()
        self._wait_for_mutations(expected_docs=max(0, before - len(keys)))

    def _wait_for_mutations(self, expected_docs: Optional[int] = None,
                            timeout_s: float = 900.0) -> float:
        """Let the index absorb a mass delete before writing into it again.

        DEL returns as soon as the key is gone; removing the vector from the
        HNSW graph is separate work, and FT.INFO reports a mutation queue
        precisely so a caller can see it.

        The queue alone is not enough to wait on, and this is the second time
        that lesson has been paid for here. `percent_indexed` does not exist,
        so a missing field read as a finished backfill and every measurement
        after it described an empty index. `mutation_queue_size` is the same
        shape of trap: absent, or present and zero while the index still
        reported 481 documents more than the keyspace held. So the document
        count is waited on too, against the number the caller knows to expect,
        and a missing field is reported rather than treated as agreement.
        """
        started = time.time()
        last_report = started
        settled_for = 0
        previous: Optional[int] = None
        announced_missing = False

        while time.time() - started < timeout_s:
            info = self._ft_info()
            queued = _optional_int(info.get("mutation_queue_size"))
            if queued is None and not announced_missing:
                announced_missing = True
                print("[valkey] FT.INFO reports no mutation_queue_size in this "
                      "build; waiting on the document count instead",
                      flush=True)
            docs = _optional_int(info.get("num_docs"))

            queue_drained = queued in (None, 0)
            if docs is None:
                # This build reports no document count either. The queue is
                # then the only signal there is; waiting on a number nobody
                # publishes would hang every churn rather than guard one.
                docs_ready = True
            elif expected_docs is None:
                # Nothing to compare against: settle instead, which needs two
                # consecutive readings so a single sample cannot end the wait.
                settled_for = settled_for + 1 if docs == previous else 0
                docs_ready = settled_for >= 2
            else:
                docs_ready = docs <= expected_docs
            previous = docs

            if queue_drained and docs_ready:
                return time.time() - started

            if time.time() - last_report >= PROGRESS_INTERVAL_S:
                print(f"[valkey] index absorbing deletes: "
                      f"num_docs={docs if docs is not None else '?'}"
                      f"{f' (target {expected_docs:,})' if expected_docs is not None else ''}"
                      f", queued={queued if queued is not None else 'n/a'}, "
                      f"{time.time() - started:.0f}s elapsed", flush=True)
                last_report = time.time()
            time.sleep(2)

        waited = time.time() - started
        print(f"[valkey] WARNING: the index had not absorbed the delete after "
              f"{waited:.0f}s. Writing into it anyway; treat what follows as a "
              f"measurement of an index that was still catching up.",
              file=sys.stderr, flush=True)
        return waited

    def insert_rows(self, ids: Sequence[int], vectors: numpy.ndarray,
                    tags: Sequence[int]) -> None:
        """Write back in batches, shrinking the batch if one does not return.

        Two facts fix the shape of this. A pipeline of a thousand HSETs is what
        the load uses and it moves 30,000 rows a second into an unindexed
        keyspace. The same pipeline into the populated index does not come back
        at all -- 302 seconds, zero rows -- while a single HSET into that same
        index, on a connection opened at the moment of the stall, returns in
        one millisecond.

        So the batch is not a constant any more. A write that does not return
        halves it and tries again on a fresh connection, down to one row at a
        time, and the size that worked is reported: at a millisecond a row even
        the floor finishes 99,000 rows inside the budget. That is a slower
        measurement than the load, which is the honest result -- writing into a
        live HNSW graph is slower than writing into a hash table -- but it is a
        measurement, where a timeout is not.
        """
        cursor = 0
        while cursor < len(ids):
            size = min(self._write_batch, len(ids) - cursor)
            stop = cursor + size
            try:
                self._write_batch_once(ids[cursor:stop], vectors[cursor:stop],
                                       tags[cursor:stop])
            except _WriteStalled:
                if self._write_batch <= 1:
                    raise RuntimeError(
                        f"a single-row write did not return either.\n"
                        f"{self._diagnose_stalled_write(ids[cursor:stop], vectors[cursor:stop], tags[cursor:stop])}"
                    )
                if self._write_batch == CHURN_BATCH:
                    # Printed once, from the first stall, while the server is
                    # still in the state that produced it.
                    print(self._diagnose_stalled_write(
                        ids[cursor:stop], vectors[cursor:stop],
                        tags[cursor:stop]), flush=True)
                self._write_batch = max(1, self._write_batch // 2)
                print(f"[valkey] batched write of {size:,} rows did not "
                      f"return; retrying at {self._write_batch:,}",
                      flush=True)
                continue
            cursor = stop

    def _write_batch_once(self, ids, vectors, tags) -> None:
        """One pipeline, one connection, a timeout a healthy write cannot hit.

        Fresh each time: redis-py disconnects a connection whose read timed
        out, and reusing one that has already failed would confuse a wedged
        socket with a wedged server -- which is the distinction this whole path
        exists to make.
        """
        conn = self._write_connection(timeout_s=CHURN_WRITE_TIMEOUT_S)
        try:
            pipe = conn.pipeline(transaction=False)
            for i, key_id in enumerate(ids):
                pipe.hset(f"{PREFIX}{int(key_id)}", mapping={
                    TAG_FIELD: int(tags[i]),
                    VECTOR_FIELD: encode_vector(vectors[i]),
                })
            try:
                self._execute_watched(pipe, len(ids))
            except Exception as exc:
                raise _WriteStalled(str(exc)) from exc
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _execute_watched(self, pipe, rows: int) -> None:
        """Run the batch, and look at the server while it is still hanging.

        Every diagnosis so far has run after the fact, and that is why five of
        them were wrong. redis-py disconnects a connection whose read timed
        out, so by the time an exception is caught the socket the server would
        have had an opinion about is already gone: CLIENT LIST shows nothing,
        connected_clients is back to one, and the only honest reading of any of
        it is "the client is no longer connected", which was never in doubt.

        So the write goes on a thread and the server is questioned from here
        while it is still outstanding. CLIENT LIST settles it in one line. If
        the stalled connection is there with a full query buffer and cmd=hset,
        the server has the write and is not finishing it. If it is not there at
        all, the server never took the connection, and nothing about the index
        or the module is involved.
        """
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            future = pool.submit(pipe.execute)
            try:
                return future.result(timeout=STALL_OBSERVE_S)
            except FuturesTimeout:
                pass
            print(f"[valkey] a write of {rows:,} rows has not returned after "
                  f"{STALL_OBSERVE_S:.0f}s. The server, while it is still "
                  f"outstanding:", flush=True)
            print(self._live_server_state(), flush=True)
            return future.result(
                timeout=max(1.0, CHURN_WRITE_TIMEOUT_S - STALL_OBSERVE_S))
        except FuturesTimeout as exc:
            raise TimeoutError(
                f"the write did not return within "
                f"{CHURN_WRITE_TIMEOUT_S:.0f}s") from exc
        finally:
            # Not waited on: the thread is blocked on a socket that is about to
            # be closed under it, and holding the run for that is the failure
            # this path exists to end.
            pool.shutdown(wait=False)

    def _live_server_state(self) -> str:
        """CLIENT LIST and the input counter, over the monitoring connection.

        Deliberately best-effort: this runs while something is already wrong.
        """
        lines = []
        try:
            clients = self._conn.execute_command("CLIENT", "LIST")
            if isinstance(clients, bytes):
                clients = clients.decode(errors="replace")
            for row in str(clients).splitlines():
                lines.append(f"    {row}")
        except Exception as exc:
            lines.append(f"    CLIENT LIST failed: {type(exc).__name__}: {exc}")
        for field in ("connected_clients", "blocked_clients",
                      "total_net_input_bytes", "instantaneous_ops_per_sec",
                      "used_memory_human"):
            try:
                lines.append(f"    INFO {field:24} = "
                             f"{self._conn.info().get(field, 'n/a')}")
            except Exception:
                break
        try:
            info = self._ft_info()
            lines.append(f"    FT.INFO mutation_queue_size = "
                         f"{info.get('mutation_queue_size', 'n/a')}, "
                         f"num_docs = {info.get('num_docs', 'n/a')}")
        except Exception:
            pass
        return "\n".join(lines)

    @property
    def write_batch_used(self) -> int:
        """The batch size the re-insert settled on, for the record.

        A churn that completed at one row per write and one that completed at a
        thousand are not the same result, and nothing else in the record would
        say which happened.
        """
        return self._write_batch

    def _diagnose_stalled_write(self, ids, vectors, tags) -> str:
        """What the server was doing when a write did not come back.

        Deliberately best-effort throughout: this runs while something is
        already wrong, and a diagnosis that raises tells nobody anything.
        """
        lines = ["[valkey] the write did not return. Server state at that moment:"]

        info = self._ft_info()
        for field in ("num_docs", "num_records", "mutation_queue_size",
                      "backfill_in_progress", "backfill_complete_percent",
                      "hash_indexing_failures", "state"):
            lines.append(f"    FT.INFO {field:26} = {info.get(field, 'n/a')}")

        try:
            server = self._conn.info()
        except Exception as exc:
            lines.append(f"    INFO failed on the monitoring connection: {exc}")
            lines.append("    -- which means the server, not one connection, "
                         "stopped answering.")
            return "\n".join(lines)

        for field in ("connected_clients", "blocked_clients", "used_memory_human",
                      "maxmemory_human", "mem_fragmentation_ratio",
                      "instantaneous_ops_per_sec", "rdb_bgsave_in_progress",
                      "loading", "total_net_input_bytes"):
            lines.append(f"    INFO    {field:26} = {server.get(field, 'n/a')}")

        # num_docs disagreeing with the keyspace is the difference between an
        # index that is behind and one that is wrong, and only one of those is
        # something to wait out.
        try:
            lines.append(f"    DBSIZE  {'keys':26} = {self._conn.dbsize():,}")
        except Exception:
            pass

        # The decisive test, and then the size at which it stops working. One
        # HSET returning while a thousand did not says the engine was never
        # what stalled; where between one and a thousand it turns over says
        # what to do about it.
        for size in _PROBE_SIZES:
            if size > len(ids):
                break
            outcome = self._probe_write(ids[:size], vectors[:size], tags[:size])
            lines.append(f"    write of {size:>5} row(s) on a fresh "
                         f"connection: {outcome}")
        return "\n".join(lines)

    def _probe_write(self, ids, vectors, tags) -> str:
        """Try one sized write and describe what happened, without raising."""
        try:
            probe = self._write_connection(timeout_s=CANARY_TIMEOUT_S)
        except Exception as exc:
            return f"could not connect: {type(exc).__name__}: {exc}"
        try:
            started = time.time()
            pipe = probe.pipeline(transaction=False)
            for i, key_id in enumerate(ids):
                pipe.hset(f"{PREFIX}{int(key_id)}", mapping={
                    TAG_FIELD: int(tags[i]),
                    VECTOR_FIELD: encode_vector(vectors[i]),
                })
            pipe.execute()
            return f"OK in {(time.time() - started) * 1000:.0f} ms"
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"
        finally:
            try:
                probe.close()
            except Exception:
                pass

    def count_rows(self) -> int:
        info = self._ft_info()
        try:
            return int(info.get("num_docs", 0))
        except (TypeError, ValueError):
            return 0

    # -- query ----------------------------------------------------------

    def set_ef_search(self, ef_search: int) -> None:
        """EF_RUNTIME is a modifier inside the query string, not a setting."""
        self._ef_runtime = int(ef_search)

    def _search(self, vector, k: int, filter_expr: str = "*") -> List[int]:
        query = (f"{filter_expr}=>[KNN {k} @{VECTOR_FIELD} $vec "
                 f"EF_RUNTIME {max(self._ef_runtime, k)}]")
        raw = self._conn.execute_command(
            "FT.SEARCH", INDEX, query,
            "PARAMS", "2", "vec", encode_vector(vector),
            "LIMIT", "0", str(k),
            # Keys only. Returning the vectors would put 6 KB per hit on the
            # wire and measure the network rather than the index.
            "NOCONTENT",
            "DIALECT", str(DIALECT),
        )
        ids = []
        for item in raw[1:]:
            key = item.decode() if isinstance(item, bytes) else str(item)
            if key.startswith(PREFIX):
                ids.append(int(key[len(PREFIX):]))
        return ids

    def query(self, vector, k: int) -> List[int]:
        return self._search(vector, k)

    def query_filtered(self, vector, k: int, tag_threshold: int) -> List[int]:
        # Exclusive upper bound, matching `WHERE tag < threshold` in the SQL
        # engines and `{"tag": {"$lt": ...}}` in Percona Search.
        return self._search(vector, k, f"@{TAG_FIELD}:[-inf ({int(tag_threshold)}]")

    def explain_uses_vector_index(self, vector, k: int,
                                  tag_threshold: Optional[int] = None) -> bool:
        """There is no plan to inspect and no scan to fall back to.

        FT.SEARCH is only answerable through the index; without one the command
        errors. What this distinguishes is an erroring configuration from a
        genuinely slow one, so the two are not both recorded as low throughput.
        """
        try:
            if tag_threshold is None:
                self._search(vector, k)
            else:
                self._search(vector, k, f"@{TAG_FIELD}:[-inf ({int(tag_threshold)}]")
            return True
        except Exception:
            return False

    # -- sizing / metadata ----------------------------------------------

    def _indexed_memory(self) -> int:
        """What the module says its own attributes cost, from FT.INFO.

        The used_memory delta across index creation is the obvious measure and
        it is not reliable: on the upstream build the index shows up there, and
        on Percona's it does not, which reported a 66 KB index beside an 82 MB
        table. FT.INFO carries user_indexed_memory per attribute, and it needs
        a nested walk because attributes is a list of lists.
        """
        total = 0

        def walk(node):
            nonlocal total
            if not isinstance(node, (list, tuple)):
                return
            for i in range(0, len(node) - 1, 2):
                key = node[i]
                key = key.decode() if isinstance(key, bytes) else key
                value = node[i + 1]
                if key == "user_indexed_memory":
                    try:
                        total += int(value)
                    except (TypeError, ValueError):
                        pass
                if isinstance(value, (list, tuple)):
                    walk(value)
                    for sub in value:
                        walk(sub)

        try:
            walk(self._conn.execute_command("FT.INFO", INDEX))
        except Exception:
            return 0
        return total

    def _used_memory(self) -> int:
        try:
            return int(self._conn.info("memory").get("used_memory", 0))
        except Exception:
            return 0

    def index_bytes(self) -> int:
        """Resident cost of the index, not a file size.

        Two measurements, and the larger wins. The used_memory delta across
        index creation covers the graph as well as the vectors but only where
        the build accounts index allocations to the main allocator, which
        Percona's does not: it reported 66 KB beside an 82 MB table. FT.INFO's
        user_indexed_memory is always reported but counts the indexed vectors
        rather than the graph built over them. Taking the larger reports the
        more complete of the two rather than whichever the build happens to
        support.
        """
        delta = max(0, self._bytes_after_index - self._bytes_before_index)
        return max(delta, self._indexed_memory())

    def table_bytes(self) -> int:
        """What the hashes themselves cost, index excluded."""
        if self._bytes_before_index:
            return self._bytes_before_index
        return max(0, self._used_memory() - self.index_bytes())

    def server_version(self) -> str:
        try:
            info = self._conn.info("server")
            version = info.get("valkey_version") or info.get("redis_version") or "?"
        except Exception:
            version = "unknown"
        return f"Valkey {version} / valkey-search"

    def capabilities(self) -> Dict[str, Any]:
        caps = super().capabilities()
        caps.update({
            "ef_construction_tunable": True,
            "in_memory_only": True,
            "hybrid_filter_planner": True,
            "backfill_seconds": round(self._backfill_seconds, 3),
        })
        return caps
