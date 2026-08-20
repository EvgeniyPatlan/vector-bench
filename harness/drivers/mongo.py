"""Ops-harness driver for Percona Search for MongoDB (mongot).

A third client stack, and a third architecture. The MySQL-family engines and
pgvector each keep their vector index inside the query engine; here `mongod`
holds the documents and a separate Lucene process, `mongot`, holds the index and
is fed by a change stream. Three consequences run through this file:

  * The index build is asynchronous. createSearchIndex returns before the index
    exists, so `create_index` polls getSearchIndexes until status is READY and
    reports the waiting as build time. Neither `incremental_index = True`
    (MHNSW, VIDX) nor `False` (pgvector's separable bulk build) describes that,
    so this driver reports a third mode.

  * The filter is applied before vector comparison rather than after. Every
    other engine in the set post-filters, which is why their filtered results
    either collapse to a fraction of a query per second or come back short.

  * Sizing a query means numCandidates, not ef_search. It is the same knob in
    spirit, but the vendor's guidance starts at 20x the limit, so the low end of
    the shared grid is further outside its intended range than it is for the
    others.
"""

from __future__ import annotations

import os
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence

import numpy

from .base import DATABASE, TABLE, ConnectionSpec, EngineDriver, IndexSpec, LoadResult

try:
    import pymongo
    from bson.binary import Binary
except ImportError:  # pragma: no cover - only the bench image has these
    pymongo = None
    Binary = None

PROGRESS_INTERVAL_S = float(os.environ.get("VB_PROGRESS_INTERVAL", "20"))

INDEX_NAME = "vector_index"
# createSearchIndex is asynchronous with no completion callback, so the only
# way to know the index exists is to ask until it says so.
READY_POLL_S = float(os.environ.get("VB_MONGOT_POLL_INTERVAL", "2"))
READY_TIMEOUT_S = float(os.environ.get("VB_MONGOT_READY_TIMEOUT", "43200"))

SIMILARITY = {"angular": "cosine", "euclidean": "euclidean"}


def encode_vector(vector) -> Any:
    """Pack a float vector as BSON BinData float32.

    A BSON array of doubles spends 8 bytes per dimension, so 990k x 1536 costs
    12.2 GB in mongod against the 5.7 GB the same vectors occupy in Lucene.
    BinData float32 halves the working set and is the encoding MongoDB
    recommends for vector search.
    """
    values = numpy.asarray(vector, dtype=numpy.float32)
    # Subtype 9 is BSON's vector type; the two header bytes are the float32
    # data-type marker and a zero padding count.
    payload = b"\x27\x00" + values.tobytes()
    return Binary(payload, 9)


class MongoDriver(EngineDriver):
    name = "mongodb"
    # Neither of the framework's two modes. The graph is not maintained on
    # INSERT and it is not built by a blocking statement after the load; it is
    # built in another process while and after the writes land.
    incremental_index = False
    async_index_build = True

    def __init__(self, spec: ConnectionSpec):
        super().__init__(spec)
        self._db = None
        self._coll = None
        self._index: Optional[IndexSpec] = None
        self._num_candidates: int = 100
        self._ready_seconds: float = 0.0

    # -- lifecycle ------------------------------------------------------

    def connect(self) -> None:
        if pymongo is None:
            raise RuntimeError("pymongo is not installed in this image")
        uri = self._uri()
        self._conn = pymongo.MongoClient(uri, serverSelectionTimeoutMS=60000)
        # Fail here rather than inside the first measured operation.
        self._conn.admin.command("ping")
        self._db = self._conn[self.spec.database or DATABASE]
        self._coll = self._db[TABLE]

    def _uri(self) -> str:
        """Authenticated, unlike every other driver here.

        mongot will not parse a config without SCRAM or x509, so mongod runs
        with auth on and the benchmark client has to authenticate as well. The
        credentials are fixed and local to an isolated container network; they
        are not a secret, they are a requirement mongot imposes.
        """
        user, password = self.spec.user, self.spec.password
        if not user:
            return f"mongodb://{self.spec.host}:{self.spec.port}/?directConnection=true"
        return (f"mongodb://{user}:{password}@{self.spec.host}:{self.spec.port}/"
                f"?directConnection=true&authSource=admin")

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # -- schema ---------------------------------------------------------

    def drop_schema(self) -> None:
        try:
            self._db.drop_collection(TABLE)
        except Exception:
            pass

    def create_schema(self, index: IndexSpec) -> None:
        """Create the collection only.

        Unlike MHNSW and VIDX there is nothing to declare here that builds a
        graph: the search index is a separate object owned by mongot, created
        in create_index.
        """
        self._index = index
        self._db.create_collection(TABLE)

    def create_index(self, index: IndexSpec) -> None:
        """Create the search index and wait for mongot to finish building it.

        The wait is the point. createSearchIndex returns in milliseconds and the
        index is unqueryable until mongot has consumed the change stream and
        written its segments, so treating the call as the build would report a
        build time of nearly zero.
        """
        self._index = index
        fields: List[Dict[str, Any]] = [{
            "type": "vector",
            "path": "embedding",
            "numDimensions": index.dim,
            "similarity": SIMILARITY[index.metric],
        }]
        if index.quantization and index.quantization != "none":
            fields[0]["quantization"] = index.quantization
        # Declared even when unused: a field can only be filtered on if it was
        # indexed as one, and adding it later would rebuild the index.
        fields.append({"type": "filter", "path": "tag"})

        self._coll.create_search_index({
            "name": INDEX_NAME,
            "type": "vectorSearch",
            "definition": {"fields": fields},
        })
        self._ready_seconds = self._wait_until_ready()

    def _wait_until_ready(self) -> float:
        started = time.time()
        last_report = started
        while True:
            status = self._index_status()
            if status == "READY":
                return time.time() - started
            if status in ("FAILED", "DOES_NOT_EXIST"):
                raise RuntimeError(
                    f"mongot index build reported {status}; "
                    "check the mongot container logs")
            waited = time.time() - started
            if waited > READY_TIMEOUT_S:
                raise TimeoutError(
                    f"mongot index still {status} after {waited / 3600:.1f} h")
            if time.time() - last_report >= PROGRESS_INTERVAL_S:
                print(f"[mongot] index {status}, {waited / 60:.1f} min elapsed",
                      flush=True)
                last_report = time.time()
            time.sleep(READY_POLL_S)

    def _index_status(self) -> str:
        try:
            for info in self._coll.list_search_indexes():
                if info.get("name") == INDEX_NAME:
                    return str(info.get("status", "UNKNOWN"))
        except Exception:
            return "UNKNOWN"
        return "DOES_NOT_EXIST"

    # -- data -----------------------------------------------------------

    def load(self, vectors: numpy.ndarray, tags: numpy.ndarray,
             threads: int = 1, start_id: int = 0) -> LoadResult:
        rows = len(vectors)
        started = time.time()
        if threads <= 1:
            self._insert_range(vectors, tags, start_id, 0, rows)
        else:
            chunk = (rows + threads - 1) // threads
            with ThreadPoolExecutor(max_workers=threads) as pool:
                futures = [
                    pool.submit(self._insert_range, vectors, tags, start_id,
                                begin, min(begin + chunk, rows))
                    for begin in range(0, rows, chunk)
                ]
                for f in futures:
                    f.result()
        return LoadResult(rows=rows, wall_seconds=time.time() - started,
                          threads=threads)

    def _insert_range(self, vectors, tags, start_id: int,
                      begin: int, end: int, batch: int = 1000) -> None:
        # Each worker opens its own client: MongoClient is thread safe but
        # shares one connection pool, and sharing it would make the thread count
        # a setting on the pool rather than a real measure of write concurrency.
        client = pymongo.MongoClient(self._uri())
        try:
            coll = client[self.spec.database or DATABASE][TABLE]
            for chunk_start in range(begin, end, batch):
                chunk_end = min(chunk_start + batch, end)
                coll.insert_many([
                    {"_id": start_id + i,
                     "tag": int(tags[i]),
                     "embedding": encode_vector(vectors[i])}
                    for i in range(chunk_start, chunk_end)
                ], ordered=False)
        finally:
            client.close()

    def delete_ids(self, ids: Sequence[int]) -> None:
        self._coll.delete_many({"_id": {"$in": list(ids)}})

    def insert_rows(self, ids: Sequence[int], vectors: numpy.ndarray,
                    tags: Sequence[int]) -> None:
        self._coll.insert_many([
            {"_id": int(ids[i]), "tag": int(tags[i]),
             "embedding": encode_vector(vectors[i])}
            for i in range(len(ids))
        ], ordered=False)

    def count_rows(self) -> int:
        return int(self._coll.count_documents({}))

    # -- query ----------------------------------------------------------

    def set_ef_search(self, ef_search: int) -> None:
        """numCandidates is the ef_search analogue.

        Not a session variable: it is a per-query argument, so this records the
        value the next query will use rather than sending anything.
        """
        self._num_candidates = int(ef_search)

    def _pipeline(self, vector, k: int,
                  tag_threshold: Optional[int] = None) -> List[Dict[str, Any]]:
        stage: Dict[str, Any] = {
            "index": INDEX_NAME,
            "path": "embedding",
            "queryVector": encode_vector(vector),
            # Never below k: numCandidates < limit cannot return limit rows,
            # and the low end of the shared grid reaches there at k=10.
            "numCandidates": max(self._num_candidates, k),
            "limit": k,
        }
        if tag_threshold is not None:
            stage["filter"] = {"tag": {"$lt": int(tag_threshold)}}
        return [{"$vectorSearch": stage}, {"$project": {"_id": 1}}]

    def query(self, vector, k: int) -> List[int]:
        return [d["_id"] for d in
                self._coll.aggregate(self._pipeline(vector, k))]

    def query_filtered(self, vector, k: int, tag_threshold: int) -> List[int]:
        return [d["_id"] for d in
                self._coll.aggregate(self._pipeline(vector, k, tag_threshold))]

    def explain_uses_vector_index(self, vector, k: int,
                                  tag_threshold: Optional[int] = None) -> bool:
        """Whether the query ran as a vector search rather than a collection scan.

        $vectorSearch has no planner fallback the way a SQL optimiser does: if
        the index is missing the aggregation errors instead of quietly scanning.
        The check is kept anyway, because an error here and a slow exact scan
        are worth telling apart in the record.
        """
        try:
            plan = self._db.command({
                "explain": {
                    "aggregate": TABLE,
                    "pipeline": self._pipeline(vector, k, tag_threshold),
                    "cursor": {},
                },
                "verbosity": "queryPlanner",
            })
        except Exception:
            return False
        return "$vectorSearch" in str(plan)

    # -- sizing / metadata ----------------------------------------------

    def index_bytes(self) -> int:
        """Size of the Lucene index, which lives in mongot rather than mongod.

        collStats cannot see it: it is another process's files. The orchestrator
        shares mongot's data directory read-only for exactly this, and the
        fallback reports what the index metadata claims rather than zero.
        """
        if self.spec.data_dir and os.path.isdir(self.spec.data_dir):
            total = 0
            for root, _dirs, files in os.walk(self.spec.data_dir):
                for name in files:
                    try:
                        total += os.path.getsize(os.path.join(root, name))
                    except OSError:
                        continue
            if total:
                return total
        try:
            for info in self._coll.list_search_indexes():
                if info.get("name") == INDEX_NAME:
                    return int(info.get("indexSizeBytes") or 0)
        except Exception:
            pass
        return 0

    def table_bytes(self) -> int:
        try:
            stats = self._db.command("collStats", TABLE)
            return int(stats.get("storageSize") or 0)
        except Exception:
            return 0

    def server_version(self) -> str:
        try:
            build = self._conn.admin.command("buildInfo")
            version = build.get("version", "?")
        except Exception:
            version = "?"
        return f"Percona Server for MongoDB {version} / mongot"

    def capabilities(self) -> Dict[str, Any]:
        caps = super().capabilities()
        caps.update({
            "async_index_build": True,
            "prefilter": True,
            "quantization": (self._index.quantization if self._index else None),
            # Reported separately from the load: the two overlap, so adding
            # them would double count and treating either alone would
            # understate the cost of having a queryable index.
            "index_ready_seconds": round(self._ready_seconds, 3),
        })
        return caps
