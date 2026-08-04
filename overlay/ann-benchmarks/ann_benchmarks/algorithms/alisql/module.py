"""AliSQL VIDX vector index, driven by ann-benchmarks.

New module — upstream ann-benchmarks has no AliSQL support.

Three AliSQL-specific facts drive this code:

1. **Vector support ships disabled.** `vidx_disabled` defaults to ON; any
   `VECTOR(N)` column or `VECTOR INDEX` fails with ER_VECTOR_DISABLED until it is
   turned off. The runtime image passes `--vidx-disabled=OFF`, and the statement
   is repeated here so the module also works against a server someone else started.

2. **Vector DML requires READ COMMITTED.** At any other isolation level the
   server raises ER_NOT_SUPPORTED_YET. Set per session, on every connection.

3. **The optimizer can silently choose a full scan.** AliSQL's own
   mysql-test/suite/rds/t/vidx_dml.test shows a 100-row table using the vector
   index at `LIMIT 16` and falling back to JT_ALL at `LIMIT 17`. A full scan
   returns exact results, so recall looks perfect while QPS collapses — the
   result would look like "AliSQL is accurate but slow" when in fact no ANN
   search happened at all. `verify_index_used` makes that loud instead of silent.

Values are bound as raw little-endian float32, which Field_vector::store()
accepts directly (it validates length == 4*dim), rather than through
VEC_FROMTEXT(). That keeps the client path identical to MariaDB's.
"""

from ...vb_mysql import Dialect, VBMySQLBase

ALISQL = Dialect(
    name="alisql",
    global_setup=("SET GLOBAL vidx_disabled = OFF",),
    session_setup=(
        "SET SESSION transaction_isolation = 'READ-COMMITTED'",
        "SET SESSION default_storage_engine = InnoDB",
    ),
    set_ef_search="SET vidx_hnsw_ef_search = {ef_search}",
    index_name="vi",
    # VIDX keeps the HNSW graph in an InnoDB auxiliary table named
    # vidx_%016lx_%02x (se_private_id, index number) — see VIDX_NAME in
    # sql/vidx/vidx_index.cc. The exact id is not known ahead of time, so match
    # on the prefix.
    index_file_globs=("vidx_*.ibd",),
    metric_names={"angular": "COSINE", "euclidean": "EUCLIDEAN"},
    verify_index_used=True,
    force_index_hint="FORCE INDEX (vi)",
)


class AliSQL(VBMySQLBase):
    dialect = ALISQL
