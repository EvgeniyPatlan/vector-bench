"""MariaDB MHNSW vector index, driven by ann-benchmarks.

Differences from the module shipped in MariaDB's ann-benchmarks fork:

* `M` is set as an index option in DDL (`VECTOR INDEX (v) M=n DISTANCE=…`)
  rather than through the `mhnsw_default_m` startup variable. 11.8 exposes both
  — `HA_IOPTION_SYSVAR("m", M, default_m)` in sql/vector_mhnsw.cc — and the DDL
  form matches how AliSQL takes M, so the two engines are configured the same way.
* InnoDB rather than MyISAM by default, because AliSQL's VIDX is InnoDB-only and
  the headline comparison has to hold storage engine constant. MyISAM remains
  available as an extra MariaDB-only curve in the tuned pass.
* The server is started through the image entrypoint, so a benchmark run and a
  by-hand `docker run` exercise identical startup.
"""

from ...vb_mysql import Dialect, VBMySQLBase

MARIADB = Dialect(
    name="mariadb",
    global_setup=(),
    session_setup=("SET SESSION default_storage_engine = InnoDB",),
    set_ef_search="SET mhnsw_ef_search = {ef_search}",
    index_name="vi",
    # MariaDB stores the MHNSW graph in a companion "high-level index" table
    # named <table>#i#<nn>. Some filesystems/versions encode '#' as @0023, so
    # both spellings are matched.
    index_file_globs=(
        "t1#i#*.ibd",
        "t1#i#*.MYI",
        "t1#i#*.MYD",
        "t1@0023i@0023*.ibd",
        "t1@0023i@0023*.MYI",
        "t1@0023i@0023*.MYD",
    ),
    metric_names={"angular": "cosine", "euclidean": "euclidean"},
    # MariaDB's optimizer has not been observed to decline the vector index for
    # the k values used here, but the check is cheap and the failure mode
    # (silently benchmarking a full scan) is severe, so it stays on.
    verify_index_used=True,
    force_index_hint="FORCE INDEX (vi)",
)


class MariaDB(VBMySQLBase):
    dialect = MARIADB
