#
# Percona Server (MySQL 9.7.1 base + native InnoDB HNSW vector index),
# built from a pinned SHA on the `vector-mvp` branch.
#
# Targets:
#   runtime  — server binaries only. Use this to run Percona Server by hand
#              (see docs/03-running-manually.md).
#   bench    — runtime + the Python stack ann-benchmarks drives.
#
# Build context is work/buildctx/percona/, produced by scripts/prepare-sources.sh.
#
# REVIEW STATUS: this file has never been built. docs/09-adding-percona-server.md
# §7 lists what the first build has to prove. Every flag below cites the line on
# the pinned tree that made it necessary; none was guessed from a commit subject.
#
# Differences from docker/alisql/Dockerfile, which is otherwise the closest
# template (both are MySQL-derived, both initialise with
# `mysqld --initialize-insecure`, both create the bench account from --init-file):
#
#   1. OUT-OF-SOURCE BUILD. AliSQL's own build.sh forces an in-source build.
#      Percona Server refuses one: CMakeLists.txt:583 is
#      `OPTION(FORCE_INSOURCE_BUILD "Allow in-source build" OFF)` and :606 says
#      "You *can* force in-source build by invoking cmake with
#      -DFORCE_INSOURCE_BUILD=1". We do not force it; cmake runs in /build.
#
#   2. NO SUBMODULE FETCH, WHICH TAKES FOUR FLAGS AND NOT TWO. The pinned
#      tree's .gitmodules lists four submodules, and prepare-sources.sh ships a
#      `git archive`, which omits submodule paths entirely. Every one of them
#      therefore has to be switched off, or cmake descends into a directory
#      that is not in the tarball:
#
#        storage/rocksdb/rocksdb  -> -DWITH_ROCKSDB=OFF
#        extra/coredumper         -> -DWITH_COREDUMPER=OFF
#                                    CMakeLists.txt:1558, default ON
#        extra/jwt-cpp            -> -DWITH_AUTH_OPENID_CONNECT=OFF
#                                    CMakeLists.txt:167, default ON
#        extra/libkmip            -> -DWITHOUT_COMPONENT_KEYRING_KMIP=1
#
#      The last one is the reason this list is four lines long. libkmip is not
#      behind an OPTION() with a default; it is behind a two-symbol condition
#      that fires when NEITHER symbol is defined — CMakeLists.txt:2295-2297:
#
#        IF(NOT DEFINED WITH_COMPONENT_KEYRING_KMIP AND NOT DEFINED WITHOUT_COMPONENT_KEYRING_KMIP
#           OR WITH_COMPONENT_KEYRING_KMIP)
#          ADD_SUBDIRECTORY(extra/libkmip)
#        ENDIF()
#
#      AND binds tighter than OR, so with neither defined that is
#      (TRUE AND TRUE) OR FALSE = TRUE — which is the original defect: an
#      earlier revision of this file enumerated all four submodules correctly,
#      said "two", and passed three flags, so the build would have failed at
#      configure time on a directory the tarball does not contain.
#
#      -DWITH_COMPONENT_KEYRING_KMIP=OFF closes the gate too. Measured, not
#      reasoned: the condition above was transcribed verbatim into a standalone
#      project and run under cmake 3.28.3 —
#
#        (no flags)                          -> WOULD_ADD_SUBDIRECTORY
#        -DWITH_COMPONENT_KEYRING_KMIP=OFF   -> SKIPPED
#        -DWITH_COMPONENT_KEYRING_KMIP=0     -> SKIPPED
#        -DWITH_COMPONENT_KEYRING_KMIP=NO    -> SKIPPED
#        -DWITH_COMPONENT_KEYRING_KMIP=ON    -> WOULD_ADD_SUBDIRECTORY
#        -DWITHOUT_COMPONENT_KEYRING_KMIP=1  -> SKIPPED
#        -DWITHOUT_COMPONENT_KEYRING_KMIP=0  -> SKIPPED
#
#      An earlier revision of this note had that backwards, on the argument
#      that defining the WITH_ symbol leaves it defined. That conflated
#      `DEFINED WITH_X` with `WITH_X`: defining it is exactly what makes the
#      AND false, and the OR's second operand is `WITH_X` evaluated as a
#      boolean, where OFF, 0 and NO are CMake false constants. The component's
#      own guard agrees — components/keyrings/keyring_kmip/CMakeLists.txt:28 is
#      `IF(NOT WITH_COMPONENT_KEYRING_KMIP) RETURN()`, keyed off the WITH_
#      spelling. Reasoning about a tool's semantics when the tool is installed
#      and the answer is one command away is the mirror image of this round's
#      other lesson about quoted hedges.
#
#      WITHOUT_ is still the better choice here, for the reason the last row of
#      that table shows: the gate tests `DEFINED WITHOUT_...` and never its
#      value, so WITHOUT_ closes it on definedness alone and cannot be broken
#      by a value spelling. WITH_=OFF relies on false-constant evaluation,
#      which is correct but is one more thing to be right about.
#
#      The alternative (listing them in percona.yml's `source.submodules:`, the
#      way mariadb.yml lists libmariadb) would make the build fetch from three
#      more hosts for plugins no benchmark here exercises. With the four flags
#      below this build needs no network beyond the base image and apt — the
#      same property docker/alisql/Dockerfile relies on.
#
#   3. BOOST NEEDS NO FLAG AND NO DOWNLOAD. cmake/boost.cmake:26-27 reads
#      "# Always use the bundled version." / SET(BOOST_SOURCE_DIR
#      ${CMAKE_SOURCE_DIR}/extra/boost), and the pinned tree carries
#      extra/boost/boost_1_87_0. So there is no -DWITH_BOOST and no
#      -DDOWNLOAD_BOOST here, deliberately.
#
#   4. NO mysql_native_password ANYWHERE. See init.sql — the plugin is gone from
#      the server in this base and requesting it is a hard error.
#

ARG BASE_IMAGE=ubuntu:24.04

# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS builder
# ---------------------------------------------------------------------------
ARG BUILD_TYPE=RelWithDebInfo
ARG MARCH=x86-64-v3
ARG CMAKE_FLAGS=""
ARG EXTRA_CXXFLAGS=""
ARG JOBS=0

# Same shape as the AliSQL builder stage. `file` is used by the strip pass at
# the end; `python3` is required by the MySQL build itself (comp_err and the
# generated-source steps), not by the benchmark.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake bison pkg-config \
      libssl-dev libncurses-dev libudev-dev libaio-dev \
      libnuma-dev zlib1g-dev libedit-dev \
      python3 file

WORKDIR /src
COPY source.tar /tmp/source.tar
RUN tar -xf /tmp/source.tar -C /src && rm -f /tmp/source.tar

# Flags mirror config/engines/percona.yml. -march is the only one that may ever
# differ between engines, and it must not: all seven engines compile SIMD
# distance kernels, and a different flag on one turns the benchmark into a
# comparison of compiler flags. The value is written to /opt/percona/.march so
# the ann overlay module can report which one the image actually used — see
# overlay/.../percona/module.py, which overrides _image_march() for exactly this.
RUN set -eux; \
    JOBS="${JOBS:-0}"; [ "$JOBS" -gt 0 ] 2>/dev/null || JOBS="$(nproc)"; \
    mkdir -p /build; \
    cd /build; \
    export CFLAGS="-O3 -g -fexceptions -fno-strict-aliasing -fno-omit-frame-pointer -march=${MARCH} ${EXTRA_CXXFLAGS}"; \
    export CXXFLAGS="$CFLAGS"; \
    cmake /src/source \
      -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
      -DCMAKE_INSTALL_PREFIX=/opt/percona \
      -DMYSQL_DATADIR=/var/lib/vbench/data \
      -DSYSCONFDIR=/opt/percona \
      -DMYSQL_SERVER_SUFFIX="-vector-bench" \
      -DWITH_COREDUMPER=OFF \
      -DWITH_AUTH_OPENID_CONNECT=OFF \
      -DWITH_ROCKSDB=OFF \
      -DWITHOUT_COMPONENT_KEYRING_KMIP=1 \
      -DWITH_NDB=OFF \
      -DWITH_ROUTER=OFF \
      -DWITH_MYSQLX=OFF \
      -DWITH_UNIT_TESTS=OFF \
      -DWITH_SSL=system \
      -DWITH_ZLIB=system \
      ${CMAKE_FLAGS} ; \
    make -j"$JOBS"; \
    make install; \
    echo "${MARCH}" > /opt/percona/.march

RUN find /opt/percona -type f -executable -exec file {} \; \
      | awk -F: '/ELF .* not stripped/{print $1}' | xargs -r strip --strip-debug

# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS runtime
# ---------------------------------------------------------------------------
LABEL org.opencontainers.image.title="vector-bench Percona Server runtime"
ARG PERCONA_TAG=unknown
ENV VB_ENGINE=percona \
    VB_ENGINE_TAG=${PERCONA_TAG} \
    VB_ROOT_DIR=/opt/percona \
    VB_DATA_DIR=/var/lib/vbench/data \
    VB_SOCKET=/var/run/vbench/percona.sock \
    PATH=/opt/percona/bin:$PATH

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      libssl3t64 libncurses6 libaio1t64 libnuma1 libudev1 zlib1g libedit2 \
      ca-certificates procps tzdata \
    && useradd --system --create-home --home-dir /var/lib/vbench --shell /usr/sbin/nologin mysql 2>/dev/null || true

COPY --from=builder /opt/percona /opt/percona
COPY entrypoint-percona.sh /usr/local/bin/vb-entrypoint
COPY init.sql /opt/vbench/init.sql
RUN chmod +x /usr/local/bin/vb-entrypoint \
    && mkdir -p "${VB_DATA_DIR}" /var/run/vbench \
    && chown -R mysql:mysql /var/lib/vbench /var/run/vbench \
    && chmod 0644 /opt/vbench/init.sql

ENV VB_INIT_SQL=/opt/vbench/init.sql \
    VB_DB_USER=bench \
    VB_DB_PASSWORD=bench

EXPOSE 3306
ENTRYPOINT ["/usr/local/bin/vb-entrypoint"]
CMD ["server"]

# ---------------------------------------------------------------------------
FROM runtime AS bench
# ---------------------------------------------------------------------------
LABEL org.opencontainers.image.title="vector-bench Percona Server + ann-benchmarks"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-dev build-essential git \
      libmariadb-dev libhdf5-dev pkg-config

# The same client stack as the MariaDB and AliSQL bench images, for the same
# reason: MariaDB Connector/C speaks the MySQL protocol, so all three
# MySQL-family engines are driven through an identical client with
# binary-protocol prepared statements, and differences in the numbers belong to
# the servers. requirements-mysql.txt pins mariadb==1.1.10 with --no-binary.
#
# MARIADB_CONFIG points at the distro's mariadb_config and not at the one on
# PATH, exactly as in the MariaDB and AliSQL images. Here PATH starts with
# /opt/percona/bin, which carries no mariadb_config at all, so the pin is
# belt-and-braces rather than load-bearing — but it keeps the three images
# compiling the extension against one Connector/C, which is the point.
#
# ONE UNPROVEN THING, and it is the auth handshake rather than the build:
# this base has no mysql_native_password (see init.sql), so the bench account
# authenticates with caching_sha2_password. Over the unix socket the ann pass
# uses (vb_mysql.py passes unix_socket=) that is a cleartext exchange the server
# permits. Over the TCP connection the ops harness uses
# (mysql_family.py passes host=/port=) it is not, and Connector/C must either
# negotiate TLS or fetch the server's RSA public key. docs/09 §7 item 2 lists this as
# the first thing the smoke run has to settle.
COPY requirements-bench.txt requirements-mysql.txt /tmp/
RUN MARIADB_CONFIG=/usr/bin/mariadb_config \
    pip3 install --no-cache-dir --break-system-packages \
        -r /tmp/requirements-bench.txt -r /tmp/requirements-mysql.txt \
    && rm -f /tmp/requirements-bench.txt /tmp/requirements-mysql.txt \
    && python3 -c "import mariadb; print('Connector/Python', mariadb.__version__, 'OK')"

WORKDIR /home/app
ENV PERCONA_ROOT_DIR=/opt/percona \
    PERCONA_DB_WORKSPACE=/var/lib/vbench \
    DO_INIT_PERCONA=yes
ENTRYPOINT ["python3", "-u", "run_algorithm.py"]
