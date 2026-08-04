#
# AliSQL (MySQL 8.0.44 base + VIDX vector index), built from a pinned tag.
#
# Targets:
#   runtime  — server binaries only. Use this to run AliSQL by hand
#              (see docs/03-running-manually.md).
#   bench    — runtime + the Python stack ann-benchmarks drives.
#
# Build context is work/buildctx/alisql/, produced by scripts/prepare-sources.sh.
#
# AliSQL vendors every third-party dependency it needs (including boost 1.77 at
# extra/boost/boost_1_77_0) and has no git submodules, so this build needs no
# network beyond the base image and apt.

ARG BASE_IMAGE=ubuntu:24.04

# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS builder
# ---------------------------------------------------------------------------
ARG BUILD_TYPE=RelWithDebInfo
ARG MARCH=x86-64-v3
ARG CMAKE_FLAGS=""
ARG EXTRA_CXXFLAGS=""
ARG JOBS=0

# git and ninja-build are for the bundled DuckDB. AliSQL's CMake calls
# MYSQL_CHECK_DUCKDB() unconditionally on Linux, so extra/duckdb is compiled as
# an ExternalProject on every build and its Makefile probes for both.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake bison pkg-config \
      libssl-dev libncurses-dev libudev-dev libaio-dev \
      libnuma-dev zlib1g-dev \
      python3 file \
      git ninja-build

WORKDIR /src
COPY source.tar /tmp/source.tar
RUN tar -xf /tmp/source.tar -C /src && rm -f /tmp/source.tar

# AliSQL's own build.sh forces an in-source build, so cmake runs in the tree.
# Flags mirror its release configuration; see config/engines/alisql.yml.
RUN set -eux; \
    JOBS="${JOBS:-0}"; [ "$JOBS" -gt 0 ] 2>/dev/null || JOBS="$(nproc)"; \
    cd /src/source; \
    export CFLAGS="-O3 -g -fexceptions -fno-strict-aliasing -fno-omit-frame-pointer -march=${MARCH} ${EXTRA_CXXFLAGS}"; \
    export CXXFLAGS="$CFLAGS"; \
    cmake . \
      -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
      -DCMAKE_INSTALL_PREFIX=/opt/alisql \
      -DMYSQL_DATADIR=/var/lib/vbench/data \
      -DSYSCONFDIR=/opt/alisql \
      -DMYSQL_SERVER_SUFFIX="-vector-bench" \
      ${CMAKE_FLAGS} ; \
    make -j"$JOBS"; \
    make install; \
    echo "${MARCH}" > /opt/alisql/.march

RUN find /opt/alisql -type f -executable -exec file {} \; \
      | awk -F: '/ELF .* not stripped/{print $1}' | xargs -r strip --strip-debug

# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS runtime
# ---------------------------------------------------------------------------
LABEL org.opencontainers.image.title="vector-bench AliSQL runtime"
ARG ALISQL_TAG=unknown
ENV VB_ENGINE=alisql \
    VB_ENGINE_TAG=${ALISQL_TAG} \
    VB_ROOT_DIR=/opt/alisql \
    VB_DATA_DIR=/var/lib/vbench/data \
    VB_SOCKET=/var/run/vbench/alisql.sock \
    PATH=/opt/alisql/bin:$PATH

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      libssl3t64 libncurses6 libaio1t64 libnuma1 libudev1 zlib1g \
      ca-certificates procps tzdata \
    && useradd --system --create-home --home-dir /var/lib/vbench --shell /usr/sbin/nologin mysql 2>/dev/null || true

COPY --from=builder /opt/alisql /opt/alisql
COPY entrypoint-alisql.sh /usr/local/bin/vb-entrypoint
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
LABEL org.opencontainers.image.title="vector-bench AliSQL + ann-benchmarks"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-dev build-essential git \
      libmariadb-dev libhdf5-dev pkg-config

# Same client stack as the MariaDB bench image on purpose: MariaDB Connector/C
# speaks the MySQL protocol, so both MySQL-family engines are driven through an
# identical driver with binary-protocol prepared statements. Differences in the
# numbers then belong to the servers, not to the clients.
COPY requirements-bench.txt requirements-mysql.txt /tmp/
# MARIADB_CONFIG must point at the distro's mariadb_config, not at the one on
# PATH. In the MariaDB image PATH starts with /opt/mariadb/bin, whose
# mariadb_config describes the server's own bundled Connector/C (3.4.9 in
# 11.8.8) and links against /opt/mariadb/lib — which is not in the loader path,
# so at runtime the distro's libmariadb.so.3 (3.3.17) is loaded instead and the
# extension refuses to import. Pinning both images to the distro's Connector/C
# fixes that and keeps the two MySQL-family engines on a genuinely identical
# client, which is the whole point of sharing one driver.
RUN MARIADB_CONFIG=/usr/bin/mariadb_config \
    pip3 install --no-cache-dir --break-system-packages \
        -r /tmp/requirements-bench.txt -r /tmp/requirements-mysql.txt \
    && rm -f /tmp/requirements-bench.txt /tmp/requirements-mysql.txt \
    && python3 -c "import mariadb; print('Connector/Python', mariadb.__version__, 'OK')"


WORKDIR /home/app
ENV ALISQL_ROOT_DIR=/opt/alisql \
    ALISQL_DB_WORKSPACE=/var/lib/vbench \
    DO_INIT_ALISQL=yes
ENTRYPOINT ["python3", "-u", "run_algorithm.py"]
