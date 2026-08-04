#
# PostgreSQL 17 + pgvector, built from a pinned pgvector release tag.
#
# Targets:
#   runtime  — stock postgres:17 plus the compiled pgvector extension.
#   bench    — runtime + the Python stack ann-benchmarks drives.
#
# Build context is work/buildctx/pgvector/, produced by scripts/prepare-sources.sh.

ARG BASE_IMAGE=postgres:17-bookworm

# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS builder
# ---------------------------------------------------------------------------
ARG MARCH=x86-64-v3
ARG OPTFLAGS="-O3 -fno-semantic-interposition"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential clang llvm postgresql-server-dev-17

WORKDIR /src
COPY source.tar /tmp/source.tar
RUN tar -xf /tmp/source.tar -C /src && rm -f /tmp/source.tar

# pgvector's distance kernels are the hot loop, same as the MHNSW/VIDX ones.
# The -march used here must match what the other engines got, or the comparison
# silently becomes a comparison of compiler flags.
RUN set -eux; \
    cd /src/source; \
    make clean; \
    make OPTFLAGS="${OPTFLAGS} -march=${MARCH}" -j"$(nproc)"; \
    make install; \
    mkdir -p /opt/pgvector-artifacts; \
    echo "${MARCH}" > /opt/pgvector-artifacts/.march

# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS runtime
# ---------------------------------------------------------------------------
LABEL org.opencontainers.image.title="vector-bench PostgreSQL + pgvector runtime"
ARG PGVECTOR_TAG=unknown
ENV VB_ENGINE=pgvector \
    VB_ENGINE_TAG=${PGVECTOR_TAG} \
    VB_DATA_DIR=/var/lib/postgresql/data \
    PGDATA=/var/lib/postgresql/data

# Only the built extension is carried over — no compiler in the runtime image.
COPY --from=builder /usr/lib/postgresql/17/lib/vector.so /usr/lib/postgresql/17/lib/
COPY --from=builder /usr/share/postgresql/17/extension/vector* /usr/share/postgresql/17/extension/
COPY --from=builder /opt/pgvector-artifacts/.march /opt/pgvector-artifacts/.march

COPY entrypoint-pgvector.sh /usr/local/bin/vb-entrypoint
RUN chmod +x /usr/local/bin/vb-entrypoint

ENV POSTGRES_HOST_AUTH_METHOD=trust \
    POSTGRES_DB=ann \
    POSTGRES_USER=postgres
EXPOSE 5432
ENTRYPOINT ["/usr/local/bin/vb-entrypoint"]
CMD ["server"]

# ---------------------------------------------------------------------------
FROM runtime AS bench
# ---------------------------------------------------------------------------
LABEL org.opencontainers.image.title="vector-bench PostgreSQL + pgvector + ann-benchmarks"
USER root

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-dev build-essential git libhdf5-dev pkg-config

# PostgreSQL necessarily uses a different client stack (psycopg3) from the
# MySQL-family engines. That is an unavoidable asymmetry and is called out in
# docs/05-methodology.md; psycopg3's binary protocol is the closest equivalent
# to Connector/C's, so it is what we use.
COPY requirements-bench.txt /tmp/requirements-bench.txt
RUN pip3 install --no-cache-dir --break-system-packages -r /tmp/requirements-bench.txt \
    && pip3 install --no-cache-dir --break-system-packages "psycopg[binary]>=3.1" pgvector \
    && rm -f /tmp/requirements-bench.txt

WORKDIR /home/app
ENTRYPOINT ["python3", "-u", "run_algorithm.py"]
