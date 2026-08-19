#
# Percona Server for MongoDB + Percona Search (mongot).
#
# Targets:
#   runtime  — the published PSMDB image plus the mongot binary and our
#              entrypoint. Nothing is compiled here.
#   bench    — runtime + the Python stack ann-benchmarks drives, plus pymongo.
#
# Build context is work/buildctx/mongodb/, produced by scripts/prepare-sources.sh.
#
# WHY THIS ONE DOES NOT BUILD FROM SOURCE
#
# Every other engine in this framework is compiled from a pinned upstream tag
# with a -march we control, so that a difference between two engines is a
# difference between the engines rather than between compiler flags. mongot is
# not published as source: Percona ships tarballs, packages and images only, and
# it runs on the JVM, so its distance kernels come from the JVM's vector API
# rather than from anything we pass to a compiler.
#
# The provenance that replaces a tag and a commit is the image digest, pinned by
# scripts/prepare-sources.sh and recorded in the run manifest alongside the JVM
# version. A run that cannot resolve a digest is not reproducible in the one
# dimension this framework exists to guarantee, so the build fails rather than
# floating on a tag.

ARG BASE_IMAGE=percona/percona-server-mongodb:8.3
ARG MONGOT_IMAGE=percona/percona-server-mongodb-mongot:1.70.3-1

# ---------------------------------------------------------------------------
FROM ${MONGOT_IMAGE} AS mongot
# ---------------------------------------------------------------------------
# Present only so the runtime stage can copy the binary and its JVM out of it.

# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS runtime
# ---------------------------------------------------------------------------
ARG MONGOT_IMAGE
USER root

ENV VB_ENGINE=mongodb \
    VB_ENGINE_FAMILY=mongo \
    VB_MONGOT_IMAGE=${MONGOT_IMAGE}

# mongot and the JRE it runs on. Copied rather than installed so the runtime
# image carries exactly the build the digest names.
COPY --from=mongot /opt/mongot /opt/mongot
COPY --from=mongot /usr/lib/jvm /usr/lib/jvm

ENV PATH="/opt/mongot/bin:${PATH}" \
    JAVA_HOME=/usr/lib/jvm/default-jvm

COPY entrypoint-mongodb.sh /usr/local/bin/vb-entrypoint
RUN chmod +x /usr/local/bin/vb-entrypoint

# Percona Search requires an initiated replica set; standalone is explicitly
# unsupported. Single node keeps the oplog write amplification that the other
# engines do not pay down to its minimum.
ENV VB_REPLICA_SET=rs0 \
    VB_MONGOT_PORT=27028 \
    VB_MONGOT_HEAP_GB=8

EXPOSE 27017 27028
ENTRYPOINT ["/usr/local/bin/vb-entrypoint"]
CMD ["server"]

# ---------------------------------------------------------------------------
FROM runtime AS bench
# ---------------------------------------------------------------------------
USER root
ENV DEBIAN_FRONTEND=noninteractive

RUN set -eux; \
    if command -v microdnf >/dev/null 2>&1; then \
      microdnf install -y python3 python3-pip; microdnf clean all; \
    elif command -v dnf >/dev/null 2>&1; then \
      dnf install -y python3 python3-pip; dnf clean all; \
    else \
      apt-get update && apt-get install -y --no-install-recommends python3 python3-pip \
        && rm -rf /var/lib/apt/lists/*; \
    fi

COPY requirements-bench.txt /tmp/requirements-bench.txt
COPY requirements-mongo.txt /tmp/requirements-mongo.txt
RUN pip3 install --no-cache-dir --break-system-packages \
        -r /tmp/requirements-bench.txt -r /tmp/requirements-mongo.txt \
    && rm -f /tmp/requirements-bench.txt /tmp/requirements-mongo.txt
