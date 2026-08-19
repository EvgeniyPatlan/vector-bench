#
# Valkey + valkey-search, installed from Percona's valkey-91 repository.
#
# Targets:
#   runtime  — Percona's Valkey packages plus our entrypoint. Nothing compiled.
#   bench    — runtime + the Python stack ann-benchmarks drives, plus the client.
#
# Build context is work/buildctx/valkey/, produced by scripts/prepare-sources.sh.
#
# PROVENANCE
#
# This engine's provenance is a package, not a git tag or an image digest.
# "whatever valkey-91 held on the day we built" is not a reproducible
# statement, so the installed versions are read back after installation and
# written into the image, where the manifest picks them up.

ARG BASE_IMAGE=debian:bookworm-slim
ARG VALKEY_REPO=valkey-91

# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS runtime
# ---------------------------------------------------------------------------
ARG VALKEY_REPO
ENV DEBIAN_FRONTEND=noninteractive \
    VB_ENGINE=valkey \
    VB_ENGINE_FAMILY=valkey

RUN apt-get update && apt-get install -y --no-install-recommends \
      curl ca-certificates gnupg lsb-release procps \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    curl -fsSL -o /tmp/percona-release.deb \
      https://repo.percona.com/apt/percona-release_latest.generic_all.deb; \
    apt-get update; \
    apt-get install -y --no-install-recommends /tmp/percona-release.deb; \
    rm -f /tmp/percona-release.deb; \
    percona-release enable "${VALKEY_REPO}" release; \
    apt-get update

# Package names come from the engine config rather than being hardcoded here,
# and a name that does not resolve fails with the list of what the repository
# actually offers. Discovering that after the corpus is loaded costs hours;
# discovering it here costs a rebuild.
ARG VALKEY_PACKAGES="percona-valkey percona-valkey-search"
RUN set -eux; \
    for pkg in ${VALKEY_PACKAGES}; do \
      if ! apt-cache show "$pkg" >/dev/null 2>&1; then \
        echo "ERROR: package '$pkg' is not in the ${VALKEY_REPO} repository." >&2; \
        echo "Packages matching 'valkey' that ARE available:" >&2; \
        apt-cache search valkey >&2 || true; \
        exit 1; \
      fi; \
    done; \
    apt-get install -y --no-install-recommends ${VALKEY_PACKAGES}; \
    rm -rf /var/lib/apt/lists/*

# Read back what actually got installed. This is the engine's provenance and
# the manifest reports it where the others report a tag and a commit.
RUN set -eux; \
    mkdir -p /opt/valkey-artifacts; \
    for pkg in ${VALKEY_PACKAGES}; do \
      printf '%s=%s\n' "$pkg" "$(dpkg-query -W -f='${Version}' "$pkg")"; \
    done > /opt/valkey-artifacts/.packages; \
    cat /opt/valkey-artifacts/.packages

COPY entrypoint-valkey.sh /usr/local/bin/vb-entrypoint
RUN chmod +x /usr/local/bin/vb-entrypoint

ENV VB_VALKEY_PORT=6379 \
    VB_MODULE_PATH=/usr/lib/valkey/modules/libsearch.so
EXPOSE 6379
ENTRYPOINT ["/usr/local/bin/vb-entrypoint"]
CMD ["server"]

# ---------------------------------------------------------------------------
FROM runtime AS bench
# ---------------------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-bench.txt /tmp/requirements-bench.txt
COPY requirements-valkey.txt /tmp/requirements-valkey.txt
RUN pip3 install --no-cache-dir --break-system-packages \
        -r /tmp/requirements-bench.txt -r /tmp/requirements-valkey.txt \
    && rm -f /tmp/requirements-bench.txt /tmp/requirements-valkey.txt
