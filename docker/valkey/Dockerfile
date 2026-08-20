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
ARG VALKEY_PACKAGES="percona-valkey-server percona-valkey-tools percona-valkey-search"
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

# Ask dpkg where the module actually is rather than hardcoding a path. Guessing
# it costs a full build to find out, and the answer is already in the package
# manifest. Recorded in the image so the entrypoint does not have to search.
RUN set -eux; \
    module="$(dpkg -L percona-valkey-search | grep -E '\.so$' | head -1)"; \
    if [ -z "$module" ]; then \
      echo "ERROR: percona-valkey-search installed no shared object." >&2; \
      echo "Files it did install:" >&2; \
      dpkg -L percona-valkey-search >&2; \
      exit 1; \
    fi; \
    echo "$module" > /opt/valkey-artifacts/.module_path; \
    echo "search module: $module"

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
# pip's --break-system-packages does not exist before pip 23. This image is
# el9-based and ships an older one, where the flag is a hard error rather than
# a no-op. Detected instead of assumed, so the same Dockerfile works whichever
# base it is built on.
RUN set -eux; \
    flags=""; \
    if pip3 install --help 2>&1 | grep -q -- --break-system-packages; then \
      flags="--break-system-packages"; \
    fi; \
    pip3 install --no-cache-dir $flags \
        -r /tmp/requirements-bench.txt -r /tmp/requirements-valkey.txt \
    && rm -f /tmp/requirements-bench.txt /tmp/requirements-valkey.txt
