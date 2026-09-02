# Web UI image.
#
# Not built from an engine bench image: this needs pyyaml, numpy, git and the
# Docker CLI, not matplotlib and h5py. Separate means it builds in a minute and
# does not go stale when an engine is rebuilt.
#
# The repo is bind-mounted at its own absolute host path at run time, and the
# container runs as the host user. Both matter -- see docs/08-web-ui.md.

# --- stage 1: the Docker CLI, verified -------------------------------------
#
# Only the client. The daemon stays on the host and is reached through its
# socket. The published tarball carries dockerd, containerd and runc as well;
# none of them belong in this image, so exactly one binary is extracted.
#
# Checksums are pinned rather than trusted from the wire. There is no published
# .sha256 sidecar for these artifacts, so a corrupt or truncated download would
# otherwise be baked into an image and only surface as a confusing runtime
# failure. Update both when DOCKER_CLI_VERSION changes.
FROM debian:bookworm-slim AS cli

ARG DOCKER_CLI_VERSION=27.3.1
ARG SHA256_X86_64=9b4f6fe406e50f9085ee474c451e2bb5adb119a03591f467922d3b4e2ddf31d3
ARG SHA256_AARCH64=4da6a6c7502b7ab561675a5ff5ac192d9b49d76d0b8847cf17ade246122279f4

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    case "$(uname -m)" in \
      x86_64)  arch=x86_64;  sha="${SHA256_X86_64}" ;; \
      aarch64) arch=aarch64; sha="${SHA256_AARCH64}" ;; \
      *) echo "unsupported architecture: $(uname -m)" >&2; exit 1 ;; \
    esac; \
    curl -fsSL --retry 3 -o /tmp/docker.tgz \
      "https://download.docker.com/linux/static/stable/${arch}/docker-${DOCKER_CLI_VERSION}.tgz"; \
    echo "${sha}  /tmp/docker.tgz" | sha256sum -c -; \
    tar -xzf /tmp/docker.tgz -C /tmp docker/docker; \
    install -D -m 0755 /tmp/docker/docker /out/docker; \
    /out/docker --version

# --- stage 2: the runtime ---------------------------------------------------
FROM debian:bookworm-slim

LABEL org.opencontainers.image.title="vector-bench web UI" \
      org.opencontainers.image.description="Run browser, record explorer and run launcher for vector-bench" \
      org.opencontainers.image.source="https://github.com/EvgeniyPatlan/vector-bench" \
      org.opencontainers.image.licenses="MIT"

# python3-numpy is for report/loaders.py, which the record explorer reuses.
# git is not optional: run-benchmark.sh re-applies the ann-benchmarks overlay on
# every run, and prepare-harness.sh needs git to do it.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-yaml python3-numpy git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=cli /out/docker /usr/local/bin/docker

# The repo arrives as a bind mount owned by the host user, and git refuses a
# repository owned by someone else. `run-benchmark.sh web` runs this container
# as that same user, which resolves it properly; this is the safety net for
# when it is not, and applies to every uid rather than one.
RUN git config --system --add safe.directory '*'

# A default that is not root. `run-benchmark.sh web` overrides it with the
# invoking user's uid so writes to the bind mount land correctly owned; a bare
# `docker run` gets an unprivileged user that can read but not write, which
# fails visibly rather than leaving root-owned files behind.
RUN useradd --system --uid 10001 --create-home --home-dir /home/vb vb
USER 10001
ENV HOME=/home/vb

ENV PYTHONUNBUFFERED=1
EXPOSE 8080

# No curl in this stage; python3 is already here and /api/health needs no
# authentication precisely so this can reach it.
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python3 -c "import sys,urllib.request; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/api/health', timeout=3).status == 200 else 1)"

ENTRYPOINT ["python3", "-m", "webui.server"]
CMD ["--host", "0.0.0.0", "--port", "8080"]
