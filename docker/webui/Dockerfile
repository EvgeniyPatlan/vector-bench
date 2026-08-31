# Web UI image.
#
# Deliberately not built from an engine bench image: the UI needs pyyaml and
# numpy, not the whole scientific stack, and it needs the docker CLI, which no
# engine image carries. Keeping it separate means it builds in under a minute
# and does not go stale when an engine is rebuilt.
#
# The repo is bind-mounted at its own absolute host path at run time. The
# orchestrator hands host paths to the Docker daemon, so a container path here
# would resolve to nothing on the host and mount an empty directory.
FROM ubuntu:24.04

ARG DOCKER_CLI_VERSION=27.3.1

# git is not optional: run-benchmark.sh re-applies the ann-benchmarks overlay
# on every run, and prepare-harness.sh needs git to do it.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-yaml python3-numpy git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# CLI only: the daemon stays on the host and is reached through its socket.
RUN set -eux; \
    case "$(uname -m)" in \
      x86_64)  arch=x86_64 ;; \
      aarch64) arch=aarch64 ;; \
      *) echo "unsupported architecture: $(uname -m)" >&2; exit 1 ;; \
    esac; \
    curl -fsSL "https://download.docker.com/linux/static/stable/${arch}/docker-${DOCKER_CLI_VERSION}.tgz" \
      | tar -xz -C /tmp docker/docker; \
    mv /tmp/docker/docker /usr/local/bin/docker; \
    rm -rf /tmp/docker; \
    docker --version

# The repo arrives as a bind mount owned by the host user. git refuses to touch
# a repository owned by someone else, and run-benchmark.sh runs git on every run
# to re-apply the ann-benchmarks overlay. The container is normally run as that
# same host user, which resolves it; this is the safety net for when it is not,
# and applies to every uid rather than one.
RUN git config --system --add safe.directory '*'

ENV PYTHONUNBUFFERED=1
EXPOSE 8080

ENTRYPOINT ["python3", "-m", "webui.server"]
CMD ["--host", "0.0.0.0", "--port", "8080"]
