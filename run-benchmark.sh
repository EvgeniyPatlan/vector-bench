#!/usr/bin/env bash
#
# vector-bench — single entrypoint.
#
#   ./run-benchmark.sh build                            build all three engine images
#   ./run-benchmark.sh fetch --datasets glove-100-angular
#   ./run-benchmark.sh run --profile smoke              validate the pipeline (~15 min)
#   ./run-benchmark.sh run --profile full --resource-pass both
#   ./run-benchmark.sh run --profile quick --engines mariadb,alisql --phases ops
#   ./run-benchmark.sh report --run-dir results/full-20260803-120000
#   ./run-benchmark.sh clean --run-id full-20260803-120000
#
# Run `./run-benchmark.sh <subcommand> --help` for the full option list.
#
# Requirements on the host: docker, python3 with PyYAML, git. Nothing else —
# the database clients and the scientific Python stack live inside the images,
# so running a benchmark does not modify the machine being benchmarked.

set -euo pipefail

VB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$VB_ROOT"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required on the host" >&2
  exit 1
fi

if ! python3 -c 'import yaml' >/dev/null 2>&1; then
  cat >&2 <<'EOF'
PyYAML is required on the host but is not installed.

  Debian/Ubuntu : sudo apt-get install -y python3-yaml
  RHEL/Fedora   : sudo dnf install -y python3-pyyaml
  any           : python3 -m pip install --user pyyaml

This is the framework's only Python dependency on the host; everything else
runs inside the engine images.
EOF
  exit 1
fi

# Refresh the ann-benchmarks working copy on EVERY run, not just the first.
#
# It is a disposable clone of the vendor checkout with our overlay copied over
# it. Doing this only when the clone was missing meant that after a `git pull`
# touching overlay/, the working copy silently kept the old algorithm modules —
# you would pull a fix and then not run it, with nothing to indicate why. The
# script reuses an existing clone and just re-applies the overlay, so this costs
# well under a second.
case "${1:-}" in
  run|render)
    "$VB_ROOT/scripts/prepare-harness.sh" >/dev/null || {
      echo "failed to prepare the ann-benchmarks working copy" >&2; exit 1; }
    ;;
esac

exec python3 -m orchestrator.cli "$@"
