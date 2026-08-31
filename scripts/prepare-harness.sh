#!/usr/bin/env bash
#
# Materialise the ann-benchmarks working copy under work/ann-benchmarks and
# apply the vector-bench overlay on top of it.
#
# The vendor ann-benchmarks checkout is READ-ONLY. We clone from it (a local
# clone, so objects are hardlinked and cost essentially no disk) and then copy
# our algorithm modules and patches into the clone. Everything we change lives
# in overlay/ and is version-controlled with vector-bench, so the working copy
# is disposable and can be regenerated at any time with --force.
#
# Usage: scripts/prepare-harness.sh [--force]

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

FORCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force) FORCE=1; shift ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

need_cmd git python3
WORKDIR="$VB_WORK/ann-benchmarks"
assert_not_vendor_repo "$WORKDIR"

UPSTREAM_ANNB="${VB_UPSTREAM_ANNB:-https://github.com/erikbern/ann-benchmarks.git}"

if [[ $FORCE -eq 1 ]]; then
  info "removing existing working copy at $WORKDIR"
  rm -rf "$WORKDIR"
fi

if [[ ! -d "$WORKDIR/.git" ]]; then
  mkdir -p "$VB_WORK"
  if [[ -d "$VB_REPO_ANNB/.git" ]]; then
    info "cloning ann-benchmarks working copy from $VB_REPO_ANNB (read-only source)"
    git clone --quiet "$VB_REPO_ANNB" "$WORKDIR"
  else
    warn "local ann-benchmarks repo not found at $VB_REPO_ANNB; cloning upstream"
    git clone --quiet --depth 1 "$UPSTREAM_ANNB" "$WORKDIR"
  fi
else
  info "reusing working copy at $WORKDIR"
fi

ANNB_SHA="$(git -C "$WORKDIR" rev-parse HEAD)"

# ---------------------------------------------------------------------------
# Apply the overlay
# ---------------------------------------------------------------------------

OVL="$VB_OVERLAY/ann-benchmarks"

info "applying overlay modules"
if [[ -d "$OVL/ann_benchmarks" ]]; then
  # -a preserves timestamps so ann-benchmarks' own caching stays sane.
  cp -a "$OVL/ann_benchmarks/." "$WORKDIR/ann_benchmarks/"
fi

# Patches are applied idempotently: each is checked with --reverse --check first
# so a re-run on an already-patched tree is a no-op instead of a failure.
if compgen -G "$OVL/patches/*.patch" >/dev/null; then
  for p in "$OVL/patches"/*.patch; do
    if git -C "$WORKDIR" apply --reverse --check "$p" >/dev/null 2>&1; then
      log "patch already applied: $(basename "$p")"
    elif git -C "$WORKDIR" apply --check "$p" >/dev/null 2>&1; then
      git -C "$WORKDIR" apply "$p"
      ok "applied patch: $(basename "$p")"
    else
      die "patch does not apply cleanly against ann-benchmarks $ANNB_SHA: $(basename "$p")"
    fi
  done
fi

# ann-benchmarks resolves data/ and results/ relative to its own root. These are
# left as REAL directories, not symlinks: the orchestrator bind-mounts the
# shared datasets/ and results/annb/ over them inside the container, and a
# symlink at a bind-mount target resolves on the host side in ways that are
# easy to get subtly wrong. Real directories make the mount unambiguous.
mkdir -p "$WORKDIR/data" "$WORKDIR/results"
# $VB_SOURCES holds the provenance record written below. It exists as soon as
# anything has been built, which is why this went unnoticed -- but on a checkout
# where only the webui image was built it does not, and writing the record
# failed with a bare FileNotFoundError.
mkdir -p "$VB_DATASETS" "$VB_RESULTS/annb" "$VB_SOURCES"

python3 - "$VB_SOURCES/annbench.source.json" "$ANNB_SHA" "$VB_REPO_ANNB" <<'PY'
import json, sys
out, sha, origin = sys.argv[1:4]
json.dump({"component": "ann-benchmarks", "commit": sha, "origin": origin},
          open(out, "w"), indent=2, sort_keys=True)
open(out, "a").write("\n")
PY

ok "ann-benchmarks working copy ready at $WORKDIR (base $ANNB_SHA)"
