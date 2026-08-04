#!/usr/bin/env bash
#
# Download the HDF5 datasets (train, test and exact ground truth) into
# vector-bench/datasets/.
#
# The same files feed both measurement paths, so ann-benchmarks and the ops
# harness score against identical ground truth. Downloads are resumable and
# verified by size, because a truncated 3.6 GB file that still parses as HDF5
# would silently produce a benchmark over partial data.
#
# Usage:
#   scripts/fetch-datasets.sh [--datasets a,b,c] [--all] [--list]

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

BASE_URL="${VB_DATASET_BASE_URL:-http://ann-benchmarks.com}"

# name|approximate size in bytes|role
DATASETS=(
  "fashion-mnist-784-euclidean|227598568|smoke — 60k x 784, euclidean"
  "glove-100-angular|485413888|main  — 1.18M x 100, angular"
  "sift-128-euclidean|525437440|main  — 1M x 128, euclidean"
  "gist-960-euclidean|3771875328|stress — 1M x 960, euclidean"
  "glove-25-angular|126959616|optional — 1.18M x 25, angular"
  "deep-image-96-angular|3843883008|optional — 9.99M x 96, angular"
)

WANTED=""
LIST=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --datasets) WANTED="$2"; shift 2 ;;
    --datasets=*) WANTED="${1#*=}"; shift ;;
    --all) WANTED="all"; shift ;;
    --list) LIST=1; shift ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

if [[ $LIST -eq 1 ]]; then
  printf '%-32s %10s  %s\n' "DATASET" "SIZE" "ROLE"
  for entry in "${DATASETS[@]}"; do
    IFS='|' read -r name size role <<<"$entry"
    printf '%-32s %10s  %s\n' "$name" "$(human_bytes "$size")" "$role"
  done
  exit 0
fi

need_cmd curl
mkdir -p "$VB_DATASETS"
assert_not_vendor_repo "$VB_DATASETS"

# Default to the four datasets the profiles actually use.
if [[ -z "$WANTED" ]]; then
  WANTED="fashion-mnist-784-euclidean,glove-100-angular,sift-128-euclidean,gist-960-euclidean"
elif [[ "$WANTED" == "all" ]]; then
  WANTED="$(printf '%s\n' "${DATASETS[@]}" | cut -d'|' -f1 | paste -sd,)"
fi

lookup_size() {
  local want="$1" entry name size role
  for entry in "${DATASETS[@]}"; do
    IFS='|' read -r name size role <<<"$entry"
    [[ "$name" == "$want" ]] && { printf '%s' "$size"; return 0; }
  done
  printf '0'
}

# Check free space before starting: running out mid-download of a 3.6 GB file
# wastes far more time than the check costs.
total_needed=0
IFS=',' read -ra NAMES <<<"$WANTED"
for name in "${NAMES[@]}"; do
  [[ -f "$VB_DATASETS/${name}.hdf5" ]] && continue
  total_needed=$(( total_needed + $(lookup_size "$name") ))
done

if [[ $total_needed -gt 0 ]]; then
  avail=$(df -B1 --output=avail "$VB_DATASETS" | tail -1 | tr -d ' ')
  info "need ~$(human_bytes "$total_needed"), $(human_bytes "$avail") free"
  # Index data needs several times the dataset size on top.
  if (( avail < total_needed * 3 )); then
    warn "less than 3x the dataset size is free. Index data for these engines "
    warn "typically needs 2-4x the raw vectors; the run may fail on disk space."
  fi
fi

for name in "${NAMES[@]}"; do
  name="$(echo "$name" | xargs)"
  [[ -n "$name" ]] || continue
  target="$VB_DATASETS/${name}.hdf5"
  expected="$(lookup_size "$name")"

  if [[ -f "$target" ]]; then
    actual=$(stat -c%s "$target")
    # Tolerate small differences: the published sizes are approximate.
    if [[ "$expected" == "0" ]] || (( actual > expected * 95 / 100 )); then
      ok "$name already present ($(human_bytes "$actual"))"
      continue
    fi
    warn "$name looks truncated ($(human_bytes "$actual") of ~$(human_bytes "$expected")); re-downloading"
    rm -f "$target"
  fi

  info "downloading $name (~$(human_bytes "$expected"))"
  # --continue-at resumes a partial download; the temp name means an aborted
  # transfer can never be mistaken for a complete dataset.
  curl --fail --location --progress-bar --continue-at - \
       --output "$target.part" "$BASE_URL/${name}.hdf5" \
    || die "download failed for $name"
  mv "$target.part" "$target"
  ok "$name -> $target ($(human_bytes "$(stat -c%s "$target")"))"
done

ok "datasets ready in $VB_DATASETS"
