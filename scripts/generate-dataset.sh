#!/usr/bin/env bash
#
# Generate a dataset that ann-benchmarks builds locally rather than publishing
# as a prebuilt HDF5.
#
# Most datasets are downloaded by scripts/fetch-datasets.sh from
# ann-benchmarks.com. A few are not published there and must be constructed —
# notably `dbpedia-openai-*-angular`, which is assembled from a HuggingFace
# dataset and is the corpus MariaDB used for its big-vector-search benchmark.
#
# Generation needs the HuggingFace `datasets` package, which is heavy and has no
# place in the engine images, so it is installed into a throwaway container here
# instead.
#
# Ground truth is computed by brute force inside ann-benchmarks' write_output().
# At a million vectors of 1536 dimensions that is a large, slow, memory-hungry
# step — budget hours and tens of GB, and see the warning printed below.
#
# Usage:
#   scripts/generate-dataset.sh dbpedia-openai-1000k-angular
#   scripts/generate-dataset.sh --list

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

if [[ "${1:-}" == "--list" ]]; then
  need_docker
  info "datasets ann-benchmarks can generate locally (not on ann-benchmarks.com)"
  cat <<'EOF'
  dbpedia-openai-100k-angular   100,000 x 1536   OpenAI embeddings of DBpedia
  dbpedia-openai-200k-angular   200,000 x 1536
  ...                           (100k steps)
  dbpedia-openai-1000k-angular  1,000,000 x 1536  <- MariaDB's big-vector benchmark
EOF
  exit 0
fi

DATASET="${1:-}"
[[ -n "$DATASET" ]] || die "usage: $0 <dataset-name> | --list"

need_docker
mkdir -p "$VB_DATASETS"
assert_not_vendor_repo "$VB_DATASETS"

WORKDIR="$VB_WORK/ann-benchmarks"
[[ -d "$WORKDIR" ]] || die "ann-benchmarks working copy missing; run scripts/prepare-harness.sh"

TARGET="$VB_DATASETS/${DATASET}.hdf5"
if [[ -f "$TARGET" ]]; then
  ok "$DATASET already present ($(human_bytes "$(stat -c%s "$TARGET")"))"
  exit 0
fi

# Pick any bench image — they all carry numpy, h5py and scikit-learn.
IMAGE=""
for candidate in vector-bench/pgvector-bench vector-bench/mariadb-bench vector-bench/alisql-bench; do
  if image_exists "$candidate"; then IMAGE="$candidate"; break; fi
done
[[ -n "$IMAGE" ]] || die "no bench image found; run: ./run-benchmark.sh build"

avail=$(df -B1 --output=avail "$VB_DATASETS" | tail -1 | tr -d ' ')
warn "Generating '$DATASET'. This is not a plain download — it fetches the source"
warn "corpus and then computes exact ground truth by brute force."
warn ""
warn "For the dbpedia family: ann-benchmarks downloads the FULL 1M-row HuggingFace"
warn "dataset and then selects the first N rows. Choosing a smaller variant does"
warn "NOT reduce the download (~6-10 GB); it only shrinks the ground-truth"
warn "computation and every later engine load."
warn ""
warn "Budget hours of computation for the 1000k variant and ~20 GB of working"
warn "space. You have $(human_bytes "$avail") free."
info "using $IMAGE"

# --network is required here, unlike every other container this framework runs:
# the source data comes from HuggingFace.
docker run --rm \
  --name "vb-gends-$$" \
  -v "$WORKDIR:/home/app:rw" \
  -v "$VB_DATASETS:/home/app/data:rw" \
  -e HF_HUB_DISABLE_TELEMETRY=1 \
  -e PYTHONUNBUFFERED=1 \
  --entrypoint bash \
  "$IMAGE" -c '
set -e
python3 -c "import datasets" 2>/dev/null || {
  echo "[gen] installing the HuggingFace datasets package into this throwaway container"
  pip3 install --quiet --no-cache-dir --break-system-packages "datasets>=2.14" "pyarrow>=12"
}
cd /home/app
echo "[gen] building '"$DATASET"' — download, split, then brute-force ground truth"
python3 create_dataset.py --dataset '"$DATASET"'
' || die "generation failed for $DATASET"

[[ -f "$TARGET" ]] || die "generation reported success but $TARGET is missing"

# The container writes as root; hand the file back.
if [[ "$(id -u)" -ne 0 ]]; then
  docker run --rm -v "$VB_DATASETS:/d" --entrypoint chown "$IMAGE" \
    -R "$(id -u):$(id -g)" /d >/dev/null 2>&1 || true
fi

ok "$DATASET ready: $TARGET ($(human_bytes "$(stat -c%s "$TARGET")"))"
