#!/usr/bin/env bash
#
# Generate a tiny synthetic dataset with exact ground truth, for exercising the
# framework without downloading anything.
#
# Pairs with config/profiles/dev.yml: 4,000 x 16 vectors runs a full ops cycle
# (build, concurrency, filtered, churn) in about a minute per engine, which
# makes it practical to validate a framework change before spending hours on a
# real dataset.
#
# The ground truth is computed by brute force, so recall figures against it are
# exact. ann-benchmarks validates --dataset against its own registry and will
# not accept this name, which is why dev.yml disables the recall/QPS path.
#
# Usage: tests/make-tiny-dataset.sh [rows] [dims] [queries]

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/lib.sh"

ROWS="${1:-4000}"
DIMS="${2:-16}"
QUERIES="${3:-200}"
NAME="tiny-${DIMS}-euclidean"

need_docker
mkdir -p "$VB_DATASETS"

# Any bench image will do — they all carry numpy and h5py.
IMAGE=""
for candidate in vector-bench/pgvector-bench vector-bench/mariadb-bench vector-bench/alisql-bench; do
  if image_exists "$candidate"; then IMAGE="$candidate"; break; fi
done
[[ -n "$IMAGE" ]] || die "no bench image found; build one first: ./run-benchmark.sh build"

info "generating $NAME ($ROWS x $DIMS, $QUERIES queries) using $IMAGE"

docker run --rm --network none \
  -v "$VB_DATASETS:/datasets" \
  -e ROWS="$ROWS" -e DIMS="$DIMS" -e QUERIES="$QUERIES" -e NAME="$NAME" \
  -u "$(id -u):$(id -g)" \
  --entrypoint python3 "$IMAGE" -c '
import os
import h5py, numpy

rows, dims, queries = int(os.environ["ROWS"]), int(os.environ["DIMS"]), int(os.environ["QUERIES"])
name = os.environ["NAME"]

# Fixed seed: the dataset must be identical on every machine, or results
# generated against it are not comparable.
rng = numpy.random.default_rng(7)
train = rng.random((rows, dims), dtype=numpy.float32)
test = rng.random((queries, dims), dtype=numpy.float32)

# Exact top-100 by brute force. ||a-b||^2 expanded; the ordering is what matters.
d2 = ((test ** 2).sum(1)[:, None] - 2 * test @ train.T + (train ** 2).sum(1)[None, :])
k = min(100, rows)
idx = numpy.argsort(d2, axis=1)[:, :k]
dist = numpy.sqrt(numpy.maximum(numpy.take_along_axis(d2, idx, 1), 0))

with h5py.File(f"/datasets/{name}.hdf5", "w") as f:
    f.create_dataset("train", data=train)
    f.create_dataset("test", data=test)
    f.create_dataset("neighbors", data=idx.astype("int32"))
    f.create_dataset("distances", data=dist.astype("float32"))
    f.attrs["distance"] = "euclidean"
    f.attrs["point_type"] = "float"
print(f"wrote {name}.hdf5: {train.shape} train, {test.shape} test, exact top-{k}")
'

ok "$NAME ready in $VB_DATASETS"
echo
echo "Run a full ops cycle against it with:"
echo "  ./run-benchmark.sh run --profile dev --engines pgvector"
