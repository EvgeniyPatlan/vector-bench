"""Dataset loading and ground-truth computation for the ops harness.

Reads the same HDF5 files ann-benchmarks uses, so both measurement paths see
identical data and identical ground truth. The one thing computed here that
ann-benchmarks does not provide is **filtered ground truth**: applying a WHERE
predicate changes the correct answer set, so reusing the unfiltered neighbours
would silently score every engine against the wrong targets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy

# Distance metrics as ann-benchmarks names them in the HDF5 `distance` attribute.
ANGULAR = "angular"
EUCLIDEAN = "euclidean"

# Datasets this framework is set up for, with the shape facts needed to plan a
# run before the file is downloaded.
KNOWN_DATASETS: Dict[str, Dict[str, object]] = {
    "fashion-mnist-784-euclidean": {
        "dim": 784, "train": 60_000, "test": 10_000,
        "metric": EUCLIDEAN, "approx_bytes": 217 * 1024**2,
        "role": "smoke",
    },
    "glove-100-angular": {
        "dim": 100, "train": 1_183_514, "test": 10_000,
        "metric": ANGULAR, "approx_bytes": 463 * 1024**2,
        "role": "main",
    },
    "sift-128-euclidean": {
        "dim": 128, "train": 1_000_000, "test": 10_000,
        "metric": EUCLIDEAN, "approx_bytes": 501 * 1024**2,
        "role": "main",
    },
    "gist-960-euclidean": {
        "dim": 960, "train": 1_000_000, "test": 1_000,
        "metric": EUCLIDEAN, "approx_bytes": 3_600 * 1024**2,
        "role": "stress",
    },
}


@dataclass
class Dataset:
    name: str
    train: numpy.ndarray
    test: numpy.ndarray
    neighbors: numpy.ndarray     # ground truth ids, shape (n_test, 100)
    distances: Optional[numpy.ndarray]
    metric: str

    @property
    def dim(self) -> int:
        return int(self.train.shape[1])

    @property
    def n_train(self) -> int:
        return int(self.train.shape[0])

    @property
    def n_test(self) -> int:
        return int(self.test.shape[0])


def dataset_path(datasets_dir: str, name: str) -> str:
    return os.path.join(datasets_dir, f"{name}.hdf5")


def load(datasets_dir: str, name: str, max_test: Optional[int] = None) -> Dataset:
    import h5py  # imported lazily: only the client container has h5py

    path = dataset_path(datasets_dir, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"dataset not found: {path}\n"
            f"run: ./run-benchmark.sh fetch --datasets {name}"
        )

    with h5py.File(path, "r") as fh:
        train = numpy.asarray(fh["train"], dtype="<f4")
        test = numpy.asarray(fh["test"], dtype="<f4")
        neighbors = numpy.asarray(fh["neighbors"], dtype=numpy.int64)
        distances = numpy.asarray(fh["distances"]) if "distances" in fh else None
        metric = fh.attrs.get("distance", KNOWN_DATASETS.get(name, {}).get("metric", EUCLIDEAN))
        if isinstance(metric, bytes):
            metric = metric.decode()

    if max_test is not None and max_test < len(test):
        test = test[:max_test]
        neighbors = neighbors[:max_test]
        if distances is not None:
            distances = distances[:max_test]

    return Dataset(
        name=name, train=train, test=test, neighbors=neighbors,
        distances=distances, metric=str(metric),
    )


# ---------------------------------------------------------------------------
# Recall
# ---------------------------------------------------------------------------

def recall_at_k(returned: Sequence[Sequence[int]],
                truth: numpy.ndarray, k: int) -> float:
    """Mean fraction of the true top-k that each result set contains.

    Matches ann-benchmarks' definition so the two measurement paths produce
    directly comparable recall numbers.
    """
    if len(returned) == 0:
        return float("nan")
    total = 0.0
    for got, want in zip(returned, truth):
        wanted = set(int(x) for x in want[:k])
        if not wanted:
            continue
        total += len(wanted.intersection(int(x) for x in got[:k])) / len(wanted)
    return total / len(returned)


# ---------------------------------------------------------------------------
# Filtered ground truth
# ---------------------------------------------------------------------------

def assign_tags(n_rows: int, buckets: int = 100) -> numpy.ndarray:
    """Deterministic scalar attribute used by the filtered-search workload.

    `tag = row_id % buckets` gives an exactly known selectivity for a predicate
    of the form `tag < t`: t/buckets of the rows qualify. Deterministic and
    uniform, so the filter's selectivity is a property of the query rather than
    of a random draw, and every engine sees the identical qualifying set.
    """
    return numpy.arange(n_rows, dtype=numpy.int64) % buckets


def filtered_ground_truth(train: numpy.ndarray, test: numpy.ndarray,
                          metric: str, k: int, tag_threshold: int,
                          buckets: int = 100,
                          batch_size: int = 256) -> numpy.ndarray:
    """Exact top-k over only the rows passing `tag < tag_threshold`.

    Brute force, in batches, because this must be exact: it is the yardstick the
    engines' filtered results are scored against. Cost is O(n_test * n_qualify *
    dim), which at these dataset sizes is minutes, not hours — and it is cached
    by the caller so it is paid once per (dataset, selectivity).
    """
    tags = assign_tags(len(train), buckets)
    qualifying = numpy.nonzero(tags < tag_threshold)[0]
    if len(qualifying) == 0:
        raise ValueError(f"no rows qualify for tag < {tag_threshold}")

    subset = train[qualifying]
    if metric == ANGULAR:
        subset = _normalize(subset)
        queries = _normalize(test)
    else:
        queries = test

    k = min(k, len(qualifying))
    out = numpy.empty((len(queries), k), dtype=numpy.int64)

    for start in range(0, len(queries), batch_size):
        batch = queries[start:start + batch_size]
        if metric == ANGULAR:
            # Cosine distance ordering is the reverse of dot-product ordering
            # once both sides are unit-normalized.
            scores = -(batch @ subset.T)
        else:
            # ||a-b||^2 expanded; the ||a||^2 term is constant per query and so
            # does not affect ordering, but is kept for numerical clarity.
            scores = (
                (batch ** 2).sum(axis=1, keepdims=True)
                - 2.0 * (batch @ subset.T)
                + (subset ** 2).sum(axis=1)[None, :]
            )
        idx = numpy.argpartition(scores, kth=k - 1, axis=1)[:, :k]
        ordered = numpy.take_along_axis(
            idx, numpy.argsort(numpy.take_along_axis(scores, idx, axis=1), axis=1), axis=1
        )
        out[start:start + len(batch)] = qualifying[ordered]

    return out


def _normalize(a: numpy.ndarray) -> numpy.ndarray:
    norms = numpy.linalg.norm(a, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (a / norms).astype("<f4")


def cached_ground_truth(cache_dir: str, dataset: Dataset, k: int,
                        indexed_rows: Optional[int] = None) -> numpy.ndarray:
    """Exact unfiltered top-k over the rows the engine actually holds.

    When the whole training set is loaded this is just the neighbours shipped in
    the HDF5 file. When a profile loads a subset, those neighbours point at rows
    the engine never received, so the correct answers have to be recomputed over
    the subset — otherwise recall is measured against unreachable targets.
    """
    n_train = len(dataset.train)
    rows = n_train if indexed_rows is None else min(indexed_rows, n_train)
    if rows >= n_train:
        return dataset.neighbors[:, :k]

    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{dataset.name}.k{k}.n{rows}.full.npy")
    if os.path.exists(path):
        return numpy.load(path)

    # `tag < buckets` selects every row, so the filtered brute force over the
    # subset is exactly the unfiltered ground truth for that subset.
    truth = filtered_ground_truth(
        dataset.train[:rows], dataset.test, dataset.metric, k,
        tag_threshold=100, buckets=100,
    )
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        numpy.save(fh, truth)
    os.replace(tmp, path)
    return truth


def selectivity_to_threshold(selectivity: float, buckets: int = 100) -> int:
    """Convert a target selectivity into a `tag <` threshold.

    Rounded up and clamped to at least 1 so a 0.5% request on 100 buckets still
    selects a non-empty set rather than silently matching nothing.
    """
    return max(1, min(buckets, int(round(selectivity * buckets))))


def cached_filtered_ground_truth(cache_dir: str, dataset: Dataset, k: int,
                                 selectivity: float, buckets: int = 100,
                                 indexed_rows: Optional[int] = None) -> numpy.ndarray:
    """Compute filtered ground truth once per (dataset, k, selectivity, rows).

    `indexed_rows` must be the number of rows actually loaded into the engine.
    Profiles can load a subset of the training set (smoke uses 20,000 of
    60,000), and ground truth computed over the full set would include rows the
    engine was never given — scoring every engine against unreachable answers
    and understating recall across the board. The row count is part of the cache
    key for the same reason.
    """
    threshold = selectivity_to_threshold(selectivity, buckets)
    rows = len(dataset.train) if indexed_rows is None else min(indexed_rows,
                                                              len(dataset.train))
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(
        cache_dir, f"{dataset.name}.k{k}.sel{threshold}of{buckets}.n{rows}.npy"
    )
    if os.path.exists(path):
        return numpy.load(path)
    truth = filtered_ground_truth(
        dataset.train[:rows], dataset.test, dataset.metric, k, threshold, buckets
    )
    # numpy.save() appends ".npy" to a *path* that lacks it, which would turn
    # "<name>.npy.tmp" into "<name>.npy.tmp.npy" and break the atomic rename.
    # Passing an open file object suppresses that behaviour.
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        numpy.save(fh, truth)
    os.replace(tmp, path)
    return truth
