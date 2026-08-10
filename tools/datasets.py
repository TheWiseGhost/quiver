"""Benchmark datasets: ann-benchmarks HDF5 files (downloaded on request) or synthetic data."""

from __future__ import annotations

import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# name -> (url, metric). Sizes are approximate download sizes.
ANN_BENCHMARKS = {
    "sift-128-euclidean": ("http://ann-benchmarks.com/sift-128-euclidean.hdf5", "l2"),        # 501 MB
    "glove-100-angular": ("http://ann-benchmarks.com/glove-100-angular.hdf5", "cosine"),      # 463 MB
    "fashion-mnist-784-euclidean": ("http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5", "l2"),  # 217 MB
    "nytimes-256-angular": ("http://ann-benchmarks.com/nytimes-256-angular.hdf5", "cosine"),  # 301 MB
}


@dataclass
class Dataset:
    name: str
    base: np.ndarray
    queries: np.ndarray
    neighbors: np.ndarray  # exact top-100 (or top-k) ids
    metric: str


def download(name: str) -> Path:
    url, _ = ANN_BENCHMARKS[name]
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / f"{name}.hdf5"
    if path.exists():
        return path
    tmp = path.with_suffix(".part")
    print(f"downloading {url} -> {path}", file=sys.stderr)

    # The server rejects urllib's default User-Agent with 403.
    req = urllib.request.Request(url, headers={"User-Agent": "quiver-benchmarks/0.1 (+https://github.com)"})
    with urllib.request.urlopen(req) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", file=sys.stderr)
    print(file=sys.stderr)
    os.replace(tmp, path)
    return path


def load(name: str, max_base: int = 0, max_queries: int = 0) -> Dataset:
    if name.startswith("synthetic"):
        # synthetic-N-D[-Q]: Gaussian blobs, ground truth by brute force.
        parts = name.split("-")
        n, d = int(parts[1]), int(parts[2])
        nq = int(parts[3]) if len(parts) > 3 else 1000
        rng = np.random.default_rng(0)
        centers = rng.standard_normal((100, d)).astype(np.float32) * 2
        base = (centers[rng.integers(0, 100, n)] + rng.standard_normal((n, d))).astype(np.float32)
        queries = (centers[rng.integers(0, 100, nq)] + rng.standard_normal((nq, d))).astype(np.float32)
        return Dataset(name, base, queries, exact_neighbors(base, queries, 100, "l2"), "l2")

    import h5py

    path = DATA_DIR / f"{name}.hdf5"
    if not path.exists():
        raise SystemExit(f"{path} not found; run: python tools/datasets.py {name}")
    with h5py.File(path, "r") as f:
        base = np.asarray(f["train"], dtype=np.float32)
        queries = np.asarray(f["test"], dtype=np.float32)
        neighbors = np.asarray(f["neighbors"])
    metric = ANN_BENCHMARKS[name][1]
    if max_queries:
        queries, neighbors = queries[:max_queries], neighbors[:max_queries]
    if max_base and max_base < len(base):
        base = base[:max_base]
        neighbors = exact_neighbors(base, queries, 100, metric)
    return Dataset(name, base, queries, neighbors, metric)


def exact_neighbors(base: np.ndarray, queries: np.ndarray, k: int, metric: str) -> np.ndarray:
    """Exact ground truth with plain numpy (float64), independent of every library under test."""
    k = min(k, len(base))
    b = base.astype(np.float64)
    if metric == "cosine":
        b /= np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-30)
    out = np.empty((len(queries), k), dtype=np.int64)
    norms = (b * b).sum(1)
    for s in range(0, len(queries), 256):
        q = queries[s:s + 256].astype(np.float64)
        if metric == "cosine":
            q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-30)
        score = norms[None, :] - 2.0 * q @ b.T if metric == "l2" else -(q @ b.T)
        part = np.argpartition(score, k - 1, axis=1)[:, :k]
        order = np.take_along_axis(score, part, 1).argsort(1)
        out[s:s + 256] = np.take_along_axis(part, order, 1)
    return out


if __name__ == "__main__":
    for n in sys.argv[1:] or list(ANN_BENCHMARKS):
        download(n)
