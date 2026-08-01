"""Quiver: SIMD-accelerated approximate nearest neighbor search.

    >>> import numpy as np, quiver
    >>> index = quiver.HnswIndex(dim=128, metric="cosine")
    >>> index.add(np.random.rand(10_000, 128))
    >>> labels, distances = index.search(np.random.rand(5, 128), k=10)

Distances are "smaller is closer": squared L2 for ``"l2"``, ``1 - <q, x>`` for ``"ip"``, and
``1 - cos(q, x)`` for ``"cosine"``. Missing results (k larger than the index or filter) come
back as label ``-1`` and distance ``inf``.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple, Union

import numpy as np

from . import _quiver
from ._quiver import simd_isa

__all__ = ["HnswIndex", "IvfPqIndex", "FlatIndex", "simd_isa"]
__version__ = "0.1.0"

ArrayLike = Union[np.ndarray, Iterable]


def _as_matrix(x: ArrayLike, dim: int) -> np.ndarray:
    a = np.ascontiguousarray(x, dtype=np.float32)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if a.ndim != 2:
        raise ValueError(f"expected a 1-D or 2-D array, got shape {a.shape}")
    if a.shape[1] != dim:
        raise ValueError(f"expected dimension {dim}, got {a.shape[1]}")
    return a


_MAX_LABEL = 2**63 - 1  # labels come back as int64, and -1 marks an empty slot


def _as_labels(labels: Optional[ArrayLike], n: Optional[int] = None) -> Optional[np.ndarray]:
    """Validate labels: integers in [0, 2**63). Accepts arrays, lists, sets, and iterators."""
    if labels is None:
        return None
    if isinstance(labels, np.ndarray):
        a = labels.reshape(-1)
        if a.dtype == object:
            return _as_labels(a.tolist(), n)
        if a.size and not np.issubdtype(a.dtype, np.integer):
            raise TypeError("labels must be integers")
        if a.size:
            if np.issubdtype(a.dtype, np.signedinteger) and a.min() < 0:
                raise ValueError("labels must be non-negative")
            if a.max() > _MAX_LABEL:
                raise ValueError("labels must be below 2**63")
        a = np.ascontiguousarray(a, dtype=np.uint64)
    else:
        # Validate as Python ints *before* numpy sees them: numpy would turn [0, 2**63] into
        # float64 and silently lose precision.
        items = [labels] if isinstance(labels, (int, np.integer)) else list(labels)
        for v in items:
            if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
                raise TypeError("labels must be integers")
            if v < 0:
                raise ValueError("labels must be non-negative")
            if v > _MAX_LABEL:
                raise ValueError("labels must be below 2**63")
        a = np.array([int(v) for v in items], dtype=np.uint64)
    if n is not None and a.shape[0] != n:
        raise ValueError(f"got {a.shape[0]} labels for {n} vectors")
    return a


def _valid_label(label) -> bool:
    try:
        v = int(label)
    except (TypeError, ValueError):
        return False
    return 0 <= v <= _MAX_LABEL


class HnswIndex:
    """Hierarchical Navigable Small World graph index.

    Args:
        dim: vector dimensionality.
        metric: ``"l2"``, ``"ip"``, or ``"cosine"``.
        M: graph out-degree (layer 0 uses ``2*M``). 16 is a good default; 32-48 for
            high-dimensional embeddings needing very high recall.
        ef_construction: build-time beam width. Higher builds a better graph, slower.
        storage: ``"float32"`` (exact) or ``"sq8"`` (8-bit, 4x less memory).
        seed: seeds level assignment for reproducible single-threaded builds.

    All methods are thread-safe. Searches never block; ``save`` waits for in-flight ``add``
    calls. Default labels are 0, 1, 2, ... in insertion order (counting removed vectors).
    """

    def __init__(self, dim: int, metric: str = "l2", M: int = 16, ef_construction: int = 200,
                 storage: str = "float32", seed: int = 0x5EED, *, _native=None):
        self._index = _native if _native is not None else _quiver.HnswIndex(
            dim, metric, M, ef_construction, storage, seed)

    def train(self, vectors: ArrayLike) -> None:
        """SQ8 only: learn quantization ranges from a representative sample (once). Without it,
        the first ``add`` trains on its own batch, which must hold at least 256 vectors."""
        self._index._train(_as_matrix(vectors, self.dim))

    def add(self, vectors: ArrayLike, labels: Optional[ArrayLike] = None, num_threads: int = 0) -> None:
        """Insert vectors (n x dim). Labels default to consecutive insertion slots."""
        x = _as_matrix(vectors, self.dim)
        self._index._add(x, _as_labels(labels, x.shape[0]), num_threads)

    def search(self, queries: ArrayLike, k: int = 10, ef: int = 64,
               filter: Optional[ArrayLike] = None, exact_scan_threshold: float = 0.02,
               num_threads: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """k-NN search. Returns (labels int64 [nq, k], distances float32 [nq, k]).

        Args:
            ef: beam width (>= k); the recall/speed knob.
            filter: iterable of allowed labels (array, list, set, ...). Very selective filters (admitting at most
                ``exact_scan_threshold`` of the index) are answered by an exact scan.
        """
        q = _as_matrix(queries, self.dim)
        if k <= 0:
            raise ValueError("k must be positive")
        allowed = _as_labels(filter) if filter is not None else None
        return self._index._search(q, int(k), int(ef), allowed, float(exact_scan_threshold), num_threads)

    def remove(self, label: int) -> bool:
        """Soft-delete; returns False if absent."""
        return _valid_label(label) and self._index.remove(int(label))

    def get_vector(self, label: int) -> np.ndarray:
        return self._index.get_vector(int(label))

    def save(self, path: str) -> None:
        """Atomically write the index to ``path``."""
        self._index.save(str(path))

    @classmethod
    def load(cls, path: str, mmap: bool = False, verify_checksums: bool = True) -> "HnswIndex":
        """Load an index. ``mmap=True`` maps the file copy-on-write: near-instant, lazily paged,
        still fully mutable, and the file is never written."""
        return cls(0, _native=_quiver.HnswIndex.load(str(path), mmap, verify_checksums))

    def stats(self) -> dict:
        return self._index.stats()

    def __len__(self) -> int:
        return len(self._index)

    def __contains__(self, label) -> bool:
        return _valid_label(label) and int(label) in self._index

    dim = property(lambda self: self._index.dim)
    metric = property(lambda self: self._index.metric)
    storage = property(lambda self: self._index.storage)
    M = property(lambda self: self._index.M)

    def __repr__(self) -> str:
        return (f"HnswIndex(dim={self.dim}, metric={self.metric!r}, M={self.M}, "
                f"storage={self.storage!r}, size={len(self)})")


class IvfPqIndex:
    """Inverted-file index over product-quantized residuals: ~``m`` bytes per vector.

    Args:
        dim: vector dimensionality (must be divisible by ``m``).
        metric: ``"l2"``, ``"ip"``, or ``"cosine"``.
        nlist: number of coarse cells (roughly ``4 * sqrt(n)``).
        m: bytes per vector (PQ sub-quantizers).
        store_vectors: keep float32 copies to enable exact re-ranking (``refine_factor``).

    Call :meth:`train` on a representative sample before :meth:`add`.
    """

    def __init__(self, dim: int, metric: str = "l2", nlist: int = 1024, m: int = 16,
                 store_vectors: bool = False, train_iterations: int = 20, seed: int = 1234,
                 *, _native=None):
        self._index = _native if _native is not None else _quiver.IvfPqIndex(
            dim, metric, nlist, m, store_vectors, train_iterations, seed)

    def train(self, vectors: ArrayLike, num_threads: int = 0) -> None:
        self._index._train(_as_matrix(vectors, self.dim), num_threads)

    def add(self, vectors: ArrayLike, labels: Optional[ArrayLike] = None, num_threads: int = 0) -> None:
        x = _as_matrix(vectors, self.dim)
        self._index._add(x, _as_labels(labels, x.shape[0]), num_threads)

    def search(self, queries: ArrayLike, k: int = 10, nprobe: int = 16, refine_factor: int = 0,
               filter: Optional[ArrayLike] = None, num_threads: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        q = _as_matrix(queries, self.dim)
        if k <= 0:
            raise ValueError("k must be positive")
        allowed = _as_labels(filter) if filter is not None else None
        return self._index._search(q, int(k), int(nprobe), int(refine_factor), allowed, num_threads)

    def save(self, path: str) -> None:
        self._index.save(str(path))

    @classmethod
    def load(cls, path: str, verify_checksums: bool = True) -> "IvfPqIndex":
        return cls(0, _native=_quiver.IvfPqIndex.load(str(path), verify_checksums))

    def __len__(self) -> int:
        return len(self._index)

    dim = property(lambda self: self._index.dim)
    is_trained = property(lambda self: self._index.is_trained)
    memory_bytes = property(lambda self: self._index.memory_bytes)

    def __repr__(self) -> str:
        return (f"IvfPqIndex(dim={self.dim}, nlist={self._index.nlist}, m={self._index.m}, "
                f"size={len(self)})")


class FlatIndex:
    """Exact brute-force index (SIMD-accelerated, multithreaded, thread-safe)."""

    def __init__(self, dim: int, metric: str = "l2", *, _native=None):
        self._index = _native if _native is not None else _quiver.FlatIndex(dim, metric)

    def add(self, vectors: ArrayLike, labels: Optional[ArrayLike] = None) -> None:
        x = _as_matrix(vectors, self.dim)
        self._index._add(x, _as_labels(labels, x.shape[0]))

    def search(self, queries: ArrayLike, k: int = 10, filter: Optional[ArrayLike] = None,
               num_threads: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        q = _as_matrix(queries, self.dim)
        if k <= 0:
            raise ValueError("k must be positive")
        allowed = _as_labels(filter) if filter is not None else None
        return self._index._search(q, int(k), allowed, num_threads)

    def save(self, path: str) -> None:
        self._index.save(str(path))

    @classmethod
    def load(cls, path: str, verify_checksums: bool = True) -> "FlatIndex":
        return cls(0, _native=_quiver.FlatIndex.load(str(path), verify_checksums))

    def __len__(self) -> int:
        return len(self._index)

    dim = property(lambda self: self._index.dim)
    metric = property(lambda self: self._index.metric)

    def __repr__(self) -> str:
        return f"FlatIndex(dim={self.dim}, metric={self.metric!r}, size={len(self)})"
