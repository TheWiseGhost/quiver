import threading

import numpy as np
import pytest

import quiver


def brute_force(base, queries, k):
    d = ((queries[:, None, :] - base[None, :, :]) ** 2).sum(-1)
    return np.argsort(d, axis=1)[:, :k]


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    return rng.standard_normal((3000, 24), dtype=np.float32), rng.standard_normal((50, 24), dtype=np.float32)


def recall(truth, got):
    return np.mean([len(set(t) & set(g)) / len(t) for t, g in zip(truth, got)])


@pytest.mark.parametrize("storage,target", [("float32", 0.97), ("sq8", 0.9)])
def test_recall(data, storage, target):
    base, queries = data
    index = quiver.HnswIndex(24, storage=storage)
    index.add(base)
    labels, dist = index.search(queries, k=10, ef=128)
    assert labels.shape == (50, 10) and labels.dtype == np.int64
    assert dist.dtype == np.float32 and np.all(np.diff(dist, axis=1) >= 0)
    assert recall(brute_force(base, queries, 10), labels) >= target


def test_float64_lists_and_single_vectors_are_accepted():
    index = quiver.HnswIndex(3)
    index.add([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    labels, _ = index.search([1.0, 2.0, 3.1], k=1)
    assert labels.tolist() == [[0]]


def test_missing_results_are_minus_one():
    index = quiver.HnswIndex(4)
    index.add(np.eye(4, dtype=np.float32))
    labels, dist = index.search(np.ones(4), k=6)
    assert labels[0, 4:].tolist() == [-1, -1] and np.isinf(dist[0, 4:]).all()


def test_labels_filter_remove(data):
    base, queries = data
    index = quiver.HnswIndex(24, metric="cosine")
    index.add(base, labels=np.arange(len(base)) * 10)
    assert 20 in index and 21 not in index
    allowed = np.arange(0, len(base) * 10, 30)
    labels, _ = index.search(queries, k=5, filter=allowed)
    assert np.isin(labels, allowed).all()
    assert index.remove(20) and not index.remove(20)
    assert 20 not in index and len(index) == len(base) - 1
    np.testing.assert_allclose(np.linalg.norm(index.get_vector(30)), 1.0, rtol=1e-5)


def test_errors():
    index = quiver.HnswIndex(8)
    with pytest.raises(ValueError, match="dimension"):
        index.add(np.zeros((2, 7)))
    index.add(np.zeros((1, 8)), labels=[5])
    with pytest.raises(ValueError, match="already"):
        index.add(np.zeros((1, 8)), labels=[5])
    with pytest.raises(ValueError):
        index.add(np.zeros((1, 8)), labels=[-1])
    with pytest.raises(ValueError):
        quiver.HnswIndex(8, metric="hamming")
    with pytest.raises(ValueError):
        quiver.HnswIndex.load("/does/not/exist.qvr")


@pytest.mark.parametrize("mmap", [False, True])
def test_save_load(tmp_path, data, mmap):
    base, queries = data
    index = quiver.HnswIndex(24, storage="sq8")
    index.add(base)
    path = tmp_path / "index.qvr"
    index.save(path)
    loaded = quiver.HnswIndex.load(path, mmap=mmap)
    assert loaded.stats()["memory_mapped"] == mmap
    assert loaded.storage == "sq8" and len(loaded) == len(index)
    np.testing.assert_array_equal(loaded.search(queries, k=10)[0], index.search(queries, k=10)[0])
    loaded.add(queries, labels=np.arange(len(queries)) + 100000)  # mapped index stays mutable
    assert loaded.search(queries[:1], k=1)[0][0, 0] == 100000


def test_concurrent_python_threads(data):
    base, queries = data
    index = quiver.HnswIndex(24)
    index.add(base[:1000])
    errors = []

    def reader():
        try:
            for _ in range(50):
                labels, _ = index.search(queries, k=5, num_threads=1)
                assert (labels >= 0).all()
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    for i in range(1000, 3000, 100):
        index.add(base[i:i + 100], labels=np.arange(i, i + 100))
    for t in threads:
        t.join()
    assert not errors and len(index) == 3000


def test_flat_index_is_exact(data, tmp_path):
    base, queries = data
    flat = quiver.FlatIndex(24)
    flat.add(base)
    labels, _ = flat.search(queries, k=10)
    np.testing.assert_array_equal(labels, brute_force(base, queries, 10))
    flat.save(tmp_path / "flat.qvr")
    assert len(quiver.FlatIndex.load(tmp_path / "flat.qvr")) == len(base)


def test_ivfpq(tmp_path):
    rng = np.random.default_rng(1)
    centers = rng.standard_normal((40, 32)).astype(np.float32) * 3
    base = (centers[rng.integers(0, 40, 6000)] + rng.standard_normal((6000, 32))).astype(np.float32)
    queries = (centers[rng.integers(0, 40, 50)] + rng.standard_normal((50, 32))).astype(np.float32)
    index = quiver.IvfPqIndex(32, nlist=64, m=16, store_vectors=True)
    with pytest.raises(ValueError):
        index.add(base)  # must train first
    index.train(base)
    index.add(base)
    assert index.is_trained and len(index) == 6000
    labels, _ = index.search(queries, k=10, nprobe=16, refine_factor=4)
    assert recall(brute_force(base, queries, 10), labels) >= 0.9
    index.save(tmp_path / "ivf.qvr")
    loaded = quiver.IvfPqIndex.load(tmp_path / "ivf.qvr")
    np.testing.assert_array_equal(loaded.search(queries, k=5)[0], index.search(queries, k=5)[0])


# ---- regressions from review -------------------------------------------------------------------

def test_label_range_and_sentinel():
    x = np.random.default_rng(2).standard_normal((5, 4)).astype(np.float32)
    for cls in (quiver.HnswIndex, quiver.FlatIndex):
        index = cls(4)
        with pytest.raises(ValueError, match="2\\*\\*63"):
            index.add(x, labels=np.array([0, 1, 2, 3, 2**64 - 1], dtype=np.uint64))
        with pytest.raises(ValueError, match="2\\*\\*63"):
            index.add(x, labels=[0, 1, 2, 3, 2**63])
        index.add(x, labels=[0, 1, 2, 3, 2**63 - 1])
        assert index.search(x[4], k=1)[0][0, 0] == 2**63 - 1
    h = quiver.HnswIndex(4)
    h.add(x)
    assert (-1 in h) is False and (2**70 in h) is False and h.remove(-1) is False


@pytest.mark.parametrize("flt", [[], set(), {1, 2}, (i for i in [1, 2]), range(3), np.array([1, 2])])
def test_filter_accepts_any_iterable(flt):
    x = np.random.default_rng(3).standard_normal((20, 4)).astype(np.float32)
    f = quiver.FlatIndex(4)
    f.add(x)
    labels, _ = f.search(x[:1], k=2, filter=flt)
    assert set(labels[0][labels[0] >= 0]) <= {0, 1, 2}


def test_huge_k_is_rejected_not_a_crash():
    x = np.random.default_rng(4).standard_normal((10, 4)).astype(np.float32)
    h = quiver.HnswIndex(4)
    h.add(x)
    with pytest.raises(ValueError, match="too large"):
        h.search(x[:3], k=2**64 // 3 + 1)
    with pytest.raises((ValueError, OverflowError)):
        h.search(x[:4], k=2**62)


def test_ivfpq_concurrent_add_and_search_do_not_deadlock():
    rng = np.random.default_rng(5)
    x = rng.standard_normal((20000, 16)).astype(np.float32)
    index = quiver.IvfPqIndex(16, nlist=32, m=4, train_iterations=3)
    index.train(x)
    index.add(x)
    q = rng.standard_normal((20000, 16)).astype(np.float32)
    done = []

    def run(fn):
        fn()
        done.append(1)

    threads = [threading.Thread(target=run, args=(f,)) for f in (
        lambda: index.search(q, k=10, nprobe=32),
        lambda: index.add(x[:5000]),
        lambda: index.add(x[:100], num_threads=1))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert len(done) == 3, "deadlock"
    assert len(index) == 25100
