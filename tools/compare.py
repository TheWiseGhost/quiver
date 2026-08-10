"""Head-to-head HNSW benchmark: Quiver vs hnswlib vs FAISS on identical data and parameters.

    python tools/compare.py --dataset sift-128-euclidean --threads 8 --out results/sift.jsonl
    python tools/compare.py --dataset synthetic-200000-128

Every library gets the same M, ef_construction, ef sweep, and thread count. QPS is measured on
the whole query batch (best of --reps); "1-thread" runs the same batch with one thread.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datasets  # noqa: E402


def performance_cores() -> int:
    """P-core count on Apple Silicon / hybrid CPUs, else all cores."""
    try:
        import subprocess

        return int(subprocess.check_output(["sysctl", "-n", "hw.perflevel0.physicalcpu"], text=True))
    except Exception:
        return os.cpu_count() or 1


def recall_at_k(truth: np.ndarray, got: np.ndarray, k: int) -> float:
    hits = 0
    for t, g in zip(truth[:, :k], got[:, :k]):
        hits += len(np.intersect1d(t, g, assume_unique=False))
    return hits / (k * len(truth))


def timed(fn, reps):
    best, out = float("inf"), None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


class Quiver:
    name = "quiver"

    def __init__(self, dim, metric, M, efc, threads, storage="float32"):
        import quiver

        self.index = quiver.HnswIndex(dim, metric=metric, M=M, ef_construction=efc, storage=storage)
        self.threads = threads
        if storage != "float32":
            self.name = f"quiver-{storage}"

    def build(self, x):
        if self.index.storage == "sq8":
            self.index.train(x[:100000])
        self.index.add(x, num_threads=self.threads)

    def search(self, q, k, ef, threads):
        return self.index.search(q, k=k, ef=ef, num_threads=threads)[0]

    def memory(self):
        s = self.index.stats()
        return s["vector_bytes"] + s["graph_bytes"]


class Hnswlib:
    name = "hnswlib"

    def __init__(self, dim, metric, M, efc, threads):
        import hnswlib

        space = {"l2": "l2", "ip": "ip", "cosine": "cosine"}[metric]
        self.index = hnswlib.Index(space=space, dim=dim)
        self.M, self.efc, self.threads = M, efc, threads

    def build(self, x):
        self.index.init_index(max_elements=len(x), M=self.M, ef_construction=self.efc, random_seed=100)
        self.index.add_items(x, np.arange(len(x)), num_threads=self.threads)

    def search(self, q, k, ef, threads):
        self.index.set_ef(max(ef, k))
        return self.index.knn_query(q, k=k, num_threads=threads)[0]

    def memory(self):
        return None


class FaissHnsw:
    name = "faiss-hnsw"

    def __init__(self, dim, metric, M, efc, threads):
        import faiss

        self.faiss = faiss
        self.cosine = metric == "cosine"
        mt = faiss.METRIC_L2 if metric == "l2" else faiss.METRIC_INNER_PRODUCT
        self.index = faiss.IndexHNSWFlat(dim, M, mt)
        self.index.hnsw.efConstruction = efc
        self.threads = threads

    def _prep(self, x):
        if self.cosine:
            x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        return np.ascontiguousarray(x, dtype=np.float32)

    def build(self, x):
        self.faiss.omp_set_num_threads(self.threads)
        self.index.add(self._prep(x))

    def search(self, q, k, ef, threads):
        self.faiss.omp_set_num_threads(threads)
        self.index.hnsw.efSearch = max(ef, k)
        return self.index.search(self._prep(q), k)[1]

    def memory(self):
        return None


class QuiverIvfPq:
    """IVF-PQ: `ef` in the sweep is nprobe."""

    def __init__(self, dim, metric, nlist, m, threads, refine=0):
        import quiver

        self.index = quiver.IvfPqIndex(dim, metric=metric, nlist=nlist, m=m, store_vectors=refine > 0)
        self.threads, self.refine = threads, refine
        self.name = "quiver-ivfpq" + (f"-refine{refine}" if refine else "")

    def build(self, x):
        rng = np.random.default_rng(0)
        self.index.train(x[rng.choice(len(x), min(len(x), 200000), replace=False)], num_threads=self.threads)
        self.index.add(x, num_threads=self.threads)

    def search(self, q, k, nprobe, threads):
        return self.index.search(q, k=k, nprobe=nprobe, refine_factor=self.refine, num_threads=threads)[0]

    def memory(self):
        return self.index.memory_bytes


class FaissIvfPq:
    name = "faiss-ivfpq"

    def __init__(self, dim, metric, nlist, m, threads):
        import faiss

        self.faiss = faiss
        self.cosine = metric == "cosine"
        mt = faiss.METRIC_L2 if metric == "l2" else faiss.METRIC_INNER_PRODUCT
        self.quantizer = faiss.IndexFlatL2(dim) if mt == faiss.METRIC_L2 else faiss.IndexFlatIP(dim)
        self.index = faiss.IndexIVFPQ(self.quantizer, dim, nlist, m, 8, mt)
        self.threads = threads

    def _prep(self, x):
        if self.cosine:
            x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        return np.ascontiguousarray(x, dtype=np.float32)

    def build(self, x):
        self.faiss.omp_set_num_threads(self.threads)
        x = self._prep(x)
        rng = np.random.default_rng(0)
        self.index.train(x[rng.choice(len(x), min(len(x), 200000), replace=False)])
        self.index.add(x)

    def search(self, q, k, nprobe, threads):
        self.faiss.omp_set_num_threads(threads)
        self.index.nprobe = nprobe
        return self.index.search(self._prep(q), k)[1]

    def memory(self):
        return None


IVF_LIBS = {"quiver-ivfpq": QuiverIvfPq, "quiver-ivfpq-refine": lambda *a: QuiverIvfPq(*a, refine=8),
            "faiss-ivfpq": FaissIvfPq}

LIBS = {"quiver": Quiver, "quiver-sq8": lambda *a: Quiver(*a, storage="sq8"),
        "hnswlib": Hnswlib, "faiss-hnsw": FaissHnsw}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="synthetic-100000-128")
    ap.add_argument("--libs", default="quiver,hnswlib,faiss-hnsw")
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--efc", type=int, default=200)
    ap.add_argument("--ef", default="10,16,24,32,48,64,96,128,192,256")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--threads", type=int, default=performance_cores(),
                    help="default: performance cores only (hybrid CPUs skew static schedulers)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-base", type=int, default=0)
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--single-thread", action="store_true", help="also measure 1-thread QPS")
    ap.add_argument("--out", default="")
    ap.add_argument("--nlist", type=int, default=1024)
    ap.add_argument("--pq-m", type=int, default=16)
    ap.add_argument("--nprobe", default="1,2,4,8,16,32,64,128")
    ap.add_argument("--rotate", type=int, default=0,
                    help="rotate the library order by N, so no library always runs first")
    args = ap.parse_args()

    ds = datasets.load(args.dataset, args.max_base, args.max_queries)
    print(f"{ds.name}: base {ds.base.shape}, queries {ds.queries.shape}, metric {ds.metric}, "
          f"threads {args.threads}", file=sys.stderr)
    efs = [int(e) for e in args.ef.split(",")]
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out = open(args.out, "a") if args.out else None

    libs = args.libs.split(",")
    libs = libs[args.rotate % len(libs):] + libs[:args.rotate % len(libs)]
    for lib in libs:
        if lib in IVF_LIBS:
            impl = IVF_LIBS[lib](ds.base.shape[1], ds.metric, args.nlist, args.pq_m, args.threads)
            sweep = [int(e) for e in args.nprobe.split(",")]
        else:
            impl = LIBS[lib](ds.base.shape[1], ds.metric, args.M, args.efc, args.threads)
            sweep = efs
        t0 = time.perf_counter()
        impl.build(ds.base)
        build_s = time.perf_counter() - t0
        print(f"\n{impl.name}: built in {build_s:.2f}s", file=sys.stderr)
        impl.search(ds.queries, args.k, sweep[-1], args.threads)  # warm caches, threads, CPU clocks
        print(f"{'ef':>6} {'recall':>8} {'QPS':>10} {'QPS 1-thr':>10}", file=sys.stderr)
        for ef in sweep:
            secs, labels = timed(lambda: impl.search(ds.queries, args.k, ef, args.threads), args.reps)
            rec = recall_at_k(ds.neighbors, labels, args.k)
            qps = len(ds.queries) / secs
            qps1 = None
            if args.single_thread:
                s1, _ = timed(lambda: impl.search(ds.queries, args.k, ef, 1), args.reps)
                qps1 = len(ds.queries) / s1
            print(f"{ef:>6} {rec:>8.4f} {qps:>10.0f} {qps1 or 0:>10.0f}", file=sys.stderr)
            row = dict(library=impl.name, dataset=ds.name, M=args.M, efc=args.efc, ef=ef, k=args.k,
                       nlist=args.nlist if lib in IVF_LIBS else None, pq_m=args.pq_m if lib in IVF_LIBS else None,
                       recall=round(rec, 5), qps=round(qps, 1), qps_1thread=qps1 and round(qps1, 1),
                       build_s=round(build_s, 3), threads=args.threads, memory_bytes=impl.memory())
            if out:
                out.write(json.dumps(row) + "\n")
                out.flush()
        del impl


if __name__ == "__main__":
    main()
