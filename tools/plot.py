"""Plot recall@k vs QPS curves from compare.py / ann_bench JSONL output.

    python tools/plot.py results/sift.jsonl -o docs/img/sift.png [--key qps_1thread]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STYLE = {
    "quiver": dict(color="#2563eb", marker="o", lw=2.2, zorder=5),
    "quiver-sq8": dict(color="#60a5fa", marker="o", lw=1.8, ls="--", zorder=4),
    "hnswlib": dict(color="#f59e0b", marker="s", lw=1.6),
    "faiss-hnsw": dict(color="#10b981", marker="^", lw=1.6),
    "quiver-ivfpq": dict(color="#2563eb", marker="o", lw=2.2),
    "quiver-ivfpq-refine8": dict(color="#60a5fa", marker="o", lw=1.8, ls="--"),
    "faiss-ivfpq": dict(color="#10b981", marker="^", lw=1.6),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--key", default="qps", help="qps or qps_1thread")
    ap.add_argument("--title", default="")
    ap.add_argument("--min-recall", type=float, default=0.7)
    ap.add_argument("--libs", default="", help="comma-separated subset to plot")
    args = ap.parse_args()

    series = defaultdict(list)
    meta = {}
    for path in args.inputs:
        for line in open(path):
            r = json.loads(line)
            if args.libs and r["library"] not in args.libs.split(","):
                continue
            if r.get(args.key):
                series[r["library"]].append((r["recall"], r[args.key]))
                meta = r
    fig, ax = plt.subplots(figsize=(7, 4.6), dpi=150)
    for lib, pts in series.items():
        pts = sorted(p for p in pts if p[0] >= args.min_recall)
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], label=lib, **STYLE.get(lib, {}))
    ax.set_yscale("log")
    ax.set_xlabel(f"Recall@{meta.get('k', 10)}")
    threads = 1 if args.key == "qps_1thread" else meta.get("threads")
    ax.set_ylabel(f"Queries per second ({threads} thread{'s' if threads != 1 else ''}, log scale)")
    ax.set_title(args.title or f"{meta.get('dataset')} - M={meta.get('M')}, efC={meta.get('efc')}")
    ax.set_xlim(right=1.0)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
