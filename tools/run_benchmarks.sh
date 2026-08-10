#!/usr/bin/env bash
# Reproduces the README benchmarks. Run on an otherwise idle machine (close VMs/browsers):
# background load easily moves QPS by 2x on laptops.
#
#   tools/run_benchmarks.sh
#
# Downloads the ann-benchmarks SIFT and GloVe datasets into data/ on first use (~1 GB total).
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-.venv/bin/python}
# Performance cores only: hybrid P/E CPUs skew statically scheduled libraries (OpenMP).
THREADS=${THREADS:-$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || nproc)}
EF=10,16,24,32,48,64,96,128,192,256,384,512
NPROBE=1,2,4,8,16,32,64,128,256
mkdir -p results docs/img

cmake --preset release >/dev/null
cmake --build --preset release --target bench_kernels >/dev/null
echo "== SIMD kernel microbenchmarks"
./build/release/bench/bench_kernels --benchmark_filter='/(128|768)$' \
  --benchmark_out=results/kernels.json --benchmark_out_format=json

"$PY" tools/datasets.py sift-128-euclidean glove-100-angular
rotate=0
for ds in sift-128-euclidean glove-100-angular; do
  m=$([ "$ds" = sift-128-euclidean ] && echo 32 || echo 25)
  rm -f "results/$ds.jsonl" "results/$ds-ivfpq.jsonl"
  echo "== $ds: HNSW"
  "$PY" tools/compare.py --dataset "$ds" --threads "$THREADS" --single-thread --ef $EF --rotate $rotate \
    --libs quiver,quiver-sq8,hnswlib,faiss-hnsw --out "results/$ds.jsonl"
  echo "== $ds: IVF-PQ (nlist=2048, m=$m)"
  "$PY" tools/compare.py --dataset "$ds" --threads "$THREADS" --single-thread --nprobe $NPROBE --rotate $rotate \
    --nlist 2048 --pq-m "$m" --libs quiver-ivfpq,quiver-ivfpq-refine,faiss-ivfpq --out "results/$ds-ivfpq.jsonl"
  "$PY" tools/plot.py "results/$ds.jsonl" -o "docs/img/$ds-hnsw.png"
  "$PY" tools/plot.py "results/$ds.jsonl" -o "docs/img/$ds-hnsw-1thread.png" --key qps_1thread
  "$PY" tools/plot.py "results/$ds-ivfpq.jsonl" -o "docs/img/$ds-ivfpq.png" --min-recall 0.3
  rotate=$((rotate + 2))
done
echo "results in results/, plots in docs/img/"
