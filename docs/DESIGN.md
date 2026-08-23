# Quiver design notes

This document explains how Quiver is built and why: the data layout, the concurrency argument,
the file format, and a log of the performance experiments (including the ones that did not pay
off). Numbers quoted here come from an Apple M5 (4P + 6E cores) and are noted with their setup.

## 1. Layering

```
python/quiver (nanobind)         integrations/recallkit (async VectorIndex adapter)
        │                                    │
        ▼                                    ▼
include/quiver/*.h   HnswIndex · IvfPqIndex · FlatIndex · ScalarQuantizer · ProductQuantizer
        │
src/internal         ChunkedStorage · CentroidTable (GEMM-style microkernel) · file format · CRC32C
src/simd             scalar / NEON / AVX2+FMA kernels, runtime dispatch
src/thread_pool.cc   fork-join pool used by every batch operation
```

Public headers expose only value types and pimpl'd index classes, so the hot internals (node
layout, kernel policies) can change without breaking the ABI or recompiling users.

## 2. Distance kernels and dispatch

Each kernel (`l2sqr`, `dot`, and the asymmetric SQ8 variants) exists in three forms:

* **scalar**: the reference. Without `-ffast-math` the compiler cannot reassociate float sums,
  so these stay genuinely scalar and make an honest baseline.
* **NEON**: four independent `float32x4_t` accumulators (16 floats/iteration) so consecutive FMAs
  do not serialize on FMA latency. SQ8 decodes 16 codes per iteration with `vmovl` widening plus
  one FMA per lane group (`x = vmin + code * vscale`).
* **AVX2+FMA**: four `__m256` accumulators (32 floats/iteration). Functions carry
  `__attribute__((target("avx2,fma")))` rather than compiling the file with `-mavx2`, so the
  binary still runs on CPUs without AVX2. Dispatch selects them only after CPUID confirms support.

`simd::active()` resolves once. `QUIVER_SIMD=scalar|neon|avx2` overrides it, which is how CI
tests every ISA. AVX2 is exercised on Apple Silicon by building x86-64 and running under
Rosetta 2, which executes AVX2 but does not advertise it in CPUID.

On AArch64 NEON is mandatory, so the HNSW search loop is instantiated with *inlined* NEON
kernel functors instead of calling through the dispatch table (`FloatSpace<kIp, Kernel>`). That
removes an indirect call per distance and lets the compiler schedule a distance computation's
loads together with the surrounding traversal.

## 3. HNSW

### Node storage: `ChunkedStorage`

Every per-node array (vectors, layer-0 adjacency, metadata) lives in a `ChunkedStorage`: a fixed
table of up to 2^15 chunk pointers, each chunk an anonymous `mmap` of 2^10 to 2^16 node slots sized
to about 64 MB. A node id maps to `chunks[id >> shift] + (id & mask) * stride`.

* **Growth never moves data.** Adding a chunk publishes one pointer, so a reader holding a valid
  id can dereference it with no lock while other threads grow the index.
* **Lazy and zeroed.** Untouched pages cost nothing, and fresh slots read as zero, which is
  exactly "empty adjacency list".
* **mmap-compatible.** A loaded file's sections are laid out contiguously, so full chunks can
  point straight into a file mapping (see §5).

Layer-0 adjacency is `[count, ids[2M]]` (uint32) per node. Upper layers (needed by ~1/M of
nodes) come from a bump arena. Node metadata (label, level, flags, a 1-byte spinlock) is a
24-byte POD record. Concurrently accessed bytes are only touched through `std::atomic_ref`, so
the record can live in zero-filled chunk memory.

### Concurrency argument

Goal: `search`, `add` and `remove` callable concurrently from any threads, with searches taking
**no locks at all**.

1. **Edges are published with release stores and read with acquire loads**
   (`std::atomic_ref<uint32_t>`). A writer appends `id` to a list by storing the id, then the
   count. A reader that observes the new count (acquire) observes the id, and transitively
   everything the inserting thread wrote before publishing: the new node's vector, metadata and
   its own adjacency lists.
2. **A node becomes reachable only after it is completely wired.** Insertion runs in two
   phases: (a) top-down, find neighbors and write the node's *own* lists on every layer; (b)
   publish the reverse edges, which is the only thing that makes the node reachable. An earlier
   version interleaved the phases per layer. A concurrent search could then reach the node on an
   upper layer, descend onto it, and start its layer-0 beam from a node whose layer-0 list was
   still empty, returning fewer than k results. A Python stress test caught this. The fix also
   brought parallel-build graph quality up to that of a sequential build (self-recall misses on
   20k nodes went from 11–13 to 1–2).
3. **Writers hold at most one per-node spinlock at a time** (own list, then each neighbor's list
   in turn), so there is no lock-ordering cycle and no deadlock. hnswlib holds the new node's lock
   for the whole insertion while locking neighbors.
4. **Readers tolerate concurrent pruning.** A reader may see a mix of old and new ids while a
   list is re-pruned. Every id it can see refers to a fully published node, so the worst case is
   a slightly different traversal, never invalid memory.
5. **The entry point** is a single packed `atomic<uint64_t>` holding (node, top level), so the
   pair is read and swapped together. Only inserts that may raise the top level take
   `entry_mu`.
6. **Labels → slots** is a hash map behind a `shared_mutex`, touched once per `add()` batch and
   on remove/contains, never on the search path (except the exact-scan filter fallback).

All of this is checked by ThreadSanitizer in CI, including a test that runs searches, inserts
and removals simultaneously and asserts every search returns exactly k results.

### Insert and search

Standard HNSW (Malkov & Yashunin, 2018). Levels come from `-ln(U) / ln(M)`, with U derived by
hashing `(seed, slot)`, so single-threaded builds are bit-for-bit reproducible. Neighbors are
chosen with the diversity heuristic (Algorithm 4). New nodes take M neighbors on every layer and
lists are capped at 2M on layer 0 (see the experiment log for why not 2M for new nodes).

The beam search (`search_layer`) runs in two passes per expanded node. Pass 1 reads the
adjacency list, filters visited ids (an epoch-tagged `uint16` array, so clearing is O(1)), and
prefetches every fresh neighbor's vector. Pass 2 computes the distances. Issuing all prefetches
before any compute lets the memory system service the misses in parallel. The search also
prefetches the adjacency list of the next frontier node. Scratch buffers (heaps, visited set)
are `thread_local`, so the search path performs no allocation.

### Filtering and deletion

`Filter` exposes `contains(label)` and, optionally, `cardinality()` plus `for_each()`. Filtered
search keeps *rejected* nodes in the frontier (they still route) but out of the results, and
keeps expanding until it has `ef` admitted results. When the filter admits at most
`exact_scan_threshold` (default 2%) of the index, graph search would wander through mostly
rejected nodes, so Quiver scans the admitted vectors exactly instead. `remove()` is a soft delete
through the same mechanism.

## 4. IVF-PQ

* Coarse k-means over `nlist` cells. PQ (m sub-quantizers × 256 centroids) on **residuals**
  `x − c(x)`, which have far less variance than raw vectors.
* **L2 precomputed tables** (the FAISS decomposition):
  `‖q − c − r‖² = ‖q − c‖² + Σⱼ (‖rⱼ‖² + 2⟨cⱼ, rⱼ⟩) − 2Σⱼ⟨qⱼ, rⱼ⟩`. The middle term depends only
  on (cell, sub-space, code) and is built once (`nlist × m × 256` floats, used when it fits in
  256 MB). Probing a cell then costs `m × 256` additions instead of a fresh `256 × d` table.
  Single-thread QPS at nprobe=64 went from 589 to 3,569.
* For inner product and cosine, `⟨q, c + r⟩ = ⟨q, c⟩ + ⟨q, r⟩`, and the second table is the
  same for every cell.
* ADC scanning interleaves four codes so their dependent table lookups overlap.
* Optional exact re-ranking of the best `k × refine_factor` candidates with stored float32 vectors.

### `CentroidTable`: k-means and PQ without per-centroid calls

Distances from a vector to many centroids (k-means assignment, PQ encoding, coarse probing) are
computed over a **transposed, padded** centroid array, so the inner loop runs across centroids.
That loop is element-wise (no horizontal reduction), so it vectorizes without fast-math, and there
is no per-centroid call. For 4-dimensional PQ sub-vectors, calling a SIMD kernel per centroid
was almost pure call overhead.

For batch assignment, `nearest_batch` uses a **register-blocked microkernel**: a 4-point ×
16-centroid tile accumulated in 16 vector registers (GCC/Clang `vector_size` types, which lower to
NEON or SSE). Each centroid value loaded feeds 4 FMAs instead of one load-FMA-store. Clang would
not auto-vectorize the plain-array version ("unsafe dependent memory operations"), and that
version was 4× slower than the simple transposed loop. The explicit vector-type version was
1.6× faster than the simple loop (coarse k-means, 100k × 128, k=512: 4.2 s → 2.6 s).

## 5. File format

```
FileHeader (64 B: magic, version, kind, endian tag, section count, file size, CRCs)
SectionEntry[n] (tag, crc32c, offset, size)
sections, each 64-byte aligned and raw little-endian
```

* **Crash-safe saves**: write `path.tmp.<pid>.<n>`, `fsync` (`F_FULLFSYNC` on macOS, because
  plain `fsync` there stops at the drive cache), `rename`, then fsync the directory.
* **Integrity**: the header and section table are always CRC32C-verified (hardware CRC32C on
  ARMv8 and SSE4.2). Per-section CRCs are verified by default. Single-bit flips anywhere are
  detected (tested).
* **Untrusted input**: even with checksums disabled, a structural validator checks every
  count, offset, level and edge (each edge `u→v` on layer `l` must point at a ready node that
  exists on layer `l`) before a search can touch them. A 300-round corruption test runs under
  ASan/UBSan, and a libFuzzer target runs in CI.
* **Zero-copy mmap loading**: with `mmap=true` the file is mapped `MAP_PRIVATE` +
  `PROT_READ|PROT_WRITE`, and full chunks of the vector and adjacency arrays alias the mapping
  directly. Loading is O(metadata), pages fault in on demand, and the index stays **fully
  mutable**: writes go to copy-on-write pages and never reach the file. The trailing partial
  chunk is copied into owned memory so appends cannot run past the mapping. Saving over the
  mapped file is safe, because the atomic rename leaves the old inode alive for the mapping.

## 6. Experiment log

Measured on 1M × 128 synthetic vectors unless noted. The development machine had heavy
background load (~5.5 cores busy), so only effects well outside ±10–15% noise are claimed. Every
A/B used the same binary with environment-variable switches, interleaved, best-of-5 per query.

| Change | Result | Kept? |
|---|---|---|
| Two-phase insert (wire node fully, then publish) | Fixed short result lists under concurrent insert. Parallel-build unreachable nodes 12 → 2 per 20k | ✅ |
| Prefetch all fresh neighbors before computing distances | p50 ≈ 100 µs vs ≈ 120 µs without (≈ 20%) | ✅ |
| Prefetch next frontier node's adjacency | Within noise | ✅ (free) |
| New nodes take 2M layer-0 neighbors (FAISS-style) | Same recall, ~10% lower QPS, 30% slower build | ❌ |
| Inline NEON kernels into the search loop | ~4% at 1M, within noise | ✅ (no cost) |
| IVF-PQ precomputed L2 tables + transposed LUT | 6× single-thread QPS at nprobe=64 | ✅ |
| PQ encode via transposed table + 2-pass argmin | 43 µs → 9.7 µs per vector | ✅ |
| k-means: auto-vectorized 2-D tile | 4× *slower* (did not vectorize) | ❌ |
| k-means: explicit vector-type microkernel | 1.6× faster than the transposed loop | ✅ |

## 7. Verification

| Layer | What runs |
|---|---|
| Unit and property tests | 75 GoogleTest cases: kernel equivalence across ISAs and awkward lengths, recall targets per metric/storage, filters at 0.5–50% selectivity, persistence round trips (read and mmap), single-bit-flip detection, 300 random corruptions |
| Concurrency | searches + inserts + removes + label lookups simultaneously, asserting every search returns exactly k results; run under ThreadSanitizer |
| Memory safety | full suite under ASan + UBSan |
| Untrusted input | libFuzzer on `HnswIndex::load` with checksums off (≈600k executions / 3 min locally, also in CI) |
| ISA coverage | `QUIVER_SIMD=scalar|neon|avx2` forces each kernel set; AVX2 also verified under Rosetta 2 |
| Python | pytest for the bindings (dtype coercion, GIL-released concurrency, mmap) and the RecallKit adapter (exact cosine top-k, filter semantics vs. Postgres) |
| Static analysis | `-Wall -Wextra -Wconversion -Wsign-conversion -Wshadow -Werror`, clang-tidy (bugprone, performance, concurrency), clang-format |

### Defects found by review, and how each was closed

Two independent review passes were run against the code (the second specifically re-checked
the first pass's fixes, and found two regressions they had introduced). Every confirmed defect has a
regression test in `tests/test_regressions.cc`.

| Defect | Root cause | Fix |
|---|---|---|
| `contains`/`remove`/`stats`/duplicate-`add` racing the first `add` crashed | slot ids were published (label map, `next_slot`) before their storage chunk existed | reserve storage inside the label lock, before publishing; `next_slot` stored with release |
| Crafted file could make later inserts corrupt validated layers | upper-layer blocks / sections were bounds-checked but not checked for aliasing | strict upper-layer layout (consecutive, node order) and no overlapping sections |
| IVF-PQ refine read `front()` of an empty heap | `k * refine_factor` overflow wrapped to 0 | saturating candidate budget |
| IVF-PQ could be half-trained after an exception, or re-trained under readers | trained state assigned piecemeal, without the lock | train into locals; commit once under the exclusive lock |
| IVF-PQ inner-product recall loss | cells assigned by L2 but probed by inner product | assign by max inner product; normalize cosine centroids |
| Cross-pool `parallel_for` nesting could deadlock | only same-pool nesting ran inline | nested calls from any pool's chunk run inline |
| A filter that itself searches corrupted the outer search | one thread-local scratch per thread | per-thread scratch *stack* (`ScratchLease`) |
| SQ8 auto-trained on a 1-vector batch collapsed recall | ranges learned from too little data | first batch must hold ≥ 256 vectors; train only once |
| NaN inputs broke heap ordering (UB) | comparators are not a strict weak order with NaN | reject non-finite inputs; map NaN distances to +inf |
| Concurrent search could return fewer than k results | a node was reachable before its layer-0 list existed | two-phase insert (§3) |
| *Second pass:* IVF-PQ `add` + `search_batch` + `add` hung forever | `add` held the index lock across `parallel_for` while pool chunks took that lock; a waiting writer blocks new readers | rule: no index lock is held across `parallel_for`, and pool chunks never take one (batch locks are taken once by the caller) |
| *Second pass:* `size()` briefly read 2^64−1 | the live counter was bumped after linking, so a racing `remove` decremented first | count the node before publishing it |
| *Second pass:* Python `search(k=huge)` overflowed a buffer | `nq * k` wrapped | overflow-checked allocation |
| *Second pass:* label 2^64−1 aliased the "empty slot" sentinel | not reserved | rejected in every `add`; Python labels limited to < 2^63 so int64 results are exact |
| *Second pass:* RecallKit adapter: lost vector on failed upsert, duplicate ids, query racing delete, JSONB edge cases | remove-before-add, map lookups after an `await`, loose containment | add-then-retire with rollback, ids resolved before awaiting, PostgreSQL-exact `@>` |

The invariant that rules out this whole class of deadlock: **no thread waits on a thread pool
while holding a lock that pool chunks may acquire.** Concretely:

| Lock | Held across `parallel_for`? | Taken inside pool chunks? |
|---|---|---|
| IVF-PQ / Flat index lock (shared) | yes, by `search_batch`'s caller | no |
| IVF-PQ index lock (exclusive) | no (append only, no pool) | no |
| HNSW `writers_mu` (shared by `add`, exclusive by `save`) | yes, by `add` | no |
| HNSW `labels_mu` | no (`claim_slots`, `remove`, lookups never use a pool) | yes (filtered exact scan) |
| HNSW per-node spinlocks, `entry_mu` | no | yes (never held while waiting on anything) |

Every row has at most one "yes", so no wait-for cycle can form through a pool.
