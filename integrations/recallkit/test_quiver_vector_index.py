"""Tests for the RecallKit adapter against in-memory repositories.

    pip install "recall @ git+https://github.com/TheWiseGhost/recall" pytest-asyncio
    pytest integrations/recallkit
"""

import uuid
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

pytest.importorskip("recall")

from recall.core.embeddings.base import EmbeddingModelInfo  # noqa: E402
from recall.core.models import Chunk, Document, SearchFilters, SourceType  # noqa: E402
from recall.core.ports import VectorIndex  # noqa: E402

from quiver_vector_index import QuiverVectorIndex, _DocAttrs, _jsonb_contains, document_matches  # noqa: E402

DIM = 32
MODEL = EmbeddingModelInfo(provider="test", model="m", dimensions=DIM)


class MemChunks:
    def __init__(self):
        self.rows = {}

    async def get_many(self, ids):
        return [self.rows[i] for i in ids if i in self.rows]


class MemDocs:
    def __init__(self):
        self.rows = {}

    async def get(self, document_id):
        return self.rows.get(document_id)


def make_corpus(n_docs=40, chunks_per_doc=25, seed=0):
    rng = np.random.default_rng(seed)
    chunks, docs = MemChunks(), MemDocs()
    vectors = {}
    base = datetime(2026, 1, 1, tzinfo=UTC)
    for d in range(n_docs):
        doc = Document.create(
            source_id=f"doc{d}", source_type=SourceType.GITHUB if d % 2 else SourceType.NOTION,
            title=f"Doc {d}", content="...", uri=f"mem://{d}",
            metadata={"author": f"a{d % 4}", "tags": ["x"] if d % 3 == 0 else ["y"], "team": {"name": f"t{d % 5}"}},
            created_at=base + timedelta(days=d))
        docs.rows[doc.id] = doc
        for c in range(chunks_per_doc):
            ch = Chunk(id=uuid.uuid4(), document_id=doc.id, content=f"chunk {d}.{c}", position=c,
                       token_count=3, checksum="x")
            chunks.rows[ch.id] = ch
            vectors[ch.id] = rng.standard_normal(DIM).astype(np.float32)
    return chunks, docs, vectors


def cosine_topk(vectors, q, k, allowed=None):
    ids = [i for i in vectors if allowed is None or i in allowed]
    m = np.stack([vectors[i] for i in ids])
    sims = (m / np.linalg.norm(m, axis=1, keepdims=True)) @ (q / np.linalg.norm(q))
    return [ids[j] for j in np.argsort(-sims)[:k]]


@pytest.fixture
async def loaded():
    chunks, docs, vectors = make_corpus()
    index = QuiverVectorIndex(chunks, docs, ef_search=200)
    ids = list(vectors)
    assert await index.upsert(ids, [vectors[i].tolist() for i in ids], MODEL) == len(ids)
    return index, chunks, docs, vectors


def test_satisfies_protocol():
    assert isinstance(QuiverVectorIndex(MemChunks(), MemDocs()), VectorIndex)


async def test_query_matches_exact_cosine(loaded):
    index, chunks, docs, vectors = loaded
    q = np.random.default_rng(9).standard_normal(DIM).astype(np.float32)
    results = await index.query(q.tolist(), top_k=10, model=MODEL)
    assert [r.rank for r in results] == list(range(1, 11))
    assert [r.chunk_id for r in results] == cosine_topk(vectors, q, 10)
    assert all(r.retriever == "dense" and r.document_title.startswith("Doc") for r in results)
    assert results[0].score >= results[-1].score


@pytest.mark.parametrize("filters", [
    SearchFilters(source_types=[SourceType.GITHUB]),
    SearchFilters(authors=["a1", "a2"], tags=["x"]),
    SearchFilters(metadata={"team": {"name": "t3"}}),
    SearchFilters(created_after=datetime(2026, 1, 30, tzinfo=UTC)),
])
async def test_filters_are_prefilters(loaded, filters):
    index, chunks, docs, vectors = loaded
    allowed = {cid for cid, ch in chunks.rows.items()
               if document_matches(ch.document_id, _DocAttrs.of(docs.rows[ch.document_id]), filters)}
    assert allowed
    q = np.random.default_rng(5).standard_normal(DIM).astype(np.float32)
    results = await index.query(q.tolist(), top_k=8, filters=filters, model=MODEL)
    # top_k keeps its meaning: the true top-8 among matching chunks, not "top-8 then discard".
    assert [r.chunk_id for r in results] == cosine_topk(vectors, q, 8, allowed)


async def test_upsert_replaces_and_delete_removes(loaded):
    index, chunks, docs, vectors = loaded
    n = await index.count()
    some = next(iter(vectors))
    target = -vectors[some]
    await index.upsert([some], [target.tolist()], MODEL)
    assert await index.count() == n
    top = await index.query(target.tolist(), top_k=1, model=MODEL)
    assert top[0].chunk_id == some and top[0].score == pytest.approx(1.0, abs=1e-5)

    doc_id = chunks.rows[some].document_id
    assert await index.delete_for_document(doc_id) == 25
    assert await index.count() == n - 25
    results = await index.query(target.tolist(), top_k=50, model=MODEL)
    assert all(r.document_id != doc_id for r in results)


async def test_dimension_mismatch_and_empty(loaded):
    from recall.core.errors import DimensionMismatchError

    index, *_ = loaded
    with pytest.raises(DimensionMismatchError):
        await index.query([0.0] * (DIM + 1), top_k=3, model=MODEL)
    empty = QuiverVectorIndex(MemChunks(), MemDocs())
    assert await empty.query([0.0] * DIM, top_k=3) == []
    assert await empty.models() == []


async def test_save_load_roundtrip(loaded, tmp_path):
    index, chunks, docs, vectors = loaded
    await index.save(tmp_path)
    await index.save(tmp_path)  # second generation replaces the first
    assert len(list(tmp_path.glob("*.qvr"))) == 1
    again = QuiverVectorIndex.load(tmp_path, chunks, docs, ef_search=200)
    q = np.random.default_rng(3).standard_normal(DIM).astype(np.float32).tolist()
    f = SearchFilters(source_types=[SourceType.NOTION])
    assert [r.chunk_id for r in await again.query(q, top_k=5, filters=f, model=MODEL)] == \
           [r.chunk_id for r in await index.query(q, top_k=5, filters=f, model=MODEL)]
    assert await again.models() == [MODEL]


# ---- regressions from review -------------------------------------------------------------------

@pytest.mark.parametrize("left,right,expected", [
    ({"a": ["x"]}, {"a": "x"}, False),            # nested scalar is not contained in an array
    ({"a": True}, {"a": 1}, False),               # booleans are not numbers
    ([1, 2, [1, 3]], [1, 3], False),              # PostgreSQL docs example
    ([1, 2, [1, 3]], [[1, 3]], True),
    (["foo", "bar"], "foo", True),                # top-level array contains a primitive
    ({"a": {"b": [1, 2]}, "c": 3}, {"a": {"b": [2]}}, True),
    ({"a": 1.0}, {"a": 1}, True),                 # numbers compare numerically
])
def test_jsonb_containment_matches_postgres(left, right, expected):
    assert _jsonb_contains(left, right) is expected


async def test_repeated_chunk_in_one_upsert_is_last_wins(loaded):
    index, chunks, docs, vectors = loaded
    n = await index.count()
    cid = next(iter(vectors))
    a, b = vectors[cid], -vectors[cid]
    assert await index.upsert([cid, cid], [a.tolist(), b.tolist()], MODEL) == 1
    assert await index.count() == n
    top = await index.query(b.tolist(), top_k=3, model=MODEL)
    assert [r.chunk_id for r in top].count(cid) == 1 and top[0].chunk_id == cid
    assert await index.delete_for_document(chunks.rows[cid].document_id) == 25
    assert await index.count() == n - 25


async def test_failed_upsert_keeps_previous_vector(loaded):
    index, chunks, docs, vectors = loaded
    n = await index.count()
    cid = next(iter(vectors))
    with pytest.raises(ValueError):
        await index.upsert([cid], [[float("nan")] * DIM], MODEL)
    assert await index.count() == n
    assert (await index.query(vectors[cid].tolist(), top_k=1, model=MODEL))[0].chunk_id == cid


async def test_query_survives_concurrent_delete(loaded):
    import asyncio

    index, chunks, docs, vectors = loaded

    class SlowChunks(MemChunks):
        def __init__(self, rows):
            super().__init__()
            self.rows = rows

        async def get_many(self, ids):
            await asyncio.sleep(0.02)
            return await super().get_many(ids)

    index._chunks = SlowChunks(chunks.rows)
    cid = next(iter(vectors))
    did = chunks.rows[cid].document_id

    async def deleter():
        await asyncio.sleep(0.005)
        await index.delete_for_document(did)

    results, _ = await asyncio.gather(index.query(vectors[cid].tolist(), top_k=5, model=MODEL), deleter())
    assert len(results) <= 5


async def test_naive_datetime_filters_are_utc(loaded):
    index, *_ = loaded
    naive = await index.query([0.1] * DIM, top_k=5, model=MODEL,
                              filters=SearchFilters(created_after=datetime(2026, 1, 30)))
    aware = await index.query([0.1] * DIM, top_k=5, model=MODEL,
                              filters=SearchFilters(created_after=datetime(2026, 1, 30, tzinfo=UTC)))
    assert [r.chunk_id for r in naive] == [r.chunk_id for r in aware] and naive


async def test_query_without_model_spans_models(loaded):
    index, chunks, docs, vectors = loaded
    other = EmbeddingModelInfo(provider="test", model="other", dimensions=DIM)
    some = list(vectors)[:10]
    target = np.ones(DIM, dtype=np.float32)
    await index.upsert(some[:1], [target.tolist()], other)
    results = await index.query(target.tolist(), top_k=3)
    assert results[0].chunk_id == some[0] and results[0].score == pytest.approx(1.0, abs=1e-5)
