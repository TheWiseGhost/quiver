"""A RecallKit ``VectorIndex`` backed by Quiver's HNSW index.

Drop-in replacement for ``PostgresVectorIndex``: chunk/document rows stay wherever RecallKit keeps
them (they are only used to hydrate results), while nearest-neighbour search runs in-process in
Quiver's C++ core, off the event loop (Quiver releases the GIL).

Filters keep RecallKit's contract - "filtering must happen in the index, never by over-fetching" -
via *pre-filtering*: ``SearchFilters`` are evaluated against cached document attributes (mirroring
``recall.storage.postgres.filters``, including PostgreSQL's ``jsonb @>`` rules), and the admitted
chunk labels are handed to Quiver, which routes the graph search through them (or scans them
exactly when very selective). ``top_k`` therefore means the true top-k among matching chunks.

    index = QuiverVectorIndex(chunks=storage.chunks, documents=storage.documents)
    retriever = DenseRetriever(embedder=embedder, index=index)
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from numbers import Number
from pathlib import Path
from typing import Any

import numpy as np

import quiver
from recall.core.embeddings.base import EmbeddingModelInfo, Vector
from recall.core.errors import DimensionMismatchError
from recall.core.models import Document, SearchFilters, SearchResult, SourceType
from recall.core.ports import ChunkRepository, DocumentRepository

_METADATA_LIST_FIELDS = (("file_types", "file_type"), ("authors", "author"), ("repositories", "repository"))


# ---- PostgreSQL semantics ---------------------------------------------------------------------

def _scalar_eq(a: Any, b: Any) -> bool:
    """jsonb scalar equality: types must match (true is not 1), numbers compare numerically."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, Number) and isinstance(b, Number):
        return a == b
    if a is None or b is None:
        return a is None and b is None
    return type(a) is type(b) and a == b


def _jsonb_contains(left: Any, right: Any, top_level: bool = True) -> bool:
    """PostgreSQL ``left @> right``.

    Objects: every key of ``right`` is in ``left`` with a contained value. Arrays: every element of
    ``right`` matches some element of ``left`` (a scalar matches an equal scalar, a container is
    contained in a container). The one special case - an array containing a primitive - applies
    only at the top level, exactly as in PostgreSQL.
    """
    if isinstance(right, Mapping):
        return isinstance(left, Mapping) and all(
            k in left and _jsonb_contains(left[k], v, top_level=False) for k, v in right.items())
    if isinstance(right, list):
        return isinstance(left, list) and all(
            any(_element_matches(item, r) for item in left) for r in right)
    if isinstance(left, list):
        return top_level and any(_scalar_eq(item, right) for item in left
                                 if not isinstance(item, (list, Mapping)))
    return not isinstance(left, Mapping) and _scalar_eq(left, right)


def _element_matches(item: Any, r: Any) -> bool:
    """Array element match: containers by containment (same kind), scalars by equality."""
    if isinstance(r, list):
        return isinstance(item, list) and _jsonb_contains(item, r, top_level=False)
    if isinstance(r, Mapping):
        return isinstance(item, Mapping) and _jsonb_contains(item, r, top_level=False)
    return not isinstance(item, (list, Mapping)) and _scalar_eq(item, r)


def _as_text(v: Any) -> str:
    """PostgreSQL ``metadata->>key``: strings unquoted, everything else as jsonb text."""
    if isinstance(v, str):
        return v
    return json.dumps(v, ensure_ascii=False, separators=(", ", ": "))


def _aware(t: datetime) -> datetime:
    """A naive bound is read like PostgreSQL reads a timestamp literal under a UTC session."""
    return t if t.tzinfo is not None else t.replace(tzinfo=UTC)


@dataclass
class _DocAttrs:
    """The document fields SearchFilters can reference, plus provenance for results."""

    source_type: str
    source_id: str
    created_at: datetime
    metadata: dict[str, Any]
    title: str
    uri: str

    @classmethod
    def of(cls, doc: Document) -> _DocAttrs:
        st = doc.source_type.value if isinstance(doc.source_type, SourceType) else str(doc.source_type)
        return cls(st, doc.source_id, _aware(doc.created_at), dict(doc.metadata), doc.title, doc.uri)


def document_matches(doc_id: uuid.UUID, doc: _DocAttrs, f: SearchFilters) -> bool:
    """Python mirror of ``build_document_predicates`` (AND across fields, OR within one)."""
    if f.source_types and doc.source_type not in {t.value for t in f.source_types}:
        return False
    if f.source_ids and doc.source_id not in set(f.source_ids):
        return False
    if f.document_ids and doc_id not in set(f.document_ids):
        return False
    if f.created_after is not None and not doc.created_at >= _aware(f.created_after):
        return False
    if f.created_before is not None and not doc.created_at <= _aware(f.created_before):
        return False
    for field_name, key in _METADATA_LIST_FIELDS:
        values = getattr(f, field_name)
        if values:
            v = doc.metadata.get(key)
            if v is None or _as_text(v) not in {str(x) for x in values}:
                return False
    if f.tags and not any(_jsonb_contains(doc.metadata, {"tags": [t]}) for t in f.tags):
        return False
    if f.metadata and not _jsonb_contains(doc.metadata, f.metadata):
        return False
    return True


# ---- the index ---------------------------------------------------------------------------------

@dataclass
class _ModelIndex:
    info: EmbeddingModelInfo
    index: quiver.HnswIndex
    next_label: int = 0
    label_of: dict[uuid.UUID, int] = field(default_factory=dict)                    # chunk -> live label
    chunk_of: dict[int, tuple[uuid.UUID, uuid.UUID]] = field(default_factory=dict)  # label -> (chunk, doc)
    labels_of_doc: dict[uuid.UUID, set[int]] = field(default_factory=dict)

    def forget(self, label: int) -> None:
        cid, did = self.chunk_of.pop(label)
        if self.label_of.get(cid) == label:
            del self.label_of[cid]
        labels = self.labels_of_doc.get(did)
        if labels is not None:
            labels.discard(label)
            if not labels:
                del self.labels_of_doc[did]


class QuiverVectorIndex:
    """Implements :class:`recall.core.ports.VectorIndex` with one Quiver HNSW index per model.

    Scores follow ``PostgresVectorIndex``: cosine similarity, ``score = 1 - cosine_distance``.
    Mutations (upsert, delete, save) are serialized by an asyncio lock; queries never take it.
    """

    def __init__(self, chunks: ChunkRepository, documents: DocumentRepository, *, M: int = 16,
                 ef_construction: int = 200, ef_search: int = 96, storage: str = "float32") -> None:
        self._chunks = chunks
        self._documents = documents
        self._params = dict(M=M, ef_construction=ef_construction, storage=storage)
        self.ef_search = ef_search
        self._models: dict[str, _ModelIndex] = {}
        self._docs: dict[uuid.UUID, _DocAttrs] = {}
        self._lock = asyncio.Lock()

    # ---- VectorIndex protocol -------------------------------------------------------------

    async def upsert(self, chunk_ids: Sequence[uuid.UUID], vectors: Sequence[Vector],
                     model: EmbeddingModelInfo) -> int:
        if not chunk_ids:
            return 0
        if len(chunk_ids) != len(vectors):
            raise ValueError(f"got {len(chunk_ids)} chunk ids but {len(vectors)} vectors")
        x = np.asarray(vectors, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != model.dimensions:
            raise DimensionMismatchError(model.dimensions, x.shape[-1] if x.ndim else 0)
        if not np.isfinite(x).all():
            raise ValueError("vectors must be finite")

        # A chunk id repeated within one batch: the last vector wins (one row per chunk).
        last = {cid: i for i, cid in enumerate(chunk_ids)}
        chunk_rows = {c.id: c for c in await self._chunks.get_many(list(last))}
        rows = [(cid, i, chunk_rows[cid]) for cid, i in last.items() if cid in chunk_rows]
        if not rows:
            return 0  # all deleted concurrently, as in PostgresVectorIndex
        await self._cache_documents({chunk.document_id for _, _, chunk in rows})

        async with self._lock:
            mi = self._model_index(model)
            new_labels = list(range(mi.next_label, mi.next_label + len(rows)))
            mi.next_label += len(rows)
            plan = [(label, cid, chunk.document_id, mi.label_of.get(cid))
                    for label, (cid, _, chunk) in zip(new_labels, rows)]
            # Register the new labels before inserting so concurrent queries can resolve them,
            # add the vectors, and only then retire the old ones: a failed upsert leaves the
            # previous embeddings intact (and a chunk briefly visible twice is deduplicated).
            for label, cid, did, _ in plan:
                mi.chunk_of[label] = (cid, did)
                mi.labels_of_doc.setdefault(did, set()).add(label)
            try:
                await asyncio.to_thread(mi.index.add, x[[i for _, i, _ in rows]],
                                        np.asarray(new_labels, dtype=np.uint64))
            except BaseException:
                for label, *_ in plan:
                    mi.forget(label)
                raise
            for label, cid, _, old in plan:
                mi.label_of[cid] = label
                if old is not None:
                    mi.index.remove(old)
                    mi.forget(old)
            return len(rows)

    async def query(self, vector: Vector, *, top_k: int, filters: SearchFilters | None = None,
                    model: EmbeddingModelInfo | None = None) -> list[SearchResult]:
        if top_k <= 0:
            return []
        if model is not None:
            if len(vector) != model.dimensions:
                raise DimensionMismatchError(model.dimensions, len(vector))
            mi = self._models.get(model.key)
            targets = [mi] if mi is not None else []
        else:
            # Like PostgresVectorIndex without a model filter: search every model of this width.
            targets = [mi for mi in self._models.values() if mi.info.dimensions == len(vector)]
        q = np.asarray(vector, dtype=np.float32)[None, :]

        hits: list[tuple[float, uuid.UUID, uuid.UUID]] = []  # (distance, chunk, document)
        for mi in targets:
            if len(mi.index) == 0:
                continue
            allowed = None
            if filters is not None and not filters.is_empty():
                labels = [label for did, attrs in list(self._docs.items())
                          if document_matches(did, attrs, filters)
                          for label in list(mi.labels_of_doc.get(did, ()))]
                if not labels:
                    continue
                allowed = np.fromiter(labels, dtype=np.uint64, count=len(labels))
            found, dist = await asyncio.to_thread(
                mi.index.search, q, top_k, max(self.ef_search, top_k), allowed)
            for label, d in zip(found[0].tolist(), dist[0].tolist()):
                owner = mi.chunk_of.get(label)  # resolved now: maps may change across the await
                if label >= 0 and owner is not None:
                    hits.append((d, owner[0], owner[1]))

        # During an upsert a chunk may briefly have old and new vectors; keep its best hit.
        best: dict[uuid.UUID, tuple[float, uuid.UUID]] = {}
        for d, cid, did in sorted(hits, key=lambda h: h[0]):
            best.setdefault(cid, (d, did))
        ranked = sorted(best.items(), key=lambda kv: kv[1][0])[:top_k]

        chunk_rows = {c.id: c for c in await self._chunks.get_many([cid for cid, _ in ranked])}
        results: list[SearchResult] = []
        for cid, (distance, did) in ranked:
            chunk = chunk_rows.get(cid)
            if chunk is None:
                continue
            doc = self._docs.get(did)
            results.append(SearchResult(
                chunk_id=cid, document_id=did, content=chunk.content,
                score=round(1.0 - distance, 6), rank=len(results) + 1, metadata=dict(chunk.metadata),
                document_title=doc.title if doc else None, document_uri=doc.uri if doc else None,
                source_type=SourceType(doc.source_type) if doc else None, retriever="dense"))
        return results

    async def delete_for_document(self, document_id: uuid.UUID) -> int:
        async with self._lock:
            removed = 0
            for mi in self._models.values():
                for label in list(mi.labels_of_doc.get(document_id, ())):
                    mi.index.remove(label)
                    mi.forget(label)
                    removed += 1
            self._docs.pop(document_id, None)
            return removed

    async def count(self) -> int:
        return sum(len(mi.index) for mi in self._models.values())

    async def models(self) -> list[EmbeddingModelInfo]:
        return [mi.info for mi in self._models.values() if len(mi.index) > 0]

    # ---- persistence ----------------------------------------------------------------------

    async def save(self, directory: str | Path) -> None:
        """Write a consistent snapshot: index files for a new generation, then the manifest
        (atomically, last), then delete the previous generation. A crash at any point leaves
        either the old or the new snapshot fully readable."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        async with self._lock:
            old = _read_manifest(d)
            generation = (old or {}).get("generation", 0) + 1
            manifest: dict[str, Any] = {
                "generation": generation,
                "docs": {str(k): {**v.__dict__, "created_at": v.created_at.isoformat()}
                         for k, v in self._docs.items()},
                "models": [],
            }
            for i, mi in enumerate(self._models.values()):
                name = f"g{generation}-model{i}.qvr"
                await asyncio.to_thread(mi.index.save, d / name)
                manifest["models"].append({
                    "file": name, "info": mi.info.model_dump(), "next_label": mi.next_label,
                    "chunks": {str(l): [str(c), str(doc)] for l, (c, doc) in mi.chunk_of.items()}})
            payload = json.dumps(manifest, default=str)
        await asyncio.to_thread(_atomic_write, d / "manifest.json", payload)
        for m in (old or {}).get("models", []):
            (d / m["file"]).unlink(missing_ok=True)

    @classmethod
    def load(cls, directory: str | Path, chunks: ChunkRepository, documents: DocumentRepository,
             *, mmap: bool = True, **kwargs: Any) -> QuiverVectorIndex:
        d = Path(directory)
        manifest = _read_manifest(d)
        if manifest is None:
            raise FileNotFoundError(d / "manifest.json")
        self = cls(chunks, documents, **kwargs)
        for k, v in manifest["docs"].items():
            self._docs[uuid.UUID(k)] = _DocAttrs(**{**v, "created_at": datetime.fromisoformat(v["created_at"])})
        for m in manifest["models"]:
            info = EmbeddingModelInfo(**m["info"])
            mi = _ModelIndex(info, quiver.HnswIndex.load(d / m["file"], mmap=mmap), m["next_label"])
            for l, (c, doc) in m["chunks"].items():
                label, cid, did = int(l), uuid.UUID(c), uuid.UUID(doc)
                mi.chunk_of[label] = (cid, did)
                mi.label_of[cid] = label
                mi.labels_of_doc.setdefault(did, set()).add(label)
            self._models[info.key] = mi
        return self

    # ---- internals ------------------------------------------------------------------------

    def _model_index(self, model: EmbeddingModelInfo) -> _ModelIndex:
        mi = self._models.get(model.key)
        if mi is None:
            mi = _ModelIndex(model, quiver.HnswIndex(model.dimensions, metric="cosine", **self._params))
            self._models[model.key] = mi
        elif mi.info.dimensions != model.dimensions:
            raise DimensionMismatchError(mi.info.dimensions, model.dimensions)
        return mi

    async def _cache_documents(self, document_ids: set[uuid.UUID]) -> None:
        missing = [did for did in document_ids if did not in self._docs]
        for did, doc in zip(missing, await asyncio.gather(*(self._documents.get(i) for i in missing))):
            if doc is not None:
                self._docs[did] = _DocAttrs.of(doc)


def _read_manifest(d: Path) -> dict[str, Any] | None:
    path = d / "manifest.json"
    return json.loads(path.read_text()) if path.exists() else None


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
