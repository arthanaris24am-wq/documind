"""Qdrant vector store wrapper supporting memory, docker, and cloud modes."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.chunking import TextChunk
from app.config import QdrantConfig


@dataclass(frozen=True)
class RetrievedChunk:
    """A retrieved chunk with similarity or reranker score."""

    chunk_id: str
    doc_id: str
    filename: str
    page: int
    chunk_index: int
    text: str
    score: float


@dataclass(frozen=True)
class DocumentSummary:
    """Aggregated metadata for an ingested document."""

    doc_id: str
    filename: str
    chunk_count: int


def build_qdrant_client(config: QdrantConfig) -> QdrantClient:
    """Create a QdrantClient configured for memory, docker, or cloud mode."""
    if config.mode == "memory":
        return QdrantClient(":memory:")
    if config.mode == "docker":
        return QdrantClient(host=config.host, port=config.port)
    if config.mode == "cloud":
        if not config.url or not config.api_key:
            raise ValueError("QDRANT_URL and QDRANT_API_KEY are required for cloud mode")
        return QdrantClient(url=config.url, api_key=config.api_key)
    raise ValueError(f"Unsupported Qdrant mode: {config.mode}")


class QdrantVectorStore:
    """Manages chunk storage, semantic search, and document lifecycle in Qdrant."""

    def __init__(
        self,
        config: QdrantConfig,
        vector_size: int,
        client: QdrantClient | None = None,
    ) -> None:
        self.config = config
        self.vector_size = vector_size
        self.collection_name = config.collection_name
        self.client = client if client is not None else build_qdrant_client(config)
        self.ensure_collection()

    def ensure_collection(self) -> None:
        """Create the Qdrant collection if it does not already exist."""
        existing = {col.name for col in self.client.get_collections().collections}
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )

    def upsert_chunks(
        self,
        chunks: Sequence[TextChunk],
        vectors: Sequence[Sequence[float]],
    ) -> int:
        """Upsert text chunks and their embedding vectors into Qdrant."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if not chunks:
            return 0

        points = [
            PointStruct(
                id=chunk.chunk_id,
                vector=list(vector),
                payload={
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "filename": chunk.filename,
                    "page": chunk.page,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)
        return len(points)

    def search(self, query_vector: Sequence[float], top_k: int = 4) -> list[RetrievedChunk]:
        """Search Qdrant for the top-k most similar chunks."""
        if top_k <= 0:
            return []
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=list(query_vector),
            limit=top_k,
            with_payload=True,
        )
        results: list[RetrievedChunk] = []
        for point in response.points:
            payload = point.payload or {}
            results.append(
                RetrievedChunk(
                    chunk_id=str(payload.get("chunk_id", point.id)),
                    doc_id=str(payload.get("doc_id", "")),
                    filename=str(payload.get("filename", "unknown")),
                    page=int(payload.get("page", 1)),
                    chunk_index=int(payload.get("chunk_index", 0)),
                    text=str(payload.get("text", "")),
                    score=float(point.score),
                )
            )
        return results

    def list_documents(self) -> list[DocumentSummary]:
        """Return unique ingested documents and their chunk counts."""
        counts: dict[str, int] = defaultdict(int)
        filenames: dict[str, str] = {}
        offset = None

        while True:
            records, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=256,
                offset=offset,
                with_payload=["doc_id", "filename"],
                with_vectors=False,
            )
            for record in records:
                payload = record.payload or {}
                doc_id = str(payload.get("doc_id", ""))
                if not doc_id:
                    continue
                counts[doc_id] += 1
                filenames.setdefault(doc_id, str(payload.get("filename", "unknown")))
            if next_offset is None:
                break
            offset = next_offset

        summaries = [
            DocumentSummary(
                doc_id=doc_id,
                filename=filenames[doc_id],
                chunk_count=chunk_count,
            )
            for doc_id, chunk_count in counts.items()
        ]
        summaries.sort(key=lambda item: (item.filename.lower(), item.doc_id))
        return summaries

    def delete_document(self, doc_id: str) -> int:
        """Delete all chunks belonging to doc_id and return deleted chunk count."""
        doc_filter = Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        )
        count_result = self.client.count(
            collection_name=self.collection_name,
            count_filter=doc_filter,
            exact=True,
        )
        deleted_count = int(count_result.count)
        if deleted_count == 0:
            return 0

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=FilterSelector(filter=doc_filter),
        )
        return deleted_count

    def count_points(self) -> int:
        """Return total number of stored chunks in the collection."""
        return int(self.client.count(collection_name=self.collection_name, exact=True).count)

