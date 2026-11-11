"""Optional cross-encoder reranker for retrieved chunks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from app.vectorstore import RetrievedChunk


class RerankerClient(ABC):
    """Abstract interface for passage reranking."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Rerank candidate chunks by relevance to query."""


class CrossEncoderReranker(RerankerClient):
    """Cross-encoder reranker backed by sentence-transformers CrossEncoder."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
    ) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.device = device
        self._model = CrossEncoder(model_name, device=device)

    def rerank(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        if not chunks or top_k <= 0:
            return []
        pairs = [(query, chunk.text) for chunk in chunks]
        scores = self._model.predict(pairs, show_progress_bar=False)
        rescored = [
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                filename=chunk.filename,
                page=chunk.page,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                score=float(score),
            )
            for chunk, score in zip(chunks, scores, strict=True)
        ]
        rescored.sort(key=lambda item: item.score, reverse=True)
        return rescored[:top_k]

