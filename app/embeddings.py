"""Embedding client abstraction and SentenceTransformers implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class EmbeddingClient(ABC):
    """Abstract interface for text embedding models."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding vector dimension."""

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of document texts."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query."""


class SentenceTransformerEmbedding(EmbeddingClient):
    """SentenceTransformer embedding model loaded once at startup."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str = "cpu",
        normalize_embeddings: bool = True,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self._model = SentenceTransformer(model_name, device=device)
        dim_fn = getattr(
            self._model,
            "get_embedding_dimension",
            self._model.get_sentence_embedding_dimension,
        )
        dim = dim_fn()
        if dim is None:
            raise ValueError(f"Could not determine embedding dimension for {model_name}")
        self._dimension = int(dim)
        self._is_bge = "bge" in model_name.lower()

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            list(texts),
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        return [vec.tolist() for vec in vectors]

    def embed_query(self, text: str) -> list[float]:
        query_text = f"{BGE_QUERY_INSTRUCTION}{text}" if self._is_bge else text
        vector = self._model.encode(
            query_text,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        return vector.tolist()

