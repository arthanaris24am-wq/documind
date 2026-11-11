"""Shared fixtures and deterministic offline test doubles."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

import pytest
from fastapi.testclient import TestClient

from app.config import QdrantConfig, Settings
from app.embeddings import EmbeddingClient
from app.llm import LLMClient
from app.main import create_app
from app.reranker import RerankerClient
from app.vectorstore import QdrantVectorStore, RetrievedChunk


class DeterministicEmbeddingClient(EmbeddingClient):
    """Offline hash-based bag-of-words embedding client for fast unit tests."""

    def __init__(self, dim: int = 32) -> None:
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def _encode_one(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        tokens = [tok.strip(".,!?:;()[]{}\"'").lower() for tok in text.split()]
        for token in tokens:
            if not token:
                continue
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:2], "big") % self._dim
            sign = 1.0 if digest[2] % 2 == 0 else -1.0
            vec[bucket] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            vec[0] = 1.0
            return vec
        return [v / norm for v in vec]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._encode_one(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._encode_one(text)


class FakeLLMClient(LLMClient):
    """Offline mock LLM client that echoes grounded context metadata."""

    def __init__(self, fixed_response: str | None = None) -> None:
        self._fixed_response = fixed_response
        self.prompts: list[str] = []

    @property
    def model_name(self) -> str:
        return "mock-gemini-2.0-flash"

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self._fixed_response is not None:
            return self._fixed_response
        return "Mocked grounded answer based on provided context [sample, p.1]."


class KeywordRerankerClient(RerankerClient):
    """Deterministic reranker prioritizing chunks with exact query keyword overlap."""

    def rerank(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        query_terms = {w.lower() for w in query.split() if len(w) > 2}
        scored: list[RetrievedChunk] = []
        for chunk in chunks:
            text_lower = chunk.text.lower()
            hits = sum(1 for term in query_terms if term in text_lower)
            scored.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    filename=chunk.filename,
                    page=chunk.page,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    score=float(hits) + chunk.score * 0.01,
                )
            )
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]


def build_minimal_pdf_bytes(pages: Sequence[str]) -> bytes:
    """Create a valid multi-page uncompressed PDF 1.4 document in pure Python."""
    objects: list[bytes] = []
    # 1: Catalog, 2: Pages, 3: Font, then 2 objects per page (Page + Content)
    page_obj_ids = [4 + i * 2 for i in range(len(pages))]
    kids_str = " ".join(f"{pid} 0 R" for pid in page_obj_ids)

    objects.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    objects.append(
        f"2 0 obj\n<< /Type /Pages /Kids [{kids_str}] /Count {len(pages)} >>\nendobj\n".encode(
            "ascii"
        )
    )
    objects.append(
        b"3 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    )

    for idx, page_text in enumerate(pages):
        page_id = 4 + idx * 2
        content_id = page_id + 1
        escaped = (
            page_text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        )
        stream = f"BT\n/F1 12 Tf\n72 720 Td\n({escaped}) Tj\nET\n".encode("latin-1")
        page_obj = (
            f"{page_id} 0 obj\n"
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>\n"
            f"endobj\n"
        ).encode("ascii")
        content_obj = (
            f"{content_id} 0 obj\n<< /Length {len(stream)} >>\nstream\n".encode("ascii")
            + stream
            + b"endstream\nendobj\n"
        )
        objects.append(page_obj)
        objects.append(content_obj)

    header = b"%PDF-1.4\n"
    offsets: list[int] = [0]
    body = bytearray(header)
    for obj in objects:
        offsets.append(len(body))
        body.extend(obj)

    xref_pos = len(body)
    body.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    body.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        body.extend(f"{off:010d} 00000 n \n".encode("ascii"))

    body.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_pos}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(body)


@pytest.fixture
def memory_settings() -> Settings:
    return Settings(
        qdrant=QdrantConfig(mode="memory", collection_name="test_collection"),
    )


@pytest.fixture
def embedding_client() -> DeterministicEmbeddingClient:
    return DeterministicEmbeddingClient(dim=32)


@pytest.fixture
def fake_llm() -> FakeLLMClient:
    return FakeLLMClient()


@pytest.fixture
def memory_store(
    memory_settings: Settings,
    embedding_client: DeterministicEmbeddingClient,
) -> QdrantVectorStore:
    return QdrantVectorStore(
        config=memory_settings.qdrant,
        vector_size=embedding_client.dimension,
    )


@pytest.fixture
def api_client(
    memory_settings: Settings,
    embedding_client: DeterministicEmbeddingClient,
    memory_store: QdrantVectorStore,
    fake_llm: FakeLLMClient,
) -> TestClient:
    app = create_app(
        settings=memory_settings,
        embedding_client=embedding_client,
        vector_store=memory_store,
        llm_client=fake_llm,
    )
    with TestClient(app) as client:
        yield client

