"""Tests for Qdrant memory-mode retrieval, document management, and reranking."""

from __future__ import annotations

from app.config import ChunkingConfig, QdrantConfig, RetrievalConfig, Settings
from app.rag import RAGService
from app.vectorstore import QdrantVectorStore
from tests.conftest import (
    DeterministicEmbeddingClient,
    FakeLLMClient,
    KeywordRerankerClient,
)


def test_qdrant_memory_upsert_search_list_delete(
    embedding_client: DeterministicEmbeddingClient,
    memory_store: QdrantVectorStore,
    memory_settings: Settings,
    fake_llm: FakeLLMClient,
) -> None:
    service = RAGService(
        settings=memory_settings,
        embedding_client=embedding_client,
        vector_store=memory_store,
        llm_client=fake_llm,
    )

    doc1 = service.ingest_bytes(
        "architecture.md",
        b"DocuMind uses FastAPI for REST APIs and Qdrant for vector storage.",
    )
    doc2 = service.ingest_bytes(
        "deploy.txt",
        b"Hugging Face Spaces hosts the Docker container on port 7860 for free.",
    )

    docs = service.list_documents()
    assert len(docs) == 2
    assert {d.doc_id for d in docs} == {doc1.doc_id, doc2.doc_id}

    retrieved, reranked = service.retrieve("Which port does Hugging Face Spaces use?", top_k=2)
    assert reranked is False
    assert len(retrieved) == 2
    assert retrieved[0].filename == "deploy.txt"
    assert retrieved[0].page == 1
    assert retrieved[0].chunk_id

    deleted_count = service.delete_document(doc1.doc_id)
    assert deleted_count == doc1.chunks_indexed
    assert service.delete_document(doc1.doc_id) == 0
    remaining = service.list_documents()
    assert [d.doc_id for d in remaining] == [doc2.doc_id]


def test_optional_reranker_reorders_candidates() -> None:
    embedder = DeterministicEmbeddingClient(dim=32)
    settings = Settings(
        chunking=ChunkingConfig(chunk_size=200, chunk_overlap=20),
        retrieval=RetrievalConfig(top_k=1, use_reranker=True, rerank_candidate_multiplier=3),
        qdrant=QdrantConfig(mode="memory", collection_name="rerank_test"),
    )
    store = QdrantVectorStore(config=settings.qdrant, vector_size=embedder.dimension)
    service = RAGService(
        settings=settings,
        embedding_client=embedder,
        vector_store=store,
        llm_client=FakeLLMClient(),
        reranker_client=KeywordRerankerClient(),
    )

    service.ingest_bytes("alpha.txt", b"General introduction to vector databases and search.")
    service.ingest_bytes(
        "beta.txt",
        b"Cross-encoder reranking scores query and passage pairs directly.",
    )

    chunks, reranked = service.retrieve("cross-encoder reranking pairs")
    assert reranked is True
    assert len(chunks) == 1
    assert chunks[0].filename == "beta.txt"

