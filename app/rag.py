"""Core RAG service orchestrating ingestion, retrieval, reranking, and generation."""

from __future__ import annotations

import time
from dataclasses import dataclass

from app.chunking import chunk_documents
from app.config import Settings
from app.embeddings import EmbeddingClient
from app.llm import LLMClient, LLMConfigurationError
from app.loaders import load_document_from_bytes
from app.prompts import INSUFFICIENT_CONTEXT_RESPONSE, build_grounded_prompt
from app.reranker import RerankerClient
from app.vectorstore import DocumentSummary, QdrantVectorStore, RetrievedChunk


@dataclass(frozen=True)
class IngestedDocumentOutcome:
    doc_id: str
    filename: str
    pages_loaded: int
    chunks_indexed: int


@dataclass(frozen=True)
class QueryOutcome:
    answer: str
    sources: list[RetrievedChunk]
    latency_ms: float
    model: str
    reranked: bool


class RAGService:
    """Orchestrates document ingestion and grounded question answering."""

    def __init__(
        self,
        settings: Settings,
        embedding_client: EmbeddingClient,
        vector_store: QdrantVectorStore,
        llm_client: LLMClient | None = None,
        reranker_client: RerankerClient | None = None,
    ) -> None:
        self.settings = settings
        self.embedding_client = embedding_client
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.reranker_client = reranker_client

    def ingest_bytes(self, filename: str, content: bytes) -> IngestedDocumentOutcome:
        """Load, chunk, embed, and upsert a document from raw bytes."""
        loaded_docs = load_document_from_bytes(filename=filename, content=content)
        if not loaded_docs:
            raise ValueError(f"Document '{filename}' contains no extractable text")

        chunks = chunk_documents(
            loaded_docs,
            chunk_size=self.settings.chunking.chunk_size,
            chunk_overlap=self.settings.chunking.chunk_overlap,
        )
        if not chunks:
            raise ValueError(f"Document '{filename}' produced no non-empty chunks")

        vectors = self.embedding_client.embed_documents([chunk.text for chunk in chunks])
        indexed_count = self.vector_store.upsert_chunks(chunks, vectors)
        return IngestedDocumentOutcome(
            doc_id=loaded_docs[0].doc_id,
            filename=loaded_docs[0].filename,
            pages_loaded=len(loaded_docs),
            chunks_indexed=indexed_count,
        )

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
    ) -> tuple[list[RetrievedChunk], bool]:
        """Retrieve relevant chunks and optionally apply cross-encoder reranking."""
        target_k = top_k if top_k is not None else self.settings.retrieval.top_k
        use_reranker = (
            self.settings.retrieval.use_reranker and self.reranker_client is not None
        )
        candidate_k = (
            target_k * self.settings.retrieval.rerank_candidate_multiplier
            if use_reranker
            else target_k
        )

        query_vector = self.embedding_client.embed_query(question)
        candidates = self.vector_store.search(query_vector=query_vector, top_k=candidate_k)
        if not candidates:
            return [], False

        if use_reranker and self.reranker_client is not None:
            reranked = self.reranker_client.rerank(question, candidates, top_k=target_k)
            return reranked, True

        return candidates[:target_k], False

    def query(self, question: str, top_k: int | None = None) -> QueryOutcome:
        """Answer a user question grounded strictly in retrieved document chunks."""
        start = time.perf_counter()
        cleaned_question = question.strip()
        if not cleaned_question:
            raise ValueError("Question must not be empty")

        chunks, reranked = self.retrieve(cleaned_question, top_k=top_k)
        if not chunks:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            model_name = (
                self.llm_client.model_name
                if self.llm_client is not None
                else self.settings.llm.model_name
            )
            return QueryOutcome(
                answer=INSUFFICIENT_CONTEXT_RESPONSE,
                sources=[],
                latency_ms=round(elapsed_ms, 2),
                model=model_name,
                reranked=reranked,
            )

        if self.llm_client is None:
            raise LLMConfigurationError(
                "GEMINI_API_KEY is not configured. Set it in .env or environment variables."
            )

        prompt = build_grounded_prompt(cleaned_question, chunks)
        answer = self.llm_client.generate(prompt)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return QueryOutcome(
            answer=answer,
            sources=chunks,
            latency_ms=round(elapsed_ms, 2),
            model=self.llm_client.model_name,
            reranked=reranked,
        )

    def list_documents(self) -> list[DocumentSummary]:
        """List all indexed documents."""
        return self.vector_store.list_documents()

    def delete_document(self, doc_id: str) -> int:
        """Delete a document and its chunks by doc_id."""
        return self.vector_store.delete_document(doc_id)

