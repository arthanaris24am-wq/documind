"""Pydantic schemas for FastAPI request and response payloads."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    app_name: str
    qdrant_mode: str
    collection_name: str
    embedding_model: str
    reranker_enabled: bool
    llm_configured: bool
    total_chunks: int


class IngestFileResult(BaseModel):
    doc_id: str
    filename: str
    pages_loaded: int
    chunks_indexed: int


class IngestResponse(BaseModel):
    files: list[IngestFileResult]
    total_chunks_indexed: int


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question to answer from documents.")
    top_k: int | None = Field(default=None, ge=1, le=20)


class SourceChunk(BaseModel):
    chunk_id: str
    doc_id: str
    filename: str
    page: int
    chunk_index: int
    text: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    latency_ms: float
    model: str
    reranked: bool


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total_documents: int


class DeleteDocumentResponse(BaseModel):
    doc_id: str
    deleted_chunks: int

