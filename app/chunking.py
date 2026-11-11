"""Configurable text chunking using LangChain text splitters."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.loaders import LoadedDocument


@dataclass(frozen=True)
class TextChunk:
    """A single text chunk with source metadata."""

    chunk_id: str
    doc_id: str
    filename: str
    page: int
    chunk_index: int
    text: str


def build_chunk_id(doc_id: str, page: int, chunk_index: int) -> str:
    """Generate a deterministic UUID string for Qdrant point storage."""
    seed = f"{doc_id}:page={page}:chunk={chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))


def chunk_documents(
    documents: Sequence[LoadedDocument],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[TextChunk]:
    """Split loaded documents into overlapping text chunks with metadata."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[TextChunk] = []
    running_index = 0
    for doc in documents:
        split_texts = splitter.split_text(doc.text)
        for text in split_texts:
            cleaned = text.strip()
            if not cleaned:
                continue
            chunk_id = build_chunk_id(doc.doc_id, doc.page, running_index)
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    filename=doc.filename,
                    page=doc.page,
                    chunk_index=running_index,
                    text=cleaned,
                )
            )
            running_index += 1
    return chunks

