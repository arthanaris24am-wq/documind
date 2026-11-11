"""Grounded prompt construction for RAG question answering."""

from __future__ import annotations

from collections.abc import Sequence

from app.vectorstore import RetrievedChunk

INSUFFICIENT_CONTEXT_RESPONSE = "I don't know based on the provided documents"

SYSTEM_INSTRUCTIONS = (
    "You are DocuMind, a precise document Q&A assistant.\n"
    "Follow these rules strictly:\n"
    "1. Answer ONLY using facts explicitly stated in the provided Context.\n"
    "2. Cite every factual claim inline using [filename, p.<page>] format.\n"
    "3. If the Context does not contain enough information to answer the question, "
    f'respond with exactly: "{INSUFFICIENT_CONTEXT_RESPONSE}"\n'
    "4. Do not extrapolate, guess, or use external knowledge."
)


def format_context(chunks: Sequence[RetrievedChunk]) -> str:
    """Format retrieved chunks into numbered source blocks."""
    blocks: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        header = (
            f"[{idx}] Source: {chunk.filename} | Page: {chunk.page} | "
            f"Chunk ID: {chunk.chunk_id} | Score: {chunk.score:.4f}"
        )
        blocks.append(f"{header}\n{chunk.text}")
    return "\n\n---\n\n".join(blocks)


def build_grounded_prompt(question: str, chunks: Sequence[RetrievedChunk]) -> str:
    """Build a strictly grounded prompt with source citations."""
    context_text = format_context(chunks) if chunks else "(No relevant context retrieved.)"
    return (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question.strip()}\n\n"
        "Answer:"
    )

