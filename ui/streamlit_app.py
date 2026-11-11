"""Streamlit frontend for DocuMind RAG Q&A."""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")


def fetch_health(base_url: str) -> dict | None:
    try:
        response = httpx.get(f"{base_url}/health", timeout=5.0)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError:
        return None


def fetch_documents(base_url: str) -> list[dict]:
    try:
        response = httpx.get(f"{base_url}/documents", timeout=10.0)
        response.raise_for_status()
        return response.json().get("documents", [])
    except httpx.HTTPError:
        return []


def ingest_uploaded_files(base_url: str, uploaded_files: list) -> dict:
    files_payload = [
        ("files", (uploaded.name, uploaded.getvalue(), "application/octet-stream"))
        for uploaded in uploaded_files
    ]
    response = httpx.post(f"{base_url}/ingest", files=files_payload, timeout=120.0)
    response.raise_for_status()
    return response.json()


def delete_document(base_url: str, doc_id: str) -> dict:
    response = httpx.delete(f"{base_url}/documents/{doc_id}", timeout=15.0)
    response.raise_for_status()
    return response.json()


def ask_question(base_url: str, question: str, top_k: int) -> dict:
    response = httpx.post(
        f"{base_url}/query",
        json={"question": question, "top_k": top_k},
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()


def main() -> None:
    st.set_page_config(page_title="DocuMind RAG Q&A", layout="wide")
    st.title("DocuMind - Document Q&A")
    st.caption("Ask grounded questions across your uploaded PDF, TXT, and Markdown documents.")

    with st.sidebar:
        st.header("System Status")
        health = fetch_health(API_BASE_URL)
        if health:
            st.success(f"API Online ({health['qdrant_mode']} mode)")
            st.text(f"Collection: {health['collection_name']}")
            st.text(f"Total Chunks: {health['total_chunks']}")
            st.text(f"Embedder: {health['embedding_model']}")
            st.text(f"Reranker: {'ON' if health['reranker_enabled'] else 'OFF'}")
        else:
            st.error(f"API unreachable at {API_BASE_URL}")

        st.divider()
        st.header("Upload Documents")
        uploaded_files = st.file_uploader(
            "Select PDF, TXT, or MD files",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
        )
        if st.button("Ingest Documents", disabled=not uploaded_files):
            try:
                result = ingest_uploaded_files(API_BASE_URL, uploaded_files)
                st.success(f"Indexed {result['total_chunks_indexed']} chunks.")
                st.rerun()
            except httpx.HTTPStatusError as exc:
                st.error(f"Ingestion failed: {exc.response.text}")
            except httpx.HTTPError as exc:
                st.error(f"Request error: {exc}")

        st.divider()
        st.header("Indexed Documents")
        docs = fetch_documents(API_BASE_URL)
        if not docs:
            st.info("No documents indexed yet.")
        else:
            for doc in docs:
                col_info, col_btn = st.columns([3, 1])
                col_info.markdown(
                    f"**{doc['filename']}**  \n`{doc['doc_id']}` ({doc['chunk_count']} chunks)"
                )
                if col_btn.button("Delete", key=f"del_{doc['doc_id']}"):
                    try:
                        delete_document(API_BASE_URL, doc["doc_id"])
                        st.rerun()
                    except httpx.HTTPError as exc:
                        st.error(f"Delete failed: {exc}")

        st.divider()
        top_k = st.slider("Top-K Chunks", min_value=1, max_value=10, value=4)

    question = st.text_input(
        "Ask a question about your documents",
        placeholder="e.g., What embedding model does DocuMind use?",
    )
    if st.button("Ask Question", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving context and generating answer..."):
                try:
                    data = ask_question(API_BASE_URL, question, top_k=top_k)
                    st.subheader("Answer")
                    st.write(data["answer"])
                    st.caption(
                        f"Model: {data['model']} | Latency: {data['latency_ms']} ms | "
                        f"Reranked: {data['reranked']}"
                    )

                    st.subheader("Sources")
                    sources = data.get("sources", [])
                    if not sources:
                        st.info("No source chunks retrieved.")
                    else:
                        for idx, src in enumerate(sources, start=1):
                            title = (
                                f"[{idx}] {src['filename']} (Page {src['page']}) "
                                f"- Score: {src['score']:.4f}"
                            )
                            with st.expander(title):
                                st.text(f"Chunk ID: {src['chunk_id']} | Doc ID: {src['doc_id']}")
                                st.write(src["text"])
                except httpx.HTTPStatusError as exc:
                    st.error(f"Query failed ({exc.response.status_code}): {exc.response.text}")
                except httpx.HTTPError as exc:
                    st.error(f"Connection error: {exc}")


if __name__ == "__main__":
    main()

