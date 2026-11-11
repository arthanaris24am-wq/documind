# DocuMind System Architecture

DocuMind is a Retrieval-Augmented Generation (RAG) document Q&A system built for local development on Windows 11 and 100% free cloud deployment.

## Core Components

- **FastAPI Backend**: Exposes `/health`, `/ingest`, `/query`, `/documents`, and `/documents/{id}` endpoints.
- **Streamlit UI**: Provides document upload, library management, and grounded Q&A with expandable source citations and similarity scores.
- **Embedding Model**: Uses `BAAI/bge-small-en-v1.5` via `sentence-transformers` on CPU, loaded once at FastAPI startup.
- **Vector Store**: Uses Qdrant in three supported modes:
  - `memory`: in-memory client for fast offline unit and integration tests.
  - `docker`: local Qdrant container for local development.
  - `cloud`: Qdrant Cloud free tier for production deployments.
- **LLM Client**: Abstracts Gemini API calls behind an `LLMClient` interface with exponential backoff and retry logic for rate limits.

