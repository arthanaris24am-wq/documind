# DocuMind

A Retrieval-Augmented Generation (RAG) document Q&A system built with FastAPI, Streamlit, Qdrant, Sentence-Transformers, LangChain, Gemini API, MLflow, DVC, RAGAS, Docker, Prometheus, and Grafana, designed for 100% free deployment on Hugging Face Spaces + Qdrant Cloud.

## Project Status

- **Phase 0 (Setup):** Initialized repository structure, `.venv`, configuration templates, and project context.
- **Phase 1 (Core RAG):** Implemented PDF/TXT/MD document loaders, LangChain chunking, `BAAI/bge-small-en-v1.5` embeddings (loaded once at startup), optional cross-encoder reranking, Qdrant vector store (`memory`/`docker`/`cloud`), abstract `LLMClient` with Gemini exponential-backoff adapter, grounded prompt with source citations, FastAPI backend, Streamlit UI, sample documents, and offline pytest suite.

## Directory Structure

```text
documind/
├── app/                # FastAPI backend and core RAG modules
├── pipelines/          # Ingestion, DVC, MLflow, and RAGAS evaluation pipelines
├── ui/                 # Streamlit frontend application
├── configs/            # YAML configuration (configs/config.yaml)
├── data/
│   └── raw_samples/    # Committed sample documents (PDF, TXT, MD)
├── tests/              # Offline unit and integration tests
├── monitoring/         # Prometheus and Grafana configurations
├── .github/workflows/  # GitHub Actions CI/CD workflows
├── Makefile            # Cross-platform task runner
├── tasks.ps1           # Windows PowerShell task runner
├── pyproject.toml      # Ruff and pytest configuration
├── PROJECT_CONTEXT.md  # Architectural rules and phase roadmap
├── HUMAN_TODO.md       # Manual setup tasks (API keys, cloud accounts, secrets)
├── .env.example        # Documented environment variables template
├── requirements.txt
└── requirements-dev.txt
```

## Quick Start (Windows 11 PowerShell)

1. Activate the virtual environment and install dependencies:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   .\tasks.ps1 install-dev
   ```
2. Copy `.env.example` to `.env` and set `GEMINI_API_KEY` (required only for live LLM queries).
3. Run linting and offline tests:
   ```powershell
   .\tasks.ps1 lint
   .\tasks.ps1 test
   ```
4. Start the FastAPI backend (port `8000`) and Streamlit UI (port `7860`) in separate terminals:
   ```powershell
   .\tasks.ps1 run-api
   .\tasks.ps1 run-ui
   ```

## API Endpoints

- `GET /health`: Service status, Qdrant mode, embedding model, reranker flag, and chunk count.
- `POST /ingest`: Upload and index `.pdf`, `.txt`, or `.md` files with `filename`, `page`, `doc_id`, `chunk_index`, and `chunk_id` metadata.
- `POST /query`: Retrieve context (with optional cross-encoder reranking) and return grounded answer, source chunks, scores, and latency in ms.
- `GET /documents`: List indexed documents and chunk counts.
- `DELETE /documents/{id}`: Remove all chunks for a document ID.
