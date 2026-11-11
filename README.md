# DocuMind

A Retrieval-Augmented Generation (RAG) document Q&A system built with FastAPI, Streamlit, Qdrant, Sentence-Transformers, LangChain, Gemini API, MLflow, DVC, RAGAS, Docker, Prometheus, and Grafana, designed for 100% free deployment on Hugging Face Spaces + Qdrant Cloud.

## Project Status

- **Phase 0 (Setup):** Initialized repository structure, `.venv`, configuration templates, and project context.
- **Phase 1 (Core RAG):** Implemented PDF/TXT/MD document loaders, LangChain chunking, `BAAI/bge-small-en-v1.5` embeddings (loaded once at startup), optional cross-encoder reranking, Qdrant vector store (`memory`/`docker`/`cloud`), abstract `LLMClient` with Gemini exponential-backoff adapter, grounded prompt with source citations, FastAPI backend, Streamlit UI, sample documents, and offline pytest suite.
- **Phase 2 (MLOps):** Added `data/eval/qa_dataset.json` (22 Q&A pairs), throttled + retryable RAGAS evaluation (`pipelines/evaluate.py`) using Gemini as judge and local embeddings, MLflow tracking, `chunk_size x top_k` grid search (`pipelines/run_experiments.py` producing `best_config.json`), DVC data versioning with local remote at `../dvc_storage`, and non-LLM unit tests.

## Directory Structure

```text
documind/
├── app/                # FastAPI backend and core RAG modules
├── pipelines/          # RAGAS evaluation and MLflow experiment grid scripts
├── ui/                 # Streamlit frontend application
├── configs/            # YAML configuration (configs/config.yaml)
├── data/
│   ├── raw_samples/    # Committed sample documents (PDF, TXT, MD)
│   ├── eval/           # Evaluation Q&A dataset (qa_dataset.json)
│   └── corpus.dvc      # DVC-tracked corpus snapshot (remote at ../dvc_storage)
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
├── requirements-dev.txt
└── requirements-mlops.txt
```

## Quick Start (Windows 11 PowerShell)

1. Activate the virtual environment and install dependencies:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   .\tasks.ps1 install-dev
   .\tasks.ps1 install-mlops
   ```
2. Copy `.env.example` to `.env` and set `GEMINI_API_KEY` (required for live LLM queries and RAGAS Gemini judge).
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

## Evaluation & Experiment Tracking (MLflow + RAGAS)

- Quick evaluation (5 questions, throttled Gemini judge + local `BAAI/bge-small-en-v1.5` embeddings):
  ```powershell
  .\.venv\Scripts\python.exe -m pipelines.evaluate --quick
  ```
- Full evaluation (all 22 Q&A pairs):
  ```powershell
  .\.venv\Scripts\python.exe -m pipelines.evaluate
  ```
- Offline retrieval-only evaluation (no `GEMINI_API_KEY` required):
  ```powershell
  .\.venv\Scripts\python.exe -m pipelines.evaluate --quick --retrieval-only
  ```
- Grid search (`chunk_size` in `256, 512, 1024` x `top_k` in `3, 5`) writing `best_config.json`:
  ```powershell
  .\.venv\Scripts\python.exe -m pipelines.run_experiments --quick
  ```
- Launch MLflow UI to inspect runs, params, metrics, and artifacts:
  ```powershell
  .\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
  ```

## DVC Data Versioning Workflow

DVC is initialized for `data/` with a local remote configured at `../dvc_storage` (`.dvc/config`).

1. Check data tracking status:
   ```powershell
   .\.venv\Scripts\python.exe -m dvc status
   ```
2. After updating files in `data/corpus/`:
   ```powershell
   .\.venv\Scripts\python.exe -m dvc add data/corpus
   .\.venv\Scripts\python.exe -m dvc push
   git add data/corpus.dvc data/.gitignore
   git commit -m "chore: update DVC tracked corpus"
   ```
3. Restore data on a clean checkout:
   ```powershell
   .\.venv\Scripts\python.exe -m dvc pull
   ```

## API Endpoints

- `GET /health`: Service status, Qdrant mode, embedding model, reranker flag, and chunk count.
- `POST /ingest`: Upload and index `.pdf`, `.txt`, or `.md` files with `filename`, `page`, `doc_id`, `chunk_index`, and `chunk_id` metadata.
- `POST /query`: Retrieve context (with optional cross-encoder reranking) and return grounded answer, source chunks, scores, and latency in ms.
- `GET /documents`: List indexed documents and chunk counts.
- `DELETE /documents/{id}`: Remove all chunks for a document ID.
