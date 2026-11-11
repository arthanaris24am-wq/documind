# DocuMind

A Retrieval-Augmented Generation (RAG) document Q&A system built with FastAPI, Streamlit, Qdrant, Sentence-Transformers, LangChain, Gemini API, MLflow, DVC, RAGAS, Docker, Prometheus, and Grafana, designed for 100% free deployment on Hugging Face Spaces + Qdrant Cloud.

## Project Status

- **Phase 0 (Setup):** Initialized repository structure, `.venv`, configuration templates, and project context.

## Directory Structure

```text
documind/
├── app/                # FastAPI backend and core RAG services
├── pipelines/          # Ingestion, DVC, MLflow, and RAGAS evaluation pipelines
├── ui/                 # Streamlit frontend application
├── configs/            # YAML/JSON configuration files
├── data/               # Sample and processed datasets (data/raw/ is gitignored)
├── tests/              # Unit and integration tests
├── monitoring/         # Prometheus and Grafana configurations
├── .github/workflows/  # GitHub Actions CI/CD workflows
├── PROJECT_CONTEXT.md  # Architectural rules and phase roadmap
├── HUMAN_TODO.md       # Manual setup tasks (API keys, cloud accounts, secrets)
├── .env.example        # Documented environment variables template
├── requirements.txt
└── requirements-dev.txt
```

## Quick Start (Phase 0)

1. Activate the virtual environment on Windows (PowerShell):
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
2. Copy `.env.example` to `.env` and populate your secrets (see `HUMAN_TODO.md`).
3. See `PROJECT_CONTEXT.md` for full architectural details and upcoming phases.
