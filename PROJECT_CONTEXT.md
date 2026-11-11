# DocuMind - Project Context

This document is the canonical project specification and architectural context for **DocuMind**, a RAG document Q&A system with MLOps and 100% free deployment. Future phases must read and follow every constraint below.

## 1. Environment & Platform Constraints
- **Operating System:** Windows 11 (development).
- **Paths & File I/O:** Always use `pathlib.Path` for filesystem paths. Always open text files with `encoding="utf-8"`.
- **Scripts & Task Runners:** No bash-only scripts for local developer workflows. Provide both `Makefile` and `tasks.ps1` (PowerShell) for cross-platform / Windows task execution.
- **Console Output:** Avoid emojis in console output / logs (prevents Windows cp1252 encoding issues).
- **Python Environment:** Python 3.10.x in `.venv`. Use **CPU-only PyTorch**. Pin all dependency versions in requirements files.

## 2. Technology Stack
- **Backend API:** FastAPI
- **Frontend UI:** Streamlit
- **Embeddings:** `sentence-transformers` with `BAAI/bge-small-en-v1.5` (CPU)
- **Vector Database:** Qdrant supporting three modes:
  - `memory`: in-memory Qdrant client for unit/integration tests
  - `docker`: local Qdrant container for local development
  - `cloud`: Qdrant Cloud free tier for production via `QDRANT_URL` and `QDRANT_API_KEY`
- **LLM Integration:** Gemini API (`GEMINI_API_KEY`) behind an abstract `LLMClient` interface with retry and exponential backoff for rate limits.
- **RAG Orchestration:** LangChain (selected over LlamaIndex for modular text splitting and clean integration with custom `LLMClient`, Qdrant, and RAGAS).
- **Experiment Tracking & Data Versioning:** MLflow + DVC
- **Evaluation:** RAGAS (using Gemini as judge, throttled to respect free-tier rate limits)
- **Containerization & CI/CD:** Docker, Docker Compose, GitHub Actions
- **Monitoring:** Prometheus + Grafana

## 3. Deployment Constraints (100% Free, No Credit Card)
- **Target Platform:** Hugging Face Spaces (Docker SDK, exposed on port `7860`) paired with **Qdrant Cloud Free Tier**.
- **Strict Prohibition:** Do **NOT** use Google Cloud or any paid service requiring a credit card.
- **Container Runtime:** The HF Space container runs FastAPI (internal port `8000`) and Streamlit (port `7860`) together via a single start script.

## 4. Security & Secrets Management
- Secrets must **only** be supplied via environment variables or a local `.env` file.
- **Never** commit `.env`, API keys, or credentials to Git.
- Document every required and optional environment variable in `.env.example`.

## 5. Git & Workflow Conventions
- **Repository:** `https://github.com/arthanaris24am-wq/documind.git` (default branch: `main`).
- **Branching:** One feature branch per phase (e.g., `feat/phase-1-core-rag`), using conventional commits (`feat:`, `chore:`, `docs:`, `test:`, `ci:`), merged into `main` only after all checks pass.
- **Push Auth:** If `git push` fails due to authentication, stop immediately and inform the user of the exact steps required.

## 6. Engineering Rules
- If a command fails, diagnose and fix the root cause and retry.
- Never stub out failing parts or tests.
- Never mark a phase complete if any test or check fails.

## 7. Implementation Phases
- **Phase 0:** Project setup, folder structure, `.venv`, `.gitignore`, `.env.example`, `PROJECT_CONTEXT.md`, `HUMAN_TODO.md`, initial push.
- **Phase 1:** Core RAG (document ingestion, chunking, embeddings, Qdrant client, `LLMClient` + Gemini adapter with retry/backoff, FastAPI endpoints, Streamlit UI, unit/integration tests with Qdrant memory mode, `Makefile` & `tasks.ps1`).
- **Phase 2:** MLOps (MLflow tracking, DVC data/pipeline versioning, throttled RAGAS evaluation with Gemini judge).
- **Phase 3:** Docker & Docker Compose (local FastAPI + Streamlit + Qdrant + start script for single-container HF Spaces compatibility).
- **Phase 4:** CI/CD & Free Deployment (GitHub Actions lint/test/build + Hugging Face Spaces Docker SDK deploy on port 7860 with Qdrant Cloud free tier).
- **Phase 5:** Monitoring & Final Docs (Prometheus metrics, Grafana dashboard configuration, final `README.md`).
