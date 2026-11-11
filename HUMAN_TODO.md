# Human To-Do List (Manual Steps & Secrets)

This file tracks manual actions that require your personal accounts, credentials, or UI confirmations outside the repository.

## Phase 0 (Current)
- [ ] Create a free Google AI Studio account (no credit card required) and generate a `GEMINI_API_KEY` at https://aistudio.google.com/apikey.
- [ ] Copy `.env.example` to `.env` in the repository root and paste your `GEMINI_API_KEY`.
- [ ] Verify GitHub push access works from this machine (or configure a GitHub Personal Access Token / Git Credential Manager if `git push` prompts for authentication).

## Upcoming Phases (Do Not Need Yet)
- [ ] **Phase 3 (Local Containers):** Ensure Docker Desktop is running on Windows 11 before testing `docker compose`.
- [ ] **Phase 4 (Free Cloud Deployment - No Credit Card):**
  - [ ] Create a free Qdrant Cloud cluster (Free Tier, 1GB, no credit card) at https://cloud.qdrant.io/ and save `QDRANT_URL` and `QDRANT_API_KEY`.
  - [ ] Create a free Hugging Face account and a new Space using the **Docker** SDK (`port 7860`) at https://huggingface.co/spaces.
  - [ ] Add `GEMINI_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, and `QDRANT_MODE=cloud` as Space Secrets in Hugging Face, and configure `HF_TOKEN` / Space deployment secrets in GitHub Actions.
