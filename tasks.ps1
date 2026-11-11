param(
    [Parameter(Position = 0)]
    [ValidateSet(
        "help",
        "install",
        "install-dev",
        "install-mlops",
        "lint",
        "format",
        "test",
        "eval",
        "eval-quick",
        "experiments",
        "experiments-quick",
        "run-api",
        "run-ui",
        "clean"
    )]
    [string]$Task = "help"
)

$ErrorActionPreference = "Stop"

$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

switch ($Task) {
    "help" {
        Write-Host "DocuMind PowerShell Tasks:"
        Write-Host "  .\tasks.ps1 install            Install pinned production dependencies (CPU PyTorch)"
        Write-Host "  .\tasks.ps1 install-dev        Install pinned dev/test/lint dependencies"
        Write-Host "  .\tasks.ps1 install-mlops      Install pinned MLOps dependencies (MLflow, DVC, RAGAS)"
        Write-Host "  .\tasks.ps1 lint               Run ruff check across codebase"
        Write-Host "  .\tasks.ps1 format             Run ruff format across codebase"
        Write-Host "  .\tasks.ps1 test               Run pytest unit and integration suite"
        Write-Host "  .\tasks.ps1 eval               Run full RAGAS evaluation pipeline"
        Write-Host "  .\tasks.ps1 eval-quick         Run quick RAGAS evaluation (5 questions)"
        Write-Host "  .\tasks.ps1 experiments        Run chunk_size x top_k grid experiments"
        Write-Host "  .\tasks.ps1 experiments-quick  Run quick grid experiments (5 questions per run)"
        Write-Host "  .\tasks.ps1 run-api            Start FastAPI backend on port 8000"
        Write-Host "  .\tasks.ps1 run-ui             Start Streamlit frontend on port 7860"
        Write-Host "  .\tasks.ps1 clean              Remove caches and temporary artifacts"
    }
    "install" {
        & $Python -m pip install -r (Join-Path $PSScriptRoot "requirements.txt")
    }
    "install-dev" {
        & $Python -m pip install -r (Join-Path $PSScriptRoot "requirements.txt")
        & $Python -m pip install -r (Join-Path $PSScriptRoot "requirements-dev.txt")
    }
    "install-mlops" {
        & $Python -m pip install -r (Join-Path $PSScriptRoot "requirements.txt")
        & $Python -m pip install -r (Join-Path $PSScriptRoot "requirements-mlops.txt")
    }
    "lint" {
        & $Python -m ruff check app ui pipelines tests
    }
    "format" {
        & $Python -m ruff format app ui pipelines tests
    }
    "test" {
        & $Python -m pytest
    }
    "eval" {
        & $Python -m pipelines.evaluate
    }
    "eval-quick" {
        & $Python -m pipelines.evaluate --quick
    }
    "experiments" {
        & $Python -m pipelines.run_experiments
    }
    "experiments-quick" {
        & $Python -m pipelines.run_experiments --quick
    }
    "run-api" {
        & $Python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    }
    "run-ui" {
        & $Python -m streamlit run (Join-Path $PSScriptRoot "ui\streamlit_app.py") --server.port 7860
    }
    "clean" {
        Get-ChildItem -Path $PSScriptRoot -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Remove-Item -Path (Join-Path $PSScriptRoot ".pytest_cache") -Recurse -Force -ErrorAction SilentlyContinue
        Remove-Item -Path (Join-Path $PSScriptRoot ".ruff_cache") -Recurse -Force -ErrorAction SilentlyContinue
    }
}
