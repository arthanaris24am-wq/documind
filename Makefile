PYTHON ?= .venv/Scripts/python.exe

.PHONY: help install install-dev lint format test run-api run-ui clean

help:
	@echo "DocuMind Makefile Targets:"
	@echo "  install      Install pinned production dependencies (CPU PyTorch)"
	@echo "  install-dev  Install pinned dev/test/lint dependencies"
	@echo "  lint         Run ruff check across codebase"
	@echo "  format       Run ruff format across codebase"
	@echo "  test         Run pytest unit and integration suite"
	@echo "  run-api      Start FastAPI backend on port 8000"
	@echo "  run-ui       Start Streamlit frontend on port 7860"
	@echo "  clean        Remove caches and temporary artifacts"

install:
	$(PYTHON) -m pip install -r requirements.txt

install-dev: install
	$(PYTHON) -m pip install -r requirements-dev.txt

lint:
	$(PYTHON) -m ruff check app ui tests

format:
	$(PYTHON) -m ruff format app ui tests

test:
	$(PYTHON) -m pytest

run-api:
	$(PYTHON) -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

run-ui:
	$(PYTHON) -m streamlit run ui/streamlit_app.py --server.port 7860

clean:
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
	$(PYTHON) -c "import shutil, pathlib; shutil.rmtree(pathlib.Path('.pytest_cache'), ignore_errors=True)"
	$(PYTHON) -c "import shutil, pathlib; shutil.rmtree(pathlib.Path('.ruff_cache'), ignore_errors=True)"
