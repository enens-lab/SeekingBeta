# Pythia API Makefile

PYTHON      := python
PIP         := pip
UVICORN     := uvicorn
APP         := api.service:app
MAIN        := main.py
MIGRATE     := storage.scripts.migrate
ENV_FILE    := .env

# Default DB URL check
export DATABASE_URL ?= $(shell grep DATABASE_URL $(ENV_FILE) | cut -d '=' -f2)

.PHONY: help install install-pga-torch migrate migrate-status run run-dev lint test clean benchmark-pga-torch train-pga-torch backfill-mlb-statcast train-mlb-torch

help:
	@echo "Available targets:"
	@echo "  install         Install dependencies from requirements.txt"
	@echo "  install-pga-torch  Create/update the Apple-Silicon PGA PyTorch env"
	@echo "  migrate         Run pending DB migrations"
	@echo "  migrate-status  Show applied and pending migrations"
	@echo "  run             Start API server (prod mode)"
	@echo "  run-dev         Start API server with reload (dev mode)"
	@echo "  lint            Run flake8 linting"
	@echo "  test            Run pytest test suite"
	@echo "  benchmark-pga-torch  Run an MPS/CPU benchmark for the PGA torch env"
	@echo "  train-pga-torch  Rebuild the PGA dataset and train the multitask torch model"
	@echo "  backfill-mlb-statcast  Collect multi-season Statcast exports into the local cache"
	@echo "  train-mlb-torch  Rebuild the MLB dataset and train the PyTorch home-win model"
	@echo "  clean           Remove cache and build artifacts"

install:
	$(PIP) install -r requirements.txt

install-pga-torch:
	bash scripts/setup_pga_torch_env.sh

migrate:
	$(PYTHON) -m $(MIGRATE)

migrate-status:
	$(PYTHON) -m $(MIGRATE) status

run:
	$(PYTHON) $(MAIN) --host 0.0.0.0 --port 8000 --log-level info

run-dev:
	$(PYTHON) $(MAIN) --reload --host 127.0.0.1 --port 8000 --log-level debug

lint:
	flake8 .

test:
	pytest -q

benchmark-pga-torch:
	.venv-torch/bin/python -m sports.pga.benchmark_torch_env --device auto

train-pga-torch:
	bash scripts/train_pga_multitask_torch.sh

backfill-mlb-statcast:
	bash scripts/backfill_mlb_statcast.sh

train-mlb-torch:
	bash scripts/train_mlb_torch.sh

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache build dist *.egg-info
