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

.PHONY: help install migrate migrate-status run run-dev lint test clean

help:
	@echo "Available targets:"
	@echo "  install         Install dependencies from requirements.txt"
	@echo "  migrate         Run pending DB migrations"
	@echo "  migrate-status  Show applied and pending migrations"
	@echo "  run             Start API server (prod mode)"
	@echo "  run-dev         Start API server with reload (dev mode)"
	@echo "  lint            Run flake8 linting"
	@echo "  test            Run pytest test suite"
	@echo "  clean           Remove cache and build artifacts"

install:
	$(PIP) install -r requirements.txt

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

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache build dist *.egg-info
