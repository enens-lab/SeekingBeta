.PHONY: install install-frontend build run run-dev dev clean help sync-sports-data

help:
	@echo "Available targets:"
	@echo "  install           - Install Python dependencies"
	@echo "  install-frontend  - Install frontend (npm) dependencies"
	@echo "  build             - Build frontend for production"
	@echo "  run               - Run production server (0.0.0.0:8001)"
	@echo "  run-dev           - Run backend with hot reload (127.0.0.1:8001)"
	@echo "  dev               - Run frontend dev server (localhost:3000)"
	@echo "  clean             - Remove cache and build files"
	@echo "  sync-sports-data  - Copy PGA schedule CSVs from pythia_divination so the"
	@echo "                      Dockerfile bakes them into the prophecy image"

install:
	pip install -r requirements.txt

install-frontend:
	cd frontend && npm install

build:
	cd frontend && npm run build

run:
	python main.py --host 0.0.0.0 --port 8001

run-dev:
	python main.py --host 127.0.0.1 --port 8001 --reload --log-level debug

dev:
	cd frontend && npm run dev

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf frontend/dist frontend/node_modules/.vite 2>/dev/null || true

# Refresh prophecy-local copies of the PGA schedule CSVs that the runtime
# golf-backtest builder reads. The prophecy container cannot see
# pythia_divination at runtime, so we ship a snapshot inside the image
# (Dockerfile already does `COPY data/`). Re-run before each release if
# the upstream schedules have changed.
sync-sports-data:
	mkdir -p data/sports/pga/normalized
	@if [ -d ../pythia_divination/data/sports/pga/normalized ]; then \
	  cp ../pythia_divination/data/sports/pga/normalized/schedule_*_latest.csv data/sports/pga/normalized/ && \
	  echo "Synced PGA schedule CSVs into data/sports/pga/normalized/"; \
	else \
	  echo "ERROR: ../pythia_divination/data/sports/pga/normalized not found" >&2; \
	  exit 1; \
	fi
