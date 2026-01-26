.PHONY: install install-frontend build run run-dev dev clean help

help:
	@echo "Available targets:"
	@echo "  install          - Install Python dependencies"
	@echo "  install-frontend - Install frontend (npm) dependencies"
	@echo "  build            - Build frontend for production"
	@echo "  run              - Run production server (0.0.0.0:8001)"
	@echo "  run-dev          - Run backend with hot reload (127.0.0.1:8001)"
	@echo "  dev              - Run frontend dev server (localhost:3000)"
	@echo "  clean            - Remove cache and build files"

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
