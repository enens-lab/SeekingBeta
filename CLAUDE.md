# CLAUDE.md - Pythia Project

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

Pythia is a personal automated trading platform with two main components:

1. **pythia_divination** - Trading engine containing ML strategies, algorithmic trading strategies, automated trade execution, backtesting, and related trading functionality (FastAPI + scikit-learn + PyTorch)
2. **pythia_prophecy** - Frontend + Backend-for-Frontend (BFF) that handles marketing, user login flow, user credentials, and user-specific information display (FastAPI + React)

## Work Guidelines

### Commit Practices

**IMPORTANT: Commit changes frequently after completing discrete steps.**

After completing any of the following, create a commit:
- Adding or modifying a feature
- Fixing a bug
- Adding/updating tests
- Updating documentation
- Database migrations
- Configuration changes

Commit message format:
```
<type>(<scope>): <short description>

<optional longer description>

Co-Authored-By: Claude <noreply@anthropic.com>
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `style`

### Daily Logging

Maintain a daily work log at `.claude/logs/YYYY-MM-DD.md` with:
- Tasks completed
- Files modified
- Key decisions made
- Issues encountered

### Project Summaries

Keep project summaries updated at:
- `PROJECT_SUMMARY.md` - High-level overview of the entire Pythia project
- Individual `CLAUDE.md` files in each subproject

## Quick Commands

```bash
# Development
cd pythia_divination && python main.py --reload    # ML API (port 8000)
cd pythia_prophecy && make run-dev                  # Frontend API (port 8001)
cd pythia_prophecy/frontend && npm run dev          # React dev (port 3000)

# Database
cd pythia_divination && python -m storage.scripts.migrate

# Training
cd pythia_divination && python -m models.train --all

# Docker
docker-compose up -d   # Start all services
```

## Architecture

```
pythia/
├── pythia_divination/          # Trading Engine (ML, algo trading, automated trades, backtesting)
│   ├── api/                    # FastAPI endpoints
│   ├── models/                 # ML models (GB, LR, RF, LSTM)
│   ├── features/               # Feature engineering
│   ├── storage/                # PostgreSQL layer
│   ├── artifacts/              # Trained model files
│   └── static/                 # Dashboard UI
│
├── pythia_prophecy/            # User Platform (Frontend + BFF: marketing, login, credentials, user info)
│   ├── api/                    # FastAPI + auth
│   ├── frontend/               # React app
│   └── data/                   # SQLite user DB
│
├── .claude/                    # Claude Code config
│   ├── settings.json
│   └── logs/                   # Daily work logs
│
├── PROJECT_SUMMARY.md          # Project overview
└── docker-compose.yml          # Container orchestration
```

## Environment Setup

1. Create `.env` files in each subproject (see `.env.example`)
2. Install Python deps: `pip install -r requirements.txt`
3. Install frontend deps: `cd pythia_prophecy/frontend && npm install`
4. Run migrations: `python -m storage.scripts.migrate`

## Key Technologies

- **Backend**: FastAPI, asyncpg, PostgreSQL
- **ML**: scikit-learn, PyTorch, pandas, numpy
- **Frontend**: React, Vite, TailwindCSS
- **Auth**: JWT, bcrypt
- **Data**: Alpaca Markets API
