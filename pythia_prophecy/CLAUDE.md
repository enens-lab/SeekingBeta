# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pythia Prophecy is the Frontend and Backend-for-Frontend (BFF) component of the Pythia platform. It handles marketing to users, the login flow, user credential management, and displays user-specific information. This is the user-facing layer that integrates with pythia_divination for predictions.

## Build & Development Commands

```bash
# Install dependencies
make install            # Python backend
make install-frontend   # React frontend (npm)

# Development
make dev                # Frontend dev server with HMR (localhost:3000)
make run-dev            # Backend with hot-reload (localhost:8001)

# Production
make build              # Build React frontend
make run                # Run production server (0.0.0.0:8001)

# Cleanup
make clean
```

## Architecture

```
pythia_prophecy/
├── api/
│   ├── service.py           # FastAPI - main app with all endpoints
│   ├── models.py            # Pydantic models (User, Tier, Token, etc.)
│   ├── database.py          # SQLite user storage
│   ├── auth.py              # Password hashing, JWT tokens
│   └── email_service.py     # Email verification (dev mode prints to console)
├── data/
│   └── pythia.db            # SQLite database (auto-created)
├── frontend/
│   ├── src/
│   │   ├── api/client.js    # API client with auth
│   │   ├── context/
│   │   │   └── AuthContext.jsx  # Auth state management
│   │   ├── components/      # Reusable UI components
│   │   ├── pages/           # Route pages
│   │   │   ├── Landing.jsx
│   │   │   ├── Signup.jsx
│   │   │   ├── Login.jsx
│   │   │   ├── VerifyEmail.jsx
│   │   │   ├── Pricing.jsx
│   │   │   └── Dashboard.jsx
│   │   ├── styles/index.css
│   │   ├── App.jsx          # Router setup
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
├── main.py
├── requirements.txt
└── Makefile
```

## Authentication Flow

1. **Signup** (`/signup`) - User provides name, email, password, and selects tier
2. **Email Verification** - Verification email sent (dev mode: prints to console)
3. **Verify Email** (`/verify-email?token=...`) - Activates account, logs user in
4. **Login** (`/login`) - Returns JWT token stored in localStorage
5. **Dashboard** (`/dashboard`) - Protected route, requires auth

## Subscription Tiers

| Tier | Price | Stocks | Timeframes |
|------|-------|--------|------------|
| Free | $0 | 3 | 1d |
| Basic | $19/mo | 10 | 1d, 1h, 30m |
| Pro | $49/mo | Unlimited | All (1m to 1d) |

Tier access is enforced on the `/predict/{ticker}` endpoint.

## Key API Endpoints

### Auth
- `POST /api/auth/signup` - Register new user
- `POST /api/auth/login` - Login, returns JWT
- `POST /api/auth/verify-email` - Verify email with token
- `POST /api/auth/resend-verification` - Resend verification email
- `GET /api/auth/me` - Get current user (requires auth)

### Tiers
- `GET /api/tiers` - List all subscription tiers
- `GET /api/tiers/{tier}` - Get specific tier info

### Predictions
- `GET /predict/{ticker}?horizon=1d` - Get prediction (tier-gated)
- `GET /api/universe` - Get available stocks for user's tier
- `GET /healthz` - Health check

## Development Workflow

For development, run frontend and backend separately:

1. Terminal 1: `make run-dev` (backend on :8001)
2. Terminal 2: `make dev` (frontend on :3000 with proxy to backend)

Email verification in dev mode prints to console instead of sending real emails.

## Environment Variables

- `JWT_SECRET_KEY` - Secret for JWT signing (default: dev key)
- `EMAIL_DEV_MODE` - Set to "false" for real email sending
- `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD` - For email
- `FRONTEND_URL` - Base URL for email links (default: http://localhost:3000)

## Integration with Pythia Divination

Integrates with sibling `pythia_divination/` component for ML predictions and trading data. Falls back to demo mode with mock data if unavailable.
