#!/usr/bin/env bash
# ==============================================================================
# ResearchLens AI — Service Startup Commands & Cheat Sheet
# ==============================================================================
# This file provides the terminal commands to start PostgreSQL, Backend,
# and Frontend services for local development.
#
# You can copy and paste the commands into separate terminal windows,
# or execute this script directly with one of the helper arguments:
#
#   ./commands.sh postgres   -> Starts the PostgreSQL container
#   ./commands.sh migrate    -> Runs Alembic database migrations
#   ./commands.sh backend    -> Runs the FastAPI backend server (Uvicorn)
#   ./commands.sh frontend   -> Runs the Vite frontend development server
#   ./commands.sh check      -> Checks health of running services
# ==============================================================================

# ------------------------------------------------------------------------------
# TERMINAL 1: POSTGRESQL (Port 5432)
# ------------------------------------------------------------------------------
# Option A: Run via Docker (Recommended, isolated)
# docker run -d \
#   --name researchlens-postgres \
#   -e POSTGRES_USER=postgres \
#   -e POSTGRES_PASSWORD=postgres \
#   -e POSTGRES_DB=researchlens \
#   -p 5432:5432 \
#   postgres:16-alpine
#
# Option B: Run via macOS Homebrew
# brew services start postgresql
# psql postgres -c "CREATE USER postgres WITH PASSWORD 'postgres' SUPERUSER;"
# createdb -U postgres researchlens 2>/dev/null || psql postgres -c "CREATE DATABASE researchlens OWNER postgres;"
#
# Run Alembic Database Migrations (apply schema):
# .venv/bin/alembic upgrade head
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# TERMINAL 2: FASTAPI BACKEND (Port 8000)
# ------------------------------------------------------------------------------
# From repository root:
# source .venv/bin/activate
# PYTHONPATH=backend uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
#
# Interactive API documentation will be available at:
#   Swagger UI: http://localhost:8000/docs
#   ReDoc:      http://localhost:8000/redoc
#   Health:     http://localhost:8000/health
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# TERMINAL 3: REACT + VITE FRONTEND (Port 5173)
# ------------------------------------------------------------------------------
# cd frontend
# npm run dev
#
# The web interface will be accessible at:
#   http://localhost:5173
# ------------------------------------------------------------------------------


# ==============================================================================
# SCRIPT EXECUTION HELPERS
# ==============================================================================
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

case "${1:-}" in
  postgres)
    echo "==> Starting PostgreSQL container via Docker..."
    if docker ps -a --format '{{.Names}}' | grep -q "^researchlens-postgres$"; then
      docker start researchlens-postgres
    else
      docker run -d \
        --name researchlens-postgres \
        -e POSTGRES_USER=postgres \
        -e POSTGRES_PASSWORD=postgres \
        -e POSTGRES_DB=researchlens \
        -p 5432:5432 \
        postgres:16-alpine
    fi
    echo "==> PostgreSQL is running on port 5432 (database: researchlens, user: postgres)."
    ;;

  migrate)
    echo "==> Running Alembic migrations..."
    "$REPO_ROOT/.venv/bin/alembic" upgrade head
    ;;

  backend)
    echo "==> Starting FastAPI Backend on http://localhost:8000..."
    PYTHONPATH="$REPO_ROOT/backend" "$REPO_ROOT/.venv/bin/uvicorn" app.main:app --reload --host 0.0.0.0 --port 8000
    ;;

  frontend)
    echo "==> Starting Frontend on http://localhost:5173..."
    cd "$REPO_ROOT/frontend" && npm run dev
    ;;

  check)
    echo "==> Checking backend health..."
    curl -s http://localhost:8000/health | grep -q "healthy" && echo "  [OK] Backend is healthy!" || echo "  [FAIL] Backend is not responding on :8000"

    echo "==> Checking frontend availability..."
    curl -s -I http://localhost:5173 | grep -q "200 OK" && echo "  [OK] Frontend is up!" || echo "  [INFO] Frontend not running or returned non-200"

    echo "==> Checking PostgreSQL port..."
    nc -z localhost 5432 2>/dev/null && echo "  [OK] PostgreSQL port 5432 is open!" || echo "  [FAIL] PostgreSQL port 5432 is closed"
    ;;

  *)
    echo "ResearchLens AI — Service Startup Commands"
    echo ""
    echo "Usage: ./commands.sh [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  postgres   Start PostgreSQL container using Docker"
    echo "  migrate    Run database migrations (alembic upgrade head)"
    echo "  backend    Start the FastAPI backend with uvicorn"
    echo "  frontend   Start the React/Vite development server"
    echo "  check      Verify if services are running"
    echo ""
    echo "Or run manually across 3 terminal tabs:"
    echo "  Tab 1 (DB):       docker start researchlens-postgres || docker run -d --name researchlens-postgres -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=researchlens -p 5432:5432 postgres:16-alpine"
    echo "  Tab 1 (Migrate):  .venv/bin/alembic upgrade head"
    echo "  Tab 2 (Backend):  PYTHONPATH=backend .venv/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
    echo "  Tab 3 (Frontend): cd frontend && npm run dev"
    ;;
esac
