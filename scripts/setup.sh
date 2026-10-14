#!/usr/bin/env bash
# setup.sh - One-command local setup for the Life Insurance RAG demo.
#
# Prerequisites: docker, python 3.10+
# Usage: bash scripts/setup.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== Life Insurance RAG - Local Setup ==="

# --- 1. Start PostgreSQL + pgvector ---
echo ""
echo "[1/5] Starting PostgreSQL + pgvector (Docker) ..."
if docker ps --format '{{.Names}}' | grep -q '^feast-postgres$'; then
  echo "       Container 'feast-postgres' already running."
else
  docker run -d \
    --name feast-postgres \
    -e POSTGRES_USER=feast \
    -e POSTGRES_PASSWORD=feast \
    -e POSTGRES_DB=feast \
    -p 5432:5432 \
    pgvector/pgvector:pg16

  echo "       Waiting for PostgreSQL to be ready ..."
  for i in $(seq 1 30); do
    if docker exec feast-postgres pg_isready -U feast > /dev/null 2>&1; then
      break
    fi
    sleep 1
  done
  echo "       PostgreSQL is ready."
fi

# Enable pgvector extension
docker exec feast-postgres psql -U feast -d feast -c "CREATE EXTENSION IF NOT EXISTS vector;" 2>/dev/null || true

# --- 2. Python virtual environment ---
echo ""
echo "[2/5] Setting up Python environment ..."
cd "$PROJECT_ROOT"
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q -r requirements.txt

# --- 3. Generate sample data ---
echo ""
echo "[3/5] Generating sample data ..."
python data/generate_sample_data.py

# --- 4. Apply Feast definitions + materialize ---
echo ""
echo "[4/5] Applying Feast feature definitions and materializing ..."
cd feature_repo
feast apply
cd "$PROJECT_ROOT"
python scripts/materialize.py

# --- 5. Done ---
echo ""
echo "[5/5] Setup complete!"
echo ""
echo "To start the demo:"
echo "  source .venv/bin/activate"
echo "  python app/demo_app.py"
echo ""
echo "The Gradio UI will be available at http://localhost:7860"
echo ""
echo "(Optional) To start the Feast MCP server:"
echo "  cd feature_repo && feast serve --host 0.0.0.0 --port 6566"
