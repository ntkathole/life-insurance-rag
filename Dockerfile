FROM python:3.12-slim

WORKDIR /app

# Install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gradio

# Copy app code
COPY feature_repo/ feature_repo/
COPY app/ app/
COPY data/ data/
COPY scripts/ scripts/

# The feature_store.yaml will be overwritten at runtime via env vars
# Generate it from environment at container start
COPY <<'ENTRYPOINT' /app/entrypoint.sh
#!/bin/bash
set -e

PG_HOST="${FEAST_PG_HOST:-postgres.life-insurance-rag.svc.cluster.local}"
PG_PORT="${FEAST_PG_PORT:-5432}"
PG_USER="${POSTGRES_USER:-feast}"
PG_PASS="${POSTGRES_PASSWORD:-feast}"
PG_DB="${POSTGRES_DB:-feast}"

cat > /app/feature_repo/feature_store.yaml << EOF
project: life_insurance_rag
provider: local

registry:
  registry_type: sql
  path: postgresql+psycopg://${PG_USER}:${PG_PASS}@${PG_HOST}:${PG_PORT}/${PG_DB}
  cache_ttl_seconds: 60

online_store:
  type: postgres
  host: ${PG_HOST}
  port: ${PG_PORT}
  database: ${PG_DB}
  db_schema: public
  user: ${PG_USER}
  password: ${PG_PASS}
  sslmode: disable
  vector_enabled: true
  enable_openai_compatible_store: true

offline_store:
  type: file

embedding_model:
  provider: sentence_transformers
  model: all-MiniLM-L6-v2

auth:
  type: no_auth

entity_key_serialization_version: 3
EOF

echo "Generated feature_store.yaml pointing to ${PG_HOST}:${PG_PORT}"

# Apply schema updates (creates openai-compatible tables if needed)
echo "Running feast apply to update schema..."
cd /app/feature_repo && python -c "
import sys
sys.path.insert(0, '.')
from feast import FeatureStore
from features import (
    insurance_docs, customer_profile, claim_status,
    interaction_memory, realtime_computed,
    customer, document,
    level0_basic_rag_service, level1_personalised_rag_service,
    level2_streaming_rag_service, level3_full_context_rag_service,
)
store = FeatureStore('.')
store.apply([
    customer, document,
    insurance_docs, customer_profile, claim_status,
    interaction_memory, realtime_computed,
    level0_basic_rag_service, level1_personalised_rag_service,
    level2_streaming_rag_service, level3_full_context_rag_service,
])
print('feast apply completed')
" 2>&1 || echo "Warning: feast apply failed, continuing anyway"
cd /app

exec python app/demo_app.py
ENTRYPOINT

RUN chmod +x /app/entrypoint.sh

# OpenShift runs as random UID — set writable cache dirs for HuggingFace/sentence-transformers
ENV HF_HOME=/tmp/hf_home \
    TRANSFORMERS_CACHE=/tmp/hf_home/transformers \
    SENTENCE_TRANSFORMERS_HOME=/tmp/hf_home/sentence_transformers \
    XDG_CACHE_HOME=/tmp/.cache

RUN chmod -R g+rwX /app

EXPOSE 7860

CMD ["/app/entrypoint.sh"]
