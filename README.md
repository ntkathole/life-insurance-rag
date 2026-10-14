# Life Insurance RAG Demo with Feast on OpenShift AI

A progressive demo showing how **Feast Feature Store** improves RAG applications for life insurance, deployed natively on **Red Hat OpenShift AI (RHOAI)**.

## RAG Maturity Levels

| Level | Feast Capability | What Changes |
|-------|-----------------|-------------|
| **0** | `retrieve_online_documents` (pgvector cosine) | Pure vector search – generic answers |
| **1** | + `get_online_features` (batch profile) + metadata filter | Personalized by plan type |
| **2** | + `store.push()` (PushSource streaming) + hybrid search | Real-time claim status awareness |
| **3** | + `@on_demand_feature_view` + `write_to_online_store` (memory) + `openai_search()` | Fully contextual, computed at request time |

## Architecture on OpenShift AI

```
Dify.ai / LLM App  <-- MCP SSE --> Feast Online Server (operator-managed)
                                          |
                               +----------+----------+
                               |          |          |
                          Batch FV   Stream FV   OnDemand FV
                          (profile)  (claims)   (computed)
                               |          |          |
                               +----+-----+----------+
                                    |
                         PostgreSQL + pgvector (online store)

OpenShift AI Cluster
+-----------------------------------------------------------+
| Namespace: life-insurance-rag                             |
|                                                           |
| [FeatureStore CR] --> Feast Operator (RHOAI-managed)      |
|    +-> feast-*-registry  (gRPC + REST + MCP)              |
|    +-> feast-*-online    (REST + MCP)                     |
|                                                           |
| [PostgreSQL + pgvector]  (Deployment + PVC)               |
| [RHOAI Workbench]        (Notebook CR, Data Science img)  |
+-----------------------------------------------------------+
```

## Repository Structure

```
life-insurance-rag/
├── manifests/                          # OpenShift AI deployment
│   ├── kustomization.yaml              # Kustomize entrypoint
│   ├── base/
│   │   ├── namespace.yaml              # Namespace with RHOAI labels
│   │   ├── feast-data-stores-secret.yaml  # DB connection strings for operator
│   │   └── featurestore.yaml           # FeatureStore CR (Feast Operator)
│   ├── postgres/
│   │   ├── secret.yaml                 # PostgreSQL credentials
│   │   ├── pvc.yaml                    # 10Gi persistent storage
│   │   └── deployment.yaml             # pgvector/pg17 + Service
│   └── workbench/
│       └── notebook.yaml               # RHOAI Workbench (Notebook CR)
│
├── feature_repo/
│   ├── feature_store.yaml              # Feast config (local dev / workbench)
│   └── features.py                     # Feature definitions (all 4 levels)
│
├── data/
│   └── generate_sample_data.py         # Generates sample data (Parquet)
│
├── app/
│   ├── config.py                       # Env-var configuration
│   ├── rag_pipeline.py                 # 4-level RAG pipeline
│   └── demo_app.py                     # Gradio demo UI
│
├── notebooks/                          # Step-by-step notebooks
│   ├── 01_setup_and_materialize.ipynb
│   ├── 02_level0_basic_rag.ipynb
│   ├── 03_level1_batch_features.ipynb
│   ├── 04_level2_streaming_hybrid.ipynb
│   └── 05_level3_realtime_full.ipynb
│
├── scripts/
│   ├── deploy.sh                       # Deploy to OpenShift AI
│   ├── setup.sh                        # Local dev setup (Docker)
│   └── materialize.py                  # Push features to online store
│
├── Dockerfile
├── requirements.txt
└── README.md
```

## Prerequisites

- OpenShift cluster with **RHOAI** (Red Hat OpenShift AI) installed
- **Feast Operator** enabled (`feastoperator: managementState: Managed` in DataScienceCluster)
- `oc` CLI logged in as cluster admin or project admin

## Quick Deploy to OpenShift AI

```bash
# 1. Clone the repo
git clone https://github.com/ntkathole/life-insurance-rag.git
cd life-insurance-rag

# 2. Deploy everything (namespace, postgres, FeatureStore CR, workbench)
bash scripts/deploy.sh

# 3. Port-forward PostgreSQL and load data
oc -n life-insurance-rag port-forward svc/postgres 5432:5432 &
export FEAST_PG_HOST=localhost
pip install 'feast[postgres]' sentence-transformers numpy pandas
python data/generate_sample_data.py
python scripts/materialize.py

# 4. Open the RHOAI Dashboard -> Data Science Projects -> life-insurance-rag
#    Start the workbench and run the notebooks in order.
```

## What Gets Deployed

| Resource | Kind | Description |
|----------|------|-------------|
| `life-insurance-rag` | Namespace | Project namespace with RHOAI labels |
| `postgres` | Deployment + Service | PostgreSQL 17 with pgvector extension |
| `postgres-data` | PVC | 10Gi persistent storage for PostgreSQL |
| `postgres-secret` | Secret | Database credentials |
| `feast-data-stores` | Secret | Connection strings for the Feast operator |
| `life-insurance-rag` | FeatureStore CR | Tells the Feast operator to deploy registry + online servers |
| `life-insurance-rag-workbench` | Notebook CR | RHOAI Workbench with Data Science image |

The **Feast Operator** automatically creates:
- `feast-life-insurance-rag-registry` — Registry server (gRPC + REST + MCP)
- `feast-life-insurance-rag-online` — Online feature server (REST + MCP)
- ConfigMaps, ServiceAccounts, Services, and Routes as needed

## Feature Definitions

Defined in `feature_repo/features.py`:

| Feature View | Type | Entity | Purpose |
|-------------|------|--------|---------|
| `insurance_docs` | FeatureView (vector) | `doc_id` | 15 insurance documents with 384-dim embeddings |
| `customer_profile` | FeatureView (batch) | `customer_id` | 5 policyholders with plan details |
| `claim_status` | FeatureView (stream) | `customer_id` | Active claims via PushSource |
| `interaction_memory` | FeatureView (r/w) | `customer_id` | Conversation state for follow-up awareness |
| `realtime_computed` | OnDemandFeatureView | `customer_id` | `estimated_payout`, `can_withdraw_claim`, `days_since_filed` |

## MCP Integration with Dify.ai

The Feast operator deploys an MCP (Model Context Protocol) server alongside the online store. To connect from Dify.ai:

1. Get the MCP endpoint:
   ```bash
   oc get svc -n life-insurance-rag -l app.kubernetes.io/name=feast-online
   # MCP SSE URL: http://<service>.<namespace>.svc.cluster.local:80/mcp/sse
   ```

2. In Dify.ai workflow builder, add an **MCP Tool** node with the SSE URL

3. Available MCP tools:
   - `get_online_features` — Fetch customer profile, claim status, computed features
   - `retrieve_online_documents` — Vector search over insurance docs
   - `push` — Push real-time claim updates
   - `write_to_online_store` — Save conversation memory

## Local Development

For local development without an OpenShift cluster:

```bash
bash scripts/setup.sh        # Starts PostgreSQL in Docker + creates venv
python app/demo_app.py        # Launches Gradio UI at http://localhost:7860
```

## Shakti Cloud Integration

For sovereign GPU infrastructure, configure the LLM endpoint:

```bash
export LLM_BASE_URL=https://your-shakti-endpoint/v1
export OPENAI_API_KEY=your-key
export LLM_MODEL=your-model-name
```
