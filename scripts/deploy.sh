#!/usr/bin/env bash
# deploy.sh - Deploy the Life Insurance RAG demo to OpenShift AI.
#
# What this script does:
#   1. Creates the namespace with RHOAI labels
#   2. Deploys PostgreSQL + pgvector
#   3. Creates the FeatureStore CR (Feast operator handles the rest)
#   4. Creates a RHOAI Workbench for running notebooks
#
# Prerequisites:
#   - oc CLI logged into an OpenShift cluster with RHOAI installed
#   - Feast Operator enabled in the DataScienceCluster
#
# Usage: bash scripts/deploy.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "================================================================"
echo "  Life Insurance RAG - OpenShift AI Deployment"
echo "================================================================"

# Detect CLI
if command -v oc &> /dev/null; then
  CLI="oc"
elif command -v kubectl &> /dev/null; then
  CLI="kubectl"
else
  echo "Error: Neither 'oc' nor 'kubectl' found."
  exit 1
fi
echo "Using CLI: $CLI"
echo ""

# --- Pre-checks ---
echo "[Pre-check] Verifying Feast Operator is running ..."
if $CLI get pods -n redhat-ods-applications -l app.kubernetes.io/name=feast-operator --no-headers 2>/dev/null | grep -q Running; then
  echo "  Feast Operator: Running"
else
  echo "  WARNING: Feast Operator not found. Ensure feastoperator is Managed in DataScienceCluster."
fi

echo ""
echo "[Pre-check] Verifying FeatureStore CRD exists ..."
if $CLI get crd featurestores.feast.dev &>/dev/null; then
  echo "  FeatureStore CRD: Available"
else
  echo "  ERROR: FeatureStore CRD not found. Install the Feast Operator first."
  exit 1
fi

# --- 1. Apply all manifests ---
echo ""
echo "[1/4] Applying manifests (namespace, postgres, feast CR, workbench) ..."
$CLI apply -k "$PROJECT_ROOT/manifests/"
echo "  Done."

# --- 2. Wait for PostgreSQL ---
echo ""
echo "[2/4] Waiting for PostgreSQL to be ready ..."
$CLI -n life-insurance-rag rollout status deployment/postgres --timeout=120s
echo "  PostgreSQL is ready."

# --- 3. Wait for FeatureStore ---
echo ""
echo "[3/4] Waiting for FeatureStore CR to be ready ..."
echo "  (The Feast operator is deploying registry + online store servers ...)"
for i in $(seq 1 60); do
  STATUS=$($CLI get featurestore life-insurance-rag -n life-insurance-rag -o jsonpath='{.status.phase}' 2>/dev/null || echo "Pending")
  if [ "$STATUS" = "Ready" ]; then
    echo "  FeatureStore is ready!"
    break
  fi
  if [ "$i" -eq 60 ]; then
    echo "  WARNING: FeatureStore not ready after 5 minutes. Check: $CLI get featurestore -n life-insurance-rag -o yaml"
  fi
  sleep 5
done

# Print FeatureStore status
echo ""
echo "[Status] FeatureStore resources:"
$CLI get featurestore -n life-insurance-rag 2>/dev/null
echo ""
$CLI get pods -n life-insurance-rag 2>/dev/null
echo ""
$CLI get services -n life-insurance-rag 2>/dev/null

# --- 4. Print next steps ---
echo ""
echo "[4/4] Deployment complete!"
echo ""
echo "================================================================"
echo "  Next Steps"
echo "================================================================"
echo ""
echo "1. Load data into the feature store:"
echo "   $CLI -n life-insurance-rag port-forward svc/postgres 5432:5432 &"
echo "   export FEAST_PG_HOST=localhost"
echo "   python data/generate_sample_data.py"
echo "   python scripts/materialize.py"
echo ""
echo "2. Open the RHOAI Dashboard and navigate to the workbench"
echo "   or port-forward to access Jupyter directly:"
echo "   $CLI -n life-insurance-rag port-forward notebook/life-insurance-rag-workbench 8888:8888"
echo ""
echo "3. Connect Dify.ai to the Feast MCP server:"
FEAST_SVC=$($CLI get svc -n life-insurance-rag -l 'app.kubernetes.io/name=feast-online' -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "feast-life-insurance-rag-online")
echo "   MCP SSE endpoint: http://$FEAST_SVC.life-insurance-rag.svc.cluster.local:80/mcp/sse"
echo ""

# Print route if on OpenShift
if [ "$CLI" = "oc" ]; then
  ROUTES=$($CLI get routes -n life-insurance-rag -o jsonpath='{.items[*].spec.host}' 2>/dev/null || echo "")
  if [ -n "$ROUTES" ]; then
    echo "  Routes:"
    for r in $ROUTES; do
      echo "    https://$r"
    done
  fi
fi
echo ""
echo "================================================================"
