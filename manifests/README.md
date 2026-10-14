# OpenShift AI Manifests

These manifests deploy the Life Insurance RAG demo on **Red Hat OpenShift AI (RHOAI)** using the **Feast Operator**.

## Components

| Manifest | What It Creates |
|----------|----------------|
| `base/namespace.yaml` | `life-insurance-rag` namespace with RHOAI dashboard labels |
| `base/feast-data-stores-secret.yaml` | Connection strings for SQL registry + PostgreSQL online store |
| `base/featurestore.yaml` | **FeatureStore CR** — the Feast operator handles all Feast deployments |
| `postgres/secret.yaml` | PostgreSQL credentials |
| `postgres/pvc.yaml` | 10Gi persistent volume for PostgreSQL data |
| `postgres/deployment.yaml` | PostgreSQL 17 with pgvector + ClusterIP Service |
| `workbench/notebook.yaml` | RHOAI Workbench (Jupyter Notebook CR) with Data Science image |

## How It Works

1. **You deploy:** PostgreSQL + pgvector, secrets, and the FeatureStore CR
2. **The Feast Operator deploys:** Registry server (gRPC + REST + MCP) and Online server (REST + MCP)
3. **The Workbench** provides a Jupyter environment to run the demo notebooks

## Deploy

```bash
# One command
oc apply -k manifests/

# Or use the deploy script
bash scripts/deploy.sh
```

## Verify

```bash
# Check FeatureStore status
oc get featurestore -n life-insurance-rag

# Check all pods
oc get pods -n life-insurance-rag

# Check services (includes operator-created ones)
oc get svc -n life-insurance-rag
```

## Customization

- **PostgreSQL credentials:** Edit `postgres/secret.yaml` and `base/feast-data-stores-secret.yaml`
- **Feast project:** Edit `base/featurestore.yaml` `.spec.feastProject`
- **Git repo:** Edit `base/featurestore.yaml` `.spec.feastProjectDir.git.url`
- **Authentication:** Replace `authz.noAuth` with `authz.oidc` or `authz.kubernetes` in the FeatureStore CR
- **Workbench image:** Change the image in `workbench/notebook.yaml` to use GPU images if needed
