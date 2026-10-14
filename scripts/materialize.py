"""
Materialize all feature views into the PostgreSQL online store.

Usage:
    python scripts/materialize.py
"""

import os
import sys

import pandas as pd

REPO_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "feature_repo")
DATA_DIR = os.path.join(REPO_DIR, "data")


def main():
    sys.path.insert(0, REPO_DIR)
    from feast import FeatureStore

    store = FeatureStore(repo_path=REPO_DIR)

    print("=" * 60)
    print("  Materializing features to PostgreSQL online store")
    print("=" * 60)

    # Level 0: Insurance documents (vector search)
    print("\n[1/4] Materializing insurance documents...")
    docs_df = pd.read_parquet(os.path.join(DATA_DIR, "insurance_docs.parquet"))
    store.write_to_online_store(feature_view_name="insurance_docs", df=docs_df)
    print(f"  Wrote {len(docs_df)} documents with embeddings")

    # Level 1: Customer profiles (batch features)
    print("\n[2/4] Materializing customer profiles...")
    profiles_df = pd.read_parquet(os.path.join(DATA_DIR, "customer_profiles.parquet"))
    store.write_to_online_store(feature_view_name="customer_profile", df=profiles_df)
    print(f"  Wrote {len(profiles_df)} customer profiles")

    # Level 2: Claim status (streaming - initial load)
    print("\n[3/4] Materializing claim status (initial load)...")
    claims_df = pd.read_parquet(os.path.join(DATA_DIR, "claim_status.parquet"))
    store.push(push_source_name="claim_status_push", df=claims_df)
    print(f"  Pushed {len(claims_df)} claim events")

    # Level 3: Interaction memory (empty scaffold)
    print("\n[4/4] Interaction memory ready (populated at runtime)")

    print("\n" + "=" * 60)
    print("  Materialization complete!")
    print("  Start server: cd feature_repo && feast serve --port 6566")
    print("=" * 60)


if __name__ == "__main__":
    main()
