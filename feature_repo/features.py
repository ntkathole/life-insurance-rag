"""
Feature definitions for the Life Insurance RAG demo.

Covers all four maturity levels:
  Level 0: insurance_docs (vector search)
  Level 1: customer_profile (batch features)
  Level 2: claim_status (streaming features via PushSource)
  Level 3: interaction_memory + realtime_computed (on-demand features)
"""

from datetime import timedelta

import pandas as pd
from feast import Entity, FeatureView, Field, FileSource, PushSource
from feast.data_format import ParquetFormat
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Array, Float32, Float64, Int64, String, UnixTimestamp, ValueType

# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------

customer = Entity(
    name="customer_id",
    description="Policyholder ID from the policy admin system",
    value_type=ValueType.STRING,
)

document = Entity(
    name="doc_id",
    description="Document chunk identifier",
    value_type=ValueType.INT64,
)

# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

customer_profile_source = FileSource(
    file_format=ParquetFormat(),
    path="data/customer_profiles.parquet",
    timestamp_field="event_timestamp",
)

insurance_docs_source = FileSource(
    file_format=ParquetFormat(),
    path="data/insurance_docs.parquet",
    timestamp_field="event_timestamp",
)

claim_status_batch_source = FileSource(
    file_format=ParquetFormat(),
    path="data/claim_status.parquet",
    timestamp_field="event_timestamp",
)

claim_status_push_source = PushSource(
    name="claim_status_push",
    batch_source=claim_status_batch_source,
)

interaction_memory_source = FileSource(
    file_format=ParquetFormat(),
    path="data/interaction_memory.parquet",
    timestamp_field="event_timestamp",
)

# ---------------------------------------------------------------------------
# Level 0: Document embeddings for vector search
# ---------------------------------------------------------------------------

insurance_docs = FeatureView(
    name="insurance_docs",
    entities=[document],
    schema=[
        Field(
            name="embedding",
            dtype=Array(Float32),
            vector_index=True,
            vector_search_metric="cosine",
        ),
        Field(name="content", dtype=String),
        Field(name="title", dtype=String),
        Field(name="category", dtype=String),
        Field(name="plan_type", dtype=String),
    ],
    source=insurance_docs_source,
    ttl=timedelta(days=7),
)

# ---------------------------------------------------------------------------
# Level 1: Customer profile (batch, materialized nightly)
# ---------------------------------------------------------------------------

customer_profile = FeatureView(
    name="customer_profile",
    entities=[customer],
    schema=[
        Field(name="customer_name", dtype=String),
        Field(name="policy_number", dtype=String),
        Field(name="plan_type", dtype=String),
        Field(name="coverage_amount", dtype=Float64),
        Field(name="premium_frequency", dtype=String),
        Field(name="nominee_name", dtype=String),
        Field(name="age", dtype=Int64),
        Field(name="city", dtype=String),
    ],
    source=customer_profile_source,
    ttl=timedelta(days=1),
)

# ---------------------------------------------------------------------------
# Level 2: Claim status (streaming via PushSource)
# ---------------------------------------------------------------------------

claim_status = FeatureView(
    name="claim_status",
    entities=[customer],
    schema=[
        Field(name="claim_number", dtype=String),
        Field(name="claim_stage", dtype=String),
        Field(name="claim_filed_date", dtype=String),
        Field(name="last_update", dtype=String),
        Field(name="documents_pending", dtype=Int64),
        Field(name="estimated_days_remaining", dtype=Int64),
    ],
    source=claim_status_push_source,
    ttl=timedelta(days=90),
)

# ---------------------------------------------------------------------------
# Level 3a: Interaction memory (read/write via online store)
# ---------------------------------------------------------------------------

interaction_memory = FeatureView(
    name="interaction_memory",
    entities=[customer],
    schema=[
        Field(name="last_topic", dtype=String),
        Field(name="last_resolution", dtype=String),
        Field(name="visit_count", dtype=Int64),
        Field(name="open_question", dtype=String),
    ],
    source=interaction_memory_source,
    ttl=timedelta(days=30),
)

# ---------------------------------------------------------------------------
# Level 3b: On-demand computed features (calculated at request time)
# ---------------------------------------------------------------------------


@on_demand_feature_view(
    sources=[customer_profile, claim_status],
    schema=[
        Field(name="estimated_payout", dtype=Float64),
        Field(name="can_withdraw_claim", dtype=String),
        Field(name="days_since_filed", dtype=Float64),
    ],
)
def realtime_computed(inputs: pd.DataFrame) -> pd.DataFrame:
    """Compute features in real-time from batch + streaming sources."""
    df = pd.DataFrame()

    # Simplified payout computation
    df["estimated_payout"] = inputs["coverage_amount"].fillna(0)

    # Can only withdraw if claim is in early stages
    df["can_withdraw_claim"] = inputs["claim_stage"].apply(
        lambda s: "yes" if s in ("filed", "docs_received") else "no"
    )

    # Days since claim was filed
    try:
        filed = pd.to_datetime(inputs["claim_filed_date"], errors="coerce")
        df["days_since_filed"] = (pd.Timestamp.now() - filed).dt.days.fillna(0).astype(float)
    except Exception:
        df["days_since_filed"] = 0.0

    return df
