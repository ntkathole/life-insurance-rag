"""
Feature definitions for the Life Insurance RAG demo.

Covers all four maturity levels:
  Level 0: insurance_docs (vector search)
  Level 1: customer_profile (batch features)
  Level 2: claim_status (streaming features via PushSource)
  Level 3: interaction_memory + realtime_computed (on-demand features)

Feature Services group features by RAG maturity level for cleaner serving.
"""

from datetime import timedelta

import pandas as pd
from feast import Entity, FeatureService, FeatureView, Field, FileSource, PushSource
from feast.data_format import ParquetFormat
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Array, Float32, Float64, Int64, String, ValueType

# ============================================================================
# Common metadata
# ============================================================================

_OWNER = "rag-platform-team@example.com"
_PROJECT_TAGS = {
    "project": "life-insurance-rag",
    "team": "ai-platform",
    "environment": "demo",
    "domain": "insurance",
}

# ============================================================================
# Entities
# ============================================================================

customer = Entity(
    name="customer_id",
    description="Unique policyholder identifier issued by the policy administration system.",
    value_type=ValueType.STRING,
    owner=_OWNER,
    tags={**_PROJECT_TAGS, "pii": "true"},
)

document = Entity(
    name="doc_id",
    description="Document chunk identifier for the insurance knowledge base.",
    value_type=ValueType.INT64,
    owner=_OWNER,
    tags={**_PROJECT_TAGS, "pii": "false"},
)

# ============================================================================
# Data Sources
# ============================================================================

customer_profile_source = FileSource(
    name="customer_profile_source",
    description="Nightly batch export of policyholder profiles from the policy admin system.",
    file_format=ParquetFormat(),
    path="data/customer_profiles.parquet",
    timestamp_field="event_timestamp",
    owner=_OWNER,
    tags={**_PROJECT_TAGS, "source_type": "batch", "refresh": "daily"},
)

insurance_docs_source = FileSource(
    name="insurance_docs_source",
    description="Insurance knowledge base documents with pre-computed embeddings (all-MiniLM-L6-v2, 384d).",
    file_format=ParquetFormat(),
    path="data/insurance_docs.parquet",
    timestamp_field="event_timestamp",
    owner=_OWNER,
    tags={**_PROJECT_TAGS, "source_type": "batch", "refresh": "weekly"},
)

claim_status_batch_source = FileSource(
    name="claim_status_batch_source",
    description="Historical claim status snapshots used as fallback for the streaming source.",
    file_format=ParquetFormat(),
    path="data/claim_status.parquet",
    timestamp_field="event_timestamp",
    owner=_OWNER,
    tags={**_PROJECT_TAGS, "source_type": "batch", "refresh": "hourly"},
)

claim_status_push_source = PushSource(
    name="claim_status_push",
    description="Real-time claim status updates pushed from the claims processing pipeline.",
    batch_source=claim_status_batch_source,
    tags={**_PROJECT_TAGS, "source_type": "streaming", "transport": "push_api"},
)

interaction_memory_source = FileSource(
    name="interaction_memory_source",
    description="Conversation memory scaffold — seeded empty, populated at runtime via online store writes.",
    file_format=ParquetFormat(),
    path="data/interaction_memory.parquet",
    timestamp_field="event_timestamp",
    owner=_OWNER,
    tags={**_PROJECT_TAGS, "source_type": "online_write", "refresh": "realtime"},
)

# ============================================================================
# Level 0: Document embeddings for vector search
# ============================================================================

insurance_docs = FeatureView(
    name="insurance_docs",
    entities=[document],
    description="Insurance knowledge base with vector embeddings for semantic search. "
    "Contains policy documents, claim guides, and FAQ content across all plan types.",
    owner=_OWNER,
    tags={
        **_PROJECT_TAGS,
        "rag_level": "0",
        "feature_type": "embedding",
        "embedding_model": "all-MiniLM-L6-v2",
        "embedding_dim": "384",
        "search_type": "vector",
    },
    schema=[
        Field(
            name="embedding",
            dtype=Array(Float32),
            description="384-dimensional sentence embedding (all-MiniLM-L6-v2) for cosine similarity search.",
            vector_index=True,
            vector_search_metric="cosine",
            tags={"role": "vector_index", "model": "all-MiniLM-L6-v2"},
        ),
        Field(
            name="content",
            dtype=String,
            description="Full text content of the document chunk.",
            tags={"role": "document_body"},
        ),
        Field(
            name="title",
            dtype=String,
            description="Document title or section heading.",
            tags={"role": "metadata"},
        ),
        Field(
            name="category",
            dtype=String,
            description="Document category: claims, benefits, premiums, riders, general.",
            tags={"role": "metadata", "filterable": "true"},
        ),
        Field(
            name="plan_type",
            dtype=String,
            description="Insurance plan type the document applies to: term, savings, pension, all.",
            tags={"role": "metadata", "filterable": "true"},
        ),
    ],
    source=insurance_docs_source,
    ttl=timedelta(days=7),
)

# ============================================================================
# Level 1: Customer profile (batch, materialized nightly)
# ============================================================================

customer_profile = FeatureView(
    name="customer_profile",
    entities=[customer],
    description="Policyholder profile features materialized nightly from the policy admin system. "
    "Used to personalize RAG responses with customer-specific context.",
    owner=_OWNER,
    tags={
        **_PROJECT_TAGS,
        "rag_level": "1",
        "feature_type": "batch",
        "refresh_cadence": "daily",
        "contains_pii": "true",
    },
    schema=[
        Field(name="customer_name", dtype=String, description="Full name of the policyholder.", tags={"pii": "true"}),
        Field(name="policy_number", dtype=String, description="Policy number in format XX-NNNN.", tags={"pii": "false"}),
        Field(name="plan_type", dtype=String, description="Insurance plan: term, savings, or pension.", tags={"pii": "false"}),
        Field(name="coverage_amount", dtype=Float64, description="Sum assured in INR.", tags={"pii": "false", "unit": "INR"}),
        Field(name="premium_frequency", dtype=String, description="Premium payment schedule: monthly, quarterly, yearly.", tags={"pii": "false"}),
        Field(name="nominee_name", dtype=String, description="Designated beneficiary name.", tags={"pii": "true"}),
        Field(name="age", dtype=Int64, description="Policyholder age at last birthday.", tags={"pii": "true"}),
        Field(name="city", dtype=String, description="City of residence.", tags={"pii": "true"}),
    ],
    source=customer_profile_source,
    ttl=timedelta(days=1),
)

# ============================================================================
# Level 2: Claim status (streaming via PushSource)
# ============================================================================

claim_status = FeatureView(
    name="claim_status",
    entities=[customer],
    description="Real-time claim tracking features pushed from the claims processing pipeline. "
    "Provides live claim stage, pending documents, and estimated resolution time.",
    owner=_OWNER,
    tags={
        **_PROJECT_TAGS,
        "rag_level": "2",
        "feature_type": "streaming",
        "latency_sla": "< 5 min",
        "push_source": "claim_status_push",
    },
    schema=[
        Field(name="claim_number", dtype=String, description="Unique claim reference number.", tags={"pii": "false"}),
        Field(name="claim_stage", dtype=String, description="Current processing stage: filed, docs_received, under_review, approved, settled.", tags={"pii": "false"}),
        Field(name="claim_filed_date", dtype=String, description="ISO date when the claim was originally filed.", tags={"pii": "false"}),
        Field(name="last_update", dtype=String, description="ISO timestamp of the most recent status change.", tags={"pii": "false"}),
        Field(name="documents_pending", dtype=Int64, description="Number of documents still required from the claimant.", tags={"pii": "false"}),
        Field(name="estimated_days_remaining", dtype=Int64, description="Estimated business days until claim resolution.", tags={"pii": "false", "unit": "days"}),
    ],
    source=claim_status_push_source,
    ttl=timedelta(days=90),
)

# ============================================================================
# Level 3a: Interaction memory (read/write via online store)
# ============================================================================

interaction_memory = FeatureView(
    name="interaction_memory",
    entities=[customer],
    description="Conversational memory for multi-turn RAG interactions. "
    "Written at the end of each conversation turn, read at the start of the next.",
    owner=_OWNER,
    tags={
        **_PROJECT_TAGS,
        "rag_level": "3",
        "feature_type": "online_readwrite",
        "write_pattern": "per_interaction",
        "contains_pii": "true",
    },
    schema=[
        Field(name="last_topic", dtype=String, description="Topic of the most recent customer question.", tags={"pii": "false"}),
        Field(name="last_resolution", dtype=String, description="Outcome of the last interaction: answered, escalated, pending.", tags={"pii": "false"}),
        Field(name="visit_count", dtype=Int64, description="Total number of interactions by this customer.", tags={"pii": "false"}),
        Field(name="open_question", dtype=String, description="Unresolved question carried over from a previous session.", tags={"pii": "false"}),
    ],
    source=interaction_memory_source,
    ttl=timedelta(days=30),
)

# ============================================================================
# Level 3b: On-demand computed features (calculated at request time)
# ============================================================================


@on_demand_feature_view(
    sources=[customer_profile, claim_status],
    schema=[
        Field(name="estimated_payout", dtype=Float64),
        Field(name="can_withdraw_claim", dtype=String),
        Field(name="days_since_filed", dtype=Float64),
    ],
    description="Real-time computed features derived from batch (customer_profile) and streaming (claim_status) sources. "
    "Calculated on every request — no materialization required.",
    owner=_OWNER,
    tags={
        **_PROJECT_TAGS,
        "rag_level": "3",
        "feature_type": "on_demand",
        "compute": "realtime",
    },
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


# ============================================================================
# Feature Services — group features by RAG maturity level
# ============================================================================

level0_basic_rag_service = FeatureService(
    name="level0_basic_rag",
    description="Level 0: Basic RAG with vector-only document search. "
    "No customer context — purely retrieves relevant documents via cosine similarity.",
    features=[insurance_docs],
    owner=_OWNER,
    tags={
        **_PROJECT_TAGS,
        "rag_level": "0",
        "serving_pattern": "vector_search",
    },
)

level1_personalised_rag_service = FeatureService(
    name="level1_personalised_rag",
    description="Level 1: Personalised RAG enriched with batch customer profile features. "
    "Queries are enhanced with plan type, coverage, and customer name for targeted responses.",
    features=[insurance_docs, customer_profile],
    owner=_OWNER,
    tags={
        **_PROJECT_TAGS,
        "rag_level": "1",
        "serving_pattern": "vector_search + batch_lookup",
    },
)

level2_streaming_rag_service = FeatureService(
    name="level2_streaming_rag",
    description="Level 2: Streaming RAG with real-time claim status and hybrid search. "
    "Combines batch profile, live claim data, and vector + keyword search.",
    features=[insurance_docs, customer_profile, claim_status],
    owner=_OWNER,
    tags={
        **_PROJECT_TAGS,
        "rag_level": "2",
        "serving_pattern": "hybrid_search + batch + streaming",
    },
)

level3_full_context_rag_service = FeatureService(
    name="level3_full_context_rag",
    description="Level 3: Full-context RAG with on-demand features and conversation memory. "
    "The complete pipeline: vector search, batch profile, streaming claims, "
    "computed features (payout, withdrawability), and multi-turn memory.",
    features=[
        insurance_docs,
        customer_profile,
        claim_status,
        interaction_memory,
        realtime_computed,
    ],
    owner=_OWNER,
    tags={
        **_PROJECT_TAGS,
        "rag_level": "3",
        "serving_pattern": "hybrid_search + batch + streaming + on_demand + memory",
    },
)
