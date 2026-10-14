"""
RAG Pipeline implementation for all four maturity levels.

Each level function returns:
  - answer: str (the generated response)
  - context: dict (features, docs, and metadata used)
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from feast import FeatureStore

from config import FEAST_REPO_PATH, EMBEDDING_DIM


def get_store() -> FeatureStore:
    return FeatureStore(repo_path=FEAST_REPO_PATH)


def embed_query(query: str) -> List[float]:
    """Embed a query using sentence-transformers."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(query).tolist()


def call_llm(prompt: str) -> str:
    """Call the LLM to generate a response."""
    from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

    if not LLM_API_KEY:
        return _demo_response(prompt)

    from openai import OpenAI

    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.3,
    )
    return response.choices[0].message.content


def _demo_response(prompt: str) -> str:
    """Generate a simulated response when no LLM API key is set."""
    if "claim" in prompt.lower():
        return (
            "[Demo mode] Based on the retrieved documents and customer context, "
            "I can help you with your claim. The documents indicate the typical "
            "claim process and current status information has been retrieved from "
            "the feature store."
        )
    return (
        "[Demo mode] I found relevant information in the knowledge base. "
        "Set OPENAI_API_KEY for real LLM responses."
    )


# ---------------------------------------------------------------------------
# Level 0: Basic RAG (vector search only)
# ---------------------------------------------------------------------------


def level0_basic_rag(query: str) -> Tuple[str, Dict[str, Any]]:
    """Pure vector search - no customer context."""
    store = get_store()
    embedding = embed_query(query)

    results = store.retrieve_online_documents(
        features=[
            "insurance_docs:embedding",
            "insurance_docs:content",
            "insurance_docs:title",
            "insurance_docs:category",
        ],
        query=embedding,
        top_k=3,
        distance_metric="cosine",
    )
    docs = results.to_dict()

    doc_context = "\n\n".join(
        f"[{t}]: {c}"
        for t, c in zip(docs.get("title", []), docs.get("content", []))
    )

    prompt = (
        "You are an insurance customer service assistant.\n\n"
        f"DOCUMENTS:\n{doc_context}\n\n"
        f"QUESTION: {query}\n\n"
        "Answer using only the documents above."
    )

    answer = call_llm(prompt)
    return answer, {"level": 0, "docs_retrieved": len(docs.get("title", [])), "docs": docs}


# ---------------------------------------------------------------------------
# Level 1: + Batch features (customer profile)
# ---------------------------------------------------------------------------


def level1_batch_features(
    query: str, customer_id: str
) -> Tuple[str, Dict[str, Any]]:
    """Vector search enriched with batch customer features."""
    store = get_store()

    # Fetch customer profile
    profile = store.get_online_features(
        features=[
            "customer_profile:customer_name",
            "customer_profile:plan_type",
            "customer_profile:coverage_amount",
            "customer_profile:premium_frequency",
            "customer_profile:policy_number",
        ],
        entity_rows=[{"customer_id": customer_id}],
    ).to_dict()

    plan_type = profile.get("plan_type", [None])[0]
    name = profile.get("customer_name", ["Customer"])[0]

    # Enrich query with customer context
    enriched = f"Customer has {plan_type} life insurance plan. Query: {query}"
    embedding = embed_query(enriched)

    # Search with metadata filter
    results = store.retrieve_online_documents_v2(
        features=[
            "insurance_docs:embedding",
            "insurance_docs:content",
            "insurance_docs:title",
            "insurance_docs:category",
        ],
        query=embedding,
        top_k=3,
        distance_metric="cosine",
        filters={"plan_type": plan_type} if plan_type else None,
    )
    docs = results.to_dict()

    doc_context = "\n\n".join(
        f"[{t}]: {c}"
        for t, c in zip(docs.get("title", []), docs.get("content", []))
    )

    coverage = profile.get("coverage_amount", [0])[0]
    prompt = (
        "You are a helpful life insurance assistant.\n\n"
        f"CUSTOMER: {name}\n"
        f"  Policy: {profile.get('policy_number', [''])[0]}\n"
        f"  Plan: {plan_type}, coverage Rs {coverage:,.0f}\n"
        f"  Premium: {profile.get('premium_frequency', [''])[0]}\n\n"
        f"DOCUMENTS:\n{doc_context}\n\n"
        f"QUESTION: {query}\n\n"
        "Answer personally and accurately."
    )

    answer = call_llm(prompt)
    return answer, {
        "level": 1,
        "profile": profile,
        "docs_retrieved": len(docs.get("title", [])),
        "docs": docs,
    }


# ---------------------------------------------------------------------------
# Level 2: + Streaming features (claim status) + hybrid search
# ---------------------------------------------------------------------------


def level2_streaming(
    query: str, customer_id: str
) -> Tuple[str, Dict[str, Any]]:
    """Batch + streaming features with hybrid search."""
    store = get_store()

    # Fetch batch + streaming features in one call
    features = store.get_online_features(
        features=[
            # Batch
            "customer_profile:customer_name",
            "customer_profile:plan_type",
            "customer_profile:coverage_amount",
            "customer_profile:policy_number",
            # Streaming
            "claim_status:claim_number",
            "claim_status:claim_stage",
            "claim_status:claim_filed_date",
            "claim_status:estimated_days_remaining",
            "claim_status:documents_pending",
        ],
        entity_rows=[{"customer_id": customer_id}],
    ).to_dict()

    plan_type = features.get("plan_type", [None])[0]
    claim_stage = features.get("claim_stage", [None])[0]

    enriched = f"{plan_type} plan customer, claim stage: {claim_stage}. Query: {query}"
    embedding = embed_query(enriched)

    # Hybrid search: vector + keyword
    keyword = f"claim {claim_stage}" if claim_stage else query
    results = store.retrieve_online_documents_v2(
        features=[
            "insurance_docs:embedding",
            "insurance_docs:content",
            "insurance_docs:title",
        ],
        query=embedding,
        query_string=keyword,
        top_k=3,
        distance_metric="cosine",
    )
    docs = results.to_dict()

    doc_context = "\n\n".join(
        f"[{t}]: {c}"
        for t, c in zip(docs.get("title", []), docs.get("content", []))
    )

    name = features.get("customer_name", ["Customer"])[0]
    claim_num = features.get("claim_number", ["N/A"])[0]
    days_left = features.get("estimated_days_remaining", [0])[0]

    prompt = (
        "You are a helpful life insurance assistant with real-time data.\n\n"
        f"CUSTOMER: {name}\n"
        f"  Plan: {plan_type}\n"
        f"  Claim: #{claim_num} - Stage: {claim_stage}\n"
        f"  Filed: {features.get('claim_filed_date', [''])[0]}\n"
        f"  Est. days remaining: {days_left}\n"
        f"  Docs pending: {features.get('documents_pending', [0])[0]}\n\n"
        f"DOCUMENTS:\n{doc_context}\n\n"
        f"QUESTION: {query}\n\n"
        "Answer with specific details about the customer's claim."
    )

    answer = call_llm(prompt)
    return answer, {
        "level": 2,
        "features": features,
        "docs_retrieved": len(docs.get("title", [])),
        "docs": docs,
        "search_type": "hybrid",
    }


# ---------------------------------------------------------------------------
# Level 3: Full real-time context (on-demand + memory + OpenAI search)
# ---------------------------------------------------------------------------


def level3_realtime(
    query: str, customer_id: str
) -> Tuple[str, Dict[str, Any]]:
    """Full pipeline: memory + batch + streaming + on-demand + hybrid search."""
    store = get_store()

    # 1. Recall conversation memory
    memory = store.get_online_features(
        features=[
            "interaction_memory:last_topic",
            "interaction_memory:visit_count",
            "interaction_memory:open_question",
        ],
        entity_rows=[{"customer_id": customer_id}],
    ).to_dict()

    # 2. Fetch all features (batch + streaming + on-demand)
    context = store.get_online_features(
        features=[
            "customer_profile:customer_name",
            "customer_profile:plan_type",
            "customer_profile:coverage_amount",
            "customer_profile:policy_number",
            "claim_status:claim_number",
            "claim_status:claim_stage",
            "claim_status:claim_filed_date",
            "claim_status:documents_pending",
            "realtime_computed:estimated_payout",
            "realtime_computed:can_withdraw_claim",
            "realtime_computed:days_since_filed",
        ],
        entity_rows=[{"customer_id": customer_id}],
    ).to_dict()

    # 3. Hybrid search
    plan_type = context.get("plan_type", [None])[0]
    embedding = embed_query(query)
    results = store.retrieve_online_documents_v2(
        features=[
            "insurance_docs:embedding",
            "insurance_docs:content",
            "insurance_docs:title",
        ],
        query=embedding,
        query_string=query,
        top_k=3,
        distance_metric="cosine",
        filters={"plan_type": plan_type} if plan_type else None,
    )
    docs = results.to_dict()

    doc_context = "\n\n".join(
        f"[{t}]: {c}"
        for t, c in zip(docs.get("title", []), docs.get("content", []))
    )

    # 4. Build the full prompt
    name = context.get("customer_name", ["Customer"])[0]
    visit_count = memory.get("visit_count", [0])[0] or 0
    last_topic = memory.get("last_topic", [None])[0]

    memory_section = f"Visit count: {visit_count}"
    if last_topic:
        memory_section += f", Last topic: {last_topic}"

    prompt = (
        "You are a life insurance assistant with full context.\n\n"
        f"MEMORY: {memory_section}\n\n"
        f"CUSTOMER: {name}\n"
        f"  Plan: {plan_type}, Coverage: Rs {context.get('coverage_amount', [0])[0]:,.0f}\n"
        f"  Claim: #{context.get('claim_number', ['N/A'])[0]}"
        f" - {context.get('claim_stage', ['no claim'])[0]}\n"
        f"  Est. payout: Rs {context.get('estimated_payout', [0])[0]:,.0f}\n"
        f"  Can withdraw: {context.get('can_withdraw_claim', ['N/A'])[0]}\n"
        f"  Days since filed: {context.get('days_since_filed', [0])[0]}\n\n"
        f"DOCUMENTS:\n{doc_context}\n\n"
        f"QUESTION: {query}\n\n"
        "If the customer is returning, acknowledge their previous interaction. "
        "Answer with full personalized context."
    )

    answer = call_llm(prompt)

    # 5. Save interaction memory
    try:
        store.write_to_online_store(
            feature_view_name="interaction_memory",
            df=pd.DataFrame(
                [
                    {
                        "customer_id": customer_id,
                        "last_topic": query[:100],
                        "last_resolution": "answered",
                        "visit_count": visit_count + 1,
                        "open_question": "",
                        "event_timestamp": datetime.utcnow(),
                    }
                ]
            ),
        )
    except Exception:
        pass  # Memory save is best-effort

    return answer, {
        "level": 3,
        "memory": memory,
        "features": context,
        "docs_retrieved": len(docs.get("title", [])),
        "docs": docs,
        "search_type": "hybrid",
    }
