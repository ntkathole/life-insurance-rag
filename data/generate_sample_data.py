"""
Generate sample data for the Life Insurance RAG demo.

Creates Parquet files for:
  - Customer profiles (5 policyholders)
  - Insurance documents (15 chunked documents with embeddings)
  - Claim status events (active claims)
  - Interaction memory scaffold (empty)

Usage:
    python data/generate_sample_data.py
"""

import os
import sys

import numpy as np
import pandas as pd

REPO_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "feature_repo")
DATA_DIR = os.path.join(REPO_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

EMBEDDING_DIM = 384
NOW = pd.Timestamp.now()


def generate_customer_profiles() -> pd.DataFrame:
    """5 sample policyholders with different plan types."""
    customers = [
        {
            "customer_id": "POL-1001",
            "customer_name": "Priya Sharma",
            "policy_number": "TL-4521",
            "plan_type": "term",
            "coverage_amount": 5000000.0,
            "premium_frequency": "yearly",
            "nominee_name": "Rahul Sharma",
            "age": 35,
            "city": "Mumbai",
        },
        {
            "customer_id": "POL-1002",
            "customer_name": "Amit Patel",
            "policy_number": "SV-7832",
            "plan_type": "savings",
            "coverage_amount": 2000000.0,
            "premium_frequency": "monthly",
            "nominee_name": "Neeta Patel",
            "age": 42,
            "city": "Ahmedabad",
        },
        {
            "customer_id": "POL-1003",
            "customer_name": "Kavita Reddy",
            "policy_number": "PN-1193",
            "plan_type": "pension",
            "coverage_amount": 3000000.0,
            "premium_frequency": "yearly",
            "nominee_name": "Suresh Reddy",
            "age": 55,
            "city": "Hyderabad",
        },
        {
            "customer_id": "POL-1004",
            "customer_name": "Rohan Desai",
            "policy_number": "TL-6204",
            "plan_type": "term",
            "coverage_amount": 10000000.0,
            "premium_frequency": "monthly",
            "nominee_name": "Sneha Desai",
            "age": 30,
            "city": "Bangalore",
        },
        {
            "customer_id": "POL-1005",
            "customer_name": "Sunita Gupta",
            "policy_number": "SV-3350",
            "plan_type": "savings",
            "coverage_amount": 1500000.0,
            "premium_frequency": "quarterly",
            "nominee_name": "Vinod Gupta",
            "age": 48,
            "city": "Delhi",
        },
    ]
    df = pd.DataFrame(customers)
    df["event_timestamp"] = NOW
    return df


def generate_insurance_docs() -> pd.DataFrame:
    """15 chunked insurance documents across categories and plan types."""
    articles = [
        # Claims - general
        {
            "doc_id": 1,
            "title": "How to file a death claim",
            "content": (
                "To file a death claim, the nominee must submit: (1) Claim form, "
                "(2) Original policy document, (3) Death certificate from municipal "
                "authority, (4) Hospital records if applicable, (5) FIR copy if "
                "accidental death. Submit all documents at the nearest branch or "
                "upload via the policyholder portal. Claims are acknowledged within "
                "48 hours of submission."
            ),
            "category": "claims",
            "plan_type": "all",
        },
        {
            "doc_id": 2,
            "title": "Claim processing stages",
            "content": (
                "Every claim goes through five stages: (1) Filed - initial submission "
                "received, (2) Documents Received - all required papers verified, "
                "(3) Under Review - medical and financial assessment in progress, "
                "(4) Approved - claim amount sanctioned, (5) Paid - funds transferred "
                "to nominee bank account. You can track status in the policyholder "
                "portal or by calling 1800-123-4567."
            ),
            "category": "claims",
            "plan_type": "all",
        },
        {
            "doc_id": 3,
            "title": "Claim settlement timelines",
            "content": (
                "As per IRDAI guidelines, life insurance claims must be settled "
                "within 30 days of receiving all required documents. If investigation "
                "is needed, the insurer has 90 days from the date of claim submission. "
                "Interest is payable on delayed settlements at a rate determined by "
                "IRDAI each year."
            ),
            "category": "claims",
            "plan_type": "all",
        },
        # Payments
        {
            "doc_id": 4,
            "title": "Premium payment options",
            "content": (
                "Premiums can be paid via: (1) Auto-debit from bank account (ECS/NACH), "
                "(2) UPI payment on the app, (3) Net banking, (4) Cheque at branch, "
                "(5) Cash at branch (up to Rs 49,999). Payment frequency depends on "
                "your policy: monthly, quarterly, half-yearly, or yearly. You can "
                "change payment frequency by submitting a request form."
            ),
            "category": "payments",
            "plan_type": "all",
        },
        {
            "doc_id": 5,
            "title": "Grace period and lapse rules",
            "content": (
                "If you miss a premium, you have a grace period: 30 days for yearly/"
                "half-yearly/quarterly premiums, 15 days for monthly premiums. During "
                "the grace period, your policy remains active. After the grace period, "
                "the policy lapses. A lapsed policy can be revived within 5 years by "
                "paying all overdue premiums with interest and submitting a health "
                "declaration."
            ),
            "category": "payments",
            "plan_type": "all",
        },
        # Term plan specific
        {
            "doc_id": 6,
            "title": "Term plan - what is covered",
            "content": (
                "A term life insurance plan provides a death benefit to the nominee "
                "if the policyholder passes away during the policy term. Key features: "
                "(1) Pure protection - no maturity benefit, (2) High coverage at low "
                "premiums, (3) Optional riders: accidental death, critical illness, "
                "waiver of premium. Term plans do not have surrender value or loan "
                "facility."
            ),
            "category": "benefits",
            "plan_type": "term",
        },
        {
            "doc_id": 7,
            "title": "Term plan riders and add-ons",
            "content": (
                "Available riders for term plans: (1) Accidental Death Benefit - "
                "additional payout if death is accidental, (2) Critical Illness - "
                "lump sum on diagnosis of specified illnesses (cancer, heart attack, "
                "stroke, etc.), (3) Waiver of Premium - premiums waived if permanently "
                "disabled, (4) Terminal Illness - advance payout if diagnosed with "
                "terminal illness. Riders can be added at policy inception or renewal."
            ),
            "category": "benefits",
            "plan_type": "term",
        },
        # Savings plan specific
        {
            "doc_id": 8,
            "title": "Savings plan - maturity benefits",
            "content": (
                "Savings life insurance plans provide both protection and returns. "
                "On maturity, you receive: (1) Sum assured + accumulated bonuses for "
                "traditional plans, (2) Fund value based on NAV for unit-linked plans. "
                "Bonuses are declared annually by the insurer. The maturity benefit "
                "is tax-free under Section 10(10D) if annual premium is less than "
                "Rs 5 lakh."
            ),
            "category": "benefits",
            "plan_type": "savings",
        },
        {
            "doc_id": 9,
            "title": "Savings plan - surrender and loan",
            "content": (
                "If you need funds before maturity: (1) Surrender - cancel the policy "
                "and receive the surrender value (available after 3 years of premium "
                "payment). Surrender value = guaranteed surrender value + bonus "
                "component. (2) Policy loan - borrow up to 80% of surrender value at "
                "competitive interest rates without cancelling the policy. Loan "
                "interest is typically 9-10% per annum."
            ),
            "category": "benefits",
            "plan_type": "savings",
        },
        # Pension plan specific
        {
            "doc_id": 10,
            "title": "Pension plan - annuity options",
            "content": (
                "At retirement (vesting age), you can: (1) Take up to 1/3 of the "
                "corpus as lump sum (tax-free), (2) Use remaining 2/3 to purchase "
                "an annuity for regular pension. Annuity options include: life annuity, "
                "life annuity with return of purchase price, joint life annuity, and "
                "annuity certain for 5/10/15/20 years. Once chosen, annuity option "
                "cannot be changed."
            ),
            "category": "benefits",
            "plan_type": "pension",
        },
        {
            "doc_id": 11,
            "title": "Pension plan - tax benefits",
            "content": (
                "Pension plan tax benefits: (1) Premiums deductible under Section 80CCC "
                "up to Rs 1.5 lakh per year, (2) Lump sum withdrawal of up to 1/3 "
                "of corpus is tax-free, (3) Annuity income is taxable as per your "
                "income tax slab. NPS subscribers get additional deduction of Rs 50,000 "
                "under Section 80CCD(1B)."
            ),
            "category": "benefits",
            "plan_type": "pension",
        },
        # General / account
        {
            "doc_id": 12,
            "title": "Updating nominee details",
            "content": (
                "To update nominee details: (1) Log into the policyholder portal, "
                "(2) Go to My Policies > Policy Details > Nominee, (3) Fill in new "
                "nominee name, relationship, and share percentage, (4) Upload identity "
                "proof of the new nominee, (5) Submit. Changes take effect within "
                "3 working days. You can also visit any branch with the nominee "
                "change form and identity documents."
            ),
            "category": "account",
            "plan_type": "all",
        },
        {
            "doc_id": 13,
            "title": "Policy document download",
            "content": (
                "Your policy document is available digitally: (1) Log into the "
                "policyholder portal, (2) Go to My Policies, (3) Click Download "
                "next to your policy. The digital document is legally valid as per "
                "IRDAI regulations. If you need a physical copy, request one through "
                "the portal or at a branch. Delivery takes 7-10 working days."
            ),
            "category": "account",
            "plan_type": "all",
        },
        {
            "doc_id": 14,
            "title": "Free-look period and cancellation",
            "content": (
                "You have a 30-day free-look period from the date you receive your "
                "policy document (15 days for offline purchase). During this period, "
                "you can cancel the policy and get a full refund minus stamp duty "
                "and medical exam charges. To cancel, submit a written request with "
                "the original policy document at any branch."
            ),
            "category": "account",
            "plan_type": "all",
        },
        {
            "doc_id": 15,
            "title": "Contacting customer support",
            "content": (
                "Reach us via: (1) Toll-free: 1800-123-4567 (8 AM to 8 PM), "
                "(2) Email: support@insurer.example.com, (3) WhatsApp: +91-98765-43210, "
                "(4) Policyholder portal chat, (5) Visit any branch. For claim-related "
                "queries, have your policy number and claim reference number ready. "
                "Enterprise policyholders (coverage > Rs 1 crore) get a dedicated "
                "relationship manager."
            ),
            "category": "support",
            "plan_type": "all",
        },
    ]

    np.random.seed(42)
    df = pd.DataFrame(articles)

    # Generate random embeddings (in production, use sentence-transformers)
    df["embedding"] = [
        np.random.randn(EMBEDDING_DIM).astype(np.float32).tolist()
        for _ in range(len(df))
    ]
    df["event_timestamp"] = NOW
    return df


def generate_claim_status() -> pd.DataFrame:
    """Active claims for some policyholders."""
    claims = [
        {
            "customer_id": "POL-1001",
            "claim_number": "CLM-9087",
            "claim_stage": "docs_received",
            "claim_filed_date": "2024-10-10",
            "last_update": "2024-10-12",
            "documents_pending": 0,
            "estimated_days_remaining": 5,
        },
        {
            "customer_id": "POL-1003",
            "claim_number": "CLM-8234",
            "claim_stage": "under_review",
            "claim_filed_date": "2024-10-01",
            "last_update": "2024-10-11",
            "documents_pending": 0,
            "estimated_days_remaining": 10,
        },
        {
            "customer_id": "POL-1005",
            "claim_number": "CLM-7601",
            "claim_stage": "approved",
            "claim_filed_date": "2024-09-15",
            "last_update": "2024-10-13",
            "documents_pending": 0,
            "estimated_days_remaining": 0,
        },
    ]
    df = pd.DataFrame(claims)
    df["event_timestamp"] = NOW
    return df


def generate_interaction_memory_scaffold() -> pd.DataFrame:
    """Empty scaffold for interaction memory."""
    return pd.DataFrame(
        {
            "customer_id": pd.Series(dtype="str"),
            "last_topic": pd.Series(dtype="str"),
            "last_resolution": pd.Series(dtype="str"),
            "visit_count": pd.Series(dtype="int64"),
            "open_question": pd.Series(dtype="str"),
            "event_timestamp": pd.Series(dtype="datetime64[ns]"),
        }
    )


def main():
    print("=" * 60)
    print("  Life Insurance RAG - Sample Data Generator")
    print("=" * 60)

    print("\n[1/4] Generating customer profiles...")
    customers_df = generate_customer_profiles()
    path = os.path.join(DATA_DIR, "customer_profiles.parquet")
    customers_df.to_parquet(path, index=False)
    print(f"  Saved {len(customers_df)} profiles to {path}")

    print("\n[2/4] Generating insurance documents with embeddings...")
    docs_df = generate_insurance_docs()
    path = os.path.join(DATA_DIR, "insurance_docs.parquet")
    docs_df.to_parquet(path, index=False)
    print(f"  Saved {len(docs_df)} documents to {path}")

    print("\n[3/4] Generating claim status data...")
    claims_df = generate_claim_status()
    path = os.path.join(DATA_DIR, "claim_status.parquet")
    claims_df.to_parquet(path, index=False)
    print(f"  Saved {len(claims_df)} claims to {path}")

    print("\n[4/4] Generating interaction memory scaffold...")
    memory_df = generate_interaction_memory_scaffold()
    path = os.path.join(DATA_DIR, "interaction_memory.parquet")
    memory_df.to_parquet(path, index=False)
    print(f"  Saved empty scaffold to {path}")

    print("\n" + "=" * 60)
    print("  Data generation complete!")
    print("  Next: cd feature_repo && feast apply")
    print("=" * 60)


if __name__ == "__main__":
    main()
