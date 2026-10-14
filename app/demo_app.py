"""
Gradio demo application for the Life Insurance RAG maturity journey.

Usage:
    python app/demo_app.py

    # With real LLM:
    OPENAI_API_KEY=sk-... python app/demo_app.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import gradio as gr

from rag_pipeline import (
    level0_basic_rag,
    level1_batch_features,
    level2_streaming,
    level3_realtime,
)
from config import APP_HOST, APP_PORT

CUSTOMERS = {
    "POL-1001": "Priya Sharma (Term plan, 50L coverage, active claim)",
    "POL-1002": "Amit Patel (Savings plan, 20L coverage)",
    "POL-1003": "Kavita Reddy (Pension plan, 30L coverage, claim under review)",
    "POL-1004": "Rohan Desai (Term plan, 1Cr coverage)",
    "POL-1005": "Sunita Gupta (Savings plan, 15L coverage, claim approved)",
}

SAMPLE_QUERIES = [
    "What is happening with my claim?",
    "How do I file a death claim?",
    "What are the payment options for my premium?",
    "Can I surrender my policy?",
    "What riders are available for my plan?",
    "What are the tax benefits of my policy?",
]


def run_query(level: str, customer_id: str, query: str):
    """Execute the RAG pipeline at the selected level."""
    if not query.strip():
        return "Please enter a question.", "{}"

    level_num = int(level.split(":")[0].replace("Level ", ""))

    try:
        if level_num == 0:
            answer, ctx = level0_basic_rag(query)
        elif level_num == 1:
            answer, ctx = level1_batch_features(query, customer_id)
        elif level_num == 2:
            answer, ctx = level2_streaming(query, customer_id)
        else:
            answer, ctx = level3_realtime(query, customer_id)
    except Exception as e:
        answer = f"Error: {e}"
        ctx = {"error": str(e)}

    # Format context for display
    ctx_display = []
    if "profile" in ctx:
        ctx_display.append("--- Customer Profile ---")
        for k, v in ctx["profile"].items():
            if k != "customer_id":
                ctx_display.append(f"  {k}: {v[0] if isinstance(v, list) else v}")

    if "features" in ctx:
        ctx_display.append("\n--- Features (batch + streaming + on-demand) ---")
        for k, v in ctx["features"].items():
            if k != "customer_id":
                ctx_display.append(f"  {k}: {v[0] if isinstance(v, list) else v}")

    if "memory" in ctx:
        ctx_display.append("\n--- Conversation Memory ---")
        for k, v in ctx["memory"].items():
            if k != "customer_id":
                ctx_display.append(f"  {k}: {v[0] if isinstance(v, list) else v}")

    if "docs" in ctx:
        ctx_display.append(f"\n--- Retrieved Documents ({ctx.get('docs_retrieved', 0)}) ---")
        titles = ctx["docs"].get("title", [])
        categories = ctx["docs"].get("category", [])
        for i, (t, c) in enumerate(zip(titles, categories)):
            ctx_display.append(f"  [{i+1}] {t} ({c})")

    ctx_display.append(f"\n--- Metadata ---")
    ctx_display.append(f"  Level: {ctx.get('level', level_num)}")
    ctx_display.append(f"  Search type: {ctx.get('search_type', 'vector')}")

    return answer, "\n".join(ctx_display)


with gr.Blocks(
    title="Life Insurance RAG - Feast Feature Store Demo",
) as demo:
    gr.Markdown(
        "# Life Insurance RAG Maturity Demo\n"
        "See how Feast feature store progressively improves RAG quality from "
        "generic vector search to fully personalized, real-time answers."
    )

    with gr.Row():
        with gr.Column(scale=1):
            level_dropdown = gr.Dropdown(
                choices=[
                    "Level 0: Basic RAG (vector search only)",
                    "Level 1: + Batch Features (customer profile)",
                    "Level 2: + Streaming (live claim status + hybrid search)",
                    "Level 3: + Real-Time (on-demand + memory + OpenAI API)",
                ],
                value="Level 0: Basic RAG (vector search only)",
                label="RAG Maturity Level",
            )
            customer_dropdown = gr.Dropdown(
                choices=list(CUSTOMERS.keys()),
                value="POL-1001",
                label="Customer",
                info="Select a policyholder (used in Levels 1-3)",
            )
            gr.Markdown(
                "**Customers:**\n"
                + "\n".join(f"- `{k}`: {v}" for k, v in CUSTOMERS.items())
            )

        with gr.Column(scale=2):
            query_input = gr.Textbox(
                label="Question",
                placeholder="What is happening with my claim?",
                lines=2,
            )
            gr.Examples(
                examples=[[q] for q in SAMPLE_QUERIES],
                inputs=[query_input],
            )
            submit_btn = gr.Button("Ask", variant="primary")

    with gr.Row():
        with gr.Column(scale=2):
            answer_output = gr.Textbox(label="Answer", lines=10, interactive=False)
        with gr.Column(scale=1):
            context_output = gr.Textbox(
                label="Context (features + docs retrieved)",
                lines=10,
                interactive=False,
            )

    submit_btn.click(
        fn=run_query,
        inputs=[level_dropdown, customer_dropdown, query_input],
        outputs=[answer_output, context_output],
    )

if __name__ == "__main__":
    demo.launch(server_name=APP_HOST, server_port=APP_PORT, share=False, theme=gr.themes.Soft())
