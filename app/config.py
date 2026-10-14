"""Configuration for the Life Insurance RAG demo."""

import os

# Feast
FEAST_REPO_PATH = os.environ.get(
    "FEAST_REPO_PATH",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "feature_repo"),
)
FEAST_SERVER_URL = os.environ.get("FEAST_SERVER_URL", "http://localhost:6566")

# LLM
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai")  # openai, ollama
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", None)
LLM_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Embedding
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM = 384

# Application
APP_HOST = os.environ.get("APP_HOST", "0.0.0.0")
APP_PORT = int(os.environ.get("APP_PORT", "7860"))
