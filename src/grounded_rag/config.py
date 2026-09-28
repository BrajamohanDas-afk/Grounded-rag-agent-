"""
Central configuration (pydantic-settings).

Every knob is read from environment variables prefixed with GRAG_ (or a .env file),
so you can run experiments without touching code:

    GRAG_TOP_K=8 GRAG_USE_JUDGE=false grag ask "..."
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GRAG_", env_file=".env", extra="ignore")

    # --- models ---
    llm_provider: str = "anthropic"
    llm_model: str = "claude-haiku-4-5-20251001"
    llm_temperature: float = 0.0
    embedding_provider: Literal["huggingface", "openai", "ollama", "fake"] = "huggingface"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- storage ---
    vector_store: Literal["chroma", "memory"] = "chroma"
    index_dir: Path = Path("./index")
    collection: str = "grounded_rag"

    # --- ingestion ---
    chunk_size: int = 800
    chunk_overlap: int = 120

    # --- retrieval ---
    top_k: int = 5                 # final chunks per query
    fetch_k: int = 20              # candidates per retriever before fusion / rerank
    bm25_weight: float = 0.4       # weight of keyword search in the fusion (dense = 1 - this)
    use_reranker: bool = False

    # --- agent loop ---
    max_sub_queries: int = 4
    max_retrieval_rounds: int = 2      # 1 = no corrective re-search
    min_relevant_chunks: int = 1       # below this -> rewrite queries & search again
    max_generation_attempts: int = 2   # 1 = no repair loop
    min_grounding: float = 0.99        # fraction of sentences that must verify
    use_judge: bool = True             # LLM entailment check on top of mechanical checks
    quote_match_threshold: float = 90.0
    history_turns: int = 6             # messages of chat history used for contextualising

    @property
    def chunks_path(self) -> Path:
        return self.index_dir / "chunks.jsonl"


@lru_cache
def get_settings() -> Settings:
    return Settings()
