"""
Model factories. LangChain's `init_chat_model` gives one interface over every provider,
so changing from Claude to GPT to a local Llama is a config change.
"""
from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from .config import Settings


def build_llm(s: Settings) -> BaseChatModel:
    from langchain.chat_models import init_chat_model
    return init_chat_model(
        s.llm_model, model_provider=s.llm_provider, temperature=s.llm_temperature,
        max_retries=3,          # provider SDK retries on 429 / 5xx
    )


def build_embeddings(s: Settings) -> Embeddings:
    p = s.embedding_provider
    if p == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=s.embedding_model,
                                     encode_kwargs={"normalize_embeddings": True})
    if p == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=s.embedding_model)
    if p == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=s.embedding_model)
    if p == "fake":
        # deterministic random vectors: only for tests, semantically meaningless
        from langchain_core.embeddings import DeterministicFakeEmbedding
        return DeterministicFakeEmbedding(size=256)
    raise ValueError(f"unknown embedding provider {p}")
