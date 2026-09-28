"""
RAGService: the one object the CLI, API, UI and evaluation all use.
Owns settings, models, index, searcher and the compiled graph.
"""
from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from .config import Settings, get_settings
from .documents import load_path
from .graph import build_graph
from .ingestion import IndexManager
from .models import build_embeddings, build_llm
from .retrieval import HybridSearcher


class RAGService:
    def __init__(self, settings: Settings | None = None, llm: BaseChatModel | None = None,
                 embeddings: Embeddings | None = None, checkpointer=None):
        self.s = settings or get_settings()
        self.embeddings = embeddings or build_embeddings(self.s)
        self.index = IndexManager(self.s, self.embeddings)
        self.searcher = HybridSearcher(self.index, self.s)
        self._llm = llm
        self._checkpointer = checkpointer
        self._graph = None

    # lazily build the LLM/graph so `ingest` and `search` work without an API key
    @property
    def llm(self) -> BaseChatModel:
        if self._llm is None:
            self._llm = build_llm(self.s)
        return self._llm

    @property
    def graph(self):
        if self._graph is None:
            self._graph = build_graph(self.llm, self.searcher, self.s, self._checkpointer)
        return self._graph

    # ---------------- ingestion ----------------
    def ingest(self, path: str | Path) -> dict:
        return self.ingest_documents(load_path(path))

    def ingest_documents(self, docs: list[Document]) -> dict:
        stats = self.index.add_documents(docs)
        self.searcher.refresh()
        return stats

    def delete_source(self, source: str) -> int:
        n = self.index.delete_source(source)
        self.searcher.refresh()
        return n

    def search(self, query: str, k: int | None = None, mode: str = "hybrid") -> list[Document]:
        return self.searcher.search(query, k, mode)

    # ---------------- asking ----------------
    def _config(self, thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}, "recursion_limit": 60}

    def ask(self, question: str, thread_id: str | None = None) -> dict:
        thread_id = thread_id or str(uuid.uuid4())
        t0 = time.perf_counter()
        state = self.graph.invoke({"question": question}, self._config(thread_id))
        return self._result(state, thread_id, time.perf_counter() - t0)

    def stream(self, question: str, thread_id: str | None = None) -> Iterator[dict]:
        """Yields {'type': 'step', 'node': ..., 'trace': [...]} per node, then {'type': 'result', ...}."""
        thread_id = thread_id or str(uuid.uuid4())
        t0 = time.perf_counter()
        for chunk in self.graph.stream({"question": question}, self._config(thread_id), stream_mode="updates"):
            for node, update in chunk.items():
                trace = update.get("trace", []) if isinstance(update, dict) else []
                yield {"type": "step", "node": node, "trace": getattr(trace, "value", trace)}
        state = self.graph.get_state(self._config(thread_id)).values
        yield {"type": "result", **self._result(state, thread_id, time.perf_counter() - t0)}

    def _result(self, state: dict, thread_id: str, seconds: float) -> dict:
        return {
            "thread_id": thread_id,
            "question": state.get("question"),
            "standalone_question": state.get("standalone_question"),
            "route": state.get("route"),
            "answer": state.get("answer", ""),
            "citations": state.get("citations", []),
            "grounding_score": state.get("grounding_score", 0.0),
            "reports": state.get("best", {}).get("reports", state.get("reports", [])),
            "queries": state.get("queries_tried", []),
            "evidence": {sid: {"label": d.metadata.get("source"), "chunk_id": d.metadata.get("chunk_id"),
                               "text": d.page_content} for sid, d in state.get("evidence", {}).items()},
            "trace": state.get("trace", []),
            "seconds": round(seconds, 2),
        }

    def history(self, thread_id: str) -> list[dict]:
        msgs = self.graph.get_state(self._config(thread_id)).values.get("messages", [])
        return [{"role": m.type, "content": m.content} for m in msgs]

    def mermaid(self) -> str:
        return self.graph.get_graph().draw_mermaid()
