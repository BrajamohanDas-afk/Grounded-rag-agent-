"""
Retrieval = BM25 (keywords) + vector search (meaning), fused with weighted
Reciprocal Rank Fusion via LangChain's EnsembleRetriever, then optionally reranked
with a cross-encoder.

Why each piece:
  * Vector search finds paraphrases ("how long do panels last" -> "degrade after 25 years").
  * BM25 nails exact tokens: product codes, acronyms, names, numbers.
  * RRF merges rankings without needing comparable scores.
  * A cross-encoder reads (query, chunk) TOGETHER and scores relevance far more
    accurately than comparing two independent vectors — but it's slow, so it only
    re-orders the top ~20 candidates.
"""
from __future__ import annotations

import re

from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, PrivateAttr
from rank_bm25 import BM25Okapi

from .config import Settings
from .ingestion import IndexManager

_STOP = set("""a an the and or of to in on for with by is are was were be been it this that
these those as at from what which who how why when where do does did can could should would
i you he she they we my your their its not no about into than then there here""".split())


def _stem(t: str) -> str:
    """Tiny, predictable stemmer: plural/verb suffixes only ('electrolysers' -> 'electrolyser').
    Swap in nltk's PorterStemmer if your corpus needs more."""
    if t.isdigit() or len(t) <= 3:
        return t
    for suf, rep in (("ies", "y"), ("sses", "ss"), ("ing", ""), ("ed", ""), ("es", "e"), ("s", "")):
        if t.endswith(suf) and len(t) - len(suf) >= 3 and not t.endswith("ss"):
            return t[: -len(suf)] + rep
    return t


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower()) if t not in _STOP]


class BM25Retriever(BaseRetriever):
    """A custom LangChain retriever: subclass BaseRetriever and implement one method."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    docs: list[Document]
    k: int = 20
    _bm25: BM25Okapi | None = PrivateAttr(default=None)

    def model_post_init(self, __context) -> None:
        if self.docs:
            self._bm25 = BM25Okapi([tokenize(d.page_content) for d in self.docs])

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[: self.k]
        return [self.docs[i] for i in ranked if scores[i] > 0]


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, docs: list[Document], k: int) -> list[Document]:
        if not docs:
            return docs
        scores = self.model.predict([(query, d.page_content) for d in docs])
        ranked = sorted(zip(docs, scores), key=lambda x: -float(x[1]))[:k]
        out = []
        for d, sc in ranked:
            d = Document(page_content=d.page_content, metadata={**d.metadata, "rerank_score": float(sc)}, id=d.id)
            out.append(d)
        return out


class HybridSearcher:
    def __init__(self, index: IndexManager, s: Settings):
        self.index = index
        self.s = s
        self.reranker = CrossEncoderReranker(s.reranker_model) if s.use_reranker else None
        self.refresh()

    def refresh(self):
        """Call after ingestion: BM25 must be rebuilt over the new chunk set."""
        chunks = self.index.all_chunks()
        dense = self.index.vector_store.as_retriever(search_kwargs={"k": self.s.fetch_k})
        self.bm25 = BM25Retriever(docs=chunks, k=self.s.fetch_k)
        self.dense = dense
        self.ensemble = EnsembleRetriever(
            retrievers=[self.bm25, dense],
            weights=[self.s.bm25_weight, 1 - self.s.bm25_weight],
            id_key="chunk_id",
        )

    def search(self, query: str, k: int | None = None, mode: str = "hybrid") -> list[Document]:
        k = k or self.s.top_k
        if not self.index.chunks:
            return []
        retriever = {"hybrid": self.ensemble, "bm25": self.bm25, "dense": self.dense}[mode]
        docs = retriever.invoke(query)
        if self.reranker and mode == "hybrid":
            return self.reranker.rerank(query, docs[: self.s.fetch_k], k)
        return docs[:k]
