"""
DAY 1 — Vector store (dense search)
DAY 2 — BM25 keyword search + Hybrid fusion (RRF)
=================================================
Why hybrid? Dense search is good at meaning ("how long do panels last" ->
"degrades ... after 25 years"); BM25 is good at exact tokens (model numbers,
names, "LFP"). Reciprocal Rank Fusion combines both rankings without having
to calibrate their very different score scales.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from .embeddings import Embedder, get_embedder
from .ingest import Chunk


@dataclass
class Hit:
    chunk: Chunk
    score: float
    method: str  # "dense" | "bm25" | "hybrid"


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


# --------------------------------------------------------------------------- #
# Dense vector store (a real project would use FAISS / Chroma / pgvector;
# a numpy matrix + dot product is exactly what they do under the hood)
# --------------------------------------------------------------------------- #
class VectorStore:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self.matrix: np.ndarray | None = None

    def build(self, chunks: list[Chunk]):
        self.chunks = chunks
        texts = [c.text for c in chunks]
        self.embedder.fit(texts)
        self.matrix = self.embedder.embed(texts)

    def search(self, query: str, k: int = 5) -> list[Hit]:
        q = self.embedder.embed([query])[0]
        scores = self.matrix @ q          # cosine similarity (vectors are normalised)
        top = np.argsort(-scores)[:k]
        return [Hit(self.chunks[i], float(scores[i]), "dense") for i in top if scores[i] > 0]


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    def search(self, query: str, k: int = 5) -> list[Hit]:
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(-scores)[:k]
        return [Hit(self.chunks[i], float(scores[i]), "bm25") for i in top if scores[i] > 0]


def reciprocal_rank_fusion(result_lists: list[list[Hit]], k: int = 60) -> list[Hit]:
    """score(doc) = sum over lists of 1 / (k + rank). k=60 is the standard constant."""
    fused: dict[str, float] = {}
    by_id: dict[str, Chunk] = {}
    for hits in result_lists:
        for rank, h in enumerate(hits):
            fused[h.chunk.chunk_id] = fused.get(h.chunk.chunk_id, 0.0) + 1.0 / (k + rank + 1)
            by_id[h.chunk.chunk_id] = h.chunk
    ordered = sorted(fused.items(), key=lambda kv: -kv[1])
    return [Hit(by_id[cid], s, "hybrid") for cid, s in ordered]


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], embedder_name: str = "tfidf"):
        self.chunks = chunks
        self.embedder_name = embedder_name
        self.dense = VectorStore(get_embedder(embedder_name))
        self.dense.build(chunks)
        self.bm25 = BM25Index(chunks)

    def search(self, query: str, k: int = 5, mode: str = "hybrid") -> list[Hit]:
        if mode == "dense":
            return self.dense.search(query, k)
        if mode == "bm25":
            return self.bm25.search(query, k)
        # fetch more candidates from each, then fuse
        return reciprocal_rank_fusion(
            [self.dense.search(query, k * 3), self.bm25.search(query, k * 3)]
        )[:k]

    # ---- persistence: save chunks so you don't re-ingest every run ----
    def save(self, index_dir: str | Path):
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        (index_dir / "chunks.json").write_text(
            json.dumps([c.to_dict() for c in self.chunks], indent=1), encoding="utf-8")
        (index_dir / "meta.json").write_text(json.dumps({"embedder": self.embedder_name}))

    @classmethod
    def load(cls, index_dir: str | Path) -> "HybridRetriever":
        index_dir = Path(index_dir)
        chunks = [Chunk(**d) for d in json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))]
        meta = json.loads((index_dir / "meta.json").read_text())
        return cls(chunks, meta["embedder"])  # re-embeds; fine for small/medium corpora
