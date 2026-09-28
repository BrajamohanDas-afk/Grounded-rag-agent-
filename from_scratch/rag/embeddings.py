"""
DAY 1 — Embeddings
==================
An embedder turns text into a vector so "similar meaning" => "nearby vectors".

We start with TF-IDF (no downloads, runs anywhere, great for understanding the
mechanics) and you upgrade to a neural model (sentence-transformers) on Day 2.
Both expose the same interface, so the rest of the code does not care.
"""
from __future__ import annotations

import pickle
from abc import ABC, abstractmethod

import numpy as np


class Embedder(ABC):
    @abstractmethod
    def fit(self, texts: list[str]) -> None: ...

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalised vectors, shape (n, dim)."""


def _normalise(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


class TfidfEmbedder(Embedder):
    """Sparse lexical vectors. Must be `fit` on the corpus vocabulary first."""

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")

    def fit(self, texts):
        self.vec.fit(texts)

    def embed(self, texts):
        return _normalise(self.vec.transform(texts).toarray().astype(np.float32))

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.vec, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.vec = pickle.load(f)


class SbertEmbedder(Embedder):
    """Dense semantic vectors. Understands synonyms ('car' ~ 'automobile')."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def fit(self, texts):
        pass  # pretrained, nothing to fit

    def embed(self, texts):
        return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

    def save(self, path):
        pass

    def load(self, path):
        pass


def get_embedder(name: str) -> Embedder:
    return {"tfidf": TfidfEmbedder, "sbert": SbertEmbedder}[name]()
