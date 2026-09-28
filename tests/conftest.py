"""
Test fixtures: a ScriptedLLM that fakes structured output, so the ENTIRE graph runs
offline and deterministically. Each schema gets a handler: (prompt_text) -> instance.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from grounded_rag.config import Settings
from grounded_rag.schemas import (Citation, CitedAnswer, CitedSentence, ChunkGrade, GradeBatch,
                                  RouteDecision, SearchPlan, StandaloneQuestion, SupportVerdict,
                                  SupportVerdicts)
from grounded_rag.service import RAGService

DOCS = Path(__file__).parent.parent / "data" / "docs"


class ScriptedLLM(GenericFakeChatModel):
    handlers: dict = {}
    calls: list = []

    def with_structured_output(self, schema, **kwargs):
        def run(prompt_value):
            text = prompt_value.to_string() if hasattr(prompt_value, "to_string") else str(prompt_value)
            self.calls.append((schema.__name__, text))
            h = self.handlers[schema.__name__]
            if isinstance(h, list):          # a queue of handlers: one per call
                h = h.pop(0) if len(h) > 1 else h[0]
            return h(text) if callable(h) else h
        return RunnableLambda(run)

    def schema_calls(self, name):
        return [t for n, t in self.calls if n == name]


# ---------- handler helpers ----------
def sources_in(prompt: str) -> dict[str, str]:
    """Parse '[S1] (label)\\ncontent' blocks out of the generation prompt."""
    blocks = re.split(r"\n(?=\[S\d+\] \()", "\n" + prompt.split("QUESTION:")[0])
    out = {}
    for b in blocks:
        m = re.match(r"\s*\[(S\d+)\] \([^)]*\)\n(.*)", b, flags=re.S)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def answer_with(*pairs: tuple[str, str]):
    """Handler producing a CitedAnswer; each (sentence, quote) is cited to whichever source contains it."""
    def h(prompt):
        srcs = sources_in(prompt)
        sents = []
        for sentence, quote in pairs:
            sid = next((k for k, v in srcs.items() if quote in v), "S1")
            sents.append(CitedSentence(text=sentence, citations=[Citation(source_id=sid, quote=quote)]))
        return CitedAnswer(sentences=sents)
    return h


def grade_all(relevant: bool = True):
    def h(prompt):
        ids = re.findall(r"^\[(C\d+)\]", prompt, flags=re.M)
        return GradeBatch(grades=[ChunkGrade(source_id=i, relevant=relevant) for i in ids])
    return h


def grade_containing(word: str):
    def h(prompt):
        blocks = re.split(r"\n(?=\[C\d+\])", prompt)
        grades = []
        for b in blocks:
            m = re.match(r"\s*\[(C\d+)\]", b)
            if m:
                grades.append(ChunkGrade(source_id=m.group(1), relevant=word.lower() in b.lower()))
        return GradeBatch(grades=grades)
    return h


def judge_all(supported: bool = True):
    def h(prompt):
        idx = [int(i) for i in re.findall(r"Claim (\d+):", prompt)]
        return SupportVerdicts(verdicts=[SupportVerdict(index=i, supported=supported, reason="test") for i in idx])
    return h


def default_handlers():
    return {
        "StandaloneQuestion": lambda p: StandaloneQuestion(question=p.split("Latest message:")[-1].strip()),
        "RouteDecision": RouteDecision(route="retrieve", reason="needs documents"),
        "SearchPlan": SearchPlan(queries=[]),
        "GradeBatch": grade_all(True),
        "CitedAnswer": CitedAnswer(insufficient_evidence=True, missing_information="n/a"),
        "SupportVerdicts": judge_all(True),
    }


@pytest.fixture
def settings(tmp_path):
    # Fake embeddings are random noise, so BM25 gets most of the fusion weight in tests.
    # With real embeddings the default (0.4) is a good starting point.
    return Settings(embedding_provider="fake", vector_store="memory", index_dir=tmp_path / "index",
                    chunk_size=300, chunk_overlap=50, top_k=4, bm25_weight=0.8, llm_provider="fake", llm_model="fake")


@pytest.fixture
def llm():
    return ScriptedLLM(messages=iter([AIMessage("Hi! Ask me about your documents.")] * 10),
                       handlers=default_handlers(), calls=[])


@pytest.fixture
def svc(settings, llm, tmp_path):
    docs = tmp_path / "docs"
    shutil.copytree(DOCS, docs)
    s = RAGService(settings=settings, llm=llm)
    s.ingest(docs)
    return s
