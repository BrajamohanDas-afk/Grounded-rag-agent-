"""
DAY 3 — The Agent
=================
"Agentic" RAG = the LLM decides WHAT to retrieve and WHEN it has enough,
instead of doing one fixed search with the raw question.

Loop:
    1. PLAN     : LLM looks at question + evidence so far -> "search <query>" or "answer"
    2. SEARCH   : hybrid retriever; new chunks get stable ids S1, S2, ...
    3. ANSWER   : grounded JSON answer with verbatim quotes
    4. VERIFY   : mechanical checks (+ optional LLM judge)
    5. REPAIR   : if grounding is weak, send the failure report back and retry
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .grounding import (ANSWER_SYSTEM, GroundedAnswer, build_answer_prompt,
                        judge_support, verify)
from .ingest import Chunk
from .llm import LLM
from .retriever import HybridRetriever

PLANNER_SYSTEM = """You are the planning step of a research agent that answers questions
from a private document collection via a search tool.

Given the question and the evidence gathered so far, decide the next action:
- {"action": "search", "query": "<short keyword-rich query>"}  -> gather more evidence.
  Break multi-part or comparison questions into one search per part. Do not repeat a
  query you already ran.
- {"action": "answer"}  -> when the evidence covers every part of the question, OR
  further searching is unlikely to help.

Reply with ONLY the JSON object."""


@dataclass
class AgentConfig:
    max_searches: int = 4
    top_k: int = 4
    min_grounding: float = 0.99   # below this, try a repair pass
    max_repairs: int = 1
    use_judge: bool = True


@dataclass
class AgentResult:
    answer: GroundedAnswer
    trace: list[str] = field(default_factory=list)


class RAGAgent:
    def __init__(self, retriever: HybridRetriever, llm: LLM, config: AgentConfig | None = None):
        self.retriever = retriever
        self.llm = llm
        self.cfg = config or AgentConfig()

    # evidence registry: chunk_id -> "S#" keeps ids stable across searches
    def _add_evidence(self, evidence: dict[str, Chunk], chunks: list[Chunk]) -> int:
        known = {c.chunk_id for c in evidence.values()}
        added = 0
        for c in chunks:
            if c.chunk_id not in known:
                evidence[f"S{len(evidence) + 1}"] = c
                known.add(c.chunk_id)
                added += 1
        return added

    def _plan(self, question: str, evidence: dict[str, Chunk], queries: list[str]) -> dict:
        ev = "\n".join(f"[{sid}] {c.text[:200]}" for sid, c in evidence.items()) or "(none yet)"
        user = (f"QUESTION: {question}\n\nQUERIES ALREADY RUN: {queries or 'none'}\n\n"
                f"EVIDENCE SO FAR:\n{ev}")
        try:
            return self.llm.complete_json(PLANNER_SYSTEM, user, max_tokens=200)
        except ValueError:
            return {"action": "answer"}

    def run(self, question: str) -> AgentResult:
        trace: list[str] = []
        evidence: dict[str, Chunk] = {}
        queries: list[str] = []

        # always start with the user's own question: cheap and usually useful
        next_query = question
        while len(queries) < self.cfg.max_searches:
            hits = self.retriever.search(next_query, k=self.cfg.top_k)
            n_new = self._add_evidence(evidence, [h.chunk for h in hits])
            queries.append(next_query)
            trace.append(f"search({next_query!r}) -> {len(hits)} hits, {n_new} new")

            decision = self._plan(question, evidence, queries)
            trace.append(f"plan -> {decision}")
            if decision.get("action") != "search":
                break
            next_query = str(decision.get("query") or "").strip()
            if not next_query or next_query in queries:
                trace.append("planner repeated itself; stopping search")
                break

        answer = self._answer(question, evidence, trace)
        return AgentResult(answer=answer, trace=trace)

    def _answer(self, question: str, evidence: dict[str, Chunk], trace: list[str]) -> GroundedAnswer:
        feedback = ""
        answer: GroundedAnswer | None = None
        for attempt in range(self.cfg.max_repairs + 1):
            try:
                raw = self.llm.complete_json(ANSWER_SYSTEM, build_answer_prompt(question, evidence, feedback))
            except ValueError as e:
                trace.append(f"answer attempt {attempt}: bad JSON ({e})")
                continue
            candidate = verify(raw, question, evidence)
            if self.cfg.use_judge:
                candidate = judge_support(candidate, self.llm)
            trace.append(f"answer attempt {attempt}: grounding {candidate.grounding_score:.0%}")
            if answer is None or candidate.grounding_score >= answer.grounding_score:
                answer = candidate
            if answer.grounding_score >= self.cfg.min_grounding:
                break
            feedback = answer.failure_report()
            trace.append(f"repairing:\n{feedback}")

        return answer or GroundedAnswer(question, [], evidence, insufficient_evidence=True,
                                        missing="The model failed to produce a valid answer.")
