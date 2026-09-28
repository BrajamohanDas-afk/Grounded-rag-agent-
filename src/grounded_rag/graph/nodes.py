"""
Graph nodes. Each node: (state) -> partial state update (a dict).
Routers: (state) -> name of the next node (or a list of Send objects for fan-out).

The nodes are methods on one class so they share dependencies (LLM, searcher, settings)
without globals — which also makes them trivial to test with a fake LLM.
"""
from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Overwrite, Send

from ..config import Settings
from ..grounding import (failure_feedback, grounding_score, judge_reports, render,
                         verify_answer)
from ..ingestion import citation_label
from ..prompts import CHITCHAT, CONTEXTUALIZE, GENERATE, GRADE, PLAN, REWRITE, ROUTE
from ..retrieval import HybridSearcher
from ..schemas import (CitedAnswer, GradeBatch, RouteDecision, SearchPlan,
                       StandaloneQuestion)
from .state import RAGState, RetrieveInput

log = logging.getLogger(__name__)

# Errors from a MALFORMED model reply (bad JSON, schema mismatch) -> fall back gracefully.
# Anything else (auth, network, rate limit after retries) is a real failure and must surface.
# (OutputParserException and pydantic.ValidationError are both ValueError subclasses.)
RECOVERABLE = (ValueError, KeyError, TypeError, IndexError)


def _dedupe(queries: list[str], exclude: list[str] = ()) -> list[str]:
    seen = {q.strip().lower() for q in exclude}
    out = []
    for q in queries:
        k = q.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(q.strip())
    return out


class RAGNodes:
    def __init__(self, llm: BaseChatModel, searcher: HybridSearcher, s: Settings):
        self.llm = llm
        self.searcher = searcher
        self.s = s

    # ------------------------------------------------------------------ helpers
    def _structured(self, prompt, schema, inputs):
        return (prompt | self.llm.with_structured_output(schema)).invoke(inputs)

    def _corpus_hint(self) -> str:
        srcs = list(self.searcher.index.sources())
        if not srcs:
            return "The document collection is currently empty."
        shown = ", ".join(srcs[:15]) + (" ..." if len(srcs) > 15 else "")
        return f"Documents in the collection: {shown}."

    def _question(self, state: RAGState) -> str:
        return state.get("standalone_question") or state["question"]

    # ------------------------------------------------------------------ 1. understand
    def contextualize(self, state: RAGState) -> dict:
        """Rewrite follow-ups ('what about its cost?') into standalone questions using memory.
        Also resets all per-turn fields, because the checkpointer keeps the previous turn's state."""
        question = state["question"].strip()
        history = state.get("messages", [])[-self.s.history_turns:]
        standalone = question
        if history:
            try:
                standalone = self._structured(CONTEXTUALIZE, StandaloneQuestion,
                                              {"history": history, "question": question}).question
            except RECOVERABLE as e:
                log.warning("contextualize failed: %s", e)
        return {
            "messages": [HumanMessage(question)],
            "standalone_question": standalone,
            "pending_queries": [], "queries_tried": [], "candidates": Overwrite([]),
            "evidence": {}, "retrieval_round": 0, "generation_attempt": 0,
            "draft": {}, "reports": [], "feedback": "", "best": {},
            "answer": "", "citations": [], "grounding_score": 0.0,
            "trace": Overwrite([f"contextualize: {standalone!r}" if standalone != question
                                else "contextualize: question is standalone"]),
        }

    def route(self, state: RAGState) -> dict:
        try:
            d: RouteDecision = self._structured(ROUTE, RouteDecision, {
                "question": self._question(state), "corpus_hint": self._corpus_hint()})
            route, why = d.route, d.reason
        except RECOVERABLE as e:
            route, why = "retrieve", f"router error ({type(e).__name__}), defaulting to retrieve"
        return {"route": route, "trace": [f"route: {route} ({why})"]}

    def chitchat(self, state: RAGState) -> dict:
        msg = (CHITCHAT | self.llm).invoke({
            "history": state.get("messages", [])[-self.s.history_turns:-1],
            "question": state["question"], "corpus_hint": self._corpus_hint()})
        text = msg.content if isinstance(msg.content, str) else str(msg.content)
        return {"answer": text, "citations": [], "grounding_score": 1.0,
                "messages": [AIMessage(text)], "trace": ["chitchat: answered directly"]}

    # ------------------------------------------------------------------ 2. retrieve
    def plan(self, state: RAGState) -> dict:
        q = self._question(state)
        try:
            planned = self._structured(PLAN, SearchPlan, {"question": q, "max_queries": self.s.max_sub_queries}).queries
        except RECOVERABLE as e:
            log.warning("plan failed, searching with the question only: %s", e)
            planned = []
        queries = _dedupe([q] + planned[: self.s.max_sub_queries])
        return {"pending_queries": queries, "queries_tried": queries,
                "trace": [f"plan: {len(queries)} queries -> {queries}"]}

    def retrieve(self, inp: RetrieveInput) -> dict:
        """Runs once PER QUERY, in parallel (fan-out via Send)."""
        docs = self.searcher.search(inp["query"])
        return {"candidates": docs, "trace": [f"retrieve({inp['query']!r}): {len(docs)} chunks"]}

    def grade(self, state: RAGState) -> dict:
        """Fan-in: merge all parallel results, drop duplicates, let the LLM filter irrelevant
        chunks (the 'corrective' in Corrective-RAG), then give survivors stable ids S1, S2..."""
        evidence = dict(state.get("evidence", {}))
        known = {d.metadata["chunk_id"] for d in evidence.values()}
        fresh: dict[str, Document] = {}
        for d in state.get("candidates", []):
            cid = d.metadata["chunk_id"]
            if cid not in known and cid not in fresh:
                fresh[cid] = d
        new_docs = list(fresh.values())
        rnd = state.get("retrieval_round", 0) + 1
        if not new_docs:
            return {"candidates": Overwrite([]), "retrieval_round": rnd,
                    "trace": [f"grade (round {rnd}): no new chunks"]}

        temp = {f"C{i + 1}": d for i, d in enumerate(new_docs)}
        listing = "\n\n".join(f"[{cid}] ({citation_label(d.metadata)})\n{d.page_content[:1200]}"
                              for cid, d in temp.items())
        try:
            grades: GradeBatch = self._structured(GRADE, GradeBatch,
                                                  {"question": self._question(state), "chunks": listing})
            keep = {g.source_id.strip("[] ") for g in grades.grades if g.relevant}
        except RECOVERABLE as e:
            keep = set(temp)
            log.warning("grader failed, keeping all: %s", e)

        for cid, d in temp.items():
            if cid in keep:
                evidence[f"S{len(evidence) + 1}"] = d
        return {"evidence": evidence, "candidates": Overwrite([]), "retrieval_round": rnd,
                "trace": [f"grade (round {rnd}): {len(keep & set(temp))}/{len(temp)} new chunks relevant; "
                          f"{len(evidence)} evidence chunks total"]}

    def rewrite(self, state: RAGState) -> dict:
        found = "\n".join(f"- {d.page_content[:200]}" for d in state.get("evidence", {}).values()) or "(nothing)"
        tried = state.get("queries_tried", [])
        try:
            new = self._structured(REWRITE, SearchPlan, {
                "question": self._question(state), "tried": tried, "found": found,
                "max_queries": self.s.max_sub_queries}).queries
        except RECOVERABLE:
            new = []
        new = _dedupe(new, exclude=tried)[: self.s.max_sub_queries]
        update = {"pending_queries": new, "queries_tried": tried + new, "feedback": "",
                  "trace": [f"rewrite: {new or 'no new queries'}"]}
        if not new:  # nothing new to try -> spend the whole retrieval budget so we can't loop
            update["retrieval_round"] = self.s.max_retrieval_rounds
        return update

    # ------------------------------------------------------------------ 3. answer & verify
    def generate(self, state: RAGState) -> dict:
        evidence = state.get("evidence", {})
        sources = "\n\n".join(f"[{sid}] ({citation_label(d.metadata)})\n{d.page_content}"
                              for sid, d in evidence.items())
        fb = state.get("feedback", "")
        feedback = (f"\n\nYOUR PREVIOUS ANSWER FAILED VERIFICATION. Fix exactly these problems "
                    f"(copy quotes verbatim, or drop the claim):\n{fb}") if fb else ""
        attempt = state.get("generation_attempt", 0) + 1
        try:
            draft = self._structured(GENERATE, CitedAnswer, {
                "sources": sources, "question": self._question(state), "feedback": feedback})
        except RECOVERABLE as e:
            log.warning("generation failed: %s", e)
            draft = CitedAnswer(insufficient_evidence=True, missing_information="")
        return {"draft": draft.model_dump(), "generation_attempt": attempt,
                "trace": [f"generate (attempt {attempt}): {len(draft.sentences)} sentences, "
                          f"insufficient={draft.insufficient_evidence}"]}

    def verify(self, state: RAGState) -> dict:
        draft = CitedAnswer.model_validate(state["draft"])
        reports = verify_answer(draft, state.get("evidence", {}), self.s.quote_match_threshold)
        if self.s.use_judge:
            reports = judge_reports(reports, self.llm)
        score = grounding_score(reports)
        best = state.get("best") or {}
        if not best or score >= best.get("score", -1):
            best = {"draft": state["draft"], "reports": reports, "score": score}
        n_ok = sum(r["supported"] for r in reports)
        return {"reports": reports, "grounding_score": score, "best": best,
                "feedback": failure_feedback(reports) if score < self.s.min_grounding else "",
                "trace": [f"verify: {n_ok}/{len(reports)} sentences grounded ({score:.0%})"]}

    def finalize(self, state: RAGState) -> dict:
        best = state.get("best") or {"draft": state.get("draft", {}), "reports": state.get("reports", []), "score": 0.0}
        draft = CitedAnswer.model_validate(best["draft"] or {})
        text, citations = render(best["reports"], state.get("evidence", {}),
                                 draft.insufficient_evidence, draft.missing_information)
        return {"answer": text, "citations": citations, "grounding_score": best["score"],
                "messages": [AIMessage(text)], "trace": [f"finalize: {len(citations)} sources cited"]}

    def abstain(self, state: RAGState) -> dict:
        text = ("I couldn't find information about this in the documents, so I won't guess. "
                f"Searched for: {', '.join(state.get('queries_tried', []))}.")
        return {"answer": text, "citations": [], "grounding_score": 0.0,
                "messages": [AIMessage(text)], "trace": ["abstain: no relevant evidence"]}

    # ------------------------------------------------------------------ routers (edges)
    def after_route(self, state: RAGState) -> str:
        return "chitchat" if state.get("route") == "chitchat" else "plan"

    def fan_out(self, state: RAGState):
        """Conditional edge returning Send objects = run `retrieve` once per query IN PARALLEL."""
        qs = state.get("pending_queries", [])
        if not qs:
            return "generate" if state.get("evidence") else "abstain"
        return [Send("retrieve", {"query": q}) for q in qs]

    def after_grade(self, state: RAGState) -> str:
        n = len(state.get("evidence", {}))
        if n >= self.s.min_relevant_chunks:
            return "generate"
        if state.get("retrieval_round", 0) < self.s.max_retrieval_rounds:
            return "rewrite"
        return "generate" if n else "abstain"

    def after_verify(self, state: RAGState) -> str:
        draft = state.get("draft", {})
        # The model says evidence is missing -> go search again (if budget allows)
        if draft.get("insufficient_evidence") and state.get("retrieval_round", 0) < self.s.max_retrieval_rounds:
            return "rewrite"
        if state.get("grounding_score", 0) >= self.s.min_grounding:
            return "finalize"
        if state.get("generation_attempt", 0) < self.s.max_generation_attempts:
            return "generate"      # repair loop, with feedback in state
        return "finalize"          # give up repairing; unsupported sentences get dropped
