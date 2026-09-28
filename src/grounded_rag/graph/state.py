"""
Graph state = the shared "memory" every node reads from and writes partial updates to.

Reducers (the second arg of Annotated) decide how an update is merged:
  * no reducer      -> the new value replaces the old one
  * add_messages    -> appends chat messages (and de-duplicates by id)   -> conversation memory
  * operator.add    -> concatenates lists; lets PARALLEL retrieve nodes all contribute
                       candidates in the same step. We reset it with `Overwrite([])`.
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class RAGState(TypedDict, total=False):
    # --- conversation (persisted across turns by the checkpointer) ---
    messages: Annotated[list[AnyMessage], add_messages]

    # --- per-turn input / understanding ---
    question: str
    standalone_question: str
    route: str

    # --- retrieval ---
    pending_queries: list[str]
    queries_tried: list[str]
    candidates: Annotated[list[Document], operator.add]    # fan-in from parallel retrieves
    evidence: dict[str, Document]                           # "S1" -> graded-relevant chunk
    retrieval_round: int

    # --- generation & verification ---
    generation_attempt: int
    draft: dict                  # CitedAnswer as dict
    reports: list[dict]          # per-sentence verification reports
    feedback: str
    best: dict                   # best attempt so far {draft, reports, score}

    # --- output ---
    answer: str
    citations: list[dict]
    grounding_score: float
    trace: Annotated[list[str], operator.add]


class RetrieveInput(TypedDict):
    """Private input of the `retrieve` node; each parallel branch gets one query via Send."""
    query: str
