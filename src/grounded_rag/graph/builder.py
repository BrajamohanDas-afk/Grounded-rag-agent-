"""
Wiring the graph.

    START -> contextualize -> route --chitchat--> chitchat -> END
                                   \\--retrieve--> plan ==Send==> retrieve (xN, parallel) -> grade
    grade --enough--> generate -> verify --grounded--> finalize -> END
          \\--none---> rewrite ==Send==> retrieve ...         |--repair--> generate
          \\--give up-> abstain -> END                        \\--insufficient--> rewrite
"""
from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from ..config import Settings
from ..retrieval import HybridSearcher
from .nodes import RAGNodes
from .state import RAGState, RetrieveInput


def build_graph(llm: BaseChatModel, searcher: HybridSearcher, s: Settings,
                checkpointer: BaseCheckpointSaver | None = None):
    n = RAGNodes(llm, searcher, s)
    llm_retry = RetryPolicy(max_attempts=3, initial_interval=1.0)  # transient API errors

    g = StateGraph(RAGState)
    g.add_node("contextualize", n.contextualize, retry_policy=llm_retry)
    g.add_node("route", n.route, retry_policy=llm_retry)
    g.add_node("chitchat", n.chitchat, retry_policy=llm_retry)
    g.add_node("plan", n.plan, retry_policy=llm_retry)
    g.add_node("retrieve", n.retrieve, input_schema=RetrieveInput)
    g.add_node("grade", n.grade, retry_policy=llm_retry)
    g.add_node("rewrite", n.rewrite, retry_policy=llm_retry)
    g.add_node("generate", n.generate, retry_policy=llm_retry)
    g.add_node("verify", n.verify)
    g.add_node("finalize", n.finalize)
    g.add_node("abstain", n.abstain)

    g.add_edge(START, "contextualize")
    g.add_edge("contextualize", "route")
    g.add_conditional_edges("route", n.after_route, ["plan", "chitchat"])
    g.add_edge("chitchat", END)
    g.add_conditional_edges("plan", n.fan_out, ["retrieve", "generate", "abstain"])
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", n.after_grade, ["generate", "rewrite", "abstain"])
    g.add_conditional_edges("rewrite", n.fan_out, ["retrieve", "generate", "abstain"])
    g.add_edge("generate", "verify")
    g.add_conditional_edges("verify", n.after_verify, ["finalize", "generate", "rewrite"])
    g.add_edge("finalize", END)
    g.add_edge("abstain", END)

    return g.compile(checkpointer=checkpointer if checkpointer is not None else InMemorySaver())
