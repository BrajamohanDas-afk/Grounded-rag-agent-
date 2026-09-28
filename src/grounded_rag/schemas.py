"""
Pydantic schemas for structured output.

`llm.with_structured_output(Schema)` turns a schema into a tool/JSON-schema definition,
and the provider is forced to return data matching it. The Field descriptions are sent
to the model — they ARE part of the prompt, so write them carefully.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RouteDecision(BaseModel):
    route: Literal["retrieve", "chitchat"] = Field(
        description="'retrieve' for any question that needs facts from the documents; "
                    "'chitchat' only for greetings, thanks, or questions about the assistant itself.")
    reason: str = Field(description="One short sentence explaining the choice.")


class StandaloneQuestion(BaseModel):
    question: str = Field(description="The user's latest question rewritten to be fully "
                                      "understandable without the chat history.")


class SearchPlan(BaseModel):
    queries: list[str] = Field(
        description="1-4 short, keyword-rich search queries. One per distinct sub-question "
                    "(e.g. one per item being compared). No duplicates.")


class ChunkGrade(BaseModel):
    source_id: str = Field(description="The id of the chunk, e.g. 'S3'.")
    relevant: bool = Field(description="True if the chunk contains information useful for "
                                       "answering ANY part of the question.")


class GradeBatch(BaseModel):
    grades: list[ChunkGrade]


class Citation(BaseModel):
    source_id: str = Field(description="Id of the source that supports the sentence, e.g. 'S2'.")
    quote: str = Field(description="An EXACT verbatim span (5-40 words) copied character-for-character "
                                   "from that source which directly supports the sentence.")


class CitedSentence(BaseModel):
    text: str = Field(description="One sentence of the answer making a single factual claim.")
    citations: list[Citation] = Field(description="At least one citation supporting this sentence.")


class CitedAnswer(BaseModel):
    sentences: list[CitedSentence] = Field(
        default_factory=list, description="The answer, split into cited sentences.")
    insufficient_evidence: bool = Field(
        default=False, description="True if the sources do not contain enough information "
                                   "to fully answer the question.")
    missing_information: str = Field(
        default="", description="If evidence is insufficient or partial, what is missing.")


class SupportVerdict(BaseModel):
    index: int = Field(description="The claim number.")
    supported: bool = Field(description="True only if the quotes alone clearly entail the claim.")
    reason: str = Field(description="Short justification.")


class SupportVerdicts(BaseModel):
    verdicts: list[SupportVerdict]


class CorrectnessGrade(BaseModel):
    """Used by the evaluation harness (LLM-as-judge against a reference answer)."""
    verdict: Literal["correct", "partially_correct", "incorrect"]
    reason: str
