"""
DAY 2 — Citation Grounding (the heart of this project)
=====================================================
Plain RAG: "here are some chunks, answer the question" -> the model may still
invent facts and sprinkle plausible-looking [1] [2] markers on them.

Grounded RAG adds three layers of defence:
  1. STRUCTURED OUTPUT  : every sentence must carry citations, and every citation
                          must include an EXACT QUOTE copied from the source.
  2. MECHANICAL CHECKS  : we verify in code that (a) the cited source exists,
                          (b) the quote really appears in it (fuzzy match), and
                          (c) numbers in the claim appear in the quotes.
  3. SEMANTIC CHECK     : (optional) an LLM judge asks "does this quote actually
                          SUPPORT this claim?" (a quote can exist but be irrelevant).
Sentences that fail are removed or flagged — the user only sees supported claims.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from .ingest import Chunk
from .llm import LLM

# --------------------------------------------------------------------------- #
# 1. Prompting
# --------------------------------------------------------------------------- #
ANSWER_SYSTEM = """You are a careful research assistant. Answer ONLY from the numbered sources provided.

Rules:
- Split your answer into short sentences. Each sentence makes ONE factual claim.
- Every sentence MUST cite at least one source id (e.g. "S2") and include a "quote":
  an EXACT, verbatim span copied character-for-character from that source (5-40 words)
  that directly supports the sentence. Do not paraphrase inside "quote".
- Never use outside knowledge. If the sources do not contain the answer, set
  "insufficient_evidence" to true and explain what is missing in "missing".
- If sources only partially answer, answer the supported part and describe the gap in "missing".

Reply with ONLY this JSON object:
{
  "sentences": [
    {"text": "<one claim>", "citations": [{"source": "S1", "quote": "<verbatim span>"}]}
  ],
  "insufficient_evidence": false,
  "missing": ""
}"""


def format_sources(sources: dict[str, Chunk]) -> str:
    return "\n\n".join(
        f"[{sid}] (from {c.citation_label()})\n{c.text}" for sid, c in sources.items()
    )


def build_answer_prompt(question: str, sources: dict[str, Chunk], feedback: str = "") -> str:
    prompt = f"SOURCES:\n{format_sources(sources)}\n\nQUESTION: {question}"
    if feedback:
        prompt += f"\n\nYOUR PREVIOUS ATTEMPT FAILED VERIFICATION:\n{feedback}\nFix these problems."
    return prompt


# --------------------------------------------------------------------------- #
# 2. Data structures
# --------------------------------------------------------------------------- #
@dataclass
class Citation:
    source: str
    quote: str
    valid: bool = False
    match_score: float = 0.0
    doc_offset: tuple[int, int] | None = None  # quote position inside the ORIGINAL document
    problem: str = ""


@dataclass
class Sentence:
    text: str
    citations: list[Citation]
    supported: bool = False
    problems: list[str] = field(default_factory=list)


@dataclass
class GroundedAnswer:
    question: str
    sentences: list[Sentence]
    sources: dict[str, Chunk]
    insufficient_evidence: bool = False
    missing: str = ""

    @property
    def grounding_score(self) -> float:
        """Fraction of sentences whose citations all verified. 1.0 = fully grounded."""
        if not self.sentences:
            return 1.0 if self.insufficient_evidence else 0.0
        return sum(s.supported for s in self.sentences) / len(self.sentences)

    def failure_report(self) -> str:
        if not self.sentences and not self.insufficient_evidence:
            return "- You returned no sentences. Answer with cited sentences, or set insufficient_evidence to true."
        lines = []
        for s in self.sentences:
            if not s.supported:
                lines.append(f'- "{s.text}": ' + "; ".join(s.problems))
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 3. Mechanical verification
# --------------------------------------------------------------------------- #
def _norm(s: str) -> str:
    s = s.lower().replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", " ", s).strip()


_NUM = re.compile(r"\d+(?:[.,]\d+)?")


def numbers_in(text: str) -> set[str]:
    return {n.replace(",", "") for n in _NUM.findall(text)}


def locate_quote(quote: str, chunk: Chunk, threshold: float = 90.0) -> tuple[bool, float, tuple[int, int] | None]:
    """Is `quote` (approximately) a substring of the chunk? Returns (ok, score, doc_offsets)."""
    q, t = _norm(quote), _norm(chunk.text)
    if len(q) < 8:
        return False, 0.0, None
    idx = t.find(q)
    if idx != -1:
        # offsets are approximate if whitespace was collapsed; good enough for highlighting
        return True, 100.0, (chunk.start_char + idx, chunk.start_char + idx + len(q))
    # fuzzy: tolerate tiny differences (punctuation, a dropped word)
    align = fuzz.partial_ratio_alignment(q, t)
    ok = align.score >= threshold
    offs = (chunk.start_char + align.dest_start, chunk.start_char + align.dest_end) if ok else None
    return ok, align.score, offs


def verify(raw: dict, question: str, sources: dict[str, Chunk], threshold: float = 90.0) -> GroundedAnswer:
    sentences: list[Sentence] = []
    for s in raw.get("sentences", []):
        cits = []
        for c in s.get("citations", []) or []:
            cit = Citation(source=str(c.get("source", "")).strip("[] "), quote=str(c.get("quote", "")))
            chunk = sources.get(cit.source)
            if chunk is None:
                cit.problem = f"cites unknown source {cit.source!r}"
            else:
                cit.valid, cit.match_score, cit.doc_offset = locate_quote(cit.quote, chunk, threshold)
                if not cit.valid:
                    cit.problem = f"quote not found in {cit.source} (match {cit.match_score:.0f}%)"
            cits.append(cit)

        sent = Sentence(text=str(s.get("text", "")).strip(), citations=cits)
        if not cits:
            sent.problems.append("no citation")
        sent.problems += [c.problem for c in cits if c.problem]

        # numeric consistency: every number in the claim must appear in a valid quote
        quoted_nums = set().union(*[numbers_in(c.quote) for c in cits if c.valid]) if cits else set()
        missing_nums = numbers_in(sent.text) - quoted_nums
        if missing_nums:
            sent.problems.append(f"numbers {sorted(missing_nums)} not in any quote")

        sent.supported = not sent.problems
        sentences.append(sent)

    return GroundedAnswer(
        question=question, sentences=sentences, sources=sources,
        insufficient_evidence=bool(raw.get("insufficient_evidence", False)),
        missing=str(raw.get("missing", "") or ""),
    )


# --------------------------------------------------------------------------- #
# 4. Semantic verification (LLM-as-judge for entailment)
# --------------------------------------------------------------------------- #
JUDGE_SYSTEM = """You check whether evidence supports claims. For each item, answer
"supported" only if the quotes alone clearly entail the claim, with no extra assumptions.
Reply ONLY with JSON: {"verdicts": [{"id": 0, "supported": true, "reason": "..."}]}"""


def judge_support(answer: GroundedAnswer, llm: LLM) -> GroundedAnswer:
    items = [(i, s) for i, s in enumerate(answer.sentences) if s.supported]
    if not items:
        return answer
    listing = "\n".join(
        f'{i}. CLAIM: {s.text}\n   QUOTES: ' + " | ".join(f'"{c.quote}"' for c in s.citations if c.valid)
        for i, s in items
    )
    try:
        verdicts = llm.complete_json(JUDGE_SYSTEM, listing).get("verdicts", [])
    except ValueError:
        return answer  # judge failed -> keep mechanical result rather than crash
    for v in verdicts:
        i = v.get("id")
        if isinstance(i, int) and 0 <= i < len(answer.sentences) and v.get("supported") is False:
            s = answer.sentences[i]
            s.supported = False
            s.problems.append(f"judge: not entailed ({v.get('reason', '')})")
    return answer


# --------------------------------------------------------------------------- #
# 5. Rendering
# --------------------------------------------------------------------------- #
def render(answer: GroundedAnswer, show_unsupported: bool = False) -> str:
    """Produce the final text with [n] markers and a reference list."""
    ref_num: dict[str, int] = {}
    body = []
    for s in answer.sentences:
        if not s.supported and not show_unsupported:
            continue
        nums = []
        for c in s.citations:
            if c.valid:
                ref_num.setdefault(c.source, len(ref_num) + 1)
                nums.append(ref_num[c.source])
        marks = "".join(f"[{n}]" for n in sorted(set(nums)))
        flag = "" if s.supported else " (⚠ unverified)"
        body.append(f"{s.text}{marks}{flag}")

    out = " ".join(body) if body else ""
    if answer.insufficient_evidence or not body:
        gap = answer.missing or "The documents do not contain enough information to answer this."
        out = (out + "\n\n" if out else "") + f"Note: {gap}"

    if ref_num:
        out += "\n\nSources:"
        for sid, n in sorted(ref_num.items(), key=lambda kv: kv[1]):
            chunk = answer.sources[sid]
            quotes = {c.quote for s in answer.sentences if s.supported or show_unsupported
                      for c in s.citations if c.source == sid and c.valid}
            out += f"\n  [{n}] {chunk.citation_label()}"
            for q in quotes:
                out += f'\n      "{q.strip()}"'
    out += f"\n\n(grounding score: {answer.grounding_score:.0%})"
    return out
