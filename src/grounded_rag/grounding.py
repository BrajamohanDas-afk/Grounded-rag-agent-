"""
Citation grounding: turning "the model put [1] after a sentence" into
"code has PROVEN the source contains this quote, and a judge agrees it supports the claim".

Checks per sentence:
  1. has at least one citation
  2. every cited source id exists in the evidence given to the model
  3. every quote is found in its source (exact, else fuzzy >= threshold)
  4. every number in the sentence appears in one of its verified quotes
  5. (optional) an LLM judge confirms the quotes ENTAIL the sentence

A sentence is "supported" only if all checks pass. Only supported sentences reach the user.
"""
from __future__ import annotations

import re

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from rapidfuzz import fuzz

from .ingestion import citation_label
from .prompts import JUDGE
from .schemas import CitedAnswer, SupportVerdicts


def _norm(s: str) -> str:
    s = (s.lower().replace("\u2019", "'").replace("\u2018", "'")
         .replace("\u201c", '"').replace("\u201d", '"').replace("\u2013", "-").replace("\u2014", "-"))
    return re.sub(r"\s+", " ", s).strip()


_NUM = re.compile(r"(?<![a-z])\d+(?:[.,]\d+)*")


def numbers_in(text: str) -> set[str]:
    return {n.replace(",", "").rstrip(".") for n in _NUM.findall(text.lower())}


def locate_quote(quote: str, text: str, threshold: float) -> tuple[bool, float, tuple[int, int] | None]:
    """Find `quote` in `text`. Returns (found, score 0-100, (start, end) in normalised text)."""
    q, t = _norm(quote), _norm(text)
    if len(q) < 10 or len(q.split()) < 3:           # too short to be meaningful evidence
        return False, 0.0, None
    i = t.find(q)
    if i >= 0:
        return True, 100.0, (i, i + len(q))
    a = fuzz.partial_ratio_alignment(q, t)
    return a.score >= threshold, float(a.score), (a.dest_start, a.dest_end) if a.score >= threshold else None


def verify_answer(answer: CitedAnswer, sources: dict[str, Document], threshold: float = 90.0) -> list[dict]:
    """Mechanical checks. Returns one report dict per sentence (JSON-serialisable for graph state)."""
    reports = []
    for sent in answer.sentences:
        problems, cites = [], []
        if not sent.citations:
            problems.append("no citation")
        for c in sent.citations:
            sid = c.source_id.strip().strip("[]")
            rec = {"source_id": sid, "quote": c.quote, "valid": False, "score": 0.0, "span": None}
            doc = sources.get(sid)
            if doc is None:
                problems.append(f"cites unknown source '{sid}'")
            else:
                ok, score, span = locate_quote(c.quote, doc.page_content, threshold)
                rec.update(valid=ok, score=score, span=span)
                if not ok:
                    problems.append(f"quote not found in {sid} (best match {score:.0f}%): \"{c.quote[:80]}\"")
            cites.append(rec)
        verified_nums = set().union(*(numbers_in(c["quote"]) for c in cites if c["valid"])) if cites else set()
        missing = numbers_in(sent.text) - verified_nums
        if missing:
            problems.append(f"numbers {sorted(missing)} do not appear in any verified quote")
        reports.append({"text": sent.text, "citations": cites, "problems": problems,
                        "supported": not problems, "judge": None})
    return reports


def judge_reports(reports: list[dict], llm: BaseChatModel) -> list[dict]:
    """Semantic check on sentences that passed the mechanical checks."""
    todo = [(i, r) for i, r in enumerate(reports) if r["supported"]]
    if not todo:
        return reports
    claims = "\n\n".join(
        f"Claim {i}: {r['text']}\nQuotes: " + " | ".join(f'"{c["quote"]}"' for c in r["citations"] if c["valid"])
        for i, r in todo)
    try:
        result: SupportVerdicts = (JUDGE | llm.with_structured_output(SupportVerdicts)).invoke({"claims": claims})
    except (ValueError, KeyError, TypeError) as e:   # malformed judge reply -> keep mechanical result
        for _, r in todo:
            r["judge"] = f"judge error: {type(e).__name__}"
        return reports
    for v in result.verdicts:
        if 0 <= v.index < len(reports) and reports[v.index]["supported"]:
            reports[v.index]["judge"] = v.reason
            if not v.supported:
                reports[v.index]["supported"] = False
                reports[v.index]["problems"].append(f"judge: quotes do not entail claim ({v.reason})")
    return reports


def grounding_score(reports: list[dict]) -> float:
    return sum(r["supported"] for r in reports) / len(reports) if reports else 0.0


def failure_feedback(reports: list[dict]) -> str:
    if not reports:
        return "You returned no sentences. Answer with cited sentences, or set insufficient_evidence=true."
    return "\n".join(f'- Sentence "{r["text"]}": ' + "; ".join(r["problems"])
                     for r in reports if not r["supported"])


def render(reports: list[dict], sources: dict[str, Document], insufficient: bool, missing: str) -> tuple[str, list[dict]]:
    """Build the user-facing answer text + a structured citation list (for UIs/APIs).
    Sources are renumbered [1], [2], ... in order of first use."""
    number: dict[str, int] = {}
    parts = []
    for r in reports:
        if not r["supported"]:
            continue
        marks = []
        for c in r["citations"]:
            if c["valid"]:
                number.setdefault(c["source_id"], len(number) + 1)
                marks.append(number[c["source_id"]])
        parts.append(r["text"].rstrip() + "".join(f"[{n}]" for n in sorted(set(marks))))

    text = " ".join(parts)
    dropped = sum(not r["supported"] for r in reports)
    if not parts:
        text = "I couldn't find this in the documents."
        if missing:
            text += f" {missing}"
    elif insufficient or missing:
        text += f"\n\nNote: {missing or 'The documents only partially answer this question.'}"
    if dropped and parts:
        text += f"\n\n({dropped} statement(s) were removed because they could not be verified against the sources.)"

    citations = []
    for sid, n in sorted(number.items(), key=lambda kv: kv[1]):
        d = sources[sid]
        quotes = sorted({c["quote"] for r in reports if r["supported"]
                         for c in r["citations"] if c["source_id"] == sid and c["valid"]})
        citations.append({"number": n, "source_id": sid, "label": citation_label(d.metadata),
                          "source": d.metadata.get("source"), "page": d.metadata.get("page"),
                          "section": d.metadata.get("section"), "chunk_id": d.metadata.get("chunk_id"),
                          "quotes": quotes, "chunk_text": d.page_content})
    return text, citations


def format_citations(citations: list[dict]) -> str:
    lines = []
    for c in citations:
        lines.append(f"[{c['number']}] {c['label']}")
        lines += [f'    "{q}"' for q in c["quotes"]]
    return "\n".join(lines)


def find_span(text: str, quote: str, threshold: float = 85.0) -> tuple[int, int] | None:
    """Character span of `quote` inside the ORIGINAL (un-normalised) text, for highlighting in a UI.
    Tolerates whitespace/case differences exactly, then falls back to fuzzy alignment."""
    words = [re.escape(w) for w in quote.split()]
    if not words:
        return None
    m = re.search(r"\s+".join(words), text, flags=re.I)
    if m:
        return m.start(), m.end()
    a = fuzz.partial_ratio_alignment(quote.lower(), text.lower())
    return (a.dest_start, a.dest_end) if a.score >= threshold else None
