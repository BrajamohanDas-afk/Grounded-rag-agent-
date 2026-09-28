"""
DAY 1 — Ingestion & Chunking
============================
Goal: turn raw files into small, citable "chunks" that remember WHERE they came from.

Key idea for citation grounding: every chunk keeps provenance metadata
(source file, page, character offsets). Without this, you can never show the
user *where* an answer came from.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class Chunk:
    chunk_id: str          # stable hash -> same text always gets same id
    text: str
    source: str            # file name
    page: int | None       # page number for PDFs, None for text files
    start_char: int        # offset inside the page/document
    end_char: int
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def citation_label(self) -> str:
        """Human readable location, e.g. 'report.pdf p.3' or 'notes.md'."""
        return f"{self.source} p.{self.page}" if self.page else self.source


# --------------------------------------------------------------------------- #
# 1. Loading
# --------------------------------------------------------------------------- #
def load_file(path: Path) -> list[tuple[int | None, str]]:
    """Return a list of (page_number, text). Text files are a single 'page'."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
    if suffix in {".txt", ".md"}:
        return [(None, path.read_text(encoding="utf-8", errors="ignore"))]
    raise ValueError(f"Unsupported file type: {path}")


# --------------------------------------------------------------------------- #
# 2. Chunking
# --------------------------------------------------------------------------- #
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Return (start, end) character spans of sentences / paragraphs."""
    spans, start = [], 0
    for m in _SENTENCE_END.finditer(text):
        if text[start:m.start()].strip():
            spans.append((start, m.start()))
        start = m.end()
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans


def chunk_text(text: str, max_chars: int = 600, overlap_sentences: int = 1) -> list[tuple[int, int]]:
    """
    Sentence-aware chunking: pack whole sentences into chunks up to max_chars,
    and repeat the last `overlap_sentences` sentences at the start of the next
    chunk so facts spanning a boundary are not lost.

    Why not fixed-size character windows? They cut sentences in half, and a
    half-sentence is a terrible thing to cite.
    """
    sents = split_sentences(text)
    chunks: list[tuple[int, int]] = []
    i = 0
    while i < len(sents):
        j = i
        # always take at least one sentence, then keep adding while under budget
        while j + 1 < len(sents) and sents[j + 1][1] - sents[i][0] <= max_chars:
            j += 1
        chunks.append((sents[i][0], sents[j][1]))
        if j + 1 >= len(sents):
            break
        i = max(j + 1 - overlap_sentences, i + 1)
    return chunks


def make_chunk_id(source: str, page, start: int, text: str) -> str:
    h = hashlib.sha1(f"{source}|{page}|{start}|{text}".encode()).hexdigest()
    return h[:12]


def ingest_directory(directory: str | Path, max_chars: int = 600) -> list[Chunk]:
    directory = Path(directory)
    chunks: list[Chunk] = []
    for path in sorted(directory.rglob("*")):
        if path.suffix.lower() not in {".pdf", ".txt", ".md"}:
            continue
        for page, text in load_file(path):
            for start, end in chunk_text(text, max_chars=max_chars):
                body = text[start:end].strip()
                if len(body) < 20:  # skip tiny fragments like lone headings
                    continue
                chunks.append(Chunk(
                    chunk_id=make_chunk_id(path.name, page, start, body),
                    text=body, source=path.name, page=page,
                    start_char=start, end_char=end,
                ))
    return chunks


if __name__ == "__main__":
    for c in ingest_directory("data/docs")[:5]:
        print(f"[{c.chunk_id}] {c.citation_label()} ({c.start_char}-{c.end_char})\n  {c.text[:100]}...\n")
