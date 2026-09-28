"""
Ingestion: Documents -> chunks -> index.

  * RecursiveCharacterTextSplitter tries to split on paragraphs, then lines, then
    sentences, then words, so chunks break at natural boundaries.
  * add_start_index=True records where each chunk starts in its page -> precise citations.
  * Every chunk gets a deterministic `chunk_id` (hash of source+page+offset+text), so
    re-ingesting the same file is a no-op (incremental indexing) and IDs are stable.
  * We keep TWO indexes: a vector store (semantic) and a JSONL chunk store (needed to
    rebuild BM25, and a human-readable record of exactly what the model can see).
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings

_HEADING = re.compile(r"^(#{1,6})\s+(.+)$", re.M)


def _section_at(text: str, offset: int) -> str:
    """Markdown heading path in effect at `offset`, e.g. 'Battery Storage > Costs'."""
    path: list[tuple[int, str]] = []
    for m in _HEADING.finditer(text):
        if m.start() > offset:
            break
        level = len(m.group(1))
        path = [p for p in path if p[0] < level] + [(level, m.group(2).strip())]
    return " > ".join(t for _, t in path)


def chunk_id(meta: dict, content: str) -> str:
    key = f"{meta.get('source')}|{meta.get('page')}|{meta.get('start_index')}|{content}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def split_documents(docs: list[Document], s: Settings) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=s.chunk_size, chunk_overlap=s.chunk_overlap, add_start_index=True,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
    )
    chunks: list[Document] = []
    for doc in docs:
        for i, ch in enumerate(splitter.split_documents([doc])):
            content = ch.page_content.strip()
            if len(content) < 30:        # lone headings / page numbers are useless to cite
                continue
            meta = {k: v for k, v in ch.metadata.items() if v is not None}
            if meta.get("file_type") == "markdown":
                sec = _section_at(doc.page_content, meta.get("start_index", 0))
                if sec:
                    meta["section"] = sec
            meta["chunk_index"] = i
            meta["chunk_id"] = chunk_id(meta, content)
            chunks.append(Document(page_content=content, metadata=meta, id=meta["chunk_id"]))
    return chunks


def citation_label(meta: dict) -> str:
    parts = [meta.get("source", "?")]
    if meta.get("page"):
        parts.append(f"p.{meta['page']}")
    if meta.get("section"):
        parts.append(f"§ {meta['section']}")
    return ", ".join(parts)


# --------------------------------------------------------------------------- #
class IndexManager:
    """Owns the vector store + chunk store and keeps them in sync."""

    def __init__(self, s: Settings, embeddings: Embeddings):
        self.s = s
        self.embeddings = embeddings
        self.s.index_dir.mkdir(parents=True, exist_ok=True)
        self.chunks: dict[str, Document] = self._read_chunk_store()
        self.vector_store: VectorStore = self._open_vector_store()

    # ---- persistence ----
    def _read_chunk_store(self) -> dict[str, Document]:
        out: dict[str, Document] = {}
        if self.s.chunks_path.exists():
            for line in self.s.chunks_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    d = json.loads(line)
                    out[d["metadata"]["chunk_id"]] = Document(
                        page_content=d["page_content"], metadata=d["metadata"], id=d["metadata"]["chunk_id"])
        return out

    def _write_chunk_store(self):
        with self.s.chunks_path.open("w", encoding="utf-8") as f:
            for d in self.chunks.values():
                f.write(json.dumps({"page_content": d.page_content, "metadata": d.metadata}) + "\n")

    def _open_vector_store(self) -> VectorStore:
        if self.s.vector_store == "chroma":
            from langchain_chroma import Chroma
            return Chroma(collection_name=self.s.collection, embedding_function=self.embeddings,
                          persist_directory=str(self.s.index_dir / "chroma"),
                          collection_metadata={"hnsw:space": "cosine"})
        from langchain_core.vectorstores import InMemoryVectorStore
        vs = InMemoryVectorStore(self.embeddings)
        if self.chunks:                           # memory store is rebuilt from the chunk store
            vs.add_documents(list(self.chunks.values()), ids=list(self.chunks))
        return vs

    # ---- operations ----
    def add_documents(self, docs: list[Document]) -> dict:
        chunks = split_documents(docs, self.s)
        new = [c for c in chunks if c.metadata["chunk_id"] not in self.chunks]
        # drop duplicates inside this batch too
        uniq = {c.metadata["chunk_id"]: c for c in new}
        if uniq:
            self.vector_store.add_documents(list(uniq.values()), ids=list(uniq))
            self.chunks.update(uniq)
            self._write_chunk_store()
        return {"documents": len(docs), "chunks": len(chunks), "added": len(uniq),
                "skipped_existing": len(chunks) - len(uniq)}

    def delete_source(self, source: str) -> int:
        ids = [cid for cid, d in self.chunks.items() if d.metadata.get("source") == source]
        if ids:
            self.vector_store.delete(ids=ids)
            for cid in ids:
                del self.chunks[cid]
            self._write_chunk_store()
        return len(ids)

    def sources(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for d in self.chunks.values():
            out[d.metadata["source"]] = out.get(d.metadata["source"], 0) + 1
        return out

    def all_chunks(self) -> list[Document]:
        return list(self.chunks.values())
