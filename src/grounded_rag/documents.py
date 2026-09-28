"""
Loading: files -> LangChain `Document` objects.

A `Document` is just `page_content: str` + `metadata: dict`. Everything we want to
show in a citation later (file, page, section, character offset) must be put into
metadata NOW — it cannot be recovered after chunking.

We write small loaders ourselves instead of using langchain-community loaders
(that package is being sunset) — it's ~50 lines and you see exactly what happens.
"""
from __future__ import annotations

import re
from pathlib import Path

from langchain_core.documents import Document

SUPPORTED = {".pdf", ".md", ".markdown", ".txt", ".html", ".htm"}


def _clean(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"-\n(\w)", r"\1", text)        # re-join words hyphenated across PDF lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(path: Path) -> list[Document]:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    title = (reader.metadata.title if reader.metadata and reader.metadata.title else path.stem)
    docs = []
    for i, page in enumerate(reader.pages):
        text = _clean(page.extract_text() or "")
        if text:
            docs.append(Document(page_content=text, metadata={
                "source": path.name, "path": str(path), "page": i + 1,
                "title": str(title), "file_type": "pdf"}))
    return docs


def load_text(path: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"^#\s+(.+)$", text, flags=re.M)
    title = m.group(1).strip() if m else path.stem
    return [Document(page_content=text.strip(), metadata={
        "source": path.name, "path": str(path), "title": title,
        "file_type": "markdown" if path.suffix in {".md", ".markdown"} else "text"})]


def load_html(path: Path) -> list[Document]:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    title = soup.title.get_text(strip=True) if soup.title else path.stem
    return [Document(page_content=_clean(soup.get_text("\n")), metadata={
        "source": path.name, "path": str(path), "title": title, "file_type": "html"})]


def load_file(path: str | Path) -> list[Document]:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        return load_pdf(path)
    if ext in {".md", ".markdown", ".txt"}:
        return load_text(path)
    if ext in {".html", ".htm"}:
        return load_html(path)
    raise ValueError(f"Unsupported file type: {path.name}")


def load_path(path: str | Path) -> list[Document]:
    """Load one file or every supported file under a directory."""
    path = Path(path)
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.suffix.lower() in SUPPORTED)
    docs: list[Document] = []
    for f in files:
        docs.extend(load_file(f))
    return docs
