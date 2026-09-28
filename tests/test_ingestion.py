from grounded_rag.documents import load_path
from grounded_rag.ingestion import citation_label, split_documents


def test_chunks_have_provenance(svc):
    chunks = svc.index.all_chunks()
    assert len(chunks) > 3
    for c in chunks:
        m = c.metadata
        assert m["source"].split(".")[-1] in {"md", "pdf", "html"} and "start_index" in m and len(m["chunk_id"]) == 16
    # markdown section tracking
    assert any(c.metadata.get("section") == "Battery Storage" for c in chunks)


def test_start_index_points_into_original(settings, tmp_path):
    from tests.conftest import DOCS
    docs = load_path(DOCS / "wind_power.md")
    for ch in split_documents(docs, settings):
        start = ch.metadata["start_index"]
        assert docs[0].page_content[start:start + 20] == ch.page_content[:20]


def test_incremental_reingest_is_noop(svc, tmp_path):
    stats = svc.ingest(tmp_path / "docs")
    assert stats["added"] == 0 and stats["skipped_existing"] > 0


def test_delete_source(svc):
    n = svc.delete_source("wind_power.md")
    assert n > 0 and "wind_power.md" not in svc.index.sources()
    assert all(d.metadata["source"] != "wind_power.md" for d in svc.search("Betz limit wind", mode="bm25"))


def test_pdf_pages(settings, tmp_path):
    from reportlab.pdfgen import canvas
    pdf = tmp_path / "report.pdf"
    c = canvas.Canvas(str(pdf))
    c.drawString(72, 720, "Page one talks about hydrogen electrolysis efficiency."); c.showPage()
    c.drawString(72, 720, "Page two says geothermal plants run around the clock."); c.showPage()
    c.save()
    docs = load_path(pdf)
    assert [d.metadata["page"] for d in docs] == [1, 2]
    chunks = split_documents(docs, settings)
    assert citation_label(chunks[1].metadata) == "report.pdf, p.2"
