def test_bm25_finds_exact_terms(svc):
    top = svc.search("Betz limit", k=1, mode="bm25")[0]
    assert top.metadata["source"] == "wind_power.md"
    top = svc.search("LFP chemistry", k=1, mode="bm25")[0]
    assert "LFP" in top.page_content


def test_hybrid_contains_keyword_hit(svc):
    # with FAKE embeddings the dense side is random, so hybrid must still surface the BM25 hit
    hits = svc.search("Betz limit turbine", k=4, mode="hybrid")
    assert any("Betz" in d.page_content for d in hits)


def test_empty_index_returns_nothing(settings, llm):
    from grounded_rag.service import RAGService
    assert RAGService(settings=settings, llm=llm).search("anything") == []
