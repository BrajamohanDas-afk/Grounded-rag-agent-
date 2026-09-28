"""Offline tests (no API key). Run: python -m pytest -q"""
import json

from rag.agent import AgentConfig, RAGAgent
from rag.grounding import locate_quote, render, verify
from rag.ingest import chunk_text, ingest_directory
from rag.llm import MockLLM
from rag.retriever import HybridRetriever

CHUNKS = ingest_directory("data/docs", max_chars=400)
RET = HybridRetriever(CHUNKS, "tfidf")


def test_chunks_keep_provenance():
    assert CHUNKS and all(c.source.endswith(".md") for c in CHUNKS)
    for c in CHUNKS:
        assert c.end_char > c.start_char


def test_chunk_overlap():
    text = "One. Two is here. Three is here. Four is here."
    spans = chunk_text(text, max_chars=25, overlap_sentences=1)
    assert len(spans) > 1 and spans[1][0] < spans[0][1]  # overlap


def test_hybrid_finds_right_doc():
    assert RET.search("Betz limit turbine", k=1)[0].chunk.source == "wind_power.md"
    assert RET.search("LFP chemistry", k=1)[0].chunk.source == "battery_storage.md"


def _sources(query):
    return {f"S{i+1}": h.chunk for i, h in enumerate(RET.search(query, k=3))}


def test_quote_verification():
    c = RET.search("Betz limit", k=1)[0].chunk
    assert locate_quote("no wind turbine can capture more than 59.3 percent", c)[0]
    assert not locate_quote("turbines capture 80 percent of wind energy", c)[0]


def test_verify_catches_hallucinations():
    src = _sources("Betz limit wind turbine")
    sid = next(k for k, v in src.items() if "Betz" in v.text)
    raw = {"sentences": [
        {"text": "Turbines can capture at most 59.3% of wind energy.",
         "citations": [{"source": sid, "quote": "no wind turbine can capture more than 59.3 percent of the kinetic energy"}]},
        {"text": "Modern turbines reach 70% efficiency.",               # fake quote
         "citations": [{"source": sid, "quote": "modern turbines reach 70 percent efficiency"}]},
        {"text": "Offshore wind is cheap.", "citations": [{"source": "S99", "quote": "x" * 20}]},  # bad id
        {"text": "The limit is 60 percent.",                            # wrong number, real quote
         "citations": [{"source": sid, "quote": "The Betz limit states that no wind turbine"}]},
    ]}
    ans = verify(raw, "q", src)
    assert [s.supported for s in ans.sentences] == [True, False, False, False]
    out = render(ans)
    assert "59.3" in out and "70%" not in out and "grounding score: 25%" in out


def test_agent_loop_with_mock_llm():
    answer = {"sentences": [{"text": "LFP is safer and lasts more cycles.",
               "citations": [{"source": "S1", "quote": "it is safer and lasts for more charge cycles"}]}]}
    llm = MockLLM([
        json.dumps({"action": "search", "query": "LFP stationary storage"}),
        json.dumps({"action": "answer"}),
        json.dumps(answer),
        json.dumps({"verdicts": [{"id": 0, "supported": True}]}),
    ])
    res = RAGAgent(RET, llm, AgentConfig()).run("LFP chemistry safer?")
    assert res.answer.grounding_score == 1.0
    assert sum("search(" in t for t in res.trace) == 2


def test_repair_loop():
    bad = {"sentences": [{"text": "LFP is great.", "citations": [{"source": "S1", "quote": "totally invented words here"}]}]}
    good = {"sentences": [{"text": "LFP is safer.", "citations": [{"source": "S1", "quote": "it is safer and lasts for more charge cycles"}]}]}
    llm = MockLLM([json.dumps({"action": "answer"}), json.dumps(bad), json.dumps(good)])
    res = RAGAgent(RET, llm, AgentConfig(use_judge=False)).run("LFP chemistry safer?")
    assert res.answer.grounding_score == 1.0
    assert "FAILED VERIFICATION" in llm.calls[-1][1]  # feedback was sent back
