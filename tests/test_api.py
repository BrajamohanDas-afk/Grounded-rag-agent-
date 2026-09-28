from fastapi.testclient import TestClient

from grounded_rag import api
from tests.conftest import answer_with
from tests.test_graph import LFP


def test_api_flow(svc, llm):
    api.set_service(svc)
    llm.handlers["CitedAnswer"] = answer_with(LFP)
    with TestClient(api.app) as c:
        assert c.get("/health").json()["chunks"] > 0
        r = c.post("/ingest", files=[("files", ("extra.md", b"# Hydro\n\nPumped hydro storage uses two reservoirs at different heights.", "text/markdown"))])
        assert r.json()["added"] >= 1 and "extra.md" in c.get("/sources").json()
        assert c.post("/ingest", files=[("files", ("x.exe", b"MZ", "application/octet-stream"))]).status_code == 400
        assert c.post("/search", json={"query": "two reservoirs", "mode": "bm25"}).json()[0]["metadata"]["source"] == "extra.md"
        res = c.post("/chat", json={"question": "Why LFP chemistry?", "thread_id": "api"}).json()
        assert res["grounding_score"] == 1.0 and res["citations"]
        body = c.post("/chat/stream", json={"question": "Why LFP chemistry?"}).text
        assert "event: step" in body and "event: result" in body
        assert len(c.get("/threads/api/history").json()) == 2
