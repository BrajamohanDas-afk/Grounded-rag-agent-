import importlib.util
from pathlib import Path

from grounded_rag.config import Settings

spec = importlib.util.spec_from_file_location("run_eval", Path(__file__).parent.parent / "eval" / "run_eval.py")
run_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_eval)


def test_presets_are_valid_settings():
    for ov in run_eval.PRESETS.values():
        Settings(**ov)


def test_dataset_loads():
    rows = run_eval.load_dataset(None)
    assert len(rows) >= 15 and any(r["type"] == "unanswerable" for r in rows)


def test_scoring():
    item = {"id": "x", "type": "comparison", "expected_sources": ["a.md", "b.pdf"]}
    res = {"grounding_score": 1.0, "seconds": 1.0, "queries": ["q"], "answer": "..",
           "evidence": {"S1": {"label": "a.md"}, "S2": {"label": "c.md"}},
           "citations": [{"source": "a.md"}, {"source": "c.md"}]}
    r = run_eval.score(item, res)
    assert r["retrieval_recall"] == 0.5 and r["citation_precision"] == 0.5 and r["false_abstain"] == 0.0
    r = run_eval.score({"id": "y", "type": "unanswerable", "expected_sources": []}, {**res, "citations": []})
    assert r["abstain_ok"] == 1.0
