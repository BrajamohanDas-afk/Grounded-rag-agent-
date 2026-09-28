"""
Evaluation harness.

    python eval/run_eval.py --retrieval-only                  # no LLM: compare bm25/dense/hybrid
    python eval/run_eval.py --preset full                      # one configuration
    python eval/run_eval.py --compare                          # ablation study across presets
    python eval/run_eval.py --set top_k=8 --set use_judge=false --tag my_experiment

Metrics (per question, then averaged):
  retrieval_recall    expected source files found in the evidence the agent gathered
  context_precision   share of evidence chunks that come from expected files (noise level)
  citation_precision  share of cited files that are expected files
  citation_recall     share of expected files that got cited
  grounding           share of answer sentences that passed verification
  correctness         LLM judge vs. reference answer: 1 / 0.5 / 0
  abstain_ok          unanswerable questions: 1 if the agent cited nothing (did not make things up)
  false_abstain       answerable questions: 1 if the agent cited nothing (too cautious)
  seconds             wall-clock latency

Results: eval/results/<tag>.jsonl and a row appended to eval/results/summary.md
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from grounded_rag.config import Settings            # noqa: E402
from grounded_rag.prompts import CORRECTNESS        # noqa: E402
from grounded_rag.schemas import CorrectnessGrade   # noqa: E402
from grounded_rag.service import RAGService         # noqa: E402

RESULTS = ROOT / "eval" / "results"

# Ablations: each preset switches features OFF to show what they contribute.
PRESETS: dict[str, dict] = {
    "naive_rag":     dict(max_sub_queries=0, max_retrieval_rounds=1, max_generation_attempts=1,
                          use_judge=False, bm25_weight=0.0, quote_match_threshold=0.0),
    "hybrid_only":   dict(max_sub_queries=0, max_retrieval_rounds=1, max_generation_attempts=1,
                          use_judge=False),
    "verified":      dict(max_sub_queries=0, max_retrieval_rounds=1, max_generation_attempts=2,
                          use_judge=True),
    "full":          dict(),   # everything on (defaults)
}
# note: naive_rag sets quote_match_threshold=0 so every citation "passes": it measures what the
# user would see WITHOUT verification.


def load_dataset(limit: int | None) -> list[dict]:
    rows = [json.loads(l) for l in (ROOT / "eval" / "dataset.jsonl").read_text().splitlines() if l.strip()]
    return rows[:limit] if limit else rows


def parse_sets(pairs: list[str]) -> dict:
    out = {}
    for p in pairs:
        k, v = p.split("=", 1)
        out[k] = {"true": True, "false": False}.get(v.lower(), v)
    return out


def correctness(svc: RAGService, item: dict, answer: str) -> float | None:
    if not item["reference"]:
        return None
    g: CorrectnessGrade = (CORRECTNESS | svc.llm.with_structured_output(CorrectnessGrade)).invoke(
        {"question": item["question"], "reference": item["reference"], "answer": answer})
    return {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}[g.verdict]


def score(item: dict, res: dict) -> dict:
    exp = set(item["expected_sources"])
    ev_sources = [e["label"] for e in res["evidence"].values()]
    cited = {c["source"] for c in res["citations"]}
    row = {"id": item["id"], "type": item["type"], "grounding": res["grounding_score"],
           "seconds": res["seconds"], "n_queries": len(res["queries"]), "answer": res["answer"],
           "cited": sorted(cited)}
    if exp:
        row["retrieval_recall"] = len(exp & set(ev_sources)) / len(exp)
        row["context_precision"] = (sum(s in exp for s in ev_sources) / len(ev_sources)) if ev_sources else 0.0
        row["citation_precision"] = len(cited & exp) / len(cited) if cited else 0.0
        row["citation_recall"] = len(cited & exp) / len(exp)
        row["false_abstain"] = float(not cited)
    else:
        row["abstain_ok"] = float(not cited)
    return row


METRICS = ["retrieval_recall", "context_precision", "citation_precision", "citation_recall",
           "grounding", "correctness", "abstain_ok", "false_abstain", "seconds"]


def run(tag: str, overrides: dict, limit: int | None, judge_correctness: bool) -> dict:
    s = Settings(**overrides)
    svc = RAGService(settings=s)
    if not svc.index.chunks:
        print("Index is empty -> ingesting data/docs")
        svc.ingest(ROOT / "data" / "docs")
    rows = []
    for item in load_dataset(limit):
        try:
            res = svc.ask(item["question"])
            row = score(item, res)
            if judge_correctness:
                row["correctness"] = correctness(svc, item, res["answer"])
        except Exception as e:  # noqa: BLE001
            row = {"id": item["id"], "type": item["type"], "error": f"{type(e).__name__}: {e}"}
        rows.append(row)
        print(f"  {row['id']} {row['type']:13s} grounding={row.get('grounding', 'ERR')!s:5.5} "
              f"cited={row.get('cited')} {row.get('error', '')}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{tag}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    summary = {m: round(st.mean(v), 3) for m in METRICS
               if (v := [r[m] for r in rows if r.get(m) is not None])}
    summary["errors"] = sum("error" in r for r in rows)
    return summary


def write_summary(results: dict[str, dict], overrides_by_tag: dict[str, dict]):
    path = RESULTS / "summary.md"
    header = "| run | " + " | ".join(METRICS) + " | errors | overrides |\n|" + "---|" * (len(METRICS) + 3) + "\n"
    new = not path.exists()
    with path.open("a") as f:
        if new:
            f.write(header)
        for tag, sm in results.items():
            f.write(f"| {tag} {time.strftime('%m-%d %H:%M')} | " +
                    " | ".join(str(sm.get(m, "")) for m in METRICS) +
                    f" | {sm['errors']} | `{overrides_by_tag[tag]}` |\n")
    print("\n" + header + "".join(
        f"| {t} | " + " | ".join(str(sm.get(m, "")) for m in METRICS) + f" | {sm['errors']} | |\n"
        for t, sm in results.items()))


def retrieval_only(k: int):
    svc = RAGService()
    if not svc.index.chunks:
        svc.ingest(ROOT / "data" / "docs")
    data = [d for d in load_dataset(None) if d["expected_sources"]]
    print(f"{'mode':8s} recall@{k}  MRR")
    for mode in ("bm25", "dense", "hybrid"):
        rec, rr = [], []
        for d in data:
            srcs = [x.metadata["source"] for x in svc.search(d["question"], k, mode)]
            exp = set(d["expected_sources"])
            rec.append(len(exp & set(srcs)) / len(exp))
            rr.append(next((1 / (i + 1) for i, s in enumerate(srcs) if s in exp), 0.0))
        print(f"{mode:8s} {st.mean(rec):9.3f}  {st.mean(rr):.3f}")


if __name__ == "__main__":
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--preset", choices=list(PRESETS), default="full")
    ap.add_argument("--compare", action="store_true", help="run every preset")
    ap.add_argument("--set", action="append", default=[], help="override a setting, e.g. --set top_k=8")
    ap.add_argument("--tag")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-correctness", action="store_true", help="skip the LLM correctness judge")
    a = ap.parse_args()

    if a.retrieval_only:
        retrieval_only(a.k)
        sys.exit()
    todo = {p: PRESETS[p] for p in PRESETS} if a.compare else {a.tag or a.preset: {**PRESETS[a.preset], **parse_sets(a.set)}}
    results = {}
    for tag, ov in todo.items():
        print(f"\n=== {tag} {ov} ===")
        results[tag] = run(tag, ov, a.limit, not a.no_correctness)
    write_summary(results, todo)
