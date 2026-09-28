"""
DAY 3 — Evaluation
==================
If you can't measure it you can't improve it. We measure:

  retrieval_recall : did search find the expected source files?        (retriever quality)
  answer_hit       : does the answer contain the key facts?             (usefulness)
  grounding        : share of answer sentences whose citations verify   (faithfulness)
  citation_prec.   : share of cited sources that are expected sources   (citation accuracy)
  abstain_ok       : for unanswerable questions, did the agent refuse?  (honesty)

Run:  python evaluate.py            (full agent, needs an LLM)
      python evaluate.py --retrieval-only   (no LLM needed)
"""
import argparse
import json
import statistics as st

from dotenv import load_dotenv

from rag.agent import AgentConfig, RAGAgent
from rag.llm import get_llm
from rag.retriever import HybridRetriever


def retrieval_eval(retriever, items, k=4):
    for mode in ("bm25", "dense", "hybrid"):
        recalls = []
        for it in items:
            if not it["expected_sources"]:
                continue
            found = {h.chunk.source for h in retriever.search(it["question"], k=k, mode=mode)}
            recalls.append(len(found & set(it["expected_sources"])) / len(it["expected_sources"]))
        print(f"  {mode:7s} recall@{k}: {st.mean(recalls):.2f}")


def full_eval(retriever, items, use_judge):
    agent = RAGAgent(retriever, get_llm(), AgentConfig(use_judge=use_judge))
    rows = []
    for it in items:
        ans = agent.run(it["question"]).answer
        supported = [s for s in ans.sentences if s.supported]
        text = " ".join(s.text for s in supported)
        cited = {ans.sources[c.source].source for s in supported for c in s.citations if c.valid}
        retrieved = {c.source for c in ans.sources.values()}
        row = {"q": it["question"][:50]}
        if it["answerable"]:
            exp = set(it["expected_sources"])
            row["retrieval_recall"] = len(retrieved & exp) / len(exp)
            row["answer_hit"] = float(all(m.lower() in text.lower() for m in it["must_contain"]))
            row["grounding"] = ans.grounding_score
            row["citation_precision"] = len(cited & exp) / len(cited) if cited else 0.0
        else:
            row["abstain_ok"] = float(ans.insufficient_evidence or not supported)
        rows.append(row)
        print(json.dumps(row))

    print("\n=== averages ===")
    for key in ("retrieval_recall", "answer_hit", "grounding", "citation_precision", "abstain_ok"):
        vals = [r[key] for r in rows if key in r]
        if vals:
            print(f"  {key:20s} {st.mean(vals):.2f}")


if __name__ == "__main__":
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--retrieval-only", action="store_true")
    p.add_argument("--no-judge", action="store_true")
    args = p.parse_args()
    items = json.load(open("data/eval_set.json"))
    r = HybridRetriever.load("index")
    print("Retrieval:")
    retrieval_eval(r, items)
    if not args.retrieval_only:
        print("\nEnd-to-end:")
        full_eval(r, items, use_judge=not args.no_judge)
