"""
Command line entry point.

  python main.py ingest data/docs          # Day 1: build the index
  python main.py search "battery cost"     # Day 1-2: inspect retrieval only (no LLM)
  python main.py ask "How efficient are solar panels?"   # Day 2-3: full agent
  python main.py chat                      # interactive loop
"""
import argparse
import os

from dotenv import load_dotenv

from rag.agent import AgentConfig, RAGAgent
from rag.grounding import render
from rag.ingest import ingest_directory
from rag.llm import get_llm
from rag.retriever import HybridRetriever

INDEX_DIR = "index"


def cmd_ingest(args):
    chunks = ingest_directory(args.path, max_chars=args.chunk_chars)
    retriever = HybridRetriever(chunks, os.getenv("EMBEDDER", "tfidf"))
    retriever.save(INDEX_DIR)
    print(f"Indexed {len(chunks)} chunks from {len({c.source for c in chunks})} files -> {INDEX_DIR}/")


def cmd_search(args):
    r = HybridRetriever.load(INDEX_DIR)
    for h in r.search(args.query, k=args.k, mode=args.mode):
        print(f"{h.score:7.4f}  {h.chunk.citation_label():25s} {h.chunk.text[:90]!r}")


def make_agent(args) -> RAGAgent:
    return RAGAgent(HybridRetriever.load(INDEX_DIR), get_llm(),
                    AgentConfig(use_judge=not args.no_judge))


def answer(agent, q, verbose):
    result = agent.run(q)
    if verbose:
        print("\n--- trace ---\n" + "\n".join(result.trace) + "\n-------------\n")
    print(render(result.answer, show_unsupported=verbose))


def main():
    load_dotenv()
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest"); s.add_argument("path"); s.add_argument("--chunk-chars", type=int, default=400)
    s = sub.add_parser("search"); s.add_argument("query"); s.add_argument("-k", type=int, default=5)
    s.add_argument("--mode", default="hybrid", choices=["hybrid", "dense", "bm25"])
    for name in ("ask", "chat"):
        s = sub.add_parser(name)
        if name == "ask":
            s.add_argument("question")
        s.add_argument("-v", "--verbose", action="store_true")
        s.add_argument("--no-judge", action="store_true")

    args = p.parse_args()
    if args.cmd == "ingest":
        cmd_ingest(args)
    elif args.cmd == "search":
        cmd_search(args)
    elif args.cmd == "ask":
        answer(make_agent(args), args.question, args.verbose)
    else:
        agent = make_agent(args)
        while (q := input("\n? ").strip()) not in {"", "exit", "quit"}:
            answer(agent, q, args.verbose)


if __name__ == "__main__":
    main()
