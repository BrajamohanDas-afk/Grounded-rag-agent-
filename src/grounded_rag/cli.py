"""
Command line interface (installed as `grag`).

  grag ingest data/docs              index files (incremental: re-running skips unchanged chunks)
  grag sources                       list indexed files
  grag search "betz limit" --mode bm25|dense|hybrid
  grag ask "question" [-v]           one question; -v shows the agent trace & verification
  grag chat [-v]                     multi-turn chat with memory
  grag graph                         print the LangGraph as Mermaid (paste into mermaid.live)
"""
from __future__ import annotations

import argparse
import logging
import sys

from dotenv import load_dotenv

from .grounding import format_citations


def _svc():
    from .service import RAGService
    return RAGService()


def _print_result(res: dict, verbose: bool):
    print("\n" + res["answer"])
    if res["citations"]:
        print("\nSources:\n" + format_citations(res["citations"]))
    print(f"\n(grounding {res['grounding_score']:.0%} · {res['seconds']}s)")
    if verbose:
        print("\n--- verification ---")
        for r in res["reports"]:
            mark = "✓" if r["supported"] else "✗"
            print(f" {mark} {r['text']}")
            for p in r["problems"]:
                print(f"     ! {p}")


def _run(svc, question: str, thread_id: str, verbose: bool):
    try:
        _stream(svc, question, thread_id, verbose)
    except Exception as e:  # noqa: BLE001 - show a readable error instead of a traceback
        print(f"\n\n❌ {type(e).__name__}: {str(e)[:300]}")
        print("   Check GRAG_LLM_PROVIDER / GRAG_LLM_MODEL and your API key in .env "
              "(or run with a local Ollama model).")


def _stream(svc, question: str, thread_id: str, verbose: bool):
    for ev in svc.stream(question, thread_id):
        if ev["type"] == "step":
            if verbose:
                for t in ev["trace"]:
                    print(f"  · {t}")
            else:
                print(f"  · {ev['node']}", end="", flush=True)
        else:
            _print_result(ev, verbose)


def main(argv=None):
    load_dotenv()
    p = argparse.ArgumentParser(prog="grag")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest").add_argument("path")
    sub.add_parser("sources")
    sp = sub.add_parser("search"); sp.add_argument("query"); sp.add_argument("-k", type=int, default=5)
    sp.add_argument("--mode", default="hybrid", choices=["hybrid", "bm25", "dense"])
    ap = sub.add_parser("ask"); ap.add_argument("question"); ap.add_argument("-v", "--verbose", action="store_true")
    cp = sub.add_parser("chat"); cp.add_argument("-v", "--verbose", action="store_true")
    sub.add_parser("graph")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    svc = _svc()
    if args.cmd == "ingest":
        print(svc.ingest(args.path))
    elif args.cmd == "sources":
        for src, n in svc.index.sources().items():
            print(f"{n:5d} chunks  {src}")
    elif args.cmd == "search":
        from .ingestion import citation_label
        for d in svc.search(args.query, args.k, args.mode):
            print(f"- {citation_label(d.metadata)}\n  {d.page_content[:160]!r}")
    elif args.cmd == "ask":
        _run(svc, args.question, "cli", args.verbose)
    elif args.cmd == "chat":
        import uuid
        tid = str(uuid.uuid4())
        print("Chat with your documents (empty line to quit)")
        while True:
            try:
                q = input("\nyou> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                break
            _run(svc, q, tid, args.verbose)
    elif args.cmd == "graph":
        print(svc.mermaid())


if __name__ == "__main__":
    sys.exit(main())
