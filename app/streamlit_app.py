"""
Streamlit UI:  streamlit run app/streamlit_app.py

  * sidebar: upload & index files, list/delete sources, tweak agent knobs live
  * chat with memory (one LangGraph thread per browser session)
  * live view of each agent step while it runs
  * answers with [n] markers; each source expands to the chunk with the quote HIGHLIGHTED
  * verification panel: which sentences passed / failed and why
"""
from __future__ import annotations

import html
import tempfile
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from grounded_rag.documents import SUPPORTED
from grounded_rag.grounding import find_span
from grounded_rag.service import RAGService

load_dotenv()
st.set_page_config(page_title="Grounded RAG Agent", page_icon="📎", layout="wide")


@st.cache_resource(show_spinner="Loading models and index…")
def get_service() -> RAGService:
    return RAGService()


svc = get_service()
ss = st.session_state
ss.setdefault("thread_id", str(uuid.uuid4()))
ss.setdefault("turns", [])          # list of (question, result)


def highlight(text: str, quotes: list[str]) -> str:
    spans = sorted(s for q in quotes if (s := find_span(text, q)))
    out, pos = [], 0
    for a, b in spans:
        if a < pos:
            continue
        out += [html.escape(text[pos:a]), f"<mark>{html.escape(text[a:b])}</mark>"]
        pos = b
    out.append(html.escape(text[pos:]))
    return "<div style='white-space:pre-wrap;font-size:0.9em'>" + "".join(out) + "</div>"


def grounding_badge(score: float) -> str:
    color = "green" if score >= 0.99 else "orange" if score >= 0.5 else "red"
    return f":{color}[**grounding {score:.0%}**]"


def show_result(res: dict):
    st.markdown(res["answer"])
    st.caption(f"{grounding_badge(res['grounding_score'])} · {res['seconds']}s · route: {res['route']}")
    for c in res["citations"]:
        with st.expander(f"[{c['number']}] {c['label']}"):
            st.markdown(highlight(c["chunk_text"], c["quotes"]), unsafe_allow_html=True)
    if res["reports"]:
        with st.expander("🔎 Verification details"):
            for r in res["reports"]:
                st.markdown(("✅ " if r["supported"] else "❌ ") + r["text"])
                for p in r["problems"]:
                    st.caption(f"↳ {p}")
    with st.expander("🧭 Agent trace"):
        if res["standalone_question"] and res["standalone_question"] != res["question"]:
            st.markdown(f"**Rewritten question:** {res['standalone_question']}")
        st.code("\n".join(res["trace"]), language=None)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("📚 Documents")
    files = st.file_uploader("Add files", type=[e.strip(".") for e in SUPPORTED], accept_multiple_files=True)
    if files and st.button("Index files", type="primary"):
        with tempfile.TemporaryDirectory() as tmp:
            for f in files:
                (Path(tmp) / Path(f.name).name).write_bytes(f.getvalue())
            with st.spinner("Chunking and embedding…"):
                stats = svc.ingest(tmp)
        st.success(f"Added {stats['added']} chunks ({stats['skipped_existing']} already indexed)")

    for src, n in svc.index.sources().items():
        c1, c2 = st.columns([5, 1])
        c1.caption(f"{src} · {n} chunks")
        if c2.button("🗑", key=f"del-{src}", help=f"Remove {src}"):
            svc.delete_source(src)
            st.rerun()

    st.header("⚙️ Agent")
    svc.s.top_k = st.slider("Chunks per query", 2, 12, svc.s.top_k)
    svc.s.max_retrieval_rounds = st.slider("Max retrieval rounds", 1, 4, svc.s.max_retrieval_rounds)
    svc.s.max_generation_attempts = st.slider("Max answer attempts (repair loop)", 1, 4, svc.s.max_generation_attempts)
    svc.s.use_judge = st.toggle("LLM judge (entailment check)", svc.s.use_judge)
    st.caption(f"LLM: `{svc.s.llm_provider}:{svc.s.llm_model}`  \nEmbeddings: `{svc.s.embedding_model}`")
    if st.button("New conversation"):
        ss.thread_id, ss.turns = str(uuid.uuid4()), []
        st.rerun()

# ---------------------------------------------------------------- chat
st.title("📎 Grounded RAG Agent")
st.caption("Every sentence is backed by a verbatim quote that was checked in code.")

if not svc.index.chunks:
    st.info("Upload documents in the sidebar to get started.")

for q, res in ss.turns:
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        show_result(res)

if question := st.chat_input("Ask about your documents…"):
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        result = None
        with st.status("Thinking…", expanded=True) as status:
            try:
                for ev in svc.stream(question, ss.thread_id):
                    if ev["type"] == "step":
                        for t in ev["trace"]:
                            st.write(f"**{ev['node']}** — {t}")
                    else:
                        result = ev
                status.update(label="Done", state="complete", expanded=False)
            except Exception as e:  # noqa: BLE001
                status.update(label="Error", state="error")
                st.error(f"{type(e).__name__}: {e}")
        if result:
            show_result(result)
            ss.turns.append((question, result))
