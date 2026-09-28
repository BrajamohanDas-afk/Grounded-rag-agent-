> **Part 1 of the learning path.** This folder is the no-framework version used in Blocks 1.2 and 2.1 of
> [`../LEARNING_PLAN.md`](../LEARNING_PLAN.md). Build intuition here, then move to the LangChain + LangGraph
> version in `../src/grounded_rag/`. Run commands from inside this folder.

# RAG Agent with Citation Grounding — from scratch in 3 days

An agent that answers questions from **your** documents, where every sentence it says
is backed by a **verbatim quote** that the code itself checks. Unsupported sentences
are removed before you ever see them.

No LangChain / LlamaIndex on purpose: you build every piece yourself so you understand
what those frameworks do. (Learn them afterwards; they'll make much more sense.)

```
question
   │
   ▼
┌──────────┐  search(q)   ┌───────────────────────────┐
│  PLANNER │ ───────────► │ HYBRID RETRIEVER          │
│  (LLM)   │ ◄─────────── │ BM25 + embeddings + RRF   │
└──────────┘  chunks S1.. └───────────────────────────┘
   │ "answer"
   ▼
┌──────────┐   JSON: sentences + source ids + exact quotes
│ ANSWERER │ ─────────────────────────────────────────┐
└──────────┘                                          ▼
   ▲                                      ┌─────────────────────┐
   │   failure report (repair loop)       │ VERIFIER            │
   └───────────────────────────────────── │ 1. source exists?   │
                                          │ 2. quote in source? │
                                          │ 3. numbers match?   │
                                          │ 4. LLM judge entail │
                                          └─────────────────────┘
                                                     │
                                                     ▼
                                   answer with [1][2] + quoted sources + grounding score
```

## Project layout

| File | Day | What it teaches |
|---|---|---|
| `rag/ingest.py` | 1 | Loading PDFs/text, sentence-aware chunking with overlap, provenance metadata |
| `rag/embeddings.py` | 1 | What an embedding is; TF-IDF baseline vs neural (sentence-transformers) |
| `rag/retriever.py` | 1–2 | Vector search as a dot product, BM25, Reciprocal Rank Fusion |
| `rag/llm.py` | 2 | Provider-agnostic LLM calls, forcing & repairing JSON output |
| `rag/grounding.py` | 2 | **The core:** citation prompt, quote verification, numeric checks, LLM judge, rendering |
| `rag/agent.py` | 3 | Agent loop: plan → search → answer → verify → repair |
| `evaluate.py` | 3 | Measuring retrieval recall, grounding, citation precision, abstention |
| `tests/test_pipeline.py` | 1–3 | Offline tests using a mock LLM (no API key needed) |
| `main.py` | all | CLI: `ingest`, `search`, `ask`, `chat` |

## Setup (30 min, before Day 1)

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # add your API key, or point to local Ollama (free)
python -m pytest -q         # 7 tests should pass with NO api key
```

Free option: install [Ollama](https://ollama.com), run `ollama pull llama3.1:8b`, and in `.env`
set `LLM_PROVIDER=openai`, `OPENAI_BASE_URL=http://localhost:11434/v1`, `OPENAI_API_KEY=ollama`,
`OPENAI_MODEL=llama3.1:8b`. Small local models follow the JSON format less reliably — which is
actually a great way to watch the repair loop and verifier earn their keep.

---

## DAY 1 — Retrieval: getting the right text (≈6 h)

**Mindset:** an LLM can only cite what you retrieve. Bad retrieval = bad answers, no matter the model.

**Morning — concepts (2 h)**
- What RAG is and why it beats fine-tuning for private/changing knowledge.
- Chunking: why chunk at all (context limits, precision), chunk size vs. context trade-off, overlap.
- Embeddings & cosine similarity. Watch/read: any intro on "vector embeddings"; then realise
  cosine similarity of normalised vectors is just a dot product.
- Lexical search (TF-IDF, BM25) vs. semantic search.

**Afternoon — build (4 h)**
1. Read and run `rag/ingest.py` (`python -m rag.ingest`). Put 5–10 of your own PDFs/notes in `data/docs/`.
2. Understand `Chunk`: why `source`, `page`, `start_char` exist → they become citations tomorrow.
3. Read `embeddings.py` and `VectorStore` in `retriever.py`. Note there is no magic: a matrix and `argsort`.
4. `python main.py ingest data/docs` then try `python main.py search "..." --mode dense`.

**Exercises**
- Change `--chunk-chars` to 150 and 1500. Search the same query. What breaks at each extreme?
- Set `overlap_sentences=0`. Find a question whose answer now straddles two chunks.
- Search `"how long do solar panels last"` — the "25 years" chunk is not first. Why? (TF-IDF has
  no idea "last" ≈ "degrade ... after 25 years".) Keep this in mind for Day 2.

**Checkpoint:** you can explain chunk → vector → nearest neighbours, and search returns sensible chunks.

---

## DAY 2 — Generation with grounded citations (≈7 h)

**Mindset:** "the model added [1] after the sentence" is **not** grounding. Grounding means
you can *prove*, in code, that the source says it.

**Morning — better retrieval (2.5 h)**
1. `BM25Index` + `reciprocal_rank_fusion` in `retriever.py`. Compare modes:
   `python main.py search "LFP" --mode dense|bm25|hybrid`.
2. Upgrade embeddings: `pip install sentence-transformers`, set `EMBEDDER=sbert`, re-ingest.
   Retry the "how long do solar panels last" query — see semantic search win.
3. `python evaluate.py --retrieval-only` to compare recall of the three modes.

**Afternoon — citation grounding (4.5 h)** — spend most of your time in `rag/grounding.py`
1. `ANSWER_SYSTEM`: why sentence-level claims, why source IDs, **why exact quotes** (they turn
   "is this supported?" from a judgement call into a string search).
2. `locate_quote`: exact match first, fuzzy (`rapidfuzz.partial_ratio`) as fallback, and it returns
   the quote's character offsets in the original document (useful for highlighting in a UI).
3. `verify`: three mechanical checks — unknown source id, quote not found, numbers in the claim
   missing from the quote (the most common hallucination: a real quote attached to a wrong number).
4. `judge_support`: a quote can exist and still not support the claim. A second LLM call checks entailment.
5. `render`: renumbers sources as [1], [2] in order of first use and prints the quotes.
6. Run `python main.py ask "What efficiency do commercial solar panels have?" -v`.

**Exercises**
- Read `test_verify_catches_hallucinations` — it lists the 4 failure types. Add a fifth (e.g. quote
  shorter than 8 chars "gaming" the check).
- Temporarily delete the "EXACT quote" rule from the prompt. Run a few questions with `-v`. How
  does the grounding score change?
- Ask something the docs don't cover. Does it set `insufficient_evidence`?

**Checkpoint:** answers show `[n]` markers with verbatim quotes, and you can make the verifier
reject a hallucinated sentence.

---

## DAY 3 — Make it an agent, then measure it (≈7 h)

**Mindset:** one search with the raw question fails on multi-part questions ("compare X and Y").
Let the model decide what to search next, and let verification failures drive a retry.

**Morning — the agent loop (3.5 h)** — `rag/agent.py`
1. Planner: returns `{"action":"search","query":...}` or `{"action":"answer"}`. Guardrails:
   `max_searches`, no repeated queries, falls back to "answer" if JSON breaks.
2. Evidence registry: chunks get stable IDs `S1, S2, …` across searches so citations don't shift.
3. Repair loop: if grounding < threshold, the verifier's `failure_report()` goes back to the model
   ("your quote for sentence X was not found in S3") for one more try — self-correction with evidence.
4. Try a comparison question with `-v` and read the trace:
   `python main.py ask "Compare solar panel efficiency with battery round-trip efficiency" -v`

**Afternoon — evaluation (3.5 h)** — `evaluate.py`
1. Write 15–20 questions for YOUR documents in `data/eval_set.json`, including 3–4 unanswerable ones.
2. `python evaluate.py` — read each metric's definition at the top of the file.
3. Run an experiment table (this is what makes it a portfolio project):

   | Variant | recall | answer_hit | grounding | cit. precision | abstain_ok |
   |---|---|---|---|---|---|
   | tfidf, no judge | | | | | |
   | sbert, no judge | | | | | |
   | sbert + judge | | | | | |
   | sbert + judge, `max_searches=1` (no agent) | | | | | |

**Checkpoint:** you can show with numbers that hybrid retrieval, verification, and the agent loop
each help (or honestly don't, on your data).

---

## Stretch goals (after the 3 days)

- **Reranker:** add a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) on the top 20 hybrid hits.
- **Real vector DB:** swap `VectorStore` for Chroma, FAISS, or pgvector; persist embeddings instead of re-computing.
- **UI:** a Streamlit/Gradio page that highlights cited spans using `Citation.doc_offset`.
- **Native tool use:** replace the JSON planner with the provider's tool-calling API.
- **NLI judge:** replace the LLM judge with a small NLI model (cheaper, deterministic).
- **Frameworks:** rebuild the same thing in LangGraph or LlamaIndex and compare.

## Common problems

| Symptom | Likely cause / fix |
|---|---|
| `LLM did not return valid JSON` | Small local model. Lower temperature (already 0), try a bigger model, or keep retries at 2+. |
| Grounding score low, answer looks right | Model paraphrased inside `quote`. Check `-v` output; tighten prompt; the repair loop should help. |
| Right doc never retrieved | Chunks too big/small, or TF-IDF vocabulary mismatch → try `sbert` and hybrid. |
| Agent searches forever | `max_searches` caps it; inspect the planner prompt and trace. |
