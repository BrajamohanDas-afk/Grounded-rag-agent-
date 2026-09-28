# 3-Day Learning Plan: Grounded RAG Agent with LangChain + LangGraph

This plan is dense on purpose. Each day is ~9–10 focused hours, split into blocks. Every block has
**Learn** (concepts), **Do** (commands or code), **Check** (how you know you understood it), and some
have **Break it** (experiments that teach more than reading ever will).

If you fall behind, skip blocks marked ⭐ (stretch) and come back later. **Never skip a Check.**

```
Day 1  Foundations & Retrieval   → you can explain and debug why a chunk was (not) retrieved
Day 2  Grounded Generation       → every sentence carries a quote, and code proves the quote is real
Day 3  Agent, Product, Evaluation → LangGraph agent + API + UI + numbers that prove it works
```

---

## Before Day 1: setup (~1 hour, do it the evening before)

1. **Python 3.10+**, Git, VS Code (or any editor). On Windows, use PowerShell or WSL.
2. Create the environment:
   ```bash
   cd grounded-rag-agent
   python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
   pip install -e ".[anthropic,local,ui,dev]"
   pip install -r from_scratch/requirements.txt
   cp .env.example .env
   ```
3. **Pick an LLM** (you can switch any time):
   - *Anthropic*: get a key at console.anthropic.com, put it in `.env`. Haiku is cheap. This entire
     plan should cost only a few dollars.
   - *Free/local*: install Ollama, `ollama pull llama3.1:8b`, set `GRAG_LLM_PROVIDER=ollama`,
     `GRAG_LLM_MODEL=llama3.1:8b`. Needs ~8 GB RAM. Slower and less obedient, which is educational.
4. **Verify:** `pytest -q` → 28 passed. `cd from_scratch && python -m pytest -q` → 7 passed.
5. Optional but recommended: free LangSmith account, then add `LANGSMITH_TRACING=true`,
   `LANGSMITH_API_KEY=...` to `.env`. You'll get a visual trace of every LLM call. Enormously
   helpful on Day 3.
6. Start a `NOTES.md` file. Every Check question: write your answer there in your own words.
   That file becomes your blog post and interview prep.

**Prerequisite knowledge check.** You should be comfortable with Python functions, classes, dicts,
list comprehensions, `pip`, and virtual environments. Type hints and Pydantic help; you'll pick them
up. You do *not* need ML theory.

---

# DAY 1: Foundations & Retrieval

> Mindset: **the model can only cite what you retrieve.** Most "the LLM is wrong" bugs are
> actually retrieval bugs. Today you learn to see and fix them.

### Block 1.1: What problem are we solving? (1 h) · Learn

Read `docs/CONCEPTS.md` sections 1–3. Then answer in `NOTES.md`:

- Why can't we just paste all documents into the prompt? (context limits, cost, "lost in the middle")
- Why not fine-tune the model on our documents instead? (knowledge updates, attribution, cost, hallucination)
- What is the difference between *a citation* and *grounding*?
- Write out the RAG pipeline from memory: `load → chunk → embed → index → retrieve → generate`.
  Then add the three things this project adds: `plan`, `grade`, `verify`.

**Check:** explain to an imaginary friend, in 3 sentences, why "the model added [1]" is not proof.

### Block 1.2: RAG from scratch, retrieval half (2 h) · Do

Frameworks hide the mechanics. Build intuition on the no-framework version first.

```bash
cd from_scratch
python -m rag.ingest                       # see chunks with ids and offsets
python main.py ingest data/docs
python main.py search "how long do solar panels last" --mode dense
python main.py search "LFP" --mode bm25
```

Read, in order, with the file open next to this plan:

1. `rag/ingest.py`: `split_sentences`, `chunk_text`, why overlap exists, `Chunk.citation_label`.
2. `rag/embeddings.py`: TF-IDF vectors. Note `_normalise`: after L2 normalisation, cosine similarity is just a dot product.
3. `rag/retriever.py`: `VectorStore.search` is literally `matrix @ query_vector` + `argsort`.
   `reciprocal_rank_fusion`: score = Σ 1/(60 + rank).

**Break it:**
- `python main.py ingest data/docs --chunk-chars 100`, then search again. Why are results worse?
- Try `--chunk-chars 3000`. Now each chunk is the whole doc. Why is that also bad for *citations*?
- Search `"how long do solar panels last"` with `--mode dense`. The "25 years" chunk isn't first:
  TF-IDF has no idea that *last* ≈ *degrade over time*. **Remember this; you'll fix it with neural embeddings.**

**Check:** compute RRF by hand: doc A is rank 1 in BM25 and rank 3 in dense; doc B is rank 2 in both.
Which wins with k=60? (A: 1/61+1/63 ≈ 0.03227; B: 2/62 ≈ 0.03226. A, barely.)

### Block 1.3: LangChain core abstractions (1.5 h) · Learn + Do

LangChain is mostly a set of **interfaces** so components are swappable. Open a Python REPL
(`python`) in the project root and run each snippet.

**Documents.** Just text + metadata:
```python
from langchain_core.documents import Document
d = Document(page_content="Betz limit is 59.3%", metadata={"source": "wind.md", "page": 1})
```

**Text splitters.** `add_start_index=True` records where each chunk came from:
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
sp = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40, add_start_index=True)
text = open("data/docs/grid_integration.md").read()
for c in sp.split_documents([Document(page_content=text, metadata={"source": "grid.md"})])[:3]:
    print(c.metadata, repr(c.page_content[:60]))
```

**Embeddings + vector store:**
```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
v = emb.embed_query("how long do solar panels last"); print(len(v), v[:5])   # 384 dims
vs = InMemoryVectorStore(emb)
vs.add_documents(sp.split_documents([Document(page_content=text, metadata={"source": "grid.md"})]))
for doc, score in vs.similarity_search_with_score("why do grid operators turn off wind farms?", k=2):
    print(round(score, 3), doc.page_content[:80])
```
Notice: the query never says "curtailment", yet the curtailment chunk comes back. That is semantic search.

**Runnables & LCEL.** Everything (prompts, models, retrievers, parsers) is a `Runnable` with
`.invoke / .batch / .stream`, and `|` pipes them:
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv; load_dotenv()
llm = init_chat_model("claude-haiku-4-5-20251001", model_provider="anthropic", temperature=0)
# Ollama: init_chat_model("llama3.1:8b", model_provider="ollama")
chain = ChatPromptTemplate.from_messages([("system", "Answer in 5 words."), ("human", "{q}")]) | llm | StrOutputParser()
print(chain.invoke({"q": "What is a vector database?"}))
retriever = vs.as_retriever(search_kwargs={"k": 2})
print(retriever.invoke("duck curve"))           # retrievers are Runnables too
```

**Check:** in `NOTES.md`, list the 5 abstractions (Document, TextSplitter, Embeddings, VectorStore,
Retriever) plus Runnable/LCEL, each with a one-line definition.

### Block 1.4: Our ingestion pipeline (2 h) · Do

Read `src/grounded_rag/documents.py` then `ingestion.py`. Key ideas to find in the code:

| Idea | Where | Why it matters for citations |
|---|---|---|
| Per-page Documents for PDFs | `load_pdf` | citations can say "p.2" |
| HTML boilerplate removal | `load_html` | nav/footer text would pollute retrieval |
| `start_index` | `split_documents` | exact position in the source, for highlighting |
| Markdown section path | `_section_at` | "§ Grid Integration > Curtailment" |
| Deterministic `chunk_id` | `chunk_id()` | re-ingest = no duplicates; stable ids for eval |
| Two stores in sync | `IndexManager` | Chroma for vectors, JSONL for BM25 + inspection |

```bash
grag ingest data/docs          # 8 documents (PDF pages count separately)
grag ingest data/docs          # again: 'added': 0. Incremental indexing works.
grag sources
head -c 600 index/chunks.jsonl # look at exactly what the model will see
```

Add your own documents (5–20 PDFs/notes on a topic you know well; that makes evaluation
easier later) to a folder like `data/mine/` and `grag ingest data/mine`.

**Break it:**
- Open a PDF chunk in `index/chunks.jsonl`. Is the text clean? Hyphenation, headers, page numbers?
  Improve `_clean()` for your PDFs. (Real-world RAG quality is 50% document cleaning.)
- Change `GRAG_CHUNK_SIZE=300` then `=2000` (delete `index/` between runs). Watch chunk counts.

**Check:** why must `chunk_id` include `start_index` and not just the text? (Two identical
sentences in different places are different citations.)

### Block 1.5: Hybrid retrieval (2 h) · Do

Read `src/grounded_rag/retrieval.py`.

- `BM25Retriever` shows how to write a **custom LangChain retriever**: subclass `BaseRetriever` and
  implement `_get_relevant_documents`. Now it works with `.invoke`, `.batch`, tracing, and `|`.
- `tokenize` + `_stem`: why "electrolysers" must match "electrolyser".
- `EnsembleRetriever` (from `langchain_classic`) = weighted Reciprocal Rank Fusion, deduped by `chunk_id`.
- `CrossEncoderReranker`: reads (query, chunk) *together*; more accurate, slower. Only re-orders the top `fetch_k`.

```bash
grag search "how long do solar panels last" --mode bm25
grag search "how long do solar panels last" --mode dense    # neural embeddings fix Day-1 TF-IDF issue
grag search "PEM" --mode dense                              # acronyms: dense often struggles
grag search "PEM" --mode bm25                               # BM25 nails exact tokens
python eval/run_eval.py --retrieval-only                    # recall@5 and MRR for each mode
```

⭐ Enable the reranker: `pip install sentence-transformers`, `GRAG_USE_RERANKER=true`, re-run the
retrieval eval. Did MRR improve? At what latency cost? (`time grag search ...`)

**Break it:** set `GRAG_BM25_WEIGHT=0.0` and `=1.0`. Re-run the retrieval eval. Plot or table it
in `NOTES.md`: where is the sweet spot on *your* documents?

**Check:** give one query type where BM25 beats dense and one where dense beats BM25, from your own
experiments, not from the plan.

### Day 1 checkpoint ✅
- [ ] You can explain chunk → embedding → nearest neighbour → fusion without looking.
- [ ] Your own documents are indexed; `grag search` returns sensible chunks.
- [ ] You have a table of recall/MRR for bm25/dense/hybrid on the sample data.
- [ ] `NOTES.md` has answers to every Check.

---

# DAY 2: Grounded Generation

> Mindset: **grounding is a property you verify, not a property you ask for.** Prompting reduces
> hallucination; checking catches what prompting missed.

### Block 2.1: From scratch, generation half (1.5 h) · Do

In `from_scratch/`: read `rag/llm.py` (JSON parsing + retry with error feedback) and
`rag/grounding.py`. Run:
```bash
cd from_scratch && python main.py ask "How efficient are solar panels?" -v
```
Read `tests/test_pipeline.py::test_verify_catches_hallucinations`: four ways a citation can lie.

**Check:** why does requiring an *exact quote* change verification from a judgement call into a
string search? What can it still not catch? (A real quote that doesn't support the claim → the judge.)

### Block 2.2: Structured output with Pydantic (1.5 h) · Learn + Do

The framework version doesn't parse JSON by hand. `llm.with_structured_output(Schema)` converts a
Pydantic model into a tool definition / JSON schema and returns a validated object.

```python
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv; load_dotenv()
class Person(BaseModel):
    name: str
    age: int = Field(description="age in whole years")
llm = init_chat_model("claude-haiku-4-5-20251001", model_provider="anthropic", temperature=0)
print(llm.with_structured_output(Person).invoke("Ravi turned thirty last week."))
```

Now read `src/grounded_rag/schemas.py`. **The `Field(description=...)` texts are sent to the model:
they are prompt engineering.** Note how `Citation.quote` says "EXACT verbatim span (5-40 words)".

**Do:** in the REPL, call `with_structured_output(CitedAnswer)` yourself with two hand-written
sources and a question. Inspect `.sentences[0].citations[0].quote`. Is it really verbatim?

**Check:** what happens at the API level when you call `with_structured_output`? (Provider tool
calling / JSON-schema mode; the reply is validated by Pydantic, and a ValidationError is raised if it doesn't fit.)

### Block 2.3: The verifier (2 h) · Do (the heart of the project)

Read `src/grounded_rag/grounding.py` top to bottom, then `tests/test_grounding.py`.

| Check | Function | Catches |
|---|---|---|
| Has citations | `verify_answer` | unsupported filler sentences |
| Source id exists | `verify_answer` | invented `[S9]` |
| Quote in source | `locate_quote` | fabricated or paraphrased "quotes" |
| Numbers match | `numbers_in` | real quote + wrong number (the most common failure!) |
| Entailment | `judge_reports` | real quote that doesn't actually support the claim |

Details worth understanding:
- `_norm` flattens curly quotes, dashes, whitespace and case, so PDF extraction quirks don't cause false failures.
- `partial_ratio_alignment` finds the best-matching *substring*; threshold 90 tolerates a dropped
  word but not a rewritten sentence. Quotes under 3 words are rejected (too easy to match by accident).
- `render` renumbers sources `[1], [2]` in order of first use and **drops** unsupported sentences.
- `find_span` maps a quote back to character offsets in the original text for UI highlighting.

**Do:** write 3 new test cases in `tests/test_grounding.py`:
1. A quote with different capitalisation and line breaks → should pass.
2. A claim with "2024" whose quote says "2023" → should fail.
3. A one-word quote → should fail.

**Break it:** set `GRAG_QUOTE_MATCH_THRESHOLD=70` and write a test showing a paraphrase that now
wrongly passes. That's why the threshold is 90.

### Block 2.4: Prompts that make verification pass (1.5 h) · Do

Read `src/grounded_rag/prompts.py`, especially `GENERATE` and `JUDGE`. Then:

```bash
grag ask "What is the Betz limit?" -v
grag ask "How much does rooftop solar cost in Bengaluru?" -v    # not in docs → must abstain
grag ask "Why is grey hydrogen carbon-free?" -v                  # false premise → must correct it
```

Read the `--- verification ---` section. For each ✗, which rule failed?

**Break it:** in `GENERATE`, delete rule 2 (exact quotes). Ask 5 questions. What happens to the
grounding score? Restore the rule. **Write the before/after numbers in NOTES.md.**

**Check:** why is temperature 0? Why does the judge see *only* the quotes and not the full chunk?
(Deterministic output; if the judge saw the whole chunk, it could approve claims the quotes don't support.)

### Block 2.5: The repair loop (1 h) · Learn

In `graph/nodes.py`, find how `verify` produces `feedback` via `failure_feedback`, and how
`generate` puts it in the prompt. Then read `tests/test_graph.py::test_repair_loop_fixes_bad_quote`
and `test_unfixable_claims_are_dropped`.

**Check:** why feed back the *specific* failure ("quote not found in S3, best match 62%") instead of
just "try again"? Why keep `best` across attempts instead of the latest?

### Block 2.6: Local vs API models ⭐ (1 h)

Switch to Ollama (`GRAG_LLM_PROVIDER=ollama`), ask the same 5 questions, compare grounding scores
and repair-loop counts with `-v`. Small models paraphrase inside quotes much more. **This is the
strongest argument for verification you can make in an interview:** the system stays honest even
with a weak model, because unverified sentences never reach the user.

### Day 2 checkpoint ✅
- [ ] You can name all 5 verification checks and what each catches.
- [ ] 3 new passing tests in `test_grounding.py`.
- [ ] You've seen an abstention, a false-premise correction, and a repair in `-v` output.
- [ ] Before/after numbers for the "delete the exact-quote rule" experiment.

---

# DAY 3: The Agent (LangGraph), Product, Evaluation

> Mindset: a fixed pipeline (`retrieve → generate`) fails on multi-part questions and gives up
> too easily. A graph with **decisions** and **loops** fixes both, and evaluation tells you whether
> each piece earns its cost.

### Block 3.1: LangGraph fundamentals (2 h) · Learn + Do

Core ideas (details in `docs/CONCEPTS.md` §6):

| Concept | Meaning |
|---|---|
| `StateGraph(State)` | a graph whose nodes share one typed state dict |
| node | function `state -> partial update dict` |
| reducer | how an update merges: replace (default), `operator.add`, `add_messages` |
| edge / conditional edge | fixed next step / a function that picks the next step |
| `Send` | fan-out: run a node N times **in parallel** with different inputs (map-reduce) |
| checkpointer + `thread_id` | saves state after every step → memory, resume, time-travel |
| `stream_mode="updates"` | yields each node's output as it finishes |
| `RetryPolicy` | automatic retry of a node on transient errors |

**Do:** save as `toy_graph.py` and run it. It's the same fan-out/fan-in pattern as our `plan → retrieve → grade`:
```python
import operator
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langgraph.checkpoint.memory import InMemorySaver

class State(TypedDict):
    topics: list[str]
    facts: Annotated[list[str], operator.add]   # reducer: parallel branches append

def plan(state):      return {"topics": ["solar", "wind", "hydrogen"]}
def research(inp):    return {"facts": [f"a fact about {inp['topic']}"]}
def fan_out(state):   return [Send("research", {"topic": t}) for t in state["topics"]]
def summarize(state): return {"facts": [f"summary of {len(state['facts'])} facts"]}

g = StateGraph(State)
g.add_node("plan", plan); g.add_node("research", research); g.add_node("summarize", summarize)
g.add_edge(START, "plan")
g.add_conditional_edges("plan", fan_out, ["research"])
g.add_edge("research", "summarize")
g.add_edge("summarize", END)
app = g.compile(checkpointer=InMemorySaver())
print(app.invoke({"topics": [], "facts": []}, {"configurable": {"thread_id": "demo"}}))
for step in app.stream({"topics": [], "facts": []}, {"configurable": {"thread_id": "2"}}, stream_mode="updates"):
    print(step)
```

**Exercises:**
1. Remove the `operator.add` reducer (make `facts: list[str]`). Run it. What error / result, and why?
2. Add a conditional edge after `summarize` that loops back to `plan` once, using a counter in state.
3. Print `app.get_graph().draw_mermaid()` and paste it into mermaid.live.

**Check:** why does `summarize` run once and not three times, even though three `research` branches
point at it? (LangGraph runs in supersteps: all parallel branches finish, *then* the next step runs once.)

### Block 3.2: Our agent graph, node by node (2 h) · Do

```bash
grag graph          # Mermaid → paste into mermaid.live, keep it open
```

Read `graph/state.py`, `graph/builder.py`, then `graph/nodes.py` **in execution order**:

1. `contextualize`: rewrites follow-ups; **resets per-turn fields** (the checkpointer keeps the last
   turn's state!). Note `Overwrite([])` to bypass the `operator.add` reducer.
2. `route`: skip retrieval for "hi" / "thanks". When the router errors, default to retrieving.
3. `plan`: query decomposition; the original question is always query #1.
4. `fan_out` → `retrieve` ×N via `Send`, in parallel.
5. `grade`: fan-in, dedupe by `chunk_id`, LLM relevance filter (Corrective RAG), assign `S1, S2…`.
6. `after_grade`: generate / rewrite / abstain.
7. `generate` → `verify` → `after_verify`: finalize / repair / re-search if the model says evidence is missing.
8. `finalize` or `abstain`.

Note `RECOVERABLE` at the top of `nodes.py`: malformed model output falls back gracefully, but auth and
network errors are **not** swallowed. (A silent "0% grounding" on a bad API key is a nightmare to debug.)

**Do:** run these with `-v` and match every trace line to a node in the Mermaid diagram:
```bash
grag ask "Compare the round-trip efficiency of lithium-ion batteries and hydrogen storage" -v
grag ask "hello!" -v
grag ask "What is the capacity factor of nuclear plants?" -v
```

**Check:** draw (on paper) the path each of those three questions took through the graph.

### Block 3.3: Memory & multi-turn (45 min) · Do

```bash
grag chat -v
you> What is the capacity factor of offshore wind?
you> and onshore?                       # watch: contextualize rewrites it into a standalone question
you> why is the first one higher?
```

Read `tests/test_graph.py::test_memory_contextualizes_follow_up`.

**Check:** why rewrite the question instead of just passing the chat history to `generate`?
(Retrieval needs a searchable query: "and onshore?" retrieves nothing useful.)

⭐ Make memory survive restarts: `pip install langgraph-checkpoint-sqlite`, then pass
`SqliteSaver.from_conn_string(...)` as the checkpointer in `RAGService`. Check the package docs
for the exact API.

### Block 3.4: Serving it: API + UI (1.5 h) · Do

```bash
uvicorn grounded_rag.api:app --reload        # open http://localhost:8000/docs and try every endpoint
curl -N localhost:8000/chat/stream -H 'content-type: application/json' -d '{"question":"What is the duck curve?"}'
streamlit run app/streamlit_app.py
```

Read `api.py` (lifespan, `run_in_threadpool`, SSE format) and `app/streamlit_app.py`
(`st.cache_resource`, `st.status` for live steps, `find_span` highlighting). In the UI, upload one of
your own PDFs, ask a question, expand a citation and see the quote highlighted.

⭐ Run it in Docker: `docker compose up --build`.

**Check:** why does the API run the graph in a thread pool? (`invoke` is blocking; running it
directly in an async endpoint would block the event loop for every other user.)

### Block 3.5: Evaluation & ablation study (2.5 h) · Do (makes it a portfolio project)

Read the docstring of `eval/run_eval.py`: every metric and what it measures.

1. Write **15–25 questions about YOUR documents** in a new JSONL (same format as `eval/dataset.jsonl`).
   Include: single facts, numbers, comparisons across files, 3+ unanswerable, 1–2 false premises.
2. Run the ablation:
   ```bash
   python eval/run_eval.py --compare              # naive_rag, hybrid_only, verified, full
   cat eval/results/summary.md
   ```
3. Run single-variable experiments:
   ```bash
   python eval/run_eval.py --set top_k=3 --tag k3
   python eval/run_eval.py --set top_k=10 --tag k10
   python eval/run_eval.py --set use_judge=false --tag nojudge
   ```
4. Open `eval/results/full.jsonl`, find the **3 worst questions**, and diagnose each: retrieval miss?
   Grader dropped the right chunk? Generation? Verifier too strict? Fix one and re-run.

Expected pattern (yours will differ, and **that's the point**): `naive_rag` shows high "grounding"
(nothing is verified) but worse correctness and fake abstentions; `verified` drops unsupported claims;
`full` wins on comparisons and multi-hop questions thanks to parallel sub-queries, at more latency and cost.

⭐ Try RAGAS or LangSmith's evaluation features on the same dataset and compare with the custom metrics.
Both evolve quickly, so follow their current docs.

**Check:** why is the LLM judge for correctness itself a source of error? How would you validate it?
(Hand-label 20 answers and measure agreement.)

### Block 3.6: Observability & write-up (1 h)

With LangSmith enabled, open one full trace: see every node, prompt, token count, and latency.
Find the slowest node, then think about how you'd speed it up (smaller model for grading? skip judge
when all quotes are exact?).

Write your `README`/blog post using the template below.

### Day 3 checkpoint ✅
- [ ] You can draw the agent graph from memory and explain every edge.
- [ ] Multi-turn chat works; you've seen a follow-up rewritten.
- [ ] API and UI both run; you've uploaded your own document through the UI.
- [ ] An ablation table on YOUR data, plus a diagnosis of the worst 3 questions.

---

## After the 3 days: stretch goals (ordered by learning value)

1. **Web-search fallback (full CRAG):** when `grade` finds nothing, search the web (e.g. Tavily or
   another search API), mark those sources as external, and verify them the same way.
2. **Human-in-the-loop:** use LangGraph `interrupt()` to ask the user to confirm sub-queries before searching.
3. **Better chunking:** semantic chunking, or parent-document retrieval (search small chunks,
   cite/show their larger parent).
4. **Tables & scanned PDFs:** use a layout-aware parser or OCR; keep tables as Markdown.
5. **NLI verifier:** replace the LLM judge with a small NLI cross-encoder, then compare cost and agreement.
6. **Answer caching** keyed on (normalised question, index version).
7. **Metadata filters:** "only search 2024 reports" → pass filters to Chroma.
8. **Streaming tokens:** stream the final answer token-by-token (`stream_mode="messages"`).
9. **Prompt versioning** in LangSmith, and a regression eval in CI (GitHub Actions running `pytest` + a small eval).
10. **Deploy:** container on Render / Fly.io / a VM; Chroma volume; auth on the API.

## Portfolio template

```
# <Project name>: RAG agent with verified citations
Problem: LLM answers over private docs can't be trusted; "citations" are often decorative.
Approach: hybrid retrieval → LangGraph agent (decompose, parallel search, corrective grading)
          → structured answers with verbatim quotes → code-level verification + LLM judge → repair loop.
Results (on N questions about <your corpus>):
  | variant | correctness | grounding | citation precision | abstain_ok | latency |
  ...
What I learned: <3 surprising findings from your experiments>
Limitations & next steps: <honest list>
```

**Resume bullets (adapt the numbers to your results):**
- Built an agentic RAG system (LangChain, LangGraph, Chroma) with parallel query decomposition, corrective
  retrieval and a self-repair loop; every answer sentence is verified against verbatim source quotes.
- Designed a citation verifier (fuzzy quote matching, numeric consistency, LLM entailment judge) that
  removed X% of unsupported claims; evaluated via ablations on a custom N-question benchmark.
- Shipped FastAPI (SSE streaming) + Streamlit UI with evidence highlighting; 28 offline tests using a scripted LLM.

## Daily time budget summary

| Day | Blocks | ≈ Hours |
|---|---|---|
| 0 | setup | 1 |
| 1 | 1.1 concepts · 1.2 from scratch · 1.3 LangChain core · 1.4 ingestion · 1.5 hybrid | 8.5 |
| 2 | 2.1 scratch gen · 2.2 structured output · 2.3 verifier · 2.4 prompts · 2.5 repair · 2.6⭐ local | 8.5 |
| 3 | 3.1 LangGraph · 3.2 agent · 3.3 memory · 3.4 serving · 3.5 eval · 3.6 write-up | 9.75 |
