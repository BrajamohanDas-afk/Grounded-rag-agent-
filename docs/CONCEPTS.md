# Concepts, Design Decisions & Interview Prep

## 1. Why RAG exists

LLMs know what was in their training data, frozen at a cutoff, and they generate *plausible* text,
not *verified* text. For questions about private, recent or niche information you have three options:

| Option | Good at | Bad at |
|---|---|---|
| Put everything in the prompt | tiny corpora | context limits, cost per query, accuracy drops for facts buried mid-context ("lost in the middle") |
| Fine-tune | style, format, domain vocabulary | injecting *facts* reliably; updating knowledge; saying *where* a fact came from |
| **RAG**: retrieve relevant passages at question time | fresh, private, attributable knowledge | depends entirely on retrieval quality |

RAG turns "remember the answer" into "read the answer from these passages", which also makes
**attribution** possible: you know exactly which passages the model saw.

## 2. The pipeline in depth

**Loading & cleaning.** Most quality problems start here: PDF hyphenation, headers/footers on every
page, tables flattened into word soup, navigation text in HTML. Garbage chunks retrieve garbage.

**Chunking.** Retrieval works on chunks, not documents.
- Too small → a chunk lacks the context to be understood or to answer anything ("It increased by 20%": what did?).
- Too large → the embedding averages many topics (blurry), fewer chunks fit in the prompt, citations become vague.
- Typical: 300–1000 characters (or 100–300 tokens) with 10–20% overlap so facts on a boundary aren't cut.
- `RecursiveCharacterTextSplitter` tries separators in order (paragraph → line → sentence → word)
  so it breaks at the most natural boundary that fits.
- Advanced: semantic chunking (split where embedding similarity drops), parent-document retrieval
  (search small chunks, return their larger parents), late chunking.

**Embeddings.** A model maps text → vector (e.g. 384 dims for MiniLM, 768–3072 for larger models)
such that similar meanings are close. Similarity is usually cosine; with normalised vectors,
cosine = dot product. Embedding models are trained contrastively on (query, relevant passage) pairs.
Bi-encoder = query and passage embedded *separately*, which makes it fast because passages are pre-computed.

**Vector index.** Exact search compares the query with every vector (fine to ~100k chunks).
Beyond that, approximate nearest neighbour (ANN) indexes like **HNSW** (a navigable graph of vectors)
trade a little recall for huge speed. Chroma uses HNSW; we set `hnsw:space=cosine`.

**BM25 (lexical search).** Scores documents by query-term matches, weighted by rarity (IDF) and
saturated by term frequency and normalised by length. Excellent for exact tokens (codes, names,
acronyms, numbers) and needs no model. Blind to synonyms.

**Hybrid + Reciprocal Rank Fusion.** Run both; fuse *ranks*, not scores (BM25 scores are unbounded,
cosine is in [-1, 1], so they can't be added):
`RRF(d) = Σ_r  w_r / (k + rank_r(d))`, with k = 60. Documents ranked well by *both* rise to the top.

**Reranking.** A cross-encoder reads `[query, passage]` together through a transformer and outputs
relevance. It's far more accurate than bi-encoder similarity (it sees interactions between words)
but too slow to run on the whole corpus, so: retrieve 20–50 cheaply → rerank → keep top 5.

## 3. Grounding & citations

**Levels of attribution (weak → strong):**
1. "Here are some sources" listed after the answer. No link between claims and sources.
2. Inline `[n]` markers the model chooses. Can point to the wrong source; unverifiable.
3. `[n]` + **verbatim quote** per claim. Verifiable mechanically: is the quote in the source?
4. Level 3 + **entailment check**: do the quotes actually support the claim?  ← this project
5. Level 4 + source reliability (is the source itself correct?). Out of scope for RAG; needs curation.

**How citations lie (all tested in this repo):** invented source id; fabricated quote; paraphrase
presented as a quote; real quote + wrong number; real quote that doesn't support the claim;
correct claim cited to the wrong source; uncited filler sentences.

**Why exact quotes work.** They convert "is claim X supported by chunk Y?" (a fuzzy judgement)
into "is string Q a substring of Y?" (a deterministic check), leaving only "does Q entail X?" for
the judge, which is a much narrower and more reliable question than judging against a whole chunk.

**Why verification beats more prompting.** Prompting shifts the probability of good behaviour;
verification *guarantees* a property for everything that reaches the user. With a weak model you
get fewer verified sentences, not more lies.

## 4. Agentic RAG patterns

| Pattern | Idea | Here |
|---|---|---|
| Query rewriting / contextualisation | turn "and onshore?" into a searchable standalone question | `contextualize` |
| Routing / Adaptive RAG | don't retrieve for "hello"; pick a strategy per question | `route` |
| Query decomposition / multi-query | one search per sub-question; union the results | `plan` + `Send` |
| Corrective RAG (CRAG) | grade retrieved docs; if poor, re-search (or search the web) | `grade`, `rewrite` |
| Self-RAG / self-reflection | model critiques its own output and retries | `verify` → `generate` with feedback |
| Abstention | refuse when evidence is missing | `abstain`, `insufficient_evidence` |

**Workflow vs. agent.** This graph is a *workflow with LLM decisions*: the paths are fixed, and the
LLM chooses among them. A fully autonomous tool-calling agent (LLM decides every action in a loop)
is more flexible but less predictable, harder to test, and more expensive. For RAG, constrained
graphs usually win on reliability; that's a deliberate design choice you should be able to defend.

## 5. Structured output

`llm.with_structured_output(PydanticModel)` sends the model's JSON schema to the provider (via
tool calling or native JSON-schema mode) and validates the reply into a Pydantic object. Benefits:
no regex parsing, typed fields, validation errors instead of silent garbage. Field descriptions
are part of the prompt. Failure modes: small local models may produce invalid structures, so every
structured call in `nodes.py` has a fallback (`RECOVERABLE` errors only).

## 6. LangGraph deep dive

- **State** is a `TypedDict`. Nodes return *partial* updates, and LangGraph merges them.
- **Reducers** define merging: default = overwrite; `operator.add` = concatenate (needed when
  parallel nodes write the same key in one step, otherwise you get `InvalidUpdateError`);
  `add_messages` = append chat messages, dedupe by id. `Overwrite(value)` bypasses a reducer once
  (we use it to reset `candidates` and `trace` each turn).
- **Execution model: supersteps** (Pregel/BSP). All nodes triggered in a step run (possibly in
  parallel); their updates are applied; then the next step's nodes run. That's why `grade` runs once
  after N parallel `retrieve`s.
- **`Send(node, payload)`**: returned from a conditional edge for dynamic fan-out; each branch
  receives its own private input (`RetrieveInput`), not the whole state.
- **Checkpointer** saves state after every superstep under a `thread_id`: conversation memory,
  crash recovery, `get_state()` inspection, human-in-the-loop (`interrupt`), time travel.
  `InMemorySaver` for dev; SQLite/Postgres savers for production.
- **Per-turn state reset:** because state persists per thread, turn-specific fields must be reset
  at the start of each turn (`contextualize` does this). Forgetting this is a classic bug: evidence
  from the previous question leaking into the next answer.
- **Streaming modes:** `updates` (per-node outputs; our UI), `values` (full state), `messages`
  (LLM tokens), `custom`.
- **RetryPolicy** per node for transient failures; `recursion_limit` caps runaway loops (every loop
  in our graph also has an explicit budget).

## 7. Evaluation

Separate **retrieval** from **generation**, or you can't tell which one failed.

| Metric | Question it answers |
|---|---|
| recall@k, MRR | did the right chunk come back, and how high? |
| context precision | how much noise did we pass to the model? |
| grounding (faithfulness) | is every claim supported by the provided context? |
| citation precision / recall | are citations pointing to the right sources, and are all needed sources cited? |
| correctness | does the answer match a reference? (LLM judge: validate it on hand labels!) |
| abstention accuracy | does it refuse when it should, and *only* then? |
| latency, cost | is the improvement worth it? |

**Ablations** (turn one feature off at a time) are how you prove each component earns its cost.
Frameworks like RAGAS and LangSmith evaluators implement many of these metrics; knowing how to
compute them yourself makes those tools much easier to trust and debug.

## 8. Production concerns

- **Prompt injection via documents.** A document can contain "ignore previous instructions…".
  Mitigations: treat sources as data (clearly delimited), never give the RAG model dangerous tools,
  and verification limits the damage since every claim must quote a source. Injected
  *instructions* can still shape tone and selection, so sanitise untrusted corpora.
- **Access control.** Filter retrieval by user permissions (metadata filters), never after generation.
- **Cost/latency.** Our full graph makes ~6 LLM calls per question (contextualize, route, plan,
  grade, generate, judge). Options: smaller model for routing/grading, skip judge when all quotes
  are exact matches and claims are short, cache.
- **Index freshness.** Deterministic chunk ids allow incremental updates and deletes.
- **Observability.** Trace every step (LangSmith/OpenTelemetry); log grounding scores; alert on drops.

## 9. Design decisions in this repo (be ready to defend them)

| Decision | Why | Alternative |
|---|---|---|
| Own loaders instead of `langchain-community` | that package is being sunset; loaders are ~50 lines | `pymupdf4llm`, `unstructured`, Docling for complex PDFs |
| JSONL chunk store alongside Chroma | BM25 needs all texts; human-inspectable; rebuild any vector store | store texts in Chroma only and read them back |
| Custom BM25 retriever | avoid deprecated package; shows the `BaseRetriever` interface | OpenSearch/Elasticsearch in production |
| Sentence-level claims | verification granularity = claim granularity | paragraph-level (coarser, easier for model) |
| Drop unsupported sentences (not flag) | user only sees verified content | flag with ⚠ (shown in `-v` / UI verification panel) |
| Workflow graph, not free agent | predictable, testable, bounded cost | ReAct tool-calling agent |
| Scripted fake LLM in tests | deterministic, free, tests every branch | recorded real responses (VCR-style) |

## 10. Interview questions (with short answers)

1. **What is RAG and why use it over fine-tuning?** Retrieve passages at query time and generate from
   them. Fresh and private knowledge without retraining, plus attribution. Fine-tuning is for behaviour/style.
2. **How do you choose chunk size?** Empirically with a retrieval eval; balance context vs. precision;
   match the typical answer span; use overlap.
3. **Dense vs sparse retrieval?** Dense = semantic, handles paraphrase, weak on exact tokens. Sparse
   (BM25) = exact tokens, no synonyms. Hybrid gets both.
4. **Why RRF instead of adding scores?** Scores live on incompatible scales; ranks don't.
5. **Bi-encoder vs cross-encoder?** Bi-encoder embeds separately (fast, precomputable) and is used for
   retrieval. Cross-encoder jointly encodes the pair (accurate, slow) and is used for reranking.
6. **How do you reduce hallucinations in RAG?** Better retrieval, instruct to use only context,
   allow abstention, structured citations with quotes, verify quotes, entailment check, repair loop, evaluate.
7. **How do you verify a citation?** Source id exists; quote is in the source (normalised/fuzzy);
   numbers consistent; judge confirms entailment.
8. **What's "lost in the middle"?** Models attend less to information in the middle of long contexts,
   so retrieve fewer, better chunks and rerank.
9. **Multi-hop / comparison questions?** Decompose into sub-queries, retrieve each (in parallel), answer
   over the union.
10. **What is Corrective RAG?** Grade retrieved docs for relevance; if insufficient, rewrite the query or
    search another source.
11. **Explain LangGraph reducers.** Functions that merge a node's update into state; needed for parallel
    writes and message history.
12. **How does LangGraph memory work?** A checkpointer persists state per `thread_id` after each
    superstep; the next invoke resumes from it.
13. **What does `Send` do?** Dynamic fan-out: run a node once per payload in parallel (map-reduce).
14. **How do you evaluate a RAG system?** Separate retrieval (recall, MRR) and generation (faithfulness,
    correctness, citation precision), test abstention, run ablations, track cost and latency.
15. **How would you test LLM code deterministically?** Fake/scripted models at the interface boundary
    (as in `tests/conftest.py`), plus a small live eval set.
16. **LLM-as-judge pitfalls?** Bias (verbosity, position, self-preference), cost, variance. Validate against
    human labels, use a strict rubric, and give the judge only the necessary evidence.
17. **How would you scale to 10M chunks?** ANN index (HNSW/IVF-PQ), managed vector DB, a search engine
    for BM25, metadata filters, caching, async batching for embeddings, tiered retrieval.
18. **How do you handle document updates?** Deterministic chunk ids; delete-by-source then re-add;
    version the index; re-run the eval.
19. **Security risks?** Prompt injection in documents, data leakage across users (filter at retrieval),
    PII in logs/traces.
20. **What would you improve next?** (Pick from your own eval findings. Interviewers love evidence-driven answers.)

## 11. Glossary

**ANN**: approximate nearest neighbour search · **BM25**: lexical ranking function ·
**Checkpointer**: persists LangGraph state per thread · **Chunk**: retrievable text unit ·
**Cross-encoder**: model that scores a (query, passage) pair jointly · **Embedding**: vector
representation of text · **Entailment**: premise logically supports hypothesis · **Faithfulness /
grounding**: claims supported by provided context · **HNSW**: graph-based ANN index · **LCEL**:
LangChain Expression Language (`a | b | c`) · **MRR**: mean reciprocal rank · **Reducer**: state-merge
function · **RRF**: reciprocal rank fusion · **Runnable**: LangChain's universal invoke/stream/batch
interface · **Superstep**: one parallel execution round in LangGraph · **Thread**: one conversation's
persisted state
