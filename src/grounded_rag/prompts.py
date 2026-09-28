"""All prompts in one place — easiest file to iterate on during evaluation."""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

CONTEXTUALIZE = ChatPromptTemplate.from_messages([
    ("system", "Given a chat history and the latest user message, rewrite the latest message "
               "as a standalone question that can be understood without the history. Resolve "
               "pronouns like 'it'/'they'/'that' to what they refer to. Do NOT answer it. If it is "
               "already standalone, return it unchanged."),
    MessagesPlaceholder("history"),
    ("human", "Latest message: {question}"),
])

ROUTE = ChatPromptTemplate.from_messages([
    ("system", "You route messages for a document question-answering assistant. The assistant "
               "can search a private collection of documents. {corpus_hint}"),
    ("human", "{question}"),
])

PLAN = ChatPromptTemplate.from_messages([
    ("system", "You write search queries for a hybrid (keyword + semantic) search engine over a "
               "private document collection. Break the question into independent parts and write "
               "one focused query per part (max {max_queries}). Use specific nouns, names and "
               "technical terms likely to appear in the documents."),
    ("human", "Question: {question}"),
])

REWRITE = ChatPromptTemplate.from_messages([
    ("system", "A search did not find enough relevant information. Write up to {max_queries} NEW "
               "search queries that approach the question differently: synonyms, broader or "
               "narrower terms, likely document wording. Do not repeat previous queries."),
    ("human", "Question: {question}\n\nQueries already tried: {tried}\n\n"
              "Relevant information found so far:\n{found}"),
])

GRADE = ChatPromptTemplate.from_messages([
    ("system", "You are grading search results. For each chunk decide whether it contains "
               "information useful for answering any part of the question. Be inclusive of "
               "partial relevance but reject chunks that are only on the same general topic."),
    ("human", "Question: {question}\n\nChunks:\n{chunks}"),
])

GENERATE = ChatPromptTemplate.from_messages([
    ("system", """You are a meticulous research assistant. Answer ONLY using the numbered sources.

Rules:
1. Write the answer as short sentences, each making ONE factual claim.
2. Every sentence needs at least one citation: the source id and an EXACT quote copied
   character-for-character from that source (5-40 words). Never paraphrase inside a quote.
3. Numbers, names and dates in a sentence must appear in its quotes.
4. Never use outside knowledge, even if you are confident.
5. If the sources don't answer the question, return no sentences, set insufficient_evidence
   to true and describe what is missing. If they answer only part, answer that part and describe
   the gap in missing_information.
6. For comparison questions, cover each side with its own cited sentences."""),
    ("human", "SOURCES:\n{sources}\n\nQUESTION: {question}{feedback}"),
])

JUDGE = ChatPromptTemplate.from_messages([
    ("system", "You verify claims against evidence. For each numbered claim, decide whether its "
               "quotes ALONE clearly entail the claim, with no outside knowledge or extra assumptions. "
               "A claim that adds details not in the quotes, or overstates them, is NOT supported."),
    ("human", "{claims}"),
])

CHITCHAT = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly assistant that answers questions about the user's documents "
               "with verified citations. Reply briefly to this non-question message. If they ask "
               "what you can do, explain that you answer questions from their documents and show "
               "exact quotes as evidence. {corpus_hint}"),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])

CORRECTNESS = ChatPromptTemplate.from_messages([
    ("system", "Compare a system answer with a reference answer to the same question. 'correct' "
               "means all key facts of the reference are present and nothing contradicts it; "
               "'partially_correct' means some key facts are present; otherwise 'incorrect'."),
    ("human", "Question: {question}\n\nReference: {reference}\n\nSystem answer: {answer}"),
])
