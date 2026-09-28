"""
FastAPI server.

    uvicorn grounded_rag.api:app --reload
    open http://localhost:8000/docs   (interactive Swagger UI)

Endpoints
    GET  /health
    GET  /sources                     list indexed files
    POST /ingest   (multipart files)  upload & index documents
    DELETE /sources/{name}
    POST /search   {query, k, mode}   retrieval only (debugging)
    POST /chat     {question, thread_id?}           -> full grounded answer
    POST /chat/stream {question, thread_id?}        -> Server-Sent Events: one event per graph node
    GET  /threads/{thread_id}/history
"""
from __future__ import annotations

import json
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .documents import SUPPORTED
from .service import RAGService

_service: RAGService | None = None


def get_service() -> RAGService:
    global _service
    if _service is None:
        load_dotenv()
        _service = RAGService()
    return _service


def set_service(svc: RAGService) -> None:
    """Used by tests to inject a service with a fake LLM."""
    global _service
    _service = svc


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_service()           # warm up models & index once at startup
    yield


app = FastAPI(title="Grounded RAG Agent", version="1.0.0", lifespan=lifespan)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    thread_id: str | None = None


class SearchRequest(BaseModel):
    query: str
    k: int = 5
    mode: str = "hybrid"


@app.get("/health")
def health():
    svc = get_service()
    return {"status": "ok", "chunks": len(svc.index.chunks), "llm": f"{svc.s.llm_provider}:{svc.s.llm_model}"}


@app.get("/sources")
def sources():
    return get_service().index.sources()


@app.post("/ingest")
async def ingest(files: list[UploadFile] = File(...)):
    svc = get_service()
    tmp = Path(tempfile.mkdtemp())
    try:
        for f in files:
            name = Path(f.filename or "upload").name
            if Path(name).suffix.lower() not in SUPPORTED:
                raise HTTPException(400, f"Unsupported file type: {name}")
            with (tmp / name).open("wb") as out:
                shutil.copyfileobj(f.file, out)
        return await run_in_threadpool(svc.ingest, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@app.delete("/sources/{name}")
def delete_source(name: str):
    n = get_service().delete_source(name)
    if not n:
        raise HTTPException(404, "source not found")
    return {"deleted_chunks": n}


@app.post("/search")
def search(req: SearchRequest):
    docs = get_service().search(req.query, req.k, req.mode)
    return [{"text": d.page_content, "metadata": d.metadata} for d in docs]


@app.post("/chat")
async def chat(req: ChatRequest):
    return await run_in_threadpool(get_service().ask, req.question, req.thread_id)


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    def events():
        try:
            for ev in get_service().stream(req.question, req.thread_id):
                yield f"event: {ev['type']}\ndata: {json.dumps(ev, default=str)}\n\n"
        except Exception as e:  # noqa: BLE001
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/threads/{thread_id}/history")
def history(thread_id: str):
    return get_service().history(thread_id)
