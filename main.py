from entity_graph import entity_graph
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from bm25_index import bm25_index
import shutil
import os
import uuid
import chromadb

from ingest import (
    ingest_pdf,
    ingest_structured,
    ingest_document,
    ingest_pptx,
    ingest_code,
    STRUCTURED_EXTENSIONS,
    DOCUMENT_EXTENSIONS,
    PPTX_EXTENSIONS,
    CODE_EXTENSIONS
)
from retriever import retrieve_with_routing
from generator import generate_answer
from conversation_manager import session_store, process_query, record_turn

client = chromadb.PersistentClient(path="./chroma_store")
collection = client.get_or_create_collection("documents")

app = FastAPI(title="OmniRAG", version="0.2.0 — Conversational")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("uploads", exist_ok=True)

# All supported extensions
PDF_EXTENSIONS = {'.pdf'}
STRUCTURED_EXTS = set(STRUCTURED_EXTENSIONS.keys())
DOCUMENT_EXTS = set(DOCUMENT_EXTENSIONS.keys())
PPTX_EXTS = PPTX_EXTENSIONS
CODE_EXTS = CODE_EXTENSIONS

@app.get("/")
def health_check():
    return {"status": "OmniRAG is running", "version": "0.2.0"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()

    all_supported = (PDF_EXTENSIONS | STRUCTURED_EXTS | DOCUMENT_EXTS |
                     PPTX_EXTS | CODE_EXTS)

    if ext not in all_supported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: {sorted(all_supported)}"
        )

    # Auto-clear previous data before every upload (single-document mode)
    global collection, client
    try:
        client.delete_collection("documents")
        collection = client.get_or_create_collection("documents")

        import retriever, ingest
        retriever.collection = collection
        ingest.collection = collection
    except Exception:
        pass

    from bm25_index import BM25_INDEX_PATH
    from entity_graph import GRAPH_PATH
    from structured_parser import SQLITE_DB_PATH

    for path in [BM25_INDEX_PATH, GRAPH_PATH, SQLITE_DB_PATH]:
        if os.path.exists(path):
            os.remove(path)

    bm25_index.corpus = []
    bm25_index.chunks = []
    bm25_index.bm25 = None
    entity_graph.graph.clear()

    file_path = f"uploads/{filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        if ext == '.pdf':
            result = ingest_pdf(file_path, filename)
        elif ext in STRUCTURED_EXTS:
            result = ingest_structured(file_path, filename)
        elif ext in PPTX_EXTS:
            result = ingest_pptx(file_path, filename)
        elif ext in CODE_EXTS:
            result = ingest_code(file_path, filename)
        else:
            result = ingest_document(file_path, filename)
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

    return result


# ── Query models ──────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    n_results: int = 5
    session_id: str | None = None   # optional; auto-created if missing


class SessionResponse(BaseModel):
    session_id: str


# ── Query endpoint ────────────────────────────────────────────────────────

@app.post("/query")
def query_documents(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Resolve or create session
    session_id = request.session_id or str(uuid.uuid4())

    # Rewrite query (typo fix + follow-up resolution) and fetch history
    rewritten_query, history = process_query(request.question, session_id)

    # Retrieve and generate — pass both original + rewritten query
    result = retrieve_with_routing(
        query=request.question,          # original — used for display & history
        n_results=request.n_results,
        rewritten_query=rewritten_query, # used for actual retrieval
        history=history                  # passed to generator for context
    )

    # Save this turn to history (original question + answer)
    record_turn(session_id, request.question, result.get("answer", ""))

    # Always echo back the session_id so the frontend can send it next time
    result["session_id"] = session_id

    # Show what the query was rewritten to (useful for debugging / UI display)
    if rewritten_query != request.question:
        result["rewritten_query"] = rewritten_query

    return result


# ── Session management endpoints ──────────────────────────────────────────

@app.post("/session/new", response_model=SessionResponse)
def new_session():
    """Create a fresh conversation session. Returns the session ID."""
    sid = session_store.new_session()
    return {"session_id": sid}


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    """Clear conversation history for a session (e.g. 'Start over')."""
    session_store.clear(session_id)
    return {"message": f"Session {session_id} cleared."}


# ── Existing endpoints (unchanged) ────────────────────────────────────────

@app.get("/graph")
def get_graph():
    return entity_graph.get_entity_summary()


@app.delete("/clear/documents")
def clear_documents_only():
    from bm25_index import BM25_INDEX_PATH
    global collection, client

    try:
        client.delete_collection("documents")
        collection = client.get_or_create_collection("documents")
        import retriever, ingest
        retriever.collection = collection
        ingest.collection = collection
    except Exception as e:
        return {"error": f"ChromaDB clear failed: {e}"}

    if os.path.exists(BM25_INDEX_PATH):
        os.remove(BM25_INDEX_PATH)
    bm25_index.corpus = []
    bm25_index.chunks = []
    bm25_index.bm25 = None

    return {"message": "Document data cleared. Structured data untouched."}


@app.delete("/clear")
def clear_all_data():
    from bm25_index import BM25_INDEX_PATH
    from entity_graph import GRAPH_PATH
    from structured_parser import SQLITE_DB_PATH

    global collection, client
    results = {}

    try:
        client.delete_collection("documents")
        collection = client.get_or_create_collection("documents")
        import retriever, ingest
        retriever.collection = collection
        ingest.collection = collection
        results["chromadb"] = "cleared"
    except Exception as e:
        results["chromadb"] = f"error: {e}"

    try:
        if os.path.exists(BM25_INDEX_PATH):
            os.remove(BM25_INDEX_PATH)
        bm25_index.corpus = []
        bm25_index.chunks = []
        bm25_index.bm25 = None
        results["bm25"] = "cleared"
    except Exception as e:
        results["bm25"] = f"error: {e}"

    try:
        if os.path.exists(SQLITE_DB_PATH):
            os.remove(SQLITE_DB_PATH)
        results["sqlite"] = "cleared"
    except Exception as e:
        results["sqlite"] = f"error: {e}"

    try:
        if os.path.exists(GRAPH_PATH):
            os.remove(GRAPH_PATH)
        entity_graph.graph.clear()
        results["entity_graph"] = "cleared"
    except Exception as e:
        results["entity_graph"] = f"error: {e}"

    return {"message": "All data cleared.", "details": results}


@app.delete("/clear/structured")
def clear_structured_only():
    from structured_parser import SQLITE_DB_PATH
    if os.path.exists(SQLITE_DB_PATH):
        os.remove(SQLITE_DB_PATH)
        return {"message": "Structured data cleared. PDF/text data untouched."}
    return {"message": "No structured data found."}


@app.get("/debug/chunks")
def debug_chunks():
    results = collection.get(limit=5, include=["documents", "metadatas"])
    return {
        "total_chunks": collection.count(),
        "sample_chunks": [
            {"text": doc[:300], "metadata": meta}
            for doc, meta in zip(results["documents"], results["metadatas"])
        ]
    }