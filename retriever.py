from entity_graph import entity_graph
from corrective_filter import filter_chunks, compute_confidence
from sql_router import try_sql_route
from sentence_transformers import SentenceTransformer
import chromadb
import re
from bm25_index import bm25_index

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./chroma_store")
collection = client.get_or_create_collection(name="documents")


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────

def reciprocal_rank_fusion(
    vector_results: list[dict],
    bm25_results: list[dict],
    k: int = 60
) -> list[dict]:
    """
    Merge two ranked result lists without needing comparable scores.

    RRF formula: score += 1 / (k + rank)

    k=60 is the empirically optimal constant from the original paper
    (Cormack, Clarke, Buettcher 2009). It prevents very high-ranked
    results from dominating — smooths the influence of rank position.

    A chunk appearing at rank 1 in both lists scores higher than
    one appearing at rank 1 in one list and rank 50 in another.
    """
    rrf_scores = {}   # chunk_text → accumulated RRF score
    chunk_data = {}   # chunk_text → full chunk dict (for returning results)

    for rank, chunk in enumerate(vector_results):
        key = chunk["text"]
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (k + rank + 1)
        chunk_data[key] = chunk

    for rank, chunk in enumerate(bm25_results):
        key = chunk["text"]
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (k + rank + 1)
        if key not in chunk_data:
            chunk_data[key] = chunk

    sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)

    merged = []
    for key in sorted_keys:
        chunk = chunk_data[key].copy()
        chunk["rrf_score"] = round(rrf_scores[key], 5)
        merged.append(chunk)

    return merged


# ── Vector search ─────────────────────────────────────────────────────────

def vector_search(query: str, n_results: int = 10) -> list[dict]:
    """
    Dense embedding search via ChromaDB.
    Finds chunks that are semantically similar to the query.
    """
    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    if not results["documents"][0]:
        return []

    chunks = []
    for i, doc in enumerate(results["documents"][0]):
        chunks.append({
            "text": doc,
            "filename": results["metadatas"][0][i]["filename"],
            "chunk_index": results["metadatas"][0][i]["chunk_index"],
            "distance": results["distances"][0][i],
            "relevance_score": round(1 - results["distances"][0][i], 3)
        })

    return chunks


# ── Main retrieve function ────────────────────────────────────────────────

def retrieve(query: str, n_results: int = 5) -> list[dict]:
    """
    Hybrid retrieval: vector search + BM25, merged via RRF.
    Uses the rewritten query for best retrieval accuracy.
    """
    fetch_n = n_results * 2

    vec_results = vector_search(query, fetch_n)
    bm25_results = bm25_index.search(query, fetch_n)

    if not bm25_results:
        return vec_results[:n_results]

    if not vec_results:
        return bm25_results[:n_results]

    merged = reciprocal_rank_fusion(vec_results, bm25_results)
    return merged[:n_results]


SUMMARY_SIGNALS = [
    r'\bsummar(y|ise|ize)\b',
    r'\boverview\b',
    r'\bwhat.{0,20}(this|the) (file|document|doc|pdf|content|presentation|spreadsheet|code)s?.{0,15}(about|contain|cover|say|mean)\b',
    r'\bwhat.{0,10}(in|inside|on)\b.{0,15}(this|the) (file|document|doc)',
    r'\btell me about (this|the) (file|document)\b',
    r'\bdescribe (this|the) (file|document)\b',
    r'\bwhat is this\b',
]

def is_summary_query(query: str) -> bool:
    q = query.lower().strip().rstrip('?!.')
    return any(re.search(p, q) for p in SUMMARY_SIGNALS)


def fetch_chunks_by_ids(chunk_ids: list[str]) -> list[dict]:
    """Fetch specific chunks from ChromaDB by their IDs."""
    try:
        result = collection.get(ids=chunk_ids, include=["documents", "metadatas"])
        chunks = []
        for i, doc in enumerate(result["documents"]):
            chunks.append({
                "text": doc,
                "filename": result["metadatas"][i].get("filename", "unknown"),
                "chunk_index": result["metadatas"][i].get("chunk_index", 0),
                "distance": 0.2,  # neutral fallback for graph-sourced chunks
            })
        return chunks
    except Exception:
        return []


def retrieve_with_routing(
    query: str,
    n_results: int = 5,
    rewritten_query: str | None = None,
    history: list[dict] | None = None
) -> dict:
    """
    Main retrieval entrypoint.

    Parameters
    ----------
    query           : the original user question (used for display/history)
    n_results       : how many chunks to return
    rewritten_query : typo-fixed, context-resolved version from conversation_manager.
                      Used for actual retrieval. Falls back to `query` if not provided.
    history         : recent conversation turns passed through to the generator
                      so the LLM can answer follow-ups coherently.
    """
    # Use rewritten query for retrieval if available, otherwise original
    retrieval_query = rewritten_query if rewritten_query else query
    history = history or []

    # ── Structured data routes (use rewritten query for better SQL matching) ──
    from sql_router import describe_structured_data
    desc_result = describe_structured_data(retrieval_query)
    if desc_result:
        return desc_result

    sql_result = try_sql_route(retrieval_query)
    if sql_result:
        return sql_result

    if collection.count() == 0:
        return {
            "answer": "No documents have been uploaded yet.",
            "confidence": 0.0, "confidence_label": "no_support",
            "sources": [], "route": "none"
        }

    # ── Summary path ──────────────────────────────────────────────────────
    def run_summary_path():
        total = collection.count()
        sample_size = min(10, total)

        if total <= sample_size:
            sample = collection.get(include=["documents", "metadatas"])
        else:
            all_ids = collection.get(include=[])["ids"]
            step = max(1, len(all_ids) // sample_size)
            picked_ids = all_ids[::step][:sample_size]
            sample = collection.get(ids=picked_ids, include=["documents", "metadatas"])

        if not sample or not sample.get("documents"):
            return None

        combined = '\n\n---\n\n'.join(sample["documents"])
        filename = sample["metadatas"][0].get("filename", "document") if sample["metadatas"] else "document"

        fake_chunks = [{"text": combined, "filename": filename, "distance": 0.1}]

        from generator import generate_answer
        result = generate_answer(
            "Provide a clear, well-organised summary of what this document covers overall, "
            "based on the passages provided.",
            fake_chunks,
            history=history   # ← pass history even on summary path
        )
        result["route"] = "summary"
        return result

    if is_summary_query(retrieval_query):
        summary_result = run_summary_path()
        if summary_result:
            return summary_result
    # ── End summary ───────────────────────────────────────────────────────

    # ── Hybrid retrieval (rewritten query) ────────────────────────────────
    raw_chunks = retrieve(retrieval_query, n_results * 2)

    # Entity graph augmentation
    graph_chunk_ids = entity_graph.query_graph(retrieval_query)
    if graph_chunk_ids:
        graph_chunks = fetch_chunks_by_ids(graph_chunk_ids)
        seen_texts = {c["text"] for c in raw_chunks}
        for gc in graph_chunks:
            if gc["text"] not in seen_texts:
                raw_chunks.append(gc)
                seen_texts.add(gc["text"])

    filtered_chunks = filter_chunks(raw_chunks, retrieval_query)

    # ── Fallback to summary if retrieval found nothing ────────────────────
    if not filtered_chunks:
        summary_result = run_summary_path()
        if summary_result:
            summary_result["route"] = "summary_fallback"
            return summary_result

    # ── Abstention with fuzzy suggestions ────────────────────────────────
    if not filtered_chunks:
        from corrective_filter import find_closest_terms

        all_text = [c["text"] for c in raw_chunks]
        suggestions = find_closest_terms(retrieval_query, all_text)

        if suggestions:
            suggestion_list = ', '.join(f'"{s}"' for s in suggestions)
            answer = (
                f"I couldn't find an exact match for your question in the uploaded document. "
                f"Did you mean one of these terms that appear in the document: {suggestion_list}? "
                f"Try rephrasing your question using one of these, or let me know if you meant "
                f"something else entirely."
            )
        else:
            answer = "This information is not in the uploaded documents."

        return {
            "answer": answer,
            "confidence": 0.0,
            "confidence_label": "no_support",
            "sources": [],
            "suggestions": suggestions,
            "route": "vector_abstained"
        }

    # ── Generate answer with history ──────────────────────────────────────
    filtered_chunks = filtered_chunks[:n_results]
    from generator import generate_answer
    result = generate_answer(query, filtered_chunks, history=history)
    result["route"] = "vector+graph"
    return result