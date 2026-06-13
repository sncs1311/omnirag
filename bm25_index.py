from rank_bm25 import BM25Okapi
import json
import os
import re
import numpy as np

BM25_INDEX_PATH = "./bm25_index.json"


# ── Tokenizer ─────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """
    Convert text to lowercase token list for BM25.
    Preserves numbers and alphanumeric sequences — critical for
    matching invoice numbers, function names, IDs, dates.
    """
    return re.findall(r'\b\w+\b', text.lower())


# ── BM25 Index class ──────────────────────────────────────────────────────

class BM25Index:
    """
    Wraps BM25Okapi with disk persistence.
    
    BM25Okapi itself has no save/load — we store the raw corpus
    as JSON and rebuild the BM25 object from it on load.
    """

    def __init__(self):
        self.corpus = []       # list of token lists — BM25's input
        self.chunks = []       # list of dicts — {text, filename, chunk_index}
        self.bm25 = None       # BM25Okapi object — rebuilt from corpus

    def add(self, chunks: list[str], metadatas: list[dict]):
        """
        Add new chunks to the index.
        Called during ingestion alongside ChromaDB storage.
        Both must receive the same chunks — they stay in sync.
        """
        for chunk, meta in zip(chunks, metadatas):
            tokens = tokenize(chunk)
            self.corpus.append(tokens)
            self.chunks.append({
                "text": chunk,
                "filename": meta.get("filename", "unknown"),
                "chunk_index": meta.get("chunk_index", 0)
            })

        # Rebuild BM25 after adding new documents
        # BM25Okapi precomputes IDF scores — must recalculate
        # when the corpus changes
        self.bm25 = BM25Okapi(self.corpus)
        self.save()

    def search(self, query: str, n_results: int = 10) -> list[dict]:
        """
        Search for query using BM25 keyword matching.
        Returns results with scores — ready for RRF merging.
        """
        if self.bm25 is None or len(self.chunks) == 0:
            return []

        query_tokens = tokenize(query)

        # get_scores returns a score for every chunk in the corpus
        scores = self.bm25.get_scores(query_tokens)

        # Sort by score descending — highest BM25 score first
        # argsort gives ascending indices, [::-1] reverses to descending
        ranked_indices = np.argsort(scores)[::-1]

        results = []
        n = min(n_results, len(self.chunks))

        for idx in ranked_indices[:n]:
            score = float(scores[idx])

            # Skip chunks with zero keyword overlap — pure noise
            if score == 0.0:
                continue

            results.append({
                "text": self.chunks[idx]["text"],
                "filename": self.chunks[idx]["filename"],
                "chunk_index": self.chunks[idx]["chunk_index"],
                "bm25_score": round(score, 4)
            })

        return results

    def save(self):
        """
        Persist index to disk as JSON.
        Called automatically after every add().
        """
        data = {
            "corpus": self.corpus,
            "chunks": self.chunks
        }
        with open(BM25_INDEX_PATH, "w") as f:
            json.dump(data, f)

    def load(self):
        """
        Load index from disk and rebuild BM25 object.
        Call this once when the app starts.
        """
        if not os.path.exists(BM25_INDEX_PATH):
            # No index yet — first run, no documents ingested
            return

        with open(BM25_INDEX_PATH, "r") as f:
            data = json.load(f)

        self.corpus = data["corpus"]
        self.chunks = data["chunks"]

        if self.corpus:
            # Rebuild the BM25 object from the loaded corpus
            self.bm25 = BM25Okapi(self.corpus)


# ── Module-level singleton ────────────────────────────────────────────────
# One BM25 index shared across the entire app
# Loaded once at startup, updated on every ingestion

bm25_index = BM25Index()
bm25_index.load()