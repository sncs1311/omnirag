from sentence_transformers import SentenceTransformer
import numpy as np
import re

model = SentenceTransformer("all-MiniLM-L6-v2")

# Batch size for encoding — tune down if you hit OOM on very large docs
ENCODE_BATCH_SIZE = 512


# ── Utility: cosine similarity between two vectors ────────────────────────

def cosine_similarity(a: list, b: list) -> float:
    """
    Measures how similar two embedding vectors are.
    Returns 0.0 (completely different) to 1.0 (identical meaning).
    
    We use this to decide: did the topic just change between
    sentence[i] and sentence[i+1]?
    """
    a = np.array(a)
    b = np.array(b)
    
    dot_product = np.dot(a, b)
    magnitude = np.linalg.norm(a) * np.linalg.norm(b)
    
    # Guard against division by zero (empty vectors)
    if magnitude == 0:
        return 0.0
    
    return float(dot_product / magnitude)


# ── Step 1: Split text into sentences ────────────────────────────────────

def split_into_sentences(text: str) -> list[str]:
    """
    Split a block of text into individual sentences.
    Sentences are the unit we compare — not words, not paragraphs.
    
    A sentence carries one complete thought.
    That's the right granularity for detecting topic shifts.
    """
    # Split after sentence-ending punctuation followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    # Remove empty strings and very short fragments (likely noise)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    
    return sentences


# ── Step 2: Detect structural hard boundaries ─────────────────────────────

def find_hard_boundaries(text: str) -> list[str]:
    """
    Split text at structural markers BEFORE semantic analysis.
    These are always chunk boundaries regardless of semantic similarity.
    
    Hard boundaries:
    - [Page N] markers we added during ingestion
    - Lines that look like headings (ALL CAPS or Title Case short lines)
    - Code blocks (``` markers)
    - Double newlines (paragraph breaks)
    """
    # Split on page markers first
    sections = re.split(r'\[Page \d+\]', text)
    
    result = []
    for section in sections:
        if not section.strip():
            continue
            
        # Further split on double newlines (paragraph breaks)
        paragraphs = re.split(r'\n\s*\n', section)
        
        for para in paragraphs:
            para = para.strip()
            if para:
                result.append(para)
    
    return result


# ── Step 3: Core semantic chunking ────────────────────────────────────────

def semantic_chunk(text: str, threshold: float = 0.3, min_chunk_size: int = 2) -> list[str]:
    """
    Main chunking function. Two-pass approach:

    Pass 1: Split at hard structural boundaries (pages, paragraphs)
    Pass 2: Within each section, split further at semantic topic changes

    KEY OPTIMISATION: all sentences across ALL sections are encoded in a
    single model.encode() call (one GPU/CPU batch). Previously this was
    called once per section, causing hundreds of sequential encode calls
    on large documents — this was the primary bottleneck for 500+ page files.

    threshold: cosine similarity below this = new chunk
        Lower (0.2) = more, smaller chunks
        Higher (0.5) = fewer, larger chunks
        0.3 is a good default for academic/business documents

    min_chunk_size: minimum sentences per chunk
        Prevents single-sentence orphan chunks with no context
    """

    # Pass 1 — structural splits
    sections = find_hard_boundaries(text)

    # ── Collect all sentences from all sections up-front ─────────────────
    # We track which sentences belong to which section so we can stitch
    # them back together after the single batch encode.

    section_sentences: list[list[str]] = []   # sentences per section
    short_sections: list[tuple[int, str]] = [] # (section_idx, text) for tiny sections

    flat_sentences: list[str] = []             # all sentences, flattened
    flat_offsets: list[int] = []               # start index in flat_sentences per section

    for idx, section in enumerate(sections):
        sentences = split_into_sentences(section)

        if len(sentences) <= min_chunk_size:
            # Too short for semantic analysis — keep as-is
            short_sections.append((idx, section.strip()))
            section_sentences.append([])        # placeholder to keep indices aligned
            flat_offsets.append(len(flat_sentences))
        else:
            section_sentences.append(sentences)
            flat_offsets.append(len(flat_sentences))
            flat_sentences.extend(sentences)

    # ── Single batch encode — the key performance fix ─────────────────────
    # All sentences from the entire document encoded in one call.
    # batch_size controls memory; 512 works well up to ~2000 pages.
    all_embeddings: list = []
    if flat_sentences:
        all_embeddings = model.encode(
            flat_sentences,
            batch_size=ENCODE_BATCH_SIZE,
            show_progress_bar=False,
        ).tolist()

    # ── Pass 2 — semantic splits using the pre-computed embeddings ────────
    all_chunks: list[str] = []

    # Reconstruct a section_idx → (is_short, text_or_sentences) map
    short_section_map = {idx: txt for idx, txt in short_sections}

    for idx, section in enumerate(sections):
        if idx in short_section_map:
            if short_section_map[idx]:
                all_chunks.append(short_section_map[idx])
            continue

        sentences = section_sentences[idx]
        offset = flat_offsets[idx]
        embeddings = all_embeddings[offset: offset + len(sentences)]

        current_chunk = [sentences[0]]

        for i in range(1, len(sentences)):
            similarity = cosine_similarity(embeddings[i - 1], embeddings[i])

            if similarity < threshold:
                if len(current_chunk) >= min_chunk_size:
                    all_chunks.append(' '.join(current_chunk))
                    current_chunk = [sentences[i]]
                else:
                    current_chunk.append(sentences[i])
            else:
                current_chunk.append(sentences[i])

        if current_chunk:
            all_chunks.append(' '.join(current_chunk))

    # Final cleanup — remove any empty chunks
    all_chunks = [c.strip() for c in all_chunks if len(c.strip()) > 20]

    return all_chunks


# ── Fallback: original fixed chunking ────────────────────────────────────
# Kept for documents where semantic chunking fails (very short text, etc.)

def fixed_chunk(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Original Phase 1 chunker. Used as fallback only.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if end < len(text):
            last_space = chunk.rfind(' ')
            if last_space != -1:
                end = start + last_space
                chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap
    return chunks