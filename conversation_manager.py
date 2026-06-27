"""
conversation_manager.py
────────────────────────────────────────────────────────────────────────────
Handles two jobs:

  1. SESSION HISTORY  — stores per-session conversation turns so follow-up
                        questions have context ("explain further", "what about
                        the second point", etc.)

  2. QUERY REWRITING  — before retrieval, sends the raw user query + recent
                        history to Mistral and asks it to:
                          • resolve pronouns / references ("that", "it", "the
                            second point") into explicit terms
                          • fix typos and spelling mistakes
                          • expand vague follow-ups into self-contained questions

     The rewritten query is what goes to retrieval.
     The original raw query is kept for display / history.
"""

import requests
import uuid
from collections import deque
from typing import Optional

OLLAMA_URL = "http://localhost:11434/api/generate"

# How many past turns to include when rewriting (keeps prompt short)
HISTORY_WINDOW = 6  # last 3 exchanges (user + assistant each)


# ── Session store ─────────────────────────────────────────────────────────

class ConversationSession:
    """
    Stores the last N messages for a single chat session.

    Each entry: {"role": "user"|"assistant", "content": str}
    """
    def __init__(self, session_id: str, maxlen: int = 20):
        self.session_id = session_id
        # deque auto-drops oldest when maxlen exceeded
        self.history: deque = deque(maxlen=maxlen)

    def add_turn(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

    def get_recent(self, n: int = HISTORY_WINDOW) -> list[dict]:
        """Return the last n messages (most recent at the end)."""
        return list(self.history)[-n:]

    def is_empty(self) -> bool:
        return len(self.history) == 0


class SessionStore:
    """
    In-memory store for all active sessions.
    Sessions are created on demand and never expire during the process
    lifetime. For production you'd back this with Redis or a DB.
    """
    def __init__(self):
        self._sessions: dict[str, ConversationSession] = {}

    def get_or_create(self, session_id: str) -> ConversationSession:
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationSession(session_id)
        return self._sessions[session_id]

    def clear(self, session_id: str):
        self._sessions.pop(session_id, None)

    def new_session(self) -> str:
        """Create a fresh session and return its ID."""
        sid = str(uuid.uuid4())
        self._sessions[sid] = ConversationSession(sid)
        return sid


# Singleton — imported everywhere
session_store = SessionStore()


# ── Query rewriter ────────────────────────────────────────────────────────

def _build_rewrite_prompt(raw_query: str, history: list[dict]) -> str:
    """
    Build the prompt that asks Mistral to rewrite the user's query
    into a self-contained, typo-free question.
    """
    if not history:
        # No history — just fix typos / casing, nothing to resolve
        return f"""You are a query rewriting assistant.
Your only job: rewrite the user's question to fix any spelling mistakes or
typos and make it grammatically clean. Do NOT change the meaning.
Output ONLY the rewritten question — no explanation, no quotes, no preamble.

Question: {raw_query}
Rewritten:"""

    # Format history as a readable transcript
    transcript_lines = []
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        # Truncate long assistant answers so the prompt stays compact
        content = msg["content"]
        if msg["role"] == "assistant" and len(content) > 300:
            content = content[:300] + "…"
        transcript_lines.append(f"{role}: {content}")

    transcript = "\n".join(transcript_lines)

    return f"""You are a query rewriting assistant working inside a document Q&A system.

Given the conversation so far and the user's latest message, rewrite the
latest message into a single, self-contained question that:
  1. Resolves all references (e.g. "it", "that", "the second point",
     "explain further", "what about X") using context from the conversation
  2. Fixes any spelling mistakes or typos
  3. Preserves the user's original intent exactly

Output ONLY the rewritten question — no explanation, no quotes, no preamble.
If the question is already clear and self-contained, output it as-is (still
fix typos).

Conversation so far:
{transcript}

User's latest message: {raw_query}
Rewritten question:"""


def rewrite_query(raw_query: str, history: list[dict]) -> str:
    """
    Send the raw query + history to Mistral for rewriting.
    Returns the rewritten query, or the original if rewriting fails.

    This is a fast call — we use a low token budget (100 tokens max)
    so it adds minimal latency.
    """
    # Don't bother rewriting if query is long and specific — likely fine
    # But always try if history exists (follow-up resolution needed)
    prompt = _build_rewrite_prompt(raw_query, history)

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 100,   # short — just one question
                    "temperature": 0.0,   # deterministic rewriting
                }
            },
            timeout=15
        )
        rewritten = response.json()["response"].strip()

        # Sanity checks — if the model hallucinated something weird, fall back
        if not rewritten:
            return raw_query
        if len(rewritten) > len(raw_query) * 5:
            # Model went off the rails and wrote an essay
            return raw_query
        # Strip any quotes the model wrapped around it
        rewritten = rewritten.strip('"\'')

        return rewritten

    except Exception:
        # Network issue, Ollama down, etc. — silently fall back
        return raw_query


# ── Public API ────────────────────────────────────────────────────────────

def process_query(raw_query: str, session_id: str) -> tuple[str, list[dict]]:
    """
    Main entry point called by the /query endpoint.

    Returns:
        rewritten_query  — what goes to retrieval (typo-fixed, context-resolved)
        history          — the recent conversation turns (for generator prompt)
    """
    session = session_store.get_or_create(session_id)
    recent_history = session.get_recent(HISTORY_WINDOW)

    # Rewrite the query using history context
    rewritten = rewrite_query(raw_query, recent_history)

    return rewritten, recent_history


def record_turn(session_id: str, user_query: str, assistant_answer: str):
    """
    After a successful response, save the turn to history.
    Call this with the ORIGINAL user query (not rewritten) for natural
    conversation display, but we store the answer verbatim.
    """
    session = session_store.get_or_create(session_id)
    session.add_turn("user", user_query)
    session.add_turn("assistant", assistant_answer)