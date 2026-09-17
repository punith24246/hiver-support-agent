"""Thin LLM layer.

Three things matter here and nothing else:
  * Disk cache keyed on (model, prompt). Without it the 15-minute reproduce
    claim is a lie on the second run, and evaluation becomes non-deterministic.
  * Defensive JSON parsing. Small open models emit fenced JSON, trailing prose,
    and occasionally a leading "Here is the JSON:". We strip all of it.
  * An offline mock so `pytest` and the smoke test run with no API key.

Rate limiting: Groq's on-demand tier caps tokens-per-minute (TPM). That limit
applies to the whole account, not per-thread, so a plain per-call retry loop
doesn't help when several ThreadPoolExecutor workers all retry at once and
re-collide on the same shrinking budget. RATE_LIMIT_TPM below paces every
call - across all threads - to stay under budget proactively, and the retry
loop also parses the "try again in Xs" the API gives you on a 429 instead of
guessing a backoff.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from typing import Any

_CACHE_LOCK = threading.Lock()
_CACHE_PATH = os.environ.get("AGENT_CACHE", "runs/llm_cache.sqlite")


def _cache_conn():
    os.makedirs(os.path.dirname(_CACHE_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(_CACHE_PATH, check_same_thread=False)
    conn.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT)")
    return conn


_CONN = _cache_conn()


def _key(model: str, system: str, user: str, temperature: float) -> str:
    raw = json.dumps([model, system, user, temperature], sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def cache_get(k: str):
    with _CACHE_LOCK:
        row = _CONN.execute("SELECT v FROM cache WHERE k=?", (k,)).fetchone()
    return row[0] if row else None


def cache_put(k: str, v: str):
    with _CACHE_LOCK:
        _CONN.execute("INSERT OR REPLACE INTO cache VALUES (?,?)", (k, v))
        _CONN.commit()


# --- global cross-thread pacing -------------------------------------------
# Groq's TPM limit is per-account, not per-thread. A token bucket shared by
# every LLM instance / thread keeps us under budget instead of just reacting
# to 429s after the fact. Tune via env vars if your tier/model differs.

_RATE_LOCK = threading.Lock()
_LAST_CALL_TS = [0.0]

_TPM_LIMIT = float(os.environ.get("AGENT_TPM_LIMIT", "8000"))
_EST_TOKENS_PER_CALL = float(os.environ.get("AGENT_EST_TOKENS_PER_CALL", "600"))
# seconds to leave between calls so (60 / interval) * est_tokens <= TPM_LIMIT
_MIN_INTERVAL = (60.0 * _EST_TOKENS_PER_CALL / _TPM_LIMIT) if _TPM_LIMIT > 0 else 0.0


def _pace():
    """Block until it's safe to make another call, shared across all threads."""
    if _MIN_INTERVAL <= 0:
        return
    with _RATE_LOCK:
        now = time.monotonic()
        wait = _LAST_CALL_TS[0] + _MIN_INTERVAL - now
        if wait > 0:
            time.sleep(wait)
        _LAST_CALL_TS[0] = time.monotonic()


_RETRY_AFTER_RE = re.compile(r"try again in ([\d.]+)s", re.I)


def _retry_after_seconds(err: Exception) -> float | None:
    """Pull the server-suggested wait time out of a 429 error message, if present."""
    m = _RETRY_AFTER_RE.search(str(err))
    return float(m.group(1)) if m else None


def _is_rate_limit(err: Exception) -> bool:
    s = str(err)
    return "429" in s or "rate_limit" in s.lower()


# ---------------------------------------------------------------------------

class LLM:
    def __init__(self, model: str, temperature: float = 0.0, mock: bool = False):
        self.model = model
        self.temperature = temperature
        self.mock = mock or os.environ.get("AGENT_MOCK") == "1"
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI  # Groq is OpenAI-API compatible
            self._client = OpenAI(
                api_key=os.environ["GROQ_API_KEY"],
                base_url="https://api.groq.com/openai/v1",
            )
        return self._client

    def complete(self, system: str, user: str, max_tokens: int = 700) -> str:
        if self.mock:
            return _mock_response(system, user)

        k = _key(self.model, system, user, self.temperature)
        hit = cache_get(k)
        if hit is not None:
            return hit

        last_err = None
        max_attempts = 8  # rate limits need more room than transient 5xx do
        for attempt in range(max_attempts):
            _pace()  # proactive: stay under TPM budget before we even try
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                out = resp.choices[0].message.content or ""
                cache_put(k, out)
                return out
            except Exception as e:  # rate limits, transient 5xx
                last_err = e
                if _is_rate_limit(e):
                    # reactive: honour the server's own "try again in Xs" if given,
                    # otherwise fall back to a longer backoff than non-rate-limit errors
                    wait = _retry_after_seconds(e)
                    if wait is None:
                        wait = min(60.0, 2 ** attempt)
                    time.sleep(wait + 0.5)  # small buffer past the boundary
                else:
                    time.sleep(min(30.0, 2 ** attempt))
        raise RuntimeError(f"LLM call failed after retries: {last_err}")

    def complete_json(self, system: str, user: str, max_tokens: int = 700) -> dict[str, Any]:
        raw = self.complete(system, user, max_tokens=max_tokens)
        return parse_json(raw)


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse_json(raw: str) -> dict[str, Any]:
    """Extract the first JSON object from a possibly chatty completion."""
    if not raw:
        return {}
    m = _FENCE.search(raw)
    if m:
        raw = m.group(1)
    start = raw.find("{")
    if start == -1:
        return {}
    depth, end = 0, None
    for i, ch in enumerate(raw[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        return {}
    try:
        return json.loads(raw[start:end])
    except json.JSONDecodeError:
        # last resort: single -> double quotes
        try:
            return json.loads(raw[start:end].replace("'", '"'))
        except Exception:
            return {}


def _mock_response(system: str, user: str) -> str:
    """Deterministic stand-in so the pipeline is testable with no network."""
    if "judge" in system.lower():
        return json.dumps({
            "groundedness": 3, "helpfulness": 3, "tone": 3,
            "safety": 5, "overall": 3, "rationale": "mock",
        })
    if "intent" in system.lower():
        return json.dumps({
            "intent": "outage", "confidence": 0.5,
            "rationale": "mock", "severity": "medium",
        })
    return json.dumps({"reply": "Sorry for the trouble — can you DM us your address?"})
