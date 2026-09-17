"""Thin LLM layer.

Three things matter here and nothing else:
  * Disk cache keyed on (model, prompt). Without it the 15-minute reproduce
    claim is a lie on the second run, and evaluation becomes non-deterministic.
  * Defensive JSON parsing. Small open models emit fenced JSON, trailing prose,
    and occasionally a leading "Here is the JSON:". We strip all of it.
  * An offline mock so `pytest` and the smoke test run with no API key.
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
        for attempt in range(4):
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
                time.sleep(2 ** attempt)
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
