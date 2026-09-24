"""Chat-completions client for Groq's free OpenAI-compatible API, with retries and a response cache.

The cache key is a hash of (model, messages, tools, temperature). Identical requests are answered from the
cache, so re-running the benchmark or a repeated demo question costs no API calls.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import httpx

GROQ_URL = "https://api.groq.com/openai/v1"
# Ordered by free-tier headroom for tool-calling models (tokens per minute and per day), then quality.
PREFERRED_MODELS = ["meta-llama/llama-4-scout-17b-16e-instruct", "llama-3.3-70b-versatile", "openai/gpt-oss-120b",
                    "qwen/qwen3-32b", "llama-3.1-8b-instant"]


class LLMError(Exception):
    pass


class ResponseCache:
    def __init__(self, path: Path | None = None):
        self.path = path
        self.mem: dict[str, dict] = {}
        self.hits = self.misses = 0
        if path and path.exists():
            self.mem = json.loads(path.read_text())

    @staticmethod
    def key(payload: dict) -> str:
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def get(self, k):
        v = self.mem.get(k)
        if v is None:
            self.misses += 1
        else:
            self.hits += 1
        return v

    def put(self, k, v):
        self.mem[k] = v

    def save(self):
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.mem))


class GroqLLM:
    def __init__(self, model: str | None = None, api_key: str | None = None, cache: ResponseCache | None = None,
                 timeout: float = 30.0):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise LLMError("GROQ_API_KEY is not set")
        self.client = httpx.Client(base_url=GROQ_URL, timeout=timeout,
                                   headers={"Authorization": f"Bearer {self.api_key}"})
        self.model = model or os.environ.get("GROQ_MODEL") or self.pick_model()
        self.cache = cache or ResponseCache()
        self.calls = 0
        self.wait_s = 0.0          # time spent backing off on 429/5xx, so latency can be reported without it
        self.limits: dict = {}     # last seen x-ratelimit-limit-* headers (free-tier quotas)

    def pick_model(self) -> str:
        try:
            ids = {m["id"] for m in self.client.get("/models").json().get("data", [])}
        except Exception:
            return PREFERRED_MODELS[0]
        return next((m for m in PREFERRED_MODELS if m in ids), PREFERRED_MODELS[0])

    def chat(self, messages: list[dict], tools: list[dict] | None = None, temperature: float = 0.0,
             max_tokens: int = 2048) -> dict:
        payload = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        k = ResponseCache.key(payload)
        hit = self.cache.get(k)
        if hit is not None:
            return {**hit, "cached": True}
        waited = 0.0
        for attempt in range(8):
            t0 = time.perf_counter()
            r = self.client.post("/chat/completions", json=payload)
            api_ms = round((time.perf_counter() - t0) * 1000, 1)
            self.limits.update({h: v for h, v in r.headers.items() if h.startswith("x-ratelimit-limit")})
            if r.status_code == 200:
                data = r.json()
                msg = data["choices"][0]["message"]
                out = {"content": msg.get("content") or "", "tool_calls": msg.get("tool_calls") or [],
                       "usage": data.get("usage", {}), "model": data.get("model", self.model), "api_ms": api_ms}
                self.calls += 1
                self.cache.put(k, out)
                return {**out, "cached": False, "wait_ms": round(waited * 1000, 1)}
            if r.status_code == 400 and "tool_use_failed" in r.text:
                # the model produced a malformed tool call; surface it as text so the graph can recover
                failed = r.json().get("error", {}).get("failed_generation", "")
                return {"content": "", "tool_calls": [], "usage": {}, "model": self.model, "cached": False,
                        "error": "malformed tool call", "failed_generation": failed[:500]}
            if r.status_code in (429, 500, 502, 503):
                wait = min(float(r.headers.get("retry-after", 0) or 0) or min(2 ** attempt, 30), 60)
                if "per day" in r.text.lower() or wait >= 60:
                    raise LLMError(f"daily or long rate limit reached: {r.text[:200]}")
                time.sleep(wait)
                waited += wait
                self.wait_s += wait
                continue
            raise LLMError(f"Groq API error {r.status_code}: {r.text[:300]}")
        raise LLMError("Groq API kept rate-limiting; try again later")


class ScriptedLLM:
    """Test double: returns pre-written responses in order (or computes them from the messages)."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.model = "scripted"
        self.cache = ResponseCache()
        self.calls = 0
        self.seen: list[list[dict]] = []

    def chat(self, messages, tools=None, temperature=0.0, max_tokens=2048):
        self.seen.append(messages)
        self.calls += 1
        r = self.responses.pop(0)
        if callable(r):
            r = r(messages, tools)
        return {"content": r.get("content", ""), "tool_calls": r.get("tool_calls", []), "usage": r.get("usage", {}),
                "model": "scripted", "cached": False}


def tool_call(name: str, args: dict, call_id: str = "call_1") -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
