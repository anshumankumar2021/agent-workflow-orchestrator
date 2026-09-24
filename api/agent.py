"""Vercel serverless endpoint for the ops agent.

GET  /api/agent?meta=1                      -> benchmark results, sample questions, recorded runs (no LLM calls)
POST /api/agent {question, role, mode}      -> run the agent live on Groq (needs GROQ_API_KEY in the environment)
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.graph import Agent  # noqa: E402
from agent.llm import GroqLLM, LLMError, ResponseCache  # noqa: E402
from agent.tools import TicketStore  # noqa: E402

MAX_QUESTION = 400
PER_IP = (8, 600)          # 8 live runs per 10 minutes per visitor
PER_INSTANCE_DAY = 300     # and a daily ceiling per server instance (Groq's free tier has its own caps too)
_hits: dict[str, deque] = defaultdict(deque)
_day = {"date": time.strftime("%Y-%m-%d"), "n": 0}
_lock = threading.Lock()
_cache = ResponseCache()
_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = GroqLLM(cache=_cache, timeout=20)
    return _llm


def _allow(ip: str) -> str | None:
    now = time.time()
    with _lock:
        if _day["date"] != time.strftime("%Y-%m-%d"):
            _day.update(date=time.strftime("%Y-%m-%d"), n=0)
        if _day["n"] >= PER_INSTANCE_DAY:
            return "The live demo has reached today's limit. Recorded runs below still work."
        q = _hits[ip]
        while q and now - q[0] > PER_IP[1]:
            q.popleft()
        if len(q) >= PER_IP[0]:
            return f"You've run {PER_IP[0]} live questions in 10 minutes. Wait a few minutes, or browse the recorded runs."
        q.append(now)
        _day["n"] += 1
    return None


def meta() -> dict:
    res_p, runs_p = ROOT / "results" / "results.json", ROOT / "results" / "runs.json"
    tasks = json.loads((ROOT / "bench" / "tasks.json").read_text())
    runs = json.loads(runs_p.read_text()) if runs_p.exists() else {}
    return {"results": json.loads(res_p.read_text()) if res_p.exists() else None,
            "tasks": [{"id": t["id"], "category": t["category"], "question": t["question"], "role": t["role"]} for t in tasks],
            "runs": runs, "live": bool(os.environ.get("GROQ_API_KEY"))}


def live(body: dict, ip: str) -> tuple[int, dict]:
    q = str(body.get("question", "")).strip()[:MAX_QUESTION]
    role = body.get("role", "analyst")
    mode = body.get("mode", "plan_verify")
    if not q:
        return 400, {"error": "Type a question first."}
    if role not in ("analyst", "viewer") or mode not in ("plan_verify", "react"):
        return 400, {"error": "role must be analyst or viewer; mode must be plan_verify or react"}
    if not os.environ.get("GROQ_API_KEY"):
        return 503, {"error": "Live runs aren't switched on yet (no Groq key on the server). Recorded runs below still work."}
    limited = _allow(ip)
    if limited:
        return 429, {"error": limited}
    try:
        return 200, Agent(_get_llm(), TicketStore()).run(q, role=role, mode=mode, guards=True)
    except LLMError as e:
        return 503, {"error": f"The model service is busy or rate-limited right now ({e}). Try again in a minute."}


class handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: dict, cache: bool = False):
        data = json.dumps(body, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "public, s-maxage=3600, max-age=300" if cache else "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        qs = parse_qs(urlparse(self.path).query)
        if qs.get("meta"):
            return self._send(200, meta(), cache=True)
        return self._send(400, {"error": "use ?meta=1 or POST a question"})

    def do_POST(self):
        try:
            n = min(int(self.headers.get("Content-Length", 0)), 10_000)
            body = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self._send(400, {"error": "send JSON: {question, role, mode}"})
        ip = (self.headers.get("x-forwarded-for") or self.client_address[0] or "?").split(",")[0].strip()
        status, out = live(body, ip)
        return self._send(status, out)
