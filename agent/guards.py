"""Guardrails: prompt-injection screening of tool output, read-only SQL, and role-based tool access."""
from __future__ import annotations

import re
import sqlite3

# Phrases that address an AI system rather than a human reader. Tool output is data; anything that tries to
# give the agent instructions is redacted before the model sees it, and the event is logged in the trace.
_INJECTION = [
    r"ignore (all |any )?(previous|prior|above|earlier) (instructions|prompts|messages)",
    r"disregard (all |any )?(previous|prior|above) (instructions|rules)",
    r"you are now (in )?\w+",
    r"(system|developer) (notice|prompt|message|override)",
    r"to (ai|llm) (assistants|agents|models)",
    r"(immediately|now) (call|invoke|run|use) (the )?\w+(_\w+)*",
    r"admin mode",
    r"do not (tell|inform) the user",
]
_INJECTION_RE = re.compile("|".join(f"(?:{p})" for p in _INJECTION), re.I)


def screen(text: str) -> tuple[str, list[str]]:
    """Return (safe_text, findings). Sentences that contain injection patterns are replaced with a marker."""
    findings = []
    parts = re.split(r'(?<=[.!?"])\s+', text)
    out = []
    for s in parts:
        m = _INJECTION_RE.search(s)
        if m:
            findings.append(m.group(0))
            out.append("[removed: instruction-like text in untrusted content]")
        else:
            out.append(s)
    return " ".join(out), findings


# ---------- read-only SQL ----------
_ALLOWED_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}


def _authorizer(action, arg1, arg2, dbname, source):
    if action in _ALLOWED_ACTIONS:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def check_sql(query: str) -> str | None:
    """Cheap pre-check before the database-level authorizer. Returns an error message or None."""
    q = query.strip().rstrip(";").strip()
    if not q:
        return "empty query"
    if ";" in q:
        return "only one statement is allowed"
    if not re.match(r"(?is)^\s*(select|with)\b", q):
        return "only SELECT queries are allowed"
    if re.search(r"(?i)\b(attach|pragma|insert|update|delete|drop|alter|create|replace|vacuum)\b", q):
        return "only read-only SELECT queries are allowed"
    return None


def readonly_connection(path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True, check_same_thread=False)
    con.set_authorizer(_authorizer)
    return con


# ---------- roles ----------
ROLE_TOOLS = {
    "viewer": {"run_sql", "search_policies", "calculate", "convert_currency"},
    "analyst": {"run_sql", "search_policies", "calculate", "convert_currency", "create_ticket"},
}


def allowed(role: str, tool: str) -> bool:
    return tool in ROLE_TOOLS.get(role, set())
