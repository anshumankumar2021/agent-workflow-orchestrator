"""Tool registry: JSON schemas the model sees, plus the Python that runs each tool."""
from __future__ import annotations

import ast
import math
import operator
import re
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

from agent.data import DB_PATH, FX, SCHEMA_DOC
from agent.guards import check_sql, readonly_connection

POLICY_DIR = Path(__file__).resolve().parent.parent / "policies"
MAX_ROWS = 50


class ToolError(Exception):
    pass


# ---------- run_sql ----------
@lru_cache(maxsize=1)
def _db():
    return readonly_connection(DB_PATH)


def run_sql(query: str) -> dict:
    err = check_sql(query)
    if err:
        raise ToolError(err)
    con = _db()
    t = time.perf_counter()
    deadline = t + 2.0
    con.set_progress_handler(lambda: 1 if time.perf_counter() > deadline else 0, 10_000)  # 2 s timeout
    try:
        cur = con.execute(query)
        cols = [d[0] for d in cur.description or []]
        rows = cur.fetchmany(MAX_ROWS + 1)
    except Exception as e:  # sqlite errors and authorizer denials come back to the model as text
        raise ToolError(f"SQL error: {e}") from None
    finally:
        con.set_progress_handler(None, 0)
    truncated = len(rows) > MAX_ROWS
    rows = [[round(v, 4) if isinstance(v, float) else v for v in r] for r in rows[:MAX_ROWS]]
    return {"columns": cols, "rows": rows, "truncated": truncated}


# ---------- search_policies (small BM25 over the policy documents) ----------
_WORD = re.compile(r"[a-z0-9.]+")
_STOP = set("a an and are as at be by for from how i in is it of on or the to what when which who why with does do".split())


def _toks(s: str) -> list[str]:
    return [w.strip(".") for w in _WORD.findall(s.lower()) if w.strip(".") and w.strip(".") not in _STOP]


@lru_cache(maxsize=1)
def _policies():
    docs = []
    for p in sorted(POLICY_DIR.glob("*.md")):
        text = p.read_text()
        title = text.splitlines()[0].lstrip("# ").strip()
        docs.append({"id": p.stem, "title": title, "text": text, "tf": Counter(_toks(text))})
    df = Counter(t for d in docs for t in d["tf"])
    avg = sum(sum(d["tf"].values()) for d in docs) / len(docs)
    return docs, df, avg


def search_policies(query: str, k: int = 2) -> dict:
    docs, df, avg = _policies()
    n = len(docs)
    scores = []
    for d in docs:
        dl = sum(d["tf"].values())
        s = 0.0
        for t in set(_toks(query)):
            if t in d["tf"]:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                tf = d["tf"][t]
                s += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * dl / avg))
        scores.append(s)
    order = sorted(range(n), key=lambda i: -scores[i])[:k]
    return {"results": [{"doc": docs[i]["id"], "title": docs[i]["title"], "text": docs[i]["text"]}
                        for i in order if scores[i] > 0]}


# ---------- calculate (safe arithmetic) ----------
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {"round": round, "abs": abs, "min": min, "max": max, "sqrt": math.sqrt}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        if isinstance(node.op, ast.Pow) and abs(_eval(node.right)) > 100:
            raise ToolError("exponent too large")
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and not node.keywords:
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise ToolError("only numbers, + - * / % **, parentheses and round/abs/min/max/sqrt are allowed")


def calculate(expression: str) -> dict:
    if len(expression) > 300:
        raise ToolError("expression too long")
    try:
        value = _eval(ast.parse(expression, mode="eval"))
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"could not evaluate: {e}") from None
    return {"value": round(value, 6) if isinstance(value, float) else value}


# ---------- convert_currency ----------
def convert_currency(amount: float, from_currency: str, to_currency: str = "USD") -> dict:
    f, t = from_currency.upper(), to_currency.upper()
    if f not in FX or t not in FX:
        raise ToolError(f"unsupported currency; supported: {', '.join(sorted(FX))}")
    usd = float(amount) / FX[f]
    return {"amount": round(usd * FX[t], 2), "currency": t, "rate_note": f"1 USD = {FX[f]} {f} = {FX[t]} {t}"}


# ---------- create_ticket (the only tool with a side effect) ----------
class TicketStore:
    def __init__(self):
        self.tickets: list[dict] = []

    def create(self, title: str, merchant_id: str, priority: str, details: str = "") -> dict:
        if priority not in ("P1", "P2", "P3"):
            raise ToolError("priority must be P1, P2 or P3")
        if not re.fullmatch(r"M\d{4}", merchant_id or ""):
            raise ToolError("merchant_id must look like M1000")
        t = {"ticket_id": f"RISK-{1001 + len(self.tickets)}", "title": title[:120], "merchant_id": merchant_id,
             "priority": priority, "details": details[:500]}
        self.tickets.append(t)
        return t


SPECS = [
    {"name": "run_sql", "side_effect": False,
     "description": "Run one read-only SQLite SELECT query on the payments database and return up to 50 rows.\n" + SCHEMA_DOC,
     "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "A single SELECT statement"}},
                    "required": ["query"]}},
    {"name": "search_policies", "side_effect": False,
     "description": "Search Northwind Payments policy documents (thresholds, timelines, definitions, reason codes). Returns the 2 best matching documents.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "calculate", "side_effect": False,
     "description": "Evaluate an arithmetic expression, e.g. '(12 / 850) * 100' or 'round(1234.567, 2)'.",
     "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}},
    {"name": "convert_currency", "side_effect": False,
     "description": "Convert an amount between currencies using the reference fx table (USD, GBP, EUR, INR, SGD, BRL, CAD, AUD, JPY).",
     "parameters": {"type": "object", "properties": {"amount": {"type": "number"}, "from_currency": {"type": "string"},
                                                     "to_currency": {"type": "string"}},
                    "required": ["amount", "from_currency", "to_currency"]}},
    {"name": "create_ticket", "side_effect": True,
     "description": "Open a risk ticket for a merchant. Only use when the user explicitly asks you to open or create a ticket.",
     "parameters": {"type": "object", "properties": {"title": {"type": "string"}, "merchant_id": {"type": "string"},
                                                     "priority": {"type": "string", "enum": ["P1", "P2", "P3"]},
                                                     "details": {"type": "string"}},
                    "required": ["title", "merchant_id", "priority"]}},
]
SPEC_BY_NAME = {s["name"]: s for s in SPECS}


def openai_tools(names: set[str]) -> list[dict]:
    return [{"type": "function", "function": {"name": s["name"], "description": s["description"],
                                              "parameters": s["parameters"]}} for s in SPECS if s["name"] in names]


def execute(name: str, args: dict, tickets: TicketStore) -> dict:
    if name == "run_sql":
        return run_sql(str(args.get("query", "")))
    if name == "search_policies":
        return search_policies(str(args.get("query", "")))
    if name == "calculate":
        return calculate(str(args.get("expression", "")))
    if name == "convert_currency":
        return convert_currency(args.get("amount", 0), str(args.get("from_currency", "")), str(args.get("to_currency", "USD")))
    if name == "create_ticket":
        return tickets.create(str(args.get("title", "")), str(args.get("merchant_id", "")), str(args.get("priority", "")),
                              str(args.get("details", "")))
    raise ToolError(f"unknown tool {name}")
