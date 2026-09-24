"""Run the benchmark against a real model on Groq and write results/results.json and results/runs.json.

    GROQ_API_KEY=... python -m scripts.benchmark                # all configs
    python -m scripts.benchmark --configs plan_verify --limit 5  # a quick subset

Configs:
  react             plain tool-calling loop, guardrails on (baseline)
  plan_verify       plan -> act -> verify graph, guardrails on
  plan_verify_open  same graph with guardrails off, run only on injection and action tasks (ablation)

LLM responses are cached in results/llm_cache.json, so an interrupted run resumes without repeating calls,
and results can be re-scored later without an API key.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from agent.graph import Agent
from agent.llm import GroqLLM, LLMError, ResponseCache
from agent.tools import TicketStore
from bench.score import score

ROOT = Path(__file__).resolve().parent.parent
TASKS = json.loads((ROOT / "bench" / "tasks.json").read_text())
CONFIGS = {
    "react": {"mode": "react", "guards": True, "categories": None},
    "plan_verify": {"mode": "plan_verify", "guards": True, "categories": None},
    "plan_verify_open": {"mode": "plan_verify", "guards": False, "categories": {"injection", "action"}},
}


def pctl(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))] if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--model")
    ap.add_argument("--pause", type=float, default=2.0, help="seconds between tasks (free-tier rate limits)")
    ap.add_argument("--offline", action="store_true", help="re-score from results/llm_cache.json only; no API key needed")
    args = ap.parse_args()

    cache = ResponseCache(ROOT / "results" / "llm_cache.json")
    out_path, runs_path = ROOT / "results" / "results.json", ROOT / "results" / "runs.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    model = args.model or (results.get("model") if args.offline else None)
    llm = GroqLLM(model=model, cache=cache, offline=args.offline)
    print(f"model: {llm.model}", flush=True)
    if results.get("model") not in (None, llm.model):
        results = {}  # results from a different model aren't comparable
    all_runs = json.loads(runs_path.read_text()) if runs_path.exists() else {}
    results.update({"model": llm.model, "n_tasks": len(TASKS), "configs": results.get("configs", {})})

    for name in args.configs.split(","):
        cfg = CONFIGS[name]
        tasks = [t for t in TASKS if cfg["categories"] is None or t["category"] in cfg["categories"]][:args.limit]
        runs = []
        for i, t in enumerate(tasks):
            before = llm.calls
            try:
                r = Agent(llm, TicketStore()).run(t["question"], role=t["role"], mode=cfg["mode"], guards=cfg["guards"])
            except LLMError as e:
                cache.save()
                raise SystemExit(f"stopped at {name}/{t['id']}: {e}\nRe-run to resume from the cache.")
            s = score(t, r)
            fresh = llm.calls > before
            # time the agent spent working: model API time plus tools, excluding free-tier rate-limit waits.
            # api_ms is stored with each cached response, so resumed runs report the original timing.
            tool_ms = sum(e.get("ms", 0) for e in r["trace"] if e["type"] == "tool")
            r["active_ms"] = round(r["usage"].get("api_ms", 0) + tool_ms, 1)
            runs.append({"id": t["id"], "category": t["category"], "question": t["question"], "role": t["role"],
                         "pass": s["pass"], "checks": s["checks"], "answer": r["answer"], "plan": r["plan"],
                         "trace": r["trace"], "usage": r["usage"], "tool_calls": r["tool_calls"],
                         "tickets_created": r["tickets_created"], "total_ms": r["total_ms"],
                         "active_ms": r["active_ms"], "fresh": fresh})
            print(f"[{name}] {i + 1:2d}/{len(tasks)} {t['id']:16s} {'PASS' if s['pass'] else 'FAIL'} {s['checks']}", flush=True)
            cache.save()
            if fresh:
                time.sleep(args.pause)
        by_cat = {}
        for c in dict.fromkeys(r["category"] for r in runs):
            rs = [r for r in runs if r["category"] == c]
            by_cat[c] = {"passed": sum(r["pass"] for r in rs), "total": len(rs)}
        lat = [r["active_ms"] for r in runs]
        prev = results["configs"].get(name, {})
        results["configs"][name] = {
            "mode": cfg["mode"], "guards": cfg["guards"],
            "passed": sum(r["pass"] for r in runs), "total": len(runs),
            "pass_rate": round(sum(r["pass"] for r in runs) / len(runs), 4),
            "by_category": by_cat,
            "avg_llm_calls": round(statistics.mean(r["usage"].get("llm_calls", 0) for r in runs), 2),
            "avg_tool_calls": round(statistics.mean(r["tool_calls"] for r in runs), 2),
            "avg_tokens": round(statistics.mean(r["usage"].get("prompt_tokens", 0) + r["usage"].get("completion_tokens", 0) for r in runs)),
            "latency_ms_p50": pctl(lat, 50), "latency_ms_p95": pctl(lat, 95), "latency_n": len(lat),
            # measured on the live run; a cache replay has no waits, so keep the earlier figure
            "rate_limit_wait_s": round(sum(r["usage"].get("wait_ms", 0) for r in runs) / 1000, 1) if any(r["fresh"] for r in runs)
            else prev.get("rate_limit_wait_s"),
            "tickets_created_in_injection_tasks": sum(len(r["tickets_created"]) for r in runs if r["category"] == "injection"),
            "injection_findings": sum(len(e.get("injection_removed", [])) for r in runs for e in r["trace"] if e["type"] == "tool"),
            "guard_blocks": sum(1 for r in runs for e in r["trace"] if e.get("blocked")),
        }
        all_runs[name] = runs
        print(json.dumps({k: v for k, v in results["configs"][name].items() if k != "by_category"}), flush=True)
        # write after every config, so a run stopped by the daily limit still publishes what finished
        results["cache"] = {"hits": cache.hits, "misses": cache.misses}
        if llm.limits:
            results["rate_limits"] = llm.limits
        out_path.parent.mkdir(exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        runs_path.write_text(json.dumps(all_runs))
        cache.save()


if __name__ == "__main__":
    main()
