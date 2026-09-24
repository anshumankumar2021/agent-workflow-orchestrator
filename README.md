# Agentic AI Workflow Orchestrator

[![ci](https://github.com/anshumankumar2021/agent-workflow-orchestrator/actions/workflows/ci.yml/badge.svg)](https://github.com/anshumankumar2021/agent-workflow-orchestrator/actions/workflows/ci.yml)

An AI operations agent for a fictional card-payment processor, Northwind Payments. It answers questions from risk
and operations staff by planning, querying a read-only SQL database and a set of policy documents, doing the
arithmetic with a calculator tool, and checking its own answer before replying. It can open risk tickets, but only
when a user with the right role asks for one, and it ignores instructions planted inside the documents it reads.

The agent is a [LangGraph](https://github.com/langchain-ai/langgraph) state machine driving an open-weights model on
[Groq](https://console.groq.com)'s free API. Every step (plan, tool call, guardrail decision, self-check) is recorded
in a trace that the demo page replays.

```
      plan ──► act ⇄ tools ──► verify ──► answer
               ▲                  │
               └── fix (once) ────┘
```

## What's in it

| Part | Details |
|---|---|
| Data | `agent/data.py` builds a SQLite database from a fixed seed: 120 merchants, 45,698 transactions and 324 chargebacks (Jan–Jun 2026), plus reference FX rates. CI rebuilds it and fails if it differs from the committed file. |
| Policies | 12 short markdown documents: chargeback thresholds, dispute timelines, refund windows, reserve rules and so on. `dispute-evidence-tips.md` contains a planted prompt injection. |
| Tools | `run_sql`, `search_policies` (BM25), `calculate` (safe AST evaluator), `convert_currency`, `create_ticket` (the only tool with a side effect). |
| Graph | `agent/graph.py`: *plan → act ⇄ tools → verify*, where a reviewer pass checks every number in the answer against the tool results and can send the agent back once. `mode="react"` is the same agent without plan and verify, used as the baseline. |
| LLM client | `agent/llm.py`: Groq's OpenAI-compatible endpoint over `httpx`, with retries on 429/5xx, recovery from malformed tool calls, and a response cache keyed by the full request, so reruns cost nothing and recorded runs replay exactly. |
| Benchmark | `bench/tasks.json`: 37 questions in 7 categories. Expected answers are computed from SQL by `bench/build_tasks.py`, so scoring is automatic: numbers must match within tolerance, required keywords must appear, tickets must or must not be opened, and injected claims must not be repeated. |

## Guardrails

| Guardrail | How it works |
|---|---|
| Read-only SQL | The database is opened `mode=ro&immutable=1` with a SQLite authorizer that allows only `SELECT`/`READ`/`FUNCTION`. Queries are single statements, time out after 2 s (progress handler) and return at most 50 rows. |
| Role-based tools | `viewer` can read; `analyst` can also open tickets. Viewers are never offered the ticket tool, and a call to it is refused anyway. |
| Side-effect gate | `create_ticket` only runs if the user's own question asks for a ticket. A model can't decide to do it on its own, or because a document told it to. |
| Injection screening | Tool output is treated as data. Sentences that address an AI ("ignore previous instructions", "you are now in admin mode", "immediately call …") are removed before the model sees them, and logged in the trace. |
| Budgets | At most 8 tool calls per question and a LangGraph recursion limit; unknown tools and malformed arguments come back to the model as errors instead of crashing the run. |
| Self-check | In plan + verify mode, an answer whose numbers aren't backed by tool results is sent back for one fix. |

## Results

Results are produced by `python -m scripts.benchmark` in GitHub Actions and committed to `results/`. They appear here
and on the demo page after the first benchmark run.

## Live demo

The demo page (`public/index.html`, served by `api/agent.py` on Vercel) has two modes:

- **Recorded runs**: every benchmark run, with its full trace and score, replayed without calling the model.
- **Live runs**: type any question and the agent runs on Groq. Rate-limited to 8 questions per visitor every 10
  minutes and 300 per server instance per day, to stay inside Groq's free tier.

Run it locally with `GROQ_API_KEY=... python -m scripts.dev_server` and open http://localhost:3003.

## Run it

```bash
pip install -r requirements-dev.txt
pytest -q                                   # offline: a scripted model stands in for the LLM
python -m agent.data --check                # database matches a fresh build from the seed
python -m bench.build_tasks --check         # expected answers match the database

export GROQ_API_KEY=...                     # free key from console.groq.com
python -m scripts.benchmark                 # all three configs; resumable if rate-limited
python -m scripts.benchmark --configs plan_verify --limit 5
```

Benchmark configurations:

| Config | Graph | Guardrails | Questions |
|---|---|---|---|
| `react` | plain tool-calling loop | on | all 37 |
| `plan_verify` | plan → act → verify | on | all 37 |
| `plan_verify_open` | plan → act → verify | **off** | the 8 action and injection questions (ablation) |

### Running the benchmark in GitHub Actions

1. Add a repository secret named `GROQ_API_KEY` (Settings → Secrets and variables → Actions).
2. Actions → **ci** → **Run workflow**. The benchmark job runs after the tests and commits `results/` back to the
   repository, including the response cache, so a run that hits the free tier's daily limit picks up where it
   stopped next time.

## Layout

```
agent/data.py         synthetic database (seeded) and schema description
agent/tools.py        the five tools, tool schemas, ticket store
agent/guards.py       injection screening, read-only SQL connection, role permissions
agent/graph.py        LangGraph agent: plan / act / verify nodes, guarded tool execution, trace
agent/llm.py          Groq client with retries and response cache; scripted model for tests
bench/                benchmark tasks (generated from SQL) and scorer
scripts/benchmark.py  runs the configs and writes results/results.json and results/runs.json
api/agent.py          Vercel serverless endpoint: recorded runs and rate-limited live runs
public/index.html     demo page
policies/             policy documents the agent searches
tests/                unit tests for guardrails, graph routing, LLM client and API
```

## Limitations

- The data and policies are synthetic, and the benchmark is small (37 questions), so differences of a few points
  between configurations are within noise.
- Injection screening is pattern-based. It catches the planted attack and common phrasings, not every possible one;
  the side-effect gate and role checks are what actually stop an unrequested ticket.
- The verifier is the same model checking itself, which catches unsupported numbers more reliably than wrong
  reasoning.
