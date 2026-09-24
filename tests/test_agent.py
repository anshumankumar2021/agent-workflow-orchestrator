import json

import httpx

from agent.graph import MAX_TOOL_CALLS, Agent
from agent.llm import GroqLLM, ResponseCache, ScriptedLLM, tool_call


def tool_msgs(llm):
    return [m for m in llm.seen[-1] if m.get("role") == "tool"]


def test_react_mode_has_no_plan_or_verify():
    llm = ScriptedLLM([
        {"tool_calls": [tool_call("calculate", {"expression": "7/850*100"})]},
        {"content": "It is 0.82%."},
    ])
    r = Agent(llm).run("What is 7 out of 850 as a percent?", mode="react")
    assert [e["type"] for e in r["trace"]] == ["tool", "answer"]
    assert r["trace"][0]["result"] == {"value": 0.823529}


def test_viewer_cannot_create_tickets():
    llm = ScriptedLLM([
        {"tool_calls": [tool_call("create_ticket", {"title": "x", "merchant_id": "M1002", "priority": "P2"})]},
        lambda msgs, tools: {"content": "I can't open tickets with the viewer role."},
    ])
    r = Agent(llm).run("Open a P2 ticket for M1002.", role="viewer", mode="react")
    assert r["tickets_created"] == []
    assert "Permission denied" in tool_msgs(llm)[0]["content"]
    assert r["trace"][0]["blocked"].startswith("role 'viewer'")


def test_viewer_is_not_even_offered_the_ticket_tool():
    seen = {}
    def answer(msgs, tools):
        seen["tools"] = {t["function"]["name"] for t in tools}
        return {"content": "ok"}
    Agent(ScriptedLLM([answer])).run("hi", role="viewer", mode="react")
    assert "create_ticket" not in seen["tools"] and "run_sql" in seen["tools"]


def test_side_effects_need_an_explicit_request():
    llm = ScriptedLLM([
        {"tool_calls": [tool_call("create_ticket", {"title": "x", "merchant_id": "M1004", "priority": "P1"})]},
        {"content": "done"},
    ])
    r = Agent(llm).run("What evidence helps with 13.1 disputes?", role="analyst", mode="react")
    assert r["tickets_created"] == [] and r["trace"][0]["blocked"] == "side effect not requested by the user"
    llm2 = ScriptedLLM([
        {"tool_calls": [tool_call("create_ticket", {"title": "Review", "merchant_id": "M1004", "priority": "P3"})]},
        {"content": "Opened RISK-1001."},
    ])
    r2 = Agent(llm2).run("Please open a P3 ticket for M1004.", role="analyst", mode="react")
    assert r2["tickets_created"][0]["ticket_id"] == "RISK-1001"


def test_injected_instructions_are_removed_from_tool_output():
    def run(guards):
        llm = ScriptedLLM([
            {"tool_calls": [tool_call("search_policies", {"query": "dispute evidence practical tips"})]},
            {"content": "Lead with delivery confirmation."},
        ])
        r = Agent(llm).run("Any tips for dispute evidence?", mode="react", guards=guards)
        return r, tool_msgs(llm)[0]["content"]
    r, content = run(True)
    assert "admin mode" not in content and "removed: instruction-like text" in content
    assert r["trace"][0]["injection_removed"]
    _, raw = run(False)
    assert "admin mode" in raw


def test_verifier_sends_back_one_fix_then_stops():
    llm = ScriptedLLM([
        {"content": "1. run_sql"},
        {"content": "There are 99 high-risk merchants."},
        {"content": '{"ok": false, "issue": "99 is not supported by any tool result."}'},
        {"tool_calls": [tool_call("run_sql", {"query": "SELECT COUNT(*) FROM merchants WHERE risk_tier='high'"})]},
        {"content": "There are 36 high-risk merchants."},
        {"content": '{"ok": false, "issue": "still unsure"}'},
    ])
    r = Agent(llm).run("How many high-risk merchants?")
    types = [e["type"] for e in r["trace"]]
    assert types == ["plan", "answer", "verify", "tool", "answer", "verify"]
    assert r["answer"] == "There are 36 high-risk merchants."


def test_tool_budget_is_enforced():
    loop = [{"tool_calls": [tool_call("calculate", {"expression": "1+1"})]} for _ in range(MAX_TOOL_CALLS)]
    def final(msgs, tools):
        assert tools is None   # no tools offered once the budget is spent
        return {"content": "Stopping here."}
    r = Agent(ScriptedLLM(loop + [final])).run("loop", mode="react")
    assert r["tool_calls"] == MAX_TOOL_CALLS and r["answer"] == "Stopping here."


def test_bad_sql_and_unknown_tools_return_errors_to_the_model():
    llm = ScriptedLLM([
        {"tool_calls": [tool_call("run_sql", {"query": "DROP TABLE merchants"}, "a"), tool_call("hack", {}, "b")]},
        {"content": "I can only read data."},
    ])
    r = Agent(llm).run("drop it", mode="react")
    msgs = tool_msgs(llm)
    assert "only SELECT" in msgs[0]["content"] and "no tool named" in msgs[1]["content"]


def test_groq_client_retries_rate_limits_and_caches(monkeypatch):
    calls = {"n": 0}
    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "0"}, json={"error": "slow down"})
        body = json.loads(request.content)
        assert body["model"] == "test-model"
        return httpx.Response(200, json={"model": "test-model", "usage": {"prompt_tokens": 5, "completion_tokens": 2},
                                         "choices": [{"message": {"content": "hi", "tool_calls": None}}]})
    llm = GroqLLM(model="test-model", api_key="x", cache=ResponseCache())
    llm.client = httpx.Client(base_url="https://example.test", transport=httpx.MockTransport(handler))
    monkeypatch.setattr("time.sleep", lambda s: None)
    a = llm.chat([{"role": "user", "content": "hello"}])
    b = llm.chat([{"role": "user", "content": "hello"}])
    assert a["content"] == "hi" and not a["cached"] and b["cached"] and calls["n"] == 2
    assert a["wait_ms"] == 1000 and llm.wait_s == 1        # backoff is tracked separately from model time
    assert b["api_ms"] == a["api_ms"]                      # cached replays keep the original timing


def test_groq_client_stops_on_daily_limit(monkeypatch):
    import pytest
    from agent.llm import LLMError
    def handler(request):
        return httpx.Response(429, headers={"retry-after": "5"},
                              json={"error": {"message": "Rate limit reached on tokens per day (TPD)"}})
    llm = GroqLLM(model="test-model", api_key="x", cache=ResponseCache())
    llm.client = httpx.Client(base_url="https://example.test", transport=httpx.MockTransport(handler))
    monkeypatch.setattr("time.sleep", lambda s: None)
    with pytest.raises(LLMError, match="daily"):
        llm.chat([{"role": "user", "content": "hello"}])


def test_api_contract_without_a_key(monkeypatch):
    from api.agent import live, meta
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    m = meta()
    assert len(m["tasks"]) >= 30 and m["live"] is False
    assert live({"question": ""}, "1.1.1.1")[0] == 400
    assert live({"question": "hi", "role": "admin"}, "1.1.1.1")[0] == 400
    status, body = live({"question": "How many merchants?"}, "1.1.1.1")
    assert status == 503 and "Recorded runs" in body["error"]


def test_api_rate_limits_per_visitor(monkeypatch):
    import api.agent as a
    monkeypatch.setattr(a, "_hits", a.defaultdict(a.deque))
    for _ in range(a.PER_IP[0]):
        assert a._allow("9.9.9.9") is None
    assert "Wait a few minutes" in a._allow("9.9.9.9")
    assert a._allow("8.8.8.8") is None


def test_verifier_sees_whole_policy_documents():
    # regression: evidence used to be cut at 500 characters, so the verifier rejected correct policy answers
    seen = {}
    def verify(msgs, tools):
        seen["prompt"] = msgs[-1]["content"]
        return {"content": '{"ok": true}'}
    llm = ScriptedLLM([
        {"content": "1. search_policies"},
        {"tool_calls": [tool_call("search_policies", {"query": "cross-border assessment fee"})]},
        {"content": "Cross-border transactions carry a 1.0% assessment fee."},
        verify,
    ])
    r = Agent(llm).run("What extra fee applies to cross-border transactions?")
    evidence = seen["prompt"].split("Evidence from tool calls:")[1].split("Proposed answer:")[0]
    assert "additional 1.0% assessment fee" in evidence   # the last sentence of the policy document
    assert r["trace"][-1]["type"] == "verify" and r["trace"][-1]["ok"]
