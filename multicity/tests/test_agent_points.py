"""Plan 4: MiniMax call points — request shape, schema, and that a well-formed answer validates."""
import importlib

import pytest


@pytest.fixture
def captured(monkeypatch):
    import llm
    import agent_points
    importlib.reload(agent_points)
    seen = []

    def fake(messages, schema, effort="low", max_tokens=16000, timeout=1800, name="x"):
        seen.append({"messages": messages, "schema": schema, "name": name, "max_tokens": max_tokens})
        return {"ok": True}
    monkeypatch.setattr(llm, "call", fake)
    return agent_points, seen


def test_review_point(captured):
    import llm
    ap, seen = captured
    ap.review({"cid": "c0001", "eta": "log(x6 + 1)"})
    s = seen[0]
    assert s["name"] == "review" and s["schema"] is ap.REVIEW_SCHEMA and "c0001" in s["messages"][1]["content"]
    ans = {"problems": [{"issue": "i", "evidence": "e"}], "cleaned_sympy": "", "changes": "", "expected_fit_cost": "0",
           "insight": "x"}
    assert llm.validate(ans, ap.REVIEW_SCHEMA) == []


def test_hypotheses_point(captured):
    import llm
    ap, seen = captured
    ap.hypotheses({"n_cities": 30, "methods": {"slope_ci": "…"}})
    s = seen[0]
    assert s["name"] == "hypotheses" and "at most 5" in s["messages"][1]["content"]
    ans = {"hypotheses": [{"statement": "s", "skeleton": "c1", "parameter": "f:c1", "covariate": "log_population",
                           "direction": "+", "test": "slope_ci", "criterion": "CI > 0", "rationale": "r"}]}
    assert llm.validate(ans, ap.HYPOTHESES_SCHEMA) == []
    assert llm.validate({"hypotheses": [{"statement": "s"}]}, ap.HYPOTHESES_SCHEMA)


def test_stuck_report_point(captured):
    import llm
    ap, seen = captured
    ap.stuck_report({"reason": "no_improvement", "top8_history": [["c1"]]})
    s = seen[0]
    assert s["name"] == "stuck_report" and "no_improvement" in s["messages"][1]["content"]
    ans = {"tried": "t", "evidence": "e", "why_framework": "w", "proposal": "p"}
    assert llm.validate(ans, ap.STUCK_SCHEMA) == [] and llm.validate({"tried": "t"}, ap.STUCK_SCHEMA)
