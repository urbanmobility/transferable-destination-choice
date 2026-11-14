"""Plan 2 Task 5: llm.py (endpoint from env, key handling, schema-validated calls with retries, input whitelist, file
interface) and the call points qc_summary / simplify / readability. A fake requests.post stands in for the server."""
import importlib
import json
import os

import pytest

from test_search_param import two_cities  # noqa: F401


class FakeResp:
    def __init__(self, status, payload):
        self.status_code, self._p = status, payload
        self.text = json.dumps(payload)

    def json(self):
        return self._p


def _server(script, seen):
    """script: one (status, content) per expected call; content may be a dict (JSON-encoded) or a raw string."""
    def post(url, json=None, headers=None, timeout=None):
        import json as _json
        seen.append({"url": url, "body": json, "headers": headers, "timeout": timeout})
        status, content = script.pop(0)
        if status != 200:
            return FakeResp(status, {"error": "boom"})
        c = content if isinstance(content, str) else _json.dumps(content)
        return FakeResp(200, {"choices": [{"message": {"content": c, "reasoning_content": "..."}}], "model": json["model"]})
    return post


@pytest.fixture
def llm_env(monkeypatch):
    for k in ("LLM_BASE_URL", "LLM_MODEL", "LLM_KEY_FILE", "OPENAI_API_KEY", "LLM_LOG_DIR"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("LLM_API", "openai")           # plan 4: the default is the Anthropic format (MiniMax)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    import llm
    importlib.reload(llm)
    monkeypatch.setattr(llm, "BACKOFF", 0.0)
    return llm


SCHEMA = {"type": "object", "properties": {"sympy": {"type": "string"}, "reason": {"type": "string"}},
          "required": ["sympy", "reason"], "additionalProperties": False}
MSG = [{"role": "user", "content": "hi"}]


def test_call_returns_validated_json_and_retries(llm_env, monkeypatch):
    llm, seen = llm_env, []
    monkeypatch.setattr(llm.requests, "post", _server([(500, None), (200, {"sympy": "log(x5 + 1.0)", "reason": "merged"})], seen))
    out = llm.call(MSG, SCHEMA)
    assert out == {"sympy": "log(x5 + 1.0)", "reason": "merged"} and len(seen) == 2 and llm.LAST["attempt"] == 2
    body = seen[-1]["body"]
    assert body["model"] == llm.DEFAULT_MODEL and body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["schema"] == SCHEMA and body["reasoning_effort"] == "low"
    assert seen[-1]["url"] == "http://127.0.0.1:8081/v1/chat/completions"
    assert seen[-1]["headers"]["Authorization"] == "Bearer sk-test" and body["messages"] == MSG


def test_call_rejects_schema_violation_after_retries(llm_env, monkeypatch):
    llm, seen = llm_env, []
    monkeypatch.setattr(llm.requests, "post", _server([(200, {"sympy": "x5"})] * 3, seen))     # "reason" missing
    with pytest.raises(llm.LLMError) as e:
        llm.call(MSG, SCHEMA)
    assert len(seen) == 3 and "sk-test" not in str(e.value) and "reason" in str(e.value)
    monkeypatch.setattr(llm.requests, "post", _server([(200, "not json {")] * 3, []))
    with pytest.raises(llm.LLMError):
        llm.call(MSG, SCHEMA)


def test_validate(llm_env):
    llm = llm_env
    assert llm.validate({"sympy": "a", "reason": "b"}, SCHEMA) == []
    assert llm.validate({"sympy": 1, "reason": "b"}, SCHEMA) and llm.validate({"sympy": "a"}, SCHEMA)
    assert llm.validate({"sympy": "a", "reason": "b", "extra": 1}, SCHEMA)
    s = {"type": "object", "required": ["score"],
         "properties": {"score": {"type": "integer", "minimum": 1, "maximum": 5},
                        "flags": {"type": "array", "items": {"type": "object", "properties": {"city": {"type": "string"}},
                                                             "required": ["city"]}},
                        "kind": {"type": "string", "enum": ["a", "b"]}}}
    assert llm.validate({"score": 3, "flags": [{"city": "x"}], "kind": "a"}, s) == []
    assert llm.validate({"score": 7}, s) and llm.validate({"score": 3, "kind": "z"}, s) and llm.validate({"score": 3, "flags": [{}]}, s)
    assert llm.validate({"score": 3.5}, s) and llm.validate({"score": True}, s) and llm.validate([], s)


def test_endpoint_from_env(llm_env, monkeypatch):
    llm = llm_env
    assert llm.endpoint() == ("http://127.0.0.1:8081/v1", llm.DEFAULT_MODEL)
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:3000/api/")
    monkeypatch.setenv("LLM_MODEL", "qwen3.8-flash-next")
    assert llm.endpoint() == ("http://localhost:3000/api", "qwen3.8-flash-next")
    seen = []
    monkeypatch.setattr(llm.requests, "post", _server([(200, {"sympy": "x1", "reason": "r"})], seen))
    llm.call(MSG, SCHEMA)
    assert seen[0]["url"] == "http://localhost:3000/api/chat/completions" and seen[0]["body"]["model"] == "qwen3.8-flash-next"


def test_key_from_file_and_never_in_errors(llm_env, monkeypatch, tmp_path):
    llm = llm_env
    monkeypatch.delenv("OPENAI_API_KEY")
    kf = tmp_path / "k"
    kf.write_text("sk-filekey\n")
    monkeypatch.setenv("LLM_KEY_FILE", str(kf))
    assert llm.key() == "sk-filekey"
    monkeypatch.setenv("LLM_KEY_FILE", str(tmp_path / "missing"))
    with pytest.raises(llm.LLMError) as e:
        llm.key()
    assert "sk-" not in str(e.value) and "missing" in str(e.value)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def boom(url, json=None, headers=None, timeout=None):
        raise ConnectionError(f"refused with headers {headers}")
    monkeypatch.setattr(llm.requests, "post", boom)
    with pytest.raises(llm.LLMError) as e:
        llm.call(MSG, SCHEMA)
    assert "sk-test" not in str(e.value) and llm.LAST["attempt"] == 3


def test_check_input_whitelist(llm_env):
    llm = llm_env
    with pytest.raises(ValueError):
        llm.check_input({"tract_vals": [1, 2, 3]})
    with pytest.raises(ValueError):
        llm.check_input({"cities": [{"unit_counts": [1, 2]}]})
    with pytest.raises(ValueError):
        llm.check_input({"city": "38060"})                        # a confirmation city id
    with pytest.raises(ValueError):
        llm.check_input({"fits": {"37980": 1.0}})
    with pytest.raises(ValueError):
        llm.check_input({"note": "Phoenix 38060 looks odd"})
    llm.check_input({"city": "31080", "tract_count": 2343, "unit_scale": "tract", "vals": [1, 2, 3], "tract_ids": ["a"]})


def test_run_point_writes_output_and_never_raises_on_rejection(llm_env, monkeypatch, tmp_path):
    llm, seen = llm_env, []
    import llm_points
    importlib.reload(llm_points)
    monkeypatch.setattr(llm.requests, "post", _server([(200, {"sympy": "x5 + 1", "reason": "r"})], seen))
    inp, out = tmp_path / "in.json", tmp_path / "out.json"
    inp.write_text(json.dumps({"cid": "c0001", "equation": "log(x5 + 0.5 + 0.5)", "variables": {"x5": "E_j"}}))
    fn = lambda p: llm_points.simplify(p, verify=lambda cid, s: {"accepted": False, "reason": "corr 0.5", "corr": 0.5})  # noqa: E731
    r = llm.run_point("simplify", inp, out, fn)
    o = json.loads(out.read_text())
    assert o == r and o["point"] == "simplify" and o["result"]["accepted"] is False and o["result"]["sympy"] == "x5 + 1"
    assert o["model"] == llm.DEFAULT_MODEL and o["effort"] == "low" and o["elapsed_s"] >= 0 and o["attempt"] == 1
    assert "sk-test" not in out.read_text()
    assert seen[0]["body"]["response_format"]["json_schema"]["schema"] == llm_points.SCHEMAS["simplify"]
    inp.write_text(json.dumps({"cid": "c0001", "city": "38060"}))
    with pytest.raises(ValueError):
        llm.run_point("simplify", inp, tmp_path / "out2.json", fn)
    assert not (tmp_path / "out2.json").exists()


def test_readability_marks_unknown_variables_invalid(llm_env, monkeypatch):
    llm = llm_env
    import llm_points
    importlib.reload(llm_points)
    good = {"score": 4, "terms": [{"term": "log(x5 + 1)", "meaning": "jobs"}], "uses_variables": ["x5", "foo"]}
    monkeypatch.setattr(llm.requests, "post", _server([(200, good)], []))
    r = llm_points.readability({"cid": "c1", "equation": "log(x5 + 1.0)", "variables": {"x5": "E_j"}})
    assert r["invalid"] is True and r["score"] == 4 and "foo" in r["invalid_reason"]
    monkeypatch.setattr(llm.requests, "post", _server([(200, {**good, "uses_variables": ["x5", "E_j"]})], []))
    assert llm_points.readability({"cid": "c1", "equation": "log(x5 + 1.0)", "variables": {"x5": "E_j"}})["invalid"] is False


def test_qc_summary_and_var_meaning(llm_env, monkeypatch):
    llm = llm_env
    import llm_points
    import sr_data
    importlib.reload(llm_points)
    monkeypatch.setattr(llm.requests, "post", _server([(200, {"flags": [{"city": "31080", "flag": "coverage", "note": "n"}]})], []))
    r = llm_points.qc_summary({"cities": [{"city": "31080", "coverage": 0.35}]})
    assert r["flags"][0]["city"] == "31080"
    assert set(llm_points.VAR_MEANING) == set(sr_data.VARS) and "always 0" in llm_points.VAR_MEANING["s_health"]
    assert llm_points.XVAR["x5"] == "E_j" and llm_points.XVAR["x10"] == "s_health"


@pytest.mark.live
@pytest.mark.skipif(os.environ.get("LLM_LIVE") != "1", reason="LLM_LIVE=1 runs against the real local service")
def test_live_simplify():
    import llm
    import llm_points
    importlib.reload(llm)
    importlib.reload(llm_points)
    r = llm_points.simplify({"cid": "c0001", "equation": "log(x5 + 0.5 + 0.5)",
                             "variables": {"x5": llm_points.VAR_MEANING["E_j"]}})
    assert isinstance(r["sympy"], str) and r["sympy"] and llm.LAST["elapsed_s"] > 0
    print("live:", r, llm.LAST)


POOL = [{"cid": "c0001", "equation": "f = -2.0 * log(#1 + 0.5 + 0.5); p1 = []; p2 = []", "template": None, "complexity": 6},
        {"cid": "c0002", "equation": "f = log(#5 + 1.0); p1 = []; p2 = []", "template": None, "complexity": 4}]


def test_propose_is_resumable_and_records_failures(llm_env, monkeypatch, tmp_path):
    """ETA review 11-03: proposals run in 2 threads for the whole pool before any verification; a candidate with an
    output file is not called again; a failed call is written, not raised."""
    import pandas as pd
    import llm_stage
    importlib.reload(llm_stage)
    llm, seen = llm_env, []
    fits = pd.DataFrame([{"cid": c, "city": "TEST1", "finite": True, "val_nll": 4.0, "within_1se": True} for c in ("c0001", "c0002")])
    monkeypatch.setattr(llm.requests, "post", _server([(200, {"sympy": "-2.0*log(x1 + 1.0)", "reason": "r"}),
                                                       (500, None), (500, None), (500, None)], seen))
    llm_stage.propose("simplify", POOL, {}, fits, tmp_path, threads=1)
    a, b = json.loads((tmp_path / "simplify_c0001.json").read_text()), json.loads((tmp_path / "simplify_c0002.json").read_text())
    assert a["result"]["sympy"] == "-2.0*log(x1 + 1.0)" and a["result"]["accepted"] is None
    assert b["result"] is None and "HTTP 500" in b["error"] and "sk-test" not in json.dumps(b)
    inp = json.loads((tmp_path / "simplify_c0001.in.json").read_text())
    assert "x1" in inp["variables"] and inp["fits"][0]["city"] == "TEST1"
    n = len(seen)
    llm_stage.propose("simplify", POOL, {}, fits, tmp_path, threads=2)
    assert len(seen) == n                                            # both outputs exist: nothing called again


def test_verify_pending_parallel(two_cities, tmp_path):
    import pandas as pd
    import llm_stage
    importlib.reload(llm_stage)
    fits = pd.DataFrame([{"cid": c, "city": "TEST1", "finite": True} for c in ("c0001", "c0002")])
    for d in (tmp_path / "s", tmp_path / "p"):
        d.mkdir()
        (d / "simplify_c0001.json").write_text(json.dumps({"point": "simplify", "result": {"cid": "c0001", "sympy": "-2.0*log(x1 + 1.0)", "reason": "r", "accepted": None}}))
        (d / "simplify_c0002.json").write_text(json.dumps({"point": "simplify", "result": {"cid": "c0002", "sympy": "1/(x5 + 1)", "reason": "r", "accepted": None}}))
    llm_stage.verify_pending(POOL, fits, ["TEST1"], tmp_path / "s", procs=1, n_orig=None)
    llm_stage.verify_pending(POOL, fits, ["TEST1"], tmp_path / "p", procs=2, n_orig=None)
    for d in ("s", "p"):
        a = json.loads((tmp_path / d / "simplify_c0001.json").read_text())["result"]
        b = json.loads((tmp_path / d / "simplify_c0002.json").read_text())["result"]
        assert a["accepted"] is True and a["verify"]["corr"] > 0.999 and a["verify_city"] == "TEST1"
        assert b["accepted"] is False and "candidate" not in b["verify"]


def test_ping_reports_reachability(llm_env, monkeypatch):
    """An unreachable endpoint must fail the LLM stage up front instead of recording 150 failed calls."""
    llm = llm_env

    def get_ok(url, headers=None, timeout=None):
        return FakeResp(200, {"data": []})

    def get_down(url, headers=None, timeout=None):
        raise ConnectionError("refused")
    monkeypatch.setattr(llm.requests, "get", get_ok)
    assert llm.ping() is True
    monkeypatch.setattr(llm.requests, "get", get_down)
    assert llm.ping() is False


# ---- plan 4 Task 8: Anthropic-format backend (MiniMax), extraction, call log, test-period guard ----

def _anth(script, seen):
    """script: one (status, content blocks or None) per call; blocks = list of dicts as the server returns them."""
    def post(url, json=None, headers=None, timeout=None):
        seen.append({"url": url, "body": json, "headers": headers, "timeout": timeout})
        status, blocks = script.pop(0)
        if status != 200:
            return FakeResp(status, {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}})
        return FakeResp(200, {"id": "m1", "type": "message", "role": "assistant", "model": json["model"],
                              "content": blocks, "stop_reason": "end_turn", "usage": {"input_tokens": 5}})
    return post


@pytest.fixture
def anth_env(monkeypatch, tmp_path):
    for k in ("LLM_BASE_URL", "LLM_MODEL", "LLM_API", "OPENAI_API_KEY", "LLM_LOG_DIR"):
        monkeypatch.delenv(k, raising=False)
    kf = tmp_path / "minimax_api_key"
    kf.write_text("sk-cp-test\n")
    monkeypatch.setenv("LLM_KEY_FILE", str(kf))
    import llm
    importlib.reload(llm)
    monkeypatch.setattr(llm, "BACKOFF", 0.0)
    return llm


THINK = {"type": "thinking", "thinking": "reasoning {not json} ...", "signature": "s"}
OK = [THINK, {"type": "text", "text": "Conclusion:\n{\"a\": 1}\n"}]
SCH_A = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}


def test_anthropic_request_shape(anth_env, monkeypatch):
    llm, seen = anth_env, []
    monkeypatch.setattr(llm.requests, "post", _anth([(200, OK)], seen))
    msgs = [{"role": "system", "content": "You are a reviewer."}, {"role": "user", "content": "Review this expression"}]
    assert llm.call(msgs, SCH_A, max_tokens=8000) == {"a": 1}
    q = seen[0]
    assert q["url"] == "https://api.minimax.cn/anthropic/v1/messages"
    assert q["headers"]["x-api-key"] == "sk-cp-test" and q["headers"]["anthropic-version"] == "2023-06-01"
    assert "Authorization" not in q["headers"]
    b = q["body"]
    assert b["model"] == "MiniMax-M3.1-Flash" and b["messages"] == [{"role": "user", "content": "Review this expression"}]
    assert b["system"].startswith("You are a reviewer.") and '"required": ["a"]' in b["system"]       # schema in the prompt
    assert b["max_tokens"] >= 32000 and "response_format" not in b and "reasoning_effort" not in b   # 11-06 seeds
    assert llm.MAX_CONCURRENT == 3 and llm.endpoint() == ("https://api.minimax.cn/anthropic", "MiniMax-M3.1-Flash")


def test_extract_json():
    import llm
    assert llm.extract_json('preface {"a": 1} afterword') == {"a": 1}
    assert llm.extract_json('```json\n{"s": "x}{y", "n": {"k": [1, 2]}}\n```') == {"s": "x}{y", "n": {"k": [1, 2]}}
    assert llm.extract_json('{bad} then {"ok": true}') == {"ok": True}
    for t in ("", "no JSON", '{"a": 1', "[1, 2]"):
        with pytest.raises(llm.LLMError):
            llm.extract_json(t)


def test_anthropic_json_extraction_and_error_file(anth_env, monkeypatch, tmp_path):
    """Text around the JSON is fine; a reply with only thinking, or truncated, fails the call after the
    retries; run_point then writes result null with the error and does not raise into the loop."""
    llm, seen = anth_env, []
    monkeypatch.setattr(llm.requests, "post", _anth([(200, OK)], seen))
    assert llm.call([{"role": "user", "content": "q"}], SCH_A) == {"a": 1}
    only_think = [THINK]
    truncated = [THINK, {"type": "text", "text": '{"a": '}]
    monkeypatch.setattr(llm.requests, "post", _anth([(200, only_think), (529, None), (200, truncated)], seen))
    with pytest.raises(llm.LLMError) as e:
        llm.call([{"role": "user", "content": "q"}], SCH_A)
    assert "no text" in str(e.value) and "HTTP 529" in str(e.value) and "no JSON" in str(e.value)
    monkeypatch.setattr(llm.requests, "post", _anth([(200, only_think)] * 3, seen))
    inp, out = tmp_path / "in.json", tmp_path / "out.json"
    inp.write_text(json.dumps({"cid": "c0001", "equation": "log(x6 + 1)"}))
    r = llm.run_point("review", inp, out, lambda p: llm.call([{"role": "user", "content": json.dumps(p)}], SCH_A))
    o = json.loads(out.read_text())
    assert o == r and o["result"] is None and "no text" in o["error"] and o["model"] == "MiniMax-M3.1-Flash"
    assert "sk-cp-test" not in out.read_text()


def test_log_file_has_no_key(anth_env, monkeypatch, tmp_path):
    llm, seen = anth_env, []
    monkeypatch.setenv("LLM_LOG_DIR", str(tmp_path / "log"))
    monkeypatch.setattr(llm.requests, "post", _anth([(200, OK), (200, [THINK])] + [(200, [THINK])] * 2, seen))
    llm.call([{"role": "system", "content": "S"}, {"role": "user", "content": "q"}], SCH_A, name="review")
    with pytest.raises(llm.LLMError):
        llm.call([{"role": "user", "content": "q"}], SCH_A, name="review")
    logs = sorted((tmp_path / "log").glob("review_*.json"))
    assert len(logs) == 2
    a, b = (json.loads(p.read_text()) for p in logs)
    ok, bad = (a, b) if a["error"] is None else (b, a)
    assert ok["text"].endswith('{"a": 1}\n') and ok["thinking_chars"] == len(THINK["thinking"]) and ok["attempts"] == 1
    assert ok["model"] == "MiniMax-M3.1-Flash" and ok["elapsed_s"] >= 0 and ok["messages"][0]["content"] == "S"
    assert ok["result"] == {"a": 1} and bad["result"] is None and bad["attempts"] == 3 and "no text" in bad["error"]
    for p in logs:
        t = p.read_text()
        assert "sk-cp-test" not in t and "x-api-key" not in t


def test_anthropic_ping(anth_env, monkeypatch):
    llm, seen = anth_env, []
    monkeypatch.setattr(llm.requests, "post", _anth([(200, [{"type": "text", "text": "hi"}])], seen))
    assert llm.ping() is True and seen[0]["body"]["max_tokens"] == 8 and seen[0]["url"].endswith("/v1/messages")

    def down(*a, **k):
        raise ConnectionError("refused")
    monkeypatch.setattr(llm.requests, "post", down)
    assert llm.ping() is False


def test_check_input_rejects_test_period_numbers(anth_env):
    llm = anth_env
    for bad in ({"test_nll": 4.5}, {"x": {"TEST_share": [0.1, 0.2]}}, {"lawtest_nll": 3.9}, {"n_test": 3}):
        with pytest.raises(ValueError, match="test"):
            llm.check_input(bad)
    for ok in ({"fold4_nll": 4.5}, {"test": "slope_ci"}, {"tests_allowed": ["slope_ci", "group_diff"]},
               {"lawtest_pass": {"H1-01": True}}):
        llm.check_input(ok)


def test_check_input_does_not_confuse_decimal_constants_with_city_ids(anth_env):
    """A refitted constant such as 38060.0 or 0.38060 is not a city id; a city id in text still is."""
    import config
    llm = anth_env
    cid = config.CONFIRM[0]
    llm.check_input({"eta": f"log(x6 + {cid}.0) - 0.{cid}*x1 + 1.{cid}e-3"})
    # first case: an id in Chinese text ("city {cid}" with a full-width full stop), as LLM inputs may contain
    for bad in (f"城市 {cid}。", f"city {cid}.", f"({cid})"):
        with pytest.raises(ValueError):
            llm.check_input({"t": bad})
