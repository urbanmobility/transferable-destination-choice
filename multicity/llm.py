"""LLM client (plan 2 Task 5; plan 4 Task 8): plain HTTP, JSON output validated on
the client, retries with back-off, at most MAX_CONCURRENT = 3 concurrent requests, an input whitelist, a call log and a
file interface. The LLM only proposes; the code decides.

LLM_API = anthropic (default; MiniMax-M3.1-Flash in the paper): POST {LLM_BASE_URL}/v1/messages with x-api-key /
anthropic-version headers; the first system message becomes
`system` (with the JSON schema appended), the reply's text blocks are joined and the first balanced JSON object in
them is taken (extract_json); thinking blocks are only counted. Key: the file named by LLM_KEY_FILE.
LLM_API = openai (plans 2-3: a locally served Qwen model): POST {LLM_BASE_URL}/chat/completions with a json_schema
response format; key OPENAI_API_KEY, else the file named by LLM_KEY_FILE.
There is no default endpoint and no default key location: LLM_BASE_URL and a key must be set in the environment
(LLM_MODEL overrides the model name).
The key is read at call time and never printed, logged, or written to any product — error messages are redacted.
LLM_LOG_DIR: when set, every call writes <dir>/<name>_<YYYYmmddTHHMMSS>_<pid>_<n>.json (messages, reply text,
thinking length, model, elapsed, attempts, error, parsed result; no header).
The schema note (_schema_note) was issued in Chinese in the original runs; the text here is an English translation.
"""
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CITIES_CSV, CONFIRM  # noqa: E402

DEFAULT_MODEL = "qwen3.8-flash-next-unsloth-ud-q4_k_xl"                # openai backend (plans 2-3)
ANTHROPIC_MODEL = "MiniMax-M3.1-Flash"                                # anthropic backend (plan 4, from 2026-11-05)
ANTHROPIC_VERSION = "2023-06-01"
MIN_TOKENS = 32000          # anthropic: thinking counts against max_tokens (11-06: seed calls hit 16000 with no text)
MAX_ATTEMPTS, BACKOFF, MAX_CONCURRENT = 3, 1.0, 3
_SEM = threading.Semaphore(MAX_CONCURRENT)
_LOG_N = {"n": 0}
_LOG_LOCK = threading.Lock()
LAST = {"attempt": 0, "elapsed_s": 0.0, "effort": None, "model": None}     # bookkeeping of the latest call
_TL = threading.local()                                                     # the same, per thread (run_point)
_TYPES = {"object": dict, "array": list, "string": str, "integer": int, "number": (int, float), "boolean": bool,
          "null": type(None)}


class LLMError(RuntimeError):
    pass


def api():
    a = os.environ.get("LLM_API", "anthropic")
    if a not in ("anthropic", "openai"):
        raise LLMError(f"LLM_API must be anthropic or openai, not {a!r}")
    return a


def model_name():
    return os.environ.get("LLM_MODEL", ANTHROPIC_MODEL if api() == "anthropic" else DEFAULT_MODEL)


def endpoint():
    base = os.environ.get("LLM_BASE_URL", "").strip()
    if not base:
        raise LLMError("LLM_BASE_URL is not set (there is no default endpoint)")
    return base.rstrip("/"), model_name()


def key():
    if api() == "openai":
        k = os.environ.get("OPENAI_API_KEY")
        if k:
            return k.strip()
    f = os.environ.get("LLM_KEY_FILE", "").strip()
    if not f:
        raise LLMError("no API key: set LLM_KEY_FILE" + (" or OPENAI_API_KEY" if api() == "openai" else ""))
    p = Path(os.path.expanduser(f))
    try:
        k = p.read_text().strip()
    except OSError as e:
        raise LLMError(f"cannot read the API key file {p} ({type(e).__name__})") from None
    if not k:
        raise LLMError(f"API key file {p} is empty")
    return k


def validate(obj, schema, path="$"):
    """Violations of a JSON schema subset (type, required, properties, additionalProperties, items, enum,
    minimum / maximum) as a list of messages; [] = valid."""
    errs = []
    t = schema.get("type")
    if t:
        exp = _TYPES.get(t)
        if t in ("integer", "number") and isinstance(obj, bool):
            return [f"{path}: expected {t}, got boolean"]
        if exp is not None and not isinstance(obj, exp):
            return [f"{path}: expected {t}, got {type(obj).__name__}"]
    if "enum" in schema and obj not in schema["enum"]:
        errs.append(f"{path}: {obj!r} not in {schema['enum']}")
    if isinstance(obj, dict):
        for r in schema.get("required", []):
            if r not in obj:
                errs.append(f"{path}: missing required key {r!r}")
        props = schema.get("properties", {})
        for k, v in obj.items():
            if k in props:
                errs += validate(v, props[k], f"{path}.{k}")
            elif schema.get("additionalProperties") is False:
                errs.append(f"{path}: unexpected key {k!r}")
    if isinstance(obj, list) and "items" in schema:
        for i, v in enumerate(obj):
            errs += validate(v, schema["items"], f"{path}[{i}]")
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        if "minimum" in schema and obj < schema["minimum"]:
            errs.append(f"{path}: {obj} below minimum {schema['minimum']}")
        if "maximum" in schema and obj > schema["maximum"]:
            errs.append(f"{path}: {obj} above maximum {schema['maximum']}")
    return errs


def _redact(text, secret):
    return text.replace(secret, "***") if secret else text


def extract_json(text):
    """The first balanced {...} in text that parses as a JSON object (strings and escapes respected)."""
    text = text or ""
    for start in (i for i, ch in enumerate(text) if ch == "{"):
        depth, in_str, esc = 0, False, False
        for j in range(start, len(text)):
            ch = text[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:j + 1])
                    except ValueError:
                        break
                    if isinstance(obj, dict):
                        return obj
                    break
    raise LLMError("no JSON object in the reply text")


def _schema_note(schema):
    return ("\n\nOutput requirement: the reply must contain exactly one complete JSON object (a short explanation may "
            "precede it), conforming to the JSON schema below:\n"
            + json.dumps(schema, ensure_ascii=False))


def _anthropic_body(messages, schema, model, max_tokens):
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    return {"model": model, "max_tokens": max(int(max_tokens), MIN_TOKENS), "system": system + _schema_note(schema),
            "messages": [m for m in messages if m["role"] != "system"]}


def _post_once(base, model, secret, messages, schema, effort, max_tokens, timeout, name, rec):
    """One HTTP exchange; returns the reply text (rec gets text / thinking_chars / stop_reason / usage)."""
    if api() == "anthropic":
        body = _anthropic_body(messages, schema, model, max_tokens)
        headers = {"x-api-key": secret, "anthropic-version": ANTHROPIC_VERSION, "content-type": "application/json"}
        with _SEM:
            r = requests.post(f"{base}/v1/messages", json=body, headers=headers, timeout=timeout)
        if r.status_code != 200:
            raise LLMError(f"HTTP {r.status_code}: {r.text[:200]}")
        j = r.json()
        blocks = j.get("content") or []
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        rec.update(text=text, thinking_chars=sum(len(b.get("thinking", "")) for b in blocks if b.get("type") == "thinking"),
                   stop_reason=j.get("stop_reason"), usage=j.get("usage"))
        if not text.strip():
            raise LLMError(f"no text block in the reply (stop_reason {j.get('stop_reason')})")
        return text
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "reasoning_effort": effort,
            "response_format": {"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": True}}}
    headers = {"Authorization": f"Bearer {secret}", "Content-Type": "application/json"}
    with _SEM:
        r = requests.post(f"{base}/chat/completions", json=body, headers=headers, timeout=timeout)
    if r.status_code != 200:
        raise LLMError(f"HTTP {r.status_code}: {r.text[:200]}")
    msg = r.json()["choices"][0]["message"]
    rec.update(text=msg["content"], thinking_chars=len(msg.get("reasoning_content") or ""))
    return msg["content"]


def _log(name, rec):
    d = os.environ.get("LLM_LOG_DIR")
    if not d:
        return
    with _LOG_LOCK:
        _LOG_N["n"] += 1
        n = _LOG_N["n"]
    p = Path(d)
    p.mkdir(parents=True, exist_ok=True)
    (p / f"{name}_{time.strftime('%Y%m%dT%H%M%S')}_{os.getpid()}_{n}.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=1, default=str))


def call(messages, schema, effort="low", max_tokens=16000, timeout=1800, name="odsr_point"):
    """One schema-constrained completion; the parsed JSON object, validated on the client. Up to MAX_ATTEMPTS
    attempts (HTTP error, no text, no JSON object or a schema violation each count as a failure), then LLMError.
    effort is used by the openai backend only."""
    base, model = endpoint()
    secret = key()
    t0, errors = time.time(), []
    rec = {"name": name, "api": api(), "model": model, "messages": messages, "text": None, "thinking_chars": 0,
           "attempts": 0, "elapsed_s": 0.0, "error": None, "result": None}
    for attempt in range(1, MAX_ATTEMPTS + 1):
        LAST.update(attempt=attempt, model=model, effort=effort)
        _TL.last = dict(LAST)
        rec["attempts"] = attempt
        try:
            text = _post_once(base, model, secret, messages, schema, effort, max_tokens, timeout, name, rec)
            obj = extract_json(text) if api() == "anthropic" else json.loads(text)
            errs = validate(obj, schema)
            if errs:
                raise LLMError("schema: " + "; ".join(errs[:5]))
            LAST["elapsed_s"] = round(time.time() - t0, 3)
            _TL.last = dict(LAST)
            rec.update(elapsed_s=LAST["elapsed_s"], result=obj)
            _log(name, rec)
            return obj
        except (LLMError, ValueError, KeyError, TypeError, IndexError, OSError, requests.RequestException) as e:
            errors.append(f"attempt {attempt}: {type(e).__name__}: {_redact(str(e), secret)[:300]}")
            if attempt < MAX_ATTEMPTS:
                time.sleep(BACKOFF * 2 ** (attempt - 1))
    LAST["elapsed_s"] = round(time.time() - t0, 3)
    _TL.last = dict(LAST)
    msg = f"LLM call failed after {MAX_ATTEMPTS} attempts: " + " | ".join(errors)
    rec.update(elapsed_s=LAST["elapsed_s"], error=msg)
    _log(name, rec)
    raise LLMError(msg)


def ping(timeout=15):
    """Is the endpoint answering? openai: GET {base}/models; anthropic: an 8-token message."""
    base, model = endpoint()
    try:
        if api() == "anthropic":
            r = requests.post(f"{base}/v1/messages", json={"model": model, "max_tokens": 8,
                                                           "messages": [{"role": "user", "content": "ping"}]},
                              headers={"x-api-key": key(), "anthropic-version": ANTHROPIC_VERSION,
                                       "content-type": "application/json"}, timeout=timeout)
        else:
            r = requests.get(f"{base}/models", headers={"Authorization": f"Bearer {key()}"}, timeout=timeout)
        return r.status_code == 200
    except (requests.RequestException, OSError, LLMError):
        return False


def confirm_ids():
    ids = set(CONFIRM)
    try:
        c = pd.read_csv(CITIES_CSV, dtype=str)
        if "role" in c.columns:
            ids |= set(c[c["role"] == "confirm"]["city"])
    except OSError:
        pass
    return ids


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def check_input(payload):
    """Whitelist of what may reach the LLM: no tract-level numeric arrays (keys starting with tract / unit holding a
    list of more than one number), no number under a key containing "test" (test period, lawtest cities; plan 4) and
    no confirmation-city id anywhere (keys, values, free text)."""
    ids = confirm_ids()
    pat = re.compile("(?<![0-9A-Za-z])(?<![0-9]\\.)(" + "|".join(re.escape(i) for i in sorted(ids)) + ")(?![0-9A-Za-z])(?!\\.[0-9])")

    def walk(o, path):
        if isinstance(o, dict):
            for k, v in o.items():
                k = str(k)
                if k in ids:
                    raise ValueError(f"{path}: key {k!r} is a confirmation city id (sealed)")
                if (k.startswith("tract") or k.startswith("unit")) and isinstance(v, list) and len(v) > 1 \
                        and all(_num(x) for x in v):
                    raise ValueError(f"{path}.{k}: tract-level numeric array is not allowed in an LLM input")
                if "test" in k.lower() and (_num(v) or (isinstance(v, list) and v and all(_num(x) for x in v))):
                    raise ValueError(f"{path}.{k}: test-period / lawtest numbers are not allowed in an LLM input")
                walk(v, f"{path}.{k}")
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, f"{path}[{i}]")
        elif isinstance(o, str):
            m = pat.search(o)
            if m:
                raise ValueError(f"{path}: mentions confirmation city {m.group(1)} (sealed)")
    walk(payload, "$")


def run_point(name, in_path, out_path, fn):
    """File interface of a call point: read JSON -> check_input -> fn(payload) -> JSON with bookkeeping. The output
    never contains the key; a rejected input raises before any call and writes nothing; a failed call (LLMError)
    writes result null with the error and does not raise."""
    payload = json.loads(Path(in_path).read_text())
    check_input(payload)
    t0 = time.time()
    _TL.last = {"attempt": 0, "effort": None, "model": model_name(), "elapsed_s": 0.0}
    try:
        result, error = fn(payload), None
    except LLMError as e:
        result, error = None, str(e)
    st = _TL.last
    out = {"point": name, "input": Path(in_path).name, "model": st["model"], "effort": st["effort"],
           "elapsed_s": round(time.time() - t0, 3), "attempt": st["attempt"], "result": result, "error": error}
    Path(out_path).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return out
