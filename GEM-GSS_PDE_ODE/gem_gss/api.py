"""At most one outbound HTTP attempt per round, including after a restart."""

import json
import time
import requests
from . import core as C
import hashlib
from . import compression


def credentials():
    import os

    key = os.environ.get("GEM_GSS_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Set GEM_GSS_API_KEY in the environment before a live run")
    if not C.SETTINGS["model"] or not C.SETTINGS["base_url"]:
        raise RuntimeError("Set GEM_GSS_MODEL and GEM_GSS_BASE_URL, or pass --model and --base-url")
    return key


def parse(text):
    text = text.strip()
    if text.startswith("```"):
        text = text[text.find("\n") + 1 : text.rfind("```")]
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        text = text[text.index("{") :]
        stack, quoted, escaped = ([], False, False)
        for c in text:
            if quoted:
                if escaped:
                    escaped = False
                elif c == "\\":
                    escaped = True
                elif c == '"':
                    quoted = False
            elif c == '"':
                quoted = True
            elif c in "{[":
                stack.append(c)
            elif c in "}]":
                expected = "{" if c == "}" else "["
                if not stack or stack.pop() != expected:
                    raise ValueError("Mismatched JSON delimiters")
        if quoted:
            raise ValueError("Truncated JSON string")
        value = json.loads(text + "".join(("}" if c == "{" else "]" for c in reversed(stack))))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    tests = value.get("tests")
    if isinstance(tests, dict):
        for key in ("edits", "revisions", "new_nodes", "reason", "generation_reason", "vetoes", "lift_vetoes"):
            if key in tests and key not in value:
                value[key] = tests.pop(key)
    return value


def once(prompt, folder, number, transport=None):
    gate = None
    if transport is None:
        audit = C.read(folder.parent / "inputs" / f"{number:02d}_graph.json")["compression_audit"]
        if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != audit["sent_prompt_sha256"]:
            raise RuntimeError("Pre-dispatch prompt hash mismatch")
        if (compression.tokens(prompt) != audit["compressed_tokens"]
                or audit["compressed_tokens"] > audit["max_input_tokens"]):
            raise RuntimeError("Pre-dispatch token gate failed; no HTTP request")
        gate = audit
    folder.mkdir(parents=True, exist_ok=True)
    prefix = folder / f"round{number:02d}"
    request_path = prefix.with_name(prefix.name + "_request.json")
    record_path = prefix.with_name(prefix.name + "_record.json")
    payload = dict(
        model=C.SETTINGS["model"],
        messages=[dict(role="user", content=prompt)],
        temperature=C.SETTINGS["temperature"],
        max_tokens=C.SETTINGS["max_tokens"],
        thinking={"type": "disabled"},
    )
    if request_path.exists() and C.read(request_path) != payload:
        raise RuntimeError("Cached request changed; do not resend this round")
    C.dump(request_path, payload)
    prefix.with_suffix(".txt").write_text(prompt, encoding="utf-8")
    if record_path.exists():
        record = C.read(record_path)
        response_path = prefix.with_name(prefix.name + "_response.json")
        if record.get("status") == "failed_no_retry" and response_path.exists():
            try:
                body = C.read(response_path)
                record["policy"] = parse(body["choices"][0]["message"]["content"])
                record["previous_status"] = record["status"]
                record["status"] = "accepted"
                record["local_schema_recovery"] = True
                C.dump(record_path, record)
            except (ValueError, KeyError, TypeError):
                pass
        # A failed/uncertain request is never resent implicitly. The search can
        # continue with explicit unavailable-feedback telemetry and quota repair.
        return (record.get("policy", {}), record)
    key = credentials() if transport is None else "test-only"
    record = dict(
        round=number,
        http_attempts=1,
        status="sent_unconfirmed",
        requested_model=C.SETTINGS["model"],
        started_unix=time.time(),
        pre_dispatch_token_gate=gate,
    )
    C.dump(record_path, record)
    try:
        response = (transport or requests.post)(
            C.SETTINGS["base_url"] + "/chat/completions",
            headers={"Authorization": "Bearer " + key},
            json=payload,
            timeout=(30, 240),
            allow_redirects=False,
        )
        record["http_status"] = response.status_code
        if response.status_code != 200:
            try:
                error = response.json()
                detail = error.get("error", {})
                record["provider_error_code"] = error.get("code") or (
                    detail.get("code") if isinstance(detail, dict) else None
                )
            except (ValueError, TypeError, AttributeError):
                pass
            raise RuntimeError("Gateway HTTP failure")
        body = response.json()
        C.dump(prefix.with_name(prefix.name + "_response.json"), body)
        record.update(returned_model=body.get("model"), usage=body.get("usage"))
        record["policy"] = parse(body["choices"][0]["message"]["content"])
        record["status"] = "accepted"
    except Exception as err:
        record.update(status="failed_no_retry", error_type=type(err).__name__, policy={})
    record["seconds"] = time.time() - record["started_unix"]
    C.dump(record_path, record)
    return (record["policy"], record)
