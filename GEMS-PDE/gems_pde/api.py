"""Cached OpenAI-compatible HTTP client with explicit offline mode.

Only non-secret payloads, prompts, replies, usage and statuses are saved.
Invalid JSON replies are logged rather than hidden from usage accounting.
"""

import json
import os
import time
import requests
from . import config
from .io import dump, read


def _validate_json(value, kind):
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    if kind == "initial":
        for layer in ("L0", "G0"):
            if not isinstance(value.get(layer), list):
                raise ValueError("Expected L0 and G0 lists")
            for node in value[layer]:
                if not isinstance(node, dict) or not isinstance(node.get("expr"), str):
                    raise ValueError("Each mechanism needs an expr string")
                if "id" in node and not isinstance(node["id"], str):
                    raise ValueError("Mechanism IDs must be strings")
                if not isinstance(node.get("parents", []), list) or any(
                    not isinstance(v, str) for v in node.get("parents", [])
                ):
                    raise ValueError("Mechanism parents must be ID lists")
    if kind == "policy":
        for key in ("focus_states", "priority_nodes", "edits"):
            if not isinstance(value.get(key, {}), dict):
                raise ValueError("Policy entries must be field mappings")
        if any(not isinstance(v, str) for v in value.get("focus_states", {}).values()):
            raise ValueError("Focus IDs must be strings")
        if any(
            not isinstance(v, list) or any(not isinstance(x, str) for x in v)
            for v in value.get("priority_nodes", {}).values()
        ):
            raise ValueError("Priority mechanisms must be ID lists")
        if any(
            not isinstance(v, list) or any(not isinstance(x, dict) for x in v)
            for v in value.get("edits", {}).values()
        ):
            raise ValueError("Edits must be lists of objects")
        for edits in value.get("edits", {}).values():
            for edit in edits:
                if edit.get("operation") not in ("add", "delete", "replace"):
                    raise ValueError("Unknown edit operation")
                for key in ("add_node_id", "remove_node_id"):
                    if key in edit and edit[key] is not None and not isinstance(edit[key], str):
                        raise ValueError("Edit mechanism IDs must be strings")
    return value


def _call(prompt, path, json_mode=False, kind=None):
    payload = {
        "model": config.SETTINGS["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": config.SETTINGS["temperature"],
        "max_tokens": config.SETTINGS["max_tokens"],
    }
    if config.SETTINGS["thinking"]:
        payload["thinking"] = {"type": config.SETTINGS["thinking"]}
    endpoint = config.endpoint()
    request = {"url": endpoint, "payload": payload}
    request_path = path.with_name(path.stem + "_request.json")
    prompt_path = path.with_name(path.stem + "_prompt.txt")
    if path.exists():
        if (
            not request_path.exists()
            or read(request_path) != request
            or not prompt_path.exists()
            or prompt_path.read_text(encoding="utf-8") != prompt
        ):
            raise RuntimeError("Cached request differs; use a fresh run directory")
        return _validate_json(read(path), kind) if json_mode else path.read_text(encoding="utf-8")
    if config.OFFLINE:
        raise RuntimeError("Offline cache miss: " + str(path))
    key = os.environ.get("PDE_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Set PDE_API_KEY or create .env from .env.example")
    if request_path.exists() and read(request_path) != request:
        raise RuntimeError("An unfinished cached request differs; use a fresh run directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    dump(request_path, request)
    prompt_path.write_text(prompt, encoding="utf-8")
    attempts_path = path.with_name(path.stem + "_attempts.json")
    attempts = read(attempts_path) if attempts_path.exists() else []
    for retry in range(3):
        record = {"attempt": len(attempts) + 1, "requested_model": payload["model"]}
        try:
            response = requests.post(
                endpoint,
                headers={"Authorization": "Bearer " + key},
                json=payload,
                timeout=180,
            )
            record["status"] = response.status_code
            if response.status_code >= 400:
                raise RuntimeError("Provider HTTP status " + str(response.status_code))
            body = response.json()
            record.update(returned_model=body.get("model"), usage=body.get("usage"))
            raw = body["choices"][0]["message"]["content"]
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError("Provider returned empty text")
            path.with_name(path.stem + f'_attempt{record["attempt"]:02d}_raw.txt').write_text(
                raw, encoding="utf-8"
            )
            if json_mode:
                text = raw.strip()
                if text.startswith("```"):
                    text = text[text.find("\n") + 1 : text.rfind("```")]
                result = _validate_json(json.loads(text), kind)
                dump(path, result)
            else:
                result = raw
                path.write_text(raw, encoding="utf-8")
            record["accepted"] = True
            attempts.append(record)
            dump(attempts_path, attempts)
            dump(
                path.with_name(path.stem + "_usage.json"),
                {
                    "requested_model": payload["model"],
                    "returned_model": body.get("model"),
                    "usage": body.get("usage"),
                    "transport_attempts": len(attempts),
                },
            )
            return result
        except Exception as error:
            record.update(accepted=False, error_type=type(error).__name__)
            attempts.append(record)
            dump(attempts_path, attempts)
            if retry == 2:
                raise RuntimeError(
                    "LLM call failed after three attempts; inspect the sanitized attempt log"
                ) from None
            time.sleep(2 * (retry + 1))


def call_json(prompt, path, kind=None):
    return _call(prompt, path, True, kind)


def call_text(prompt, path):
    return _call(prompt, path)
