"""Try provider-prefixed spellings, in case TypeSafe/Jev is reachable another way."""
import json
import os
import re
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env = {}
for line in open(os.path.join(ROOT, ".env"), encoding="utf-8"):
    match = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(\S.*)$", line)
    if match:
        env[match.group(1)] = match.group(2).strip()
key = env.get("TEAMOROUTER_API_KEY") or env.get("API_KEY")
HEAD = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

VARIANTS = ["typesafe/jev", "tev", "type-safe", "typesafe", "jev-pro",
            "claude-fable-5", "gemini-3.5-flash-lite", "glm-5.3-flash"]
for model in VARIANTS:
    body = json.dumps({"model": model, "max_tokens": 10,
                       "messages": [{"role": "user", "content": "say ok"}]}).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(
                "https://api.teamorouter.cn/v1/chat/completions", data=body,
                headers=HEAD), timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
        print(f"  {model:<24} OK -> {payload['choices'][0]['message']['content'][:30]!r}")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")
        kind = re.search(r'"type":"([^"]+)"', detail)
        print(f"  {model:<24} HTTP {error.code} {kind.group(1) if kind else ''}")
    except Exception as error:                                     # noqa: BLE001
        print(f"  {model:<24} {type(error).__name__}")
