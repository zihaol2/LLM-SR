"""Find out what 'jev' refers to on the gateway: list ids, then probe spellings."""
import json
import os
import re
import time
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

with urllib.request.urlopen(urllib.request.Request(
        "https://api.teamorouter.cn/v1/models", headers=HEAD), timeout=60) as response:
    payload = json.loads(response.read().decode("utf-8"))
ids = sorted(item["id"] for item in payload.get("data", []))
print(f"gateway serves {len(ids)} ids:")
print("  " + ", ".join(ids))

for needle in ("jev", "jv", "grok", "kimi", "qwen", "claude"):
    hits = [i for i in ids if needle in i.lower()]
    print(f"\nids containing '{needle}': {hits or 'none'}")

print("\nprobing spellings directly:")
for candidate in ("jev", "Jev", "jev-1", "jev-v1", "j-ev", "jev5"):
    body = json.dumps({"model": candidate, "max_tokens": 10,
                       "messages": [{"role": "user", "content": "say ok"}]}).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(
                "https://api.teamorouter.cn/v1/chat/completions", data=body,
                headers=HEAD), timeout=60) as response:
            text = json.loads(response.read().decode("utf-8"))
        print(f"  {candidate:<8} OK -> {text['choices'][0]['message']['content'][:40]!r}")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:160]
        print(f"  {candidate:<8} HTTP {error.code}: {detail}")
    except Exception as error:                                     # noqa: BLE001
        print(f"  {candidate:<8} {type(error).__name__}: {error}")
    time.sleep(0.4)
