"""Which candidate arms can actually be called right now, and how fast?"""
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

PROMPT = ("Complete this python function: def u_t_pred(x, u, params, d_dx):\n"
          "    return params[0] * d_dx(u, order=2)\n"
          "Reply with code only.")
MODELS = ["glm-5.3-flash-free", "glm-5.3-flash", "glm-5.3", "deepseek-v4-flash-free",
          "deepseek-flash-free", "deepseek-v4-flash", "deepseek-v4-pro",
          "gemini-3.5-flash-lite", "gemini-3.6-flash", "gpt-5.4-mini",
          "claude-haiku-4-5", "kimi-k3", "grok-4.6"]

print(f"{'model':<24} {'status':<26} {'latency':>8}  code?")
for model in MODELS:
    body = json.dumps({"model": model, "max_tokens": 200, "temperature": 0.2,
                       "messages": [{"role": "user", "content": PROMPT}]}).encode()
    start = time.time()
    try:
        with urllib.request.urlopen(urllib.request.Request(
                "https://api.teamorouter.cn/v1/chat/completions", data=body,
                headers=HEAD), timeout=90) as response:
            payload = json.loads(response.read().decode("utf-8"))
        text = payload["choices"][0]["message"]["content"]
        print(f"{model:<24} {'OK':<26} {time.time() - start:7.1f}s  {'d_dx' in text}")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")
        kind = re.search(r'"type":"([^"]+)"', detail)
        message = re.search(r'"message":"([^"]{0,60})', detail)
        label = f"HTTP {error.code} {kind.group(1) if kind else ''}"
        print(f"{model:<24} {label:<26} {time.time() - start:7.1f}s  "
              f"{message.group(1) if message else ''}")
    except Exception as error:                                     # noqa: BLE001
        print(f"{model:<24} {type(error).__name__:<26} {time.time() - start:7.1f}s")
    time.sleep(0.4)
