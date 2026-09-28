"""What is the 'TypeSafe (Jev)' entry, and does it have a callable API path?"""
import re
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (compatible; catalogue-check/1.0)"}
with urllib.request.urlopen(urllib.request.Request(
        "https://api.teamorouter.cn/", headers=UA), timeout=45) as response:
    page = response.read().decode("utf-8", "replace")

print(f"page length {len(page)}")
for match in re.finditer(r"Jev", page, re.I):
    start = max(0, match.start() - 500)
    snippet = page[start:match.end() + 500]
    print("\n--- context ---")
    print(" ".join(snippet.replace("\\u003c", "<").replace("\\u003e", ">").split())[:900])

print("\n--- entries that DO have an api path near 'TypeSafe' ---")
for match in re.finditer(r"data-path=\\?\"([^\"\\]+)\\?\"", page):
    value = match.group(1)
    if value and ("type" in value.lower() or "safe" in value.lower() or "jev" in value.lower()):
        print("   ", value)

print("\n--- count of non-empty data-path attributes ---")
paths = re.findall(r"data-path=\\?\"([^\"\\]+)\\?\"", page)
print(f"   {len(paths)} entries, e.g. {sorted(set(paths))[:12]}")
