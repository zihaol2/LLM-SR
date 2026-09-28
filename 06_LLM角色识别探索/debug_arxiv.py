"""Find an arXiv query form that the endpoint accepts."""
import re
import urllib.parse
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (compatible; lit-scan/1.0)"}
CANDIDATES = [
    ("quoted-single", 'all:"equation discovery"'),
    ("quoted-and", 'all:"physical role" AND all:"equation discovery"'),
    ("abs-quoted", 'abs:"variable identification"'),
    ("plain", "physical role equation discovery"),
    ("ti-abs", 'ti:"equation discovery" AND abs:"language model"'),
]

URL_FORMS = [
    ("literal-space", "search_query=all:equation discovery&max_results=2"),
    ("literal-space2", "search_query=all:equation discovery AND all:language model&max_results=2"),
]

for name, suffix in URL_FORMS:
    url = "https://export.arxiv.org/api/query?" + suffix
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                    timeout=60) as response:
            body = response.read().decode("utf-8", "replace")
        titles = [t.strip()[:60] for t in re.findall(r"<title>(.*?)</title>", body, re.S)[1:4]]
        print(f"{name}: OK entries={body.count('<entry>')} {titles}")
    except Exception as error:                              # noqa: BLE001
        print(f"{name}: FAIL {type(error).__name__} {error}")
    print(f"   url: {url[:160]}")

import json

UA2 = {"User-Agent": "Mozilla/5.0 (compatible; lit-scan/1.0; mailto:research@example.org)"}
oa = ("https://api.openalex.org/works?" + urllib.parse.urlencode({
    "filter": "title_and_abstract.search:identify physical variables governing equations language model",
    "per-page": 8, "mailto": "research@example.org",
    "select": "display_name,publication_year,doi,abstract_inverted_index"}))
try:
    with urllib.request.urlopen(urllib.request.Request(oa, headers=UA2), timeout=60) as r:
        payload = json.loads(r.read().decode("utf-8"))
    print(f"\nOpenAlex filter form: {payload['meta']['count']} results")
    for work in payload.get("results", [])[:8]:
        print(f"  [{work.get('publication_year')}] {work.get('display_name','')[:90]}")
except Exception as error:                                   # noqa: BLE001
    print(f"\nOpenAlex filter form: FAIL {type(error).__name__} {error}")
