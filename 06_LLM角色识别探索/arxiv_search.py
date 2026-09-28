"""Query the arXiv API for prior work on role / variable discovery with LLMs."""
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

QUERIES = [
    ('role discovery + equations',
     'all:"role" AND all:"equation discovery" AND all:"language model"'),
    ('physical variable identification',
     'abs:"identify the physical" AND abs:"symbolic regression"'),
    ('variable semantics / names',
     'abs:"variable names" AND abs:"symbolic regression"'),
    ('discover variables then equations',
     'abs:"discover" AND abs:"variables" AND abs:"governing equations" AND abs:"language model"'),
    ('quantity discovery',
     'all:"quantity discovery" OR all:"physical quantity identification"'),
    ('PDE + LLM + prior knowledge',
     'abs:"partial differential equations" AND abs:"large language model" AND abs:"discovery"'),
    ('unit / dimension aware SR',
     'abs:"dimensional analysis" AND abs:"symbolic regression"'),
    ('latent variable / coordinate discovery',
     'abs:"discovering coordinates" OR abs:"latent variables" AND abs:"governing equations"'),
    ('semantics of measurements',
     'abs:"semantic" AND abs:"equations" AND abs:"large language model" AND abs:"discovery"'),
    ('anonymous / anonymised benchmarks',
     'abs:"anonymised" OR abs:"anonymized" AND abs:"equation discovery"'),
]

ATOM = "{http://www.w3.org/2005/Atom}"


def query(expr: str, limit: int = 15) -> list[dict]:
    url = ("http://export.arxiv.org/api/query?"
           + urllib.parse.urlencode({"search_query": expr, "start": 0,
                                     "max_results": limit,
                                     "sortBy": "relevance"}))
    request = urllib.request.Request(url, headers={"User-Agent": "role-probe/0.1"})
    with urllib.request.urlopen(request, timeout=90) as response:
        tree = ET.fromstring(response.read())
    out = []
    for entry in tree.findall(f"{ATOM}entry"):
        out.append({
            "id": entry.findtext(f"{ATOM}id", ""),
            "title": " ".join((entry.findtext(f"{ATOM}title", "") or "").split()),
            "published": (entry.findtext(f"{ATOM}published", "") or "")[:7],
            "summary": " ".join((entry.findtext(f"{ATOM}summary", "") or "").split()),
        })
    return out


def main() -> int:
    seen = {}
    for label, expr in QUERIES:
        try:
            hits = query(expr)
        except Exception as error:
            print(f"### {label}: FAILED {type(error).__name__}: {error}")
            continue
        print(f"\n### {label}   ({len(hits)} hits)")
        for hit in hits:
            key = hit["id"].rsplit("/", 1)[-1]
            if key in seen:
                continue
            seen[key] = hit
            print(f"  [{hit['published']}] {hit['title'][:96]}")
            print(f"      {hit['id'].rsplit('/', 1)[-1]}  {hit['summary'][:200]}")
        time.sleep(3)
    here = os.path.dirname(os.path.abspath(__file__))
    json.dump(list(seen.values()), open(os.path.join(here, "arxiv_hits.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\ntotal unique papers: {len(seen)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
