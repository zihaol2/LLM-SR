"""Prior-work scan: has anyone made an LLM discover the *roles* of variables first?

OpenAlex is used because the arXiv API endpoint refuses multi-word queries through
this network. OpenAlex indexes arXiv preprints as well.
"""
import json
import os
import time
import urllib.parse
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (compatible; lit-scan/1.0; mailto:research@example.org)"}
MAIL = "research@example.org"

QUERIES = [
    "identify physical variables governing equations language model",
    "discover the roles of variables equation discovery",
    "variable identification symbolic regression scientific discovery",
    "semantic interpretation of measured variables equation discovery",
    "discover variables and governing equations jointly",
    "physical meaning of variables symbolic regression",
    "equation discovery without prior knowledge of variables",
    "latent variable discovery governing equations",
    "dimensional analysis units symbolic regression",
    "variable names leakage benchmark equation discovery",
    "unknown physical quantities discovery machine learning",
    "role of variables in scientific equations interpretation LLM",
]


def abstract_of(work: dict) -> str:
    inverted = work.get("abstract_inverted_index") or {}
    pairs = []
    for word, positions in inverted.items():
        pairs.extend((position, word) for position in positions)
    return " ".join(word for _, word in sorted(pairs))


def search(query: str, limit: int = 8) -> dict:
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode({
        "filter": f"title_and_abstract.search:{query}",
        "per-page": limit,
        "mailto": MAIL,
        "select": "display_name,publication_year,doi,abstract_inverted_index,primary_location",
    })
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> int:
    seen = {}
    for query in QUERIES:
        try:
            payload = search(query)
        except Exception as error:                                   # noqa: BLE001
            print(f"## {query}: FAILED {type(error).__name__}")
            time.sleep(2)
            continue
        print(f"\n## {query}   (total {payload['meta']['count']})")
        for work in payload.get("results", []):
            title = (work.get("display_name") or "").strip()
            key = title.lower()[:60]
            if key in seen:
                continue
            seen[key] = {
                "title": title, "year": work.get("publication_year"),
                "doi": work.get("doi"), "abstract": abstract_of(work),
            }
            print(f"   [{work.get('publication_year')}] {title[:92]}")
            print(f"      {work.get('doi')}")
            print(f"      {abstract_of(work)[:230]}")
        time.sleep(1.5)
    here = os.path.dirname(os.path.abspath(__file__))
    json.dump(list(seen.values()), open(os.path.join(here, "lit_hits.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\nunique works: {len(seen)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
