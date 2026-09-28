"""Second pass: phrase-anchored queries, then pull the abstracts of the survivors."""
import json
import os
import time
import urllib.parse
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (compatible; lit-scan/1.0; mailto:research@example.org)"}
MAIL = "research@example.org"

PHRASES = [
    '"state variables" "equation discovery" "language model"',
    '"physical variables" "symbolic regression"',
    '"identify" "physical quantities" "observational data"',
    '"latent dynamics" "equation discovery"',
    '"governing equations" "variable roles"',
    '"semantic roles" equations scientific discovery',
    '"discover" "physical laws" "minimal prior knowledge"',
    '"coordinates" "governing equations" autoencoder',
]


def abstract_of(work: dict) -> str:
    inverted = work.get("abstract_inverted_index") or {}
    pairs = []
    for word, positions in inverted.items():
        pairs.extend((position, word) for position in positions)
    return " ".join(word for _, word in sorted(pairs))


def main() -> int:
    seen = {}
    for phrase in PHRASES:
        url = "https://api.openalex.org/works?" + urllib.parse.urlencode({
            "search": phrase, "per-page": 6, "mailto": MAIL,
            "sort": "relevance_score:desc",
            "select": "display_name,publication_year,doi,abstract_inverted_index,cited_by_count",
        })
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                        timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as error:                                   # noqa: BLE001
            print(f"## {phrase}: FAILED {type(error).__name__}")
            time.sleep(2)
            continue
        print(f"\n## {phrase}   (total {payload['meta']['count']})")
        for work in payload.get("results", []):
            title = (work.get("display_name") or "").strip()
            key = title.lower()[:60]
            if key in seen or not title:
                continue
            abstract = abstract_of(work)
            seen[key] = {"title": title, "year": work.get("publication_year"),
                         "doi": work.get("doi"), "abstract": abstract,
                         "cited": work.get("cited_by_count")}
            print(f"   [{work.get('publication_year')}] {title[:95]}")
            print(f"      {work.get('doi')}  cited={work.get('cited_by_count')}")
            print(f"      {abstract[:260]}")
        time.sleep(1.5)
    here = os.path.dirname(os.path.abspath(__file__))
    json.dump(list(seen.values()), open(os.path.join(here, "lit_hits2.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\nunique works: {len(seen)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
