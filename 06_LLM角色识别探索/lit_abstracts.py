"""Pull the full abstracts of the three closest candidates found in the scan."""
import json
import time
import urllib.parse
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (compatible; lit-scan/1.0; mailto:research@example.org)"}
DOIS = [
    "10.48550/arxiv.2603.06869",   # Symmetry-constrained language-guided program synthesis
    "10.48550/arxiv.2602.19516",   # Pixel2Phys: distilling governing laws from visual dynamics
    "10.48550/arxiv.2510.12618",   # coarse-graining + equation discovery, latent variables
    "10.48550/arxiv.2607.13608",   # automatic ODE discovery for biological systems with LLMs
]


def abstract_of(work: dict) -> str:
    inverted = work.get("abstract_inverted_index") or {}
    pairs = []
    for word, positions in inverted.items():
        pairs.extend((position, word) for position in positions)
    return " ".join(word for _, word in sorted(pairs))


for doi in DOIS:
    url = "https://api.openalex.org/works/https://doi.org/" + urllib.parse.quote(doi) + \
          "?mailto=research@example.org"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                    timeout=60) as response:
            work = json.loads(response.read().decode("utf-8"))
        print("=" * 78)
        print(f"[{work.get('publication_year')}] {work.get('display_name')}")
        print(f"doi: {work.get('doi')}   cited: {work.get('cited_by_count')}")
        print(abstract_of(work)[:1500])
    except Exception as error:                                       # noqa: BLE001
        print(f"{doi}: FAILED {type(error).__name__} {error}")
    time.sleep(1.5)
