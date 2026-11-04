"""Predeclared, pre-dispatch serialization; no responses/references are read."""

import hashlib
import json
from pathlib import Path
import sys
from . import core as C
import gzip
from tokenizers import Tokenizer

TOKENIZER_PATH = C.ASSETS / "tokenizer.json.gz"
with gzip.open(TOKENIZER_PATH, "rt", encoding="utf8") as file:
    TOKENIZER = Tokenizer.from_str(file.read())
PREFIX = "Complete sampled-root and one-hop graph packet: "
HIST = "Historical partial solutions: "
FEED = "Per-route residual, contribution and edit feedback: "
RECENT = "Recent completed rounds: "


def tokens(text):
    return len(TOKENIZER.encode(text, add_special_tokens=False).ids) + 4


def compact(packet):
    nodes = [
        [mid, n["expr"], n["order"], n["depth"], int(n["candidate"])]
        for mid, n in packet["nodes"].items()
    ]
    edges = [
        [a, b, [i for i, o in enumerate(packet["nodes"][b]["operands"]) if o.get("node_id") == a]]
        for a, b in packet["dependency_edges"]
    ]
    result = dict(
        node_columns=["id", "expression", "spatial_order", "construction_depth", "selectable"],
        nodes=nodes,
        edge_columns=["operand", "expression", "operand_positions"],
        edges=edges,
        sampled_ids=packet["sampled_ids"],
        history_columns=["sampled_rounds", "tested_rounds"],
        history={
            mid: [
                packet["nodes"][mid]["history"].get("sampled", 0),
                packet["nodes"][mid]["history"].get("tested", 0),
            ]
            for mid in packet["sampled_ids"]
        },
    )
    assert {r[0]: r[1] for r in nodes} == {mid: n["expr"] for mid, n in packet["nodes"].items()}
    assert [[a, b] for a, b, _ in edges] == packet["dependency_edges"]
    assert all((positions for _, _, positions in edges))
    return result


def rounded(value):
    if isinstance(value, float):
        return float(format(value, ".7g"))
    if isinstance(value, list):
        return [rounded(x) for x in value]
    if isinstance(value, dict):
        return {k: rounded(v) for k, v in value.items()}
    return value


def encode(raw, packet):
    start = raw.index(PREFIX) + len(PREFIX)
    end = raw.index("\n" + HIST, start)
    h0 = end + 1 + len(HIST)
    h1 = raw.index("\n" + FEED, h0)
    f0 = h1 + 1 + len(FEED)
    f1 = raw.index("\n" + RECENT, f0)
    histories = json.loads(raw[h0:h1])
    short_history = {
        f: [
            dict(
                id=x["id"],
                terms=[
                    dict(id=t["id"], coefficient=t["coefficient"], params=t["params"])
                    for t in x["terms"]
                ],
                MSE=x["MSE"],
                C=x["C"],
                score=x["score"],
                visits=x["visits"],
                budget_exhausted=any(
                    (s.get("status") == 0 for s in x.get("optimization", {}).get("starts", []))
                ),
            )
            for x in rows
        ]
        for f, rows in histories.items()
    }
    active = {t["id"] for rows in histories.values() for x in rows for t in x["terms"]}
    assert active <= packet["nodes"].keys()
    text = (
        raw[:start]
        + json.dumps(compact(packet), separators=(",", ":"))
        + "\n"
        + HIST
        + json.dumps(rounded(short_history), separators=(",", ":"))
        + raw[h1:]
    )
    audit = dict(
        schema="GSS-table-v1",
        raw_tokens=tokens(raw),
        compressed_tokens=tokens(text),
        input_reduction=1 - tokens(text) / tokens(raw),
        all_node_expressions_preserved=True,
        sampled_roots=len(packet["sampled_ids"]),
        dictionary_nodes=len(packet["nodes"]),
        edges=len(packet["dependency_edges"]),
        all_incident_edges_and_positions_preserved=True,
        historical_model_parameters_full_precision_private=True,
        before_API=True,
        reference_access=False,
        tokenizer_sha256=hashlib.sha256(TOKENIZER_PATH.read_bytes()).hexdigest(),
    )
    audit["raw_prompt_sha256"] = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    audit["sent_prompt_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if audit["compressed_tokens"] > audit["raw_tokens"] * 0.5:
        raise RuntimeError("Pre-dispatch compression below required 50%; no request may be sent")
    return (text, audit)
