"""Self-contained planning prompts with verified lossless payload compression."""

import gzip
import hashlib
import json
from functools import lru_cache

from tokenizers import Tokenizer

from . import core as C
from .feedback import diagnose
from .final_selection import cards_for, selection_prompt
from .hypothesis import forced_for, quota
from .wire import (
    BLOCK_INSTRUCTIONS,
    TEMPLATE_INSTRUCTIONS,
    decode_blocks,
    decode_search,
    decode_templates,
    encode_blocks,
    encode_search,
    encode_templates,
)
from .wire import INSTRUCTIONS as WIRE_INSTRUCTIONS


@lru_cache(maxsize=1)
def tokenizer():
    with gzip.open(C.ASSETS / "tokenizer.json.gz", "rt", encoding="utf8") as file:
        return Tokenizer.from_str(file.read())


def tokens(text):
    return len(tokenizer().encode(text, add_special_tokens=False).ids) + 4


def serialize(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def search_packet(registry, archives, sampled, stats, history, number, preview):
    working = {field: cards_for(archive) for field, archive in archives.items()}
    active = {
        term["id"] for rows in working.values() for row in rows for term in row["fitted_terms"]
    }
    preview_ids = {
        mid
        for info in preview.values()
        for example in info.get("examples", [])
        for mid in example["support"]
    }
    ids = set(sampled) | active | preview_ids
    return dict(
        schema="frontier-search-v1",
        completed_round=number,
        planning_round=number + 1,
        sampled_ids=sampled,
        edges=[],
        one_hop_neighbors_included=False,
        node_columns=[
            "id",
            "expression",
            "spatial_order",
            "construction_depth",
            "selectable",
        ],
        nodes=[
            [
                mid,
                registry.nodes[mid]["expr"],
                registry.nodes[mid]["order"],
                registry.nodes[mid]["depth"],
                int(registry.nodes[mid]["candidate"]),
            ]
            for mid in sorted(ids)
        ],
        working_set=working,
        feedback={
            field: [diagnose(archive.data, field, state) for state in archive.view()]
            for field, archive in archives.items()
        },
        history_preview=preview,
        sampled_history={mid: stats.get(mid, {}) for mid in sampled},
        forced_untested={
            field: forced_for(archive, sampled, number + 1) for field, archive in archives.items()
        },
        last_planning_execution=history[-1].get("edit_counts", {}) if history else {},
        last_planning_errors=(
            history[-1].get("policy_normalization_errors", []) if history else []
        ),
    )


def search_instructions(case, sampled, number):
    schedule = (
        "Ten initial groups, two passes: test about 10% of this displayed group (about 1% global). "
        "Prefer untested IDs when enough exist; no second-pass catch-up. "
        if C.SETTINGS["coverage_schedule"] == "global_1pct"
        else "Two passes over ten initial groups; reserve forced_untested first. "
    )
    return f"""Plan round {number + 1} for observed fields {case['fields']} on ordered coordinates {case['coordinates']} and time t.
{case['prior']}
Metadata: {serialize(case.get('metadata', {}))}
{schedule}Choose EXACTLY {quota(sampled)} distinct sampled IDs per field, shared across all parents.
Parents and keep IDs must come from the supplied working_set (at most {C.SETTINGS['frontier_limit']} equations/field).
Tests: parent_state, complete keep subset, candidates, joint. joint=false fits candidates separately with keep;
joint=true fits the block together. Maximum {C.SETTINGS['max_terms']} mechanisms/equation; remove terms from full parents.
Missing/invalid tests are repaired using these same parents. No round winner or veto.
Propose up to {C.SETTINGS['max_revision_requests']} revisions TOTAL across fields: parent_state, keep, add, reason;
combine/replace/delete terms. edits is an alias. A replacement may instead give new="COMPLETE_EXPRESSION".
Propose up to {C.SETTINGS['new_roots_per_call']} new_nodes TOTAL: id=new1/new2, expr, reason; use aliases in revision add.
If none, give generation_reason. Proposals are fitted or logged rejected.
Separate history recombination uses all archived fits and shares {C.SETTINGS['history_fits_per_round']} fits across fields.
All valid round fits join previous working-set fits. Repairs fully explore up to 10 preliminary frontier routes:
recursive deletions, graph-polynomial prefixes, existing-core replacements, without an eight-fit cap.
Intermediate repairs may exceed term capacity; parents and output equations may not.
The next working_set uses numerical MSE/complexity Pareto filtering, retaining endpoints and normalized
complexity/log-MSE bends when over capacity. Scalar Score ranks history/repairs only. Final LLM choice follows
round {C.SETTINGS['rounds']}; no terminal fitting. Do not expand chain-rule derivatives.
Feedback contains aggregate errors and term contributions. A small coefficient alone does not imply redundancy.
No reference equation, benchmark identity, previous result or correctness label is available.
Grammar: declared fields/coordinates,t; parameters a,b,k,p0,p1; +,-,*,/,**;
Exp,Log,Sin,Cos,Tan,ArcSin,ArcCos,ArcTan,Sinh,Cosh,Tanh,ArcSinh,ArcCosh,ArcTanh,Sqrt,CubeRoot;
PD(E,alpha), ordered spatial integer multi-index, total spatial order<=4; <=40 operations, <=600 characters.
Internal parameters are independent per mechanism. Return JSON only using actual IDs:
{{"tests":{{"FIELD":[{{"parent_state":"STATE","keep":["ID"],"candidates":["SAMPLED_ID"],"joint":false}}]}},
"revisions":{{"FIELD":[{{"parent_state":"STATE","keep":["ID"],"add":["new1"],"reason":"edit"}}]}},
"new_nodes":[{{"id":"new1","expr":"EXPRESSION","reason":"hypothesis"}}],"generation_reason":"rationale","reason":"plan"}}
"""


def compress_search(packet, instructions):
    """Choose the cheapest exact representation, preserving IDs and numeric precision."""
    original = instructions + "INPUT: " + serialize(packet)
    best = (original, "original", None)
    errors = []
    codecs = [
        (encode_search, decode_search, WIRE_INSTRUCTIONS, "moderate-reversible-v1"),
        (
            encode_templates,
            decode_templates,
            WIRE_INSTRUCTIONS + TEMPLATE_INSTRUCTIONS,
            "lossless-expression-templates-v2",
        ),
    ]
    codecs.append(
        (
            encode_blocks,
            decode_blocks,
            WIRE_INSTRUCTIONS + TEMPLATE_INSTRUCTIONS + BLOCK_INSTRUCTIONS,
            "lossless-node-blocks-v1",
        )
    )
    for encode, decode, notation, name in codecs:
        try:
            encoded = encode(packet)
            if decode(encoded) != packet:
                raise ValueError("Exact semantic roundtrip failed")
            text = instructions + notation + "\nINPUT: " + serialize(encoded)
            if tokens(text) < tokens(best[0]):
                best = (text, name, encoded)
        except (ValueError, KeyError, TypeError, SyntaxError) as error:
            errors.append(f"{name}: {error}")
    return best, errors


def make_prompt(case, registry, archives, sampled, stats, history, number, preview):
    final = number == C.SETTINGS["rounds"]
    codec, wire_packet, errors = "original", None, []
    if final:
        working = {field: cards_for(archive) for field, archive in archives.items()}
        packet = dict(
            schema="frontier-final-v1",
            completed_round=number,
            candidates=working,
            reference_access=False,
        )
        text = selection_prompt(working)
        uncompressed_tokens = tokens(text)
    else:
        packet = search_packet(registry, archives, sampled, stats, history, number, preview)
        instructions = search_instructions(case, sampled, number)
        text = instructions + "INPUT: " + serialize(packet)
        uncompressed_tokens = tokens(text)
        if C.SETTINGS["compact_prompt"]:
            (text, codec, wire_packet), errors = compress_search(packet, instructions)
        if wire_packet is not None:
            packet["wire_packet"] = wire_packet
    audit = dict(
        schema=packet["schema"],
        compressed_tokens=tokens(text),
        max_input_tokens=C.SETTINGS["max_input_tokens"],
        sent_prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),
        reference_access=False,
        before_API=True,
        codec=codec,
        codec_error="; ".join(errors) or None,
        uncompressed_tokens=uncompressed_tokens,
        semantic_roundtrip=wire_packet is not None,
    )
    if audit["compressed_tokens"] > audit["max_input_tokens"]:
        raise RuntimeError(
            f"Input {audit['compressed_tokens']} exceeds token limit {audit['max_input_tokens']}; no request sent"
        )
    packet["compression_audit"] = audit
    return text, packet
