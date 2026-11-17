"""One bounded working set is both the displayed equations and legal parents."""
import gzip
import hashlib
import json
import math
from tokenizers import Tokenizer
from . import core as C, rotation
from .feedback import diagnose
from .final_selection import cards_for, RULES, selection_prompt
from .wire import encode_search, decode_search, INSTRUCTIONS as WIRE_INSTRUCTIONS
from .wire import encode_templates, decode_templates, TEMPLATE_INSTRUCTIONS

with gzip.open(C.ASSETS/'tokenizer.json.gz', 'rt', encoding='utf8') as file:
    TOKENIZER = Tokenizer.from_str(file.read())


def tokens(text):
    return len(TOKENIZER.encode(text, add_special_tokens=False).ids)+4


def make_prompt(case, registry, archives, sampled, stats, history, number, preview, cards=None):
    final = number == C.SETTINGS['rounds']
    working = {field:cards_for(archive) for field, archive in archives.items()}
    if final:
        packet = dict(schema='frontier-final-v1', completed_round=number, candidates=working,
                      reference_access=False)
        instructions = RULES+'\nCANDIDATES: '
        payload = working
    else:
        active = {term['id'] for rows in working.values() for row in rows for term in row['fitted_terms']}
        preview_ids = {mid for info in preview.values() for example in info.get('examples', []) for mid in example['support']}
        ids = set(sampled) | active | preview_ids
        packet = dict(schema='frontier-search-v1', completed_round=number, planning_round=number+1,
            sampled_ids=sampled, edges=[], one_hop_neighbors_included=False,
            node_columns=['id','expression','spatial_order','construction_depth','selectable'],
            nodes=[[mid,registry.nodes[mid]['expr'],registry.nodes[mid]['order'],registry.nodes[mid]['depth'],
                    int(registry.nodes[mid]['candidate'])] for mid in sorted(ids)],
            working_set=working,
            feedback={field:[diagnose(archive.data,field,state) for state in archive.view()] for field,archive in archives.items()},
            history_preview=preview,
            sampled_history={mid:stats.get(mid,{}) for mid in sampled},
            forced_untested={field:rotation.forced_for(archive,sampled,number+1) for field,archive in archives.items()},
            last_planning_execution=history[-1].get('edit_counts',{}) if history else {},
            last_planning_errors=history[-1].get('policy_normalization_errors',[]) if history else [])
        quota = rotation.quota(sampled)
        schedule_note=('The original ten-group, two-pass display schedule is preserved: each round displays '
            'one 10% group, and all initial roots are displayed over ten rounds. YOU choose which '
            'approximately 10% of the displayed group to test, approximately 1% of the global library. '
            'Choose untested IDs (sampled_history tested=0) when there are enough; their parents and '
            'kept terms are also your choice. The second pass has the SAME reduced quota, with no '
            'catch-up to full coverage. History recombination and online revisions remain separate.\n'
            if C.SETTINGS.get('coverage_schedule')=='global_1pct' else '')
        instructions = schedule_note+f'''Propose mathematical mechanisms and equation edits for observed fields {case['fields']}.
{case['prior']}
Metadata: {json.dumps(case['metadata'],separators=(',',':'))}
Plan round {number+1}. The supplied working_set is EXACTLY the previous round's output,
at most {C.SETTINGS['frontier_limit']} equations per field. It is the sole pool of parents
for coverage tests and explicit edits. There is no separate six-equation summary, no
single round winner, and no candidate veto. Full history remains available to the separate
{C.SETTINGS['history_fits_per_round']}-fit history recombination procedure.
Choose EXACTLY {quota} different sampled IDs per field, reserving forced_untested first.
For each group copy parent_state from working_set.state and a complete keep subset of
that parent's fitted_terms IDs. With joint=false, each candidate is fitted separately
with keep; with joint=true the block is fitted together. Do not test every ID on every
parent: all parents SHARE the same quota. Distribute sensible combinations across the
working set. A coverage equation has <=8 mechanisms; remove terms when a parent is full.
Invalid or missing tests are filled across these SAME parents, never undisplayed parents.
Actively propose up to 8 revisions TOTAL across fields, using parent_state, keep, add,
reason. Combine/replace/delete mechanisms; temporarily worse historical structures can
be recovered by history recombination. Propose up to {C.SETTINGS['new_roots_per_call']}
new_nodes TOTAL with id=new1/new2, expr and reason, then reference aliases in revision add.
If no revisions/new nodes, state generation_reason. All proposals are fitted or explicitly
logged as rejected. Raw edits is accepted as an alias for revisions. A complete expression
replacement may use parent_state and new="COMPLETE_EXPRESSION".
The round stores every valid fit. It fully repairs up to 10 preliminary frontier routes:
recursive deletions, all available graph-polynomial prefixes and existing-core replacements.
There is no eight-fit repair cap. Intermediate repairs may exceed eight mechanisms; parents
and output equations may not. Keep whole expressions compact; do not expand chain-rule derivatives.
The next working set comes ONLY from previous working-set equations plus all valid fits
of this round, including history recombinations and every repair intermediate within capacity.
MSE/complexity Pareto filtering is numerical, without LLM winner selection. If more than
10 remain, endpoints and geometric bends in normalized complexity/log-MSE are retained.
The original scalar Score guides the inherited numeric repair routines and history ranking,
but does NOT choose the working set. Only after round 20 does an LLM select the final equation.
No reference equation, benchmark identity, prior result or correctness label is available.
Feedback has aggregate MSE, target-normalized error and term contributions; no regional or
time-bin residual breakdowns. A small coefficient alone does not establish redundancy.
Grammar: fields {case['fields']}, coordinates {case['coordinates']}, t; constants a,b,k,p0,p1;
+,-,*,/,**; Exp,Log,Sin,Cos,Tan,ArcSin,ArcCos,ArcTan,Sinh,Cosh,Tanh,ArcSinh,ArcCosh,ArcTanh,
Sqrt,CubeRoot; PD(E,alpha) with ordered spatial derivative multi-index, order<=4.
Expressions<=40 operations and <=600 characters; independent internal term parameters.
Return JSON ONLY with actual IDs, for example schema (replace ALL placeholders):
{{"tests":{{"u":[{{"parent_state":"SHOWN_STATE","keep":["KEPT_ID"],"candidates":["SAMPLED_ID"],"joint":false}}]}},
"revisions":{{"u":[{{"parent_state":"SHOWN_STATE","keep":["KEPT_ID"],"add":["new1"],"reason":"mathematical edit"}}]}},
"new_nodes":[{{"id":"new1","expr":"EXPRESSION","reason":"hypothesis"}}],
"generation_reason":"brief mathematical rationale","reason":"brief plan"}}
INPUT: '''
        payload = packet
    codec='original';codec_error=None
    uncompressed_tokens=tokens(instructions+json.dumps(payload,ensure_ascii=False,separators=(',',':')))
    if not final and C.SETTINGS.get('compact_prompt',False):
        try:
            encoded=encode_search(packet)
            if decode_search(encoded)!=packet:
                raise ValueError('Exact semantic roundtrip failed')
            candidate_instructions=instructions.removesuffix('INPUT: ')+WIRE_INSTRUCTIONS+'\nINPUT: '
            candidate_text=candidate_instructions+json.dumps(encoded,ensure_ascii=False,separators=(',',':'))
            if tokens(candidate_text)<uncompressed_tokens:
                payload=encoded;instructions=candidate_instructions;codec='moderate-reversible-v1'
                packet['wire_packet']=encoded
            if len(case['coordinates'])==3:
                templated=encode_templates({k:v for k,v in packet.items() if k!='wire_packet'})
                if decode_templates(templated)!={k:v for k,v in packet.items() if k!='wire_packet'}:
                    raise ValueError('Exact template expansion failed')
                template_instructions=candidate_instructions.removesuffix('INPUT: ')+TEMPLATE_INSTRUCTIONS+'\nINPUT: '
                template_text=template_instructions+json.dumps(templated,ensure_ascii=False,separators=(',',':'))
                if tokens(template_text)<tokens(instructions+json.dumps(payload,ensure_ascii=False,separators=(',',':'))):
                    payload=templated;instructions=template_instructions;codec='lossless-expression-templates-v2'
                    packet['wire_packet']=templated
        except (ValueError,KeyError,TypeError) as error:
            # Unexpected metadata never disappears to save tokens.
            codec_error=str(error)
    text = instructions+json.dumps(payload,ensure_ascii=False,separators=(',',':'))
    if final:
        text=selection_prompt(working)
        uncompressed_tokens=tokens(text)
    audit = dict(schema=packet['schema'],compressed_tokens=tokens(text),max_input_tokens=C.SETTINGS['max_input_tokens'],
        sent_prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),reference_access=False,before_API=True,
        codec=codec,codec_error=codec_error,uncompressed_tokens=uncompressed_tokens,
        semantic_roundtrip=codec in ('moderate-reversible-v1','lossless-expression-templates-v1','lossless-expression-templates-v2'))
    if audit['compressed_tokens'] > audit['max_input_tokens']:
        raise RuntimeError(f"Input {audit['compressed_tokens']} exceeds token limit {audit['max_input_tokens']}; no request sent")
    packet['compression_audit'] = audit
    return text, packet
