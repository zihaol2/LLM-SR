"""Reversible, moderate search-prompt compression. No numeric rounding or ID aliases."""

import ast
import copy
import re
from collections import Counter

from .memory import fit_version

INSTRUCTIONS = """Compact notation: nodes is the expression dictionary; all listed nodes are selectable.
working_set.fitted_terms rows are [ID, coefficient, params, optional fixed_coefficient=false].
Join expressions by ID, and feedback by field/state. Feedback contributions are [ID, relative_contribution];
errors/complexity appear once in working_set. Values retain full stored precision.
Missing sampled_history means {} unless sampled_history_default is supplied.
Reply with original IDs. Group tests sharing parent_state, keep and joint=false in one candidates list;
respect the distinct-ID quota and give brief concrete reasons.
"""


def encode_search(packet):
    if packet.get("schema") != "frontier-search-v1":
        raise ValueError("Only search packets support compact encoding")
    if packet.get("edges") != [] or packet.get("one_hop_neighbors_included") is not False:
        raise ValueError("Cannot drop nonempty graph information")
    out = copy.deepcopy(packet)
    out["schema"] = "frontier-search-compact-v1"
    out.pop("edges")
    out.pop("one_hop_neighbors_included")
    out["sampled_history"] = {mid: value for mid, value in out["sampled_history"].items() if value}
    if out["node_columns"] != [
        "id",
        "expression",
        "spatial_order",
        "construction_depth",
        "selectable",
    ]:
        raise ValueError("Unknown node columns")
    if not all(row[-1] == 1 for row in out["nodes"]):
        raise ValueError("Non-selectable node requires uncompressed input")
    out["nodes"] = [row[:-1] for row in out["nodes"]]
    out["node_columns"] = out["node_columns"][:-1]
    expressions = {row[0]: row[1] for row in out["nodes"]}
    for field, rows in out["working_set"].items():
        for index, row in enumerate(rows):
            if row["candidate"] != f"K{index+1:02d}" or row["mechanism_count"] != len(
                row["fitted_terms"]
            ):
                raise ValueError("Unexpected candidate metadata")
            if row["model_version"] != fit_version(dict(terms=row["fitted_terms"])):
                raise ValueError("Candidate hash does not match terms")
            row.pop("mechanism_count")
            row.pop("model_version")
            terms = []
            for term in row["fitted_terms"]:
                if set(term) != {
                    "id",
                    "expr",
                    "params",
                    "coefficient",
                    "fixed_coefficient",
                }:
                    raise ValueError("Unknown fitted-term field; preserve full packet instead")
                if expressions.get(term["id"]) != term["expr"]:
                    raise ValueError("Dictionary and fitted expression differ")
                value = [term["id"], term["coefficient"], term["params"]]
                if term["fixed_coefficient"]:
                    value.append(True)
                terms.append(value)
            row["fitted_terms"] = terms
        states = {row["state"]: row for row in rows}
        for row in out["feedback"][field]:
            parent = states[row["state"]]
            for key in ["MSE", "normalized_MSE"]:
                if row[key] != parent[key]:
                    raise ValueError("Feedback mismatch")
            if row["complexity"] != parent["complexity"]:
                raise ValueError("Feedback mismatch")
            for key in ["MSE", "normalized_MSE", "complexity"]:
                row.pop(key)
            coefficients = {term[0]: term[1] for term in parent["fitted_terms"]}
            values = []
            for term in row["contributions"]:
                if (
                    set(term) != {"id", "coefficient", "relative_contribution"}
                    or term["coefficient"] != coefficients[term["id"]]
                ):
                    raise ValueError("Contribution metadata mismatch")
                values.append([term["id"], term["relative_contribution"]])
            row["contributions"] = values
    return out


def decode_search(packet):
    if packet.get("schema") != "frontier-search-compact-v1":
        raise ValueError("Wrong wire schema")
    out = copy.deepcopy(packet)
    out["schema"] = "frontier-search-v1"
    out["edges"] = []
    out["one_hop_neighbors_included"] = False
    out["nodes"] = [row + [1] for row in out["nodes"]]
    out["node_columns"] = out["node_columns"] + ["selectable"]
    out["sampled_history"] = {
        mid: out["sampled_history"].get(mid, {}) for mid in out["sampled_ids"]
    }
    expressions = {row[0]: row[1] for row in out["nodes"]}
    for field, rows in out["working_set"].items():
        for index, row in enumerate(rows):
            row["fitted_terms"] = [
                dict(
                    id=t[0],
                    expr=expressions[t[0]],
                    coefficient=t[1],
                    params=t[2],
                    fixed_coefficient=t[3] if len(t) == 4 else False,
                )
                for t in row["fitted_terms"]
            ]
            row["candidate"] = f"K{index+1:02d}"
            row["mechanism_count"] = len(row["fitted_terms"])
            row["model_version"] = fit_version(dict(terms=row["fitted_terms"]))
        states = {row["state"]: row for row in rows}
        for row in out["feedback"][field]:
            parent = states[row["state"]]
            for key in ["MSE", "normalized_MSE", "complexity"]:
                row[key] = parent[key]
            coefficients = {term["id"]: term["coefficient"] for term in parent["fitted_terms"]}
            row["contributions"] = [
                dict(id=t[0], coefficient=coefficients[t[0]], relative_contribution=t[1])
                for t in row["contributions"]
            ]
    return out


TEMPLATE_INSTRUCTIONS = """
expression_templates defines @E tokens: recursively substitute them in expressions and other templates.
Templates are subexpressions, not candidate nodes. Spatial derivatives remain explicit PD applications:
alpha is an integer tuple ordered by the declared coordinates; (1,0,0) means FIRST-axis derivative,
not a Laplacian. Expand templates or reference an original node ID in reply expressions.
When sampled_count is present, only the first sampled_count node rows are eligible sampled IDs, in order.
sampled_history_columns names array columns. sampled_history_default applies unless an ID overrides it;
null means {}. This default overrides the general missing-history rule.
"""


def encode_templates(packet):
    """Factor repeated expression substrings; exact byte-for-byte expansion."""
    out = encode_search(packet)
    out["schema"] = "frontier-search-templates-v2"
    counts = Counter()
    for row in out["nodes"]:
        text = row[1]
        tree = ast.parse(text, mode="eval")
        seen = set()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Call, ast.BinOp, ast.UnaryOp)):
                continue
            derivative_names = {
                "PD",
                "d_axis",
                "D",
                "Dx",
                "Dy",
                "Dz",
                "d_dx",
                "d_dy",
                "d_dz",
                "Lap",
                "lap",
                "laplace",
                "laplacian",
                "Laplacian",
            }
            if any(
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id in derivative_names
                for child in ast.walk(node)
            ):
                continue
            part = ast.get_source_segment(text, node)
            if part and len(part) >= 12:
                seen.add(part)
        counts.update(seen)
    phrases = [text for text, n in counts.items() if n >= 4]
    phrases.sort(key=lambda text: (-(counts[text] - 1) * (len(text) - 5) + len(text), text))
    phrases = phrases[:128]
    names = {text: f"@E{i+1}" for i, text in enumerate(phrases)}

    def substitute(text, allowed):
        allowed = sorted(allowed, key=lambda part: (-len(part), part))
        if not allowed:
            return text
        pattern = re.compile("|".join(re.escape(part) for part in allowed))
        return pattern.sub(lambda match: names[match.group(0)], text)

    out["expression_templates"] = {
        names[text]: substitute(text, [p for p in phrases if len(p) < len(text)])
        for text in phrases
    }
    for row in out["nodes"]:
        row[1] = substitute(row[1], phrases)
    tuple_pattern = re.compile(r"\(\s*[0-4]\s*,\s*[0-4]\s*,\s*[0-4]\s*\)")
    tuple_texts = set()
    for text in [row[1] for row in out["nodes"]] + list(out["expression_templates"].values()):
        tuple_texts.update(tuple_pattern.findall(text))
    spellings = {re.sub(r"\s+", "", text): text for text in sorted(tuple_texts)}
    if len(spellings) != len(tuple_texts):
        raise ValueError("Mixed tuple formatting requires original encoding")
    out["multi_index_spelling"] = spellings

    def compress_tuples(text):
        return tuple_pattern.sub(lambda match: re.sub(r"\s+", "", match.group(0)), text)

    for row in out["nodes"]:
        row[1] = compress_tuples(row[1])
    out["expression_templates"] = {
        name: compress_tuples(text) for name, text in out["expression_templates"].items()
    }
    history = out["sampled_history"]
    if any(set(value) != {"sampled", "tested"} for value in history.values()):
        raise ValueError("Unknown sampled history fields")
    out["sampled_history_columns"] = ["sampled", "tested"]
    values = {
        mid: ((history[mid]["sampled"], history[mid]["tested"]) if mid in history else None)
        for mid in out["sampled_ids"]
    }
    default = Counter(values.values()).most_common(1)[0][0] if values else None
    out["sampled_history_default"] = list(default) if default is not None else None
    out["sampled_history"] = {
        mid: list(value) if value is not None else None
        for mid, value in values.items()
        if value != default
    }
    if out["nodes"] != sorted(out["nodes"], key=lambda row: row[0]):
        raise ValueError("Unrecognized original node ordering")
    by_id = {row[0]: row for row in out["nodes"]}
    sampled = out.pop("sampled_ids")
    sampled_set = set(sampled)
    out["sampled_count"] = len(sampled)
    out["nodes"] = [by_id[mid] for mid in sampled] + [
        row for row in out["nodes"] if row[0] not in sampled_set
    ]
    return out


def decode_templates(packet):
    if packet.get("schema") not in (
        "frontier-search-templates-v1",
        "frontier-search-templates-v2",
    ):
        raise ValueError("Wrong template schema")
    out = copy.deepcopy(packet)
    out["schema"] = "frontier-search-compact-v1"
    if "sampled_count" in out:
        count = out.pop("sampled_count")
        out["sampled_ids"] = [row[0] for row in out["nodes"][:count]]
        out["nodes"].sort(key=lambda row: row[0])
    spellings = out.pop("multi_index_spelling", {})
    templates = out.pop("expression_templates")
    templates.update(out.pop("multi_indices", {}))
    cache = {}
    pending = set()

    def expand(text):
        def replacement(match):
            name = match.group(0)
            if name in cache:
                return cache[name]
            if name not in templates or name in pending:
                raise ValueError("Invalid expression template dependency")
            pending.add(name)
            value = expand(templates[name])
            pending.remove(name)
            cache[name] = value
            return value

        return re.sub(r"@[EQ]\d+", replacement, text)

    for row in out["nodes"]:
        row[1] = expand(row[1])
        row[1] = re.sub(
            r"\([0-4],[0-4],[0-4]\)",
            lambda match: spellings.get(match.group(0), match.group(0)),
            row[1],
        )
    columns = out.pop("sampled_history_columns")
    if columns != ["sampled", "tested"]:
        raise ValueError("Unknown history columns")
    default = out.pop("sampled_history_default")
    overrides = out["sampled_history"]
    out["sampled_history"] = {
        mid: (
            dict(zip(columns, overrides.get(mid, default)))
            if overrides.get(mid, default) is not None
            else {}
        )
        for mid in out["sampled_ids"]
    }
    return decode_search(out)


def expand_response_expressions(policy, wire_packet):
    """Expand only mathematical reply fields, using this request's definitions.

    Raw paid replies remain unchanged. Never infer derivative orders from prose.
    Unknown references remain rejected, with an explicit per-expression audit.
    """
    result = copy.deepcopy(policy)
    definitions = dict(wire_packet.get("expression_templates", {}))
    definitions.update(wire_packet.get("multi_indices", {}))
    audit = []

    def expand(text, stack=()):
        def replace(match):
            name = match.group(0)
            if name not in definitions or name in stack:
                raise ValueError("Unknown or cyclic reply template " + name)
            return expand(definitions[name], stack + (name,))

        return re.sub(r"@[EQ]\d+", replace, text)

    def item_fields(item, path):
        if not isinstance(item, dict):
            return
        for field in ["expr", "new"]:
            before = item.get(field)
            if not isinstance(before, str) or not re.search(r"@[EQ]\d+", before):
                continue
            try:
                after = expand(before)
                item[field] = after
                audit.append(
                    dict(
                        path=path + "." + field,
                        before=before,
                        after=after,
                        status="expanded",
                    )
                )
            except ValueError as error:
                audit.append(
                    dict(
                        path=path + "." + field,
                        before=before,
                        status="rejected",
                        reason=str(error),
                    )
                )

    proposals = result.get("new_nodes", [])
    if isinstance(proposals, list):
        for i, item in enumerate(proposals):
            item_fields(item, f"new_nodes[{i}]")
    for section in ["revisions", "edits"]:
        mapping = result.get(section, {})
        if not isinstance(mapping, dict):
            continue
        for field, items in mapping.items():
            if isinstance(items, list):
                for i, item in enumerate(items):
                    item_fields(item, f"{section}.{field}[{i}]")
    return result, audit


BLOCK_INSTRUCTIONS = """
Node blocks: node_columns applies after reconstruction. Each node_blocks item has
partial, spatial_order, construction_depth, and rows [original_ID, expression].
If partial is a tuple string, the actual expression is PD(expression,partial);
if partial is null, use expression directly. The tuple is an EXPLICIT ordered
spatial multi-index, never a Laplacian abbreviation. Order/depth apply to every
row in the block. Sort reconstructed nodes by original_ID; sampled_positions
lists zero-based positions of eligible sampled IDs in that sorted dictionary,
in display order. All other nodes are working/history terms only.
Reply with the original IDs. Blocks and templates are notation, not new nodes.
"""


def encode_blocks(packet):
    """Share derivative multi-indices and node metadata without changing expressions."""
    out = encode_templates(packet)
    out["schema"] = "frontier-search-blocks-v1"
    nodes = out.pop("nodes")
    count = out.pop("sampled_count")
    sampled = [row[0] for row in nodes[:count]]
    positions = {row[0]: index for index, row in enumerate(sorted(nodes))}
    out["sampled_positions"] = [positions[mid] for mid in sampled]
    blocks = {}
    for mid, expression, order, depth in nodes:
        match = re.fullmatch(r"PD\((.+),(\([0-4, ]+\))\)", expression)
        operand, alpha = match.groups() if match else (expression, None)
        key = (alpha, order, depth)
        block = blocks.setdefault(
            key,
            dict(partial=alpha, spatial_order=order, construction_depth=depth, rows=[]),
        )
        block["rows"].append([mid, operand])
    out["node_blocks"] = list(blocks.values())
    return out


def decode_blocks(packet):
    if packet.get("schema") != "frontier-search-blocks-v1":
        raise ValueError("Wrong node-block schema")
    out = copy.deepcopy(packet)
    out["schema"] = "frontier-search-templates-v2"
    nodes = []
    for block in out.pop("node_blocks"):
        for mid, expression in block["rows"]:
            if block["partial"] is not None:
                expression = "PD(" + expression + "," + block["partial"] + ")"
            nodes.append([mid, expression, block["spatial_order"], block["construction_depth"]])
    nodes.sort(key=lambda row: row[0])
    positions = out.pop("sampled_positions")
    chosen = set(positions)
    out["nodes"] = [nodes[index] for index in positions] + [
        row for index, row in enumerate(nodes) if index not in chosen
    ]
    out["sampled_count"] = len(positions)
    return decode_templates(out)
