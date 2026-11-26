"""Requirement-DAG construction, validation, fallback, and graph differences."""

from __future__ import annotations

import ast
import difflib
import hashlib
import json
import re
from collections import Counter, deque
from dataclasses import dataclass
from typing import Protocol

from .operations import (
    DEFAULT_OPERATION_SPECS,
    OperationSpec,
    canonical_operation,
    is_semantic_operation,
    is_transparent_operation,
    operation_catalog,
    operation_spec,
)
from .program_graph import output_dependency_subgraph

REQUIREMENT_GRAPH_PROMPT_VERSION = "requirement_dag_v22_explicit_semantic_coverage"


REQUIREMENT_GRAPH_SCHEMA = {
    "type": "object",
    "required": ["nodes", "output"],
    "additionalProperties": False,
    "properties": {
        "nodes": {
            "type": "array", "minItems": 1,
            "items": {
                "type": "object", "required": ["id", "op", "inputs", "span"],
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string", "minLength": 1},
                    "op": {"type": "string", "minLength": 1},
                    "inputs": {"type": "array", "items": {"type": "string"}},
                    "span": {"type": "string"},
                    "roles": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "output": {
            "oneOf": [
                {"type": "string", "minLength": 1},
                {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 1}},
            ],
        },
    },
}


class LLMClient(Protocol):
    def complete(self, system: str, user: str, max_tokens: int = 2600): ...


def _extract_json(text: str) -> dict:
    stripped = text.strip()
    if "```" in stripped:
        match = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.I | re.S)
        if match:
            stripped = match.group(1)
    start, end = stripped.find("{"), stripped.rfind("}")
    if start < 0 or end < start:
        raise ValueError("response_contains_no_json_object")
    value = json.loads(stripped[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("requirement_graph_must_be_object")
    return value


def validate_requirement_graph(
    graph: dict, registry: dict[str, OperationSpec] | None = None,
) -> list[str]:
    registry = registry or DEFAULT_OPERATION_SPECS
    issues: list[str] = []
    if not isinstance(graph, dict) or set(graph) - {"nodes", "output"}:
        issues.append("schema:root_properties")
    nodes = graph.get("nodes") if isinstance(graph, dict) else None
    if not isinstance(nodes, list) or not nodes:
        return sorted(set([*issues, "schema:nodes_nonempty_array"]))
    ids: list[str] = []
    by_id: dict[str, dict] = {}
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            issues.append(f"schema:node_{index}_object")
            continue
        if set(node) - {"id", "op", "inputs", "span", "roles"}:
            issues.append(f"schema:node_{index}_properties")
        node_id = node.get("id")
        op = node.get("op")
        inputs = node.get("inputs")
        span = node.get("span")
        roles = node.get("roles", [])
        if not isinstance(node_id, str) or not node_id:
            issues.append(f"schema:node_{index}_id")
            continue
        if node_id in by_id:
            issues.append(f"duplicate_node_id:{node_id}")
        ids.append(node_id)
        by_id[node_id] = node
        # Transparent CadQuery context operations (workplanes, selectors and
        # sketch positioning) are legal dependency carriers.  They are ignored
        # by semantic graph counters, but rejecting them here forced otherwise
        # correct LLM DAGs into the rule fallback.
        if not isinstance(op, str) or canonical_operation(op) not in registry:
            issues.append(f"unknown_operation:{op}")
        if not isinstance(inputs, list) or not all(isinstance(item, str) for item in inputs):
            issues.append(f"schema:node_{node_id}_inputs")
            inputs = []
        if not isinstance(span, str):
            issues.append(f"schema:node_{node_id}_span")
        if not isinstance(roles, list) or not all(isinstance(item, str) for item in roles):
            issues.append(f"schema:node_{node_id}_roles")
            roles = []
        spec = registry.get(canonical_operation(op)) if isinstance(op, str) else None
        if spec:
            if len(inputs) < spec.min_inputs or (spec.max_inputs is not None and len(inputs) > spec.max_inputs):
                issues.append(f"invalid_input_count:{node_id}:{len(inputs)}")
            if spec.roles:
                if len(roles) != len(inputs):
                    issues.append(f"invalid_role_count:{node_id}")
                elif not roles or roles[0] != "base" or any(role != "tool" for role in roles[1:]):
                    issues.append(f"invalid_boolean_roles:{node_id}")
            elif roles:
                issues.append(f"roles_not_allowed:{node_id}")
    for node in nodes:
        if not isinstance(node, dict) or not isinstance(node.get("inputs"), list):
            continue
        for parent in node["inputs"]:
            if not isinstance(parent, str):
                continue
            if parent not in by_id:
                issues.append(f"missing_input:{node.get('id')}:{parent}")
    output = graph.get("output")
    if isinstance(output, str):
        outputs = [output]
    elif isinstance(output, list) and output and all(isinstance(value, str) for value in output):
        outputs = list(dict.fromkeys(output))
    else:
        outputs = []
        issues.append(f"invalid_output:{output}")
    for value in outputs:
        if value not in by_id:
            issues.append(f"invalid_output:{value}")
    colour: dict[str, int] = {}
    def visit(node_id: str) -> None:
        colour[node_id] = 1
        for parent in by_id[node_id].get("inputs", []):
            if not isinstance(parent, str):
                continue
            if parent not in by_id:
                continue
            if colour.get(parent) == 1:
                issues.append(f"cycle:{parent}->{node_id}")
            elif colour.get(parent, 0) == 0:
                visit(parent)
        colour[node_id] = 2
    for node_id in ids:
        if colour.get(node_id, 0) == 0:
            visit(node_id)
    valid_outputs = [value for value in outputs if value in by_id]
    if valid_outputs:
        reachable: set[str] = set()
        frontier = list(valid_outputs)
        while frontier:
            node_id = frontier.pop()
            if node_id in reachable or node_id not in by_id:
                continue
            reachable.add(node_id)
            frontier.extend(str(parent) for parent in by_id[node_id].get("inputs") or [])
        for node_id in ids:
            if node_id not in reachable:
                issues.append(f"node_not_in_output_dependency:{node_id}")
    return sorted(set(issues))


_CLOSES_TO_START = re.compile(
    r"\b(?:back|return(?:ing)?)\s+to\s+(?:the\s+)?(?:origin|start(?:ing)?\s+point)\b",
    flags=re.IGNORECASE,
)
_COORDINATE_PAIR = re.compile(
    r"[\[(]\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*[,，]\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*[\])]"
)
_GENERIC_SKETCH_CONTAINER = re.compile(
    r"(?:\b(?:new\s+)?face\b|\btwo[- ]dimensional\s+sketch\b|\b2d\s+sketch\b|"
    r"\bcoordinate\s+system\b|二维草图|新(?:的)?面|草图平面|坐标系)",
    flags=re.IGNORECASE,
)
_EXPLICIT_PROFILE_EVIDENCE = re.compile(
    r"(?:\b(?:polygon|polyline|rectangle|rectangular|outline|contour|vertices?|points?|"
    r"lines?|segments?|arcs?|slots?)\b|多边形|矩形|轮廓|顶点|坐标|线段|圆弧|槽)",
    flags=re.IGNORECASE,
)


def canonicalize_requirement_graph(graph: dict) -> dict:
    """Canonicalise sketch closure without inventing geometry operations.

    Besides removing a duplicated return-to-start line, this converts an LLM's
    synthetic "close the whole multi-loop sketch" node into ordinary CadQuery
    data flow.  Circles, ellipses, profiles and slots are already closed, while
    a multi-input close merely assembles contours; neither is a real additional
    ``close()`` operation in the program graph.
    """
    nodes = [dict(node) for node in graph.get("nodes", []) if isinstance(node, dict)]
    by_id = {str(node.get("id")): node for node in nodes}
    consumers: dict[str, set[str]] = {}
    for node in nodes:
        for source in node.get("inputs", []):
            consumers.setdefault(str(source), set()).add(str(node.get("id")))
    # An LLM occasionally emits a generic ``profile`` merely for creating a
    # sketch/face and then emits the real circle/line/arc operation beneath it.
    # That container is not a second geometric contour.  Remove it only when
    # its own span is generic, it has concrete sketch consumers, and it has no
    # explicit profile evidence.  This keeps real polygon/profile operations.
    concrete_sketch_ops = {"circle", "ellipse", "line", "arc", "spline", "slot"}
    container_ids: set[str] = set()
    for node in nodes:
        node_id = str(node.get("id"))
        span = str(node.get("span", ""))
        child_ids = consumers.get(node_id, set())
        if (
            canonical_operation(node.get("op", "")) == "profile"
            and child_ids
            and _GENERIC_SKETCH_CONTAINER.search(span)
            and not _EXPLICIT_PROFILE_EVIDENCE.search(span)
            and all(
                canonical_operation(by_id.get(child_id, {}).get("op", "")) in concrete_sketch_ops
                for child_id in child_ids
            )
        ):
            container_ids.add(node_id)
    if container_ids:
        for candidate in nodes:
            rewritten: list[str] = []
            for source in candidate.get("inputs", []):
                source_id = str(source)
                if source_id in container_ids:
                    rewritten.extend(str(value) for value in by_id[source_id].get("inputs", []))
                else:
                    rewritten.append(source_id)
            candidate["inputs"] = list(dict.fromkeys(rewritten))
        nodes = [node for node in nodes if str(node.get("id")) not in container_ids]
        by_id = {str(node.get("id")): node for node in nodes}
        consumers = {}
        for node in nodes:
            for source in node.get("inputs", []):
                consumers.setdefault(str(source), set()).add(str(node.get("id")))
    removed: set[str] = set()
    for node in nodes:
        inputs = list(node.get("inputs", []))
        if canonical_operation(node.get("op", "")) != "close" or len(inputs) != 1:
            continue
        previous = by_id.get(str(inputs[0]))
        if previous is None or canonical_operation(previous.get("op", "")) != "line":
            continue
        previous_id = str(previous.get("id"))
        if consumers.get(previous_id, set()) != {str(node.get("id"))}:
            continue
        evidence = f"{previous.get('span', '')} {node.get('span', '')}"
        closes_by_coordinates = False
        first = previous
        seen: set[str] = set()
        while canonical_operation(first.get("op", "")) == "line":
            first_id = str(first.get("id"))
            if first_id in seen:
                break
            seen.add(first_id)
            parents = [str(value) for value in first.get("inputs", [])]
            if len(parents) != 1 or canonical_operation(by_id.get(parents[0], {}).get("op", "")) != "line":
                break
            first = by_id[parents[0]]
        first_pairs = _COORDINATE_PAIR.findall(str(first.get("span", "")))
        previous_pairs = _COORDINATE_PAIR.findall(str(previous.get("span", "")))
        if first_pairs and previous_pairs:
            start = tuple(float(value) for value in first_pairs[0])
            end = tuple(float(value) for value in previous_pairs[-1])
            closes_by_coordinates = all(abs(left - right) <= 1e-9 for left, right in zip(start, end))
        if not (_CLOSES_TO_START.search(evidence) or closes_by_coordinates):
            continue
        node["inputs"] = list(previous.get("inputs", []))
        removed.add(previous_id)
    nodes = [node for node in nodes if str(node.get("id")) not in removed]

    def redirect(old_id: str, new_id: str) -> None:
        for candidate in nodes:
            candidate["inputs"] = [
                new_id if str(source) == old_id else source
                for source in candidate.get("inputs", [])
            ]

    output = graph.get("output")
    def replace_output_id(old_id: str, new_id: str) -> None:
        nonlocal output
        if isinstance(output, list):
            output = list(dict.fromkeys(
                new_id if str(value) == old_id else str(value) for value in output
            ))
        elif str(output) == old_id:
            output = new_id
    def _ancestors(start_id: str, node_map: dict[str, dict]) -> set[str]:
        seen_ids: set[str] = set()
        stack = [str(src) for src in node_map.get(start_id, {}).get("inputs", [])]
        while stack:
            curr = stack.pop()
            if curr in seen_ids or curr not in node_map:
                continue
            seen_ids.add(curr)
            stack.extend(str(src) for src in node_map[curr].get("inputs", []))
        return seen_ids

    def _chain_contours(contour_ids: list[str], node_map: dict[str, dict]) -> None:
        """Link sibling sketch contours in order without exceeding max_inputs=1."""
        for previous_id, current_id in zip(contour_ids, contour_ids[1:]):
            if previous_id == current_id or previous_id not in node_map or current_id not in node_map:
                continue
            if current_id in _ancestors(previous_id, node_map):
                continue
            current = node_map[current_id]
            current_inputs = [str(value) for value in current.get("inputs", []) if str(value) in node_map]
            if previous_id in current_inputs:
                continue
            prev_anc = _ancestors(previous_id, node_map)
            # If current's existing inputs are already ancestors of previous_id
            # (e.g. both hang off the same workplane/translate), replace them
            # with [previous_id] so single-input sketch ops keep max_inputs=1.
            if current_inputs and all(src in prev_anc for src in current_inputs):
                current["inputs"] = [previous_id]
            elif not current_inputs:
                current["inputs"] = [previous_id]
            else:
                spec = operation_spec(current.get("op", ""))
                if spec is None or spec.max_inputs is None or len(current_inputs) < spec.max_inputs:
                    current["inputs"] = [*current_inputs, previous_id]

    changed = True
    while changed:
        changed = False
        by_id = {str(node.get("id")): node for node in nodes}
        for node in list(nodes):
            if canonical_operation(node.get("op", "")) != "close":
                continue
            node_id = str(node.get("id"))
            inputs = [str(source) for source in node.get("inputs", []) if str(source) in by_id]
            if len(inputs) > 1:
                _chain_contours(inputs, by_id)
                replacement = inputs[-1]
            elif len(inputs) == 1 and canonical_operation(
                by_id[inputs[0]].get("op", "")
            ) in {"circle", "ellipse", "profile", "slot"}:
                replacement = inputs[0]
            else:
                continue
            redirect(node_id, replacement)
            replace_output_id(node_id, replacement)
            nodes.remove(node)
            changed = True
            break

    # Linearize multi-input profile/revolve nodes whose inputs are sibling sketch
    # contours (so single-input specs do not fail invalid_input_count).
    sketch_ops = {"circle", "ellipse", "line", "arc", "spline", "slot", "profile", "close"}
    by_id = {str(node.get("id")): node for node in nodes}
    for node in nodes:
        op = canonical_operation(node.get("op", ""))
        inputs = [str(source) for source in node.get("inputs", []) if str(source) in by_id]
        if op in {"profile", "revolve"} and len(inputs) > 1:
            if all(canonical_operation(by_id[src].get("op", "")) in sketch_ops for src in inputs):
                _chain_contours(inputs, by_id)
                node["inputs"] = [inputs[-1]]

    # A closed line/arc chain is already a profile.  Some requirement parses
    # wrap it in an additional ``profile`` node, which has no CadQuery call and
    # otherwise creates a permanent false missing-operation penalty.
    changed = True
    while changed:
        changed = False
        by_id = {str(node.get("id")): node for node in nodes}
        for node in list(nodes):
            node_id = str(node.get("id"))
            inputs = [str(value) for value in node.get("inputs", [])]
            if (
                canonical_operation(node.get("op", "")) == "profile"
                and len(inputs) == 1
                and canonical_operation(by_id.get(inputs[0], {}).get("op", "")) == "close"
            ):
                replacement = inputs[0]
                redirect(node_id, replacement)
                replace_output_id(node_id, replacement)
                nodes.remove(node)
                changed = True
                break

    # Attach orphan parallel sketch loops (e.g. inner holes or multiple faces
    # drawn on the same workplane/transform frame) to the consumed sibling
    # sketch leaf on that same frame by chaining them in order, regardless of
    # whether the downstream consumer is extrude/revolve/cut/pocket.
    by_id = {str(node.get("id")): node for node in nodes}
    consumed_ids = {str(source) for node in nodes for source in node.get("inputs", [])}
    output_set = set(output) if isinstance(output, list) else ({str(output)} if output else set())
    frame_ops = {"workplane", "plane", "translate", "rotate", "scale"}
    consumer_feature_ops = {
        "extrude", "revolve", "loft", "sweep", "twist_extrude",
        "cut", "union", "intersect", "pocket", "hole",
    }

    def _nearest_frame(node_id: str) -> str | None:
        frontier = [node_id]
        seen_local: set[str] = set()
        while frontier:
            curr = frontier.pop(0)
            if curr in seen_local or curr not in by_id:
                continue
            seen_local.add(curr)
            curr_op = canonical_operation(by_id[curr].get("op", ""))
            if curr_op in frame_ops:
                return curr
            for src in by_id[curr].get("inputs", []):
                src_id = str(src)
                if src_id in by_id and src_id not in seen_local:
                    frontier.append(src_id)
        return None

    frame_to_consumed_sketch: dict[str, list[str]] = {}
    for node in nodes:
        if canonical_operation(node.get("op", "")) in consumer_feature_ops:
            for src in node.get("inputs", []):
                src_id = str(src)
                if src_id in by_id and canonical_operation(by_id[src_id].get("op", "")) in sketch_ops:
                    frame_id = _nearest_frame(src_id)
                    if frame_id is not None:
                        bucket = frame_to_consumed_sketch.setdefault(frame_id, [])
                        if src_id not in bucket:
                            bucket.append(src_id)

    orphans_by_frame: dict[str, list[str]] = {}
    for node in nodes:
        node_id = str(node.get("id"))
        if node_id in consumed_ids or node_id in output_set:
            continue
        if canonical_operation(node.get("op", "")) not in sketch_ops:
            continue
        frame_id = _nearest_frame(node_id)
        if frame_id is not None:
            orphans_by_frame.setdefault(frame_id, []).append(node_id)

    def _sketch_chain_head(sink_id: str, frame_id: str) -> str:
        """Walk upstream along single-input sketch nodes to find the head attached to frame_id."""
        curr = sink_id
        seen_local: set[str] = set()
        while curr not in seen_local and curr in by_id:
            seen_local.add(curr)
            parents = [str(src) for src in by_id[curr].get("inputs", []) if str(src) in by_id]
            if not parents:
                return curr
            if frame_id in parents:
                return curr
            sketch_parents = [
                p for p in parents
                if canonical_operation(by_id[p].get("op", "")) in sketch_ops
            ]
            if len(sketch_parents) == 1:
                curr = sketch_parents[0]
            else:
                return curr
        return curr

    for frame_id, orphan_ids in orphans_by_frame.items():
        anchors = frame_to_consumed_sketch.get(frame_id) or []
        if not anchors:
            continue
        anchor_id = anchors[-1]
        prev_tail = anchor_id
        for orphan_sink in orphan_ids:
            if orphan_sink == prev_tail or orphan_sink in _ancestors(prev_tail, by_id):
                continue
            head_id = _sketch_chain_head(orphan_sink, frame_id)
            if head_id == prev_tail or prev_tail in _ancestors(head_id, by_id):
                prev_tail = orphan_sink
                continue
            head_node = by_id[head_id]
            head_inputs = [str(src) for src in head_node.get("inputs", []) if str(src) in by_id]
            prev_anc = _ancestors(prev_tail, by_id) | {prev_tail}
            if not head_inputs or all(src in prev_anc for src in head_inputs):
                head_node["inputs"] = [prev_tail]
                prev_tail = orphan_sink
        tail_id = prev_tail
        if tail_id != anchor_id:
            attached_set = set(orphan_ids)
            for candidate in nodes:
                cand_id = str(candidate.get("id"))
                if cand_id in attached_set or cand_id in _ancestors(tail_id, by_id):
                    continue
                candidate["inputs"] = [
                    tail_id if str(source) == anchor_id else source
                    for source in candidate.get("inputs", [])
                ]
            replace_output_id(anchor_id, tail_id)

    # Preserve transform/extrude order exactly.  Rotating or translating a
    # pending sketch changes the workplane used by a later extrusion and is not
    # generally equivalent to transforming the finished solid.
    by_id = {str(node.get("id")): node for node in nodes}
    consumers = {str(source) for node in nodes for source in node.get("inputs", [])}
    sink_ids = [str(node.get("id")) for node in nodes if str(node.get("id")) not in consumers]
    solid_outputs = {
        "extrude", "revolve", "sweep", "loft", "union", "cut", "intersect",
        "rotate", "translate", "mirror", "pattern", "fillet", "chamfer", "scale",
    }
    if (
        len(sink_ids) > 1
        and all(canonical_operation(by_id.get(node_id, {}).get("op", "")) in solid_outputs for node_id in sink_ids)
    ):
        output = sink_ids
    return {**graph, "nodes": nodes, "output": output}


def repair_close_evidence_spans(graph: dict, text: str) -> tuple[dict, list[dict]]:
    """Repair only unsupported ``close`` evidence without inventing evidence.

    Natural-language descriptions often imply closure by making the final
    segment end at the starting point. An LLM may then paraphrase that as
    "back to the start", which fails the exact-substring evidence rule. The
    predecessor segment is direct evidence for the close operation, so reuse
    its already verified span. No operation or dependency is added.
    """
    nodes = [dict(node) for node in graph.get("nodes", []) if isinstance(node, dict)]
    by_id = {str(node.get("id")): node for node in nodes}
    repairs: list[dict] = []
    for node in nodes:
        span = node.get("span")
        if (
            canonical_operation(node.get("op", "")) != "close"
            or not isinstance(span, str)
            or not span
            or span in text
        ):
            continue
        inputs = [str(value) for value in node.get("inputs", [])]
        if len(inputs) != 1:
            continue
        predecessor = by_id.get(inputs[0])
        predecessor_span = predecessor.get("span") if predecessor else None
        if not isinstance(predecessor_span, str) or not predecessor_span or predecessor_span not in text:
            continue
        node["span"] = predecessor_span
        repairs.append({
            "node_id": str(node.get("id")),
            "old_span": span,
            "new_span": predecessor_span,
            "source_node_id": inputs[0],
        })
    return {**graph, "nodes": nodes}, repairs


_ALIGNED_CALL_EVIDENCE = re.compile(
    r"(?m)^Step\s+\d+\s*:\s*(Execute\s+the\s+original\s+reference\s+call\s+"
    r"`(?P<label>[^`]+)`[^\r\n]*)",
    flags=re.IGNORECASE,
)


def repair_aligned_evidence_spans(graph: dict, text: str) -> tuple[dict, list[dict]]:
    """Restore exact evidence spans in the deterministic CAD-Coder text format.

    Nodes are matched only to source steps with the same canonical operation
    and the same occurrence order.  This changes evidence metadata only; it
    never creates, deletes, or reconnects an operation node.
    """
    if "Ordered construction:" not in text:
        return graph, []
    candidates: dict[str, list[str]] = {}
    for match in _ALIGNED_CALL_EVIDENCE.finditer(text):
        op = canonical_operation(match.group("label"))
        if op not in DEFAULT_OPERATION_SPECS:
            continue
        candidates.setdefault(op, []).append(match.group(1).strip())
    if not candidates:
        return graph, []

    nodes = [dict(node) for node in graph.get("nodes", []) if isinstance(node, dict)]
    occurrence: Counter[str] = Counter()
    repairs: list[dict] = []
    for node in nodes:
        op = canonical_operation(node.get("op", ""))
        ordinal = occurrence[op]
        occurrence[op] += 1
        span = node.get("span")
        if isinstance(span, str) and span and span in text:
            continue
        op_candidates = candidates.get(op, [])
        if ordinal >= len(op_candidates):
            continue
        replacement = op_candidates[ordinal]
        node["span"] = replacement
        repairs.append({
            "node_id": str(node.get("id")),
            "operation": op,
            "occurrence": ordinal,
            "old_span": span,
            "new_span": replacement,
        })
    return {**graph, "nodes": nodes}, repairs


def repair_near_exact_evidence_spans(graph: dict, text: str) -> tuple[dict, list[dict]]:
    """Repair a high-overlap paraphrase by copying only its exact source substring.

    This is intentionally limited to evidence metadata.  It never changes an
    operation, dependency, role, multiplicity, or output.  The longest common
    contiguous block must cover at least 65% of the LLM span and contain four
    words, which handles small prefixes such as ``create`` versus ``repeat ...
    with`` without accepting a loosely related sentence.
    """
    nodes = [dict(node) for node in graph.get("nodes", []) if isinstance(node, dict)]
    repairs: list[dict] = []
    for node in nodes:
        span = node.get("span")
        if not isinstance(span, str) or not span or span in text:
            continue
        match = difflib.SequenceMatcher(a=span, b=text, autojunk=False).find_longest_match()
        coverage = match.size / max(len(span.strip()), 1)
        replacement = text[match.b:match.b + match.size].strip()
        strategy = "longest_contiguous_block"
        # Coordinate-list descriptions are frequently correct except for a
        # misplaced conjunction (for example `, and p1, p2` versus
        # `p1, and p2`).  When the ordinary contiguous match is too short,
        # anchor the evidence to the exact coordinate tokens in their original
        # order and copy the shortest corresponding source substring.  This
        # repairs evidence metadata only and cannot invent an operation.
        if coverage < 0.65:
            coordinate_tokens = _EXPLICIT_SKETCH_COORDINATES.findall(span)
            anchored_candidates: list[str] = []
            if len(coordinate_tokens) >= 2:
                search_from = 0
                while True:
                    start = text.find(coordinate_tokens[0], search_from)
                    if start < 0:
                        break
                    cursor = start + len(coordinate_tokens[0])
                    valid = True
                    for token in coordinate_tokens[1:]:
                        position = text.find(token, cursor)
                        if position < 0 or position - start > 320:
                            valid = False
                            break
                        cursor = position + len(token)
                    if valid:
                        anchored_candidates.append(text[start:cursor].strip())
                    search_from = start + 1
            if anchored_candidates:
                replacement = min(anchored_candidates, key=len)
                coverage = 1.0
                strategy = "ordered_coordinate_anchors"
        if (
            coverage < 0.65
            or len(replacement) < 24
            or len(re.findall(r"[A-Za-z0-9]+", replacement)) < 4
            or replacement not in text
        ):
            continue
        node["span"] = replacement
        repairs.append({
            "node_id": str(node.get("id")),
            "operation": canonical_operation(node.get("op", "")),
            "old_span": span,
            "new_span": replacement,
            "coverage": coverage,
            "strategy": strategy,
        })
    return {**graph, "nodes": nodes}, repairs


_TRANSPARENT_HELPER_EVIDENCE = re.compile(
    r"`(?:moveTo|move|center)`|\b(?:moveTo|sketch[-_ ]position)\b",
    flags=re.IGNORECASE,
)


def repair_transparent_alias_nodes(graph: dict) -> tuple[dict, list[dict]]:
    """Remove only proven variable aliases or transparent sketch helpers.

    Unknown operations remain unknown unless a single predecessor carries the
    identical evidence span (a pure assignment alias), or the node evidence
    explicitly names a transparent sketch-position helper.
    """
    nodes = [dict(node) for node in graph.get("nodes", []) if isinstance(node, dict)]
    output = graph.get("output")
    repairs: list[dict] = []

    def replace_output(old_id: str, new_id: str) -> None:
        nonlocal output
        if isinstance(output, list):
            output = list(dict.fromkeys(
                new_id if str(value) == old_id else str(value) for value in output
            ))
        elif str(output) == old_id:
            output = new_id

    container_alias_names = {
        "solid", "body", "part", "shape", "compound", "feature", "result",
        "object", "component", "sketch", "face", "wire", "loop", "contour",
        "model", "geometry", "output",
    }

    changed = True
    while changed:
        changed = False
        by_id = {str(node.get("id")): node for node in nodes}
        for node in list(nodes):
            raw_op = str(node.get("op", ""))
            raw_lower = raw_op.strip().lower().replace("_", "")
            op = canonical_operation(raw_op)
            inputs = [str(value) for value in node.get("inputs", [])]
            if len(inputs) != 1 or inputs[0] not in by_id:
                continue
            predecessor = by_id[inputs[0]]
            span = str(node.get("span", ""))
            same_evidence = bool(span) and span == str(predecessor.get("span", ""))
            transparent_helper = bool(_TRANSPARENT_HELPER_EVIDENCE.search(span))
            container_alias = op == "unknown_op" and (
                raw_lower in container_alias_names
                or raw_op.strip().lower() in container_alias_names
            )
            unary_union = op == "union"
            label_match = re.search(r"`([^`]+)`", span)
            evidence_op = (
                canonical_operation(label_match.group(1)) if label_match else "unknown_op"
            )
            boolean_alias = (
                op in {"cut", "union", "intersect"}
                and evidence_op == canonical_operation(predecessor.get("op", ""))
                and evidence_op != op
            )
            if op != "unknown_op" and not boolean_alias and not unary_union:
                continue
            if not (same_evidence or transparent_helper or boolean_alias or container_alias or unary_union):
                continue
            node_id = str(node.get("id"))
            replacement = inputs[0]
            for consumer in nodes:
                consumer["inputs"] = [
                    replacement if str(source) == node_id else str(source)
                    for source in consumer.get("inputs", [])
                ]
            replace_output(node_id, replacement)
            nodes.remove(node)
            repairs.append({
                "kind": "transparent_alias",
                "node_id": node_id,
                "replacement": replacement,
                "raw_operation": raw_op,
                "reason": (
                    "same_evidence" if same_evidence
                    else "transparent_helper" if transparent_helper
                    else "boolean_alias" if boolean_alias
                    else "container_alias" if container_alias
                    else "unary_union"
                ),
            })
            changed = True
            break
    return {**graph, "nodes": nodes, "output": output}, repairs


_CADCODER_TRANSFORM_NARRATIVE = re.compile(
    r"\bcoordinate\s+system\b.*\bEuler\s+angles?\b|"
    r"\bfirst\s+face\b.*\bscale\s+factor\b",
    flags=re.IGNORECASE | re.DOTALL,
)
_IDENTITY_ROTATION_EVIDENCE = re.compile(
    r"Euler\s+angles?\s*(?:set\s+to|of|are|is)?\s*[:：=]?\s*[\[(]?\s*(?:"
    r"(?:0(?:\.0+)?|zero)(?:\s+degrees?)?\s*[,，]\s*"
    r"(?:0(?:\.0+)?|zero)(?:\s+degrees?)?\s*[,，]\s*(?:and\s+)?"
    r"(?:0(?:\.0+)?|zero)(?:\s+degrees?)?|"
    r"zero(?:\s+degrees?)?(?!\s*[,，])"
    r")",
    flags=re.IGNORECASE,
)
_IDENTITY_TRANSLATION_EVIDENCE = re.compile(
    r"(?:translation\s+vector[^.;]{0,50}(?:set\s+to\s+|also\s+set\s+to\s+)?zero|"
    r"translat\w*[^.;]{0,50}\(\s*0(?:\.0+)?\s*,\s*0(?:\.0+)?\s*,\s*0(?:\.0+)?\s*\))",
    flags=re.IGNORECASE,
)
_NARRATIVE_SCALE_EVIDENCE = re.compile(
    r"\b(?:apply\s+)?(?:a\s+)?scale\s+factor\b[^;\n]{0,80}\bsketch\b|"
    r"\bscale\b[^;\n]{0,40}\bsketch\b[^;\n]{0,40}\bfactor\b|"
    r"\bscale\s+factor\s+of\b",
    flags=re.IGNORECASE,
)
_NONZERO_ANGLE_WORD = re.compile(
    r"\b(?:negative|minus|ninety|forty|thirty|sixty|forty[- ]five|one\s+hundred|two\s+hundred|eighty|twenty|ten)\b",
    flags=re.IGNORECASE,
)
_ANY_NUMBER_LITERAL = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def _is_identity_rotation_span(span: str) -> bool:
    if not span:
        return False
    triple = _EXPLICIT_EULER_TRIPLE.search(span)
    if triple:
        return all(abs(float(triple.group(i))) <= 1e-9 for i in (1, 2, 3))
    if not _IDENTITY_ROTATION_EVIDENCE.search(span):
        return False
    if _NONZERO_ANGLE_WORD.search(span):
        return False
    for match in _ANY_NUMBER_LITERAL.finditer(span):
        try:
            if abs(float(match.group(0))) > 1e-9:
                return False
        except ValueError:
            continue
    return True


def _is_identity_translation_span(span: str) -> bool:
    if not span:
        return False
    triple = _EXPLICIT_TRANSLATION_TRIPLE.search(span)
    if triple:
        return all(abs(float(triple.group(i))) <= 1e-9 for i in (1, 2, 3))
    if not _IDENTITY_TRANSLATION_EVIDENCE.search(span):
        return False
    if _NONZERO_ANGLE_WORD.search(span):
        return False
    for match in _ANY_NUMBER_LITERAL.finditer(span):
        try:
            if abs(float(match.group(0))) > 1e-9:
                return False
        except ValueError:
            continue
    return True
_EXPLICIT_SKETCH_COORDINATES = re.compile(
    r"\(\s*[-+]?\d+(?:\.\d+)?\s*,\s*[-+]?\d+(?:\.\d+)?\s*\)"
)

_SIGNED_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_EXPLICIT_EULER_TRIPLE = re.compile(
    rf"Euler\s+angles?\s*(?:set\s+to|of|are|is)?\s*[:：=]?\s*[\[(]?\s*"
    rf"({_SIGNED_NUMBER})\s*(?:degrees?)?\s*[,，]\s*"
    rf"({_SIGNED_NUMBER})\s*(?:degrees?)?\s*[,，]\s*(?:and\s+)?"
    rf"({_SIGNED_NUMBER})\s*(?:degrees?)?",
    flags=re.IGNORECASE,
)
_EXPLICIT_TRANSLATION_TRIPLE = re.compile(
    rf"(?:translation\s+vector|平移向量)[^\d+\-.]{{0,40}}"
    rf"[\[(]?\s*({_SIGNED_NUMBER})\s*[,，]\s*({_SIGNED_NUMBER})\s*[,，]\s*"
    rf"(?:and\s+)?({_SIGNED_NUMBER})",
    flags=re.IGNORECASE,
)
_COORDINATE_FRAME_BEFORE_SKETCH = re.compile(
    r"(?:start|begin|first)[\s\S]{0,120}(?:coordinate\s+system|workplane)[\s\S]{0,180}"
    r"(?:Euler\s+angles?|translation\s+vector)[\s\S]{0,180}(?:next|then)[\s\S]{0,80}"
    r"(?:draw|sketch|create\s+(?:a\s+)?(?:line|circle|arc|profile))",
    flags=re.IGNORECASE,
)


def explicit_semantic_coverage_issues(graph: dict, text: str) -> list[str]:
    """Report explicit spatial semantics omitted or misplaced by an LLM DAG.

    The checker never adds graph nodes.  It only creates actionable retry
    issues, so an accepted requirement DAG still originates from the LLM.
    """
    nodes = [node for node in graph.get("nodes", []) if isinstance(node, dict)]
    by_id = {str(node.get("id")): node for node in nodes}
    operations = Counter(canonical_operation(node.get("op", "")) for node in nodes)
    issues: list[str] = []
    nonidentity_euler = [
        match for match in _EXPLICIT_EULER_TRIPLE.finditer(text)
        if any(abs(float(match.group(index))) > 1e-9 for index in (1, 2, 3))
    ]
    nonidentity_translation = [
        match for match in _EXPLICIT_TRANSLATION_TRIPLE.finditer(text)
        if any(abs(float(match.group(index))) > 1e-9 for index in (1, 2, 3))
    ]
    if nonidentity_euler and operations["rotate"] == 0:
        wp_hints = [
            str(n.get("id")) for n in nodes
            if canonical_operation(n.get("op", "")) == "workplane"
            and any(
                any(abs(float(m.group(i))) > 1e-9 for i in (1, 2, 3))
                for m in _EXPLICIT_EULER_TRIPLE.finditer(str(n.get("span", "")))
            )
        ]
        issues.append(
            f"explicit_operation_missing:rotate:after_{','.join(wp_hints)}"
            if wp_hints else "explicit_operation_missing:rotate"
        )
    if nonidentity_translation and operations["translate"] == 0:
        wp_hints = [
            str(n.get("id")) for n in nodes
            if canonical_operation(n.get("op", "")) == "workplane"
            and any(
                any(abs(float(m.group(i))) > 1e-9 for i in (1, 2, 3))
                for m in _EXPLICIT_TRANSLATION_TRIPLE.finditer(str(n.get("span", "")))
            )
        ]
        issues.append(
            f"explicit_operation_missing:translate:after_{','.join(wp_hints)}"
            if wp_hints else "explicit_operation_missing:translate"
        )

    if _COORDINATE_FRAME_BEFORE_SKETCH.search(text):
        children: dict[str, set[str]] = {node_id: set() for node_id in by_id}
        for node_id, node in by_id.items():
            for parent in node.get("inputs") or []:
                parent_id = str(parent)
                if parent_id in children:
                    children[parent_id].add(node_id)

        def reaches_sketch(start: str) -> bool:
            frontier = deque([start])
            visited = {start}
            while frontier:
                current = frontier.popleft()
                for child in children.get(current, ()):
                    if child in visited:
                        continue
                    visited.add(child)
                    operation = canonical_operation(by_id[child].get("op", ""))
                    if operation in {"line", "arc", "circle", "profile", "polygon", "rectangle"}:
                        return True
                    frontier.append(child)
            return False

        if nonidentity_euler and operations["rotate"] and not any(
            reaches_sketch(node_id)
            for node_id, node in by_id.items()
            if canonical_operation(node.get("op", "")) == "rotate"
        ):
            issues.append("coordinate_frame_rotation_not_upstream_of_sketch")
        if nonidentity_translation and operations["translate"] and not any(
            reaches_sketch(node_id)
            for node_id, node in by_id.items()
            if canonical_operation(node.get("op", "")) == "translate"
        ):
            issues.append("coordinate_frame_translation_not_upstream_of_sketch")
    return issues


def repair_semantic_transform_nodes(graph: dict, text: str) -> tuple[dict, list[dict]]:
    """Contract proven identity or already-absorbed narrative transforms.

    CAD-Coder natural descriptions retain source-format transform metadata.
    Zero rotations/translations do not change geometry, while their narrative
    scale factors are already reflected in the explicitly listed sketch
    coordinates.  Keeping those nodes as mandatory repair targets makes A*
    alter otherwise correct geometry.  The original evidence is retained in
    the repair log; non-identity and non-CAD-Coder transforms remain untouched.
    """
    nodes = [dict(node) for node in graph.get("nodes", []) if isinstance(node, dict)]
    output = graph.get("output")
    repairs: list[dict] = []
    cadcoder_narrative = bool(
        _CADCODER_TRANSFORM_NARRATIVE.search(text)
        and _EXPLICIT_SKETCH_COORDINATES.search(text)
    )

    def replace_output(old_id: str, replacement: str) -> None:
        nonlocal output
        if isinstance(output, list):
            output = list(dict.fromkeys(
                replacement if str(value) == old_id else str(value)
                for value in output
            ))
        elif str(output) == old_id:
            output = replacement

    changed = True
    while changed:
        changed = False
        by_id = {str(node.get("id")): node for node in nodes}
        for node in list(nodes):
            node_id = str(node.get("id"))
            op = canonical_operation(node.get("op", ""))
            inputs = [str(value) for value in node.get("inputs", [])]
            if len(inputs) != 1 or inputs[0] not in by_id:
                continue
            span = str(node.get("span", ""))
            reason = None
            if op == "rotate" and _is_identity_rotation_span(span):
                reason = "identity_rotation"
            elif op == "translate" and _is_identity_translation_span(span):
                reason = "identity_translation"
            elif (
                op == "scale" and cadcoder_narrative
                and _NARRATIVE_SCALE_EVIDENCE.search(span)
            ):
                reason = "cadcoder_scale_absorbed_in_explicit_coordinates"
            if reason is None:
                continue
            replacement = inputs[0]
            for consumer in nodes:
                consumer["inputs"] = [
                    replacement if str(source) == node_id else str(source)
                    for source in consumer.get("inputs", [])
                ]
            replace_output(node_id, replacement)
            nodes.remove(node)
            repairs.append({
                "kind": "semantic_transform_contraction",
                "node_id": node_id,
                "operation": op,
                "replacement": replacement,
                "reason": reason,
                "span": span,
            })
            changed = True
            break
    return {**graph, "nodes": nodes, "output": output}, repairs


def repair_requirement_graph_metadata(graph: dict) -> tuple[dict, list[dict]]:
    """Repair validator-only metadata without inventing CAD operations.

    The operation nodes and dependency edges are left untouched.  Repairs are
    limited to role annotations dictated by the operation vocabulary and to a
    uniquely determined terminal output node.
    """
    nodes = [dict(node) for node in graph.get("nodes", []) if isinstance(node, dict)]
    repairs: list[dict] = []
    boolean_ops = {"cut", "union", "intersect"}
    for node in nodes:
        node_id = str(node.get("id"))
        op = canonical_operation(node.get("op", ""))
        inputs = list(node.get("inputs", []))
        roles = node.get("roles")
        if op not in boolean_ops and roles is not None:
            node.pop("roles", None)
            repairs.append({"kind": "remove_roles", "node_id": node_id})
        elif op in boolean_ops and len(inputs) == 2 and roles != ["base", "tool"]:
            node["roles"] = ["base", "tool"]
            repairs.append({"kind": "boolean_roles", "node_id": node_id})

    node_ids = {str(node.get("id")) for node in nodes}
    consumed = {
        str(source)
        for node in nodes
        for source in node.get("inputs", [])
        if str(source) in node_ids
    }
    sinks = [str(node.get("id")) for node in nodes if str(node.get("id")) not in consumed]
    if len(sinks) == 1 and graph.get("output") != sinks[0]:
        repairs.append({
            "kind": "unique_terminal_output",
            "old_output": graph.get("output"),
            "new_output": sinks[0],
        })
        output: str | list[str] | None = sinks[0]
    else:
        output = graph.get("output")
    return {**graph, "nodes": nodes, "output": output}, repairs


def requirement_graph_retry_guidance(issues: list[str]) -> list[str]:
    """Translate deterministic validator errors into focused repair guidance."""
    categories = {str(issue).split(":", 1)[0] for issue in issues}
    guidance: list[str] = []
    if "invalid_input_count" in categories:
        guidance.append(
            "Fix input arity exactly. Chain successive sketch operations in construction "
            "order so an extrude/revolve receives one immediate sketch predecessor. For "
            "add/remove/intersect material, first create the tool solid with extrude/revolve, "
            "then use a binary union/cut/intersect node with exactly [base, tool]. When the "
            "requirement gives an exact expression such as "
            "`receiver.merge-operation(tool)` or `receiver.difference-operation(tool)`, "
            "the receiver is the base input and the argument is the tool input. Do not omit "
            "an explicitly listed Workplane receiver merely because it is an empty starting "
            "context. A rotate/translate operation cannot be a root node: when a coordinate "
            "system is introduced before the sketch, create a workplane node first, then make "
            "rotate/translate depend on that workplane, and make the first sketch operation "
            "depend on the transform."
        )
    if {"invalid_role_count", "roles_not_allowed", "invalid_role"} & categories:
        guidance.append(
            "Only cut/union/intersect may contain roles, and their two roles must be exactly "
            "[\"base\", \"tool\"] in the same order as their two inputs. Omit roles on every "
            "other operation."
        )
    if "node_not_in_output_dependency" in categories or "invalid_output" in categories:
        guidance.append(
            "Set output to the final terminal operation only. Do not list intermediate solids. "
            "Every retained operation must have a directed path to that output, except genuinely "
            "independent final solids explicitly requested by the text."
        )
    if "span_not_in_text" in categories:
        guidance.append(
            "Replace every unsupported span with the shortest exact contiguous substring copied "
            "verbatim from the requirement; preserve capitalization and punctuation."
        )
    if any(category.startswith("aligned_") for category in categories):
        guidance.append(
            "Preserve the explicit operation instances and their order from the requirement; "
            "repair only the reported missing, extra, or misconnected nodes."
        )
    if "explicit_operation_missing" in categories:
        guidance.append(
            "Add every explicitly reported missing semantic operation as its own node using "
            "an exact evidence span from the requirement. For a non-zero Euler coordinate "
            "frame, the required chain is workplane -> rotate -> first sketch operation. "
            "For a non-zero coordinate-frame translation, insert translate in that same "
            "pre-sketch frame chain. Do not satisfy this issue by mentioning the transform "
            "only inside a workplane span."
        )
    if {
        "coordinate_frame_rotation_not_upstream_of_sketch",
        "coordinate_frame_translation_not_upstream_of_sketch",
    } & categories:
        guidance.append(
            "Move the reported coordinate-frame transform before the sketch: make rotate "
            "or translate depend on the explicit workplane node, then make the first "
            "line/arc/circle/profile depend on that transform. Do not attach the coordinate-"
            "frame transform after close, extrude, or the final solid. Preserve the transform "
            "node while rewiring its dependencies."
        )
    if "json" in categories:
        guidance.append("Return one complete JSON object only, with no Markdown fence or commentary.")
    return guidance


_RULE_PATTERNS = (
    ("box", r"长方体|方块|\bbox\b|rectangular (?:block|prism|base)"),
    ("cylinder", r"圆柱|\bcylinder\b|cylindrical"), ("sphere", r"球体|\bsphere\b"),
    ("profile", r"草图|轮廓|\bsketch\b|\bprofile\b"),
    ("hole", r"圆孔|通孔|盲孔|\bholes?\b"),
    ("circle", r"圆形|\bcircles?\b|circular"),
    ("line", r"线段|直线|\blines?\b|\bsegments?\b"), ("arc", r"圆弧|\barcs?\b"),
    ("close", r"闭合|\bclos(?:e|ed)\b"), ("extrude", r"拉伸|\bextrud\w*\b"),
    ("revolve", r"旋转体|\brevolv\w*\b"), ("sweep", r"扫掠|\bsweep\w*\b"),
    ("loft", r"放样|\bloft\w*\b"), ("fillet", r"圆角|\bfillets?\b"),
    ("chamfer", r"倒角|\bchamfers?\b|\bbevels?\b"),
    ("union", r"并集|合并|\bunion\b|\bcombine\b|\bjoin\b"),
    ("pocket", r"口袋|槽|\bpockets?\b"),
    ("cut", r"差集|切除|\bcut\b|\bremove\b|\bsubtract\w*\b"),
    ("intersect", r"交集|相交|\bintersect\w*\b"), ("mirror", r"镜像|\bmirror\w*\b"),
    ("rotate", r"旋转|\brotat\w*\b"), ("translate", r"平移|\btranslat\w*\b"),
    ("scale", r"缩放|\bscal(?:e|ed|ing)\b"),
    ("pattern", r"阵列|重复|\bpatterns?\b|\barrays?\b|\brepeat\w*\b"),
)

_ALIGNED_STEP = re.compile(
    r"(?ms)^Step\s+\d+\s*:\s*(.*?)(?=^Step\s+\d+\s*:|^Operation note:|\Z)",
)
_ALIGNED_TARGET = re.compile(r"Create or update\s+`([^`]+)`\s*:", re.I)
_ALIGNED_SOURCE = re.compile(r"start from\s+`([^`]+)`", re.I)
_ALIGNED_ASSIGN = re.compile(
    r"Set\s+`([^`]+)`\s+to the exact expression\s+`([^`]+)`", re.I,
)
_ALIGNED_OUTPUT = re.compile(
    r"Use\s+`([^`]+)`\s+as the final output\s+`result`", re.I,
)
_ALIGNED_METHOD = re.compile(r"(?:using|apply)\s+`([^`]+)`", re.I)
_ALIGNED_ARGUMENT = re.compile(
    r"\barg1\s*=\s*(.+?)(?=;\s*then\b|,\s*arg\d+\s*=|\.\s*$|\Z)",
    re.I | re.S,
)
_ALIGNED_HELPER_DEF = re.compile(
    r"Define helper\s+`([A-Za-z_]\w*)\s*\(([^`]*)\)`", re.I,
)
_ALIGNED_RETURN = re.compile(r"Return\s+`([^`]+)`", re.I)
_ALIGNED_REDUCE = re.compile(
    r"apply\s+`reduce`\s+with\s+arg1=lambda\s+a,\s*b:\s*"
    r"a\.shared-volume-operation\(b\),\s*arg2=\[([^\]]*)\]",
    re.I,
)
_ALIGNED_BOOLEAN_MARKERS = {
    "merge the supplied geometry into the current solid": "union",
    "take the supplied tool geometry away from the current solid": "cut",
    "keep only the volume shared with the supplied tool geometry": "intersect",
    "append the supplied geometry to the current object": "union",
    "merge the pending solids into one object": "union",
}


def _output_dependency_nodes(nodes: list[dict], output: str | list[str] | None) -> list[dict]:
    """Keep only operations that can influence the explicitly requested output."""
    by_id = {str(node.get("id")): node for node in nodes}
    outputs = [output] if isinstance(output, str) else list(output or [])
    reachable: set[str] = set()
    frontier = [str(value) for value in outputs if str(value) in by_id]
    while frontier:
        node_id = frontier.pop()
        if node_id in reachable:
            continue
        reachable.add(node_id)
        frontier.extend(
            str(value) for value in by_id[node_id].get("inputs", [])
            if str(value) in by_id
        )
    return [node for node in nodes if str(node.get("id")) in reachable]


def _split_compact_statements(value: str) -> list[str]:
    """Split the compact Op-CAD DSL on top-level semicolons only."""
    parts: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    for index, character in enumerate(value):
        if quote is not None:
            if character == quote and (index == 0 or value[index - 1] != "\\"):
                quote = None
            continue
        if character in {"'", '"'}:
            quote = character
        elif character in "([{" :
            depth += 1
        elif character in ")]}" and depth:
            depth -= 1
        elif character == ";" and depth == 0:
            part = value[start:index].strip().strip(".")
            if part:
                parts.append(part)
            start = index + 1
    tail = value[start:].strip().strip(".")
    if tail:
        parts.append(tail)
    return parts


def _compact_workplane_placeholders(expression: str) -> str:
    """Replace numeric W(origin;normal;xDir) payloads with parseable W()."""
    value = expression
    cursor = 0
    while True:
        start = value.find("W(", cursor)
        if start < 0:
            return value
        depth = 0
        end = None
        for index in range(start + 1, len(value)):
            character = value[index]
            if character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
                if depth == 0:
                    end = index
                    break
        if end is None:
            return value
        value = value[:start] + "W()" + value[end + 1:]
        cursor = start + 3


def _compact_dsl_fallback_graph(text: str) -> dict | None:
    """Parse the lossless compact Op-CAD text without reference artifacts."""
    if "DSL:Code statements execute left-to-right" not in text:
        return None
    steps = _ALIGNED_STEP.findall(text)
    if not steps:
        return None

    dsl_match = re.search(r"(?m)^DSL:(.*)$", text)
    aliases: dict[str, str] = {"W": "Workplane"}
    if dsl_match:
        for part in _split_compact_statements(dsl_match.group(1)):
            match = re.fullmatch(r"([A-Za-z]\w*)=([A-Za-z_]\w*)", part)
            if match:
                aliases[match.group(1)] = match.group(2)

    macros: dict[str, str] = {}
    definitions_match = re.search(r"(?m)^Defs:(.*)$", text)
    if definitions_match:
        for definition in _split_compact_statements(definitions_match.group(1)):
            match = re.fullmatch(r"(D\d+)=(.*)", definition, flags=re.S)
            if match:
                macros[match.group(1)] = match.group(2).strip()

    nodes: list[dict] = []
    variables: dict[str, str] = {}
    result_index = 0

    def bind_result(target: str, value: str) -> str | None:
        """Bind compact result variables while preserving their true numbering."""
        nonlocal result_index
        variables[target] = value
        explicit = re.fullmatch(r"r(\d+)", target, flags=re.I)
        if explicit:
            result_index = max(result_index, int(explicit.group(1)) + 1)
            return target
        if target.lower() == "r":
            alias = f"r{result_index}"
            variables[alias] = value
            result_index += 1
            return alias
        return None

    def append_node(op: str, span: str, base: str | None = None,
                    tools: list[str] | None = None) -> str:
        spec = DEFAULT_OPERATION_SPECS[op]
        inputs: list[str] = []
        roles: list[str] = []
        if spec.roles:
            if base:
                inputs.append(base)
                roles.append("base")
            for tool in tools or []:
                if tool:
                    inputs.append(tool)
                    roles.append("tool")
        elif base and spec.max_inputs != 0:
            inputs.append(base)
        node_id = f"r{len(nodes)}"
        node = {"id": node_id, "op": op, "inputs": inputs, "span": span}
        if roles:
            node["roles"] = roles
        nodes.append(node)
        return node_id

    def resolve_expression(expression: str) -> str | None:
        value = _compact_workplane_placeholders(expression.strip().strip("`"))
        for short, long_name in sorted(aliases.items(), key=lambda item: -len(item[0])):
            value = re.sub(rf"\.{re.escape(short)}(?=\s*\()", f".{long_name}", value)
        try:
            parsed = ast.parse(value, mode="eval").body
        except (SyntaxError, ValueError):
            return variables.get(value)

        def resolve(node: ast.AST) -> str | None:
            if isinstance(node, ast.Name):
                return variables.get(node.id)
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                base = resolve(node.left)
                tool = resolve(node.right)
                return append_node("union", "+", base, [tool] if tool else []) if base and tool else base or tool
            if not isinstance(node, ast.Call):
                return None
            receiver: str | None = None
            label = ""
            if isinstance(node.func, ast.Attribute):
                receiver = resolve(node.func.value)
                label = aliases.get(node.func.attr, node.func.attr)
            elif isinstance(node.func, ast.Name):
                label = aliases.get(node.func.id, node.func.id)
            op = canonical_operation(label)
            if op not in DEFAULT_OPERATION_SPECS:
                return receiver
            if op == "workplane":
                receiver = None
            if op in {"cut", "union", "intersect"}:
                tools = [value for argument in node.args if (value := resolve(argument))]
                if receiver is None:
                    # Compact targets can omit the accumulator index.  Let the
                    # explicit Rel line below supply the base/tool connection.
                    return tools[-1] if tools else None
                if not tools:
                    return receiver
                current = receiver
                for tool in tools:
                    current = append_node(op, f".{node.func.attr if isinstance(node.func, ast.Attribute) else label}(", current, [tool])
                return current
            return append_node(op, f".{node.func.attr if isinstance(node.func, ast.Attribute) else label}(", receiver)

        return resolve(parsed)

    accumulator: str | None = None
    for step_index, body in enumerate(steps, start=1):
        before = len(nodes)
        previous_accumulator = accumulator
        step_result_aliases: list[str] = []
        code_match = re.search(r"(?m)^\s*Code:(.*)$", body)
        statements = _split_compact_statements(code_match.group(1)) if code_match else []

        def expand_statement(statement: str, stack: tuple[str, ...] = ()) -> list[str]:
            """Expand nested D# macros once per occurrence without recursion loops."""
            if statement not in macros or statement in stack:
                return [statement]
            expanded_parts: list[str] = []
            for part in _split_compact_statements(macros[statement]):
                expanded_parts.extend(expand_statement(part, (*stack, statement)))
            return expanded_parts

        expanded = [
            part
            for statement in statements
            for part in expand_statement(statement)
        ]
        last_output: str | None = None
        for statement in expanded:
            grouped_update = re.match(
                r"Create\s+or\s+update\s+(r\d*):\s*apply\s+reduce\b",
                statement, re.I,
            )
            if grouped_update:
                # Preserve the explicit shared-volume reduction as a
                # role-aware intersection chain, at its exact statement
                # position so later rN references see the resulting solid.
                argument_list = re.search(r"arg2\s*=\s*\[([^\]]*)\]", statement)
                values = []
                if argument_list:
                    values = [
                        variables[name.strip()]
                        for name in argument_list.group(1).split(",")
                        if name.strip() in variables
                    ]
                current = values[0] if values else None
                for tool in values[1:]:
                    current = append_node(
                        "intersect", "shared-volume-operation", current, [tool],
                    )
                if current:
                    alias = bind_result(grouped_update.group(1), current)
                    if alias:
                        step_result_aliases.append(alias)
                    last_output = current
                continue
            output_match = re.fullmatch(
                r"Use\s+([A-Za-z_]\w*)\s+as\s+the\s+final\s+output\s+([A-Za-z_]\w*)",
                statement, flags=re.I,
            )
            if output_match:
                source, target = output_match.groups()
                if source in variables:
                    alias = bind_result(target, variables[source])
                    if alias:
                        step_result_aliases.append(alias)
                    last_output = variables[source]
                continue
            assignment = re.fullmatch(r"([A-Za-z_]\w*)=(.*)", statement, flags=re.S)
            if not assignment:
                continue
            target, expression = assignment.groups()
            resolved = resolve_expression(expression)
            if resolved:
                alias = bind_result(target, resolved)
                if alias:
                    step_result_aliases.append(alias)
                last_output = resolved

        new_boolean = next(
            (str(node["id"]) for node in reversed(nodes[before:])
             if node.get("op") in {"cut", "union", "intersect"}),
            None,
        )
        relation_match = re.search(r"(?m)^\s*Rel:([^;\.]+)", body)
        relation = relation_match.group(1).strip().lower() if relation_match else ""
        # Use the last value produced by this step before consulting the
        # cross-step ``h`` scratch variable.  Preferring stale ``h`` silently
        # discarded single fillet/chamfer steps whose output was named h1/h2.
        tool = last_output or variables.get("h")
        if new_boolean is not None:
            accumulator = new_boolean
        elif previous_accumulator and tool and tool != previous_accumulator:
            relation_op = {
                "union": "union", "subtract": "cut", "intersection": "intersect",
                "intersect": "intersect",
            }.get(relation)
            accumulator = (
                append_node(relation_op, f"Rel:{relation}", previous_accumulator, [tool])
                if relation_op else tool
            )
        else:
            accumulator = tool or previous_accumulator
        if accumulator:
            variables["r"] = accumulator
            if step_result_aliases:
                variables[step_result_aliases[-1]] = accumulator
            else:
                alias = bind_result("r", accumulator)
                if alias:
                    step_result_aliases.append(alias)

    if not nodes or accumulator is None:
        return None
    active_nodes = _output_dependency_nodes(nodes, accumulator)
    return {"nodes": active_nodes, "output": accumulator}


def _aligned_text_fallback_graph(text: str) -> dict | None:
    """Parse the deterministic AST-to-natural-language benchmark format.

    This is a failure-only fallback and reads natural language exclusively. It
    preserves variable def-use, repeated operation instances and Boolean
    base/tool roles without consulting reference code or geometry.
    """
    if "Ordered construction:" not in text:
        return None
    steps = _ALIGNED_STEP.findall(text)
    if not steps:
        return None
    nodes: list[dict] = []
    variables: dict[str, str] = {}
    explicit_outputs: list[str] = []

    # Helper definitions are descriptions, not executions.  Save their bodies
    # and replay the operations at each call site so repeated calls remain
    # repeated operation instances and formal parameters do not become bogus
    # root nodes.
    helpers: dict[str, list[str]] = {}
    # Keep the helper table that was in scope at each call site.  Benchmark
    # programs may redefine the same Python function; resolving every call
    # against the last definition changes real def-use semantics.
    main_steps: list[tuple[str, dict[str, list[str]]]] = []
    active_helper: str | None = None
    for body in steps:
        helper_match = _ALIGNED_HELPER_DEF.search(body)
        if helper_match:
            active_helper = helper_match.group(1)
            # Python name binding replaces an earlier function definition.
            # Keeping all historical bodies under the same name made a later
            # call replay every redefinition, inventing thousands of bogus
            # operations for generated benchmark descriptions.
            helpers[active_helper] = []
            continue
        if active_helper is not None:
            if _ALIGNED_OUTPUT.search(body):
                # An incomplete/truncated helper at the end of a source file
                # does not absorb the top-level result assignment.
                active_helper = None
                main_steps.append((body, {name: list(parts) for name, parts in helpers.items()}))
                continue
            if _ALIGNED_RETURN.search(body):
                # The deterministic description puts the complete returned
                # fluent chain on the Return step.  It is part of the helper
                # body and must be replayed at each call site.
                helpers[active_helper].append(body)
                active_helper = None
            else:
                helpers[active_helper].append(body)
            continue
        main_steps.append((body, {name: list(parts) for name, parts in helpers.items()}))

    def append_node(op: str, span: str, current: str | None, tool: str | None) -> str:
        spec = DEFAULT_OPERATION_SPECS[op]
        inputs: list[str] = []
        roles: list[str] = []
        if spec.roles:
            if current:
                inputs.append(current)
            if tool:
                inputs.append(tool)
            if inputs:
                roles = ["base", *(["tool"] * (len(inputs) - 1))]
        elif current and spec.max_inputs != 0:
            inputs = [current]
        node_id = f"r{len(nodes)}"
        node = {"id": node_id, "op": op, "inputs": inputs, "span": span}
        if roles:
            node["roles"] = roles
        nodes.append(node)
        return node_id

    def argument_after(body: str, end: int) -> str | None:
        tail = body[end:]
        match = _ALIGNED_ARGUMENT.search(tail)
        return match.group(1).strip() if match else None

    def inline_expression(
        expr: str | None, helper_table: dict[str, list[str]] | None = None,
    ) -> str | None:
        if not expr:
            return None
        value = expr.strip().strip("`")
        if value in variables:
            return variables[value]
        value = (
            value.replace(".difference-operation(", ".cut(")
            .replace(".shared-volume-operation(", ".intersect(")
            .replace(".union-operation(", ".union(")
        )
        try:
            parsed = ast.parse(value, mode="eval").body
        except (SyntaxError, ValueError):
            return None

        def resolve(node: ast.AST) -> str | None:
            if isinstance(node, ast.Name):
                return variables.get(node.id)
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                base = resolve(node.left)
                tool = resolve(node.right)
                if base and tool:
                    return append_node("union", "+", base, tool)
                return base or tool
            if not isinstance(node, ast.Call):
                return None
            receiver: str | None = None
            label = ""
            if isinstance(node.func, ast.Attribute):
                receiver = resolve(node.func.value)
                label = node.func.attr
            elif isinstance(node.func, ast.Name):
                label = node.func.id
            argument_nodes = [
                dependency for argument in node.args
                if (dependency := resolve(argument)) is not None
            ]
            if helper_table and label in helper_table:
                base = receiver or (argument_nodes[0] if argument_nodes else None)
                return replay_helper(label, base, helper_table)
            op = canonical_operation(label)
            if op not in DEFAULT_OPERATION_SPECS:
                return receiver or (argument_nodes[0] if argument_nodes else None)
            if op == "workplane":
                receiver = None
            if op in {"cut", "union", "intersect"}:
                current = receiver
                for tool_node in argument_nodes:
                    current = append_node(op, f".{label}(", current, tool_node)
                return current
            return append_node(op, f".{label}(", receiver, None)

        return resolve(parsed)

    def known_events(body: str, helper_table: dict[str, list[str]]) -> list[tuple[int, int, str, str, str | None]]:
        events: list[tuple[int, int, str, str, str | None]] = []
        for match in _ALIGNED_METHOD.finditer(body):
            label = match.group(1)
            op = canonical_operation(label)
            argument = argument_after(body, match.end())
            if op in DEFAULT_OPERATION_SPECS:
                events.append((match.start(), match.end(), op, match.group(0), argument))
            elif label in helper_table:
                events.append((match.start(), match.end(), f"__helper__:{label}", match.group(0), argument))
        for match in _ALIGNED_REDUCE.finditer(body):
            names = [
                value.strip() for value in match.group(1).split(",")
                if re.fullmatch(r"[A-Za-z_][\w-]*", value.strip())
            ]
            events.append((
                match.start(), match.end(), "intersect", match.group(0),
                "__reduce__:" + ",".join(names),
            ))
        lower = body.lower()
        for phrase, op in _ALIGNED_BOOLEAN_MARKERS.items():
            start = 0
            while True:
                position = lower.find(phrase, start)
                if position < 0:
                    break
                end = position + len(phrase)
                events.append((position, end, op, body[position:end], argument_after(body, end)))
                start = end
        return sorted(events, key=lambda value: (value[0], value[1], value[2]))

    def replay_helper(name: str, current: str | None,
                      helper_table: dict[str, list[str]]) -> str | None:
        for helper_body in helper_table.get(name, []):
            return_match = _ALIGNED_RETURN.search(helper_body)
            if return_match:
                # Return steps can contain a fluent expression rather than
                # prose such as "apply `rotate`".  Preserve the chain order.
                expression = return_match.group(1)
                for match in re.finditer(r"(?:^|\.)([A-Za-z_]\w*)\s*\(", expression):
                    label = match.group(1)
                    op = canonical_operation(label)
                    if op not in DEFAULT_OPERATION_SPECS:
                        continue
                    if op == "workplane":
                        current = None
                    current = append_node(op, f".{label}(", current, None)
                continue
            for _, _, op, span, _ in known_events(helper_body, helper_table):
                if op.startswith("__helper__:"):
                    current = replay_helper(op.split(":", 1)[1], current, helper_table)
                    continue
                if op == "workplane":
                    current = None
                current = append_node(op, span, current, None)
        return current

    for body, helper_table in main_steps:
        assignment = _ALIGNED_ASSIGN.search(body)
        if assignment:
            target, source = assignment.groups()
            resolved = inline_expression(source, helper_table)
            if resolved:
                variables[target] = resolved
        target_match = _ALIGNED_TARGET.search(body)
        source_match = _ALIGNED_SOURCE.search(body)
        target = target_match.group(1) if target_match else None
        source = source_match.group(1) if source_match else None
        current = variables.get(source or "")
        events = known_events(body, helper_table)
        for _, _, op, span, tool_expr in events:
            if op.startswith("__helper__:"):
                argument_node = inline_expression(tool_expr, helper_table)
                current = replay_helper(op.split(":", 1)[1], argument_node, helper_table)
                continue
            if tool_expr and tool_expr.startswith("__reduce__:"):
                names = [value for value in tool_expr.split(":", 1)[1].split(",") if value]
                operands = [variables[value] for value in names if value in variables]
                if operands:
                    current = operands[0]
                for tool_node in operands[1:]:
                    current = append_node(op, span, current, tool_node)
                continue
            if op == "union" and "pending solids" in span.lower():
                if current:
                    current = append_node(op, span, current, current)
                    continue
            tool_node = inline_expression(tool_expr, helper_table)
            current = append_node(op, span, current, tool_node)
        if target and current:
            variables[target] = current
        output_match = _ALIGNED_OUTPUT.search(body)
        if output_match and output_match.group(1) in variables:
            explicit_outputs.append(variables[output_match.group(1)])

    if not nodes:
        return None
    consumers = {
        str(parent) for node in nodes for parent in node.get("inputs", [])
    }
    sinks = [str(node["id"]) for node in nodes if str(node["id"]) not in consumers]
    outputs = list(dict.fromkeys(explicit_outputs or sinks))
    output: str | list[str] | None = (
        outputs[0] if len(outputs) == 1 else outputs if outputs else None
    )
    active_nodes = _output_dependency_nodes(nodes, output)
    return {"nodes": active_nodes, "output": output}


def build_rule_fallback_graph(text: str) -> dict:
    compact = _compact_dsl_fallback_graph(text)
    if compact is not None:
        return compact
    aligned = _aligned_text_fallback_graph(text)
    if aligned is not None:
        return aligned
    matches: list[tuple[int, int, str, str]] = []
    for op, pattern in _RULE_PATTERNS:
        for match in re.finditer(pattern, text, flags=re.I):
            matches.append((match.start(), match.end(), op, match.group(0)))
    matches.sort(key=lambda item: (item[0], item[1], item[2]))
    deduped: list[tuple[int, int, str, str]] = []
    for item in matches:
        if deduped and item[0] < deduped[-1][1] and item[2] == deduped[-1][2]:
            continue
        deduped.append(item)
    nodes: list[dict] = []
    produced: list[str] = []
    for _, _, op, span in deduped:
        spec = DEFAULT_OPERATION_SPECS[op]
        node_id = f"r{len(nodes)}"
        inputs: list[str] = []
        roles: list[str] = []
        if spec.roles:
            inputs = produced[-2:]
            roles = ["base", "tool"][:len(inputs)]
        elif produced and spec.max_inputs != 0:
            inputs = [produced[-1]]
        node = {"id": node_id, "op": op, "inputs": inputs, "span": span}
        if roles:
            node["roles"] = roles
        nodes.append(node)
        produced.append(node_id)
    consumers = {str(parent) for node in nodes for parent in node.get("inputs", [])}
    sinks = [str(node["id"]) for node in nodes if str(node["id"]) not in consumers]
    output: str | list[str] | None = (
        sinks[0] if len(sinks) == 1 else sinks if sinks else None
    )
    return {"nodes": nodes, "output": output}


@dataclass(frozen=True)
class RequirementGraphResult:
    graph: dict
    calls: list[dict]
    issues: list[str]
    fallback_used: bool
    fallback_validation_issues: list[str]

    @property
    def valid(self) -> bool:
        """Whether the graph may safely enter retrieval, generation, or search."""
        return not self.fallback_validation_issues and not validate_requirement_graph(
            self.graph,
        )


class RequirementGraphBuilder:
    def __init__(self, client: LLMClient, *, max_retries: int = 2,
                 registry: dict[str, OperationSpec] | None = None):
        self.client = client
        self.max_retries = max_retries
        self.registry = registry or DEFAULT_OPERATION_SPECS

    def build(self, text: str) -> RequirementGraphResult:
        calls: list[dict] = []
        issues: list[str] = []
        previous: dict | None = None
        aligned_candidate = (
            _compact_dsl_fallback_graph(text)
            or _aligned_text_fallback_graph(text)
        )
        aligned_reference = (
            aligned_candidate
            if aligned_candidate is not None
            and not validate_requirement_graph(aligned_candidate, self.registry)
            else None
        )
        estimated_operations = max(
            len(_ALIGNED_STEP.findall(text)),
            sum(len(re.findall(pattern, text, flags=re.I)) for _, pattern in _RULE_PATTERNS),
            1,
        )
        # A DAG node with its evidence span commonly occupies 120--180 output
        # tokens.  The former 48-token allowance truncated otherwise valid
        # JSON for long CAD programs and forced the rule fallback after every
        # retry.  Size the response budget from the required instance count;
        # this is an output allowance, not a limit on the requirement text.
        graph_max_tokens = min(24000, max(6000, 2000 + 160 * estimated_operations))
        for attempt in range(self.max_retries + 1):
            payload = {
                "prompt_version": REQUIREMENT_GRAPH_PROMPT_VERSION,
                "text": text, "operation_catalog": operation_catalog(self.registry),
                "json_schema": REQUIREMENT_GRAPH_SCHEMA, "previous_graph": previous,
                "issues": issues,
                "repair_instructions": requirement_graph_retry_guidance(issues),
                "rules": [
                    "Keep one node per operation instance, including repeated operations.",
                    "Keep only operations and directed dependencies; do not copy dimensions or coordinates.",
                    "The inputs array contains only predecessor operation node IDs. Never put API arguments, dimensions, coordinates, plane names, selectors, or other literals in inputs.",
                    "The output field contains a terminal operation node ID, or an array of terminal node IDs only for genuinely independent final solids.",
                    "Preserve every explicit semantic method occurrence with its canonical meaning: Workplane->workplane, lineTo->line, close->close, extrude->extrude, translate->translate, and rotate->rotate. moveTo/move/center are transparent sketch-position helpers and are not operation nodes.",
                    "A final assignment such as 'Use `r` as the final output `result`' only selects the output node; never create an extra operation node for result assignment.",
                    "For cut/union/intersect, inputs and roles are ordered base then tool.",
                    "Do not add a generic profile node when explicit line/arc/circle nodes already fully describe that same contour; profile is for an atomic or otherwise unspecified sketch.",
                    "A close operation already creates the final segment back to the origin/start; do not also emit a separate line node for that same return segment.",
                    "A narrative statement that the same process appears on the second/opposite face of one extrusion is an extrusion consequence, not a second independent sketch. Duplicate operations only when distinct sketch, coordinate-system, placement, or feature evidence is provided.",
                    "If explicit sketch coordinates and final dimensions are both given, a narrative scale factor is already baked into those values and is metadata, not a separate scale operation. Keep scale only when it independently changes geometry.",
                    "A non-zero Euler-angle or translation-vector instruction is a semantic rotate or translate node. When it defines a coordinate system before a sketch, place that transform between the workplane and the first sketch operation; when it explicitly transforms the finished body, place it after the solid-producing operation.",
                    "A coordinate-system rotate/translate node must have exactly one input. Create an explicit workplane node as its predecessor; never emit rotate or translate as a root node.",
                    "When the requirement describes multiple independent solid parts without an explicit Boolean combination, retain every part and return their terminal node IDs as an output array instead of deleting parts or inventing a Boolean operation.",
                    "Every span must be direct evidence copied from the requirement.",
                    "Every span must be one exact contiguous substring of text. Copy only the shortest operation phrase; never insert 'Step N:', ellipses ('...'), paraphrases, or omitted text.",
                    "For a close node whose final segment already returns to the starting point, copy that final line/arc phrase verbatim as the close span; never add words such as 'back to' unless those exact words occur in text.",
                    "Return JSON only.",
                ],
            }
            try:
                system_prompt = (
                    "Extract a validated, number-free CadQuery requirement operation DAG."
                    if attempt == 0
                    else "Repair the previous requirement DAG. Address every reported validator "
                    "issue and follow repair_instructions exactly; preserve already-correct nodes."
                )
                result = self.client.complete(
                    system_prompt,
                    json.dumps(payload, ensure_ascii=False), max_tokens=graph_max_tokens,
                )
            except Exception as error:
                calls.append({
                    "success": False, "stage": f"requirement_graph_request_{attempt + 1}",
                    "error_type": type(error).__name__, "error": str(error),
                    "attempts": list(getattr(error, "attempts", [])),
                })
                issues = [f"llm:{type(error).__name__}:{error}"]
                continue
            calls.append(result.to_dict())
            try:
                graph = _extract_json(result.content)
                graph = canonicalize_requirement_graph(graph)
                graph, span_repairs = repair_close_evidence_spans(graph, text)
                if span_repairs:
                    calls[-1]["close_evidence_repairs"] = span_repairs
                graph, aligned_span_repairs = repair_aligned_evidence_spans(graph, text)
                if aligned_span_repairs:
                    calls[-1]["aligned_evidence_repairs"] = aligned_span_repairs
                graph, near_exact_span_repairs = repair_near_exact_evidence_spans(graph, text)
                if near_exact_span_repairs:
                    calls[-1]["near_exact_evidence_repairs"] = near_exact_span_repairs
                graph, alias_repairs = repair_transparent_alias_nodes(graph)
                if alias_repairs:
                    calls[-1]["transparent_alias_repairs"] = alias_repairs
                for node in graph.get("nodes", []):
                    if isinstance(node, dict) and isinstance(node.get("op"), str):
                        canonical_op = canonical_operation(node["op"])
                        if canonical_op != "unknown_op":
                            node["op"] = canonical_op
                graph, transform_repairs = repair_semantic_transform_nodes(graph, text)
                if transform_repairs:
                    calls[-1]["semantic_transform_repairs"] = transform_repairs
                graph, metadata_repairs = repair_requirement_graph_metadata(graph)
                if metadata_repairs:
                    calls[-1]["metadata_repairs"] = metadata_repairs
                issues = validate_requirement_graph(graph, self.registry)
                issues.extend(explicit_semantic_coverage_issues(graph, text))
                for node in graph.get("nodes", []):
                    span = node.get("span") if isinstance(node, dict) else None
                    if isinstance(span, str) and span and span not in text:
                        issues.append(f"span_not_in_text:{node.get('id')}")
                if not issues and aligned_reference is not None:
                    required_gap = graph_difference(aligned_reference, graph)
                    extra_gap = graph_difference(graph, aligned_reference)
                    if required_gap["M_op"]:
                        issues.append(
                            "aligned_missing_operations:"
                            + json.dumps(required_gap["missing_operations"], sort_keys=True)
                        )
                    if extra_gap["M_op"]:
                        issues.append(
                            "aligned_extra_operations:"
                            + json.dumps(extra_gap["missing_operations"], sort_keys=True)
                        )
                    if required_gap["M_dep"] or extra_gap["M_dep"]:
                        issues.append(
                            "aligned_dependency_mismatch:"
                            f"missing_rate={required_gap['M_dep']:.6f},"
                            f"extra_rate={extra_gap['M_dep']:.6f}"
                        )
            except (ValueError, json.JSONDecodeError) as error:
                graph = {}
                issues = [f"json:{type(error).__name__}:{error}"]
            if not issues:
                return RequirementGraphResult(graph, calls, [], False, [])
            previous = graph
        fallback = build_rule_fallback_graph(text)
        fallback_issues = [
            *validate_requirement_graph(fallback, self.registry),
            *explicit_semantic_coverage_issues(fallback, text),
        ]
        return RequirementGraphResult(
            fallback, calls, sorted(set([*issues, *fallback_issues])), True, fallback_issues,
        )


def _role_for(node: dict, index: int) -> str:
    roles = node.get("roles") or []
    return str(roles[index]) if index < len(roles) else "input"


def graph_counters(graph: dict) -> tuple[Counter[str], Counter[tuple[str, str, str]]]:
    nodes = graph.get("nodes") or []
    by_id = {str(node.get("id")): node for node in nodes}
    def op_of(node: dict) -> str:
        return canonical_operation(node.get("op") or ((node.get("operations") or ["unknown_op"])[-1]))
    def multiplicity(node: dict) -> int:
        value = node.get("multiplicity", 1)
        return int(value) if isinstance(value, int) and 1 <= value <= 128 else 1
    operations: Counter[str] = Counter()
    for node in nodes:
        op = op_of(node)
        if is_semantic_operation(op) or op == "unknown_op":
            operations[op] += multiplicity(node)
    def resolved_parents(node_id: str, seen: set[str] | None = None) -> list[str]:
        seen = set(seen or set())
        if node_id in seen or node_id not in by_id:
            return []
        seen.add(node_id)
        node = by_id[node_id]
        if not is_transparent_operation(op_of(node)):
            return [node_id]
        resolved: list[str] = []
        for parent in node.get("inputs") or []:
            for value in resolved_parents(str(parent), seen):
                if value not in resolved:
                    resolved.append(value)
        return resolved
    dependencies: Counter[tuple[str, str, str]] = Counter()
    for node in nodes:
        child_op = op_of(node)
        if is_transparent_operation(child_op):
            continue
        for index, parent_id in enumerate(node.get("inputs") or []):
            for resolved_id in resolved_parents(str(parent_id)) or [str(parent_id)]:
                parent = by_id.get(resolved_id, {})
                parent_op = op_of(parent)
                if is_semantic_operation(parent_op) or parent_op == "unknown_op":
                    dependencies[(parent_op, child_op, _role_for(node, index))] += max(
                        multiplicity(parent), multiplicity(node),
                    )
    return operations, dependencies


def _missing_rate(required: Counter, observed: Counter) -> float:
    total = sum(required.values())
    return 0.0 if total == 0 else sum((required - observed).values()) / total


def _semantic_topology(graph: dict) -> tuple[dict[str, str], Counter[tuple[str, str, str]]]:
    raw_nodes = graph.get("nodes") or []
    by_id = {str(node.get("id")): node for node in raw_nodes}

    def op_of(node: dict) -> str:
        return canonical_operation(node.get("op") or ((node.get("operations") or ["unknown_op"])[-1]))

    def expanded_ids(node_id: str) -> list[str]:
        value = by_id[node_id].get("multiplicity", 1)
        count = int(value) if isinstance(value, int) and 1 <= value <= 128 else 1
        return [node_id] if count == 1 else [f"{node_id}#{index}" for index in range(count)]

    def resolve(node_id: str, seen: set[str] | None = None) -> list[str]:
        seen = set(seen or set())
        if node_id in seen or node_id not in by_id:
            return []
        seen.add(node_id)
        if not is_transparent_operation(op_of(by_id[node_id])):
            return expanded_ids(node_id)
        result: list[str] = []
        for parent in by_id[node_id].get("inputs") or []:
            for resolved in resolve(str(parent), seen):
                if resolved not in result:
                    result.append(resolved)
        return result

    nodes: dict[str, str] = {}
    for node_id, node in by_id.items():
        op = op_of(node)
        if is_semantic_operation(op) or op == "unknown_op":
            for expanded_id in expanded_ids(node_id):
                nodes[expanded_id] = op
    edges: Counter[tuple[str, str, str]] = Counter()
    for child_id, child in by_id.items():
        child_ids = expanded_ids(child_id)
        if not child_ids or child_ids[0] not in nodes:
            continue
        for index, parent_id in enumerate(child.get("inputs") or []):
            role = _role_for(child, index)
            source_ids = [source_id for source_id in resolve(str(parent_id)) if source_id in nodes]
            if len(source_ids) == len(child_ids) and len(source_ids) > 1:
                pairs = zip(source_ids, child_ids)
            elif len(child_ids) == 1:
                pairs = ((source_id, child_ids[0]) for source_id in source_ids)
            elif len(source_ids) == 1:
                pairs = ((source_ids[0], target_id) for target_id in child_ids)
            else:
                pairs = ((source_id, target_id) for source_id in source_ids for target_id in child_ids)
            for source_id, target_id in pairs:
                edges[(source_id, target_id, role)] += 1
    return nodes, edges


def _local_signature(
    node_id: str, nodes: dict[str, str], edges: Counter[tuple[str, str, str]],
) -> Counter[tuple[str, str, str]]:
    signature: Counter[tuple[str, str, str]] = Counter()
    for (source, target, role), count in edges.items():
        if target == node_id:
            signature[("in", nodes[source], role)] += count
        if source == node_id:
            signature[("out", nodes[target], role)] += count
    return signature


def _counter_jaccard(left: Counter, right: Counter) -> float:
    keys = set(left) | set(right)
    return 1.0 if not keys else sum(min(left[key], right[key]) for key in keys) / max(
        sum(max(left[key], right[key]) for key in keys), 1,
    )


def _node_mapping(required_graph: dict, observed_graph: dict) -> dict[str, str]:
    required_nodes, required_edges = _semantic_topology(required_graph)
    observed_nodes, observed_edges = _semantic_topology(observed_graph)

    def boundary_distances(
        nodes: dict[str, str], edges: Counter[tuple[str, str, str]], *, reverse: bool,
    ) -> dict[str, int]:
        parents: dict[str, set[str]] = {node_id: set() for node_id in nodes}
        children: dict[str, set[str]] = {node_id: set() for node_id in nodes}
        for source, target, _ in edges:
            left, right = (target, source) if reverse else (source, target)
            if left not in nodes or right not in nodes:
                continue
            parents[right].add(left)
            children[left].add(right)
        indegree = {node_id: len(values) for node_id, values in parents.items()}
        queue = deque(sorted(
            node_id for node_id, degree in indegree.items() if degree == 0
        ))
        distances = {node_id: 0 for node_id in nodes}
        while queue:
            node_id = queue.popleft()
            for child in sorted(children.get(node_id, ())):
                distances[child] = max(
                    distances[child], distances[node_id] + 1,
                )
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        # Valid requirement and program operation graphs are DAGs.  If a
        # defensive caller supplies a cycle, unresolved nodes retain distance
        # zero instead of recursing indefinitely or overflowing Python's stack.
        return distances

    required_depth = boundary_distances(required_nodes, required_edges, reverse=False)
    observed_depth = boundary_distances(observed_nodes, observed_edges, reverse=False)
    required_height = boundary_distances(required_nodes, required_edges, reverse=True)
    observed_height = boundary_distances(observed_nodes, observed_edges, reverse=True)
    required_signatures = {
        node_id: _local_signature(node_id, required_nodes, required_edges)
        for node_id in required_nodes
    }
    observed_signatures = {
        node_id: _local_signature(node_id, observed_nodes, observed_edges)
        for node_id in observed_nodes
    }
    observed_by_operation: dict[str, list[str]] = {}
    for observed_id, observed_op in observed_nodes.items():
        observed_by_operation.setdefault(observed_op, []).append(observed_id)
    for values in observed_by_operation.values():
        values.sort()

    candidates_by_required: dict[str, list[str]] = {}
    static_score: dict[tuple[str, str], float] = {}
    for required_id, required_op in required_nodes.items():
        if required_op == "unknown_op":
            continue
        candidates_by_required[required_id] = list(
            observed_by_operation.get(required_op, ())
        )
        required_signature = required_signatures[required_id]
        required_total = sum(required_signature.values())
        for observed_id in candidates_by_required[required_id]:
            observed_signature = observed_signatures[observed_id]
            # Requirement graphs are often a valid subgraph of a program with
            # additional parts.  Score local-signature containment so extra
            # observed neighbours do not disqualify the correct instance.
            local = 1.0 if required_total == 0 else sum(
                min(count, observed_signature[key])
                for key, count in required_signature.items()
            ) / required_total
            depth = 1.0 / (1.0 + abs(required_depth[required_id] - observed_depth[observed_id]))
            height = 1.0 / (1.0 + abs(required_height[required_id] - observed_height[observed_id]))
            static_score[(required_id, observed_id)] = 0.7 * local + 0.15 * depth + 0.15 * height

    required_degree = Counter()
    required_neighbours: dict[str, set[str]] = {
        node_id: set() for node_id in candidates_by_required
    }
    required_incident: dict[str, list[tuple[str, str, str, int]]] = {
        node_id: [] for node_id in candidates_by_required
    }
    for (source, target, role), count in required_edges.items():
        required_degree[source] += 1
        required_degree[target] += 1
        if source in required_neighbours and target in required_neighbours:
            required_neighbours[source].add(target)
            required_neighbours[target].add(source)
        if source in required_incident:
            required_incident[source].append((source, target, role, count))
        if target in required_incident:
            required_incident[target].append((source, target, role, count))
    # Start from rare/anchored operations and then expand to their neighbours.
    # This prevents repeated line/profile nodes from being paired by ID order.
    ordering: list[str] = []
    remaining = set(candidates_by_required)
    frontier: set[str] = set()
    while remaining:
        pool = frontier & remaining
        if not pool:
            pool = remaining
        def order_key(node_id: str):
            return (
                len(candidates_by_required.get(node_id, [])),
                -required_degree[node_id],
                node_id,
            )
        chosen = min(pool, key=order_key)
        ordering.append(chosen)
        remaining.remove(chosen)
        frontier.discard(chosen)
        frontier.update(required_neighbours.get(chosen, ()))

    observed_adjacency: dict[str, list[tuple[str, str]]] = {}
    for source, target, role in observed_edges:
        observed_adjacency.setdefault(source, []).append((target, role))
    reachable: set[tuple[str, str, str]] = set()
    for source in observed_nodes:
        frontier_queue = deque([(source, 0)])
        best_depth = {source: 0}
        while frontier_queue:
            current, depth = frontier_queue.popleft()
            if depth >= 3:
                continue
            for target, role in observed_adjacency.get(current, []):
                reachable.add((source, target, role))
                next_depth = depth + 1
                if next_depth < best_depth.get(target, 4):
                    best_depth[target] = next_depth
                    frontier_queue.append((target, next_depth))

    candidate_pairs = sum(len(values) for values in candidates_by_required.values())
    if len(ordering) > 128 or candidate_pairs > 10_000:
        # Fixed-width beam search copies an ever-growing mapping for every
        # repeated line/arc/profile candidate.  On thousand-node CAD graphs
        # that becomes super-quadratic and can stall every A* evaluation.
        # The large-graph path keeps the same operation, local-signature,
        # boundary-distance and 3-hop topology evidence, but commits the best
        # deterministic candidate at each topology-guided step.
        mapping: dict[str, str] = {}
        used: set[str] = set()
        for required_id in ordering:
            options = [
                observed_id
                for observed_id in candidates_by_required[required_id]
                if observed_id not in used
            ]
            if not options:
                continue
            scored: list[tuple[float, str]] = []
            for observed_id in options:
                relation = 0.0
                for source, target, role, count in required_incident[required_id]:
                    if source == required_id and target in mapping:
                        observed_edge = (observed_id, mapping[target], role)
                    elif target == required_id and source in mapping:
                        observed_edge = (mapping[source], observed_id, role)
                    else:
                        continue
                    if observed_edges.get(observed_edge, 0):
                        relation += 6.0 * count
                    elif observed_edge in reachable:
                        relation += 2.0 * count
                    else:
                        relation -= 5.0 * count
                scored.append((
                    static_score[(required_id, observed_id)] + relation,
                    observed_id,
                ))
            _, selected = min(scored, key=lambda item: (-item[0], item[1]))
            mapping[required_id] = selected
            used.add(selected)
        return mapping

    beams: list[tuple[float, dict[str, str], frozenset[str]]] = [(0.0, {}, frozenset())]
    for required_id in ordering:
        expanded: list[tuple[float, dict[str, str], frozenset[str]]] = []
        for score, mapping, used in beams:
            options = [value for value in candidates_by_required[required_id] if value not in used]
            if not options:
                expanded.append((score - 8.0, dict(mapping), used))
                continue
            for observed_id in options:
                relation = 0.0
                for source, target, role, count in required_incident[required_id]:
                    if source == required_id and target in mapping:
                        observed_edge = (observed_id, mapping[target], role)
                    elif target == required_id and source in mapping:
                        observed_edge = (mapping[source], observed_id, role)
                    else:
                        continue
                    if observed_edges.get(observed_edge, 0):
                        relation += 6.0 * count
                    elif observed_edge in reachable:
                        relation += 2.0 * count
                    else:
                        relation -= 5.0 * count
                updated = dict(mapping)
                updated[required_id] = observed_id
                expanded.append((
                    score + static_score[(required_id, observed_id)] + relation,
                    updated,
                    used | {observed_id},
                ))
        beams = sorted(
            expanded,
            key=lambda item: (-item[0], tuple(sorted(item[1].items()))),
        )[:64]
    return beams[0][1] if beams else {}


def requirement_program_node_mapping(
    requirement_graph: dict, program_graph: dict,
) -> dict[str, str]:
    """Return the deterministic one-to-one semantic node correspondence."""
    return _node_mapping(
        requirement_graph, output_dependency_subgraph(program_graph),
    )


def _dependency_match(required_graph: dict, observed_graph: dict) -> tuple[int, int, int, list[dict]]:
    required_nodes, required_edges = _semantic_topology(required_graph)
    _, observed_edges = _semantic_topology(observed_graph)
    mapping = _node_mapping(required_graph, observed_graph)
    remaining = observed_edges.copy()
    observed_adjacency: dict[str, list[tuple[str, str]]] = {}
    for edge_source, edge_target, edge_role in observed_edges:
        observed_adjacency.setdefault(edge_source, []).append(
            (edge_target, edge_role)
        )

    def connected_by_semantic_path(source: str, target: str, final_role: str) -> bool:
        # Requirement graphs intentionally omit low-level sketch operations.
        # Therefore a non-Boolean dependency may correspond to an arbitrary
        # finite directed path in the program graph (for example the first
        # ``close`` in a multi-loop sketch reaches ``extrude`` through the
        # second loop).  Boolean operands are different: ``base`` and ``tool``
        # are semantic ports and must be matched by a direct role-aware edge.
        if final_role in {"base", "tool"}:
            return False
        frontier = deque([source])
        visited = {source}
        while frontier:
            current = frontier.popleft()
            for edge_target, edge_role in observed_adjacency.get(current, ()):
                if edge_target == target and edge_role == final_role:
                    return True
                if edge_target not in visited:
                    visited.add(edge_target)
                    frontier.append(edge_target)
        return False
    matched = 0
    missing: list[dict] = []
    for (source, target, role), count in sorted(required_edges.items()):
        mapped = (mapping.get(source, ""), mapping.get(target, ""), role)
        available = min(count, remaining.get(mapped, 0)) if mapped[0] and mapped[1] else 0
        direct_available = available
        if (
            not available
            and mapped[0]
            and mapped[1]
            and connected_by_semantic_path(*mapped)
        ):
            available = 1
        matched += available
        if direct_available:
            remaining[mapped] -= available
        missing_count = count - available
        if missing_count:
            missing.append({
                "from_id": source, "to_id": target,
                "from": required_nodes[source], "to": required_nodes[target],
                "role": role, "count": missing_count,
            })
    return matched, sum(required_edges.values()), sum(observed_edges.values()), missing


def graph_difference(requirement_graph: dict, program_graph: dict) -> dict:
    program_graph = output_dependency_subgraph(program_graph)
    req_ops, req_deps = graph_counters(requirement_graph)
    prog_ops, prog_deps = graph_counters(program_graph)
    missing_ops = req_ops - prog_ops
    matched_deps, required_dep_count, _, missing_deps = _dependency_match(
        requirement_graph, program_graph,
    )
    unknown = prog_ops.get("unknown_op", 0)
    return {
        "M_op": _missing_rate(req_ops, prog_ops),
        "M_dep": 0.0 if required_dep_count == 0 else 1.0 - matched_deps / required_dep_count,
        "missing_operations": dict(missing_ops),
        "missing_dependencies": missing_deps,
        "unknown_op_count": unknown,
        "unknown_op_uncertainty": bool(unknown),
        "active_program_node_ids": [str(node.get("id")) for node in program_graph.get("nodes") or []],
        "inactive_program_node_ids": list(program_graph.get("inactive_node_ids") or []),
    }


_PATTERN_ANCHOR_CATEGORIES = {"primitive", "feature", "boolean", "transform"}


def canonical_anchor_graph(graph: dict) -> dict:
    """Create a path-invariant view while preserving the exact source DAG."""
    raw_nodes = [dict(node) for node in graph.get("nodes") or [] if isinstance(node, dict)]
    by_id = {str(node.get("id")): node for node in raw_nodes}

    def operation(node: dict) -> str:
        return canonical_operation(
            node.get("op") or ((node.get("operations") or ["unknown_op"])[-1])
        )

    def is_anchor(node: dict) -> bool:
        op = operation(node)
        spec = DEFAULT_OPERATION_SPECS.get(op)
        return op == "unknown_op" or bool(
            spec and spec.category in _PATTERN_ANCHOR_CATEGORIES
        )

    anchor_ids = {node_id for node_id, node in by_id.items() if is_anchor(node)}

    def nearest_anchors(node_id: str, seen: set[str] | None = None) -> list[str]:
        seen = set(seen or set())
        if node_id in seen or node_id not in by_id:
            return []
        if node_id in anchor_ids:
            return [node_id]
        seen.add(node_id)
        resolved: list[str] = []
        for parent in by_id[node_id].get("inputs") or []:
            for value in nearest_anchors(str(parent), seen):
                if value not in resolved:
                    resolved.append(value)
        return resolved

    nodes: list[dict] = []
    for node_id in sorted(anchor_ids):
        source = by_id[node_id]
        op = operation(source)
        inputs: list[str] = []
        roles: list[str] = []
        source_roles = list(source.get("roles") or [])
        for index, parent in enumerate(source.get("inputs") or []):
            role = str(source_roles[index]) if index < len(source_roles) else "input"
            for resolved in nearest_anchors(str(parent)):
                if resolved not in inputs:
                    inputs.append(resolved)
                    roles.append(role)
        row = {
            "id": node_id,
            "op": op,
            "inputs": inputs,
            "multiplicity": source.get("multiplicity", 1),
        }
        if roles and (
            op in {"cut", "union", "intersect"}
            or any(role != "input" for role in roles)
        ):
            row["roles"] = roles
        nodes.append(row)

    raw_outputs = graph.get("output_nodes") or graph.get("output") or []
    if not isinstance(raw_outputs, list):
        raw_outputs = [raw_outputs]
    outputs: list[str] = []
    for output in raw_outputs:
        for resolved in nearest_anchors(str(output)):
            if resolved not in outputs:
                outputs.append(resolved)
    output: str | list[str] | None = (
        outputs[0] if len(outputs) == 1 else outputs if outputs else None
    )
    return {
        "nodes": nodes,
        "output": output,
        "output_nodes": outputs,
        "graph_kind": "canonical_cad_pattern_anchor_dag",
        "dynamic_control_flow": bool(graph.get("dynamic_control_flow")),
        "source_node_count": len(raw_nodes),
    }


def asymmetric_graph_difference(requirement_graph: dict, program_graph: dict) -> dict:
    """Measure one-way requirement containment on canonical anchor graphs."""
    requirement_view = canonical_anchor_graph(requirement_graph)
    program_view = canonical_anchor_graph(output_dependency_subgraph(program_graph))
    gap = graph_difference(requirement_view, program_view)
    gap["view"] = "asymmetric_canonical_anchor_containment_v1"
    gap["requirement_anchor_graph"] = requirement_view
    gap["program_anchor_graph"] = program_view
    gap["mapping_confidence"] = 0.0 if gap["unknown_op_count"] else (
        0.65 if program_view.get("dynamic_control_flow") else 1.0
    )
    return gap


def cross_graph_alignment_gate(
    gap: dict,
    *,
    minimum_operation_missing_rate: float = 0.05,
) -> dict:
    """Decide whether a graph difference is reliable enough to guide repair.

    Program ASTs and natural-language requirement DAGs often encode equivalent
    construction orders differently.  In particular, a dependency-only gap can
    arise when a transform is applied to an already-built profile instead of to
    its workplane.  Treating that representation difference as a mandatory CAD
    edit was observed to damage otherwise repairable programs.

    Cross-graph guidance is therefore enabled only when it is anchored by a
    missing operation instance.  Dependency and role gaps remain part of the
    heuristic once that anchor is present.  Unknown operations never make the
    alignment more trustworthy.
    """
    try:
        operation_rate = max(0.0, min(1.0, float(gap.get("M_op", 0.0))))
    except (TypeError, ValueError):
        operation_rate = 0.0
    missing_operations = {
        str(operation): int(count)
        for operation, count in (gap.get("missing_operations") or {}).items()
        if operation != "unknown_op" and isinstance(count, int) and count > 0
    }
    unknown_count = int(gap.get("unknown_op_count", 0) or 0)
    threshold = max(0.0, min(1.0, float(minimum_operation_missing_rate)))
    actionable = bool(missing_operations) and operation_rate + 1e-12 >= threshold
    if actionable:
        reason = "operation_anchored_structure_gap"
    elif unknown_count:
        reason = "unknown_operation_uncertainty"
    elif missing_operations:
        reason = "operation_gap_below_minimum"
    elif float(gap.get("M_dep", 0.0) or 0.0) > 0.0:
        reason = "dependency_only_representation_gap"
    else:
        reason = "no_structure_gap"
    return {
        "actionable": actionable,
        "reason": reason,
        "operation_missing_rate": operation_rate,
        "minimum_operation_missing_rate": threshold,
        "missing_operation_instances": sum(missing_operations.values()),
        "unknown_op_count": unknown_count,
    }


def graph_consistency(requirement_graph: dict, program_graph: dict) -> float:
    program_graph = output_dependency_subgraph(program_graph)
    req_ops, req_deps = graph_counters(requirement_graph)
    prog_ops, prog_deps = graph_counters(program_graph)
    unknown = prog_ops.pop("unknown_op", 0)
    if "workplane" not in req_ops:
        prog_ops.pop("workplane", None)
    matched_deps, required_dep_count, observed_dep_count, _ = _dependency_match(
        requirement_graph, program_graph,
    )
    def f1(required: Counter, observed: Counter) -> float:
        overlap = sum((required & observed).values())
        if not required and not observed:
            return 1.0
        precision = overlap / max(sum(observed.values()), 1)
        recall = overlap / max(sum(required.values()), 1)
        return 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    dep_precision = matched_deps / max(observed_dep_count, 1)
    dep_recall = matched_deps / max(required_dep_count, 1)
    dep_f1 = (
        1.0 if not required_dep_count and not observed_dep_count else
        0.0 if dep_precision + dep_recall == 0 else
        2.0 * dep_precision * dep_recall / (dep_precision + dep_recall)
    )
    base = 0.6 * f1(req_ops, prog_ops) + 0.4 * dep_f1
    return base / (1.0 + float(unknown))


def graph_sha256(graph: dict) -> str:
    return hashlib.sha256(json.dumps(graph, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
