"""Dependency-closure localization for full-program CAD repair prompts."""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Sequence

from .graphs import graph_difference, requirement_program_node_mapping
from .operations import canonical_operation, is_transparent_operation
from .program_graph import output_dependency_subgraph


_ISSUE_OPERATION_HINTS = {
    "rotation_mismatch": {"rotate"},
    "translation_mismatch": {"translate"},
    "transform_order_mismatch": {"rotate", "translate"},
    "position_mismatch": {"workplane", "translate", "rotate"},
    "dimension_mismatch": {"box", "cylinder", "sphere", "extrude", "revolve", "loft"},
    "hole_count_mismatch": {"hole", "counterbore_hole", "countersink_hole", "circle", "cut", "pocket"},
    "hole_radius_mismatch": {"hole", "counterbore_hole", "countersink_hole", "circle"},
}


def _error_program_hints(failure: dict) -> tuple[set[int], set[str]]:
    text = " ".join(str(failure.get(key) or "") for key in (
        "stage", "error_type", "error", "stderr",
    ))
    lines = {
        int(value) for value in re.findall(
            r"(?:user_program\.py[^\n]*?|\bline\s+)(?:,?\s*line\s+)?(\d+)",
            text, flags=re.IGNORECASE,
        )
    }
    lowered = text.lower()
    operations: set[str] = set()
    keyword_groups = (
        (("bop", "boolean", "cut", "fuse", "union", "common"), {"cut", "union", "intersect"}),
        (("wire", "profile", "face", "pending"), {"line", "arc", "circle", "close", "profile"}),
        (("extrud", "solid", "volume"), {"extrude", "revolve", "loft", "sweep"}),
        (("fillet",), {"fillet"}),
        (("chamfer", "bevel"), {"chamfer"}),
        (("rotat", "euler"), {"rotate"}),
        (("translat", "position", "location"), {"translate", "workplane"}),
    )
    for keywords, values in keyword_groups:
        if any(keyword in lowered for keyword in keywords):
            operations.update(values)
    return lines, operations


def _issue_operations(issues: Sequence[str]) -> set[str]:
    operations: set[str] = set()
    for issue in issues:
        value = str(issue)
        operations.update(_ISSUE_OPERATION_HINTS.get(value, ()))
        for prefix in (
            "primitive_count_mismatch:", "boolean_relation_mismatch:",
            "unexpected_spatial_operation:",
        ):
            if value.startswith(prefix):
                operations.add(canonical_operation(value[len(prefix):].split(":", 1)[0]))
    return operations


def _graph_relations(graph: dict) -> tuple[dict[str, dict], dict[str, set[str]], dict[str, set[str]]]:
    nodes = {
        str(node.get("id")): node
        for node in graph.get("nodes", [])
        if isinstance(node, dict) and node.get("id") is not None
    }
    parents = {node_id: set() for node_id in nodes}
    children = {node_id: set() for node_id in nodes}
    for node_id, node in nodes.items():
        for parent in node.get("inputs") or []:
            parent_id = str(parent)
            if parent_id in nodes:
                parents[node_id].add(parent_id)
                children[parent_id].add(node_id)
    return nodes, parents, children


def _closure(starts: set[str], adjacency: dict[str, set[str]]) -> set[str]:
    visited = set(starts)
    frontier = deque(sorted(starts))
    while frontier:
        current = frontier.popleft()
        for neighbour in sorted(adjacency.get(current, ())):
            if neighbour not in visited:
                visited.add(neighbour)
                frontier.append(neighbour)
    return visited


def dependency_closure_analysis(
    requirement_graph: dict,
    program_graph: dict,
    *,
    constraint_issues: Sequence[str] = (),
    execution_failure: dict | None = None,
) -> dict:
    """Locate requirement and program regions implicated by observable gaps.

    The result is analysis only: repair candidates remain complete programs.
    It identifies the smallest dependency-connected region that contains the
    missing/mismatched requirement nodes, their prerequisites, and their path
    to a declared output.  Program lines outside the mapped region are marked
    as protected guidance for the LLM and measured after repair.
    """
    active_program = output_dependency_subgraph(program_graph)
    gap = graph_difference(requirement_graph, active_program)
    mapping = requirement_program_node_mapping(requirement_graph, active_program)
    req_nodes, req_parents, req_children = _graph_relations(requirement_graph)
    prog_nodes, prog_parents, prog_children = _graph_relations(active_program)

    semantic_req_ids = {
        node_id for node_id, node in req_nodes.items()
        if not is_transparent_operation(canonical_operation(node.get("op", "")))
    }
    target_req_ids = {node_id for node_id in semantic_req_ids if node_id not in mapping}
    for dependency in gap.get("missing_dependencies") or []:
        target_req_ids.update(
            value for value in (
                str(dependency.get("from_id", "")),
                str(dependency.get("to_id", "")),
            ) if value in req_nodes
        )
    hinted_operations = _issue_operations(constraint_issues)
    error_lines, error_operations = _error_program_hints(execution_failure or {})
    hinted_operations.update(error_operations)
    target_program_ids = {
        node_id for node_id, node in prog_nodes.items()
        if (
            isinstance(node.get("lineno"), int)
            and isinstance(node.get("end_lineno", node.get("lineno")), int)
            and any(
                int(node["lineno"]) <= line <= int(node.get("end_lineno", node["lineno"]))
                for line in error_lines
            )
        )
        or canonical_operation(node.get("op", "")) in error_operations
    }
    inverse_mapping = {program_id: requirement_id for requirement_id, program_id in mapping.items()}
    target_req_ids.update(
        inverse_mapping[node_id] for node_id in target_program_ids if node_id in inverse_mapping
    )
    target_req_ids.update(
        node_id for node_id, node in req_nodes.items()
        if canonical_operation(node.get("op", "")) in hinted_operations
    )

    outputs = requirement_graph.get("output")
    output_ids = {
        str(value) for value in (outputs if isinstance(outputs, list) else [outputs])
        if value is not None and str(value) in req_nodes
    }
    if not target_req_ids:
        target_req_ids.update(output_ids)

    upstream = _closure(target_req_ids, req_parents)
    downstream = _closure(target_req_ids, req_children)
    output_reachable = downstream & _closure(output_ids, req_parents) if output_ids else downstream
    requirement_closure = upstream | output_reachable | target_req_ids

    affected_program_ids = {
        mapping[node_id] for node_id in requirement_closure if node_id in mapping
    }
    # Extra operations named by a deterministic constraint are directly
    # repairable even when no requirement node can map to them.
    affected_program_ids.update(
        node_id for node_id, node in prog_nodes.items()
        if canonical_operation(node.get("op", "")) in hinted_operations
    )
    if target_program_ids:
        program_outputs = active_program.get("output_nodes") or [active_program.get("output")]
        program_output_ids = {
            str(value) for value in program_outputs
            if value is not None and str(value) in prog_nodes
        }
        program_upstream = _closure(target_program_ids, prog_parents)
        program_downstream = _closure(target_program_ids, prog_children)
        program_to_output = (
            program_downstream & _closure(program_output_ids, prog_parents)
            if program_output_ids else program_downstream
        )
        affected_program_ids.update(program_upstream | program_to_output)
    if not affected_program_ids:
        program_outputs = active_program.get("output_nodes") or [active_program.get("output")]
        affected_program_ids.update(
            str(value) for value in program_outputs if value is not None and str(value) in prog_nodes
        )

    def line_span(node: dict) -> list[int] | None:
        start = node.get("lineno")
        end = node.get("end_lineno", start)
        if isinstance(start, int) and isinstance(end, int):
            return [start, max(start, end)]
        return None

    affected_spans = sorted({
        tuple(span) for node_id in affected_program_ids
        if (span := line_span(prog_nodes.get(node_id, {}))) is not None
    })
    protected_program_ids = sorted(set(prog_nodes) - affected_program_ids)
    protected_spans = sorted({
        tuple(span) for node_id in protected_program_ids
        if (span := line_span(prog_nodes[node_id])) is not None
    })

    def requirement_view(node_id: str) -> dict:
        node = req_nodes[node_id]
        return {
            "id": node_id,
            "op": canonical_operation(node.get("op", "")),
            "span": str(node.get("span", "")),
        }

    semantic_requirement_closure = requirement_closure & semantic_req_ids
    mapping_coverage = len(mapping) / max(len(semantic_req_ids), 1)
    closure_program_ratio = len(affected_program_ids) / max(len(prog_nodes), 1)
    closure_requirement_ratio = (
        len(semantic_requirement_closure) / max(len(semantic_req_ids), 1)
    )
    seed_strength = 1.0 if (target_program_ids or constraint_issues) else (
        0.5 if target_req_ids else 0.0
    )
    localization_confidence = (
        0.45 * mapping_coverage
        + 0.35 * max(0.0, 1.0 - closure_program_ratio)
        + 0.20 * seed_strength
    )
    localization_reliable = bool(
        affected_spans
        and (target_req_ids or target_program_ids)
        # A closure that spans most of the program no longer localizes a
        # repair.  Feeding that broad region to the LLM duplicated the full
        # structure gap while over-constraining otherwise valid code.  Require
        # both a minority program region and sufficiently trustworthy
        # requirement-to-program alignment before enabling closure guidance.
        and closure_program_ratio <= 0.50
        and mapping_coverage >= 0.75
        and localization_confidence >= 0.55
    )

    return {
        "target_requirement_nodes": [
            requirement_view(node_id) for node_id in sorted(target_req_ids)
        ],
        "upstream_requirement_node_ids": sorted(upstream),
        "downstream_to_output_node_ids": sorted(output_reachable),
        "requirement_closure_node_ids": sorted(requirement_closure),
        "affected_program_node_ids": sorted(affected_program_ids),
        "error_seed_program_node_ids": sorted(target_program_ids),
        "error_line_numbers": sorted(error_lines),
        "affected_line_spans": [list(span) for span in affected_spans],
        "protected_program_node_ids": protected_program_ids,
        "protected_line_spans": [list(span) for span in protected_spans],
        "requirement_to_program_mapping": {
            key: mapping[key] for key in sorted(mapping)
        },
        "closure_requirement_ratio": closure_requirement_ratio,
        "closure_program_ratio": closure_program_ratio,
        "mapping_coverage": mapping_coverage,
        "localization_confidence": localization_confidence,
        "trigger_constraint_issues": list(dict.fromkeys(map(str, constraint_issues))),
        "localization_reliable": localization_reliable,
        "structure_gap": gap,
    }
