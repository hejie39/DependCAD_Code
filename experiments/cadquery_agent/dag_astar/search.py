"""Budgeted structural A* repair with reference-free candidate acceptance."""

from __future__ import annotations

import ast
import difflib
import hashlib
import heapq
import inspect
import queue as queue_module
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

from .constraints import validate_requirement_constraints
from .graphs import (
    asymmetric_graph_difference,
    cross_graph_alignment_gate,
    graph_counters,
)
from .program_graph import analyse_static, build_program_graph, output_dependency_subgraph


Executor = Callable[..., dict]
CandidateProvider = Callable[..., tuple[list[str], dict]]
ConstraintValidator = Callable[[str, dict, dict], dict]


@dataclass(frozen=True)
class SearchBudget:
    max_expansions: int = 6
    max_candidate_executions: int = 5
    max_llm_calls: int = 2
    max_seconds: float = 180.0


@dataclass(frozen=True)
class HeuristicConfig:
    static_weight: float = 3.0
    operation_weight: float = 8.0
    dependency_weight: float = 5.0
    constraint_weight: float = 4.0
    role_dependency_weight: float = 6.0
    path_dependency_weight: float = 4.0
    output_dependency_weight: float = 5.0
    extra_operation_weight: float = 3.0
    outside_closure_edit_weight: float = 1.5
    minimum_operation_gap_for_alignment: float = 0.05
    minimum_graph_confidence: float = field(default_factory=lambda: float(__import__("os").environ.get("TRUSTCAD_CONFIDENCE_TAU", "0.85")))
    minimum_mapping_confidence: float = field(default_factory=lambda: float(__import__("os").environ.get("TRUSTCAD_CONFIDENCE_TAU", "0.85")))
    use_graph_alignment: bool = True
    rank_best_by_structural_score: bool = True
    continue_after_acceptance: bool = True
    failure_penalties: dict[str, float] = field(default_factory=lambda: {
        # A program that cannot produce a valid solid must not outrank an
        # executable candidate with one non-structural requirement mismatch.
        # The latter costs constraint(4) + failure(4) = 8 by default.
        "static": 12.0, "timeout": 11.0, "empty": 10.0,
        "geometry": 10.0, "constraint": 4.0, "other": 10.0,
    })


def code_fingerprint(code: str) -> str:
    try:
        canonical = ast.dump(ast.parse(code), annotate_fields=True, include_attributes=False)
    except SyntaxError:
        canonical = code
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _safe_program_graph(code: str) -> dict:
    """Represent a syntactically invalid repair root without aborting search."""
    try:
        return build_program_graph(code)
    except SyntaxError as error:
        return {
            "nodes": [{"id": "p0", "op": "unknown_op", "inputs": []}],
            "output": None,
            "output_nodes": [],
            "graph_kind": "cadquery_program_operation_dag",
            "dynamic_control_flow": False,
            "source_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
            "active_node_ids": [],
            "inactive_node_ids": ["p0"],
            "parse_error": f"{type(error).__name__}:{error}",
        }


def edit_similarity(left: str, right: str) -> float:
    return difflib.SequenceMatcher(a=left.splitlines(), b=right.splitlines()).ratio()


def closure_edit_locality(parent: str, candidate: str, closure: dict | None) -> dict:
    """Measure how much of a repair edit lies outside localized source lines."""
    if not closure or not closure.get("localization_reliable"):
        return {"measured": False, "outside_edit_ratio": None, "changed_units": 0}
    affected: set[int] = set()
    for span in closure.get("affected_line_spans") or []:
        if isinstance(span, (list, tuple)) and len(span) == 2:
            try:
                start, end = int(span[0]), int(span[1])
            except (TypeError, ValueError):
                continue
            affected.update(range(max(1, start), max(start, end) + 1))
    if not affected:
        return {"measured": False, "outside_edit_ratio": None, "changed_units": 0}
    before = parent.splitlines()
    after = candidate.splitlines()
    changed_units = 0
    outside_units = 0.0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=before, b=after).get_opcodes():
        if tag == "equal":
            continue
        units = max(i2 - i1, j2 - j1, 1)
        old_lines = set(range(i1 + 1, i2 + 1))
        if old_lines:
            outside_fraction = len(old_lines - affected) / len(old_lines)
        else:
            anchors = {max(1, i1), min(len(before), i1 + 1)} if before else set()
            outside_fraction = 0.0 if anchors & affected else 1.0
        changed_units += units
        outside_units += units * outside_fraction
    return {
        "measured": True,
        "outside_edit_ratio": outside_units / max(changed_units, 1),
        "changed_units": changed_units,
        "outside_changed_units": outside_units,
    }


def failure_category(execution: dict) -> str:
    if execution.get("success"):
        return "success"
    text = " ".join(str(execution.get(key) or "") for key in (
        "stage", "error_type", "error", "stderr",
    )).lower()
    if "requirementconstraint" in text or "constraint" in text:
        return "constraint"
    if "syntax" in text or "validation" in text:
        return "static"
    if "timeout" in text:
        return "timeout"
    if "empty" in text or "no exportable" in text or "result" in text:
        return "empty"
    if any(term in text for term in ("topology", "invalid", "solid", "occ", "bop", "volume")):
        return "geometry"
    return "other"


def _effective_execution(execution: dict, acceptance: dict) -> dict:
    if acceptance.get("accepted"):
        return execution
    effective = dict(execution)
    effective["constraint_validation"] = acceptance
    # Preserve the real OCC/static/timeout failure class.  Replacing every
    # rejected candidate with a constraint failure gives non-executable code
    # the cheap constraint penalty and can make it outrank executable nodes.
    if not execution.get("success"):
        return effective
    effective.update({
        "success": False,
        "stage": "constraint",
        "error_type": "RequirementConstraintViolation",
        "error": "|".join(acceptance.get("issues") or ["candidate_not_accepted"]),
    })
    return effective


def _constraint_issue_severity(issue: str) -> int:
    """Return deterministic issue multiplicity for the heuristic.

    Count-bearing diagnostics represent several observable surplus operations.
    Treating ``rotate:3`` and ``rotate:1`` as one issue erases information that
    the requirement validator already established and can reverse heap order.
    """
    if issue.startswith(("unexpected_spatial_operation:", "unknown_operations:")):
        try:
            return max(1, int(issue.rsplit(":", 1)[-1]))
        except ValueError:
            return 1
    return 1


_OPERATION_IMPORTANCE = {
    "cut": 3.0, "union": 3.0, "intersect": 3.0,
    "hole": 2.5, "counterbore_hole": 2.5, "countersink_hole": 2.5,
    "extrude": 2.0, "revolve": 2.0, "loft": 2.0, "sweep": 2.0,
    "box": 2.0, "cylinder": 2.0, "sphere": 2.0, "pocket": 2.0,
    "rotate": 1.5, "translate": 1.5, "scale": 1.5,
    "mirror": 1.5, "pattern": 1.5,
    "workplane": 0.5, "close": 0.5,
}


def _weighted_operation_missing_rate(requirement_graph: dict, program_graph: dict) -> float:
    required, _ = graph_counters(requirement_graph)
    observed, _ = graph_counters(output_dependency_subgraph(program_graph))
    denominator = sum(
        count * _OPERATION_IMPORTANCE.get(operation, 1.0)
        for operation, count in required.items()
        if operation != "unknown_op"
    )
    if denominator <= 0:
        return 0.0
    numerator = sum(
        max(0, count - observed[operation]) * _OPERATION_IMPORTANCE.get(operation, 1.0)
        for operation, count in required.items()
        if operation != "unknown_op"
    )
    return min(1.0, numerator / denominator)


def _dependency_missing_rates(requirement_graph: dict, gap: dict) -> tuple[float, float]:
    _, dependencies = graph_counters(requirement_graph)
    role_total = sum(
        count for (_, _, role), count in dependencies.items()
        if role in {"base", "tool"}
    )
    path_total = sum(
        count for (_, _, role), count in dependencies.items()
        if role not in {"base", "tool"}
    )
    role_missing = sum(
        int(item.get("count", 1)) for item in gap.get("missing_dependencies") or []
        if item.get("role") in {"base", "tool"}
    )
    path_missing = sum(
        int(item.get("count", 1)) for item in gap.get("missing_dependencies") or []
        if item.get("role") not in {"base", "tool"}
    )
    return (
        0.0 if role_total == 0 else min(1.0, role_missing / role_total),
        0.0 if path_total == 0 else min(1.0, path_missing / path_total),
    )


def _output_dependency_missing_rate(requirement_graph: dict, program_graph: dict) -> float:
    required, _ = graph_counters(requirement_graph)
    full, _ = graph_counters(program_graph)
    active, _ = graph_counters(output_dependency_subgraph(program_graph))
    required_total = sum(count for operation, count in required.items() if operation != "unknown_op")
    if required_total == 0:
        return 0.0
    disconnected = 0
    for operation, count in required.items():
        if operation == "unknown_op":
            continue
        missing_active = max(0, count - active[operation])
        available_inactive = max(0, full[operation] - active[operation])
        disconnected += min(missing_active, available_inactive)
    return min(1.0, disconnected / required_total)


def heuristic(
    code: str, execution: dict, requirement_graph: dict,
    config: HeuristicConfig | None = None, acceptance: dict | None = None,
) -> tuple[float, dict]:
    config = config or HeuristicConfig()
    static = analyse_static(code)
    if config.use_graph_alignment:
        program_graph = _safe_program_graph(code)
        gap = asymmetric_graph_difference(requirement_graph, program_graph)
        alignment_gate = cross_graph_alignment_gate(
            gap,
            minimum_operation_missing_rate=config.minimum_operation_gap_for_alignment,
        )
        weighted_operation_gap = _weighted_operation_missing_rate(
            gap["requirement_anchor_graph"], gap["program_anchor_graph"],
        )
        role_gap, path_gap = _dependency_missing_rates(
            gap["requirement_anchor_graph"], gap,
        )
        output_gap = _output_dependency_missing_rate(
            gap["requirement_anchor_graph"], program_graph,
        )
    else:
        gap = {
            "M_op": 0.0, "M_dep": 0.0,
            "missing_operations": [], "missing_dependencies": [],
            "unknown_op_count": 0,
        }
        alignment_gate = {
            "actionable": False,
            "reason": "cross_graph_alignment_disabled",
            "operation_missing_rate": 0.0,
            "minimum_operation_missing_rate": config.minimum_operation_gap_for_alignment,
            "missing_operation_instances": 0,
            "unknown_op_count": 0,
        }
        weighted_operation_gap = role_gap = path_gap = output_gap = 0.0
    category = failure_category(execution)
    penalty = 0.0 if category == "success" else config.failure_penalties[category]
    nonstructural_constraints = []
    if acceptance:
        nonstructural_constraints = [
            issue for issue in acceptance.get("issues", [])
            if not issue.startswith(("missing_operations:", "missing_dependencies:", "execution:"))
        ]
    constraint_severity = sum(
        _constraint_issue_severity(issue) for issue in nonstructural_constraints
    )
    constraint_fidelity = float(
        acceptance.get("constraint_fidelity", acceptance.get("constraint_pass_rate", 0.0))
        if acceptance else 0.0
    )
    constraint_residual = max(0.0, min(1.0, 1.0 - constraint_fidelity))
    extra_severity = sum(
        _constraint_issue_severity(issue) for issue in nonstructural_constraints
        if issue.startswith(("unexpected_spatial_operation:", "unexpected_boolean_operation:"))
    )
    required_operations, _ = graph_counters(requirement_graph)
    extra_rate = min(1.0, extra_severity / max(sum(required_operations.values()), 1))
    graph_confidence = float(
        acceptance.get("graph_confidence", 1.0) if acceptance else 1.0
    )
    mapping_confidence = float(gap.get("mapping_confidence", 0.0))
    confidence_gate = bool(
        graph_confidence >= config.minimum_graph_confidence
        and mapping_confidence >= config.minimum_mapping_confidence
    )
    structural_confidence = (
        graph_confidence * mapping_confidence
        if alignment_gate["actionable"] and confidence_gate else 0.0
    )
    components = {
        "N_static": len(static.countable),
        "N_constraint": constraint_severity,
        "R_constraint": constraint_residual,
        "constraint_fidelity": constraint_fidelity,
        "constraint_issues": nonstructural_constraints,
        "M_op": gap["M_op"], "M_op_weighted": weighted_operation_gap,
        "M_dep": gap["M_dep"], "M_role": role_gap, "M_path": path_gap,
        "M_output": output_gap, "M_extra": extra_rate,
        "structural_confidence": structural_confidence,
        "graph_confidence": graph_confidence,
        "mapping_confidence": mapping_confidence,
        "confidence_gate": confidence_gate,
        "alignment_gate": alignment_gate,
        "P_fail": penalty, "failure_category": category,
        "missing_operations": gap["missing_operations"],
        "missing_dependencies": gap["missing_dependencies"],
        "unknown_op_count": gap["unknown_op_count"],
    }
    total = (
        config.static_weight * components["N_static"]
        + config.constraint_weight * (
            components["N_constraint"] + components["R_constraint"]
        )
        + structural_confidence * (
            config.operation_weight * components["M_op_weighted"]
            + config.dependency_weight * components["M_dep"]
            + config.role_dependency_weight * components["M_role"]
            + config.path_dependency_weight * components["M_path"]
            + config.output_dependency_weight * components["M_output"]
            # Unexpected geometry-changing operations are already counted in
            # the reference-free constraint term and are not a symmetric graph
            # edit penalty.
        )
        + components["P_fail"]
    )
    components["total"] = float(total)
    return float(total), components


@dataclass
class SearchNode:
    node_id: str
    code: str
    parent: str | None
    program_graph: dict
    error: dict
    acceptance: dict
    execution: dict
    g: float
    h: float
    f: float
    depth: int
    expanded: bool
    hash: str
    heuristic: dict
    children: list[str] = field(default_factory=list)


@dataclass
class SearchResult:
    success: bool
    code: str
    execution: dict
    acceptance: dict
    terminal_node: str
    tree: dict[str, dict]
    trace: list[dict]
    expansions: int
    candidate_executions: int
    llm_calls: int
    provider_errors: int
    elapsed_seconds: float
    exhausted: str | None

    def to_dict(self) -> dict:
        return asdict(self)


class HeuristicAStarRepair:
    def __init__(
        self, executor: Executor, requirement_graph: dict, *,
        budget: SearchBudget | None = None,
        heuristic_config: HeuristicConfig | None = None,
        constraint_validator: ConstraintValidator = validate_requirement_constraints,
    ):
        self.executor = executor
        self.requirement_graph = requirement_graph
        self.budget = budget or SearchBudget()
        self.heuristic_config = heuristic_config or HeuristicConfig()
        self.constraint_validator = constraint_validator

    def _accept(self, code: str, execution: dict) -> dict:
        return self.constraint_validator(code, execution, self.requirement_graph)

    @staticmethod
    def _accepts_timeout(function: Callable, positional_without_timeout: int) -> bool:
        try:
            parameters = list(inspect.signature(function).parameters.values())
        except (TypeError, ValueError):
            return False
        return any(item.kind == item.VAR_POSITIONAL for item in parameters) or len(parameters) > positional_without_timeout

    @classmethod
    def _deadline_call(
        cls, function: Callable, args: tuple, remaining: float, *, label: str,
    ):
        if remaining <= 0:
            raise TimeoutError(f"{label}_global_budget_exhausted")
        output: queue_module.Queue = queue_module.Queue(maxsize=1)
        call_args = (*args, remaining) if cls._accepts_timeout(function, len(args)) else args
        def target() -> None:
            try:
                output.put((True, function(*call_args)))
            except BaseException as error:  # propagate provider/runtime failures to search
                output.put((False, error))
        worker = threading.Thread(target=target, daemon=True, name=f"cad_{label}_deadline")
        worker.start()
        try:
            ok, value = output.get(timeout=max(0.001, remaining))
        except queue_module.Empty as error:
            raise TimeoutError(f"{label}_exceeded_global_budget:{remaining:.3f}s") from error
        if not ok:
            raise value
        return value

    def _best_rank(self, node: SearchNode) -> tuple:
        if not self.heuristic_config.rank_best_by_structural_score:
            return (
                0 if node.execution.get("success") else 1,
                node.g, node.depth, node.node_id,
            )
        if node.acceptance.get("accepted"):
            return (
                0,
                0 if node.execution.get("success") else 1,
                -float(node.acceptance.get("constraint_fidelity", 0.0)),
                -float(node.acceptance.get("constraint_pass_rate", 0.0)),
                -float(node.acceptance.get("structural_score", 0.0)),
                node.g, node.depth, node.node_id,
            )
        return (
            1,
            0 if node.execution.get("success") else 1,
            -float(node.acceptance.get("constraint_pass_rate", 0.0)),
            -float(node.acceptance.get("constraint_fidelity", 0.0)),
            node.h, node.f,
            -float(node.acceptance.get("structural_score", 0.0)),
            node.g, node.depth, node.node_id,
        )

    def search(self, code: str, root_execution: dict, provider: CandidateProvider) -> SearchResult:
        started = time.perf_counter()
        root_acceptance = self._accept(code, root_execution)
        root_effective = _effective_execution(root_execution, root_acceptance)
        root_h, root_components = heuristic(
            code, root_effective, self.requirement_graph, self.heuristic_config, root_acceptance,
        )
        root = SearchNode(
            "n0", code, None, _safe_program_graph(code), root_effective, root_acceptance,
            root_execution, 0.0, root_h, root_h, 0, False, code_fingerprint(code), root_components,
        )
        nodes = {root.node_id: root}
        if root_acceptance.get("accepted"):
            return self._result(True, root, nodes, [], 0, 0, 0, 0, started, None)
        queue: list[tuple[float, float, int, str]] = [(root.f, root.g, 0, root.node_id)]
        best_g = {root.hash: 0.0}
        trace: list[dict] = [{
            "event": "root", "node_id": root.node_id, "g": 0.0, "h": root.h,
            "f": root.f, "acceptance": root_acceptance,
        }]
        insertion_id = 0
        expansions = candidate_executions = llm_calls = provider_errors = 0
        best = root
        exhausted: str | None = None

        while queue:
            if time.perf_counter() - started >= self.budget.max_seconds:
                exhausted = "time"
                break
            if expansions >= self.budget.max_expansions:
                exhausted = "expansions"
                break
            if llm_calls >= self.budget.max_llm_calls:
                exhausted = "llm_calls"
                break
            _, _, _, node_id = heapq.heappop(queue)
            current = nodes[node_id]
            if current.expanded or current.g > best_g.get(current.hash, float("inf")):
                continue
            current.expanded = True
            expansions += 1
            llm_calls += 1
            try:
                remaining = self.budget.max_seconds - (time.perf_counter() - started)
                candidates, record = self._deadline_call(
                    provider, (current.code, current.error), remaining, label="llm_request",
                )
                if not isinstance(candidates, (list, tuple)):
                    raise TypeError("candidate_provider_must_return_a_sequence")
            except Exception as error:
                provider_errors += 1
                trace.append({
                    "event": "provider_error", "node_id": node_id,
                    "error_type": type(error).__name__, "error": str(error),
                    "attempts": list(getattr(error, "attempts", [])),
                })
                if isinstance(error, TimeoutError):
                    exhausted = "time"
                    break
                if llm_calls < self.budget.max_llm_calls and expansions < self.budget.max_expansions:
                    current.expanded = False
                    insertion_id += 1
                    heapq.heappush(queue, (current.f, current.g, insertion_id, current.node_id))
                continue
            trace.append({
                "event": "expand", "node_id": node_id,
                "candidate_count": min(len(candidates), 2), "llm_record": record,
            })
            closure = record.get("dependency_closure") if isinstance(record, dict) else None
            child_count_before = len(current.children)
            rejection_feedback: list[str] = []
            for candidate in candidates[:2]:
                if not isinstance(candidate, str) or not candidate.strip():
                    trace.append({"event": "reject_malformed", "parent": node_id})
                    rejection_feedback.append("malformed_or_empty_candidate")
                    continue
                if time.perf_counter() - started >= self.budget.max_seconds:
                    exhausted = "time"
                    break
                static = analyse_static(candidate)
                if static.fatal:
                    trace.append({"event": "reject_static", "parent": node_id, "issues": static.fatal})
                    rejection_feedback.extend(static.fatal)
                    continue
                marker = code_fingerprint(candidate)
                similarity = edit_similarity(candidate, current.code)
                locality = closure_edit_locality(current.code, candidate, closure)
                outside_ratio = locality.get("outside_edit_ratio")
                locality_penalty = (
                    self.heuristic_config.outside_closure_edit_weight * float(outside_ratio)
                    if isinstance(outside_ratio, (int, float)) else 0.0
                )
                step_cost = 2.0 + 4.0 * (1.0 - similarity) + locality_penalty
                next_g = current.g + step_cost
                if marker in best_g and next_g >= best_g[marker] - 1e-12:
                    trace.append({"event": "reject_duplicate", "parent": node_id, "hash": marker})
                    rejection_feedback.append("duplicate_candidate")
                    continue
                if candidate_executions >= self.budget.max_candidate_executions:
                    exhausted = "candidate_executions"
                    break
                try:
                    remaining = self.budget.max_seconds - (time.perf_counter() - started)
                    execution = self._deadline_call(
                        self.executor, (candidate, candidate_executions), remaining,
                        label="candidate_execution",
                    )
                    if not isinstance(execution, dict):
                        raise TypeError("executor_result_must_be_a_dict")
                except Exception as error:
                    execution = {
                        "success": False, "stage": "executor_wrapper",
                        "error_type": type(error).__name__, "error": str(error),
                    }
                candidate_executions += 1
                if time.perf_counter() - started >= self.budget.max_seconds:
                    exhausted = "time"
                acceptance = self._accept(candidate, execution)
                effective = _effective_execution(execution, acceptance)
                next_h, components = heuristic(
                    candidate, effective, self.requirement_graph, self.heuristic_config, acceptance,
                )
                child_id = f"n{len(nodes)}"
                child = SearchNode(
                    child_id, candidate, node_id, _safe_program_graph(candidate), effective, acceptance,
                    execution, next_g, next_h, next_g + next_h, current.depth + 1, False, marker, components,
                )
                nodes[child_id] = child
                current.children.append(child_id)
                best_g[marker] = next_g
                trace.append({
                    "event": "candidate", "node_id": child_id, "parent": node_id,
                    "execution_success": bool(execution.get("success")),
                    "accepted": bool(acceptance.get("accepted")), "acceptance": acceptance,
                    "S_edit": similarity, "step_cost": step_cost,
                    "g": child.g, "h": child.h, "f": child.f,
                    "closure_edit_locality": locality,
                    "closure_edit_penalty": locality_penalty,
                })
                if self._best_rank(child) < self._best_rank(best):
                    best = child
                if acceptance.get("accepted"):
                    trace.append({
                        "event": "accepted_frontier_candidate",
                        "node_id": child_id,
                        "continue_search": self.heuristic_config.continue_after_acceptance,
                    })
                    if not self.heuristic_config.continue_after_acceptance:
                        return self._result(
                            True, child, nodes, trace, expansions, candidate_executions,
                            llm_calls, provider_errors, started, None,
                        )
                    continue
                insertion_id += 1
                heapq.heappush(queue, (child.f, child.g, insertion_id, child_id))
            if exhausted in {"time", "candidate_executions"}:
                break
            if (
                len(current.children) == child_count_before
                and llm_calls < self.budget.max_llm_calls
                and expansions < self.budget.max_expansions
            ):
                if not rejection_feedback:
                    rejection_feedback.append("no_novel_candidates")
                current.error = dict(current.error)
                previous = str(current.error.get("error") or "")
                feedback = "previous_repair_rejected:" + ",".join(sorted(set(rejection_feedback)))
                current.error["error"] = "|".join(value for value in (previous, feedback) if value)
                current.expanded = False
                insertion_id += 1
                heapq.heappush(queue, (current.f, current.g, insertion_id, current.node_id))
                trace.append({
                    "event": "retry_no_viable_candidates", "node_id": current.node_id,
                    "issues": sorted(set(rejection_feedback)),
                })
        return self._result(
            bool(best.acceptance.get("accepted")), best, nodes, trace,
            expansions, candidate_executions,
            llm_calls, provider_errors, started, exhausted or "queue_empty",
        )

    @staticmethod
    def _result(
        success: bool, terminal: SearchNode, nodes: dict[str, SearchNode], trace: list[dict],
        expansions: int, candidate_executions: int, llm_calls: int, provider_errors: int,
        started: float, exhausted: str | None,
    ) -> SearchResult:
        return SearchResult(
            success, terminal.code, terminal.execution, terminal.acceptance, terminal.node_id,
            {key: asdict(value) for key, value in nodes.items()}, trace, expansions,
            candidate_executions, llm_calls, provider_errors,
            time.perf_counter() - started, exhausted,
        )


class GreedyRetryRepair:
    """Sequential repair baseline without a heap or structural candidate ranking.

    Candidates are executed in provider order.  After a failed round, the last
    viable candidate becomes the input to the next LLM call and the final
    fallback.  The class intentionally never computes ``g``, ``h`` or ``f``.
    """

    def __init__(
        self, executor: Executor, requirement_graph: dict, *,
        budget: SearchBudget | None = None,
        constraint_validator: ConstraintValidator = validate_requirement_constraints,
    ):
        self.executor = executor
        self.requirement_graph = requirement_graph
        self.budget = budget or SearchBudget()
        self.constraint_validator = constraint_validator

    def _accept(self, code: str, execution: dict) -> dict:
        return self.constraint_validator(code, execution, self.requirement_graph)

    def search(self, code: str, root_execution: dict, provider: CandidateProvider) -> SearchResult:
        started = time.perf_counter()
        root_acceptance = self._accept(code, root_execution)
        root_error = _effective_execution(root_execution, root_acceptance)
        root_hash = code_fingerprint(code)
        nodes: dict[str, dict] = {
            "n0": {
                "node_id": "n0", "code": code, "parent": None,
                "execution": root_execution, "error": root_error,
                "acceptance": root_acceptance, "hash": root_hash,
                "sequence_index": 0, "strategy": "sequential_greedy",
                "children": [],
            },
        }
        trace: list[dict] = [{
            "event": "root", "node_id": "n0", "acceptance": root_acceptance,
            "strategy": "sequential_greedy",
        }]
        if root_acceptance.get("accepted"):
            return SearchResult(
                True, code, root_execution, root_acceptance, "n0", nodes, trace,
                0, 0, 0, 0, time.perf_counter() - started, None,
            )

        current_id = "n0"
        current_code = code
        current_execution = root_execution
        current_acceptance = root_acceptance
        current_error = root_error
        seen = {root_hash}
        candidate_executions = llm_calls = provider_errors = 0
        exhausted: str | None = None

        while llm_calls < self.budget.max_llm_calls:
            elapsed = time.perf_counter() - started
            if elapsed >= self.budget.max_seconds:
                exhausted = "time"
                break
            if candidate_executions >= self.budget.max_candidate_executions:
                exhausted = "candidate_executions"
                break
            provider_parent = current_id
            llm_calls += 1
            try:
                remaining = self.budget.max_seconds - elapsed
                candidates, record = HeuristicAStarRepair._deadline_call(
                    provider, (current_code, current_error), remaining, label="llm_request",
                )
                if not isinstance(candidates, (list, tuple)):
                    raise TypeError("candidate_provider_must_return_a_sequence")
            except Exception as error:
                provider_errors += 1
                trace.append({
                    "event": "provider_error", "node_id": current_id,
                    "error_type": type(error).__name__, "error": str(error),
                    "attempts": list(getattr(error, "attempts", [])),
                })
                if isinstance(error, TimeoutError):
                    exhausted = "time"
                    break
                continue

            trace.append({
                "event": "retry", "node_id": current_id,
                "candidate_count": min(len(candidates), 2), "llm_record": record,
                "strategy": "sequential_greedy",
            })
            viable_in_round = 0
            rejection_feedback: list[str] = []
            for candidate_index, candidate in enumerate(candidates[:2]):
                if time.perf_counter() - started >= self.budget.max_seconds:
                    exhausted = "time"
                    break
                if candidate_executions >= self.budget.max_candidate_executions:
                    exhausted = "candidate_executions"
                    break
                if not isinstance(candidate, str) or not candidate.strip():
                    rejection_feedback.append("malformed_or_empty_candidate")
                    trace.append({
                        "event": "reject_malformed", "parent": provider_parent,
                        "candidate_index": candidate_index,
                    })
                    continue
                static = analyse_static(candidate)
                if static.fatal:
                    rejection_feedback.extend(static.fatal)
                    trace.append({
                        "event": "reject_static", "parent": provider_parent,
                        "candidate_index": candidate_index, "issues": static.fatal,
                    })
                    continue
                marker = code_fingerprint(candidate)
                if marker in seen:
                    rejection_feedback.append("duplicate_candidate")
                    trace.append({
                        "event": "reject_duplicate", "parent": provider_parent,
                        "candidate_index": candidate_index, "hash": marker,
                    })
                    continue
                try:
                    remaining = self.budget.max_seconds - (time.perf_counter() - started)
                    execution = HeuristicAStarRepair._deadline_call(
                        self.executor, (candidate, candidate_executions), remaining,
                        label="candidate_execution",
                    )
                    if not isinstance(execution, dict):
                        raise TypeError("executor_result_must_be_a_dict")
                except Exception as error:
                    execution = {
                        "success": False, "stage": "executor_wrapper",
                        "error_type": type(error).__name__, "error": str(error),
                    }
                candidate_executions += 1
                acceptance = self._accept(candidate, execution)
                effective = _effective_execution(execution, acceptance)
                child_id = f"n{len(nodes)}"
                nodes[child_id] = {
                    "node_id": child_id, "code": candidate, "parent": provider_parent,
                    "execution": execution, "error": effective,
                    "acceptance": acceptance, "hash": marker,
                    "sequence_index": candidate_executions,
                    "provider_candidate_index": candidate_index,
                    "strategy": "sequential_greedy", "children": [],
                }
                nodes[provider_parent]["children"].append(child_id)
                seen.add(marker)
                viable_in_round += 1
                current_id = child_id
                current_code = candidate
                current_execution = execution
                current_acceptance = acceptance
                current_error = effective
                trace.append({
                    "event": "candidate", "node_id": child_id,
                    "parent": provider_parent, "candidate_index": candidate_index,
                    "execution_success": bool(execution.get("success")),
                    "accepted": bool(acceptance.get("accepted")),
                    "acceptance": acceptance,
                    "selection": "last_viable_candidate",
                })
                if acceptance.get("accepted"):
                    return SearchResult(
                        True, current_code, current_execution, current_acceptance,
                        current_id, nodes, trace, llm_calls, candidate_executions,
                        llm_calls, provider_errors, time.perf_counter() - started, None,
                    )
            if exhausted in {"time", "candidate_executions"}:
                break
            if viable_in_round == 0:
                if not rejection_feedback:
                    rejection_feedback.append("no_novel_candidates")
                current_error = dict(current_error)
                previous = str(current_error.get("error") or "")
                feedback = "previous_repair_rejected:" + ",".join(
                    sorted(set(rejection_feedback))
                )
                current_error["error"] = "|".join(
                    value for value in (previous, feedback) if value
                )
                nodes[current_id]["error"] = current_error
                trace.append({
                    "event": "retry_no_viable_candidates", "node_id": current_id,
                    "issues": sorted(set(rejection_feedback)),
                })

        if exhausted is None:
            exhausted = "llm_calls"
        return SearchResult(
            False, current_code, current_execution, current_acceptance, current_id,
            nodes, trace, llm_calls, candidate_executions, llm_calls,
            provider_errors, time.perf_counter() - started, exhausted,
        )
