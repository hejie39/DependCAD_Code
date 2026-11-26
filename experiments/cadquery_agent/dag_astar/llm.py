"""Time-policy LLM routing plus requirement/code/repair prompt adapters."""

from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence
from zoneinfo import ZoneInfo

from ..ast_graph import normalize_cadquery_code, strip_code_fence
from ..llm_client import LLMRequestError, OpenAICompatibleClient
from .closure import dependency_closure_analysis
from .graphs import asymmetric_graph_difference, cross_graph_alignment_gate
from .program_graph import analyse_static, build_program_graph


SHANGHAI = ZoneInfo("Asia/Shanghai")
PROMPT_VERSIONS = {
    "code_generation": "cadquery_codegen_v23_requirement_dag_and_blueprints",
    "repair": "cadquery_structural_repair_v29_asymmetric_anchor_dual_gate",
    "replan": "cadquery_replan_v18_no_display_helpers_static_type_guard",
}


def compact_blueprint_for_prompt(graph: dict) -> dict:
    """Serialize structure-only examples without repeating JSON field names per node."""

    return {
        "format": "node_rows=[id,op,inputs,roles]",
        "node_rows": [
            [
                str(node.get("id")),
                str(node.get("op")),
                [str(value) for value in node.get("inputs", [])],
                [str(value) for value in node.get("roles", [])],
            ]
            for node in graph.get("nodes", [])
        ],
        "output": graph.get("output"),
    }


def is_peak_time(now: datetime | None = None) -> bool:
    local = (now or datetime.now(SHANGHAI)).astimezone(SHANGHAI)
    minute = local.hour * 60 + local.minute
    return 9 * 60 <= minute < 12 * 60 or 14 * 60 <= minute < 18 * 60


class PolicyRoutedClient:
    """Qwen-only at peak; Qwen then explicit DeepSeek fallback off peak."""

    def __init__(self, *, qwen_model: str = "qwen3.7-flash", deepseek_model: str = "deepseek-chat"):
        self.qwen = OpenAICompatibleClient(provider="qwen", model=qwen_model)
        self.deepseek_model = deepseek_model
        self._deepseek: OpenAICompatibleClient | None = None

    def cache_identity(self) -> dict:
        return {
            "router": "shanghai_peak_policy_v2",
            "qwen": self.qwen.cache_identity(),
            "deepseek_model": self.deepseek_model,
        }

    def complete(
        self, system: str, user: str, max_tokens: int = 2600,
        timeout_seconds: float | None = None,
    ):
        started = time.perf_counter()
        try:
            return self.qwen.complete(system, user, max_tokens, timeout_seconds)
        except LLMRequestError as qwen_error:
            if is_peak_time():
                raise LLMRequestError(f"peak_qwen_required_and_failed:{qwen_error}", qwen_error.attempts) from qwen_error
            if not os.getenv("DEEPSEEK_API_KEY", "").strip():
                raise LLMRequestError(
                    f"offpeak_qwen_failed_and_deepseek_key_missing:{qwen_error}", qwen_error.attempts,
                ) from qwen_error
            if self._deepseek is None:
                self._deepseek = OpenAICompatibleClient(provider="deepseek", model=self.deepseek_model)
            try:
                remaining = None if timeout_seconds is None else max(
                    0.0, timeout_seconds - (time.perf_counter() - started),
                )
                result = self._deepseek.complete(system, user, max_tokens, remaining)
                result.attempts = [*qwen_error.attempts, *result.attempts]
                return result
            except LLMRequestError as deepseek_error:
                raise LLMRequestError(
                    f"offpeak_qwen_and_deepseek_failed:qwen={qwen_error};deepseek={deepseek_error}",
                    [*qwen_error.attempts, *deepseek_error.attempts],
                ) from deepseek_error

    def complete_many(self, requests: Sequence[tuple[str, str, int]]) -> list:
        """Submit independent prompts as one bounded batch and preserve input order."""
        if len(requests) <= 1:
            return [self.complete(*request) for request in requests]
        with ThreadPoolExecutor(max_workers=min(4, len(requests))) as pool:
            futures = [pool.submit(self.complete, *request) for request in requests]
            return [future.result() for future in futures]


def _json_object(text: str) -> dict:
    stripped = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.I | re.S)
    if match:
        stripped = match.group(1)
    start, end = stripped.find("{"), stripped.rfind("}")
    if start < 0 or end < start:
        raise ValueError("response_contains_no_json_object")
    candidate_json = stripped[start:end + 1]
    try:
        value = json.loads(candidate_json)
    except json.JSONDecodeError as original_error:
        # Small models sometimes emit valid JSON string values for each
        # complete program but omit a comma between candidate objects. Recover
        # only independently decodable strings; never alter generated code.
        decoder = json.JSONDecoder()
        recovered: list[dict[str, str]] = []
        for field in re.finditer(
            r'"(?:code|program|source|python)"\s*:\s*', candidate_json,
        ):
            try:
                source, _ = decoder.raw_decode(candidate_json[field.end():])
            except json.JSONDecodeError:
                continue
            if isinstance(source, str):
                recovered.append({"code": source})
        if not recovered:
            raise original_error
        value = {"candidates": recovered}
    if not isinstance(value, dict):
        raise ValueError("response_json_must_be_object")
    return value


def _candidate_source(value) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("code", "program", "source", "python"):
            source = value.get(key)
            if isinstance(source, str) and source.strip():
                return source
    return None


CADQUERY_RULES = [
    "Use only `import cadquery as cq` plus math/numpy when necessary.",
    "Assign the final exportable Workplane/Shape/Assembly to top-level variable `result`.",
    "Do not read files, use networking, spawn processes, or export files.",
    "Do not call `show_object`, `show`, `display`, or notebook/UI helpers; the isolated runtime only reads the top-level `result` variable.",
    "All dimensions and coordinates must come from the current requirement, never from blueprints.",
    "Use valid CadQuery 2.x calls and preserve base/tool order in Boolean operations.",
    "CadQuery Workplane methods return new Workplanes: chain every sketch/solid call or reassign its return value (for example `wp = wp.moveTo(...).lineTo(...)`); never discard a bare `wp.method(...)` result.",
    "CadQuery Workplane has no `scale()` or `copy()` method: apply requested scale factors to numeric sketch coordinates/dimensions before constructing geometry, and reuse immutable Workplanes by assignment.",
    "Use `math.pi` for pi; CadQuery has no `cq.PI` constant.",
    "When explicit sketch vertices, circle centers, Euler angles, or translations are given, preserve their absolute coordinate frame; do not replace them with centered primitives unless the resulting placement is exactly equivalent.",
    "In CAD-Coder-style requirements, listed sketch coordinates and stated final dimensions are authoritative; descriptive scale factors are already reflected in those values and must not be applied a second time.",
    "Prefer a standard `cq.Workplane('XY'/'XZ'/'YZ')`. If an exact custom plane is required, construct `cq.Plane` with qualified `cq.Vector` values; never use bare `Plane`/`Vector`, `Workplane(plane=...)`, `cq.Location`, or `Workplane.transform`.",
    "CadQuery `translate` takes exactly one vector argument: use `.translate((x, y, z))`, never `.translate(x, y, z)`. Do not use `translate` on an empty Workplane to position a future sketch because it does not move the drawing frame; use `.transformed(offset=(x, y, z), rotate=(rx, ry, rz))`, or extrude first and then translate the solid. Never use the nonexistent `transformOffset`.",
    "When the requirement explicitly introduces a coordinate system or workplane with Euler angles/translation before drawing the sketch, apply that frame to a fresh empty Workplane with `.transformed(offset=(...), rotate=(...))` before `moveTo`/`lineTo`/`circle`. This is different from transforming a Workplane that already contains pending wires. Apply transforms to the finished solid only when the requirement describes a final-body transform after solid creation.",
    "Boolean `union`, `cut`, and `intersect` require solid operands, not pending 2D sketch Workplanes. Either chain all intended contours on one Workplane and extrude once, or extrude each profile from an independent fresh Workplane before applying Booleans to the solids.",
    "To place a circle at (x, y), use `.center(x, y).circle(radius)` or `.moveTo(x, y).circle(radius)`, never `circle(..., center=...)`.",
    "Successive `.center(x, y)` calls are relative and accumulate; for multiple absolute circle centers, use `.pushPoints([(x1, y1), ...]).circle(radius)` or restart from an unchanged Workplane.",
    "For an explicitly listed closed polygon, call `moveTo` on the first start point, connect the listed endpoints in order with `lineTo`, and call `close()` exactly once before extrusion.",
    "Call `rotate(axisStartPoint, axisEndPoint, angleDegrees)` with exactly an axis start, axis end, and scalar angle; an Euler tuple alone is not a valid Workplane.rotate call.",
    "Call `revolve(angleDegrees=..., axisStart=..., axisEnd=...)`; never pass the requirement label `angle_degrees` as a CadQuery keyword.",
    "Call `circle(radius)` after positioning the Workplane; do not pass center coordinates as extra positional arguments.",
    "Use `circle(radius)` rather than `circle(r=...)`. Use `extrude(distance)` or `extrude(until=distance)`, never `extrude(distance=...)`; for subtractive extrusion use `combine='cut'` or build a tool and call `cut`, never `extrude(cut=...)`.",
    "Call `lineTo(x, y)` with two scalar coordinates, never with one tuple.",
    "CadQuery Workplane has no generic `arc()` or `filterBy()` method; use `threePointArc`, `radiusArc`, or `sagittaArc`, and use supported selectors such as `faces`/`edges` with selector strings.",
    "Do not rotate or translate a Workplane after drawing pending sketch wires and before extruding them; first extrude the closed sketch into a solid, then apply the requested rotation and translation to that solid.",
    "Return compact executable source with no comments, step restatements, Markdown fences, or explanatory prose. Use data lists, helper functions, and loops when they preserve the exact requested repeated operations and values.",
]


def _compact_dsl_contract(text: str) -> str:
    """Return a high-salience conversion contract for compact Op-CAD text."""
    if "DSL:" not in text or "Code:" not in text:
        return ""
    return (
        "MANDATORY INPUT-FORMAT CONTRACT: `DSL:` and `Code:` fragments are ordered "
        "pseudocode, never Python to copy. Expand referenced D# macros at their use sites and "
        "translate all aliases to ordinary qualified CadQuery calls. Do not define replacement "
        "aliases or lambdas named W, m, o, e, u, k, a, l, c, v, d, N, f, b, or h. Do not emit "
        "`.W`, `.m`, `.o`, `.e`, `.u`, `.k`, `.a`, `.l`, `.c`, `.v`, `.d`, `.N`, `.f`, "
        "`.b`, bare W/Plane/Vector, or semicolon-separated "
        "arguments inside a Python call. A syntax-only example is: DSL input "
        "`w=W((0,0,0);(0,0,1);(1,0,0));s=w.m(2,3).o(r=4);h=s.e(5);Use h as "
        "the final output r.` becomes `import cadquery as cq`, then "
        "`plane = cq.Plane(origin=cq.Vector(0,0,0), normal=cq.Vector(0,0,1), "
        "xDir=cq.Vector(1,0,0))`, `w = cq.Workplane(plane)`, "
        "`s = w.moveTo(2,3).circle(4)`, `h = s.extrude(5)`, `result = h`. "
        "The example numbers are syntax demonstrations only: never copy them; use only values "
        "from the current requirement. Preserve each rN definition, dependency, operation order, "
        "and Boolean base/tool role. The returned program must start with `import cadquery as cq` "
        "and end with a top-level `result` geometry."
    )


def _requirement_specific_rules(text: str) -> list[str]:
    """Add input-format guidance without changing the requested geometry.

    Op-CAD requirements contain a self-describing compact DSL.  Small models
    otherwise tend to copy that DSL verbatim and produce expressions such as
    ``cq.Workplane(...).W(...).m(...).e(...)``.  The rule below only explains
    how to translate the already-visible requirement; it does not inject
    dimensions, coordinates, reference programs, or evaluation information.
    """
    contract = _compact_dsl_contract(text)
    return [contract] if contract else []


def _program_output_budget(
    text: str, requirement_graph: dict | None = None, code: str = "", *,
    candidate_count: int = 1,
) -> int:
    """Scale output capacity from observable task complexity, never a length bin."""
    graph_nodes = len((requirement_graph or {}).get("nodes") or [])
    operation_markers = (
        text.lower().count("; then")
        + len(re.findall(r"(?m)^Step\s+\d+\s*:", text))
    )
    complexity = max(graph_nodes, operation_markers, 1)
    estimate = max(
        9000,
        4000 + 110 * complexity,
        3000 + len(text) // 2,
        3000 + len(code) // 2,
    )
    if candidate_count > 1:
        estimate = int(estimate * min(1.6, 1.0 + 0.3 * (candidate_count - 1)))
    return min(32000, estimate)


def _repair_hints(execution: dict, constraint_validation: dict) -> list[str]:
    observed = " ".join([
        str(execution.get("error_type") or ""),
        str(execution.get("error") or execution.get("stderr") or ""),
        " ".join(str(value) for value in constraint_validation.get("issues") or []),
    ]).lower()
    hints: list[str] = []
    if "gp_vec::normalized()" in observed and "zero norm" in observed:
        hints.append(
            "CadQuery Plane positional arguments are (origin, xDir, normal), not "
            "(origin, normal). Never call cq.Plane(origin, normal) because a Z-axis "
            "value is then misread as xDir and can be parallel to the default normal. "
            "For axis-aligned sketch placement use "
            "cq.Workplane('XY'/'XZ'/'YZ').transformed(offset=(x, y, z)); do not call "
            "translate on an empty Workplane because it does not move the future sketch frame; "
            "otherwise use named cq.Plane(origin=..., xDir=..., normal=...) with nonparallel "
            "xDir and normal vectors."
        )
    if "unsupported_cadquery_plane" in observed or "invalid_workplane_plane_factory" in observed:
        hints.append(
            "Replace cq.Workplane(cq.Plane...) or cq.Workplane(cq.Plane()) with a standard "
            "cq.Workplane('XY'/'XZ'/'YZ'); extrude first, then rotate/translate the solid."
        )
    if "outer wire is not closed" in observed or "inner wire is not closed" in observed:
        hints.append(
            "Rebuild every polygon wire from its exact first listed start point, connect each "
            "listed endpoint in sequence, and close it once; do not close circles separately."
        )
    if "hole_count_mismatch" in observed:
        hints.append(
            "Construct required holes as inner closed wires on the same outer-profile Workplane "
            "using absolute pushPoints, or subtract explicit cylindrical tools from the base."
        )
    if "invalidtopology" in observed or "invalid_topology" in observed:
        hints.append(
            "Avoid self-intersecting or nested fluent chains: build each closed profile/solid "
            "separately, verify it is non-empty, then apply the required Boolean operation."
        )
    if "no pending wires present" in observed:
        hints.append(
            "A prior extrusion consumed shared pending-wire context. Rebuild every profile that "
            "will be extruded separately from its own fresh cq.Workplane(...) instance; do not "
            "derive several sketch variables from one Workplane and extrude them sequentially. "
            "Alternatively chain all outer/inner contours on one Workplane and extrude once."
        )
    if "must have at least one solid on the stack" in observed:
        hints.append(
            "Do not call union/cut/intersect on unextruded sketch Workplanes. Extrude each profile "
            "from an independent fresh Workplane first, then Boolean the resulting solids."
        )
    if "revolve" in observed and "angle_degrees" in observed:
        hints.append(
            "Replace revolve(angle_degrees=...) with CadQuery's exact keyword "
            "revolve(angleDegrees=...); preserve the requested numeric angle."
        )
    if "workplane" in observed and "keyword argument 'plane'" in observed:
        hints.append(
            "Replace Workplane(plane=...) with Workplane(inPlane=...) or a positional "
            "plane argument; qualify custom constructors as cq.Plane and cq.Vector."
        )
    if "extrude" in observed and "keyword argument 'distance'" in observed:
        hints.append(
            "Replace extrude(distance=value) with extrude(value) or extrude(until=value), "
            "preserving the exact signed distance."
        )
    if "extrude" in observed and "keyword argument 'cut'" in observed:
        hints.append(
            "Replace extrude(cut=...) with combine='cut' when appropriate, or extrude a "
            "separate tool and subtract it using base.cut(tool)."
        )
    if "circle" in observed and "keyword argument 'r'" in observed:
        hints.append("Replace circle(r=value) with circle(value), preserving the radius.")
    if "name 'plane' is not defined" in observed or "name 'vector' is not defined" in observed:
        hints.append("Qualify CadQuery constructors as cq.Plane and cq.Vector.")
    return hints


def _repair_output_rules(count: int) -> list[str]:
    if count <= 1:
        return [
            'Return JSON only: {"candidates":[{"code":"..."}]} with exactly one complete program.',
        ]
    return [
        f'Return JSON only: {{"candidates":[{{"code":"..."}},...]}} with exactly {count} complete, distinct programs.',
        "Candidate 1 must be the smallest justified repair of the observed failure.",
        "Candidate 2 must use a genuinely different repair strategy while preserving every stated requirement.",
        "Do not return explanations, patches, omitted sections, or duplicate programs.",
    ]


@dataclass(frozen=True)
class GeneratedProgram:
    code: str | None
    calls: list[dict]
    static_issues: list[str]
    error: dict | None = None


class DualCodeGenerator:
    def __init__(self, client, *, max_regenerations: int = 2):
        self.client = client
        self.max_regenerations = max_regenerations

    @staticmethod
    def _request(
        text: str, blueprints: list[dict], requirement_graph: dict | None = None,
        previous: str = "", issues: list[str] | None = None,
    ):
        structure_source = (
            "retrieved_case_blueprints" if blueprints
            else "requirement_dag" if requirement_graph else "none"
        )
        input_format_contract = _compact_dsl_contract(text)
        payload = {
            "prompt_version": PROMPT_VERSIONS["code_generation"],
            "input_format_contract": input_format_contract or None,
            "requirement": text,
            "structural_blueprints": [
                compact_blueprint_for_prompt(blueprint) for blueprint in blueprints
            ],
            "requirement_dag": requirement_graph,
            "structure_source": structure_source,
            "previous_invalid_code": previous[-12000:], "static_issues": issues or [],
            "rules": [*CADQUERY_RULES, *_requirement_specific_rules(text)],
        }
        system = (
            "Generate one complete executable CadQuery Python program. Any supplied requirement DAG or "
            "retrieved blueprints contain structure only; derive every number from the requirement. "
            "Return compact Python source only, without comments, Markdown fences, or prose."
        )
        if input_format_contract:
            system += " " + input_format_contract
        return (
            system,
            json.dumps(payload, ensure_ascii=False),
            _program_output_budget(text, requirement_graph, previous),
        )

    @staticmethod
    def _failure(error: Exception, stage: str) -> dict:
        return {
            "success": False, "stage": stage, "error_type": type(error).__name__,
            "error": str(error), "attempts": list(getattr(error, "attempts", [])),
        }

    def _generate(
        self, text: str, blueprints: list[dict], requirement_graph: dict | None = None,
    ) -> GeneratedProgram:
        calls: list[dict] = []
        previous = ""
        issues: list[str] = []
        for attempt in range(self.max_regenerations + 1):
            try:
                result = self.client.complete(*self._request(
                    text, blueprints, requirement_graph, previous, issues,
                ))
            except Exception as error:
                calls.append(self._failure(error, f"generation_request_{attempt + 1}"))
                continue
            calls.append(result.to_dict())
            code = normalize_cadquery_code(text, strip_code_fence(result.content))
            analysis = analyse_static(code)
            issues = [*analysis.fatal, *analysis.countable]
            if not analysis.fatal:
                return GeneratedProgram(code, calls, issues)
            previous = code
        error = {
            "success": False, "stage": "generation", "error_type": "CodeGenerationError",
            "error": "static_codegen_failed_after_regeneration:" + "|".join(issues),
        }
        # Preserve a concrete but statically rejected program as an A* repair
        # root.  It remains blocked from OCC execution by the whitelist; only
        # provider failures with no source at all stay unavailable.
        return GeneratedProgram(previous or None, calls, issues, error)

    def generate_branch(
        self, text: str, blueprints: list[dict], *, requirement_graph: dict | None = None,
    ) -> GeneratedProgram:
        """Generate one branch without the paired P0 call used by the main experiment."""
        return self._generate(text, blueprints, requirement_graph)

    def generate_from_requirement_graph(
        self, text: str, requirement_graph: dict,
    ) -> GeneratedProgram:
        """A1 ablation: use the requirement DAG directly, with no retrieved cases."""
        return self._generate(text, [], requirement_graph)

    def generate_pair(
        self, text: str, blueprints: list[dict], *,
        requirement_graph: dict | None = None, rag_available: bool = True,
    ) -> dict[str, GeneratedProgram]:
        if not rag_available or not blueprints:
            p0 = self._generate(text, [])
            failure = {
                "success": False, "stage": "retrieval",
                "error_type": "RetrievalUnavailable",
                "error": "PR_not_generated_without_successful_retrieval",
            }
            return {
                "P0": p0,
                "PR": GeneratedProgram(None, [], [], failure),
            }
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {
                "P0": pool.submit(self._generate, text, []),
                "PR": pool.submit(self._generate, text, blueprints, requirement_graph),
            }
            output: dict[str, GeneratedProgram] = {}
            for branch, future in futures.items():
                try:
                    output[branch] = future.result()
                except Exception as error:
                    failure = self._failure(error, f"{branch.lower()}_generation_worker")
                    output[branch] = GeneratedProgram(None, [failure], [], failure)
            return output

    def repair_candidates(self, text: str, code: str, execution: dict,
                          requirement_graph: dict, *, count: int = 2,
                          timeout_seconds: float | None = None,
                          use_dependency_closure: bool = True) -> tuple[list[str], dict]:
        try:
            program_graph = build_program_graph(code)
        except SyntaxError:
            program_graph = {"nodes": [], "output": None}
        gap = asymmetric_graph_difference(requirement_graph, program_graph)
        alignment_gate = cross_graph_alignment_gate(gap)
        constraint_validation = execution.get("constraint_validation") or {}
        graph_confidence = float(constraint_validation.get("graph_confidence", 1.0))
        mapping_confidence = float(gap.get("mapping_confidence", 0.0))
        tau = float(__import__("os").environ.get("TRUSTCAD_CONFIDENCE_TAU", "0.85"))
        confidence_gate = graph_confidence >= tau and mapping_confidence >= tau
        exec_first_gate = __import__("os").environ.get("TRUSTCAD_EXEC_FIRST_GATE", "1") == "1"
        exec_ok = bool(
            execution.get("success")
            or execution.get("execution_success")
            or execution.get("ok")
            or execution.get("stage") == "constraint"
        )
        if not alignment_gate["actionable"] or not confidence_gate or (exec_first_gate and not exec_ok):
            candidates, record = self.repair_candidates_greedy(
                text, code, execution, count=count, timeout_seconds=timeout_seconds,
            )
            record["cross_graph_alignment_requested"] = True
            record["cross_graph_alignment_enabled"] = False
            record["cross_graph_alignment_gate"] = alignment_gate
            record["cross_graph_confidence_gate"] = {
                "enabled": confidence_gate,
                "requirement_graph_confidence": graph_confidence,
                "mapping_confidence": mapping_confidence,
                "minimum_requirement_graph_confidence": tau,
                "minimum_mapping_confidence": tau,
            }
            record["dependency_closure_requested"] = use_dependency_closure
            record["dependency_closure_enabled"] = False
            return candidates, record
        closure_analysis = (
            dependency_closure_analysis(
                requirement_graph, program_graph,
                constraint_issues=list(constraint_validation.get("issues") or []),
                execution_failure=execution,
            )
            if use_dependency_closure else None
        )
        closure_enabled = bool(
            use_dependency_closure
            and closure_analysis
            and closure_analysis.get("localization_reliable")
        )
        closure = closure_analysis if closure_enabled else None
        payload = {
            "prompt_version": PROMPT_VERSIONS["repair"] + (
                "_closure" if closure_enabled else "_no_closure"
            ),
            "requirement": text, "failed_program": code,
            "observable_failure": {
                "stage": execution.get("stage"), "error_type": execution.get("error_type"),
                "error": str(execution.get("error") or execution.get("stderr") or "")[-4000:],
                "constraint_issues": list(constraint_validation.get("issues") or []),
                "constraint_checks": list(constraint_validation.get("constraint_checks") or []),
            },
            "structure_gap": gap,
            "dependency_closure": closure,
            "repair_hints": _repair_hints(execution, constraint_validation),
            "rules": [
                *CADQUERY_RULES,
                *_requirement_specific_rules(text),
                *_repair_output_rules(count),
                "Repair the concrete error and missing operations/dependencies with the smallest justified change.",
                *(
                    [
                        "Use dependency_closure as the primary repair scope. Preserve protected program lines and variables unless a change outside the closure is strictly required for executable correctness.",
                        "Restore every target requirement node and its path to the final result; do not delete unrelated working branches.",
                        "Candidate 1 must be closure-local and minimal. Candidate 2 must independently reason over the full program and full structure gap; the closure is advisory for Candidate 2.",
                    ] if closure_enabled else [
                        "No dependency-closure localization is provided in this ablation; reason from the full program and full structure gap.",
                    ]
                ),
                "Current implementation returns complete programs, not local patches.",
            ],
        }
        repair_system = (
            "Repair a failed CadQuery program using real OCC error and requirement-program structure differences."
        )
        if contract := _compact_dsl_contract(text):
            repair_system += " " + contract
        request = (
            repair_system,
            json.dumps(payload, ensure_ascii=False),
            _program_output_budget(
                text, requirement_graph, code, candidate_count=count,
            ),
        )
        result = (
            self.client.complete(*request, timeout_seconds=timeout_seconds)
            if timeout_seconds is not None else self.client.complete(*request)
        )
        try:
            parsed = _json_object(result.content)
        except (ValueError, json.JSONDecodeError) as error:
            raise LLMRequestError(
                f"repair_response_parse_failed:{error}", [result.to_dict()],
            ) from error
        candidates: list[str] = []
        for value in (parsed.get("candidates") or [])[:count]:
            source = _candidate_source(value)
            if source is None:
                continue
            candidate = normalize_cadquery_code(text, strip_code_fence(source))
            if candidate != code and candidate not in candidates:
                candidates.append(candidate)
        record = result.to_dict()
        record["cross_graph_alignment_requested"] = True
        record["cross_graph_alignment_enabled"] = True
        record["cross_graph_alignment_gate"] = alignment_gate
        record["cross_graph_confidence_gate"] = {
            "enabled": True,
            "requirement_graph_confidence": graph_confidence,
            "mapping_confidence": mapping_confidence,
            "minimum_requirement_graph_confidence": tau,
            "minimum_mapping_confidence": tau,
        }
        record["dependency_closure_requested"] = use_dependency_closure
        record["dependency_closure_enabled"] = closure_enabled
        if closure_analysis is not None:
            record["dependency_closure_analysis"] = closure_analysis
        if closure is not None:
            record["dependency_closure"] = closure
        return candidates, record

    def repair_candidates_greedy(
        self, text: str, code: str, execution: dict, *, count: int = 2,
        timeout_seconds: float | None = None,
    ) -> tuple[list[str], dict]:
        """Generate greedy baseline repairs without a requirement DAG or graph gap."""
        constraint_validation = execution.get("constraint_validation") or {}
        payload = {
            "prompt_version": PROMPT_VERSIONS["repair"] + "_greedy_no_dag",
            "requirement": text,
            "failed_program": code,
            "observable_failure": {
                "stage": execution.get("stage"),
                "error_type": execution.get("error_type"),
                "error": str(execution.get("error") or execution.get("stderr") or "")[-4000:],
                "constraint_issues": list(constraint_validation.get("issues") or []),
                "constraint_checks": list(constraint_validation.get("constraint_checks") or []),
            },
            "repair_hints": _repair_hints(execution, constraint_validation),
            "rules": [
                *CADQUERY_RULES,
                *_requirement_specific_rules(text),
                *_repair_output_rules(count),
                "Repair the latest concrete execution or deterministic text-constraint error.",
                "Do not infer, construct, or use a requirement operation graph.",
            ],
        }
        repair_system = (
            "Greedily repair a failed CadQuery program from the requirement and latest observed error only."
        )
        if contract := _compact_dsl_contract(text):
            repair_system += " " + contract
        request = (
            repair_system,
            json.dumps(payload, ensure_ascii=False),
            _program_output_budget(text, None, code, candidate_count=count),
        )
        result = (
            self.client.complete(*request, timeout_seconds=timeout_seconds)
            if timeout_seconds is not None else self.client.complete(*request)
        )
        try:
            parsed = _json_object(result.content)
        except (ValueError, json.JSONDecodeError) as error:
            raise LLMRequestError(
                f"greedy_repair_response_parse_failed:{error}", [result.to_dict()],
            ) from error
        candidates: list[str] = []
        for value in (parsed.get("candidates") or [])[:count]:
            source = _candidate_source(value)
            if source is None:
                continue
            candidate = normalize_cadquery_code(text, strip_code_fence(source))
            if candidate != code and candidate not in candidates:
                candidates.append(candidate)
        return candidates, result.to_dict()

    def replan(self, text: str, code: str, execution: dict, requirement_graph: dict) -> tuple[str, dict]:
        gap = graph_difference(requirement_graph, build_program_graph(code))
        payload = {
            "prompt_version": PROMPT_VERSIONS["replan"],
            "requirement": text, "failed_program": code,
            "observable_failure": {
                "stage": execution.get("stage"), "error_type": execution.get("error_type"),
                "error": str(execution.get("error") or execution.get("stderr") or "")[-4000:],
            },
            "structure_gap": gap,
            "rules": [*CADQUERY_RULES, *_requirement_specific_rules(text)],
        }
        replan_system = "Replan once and return one complete corrected CadQuery Python program only."
        if contract := _compact_dsl_contract(text):
            replan_system += " " + contract
        result = self.client.complete(
            replan_system,
            json.dumps(payload, ensure_ascii=False),
            max_tokens=_program_output_budget(text, requirement_graph, code),
        )
        candidate = normalize_cadquery_code(text, strip_code_fence(result.content))
        fatal = analyse_static(candidate).fatal
        if fatal:
            raise RuntimeError("fallback_replan_static_invalid:" + "|".join(fatal))
        return candidate, result.to_dict()
