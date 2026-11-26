"""Deterministic CadQuery AST-to-operation graph with branch-aware def-use."""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass

from ..ast_graph import validate_cadquery_code
from .operations import canonical_operation


def _call_label(call: ast.Call) -> str:
    target = call.func
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return "unknown_op"


def _target_names(target: ast.AST) -> list[str]:
    return [
        node.id for node in ast.walk(target)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    ]


def _loaded_names(node: ast.AST) -> list[str]:
    return sorted({
        item.id for item in ast.walk(node)
        if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)
    })


def _receiver_root_name(call: ast.Call) -> str | None:
    """Return the variable at the root of a fluent method-call expression."""
    if not isinstance(call.func, ast.Attribute):
        return None
    value: ast.AST = call.func.value
    while True:
        if isinstance(value, ast.Name):
            return value.id
        if isinstance(value, ast.Attribute):
            value = value.value
            continue
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
            value = value.func.value
            continue
        return None


def _literal(node: ast.AST):
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return None


def _contains_nonzero_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return abs(float(value)) > 1e-12
    if isinstance(value, (list, tuple)):
        return any(_contains_nonzero_number(item) for item in value)
    return False


def _fluent_point_multiplicity(call: ast.Call) -> int:
    """Return a statically proven pushPoints fan-out for a fluent operation."""
    if not isinstance(call.func, ast.Attribute):
        return 1
    value: ast.AST = call.func.value
    while isinstance(value, ast.Call):
        if _call_label(value).lower() == "pushpoints" and value.args:
            points = _literal(value.args[0])
            if isinstance(points, (list, tuple)) and points:
                return min(len(points), 128)
            return 1
        if isinstance(value.func, ast.Attribute):
            value = value.func.value
        else:
            break
    return 1


@dataclass(frozen=True)
class StaticAnalysis:
    fatal: list[str]
    countable: list[str]


def analyse_static(code: str) -> StaticAnalysis:
    fatal = validate_cadquery_code(code)
    if fatal and any(item.startswith("syntax:") for item in fatal):
        return StaticAnalysis(fatal, [])
    graph = build_program_graph(code)
    effective = output_dependency_subgraph(graph)
    unknown = sum(node.get("op") == "unknown_op" for node in effective["nodes"])
    countable = [f"unknown_operations:{unknown}"] if unknown else []
    if graph.get("output") is None:
        countable.append("unresolved_result_output")
    if graph.get("dynamic_control_flow"):
        countable.append("dynamic_control_flow")
    return StaticAnalysis(sorted(set(fatal)), countable)


class _GraphBuilder:
    def __init__(self):
        self.nodes: list[dict] = []
        self.env: dict[str, tuple[str, ...]] = {}
        self.contours: dict[str, tuple[str, ...]] = {}
        # Literal values are tracked separately from CAD def-use outputs.  They
        # are used only for conservative control-flow expansion such as
        # ``range(1, len(points))`` when ``points`` is a literal list.
        self.literal_env: dict[str, object] = {}
        self.functions: dict[str, ast.FunctionDef] = {}
        self.call_stack: list[str] = []
        self.dynamic_control_flow = False

    @staticmethod
    def _unique(values: list[str]) -> list[str]:
        return list(dict.fromkeys(values))

    def _name_dependencies(self, node: ast.AST) -> list[str]:
        values: list[str] = []
        for name in _loaded_names(node):
            values.extend(self.env.get(name, ()))
        return self._unique(values)

    def _resolved_literal(self, node: ast.AST) -> object | None:
        """Resolve a literal or a literal-valued name without executing code."""
        value = _literal(node)
        if value is not None or isinstance(node, ast.Constant):
            return value
        if isinstance(node, ast.Name):
            return self.literal_env.get(node.id)
        if isinstance(node, (ast.Tuple, ast.List)):
            items = [self._resolved_literal(item) for item in node.elts]
            if all(item is not None for item in items):
                return tuple(items) if isinstance(node, ast.Tuple) else items
        return None

    def _resolve_expression(self, value: ast.AST) -> list[str]:
        if isinstance(value, ast.Call):
            if isinstance(value.func, ast.Name) and value.func.id in self.functions:
                return self._invoke_function(value.func.id, value)
            if _call_label(value).lower() == "reduce":
                reduced = self._resolve_reduce(value)
                if reduced is not None:
                    return reduced
            return [self._add_call(value)]
        if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add):
            left = self._resolve_expression(value.left)
            right = self._resolve_expression(value.right)
            if left and right:
                base_inputs = self._unique(left)
                tool_inputs = self._unique(right)
                node_id = f"p{len(self.nodes)}"
                self.nodes.append({
                    "id": node_id,
                    "op": "union",
                    "inputs": [*base_inputs, *tool_inputs],
                    "roles": [
                        *(["base"] * len(base_inputs)),
                        *(["tool"] * len(tool_inputs)),
                    ],
                    "defines": [],
                    "uses": [name for name in _loaded_names(value) if name in self.env],
                    "lineno": getattr(value, "lineno", None),
                    "end_lineno": getattr(value, "end_lineno", None),
                    "source_call": "__add__",
                    "multiplicity": 1,
                    "literal_args": [],
                    "literal_kwargs": {},
                })
                return [node_id]
            return self._unique([*left, *right])
        if isinstance(value, ast.Name):
            return list(self.env.get(value.id, ()))
        if isinstance(value, ast.IfExp):
            return self._unique([
                *self._resolve_expression(value.body), *self._resolve_expression(value.orelse),
            ])
        outputs: list[str] = []
        for child in ast.iter_child_nodes(value):
            if isinstance(child, ast.expr):
                outputs.extend(self._resolve_expression(child))
        if outputs:
            return self._unique(outputs)
        return self._name_dependencies(value)

    def _resolve_reduce(self, call: ast.Call) -> list[str] | None:
        """Model a static ``reduce`` Boolean as an explicit role-aware chain."""
        if len(call.args) < 2 or not isinstance(call.args[0], ast.Lambda):
            return None
        body = call.args[0].body
        operation: str | None = None
        if (
            isinstance(body, ast.Call)
            and isinstance(body.func, ast.Attribute)
        ):
            candidate = canonical_operation(body.func.attr)
            if candidate in {"cut", "union", "intersect"}:
                operation = candidate
        elif isinstance(body, ast.BinOp) and isinstance(body.op, ast.Add):
            operation = "union"
        if operation is None:
            return None

        sequence = call.args[1]
        groups: list[list[str]] = []
        if isinstance(sequence, (ast.List, ast.Tuple)):
            groups = [self._resolve_expression(element) for element in sequence.elts]
        elif isinstance(sequence, ast.Name):
            groups = [[value] for value in self.env.get(sequence.id, ())]
        groups = [self._unique(group) for group in groups if group]
        if not groups:
            return []
        current = groups[0]
        for tools in groups[1:]:
            node_id = f"p{len(self.nodes)}"
            self.nodes.append({
                "id": node_id,
                "op": operation,
                "inputs": [*current, *tools],
                "roles": [
                    *(["base"] * len(current)),
                    *(["tool"] * len(tools)),
                ],
                "defines": [],
                "uses": [name for name in _loaded_names(call) if name in self.env],
                "lineno": getattr(call, "lineno", None),
                "end_lineno": getattr(call, "end_lineno", None),
                "source_call": "reduce",
                "multiplicity": 1,
                "literal_args": [],
                "literal_kwargs": {},
            })
            current = [node_id]
        return current

    def _invoke_function(self, name: str, call: ast.Call) -> list[str]:
        if name in self.call_stack:
            self.dynamic_control_flow = True
            return []
        function = self.functions[name]
        argument_outputs = [self._resolve_expression(argument) for argument in call.args]
        argument_literals = [self._resolved_literal(argument) for argument in call.args]
        keyword_outputs = {
            keyword.arg: self._resolve_expression(keyword.value)
            for keyword in call.keywords if keyword.arg is not None
        }
        keyword_literals = {
            keyword.arg: self._resolved_literal(keyword.value)
            for keyword in call.keywords if keyword.arg is not None
        }
        caller_env = self.env
        caller_literal_env = self.literal_env
        local_env = dict(caller_env)
        local_literal_env = dict(caller_literal_env)
        parameters = [*function.args.posonlyargs, *function.args.args]
        for index, parameter in enumerate(parameters):
            values = argument_outputs[index] if index < len(argument_outputs) else keyword_outputs.get(parameter.arg, [])
            local_env[parameter.arg] = tuple(values)
            literal = (
                argument_literals[index]
                if index < len(argument_literals)
                else keyword_literals.get(parameter.arg)
            )
            if literal is None:
                local_literal_env.pop(parameter.arg, None)
            else:
                local_literal_env[parameter.arg] = literal
        self.env = local_env
        self.literal_env = local_literal_env
        self.call_stack.append(name)
        outputs: list[str] = []
        try:
            for statement in function.body:
                if isinstance(statement, ast.Return):
                    if statement.value is not None:
                        outputs.extend(self._resolve_expression(statement.value))
                    break
                self.process(statement)
        finally:
            self.call_stack.pop()
            self.env = caller_env
            self.literal_env = caller_literal_env
        return self._unique(outputs)

    def _add_call(self, call: ast.Call) -> str:
        receiver_dependencies: list[str] = []
        if isinstance(call.func, ast.Attribute):
            receiver_dependencies = self._resolve_expression(call.func.value)
        argument_dependencies: list[str] = []
        for argument in [*call.args, *[keyword.value for keyword in call.keywords]]:
            argument_dependencies.extend(self._resolve_expression(argument))
        receiver_dependencies = self._unique(receiver_dependencies)
        argument_dependencies = self._unique(argument_dependencies)
        label = _call_label(call)
        op = canonical_operation(label)
        receiver_root = _receiver_root_name(call)
        literal_kwargs = {
            keyword.arg: self._resolved_literal(keyword.value)
            for keyword in call.keywords if keyword.arg is not None
        }
        if label.lower() == "transformed":
            # CadQuery's transformed() is one API call but may encode two
            # semantic operations.  Expand the observable non-identity parts
            # so requirement-DAG translate/rotate nodes align with the AST
            # program graph instead of being collapsed into another Workplane.
            transformed_ops: list[str] = []
            if _contains_nonzero_number(literal_kwargs.get("rotate")):
                transformed_ops.append("rotate")
            if _contains_nonzero_number(literal_kwargs.get("offset")):
                transformed_ops.append("translate")
            if not transformed_ops:
                transformed_ops.append("workplane")
            dependencies = self._unique([
                *receiver_dependencies, *argument_dependencies,
            ])
            last_node_id = ""
            for transformed_op in transformed_ops:
                node_id = f"p{len(self.nodes)}"
                node = {
                    "id": node_id,
                    "op": transformed_op,
                    "inputs": dependencies,
                    "roles": ["input"] * len(dependencies),
                    "defines": [],
                    "uses": [name for name in _loaded_names(call) if name in self.env],
                    "lineno": getattr(call, "lineno", None),
                    "end_lineno": getattr(call, "end_lineno", None),
                    "source_call": _call_label(call),
                    "multiplicity": 1,
                    "literal_args": [self._resolved_literal(argument) for argument in call.args],
                    "literal_kwargs": literal_kwargs,
                }
                self.nodes.append(node)
                last_node_id = node_id
                dependencies = [node_id]
            return last_node_id
        if op in {"workplane", "plane"} and _contains_nonzero_number(literal_kwargs.get("origin")):
            # CadQuery Workplane("XY", origin=(x, y, z)) / Plane(origin=...)
            # encodes a non-identity translation of the sketch frame.
            dependencies = self._unique([
                *receiver_dependencies, *argument_dependencies,
            ])
            wp_node_id = f"p{len(self.nodes)}"
            self.nodes.append({
                "id": wp_node_id,
                "op": op,
                "inputs": dependencies,
                "roles": ["input"] * len(dependencies),
                "defines": [],
                "uses": [name for name in _loaded_names(call) if name in self.env],
                "lineno": getattr(call, "lineno", None),
                "end_lineno": getattr(call, "end_lineno", None),
                "source_call": _call_label(call),
                "multiplicity": 1,
                "literal_args": [self._resolved_literal(argument) for argument in call.args],
                "literal_kwargs": literal_kwargs,
            })
            tr_node_id = f"p{len(self.nodes)}"
            self.nodes.append({
                "id": tr_node_id,
                "op": "translate",
                "inputs": [wp_node_id],
                "roles": ["input"],
                "defines": [],
                "uses": [],
                "lineno": getattr(call, "lineno", None),
                "end_lineno": getattr(call, "end_lineno", None),
                "source_call": f"{_call_label(call)}(origin)",
                "multiplicity": 1,
                "literal_args": [literal_kwargs.get("origin")],
                "literal_kwargs": {"offset": literal_kwargs.get("origin")},
            })
            return tr_node_id
        if op in {"extrude", "revolve", "loft", "sweep"} and receiver_root is not None:
            receiver_dependencies = self._unique([
                *receiver_dependencies, *self.contours.get(receiver_root, ()),
            ])
        if (
            op == "unknown_op" and label.upper() in {"XY", "XZ", "YZ", "FRONT", "BACK", "LEFT", "RIGHT", "TOP", "BOTTOM"}
            and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Attribute)
            and call.func.value.attr == "Plane"
        ):
            op = "workplane"
        inputs: list[str] = []
        roles: list[str] = []
        if op in {"cut", "union", "intersect"}:
            # Workplane.combine() fuses multiple pending solids stored inside
            # one receiver stack and therefore has no explicit Python tool
            # argument.  Encode the receiver's internal base/tool ports as a
            # repeated dependency so it aligns with the requirement DAG's
            # explicit pending-solids union rather than looking unary/invalid.
            pending_stack_union = (
                op == "union" and label.lower() == "combine"
                and receiver_dependencies and not argument_dependencies
            )
            if pending_stack_union:
                inputs.extend([receiver_dependencies[-1], receiver_dependencies[-1]])
                roles.extend(["base", "tool"])
            else:
                for dependency in receiver_dependencies:
                    inputs.append(dependency)
                    roles.append("base")
                for dependency in argument_dependencies:
                    inputs.append(dependency)
                    roles.append("tool")
        else:
            inputs = self._unique([*receiver_dependencies, *argument_dependencies])
            roles = ["input"] * len(inputs)
        node_id = f"p{len(self.nodes)}"
        node = {
            "id": node_id,
            "op": op,
            "inputs": inputs,
            "roles": roles,
            "defines": [],
            "uses": [name for name in _loaded_names(call) if name in self.env],
            "lineno": getattr(call, "lineno", None),
            "end_lineno": getattr(call, "end_lineno", None),
            "source_call": _call_label(call),
            "multiplicity": (
                _fluent_point_multiplicity(call)
                if op in {"circle", "ellipse", "profile", "slot"} else 1
            ),
            "literal_args": [self._resolved_literal(argument) for argument in call.args],
            "literal_kwargs": literal_kwargs,
        }
        self.nodes.append(node)
        return node_id

    def _assign(self, targets: list[ast.AST], value: ast.AST | None) -> None:
        if value is None:
            return
        literal_value = self._resolved_literal(value)
        first_new_node = len(self.nodes)
        outputs = self._resolve_expression(value)
        new_nodes = self.nodes[first_new_node:]
        names = [name for target in targets for name in _target_names(target)]
        for name in names:
            self.env[name] = tuple(outputs)
            if literal_value is None:
                self.literal_env.pop(name, None)
            else:
                self.literal_env[name] = literal_value
        receiver_root = _receiver_root_name(value) if isinstance(value, ast.Call) else None
        inherited_contours = list(self.contours.get(receiver_root, ())) if receiver_root else []
        if isinstance(value, ast.Name):
            inherited_contours = list(self.contours.get(value.id, ()))
        if any(node.get("op") == "workplane" for node in new_nodes) and receiver_root is None:
            inherited_contours = []
        new_contours = [
            str(node["id"]) for node in new_nodes
            if node.get("op") in {"close", "circle", "ellipse", "profile", "slot"}
        ]
        for name in names:
            combined = self._unique([*inherited_contours, *new_contours])
            if combined:
                self.contours[name] = tuple(combined)
            else:
                self.contours.pop(name, None)
        for output in outputs:
            node = next((item for item in self.nodes if item["id"] == output), None)
            if node is not None:
                node["defines"] = self._unique([*node["defines"], *names])

    def _merge_branch_envs(self, base: dict[str, tuple[str, ...]],
                           left: dict[str, tuple[str, ...]], right: dict[str, tuple[str, ...]]) -> None:
        merged: dict[str, tuple[str, ...]] = {}
        for name in set(base) | set(left) | set(right):
            values = self._unique([
                *left.get(name, base.get(name, ())),
                *right.get(name, base.get(name, ())),
            ])
            if values:
                merged[name] = tuple(values)
        self.env = merged

    def _static_integer(self, node: ast.AST) -> int | None:
        value = _literal(node)
        if isinstance(value, int) and not isinstance(value, bool):
            return int(value)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "len"
            and len(node.args) == 1
            and not node.keywords
            and isinstance(node.args[0], ast.Name)
        ):
            literal = self.literal_env.get(node.args[0].id)
            if isinstance(literal, (list, tuple, dict, set, str, bytes)):
                return len(literal)
        return None

    def _static_range_count(self, node: ast.For) -> int | None:
        iterator = node.iter
        if not (
            isinstance(iterator, ast.Call) and isinstance(iterator.func, ast.Name)
            and iterator.func.id == "range" and 1 <= len(iterator.args) <= 3
            and not iterator.keywords
        ):
            return None
        values = [self._static_integer(arg) for arg in iterator.args]
        if any(value is None for value in values):
            return None
        try:
            return len(range(*(int(value) for value in values)))
        except ValueError:
            return None

    def process(self, statement: ast.stmt) -> None:
        if isinstance(statement, ast.FunctionDef):
            self.functions[statement.name] = statement
            return
        if isinstance(statement, ast.Assign):
            self._assign(list(statement.targets), statement.value)
            return
        if isinstance(statement, ast.AnnAssign):
            self._assign([statement.target], statement.value)
            return
        if isinstance(statement, ast.AugAssign):
            self._assign([statement.target], statement.value)
            return
        if isinstance(statement, ast.Expr):
            first_new_node = len(self.nodes)
            outputs = self._resolve_expression(statement.value)
            # CadQuery frequently uses an unassigned fluent statement, e.g.
            # ``wp.moveTo(...).lineTo(...).close()``.  Those operations mutate
            # the Workplane state observed by a later ``wp.extrude(...)``.  If
            # the environment is not advanced here, every such sketch node is
            # incorrectly pruned from the final-result dependency subgraph.
            if isinstance(statement.value, ast.Call) and outputs:
                root = _receiver_root_name(statement.value)
                if root is not None and root in self.env:
                    self.env[root] = tuple(outputs)
                    new_contours = [
                        str(node["id"]) for node in self.nodes[first_new_node:]
                        if node.get("op") in {"close", "circle", "ellipse", "profile", "slot"}
                    ]
                    if new_contours:
                        self.contours[root] = tuple(self._unique([
                            *self.contours.get(root, ()), *new_contours,
                        ]))
                    for output in outputs:
                        node = next((item for item in self.nodes if item["id"] == output), None)
                        if node is not None:
                            node["defines"] = self._unique([*node["defines"], root])
            return
        if isinstance(statement, ast.If):
            base = dict(self.env)
            self.env = dict(base)
            for child in statement.body:
                self.process(child)
            left = dict(self.env)
            self.env = dict(base)
            for child in statement.orelse:
                self.process(child)
            right = dict(self.env)
            self._merge_branch_envs(base, left, right)
            return
        if isinstance(statement, ast.For):
            repetitions = self._static_range_count(statement)
            if repetitions is None or repetitions > 32:
                self.dynamic_control_flow = True
                repetitions = 1
            for _ in range(repetitions):
                for child in statement.body:
                    self.process(child)
            for child in statement.orelse:
                self.process(child)
            return
        if isinstance(statement, (ast.While, ast.Try, ast.With, ast.Match)):
            self.dynamic_control_flow = True
        for child in ast.iter_child_nodes(statement):
            if isinstance(child, ast.stmt):
                self.process(child)


def build_program_graph(code: str) -> dict:
    tree = ast.parse(code)
    builder = _GraphBuilder()
    for statement in tree.body:
        if not isinstance(statement, (ast.Import, ast.ImportFrom)):
            builder.process(statement)
    outputs = list(builder.env.get("result", ()))
    graph = {
        "nodes": builder.nodes,
        "output": outputs[0] if len(outputs) == 1 else outputs or None,
        "output_nodes": outputs,
        "graph_kind": "cadquery_program_operation_dag",
        "dynamic_control_flow": builder.dynamic_control_flow,
        "source_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
    }
    effective = output_dependency_subgraph(graph)
    graph["active_node_ids"] = [node["id"] for node in effective["nodes"]]
    active = set(graph["active_node_ids"])
    graph["inactive_node_ids"] = [node["id"] for node in builder.nodes if node["id"] not in active]
    return graph


def output_dependency_subgraph(graph: dict) -> dict:
    """Return only operations that can contribute to the top-level ``result``."""
    nodes = list(graph.get("nodes") or [])
    by_id = {str(node.get("id")): node for node in nodes}
    outputs = graph.get("output_nodes")
    if outputs is None:
        raw_output = graph.get("output")
        outputs = raw_output if isinstance(raw_output, list) else [raw_output] if raw_output else []
    frontier = [str(value) for value in outputs if str(value) in by_id]
    reachable: set[str] = set()
    while frontier:
        node_id = frontier.pop()
        if node_id in reachable:
            continue
        reachable.add(node_id)
        frontier.extend(
            str(parent) for parent in by_id[node_id].get("inputs") or []
            if str(parent) in by_id and str(parent) not in reachable
        )
    ordered = [node for node in nodes if str(node.get("id")) in reachable]
    return {
        "nodes": ordered,
        "output": graph.get("output"),
        "output_nodes": [value for value in outputs if str(value) in reachable],
        "graph_kind": graph.get("graph_kind"),
        "dynamic_control_flow": bool(graph.get("dynamic_control_flow")),
        "source_sha256": graph.get("source_sha256"),
        "inactive_node_ids": [str(node.get("id")) for node in nodes if str(node.get("id")) not in reachable],
    }
