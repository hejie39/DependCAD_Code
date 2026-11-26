"""Safe CadQuery-Python validation and post-hoc AST dependency graphs."""

from __future__ import annotations

import ast
import hashlib
import re


ALLOWED_IMPORTS = {"cadquery", "functools", "math", "numpy"}
FORBIDDEN_NAMES = {
    "breakpoint", "compile", "eval", "exec", "globals", "help", "input",
    "locals", "open", "quit", "exit", "__import__", "getattr", "setattr",
    "delattr", "vars",
}
FORBIDDEN_CALLS = {
    "export", "exportShape", "exportDXF", "toString", "readAndDeleteFile",
    "exportBrep", "exportBin", "exportStep", "exportStl", "importStep", "importDXF",
    "importShape", "load", "loadtxt", "save", "savetxt", "savez", "savez_compressed",
    "memmap", "fromfile", "tofile",
}
UNSUPPORTED_DISPLAY_CALLS = {"display", "show", "show_object"}
SKETCH_INCOMPATIBLE_WORKPLANE_METHODS = {
    "moveTo", "lineTo", "threePointArc", "sagittaArc", "radiusArc",
    "close", "extrude", "revolve", "sweep", "loft", "hole",
    "cutBlind", "cutThruAll",
}
# Keep OCC out of the controller process.  Importing CadQuery loads native OCC
# libraries and, on some Windows builds, can crash during interpreter teardown.
# Generated geometry is validated against this explicit CadQuery 2.x surface and
# executed only by the isolated runtime subprocess.
PUBLIC_ROOTS = {
    "Assembly", "BoundBox", "Color", "Compound", "CQ", "Edge", "Face",
    "Location", "Matrix", "NearestToPointSelector", "Plane", "Shape", "Shell",
    "Sketch", "Solid", "Vector", "Vertex", "Wire", "Workplane",
}
PUBLIC_METHODS = {
    "Workplane", "add", "all", "ancestors", "arc", "box", "cboreHole",
    "center", "chamfer", "circle", "clean", "close", "combine", "cone",
    "consolidateWires", "copyWorkplane", "cskHole", "cut", "cutBlind",
    "cutEach", "cutThruAll", "cylinder", "each", "eachpoint", "edges",
    "ellipse", "end", "extrude", "faces", "fillet", "filter", "findSolid",
    "first", "hLine", "hLineTo", "hole", "intersect", "invoke", "item",
    "last", "line", "lineTo", "loft", "map", "mirror", "mirrorX", "mirrorY",
    "move", "moveTo", "offset2D", "parametricCurve", "parametricSpline",
    "placeSketch", "polarArray", "polygon", "polyline", "pushPoints", "rarray",
    "radiusArc", "rect", "revolve", "rotate", "rotateAboutCenter", "sagittaArc",
    "section", "shell", "shells", "slot2D", "solids", "sphere", "spline",
    "split", "sweep", "tag", "tangentArcPoint", "text", "threePointArc",
    "toPending", "torus", "transformed", "translate", "twistExtrude", "union",
    "vLine", "vLineTo", "vertices", "wedge", "wire", "wires", "workplane",
    "workplaneFromTagged",
}
# These calls depend on the returned Workplane's cursor/selection/solid state.
# CadQuery shares some pending-profile context (for example a discarded
# ``wp.circle(...)`` can still leave a pending wire), so rejecting every bare
# fluent call would incorrectly reject valid multi-profile programs.  Limit the
# fatal rule to operations whose discarded return is deterministically unsafe.
DISCARD_UNSAFE_WORKPLANE_METHODS = {
    "center", "move", "moveTo", "line", "lineTo", "hLine", "hLineTo",
    "vLine", "vLineTo", "threePointArc", "sagittaArc", "radiusArc",
    "tangentArcPoint", "spline", "parametricCurve", "parametricSpline",
    "polyline", "transformed", "workplane", "workplaneFromTagged",
    "copyWorkplane", "translate", "rotate", "rotateAboutCenter", "mirror",
    "mirrorX", "mirrorY", "extrude", "twistExtrude", "revolve", "sweep",
    "loft", "cut", "union", "intersect", "fillet", "chamfer", "shell",
    "split", "hole", "cboreHole", "cskHole", "cutBlind", "cutThruAll",
}

INVALID_CADQUERY_KEYWORDS = {
    "Workplane": {
        "plane": "invalid_cadquery_kwarg:Workplane:plane_use_inPlane_or_positional",
    },
    "circle": {
        "r": "invalid_cadquery_kwarg:circle:r_use_positional_radius",
    },
    "extrude": {
        "distance": "invalid_cadquery_kwarg:extrude:distance_use_positional_until",
        "cut": "invalid_cadquery_kwarg:extrude:cut_use_combine_or_explicit_cut",
    },
}

# Compact Op-CAD aliases are input-language tokens, not executable CadQuery
# methods or helper functions.  Treating them as merely ``unknown_op`` lets a
# copied DSL program pass the static gate and waste an OCC execution.  The
# formal prompt requires direct CadQuery calls, so any residual alias is a
# deterministic code-generation error and must trigger regeneration/repair.
UNTRANSLATED_COMPACT_DSL_CALLS = {
    "W", "m", "l", "a", "c", "v", "e", "u", "k", "o", "d", "N", "f", "b", "h",
}

COMPACT_DSL_METHOD_MAP = {
    "m": "moveTo",
    "l": "lineTo",
    "a": "threePointArc",
    "c": "close",
    "v": "revolve",
    "e": "extrude",
    "u": "union",
    "k": "cut",
    "o": "circle",
    "d": "edges",
    "f": "fillet",
    "b": "box",
    "h": "chamfer",
}

# Recursive ``ast.NodeTransformer`` and ``ast.unparse`` calls consume several
# Python stack frames for every fluent CadQuery call.  LLMs can emit thousands
# of chained calls in a syntactically valid expression, so reject such code at
# the static gate instead of allowing a controller-level ``RecursionError`` to
# abort the whole benchmark task.
MAX_SAFE_AST_DEPTH = 300


class _CompactDslAliasExpander(ast.NodeTransformer):
    """Expand only syntax aliases explicitly advertised by compact Op-CAD text."""

    def __init__(self) -> None:
        self.changed = False

    @staticmethod
    def _cq_attribute(name: str) -> ast.Attribute:
        return ast.Attribute(value=ast.Name(id="cq", ctx=ast.Load()), attr=name, ctx=ast.Load())

    def visit_Call(self, node: ast.Call):  # noqa: N802 - AST visitor API
        node = self.generic_visit(node)
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "Workplane"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "cq"
            and len(node.args) == 3
            and not node.keywords
        ):
            origin, normal, xdir = node.args
            plane = ast.Call(
                func=self._cq_attribute("Plane"),
                args=[],
                keywords=[
                    ast.keyword(arg="origin", value=ast.Call(func=self._cq_attribute("Vector"), args=[origin], keywords=[])),
                    ast.keyword(arg="normal", value=ast.Call(func=self._cq_attribute("Vector"), args=[normal], keywords=[])),
                    ast.keyword(arg="xDir", value=ast.Call(func=self._cq_attribute("Vector"), args=[xdir], keywords=[])),
                ],
            )
            node.args = [plane]
            self.changed = True
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "union"
            and not node.args
            and not node.keywords
        ):
            self.changed = True
            return node.func.value
        if isinstance(node.func, ast.Name) and node.func.id == "N":
            node.func = self._cq_attribute("NearestToPointSelector")
            self.changed = True
        elif isinstance(node.func, ast.Name) and node.func.id == "W" and len(node.args) == 3:
            origin, normal, xdir = node.args
            plane = ast.Call(
                func=self._cq_attribute("Plane"),
                args=[],
                keywords=[
                    ast.keyword(arg="origin", value=ast.Call(func=self._cq_attribute("Vector"), args=[origin], keywords=[])),
                    ast.keyword(arg="normal", value=ast.Call(func=self._cq_attribute("Vector"), args=[normal], keywords=[])),
                    ast.keyword(arg="xDir", value=ast.Call(func=self._cq_attribute("Vector"), args=[xdir], keywords=[])),
                ],
            )
            node.func = self._cq_attribute("Workplane")
            node.args = [plane]
            self.changed = True
        elif (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "N"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "cq"
        ):
            node.func.attr = "NearestToPointSelector"
            self.changed = True
        elif isinstance(node.func, ast.Attribute) and node.func.attr in COMPACT_DSL_METHOD_MAP:
            alias = node.func.attr
            node.func.attr = COMPACT_DSL_METHOD_MAP[alias]
            self.changed = True
            if alias == "o":
                radius = [item for item in node.keywords if item.arg == "r"]
                if len(radius) == 1:
                    node.args.append(radius[0].value)
                    node.keywords = [item for item in node.keywords if item.arg != "r"]
            elif alias == "h":
                length = [item for item in node.keywords if item.arg == "len_param"]
                if len(length) == 1:
                    node.args.append(length[0].value)
                    node.keywords = [item for item in node.keywords if item.arg != "len_param"]
        return node


def strip_code_fence(text: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # A provider may stop at its output limit before closing the fence.  Strip
    # the opening marker so static diagnostics report the actual incomplete
    # Python construct rather than a misleading line-1 Markdown syntax error.
    return re.sub(r"^\s*```(?:python)?\s*", "", text, count=1, flags=re.I).strip()


def _assigned_names(node: ast.AST) -> set[str]:
    return {
        item.id for item in ast.walk(node)
        if isinstance(item, ast.Name) and isinstance(item.ctx, (ast.Store, ast.Param))
    }


def _loaded_names(node: ast.AST) -> set[str]:
    return {
        item.id for item in ast.walk(node)
        if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)
    }


def _call_name(call: ast.Call) -> str:
    target = call.func
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        pieces = [target.attr]
        value = target.value
        while isinstance(value, ast.Attribute):
            pieces.append(value.attr)
            value = value.value
        if isinstance(value, ast.Name):
            pieces.append(value.id)
        return ".".join(reversed(pieces))
    return "call"


def _cadquery_workplane_value(
    value: ast.AST | None,
    known_workplanes: set[str],
    import_aliases: set[str],
) -> bool:
    """Return whether an expression produces a CadQuery Workplane value.

    CadQuery's fluent Workplane API returns new objects.  A frequent LLM error
    is therefore to emit ``wp.moveTo(...)`` as a bare expression and then keep
    using the unchanged ``wp``.  Track the small amount of local type
    information needed to reject that silent no-op before OCC execution.
    """
    current = value
    while True:
        if isinstance(current, ast.Name):
            return current.id in known_workplanes
        if not isinstance(current, ast.Call):
            return False
        final = _call_name(current).split(".")[-1]
        if final == "Workplane" and isinstance(current.func, ast.Attribute):
            return (
                isinstance(current.func.value, ast.Name)
                and current.func.value.id in import_aliases
            )
        if not isinstance(current.func, ast.Attribute) or final not in PUBLIC_METHODS:
            return False
        current = current.func.value


def _ast_depth(root: ast.AST) -> int:
    """Return AST depth without recursive Python calls."""
    maximum = 1
    stack: list[tuple[ast.AST, int]] = [(root, 1)]
    while stack:
        node, depth = stack.pop()
        maximum = max(maximum, depth)
        stack.extend((child, depth + 1) for child in ast.iter_child_nodes(node))
    return maximum


def _bounded_unparse(tree: ast.AST, original: str) -> str:
    """Unparse a normal-sized tree without allowing a task-level exception."""
    try:
        ast.fix_missing_locations(tree)
        return ast.unparse(tree).strip() + "\n"
    except (RecursionError, MemoryError):
        return original.strip() + "\n"


def _discarded_workplane_result_issues(
    statements: list[ast.stmt], import_aliases: set[str],
) -> list[str]:
    """Find fluent Workplane calls whose returned geometry is discarded."""
    issues: list[str] = []
    known_workplanes: set[str] = set()
    for statement in statements:
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            call = statement.value
            if (
                isinstance(call.func, ast.Attribute)
                and _call_name(call).split(".")[-1] in DISCARD_UNSAFE_WORKPLANE_METHODS
                and _cadquery_workplane_value(call.func.value, known_workplanes, import_aliases)
            ):
                issues.append(
                    "discarded_cadquery_workplane_result:"
                    f"{_call_name(call).split('.')[-1]}:line_{getattr(statement, 'lineno', '?')}"
                )

        value: ast.AST | None = None
        targets: set[str] = set()
        if isinstance(statement, ast.Assign):
            value = statement.value
            targets = set().union(*(_assigned_names(target) for target in statement.targets))
        elif isinstance(statement, ast.AnnAssign):
            value = statement.value
            targets = _assigned_names(statement.target)
        if targets:
            if _cadquery_workplane_value(value, known_workplanes, import_aliases):
                known_workplanes.update(targets)
            else:
                known_workplanes.difference_update(targets)

        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            issues.extend(_discarded_workplane_result_issues(statement.body, import_aliases))
    return issues


def _sketch_workplane_method_issues(tree: ast.AST) -> list[str]:
    """Reject Workplane-only methods invoked on values created by ``.sketch()``.

    CAD-Coder descriptions sometimes use the ordinary variable name
    ``sketch`` for a Workplane.  Models can therefore over-interpret the name,
    insert ``wp.sketch()``, and then call Workplane methods such as ``moveTo``
    on the resulting :class:`cadquery.Sketch`.  That program parses cleanly
    but is guaranteed to fail at runtime.  This check uses only local AST type
    evidence and never guesses from variable names.
    """
    sketch_values: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if not isinstance(value, ast.Call):
            continue
        if _call_name(value).split(".")[-1] == "finalize":
            # Sketch.finalize() deliberately converts the builder back to the
            # owning Workplane, on which extrusion is valid.
            continue
        calls_sketch = any(
            isinstance(item, ast.Call)
            and _call_name(item).split(".")[-1] == "sketch"
            for item in ast.walk(value)
        )
        if not calls_sketch:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            sketch_values.update(_assigned_names(target))

    issues: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        method = node.func.attr
        if method not in SKETCH_INCOMPATIBLE_WORKPLANE_METHODS:
            continue
        receiver = node.func.value
        receiver_is_sketch = (
            isinstance(receiver, ast.Name) and receiver.id in sketch_values
        ) or any(
            isinstance(item, ast.Call)
            and _call_name(item).split(".")[-1] == "sketch"
            for item in ast.walk(receiver)
        )
        if receiver_is_sketch:
            issues.append(
                f"invalid_cadquery_object_method:Sketch.{method}_use_Workplane"
            )
    return sorted(set(issues))


def _pending_profile_boolean_issues(tree: ast.Module) -> list[str]:
    """Reject Booleans whose local operands are provably unextruded profiles."""
    environment: dict[str, str] = {}
    issues: list[str] = []
    profile_methods = {
        "moveTo", "lineTo", "polyline", "close", "circle", "ellipse", "rect",
        "polygon", "spline", "threePointArc", "radiusArc", "sagittaArc", "center",
        "pushPoints", "workplane",
    }
    solid_methods = {
        "extrude", "revolve", "sweep", "loft", "twistExtrude", "box", "cylinder",
        "sphere", "torus", "wedge", "hole", "cboreHole", "cskHole",
    }
    preserving_methods = {
        "translate", "rotate", "rotateAboutCenter", "mirror", "transformed",
        "faces", "edges", "vertices", "solids", "shells", "wires", "tag",
    }

    def infer(value: ast.AST | None, depth: int = 0) -> str:
        if depth > MAX_SAFE_AST_DEPTH:
            return "unknown"
        chain: list[ast.Call] = []
        current = value
        while isinstance(current, ast.Call):
            chain.append(current)
            if isinstance(current.func, ast.Attribute):
                current = current.func.value
            else:
                current = None
                break
        state = environment.get(current.id, "unknown") if isinstance(current, ast.Name) else "unknown"
        for call in reversed(chain):
            final = _call_name(call).split(".")[-1]
            if final == "Workplane":
                state = "workplane"
            elif final in profile_methods and state in {"workplane", "profile"}:
                state = "profile"
            elif final in solid_methods:
                state = "solid"
            elif final in {"union", "cut", "intersect"}:
                operand_types = [infer(argument, depth + 1) for argument in call.args]
                if state == "profile" or "profile" in operand_types:
                    issues.append(
                        f"invalid_boolean_operand:{final}_requires_solid_not_pending_profile:"
                        f"line_{getattr(call, 'lineno', '?')}"
                    )
                state = "solid" if state == "solid" else state
            elif final in preserving_methods:
                if final == "translate" and state == "workplane":
                    issues.append(
                        "ineffective_empty_workplane_translate_use_transformed:"
                        f"line_{getattr(call, 'lineno', '?')}"
                    )
            else:
                state = state if state != "unknown" else "unknown"
        return state

    for statement in tree.body:
        value: ast.AST | None = None
        targets: set[str] = set()
        if isinstance(statement, ast.Assign):
            value = statement.value
            targets = set().union(*(_assigned_names(target) for target in statement.targets))
        elif isinstance(statement, ast.AnnAssign):
            value = statement.value
            targets = _assigned_names(statement.target)
        inferred = infer(value)
        for target in targets:
            environment[target] = inferred
    return sorted(set(issues))


def normalize_cadquery_code(requirement: str, code: str) -> str:
    """Canonicalise syntax, strip top-level export side effects, and add `result`."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, RecursionError, MemoryError):
        return code
    changed = False
    tree_depth = _ast_depth(tree)
    if (
        "DSL:" in requirement
        and "Code:" in requirement
        and tree_depth <= MAX_SAFE_AST_DEPTH
    ):
        expander = _CompactDslAliasExpander()
        tree = expander.visit(tree)
        changed = expander.changed
    safe_body: list[ast.stmt] = []
    source_rewrite = False
    added_result_name: str | None = None
    for statement in tree.body:
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            name = _call_name(statement.value)
            if name.split(".")[-1] in FORBIDDEN_CALLS or "exporters" in name:
                changed = True
                source_rewrite = True
                continue
        value = None
        if isinstance(statement, ast.Assign):
            value = statement.value
        elif isinstance(statement, ast.AnnAssign):
            value = statement.value
        if isinstance(value, ast.Call):
            name = _call_name(value)
            final = name.split(".")[-1]
            passthrough = None
            if final in {
                "export", "exportShape", "exportDXF", "exportBrep",
                "exportBin", "exportStep", "exportStl",
            } and isinstance(value.func, ast.Attribute):
                passthrough = value.func.value
            elif "exporters" in name and value.args:
                passthrough = value.args[0]
            if passthrough is not None:
                statement.value = passthrough
                changed = True
                source_rewrite = True
        if isinstance(statement, ast.Import):
            original_count = len(statement.names)
            statement.names = [
                alias for alias in statement.names
                if not alias.name.startswith("cadquery.exporters")
                and not alias.name.startswith("cadquery.importers")
            ]
            changed = changed or len(statement.names) != original_count
            source_rewrite = source_rewrite or len(statement.names) != original_count
            if not statement.names:
                continue
        if isinstance(statement, ast.ImportFrom) and (statement.module or "") == "cadquery":
            original_count = len(statement.names)
            statement.names = [
                alias for alias in statement.names
                if alias.name not in {"exporters", "importers"}
            ]
            changed = changed or len(statement.names) != original_count
            source_rewrite = source_rewrite or len(statement.names) != original_count
            if not statement.names:
                continue
        safe_body.append(statement)
    tree.body = safe_body
    top_level_names: list[str] = []
    has_result = False
    for statement in tree.body:
        targets: list[ast.expr] = []
        if isinstance(statement, ast.Assign):
            targets = list(statement.targets)
        elif isinstance(statement, ast.AnnAssign):
            targets = [statement.target]
        for target in targets:
            for name in _assigned_names(target):
                top_level_names.append(name)
                has_result = has_result or name == "result"
    if not has_result and top_level_names:
        preferred_terms = ("result", "final", "model", "part", "body", "solid", "shape", "assembly", "base", "workplane", "wp")
        preferred = [name for name in top_level_names if any(term in name.lower() for term in preferred_terms)]
        candidate = (preferred or top_level_names)[-1]
        uses_cadquery = any(isinstance(node, ast.Name) and node.id in {"cq", "cadquery"} for node in ast.walk(tree))
        if uses_cadquery:
            tree.body.append(
                ast.Assign(
                    targets=[ast.Name(id="result", ctx=ast.Store())],
                    value=ast.Name(id=candidate, ctx=ast.Load()),
                )
            )
            changed = True
            added_result_name = candidate
    if not changed:
        return code.strip() + "\n"
    if tree_depth > MAX_SAFE_AST_DEPTH:
        if added_result_name is not None and not source_rewrite:
            return code.strip() + f"\nresult = {added_result_name}\n"
        return code.strip() + "\n"
    return _bounded_unparse(tree, code)


def validate_cadquery_code(code: str) -> list[str]:
    issues: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        return [f"syntax:{error.msg}:line_{error.lineno}"]
    except (RecursionError, MemoryError):
        return ["ast_parse_resource_limit"]
    result_assigned = False
    has_cadquery_call = False
    import_aliases = {"cq"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                    issues.append(f"forbidden_import:{alias.name}")
                if alias.name == "cadquery":
                    import_aliases.add(alias.asname or "cadquery")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                issues.append(f"forbidden_import:{node.module}")
            if (node.module or "").split(".")[0] == "cadquery":
                for alias in node.names:
                    if alias.name in {"exporters", "importers"}:
                        issues.append(f"forbidden_cadquery_io_import:{alias.name}")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            issues.append(f"forbidden_name:{node.id}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            issues.append(f"forbidden_dunder:{node.attr}")
        if isinstance(node, ast.Call):
            name = _call_name(node)
            final = name.split(".")[-1]
            root = name.split(".")[0]
            if isinstance(node.func, ast.Name) and final in UNSUPPORTED_DISPLAY_CALLS:
                issues.append(f"unsupported_display_call:{final}")
            if isinstance(node.func, ast.Name) and final in {"Plane", "Vector"}:
                issues.append(f"unqualified_cadquery_constructor:{final}:use_cq.{final}")
            if final in UNTRANSLATED_COMPACT_DSL_CALLS:
                issues.append(f"untranslated_compact_dsl_alias:{final}")
            if final == "Workplane" and node.args:
                plane = node.args[0]
                if (
                    isinstance(plane, ast.Attribute)
                    and isinstance(plane.value, ast.Attribute)
                    and plane.value.attr == "Plane"
                    and plane.attr.upper() in {
                        "XY", "XZ", "YZ", "FRONT", "BACK", "LEFT", "RIGHT", "TOP", "BOTTOM",
                    }
                ):
                    issues.append("invalid_workplane_plane_factory_not_called")
            if final == "Workplane" and len(node.args) > 2:
                issues.append("invalid_cadquery_arity:Workplane_use_cq_Plane_for_custom_frame")
            if final in FORBIDDEN_CALLS or "exporters" in name:
                issues.append(f"forbidden_export_call:{name}")
            if name == "cq.Sketch":
                issues.append("unsupported_cadquery_sketch_use_workplane")
            if name == "cq.Location":
                issues.append("unsupported_cadquery_location_use_rotate_translate")
            if final == "Plane" and root in import_aliases:
                plane_keywords = {keyword.arg for keyword in node.keywords if keyword.arg}
                if plane_keywords - {"origin", "xDir", "normal"}:
                    issues.append("unsupported_cadquery_plane_use_workplane_rotate_translate")
            keyword_names = {keyword.arg for keyword in node.keywords if keyword.arg}
            for keyword, issue in INVALID_CADQUERY_KEYWORDS.get(final, {}).items():
                if keyword in keyword_names:
                    issues.append(issue)
            if final == "circle" and "center" in keyword_names:
                issues.append("invalid_cadquery_kwarg:circle:center_use_center_method")
            if final == "circle" and len(node.args) > 2:
                issues.append("invalid_cadquery_arity:circle_accepts_radius_and_for_construction")
            if final == "rotate" and len(node.args) < 3 and not {
                "axisStartPoint", "axisEndPoint", "angleDegrees",
            }.issubset(keyword_names):
                issues.append("invalid_cadquery_arity:rotate_requires_axis_start_axis_end_angle")
            if final == "revolve" and "angle_degrees" in keyword_names:
                issues.append(
                    "invalid_cadquery_kwarg:revolve:angle_degrees_use_angleDegrees"
                )
            if final == "lineTo" and len(node.args) < 2 and not {"x", "y"}.issubset(keyword_names):
                issues.append("invalid_cadquery_arity:lineTo_requires_x_y")
            if final == "moveTo" and not (
                len(node.args) == 2 or (not node.args and {"x", "y"}.issubset(keyword_names))
            ):
                issues.append("invalid_cadquery_arity:moveTo_requires_exactly_x_y")
            if final == "translate" and not (
                len(node.args) == 1 or (not node.args and "vector" in keyword_names)
            ):
                issues.append("invalid_cadquery_arity:translate_requires_one_vector")
            if final == "transformOffset":
                issues.append("unsupported_cadquery_method:transformOffset_use_transformed")
            if final in {"cut", "union", "intersect"} and not node.args:
                issues.append(f"invalid_cadquery_arity:{final}_requires_tool_operand")
            if final == "threePointArc" and not (
                len(node.args) == 2
                or (
                    len(node.args) == 3
                    and isinstance(node.args[2], ast.Constant)
                    and isinstance(node.args[2].value, bool)
                )
            ):
                issues.append("invalid_cadquery_arity:threePointArc_requires_two_points")
            if final == "arc":
                issues.append("unsupported_cadquery_method:arc_use_threePointArc_radiusArc_or_sagittaArc")
            if final == "filterBy":
                issues.append("unsupported_cadquery_method:filterBy")
            if final in {"located", "moved", "fuse", "rotated", "scale", "copy", "transform"}:
                issues.append(f"unsupported_ambiguous_method:{final}")
            if "importers" in name:
                issues.append(f"forbidden_import_call:{name}")
            if root in import_aliases:
                has_cadquery_call = True
                if len(name.split(".")) == 2 and final not in PUBLIC_ROOTS:
                    issues.append(f"unknown_cadquery_root_call:{name}")
            elif isinstance(node.func, ast.Attribute) and final in PUBLIC_METHODS:
                has_cadquery_call = True
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and "result" in _assigned_names(node):
            result_assigned = True
    if not result_assigned:
        issues.append("missing_top_level_result_assignment")
    if not has_cadquery_call:
        issues.append("missing_cadquery_call")
    issues.extend(_sketch_workplane_method_issues(tree))
    issues.extend(_pending_profile_boolean_issues(tree))
    issues.extend(_discarded_workplane_result_issues(tree.body, import_aliases))
    if len(code) > 100_000:
        issues.append("code_too_large")
    return sorted(set(issues))


def build_ast_graph(code: str) -> dict:
    tree = ast.parse(code)
    nodes: list[dict] = []
    latest: dict[str, str] = {}
    for index, statement in enumerate(tree.body):
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            continue
        node_id = f"s{index:03d}"
        defines = sorted(_assigned_names(statement))
        uses = sorted(_loaded_names(statement) - set(defines))
        inputs = sorted({latest[name] for name in uses if name in latest})
        operations = [_call_name(item) for item in ast.walk(statement) if isinstance(item, ast.Call)]
        nodes.append({
            "id": node_id, "statement_type": type(statement).__name__, "operations": operations,
            "defines": defines, "uses": uses, "inputs": inputs,
            "lineno": getattr(statement, "lineno", None), "end_lineno": getattr(statement, "end_lineno", None),
        })
        for name in defines:
            latest[name] = node_id
    return {"nodes": nodes, "output": latest.get("result"),
            "source_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest()}


def generic_operation(name: str) -> str:
    lowered = name.lower().split(".")[-1]
    aliases = {
        "workplane": "workplane", "box": "box", "cylinder": "cylinder", "sphere": "sphere",
        "circle": "circle", "ellipse": "circle", "rect": "profile", "polygon": "profile",
        "polyline": "profile", "spline": "profile", "moveto": "move", "lineto": "line",
        "threepointarc": "arc", "sagittaarc": "arc", "radiusarc": "arc", "close": "close",
        "extrude": "extrude", "cutblind": "cut", "cutthruall": "cut", "hole": "cut",
        "cborehole": "cut", "cskhole": "cut", "revolve": "revolve", "sweep": "sweep",
        "loft": "loft", "union": "union", "combine": "union", "cut": "cut",
        "intersect": "intersect", "fillet": "fillet", "chamfer": "chamfer",
        "rotate": "rotate", "rotateaboutcenter": "rotate", "translate": "translate",
        "mirror": "mirror", "rarray": "pattern", "polararray": "pattern", "eachpoint": "pattern",
        "transformed": "transform", "workplanefromtag": "workplane",
    }
    return aliases.get(lowered, lowered)


def graph_features(graph: dict | None) -> dict:
    nodes = (graph or {}).get("nodes") or []
    operations = {generic_operation(op) for node in nodes for op in node.get("operations", []) if op}
    by_id = {str(node.get("id")): node for node in nodes}
    edges = {
        f"{generic_operation((by_id.get(str(parent), {}).get('operations') or ['?'])[-1])}->{generic_operation((node.get('operations') or ['?'])[-1])}"
        for node in nodes for parent in node.get("inputs", [])
    }
    return {"operations": operations, "edges": edges, "node_count": len(nodes)}
