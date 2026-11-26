"""Reference-free structural and natural-language CAD acceptance checks."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Sequence

from .graphs import (
    asymmetric_graph_difference,
    graph_counters,
    graph_difference,
    validate_requirement_graph,
)
from .program_graph import analyse_static, build_program_graph, output_dependency_subgraph


_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
_UNIT = r"mm|cm|m|in(?:ch(?:es)?)?|units?|毫米|厘米|米"
_COUNT_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}
_UNIT_SCALE = {"mm": 1.0, "millimeter": 1.0, "millimeters": 1.0,
               "毫米": 1.0, "cm": 10.0, "厘米": 10.0,
               "m": 1000.0, "米": 1000.0,
               "in": 25.4, "inch": 25.4, "inches": 25.4,
               "unit": 1.0, "units": 1.0}
_NUMBER_WORD_VALUES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_NUMBER_WORD_PATTERN = re.compile(
    r"\b(?:negative\s+|minus\s+)?(?:" + "|".join(_NUMBER_WORD_VALUES) + r")"
    r"(?:[ -]+(?:and[ -]+)?(?:" + "|".join(_NUMBER_WORD_VALUES) + r"|hundred|thousand|point))*\b",
    re.I,
)


def _normalize_spoken_numbers(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        phrase = match.group(0).lower().replace("-", " ")
        tokens = [token for token in phrase.split() if token != "and"]
        negative = bool(tokens and tokens[0] in {"negative", "minus"})
        if negative:
            tokens = tokens[1:]
        if "point" in tokens:
            split = tokens.index("point")
            integer_tokens, decimal_tokens = tokens[:split], tokens[split + 1:]
        else:
            integer_tokens, decimal_tokens = tokens, []
        total = current = 0
        for token in integer_tokens:
            if token == "hundred":
                current = max(current, 1) * 100
            elif token == "thousand":
                total += max(current, 1) * 1000
                current = 0
            else:
                current += _NUMBER_WORD_VALUES[token]
        integer = total + current
        decimals = "".join(
            str(_NUMBER_WORD_VALUES[token]) for token in decimal_tokens
            if token in _NUMBER_WORD_VALUES
        )
        rendered = str(integer) + (f".{decimals}" if decimals else "")
        return f"-{rendered}" if negative else rendered
    return _NUMBER_WORD_PATTERN.sub(replace, text)


@dataclass(frozen=True)
class ConstraintConfig:
    max_operation_missing_rate: float = 0.0
    max_dependency_missing_rate: float = 0.0
    minimum_volume: float = 1e-9
    reject_unknown_operations: bool = True
    dimension_relative_tolerance: float = 0.03
    dimension_absolute_tolerance: float = 0.05
    position_relative_tolerance: float = 0.03
    position_absolute_tolerance: float = 0.05
    radius_relative_tolerance: float = 0.03
    angle_absolute_tolerance: float = 0.1
    require_trusted_requirement_graph: bool = True
    linear_unit_policy: str = "cadquery_mm"


def _count_value(value: str) -> int | None:
    lowered = value.lower()
    if lowered.isdigit():
        return int(lowered)
    return _COUNT_WORDS.get(lowered)


def _scaled(value: str, unit: str | None, linear_unit_policy: str) -> float:
    if linear_unit_policy == "native_numeric":
        return float(value)
    if linear_unit_policy != "cadquery_mm":
        raise ValueError(f"unsupported_linear_unit_policy:{linear_unit_policy}")
    return float(value) * _UNIT_SCALE.get((unit or "").lower(), 1.0)


def extract_requirement_facts(
    text: str, *, linear_unit_policy: str = "cadquery_mm",
) -> dict:
    """Extract only high-confidence, deterministic constraints from the request."""
    facts: dict = {
        "dimensions": [], "overall_centers": [], "hole_count": None,
        "hole_radii": [], "extrude_directions": [], "primitive_counts": {},
        "boolean_operations": {}, "translations": [], "euler_angles": [],
        "transform_orders": [],
    }
    if not text.strip():
        return facts
    lowered = _normalize_spoken_numbers(text.lower().replace("×", "x"))
    seen_transform_orders: set[tuple[str, str]] = set()
    for sentence in re.split(r"[.;。；]", lowered):
        rotate_match = re.search(r"\brotate(?:d|s|ing)?\b", sentence)
        translate_match = re.search(r"\btranslate(?:d|s|ing)?\b", sentence)
        if not rotate_match or not translate_match:
            continue
        order = (
            ("rotate", "translate")
            if rotate_match.start() < translate_match.start()
            else ("translate", "rotate")
        )
        if order not in seen_transform_orders:
            facts["transform_orders"].append({
                "operations": list(order), "evidence": sentence.strip(),
            })
            seen_transform_orders.add(order)
    triple_pattern = re.compile(
        rf"(?:dimensions?|size|box|block|cuboid|prism|长方体|尺寸)[^\d+-]{{0,30}}"
        rf"({_NUMBER})\s*({_UNIT})?\s*(?:by|x|乘)\s*"
        rf"({_NUMBER})\s*({_UNIT})?\s*(?:by|x|乘)\s*"
        rf"({_NUMBER})\s*({_UNIT})?",
        re.I,
    )
    for match in triple_pattern.finditer(lowered):
        trailing_unit = match.group(6) or match.group(4) or match.group(2)
        values = [
            _scaled(match.group(1), match.group(2) or trailing_unit, linear_unit_policy),
            _scaled(match.group(3), match.group(4) or trailing_unit, linear_unit_policy),
            _scaled(match.group(5), match.group(6) or trailing_unit, linear_unit_policy),
        ]
        context = lowered[max(0, match.start() - 48):match.end() + 16]
        scope = "overall" if re.search(
            r"overall|whole\s+model|final\s+model|model\s+(?:dimensions?|size)|bounding\s+box|整体|总体|模型尺寸",
            context,
        ) else "unspecified"
        facts["dimensions"].append({
            "values": values, "ordered": False, "evidence": match.group(0), "scope": scope,
        })

    named_patterns = {
        "x": [
            rf"(?:length\b|x\s*(?:size|dimension)\b|长度?|长)\s*(?:is|of|=|为|[:：])?\s*({_NUMBER})\s*({_UNIT})?",
            rf"({_NUMBER})\s*({_UNIT})?\s*(?:long|in\s+length|长度?|长)\b",
        ],
        "y": [
            rf"(?:width\b|y\s*(?:size|dimension)\b|宽度?|宽)\s*(?:is|of|=|为|[:：])?\s*({_NUMBER})\s*({_UNIT})?",
            rf"({_NUMBER})\s*({_UNIT})?\s*(?:wide|in\s+width|宽度?|宽)\b",
        ],
        "z": [
            rf"(?:height\b|thickness\b|z\s*(?:size|dimension)\b|高度?|高|厚度?|厚)\s*(?:is|of|=|为|[:：])?\s*({_NUMBER})\s*({_UNIT})?",
            rf"({_NUMBER})\s*({_UNIT})?\s*(?:high|thick|in\s+height|in\s+thickness|高度?|高|厚度?|厚)\b",
        ],
    }
    named: dict[str, float] = {}
    evidence: list[str] = []
    for axis, patterns in named_patterns.items():
        for pattern in patterns:
            match = re.search(pattern, lowered, re.I)
            if match:
                named[axis] = _scaled(match.group(1), match.group(2), linear_unit_policy)
                evidence.append(match.group(0))
                break
    for axis_pair, label in (("xz", "length and height"), ("xy", "length and width"), ("yz", "width and height")):
        match = re.search(
            rf"({_NUMBER})\s*({_UNIT})?\s*(?:in\s+)?{label}\b", lowered, re.I,
        )
        if match:
            value = _scaled(match.group(1), match.group(2), linear_unit_policy)
            for axis in axis_pair:
                named.setdefault(axis, value)
            evidence.append(match.group(0))
    if len(named) == 3:
        facts["dimensions"].append({
            "values": [named["x"], named["y"], named["z"]],
            "ordered": False, "evidence": "; ".join(evidence), "scope": "unspecified",
        })

    multi_part = bool(re.search(
        r"\b(?:first|second|third|fourth|fifth)\s+part\b|\b(?:part|component|feature)\s*\d+\b|"
        r"\bfor\s+the\s+(?:first|second|third|fourth|fifth)\s+(?:part|component)\b|"
        r"第一部分|第二部分|第三部分|第[一二三四五六七八九十]+个部件",
        lowered,
    ))
    explicit_dimensions = [item for item in facts["dimensions"] if item.get("scope") == "overall"]
    candidates = explicit_dimensions or ([] if multi_part else facts["dimensions"])
    unique_dimensions: list[dict] = []
    seen_dimensions: set[tuple[float, float, float]] = set()
    for item in candidates:
        key = tuple(round(float(value), 12) for value in item["values"])
        if key not in seen_dimensions:
            unique_dimensions.append(item)
            seen_dimensions.add(key)
    facts["dimensions"] = unique_dimensions

    translation_pattern = re.compile(
        rf"(?:translation\s+vector|平移向量)\s*(?:of|to|is|=|为|[:：])?\s*[\[(]?\s*"
        rf"({_NUMBER})\s*[,，]\s*({_NUMBER})\s*[,，]\s*(?:and\s+)?({_NUMBER})",
        re.I,
    )
    for match in translation_pattern.finditer(lowered):
        facts["translations"].append({
            "values": [float(match.group(1)), float(match.group(2)), float(match.group(3))],
            "evidence": match.group(0),
        })

    euler_pattern = re.compile(
        rf"Euler\s+angles?\s*(?:set\s+to|of|are|is)?\s*(?:[:：=])?\s*[\[(]?\s*"
        rf"({_NUMBER})\s*(?:degrees?)?\s*[,，]\s*"
        rf"({_NUMBER})\s*(?:degrees?)?\s*[,，]\s*(?:and\s+)?"
        rf"({_NUMBER})\s*(?:degrees?)?",
        re.I,
    )
    seen_euler: set[tuple[float, float, float]] = set()
    for match in euler_pattern.finditer(lowered):
        values = tuple(float(match.group(index)) for index in (1, 2, 3))
        key = tuple(round(value, 9) for value in values)
        if key in seen_euler:
            continue
        seen_euler.add(key)
        facts["euler_angles"].append({
            "values": list(values),
            "evidence": match.group(0),
        })

    center_pattern = re.compile(
        rf"(?:overall\s+center|whole\s+model\s+center|model\s+center|body\s+center|模型中心|整体中心)"
        rf"\s*(?:at|is|=|为|位于)?\s*[:：]?\s*[\[(]?\s*({_NUMBER})\s*[,，]\s*({_NUMBER})(?:\s*[,，]\s*({_NUMBER}))?",
        re.I,
    )
    for match in center_pattern.finditer(lowered):
        context = lowered[max(0, match.start() - 24):match.start()]
        if "hole" in context or "孔" in context:
            continue
        values = [float(match.group(1)), float(match.group(2))]
        if match.group(3) is not None:
            values.append(float(match.group(3)))
        facts["overall_centers"].append({"values": values, "evidence": match.group(0)})

    count_token = r"\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|[一二两三四五六七八九十]"
    hole_patterns = [
        re.compile(rf"\b({count_token})\s+(?:circular\s+|through\s+|blind\s+)?holes?\b", re.I),
        re.compile(rf"({count_token})\s*个?\s*(?:圆|通|盲|沉头|沉孔)?孔", re.I),
    ]
    counts = []
    for pattern in hole_patterns:
        counts.extend(value for match in pattern.finditer(lowered)
                      if (value := _count_value(match.group(1))) is not None)
    if counts:
        facts["hole_count"] = max(counts)

    for match in re.finditer(
        rf"(?:hole\s+)?(?:radius|半径)\s*(?:is|=|为|[:：])?\s*({_NUMBER})\s*({_UNIT})?",
        lowered, re.I,
    ):
        facts["hole_radii"].append(
            _scaled(match.group(1), match.group(2), linear_unit_policy)
        )
    for match in re.finditer(
        rf"(?:hole\s+)?(?:diameter|直径)\s*(?:is|=|为|[:：])?\s*({_NUMBER})\s*({_UNIT})?",
        lowered, re.I,
    ):
        facts["hole_radii"].append(
            _scaled(match.group(1), match.group(2), linear_unit_policy) / 2.0
        )

    direction_patterns = [
        re.compile(r"(?:extrud\w*|拉伸)[^.;。]{0,60}?(positive|negative|正|负)?\s*([xyz])\s*(?:axis|轴)", re.I),
        re.compile(r"沿\s*(正|负)?\s*([xyz])\s*轴[^.;。]{0,30}?(?:extrud\w*|拉伸)", re.I),
    ]
    for pattern in direction_patterns:
        for match in pattern.finditer(lowered):
            sign = -1 if (match.group(1) or "").lower() in {"negative", "负"} else 1
            facts["extrude_directions"].append({"axis": match.group(2).lower(), "sign": sign, "evidence": match.group(0)})

    for op, noun in (("box", r"boxes|box|blocks?|cuboids?|长方体|方块"),
                     ("cylinder", r"cylinders?|圆柱"), ("sphere", r"spheres?|球体")):
        matches = re.findall(rf"\b({count_token})\s+(?:{noun})\b", lowered, re.I)
        chinese = re.findall(rf"({count_token})\s*个?\s*(?:{noun})", lowered, re.I)
        values = [_count_value(value) for value in [*matches, *chinese]]
        values = [value for value in values if value is not None]
        if values:
            facts["primitive_counts"][op] = max(values)
    boolean_patterns = {
        "cut": r"差集|切除|减去|\bcut\b|\bsubtract\w*\b|\bremove\b",
        "union": r"并集|合并|\bunion\b|\bcombine\b|\bjoin\b",
        "intersect": r"交集|求交|\bintersect(?:\s+(?:the|with)\b|\b)|\bcommon\b",
    }
    for op, pattern in boolean_patterns.items():
        count = len(re.findall(pattern, lowered, re.I))
        if count:
            facts["boolean_operations"][op] = count
    return facts


def _close(actual: float, expected: float, relative: float, absolute: float) -> bool:
    return math.isfinite(actual) and abs(actual - expected) <= max(absolute, abs(expected) * relative)


def _active_operation_counts(program_graph: dict) -> Counter[str]:
    operations, _ = graph_counters(output_dependency_subgraph(program_graph))
    return operations


def _identity_spatial_counts(program_graph: dict, config: ConstraintConfig) -> Counter[str]:
    """Count no-op transforms so they do not trigger unexpected-op penalties."""
    counts: Counter[str] = Counter()
    for node in output_dependency_subgraph(program_graph).get("nodes", []):
        operation = node.get("op")
        literal_args = node.get("literal_args") or []
        literal_kwargs = node.get("literal_kwargs") or {}
        if operation == "rotate":
            candidate = literal_kwargs.get("rotate")
            if isinstance(candidate, (list, tuple)) and len(candidate) == 3:
                if all(isinstance(value, (int, float)) and
                       abs(float(value)) <= config.angle_absolute_tolerance
                       for value in candidate):
                    counts[operation] += 1
            elif len(literal_args) >= 3 and isinstance(literal_args[2], (int, float)):
                if abs(float(literal_args[2])) <= config.angle_absolute_tolerance:
                    counts[operation] += 1
        elif operation == "translate" and literal_args:
            candidate = literal_args[0]
            if isinstance(candidate, (list, tuple)) and len(candidate) == 3 and all(
                isinstance(value, (int, float)) and
                abs(float(value)) <= config.position_absolute_tolerance
                for value in candidate
            ):
                counts[operation] += 1
        elif operation == "scale" and literal_args:
            candidate = literal_args[0]
            values = candidate if isinstance(candidate, (list, tuple)) else [candidate]
            if values and all(isinstance(value, (int, float)) and
                              abs(float(value) - 1.0) <= config.dimension_relative_tolerance
                              for value in values):
                counts[operation] += 1
    return counts


def _check_natural_language_constraints(
    facts: dict, program_graph: dict, metrics: dict, config: ConstraintConfig,
) -> tuple[list[str], list[dict]]:
    issues: list[str] = []
    checks: list[dict] = []
    bbox = metrics.get("bbox") or {}
    bbox_size = bbox.get("size")
    bbox_min, bbox_max = bbox.get("min"), bbox.get("max")
    for item in facts["dimensions"]:
        expected = [float(value) for value in item["values"]]
        if not isinstance(bbox_size, list) or len(bbox_size) != 3:
            issues.append("unverifiable_dimensions")
            checks.append({"kind": "dimensions", "passed": False, "reason": "missing_bbox"})
            continue
        actual = [float(value) for value in bbox_size]
        left, right = (actual, expected) if item["ordered"] else (sorted(actual), sorted(expected))
        passed = all(_close(a, e, config.dimension_relative_tolerance, config.dimension_absolute_tolerance)
                     for a, e in zip(left, right))
        checks.append({"kind": "dimensions", "passed": passed, "actual": actual,
                       "expected": expected, "ordered": item["ordered"], "evidence": item["evidence"]})
        if not passed:
            issues.append("dimension_mismatch")

    for item in facts["overall_centers"]:
        expected = [float(value) for value in item["values"]]
        if not (isinstance(bbox_min, list) and isinstance(bbox_max, list) and len(bbox_min) >= len(expected)):
            issues.append("unverifiable_overall_center")
            checks.append({"kind": "overall_center", "passed": False, "reason": "missing_bbox"})
            continue
        actual = [(float(bbox_min[index]) + float(bbox_max[index])) / 2.0 for index in range(len(expected))]
        passed = all(_close(a, e, config.position_relative_tolerance, config.position_absolute_tolerance)
                     for a, e in zip(actual, expected))
        checks.append({"kind": "overall_center", "passed": passed, "actual": actual,
                       "expected": expected, "evidence": item["evidence"]})
        if not passed:
            issues.append("position_mismatch")

    operation_counts = _active_operation_counts(program_graph)
    active_nodes = output_dependency_subgraph(program_graph)["nodes"]
    active_by_id = {str(node.get("id")): node for node in active_nodes}
    operation_positions: dict[str, list[int]] = defaultdict(list)
    for index, node in enumerate(active_nodes):
        operation_positions[str(node.get("op") or "")].append(index)
    solid_boundary_ops = {
        "extrude", "revolve", "sweep", "loft", "twist_extrude",
        "union", "cut", "intersect", "box", "cylinder", "cone", "sphere", "torus", "wedge",
    }

    def _has_same_chain_order(upstream_op: str, downstream_op: str) -> bool:
        """True if downstream_op has upstream_op as an ancestor within the same feature branch."""
        for node in active_nodes:
            if str(node.get("op") or "") != downstream_op:
                continue
            frontier = [str(src) for src in (node.get("inputs") or []) if str(src) in active_by_id]
            visited: set[str] = set()
            while frontier:
                current_id = frontier.pop()
                if current_id in visited:
                    continue
                visited.add(current_id)
                current = active_by_id[current_id]
                current_op = str(current.get("op") or "")
                if current_op == upstream_op:
                    return True
                if current_op in solid_boundary_ops:
                    continue
                for src in current.get("inputs") or []:
                    src_id = str(src)
                    if src_id in active_by_id and src_id not in visited:
                        frontier.append(src_id)
        return False

    for item in facts.get("transform_orders") or []:
        first, second = item["operations"]
        first_positions = operation_positions.get(first, [])
        second_positions = operation_positions.get(second, [])
        has_forward = _has_same_chain_order(first, second)
        has_inverted = _has_same_chain_order(second, first)
        passed = bool(
            first_positions
            and second_positions
            and (
                (has_forward and not has_inverted)
                or (not has_inverted and max(first_positions) < min(second_positions))
            )
        )
        checks.append({
            "kind": "transform_order",
            "passed": passed,
            "expected": [first, second],
            "actual_positions": {
                first: first_positions, second: second_positions,
            },
            "evidence": item["evidence"],
        })
        if not passed:
            issues.append("transform_order_mismatch")
    observed_translations: list[list[float]] = []
    for node in active_nodes:
        if node.get("op") != "translate":
            continue
        literal_kwargs = node.get("literal_kwargs") or {}
        literal_args = node.get("literal_args") or []
        candidate = literal_kwargs.get("offset")
        if candidate is None:
            candidate = literal_args[0] if literal_args else None
        if isinstance(candidate, (list, tuple)) and len(candidate) == 3 and all(
            isinstance(value, (int, float)) for value in candidate
        ):
            observed_translations.append([float(value) for value in candidate])
    for expected_item in facts.get("translations") or []:
        expected = [float(value) for value in expected_item["values"]]
        identity = all(
            _close(0.0, wanted, config.position_relative_tolerance, config.position_absolute_tolerance)
            for wanted in expected
        )
        passed = identity or any(
            all(_close(actual, wanted, config.position_relative_tolerance, config.position_absolute_tolerance)
                for actual, wanted in zip(candidate, expected))
            for candidate in observed_translations
        )
        checks.append({
            "kind": "translation_vector", "passed": passed, "expected": expected,
            "actual": observed_translations, "evidence": expected_item["evidence"],
        })
        if not passed:
            issues.append("translation_mismatch")

    observed_euler: list[list[float]] = []
    observed_axis_angles: dict[int, list[float]] = {0: [], 1: [], 2: []}
    for node in active_nodes:
        if node.get("op") != "rotate":
            continue
        literal_kwargs = node.get("literal_kwargs") or {}
        transformed_rotate = literal_kwargs.get("rotate")
        if isinstance(transformed_rotate, (list, tuple)) and len(transformed_rotate) == 3 and all(
            isinstance(value, (int, float)) for value in transformed_rotate
        ):
            observed_euler.append([float(value) for value in transformed_rotate])
        literal_args = node.get("literal_args") or []
        if len(literal_args) < 3 or not isinstance(literal_args[2], (int, float)):
            continue
        start, end = literal_args[0], literal_args[1]
        if not (
            isinstance(start, (list, tuple)) and isinstance(end, (list, tuple))
            and len(start) == 3 and len(end) == 3
            and all(isinstance(value, (int, float)) for value in [*start, *end])
        ):
            continue
        direction = [float(end[index]) - float(start[index]) for index in range(3)]
        axis = max(range(3), key=lambda index: abs(direction[index]))
        if abs(direction[axis]) <= 1e-12 or any(
            abs(direction[index]) > 1e-9 for index in range(3) if index != axis
        ):
            continue
        signed_angle = float(literal_args[2]) * (1.0 if direction[axis] > 0 else -1.0)
        observed_axis_angles[axis].append(signed_angle)
    for expected_item in facts.get("euler_angles") or []:
        expected = [float(value) for value in expected_item["values"]]
        identity = all(abs(value) <= config.angle_absolute_tolerance for value in expected)
        tuple_match = any(
            all(abs(actual - wanted) <= config.angle_absolute_tolerance
                for actual, wanted in zip(candidate, expected))
            for candidate in observed_euler
        )
        axis_match = all(
            abs(wanted) <= config.angle_absolute_tolerance
            or any(abs(actual - wanted) <= config.angle_absolute_tolerance
                   for actual in observed_axis_angles[index])
            for index, wanted in enumerate(expected)
        )
        passed = identity or tuple_match or axis_match
        checks.append({
            "kind": "euler_angles", "passed": passed, "expected": expected,
            "observed_euler": observed_euler,
            "observed_axis_angles": observed_axis_angles,
            "evidence": expected_item["evidence"],
        })
        if not passed:
            issues.append("rotation_mismatch")
    cylinders = (metrics.get("topology") or {}).get("cylindrical_faces") or []
    hole_count = facts.get("hole_count")
    if hole_count is not None:
        structural_hole = sum(operation_counts[op] for op in ("hole", "counterbore_hole", "countersink_hole", "cut", "pocket"))
        sketch_hole = operation_counts["circle"] >= hole_count and operation_counts["extrude"] >= 1
        actual_cylinders = len(cylinders)
        passed = actual_cylinders >= hole_count and (structural_hole >= 1 or sketch_hole)
        checks.append({"kind": "hole_count", "passed": passed, "expected": hole_count,
                       "cylindrical_faces": actual_cylinders, "structural_hole_ops": structural_hole,
                       "sketch_hole_profile": sketch_hole})
        if not passed:
            issues.append("hole_count_mismatch")

    for expected_radius in facts.get("hole_radii") or []:
        radii = [float(item["radius"]) for item in cylinders if isinstance(item, dict) and item.get("radius") is not None]
        passed = any(_close(radius, expected_radius, config.radius_relative_tolerance,
                            config.dimension_absolute_tolerance) for radius in radii)
        checks.append({"kind": "hole_radius", "passed": passed, "expected": expected_radius, "actual": radii})
        if not passed:
            issues.append("hole_radius_mismatch")

    active = output_dependency_subgraph(program_graph)["nodes"]
    planes = {
        str((node.get("literal_args") or [None])[0]).upper()
        for node in active if node.get("op") == "workplane" and (node.get("literal_args") or [])
    }
    extrudes = [node for node in active if node.get("op") in {"extrude", "twist_extrude"}]
    expected_plane = {"x": "YZ", "y": "XZ", "z": "XY"}
    for direction in facts.get("extrude_directions") or []:
        signs = []
        for node in extrudes:
            argument = (node.get("literal_args") or [None])[0]
            if isinstance(argument, (int, float)):
                signs.append(-1 if float(argument) < 0 else 1)
        passed = expected_plane[direction["axis"]] in planes and (
            direction["sign"] in signs or not signs
        )
        checks.append({"kind": "extrude_direction", "passed": passed, "expected": direction,
                       "planes": sorted(planes), "signs": signs})
        if not passed:
            issues.append("extrude_direction_mismatch")

    for op, expected in facts.get("primitive_counts", {}).items():
        actual = operation_counts[op]
        if op == "box":
            equivalent = min(
                operation_counts["extrude"],
                operation_counts["rectangle"] + operation_counts["polygon"] + operation_counts["profile"] + operation_counts["line"],
            )
        elif op == "cylinder":
            equivalent = min(
                operation_counts["circle"],
                operation_counts["extrude"] + operation_counts["revolve"],
            )
        elif op == "sphere":
            equivalent = operation_counts["revolve"]
        else:
            equivalent = 0
        passed = actual + equivalent >= expected or operation_counts["pattern"] > 0
        checks.append({"kind": "primitive_count", "operation": op, "passed": passed,
                       "expected": expected, "actual": actual, "equivalent": equivalent})
        if not passed:
            issues.append(f"primitive_count_mismatch:{op}")
    for op, expected in facts.get("boolean_operations", {}).items():
        actual = operation_counts[op]
        passed = actual >= expected
        checks.append({"kind": "boolean_relation", "operation": op, "passed": passed,
                       "expected": expected, "actual": actual})
        if not passed:
            issues.append(f"boolean_relation_mismatch:{op}")
    return issues, checks


def _continuous_constraint_fidelity(
    checks: Sequence[dict], config: ConstraintConfig,
) -> float:
    """Continuously score observable text constraints without reference geometry.

    Binary pass/fail remains the acceptance rule. This score only orders two
    candidates that miss a stated numeric or count constraint, so the one
    closer to the value written in the requirement is preferred.
    """
    if not checks:
        return 1.0

    def vector_score(actual, expected, *, ordered: bool = True) -> float | None:
        if not (
            isinstance(actual, (list, tuple))
            and isinstance(expected, (list, tuple))
            and len(actual) == len(expected)
            and actual
        ):
            return None
        try:
            left = [float(value) for value in actual]
            right = [float(value) for value in expected]
        except (TypeError, ValueError):
            return None
        if not ordered:
            left, right = sorted(left), sorted(right)
        residuals = [
            abs(observed - wanted)
            / max(abs(wanted), config.dimension_absolute_tolerance, 1e-9)
            for observed, wanted in zip(left, right)
        ]
        return 1.0 / (1.0 + sum(residuals) / len(residuals))

    scores: list[float] = []
    for check in checks:
        kind = str(check.get("kind") or "")
        score: float | None = None
        if kind in {"dimensions", "overall_center"}:
            score = vector_score(
                check.get("actual"), check.get("expected"),
                ordered=bool(check.get("ordered", True)),
            )
        elif kind == "translation_vector":
            expected = check.get("expected")
            candidates = check.get("actual") or []
            candidate_scores = [
                value for candidate in candidates
                if (value := vector_score(candidate, expected)) is not None
            ]
            score = max(candidate_scores, default=1.0 if check.get("passed") else 0.0)
        elif kind == "hole_radius":
            try:
                wanted = float(check.get("expected"))
                residuals = [
                    abs(float(value) - wanted)
                    / max(abs(wanted), config.dimension_absolute_tolerance, 1e-9)
                    for value in (check.get("actual") or [])
                ]
                score = 1.0 / (1.0 + min(residuals)) if residuals else 0.0
            except (TypeError, ValueError):
                score = None
        elif kind in {"primitive_count", "boolean_relation"}:
            try:
                expected = max(float(check.get("expected") or 0.0), 1.0)
                observed = float(check.get("actual") or 0.0) + float(
                    check.get("equivalent") or 0.0
                )
                score = min(1.0, max(0.0, observed / expected))
            except (TypeError, ValueError):
                score = None
        if score is None:
            score = 1.0 if check.get("passed") else 0.0
        check["fidelity"] = min(1.0, max(0.0, float(score)))
        scores.append(check["fidelity"])
    return sum(scores) / len(scores)


def validate_requirement_constraints(
    code: str, execution: dict, requirement_graph: dict,
    config: ConstraintConfig | None = None, *, requirement_text: str = "",
    graph_build_issues: Sequence[str] = (), fallback_used: bool = False,
) -> dict:
    config = config or ConstraintConfig()
    issues: list[str] = []
    static = analyse_static(code)
    issues.extend(f"static:{value}" for value in static.fatal)
    try:
        program_graph = build_program_graph(code)
    except SyntaxError as error:
        program_graph = {"nodes": [], "output": None, "output_nodes": []}
        issues.append(f"program_graph:{error}")
    effective_graph = output_dependency_subgraph(program_graph)
    exact_gap = graph_difference(requirement_graph, program_graph)
    gap = asymmetric_graph_difference(requirement_graph, program_graph)
    facts = extract_requirement_facts(
        requirement_text, linear_unit_policy=config.linear_unit_policy,
    )
    if gap["M_op"] > config.max_operation_missing_rate:
        issues.append(f"missing_operations:{gap['M_op']:.6f}")
    if gap["M_dep"] > config.max_dependency_missing_rate:
        issues.append(f"missing_dependencies:{gap['M_dep']:.6f}")
    if config.reject_unknown_operations and gap["unknown_op_count"]:
        issues.append(f"unknown_operations:{gap['unknown_op_count']}")
    required_operations, _ = graph_counters(requirement_graph)
    observed_operations, _ = graph_counters(effective_graph)
    identity_spatial_operations = _identity_spatial_counts(program_graph, config)
    text_spatial_requirements = {
        "rotate": max(
            (
                sum(abs(float(value)) > config.angle_absolute_tolerance
                    for value in item.get("values", ()))
                for item in facts.get("euler_angles") or []
            ),
            default=0,
        ),
        "translate": sum(
            any(abs(float(value)) > config.position_absolute_tolerance
                for value in item.get("values", ()))
            for item in facts.get("translations") or []
        ),
    }
    for operation in ("rotate", "translate", "scale", "mirror", "pattern"):
        # Text facts supplement a requirement DAG that omitted an explicit
        # spatial operation; they never reduce an operation already required
        # by the DAG.  This prevents a correct rotation/translation from being
        # simultaneously accepted by the deterministic text check and marked
        # as an unexpected program operation.
        text_allowance = max(
            0, text_spatial_requirements.get(operation, 0) - required_operations[operation],
        )
        excess = (
            observed_operations[operation]
            - identity_spatial_operations[operation]
            - required_operations[operation]
            - text_allowance
        )
        if excess > 0:
            issues.append(f"unexpected_spatial_operation:{operation}:{excess}")
    if not program_graph.get("output") or not effective_graph.get("nodes"):
        issues.append("missing_program_output")
    if not execution.get("success"):
        issues.append(f"execution:{execution.get('error_type') or 'failed'}")
    metrics = execution.get("metrics") or {}
    if execution.get("success") and not metrics:
        issues.append("missing_geometry_metrics")
    if metrics:
        if metrics.get("is_valid") is not True:
            issues.append("invalid_topology")
        if int(metrics.get("solid_count", 0)) < 1:
            issues.append("no_solid")
        try:
            volume = float(metrics.get("volume", 0.0))
            if not math.isfinite(volume) or volume <= config.minimum_volume:
                issues.append("zero_volume")
        except (TypeError, ValueError):
            issues.append("invalid_volume")

    requirement_issues = validate_requirement_graph(requirement_graph)
    external_graph_issues = [str(issue) for issue in graph_build_issues]
    graph_trusted = not requirement_issues and not external_graph_issues
    nodes = [node for node in requirement_graph.get("nodes") or [] if isinstance(node, dict)]
    evidence_coverage = (
        sum(bool(str(node.get("span") or "").strip()) for node in nodes) / len(nodes)
        if nodes else 0.0
    )
    graph_confidence = (
        0.55 + 0.30 * evidence_coverage + 0.15 * bool(requirement_graph.get("output"))
        if graph_trusted else 0.25 * evidence_coverage
    )
    if fallback_used:
        graph_confidence = min(graph_confidence, 0.5)
    graph_confidence = min(1.0, max(0.0, graph_confidence))
    if config.require_trusted_requirement_graph and not graph_trusted:
        issues.append("untrusted_requirement_graph")
    if not requirement_graph.get("nodes"):
        issues.append("untrusted_empty_requirement_graph")

    natural_issues, checks = _check_natural_language_constraints(facts, program_graph, metrics, config)
    issues.extend(natural_issues)
    issues = list(dict.fromkeys(issues))
    constraint_pass_rate = (
        sum(bool(check.get("passed")) for check in checks) / len(checks) if checks else 1.0
    )
    constraint_fidelity = _continuous_constraint_fidelity(checks, config)
    structural_score = max(
        0.0,
        (1.0 - 0.6 * gap["M_op"] - 0.4 * gap["M_dep"])
        * (0.5 * constraint_pass_rate + 0.5 * constraint_fidelity),
    )
    return {
        "accepted": not issues,
        "issues": issues,
        "structural_score": structural_score,
        "constraint_pass_rate": constraint_pass_rate,
        "constraint_fidelity": constraint_fidelity,
        "constraint_checks": checks,
        "requirement_facts": facts,
        "graph_trusted": graph_trusted,
        "graph_confidence": graph_confidence,
        "graph_evidence_coverage": evidence_coverage,
        "fallback_used": fallback_used,
        "requirement_graph_issues": requirement_issues,
        "graph_build_issues": external_graph_issues,
        "gap": gap,
        "exact_gap": exact_gap,
        "program_graph": program_graph,
        "effective_program_graph": effective_graph,
    }


def validate_natural_language_constraints(
    code: str, execution: dict, config: ConstraintConfig | None = None, *,
    requirement_text: str = "",
) -> dict:
    """Validate a baseline without constructing or consuming a requirement DAG.

    The baseline still uses the same static, OCC-validity, and deterministic
    text-fact checks as the proposed methods.  Operation/dependency gaps,
    requirement-graph trust, and structural scores are deliberately absent.
    """
    config = config or ConstraintConfig()
    issues: list[str] = []
    static = analyse_static(code)
    issues.extend(f"static:{value}" for value in static.fatal)
    try:
        program_graph = build_program_graph(code)
    except SyntaxError as error:
        program_graph = {"nodes": [], "output": None, "output_nodes": []}
        issues.append(f"program_graph:{error}")
    effective_graph = output_dependency_subgraph(program_graph)
    if not program_graph.get("output") or not effective_graph.get("nodes"):
        issues.append("missing_program_output")
    if not execution.get("success"):
        issues.append(f"execution:{execution.get('error_type') or 'failed'}")
    metrics = execution.get("metrics") or {}
    if execution.get("success") and not metrics:
        issues.append("missing_geometry_metrics")
    if metrics:
        if metrics.get("is_valid") is not True:
            issues.append("invalid_topology")
        if int(metrics.get("solid_count", 0)) < 1:
            issues.append("no_solid")
        try:
            volume = float(metrics.get("volume", 0.0))
            if not math.isfinite(volume) or volume <= config.minimum_volume:
                issues.append("zero_volume")
        except (TypeError, ValueError):
            issues.append("invalid_volume")

    facts = extract_requirement_facts(
        requirement_text, linear_unit_policy=config.linear_unit_policy,
    )
    natural_issues, checks = _check_natural_language_constraints(
        facts, program_graph, metrics, config,
    )
    issues.extend(natural_issues)
    issues = list(dict.fromkeys(issues))
    constraint_pass_rate = (
        sum(bool(check.get("passed")) for check in checks) / len(checks)
        if checks else 1.0
    )
    constraint_fidelity = _continuous_constraint_fidelity(checks, config)
    return {
        "accepted": not issues,
        "issues": issues,
        "constraint_pass_rate": constraint_pass_rate,
        "constraint_fidelity": constraint_fidelity,
        "constraint_checks": checks,
        "requirement_facts": facts,
        "dag_used": False,
        "program_graph": program_graph,
        "effective_program_graph": effective_graph,
    }
