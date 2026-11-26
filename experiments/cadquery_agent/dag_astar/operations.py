"""Shared, extensible CadQuery operation vocabulary and port semantics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OperationSpec:
    name: str
    min_inputs: int = 0
    max_inputs: int | None = 1
    roles: tuple[str, ...] = ()
    category: str = "feature"
    semantic: bool = True
    transparent: bool = False


def _spec(name: str, minimum: int = 0, maximum: int | None = 1, *,
          roles: tuple[str, ...] = (), category: str = "feature",
          semantic: bool = True, transparent: bool = False) -> OperationSpec:
    return OperationSpec(name, minimum, maximum, roles, category, semantic, transparent)


DEFAULT_OPERATION_SPECS: dict[str, OperationSpec] = {
    # Context and selectors preserve data flow but are not requirement features.
    "workplane": _spec("workplane", 0, 1, category="context", semantic=False, transparent=True),
    "select_faces": _spec("select_faces", 1, 1, category="selector", semantic=False, transparent=True),
    "select_edges": _spec("select_edges", 1, 1, category="selector", semantic=False, transparent=True),
    "select_vertices": _spec("select_vertices", 1, 1, category="selector", semantic=False, transparent=True),
    "select_wires": _spec("select_wires", 1, 1, category="selector", semantic=False, transparent=True),
    "select_solids": _spec("select_solids", 1, 1, category="selector", semantic=False, transparent=True),
    "select_shells": _spec("select_shells", 1, 1, category="selector", semantic=False, transparent=True),
    "selection": _spec("selection", 1, 1, category="selector", semantic=False, transparent=True),
    "selector": _spec("selector", 0, None, category="selector", semantic=False, transparent=True),
    "vector": _spec("vector", 0, None, category="context", semantic=False, transparent=True),
    "plane": _spec("plane", 0, None, category="context", semantic=False, transparent=True),
    "sketch_context": _spec(
        "sketch_context", 0, None, category="context", semantic=False, transparent=True,
    ),
    "place_sketch": _spec(
        "place_sketch", 1, None, category="context", semantic=False, transparent=True,
    ),
    "sketch_constraint": _spec(
        "sketch_constraint", 1, None, category="context", semantic=False, transparent=True,
    ),
    "sketch_position": _spec(
        "sketch_position", 0, 1, category="context", semantic=False, transparent=True,
    ),
    # Primitives and sketches.
    "box": _spec("box", 0, 1, category="primitive"),
    "cylinder": _spec("cylinder", 0, 1, category="primitive"),
    "cone": _spec("cone", 0, 1, category="primitive"),
    "sphere": _spec("sphere", 0, 1, category="primitive"),
    "torus": _spec("torus", 0, 1, category="primitive"),
    "wedge": _spec("wedge", 0, 1, category="primitive"),
    "profile": _spec("profile", 0, 1, category="sketch"),
    "circle": _spec("circle", 0, 1, category="sketch"),
    "ellipse": _spec("ellipse", 0, 1, category="sketch"),
    "line": _spec("line", 0, 1, category="sketch"),
    "arc": _spec("arc", 0, 1, category="sketch"),
    "spline": _spec("spline", 0, 1, category="sketch"),
    "slot": _spec("slot", 0, 1, category="sketch"),
    "close": _spec("close", 1, 1, category="sketch"),
    # Solid features.
    # A single feature may consume several closed contours (outer wire plus
    # holes); the AST program still commonly represents them as one Workplane.
    "extrude": _spec("extrude", 1, None),
    "revolve": _spec("revolve", 1, 1),
    "sweep": _spec("sweep", 1, 2),
    "loft": _spec("loft", 1, None),
    "twist_extrude": _spec("twist_extrude", 1, 1),
    "fillet": _spec("fillet", 1, 1),
    "chamfer": _spec("chamfer", 1, 1),
    "shell": _spec("shell", 1, 1),
    "hole": _spec("hole", 1, 1),
    "counterbore_hole": _spec("counterbore_hole", 1, 1),
    "countersink_hole": _spec("countersink_hole", 1, 1),
    "pocket": _spec("pocket", 1, 1),
    # Explicit Boolean ports are ordered base/tool.
    "cut": _spec("cut", 2, 2, roles=("base", "tool"), category="boolean"),
    "union": _spec("union", 2, None, roles=("base", "tool"), category="boolean"),
    "intersect": _spec("intersect", 2, 2, roles=("base", "tool"), category="boolean"),
    # Transforms and repetitions are semantic and must never be removed.
    "translate": _spec("translate", 1, 1, category="transform"),
    "rotate": _spec("rotate", 1, 1, category="transform"),
    "mirror": _spec("mirror", 1, 1, category="transform"),
    "scale": _spec("scale", 1, 1, category="transform"),
    "pattern": _spec("pattern", 1, 1, category="transform"),
}


_ALIASES = {
    "workplane": "workplane", "workplanefromtagged": "workplane", "transformed": "workplane",
    "faces": "select_faces", "edges": "select_edges", "vertices": "select_vertices",
    "wires": "select_wires", "solids": "select_solids", "shells": "select_shells",
    "all": "selection", "first": "selection", "last": "selection", "item": "selection",
    "filter": "selection", "tag": "selection", "end": "selection",
    "val": "selection", "vals": "selection", "clean": "selection", "compounds": "selection",
    "nearesttopointselector": "selector", "boxselector": "selector",
    "radiusnthselector": "selector", "directionselector": "selector",
    "directionminmaxselector": "selector", "typeselector": "selector",
    "stringsyntaxselector": "selector", "andselector": "selector",
    "orselector": "selector", "inverseselector": "selector", "sumselector": "selector",
    "vector": "vector", "plane": "plane",
    "sketch": "sketch_context", "placesketch": "place_sketch",
    "assemble": "sketch_constraint", "constrain": "sketch_constraint",
    "box": "box", "makebox": "box", "cylinder": "cylinder", "makecylinder": "cylinder",
    "cone": "cone", "makecone": "cone", "sphere": "sphere", "makesphere": "sphere",
    "torus": "torus", "maketorus": "torus", "wedge": "wedge", "makewedge": "wedge",
    "rect": "profile", "polygon": "profile", "polyline": "profile",
    "moveto": "sketch_position", "move": "sketch_position", "center": "sketch_position",
    "pushpoints": "sketch_position",
    "circle": "circle", "ellipse": "ellipse", "line": "line", "lineto": "line",
    "segment": "line",
    "hline": "line", "vline": "line", "hlineto": "line", "vlineto": "line",
    "threepointarc": "arc", "sagittaarc": "arc", "radiusarc": "arc", "tangentarcpoint": "arc",
    "spline": "spline", "parametricspline": "spline", "slot2d": "slot", "close": "close",
    "extrude": "extrude", "twistextrude": "twist_extrude", "revolve": "revolve",
    "sweep": "sweep", "loft": "loft", "fillet": "fillet", "chamfer": "chamfer",
    "shell": "shell", "hole": "hole", "cborehole": "counterbore_hole",
    "cskhole": "countersink_hole", "cutblind": "pocket", "cutthruall": "pocket",
    "cut": "cut", "subtract": "cut", "union": "union", "combine": "union",
    "add": "union", "fuse": "union", "makecompound": "union",
    "intersect": "intersect", "common": "intersect",
    "translate": "translate", "rotate": "rotate", "rotateaboutcenter": "rotate",
    "mirror": "mirror", "mirrorx": "mirror", "mirrory": "mirror", "scale": "scale",
    "rarray": "pattern", "polararray": "pattern", "eachpoint": "pattern",
}


def canonical_operation(name: str) -> str:
    final = str(name).lower().split(".")[-1]
    if final in DEFAULT_OPERATION_SPECS:
        return final
    compact = final.replace("_", "")
    return _ALIASES.get(compact, "unknown_op")


def operation_spec(name: str, registry: dict[str, OperationSpec] | None = None) -> OperationSpec | None:
    return (registry or DEFAULT_OPERATION_SPECS).get(canonical_operation(name))


def is_semantic_operation(name: str, registry: dict[str, OperationSpec] | None = None) -> bool:
    spec = operation_spec(name, registry)
    return bool(spec and spec.semantic)


def is_transparent_operation(name: str, registry: dict[str, OperationSpec] | None = None) -> bool:
    spec = operation_spec(name, registry)
    return bool(spec and spec.transparent)


def operation_catalog(registry: dict[str, OperationSpec] | None = None) -> list[dict]:
    return [
        {"op": spec.name, "min_inputs": spec.min_inputs, "max_inputs": spec.max_inputs,
         "roles": list(spec.roles), "category": spec.category}
        for spec in (registry or DEFAULT_OPERATION_SPECS).values() if spec.semantic
    ]
