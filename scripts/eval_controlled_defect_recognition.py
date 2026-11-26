"""Controlled Executable CAD Defect Recognition & Operation-Matched Attribution Benchmark.

Rigorous protocol:
1. Starts from verified IoU>=0.99 Boolean+extrude CAD-Coder test programs.
2. Injects 4 controlled single-defect perturbations (omitted Boolean feature,
   Boolean polarity flip .cut <-> .union, 1.35x extrusion dimension drift,
   and negated extrusion axis flip) while recording the exact injected operation.
3. Strictly verifies EVERY variant in OpenCascade:
   - Compiles cleanly with positive volume and real OCCT B-rep validity (`shape.isValid()`).
   - Every injected defect is verified to actually alter the exported 3D solid
     (volume or bounding-box coordinates differ from the clean solid by > 1e-5).
4. Evaluates 5 cleanly separated diagnostic layers, strictly isolating the
   dual-confidence threshold (tau=0.0 vs. tau=0.85) under the IDENTICAL
   `actionable` gate rule:
   - AST Static Rules Only
   - Kernel + Independent NL Constraint Validator Only
   - Raw 3D-Anchor Alignment (no actionable filter, tau=0.0)
   - Actionable 3D-Anchor Alignment (same actionable rule, tau=0.0)
   - DependCAD Gated Diagnostics (same actionable rule, tau=0.85)
5. Reports Clean FPR, per-class recall, overall Precision/Recall/F1, and
   exact operation/type-matched diagnostic hit rate on the 3 structural/parametric
   defect classes (explicit denominator 3 * N_base) as well as all 4 classes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import cadquery as cq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.cadquery_agent.dag_astar.constraints import (
    validate_natural_language_constraints,
    validate_requirement_constraints,
)
from experiments.cadquery_agent.dag_astar.graphs import canonicalize_requirement_graph
from experiments.cadquery_agent.dag_astar.program_graph import analyse_static
from experiments.cadquery_agent.dag_astar.search import HeuristicConfig, heuristic

DEFAULT_TASK_DIRS = [
    REPO_ROOT / "artifacts" / "dependcad_full1053" / "cad-coder" / "tasks",
]
DEFAULT_OUT_JSON = REPO_ROOT / "artifacts" / "controlled_defect_recognition.json"


def exec_cadquery_metrics(code: str) -> dict[str, Any]:
    """Execute CadQuery code in-process with real OpenCascade B-rep validity check."""
    ns: dict[str, Any] = {"cq": cq}
    try:
        exec(code, ns, ns)
        res = ns.get("result")
        if res is None:
            return {"success": False, "error": "missing result"}
        shape = res.val() if hasattr(res, "val") else res
        solids = shape.Solids() if hasattr(shape, "Solids") else [shape]
        if not solids:
            return {"success": False, "error": "empty solid"}
        is_valid = bool(shape.isValid()) if hasattr(shape, "isValid") else False
        if not is_valid:
            return {"success": False, "error": "invalid B-rep topology"}
        bb = shape.BoundingBox()
        vol = float(shape.Volume())
        if vol <= 1e-9:
            return {"success": False, "error": "zero volume"}
        return {
            "success": True,
            "returncode": 0,
            "error": "",
            "error_type": None,
            "metrics": {
                "is_valid": is_valid,
                "shape_count": 1,
                "solid_count": len(solids),
                "volume": vol,
                "bbox": {
                    "min": [float(bb.xmin), float(bb.ymin), float(bb.zmin)],
                    "max": [float(bb.xmax), float(bb.ymax), float(bb.zmax)],
                    "size": [float(bb.xlen), float(bb.ylen), float(bb.zlen)],
                },
            },
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def geometry_changed(clean_exec: dict[str, Any], mut_exec: dict[str, Any], require_bbox_shift: bool = False) -> bool:
    """Verify the mutated solid differs geometrically from the clean solid."""
    cm = clean_exec["metrics"]
    mm = mut_exec["metrics"]
    vol_diff = abs(mm["volume"] - cm["volume"])
    c_bb = cm["bbox"]["min"] + cm["bbox"]["max"]
    m_bb = mm["bbox"]["min"] + mm["bbox"]["max"]
    bb_diff = max(abs(a - b) for a, b in zip(c_bb, m_bb))
    if require_bbox_shift:
        return bb_diff > 1e-5
    return (vol_diff > 1e-6) or (bb_diff > 1e-5)


def mutate_omit_boolean(code: str) -> tuple[str, str] | None:
    """Drop the last .cut(...) or .union(...) call on an assignment; return (code, dropped_op)."""
    lines = code.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        m = re.match(r'^(\s*\w+\s*=\s*)(\w+)\.(cut|union)\(([^()]+)\)\s*$', lines[i])
        if m:
            new_lines = lines.copy()
            new_lines[i] = f"{m.group(1)}{m.group(2)}"
            return "\n".join(new_lines), m.group(3)
    return None


def mutate_flip_boolean(code: str) -> tuple[str, str] | None:
    """Swap the last .cut(...) <-> .union(...); return (code, original_op)."""
    lines = code.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        m = re.match(r'^(\s*\w+\s*=\s*\w+\.)(cut|union)(\([^()]+\)\s*)$', lines[i])
        if m:
            orig_op = m.group(2)
            swapped = "union" if orig_op == "cut" else "cut"
            new_lines = lines.copy()
            new_lines[i] = f"{m.group(1)}{swapped}{m.group(3)}"
            return "\n".join(new_lines), orig_op
    return None


def mutate_dimension_drift(code: str) -> tuple[str, str] | None:
    """Perturb the first numeric .extrude(val) by 1.35x; return (code, 'extrude')."""
    m = re.search(r'(\.extrude\(\s*)(-?\d+(?:\.\d+)?)(\s*\))', code)
    if not m:
        return None
    val = float(m.group(2))
    if abs(val) < 1e-6:
        return None
    new_val = round(val * 1.35, 4)
    return code[:m.start()] + f"{m.group(1)}{new_val}{m.group(3)}" + code[m.end():], "extrude"


def mutate_orientation_flip(code: str) -> tuple[str, str] | None:
    """Negate the first numeric .extrude(val) direction; return (code, 'extrude_sign')."""
    m = re.search(r'(\.extrude\(\s*)(-?\d+(?:\.\d+)?)(\s*\))', code)
    if not m:
        return None
    val = float(m.group(2))
    if abs(val) < 1e-6:
        return None
    new_val = -val
    return code[:m.start()] + f"{m.group(1)}{new_val}{m.group(3)}" + code[m.end():], "extrude_sign"


def evaluate_detectors(
    code: str,
    execution: dict[str, Any],
    req_graph: dict[str, Any],
    req_text: str,
    rgb_issues: list[str],
    rgb_fallback: bool,
    gt_class: str,
    injected_op: str,
) -> tuple[dict[str, dict[str, bool]], dict[str, Any]]:
    """Run all 5 cleanly separated diagnostic layers using real graph_confidence & mapping_confidence."""
    # 1. Static AST rules only
    st = analyse_static(code)
    static_flag = bool(st.fatal or st.countable)

    # 2. Kernel + independent NL constraint validator only (no 3D-anchor alignment)
    acc = validate_natural_language_constraints(code, execution, requirement_text=req_text)
    const_issues = [str(x).lower() for x in (acc.get("issues") or [])]
    const_flag = (not execution.get("success", False)) or bool(const_issues)

    # Compute the REAL requirement graph confidence (c_r) from req_graph + build metadata
    full_acc = validate_requirement_constraints(
        code, execution, req_graph,
        requirement_text=req_text,
        graph_build_issues=rgb_issues,
        fallback_used=rgb_fallback,
    )
    real_gc = float(full_acc.get("graph_confidence", 1.0))
    acc_with_gc = dict(acc)
    acc_with_gc["graph_confidence"] = real_gc

    # Helper to extract structural signals from heuristic() details
    def parse_struct(det: dict[str, Any], require_actionable: bool, require_conf: bool) -> dict[str, Any]:
        ag = det.get("alignment_gate") or {}
        actionable_ok = bool(ag.get("actionable", False)) if require_actionable else True
        conf_ok = bool(det.get("confidence_gate", False)) if require_conf else True
        miss_ops_raw = [str(x).lower() for x in (det.get("missing_operations") or [])]
        miss_deps_raw = [str(x).lower() for x in (det.get("missing_dependencies") or [])]
        has_ops = bool(miss_ops_raw) or (det.get("M_op_weighted", 0.0) > 0.0)
        has_role = det.get("M_role", 0.0) > 0.0
        has_dep = bool(miss_deps_raw) or (det.get("M_dep", 0.0) > 0.0) or (det.get("M_output", 0.0) > 0.0)
        active = actionable_ok and conf_ok and (has_ops or has_role or has_dep)
        return {
            "active": active,
            "has_ops": active and has_ops,
            "has_role": active and has_role,
            "has_dep": active and has_dep,
            "miss_ops_str": " ".join(miss_ops_raw),
            "miss_deps_str": " ".join(miss_deps_raw),
        }

    # Run heuristic at tau=0.0 and tau=0.85 with REAL graph_confidence (c_r) and mapping_confidence (c_m)
    _, det_tau0 = heuristic(
        code, execution, req_graph,
        config=HeuristicConfig(minimum_graph_confidence=0.0, minimum_mapping_confidence=0.0),
        acceptance=acc_with_gc,
    )
    _, det_tau85 = heuristic(
        code, execution, req_graph,
        config=HeuristicConfig(minimum_graph_confidence=0.85, minimum_mapping_confidence=0.85),
        acceptance=acc_with_gc,
    )
    conf_meta = {
        "graph_confidence": float(det_tau85.get("graph_confidence", real_gc)),
        "mapping_confidence": float(det_tau85.get("mapping_confidence", 0.0)),
        "actionable": bool((det_tau85.get("alignment_gate") or {}).get("actionable", False)),
        "blocked_by_gc": float(det_tau85.get("graph_confidence", real_gc)) < 0.85,
        "blocked_by_mc": float(det_tau85.get("mapping_confidence", 0.0)) < 0.85,
    }

    s_raw = parse_struct(det_tau0, require_actionable=False, require_conf=False)
    s_act_tau0 = parse_struct(det_tau0, require_actionable=True, require_conf=False)
    s_act_tau85 = parse_struct(det_tau85, require_actionable=True, require_conf=True)

    # Exact operation/category-matched diagnostic attribution check
    def op_matched(flag: bool, s_info: dict[str, Any] | None) -> bool:
        if not flag:
            return False
        s_ops = (s_info["miss_ops_str"] if s_info and s_info["active"] else "")
        s_deps = (s_info["miss_deps_str"] if s_info and s_info["active"] else "")
        s_role = bool(s_info and s_info["has_role"])
        if gt_class == "omit_feature":
            # Must specifically report the exact dropped Boolean operation ('cut' or 'union')
            return (
                (injected_op in s_ops)
                or (injected_op in s_deps)
                or any(injected_op in x for x in const_issues)
            )
        if gt_class == "flip_boolean":
            # Must specifically report a role mismatch OR the exact missing original Boolean op ('cut'/'union')
            return (
                s_role
                or (injected_op in s_ops)
                or (injected_op in s_deps)
                or any(("boolean" in x or injected_op in x) for x in const_issues)
            )
        if gt_class == "dim_drift":
            # Must specifically report a dimensional/extent/bbox/extrude mismatch
            return any(
                k in x
                for x in const_issues
                for k in ("dimension", "extent", "extrude", "height", "length", "bbox", "size", "thickness", "width", "depth")
            )
        if gt_class == "orient_flip":
            # Must specifically report an orientation/rotation/direction/axis mismatch
            return any(
                k in x
                for x in const_issues
                for k in ("orient", "rotat", "euler", "axis", "direction", "normal", "sign")
            )
        return False

    return {
        "static_only": {"detected": static_flag, "matched": False},
        "kernel_const": {"detected": const_flag, "matched": op_matched(const_flag, None)},
        "raw_tau0": {
            "detected": const_flag or s_raw["active"],
            "matched": op_matched(const_flag or s_raw["active"], s_raw),
        },
        "actionable_tau0": {
            "detected": const_flag or s_act_tau0["active"],
            "matched": op_matched(const_flag or s_act_tau0["active"], s_act_tau0),
        },
        "gated_tau85": {
            "detected": const_flag or s_act_tau85["active"],
            "matched": op_matched(const_flag or s_act_tau85["active"], s_act_tau85),
        },
    }, conf_meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled Executable CAD Defect Recognition Benchmark")
    parser.add_argument(
        "--task-dirs",
        "--v62-dirs",
        dest="v62_dirs",
        nargs="+",
        type=Path,
        default=DEFAULT_TASK_DIRS,
        help="Directories containing per-task DependCAD JSON outputs (e.g., artifacts/dependcad_full1053/cad-coder/tasks)",
    )
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON, help="Output summary JSON path")
    args = parser.parse_args()

    candidates = []
    for d in args.v62_dirs:
        for p in sorted(d.glob("cadcoder_*.json")):
            data = json.loads(p.read_text(encoding="utf-8"))
            a0 = data.get("conditions", {}).get("A0_full", {})
            if (
                a0.get("execution_success")
                and (a0.get("volumetric_iou") or 0.0) >= 0.99
                and a0.get("code")
                and data.get("requirement_graph")
            ):
                code = a0["code"]
                if (".cut(" in code or ".union(" in code) and ".extrude(" in code:
                    rgb = data.get("requirement_graph_build") or {}
                    candidates.append((
                        data["case_id"],
                        data["text"],
                        data["requirement_graph"],
                        list(rgb.get("issues") or []),
                        bool(rgb.get("fallback_used", False)),
                        code,
                    ))

    print(f"Found {len(candidates)} verified IoU>=0.99 Boolean+extrude candidates.")
    selected = []
    rejected_geom = 0
    for cid, text, rg_raw, rgb_issues, rgb_fallback, code in candidates:
        m_omit = mutate_omit_boolean(code)
        m_flip = mutate_flip_boolean(code)
        m_dim = mutate_dimension_drift(code)
        m_ori = mutate_orientation_flip(code)
        if not (m_omit and m_flip and m_dim and m_ori):
            continue
        c_omit, op_omit = m_omit
        c_flip, op_flip = m_flip
        c_dim, op_dim = m_dim
        c_ori, op_ori = m_ori

        ex_clean = exec_cadquery_metrics(code)
        if not ex_clean["success"]:
            continue
        ex_omit = exec_cadquery_metrics(c_omit)
        ex_flip = exec_cadquery_metrics(c_flip)
        ex_dim = exec_cadquery_metrics(c_dim)
        ex_ori = exec_cadquery_metrics(c_ori)
        if not (ex_omit["success"] and ex_flip["success"] and ex_dim["success"] and ex_ori["success"]):
            continue

        # Strict geometric-change verification: every injected defect MUST alter the 3D solid!
        if not (
            geometry_changed(ex_clean, ex_omit)
            and geometry_changed(ex_clean, ex_flip)
            and geometry_changed(ex_clean, ex_dim)
            and geometry_changed(ex_clean, ex_ori, require_bbox_shift=True)
        ):
            rejected_geom += 1
            continue

        rg = canonicalize_requirement_graph(rg_raw)
        selected.append((cid, text, rg, rgb_issues, rgb_fallback, [
            ("clean", "none", code, ex_clean),
            ("omit_feature", op_omit, c_omit, ex_omit),
            ("flip_boolean", op_flip, c_flip, ex_flip),
            ("dim_drift", op_dim, c_dim, ex_dim),
            ("orient_flip", op_ori, c_ori, ex_ori),
        ]))

    n_base = len(selected)
    print(f"Verified {n_base} base cases x 5 valid+geom-changed variants = {n_base * 5} solids (excluded {rejected_geom} base cases by geometric-change filter).")

    detectors = ["static_only", "kernel_const", "raw_tau0", "actionable_tau0", "gated_tau85"]
    classes = ["clean", "omit_feature", "flip_boolean", "dim_drift", "orient_flip"]
    stats = {det: {c: {"n": 0, "det": 0, "matched": 0} for c in classes} for det in detectors}
    gate_audit = {"total": 0, "actionable": 0, "blocked_gc_lt_85": 0, "blocked_mc_lt_85": 0, "blocked_either_on_actionable": 0, "min_gc": 1.0, "min_mc": 1.0}

    for cid, text, rg, rgb_issues, rgb_fallback, variants in selected:
        for cls_name, inj_op, v_code, v_exec in variants:
            res, cmeta = evaluate_detectors(v_code, v_exec, rg, text, rgb_issues, rgb_fallback, cls_name, inj_op)
            gate_audit["total"] += 1
            gate_audit["min_gc"] = min(gate_audit["min_gc"], cmeta["graph_confidence"])
            gate_audit["min_mc"] = min(gate_audit["min_mc"], cmeta["mapping_confidence"])
            if cmeta["actionable"]:
                gate_audit["actionable"] += 1
                if cmeta["blocked_by_gc"]:
                    gate_audit["blocked_gc_lt_85"] += 1
                if cmeta["blocked_by_mc"]:
                    gate_audit["blocked_mc_lt_85"] += 1
                if cmeta["blocked_by_gc"] or cmeta["blocked_by_mc"]:
                    gate_audit["blocked_either_on_actionable"] += 1
            for det in detectors:
                stats[det][cls_name]["n"] += 1
                if res[det]["detected"]:
                    stats[det][cls_name]["det"] += 1
                if res[det]["matched"]:
                    stats[det][cls_name]["matched"] += 1

    print("Dual-confidence gate audit:", gate_audit)
    summary: dict[str, Any] = {
        "n_base_cases": n_base,
        "total_executable_programs": n_base * 5,
        "excluded_base_cases_geom_filter": rejected_geom,
        "n_struct3_defects": n_base * 3,
        "n_all4_defects": n_base * 4,
        "dual_confidence_audit": gate_audit,
        "detectors": {},
    }
    print("\n=== RIGOROUS CONTROLLED EXECUTABLE DEFECT RECOGNITION BENCHMARK ===")
    for det in detectors:
        st = stats[det]
        fp = st["clean"]["det"]
        n_clean = st["clean"]["n"]
        fpr = fp / max(n_clean, 1)

        tp_all = sum(st[c]["det"] for c in ["omit_feature", "flip_boolean", "dim_drift", "orient_flip"])
        n_pos_all = sum(st[c]["n"] for c in ["omit_feature", "flip_boolean", "dim_drift", "orient_flip"])
        m_struct3 = sum(st[c]["matched"] for c in ["omit_feature", "flip_boolean", "dim_drift"])
        n_struct3 = sum(st[c]["n"] for c in ["omit_feature", "flip_boolean", "dim_drift"])
        m_all4 = sum(st[c]["matched"] for c in ["omit_feature", "flip_boolean", "dim_drift", "orient_flip"])

        prec_all = tp_all / max(tp_all + fp, 1)
        rec_all = tp_all / max(n_pos_all, 1)
        f1_all = 2 * prec_all * rec_all / max(prec_all + rec_all, 1e-9)

        rec_by_cls = {c: st[c]["det"] / max(st[c]["n"], 1) for c in classes if c != "clean"}
        match_by_cls = {c: st[c]["matched"] / max(st[c]["n"], 1) for c in classes if c != "clean"}

        summary["detectors"][det] = {
            "fpr": fpr,
            "precision_all": prec_all,
            "recall_all": rec_all,
            "f1_all": f1_all,
            "op_match_struct3": m_struct3 / max(n_struct3, 1),
            "op_match_all4": m_all4 / max(n_pos_all, 1),
            "recall_by_class": rec_by_cls,
            "match_by_class": match_by_cls,
        }
        print(
            f"{det:16s} | FPR={fpr*100:5.1f}% | "
            f"Omit={rec_by_cls['omit_feature']*100:5.1f}% | "
            f"FlipBool={rec_by_cls['flip_boolean']*100:5.1f}% | "
            f"DimDrift={rec_by_cls['dim_drift']*100:5.1f}% | "
            f"Orient={rec_by_cls['orient_flip']*100:5.1f}% | "
            f"Prec={prec_all*100:5.1f}% F1={f1_all*100:5.1f}% | "
            f"OpMatch3={m_struct3/max(n_struct3,1)*100:5.1f}% OpMatch4={m_all4/max(n_pos_all,1)*100:5.1f}%"
        )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Saved:", args.out_json)


if __name__ == "__main__":
    main()
