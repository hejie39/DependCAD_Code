"""Build a frozen, leakage-filtered requirement-DAG case library from JSONL records."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

from .graphs import RequirementGraphBuilder
from .llm import PolicyRoutedClient


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*|[\u4e00-\u9fff]", text.lower()))


def _near_duplicate(left: str, right: str, threshold: float) -> bool:
    a, b = _tokens(left), _tokens(right)
    return len(a & b) / max(len(a | b), 1) >= threshold


def _identity(row: dict, *names: str) -> str:
    for name in names:
        value = str(row.get(name) or "").strip()
        if value:
            return value
    return ""


def _atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        Path(name).replace(path)
    finally:
        Path(name).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.92)
    parser.add_argument(
        "--allowed-split", action="append",
        default=["train", "development", "dev", "history", "synthetic"],
    )
    args = parser.parse_args()
    if not os.getenv("QWEN_API_KEY", "").strip():
        raise RuntimeError("QWEN_API_KEY is required; keys are never stored in source code")
    evaluation = _read_jsonl(args.evaluation_manifest)
    if not evaluation:
        raise ValueError("evaluation_manifest_must_not_be_empty")
    for index, row in enumerate(evaluation):
        if not _identity(row, "case_id", "task_id", "id"):
            raise ValueError(f"evaluation_manifest_missing_case_id:{index}")
        if not _identity(row, "model_id", "source_model_id"):
            raise ValueError(f"evaluation_manifest_missing_model_id:{index}")
        if not str(row.get("text") or row.get("requirement") or "").strip():
            raise ValueError(f"evaluation_manifest_missing_text:{index}")
    evaluation_ids = {_identity(row, "case_id", "task_id", "id") for row in evaluation}
    evaluation_model_ids = {_identity(row, "model_id", "source_model_id") for row in evaluation}
    evaluation_geometry_hashes = {
        _identity(row, "geometry_hash", "shape_hash", "reference_hash") for row in evaluation
    }
    evaluation_ids.discard("")
    evaluation_model_ids.discard("")
    evaluation_geometry_hashes.discard("")
    evaluation_texts = [str(row.get("text") or row.get("requirement") or "") for row in evaluation]
    builder = RequirementGraphBuilder(PolicyRoutedClient(), max_retries=2)
    output: list[dict] = []
    accepted_texts: list[str] = []
    accepted_models: set[str] = set()
    accepted_geometry: set[str] = set()
    accepted_ids: set[str] = set()
    skipped: dict[str, int] = {}

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for row in _read_jsonl(args.input):
        forbidden = {
            "reference_stl", "reference_step", "target_stl", "target_step",
            "geometry_metrics", "volumetric_iou", "surface_f1", "chamfer_distance",
            "strict_success", "strict_match",
        } & set(row)
        if forbidden:
            skip("contains_evaluation_data")
            continue
        case_id = _identity(row, "case_id", "task_id", "id")
        model_id = _identity(row, "model_id", "source_model_id")
        geometry_hash = _identity(row, "geometry_hash", "shape_hash")
        text = str(row.get("text") or row.get("requirement") or "")
        split = str(row.get("split") or "").strip().lower()
        if split not in {value.lower() for value in args.allowed_split}:
            skip("disallowed_or_missing_split")
            continue
        if not case_id or not model_id or not text or not bool(row.get("execution_success", False)):
            skip("incomplete_or_unsuccessful")
            continue
        if (
            case_id in evaluation_ids or model_id in evaluation_model_ids
            or (geometry_hash and geometry_hash in evaluation_geometry_hashes)
            or case_id in accepted_ids or (model_id and model_id in accepted_models)
            or (geometry_hash and geometry_hash in accepted_geometry)
        ):
            skip("identity_or_geometry_leakage")
            continue
        if any(
            candidate and _near_duplicate(text, candidate, args.near_duplicate_threshold)
            for candidate in [*evaluation_texts, *accepted_texts]
        ):
            skip("near_duplicate")
            continue
        built = builder.build(text)
        if built.fallback_used or built.issues:
            skip("untrusted_requirement_graph")
            continue
        output.append({
            "case_id": case_id, "text": text, "graph": built.graph,
            "execution_success": True, "source_split": split,
            "model_id": model_id, "geometry_hash": geometry_hash,
            "graph_fallback_used": False,
            "source_record_sha256": hashlib.sha256(json.dumps(
                row, ensure_ascii=False, sort_keys=True,
            ).encode("utf-8")).hexdigest(),
        })
        accepted_ids.add(case_id)
        accepted_texts.append(text)
        if model_id:
            accepted_models.add(model_id)
        if geometry_hash:
            accepted_geometry.add(geometry_hash)
    library_version = hashlib.sha256(json.dumps(
        output, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    for item in output:
        item["library_version"] = library_version
    _atomic_jsonl(args.output, output)
    print(json.dumps({
        "output": str(args.output.resolve()), "cases": len(output), "skipped": skipped,
        "library_version": library_version,
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
