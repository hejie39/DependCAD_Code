"""Command-line entry point for one real CAD generation task."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .llm import PolicyRoutedClient
from .pipeline import CadQueryDAGAStarAgent, PipelineConfig
from .retrieval import (
    BGEM3Encoder,
    DEFAULT_SERIALIZATION_VERSION,
    RequirementGraphRAG,
    SERIALIZATION_VERSIONS,
    query_serializer,
)


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected JSON object")
        rows.append(value)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--case-library", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/dag_astar"))
    parser.add_argument("--reference-stl", type=Path)
    parser.add_argument("--bge-device", choices=("cuda", "cpu"), default=None)
    parser.add_argument("--bge-revision", default=None)
    parser.add_argument(
        "--retrieval-serialization", choices=SERIALIZATION_VERSIONS,
        default=DEFAULT_SERIALIZATION_VERSION,
    )
    parser.add_argument("--evaluation-manifest", type=Path)
    parser.add_argument("--exclude-id", action="append", default=[])
    parser.add_argument("--exclude-model-id", action="append", default=[])
    parser.add_argument("--exclude-geometry-hash", action="append", default=[])
    args = parser.parse_args()
    if not os.getenv("QWEN_API_KEY", "").strip():
        raise RuntimeError("QWEN_API_KEY is required; keys are never stored in source code")
    cases = _load_jsonl(args.case_library)
    evaluation = _load_jsonl(args.evaluation_manifest) if args.evaluation_manifest else []
    evaluation_ids = {
        str(row.get("case_id") or row.get("task_id") or row.get("id") or "") for row in evaluation
    }
    evaluation_models = {
        str(row.get("model_id") or row.get("source_model_id") or "") for row in evaluation
    }
    evaluation_geometry = {
        str(row.get("geometry_hash") or row.get("shape_hash") or row.get("reference_hash") or "")
        for row in evaluation
    }
    evaluation_texts = [str(row.get("text") or row.get("requirement") or "") for row in evaluation]
    retriever = RequirementGraphRAG(
        cases, encoder=BGEM3Encoder(device=args.bge_device, revision=args.bge_revision),
        excluded_case_ids={args.task_id, *args.exclude_id, *evaluation_ids} - {""},
        excluded_model_ids={*args.exclude_model_id, *evaluation_models} - {""},
        excluded_geometry_hashes={*args.exclude_geometry_hash, *evaluation_geometry} - {""},
        excluded_texts=[args.text, *evaluation_texts],
        serializer=query_serializer(args.retrieval_serialization),
        serialization_version=args.retrieval_serialization,
    )
    agent = CadQueryDAGAStarAgent(
        output_dir=args.output_dir, client=PolicyRoutedClient(), retriever=retriever,
        config=PipelineConfig(),
    )
    result = agent.run(args.text, task_id=args.task_id, reference_stl=args.reference_stl)
    print(json.dumps({
        "task_id": result["task_id"], "success": result["success"],
        "final_source": result["final_source"],
        "step": (result["final_execution"].get("metrics") or {}).get("step"),
        "stl": (result["final_execution"].get("metrics") or {}).get("stl"),
        "run_json": str((args.output_dir / args.task_id / "run.json").resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
