#!/usr/bin/env python3
"""Run paired P0/PR x A*-off/on CAD-Agent experiments on frozen datasets."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import shutil
import sys
import threading
import time
import traceback
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np


VERSION = "dag_astar_formal_v39"
DEFAULT_BGE_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
CONDITIONS = ("P0_no_astar", "PR_no_astar", "P0_astar", "PR_astar")
ALL_BINS = (
    "01_1_10", "02_11_20", "03_21_30",
    "04_31_40", "05_41_50", "06_51_plus",
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path,
        default=repo_root / "datasets" / "cadcoder_natural_formal_v1",
    )
    parser.add_argument(
        "--method-root", type=Path,
        default=repo_root,
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--provider", choices=("ollama", "qwen", "siliconflow"), required=True,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", action="append", choices=("op-cad", "cad-coder"))
    parser.add_argument("--bin", action="append", choices=ALL_BINS)
    parser.add_argument(
        "--case-id", action="append", dest="case_ids",
        help="Run only the explicitly named evaluation case; may be repeated.",
    )
    parser.add_argument(
        "--evaluation-fraction", type=float,
        help="Deterministically retain this stratified fraction of evaluation rows.",
    )
    parser.add_argument(
        "--evaluation-part", choices=("primary", "complement"), default="primary",
        help="Select the retained fraction or its disjoint complement.",
    )
    parser.add_argument("--limit-per-bin", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--bge-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--bge-revision", default=DEFAULT_BGE_REVISION)
    parser.add_argument(
        "--embedding-cache-dir", type=Path,
        default=repo_root / "cache" / "bge_m3",
    )
    parser.add_argument("--llm-timeout", type=float, default=300.0)
    parser.add_argument("--llm-attempts", type=int, default=3)
    parser.add_argument("--occ-timeout", type=float, default=120.0)
    parser.add_argument("--occ-memory-mb", type=int, default=4096)
    parser.add_argument("--max-expansions", type=int, default=6)
    parser.add_argument("--max-candidate-executions", type=int, default=5)
    parser.add_argument("--max-llm-calls", type=int, default=2)
    parser.add_argument("--search-seconds", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--overwrite-failed", action="store_true")
    parser.add_argument(
        "--artifact-policy", choices=("all", "sampled", "none"), default="sampled",
        help=(
            "Retention policy for raw OCC STEP/STL run directories. JSON records, code, "
            "errors, search traces and metrics are always retained."
        ),
    )
    return parser.parse_args()


def stratified_partition(
    rows: list[dict[str, Any]], labels: list[str], fraction: float,
    *, seed: int, namespace: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return an exact-size, deterministic stratified split and its complement."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("evaluation_fraction_must_be_between_zero_and_one")
    target = math.floor(len(rows) * fraction + 0.5)
    groups = {label: [row for row in rows if row["length_bin"] == label] for label in labels}
    quotas = {label: math.floor(len(group) * fraction) for label, group in groups.items()}
    remaining = target - sum(quotas.values())
    allocation_order = sorted(
        labels,
        key=lambda label: (-(len(groups[label]) * fraction - quotas[label]), label),
    )
    for label in allocation_order[:remaining]:
        quotas[label] += 1
    selected_ids: set[str] = set()
    for label in labels:
        ordered = sorted(
            groups[label],
            key=lambda row: hashlib.sha256(
                f"split|{seed}|{namespace}|{label}|{row['case_id']}".encode()
            ).hexdigest(),
        )
        selected_ids.update(str(row["case_id"]) for row in ordered[:quotas[label]])
    primary = [row for row in rows if str(row["case_id"]) in selected_ids]
    complement = [row for row in rows if str(row["case_id"]) not in selected_ids]
    if len(primary) != target or len(primary) + len(complement) != len(rows):
        raise RuntimeError("invalid_stratified_partition")
    return primary, complement


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluation_manifest(dataset_directory: Path) -> Path:
    """Resolve the frozen evaluation manifest without rewriting the dataset."""
    for name in ("evaluation_ready.jsonl", "evaluation.jsonl"):
        candidate = dataset_directory / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"evaluation_manifest_missing:{dataset_directory}")


def resolve_dataset_artifact(value: Any, fallback_directory: Path) -> str:
    """Resolve frozen JSON paths after moving a dataset across operating systems."""
    raw = str(value or "").strip()
    if not raw:
        raise FileNotFoundError(f"dataset_artifact_path_empty:{fallback_directory}")
    original = Path(raw)
    if original.is_file():
        return str(original.resolve())
    # PureWindowsPath also extracts the basename correctly on POSIX hosts when
    # the frozen manifest contains a Windows drive path with backslashes.
    filename = PureWindowsPath(raw).name
    candidate = fallback_directory.resolve() / filename
    if candidate.is_file():
        return str(candidate.resolve())
    raise FileNotFoundError(
        f"dataset_artifact_missing:manifest={raw}:fallback={candidate}"
    )


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_fingerprint(code: str | None) -> str | None:
    return hashlib.sha256(code.encode("utf-8")).hexdigest() if code else None


def provider_model_identity(provider: str, model: str) -> dict[str, Any]:
    identity: dict[str, Any] = {"provider": provider, "requested_model": model}
    if provider != "ollama":
        return identity
    chat_url = os.getenv("OLLAMA_API_URL", "http://127.0.0.1:11434/api/chat").strip()
    tags_url = chat_url.rsplit("/", 1)[0] + "/tags"
    with urllib.request.urlopen(tags_url, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
    matches = [item for item in payload.get("models", []) if item.get("name") == model]
    if len(matches) != 1 or not matches[0].get("digest"):
        raise RuntimeError(f"ollama_model_identity_not_resolved:{model}")
    selected = matches[0]
    identity.update({
        "digest": str(selected["digest"]),
        "size_bytes": int(selected.get("size", 0)),
        "modified_at": str(selected.get("modified_at") or ""),
        "details": selected.get("details") or {},
    })
    return identity


def seed_everything(seed: int, *, seed_cuda: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if not seed_cuda:
        return
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def execution_host_identity() -> dict[str, Any]:
    packages = {}
    for name in (
        "cadquery", "faiss-cpu", "numpy", "psutil", "requests", "scipy",
        "sentence-transformers", "torch", "trimesh",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "host_label": os.getenv("CAD_AGENT_HOST_LABEL", "unspecified"),
        "packages": packages,
    }


def _collect_run_directories(value: Any) -> set[Path]:
    found: set[Path] = set()
    if isinstance(value, dict):
        run_dir = value.get("run_dir")
        if isinstance(run_dir, str) and run_dir.strip():
            found.add(Path(run_dir))
        for child in value.values():
            found.update(_collect_run_directories(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_collect_run_directories(child))
    return found


def cleanup_occ_artifacts(result: dict[str, Any], runtime) -> dict[str, Any]:
    """Delete only run directories owned by this runtime after metrics are serialized."""
    root = Path(runtime.output_dir).resolve()
    removed: list[str] = []
    for raw_path in sorted(_collect_run_directories(result), key=lambda item: str(item)):
        candidate = raw_path.resolve()
        if candidate == root or not candidate.is_relative_to(root):
            raise RuntimeError(f"refusing_to_remove_unowned_occ_artifact:{candidate}")
        if candidate.is_dir():
            shutil.rmtree(candidate)
            removed.append(str(candidate))
    return {
        "retained": False,
        "removed_run_directories": len(removed),
        "runtime_root": str(root),
    }


def summarise_llm_records(
    records: list[dict], *, logical_calls: int | None = None,
    failed_logical_calls: int | None = None,
) -> dict[str, Any]:
    return {
        "logical_llm_records": len(records) if logical_calls is None else int(logical_calls),
        "failed_logical_llm_records": (
            sum(record.get("success") is False for record in records)
            if failed_logical_calls is None else int(failed_logical_calls)
        ),
        "transport_attempts": sum(max(1, len(item.get("attempts") or [])) for item in records),
        "prompt_tokens": sum(int(item.get("prompt_tokens", 0)) for item in records),
        "completion_tokens": sum(int(item.get("completion_tokens", 0)) for item in records),
        "total_tokens": sum(int(item.get("total_tokens", 0)) for item in records),
        "llm_latency_seconds": sum(float(item.get("latency_s", 0.0)) for item in records),
    }


def validation_result(
    validate_requirement_constraints, code: str | None, execution: dict,
    requirement_graph: dict, text: str, graph_result,
) -> dict:
    return validate_requirement_constraints(
        code or "", execution, requirement_graph, requirement_text=text,
        graph_build_issues=graph_result.fallback_validation_issues,
        fallback_used=graph_result.fallback_used,
    )


def offline_metrics(
    runtime, code: str | None, reference_stl: str, task_id: str,
    acceptance: dict,
) -> dict:
    if not code:
        execution = {
            "success": False, "stage": "generation", "error_type": "ProgramUnavailable",
            "error": "condition_has_no_program", "metrics": {}, "geometry_metrics": None,
        }
    else:
        execution = runtime.execute_code(
            code, task_id=task_id, attempt=0, reference_stl=reference_stl,
        )
    geometry = execution.get("geometry_metrics") or {}
    if geometry.get("error"):
        geometry = {"error": geometry["error"]}
    protocol = geometry.get("protocol") or {}
    iou = geometry.get("volumetric_iou")
    f1 = geometry.get("f1_score")
    iou_threshold = float(protocol.get("strict_iou_threshold", 0.50))
    f1_threshold = float(protocol.get("strict_f1_threshold", 0.50))
    strict = bool(
        acceptance.get("accepted") and execution.get("success")
        and isinstance(iou, (int, float)) and float(iou) >= iou_threshold
        and isinstance(f1, (int, float)) and float(f1) >= f1_threshold
    )
    return {
        "code": code,
        "code_sha256": code_fingerprint(code),
        "acceptance": acceptance,
        "execution": execution,
        "execution_success": bool(execution.get("success")),
        "requirement_accepted": bool(acceptance.get("accepted")),
        "volumetric_iou": iou,
        "surface_f1": f1,
        "chamfer_distance": geometry.get("chamfer_distance"),
        "hausdorff_distance": geometry.get("hausdorff_distance"),
        "hausdorff_95_distance": geometry.get("hausdorff_95_distance"),
        "surface_precision": geometry.get("precision"),
        "surface_recall": geometry.get("recall"),
        "bbox_center_error": geometry.get("bbox_center_error"),
        "bbox_extent_relative_error": geometry.get("bbox_extent_relative_error"),
        "volume_relative_error": geometry.get("volume_relative_error"),
        "strict_success": strict,
        "metric_error": geometry.get("error"),
        "reused_offline_evaluation": False,
    }


def run_task(context: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    graph_builder_class = context["RequirementGraphBuilder"]
    generator_class = context["DualCodeGenerator"]
    repair_class = context["HeuristicAStarRepair"]
    validate = context["validate_requirement_constraints"]
    budget = context["budget"]
    runtime = context["runtime"]
    client = context["client"]
    retriever = context["retriever"]
    retrieval_lock = context["retrieval_lock"]
    fingerprint = context["fingerprint"]
    output_path: Path = context["task_dir"] / f"{safe_name(task['case_id'])}.json"
    if output_path.is_file():
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        reusable = previous.get("run_fingerprint") == fingerprint and previous.get("status") == "completed"
        if reusable and not (context["overwrite_failed"] and not previous.get("task_completed")):
            previous["resumed"] = True
            return previous

    started = time.perf_counter()
    text = str(task["text"])
    reference_stl = str(task["reference_stl"])
    result: dict[str, Any] = {
        "version": VERSION,
        "run_fingerprint": fingerprint,
        "status": "running",
        "task_completed": False,
        "resumed": False,
        "dataset": task["dataset"],
        "case_id": task["case_id"],
        "model_id": task["model_id"],
        "length": task["length"],
        "length_bin": task["length_bin"],
        "length_metric": task["length_metric"],
        "text": text,
        "reference_stl": reference_stl,
        "reference_hash": task["reference_hash"],
    }
    try:
        graph_result = graph_builder_class(client, max_retries=2).build(text)
        requirement_graph = graph_result.graph
        try:
            with retrieval_lock:
                retrieval = retriever.retrieve(text, requirement_graph, top_n=20, top_k=3)
        except Exception as error:
            retrieval = {
                "success": False, "strategy": "failed", "selected": [],
                "selected_case_ids": [], "error_type": type(error).__name__, "error": str(error),
            }
        blueprints = [item["blueprint"] for item in retrieval.get("selected", [])]
        rag_available = bool(retrieval.get("success") and blueprints)
        generator = generator_class(client, max_regenerations=2)
        pair_parameters = inspect.signature(generator.generate_pair).parameters
        pair_kwargs = {"rag_available": rag_available}
        if "requirement_graph" in pair_parameters:
            pair_kwargs["requirement_graph"] = requirement_graph
        generated = generator.generate_pair(text, blueprints, **pair_kwargs)
        programs = {branch: generated[branch].code for branch in ("P0", "PR")}

        initial_execution: dict[str, dict] = {}
        initial_validation: dict[str, dict] = {}
        no_astar: dict[str, dict] = {}
        for branch in ("P0", "PR"):
            code = programs[branch]
            if code:
                observed = runtime.execute_code(
                    code, task_id=f"{safe_name(task['case_id'])}_{branch.lower()}_initial",
                    attempt=0, reference_stl=None,
                )
            else:
                observed = {
                    "success": False, "stage": "generation", "error_type": "ProgramUnavailable",
                    "error": "branch_did_not_produce_code", "metrics": {},
                }
            acceptance = validation_result(
                validate, code, observed, requirement_graph, text, graph_result,
            )
            initial_execution[branch] = observed
            initial_validation[branch] = acceptance
            no_astar[f"{branch}_no_astar"] = offline_metrics(
                runtime, code, reference_stl,
                f"{safe_name(task['case_id'])}_{branch.lower()}_no_astar_eval",
                acceptance,
            )

        astar: dict[str, dict] = {}
        for branch in ("P0", "PR"):
            code = programs[branch]
            condition = f"{branch}_astar"
            if not code:
                astar[condition] = {
                    **no_astar[f"{branch}_no_astar"],
                    "search": None, "reused_offline_evaluation": True,
                }
                continue

            def execute(candidate: str, attempt: int, remaining: float | None = None) -> dict:
                return runtime.execute_code(
                    candidate,
                    task_id=f"{safe_name(task['case_id'])}_{branch.lower()}_repair",
                    attempt=attempt, reference_stl=None, timeout_seconds=remaining,
                )

            def provide(candidate: str, failure: dict, remaining: float | None = None):
                return generator.repair_candidates(
                    text, candidate, failure, requirement_graph, count=2,
                    timeout_seconds=remaining,
                )

            def constraint(candidate: str, observed: dict, graph: dict) -> dict:
                return validate(
                    candidate, observed, graph, requirement_text=text,
                    graph_build_issues=graph_result.fallback_validation_issues,
                    fallback_used=graph_result.fallback_used,
                )

            try:
                search = repair_class(
                    execute, requirement_graph, budget=budget, constraint_validator=constraint,
                ).search(code, initial_execution[branch], provide)
            except Exception as error:
                base_record = no_astar[f"{branch}_no_astar"]
                astar[condition] = {
                    **base_record,
                    "search": {
                        "success": False,
                        "code": code,
                        "execution": initial_execution[branch],
                        "acceptance": initial_validation[branch],
                        "terminal_node": "n0",
                        "tree": {},
                        "trace": [{
                            "event": "search_internal_error",
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }],
                        "expansions": 0,
                        "candidate_executions": 0,
                        "llm_calls": 0,
                        "provider_errors": 0,
                        "elapsed_seconds": 0.0,
                        "exhausted": "internal_error",
                    },
                    "search_error": f"{type(error).__name__}:{error}",
                    "reused_offline_evaluation": True,
                }
                continue
            search_record = search.to_dict()
            final_code = search.code
            final_acceptance = search.acceptance
            base_record = no_astar[f"{branch}_no_astar"]
            if code_fingerprint(final_code) == base_record.get("code_sha256"):
                evaluated = {**base_record, "reused_offline_evaluation": True}
            else:
                evaluated = offline_metrics(
                    runtime, final_code, reference_stl,
                    f"{safe_name(task['case_id'])}_{branch.lower()}_astar_eval",
                    final_acceptance,
                )
            astar[condition] = {**evaluated, "search": search_record}

        conditions = {**no_astar, **astar}
        graph_calls = [*graph_result.calls]
        branch_generation_calls: dict[str, list[dict]] = {}
        branch_search_calls: dict[str, list[dict]] = {}
        branch_search_logical: dict[str, int] = {}
        branch_search_failed: dict[str, int] = {}
        for branch in ("P0", "PR"):
            branch_generation_calls[branch] = list(generated[branch].calls)
            search = astar[f"{branch}_astar"].get("search") or {}
            branch_search_logical[branch] = int(search.get("llm_calls", 0))
            branch_search_failed[branch] = int(search.get("provider_errors", 0))
            records: list[dict] = []
            for item in search.get("trace", []):
                if item.get("event") == "expand" and isinstance(item.get("llm_record"), dict):
                    records.append(item["llm_record"])
                elif item.get("event") == "provider_error":
                    records.extend(
                        attempt for attempt in item.get("attempts", [])
                        if isinstance(attempt, dict)
                    )
            branch_search_calls[branch] = records
        base_calls = [*graph_calls, *branch_generation_calls["P0"], *branch_generation_calls["PR"]]
        search_calls = [*branch_search_calls["P0"], *branch_search_calls["PR"]]
        calls = [*base_calls, *search_calls]
        search_logical_calls = sum(branch_search_logical.values())
        failed_search_calls = sum(branch_search_failed.values())
        failed_base_calls = sum(record.get("success") is False for record in base_calls)
        condition_accounting: dict[str, dict[str, Any]] = {}
        for branch in ("P0", "PR"):
            shared_and_generation = [*graph_calls, *branch_generation_calls[branch]]
            generation_failed = sum(
                record.get("success") is False for record in shared_and_generation
            )
            no_astar_key = f"{branch}_no_astar"
            astar_key = f"{branch}_astar"
            initial_seconds = float(initial_execution[branch].get("runtime_seconds", 0.0))
            no_astar_eval_seconds = float(
                (no_astar[no_astar_key].get("execution") or {}).get("runtime_seconds", 0.0)
            )
            astar_eval_seconds = float(
                (astar[astar_key].get("execution") or {}).get("runtime_seconds", 0.0)
            )
            no_astar_accounting = summarise_llm_records(
                shared_and_generation, failed_logical_calls=generation_failed,
            )
            no_astar_accounting.update({
                "initial_execution_seconds": initial_seconds,
                "search_elapsed_seconds": 0.0,
                "offline_evaluation_seconds": no_astar_eval_seconds,
            })
            astar_records = [*shared_and_generation, *branch_search_calls[branch]]
            astar_accounting = summarise_llm_records(
                astar_records,
                logical_calls=len(shared_and_generation) + branch_search_logical[branch],
                failed_logical_calls=generation_failed + branch_search_failed[branch],
            )
            astar_accounting.update({
                "initial_execution_seconds": initial_seconds,
                "search_elapsed_seconds": float(
                    (astar[astar_key].get("search") or {}).get("elapsed_seconds", 0.0)
                ),
                "offline_evaluation_seconds": astar_eval_seconds,
            })
            condition_accounting[no_astar_key] = no_astar_accounting
            condition_accounting[astar_key] = astar_accounting
        result.update({
            "status": "completed",
            "task_completed": True,
            "requirement_graph": requirement_graph,
            "requirement_graph_build": {
                "fallback_used": graph_result.fallback_used,
                "issues": graph_result.issues,
                "fallback_validation_issues": graph_result.fallback_validation_issues,
                "llm_calls": graph_result.calls,
            },
            "retrieval": retrieval,
            "rag_used": rag_available,
            "blueprints": blueprints,
            "generation": {
                branch: {
                    "code": generated[branch].code,
                    "static_issues": generated[branch].static_issues,
                    "error": generated[branch].error,
                    "llm_calls": generated[branch].calls,
                } for branch in ("P0", "PR")
            },
            "initial_execution": initial_execution,
            "initial_validation": initial_validation,
            "conditions": conditions,
            "condition_accounting": condition_accounting,
            "accounting": summarise_llm_records(
                calls,
                logical_calls=len(base_calls) + search_logical_calls,
                failed_logical_calls=failed_base_calls + failed_search_calls,
            ),
        })
    except Exception as error:
        result.update({
            "status": "completed", "task_completed": False,
            "error_type": type(error).__name__, "error": str(error),
            "traceback": traceback.format_exc(),
            "conditions": result.get("conditions", {}),
        })
    result["runtime_seconds"] = time.perf_counter() - started
    retain_artifacts = bool(context["retain_artifacts"])
    if retain_artifacts:
        result["occ_artifacts"] = {
            "retained": True,
            "policy": context["artifact_policy"],
            "runtime_root": str(Path(runtime.output_dir).resolve()),
        }
    else:
        cleanup_record = cleanup_occ_artifacts(result, runtime)
        cleanup_record["policy"] = context["artifact_policy"]
        result["occ_artifacts"] = cleanup_record
    atomic_json(output_path, result)
    return result


def main() -> None:
    args = parse_args()
    if args.case_ids and args.evaluation_fraction is not None:
        raise ValueError("case_ids_and_evaluation_fraction_are_mutually_exclusive")
    if args.evaluation_part == "complement" and args.evaluation_fraction is None:
        raise ValueError("evaluation_part_complement_requires_evaluation_fraction")
    seed_everything(args.seed, seed_cuda=args.bge_device == "cuda")
    dataset_root = args.dataset_root.resolve()
    method_root = args.method_root.resolve()
    output_root = args.output_root.resolve()
    embedding_cache_dir = args.embedding_cache_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(method_root))

    from experiments.cadquery_agent.dag_astar.constraints import validate_requirement_constraints
    from experiments.cadquery_agent.dag_astar.graphs import RequirementGraphBuilder
    from experiments.cadquery_agent.dag_astar.llm import DualCodeGenerator, PROMPT_VERSIONS
    from experiments.cadquery_agent.dag_astar.metrics import DEFAULT_PROTOCOL
    from experiments.cadquery_agent.dag_astar.retrieval import BGEM3Encoder, RequirementGraphRAG
    from experiments.cadquery_agent.dag_astar.search import HeuristicAStarRepair, SearchBudget
    from experiments.cadquery_agent.llm_client import OpenAICompatibleClient
    from experiments.cadquery_agent.runtime import CadQueryRuntime

    if args.provider == "qwen" and not os.getenv("QWEN_API_KEY", "").strip():
        raise RuntimeError("QWEN_API_KEY is required")
    if args.provider == "siliconflow" and not os.getenv("SILICONFLOW_API_KEY", "").strip():
        raise RuntimeError("SILICONFLOW_API_KEY is required")
    client = OpenAICompatibleClient(
        provider=args.provider, model=args.model,
        timeout=int(args.llm_timeout), max_attempts=args.llm_attempts,
    )
    client.timeout = args.llm_timeout
    model_identity = provider_model_identity(args.provider, args.model)
    budget = SearchBudget(
        max_expansions=args.max_expansions,
        max_candidate_executions=args.max_candidate_executions,
        max_llm_calls=args.max_llm_calls,
        max_seconds=args.search_seconds,
    )
    selected_datasets = args.dataset or ["op-cad", "cad-coder"]
    selected_bins = args.bin or list(ALL_BINS)
    agent_root = method_root / "experiments" / "cadquery_agent"
    code_files = sorted((agent_root / "dag_astar").glob("*.py")) + [
        agent_root / "ast_graph.py",
        agent_root / "llm_client.py",
        agent_root / "runtime.py",
    ]
    code_hash = hashlib.sha256("".join(file_sha256(path) for path in code_files).encode()).hexdigest()
    dataset_hash = hashlib.sha256("".join(
        file_sha256(evaluation_manifest(dataset_root / name))
        + file_sha256(dataset_root / name / "case_library.jsonl")
        for name in selected_datasets
    ).encode()).hexdigest()
    config = {
        "version": VERSION,
        "provider": args.provider,
        "model": args.model,
        "execution_host": execution_host_identity(),
        "model_identity": model_identity,
        "llm_client": client.cache_identity(),
        "ollama_generation": (
            {
                "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "16384")),
                "max_tokens": int(os.getenv("OLLAMA_MAX_TOKENS", "4096")),
                "keep_alive": os.getenv("OLLAMA_KEEP_ALIVE", "15m"),
                "think": os.getenv("OLLAMA_THINK", "false").strip().lower()
                in {"1", "true", "yes", "on"},
            }
            if args.provider == "ollama" else None
        ),
        "dataset_root": str(dataset_root),
        "datasets": selected_datasets,
        "bins": selected_bins,
        "case_ids": sorted(set(args.case_ids or [])),
        "evaluation_fraction": args.evaluation_fraction,
        "evaluation_part": args.evaluation_part,
        "limit_per_bin": args.limit_per_bin,
        "workers": args.workers,
        "artifact_policy": args.artifact_policy,
        "bge_model": "BAAI/bge-m3",
        "bge_revision": args.bge_revision,
        "bge_device": args.bge_device,
        "embedding_cache_dir": str(embedding_cache_dir),
        "seed": args.seed,
        "search_budget": asdict(budget),
        "occ_runtime": {
            "timeout_seconds": args.occ_timeout,
            "max_memory_mb": args.occ_memory_mb,
            "python_executable": os.getenv("CADQUERY_RUNTIME_PYTHON", "") or sys.executable,
        },
        "metric_protocol": asdict(DEFAULT_PROTOCOL),
        "prompt_versions": PROMPT_VERSIONS,
        "method_code_sha256": code_hash,
        "dataset_sha256": dataset_hash,
    }
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    config["run_fingerprint"] = fingerprint
    config_path = output_root / "run_config.json"
    if config_path.is_file():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing.get("run_fingerprint") != fingerprint:
            raise RuntimeError("output_root_contains_different_experiment_configuration")
    else:
        atomic_json(config_path, config)

    print(json.dumps({"event": "load_embedding", "model": "BAAI/bge-m3", "device": args.bge_device}), flush=True)
    encoder = BGEM3Encoder(
        device=args.bge_device, model_name="BAAI/bge-m3", revision=args.bge_revision,
    )
    runtime = CadQueryRuntime(
        output_root / "occ_runs", timeout_seconds=args.occ_timeout,
        max_memory_mb=args.occ_memory_mb,
    )
    all_jobs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    # Both dataset retrievers share one SentenceTransformer instance.  A single
    # lock is therefore required across datasets; per-retriever locks still
    # permit concurrent calls into the same PyTorch module on Windows.
    shared_retrieval_lock = threading.Lock()
    requested_case_ids = set(args.case_ids or [])
    found_case_ids: set[str] = set()
    for dataset_name in selected_datasets:
        root = dataset_root / dataset_name
        cases = read_jsonl(root / "case_library.jsonl")
        full_evaluation = read_jsonl(evaluation_manifest(root))
        for row in full_evaluation:
            row["reference_stl"] = resolve_dataset_artifact(
                row.get("reference_stl"), root / "reference_stl",
            )
            if row.get("reference_code_path"):
                row["reference_code_path"] = resolve_dataset_artifact(
                    row["reference_code_path"], root / "reference_programs",
                )
        evaluation = [
            row for row in full_evaluation
            if row["length_bin"] in selected_bins
            and (not requested_case_ids or str(row["case_id"]) in requested_case_ids)
        ]
        if args.evaluation_fraction is not None:
            primary, complement = stratified_partition(
                evaluation, selected_bins, args.evaluation_fraction,
                seed=args.seed, namespace=dataset_name,
            )
            evaluation = primary if args.evaluation_part == "primary" else complement
        found_case_ids.update(str(row["case_id"]) for row in evaluation)
        if args.limit_per_bin is not None:
            kept: list[dict[str, Any]] = []
            for label in selected_bins:
                rows = [row for row in evaluation if row["length_bin"] == label]
                rows.sort(key=lambda row: hashlib.sha256(
                    f"{args.seed}|{dataset_name}|{label}|{row['case_id']}".encode()
                ).hexdigest())
                kept.extend(rows[:args.limit_per_bin])
            evaluation = kept
        sampled_artifact_ids: set[str] = set()
        if args.artifact_policy == "sampled":
            for label in selected_bins:
                rows = [row for row in evaluation if row["length_bin"] == label]
                if rows:
                    selected = min(
                        rows,
                        key=lambda row: hashlib.sha256(
                            f"artifact|{args.seed}|{dataset_name}|{label}|{row['case_id']}".encode()
                        ).hexdigest(),
                    )
                    sampled_artifact_ids.add(str(selected["case_id"]))
        retriever = RequirementGraphRAG(
            cases, encoder=encoder,
            excluded_case_ids={str(row["case_id"]) for row in full_evaluation},
            excluded_model_ids={str(row["model_id"]) for row in full_evaluation},
            excluded_geometry_hashes={str(row.get("reference_hash") or "") for row in full_evaluation} - {""},
            excluded_texts=[str(row["text"]) for row in full_evaluation],
            # Embeddings depend on encoder + filtered corpus + query serialisation,
            # not on orchestration/accounting-only runner revisions.
            cache_namespace=f"dag_astar_formal_v4:{dataset_name}",
            embedding_cache_dir=embedding_cache_dir,
        )
        if not retriever.cases:
            raise RuntimeError(f"{dataset_name}:retrieval_corpus_empty_after_formal_isolation")
        context = {
            "RequirementGraphBuilder": RequirementGraphBuilder,
            "DualCodeGenerator": DualCodeGenerator,
            "HeuristicAStarRepair": HeuristicAStarRepair,
            "validate_requirement_constraints": validate_requirement_constraints,
            "budget": budget,
            "runtime": runtime,
            "client": client,
            "retriever": retriever,
            "retrieval_lock": shared_retrieval_lock,
            "fingerprint": fingerprint,
            "task_dir": output_root / dataset_name / "tasks",
            "overwrite_failed": args.overwrite_failed,
            "artifact_policy": args.artifact_policy,
        }
        for row in evaluation:
            task_context = dict(context)
            task_context["retain_artifacts"] = bool(
                args.artifact_policy == "all"
                or (
                    args.artifact_policy == "sampled"
                    and str(row["case_id"]) in sampled_artifact_ids
                )
            )
            all_jobs.append((task_context, row))
        print(json.dumps({
            "event": "dataset_ready", "dataset": dataset_name,
            "tasks": len(evaluation), "rag_cases_input": len(cases),
            "rag_cases_retained": len(retriever.cases),
        }), flush=True)

    missing_case_ids = requested_case_ids - found_case_ids
    if missing_case_ids:
        raise RuntimeError(f"requested_case_ids_not_found:{sorted(missing_case_ids)}")

    progress_lock = threading.Lock()
    completed = successful = 0
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(run_task, context, task): task for context, task in all_jobs}
        for future in as_completed(futures):
            row = future.result()
            with progress_lock:
                completed += 1
                successful += int(row.get("task_completed") is True)
                print(json.dumps({
                    "event": "task_complete", "completed": completed, "total": len(all_jobs),
                    "task_completed": row.get("task_completed"), "successful_tasks": successful,
                    "dataset": row.get("dataset"), "case_id": row.get("case_id"),
                    "length_bin": row.get("length_bin"),
                    "runtime_seconds": row.get("runtime_seconds"),
                    "elapsed_seconds": time.perf_counter() - started,
                }, ensure_ascii=False), flush=True)
    atomic_json(output_root / "run_status.json", {
        "status": "completed", "tasks": len(all_jobs), "task_completed": successful,
        "elapsed_seconds": time.perf_counter() - started, "run_fingerprint": fingerprint,
    })
    print(json.dumps({
        "event": "run_complete", "tasks": len(all_jobs), "task_completed": successful,
        "elapsed_seconds": time.perf_counter() - started,
    }), flush=True)


if __name__ == "__main__":
    main()
