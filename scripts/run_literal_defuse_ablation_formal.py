#!/usr/bin/env python3
"""Run the DependCAD internal ablations with matched execution budgets."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

from run_dag_astar_formal import (
    ALL_BINS,
    DEFAULT_BGE_REVISION,
    atomic_json,
    cleanup_occ_artifacts,
    code_fingerprint,
    evaluation_manifest,
    execution_host_identity,
    file_sha256,
    offline_metrics,
    provider_model_identity,
    read_jsonl,
    resolve_dataset_artifact,
    safe_name,
    seed_everything,
    stratified_partition,
    summarise_llm_records,
)


VERSION = "trustcad_six_ablation_v33_literal_defuse_constraints"
RETRIEVAL_SERIALIZATION_VERSIONS = (
    "text_plus_requirement_dag_v1",
    "structure_first_compact_text_v1",
)
CONDITIONS = (
    "A0_full",
    "A1_no_graphrag",
    "A2_no_cross_graph_alignment",
    "A3_greedy_repair",
    "A4_astar_no_heuristic",
    "A5_astar_no_dependency_closure",
)
MATRIX_DEFINITION = {
    "A0_full": {
        "generation": "p0_pr_reference_free_selection", "cross_graph_alignment": True,
        "repair": "heuristic_astar", "heuristic": "cad_structural",
    },
    "A1_no_graphrag": {
        "generation": "requirement_dag_only", "cross_graph_alignment": True,
        "repair": "heuristic_astar", "heuristic": "cad_structural",
    },
    "A2_no_cross_graph_alignment": {
        "generation": "p0_pr_reference_free_selection", "cross_graph_alignment": False,
        "repair": "astar", "heuristic": "nonstructural_only",
    },
    "A3_greedy_repair": {
        "generation": "p0_pr_reference_free_selection", "cross_graph_alignment": True,
        "repair": "sequential_greedy", "heuristic": None,
    },
    "A4_astar_no_heuristic": {
        "generation": "p0_pr_reference_free_selection", "cross_graph_alignment": True,
        "repair": "uniform_cost_g_only", "heuristic": "zero",
    },
    "A5_astar_no_dependency_closure": {
        "generation": "p0_pr_reference_free_selection", "cross_graph_alignment": True,
        "repair": "heuristic_astar_no_dependency_closure", "heuristic": "cad_structural",
    },
}


HOST_SPECIFIC_DATASET_KEYS = frozenset({
    "library_version", "reference_code_path", "reference_stl",
})


def semantic_jsonl_sha256(path: Path) -> str:
    """Hash experiment semantics without host-specific absolute paths.

    The referenced artifact contents are already identified by reference_hash
    and reference_code_sha256. Absolute Windows/Linux paths and a legacy
    path-derived library_version must not split an otherwise identical formal
    dataset across execution hosts.
    """
    digest = hashlib.sha256()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"dataset_row_not_object:{path}:{line_number}")
            canonical = {
                key: value for key, value in row.items()
                if key not in HOST_SPECIFIC_DATASET_KEYS
            }
            digest.update(json.dumps(
                canonical, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--method-root", type=Path,
        default=repo_root,
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--experiment-version", default=VERSION,
        help="Stable version string included in task records and cache fingerprints.",
    )
    parser.add_argument(
        "--provider", choices=("ollama", "qwen", "siliconflow"), required=True,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", action="append", choices=("op-cad", "cad-coder"))
    parser.add_argument("--bin", action="append", choices=ALL_BINS)
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument("--condition", action="append", choices=CONDITIONS)
    parser.add_argument("--evaluation-fraction", type=float)
    parser.add_argument("--evaluation-part", choices=("primary", "complement"), default="primary")
    parser.add_argument("--limit-per-bin", type=int)
    parser.add_argument(
        "--limit-per-dataset", type=int,
        help="Deterministically sample exactly this many cases from each dataset, "
        "allocated proportionally across the selected length bins.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--bge-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--bge-batch-size", type=int, default=8)
    parser.add_argument("--bge-revision", default=DEFAULT_BGE_REVISION)
    parser.add_argument(
        "--retrieval-serialization",
        choices=RETRIEVAL_SERIALIZATION_VERSIONS,
        default=RETRIEVAL_SERIALIZATION_VERSIONS[0],
    )
    parser.add_argument("--prompt-top-k", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--retrieval-min-score", type=float, default=0.0)
    parser.add_argument("--retrieval-min-topology", type=float, default=0.0)
    parser.add_argument("--retrieval-vector-weight", type=float, default=0.40)
    parser.add_argument("--retrieval-operation-weight", type=float, default=0.20)
    parser.add_argument("--retrieval-dependency-weight", type=float, default=0.15)
    parser.add_argument("--retrieval-topology-weight", type=float, default=0.25)
    parser.add_argument(
        "--embedding-cache-dir", type=Path,
        default=repo_root / "cache" / "bge_m3",
    )
    parser.add_argument("--llm-timeout", type=float, default=300.0)
    parser.add_argument("--llm-attempts", type=int, default=3)
    parser.add_argument("--occ-timeout", type=float, default=120.0)
    parser.add_argument("--occ-memory-mb", type=int, default=8192)
    parser.add_argument("--max-expansions", type=int, default=6)
    parser.add_argument("--max-candidate-executions", type=int, default=5)
    parser.add_argument("--max-llm-calls", type=int, default=2)
    parser.add_argument("--search-seconds", type=float, default=180.0)
    parser.add_argument("--heuristic-static-weight", type=float, default=3.0)
    parser.add_argument("--heuristic-operation-weight", type=float, default=8.0)
    parser.add_argument("--heuristic-dependency-weight", type=float, default=5.0)
    parser.add_argument("--heuristic-constraint-weight", type=float, default=4.0)
    parser.add_argument("--heuristic-role-weight", type=float, default=6.0)
    parser.add_argument("--heuristic-path-weight", type=float, default=4.0)
    parser.add_argument("--heuristic-output-weight", type=float, default=5.0)
    parser.add_argument("--heuristic-extra-weight", type=float, default=3.0)
    parser.add_argument("--outside-closure-edit-weight", type=float, default=1.5)
    parser.add_argument(
        "--linear-unit-policy",
        choices=("native_numeric", "cadquery_mm"),
        default="native_numeric",
        help=(
            "How explicit linear units in text map to CadQuery coordinates. "
            "CAD-Coder reference programs use the numeric magnitudes verbatim."
        ),
    )
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--overwrite-failed", action="store_true")
    parser.add_argument(
        "--require-llm-requirement-graph",
        action="store_true",
        help=(
            "Reject a task when requirement-DAG construction falls back to the "
            "rule parser. This guarantees that every scored task uses a valid "
            "DAG returned by the configured LLM."
        ),
    )
    parser.add_argument("--artifact-policy", choices=("all", "sampled", "none"), default="sampled")
    return parser.parse_args()


def validate_ollama_service_contract(workers: int) -> dict[str, int]:
    """Require enough server context for every concurrent formal request."""
    request_num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "0"))
    request_max_tokens = int(os.getenv("OLLAMA_MAX_TOKENS", "0"))
    server_context_length = int(os.getenv("OLLAMA_SERVER_CONTEXT_LENGTH", "0"))
    server_num_parallel = int(os.getenv("OLLAMA_SERVER_NUM_PARALLEL", "0"))
    model_context_length = int(os.getenv("OLLAMA_MODEL_CONTEXT_LENGTH", "0"))
    if request_num_ctx <= 0 or request_max_tokens <= 0:
        raise RuntimeError("ollama_request_context_contract_missing")
    if server_num_parallel < workers:
        raise RuntimeError(
            f"ollama_server_parallelism_too_small:{server_num_parallel}<{workers}"
        )
    if model_context_length <= 0:
        raise RuntimeError("ollama_model_context_contract_missing")
    if request_num_ctx > model_context_length:
        raise RuntimeError(
            "ollama_request_exceeds_model_context:"
            f"{request_num_ctx}>{model_context_length}"
        )
    required_context = request_num_ctx * workers
    if server_context_length < required_context:
        raise RuntimeError(
            "ollama_server_context_too_small:"
            f"{server_context_length}<{request_num_ctx}*{workers}"
        )
    return {
        "request_num_ctx": request_num_ctx,
        "request_max_tokens": request_max_tokens,
        "server_context_length": server_context_length,
        "server_num_parallel": server_num_parallel,
        "model_context_length": model_context_length,
    }


def _failure_execution(error: str) -> dict:
    return {
        "success": False,
        "stage": "generation",
        "error_type": "ProgramUnavailable",
        "error": error,
        "metrics": {},
    }


def _branch_selection_key(
    acceptance: dict, execution: dict, *, branch: str,
) -> tuple:
    """Reference-free P0/PR ordering used before any repair search.

    The key follows the method contract: static/constraint validity, real OCC
    execution and requirement-program alignment.  Retrieval rank, reference
    geometry and offline metrics are deliberately absent.  PR is only the
    deterministic final tie-break because P0 and PR have already tied on every
    observable acceptance field.
    """
    static_check = acceptance.get("static_check") or {}
    countable_static = static_check.get("countable_issues") or []
    return (
        int(bool(acceptance.get("accepted"))),
        int(bool(execution.get("success"))),
        float(acceptance.get("constraint_pass_rate") or 0.0),
        float(acceptance.get("constraint_fidelity") or 0.0),
        float(acceptance.get("structural_score") or 0.0),
        -len(countable_static),
        -len(acceptance.get("issues") or []),
        int(branch == "GraphRAG"),
    )


def _search_records(search: dict, event: str) -> list[dict]:
    records: list[dict] = []
    for item in search.get("trace") or []:
        if item.get("event") == event and isinstance(item.get("llm_record"), dict):
            records.append(item["llm_record"])
        elif item.get("event") == "provider_error":
            records.extend(
                attempt for attempt in item.get("attempts") or []
                if isinstance(attempt, dict)
            )
    return records


def _condition_accounting(
    base_calls: list[dict], execution: dict, evaluation: dict, *, search: dict | None = None,
) -> dict:
    search = search or {}
    search_calls: list[dict] = []
    for item in search.get("trace") or []:
        if item.get("event") in {"retry", "expand"} and isinstance(item.get("llm_record"), dict):
            search_calls.append(item["llm_record"])
        elif item.get("event") == "provider_error":
            search_calls.extend(
                attempt for attempt in item.get("attempts") or []
                if isinstance(attempt, dict)
            )
    records = [*base_calls, *search_calls]
    output = summarise_llm_records(
        records,
        logical_calls=len(base_calls) + int(search.get("llm_calls", 0)),
        failed_logical_calls=(
            sum(item.get("success") is False for item in base_calls)
            + int(search.get("provider_errors", 0))
        ),
    )
    output.update({
        "initial_execution_seconds": float(execution.get("runtime_seconds", 0.0)),
        "search_elapsed_seconds": float(search.get("elapsed_seconds", 0.0)),
        "offline_evaluation_seconds": float(
            (evaluation.get("execution") or {}).get("runtime_seconds", 0.0)
        ),
        "shared_candidate_cache_hits": sum(
            bool(item.get("shared_candidate_cache_hit")) for item in search_calls
        ),
        "physical_search_llm_calls": sum(
            item.get("physical_llm_call", True) is not False for item in search_calls
        ),
    })
    return output


_OCC_TEMP_PATH_RE = re.compile(
    r'["\']?[A-Za-z]?:?[/\\][^"\'\n]*?occ_runs[/\\][^"\'\n]*?runner\.py["\']?',
    flags=re.IGNORECASE,
)


def _normalize_error_for_state_key(text: str) -> str:
    return _OCC_TEMP_PATH_RE.sub("<runner.py>", text)


def _candidate_state_key(code: str, failure: dict) -> str:
    """Fingerprint the exact repair prompt state shared by matched ablations."""
    validation = failure.get("constraint_validation") or {}
    raw_error = str(failure.get("error") or failure.get("stderr") or "")[-4000:]
    payload = {
        "code_sha256": code_fingerprint(code),
        "stage": failure.get("stage"),
        "error_type": failure.get("error_type"),
        "error": _normalize_error_for_state_key(raw_error),
        "constraint_issues": list(validation.get("issues") or []),
        "constraint_checks": list(validation.get("constraint_checks") or []),
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class PromptExactCacheClient:
    """Task-local prompt-exact single-flight cache across matched ablations.

    When two ablation conditions construct the exact same (system, user,
    max_tokens) repair prompt (e.g., when a gate falls back to the same greedy
    repair prompt on a pure execution error), returning the cached LLM response
    eliminates pure API sampling noise while preserving distinct calls whenever
    prompts actually differ.
    """

    def __init__(self, client):
        self._client = client
        self._cache: dict[str, Any] = {}
        self._inflight: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def __getattr__(self, name: str):
        return getattr(self._client, name)

    @staticmethod
    def _prompt_key(system_prompt: str, user_prompt: str, max_tokens: int | None) -> str:
        payload = {
            "system": system_prompt,
            "user": _normalize_error_for_state_key(user_prompt),
            "max_tokens": max_tokens,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        *,
        timeout_seconds: float | None = None,
    ):
        key = self._prompt_key(system_prompt, user_prompt, max_tokens)
        while True:
            with self._lock:
                if key in self._cache:
                    return copy.deepcopy(self._cache[key])
                event = self._inflight.get(key)
                if event is None:
                    event = threading.Event()
                    self._inflight[key] = event
                    owner = True
                    break
            event.wait()
        try:
            kwargs: dict[str, Any] = {}
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            if timeout_seconds is not None:
                kwargs["timeout_seconds"] = timeout_seconds
            result = self._client.complete(system_prompt, user_prompt, **kwargs)
            with self._lock:
                self._cache[key] = copy.deepcopy(result)
            return result
        finally:
            with self._lock:
                self._inflight.pop(key, None)
                event.set()


class SharedCandidateOracle:
    """Use common LLM candidates whenever two ablations reach the same state.

    A0, A3 and A4 differ only in search policy or heuristic. Independent LLM
    draws at an identical state confound that comparison, even at temperature
    zero. This task-local single-flight cache gives those methods the exact same
    candidate list while retaining a logical LLM-call record for each method.
    """

    def __init__(self, provider):
        self.provider = provider
        self._cache: dict[str, tuple[list[str], dict]] = {}
        self._inflight: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self.physical_calls = 0
        self.cache_hits = 0

    @staticmethod
    def _annotate(record: dict, key: str, *, cache_hit: bool) -> dict:
        output = copy.deepcopy(record)
        output.update({
            "shared_candidate_state_key": key,
            "shared_candidate_cache_hit": cache_hit,
            "physical_llm_call": not cache_hit,
        })
        return output

    def __call__(
        self, code: str, failure: dict, remaining: float | None = None,
    ) -> tuple[list[str], dict]:
        key = _candidate_state_key(code, failure)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self.cache_hits += 1
                candidates, record = cached
                return list(candidates), self._annotate(record, key, cache_hit=True)
            event = self._inflight.get(key)
            owner = event is None
            if owner:
                event = threading.Event()
                self._inflight[key] = event

        if not owner:
            if not event.wait(timeout=remaining):
                raise TimeoutError("shared_candidate_oracle_wait_timeout")
            with self._lock:
                cached = self._cache.get(key)
                if cached is None:
                    raise RuntimeError("shared_candidate_oracle_source_failed")
                self.cache_hits += 1
                candidates, record = cached
            return list(candidates), self._annotate(record, key, cache_hit=True)

        try:
            candidates, record = self.provider(code, failure, remaining)
            stored = (list(candidates), copy.deepcopy(record))
            with self._lock:
                self._cache[key] = stored
                self.physical_calls += 1
            return list(candidates), self._annotate(record, key, cache_hit=False)
        finally:
            with self._lock:
                self._inflight.pop(key, None)
                event.set()

    def stats(self) -> dict:
        with self._lock:
            return {
                "entries": len(self._cache),
                "physical_calls": self.physical_calls,
                "cache_hits": self.cache_hits,
            }


def stratified_round_robin(
    jobs: list[tuple[dict[str, Any], dict[str, Any]]], *,
    datasets: list[str], bins: list[str], seed: int,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Interleave dataset x length strata so partial runs remain representative."""
    strata = [(dataset.lower(), label) for label in bins for dataset in datasets]
    grouped: dict[tuple[str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = {
        key: [] for key in strata
    }
    for job in jobs:
        row = job[1]
        key = (str(row["dataset"]).lower(), str(row["length_bin"]))
        grouped.setdefault(key, []).append(job)
    for key, rows in grouped.items():
        rows.sort(key=lambda job: hashlib.sha256(
            f"round_robin|{seed}|{key[0]}|{key[1]}|{job[1]['case_id']}".encode()
        ).hexdigest())
    ordered: list[tuple[dict[str, Any], dict[str, Any]]] = []
    offsets = {key: 0 for key in strata}
    while True:
        added = False
        for key in strata:
            offset = offsets[key]
            rows = grouped.get(key) or []
            if offset < len(rows):
                ordered.append(rows[offset])
                offsets[key] = offset + 1
                added = True
        if not added:
            break
    if len(ordered) != len(jobs):
        raise RuntimeError(f"stratified_schedule_lost_jobs:{len(ordered)}!={len(jobs)}")
    return ordered


def proportional_stratified_sample(
    rows: list[dict[str, Any]], *, bins: list[str], target: int,
    seed: int, namespace: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Select exactly ``target`` rows with largest-remainder bin allocation."""
    if target <= 0:
        raise ValueError("limit_per_dataset_must_be_positive")
    if target > len(rows):
        raise ValueError(
            f"limit_per_dataset_exceeds_available:{namespace}:{target}>{len(rows)}"
        )
    grouped = {
        label: [row for row in rows if str(row["length_bin"]) == label]
        for label in bins
    }
    total = sum(len(items) for items in grouped.values())
    if total != len(rows):
        raise ValueError(f"rows_outside_selected_bins:{namespace}:{len(rows) - total}")

    exact = {label: target * len(grouped[label]) / total for label in bins}
    quotas = {label: int(exact[label]) for label in bins}
    remaining = target - sum(quotas.values())
    bin_order = {label: index for index, label in enumerate(bins)}
    ranked = sorted(
        bins,
        key=lambda label: (-(exact[label] - quotas[label]), bin_order[label]),
    )
    for label in ranked[:remaining]:
        quotas[label] += 1

    selected: list[dict[str, Any]] = []
    for label in bins:
        candidates = sorted(
            grouped[label],
            key=lambda row: hashlib.sha256(
                f"proportional|{seed}|{namespace}|{label}|{row['case_id']}".encode()
            ).hexdigest(),
        )
        selected.extend(candidates[:quotas[label]])
    if len(selected) != target:
        raise RuntimeError(
            f"proportional_sample_wrong_size:{namespace}:{len(selected)}!={target}"
        )
    return selected, quotas


def run_task(context: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    output_path: Path = context["task_dir"] / f"{safe_name(task['case_id'])}.json"
    if output_path.is_file():
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        reusable = (
            previous.get("run_fingerprint") == context["fingerprint"]
            and previous.get("status") == "completed"
        )
        if reusable and not (context["overwrite_failed"] and not previous.get("task_completed")):
            previous["resumed"] = True
            return previous

    started = time.perf_counter()
    text = str(task["text"])
    reference_stl = str(task["reference_stl"])
    runtime = context["runtime"]
    task_slug = safe_name(task["case_id"])
    result: dict[str, Any] = {
        "version": context.get("version", VERSION),
        "run_fingerprint": context["fingerprint"],
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
        graph_result = context["RequirementGraphBuilder"](
            context["client"], max_retries=2,
        ).build(text)
        if context["require_llm_requirement_graph"] and (
            graph_result.fallback_used or not graph_result.calls
        ):
            result["requirement_graph_build"] = {
                "fallback_used": graph_result.fallback_used,
                "issues": graph_result.issues,
                "fallback_validation_issues": graph_result.fallback_validation_issues,
                "llm_calls": graph_result.calls,
            }
            raise ValueError("llm_requirement_graph_required_but_fallback_used")
        if not graph_result.valid:
            raise ValueError(
                "invalid_requirement_graph_after_fallback:"
                + "|".join(graph_result.fallback_validation_issues)
            )
        requirement_graph = graph_result.graph
        try:
            with context["retrieval_lock"]:
                retrieval = context["retriever"].retrieve(
                    text, requirement_graph, top_n=20, top_k=3,
                )
        except Exception as error:
            retrieval = {
                "success": False, "strategy": "failed", "selected": [],
                "selected_case_ids": [], "error_type": type(error).__name__,
                "error": str(error),
            }
        selected_hits = list(retrieval.get("selected") or [])
        top_hit = selected_hits[0] if selected_hits else {}
        top_components = top_hit.get("components") or {}
        retrieval_gate = {
            "min_score": context["retrieval_min_score"],
            "min_topology": context["retrieval_min_topology"],
            "top_score": float(top_hit.get("score") or 0.0),
            "top_topology": float(top_components.get("topology") or 0.0),
        }
        retrieval_gate["passed"] = bool(
            retrieval.get("success")
            and selected_hits
            and retrieval_gate["top_score"] >= retrieval_gate["min_score"]
            and retrieval_gate["top_topology"] >= retrieval_gate["min_topology"]
        )
        prompt_hits = selected_hits[:context["prompt_top_k"]]
        blueprints = [item["blueprint"] for item in prompt_hits]
        rag_available = bool(retrieval_gate["passed"] and blueprints)
        task_client = PromptExactCacheClient(context["client"])
        generator = context["DualCodeGenerator"](task_client, max_regenerations=2)

        with ThreadPoolExecutor(max_workers=2) as generation_pool:
            dag_future = generation_pool.submit(
                generator.generate_from_requirement_graph, text, requirement_graph,
            )
            rag_future = (
                generation_pool.submit(
                    generator.generate_branch,
                    text,
                    blueprints,
                    requirement_graph=requirement_graph,
                )
                if rag_available else None
            )
            dag_generated = dag_future.result()
            rag_generated = rag_future.result() if rag_future is not None else None

        rag_code = rag_generated.code if rag_generated is not None else None
        dag_code = dag_generated.code
        rag_execution = (
            runtime.execute_code(
                rag_code, task_id=f"{task_slug}_rag_initial",
                attempt=0, reference_stl=None,
            ) if rag_code else _failure_execution("graphrag_generation_unavailable")
        )
        dag_execution = (
            runtime.execute_code(
                dag_code, task_id=f"{task_slug}_dag_only_initial",
                attempt=0, reference_stl=None,
            ) if dag_code else _failure_execution("dag_only_generation_failed")
        )

        def validate_structural(code: str, execution: dict, graph: dict) -> dict:
            return context["validate_structural"](
                code, execution, graph, config=context["constraint_config"],
                requirement_text=text,
                graph_build_issues=graph_result.fallback_validation_issues,
                fallback_used=graph_result.fallback_used,
            )

        def validate_natural(code: str, execution: dict, _graph: dict) -> dict:
            return context["validate_natural"](
                code, execution, config=context["constraint_config"],
                requirement_text=text,
            )

        dag_initial_acceptance = validate_structural(
            dag_code or "", dag_execution, requirement_graph,
        )
        initial_candidates = [{
            "branch": "DAGOnly",
            "code": dag_code,
            "execution": dag_execution,
            "acceptance": dag_initial_acceptance,
        }]
        if rag_available:
            rag_initial_acceptance = validate_structural(
                rag_code or "", rag_execution, requirement_graph,
            )
            initial_candidates.append({
                "branch": "GraphRAG",
                "code": rag_code,
                "execution": rag_execution,
                "acceptance": rag_initial_acceptance,
            })
        else:
            rag_initial_acceptance = validate_structural(
                rag_code or "", rag_execution, requirement_graph,
            )
        selected_initial = max(
            initial_candidates,
            key=lambda item: _branch_selection_key(
                item["acceptance"], item["execution"], branch=item["branch"],
            ),
        )
        selected_code = selected_initial["code"]
        selected_execution = selected_initial["execution"]

        def raw_aligned_provider(
            code: str, failure: dict, remaining: float | None = None,
        ):
            return generator.repair_candidates(
                text, code, failure, requirement_graph,
                count=2, timeout_seconds=remaining,
            )

        aligned_provider = SharedCandidateOracle(raw_aligned_provider)

        def raw_no_closure_provider(
            code: str, failure: dict, remaining: float | None = None,
        ):
            return generator.repair_candidates(
                text, code, failure, requirement_graph,
                count=2, timeout_seconds=remaining,
                use_dependency_closure=False,
            )

        no_closure_provider = SharedCandidateOracle(raw_no_closure_provider)

        def unaligned_provider(
            code: str, failure: dict, remaining: float | None = None,
        ):
            return generator.repair_candidates_greedy(
                text, code, failure, count=2, timeout_seconds=remaining,
            )

        task_exec_cache: dict[str, dict] = {}
        task_exec_lock = threading.Lock()
        if rag_code:
            task_exec_cache[code_fingerprint(rag_code)] = copy.deepcopy(rag_execution)
        if dag_code:
            task_exec_cache[code_fingerprint(dag_code)] = copy.deepcopy(dag_execution)

        def run_search(
            condition: str, initial_code: str | None, initial_execution: dict, *,
            strategy: str, aligned: bool, heuristic_config=None,
            dependency_closure: bool = True,
        ) -> tuple[dict | None, str | None, dict, dict]:
            validator = validate_structural if aligned else validate_natural
            initial_acceptance = validator(
                initial_code or "", initial_execution, requirement_graph,
            )
            if not initial_code:
                return None, initial_code, initial_execution, initial_acceptance

            def execute(
                candidate: str, attempt: int, remaining: float | None = None,
            ) -> dict:
                fp = code_fingerprint(candidate)
                with task_exec_lock:
                    cached_exec = task_exec_cache.get(fp)
                if cached_exec is not None:
                    return copy.deepcopy(cached_exec)
                outcome = runtime.execute_code(
                    candidate, task_id=f"{task_slug}_{condition}", attempt=attempt,
                    reference_stl=None, timeout_seconds=remaining,
                )
                with task_exec_lock:
                    task_exec_cache[fp] = copy.deepcopy(outcome)
                return outcome

            provider = (
                aligned_provider if aligned and dependency_closure
                else no_closure_provider if aligned
                else unaligned_provider
            )
            if strategy == "greedy":
                searcher = context["GreedyRetryRepair"](
                    execute, requirement_graph, budget=context["budget"],
                    constraint_validator=validator,
                )
            else:
                searcher = context["HeuristicAStarRepair"](
                    execute, requirement_graph, budget=context["budget"],
                    heuristic_config=heuristic_config,
                    constraint_validator=validator,
                )
            search_result = searcher.search(initial_code, initial_execution, provider)
            return (
                search_result.to_dict(), search_result.code,
                search_result.execution, search_result.acceptance,
            )

        no_alignment_config = context["HeuristicConfig"](
            operation_weight=0.0, dependency_weight=0.0,
            use_graph_alignment=False, rank_best_by_structural_score=False,
        )
        zero_penalties = {
            "static": 0.0, "timeout": 0.0, "empty": 0.0,
            "geometry": 0.0, "constraint": 0.0, "other": 0.0,
        }
        zero_heuristic_config = context["HeuristicConfig"](
            static_weight=0.0, operation_weight=0.0,
            dependency_weight=0.0, constraint_weight=0.0,
            role_dependency_weight=0.0, path_dependency_weight=0.0,
            output_dependency_weight=0.0, extra_operation_weight=0.0,
            failure_penalties=zero_penalties,
            use_graph_alignment=True, rank_best_by_structural_score=False,
        )

        selected_conditions = context["conditions"]
        search_outputs = {}
        if "A0_full" in selected_conditions:
            search_outputs["A0_full"] = run_search(
                "A0_full", selected_code, selected_execution,
                strategy="astar", aligned=True,
                heuristic_config=context["full_heuristic_config"],
            )
        if "A1_no_graphrag" in selected_conditions:
            search_outputs["A1_no_graphrag"] = run_search(
                "A1_no_graphrag", dag_code, dag_execution,
                strategy="astar", aligned=True,
                heuristic_config=context["full_heuristic_config"],
            )
        if "A2_no_cross_graph_alignment" in selected_conditions:
            search_outputs["A2_no_cross_graph_alignment"] = run_search(
                "A2_no_cross_graph_alignment", selected_code, selected_execution,
                strategy="astar", aligned=False,
                heuristic_config=no_alignment_config,
            )
        if "A3_greedy_repair" in selected_conditions:
            search_outputs["A3_greedy_repair"] = run_search(
                "A3_greedy_repair", selected_code, selected_execution,
                strategy="greedy", aligned=True,
            )
        if "A4_astar_no_heuristic" in selected_conditions:
            search_outputs["A4_astar_no_heuristic"] = run_search(
                "A4_astar_no_heuristic", selected_code, selected_execution,
                strategy="astar", aligned=True,
                heuristic_config=zero_heuristic_config,
            )
        if "A5_astar_no_dependency_closure" in selected_conditions:
            search_outputs["A5_astar_no_dependency_closure"] = run_search(
                "A5_astar_no_dependency_closure", selected_code, selected_execution,
                strategy="astar", aligned=True,
                heuristic_config=context["full_heuristic_config"],
                dependency_closure=False,
            )

        evaluation_cache: dict[tuple, dict] = {}
        conditions: dict[str, dict] = {}
        initial_executions = {
            "A0_full": selected_execution,
            "A1_no_graphrag": dag_execution,
            "A2_no_cross_graph_alignment": selected_execution,
            "A3_greedy_repair": selected_execution,
            "A4_astar_no_heuristic": selected_execution,
            "A5_astar_no_dependency_closure": selected_execution,
        }
        for condition, (search_record, code, terminal_execution, search_acceptance) in search_outputs.items():
            common_acceptance = context["validate_natural"](
                code or "", terminal_execution,
                config=context["constraint_config"], requirement_text=text,
            )
            evaluation_key = (
                code_fingerprint(code), bool(common_acceptance.get("accepted")),
                tuple(common_acceptance.get("issues") or []),
            )
            if evaluation_key in evaluation_cache:
                evaluation = json.loads(json.dumps(evaluation_cache[evaluation_key]))
                evaluation["reused_offline_evaluation"] = True
            else:
                evaluation = offline_metrics(
                    runtime, code, reference_stl,
                    f"{task_slug}_{condition}_eval", common_acceptance,
                )
                evaluation_cache[evaluation_key] = evaluation
            definition = MATRIX_DEFINITION[condition]
            conditions[condition] = {
                **evaluation,
                "search": search_record,
                "search_acceptance": search_acceptance,
                "generation": definition["generation"],
                "cross_graph_alignment": definition["cross_graph_alignment"],
                "repair_strategy": definition["repair"],
                "heuristic": definition["heuristic"],
                "uses_rag": condition != "A1_no_graphrag" and rag_available,
                "retrieval_success": bool(retrieval.get("success")),
            }

        graph_calls = list(graph_result.calls)
        rag_generation_calls = list(rag_generated.calls) if rag_generated is not None else []
        dag_generation_calls = list(dag_generated.calls)
        condition_accounting: dict[str, dict] = {}
        all_search_calls: list[dict] = []
        for condition in selected_conditions:
            search_record = search_outputs[condition][0] or {}
            generation_calls = (
                dag_generation_calls if condition == "A1_no_graphrag"
                else [*dag_generation_calls, *rag_generation_calls]
            )
            condition_accounting[condition] = _condition_accounting(
                [*graph_calls, *generation_calls],
                initial_executions[condition], conditions[condition],
                search=search_record,
            )
            event = "retry" if condition == "A3_greedy_repair" else "expand"
            all_search_calls.extend(_search_records(search_record, event))

        all_calls = [
            *graph_calls, *rag_generation_calls, *dag_generation_calls,
            *all_search_calls,
        ]
        result.update({
            "status": "completed",
            "task_completed": True,
            "matrix_definition": {
                condition: MATRIX_DEFINITION[condition]
                for condition in selected_conditions
            },
            "requirement_graph": requirement_graph,
            "requirement_graph_build": {
                "fallback_used": graph_result.fallback_used,
                "issues": graph_result.issues,
                "fallback_validation_issues": graph_result.fallback_validation_issues,
                "llm_calls": graph_result.calls,
            },
            "retrieval": retrieval,
            "rag_used": rag_available,
            "retrieval_gate": retrieval_gate,
            "prompt_retrieved_case_ids": [item.get("case_id") for item in prompt_hits],
            "blueprints": blueprints,
            "initial_selection": {
                "selected_branch": selected_initial["branch"],
                "selection_key": list(_branch_selection_key(
                    selected_initial["acceptance"], selected_initial["execution"],
                    branch=selected_initial["branch"],
                )),
                "candidates": [{
                    "branch": item["branch"],
                    "code_sha256": (
                        code_fingerprint(item["code"]) if item["code"] else None
                    ),
                    "execution_success": bool(item["execution"].get("success")),
                    "acceptance": item["acceptance"],
                    "selection_key": list(_branch_selection_key(
                        item["acceptance"], item["execution"], branch=item["branch"],
                    )),
                } for item in initial_candidates],
                "uses_reference_geometry": False,
            },
            "shared_candidate_oracle": aligned_provider.stats(),
            "no_closure_candidate_oracle": no_closure_provider.stats(),
            "generation": {
                "GraphRAG": {
                    "code": rag_code,
                    "static_issues": rag_generated.static_issues if rag_generated else [],
                    "error": rag_generated.error if rag_generated else {
                        "error_type": "RetrievalUnavailable",
                        "error": "GraphRAG branch not generated without retrieved blueprints",
                    },
                    "llm_calls": rag_generation_calls,
                    "uses_requirement_dag": True,
                    "uses_blueprints": rag_available,
                },
                "DAGOnly": {
                    "code": dag_code,
                    "static_issues": dag_generated.static_issues,
                    "error": dag_generated.error,
                    "llm_calls": dag_generation_calls,
                    "uses_requirement_dag": True,
                    "uses_blueprints": False,
                },
            },
            "initial_execution": {
                "GraphRAG": rag_execution, "DAGOnly": dag_execution,
            },
            "conditions": conditions,
            "condition_accounting": condition_accounting,
            "accounting": summarise_llm_records(all_calls),
        })
    except Exception as error:
        result.update({
            "status": "completed", "task_completed": False,
            "error_type": type(error).__name__, "error": str(error),
            "traceback": traceback.format_exc(),
            "conditions": result.get("conditions", {}),
        })
    result["runtime_seconds"] = time.perf_counter() - started
    if context["retain_artifacts"]:
        result["occ_artifacts"] = {
            "retained": True, "policy": context["artifact_policy"],
            "runtime_root": str(Path(runtime.output_dir).resolve()),
        }
    else:
        cleanup = cleanup_occ_artifacts(result, runtime)
        cleanup["policy"] = context["artifact_policy"]
        result["occ_artifacts"] = cleanup
    atomic_json(output_path, result)
    return result


def main() -> None:
    args = parse_args()
    ollama_service_contract = (
        validate_ollama_service_contract(args.workers)
        if args.provider == "ollama" else None
    )
    selection_modes = sum(value is not None for value in (
        args.evaluation_fraction,
        args.limit_per_bin,
        args.limit_per_dataset,
        args.case_ids,
    ))
    if selection_modes > 1:
        raise ValueError("evaluation_selection_modes_are_mutually_exclusive")
    if args.evaluation_part == "complement" and args.evaluation_fraction is None:
        raise ValueError("evaluation_part_complement_requires_evaluation_fraction")
    if args.limit_per_dataset is not None and args.limit_per_dataset <= 0:
        raise ValueError("limit_per_dataset_must_be_positive")
    if args.bge_batch_size <= 0:
        raise ValueError("bge_batch_size_must_be_positive")
    retrieval_weights = {
        "vector": args.retrieval_vector_weight,
        "operation": args.retrieval_operation_weight,
        "dependency": args.retrieval_dependency_weight,
        "topology": args.retrieval_topology_weight,
    }
    if any(value < 0.0 for value in retrieval_weights.values()):
        raise ValueError("retrieval_weights_must_be_nonnegative")
    if not math.isclose(sum(retrieval_weights.values()), 1.0, abs_tol=1e-8):
        raise ValueError("retrieval_weights_must_sum_to_one")
    if not 0.0 <= args.retrieval_min_score <= 1.0:
        raise ValueError("retrieval_min_score_must_be_in_unit_interval")
    if not 0.0 <= args.retrieval_min_topology <= 1.0:
        raise ValueError("retrieval_min_topology_must_be_in_unit_interval")
    heuristic_weights = {
        "static": args.heuristic_static_weight,
        "operation": args.heuristic_operation_weight,
        "dependency": args.heuristic_dependency_weight,
        "constraint": args.heuristic_constraint_weight,
        "role": args.heuristic_role_weight,
        "path": args.heuristic_path_weight,
        "output": args.heuristic_output_weight,
        "extra": args.heuristic_extra_weight,
        "outside_closure_edit": args.outside_closure_edit_weight,
    }
    if any(value < 0.0 for value in heuristic_weights.values()):
        raise ValueError("heuristic_weights_must_be_nonnegative")
    if args.provider == "qwen" and not os.getenv("QWEN_API_KEY", "").strip():
        raise RuntimeError("QWEN_API_KEY is required")
    if args.provider == "siliconflow" and not os.getenv("SILICONFLOW_API_KEY", "").strip():
        raise RuntimeError("SILICONFLOW_API_KEY is required")

    seed_everything(args.seed, seed_cuda=args.bge_device == "cuda")
    dataset_root = args.dataset_root.resolve()
    method_root = args.method_root.resolve()
    output_root = args.output_root.resolve()
    embedding_cache_dir = args.embedding_cache_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(method_root))

    from experiments.cadquery_agent.dag_astar.constraints import (
        ConstraintConfig,
        validate_natural_language_constraints,
        validate_requirement_constraints,
    )
    from experiments.cadquery_agent.dag_astar.graphs import (
        REQUIREMENT_GRAPH_PROMPT_VERSION,
        RequirementGraphBuilder,
    )
    from experiments.cadquery_agent.dag_astar.llm import DualCodeGenerator, PROMPT_VERSIONS
    from experiments.cadquery_agent.dag_astar.metrics import DEFAULT_PROTOCOL
    from experiments.cadquery_agent.dag_astar.retrieval import (
        BGEM3Encoder,
        RequirementGraphRAG,
        query_serializer,
    )
    from experiments.cadquery_agent.dag_astar.search import (
        GreedyRetryRepair,
        HeuristicConfig,
        HeuristicAStarRepair,
        SearchBudget,
    )
    from experiments.cadquery_agent.llm_client import OpenAICompatibleClient
    from experiments.cadquery_agent.runtime import CadQueryRuntime

    client = OpenAICompatibleClient(
        provider=args.provider,
        model=args.model,
        timeout=int(args.llm_timeout),
        max_attempts=args.llm_attempts,
        seed=args.seed,
    )
    client.timeout = args.llm_timeout
    budget = SearchBudget(
        max_expansions=args.max_expansions,
        max_candidate_executions=args.max_candidate_executions,
        max_llm_calls=args.max_llm_calls,
        max_seconds=args.search_seconds,
    )
    full_heuristic_config = HeuristicConfig(
        static_weight=args.heuristic_static_weight,
        operation_weight=args.heuristic_operation_weight,
        dependency_weight=args.heuristic_dependency_weight,
        constraint_weight=args.heuristic_constraint_weight,
        role_dependency_weight=args.heuristic_role_weight,
        path_dependency_weight=args.heuristic_path_weight,
        output_dependency_weight=args.heuristic_output_weight,
        extra_operation_weight=args.heuristic_extra_weight,
        outside_closure_edit_weight=args.outside_closure_edit_weight,
    )
    constraint_config = ConstraintConfig(linear_unit_policy=args.linear_unit_policy)
    selected_datasets = args.dataset or ["op-cad", "cad-coder"]
    selected_bins = args.bin or list(ALL_BINS)
    selected_conditions = args.condition or list(CONDITIONS)
    agent_root = method_root / "experiments" / "cadquery_agent"
    code_files = sorted((agent_root / "dag_astar").glob("*.py")) + [
        agent_root / "ast_graph.py",
        agent_root / "llm_client.py",
        agent_root / "runtime.py",
        Path(__file__).resolve(),
    ]
    code_hash = hashlib.sha256(
        "".join(file_sha256(path) for path in code_files).encode()
    ).hexdigest()
    dataset_storage_hash = hashlib.sha256("".join(
        file_sha256(evaluation_manifest(dataset_root / name))
        + file_sha256(dataset_root / name / "case_library.jsonl")
        for name in selected_datasets
    ).encode()).hexdigest()
    dataset_hash = hashlib.sha256("".join(
        semantic_jsonl_sha256(evaluation_manifest(dataset_root / name))
        + semantic_jsonl_sha256(dataset_root / name / "case_library.jsonl")
        for name in selected_datasets
    ).encode()).hexdigest()
    config = {
        "version": args.experiment_version,
        "conditions": selected_conditions,
        "matrix_definition": {
            condition: MATRIX_DEFINITION[condition]
            for condition in selected_conditions
        },
        "provider": args.provider,
        "model": args.model,
        "model_identity": provider_model_identity(args.provider, args.model),
        "llm_client": client.cache_identity(),
        "ollama_service_contract": ollama_service_contract,
        "execution_host": execution_host_identity(),
        "dataset_root": str(dataset_root),
        "datasets": selected_datasets,
        "bins": selected_bins,
        "case_ids": sorted(set(args.case_ids or [])),
        "evaluation_fraction": args.evaluation_fraction,
        "evaluation_part": args.evaluation_part,
        "limit_per_bin": args.limit_per_bin,
        "limit_per_dataset": args.limit_per_dataset,
        "workers": args.workers,
        "task_order": "dataset_length_round_robin_v1",
        "artifact_policy": args.artifact_policy,
        "require_llm_requirement_graph": args.require_llm_requirement_graph,
        "bge_model": "BAAI/bge-m3",
        "bge_revision": args.bge_revision,
        "bge_device": args.bge_device,
        "bge_batch_size": args.bge_batch_size,
        "retrieval_serialization_version": args.retrieval_serialization,
        "retrieval_weights": retrieval_weights,
        "retrieval_gate": {
            "min_score": args.retrieval_min_score,
            "min_topology": args.retrieval_min_topology,
        },
        "prompt_top_k": args.prompt_top_k,
        "embedding_cache_dir": str(embedding_cache_dir),
        "seed": args.seed,
        "search_budget": asdict(budget),
        "heuristic_config": {
            **heuristic_weights,
            "failure_penalties": dict(full_heuristic_config.failure_penalties),
            "use_graph_alignment": full_heuristic_config.use_graph_alignment,
            "rank_best_by_structural_score": (
                full_heuristic_config.rank_best_by_structural_score
            ),
            "continue_after_acceptance": (
                full_heuristic_config.continue_after_acceptance
            ),
        },
        "constraint_config": asdict(constraint_config),
        "occ_runtime": {
            "timeout_seconds": args.occ_timeout,
            "max_memory_mb": args.occ_memory_mb,
            "python_executable": os.getenv("CADQUERY_RUNTIME_PYTHON", "") or sys.executable,
        },
        "metric_protocol": asdict(DEFAULT_PROTOCOL),
        "prompt_versions": {
            **PROMPT_VERSIONS,
            "requirement_graph": REQUIREMENT_GRAPH_PROMPT_VERSION,
        },
        "method_code_sha256": code_hash,
        "dataset_sha256": dataset_hash,
        "dataset_storage_sha256": dataset_storage_hash,
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

    print(json.dumps({
        "event": "load_embedding", "model": "BAAI/bge-m3", "device": args.bge_device,
        "batch_size": args.bge_batch_size,
    }), flush=True)
    encoder = BGEM3Encoder(
        device=args.bge_device,
        batch_size=args.bge_batch_size,
        model_name="BAAI/bge-m3",
        revision=args.bge_revision,
    )
    runtime = CadQueryRuntime(
        output_root / "occ_runs",
        timeout_seconds=args.occ_timeout,
        max_memory_mb=args.occ_memory_mb,
    )
    requested_case_ids = set(args.case_ids or [])
    found_case_ids: set[str] = set()
    shared_retrieval_lock = threading.Lock()
    all_jobs: list[tuple[dict[str, Any], dict[str, Any]]] = []
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
            limited: list[dict[str, Any]] = []
            for label in selected_bins:
                rows = [row for row in evaluation if row["length_bin"] == label]
                rows.sort(key=lambda row: hashlib.sha256(
                    f"{args.seed}|{dataset_name}|{label}|{row['case_id']}".encode()
                ).hexdigest())
                limited.extend(rows[:args.limit_per_bin])
            evaluation = limited
        if args.limit_per_dataset is not None:
            evaluation, quotas = proportional_stratified_sample(
                evaluation,
                bins=selected_bins,
                target=args.limit_per_dataset,
                seed=args.seed,
                namespace=dataset_name,
            )
            print(json.dumps({
                "event": "dataset_proportional_sample",
                "dataset": dataset_name,
                "target": args.limit_per_dataset,
                "quotas": quotas,
            }, ensure_ascii=False), flush=True)

        sampled_artifact_ids: set[str] = set()
        if args.artifact_policy == "sampled":
            for label in selected_bins:
                rows = [row for row in evaluation if row["length_bin"] == label]
                if rows:
                    selected = min(rows, key=lambda row: hashlib.sha256(
                        f"artifact|{args.seed}|{dataset_name}|{label}|{row['case_id']}".encode()
                    ).hexdigest())
                    sampled_artifact_ids.add(str(selected["case_id"]))

        retriever = RequirementGraphRAG(
            cases,
            encoder=encoder,
            weights=retrieval_weights,
            excluded_case_ids={str(row["case_id"]) for row in full_evaluation},
            excluded_model_ids={str(row["model_id"]) for row in full_evaluation},
            excluded_geometry_hashes={
                str(row.get("reference_hash") or "") for row in full_evaluation
            } - {""},
            excluded_texts=[str(row["text"]) for row in full_evaluation],
            cache_namespace=f"dag_astar_formal_v4:{dataset_name}",
            embedding_cache_dir=embedding_cache_dir,
            serializer=query_serializer(args.retrieval_serialization),
            serialization_version=args.retrieval_serialization,
        )
        if not retriever.cases:
            raise RuntimeError(f"{dataset_name}:retrieval_corpus_empty_after_formal_isolation")
        base_context = {
            "RequirementGraphBuilder": RequirementGraphBuilder,
            "DualCodeGenerator": DualCodeGenerator,
            "GreedyRetryRepair": GreedyRetryRepair,
            "HeuristicConfig": HeuristicConfig,
            "HeuristicAStarRepair": HeuristicAStarRepair,
            "validate_natural": validate_natural_language_constraints,
            "validate_structural": validate_requirement_constraints,
            "budget": budget,
            "full_heuristic_config": full_heuristic_config,
            "constraint_config": constraint_config,
            "runtime": runtime,
            "client": client,
            "retriever": retriever,
            "retrieval_lock": shared_retrieval_lock,
            "fingerprint": fingerprint,
            "version": args.experiment_version,
            "task_dir": output_root / dataset_name / "tasks",
            "overwrite_failed": args.overwrite_failed,
            "artifact_policy": args.artifact_policy,
            "require_llm_requirement_graph": args.require_llm_requirement_graph,
            "conditions": selected_conditions,
            "prompt_top_k": args.prompt_top_k,
            "retrieval_min_score": args.retrieval_min_score,
            "retrieval_min_topology": args.retrieval_min_topology,
        }
        for row in evaluation:
            task_context = dict(base_context)
            task_context["retain_artifacts"] = bool(
                args.artifact_policy == "all"
                or (
                    args.artifact_policy == "sampled"
                    and str(row["case_id"]) in sampled_artifact_ids
                )
            )
            all_jobs.append((task_context, row))
        print(json.dumps({
            "event": "dataset_ready",
            "dataset": dataset_name,
            "tasks": len(evaluation),
            "rag_cases_input": len(cases),
            "rag_cases_retained": len(retriever.cases),
        }), flush=True)

    missing = requested_case_ids - found_case_ids
    if missing:
        raise RuntimeError(f"requested_case_ids_not_found:{sorted(missing)}")

    all_jobs = stratified_round_robin(
        all_jobs, datasets=selected_datasets, bins=selected_bins, seed=args.seed,
    )
    print(json.dumps({
        "event": "schedule_ready",
        "strategy": "dataset_length_round_robin_v1",
        "tasks": len(all_jobs),
        "preview": [
            {
                "dataset": task["dataset"],
                "length_bin": task["length_bin"],
                "case_id": task["case_id"],
            }
            for _, task in all_jobs[:min(24, len(all_jobs))]
        ],
    }, ensure_ascii=False), flush=True)

    completed = successful = 0
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(run_task, context, task): task for context, task in all_jobs}
        for future in as_completed(futures):
            row = future.result()
            completed += 1
            successful += int(row.get("task_completed") is True)
            print(json.dumps({
                "event": "task_complete",
                "completed": completed,
                "total": len(all_jobs),
                "task_completed": row.get("task_completed"),
                "successful_tasks": successful,
                "dataset": row.get("dataset"),
                "case_id": row.get("case_id"),
                "length_bin": row.get("length_bin"),
                "runtime_seconds": row.get("runtime_seconds"),
                "elapsed_seconds": time.perf_counter() - started,
            }, ensure_ascii=False), flush=True)
    status = {
        "status": "completed",
        "tasks": len(all_jobs),
        "task_completed": successful,
        "elapsed_seconds": time.perf_counter() - started,
        "run_fingerprint": fingerprint,
    }
    atomic_json(output_root / "run_status.json", status)
    print(json.dumps({"event": "run_complete", **status}), flush=True)


if __name__ == "__main__":
    main()
