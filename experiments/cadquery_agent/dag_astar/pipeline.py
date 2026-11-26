"""Fault-isolated end-to-end requirement-DAG, Graph-RAG and A* pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..runtime import CadQueryRuntime
from .constraints import validate_requirement_constraints
from .graphs import (
    REQUIREMENT_GRAPH_PROMPT_VERSION, RequirementGraphBuilder,
    graph_consistency, graph_difference,
)
from .llm import PROMPT_VERSIONS, DualCodeGenerator
from .metrics import DEFAULT_PROTOCOL
from .program_graph import analyse_static, build_program_graph
from .retrieval import RequirementGraphRAG
from .search import HeuristicAStarRepair, SearchBudget


@dataclass(frozen=True)
class PipelineConfig:
    top_n: int = 20
    top_k: int = 3
    graph_retries: int = 2
    code_regenerations: int = 2
    enable_repair: bool = True
    enable_fallback_replan: bool = True
    resume_completed: bool = True
    strict_iou_threshold: float = 0.50
    strict_f1_threshold: float = 0.50
    repair_budget: SearchBudget = field(default_factory=SearchBudget)
    algorithm_version: str = "dag_graph_rag_astar_20261120_v60_asymmetric_anchor_dual_gate"
    evaluation_protocol_version: str = "solid_surface_protocol_v3"


def _safe_task_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    if not safe:
        raise ValueError("task_id_has_no_safe_characters")
    return safe[:120]


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
        if not handle:
            return False
        try:
            return kernel32.WaitForSingleObject(handle, 0) == 0x00000102  # WAIT_TIMEOUT
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _program_graph(code: str | None) -> dict:
    if not code:
        return {"nodes": [], "output": None, "output_nodes": []}
    try:
        return build_program_graph(code)
    except SyntaxError:
        return {"nodes": [], "output": None, "output_nodes": []}


def _consistency(code: str | None, requirement_graph: dict) -> tuple[float, dict]:
    program_graph = _program_graph(code)
    gap = graph_difference(requirement_graph, program_graph)
    return graph_consistency(requirement_graph, program_graph), gap


def select_initial_program(
    programs: dict[str, str | None], executions: dict[str, dict], requirement_graph: dict,
    validations: dict[str, dict] | None = None,
) -> tuple[str, dict]:
    validations = validations or {
        branch: validate_requirement_constraints(programs[branch] or "", executions[branch], requirement_graph)
        for branch in ("P0", "PR")
    }
    rows: dict[str, dict] = {}
    for branch in ("P0", "PR"):
        code = programs.get(branch)
        static = analyse_static(code) if code else None
        consistency, gap = _consistency(code, requirement_graph)
        rows[branch] = {
            "available": bool(code),
            "fatal_static_count": len(static.fatal) if static else 1_000_000,
            "countable_static_count": len(static.countable) if static else 1_000_000,
            "execution_success": bool(executions[branch].get("success")),
            "constraint_accepted": bool(validations[branch].get("accepted")),
            "constraint_issues": validations[branch].get("issues", []),
            "graph_consistency": consistency, "graph_gap": gap,
        }
    selected = min(
        ("P0", "PR"),
        key=lambda branch: (
            0 if rows[branch]["available"] else 1,
            rows[branch]["fatal_static_count"],
            0 if rows[branch]["constraint_accepted"] else 1,
            0 if rows[branch]["execution_success"] else 1,
            -rows[branch]["graph_consistency"],
            0 if branch == "P0" else 1,
        ),
    )
    return selected, {
        "rule": "available_then_static_then_reference_free_constraints_then_execution_then_graph",
        "branches": rows,
    }


class CadQueryDAGAStarAgent:
    def __init__(
        self, *, output_dir: str | Path, client, retriever: RequirementGraphRAG,
        runtime: CadQueryRuntime | None = None, config: PipelineConfig | None = None,
    ):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or PipelineConfig()
        self.client = client
        self.runtime = runtime or CadQueryRuntime(self.output_dir / "occ_runs")
        self.graph_builder = RequirementGraphBuilder(client, max_retries=self.config.graph_retries)
        self.generator = DualCodeGenerator(client, max_regenerations=self.config.code_regenerations)
        self.retriever = retriever

    def _execute(
        self, code: str | None, task_id: str, attempt: int,
        timeout_seconds: float | None = None,
    ) -> dict:
        if not code:
            return {
                "success": False, "stage": "generation", "error_type": "ProgramUnavailable",
                "error": "branch_did_not_produce_code",
            }
        try:
            return self.runtime.execute_code(
                code, task_id=task_id, attempt=attempt, reference_stl=None,
                timeout_seconds=timeout_seconds,
            )
        except Exception as error:
            return {
                "success": False, "stage": "runtime_wrapper", "error_type": type(error).__name__,
                "error": str(error),
            }

    def _repair(
        self, text: str, task_id: str, branch: str, code: str, execution: dict,
        requirement_graph: dict, *, graph_build_issues: list[str], fallback_used: bool,
    ) -> dict:
        def execute(candidate: str, attempt: int, remaining: float | None = None) -> dict:
            return self._execute(
                candidate, f"{task_id}_{branch}_repair", attempt,
                timeout_seconds=remaining,
            )
        def provide(candidate: str, failure: dict, remaining: float | None = None):
            return self.generator.repair_candidates(
                text, candidate, failure, requirement_graph, count=2,
                timeout_seconds=remaining,
            )
        def validate(candidate: str, observed: dict, graph: dict) -> dict:
            return validate_requirement_constraints(
                candidate, observed, graph, requirement_text=text,
                graph_build_issues=graph_build_issues, fallback_used=fallback_used,
            )
        return HeuristicAStarRepair(
            execute, requirement_graph, budget=self.config.repair_budget,
            constraint_validator=validate,
        ).search(code, execution, provide).to_dict()

    @staticmethod
    def _candidate_rank(candidate: dict) -> tuple:
        validation = candidate["validation"]
        return (
            0 if validation.get("accepted") else 1,
            0 if candidate["execution"].get("success") else 1,
            -float(validation.get("structural_score", 0.0)),
            len(validation.get("issues", [])),
            candidate["source"],
        )

    def _task_dir(self, task_id: str) -> Path:
        target = (self.output_dir / _safe_task_id(task_id)).resolve()
        if self.output_dir not in target.parents:
            raise ValueError("task_path_escapes_output_dir")
        target.mkdir(parents=True, exist_ok=True)
        return target

    @staticmethod
    def _atomic_json(path: Path, value: dict) -> None:
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    @staticmethod
    def _atomic_text(path: Path, value: str) -> None:
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(value, encoding="utf-8")
        temporary.replace(path)

    @staticmethod
    def _acquire_lock(lock_path: Path) -> None:
        for _ in range(2):
            try:
                descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    owner = json.loads(lock_path.read_text(encoding="utf-8"))
                    owner_pid = int(owner.get("pid", -1))
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    owner_pid = -1
                if _pid_exists(owner_pid):
                    raise RuntimeError(f"task_already_running:{lock_path.parent}")
                lock_path.unlink(missing_ok=True)
                continue
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump({"pid": os.getpid(), "created_unix": time.time()}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            return
        raise RuntimeError(f"task_lock_unavailable:{lock_path.parent}")

    def _cache_components(self, text: str, task_id: str) -> dict:
        client_identity = (
            self.client.cache_identity() if callable(getattr(self.client, "cache_identity", None))
            else {
                "class": f"{type(self.client).__module__}.{type(self.client).__qualname__}",
                "provider": getattr(self.client, "provider", None),
                "model": getattr(self.client, "model", None),
                "version": getattr(self.client, "version", None),
            }
        )
        protocol = asdict(getattr(self.runtime, "metric_protocol", DEFAULT_PROTOCOL))
        return {
            "text": text, "task_id": task_id, "config": asdict(self.config),
            "retrieval_corpus": getattr(self.retriever, "corpus_fingerprint", None),
            "retrieval_library_version": getattr(self.retriever, "library_version", None),
            "retrieval_cache_namespace": getattr(self.retriever, "cache_namespace", None),
            "retrieval_serialization_version": getattr(
                self.retriever, "serialization_version", None,
            ),
            "retrieval_serialization_fingerprint": getattr(
                self.retriever, "serialization_fingerprint", None,
            ),
            "embedding_model": getattr(getattr(self.retriever, "encoder", None), "model_name", None),
            "embedding_revision": getattr(getattr(self.retriever, "encoder", None), "model_revision", None),
            "llm": client_identity,
            "prompt_versions": {
                "requirement_graph": REQUIREMENT_GRAPH_PROMPT_VERSION, **PROMPT_VERSIONS,
            },
            "evaluation_protocol": protocol,
        }
    def _input_hash(self, text: str, task_id: str) -> str:
        payload = self._cache_components(text, task_id)
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

    @staticmethod
    def _reference_hash(reference_stl: str | Path | None) -> str | None:
        if reference_stl is None:
            return None
        path = Path(reference_stl)
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None

    def _refresh_offline_evaluation(
        self, target: Path, previous: dict, reference_stl: str | Path | None,
        reference_hash: str | None,
    ) -> dict:
        result = dict(previous)
        offline = None
        if result.get("final_validation", {}).get("accepted") and reference_stl is not None:
            try:
                offline = self.runtime.execute_code(
                    result.get("final_code", ""), task_id=f"{result['task_id']}_offline_evaluation",
                    attempt=int(result.get("accounting", {}).get("offline_evaluation_refreshes", 0)) + 1,
                    reference_stl=reference_stl,
                )
            except Exception as error:
                offline = {
                    "success": False, "stage": "offline_evaluation",
                    "error_type": type(error).__name__, "error": str(error),
                }
        geometry = (offline or {}).get("geometry_metrics") or {}
        protocol = geometry.get("protocol") or {}
        iou, surface_f1 = geometry.get("volumetric_iou"), geometry.get("f1_score")
        iou_threshold = float(protocol.get("strict_iou_threshold", self.config.strict_iou_threshold))
        f1_threshold = float(protocol.get("strict_f1_threshold", self.config.strict_f1_threshold))
        result["offline_evaluation"] = offline
        result["evaluation_reference_sha256"] = reference_hash
        result["metrics"] = {
            **result.get("metrics", {}),
            "volumetric_iou": iou, "surface_f1": surface_f1,
            "chamfer_distance": geometry.get("chamfer_distance"),
            "hausdorff_distance": geometry.get("hausdorff_distance"),
            "hausdorff_95_distance": geometry.get("hausdorff_95_distance"),
            "surface_precision": geometry.get("precision"),
            "surface_recall": geometry.get("recall"),
            "bbox_center_error": geometry.get("bbox_center_error"),
            "bbox_extent_relative_error": geometry.get("bbox_extent_relative_error"),
            "volume_relative_error": geometry.get("volume_relative_error"),
            "strict_success": bool(
                result.get("final_validation", {}).get("accepted")
                and isinstance(iou, (int, float)) and float(iou) >= iou_threshold
                and isinstance(surface_f1, (int, float)) and float(surface_f1) >= f1_threshold
            ),
            "strict_iou_threshold": iou_threshold,
            "strict_f1_threshold": f1_threshold,
        }
        accounting = dict(result.get("accounting", {}))
        accounting["occ_executions"] = int(accounting.get("occ_executions", 0)) + int(offline is not None)
        accounting["offline_evaluation_refreshes"] = int(accounting.get("offline_evaluation_refreshes", 0)) + 1
        result["accounting"] = accounting
        result["resumed"] = True
        self._atomic_json(target / "run.json", result)
        return result

    def run(self, text: str, *, task_id: str, reference_stl: str | Path | None = None) -> dict:
        target = self._task_dir(task_id)
        input_hash = self._input_hash(text, task_id)
        reference_hash = self._reference_hash(reference_stl)
        result_path, status_path, lock_path = target / "run.json", target / "status.json", target / ".lock"
        if self.config.resume_completed and result_path.is_file():
            previous = json.loads(result_path.read_text(encoding="utf-8"))
            if previous.get("input_hash") == input_hash and previous.get("status") == "completed":
                if previous.get("evaluation_reference_sha256") == reference_hash:
                    previous["resumed"] = True
                    return previous
                self._acquire_lock(lock_path)
                try:
                    return self._refresh_offline_evaluation(
                        target, previous, reference_stl, reference_hash,
                    )
                finally:
                    lock_path.unlink(missing_ok=True)
        self._acquire_lock(lock_path)
        self._atomic_json(status_path, {
            "status": "running", "task_id": task_id, "input_hash": input_hash,
            "started_unix": time.time(),
        })
        try:
            result = self._run_once(text, task_id=task_id, reference_stl=reference_stl)
            result.update({
                "input_hash": input_hash, "evaluation_reference_sha256": reference_hash,
                "status": "completed", "resumed": False,
            })
            self._write_artifacts(target, result)
            self._atomic_json(status_path, {
                "status": "completed", "task_id": task_id, "input_hash": input_hash,
                "success": result["success"], "finished_unix": time.time(),
            })
            return result
        except Exception as error:
            self._atomic_json(status_path, {
                "status": "failed", "task_id": task_id, "input_hash": input_hash,
                "error_type": type(error).__name__, "error": str(error),
                "finished_unix": time.time(),
            })
            raise
        finally:
            lock_path.unlink(missing_ok=True)

    def _run_once(self, text: str, *, task_id: str, reference_stl: str | Path | None) -> dict:
        started = time.perf_counter()
        graph_result = self.graph_builder.build(text)
        if not graph_result.valid:
            raise ValueError(
                "invalid_requirement_graph_after_fallback:"
                + "|".join(graph_result.fallback_validation_issues)
            )
        requirement_graph = graph_result.graph
        try:
            retrieval = self.retriever.retrieve(
                text, requirement_graph, top_n=self.config.top_n, top_k=self.config.top_k,
            )
        except Exception as error:
            retrieval = {
                "success": False, "strategy": "failed", "selected": [],
                "selected_case_ids": [], "top_n": 0, "top_k": 0,
                "error_type": type(error).__name__, "error": str(error),
            }
        blueprints = [hit["blueprint"] for hit in retrieval.get("selected", [])]
        retrieval_success = bool(retrieval.get("success") and blueprints)
        generated = self.generator.generate_pair(
            text, blueprints, requirement_graph=requirement_graph,
            rag_available=retrieval_success,
        )
        programs = {branch: generated[branch].code for branch in ("P0", "PR")}
        executions = {
            branch: self._execute(programs[branch], f"{task_id}_{branch.lower()}", 0)
            for branch in ("P0", "PR")
        }
        validations = {
            branch: validate_requirement_constraints(
                programs[branch] or "", executions[branch], requirement_graph,
                requirement_text=text,
                graph_build_issues=graph_result.fallback_validation_issues,
                fallback_used=graph_result.fallback_used,
            ) for branch in ("P0", "PR")
        }
        selected, selection = select_initial_program(programs, executions, requirement_graph, validations)
        alternate = "PR" if selected == "P0" else "P0"
        candidates: list[dict] = [
            {"source": branch, "code": programs[branch], "execution": executions[branch],
             "validation": validations[branch]}
            for branch in ("P0", "PR") if programs[branch]
        ]
        repairs: dict[str, dict] = {}

        repair_order = () if validations[selected].get("accepted") else (selected, alternate)
        for branch in repair_order:
            code = programs.get(branch)
            if not code or validations[branch].get("accepted") or not self.config.enable_repair:
                continue
            repairs[branch] = self._repair(
                text, task_id, branch, code, executions[branch], requirement_graph,
                graph_build_issues=graph_result.fallback_validation_issues,
                fallback_used=graph_result.fallback_used,
            )
            repaired = repairs[branch]
            candidates.append({
                "source": f"{branch}_repair", "code": repaired["code"],
                "execution": repaired["execution"], "validation": repaired["acceptance"],
            })
            if repaired.get("success"):
                break

        if candidates:
            best = min(candidates, key=self._candidate_rank)
        else:
            best = {
                "source": "none", "code": None,
                "execution": {"success": False, "error_type": "NoProgram", "error": "no_branch_produced_code"},
                "validation": {"accepted": False, "issues": ["no_program"], "structural_score": 0.0},
            }

        replan = None
        if not best["validation"].get("accepted") and best["code"] and self.config.enable_fallback_replan:
            try:
                replanned, record = self.generator.replan(
                    text, best["code"], best["execution"], requirement_graph,
                )
                replanned_execution = self._execute(replanned, f"{task_id}_replan", 0)
                replanned_validation = validate_requirement_constraints(
                    replanned, replanned_execution, requirement_graph,
                    requirement_text=text,
                    graph_build_issues=graph_result.fallback_validation_issues,
                    fallback_used=graph_result.fallback_used,
                )
                replan = {
                    "code": replanned, "llm_call": record, "execution": replanned_execution,
                    "validation": replanned_validation,
                }
                candidates.append({
                    "source": "fallback_replan", "code": replanned,
                    "execution": replanned_execution, "validation": replanned_validation,
                })
                best = min(candidates, key=self._candidate_rank)
            except Exception as error:
                replan = {
                    "error_type": type(error).__name__, "error": str(error),
                    "attempts": list(getattr(error, "attempts", [])),
                }

        final_code = best["code"] or ""
        final_execution = best["execution"]
        final_validation = best["validation"]
        offline_evaluation = None
        if final_validation.get("accepted") and reference_stl is not None:
            try:
                offline_evaluation = self.runtime.execute_code(
                    final_code, task_id=f"{task_id}_offline_evaluation", attempt=0,
                    reference_stl=reference_stl,
                )
            except Exception as error:
                offline_evaluation = {
                    "success": False, "stage": "offline_evaluation",
                    "error_type": type(error).__name__, "error": str(error),
                }

        call_records = [*graph_result.calls]
        for branch in ("P0", "PR"):
            call_records.extend(generated[branch].calls)
        for repair in repairs.values():
            call_records.extend(
                event["llm_record"] for event in repair.get("trace", [])
                if event.get("event") == "expand" and isinstance(event.get("llm_record"), dict)
            )
            call_records.extend(
                {"success": False, **event} for event in repair.get("trace", [])
                if event.get("event") == "provider_error"
            )
        if isinstance(replan, dict) and isinstance(replan.get("llm_call"), dict):
            call_records.append(replan["llm_call"])
        geometry = ((offline_evaluation or {}).get("geometry_metrics") or {})
        iou, surface_f1 = geometry.get("volumetric_iou"), geometry.get("f1_score")
        protocol = geometry.get("protocol") or {}
        strict_iou_threshold = float(protocol.get(
            "strict_iou_threshold", self.config.strict_iou_threshold,
        ))
        strict_f1_threshold = float(protocol.get(
            "strict_f1_threshold", self.config.strict_f1_threshold,
        ))
        strict_success = bool(
            final_validation.get("accepted")
            and isinstance(iou, (int, float)) and float(iou) >= strict_iou_threshold
            and isinstance(surface_f1, (int, float)) and float(surface_f1) >= strict_f1_threshold
        )
        candidate_executions = sum(int(repair.get("candidate_executions", 0)) for repair in repairs.values())
        occ_executions = sum(int(programs[branch] is not None) for branch in ("P0", "PR")) + candidate_executions
        occ_executions += int(bool(replan and replan.get("execution"))) + int(offline_evaluation is not None)
        transport_attempts = sum(max(1, len(call.get("attempts") or [])) for call in call_records)

        return {
            "task_id": task_id, "text": text,
            "success": bool(final_validation.get("accepted")),
            "G_req": requirement_graph, "Top3_blueprints": blueprints,
            "P0": programs["P0"], "PR": programs["PR"],
            "P_init": programs[selected], "P_init_branch": selected,
            "requirement_graph": requirement_graph,
            "requirement_graph_build": {
                "fallback_used": graph_result.fallback_used, "issues": graph_result.issues,
                "fallback_validation_issues": graph_result.fallback_validation_issues,
                "llm_calls": graph_result.calls,
            },
            "retrieval": retrieval, "retrieval_success": retrieval_success,
            "rag_used": retrieval_success,
            "actual_rag_case_ids": retrieval.get("selected_case_ids", []) if retrieval_success else [],
            "blueprints": blueprints,
            "programs": {
                branch: {
                    "code": generated[branch].code, "llm_calls": generated[branch].calls,
                    "static_issues": generated[branch].static_issues,
                    "generation_error": generated[branch].error,
                    "execution": executions[branch], "validation": validations[branch],
                } for branch in ("P0", "PR")
            },
            "initial_selection": {"selected": selected, **selection},
            "repairs": repairs, "fallback_replan": replan,
            "candidate_summaries": [
                {"source": item["source"], "accepted": item["validation"].get("accepted"),
                 "execution_success": item["execution"].get("success"),
                 "structural_score": item["validation"].get("structural_score"),
                 "issues": item["validation"].get("issues", [])}
                for item in candidates
            ],
            "final_source": best["source"], "final_code": final_code,
            "final_program_graph": _program_graph(final_code),
            "final_execution": final_execution, "final_validation": final_validation,
            "offline_evaluation": offline_evaluation,
            "metrics": {
                "execution_success": bool(final_execution.get("success")),
                "requirement_accepted": bool(final_validation.get("accepted")),
                "volumetric_iou": iou, "surface_f1": surface_f1,
                "chamfer_distance": geometry.get("chamfer_distance"),
                "hausdorff_distance": geometry.get("hausdorff_distance"),
                "hausdorff_95_distance": geometry.get("hausdorff_95_distance"),
                "surface_precision": geometry.get("precision"),
                "surface_recall": geometry.get("recall"),
                "bbox_center_error": geometry.get("bbox_center_error"),
                "bbox_extent_relative_error": geometry.get("bbox_extent_relative_error"),
                "volume_relative_error": geometry.get("volume_relative_error"),
                "strict_success": strict_success,
                "strict_iou_threshold": strict_iou_threshold,
                "strict_f1_threshold": strict_f1_threshold,
            },
            "accounting": {
                "logical_llm_records": len(call_records), "transport_attempts": transport_attempts,
                "prompt_tokens": sum(int(call.get("prompt_tokens", 0)) for call in call_records),
                "completion_tokens": sum(int(call.get("completion_tokens", 0)) for call in call_records),
                "total_tokens": sum(int(call.get("total_tokens", 0)) for call in call_records),
                "llm_latency_seconds": sum(float(call.get("latency_s", 0.0)) for call in call_records),
                "occ_executions": occ_executions,
            },
            "runtime_seconds": time.perf_counter() - started,
            "config": asdict(self.config),
            "cache_fingerprint_components": self._cache_components(text, task_id),
        }

    def _write_artifacts(self, target: Path, result: dict) -> None:
        self._atomic_json(target / "requirement_graph.json", result["requirement_graph"])
        self._atomic_json(target / "blueprints.json", {"blueprints": result["blueprints"]})
        for branch in ("P0", "PR"):
            code = result["programs"][branch]["code"]
            if code:
                self._atomic_text(target / f"{branch.lower()}.py", code)
        if result["final_code"]:
            self._atomic_text(target / "final.py", result["final_code"])
        self._atomic_json(target / "run.json", result)
