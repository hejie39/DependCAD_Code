"""BGE-M3 recall plus role-aware node correspondence and topology reranking."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np

from .graphs import (
    canonical_anchor_graph,
    graph_counters,
    graph_sha256,
    validate_requirement_graph,
)
from .operations import canonical_operation, is_semantic_operation, is_transparent_operation


class Encoder(Protocol):
    dimension: int
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class BGEM3Encoder:
    def __init__(
        self, device: str | None = None, batch_size: int = 8, *,
        model_name: str = "BAAI/bge-m3", revision: str | None = None,
    ):
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError("BGE-M3 requires sentence-transformers and torch") from error
        self.model_name = model_name
        self.model_revision = revision
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.model = SentenceTransformer(
            self.model_name, device=self.device, revision=self.model_revision,
            trust_remote_code=False,
        )
        getter = getattr(self.model, "get_embedding_dimension", None)
        self.dimension = int(getter() if getter else self.model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.asarray(self.model.encode(
            list(texts), batch_size=self.batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        ), dtype=np.float32)
        if str(self.device).startswith("cuda"):
            # SentenceTransformer can otherwise retain the temporary attention
            # workspace for a long corpus and starve the colocated CAD LLM.
            import torch
            torch.cuda.empty_cache()
        return vectors


class NumpyFlatIP:
    """Exact IP index for unit tests; production uses FAISS IndexFlatIP."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.vectors = np.empty((0, dimension), dtype=np.float32)

    def add(self, vectors: np.ndarray) -> None:
        self.vectors = np.asarray(vectors, dtype=np.float32)

    @property
    def ntotal(self) -> int:
        return int(len(self.vectors))

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        scores = np.asarray(queries, dtype=np.float32) @ self.vectors.T
        order = np.argsort(-scores, axis=1)[:, :k]
        return np.take_along_axis(scores, order, axis=1), order


DEFAULT_WEIGHTS = {
    "vector": 0.40,
    "operation": 0.20,
    "dependency": 0.15,
    "topology": 0.25,
}
DEFAULT_SERIALIZATION_VERSION = "text_plus_requirement_dag_v1"
STRUCTURE_FIRST_SERIALIZATION_VERSION = "structure_first_compact_text_v1"
SERIALIZATION_VERSIONS = (
    DEFAULT_SERIALIZATION_VERSION,
    STRUCTURE_FIRST_SERIALIZATION_VERSION,
)
_SERIALIZATION_NUMBER_RE = re.compile(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?", re.I,
)
_SERIALIZATION_VARIABLE_RE = re.compile(r"`[^`]+`")
_SERIALIZATION_SPACE_RE = re.compile(r"\s+")
_SERIALIZATION_FIXED_PREFIXES = (
    "build the cad model exactly",
    "use every value as written",
    "overall dimensions",
    "overall center",
    "ordered construction",
)
FORBIDDEN_CASE_KEYS = {
    "reference_stl", "reference_step", "target_stl", "target_step",
    "geometry_metrics", "volumetric_iou", "surface_f1", "chamfer_distance",
    "strict_success", "strict_match",
}


def _normalise(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def _multiset_jaccard(left: Counter, right: Counter) -> float:
    keys = set(left) | set(right)
    return 1.0 if not keys else sum(min(left[key], right[key]) for key in keys) / sum(
        max(left[key], right[key]) for key in keys
    )


def _multiset_f1(left: Counter, right: Counter) -> float:
    overlap = sum((left & right).values())
    if not left and not right:
        return 1.0
    precision = overlap / max(sum(right.values()), 1)
    recall = overlap / max(sum(left.values()), 1)
    return 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)


def _tokens(text: str) -> set[str]:
    return set(re.findall(
        r"[A-Za-z_][A-Za-z0-9_]*|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?|[\u4e00-\u9fff]",
        text.lower(),
    ))


def _text_jaccard(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    return len(a & b) / max(len(a | b), 1)


def _token_jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / max(len(left | right), 1)


def _tokens_meet_jaccard_threshold(
    left: set[str], right: set[str], threshold: float,
) -> bool:
    """Exact Jaccard test with a cardinality upper-bound rejection."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("jaccard_threshold_out_of_range")
    if not left or not right:
        return _token_jaccard(left, right) >= threshold
    smaller, larger = (left, right) if len(left) <= len(right) else (right, left)
    if len(smaller) / len(larger) < threshold:
        return False
    intersection = len(smaller & larger)
    return intersection / (len(left) + len(right) - intersection) >= threshold


def structure_blueprint(graph: dict) -> dict:
    return {
        "nodes": [
            {**{"id": str(node["id"]), "op": str(node["op"]), "inputs": list(node.get("inputs") or [])},
             **({"roles": list(node["roles"])} if node.get("roles") else {})}
            for node in graph.get("nodes") or []
        ],
        "output": graph.get("output"),
    }


def serialise_query(text: str, graph: dict) -> str:
    return text.strip() + "\n<requirement_dag>\n" + json.dumps(
        structure_blueprint(graph), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _normalise_retrieval_evidence(value: str) -> str:
    value = _SERIALIZATION_VARIABLE_RE.sub(" ", value)
    value = _SERIALIZATION_NUMBER_RE.sub(" ", value)
    value = value.replace("Create or update", " ").replace("Set", " ")
    return _SERIALIZATION_SPACE_RE.sub(" ", value).strip(" .;:,()")


def serialise_structure_first_query(text: str, graph: dict) -> str:
    """Serialise the number-free DAG before non-boilerplate operation evidence.

    This variant is intentionally retrieval-only: numeric dimensions and
    coordinates remain in the original requirement passed to code generation,
    but cannot dominate BGE-M3 case recall.
    """
    operations = [
        str(node.get("op") or "unknown_op") for node in graph.get("nodes") or []
    ]
    operation_tokens = {operation.lower() for operation in operations}
    evidence: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.lower().startswith(_SERIALIZATION_FIXED_PREFIXES):
            continue
        for clause in re.split(r"[;.]", line):
            normalised = _normalise_retrieval_evidence(clause)
            lowered = normalised.lower()
            if normalised and any(token in lowered for token in operation_tokens):
                evidence.append(normalised)
    for node in graph.get("nodes") or []:
        span = _normalise_retrieval_evidence(str(node.get("span") or ""))
        if span:
            evidence.append(span)
    structure = {
        "operations": operations,
        **structure_blueprint(graph),
    }
    return "<requirement_dag>\n" + json.dumps(
        structure, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ) + "\n<text_evidence>\n" + " | ".join(dict.fromkeys(evidence))


def query_serializer(version: str) -> Callable[[str, dict], str]:
    serializers = {
        DEFAULT_SERIALIZATION_VERSION: serialise_query,
        STRUCTURE_FIRST_SERIALIZATION_VERSION: serialise_structure_first_query,
    }
    try:
        return serializers[str(version)]
    except KeyError as error:
        raise ValueError(f"unknown_retrieval_serialization:{version}") from error


def _topology(graph: dict) -> tuple[dict[str, str], list[tuple[str, str, str]]]:
    raw_nodes = graph.get("nodes") or []
    by_id = {str(node.get("id")): node for node in raw_nodes}
    op_by_id = {
        node_id: canonical_operation(node.get("op") or ((node.get("operations") or ["unknown_op"])[-1]))
        for node_id, node in by_id.items()
    }

    def resolve(node_id: str, seen: set[str] | None = None) -> list[str]:
        seen = set(seen or set())
        if node_id in seen or node_id not in by_id:
            return []
        seen.add(node_id)
        if not is_transparent_operation(op_by_id[node_id]):
            return [node_id]
        values: list[str] = []
        for parent in by_id[node_id].get("inputs") or []:
            values.extend(resolve(str(parent), seen))
        return list(dict.fromkeys(values))

    nodes = {
        node_id: op for node_id, op in op_by_id.items()
        if is_semantic_operation(op) or op == "unknown_op"
    }
    edges: list[tuple[str, str, str]] = []
    for child_id, child_op in op_by_id.items():
        if child_id not in nodes:
            continue
        roles = by_id[child_id].get("roles") or []
        for index, parent in enumerate(by_id[child_id].get("inputs") or []):
            role = str(roles[index]) if index < len(roles) else "input"
            for source_id in resolve(str(parent)):
                if source_id in nodes:
                    edges.append((source_id, child_id, role))
    return nodes, edges


def _node_signature(node_id: str, nodes: dict[str, str], edges: list[tuple[str, str, str]]) -> Counter:
    signature: Counter = Counter()
    for source, target, role in edges:
        if target == node_id:
            signature[("in", nodes.get(source, "unknown_op"), role)] += 1
        if source == node_id:
            signature[("out", nodes.get(target, "unknown_op"), role)] += 1
    return signature


def _node_correspondence(left: dict, right: dict) -> tuple[dict[str, str], float]:
    left_nodes, left_edges = _topology(left)
    right_nodes, right_edges = _topology(right)
    candidates: list[tuple[float, str, str]] = []
    for left_id, left_op in left_nodes.items():
        for right_id, right_op in right_nodes.items():
            if left_op != right_op:
                continue
            score = _multiset_jaccard(
                _node_signature(left_id, left_nodes, left_edges),
                _node_signature(right_id, right_nodes, right_edges),
            )
            candidates.append((score, left_id, right_id))
    mapping: dict[str, str] = {}
    used_right: set[str] = set()
    for score, left_id, right_id in sorted(candidates, key=lambda item: (-item[0], item[1], item[2])):
        if left_id not in mapping and right_id not in used_right:
            mapping[left_id] = right_id
            used_right.add(right_id)
    node_recall = len(mapping) / max(len(left_nodes), 1)
    node_precision = len(mapping) / max(len(right_nodes), 1)
    node_f1 = 0.0 if node_recall + node_precision == 0 else (
        2.0 * node_recall * node_precision / (node_recall + node_precision)
    )
    return mapping, node_f1


def _path_counter(graph: dict) -> Counter:
    nodes, edges = _topology(graph)
    outgoing: dict[str, list[tuple[str, str]]] = {}
    for source, target, role in edges:
        outgoing.setdefault(source, []).append((target, role))
    paths: Counter = Counter()
    for source, middle, first_role in edges:
        for target, second_role in outgoing.get(middle, []):
            paths[(nodes[source], first_role, nodes[middle], second_role, nodes[target])] += 1
    return paths


def topology_similarity(left: dict, right: dict) -> float:
    left_nodes, left_edges = _topology(left)
    right_nodes, right_edges = _topology(right)
    mapping, node_f1 = _node_correspondence(left, right)
    mapped_left = Counter(
        (mapping[source], mapping[target], role)
        for source, target, role in left_edges if source in mapping and target in mapping
    )
    right_counter = Counter(right_edges)
    edge_f1 = _multiset_f1(mapped_left, right_counter)
    path_score = _multiset_jaccard(_path_counter(left), _path_counter(right))
    return 0.25 * node_f1 + 0.50 * edge_f1 + 0.25 * path_score


@dataclass(frozen=True)
class RetrievalHit:
    case_id: str
    rank: int
    score: float
    components: dict[str, float]
    blueprint: dict


class RequirementGraphRAG:
    """Immutable successful-case index with explicit leakage boundaries."""

    def __init__(
        self, cases: Sequence[dict], *, encoder: Encoder | None = None, index=None,
        weights: dict[str, float] | None = None, excluded_case_ids: set[str] | None = None,
        excluded_model_ids: set[str] | None = None, excluded_geometry_hashes: set[str] | None = None,
        excluded_texts: Sequence[str] = (), near_duplicate_threshold: float = 0.92,
        cache_namespace: str = "default",
        library_version: str | None = None,
        embedding_cache_dir: str | Path | None = None,
        serializer: Callable[[str, dict], str] | None = None,
        serialization_version: str = DEFAULT_SERIALIZATION_VERSION,
    ):
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        if not str(serialization_version).strip():
            raise ValueError("retrieval_serialization_version_required")
        self.serializer = serializer or serialise_query
        self.serialization_version = str(serialization_version).strip()
        self.serialization_fingerprint = hashlib.sha256(
            self.serialization_version.encode("utf-8")
        ).hexdigest()
        if set(self.weights) != set(DEFAULT_WEIGHTS) or not math.isclose(sum(self.weights.values()), 1.0, abs_tol=1e-8):
            raise ValueError("retrieval_weights_must_match_components_and_sum_to_one")
        excluded_ids = {str(value) for value in (excluded_case_ids or set())}
        excluded_models = {str(value) for value in (excluded_model_ids or set())}
        excluded_geometry = {str(value) for value in (excluded_geometry_hashes or set()) if value}
        filtered: list[dict] = []
        seen_ids: set[str] = set()
        seen_models: set[str] = set()
        seen_geometry: set[str] = set()
        excluded_text_tokens = [_tokens(str(value)) for value in excluded_texts if str(value).strip()]
        for raw in cases:
            case = dict(raw)
            prohibited = FORBIDDEN_CASE_KEYS & set(case)
            if prohibited:
                raise ValueError(f"retrieval_case_contains_evaluation_data:{sorted(prohibited)}")
            case_id = str(case.get("case_id") or case.get("task_id") or "")
            model_id = str(case.get("model_id") or "")
            geometry_hash = str(case.get("geometry_hash") or "")
            text = str(case.get("text") or case.get("requirement") or "")
            text_tokens = _tokens(text)
            graph = case.get("graph") or case.get("requirement_graph") or case.get("dag")
            if (
                not case_id or case_id in excluded_ids or model_id in excluded_models
                or (geometry_hash and geometry_hash in excluded_geometry)
                or case_id in seen_ids or (model_id and model_id in seen_models)
                or (geometry_hash and geometry_hash in seen_geometry)
                or not text or case.get("execution_success") is not True
                or validate_requirement_graph(graph or {})
            ):
                continue
            if any(
                _tokens_meet_jaccard_threshold(
                    text_tokens, excluded, near_duplicate_threshold,
                )
                for excluded in excluded_text_tokens
            ):
                continue
            filtered.append({
                "case_id": case_id, "model_id": model_id, "geometry_hash": geometry_hash,
                "text": text, "graph": graph, "execution_success": True,
                "library_version": str(case.get("library_version") or library_version or "unversioned"),
            })
            seen_ids.add(case_id)
            if model_id:
                seen_models.add(model_id)
            if geometry_hash:
                seen_geometry.add(geometry_hash)
        if not filtered:
            raise ValueError("retrieval_corpus_empty_after_leakage_filter")
        self.cases = tuple(filtered)
        versions = {case["library_version"] for case in self.cases}
        if len(versions) != 1:
            raise ValueError(f"mixed_retrieval_library_versions:{sorted(versions)}")
        self.library_version = next(iter(versions))
        self.encoder = encoder or BGEM3Encoder()
        fingerprint_payload = [
            (case["case_id"], case["model_id"], case["geometry_hash"],
             graph_sha256(case["graph"]),
             hashlib.sha256(case["text"].encode("utf-8")).hexdigest())
            for case in self.cases
        ]
        self.corpus_fingerprint = hashlib.sha256(json.dumps(
            fingerprint_payload, ensure_ascii=False, sort_keys=True,
        ).encode()).hexdigest()
        encoder_identity = {
            "model": getattr(self.encoder, "model_name", type(self.encoder).__name__),
            "revision": getattr(self.encoder, "model_revision", None),
            "dimension": int(getattr(self.encoder, "dimension", 0)),
        }
        self.encoder_fingerprint = hashlib.sha256(json.dumps(
            encoder_identity, ensure_ascii=False, sort_keys=True,
        ).encode()).hexdigest()
        self.cache_namespace = (
            f"{cache_namespace}:{self.encoder_fingerprint}:"
            f"{self.serialization_fingerprint}:{self.corpus_fingerprint}"
        )
        embeddings: np.ndarray | None = None
        cache_path: Path | None = None
        if embedding_cache_dir is not None:
            cache_root = Path(embedding_cache_dir)
            cache_root.mkdir(parents=True, exist_ok=True)
            cache_key = hashlib.sha256(self.cache_namespace.encode()).hexdigest()
            cache_path = cache_root / f"{cache_key}.npy"
            if cache_path.is_file():
                try:
                    cached = np.load(cache_path, allow_pickle=False)
                    if cached.shape == (len(self.cases), self.encoder.dimension):
                        embeddings = _normalise(np.asarray(cached, dtype=np.float32))
                except (OSError, ValueError):
                    embeddings = None
        if embeddings is None:
            embeddings = _normalise(np.asarray(
                self.encoder.encode([
                    self.serializer(case["text"], case["graph"])
                    for case in self.cases
                ]),
                dtype=np.float32,
            ))
            if cache_path is not None:
                temporary = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
                with temporary.open("wb") as handle:
                    np.save(handle, embeddings, allow_pickle=False)
                    handle.flush()
                    os.fsync(handle.fileno())
                temporary.replace(cache_path)
        if index is None:
            try:
                import faiss
            except ImportError as error:
                raise RuntimeError("RequirementGraphRAG requires faiss-cpu") from error
            index = faiss.IndexFlatIP(int(embeddings.shape[1]))
        if int(getattr(index, "ntotal", 0)) != 0:
            raise ValueError("retrieval_index_must_be_empty")
        if int(getattr(index, "d", getattr(index, "dimension", embeddings.shape[1]))) != int(embeddings.shape[1]):
            raise ValueError("retrieval_index_dimension_mismatch")
        self.index = index
        self.index.add(embeddings)
        self.features = tuple(
            graph_counters(canonical_anchor_graph(case["graph"]))
            for case in self.cases
        )

    def retrieve(self, text: str, graph: dict, *, top_n: int = 20, top_k: int = 3) -> dict:
        if top_n <= 0 or top_k <= 0:
            raise ValueError("top_n_and_top_k_must_be_positive")
        if top_k > top_n:
            raise ValueError("top_k_must_not_exceed_top_n")
        started = time.perf_counter()
        query_serialised = self.serializer(text, graph)
        query = _normalise(np.asarray(self.encoder.encode([query_serialised]), dtype=np.float32))
        count = min(top_n, len(self.cases))
        scores, indices = self.index.search(query, count)
        query_pattern = canonical_anchor_graph(graph)
        query_ops, query_deps = graph_counters(query_pattern)
        hits: list[RetrievalHit] = []
        for dense, raw_index in zip(scores[0], indices[0]):
            index = int(raw_index)
            if index < 0:
                continue
            case = self.cases[index]
            candidate_ops, candidate_deps = self.features[index]
            components = {
                "vector": max(0.0, min(1.0, (float(dense) + 1.0) / 2.0)),
                "operation": _multiset_jaccard(query_ops, candidate_ops),
                "dependency": _multiset_jaccard(query_deps, candidate_deps),
                "topology": topology_similarity(
                    query_pattern, canonical_anchor_graph(case["graph"]),
                ),
            }
            score = sum(self.weights[key] * components[key] for key in self.weights)
            hits.append(RetrievalHit(
                case["case_id"], 0, score, components, structure_blueprint(case["graph"]),
            ))
        hits.sort(key=lambda hit: (-hit.score, hit.case_id))
        selected = [RetrievalHit(hit.case_id, rank, hit.score, hit.components, hit.blueprint)
                    for rank, hit in enumerate(hits[:top_k], 1)]
        return {
            "success": bool(selected),
            "strategy": "bge_m3_top20_canonical_anchor_topology_rerank",
            "model": getattr(self.encoder, "model_name", type(self.encoder).__name__),
            "model_revision": getattr(self.encoder, "model_revision", None),
            "device": getattr(self.encoder, "device", None),
            "encoder_fingerprint": self.encoder_fingerprint,
            "serialization_version": self.serialization_version,
            "serialization_fingerprint": self.serialization_fingerprint,
            "query_sha256": hashlib.sha256(query_serialised.encode()).hexdigest(),
            "corpus_fingerprint": self.corpus_fingerprint,
            "library_version": self.library_version,
            "cache_namespace": self.cache_namespace,
            "top_n": count, "top_k": len(selected), "weights": dict(self.weights),
            "recalled": [asdict(hit) for hit in hits],
            "selected": [asdict(hit) for hit in selected],
            "selected_case_ids": [hit.case_id for hit in selected],
            "latency_ms": (time.perf_counter() - started) * 1000.0,
        }
