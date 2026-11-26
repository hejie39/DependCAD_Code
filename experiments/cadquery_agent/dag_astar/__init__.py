"""Shared requirement-DAG Graph-RAG and heuristic A* CadQuery agent."""

from __future__ import annotations

from importlib import import_module
from typing import Any


__all__ = [
    "CadQueryDAGAStarAgent", "GreedyRetryRepair", "HeuristicAStarRepair", "PipelineConfig",
    "RequirementGraphBuilder", "RequirementGraphRAG", "SearchBudget",
    "build_program_graph", "graph_difference", "validate_requirement_graph",
]

_EXPORTS = {
    "CadQueryDAGAStarAgent": (".pipeline", "CadQueryDAGAStarAgent"),
    "PipelineConfig": (".pipeline", "PipelineConfig"),
    "RequirementGraphBuilder": (".graphs", "RequirementGraphBuilder"),
    "graph_difference": (".graphs", "graph_difference"),
    "validate_requirement_graph": (".graphs", "validate_requirement_graph"),
    "build_program_graph": (".program_graph", "build_program_graph"),
    "RequirementGraphRAG": (".retrieval", "RequirementGraphRAG"),
    "HeuristicAStarRepair": (".search", "HeuristicAStarRepair"),
    "GreedyRetryRepair": (".search", "GreedyRetryRepair"),
    "SearchBudget": (".search", "SearchBudget"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
