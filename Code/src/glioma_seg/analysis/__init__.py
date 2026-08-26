"""Evidence-based ranking and failure-pattern analysis."""

from typing import Any

__all__ = [
    "FailureAnalysisConfig",
    "FailureEvidence",
    "RankedCase",
    "classify_case_failures",
    "rank_worst_cases",
    "select_representative_cases",
]


def __getattr__(name: str) -> Any:
    """Keep executable submodules unloaded until their public API is requested."""

    if name in {"RankedCase", "rank_worst_cases", "select_representative_cases"}:
        from . import case_ranking

        return getattr(case_ranking, name)
    if name in {"FailureAnalysisConfig", "FailureEvidence", "classify_case_failures"}:
        from . import failure_analysis

        return getattr(failure_analysis, name)
    raise AttributeError(name)
