"""Artifact-backed Markdown report generation."""

from typing import Any

__all__ = [
    "ReportInputs",
    "generate_reports",
    "generate_summary_report",
    "generate_weekly_discussion",
    "load_metric_summary",
    "metric_markdown_table",
]


def __getattr__(name: str) -> Any:
    """Keep report CLI module unloaded until its API is requested."""

    if name in {"load_metric_summary", "metric_markdown_table"}:
        from . import tables

        return getattr(tables, name)
    if name in {
        "ReportInputs",
        "generate_reports",
        "generate_summary_report",
        "generate_weekly_discussion",
    }:
        from . import report

        return getattr(report, name)
    raise AttributeError(name)
