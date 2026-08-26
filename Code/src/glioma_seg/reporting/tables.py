"""Strict ET/TC/WT table formatting with validity denominators."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from glioma_seg.evaluation.regions import REGION_ORDER


def load_metric_summary(path: str | Path) -> list[dict[str, str]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Metric summary does not exist: {source}")
    with source.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Metric summary is empty: {source}")
    required = {"metric", "ET", "TC", "WT"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Metric summary {source} is missing columns: {sorted(missing)}")
    return rows


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _count(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _cell(row: Mapping[str, Any], region: str, decimals: int) -> str:
    value = _number(row.get(region))
    valid = _count(row.get(f"{region}_n_valid"))
    total = _count(row.get("total_cases"))
    denominator = ""
    if valid is not None and total is not None:
        denominator = f" (n={valid}/{total})"
    if not np.isfinite(value):
        return f"undefined{denominator}"
    return f"{value:.{decimals}f}{denominator}"


def metric_markdown_table(
    rows: Sequence[Mapping[str, Any]], *, title_prefix: str = "", decimals: int = 4
) -> str:
    """Format semantic metric rows in canonical ET, TC, WT order."""

    by_name = {str(row.get("metric", "")).strip().lower(): row for row in rows}
    dice = by_name.get("dice")
    hd95 = by_name.get("hd95") or by_name.get("hd95_mm")
    if dice is None or hd95 is None:
        raise ValueError("Metric summary must contain both Dice and HD95 rows")
    label_prefix = f"{title_prefix} " if title_prefix else ""
    header = "| Metric | " + " | ".join(REGION_ORDER) + " |"
    separator = "|---|" + "---:|" * len(REGION_ORDER)
    dice_cells = " | ".join(_cell(dice, region, decimals) for region in REGION_ORDER)
    hd_cells = " | ".join(_cell(hd95, region, decimals) for region in REGION_ORDER)
    return "\n".join(
        (
            header,
            separator,
            f"| {label_prefix}Dice ↑ | {dice_cells} |",
            f"| {label_prefix}HD95 (mm) ↓ | {hd_cells} |",
        )
    )


def key_value_markdown_table(rows: Sequence[tuple[str, Any]]) -> str:
    lines = ["| Field | Recorded value |", "|---|---|"]
    for key, value in rows:
        rendered = str(value).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {key} | {rendered} |")
    return "\n".join(lines)
