"""Rank poor cases without hiding undefined empty-mask failures."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from glioma_seg.evaluation.regions import REGION_ORDER


@dataclass(frozen=True)
class RankedCase:
    case_id: str
    region: str
    criterion: str
    value: float
    status: str
    failure_type: str
    rank: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "region": self.region,
            "criterion": self.criterion,
            "value": self.value,
            "status": self.status,
            "failure_type": self.failure_type,
            "rank": self.rank,
        }


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _case_id(row: Mapping[str, Any]) -> str:
    case_id = str(row.get("case_id", "")).strip()
    if not case_id:
        raise ValueError("Every per-case metric row must contain a non-empty case_id")
    return case_id


def _rank_region_metric(
    rows: Sequence[Mapping[str, Any]], region: str, metric: str, n: int
) -> list[RankedCase]:
    key = f"dice_{region.lower()}" if metric == "dice" else f"hd95_{region.lower()}_mm"
    status_key = f"{metric}_{region.lower()}_status"
    failure_key = f"{region.lower()}_failure_type"
    candidates: list[tuple[tuple[float, float, str], Mapping[str, Any], float, str, str]] = []
    for row in rows:
        case_id = _case_id(row)
        value = _float(row.get(key))
        status = str(row.get(status_key, "status_not_recorded"))
        failure_type = str(row.get(failure_key, "not_recorded"))
        both_empty = "both_empty" in status or failure_type == "both_empty"
        if both_empty:
            # No target and no prediction is not an observed segmentation failure.
            continue
        if metric == "dice":
            if not np.isfinite(value):
                # Unexpected undefined Dice is retained after valid numeric values,
                # rather than silently disappearing.
                sort_key = (1.0, 0.0, case_id)
            else:
                sort_key = (0.0, value, case_id)
        else:
            if not np.isfinite(value):
                # One-sided empty masks make HD95 undefined and represent a
                # detection failure.  Rank them before large finite distances.
                one_sided_empty = failure_type in {"false_positive", "false_negative"}
                sort_key = (0.0 if one_sided_empty else 2.0, 0.0, case_id)
            else:
                sort_key = (1.0, -value, case_id)
        candidates.append((sort_key, row, value, status, failure_type))

    candidates.sort(key=lambda item: item[0])
    ranked: list[RankedCase] = []
    for index, (_, row, value, status, failure_type) in enumerate(candidates[:n], start=1):
        ranked.append(
            RankedCase(
                case_id=_case_id(row),
                region=region,
                criterion=f"worst_{region.lower()}_{metric}",
                value=value,
                status=status,
                failure_type=failure_type,
                rank=index,
            )
        )
    return ranked


def rank_worst_cases(
    rows: Sequence[Mapping[str, Any]], *, n_per_metric: int = 5
) -> dict[str, list[RankedCase]]:
    """Return top-N worst lists for Dice and HD95 in ET, TC, WT order."""

    if n_per_metric < 1:
        raise ValueError("n_per_metric must be positive")
    rankings: dict[str, list[RankedCase]] = {}
    for metric in ("dice", "hd95"):
        for region in REGION_ORDER:
            key = f"worst_{region.lower()}_{metric}"
            rankings[key] = _rank_region_metric(rows, region, metric, n_per_metric)
    return rankings


def select_representative_cases(
    rankings: Mapping[str, Sequence[RankedCase]], *, max_cases: int = 15
) -> list[dict[str, Any]]:
    """Round-robin the six rankings and deduplicate case IDs.

    Fewer than eight cases may be returned when the evaluation fold itself or
    the union of ranked failures has fewer than eight unique cases; no cases
    are fabricated to meet the requested presentation range.
    """

    if max_cases < 1:
        raise ValueError("max_cases must be positive")
    criterion_order = [
        f"worst_{region.lower()}_{metric}" for metric in ("dice", "hd95") for region in REGION_ORDER
    ]
    selected: dict[str, dict[str, Any]] = {}
    max_depth = max((len(rankings.get(key, ())) for key in criterion_order), default=0)
    for depth in range(max_depth):
        for criterion in criterion_order:
            entries = rankings.get(criterion, ())
            if depth >= len(entries):
                continue
            entry = entries[depth]
            if entry.case_id not in selected:
                selected[entry.case_id] = {
                    "case_id": entry.case_id,
                    "selection_reasons": [],
                    "primary_region": entry.region,
                    "primary_criterion": entry.criterion,
                    "primary_value": entry.value,
                    "primary_status": entry.status,
                    "primary_failure_type": entry.failure_type,
                }
            selected[entry.case_id]["selection_reasons"].append(
                f"{entry.criterion}:rank={entry.rank}"
            )
            if len(selected) >= max_cases:
                return _finalize_representatives(selected.values())
    return _finalize_representatives(selected.values())


def _finalize_representatives(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for record in records:
        copy = dict(record)
        copy["selection_reasons"] = "; ".join(copy["selection_reasons"])
        result.append(copy)
    return result


def load_metrics_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_representative_csv(records: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "selection_reasons",
        "primary_region",
        "primary_criterion",
        "primary_value",
        "primary_status",
        "primary_failure_type",
    ]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    return destination
