"""Wall-clock records and robust runtime summaries."""

from __future__ import annotations

import datetime as dt
import json
import statistics
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RuntimeRecord:
    stage: str
    started_at: str
    ended_at: str
    total_seconds: float
    total_hours: float
    number_of_epochs: int | None
    average_seconds_per_epoch: float | None
    epoch_seconds_min: float | None
    epoch_seconds_median: float | None
    epoch_seconds_max: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_runtime_record(
    *,
    stage: str,
    started_at: str,
    ended_at: str,
    total_seconds: float,
    number_of_epochs: int | None = None,
    epoch_durations_seconds: Sequence[float] = (),
) -> RuntimeRecord:
    durations = [float(value) for value in epoch_durations_seconds]
    average = None
    if durations:
        average = statistics.fmean(durations)
    elif number_of_epochs and number_of_epochs > 0:
        average = total_seconds / number_of_epochs
    return RuntimeRecord(
        stage=stage,
        started_at=started_at,
        ended_at=ended_at,
        total_seconds=total_seconds,
        total_hours=total_seconds / 3600.0,
        number_of_epochs=number_of_epochs,
        average_seconds_per_epoch=average,
        epoch_seconds_min=min(durations) if durations else None,
        epoch_seconds_median=statistics.median(durations) if durations else None,
        epoch_seconds_max=max(durations) if durations else None,
    )


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write small metadata atomically; never overwrite scientific image data."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


class StageTimer:
    def __init__(self, stage: str) -> None:
        self.stage = stage
        self.started_at: str | None = None
        self.ended_at: str | None = None
        self.total_seconds: float | None = None
        self._started_monotonic: float | None = None

    def __enter__(self) -> StageTimer:
        self.started_at = dt.datetime.now(dt.timezone.utc).isoformat()
        self._started_monotonic = time.monotonic()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        assert self._started_monotonic is not None
        self.total_seconds = time.monotonic() - self._started_monotonic
        self.ended_at = dt.datetime.now(dt.timezone.utc).isoformat()
