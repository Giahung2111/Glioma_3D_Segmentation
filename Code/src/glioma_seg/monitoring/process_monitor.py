"""Compact nnU-Net status heartbeats layered over unmodified upstream output."""

from __future__ import annotations

import time
from dataclasses import dataclass

from glioma_seg.backends.nnunet.parser import TrainingProgress
from glioma_seg.monitoring.gpu_monitor import GPUMonitor


@dataclass
class NNUNetProcessMonitor:
    experiment_id: str
    fold: int
    trainer: str
    gpu_monitor: GPUMonitor | None = None

    def __post_init__(self) -> None:
        self.started_monotonic = time.monotonic()
        self.progress = TrainingProgress()

    def consume_line(self, stream_name: str, line: str) -> None:
        self.progress.update(line)

    def status_line(self) -> str:
        elapsed = time.monotonic() - self.started_monotonic
        values = [
            "[TRAIN]",
            f"Experiment={self.experiment_id}",
            f"Fold={self.fold}",
            f"Trainer={self.trainer}",
            f"Elapsed={elapsed / 60:.1f}m",
        ]
        if self.progress.current_epoch is not None:
            values.append(f"Epoch={self.progress.current_epoch}")
        if self.progress.latest_train_loss is not None:
            values.append(f"TrainLoss={self.progress.latest_train_loss:g}")
        if self.progress.latest_validation_loss is not None:
            values.append(f"ValLoss={self.progress.latest_validation_loss:g}")
        if self.progress.latest_pseudo_dice:
            values.append(f"PseudoDice={self.progress.latest_pseudo_dice}")
        snapshot = self.gpu_monitor.latest if self.gpu_monitor else None
        if snapshot:
            values.extend(
                [
                    f"GPU={snapshot.gpu_utilization_percent:.0f}%",
                    f"VRAM={snapshot.memory_used_mb:.0f}/{snapshot.memory_total_mb:.0f}MiB",
                    f"Temp={snapshot.temperature_c:.0f}C",
                ]
            )
        return " ".join(values)
