"""Compact nnU-Net status heartbeats layered over unmodified upstream output."""

from __future__ import annotations

import time
from dataclasses import dataclass
from statistics import fmean

from glioma_seg.backends.nnunet.parser import TrainingProgress
from glioma_seg.monitoring.gpu_monitor import GPUMonitor


@dataclass
class NNUNetProcessMonitor:
    experiment_id: str
    fold: int
    trainer: str
    gpu_monitor: GPUMonitor | None = None
    target_epochs: int | None = None
    expected_validation_cases: int | None = None

    def __post_init__(self) -> None:
        self.started_monotonic = time.monotonic()
        self.progress = TrainingProgress()

    def consume_line(self, stream_name: str, line: str) -> None:
        self.progress.update(line)

    def status_line(self) -> str:
        elapsed = time.monotonic() - self.started_monotonic
        values = [
            "[NNUNET]",
            f"Experiment={self.experiment_id}",
            f"Fold={self.fold}",
            f"Trainer={self.trainer}",
            f"Elapsed={elapsed / 60:.1f}m",
        ]
        if self.progress.phase == "final_validation":
            validation_progress = str(self.progress.final_validation_cases_started)
            if self.expected_validation_cases:
                validation_progress += f"/{self.expected_validation_cases}"
            values.extend(["Phase=FINAL_VALIDATION", f"Cases={validation_progress}"])
        elif self.progress.phase == "complete":
            values.append("Phase=COMPLETE")
        elif self.progress.current_epoch is not None:
            displayed_epoch = self.progress.current_epoch + 1
            epoch_progress = str(displayed_epoch)
            if self.target_epochs:
                epoch_progress += f"/{self.target_epochs}"
                percent = min(100.0, displayed_epoch * 100.0 / self.target_epochs)
                values.append(f"Progress={percent:.1f}%")
            values.append(f"Epoch={epoch_progress}")
            durations = self.progress.epoch_durations_seconds or []
            if self.target_epochs and durations:
                remaining_epochs = max(0, self.target_epochs - displayed_epoch)
                eta_seconds = fmean(durations) * remaining_epochs
                values.append(f"TrainETA={eta_seconds / 3600:.2f}h")
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
            if snapshot.power_w is not None:
                values.append(f"Power={snapshot.power_w:.0f}W")
        return " ".join(values)
