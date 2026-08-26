"""Pure builders for official nnU-Net v2 console commands.

No nnU-Net implementation is copied or imported here. Keeping construction in
one module makes the exact scientific command easy to snapshot and review.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

PLAN_AND_PREPROCESS = "nnUNetv2_plan_and_preprocess"
TRAIN = "nnUNetv2_train"
PREDICT = "nnUNetv2_predict"
ACCUMULATE_CV = "nnUNetv2_accumulate_crossval_results"


@dataclass(frozen=True)
class CommandSpec:
    """An argv-safe command description (never executed through a shell)."""

    executable: str
    arguments: tuple[str, ...]
    stage: str

    @property
    def argv(self) -> tuple[str, ...]:
        return (self.executable, *self.arguments)

    def with_executable(self, executable: str | Path) -> CommandSpec:
        return CommandSpec(str(executable), self.arguments, self.stage)


def build_plan_and_preprocess(
    dataset_id: int,
    configuration: str = "3d_fullres",
    *,
    verify_dataset_integrity: bool = True,
) -> CommandSpec:
    args = ["-d", str(dataset_id), "-c", configuration]
    if verify_dataset_integrity:
        args.append("--verify_dataset_integrity")
    return CommandSpec(PLAN_AND_PREPROCESS, tuple(args), "plan_and_preprocess")


def build_train(
    dataset_id: int,
    configuration: str,
    fold: int,
    *,
    trainer: str = "nnUNetTrainer",
    plans: str = "nnUNetPlans",
    save_probabilities: bool = True,
    continue_training: bool = False,
    validation_only: bool = False,
) -> CommandSpec:
    if fold not in range(5):
        raise ValueError("fold must be one of 0, 1, 2, 3, 4")
    if continue_training and validation_only:
        raise ValueError("nnU-Net does not allow --c and --val together")

    args = [str(dataset_id), configuration, str(fold)]
    # Omit defaults for the final baseline so its command is visibly standard.
    if trainer != "nnUNetTrainer":
        args.extend(["-tr", trainer])
    if plans != "nnUNetPlans":
        args.extend(["-p", plans])
    if save_probabilities:
        args.append("--npz")
    if continue_training:
        args.append("--c")
    if validation_only:
        args.append("--val")
    return CommandSpec(TRAIN, tuple(args), "train")


def build_benchmark(
    dataset_id: int,
    configuration: str = "3d_fullres",
    fold: int = 0,
) -> CommandSpec:
    training_command = build_train(
        dataset_id,
        configuration,
        fold,
        trainer="nnUNetTrainerBenchmark_5epochs",
        save_probabilities=False,
    )
    return CommandSpec(training_command.executable, training_command.arguments, "benchmark")


def build_predict(
    dataset_id: int,
    configuration: str,
    folds: Iterable[int],
    input_dir: Path,
    output_dir: Path,
    *,
    trainer: str = "nnUNetTrainer",
    plans: str = "nnUNetPlans",
    disable_tta: bool = True,
    save_probabilities: bool = False,
    continue_prediction: bool = False,
    checkpoint: str = "checkpoint_final.pth",
) -> CommandSpec:
    normalized_folds = tuple(int(fold) for fold in folds)
    if not normalized_folds or any(fold not in range(5) for fold in normalized_folds):
        raise ValueError("folds must be a non-empty subset of {0, 1, 2, 3, 4}")
    args = [
        "-i",
        str(input_dir),
        "-o",
        str(output_dir),
        "-d",
        str(dataset_id),
        "-c",
        configuration,
        "-f",
        *(str(fold) for fold in normalized_folds),
    ]
    if trainer != "nnUNetTrainer":
        args.extend(["-tr", trainer])
    if plans != "nnUNetPlans":
        args.extend(["-p", plans])
    if checkpoint != "checkpoint_final.pth":
        args.extend(["-chk", checkpoint])
    if disable_tta:
        args.append("--disable_tta")
    if save_probabilities:
        args.append("--save_probabilities")
    if continue_prediction:
        args.append("--continue_prediction")
    return CommandSpec(PREDICT, tuple(args), "predict")


def build_accumulate_cross_validation(
    dataset_id: int,
    configuration: str,
    output_dir: Path,
    *,
    folds: Iterable[int] = (0, 1, 2, 3, 4),
    trainer: str = "nnUNetTrainer",
    plans: str = "nnUNetPlans",
) -> CommandSpec:
    normalized_folds = tuple(int(fold) for fold in folds)
    if normalized_folds != (0, 1, 2, 3, 4):
        raise ValueError("The standard full-CV baseline must accumulate folds 0, 1, 2, 3, 4")
    args = [
        str(dataset_id),
        "-c",
        configuration,
        "-o",
        str(output_dir),
        "-f",
        *(str(fold) for fold in normalized_folds),
    ]
    if trainer != "nnUNetTrainer":
        args.extend(["-tr", trainer])
    if plans != "nnUNetPlans":
        args.extend(["-p", plans])
    return CommandSpec(ACCUMULATE_CV, tuple(args), "accumulate_crossval")
