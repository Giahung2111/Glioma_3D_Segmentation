"""Safe YAML loading with environment-variable expansion."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from .schema import BaseConfig, ConfigError, DatasetConfig


def _expand_environment(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    return value


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping without constructing arbitrary Python objects."""

    config_path = Path(path).resolve()
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Could not load {config_path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigError(f"Top-level YAML value in {config_path} must be a mapping")
    expanded = _expand_environment(loaded)
    if not isinstance(expanded, dict):  # defensive for type checkers
        raise ConfigError(f"Top-level YAML value in {config_path} must be a mapping")
    return expanded


def load_base_config(path: str | Path) -> BaseConfig:
    config_path = Path(path).resolve()
    return BaseConfig.from_mapping(load_yaml(config_path), config_directory=config_path.parent)


def load_dataset_config(path: str | Path) -> DatasetConfig:
    return DatasetConfig.from_mapping(load_yaml(path))
