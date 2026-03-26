import argparse
import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any


def load_json_config(config_path: str | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def merge_config(
    config_data: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(config_data)
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
    return merged


def build_dataclass_config(
    config_cls,
    config_data: dict[str, Any],
    overrides: dict[str, Any],
):
    merged = merge_config(config_data, overrides)
    valid_keys = {field.name for field in fields(config_cls)}
    filtered = {key: value for key, value in merged.items() if key in valid_keys}
    return config_cls(**filtered)


def parse_json_config_arg(description: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True)
    return parser


def load_run_config(config_path: str) -> dict[str, Any]:
    config = load_json_config(config_path)
    config["config_path"] = str(Path(config_path))
    return config
