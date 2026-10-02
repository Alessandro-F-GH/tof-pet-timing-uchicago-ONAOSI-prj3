from __future__ import annotations

import copy
import json
from pathlib import Path

DEFAULT_REPORTING_CONFIG = Path(__file__).resolve().parents[1] / "config" / "reporting.json"


def _read_json(path: Path) -> dict:
    path = Path(path).expanduser().resolve()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Reporting configuration not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid reporting configuration {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("Reporting configuration must contain a JSON object")
    return value


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _validate(config: dict) -> None:
    required_top = {"global", "formulations", "reference", "plots"}
    missing = required_top - set(config)
    if missing:
        raise ValueError(f"Reporting configuration missing sections: {sorted(missing)}")
    for formulation in ("shared", "direct"):
        style = config["formulations"].get(formulation)
        if not isinstance(style, dict):
            raise ValueError(f"Missing reporting style for formulation {formulation}")
        for key in ("label", "color", "marker"):
            if key not in style:
                raise ValueError(f"formulations.{formulation}.{key} is required")
    for plot_class in ("comparison", "scatter", "bar", "correlation_heatmap", "pareto"):
        if not isinstance(config["plots"].get(plot_class), dict):
            raise ValueError(f"plots.{plot_class} must be configured")


def load_reporting_config(override=None) -> dict:
    config = _read_json(DEFAULT_REPORTING_CONFIG)
    if override is not None:
        patch = copy.deepcopy(override) if isinstance(override, dict) else _read_json(Path(override))
        config = _deep_merge(config, patch)
    _validate(config)
    return config


def rc_params(config: dict) -> dict:
    global_style = config["global"]
    return {
        "font.family": global_style["font_family"],
        "font.size": float(global_style["font_size"]),
        "axes.titlesize": float(global_style["title_size"]),
        "axes.labelsize": float(global_style["label_size"]),
        "xtick.labelsize": float(global_style["tick_size"]),
        "ytick.labelsize": float(global_style["tick_size"]),
        "legend.fontsize": float(global_style["legend_size"]),
        "figure.dpi": float(global_style["dpi"]),
        "savefig.dpi": float(global_style["dpi"]),
        "lines.linewidth": float(global_style["line_width"]),
    }


def formulation_style(config: dict, formulation: str) -> dict:
    key = str(formulation).strip().lower()
    try:
        return config["formulations"][key]
    except KeyError as exc:
        raise ValueError(f"Unknown estimator formulation {formulation!r}") from exc


def plot_style(config: dict, plot_class: str) -> dict:
    try:
        return config["plots"][str(plot_class)]
    except KeyError as exc:
        raise ValueError(f"Unknown reporting plot class {plot_class!r}") from exc
