"""Export frozen batch-study CTR tables and copy existing blind CTR plots.

Example, from the repository root:
    python report/generate_results_tables.py --uc UC_BATCH --fbk FBK_BATCH

Only persisted metrics/plots are read. No training or statistics are rerun.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from waveform_analysis.reporting.model_labels import model_label  # noqa: E402

MODES = {"energy_to_energy": "energy", "timing_to_timing": "timing"}
WINDOWS = ("onishi", "wide")


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing study metadata: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return result


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing run index: {path}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def number(values: dict, field: str, context: str) -> float:
    try:
        result = float(values[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing/invalid {field}: {context}") from exc
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {field}: {context}")
    return result


def latex(value: str) -> str:
    reserved = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(reserved.get(char, char) for char in value)


def interval(center: float, uncertainty: float) -> str:
    return "$" + f"{center:.2f}" + r" \pm " + f"{uncertainty:.2f}" + "$"


def load_batch(batch: Path, board: str) -> dict[tuple[str, str], list[dict]]:
    config = read_json(batch / "config.json")
    folder = str(config.get("results", {}).get("folder", ""))
    parts = {part.upper() for part in folder.replace("\\", "/").split("/")}
    if board not in parts:
        raise ValueError(
            f"{batch}: config results.folder={folder!r} does not identify board "
            f"{board}; refusing to infer identity"
        )
    index = batch / "tables" / "runs.csv"
    if not index.exists() and (batch / "runs.csv").exists():
        index = batch / "runs.csv"  # Read legacy index without moving it.
    groups = defaultdict(list)
    seen = set()
    for indexed in read_csv(index):
        relative = Path(indexed["path"])
        directory = (batch / relative).resolve()
        if not directory.is_relative_to(batch):
            raise ValueError(f"Run index has path outside batch: {relative}")
        manifest_path = directory / "metadata" / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = read_json(manifest_path)
        if manifest.get("status") != "complete":
            continue
        raw_mode = manifest.get("mode")
        if raw_mode not in MODES:
            raise ValueError(f"Unknown mode {raw_mode!r} in {manifest_path}")
        mode = MODES[raw_mode]
        window = str(manifest.get("window_name", ""))
        if window not in WINDOWS:
            raise ValueError(f"Unknown window {window!r} in {manifest_path}")
        model = str(manifest.get("model", ""))
        formulation = str(manifest.get("estimator_formulation", ""))
        identity = f"{board}/{mode}/{window}/{model}/{formulation}"
        if not model or formulation not in {"direct", "shared"}:
            raise ValueError(f"Invalid model identity: {identity}")
        key = (mode, window, model, formulation)
        if key in seen:
            raise ValueError(f"Duplicate completed run: {identity}")
        seen.add(key)

        best = read_json(directory / "metadata" / "best.json")
        blind = read_json(directory / "metadata" / "blind.json")
        bootstrap = read_json(directory / "metadata" / "bootstrap.json")
        selection = manifest.get("selection_method", "development_cv")
        if selection == "ridge_cv":
            cv_ctr = cv_std = None
        elif selection == "development_cv":
            cv_ctr = number(best, "validation_ctr_mean_ps", identity)
            cv_std = number(best, "validation_ctr_std_ps", identity)
            if cv_std < 0:
                raise ValueError(f"Negative CV standard deviation: {identity}")
        else:
            raise ValueError(f"Unknown selection method {selection!r}: {identity}")
        row = {
            "model": model, "formulation": formulation,
            "validation_ctr": cv_ctr, "validation_std": cv_std,
            "ctr": number(blind, "ctr_ps", identity),
            "ctr_std": number(bootstrap, "ctr_bootstrap_std_ps", identity),
            "rmse": number(blind, "rmse_ps", identity),
            "rmse_std": number(bootstrap, "rmse_bootstrap_std_ps", identity),
            "led_ctr": number(blind, "led_ctr_ps", identity),
            "led_ctr_std": number(bootstrap, "led_ctr_bootstrap_std_ps", identity),
            "led_rmse": number(blind, "led_rmse_ps", identity),
            "led_rmse_std": number(bootstrap, "led_rmse_bootstrap_std_ps", identity),
            "n_events": int(number(blind, "n_events", identity)),
        }
        row["code"], row["display_name"] = model_label(model)
        if any(row[field] < 0 for field in
               ("ctr_std", "rmse_std", "led_ctr_std", "led_rmse_std")):
            raise ValueError(f"Negative bootstrap uncertainty: {identity}")
        groups[(mode, window)].append(row)
    if not groups:
        raise ValueError(f"No completed model runs in {batch}")
    return groups

