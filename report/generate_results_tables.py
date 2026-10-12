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
    return "$" + f"{center:.1f}" + r" \pm " + f"{uncertainty:.1f}" + "$"


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


def led_reference(rows: list[dict], identity: str) -> dict:
    """Require a common test population and identical LED central values.

    Individual bootstrap seeds can differ by model, so use the saved LED
    uncertainty of the lexicographically first model rather than averaging
    independently bootstrapped standard deviations.
    """
    first = min(rows, key=lambda row: row["model"])
    for row in rows:
        if row["n_events"] != first["n_events"]:
            raise ValueError(f"Different test-event counts in {identity}")
        for field in ("led_ctr", "led_rmse"):
            if not math.isclose(row[field], first[field], abs_tol=1e-6, rel_tol=0):
                raise ValueError(f"Inconsistent LED {field} reference in {identity}")
    return first


def table_text(board: str, mode: str, window: str, rows: list[dict]) -> str:
    led = led_reference(rows, f"{board}/{mode}/{window}")
    ordered = sorted(rows, key=lambda row: (row["ctr"], row["model"]))
    lines = [
        "% Generated by report/generate_results_tables.py; do not edit manually.",
        "% LED bootstrap uncertainty source model: " + led["model"],
        r"\begin{table}[tbp]", r"\centering", r"\scriptsize",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
        "Model & CV CTR [ps] & Test CTR [ps] & Test RMSE [ps] " + r"\\",
        r"\midrule",
        ("LED & --- & "
         + interval(led["led_ctr"], led["led_ctr_std"])
         + " & " + interval(led["led_rmse"], led["led_rmse_std"])
         + " " + r"\\"),
        r"\midrule",
    ]
    for row in ordered:
        cv = ("n/a" if row["validation_ctr"] is None
              else interval(row["validation_ctr"], row["validation_std"]))
        lines.append(
            f"{latex(row['code'])} & {cv} & "
            + interval(row["ctr"], row["ctr_std"]) + " & "
            + interval(row["rmse"], row["rmse_std"]) + " " + r"\\"
        )
    lines.extend([
        r"\bottomrule", r"\end{tabular}", r"}",
        (rf"\caption{{{board} board, {mode} waveforms, {window} window. "
         r"Validation CTR is the mean $\pm$ standard deviation across held-out "
         r"CV folds. Test CTR and RMSE are shown with test-event bootstrap "
         r"standard deviations. LED has no validation score; linear ridge "
         r"uses internal MSE-based CV, not outer CTR-based validation. "
         r"Models are ordered by unrounded test CTR; codes are defined in Table~\ref{tab:model-inventory}.}"),
        rf"\label{{tab:{board.lower()}-{mode}-{window}}}",
        r"\end{table}", "",
    ])
    return "\n".join(lines)


def csv_text(rows: list[dict]) -> str:
    fields = [
        "code", "model", "formulation", "validation_ctr_mean_ps",
        "validation_ctr_std_ps", "test_ctr_ps", "test_ctr_bootstrap_std_ps",
        "test_rmse_ps", "test_rmse_bootstrap_std_ps",
    ]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in sorted(rows, key=lambda item: (item["ctr"], item["model"])):
        writer.writerow(dict(zip(fields, [
            row["code"], row["model"], row["formulation"],
            row["validation_ctr"], row["validation_std"],
            row["ctr"], row["ctr_std"], row["rmse"], row["rmse_std"],
        ])))
    return stream.getvalue()


def export(batch: Path, board: str, manuscript: Path, dry_run: bool) -> None:
    batch = batch.expanduser().resolve()
    grouped = load_batch(batch, board)
    targets = []
    includes = []
    for mode in ("energy", "timing"):
        for window in WINDOWS:
            rows = grouped.get((mode, window))
            if not rows:
                print(f"SKIP {board}/{mode}/{window}: no completed runs")
                continue
            stem = f"{board}_{mode}_{window}"
            plot = batch / "report" / "plots" / mode / window / "blind_ctr.png"
            if not plot.is_file():
                raise FileNotFoundError(
                    f"Required CTR plot missing: {plot}; generate batch report first"
                )
            # Validate all rows before writing any results to the manuscript.
            tex = table_text(board, mode, window, rows)
            targets.extend([
                (manuscript / "tables/results" / f"{stem}.tex", tex),
                (manuscript / "tables/results" / f"{stem}.csv", csv_text(rows)),
                (manuscript / "figures/results" / board
                 / f"{stem}_blind_ctr.png", plot),
            ])
            includes.append(rf"\input{{tables/results/{stem}}}")
    targets.append((
        manuscript / "tables/results" / f"{board}_ctr.tex",
        "\n".join(includes) + "\n",
    ))
    for destination, source in targets:
        action = "WOULD WRITE" if dry_run else "WRITE"
        print(f"{action} {destination.relative_to(manuscript)}")
        if dry_run:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(source, Path):
            shutil.copy2(source, destination)
        else:
            destination.write_text(source, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--uc", type=Path, help="UC batch result root")
    parser.add_argument("--fbk", type=Path, help="FBK batch result root")
    parser.add_argument("--manuscript", type=Path, default=ROOT / "report",
                        help="Output manuscript directory (default: repo/report)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate inputs and list outputs without writing")
    options = parser.parse_args()
    if not options.uc and not options.fbk:
        parser.error("At least one of --uc or --fbk is required")
    manuscript = options.manuscript.expanduser().resolve()
    if options.uc:
        export(options.uc, "UC", manuscript, options.dry_run)
    if options.fbk:
        export(options.fbk, "FBK", manuscript, options.dry_run)


if __name__ == "__main__":
    main()
