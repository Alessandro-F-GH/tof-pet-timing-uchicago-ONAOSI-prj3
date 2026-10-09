"""Refresh manuscript appendix tables from the configured waveform model spaces."""

from __future__ import annotations
import argparse
import json
from itertools import groupby
from pathlib import Path

REPORT = Path(__file__).resolve().parent
SPACES = REPORT.parent / "waveform_analysis/config/model_spaces"
NAMES = {
    "antisymmetric_mlp": "Shared dense MLP",
    "direct_mlp": "Direct dense MLP",
    "locally_connected_mlp": "Shared locally connected MLP",
    "shared_cnn1d": "Shared temporal CNN",
    "independent_cnn1d": "Independent temporal CNN",
    "onishi_cnn": "Onishi-inspired paired CNN",
    "shared_linear_ridge": "Shared linear ridge",
    "direct_linear_ridge": "Direct linear ridge",
    "shared_minirocket": "Shared MiniRocket",
    "direct_minirocket": "Direct MiniRocket",
}
LABELS = {
    "architecture": "Dense hidden widths",
    "activation": "Activation",
    "learning_rate": "Learning rate",
    "batch_size": "Batch size",
    "conv_channels": "Convolutional channels",
    "kernel_samples": "Temporal kernel sizes",
    "layer1_kernel_samples": "First local kernel size",
    "layer1_stride_samples": "First local stride",
    "layer2_kernel_positions": "Second local kernel size",
    "layer2_stride_positions": "Second local stride",
    "max_correction_ps": "Smooth correction bound [ps]",
    "num_kernels": "Requested MiniRocket features",
    "ridge_alpha": "Ridge regularization strengths",
    "epochs": "Maximum epochs",
    "patience": "Loss-stop patience [epochs]",
    "min_delta": "Minimum loss improvement [ps]",
    "early_stopping_fraction": "Internal holdout fraction",
    "gradient_clip_norm": "Gradient clipping norm",
    "gradient_min_norm": "Gradient-stop threshold",
    "gradient_patience": "Gradient-stop patience [epochs]",
    "lr_decay_epochs": "Learning-rate milestones [epochs]",
    "lr_decay_factor": "Learning-rate decay factor",
    "channels": "Convolutional channels",
    "kernels": "Temporal kernel sizes",
    "dense_units": "Dense hidden width",
}


SHARED_MODELS = {
    "antisymmetric_mlp",
    "locally_connected_mlp",
    "shared_cnn1d",
    "shared_linear_ridge",
    "shared_minirocket",
}


def formulation(name):
    return "Shared" if name in SHARED_MODELS else "Direct"


def estimator_label(name):
    label = NAMES[name].removeprefix("Shared ").removeprefix("Direct ")
    return label[0].upper() + label[1:]


def model_groups(names):
    """Keep model order within each formulation, with shared estimators first."""
    return groupby(
        sorted(names, key=lambda name: name not in SHARED_MODELS), formulation
    )


def value(raw):
    if isinstance(raw, dict):
        if raw.get("type") == "fixed":
            return value(raw["value"])
        if raw.get("type") == "categorical":
            return "\\{" + ", ".join(value(v) for v in raw["choices"]) + "\\}"
    if isinstance(raw, list):
        return "[" + ", ".join(value(v) for v in raw) + "]"
    if isinstance(raw, float):
        if raw and abs(raw) < 0.001:
            mantissa, exponent = f"{raw:.6e}".split("e")
            mantissa = f"{float(mantissa):g}"
            factor = "" if float(mantissa) == 1 else mantissa + r"\times "
            return rf"${factor}10^{{{int(exponent)}}}$"
        return f"{raw:g}"
    return "ReLU" if raw == "relu" else str(raw)


def rows(name):
    space = json.loads((SPACES / f"{name}.json").read_text())
    result = []
    if "ridge_cv" in space:
        grid = space["ridge_cv"]["alphas"]
        result.extend(
            [
                (
                    "Regularization grid",
                    f"{grid['num']} logarithmically spaced values from {value(grid['low'])} to {value(grid['high'])}",
                ),
                ("Internal selection", "Leave-one-out CV; MSE"),
                ("Intercept", "Included" if name.startswith("direct") else "None"),
            ]
        )
    else:
        for key, raw in space.get("architecture", {}).items():
            result.append((LABELS[key], value(raw)))
        for key, raw in space.get("parameters", {}).items():
            result.append((LABELS[key], value(raw)))
        if "minirocket" in name:
            result.extend(
                [
                    (
                        "Feature fitting",
                        "Control acquisition; scaling without centring",
                    ),
                    (
                        "Regression intercept",
                        "Included" if name.startswith("direct") else "None",
                    ),
                    ("Selection", "Outer CV; mean CTR"),
                ]
            )
        else:
            training = space["training"]
            optimizer = training.get(
                "optimizer", "Adam" if name == "onishi_cnn" else "sgd_nesterov"
            )
            result.append(
                (
                    "Optimizer",
                    "Nesterov SGD; momentum 0.9"
                    if optimizer == "sgd_nesterov"
                    else "Adam",
                )
            )
            for key in (
                "epochs",
                "early_stopping_fraction",
                "patience",
                "min_delta",
                "gradient_clip_norm",
                "gradient_min_norm",
                "gradient_patience",
                "lr_decay_epochs",
                "lr_decay_factor",
            ):
                if key in training:
                    result.append((LABELS[key], value(training[key])))
    return result


def table(filename, names, caption):
    lines = [
        "% Generated by report/generate_appendix.py; refresh after changing model spaces.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabularx}{\textwidth}{@{}p{0.09\textwidth}p{0.22\textwidth}p{0.27\textwidth}X@{}}",
        r"\toprule",
        r"Formulation & Estimator & Parameter & Configured space or value \\",
        r"\midrule",
    ]
    for group_index, (group_label, group) in enumerate(model_groups(names)):
        if group_index:
            lines.extend([r"\addlinespace[3pt]", r"\hdashline", r"\addlinespace[3pt]"])
        group = [(name, rows(name)) for name in group]
        row_count = sum(len(parameters) for _, parameters in group)
        for model_index, (name, parameters) in enumerate(group):
            if model_index:
                lines.extend(
                    [r"\addlinespace[3pt]", r"\cdashline{2-4}", r"\addlinespace[3pt]"]
                )
            for row_index, (parameter, choice) in enumerate(parameters):
                group_cell = (
                    rf"\multirow[t]{{{row_count}}}{{*}}{{{group_label}}}"
                    if model_index == row_index == 0
                    else ""
                )
                model_cell = (
                    rf"\multirow[t]{{{len(parameters)}}}{{=}}{{{estimator_label(name)}}}"
                    if row_index == 0
                    else ""
                )
                lines.append(
                    f"{group_cell} & {model_cell} & {parameter} & {choice} " + r"\\"
                )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            rf"\caption{{{caption}}}",
            rf"\label{{tab:space-{filename.replace('_', '-')}}}",
            r"\end{table*}",
        ]
    )
    (REPORT / "tables/hyperparameters" / f"{filename}.tex").write_text(
        "\n".join(lines) + "\n"
    )


def result_placeholders(board):
    """Reserve one result table per configured mode/window; insert no measurements."""
    batch = json.loads(
        (
            REPORT.parent / f"waveform_analysis/config/batches/benchmark_{board}.json"
        ).read_text()
    )
    inputs = []
    for mode in batch["sweep"]["modes"]:
        mode_label = {"timing_to_timing": "Timing", "energy_to_energy": "Energy"}[mode]
        for window_name, window in batch["sweep"]["windows"].items():
            filename = f"{board}_{mode_label.lower()}_{window_name}"
            inputs.append(rf"\input{{tables/results/{filename}}}")
            lines = [
                "% Results placeholder: replace dashes only with frozen study outputs.",
                r"\begin{table*}[t]",
                r"\centering",
                r"\small",
                r"\begin{tabularx}{\textwidth}{@{}lXrrr@{}}",
                r"\toprule",
                r"Formulation & Estimator & Validation CTR [ps] & Blind CTR [ps] & Blind LED CTR [ps] \\",
                r"\midrule",
            ]
            for group_index, (group_label, group) in enumerate(
                model_groups(batch["sweep"]["models"])
            ):
                if group_index:
                    lines.extend(
                        [r"\addlinespace[3pt]", r"\hdashline", r"\addlinespace[3pt]"]
                    )
                group = list(group)
                for row_index, name in enumerate(group):
                    validation = "n/a" if name.endswith("linear_ridge") else "---"
                    group_cell = (
                        rf"\multirow{{{len(group)}}}{{*}}{{{group_label}}}"
                        if row_index == 0
                        else ""
                    )
                    lines.append(
                        f"{group_cell} & {estimator_label(name)} & {validation} & --- & --- "
                        + r"\\"
                    )
            lines.extend(
                [
                    r"\bottomrule",
                    r"\end{tabularx}",
                    rf"\caption{{{board}, {mode_label.lower()} waveforms, {window['start']:g} to {window['end']:g} ns input interval. Validation CTR is summarized by the outer-CV mean and fold dispersion; blind CTR uses F1 with bootstrap uncertainty. Outer validation is not applicable to linear ridge (n/a).}}",
                    rf"\label{{tab:{filename.lower().replace('_', '-')}}}",
                    r"\end{table*}",
                ]
            )
            (REPORT / "tables/results" / f"{filename}.tex").write_text(
                "\n".join(lines) + "\n"
            )
    (REPORT / "tables/results" / f"{board}_ctr.tex").write_text(
        "\n".join(inputs) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-placeholders",
        action="store_true",
        help="Replace result tables with blank templates from benchmark axes.",
    )
    args = parser.parse_args()
    if args.result_placeholders:
        for board in ("UC", "FBK"):
            result_placeholders(board)
    table(
        "dense_and_local",
        ["antisymmetric_mlp", "direct_mlp", "locally_connected_mlp"],
        "Dense and locally connected neural model spaces. Neural losses use RMSE. Gradient-based stopping is enabled with the listed threshold and patience.",
    )
    table(
        "temporal_cnn",
        ["shared_cnn1d", "independent_cnn1d"],
        "Temporal CNN model spaces. Kernels are valid convolutions without pooling; the dense head follows flattened convolutional features. Neural losses use RMSE.",
    )
    table(
        "paired_cnn",
        ["onishi_cnn"],
        "Paired CNN settings. The first kernel spans both detectors; later kernels span one detector-height position. Training uses MSE on the complete fitting subset, without an internal early-stopping holdout.",
    )
    table(
        "linear_and_minirocket",
        [
            "shared_linear_ridge",
            "direct_linear_ridge",
            "shared_minirocket",
            "direct_minirocket",
        ],
        "Linear and MiniRocket spaces. Only the two linear estimators use internal ridge CV. MiniRocket transforms are frozen before outer supervised validation.",
    )


if __name__ == "__main__":
    main()
