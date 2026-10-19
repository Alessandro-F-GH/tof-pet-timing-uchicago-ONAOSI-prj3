from __future__ import annotations
import csv, json, shutil
from pathlib import Path
from typing import Any
import numpy as np
from waveform_analysis.core.io import atomic_json, write_csv
from waveform_analysis.reporting.plotting import (
    blind_metric_bar,
    blind_residual_boxplot,
    grouped_bar,
    heatmap,
    load_plot_config,
    output_path,
    render_run_plots,
    scatter_with_labels,
)
from waveform_analysis.data.splits import semantic_seed
from waveform_analysis.reporting.stats import (
    align_by_event_id,
    metric_values,
    paired_model_bootstrap,
    pearson_r,
)
from waveform_analysis.data.results import discover_runs, resolve_run_config
from waveform_analysis.reporting.latex_tables import (
    DATASET_TABLE_VERSION,
    export_dataset_tables,
    export_blind_metric_table,
)
from waveform_analysis.reporting.model_labels import model_label
from waveform_analysis.reporting.metric_rows import blind_metric_rows as _blind_metric_rows
from waveform_analysis.engine.diagnostics import DEVELOPMENT_DISTRIBUTION_VERSION


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as s:
        return list(csv.DictReader(s))


def _matrix(path, labels, matrix):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as s:
        w = csv.writer(s)
        w.writerow(["model", *labels])
        for label, row in zip(labels, np.asarray(matrix)):
            w.writerow([label, *row.tolist()])


def collect_runs(root):
    runs = discover_runs(root)
    if not runs:
        raise RuntimeError("No complete canonical results found; convert historical results first")
    return runs


def _uses_development_cv(run):
    return run["manifest"].get("selection_method", "development_cv") != "ridge_cv"


def _summaries(runs):
    validation = []
    blind = []
    for run in runs:
        m, b, f, u = run["manifest"], run["best"], run["blind"], run["bootstrap"]
        base = {
            "mode": m["mode"],
            "window": m.get("window_name"),
            "model": m["model"],
            "formulation": m["estimator_formulation"],
            "candidate_id": b["candidate_id"],
        }
        if _uses_development_cv(run):
            validation.append(
                {
                    **base,
                    "selection_metric": b["selection_metric"],
                    "ctr_mean_ps": b["validation_ctr_mean_ps"],
                    "ctr_std_ps": b["validation_ctr_std_ps"],
                    "rmse_mean_ps": b["validation_rmse_mean_ps"],
                    "rmse_std_ps": b["validation_rmse_std_ps"],
                    "folds": b["folds"],
                }
            )
        blind.append(
            {
                **base,
                "n_events": f["n_events"],
                "ctr_ps": f["ctr_ps"],
                "ctr_std_ps": u["ctr_bootstrap_std_ps"],
                "rmse_ps": f["rmse_ps"],
                "rmse_std_ps": u["rmse_bootstrap_std_ps"],
                "led_ctr_ps": f["led_ctr_ps"],
                "led_rmse_ps": f["led_rmse_ps"],
                "ctr_improvement_ps": f["ctr_improvement_ps"],
                "ctr_improvement_std_ps": u["ctr_improvement_bootstrap_std_ps"],
                "rmse_improvement_ps": f["rmse_improvement_ps"],
                "rmse_improvement_std_ps": u["rmse_improvement_bootstrap_std_ps"],
            }
        )
    return validation, blind


def _groups(runs):
    out = {}
    for run in runs:
        out.setdefault(
            (run["manifest"]["mode"], run["manifest"].get("window_name")), []
        ).append(run)
    return out


def _payload(run):
    with np.load(run["directory"] / "artifacts" / "pred.npz") as d:
        return {k: np.asarray(d[k]) for k in d.files}


def _correlation(group):
    labels = [r["manifest"]["model"] for r in group]
    payloads = [_payload(r) for r in group]
    matrix = np.eye(len(group))
    counts = np.zeros((len(group), len(group)), int)
    for i in range(len(group)):
        counts[i, i] = len(payloads[i]["event_id"])
        for j in range(i + 1, len(group)):
            ids, a, b = align_by_event_id(
                payloads[i]["event_id"],
                payloads[i]["prediction_ps"],
                payloads[j]["event_id"],
                payloads[j]["prediction_ps"],
            )
            matrix[i, j] = matrix[j, i] = pearson_r(a, b)
            counts[i, j] = counts[j, i] = len(ids)
    return labels, matrix, counts, payloads


def _paired(group, payloads, metric, root_config, seed):
    labels = [r["manifest"]["model"] for r in group]
    central = np.zeros((len(group), len(group)))
    std = np.zeros_like(central)
    counts = np.zeros_like(central, dtype=int)
    fit = resolve_run_config(group[0]["directory"])["fit"]
    mode = group[0]["manifest"]["mode"]
    n = int(root_config["settings"]["bootstrap"]["n_resamples"])
    for i in range(len(group)):
        counts[i, i] = len(payloads[i]["event_id"])
        for j in range(i + 1, len(group)):
            r = paired_model_bootstrap(
                payloads[i]["event_id"],
                payloads[i]["corrected_ps"],
                payloads[j]["event_id"],
                payloads[j]["corrected_ps"],
                fit,
                metric=metric,
                n_resamples=n,
                seed=semantic_seed(
                    seed,
                    mode,
                    group[0]["manifest"].get("window_name"),
                    metric,
                    labels[i],
                    labels[j],
                    "paired_model",
                ),
            )
            central[i, j] = r["difference_ps"]
            central[j, i] = -r["difference_ps"]
            std[i, j] = std[j, i] = r["bootstrap_std_ps"]
            counts[i, j] = counts[j, i] = r["n_matched"]
    return labels, central, std, counts


def _ensemble_ctr(group, payloads, seed):
    """Equal-weight saved predictions only; no model loading or calibration."""
    labels = [run["manifest"]["model"] for run in group]
    matrix = np.full((len(group), len(group)), np.nan)
    counts = np.zeros(matrix.shape, dtype=int)
    unavailable = []
    fit = resolve_run_config(group[0]["directory"])["fit"]
    for i, run in enumerate(group):
        # Preserve the published single-model value and its full population.
        matrix[i, i] = float(run["blind"]["ctr_ps"])
        counts[i, i] = int(run["blind"]["n_events"])
        for j in range(i + 1, len(group)):
            first, second = payloads[i], payloads[j]
            try:
                ids, a, b = align_by_event_id(
                    first["event_id"], first["prediction_ps"],
                    second["event_id"], second["prediction_ps"],
                )
                led_ids, led_a, led_b = align_by_event_id(
                    first["event_id"], first["led_residual_ps"],
                    second["event_id"], second["led_residual_ps"],
                )
                matched, prediction_idx, led_idx = np.intersect1d(
                    ids, led_ids, return_indices=True,
                )
                counts[i, j] = counts[j, i] = len(matched)
                if len(matched) < 5:
                    raise ValueError("fewer than five finite matched blind events")
                if not np.allclose(led_a[led_idx], led_b[led_idx], rtol=1e-10, atol=1e-8):
                    raise ValueError("saved LED residuals differ on matched events")
                # Average already-saved predictions; do not clip or recenter.
                residual = led_a[led_idx] - (0.5 * a[prediction_idx] + 0.5 * b[prediction_idx])
                value = metric_values(residual, fit, seed=semantic_seed(
                    seed, run["manifest"]["mode"], run["manifest"].get("window_name"),
                    *sorted((labels[i], labels[j])), "ensemble_ctr",
                ))["ctr_ps"]
                matrix[i, j] = matrix[j, i] = value
            except (ValueError, KeyError) as exc:
                unavailable.append({"row": labels[i], "column": labels[j], "reason": str(exc)})
    return labels, matrix, counts, unavailable


def _scatters(group, directory, cfg):
    labels = [r["manifest"]["model"] for r in group]
    ctr = np.asarray([r["blind"]["ctr_ps"] for r in group], float)
    rmse = np.asarray([r["blind"]["rmse_ps"] for r in group], float)
    ctr_e = np.asarray([r["bootstrap"]["ctr_bootstrap_std_ps"] for r in group], float)
    rmse_e = np.asarray([r["bootstrap"]["rmse_bootstrap_std_ps"] for r in group], float)
    r = pearson_r(rmse, ctr)
    led_ctr = float(group[0]["blind"]["led_ctr_ps"])
    led_rmse = float(group[0]["blind"]["led_rmse_ps"])
    scatter_with_labels(
        np.append(rmse, led_rmse),
        np.append(ctr, led_ctr),
        labels + ["LED reference"],
        output_path(directory, "rmse_vs_ctr", cfg),
        cfg,
        xlabel="Blind RMSE [ps]",
        ylabel="Blind CTR [ps]",
        title="Blind RMSE vs CTR across models",
        xerr=np.append(rmse_e, 0.0),
        yerr=np.append(ctr_e, 0.0),
        annotation=f"Model-only Pearson r = {r:.3f}"
        if np.isfinite(r)
        else "Model-only Pearson r = n/a",
    )
    group = [run for run in group if _uses_development_cv(run)]
    if not group:
        for metric in ("ctr", "rmse"):
            output_path(directory, f"validation_vs_blind_{metric}", cfg).unlink(
                missing_ok=True
            )
        return
    labels = [run["manifest"]["model"] for run in group]
    ctr = np.asarray([run["blind"]["ctr_ps"] for run in group], float)
    rmse = np.asarray([run["blind"]["rmse_ps"] for run in group], float)
    ctr_e = np.asarray(
        [run["bootstrap"]["ctr_bootstrap_std_ps"] for run in group], float
    )
    rmse_e = np.asarray(
        [run["bootstrap"]["rmse_bootstrap_std_ps"] for run in group], float
    )
    vctr = np.asarray([r["best"]["validation_ctr_mean_ps"] for r in group])
    vrmse = np.asarray([r["best"]["validation_rmse_mean_ps"] for r in group])
    for metric, x, y, e in (("ctr", vctr, ctr, ctr_e), ("rmse", vrmse, rmse, rmse_e)):
        rr = pearson_r(x, y)
        scatter_with_labels(
            x,
            y,
            labels,
            output_path(directory, f"validation_vs_blind_{metric}", cfg),
            cfg,
            xlabel=f"Development CV {metric.upper()} mean [ps]",
            ylabel=f"Blind {metric.upper()} [ps]",
            title=f"Development validation vs blind {metric.upper()}",
            yerr=e,
            annotation=f"Pearson r = {rr:.3f}"
            if np.isfinite(rr)
            else "Pearson r = n/a",
        )


def _windows(runs, root, cfg):
    for mode in sorted({r["manifest"]["mode"] for r in runs}):
        mr = [r for r in runs if r["manifest"]["mode"] == mode]
        windows = sorted({r["manifest"].get("window_name") for r in mr})
        directory = (
            root / "plots" / ("energy" if mode == "energy_to_energy" else "timing")
        )
        for stale in directory.glob("windows_*"):
            stale.unlink()
        if len(windows) < 2:
            continue
        for model in sorted(
            {r["manifest"]["model"] for r in mr if not _uses_development_cv(r)}
        ):
            model_runs = sorted(
                [r for r in mr if r["manifest"]["model"] == model],
                key=lambda r: str(r["manifest"].get("window_name")),
            )
            for metric in ("ctr", "rmse"):
                grouped_bar(
                    [str(r["manifest"].get("window_name")) for r in model_runs],
                    [
                        {
                            "label": model,
                            "values": [r["blind"][f"{metric}_ps"] for r in model_runs],
                            "errors": [
                                r["bootstrap"][f"{metric}_bootstrap_std_ps"]
                                for r in model_runs
                            ],
                        }
                    ],
                    output_path(directory, f"windows_{model}_{metric}", cfg),
                    cfg,
                    ylabel=f"Blind {metric.upper()} [ps]",
                    title=f"{model}: all waveform windows (RidgeCV lambda selection)",
                )
        if not any(_uses_development_cv(r) for r in mr):
            continue
        for metric in ("ctr", "rmse"):
            labels = []
            values = {"shared": [], "direct": []}
            errors = {"shared": [], "direct": []}
            for window in windows:
                wr = [r for r in mr if r["manifest"].get("window_name") == window]
                selected = {}
                for form in ("shared", "direct"):
                    candidates = [
                        r
                        for r in wr
                        if r["manifest"]["estimator_formulation"] == form
                        and _uses_development_cv(r)
                    ]
                    if candidates:
                        selection_metric = str(
                            candidates[0]["best"]["selection_metric"]
                        )
                        selected[form] = min(
                            candidates,
                            key=lambda r: float(
                                r["best"][f"validation_{selection_metric}_mean_ps"]
                            ),
                        )
                labels.append(
                    str(window)
                    + "\n"
                    + " ".join(
                        f"{f[0].upper()}:{selected[f]['manifest']['model']}"
                        for f in ("shared", "direct")
                        if f in selected
                    )
                )
                for form in ("shared", "direct"):
                    run = selected.get(form)
                    values[form].append(
                        float(run["blind"][f"{metric}_ps"]) if run else np.nan
                    )
                    errors[form].append(
                        float(run["bootstrap"][f"{metric}_bootstrap_std_ps"])
                        if run
                        else np.nan
                    )
            led_values = []
            led_errors = []
            for window in windows:
                wr = [run for run in mr if run["manifest"].get("window_name") == window]
                if wr:
                    led_values.append(float(wr[0]["blind"][f"led_{metric}_ps"]))
                    led_errors.append(0.0)
                else:
                    led_values.append(np.nan)
                    led_errors.append(np.nan)
            series = [
                {
                    "label": cfg["formulations"][f]["label"],
                    "values": values[f],
                    "errors": errors[f],
                }
                for f in ("shared", "direct")
                if np.any(np.isfinite(values[f]))
            ]
            series.append(
                {
                    "label": cfg["reference"]["label"],
                    "values": led_values,
                    "errors": led_errors,
                }
            )
            grouped_bar(
                labels,
                series,
                output_path(directory, f"windows_{metric}", cfg),
                cfg,
                ylabel=f"Blind {metric.upper()} [ps]",
                title=f"Waveform-window comparison — winners selected by development CV {metric.upper()}",
            )


def _read_matrix(path):
    rows = _csv(path)
    return [r["model"] for r in rows], np.asarray(
        [[float(v) for k, v in r.items() if k != "model"] for r in rows], dtype=float
    )


def _matrix_order(group: list[dict[str, Any]]) -> list[str]:
    """Order display axes by formulation, then model name, without refitting."""
    ordered = sorted(
        group,
        key=lambda r: (
            r["manifest"]["estimator_formulation"] != "direct",
            r["manifest"]["model"],
        ),
    )
    return [r["manifest"]["model"] for r in ordered]


def _reorder_matrix(
    labels: list[str], matrix: np.ndarray, ordered: list[str]
) -> np.ndarray:
    """Reindex both matrix axes, preserving entries for retained models."""
    indices = [labels.index(name) for name in ordered]
    return np.asarray(matrix)[np.ix_(indices, indices)]



def generate_report(result_root, *, logger=None, reuse_numeric=False, exclude_models=()):
    root = Path(result_root).resolve()
    cfg = load_plot_config(root)
    root_config = _json(root / "config.json")
    runs = collect_runs(root)
    excluded = set(exclude_models)
    known = {r["manifest"]["model"] for r in runs} | set(
        root_config.get("sweep", {}).get("models", []))
    if excluded - known:
        raise ValueError(f"Unknown excluded models: {sorted(excluded - known)}")
    all_runs = runs
    runs = [r for r in runs if r["manifest"]["model"] not in excluded]
    if not runs:
        raise ValueError("No complete models remain after reporting exclusions")
    report = root / "report"
    tables = report / "tables"
    plots = report / "plots"
    tables.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    # Remove obsolete aggregate groups when all of their models are excluded.
    for mode, window in _groups(all_runs).keys() - _groups(runs).keys():
        mode_name = "energy" if mode == "energy_to_energy" else "timing"
        for directory in (tables / mode_name / str(window), plots / mode_name / str(window)):
            if directory.is_dir():
                shutil.rmtree(directory)
    for run in runs:
        render_run_plots(run["directory"], cfg)
    dataset_tables = export_dataset_tables(root, runs, root_config)
    validation, blind = _summaries(runs)
    if validation:
        write_csv(tables / "validation.csv", validation)
    else:
        (tables / "validation.csv").unlink(missing_ok=True)
    ridge_rows = [
        {
            "mode": r["manifest"]["mode"],
            "window": r["manifest"].get("window_name"),
            "model": r["manifest"]["model"],
            "ridge_alpha": r["best"]["parameters"]["ridge_alpha"],
            "internal_cv": r["best"]["internal_cv"],
            "internal_cv_mse_ps2": r["best"]["internal_cv_mse_ps2"],
            "n_train": r["best"]["n_train"],
        }
        for r in runs
        if not _uses_development_cv(r)
    ]
    if ridge_rows:
        write_csv(tables / "ridge_cv.csv", ridge_rows)
    else:
        (tables / "ridge_cv.csv").unlink(missing_ok=True)
    write_csv(tables / "blind.csv", blind)
    seed = int(root_config["settings"]["seed"])
    for (mode, window), group in _groups(runs).items():
        mode_name = "energy" if mode == "energy_to_energy" else "timing"
        table_dir = tables / mode_name / str(window)
        plot_dir = report / "plots" / mode_name / str(window)
        table_dir.mkdir(parents=True, exist_ok=True)
        plot_dir.mkdir(parents=True, exist_ok=True)
        labels, corr, corr_n, payloads = _correlation(group)
        ordered = _matrix_order(group)
        corr = _reorder_matrix(labels, corr, ordered)
        corr_n = _reorder_matrix(labels, corr_n, ordered)
        _matrix(table_dir / "output_correlation.csv", ordered, corr)
        _matrix(table_dir / "output_correlation_n.csv", ordered, corr_n)
        heatmap(
            corr,
            ordered,
            output_path(plot_dir, "output_correlation", cfg),
            cfg,
            title="Blind model-output correlation",
            correlation=True,
            value_format=".2f",
        )
        ensemble_labels, ensemble, ensemble_n, unavailable = _ensemble_ctr(group, payloads, seed)
        ensemble = _reorder_matrix(ensemble_labels, ensemble, ordered)
        ensemble_n = _reorder_matrix(ensemble_labels, ensemble_n, ordered)
        _matrix(table_dir / "ensemble_ctr.csv", ordered, ensemble)
        _matrix(table_dir / "ensemble_ctr_n.csv", ordered, ensemble_n)
        atomic_json(table_dir / "ensemble_ctr.json", {
            "prediction": "arithmetic_mean_of_two_saved_predictions",
            "diagonal": "saved_single_model_ctr",
            "population": "pairwise_finite_matched_blind_event_ids",
            "unavailable_pairs": unavailable,
        })
        heatmap(
            ensemble, ordered, output_path(plot_dir, "ensemble_ctr", cfg), cfg,
            title="Blind ensemble CTR [ps] — equal-weight mean\nDiagonal: single model; pairs: matched events",
            absolute=True, colorbar_label="CTR [ps] (lower is better)",
        )
        for metric in ("ctr", "rmse"):
            central_path = table_dir / f"paired_{metric}.csv"
            std_path = table_dir / f"paired_{metric}_std.csv"
            count_path = table_dir / f"paired_{metric}_n.csv"
            if (
                reuse_numeric
                and central_path.is_file()
                and count_path.is_file()
            ):
                labels, central = _read_matrix(central_path)
                count_labels, counts = _read_matrix(count_path)
                can_reuse = set(ordered).issubset(labels) and count_labels == labels
            else:
                can_reuse = False
            if not can_reuse:
                labels, central, _, counts = _paired(
                    group, payloads, metric, root_config, seed
                )
            central = _reorder_matrix(labels, central, ordered)
            counts = _reorder_matrix(labels, counts, ordered)
            _matrix(central_path, ordered, central)
            _matrix(count_path, ordered, counts)
            std_path.unlink(missing_ok=True)
            for stale in plot_dir.glob(f"paired_{metric}_std.*"):
                stale.unlink()
            heatmap(
                central,
                ordered,
                output_path(plot_dir, f"paired_{metric}", cfg),
                cfg,
                title=f"Paired blind Δ {metric.upper()} (row − column) [ps]",
            )
        for metric in ("ctr", "rmse"):
            rows = _blind_metric_rows(group, metric)
            led_value = float(group[0]["blind"][f"led_{metric}_ps"])
            write_csv(table_dir / f"blind_{metric}.csv", rows)
            export_blind_metric_table(
                table_dir / f"blind_{metric}.tex", rows, led_value, metric=metric,
                mode=mode_name, window=str(window),
                report_id=root_config.get("results", {}).get("folder", root.name),
            )
            blind_metric_bar(rows, output_path(plot_dir, f"blind_{metric}", cfg), cfg,
                             metric=metric, led_value=led_value)
        _scatters(group, plot_dir, cfg)
        payload_by_model = {run["manifest"]["model"]: payload
                            for run, payload in zip(group, payloads)}
        blind_residual_boxplot(
            [{"label": model_label(name)[0],
              "led": payload_by_model[name]["led_residual_ps"],
              "corrected": payload_by_model[name]["corrected_ps"]} for name in ordered],
            output_path(plot_dir, "blind_boxplot", cfg), cfg,
            title="Blind LED and corrected residual distributions by model",
        )
    _windows(runs, report, cfg)
    atomic_json(
        report / "manifest.json",
        {
            "source_result_root": str(root),
            "runs": len(runs),
            "excluded_models": sorted(excluded),
            "matrix_order": "direct_then_shared_alphabetical",
            "dataset_table_version": DATASET_TABLE_VERSION,
            "development_distribution_version": DEVELOPMENT_DISTRIBUTION_VERSION,
            "blind_boxplot_version": 1,
            "ensemble_ctr_version": 1,
            "dataset_tables": [
                str(path.relative_to(report)) for path in dataset_tables
            ],
            "plot_regeneration_requires_training": False,
            "pairwise_difference_convention": "metric(row)-metric(column); negative means row model is better",
            "window_winner_source": "development_cv_only",
        },
    )
    if logger:
        logger.info("Report generated from persisted results only | %s", report)
    return report
