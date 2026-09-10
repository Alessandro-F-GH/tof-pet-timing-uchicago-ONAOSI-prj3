from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from utils_fit import fit_ctr_ps

from .common import voltage_from_name
from .dataset import load_prepared_dataset
from .shapelet_reporting import plot_fixed_shapelets
from .splits import semantic_seed
from .view import inverse_pair, waveform_view

MODEL_ORDER = ("led", "cfd", "linear_svr", "cnn", "difference_cnn", "difference_shapelet", "difference_knn")
LABELS = {
    "led": "LED",
    "cfd": "CFD",
    "linear_svr": "Linear SVR",
    "cnn": "CNN",
    "difference_cnn": "Difference CNN",
    "difference_shapelet": "Fixed-shapelet regressor",
    "difference_knn": "Difference k-NN",
}


def read_results(run_dir: str | Path) -> list[dict[str, Any]]:
    with (Path(run_dir) / "results.csv").open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _float(value, default=float("nan")):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _voltage(row):
    value = _float(row.get("voltage_V"))
    return value if np.isfinite(value) else voltage_from_name(row.get("dataset", ""))


def _residual(run, dataset, method, stage="test"):
    path = run / "artifacts" / dataset / f"{method}_{stage}_residuals_ps.npy"
    return np.asarray(np.load(path), dtype=float) if path.is_file() else None


def _model_output(run, dataset, model, stage):
    path = run / "artifacts" / dataset / f"{model}_{stage}_model_output_ps.npy"
    return np.asarray(np.load(path), dtype=float) if path.is_file() else None


def _robust_display_range(samples, *, quantiles=(0.02, 0.98), margin_fraction=0.06):
    finite = []
    for sample in samples:
        values = np.asarray(sample, dtype=float).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size:
            finite.append(values)
    if not finite:
        return -1.0, 1.0
    pooled = np.concatenate(finite)
    lo = float(np.quantile(pooled, float(quantiles[0])))
    hi = float(np.quantile(pooled, float(quantiles[1])))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return -1.0, 1.0
    if hi <= lo:
        pad = max(abs(lo) * 0.05, 1.0)
        return lo - pad, hi + pad
    margin = float(margin_fraction) * (hi - lo)
    return lo - margin, hi + margin


def _outside_count(values, xlim):
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    return int(np.count_nonzero((values < float(xlim[0])) | (values > float(xlim[1]))))


def _measurement_text(value, uncertainty):
    value = float(value)
    uncertainty = float(uncertainty)
    if not np.isfinite(value):
        return "nan"
    if not np.isfinite(uncertainty) or uncertainty <= 0:
        return f"{value:.1f}"
    rounded_unc = float(f"{uncertainty:.1g}")
    exponent = int(np.floor(np.log10(abs(rounded_unc)))) if rounded_unc else 0
    decimals = max(0, -exponent)
    return f"{value:.{decimals}f} ± {rounded_unc:.{decimals}f}"


def _distribution_methods(rows, dataset, stage):
    available = {r["method"] for r in rows if r["dataset"] == dataset and r.get("stage") == stage}
    ordered = ["led"] + [m for m in MODEL_ORDER if m not in {"led", "cfd"}]
    ordered.extend(sorted(available - set(ordered) - {"cfd"}))
    return [m for m in ordered if m in available]


def _median_centered_display_edges(values, xlim, n_bins=20):
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    low, high = float(xlim[0]), float(xlim[1])
    width = (high - low) / max(1, int(n_bins))
    if not values.size or not np.isfinite(width) or width <= 0:
        return np.linspace(low, high, int(n_bins) + 1)
    median = float(np.median(values))
    anchor = median - 0.5 * width
    steps_left = max(0, int(np.ceil((anchor - low) / width)))
    start = anchor - steps_left * width
    count = max(3, int(np.ceil((high - start) / width)))
    edges = start + np.arange(count + 1, dtype=float) * width
    if edges[-1] < high - 1e-12:
        edges = np.append(edges, edges[-1] + width)
    return edges


def _stripe_importance(time_ns, importance, width_ns=1.0):
    t = np.asarray(time_ns, dtype=float)
    imp = np.asarray(importance, dtype=float)
    if t.size == 0:
        return []
    edges = np.arange(
        np.floor(np.nanmin(t) / width_ns) * width_ns,
        np.ceil(np.nanmax(t) / width_ns) * width_ns + width_ns,
        width_ns,
    )
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        mask = (t >= a) & (t < (b if b < edges[-1] else b + 1e-12))
        values = imp[mask]
        out.append((float(a), float(b), float(np.nanmean(values)) if values.size else 0.0))
    maximum = max((v for _, _, v in out), default=0.0)
    return [(a, b, v / maximum if maximum > 0 else 0.0) for a, b, v in out]


def _xai_plot(output, artifact, mode, model, paths):
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    dataset = artifact.parent.name
    with np.load(artifact) as data:
        time = np.asarray(data["time_ps"], dtype=float) / 1000.0
        importance = np.asarray(data["importance"], dtype=float)
        pair = np.asarray(data["example_pair_mV"], dtype=float)
    if not time.size or not importance.size:
        return
    stripes = _stripe_importance(time, importance, 1.0)
    cmap = LinearSegmentedColormap.from_list("xai", ["white", "orange", "red"])
    norm = Normalize(0, 1)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8.6, 5.8), sharex=True, height_ratios=(2, 1))
    for a, b, value in stripes:
        top.axvspan(a, b, color=cmap(norm(value)), alpha=.7, lw=0)
    top.plot(time, pair[0], label="detector 1")
    top.plot(time, pair[1], label="detector 2")
    top.set_ylabel("Signal [mV]")
    top.legend()
    top.grid(True, alpha=.2)
    centers = np.asarray([(a + b) / 2 for a, b, _ in stripes])
    values = np.asarray([v for _, _, v in stripes])
    bottom.plot(centers, values, marker="o")
    bottom.set_ylim(0, 1.05)
    bottom.set_xlabel("Time relative to LED anchor [ns]")
    bottom.set_ylabel("1 ns mean importance")
    bottom.grid(True, alpha=.2)
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=top, pad=.015, fraction=.04)
    cbar.set_label("Normalized importance")
    voltage = voltage_from_name(dataset)
    label = f"{voltage:g} V" if np.isfinite(voltage) else dataset
    fig.suptitle(f"{LABELS.get(model, model)} · {mode.replace('_', ' ')} · {label}")
    fig.tight_layout()
    target = output / f"xai_{dataset}_{model}.pdf"
    fig.savefig(target)
    plt.close(fig)
    paths.append(target)


def _correction_rankings(run, model, dataset):
    led = _residual(run, dataset, "led", "test")
    corrected = _residual(run, dataset, model, "test")
    if led is None or corrected is None or led.size != corrected.size:
        return [], []
    finite = np.isfinite(led) & np.isfinite(corrected)
    if not np.any(finite):
        return [], []
    center = float(np.median(led[finite]))
    improvement = np.abs(led - center) - np.abs(corrected - center)
    event_indices = np.full(led.size, -1, dtype=np.int64)
    try:
        with np.load(run / "splits" / f"{dataset}.npz") as split:
            test = np.asarray(split["test"], dtype=np.int64)
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        prepared = load_prepared_dataset(manifest["datasets"][dataset]["prepared_dir"])
        event_indices = np.asarray(prepared.event_index[test], dtype=np.int64)
    except Exception:
        pass
    rows = [
        {
            "dataset": dataset,
            "voltage_V": voltage_from_name(dataset),
            "position": int(i),
            "event_index": int(event_indices[i]) if i < event_indices.size else -1,
            "led_residual_ps": float(led[i]),
            "corrected_residual_ps": float(corrected[i]),
            "led_bias_ps": center,
            "improvement_ps": float(improvement[i]),
        }
        for i in np.flatnonzero(finite)
    ]
    rows.sort(key=lambda r: r["improvement_ps"], reverse=True)
    return rows[:3], rows[-3:]


def _write_rankings(output, dataset, model, top, worst):
    target = output / f"correction_top_worst_{dataset}_{model}.csv"
    fields = [
        "rank_group", "rank", "dataset", "voltage_V", "event_index",
        "led_residual_ps", "corrected_residual_ps", "led_bias_ps", "improvement_ps",
    ]
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for group, items in (("top", top), ("worst", list(reversed(worst)))):
            for rank, row in enumerate(items, 1):
                writer.writerow({"rank_group": group, "rank": rank, **{k: row[k] for k in fields if k not in {"rank_group", "rank"}}})
    return target


def _correction_examples(output, run, mode, model, dataset, top, worst, paths):
    import matplotlib.pyplot as plt

    selected = [("Top", r) for r in top] + [("Worst", r) for r in reversed(worst)]
    if not selected:
        return
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), squeeze=False)
    for ax, (group, row) in zip(axes.flat, selected):
        try:
            prepared = load_prepared_dataset(manifest["datasets"][dataset]["prepared_dir"])
            with np.load(run / "splits" / f"{dataset}.npz") as split:
                test = np.asarray(split["test"], dtype=np.int64)
            index = int(test[row["position"]])
            view = waveform_view(prepared, mode, np.asarray([index]))
            pair = inverse_pair(prepared, mode, view.materialize())[0]
            time = np.asarray(view.time_ps) / 1000.0
            ax.plot(time, pair[0], label="detector 1")
            ax.plot(time, pair[1], label="detector 2")
            ax.axvline(0.0, ls="--", lw=1.0, alpha=.8)
            ax.set_title(f"{group} #{row['event_index']} | improvement {row['improvement_ps']:.1f} ps")
            ax.set_xlabel("Time [ns]")
            ax.set_ylabel("Signal [mV]")
            ax.grid(True, alpha=.2)
        except Exception as exc:
            ax.text(.5, .5, f"Unable to load example\n{exc}", ha="center", va="center", transform=ax.transAxes)
    axes[0, 0].legend()
    fig.suptitle(f"{dataset} · {mode.replace('_', ' ')} · {LABELS.get(model, model)} top/worst corrections")
    fig.tight_layout()
    target = output / f"correction_examples_{dataset}_{model}.pdf"
    fig.savefig(target)
    plt.close(fig)
    paths.append(target)


def _distribution_plot(output, run, rows, mode, dataset, stage, paths):
    import matplotlib.pyplot as plt

    series = []
    for method in _distribution_methods(rows, dataset, stage):
        residual = _residual(run, dataset, method, stage)
        if residual is None:
            continue
        residual = residual[np.isfinite(residual)]
        if not residual.size:
            continue
        row = next((r for r in rows if r["dataset"] == dataset and r["method"] == method and r.get("stage") == stage), None)
        if row is not None:
            series.append((method, residual, row))
    if not series:
        return

    xlim = _robust_display_range([r for _, r, _ in series], quantiles=(0.005, 0.995), margin_fraction=0.06)
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    for index, (method, residual, row) in enumerate(series):
        color = colors[index % len(colors)] if colors else None
        outside = _outside_count(residual, xlim)
        label = f"{LABELS.get(method, method)} · CTR {_measurement_text(_float(row.get('ctr_ps')), _float(row.get('ctr_uncertainty_ps')))} ps · outside {outside}"
        visible = residual[(residual >= xlim[0]) & (residual <= xlim[1])]
        bins = _median_centered_display_edges(visible, xlim, 20)
        ax.hist(visible, bins=bins, histtype="stepfilled", alpha=.4, color=color, edgecolor=color, linewidth=1.35, label=label)
        left = _float(row.get("fwhm_left_ps"))
        right = _float(row.get("fwhm_right_ps"))
        if np.isfinite(left):
            ax.axvline(left, color=color, ls="--", lw=1.45, alpha=.9)
        if np.isfinite(right):
            ax.axvline(right, color=color, ls="--", lw=1.45, alpha=.9)
    ax.set_xlim(*xlim)
    ax.set_xlabel(f"{stage.capitalize()} residual [ps]")
    ax.set_ylabel("Events / bin")
    ax.set_title(f"{mode.replace('_', ' ')} · {dataset} · {stage}")
    ax.legend()
    ax.grid(True, alpha=.2)
    fig.tight_layout()
    target = output / f"ctr_distribution_{stage}_{dataset}.pdf"
    fig.savefig(target)
    plt.close(fig)
    paths.append(target)


def _model_output_plot(output, run, mode, dataset, model, paths):
    import matplotlib.pyplot as plt

    train = _model_output(run, dataset, model, "train")
    test = _model_output(run, dataset, model, "test")
    if train is None and test is None:
        return
    train = np.asarray([] if train is None else train, dtype=float)
    test = np.asarray([] if test is None else test, dtype=float)
    train = train[np.isfinite(train)]
    test = test[np.isfinite(test)]
    if not train.size and not test.size:
        return
    xlim = _robust_display_range([train, test], quantiles=(0.02, 0.98), margin_fraction=0.06)
    bins = np.linspace(xlim[0], xlim[1], 21)
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 6.2), sharex=True)
    for ax, stage, values in ((axes[0], "train", train), (axes[1], "test", test)):
        outside = _outside_count(values, xlim)
        if values.size:
            visible = values[(values >= xlim[0]) & (values <= xlim[1])]
            ax.hist(visible, bins=bins, histtype="step", label=f"n={values.size} · outside display={outside}")
            ax.axvline(float(np.mean(values)), ls="--", lw=1.0, label=f"mean {np.mean(values):+.1f} ps")
            ax.legend()
        ax.set_xlim(*xlim)
        ax.set_ylabel("Events / bin")
        ax.set_title(stage.capitalize())
        ax.grid(True, alpha=.2)
    axes[1].set_xlabel(r"Learned correction $y_\theta(s_1,s_2)$ [ps]")
    fig.suptitle(f"{LABELS.get(model, model)} model output · {mode.replace('_', ' ')} · {dataset}")
    fig.tight_layout()
    target = output / f"model_output_{dataset}.pdf"
    fig.savefig(target)
    plt.close(fig)
    paths.append(target)


def _ctr_vs_voltage_bar_plot(run: Path, test_rows: list[dict[str, Any]], mode: str, paths: list[Path]) -> None:
    import matplotlib.pyplot as plt

    available_methods = {r["method"] for r in test_rows}
    methods = [m for m in MODEL_ORDER if m in available_methods]
    methods.extend(sorted(available_methods - set(methods)))
    if not methods:
        return
    voltages = sorted({_voltage(row) for row in test_rows if np.isfinite(_voltage(row))})
    if not voltages:
        return

    x = np.arange(len(voltages), dtype=float)
    group_width = 0.82
    bar_width = group_width / len(methods)
    offsets = (np.arange(len(methods), dtype=float) - 0.5 * (len(methods) - 1)) * bar_width
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    for method_index, method in enumerate(methods):
        ctr, error = [], []
        for voltage in voltages:
            row = next((r for r in test_rows if r["method"] == method and np.isfinite(_voltage(r)) and np.isclose(_voltage(r), voltage, rtol=0.0, atol=1e-9)), None)
            ctr.append(_float(row.get("ctr_ps")) if row is not None else np.nan)
            error.append(_float(row.get("ctr_uncertainty_ps")) if row is not None else np.nan)
        ctr = np.asarray(ctr, dtype=float)
        error = np.asarray(error, dtype=float)
        finite = np.isfinite(ctr)
        if not np.any(finite):
            continue
        safe_error = np.where(np.isfinite(error), error, 0.0)
        ax.bar(x[finite] + offsets[method_index], ctr[finite], width=bar_width * 0.92, yerr=safe_error[finite], capsize=3, alpha=0.85, label=LABELS.get(method, method))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:g} V" for v in voltages])
    ax.set_xlabel("Bias voltage")
    ax.set_ylabel("CTR FWHM [ps]")
    ax.set_ylim(bottom=0.0)
    ax.set_title(mode.replace("_", " "))
    ax.grid(axis="y", alpha=.22)
    ax.legend()
    fig.tight_layout()
    target = run / "ctr_vs_voltage.pdf"
    fig.savefig(target)
    plt.close(fig)
    paths.append(target)


def _paired_relative_improvement(reference, method, fit_config, *, samples, seed):
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    method = np.asarray(method, dtype=np.float64).reshape(-1)
    if reference.shape != method.shape:
        raise ValueError("Paired bootstrap requires aligned residual arrays")
    finite = np.isfinite(reference) & np.isfinite(method)
    reference = reference[finite]
    method = method[finite]
    minimum = int(fit_config.get("min_events", 100))
    if reference.size < minimum:
        raise ValueError(f"Only {reference.size} common finite residuals")
    full_ref = fit_ctr_ps(reference, fit_config, bootstrap=False).ctr_ps
    full_method = fit_ctr_ps(method, fit_config, bootstrap=False).ctr_ps
    central = 100.0 * (full_ref - full_method) / full_ref
    rng = np.random.default_rng(int(seed))
    bootstrap = []
    for _ in range(int(samples)):
        idx = rng.integers(0, reference.size, size=reference.size)
        try:
            ref_ctr = fit_ctr_ps(reference[idx], fit_config, bootstrap=False).ctr_ps
            method_ctr = fit_ctr_ps(method[idx], fit_config, bootstrap=False).ctr_ps
        except ValueError:
            continue
        if np.isfinite(ref_ctr) and ref_ctr > 0 and np.isfinite(method_ctr):
            bootstrap.append(100.0 * (ref_ctr - method_ctr) / ref_ctr)
    uncertainty = float(np.std(bootstrap, ddof=1)) if len(bootstrap) > 1 else float("nan")
    return central, uncertainty, len(bootstrap)


def _relative_improvement_plot(run, test_rows, manifest, mode, paths):
    import matplotlib.pyplot as plt

    fit_config = dict((manifest.get("config") or {}).get("fit") or {})
    samples = int(fit_config.get("bootstrap_samples", 100))
    base_seed = int(((manifest.get("config") or {}).get("validation") or {}).get("seed", 0))
    models = [m for m in MODEL_ORDER if m not in {"led", "cfd"} and any(r["method"] == m for r in test_rows)]
    models.extend(sorted({r["method"] for r in test_rows} - set(MODEL_ORDER) - {"led", "cfd"}))
    voltages = sorted({_voltage(row) for row in test_rows if np.isfinite(_voltage(row))})
    if not models or not voltages:
        return

    records = []
    for voltage in voltages:
        dataset_rows = [r for r in test_rows if np.isfinite(_voltage(r)) and np.isclose(_voltage(r), voltage, rtol=0.0, atol=1e-9)]
        dataset = next((r["dataset"] for r in dataset_rows if r["method"] == "led"), None)
        if dataset is None:
            continue
        reference = _residual(run, dataset, "led", "test")
        if reference is None:
            continue
        for model in models:
            residual = _residual(run, dataset, model, "test")
            if residual is None:
                continue
            central, uncertainty, successful = _paired_relative_improvement(
                reference,
                residual,
                fit_config,
                samples=samples,
                seed=semantic_seed(base_seed, dataset, model, "paired_relative_ctr_bootstrap"),
            )
            records.append({
                "dataset": dataset,
                "voltage_V": voltage,
                "method": model,
                "relative_improvement_percent": central,
                "paired_bootstrap_uncertainty_percent": uncertainty,
                "bootstrap_successful": successful,
            })
    if not records:
        return

    with (run / "relative_improvement.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    x = np.arange(len(voltages), dtype=float)
    group_width = 0.76
    bar_width = group_width / max(1, len(models))
    offsets = (np.arange(len(models), dtype=float) - 0.5 * (len(models) - 1)) * bar_width
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    for model_index, model in enumerate(models):
        values, errors = [], []
        for voltage in voltages:
            row = next((r for r in records if r["method"] == model and np.isclose(r["voltage_V"], voltage, rtol=0.0, atol=1e-9)), None)
            values.append(float(row["relative_improvement_percent"]) if row else np.nan)
            errors.append(float(row["paired_bootstrap_uncertainty_percent"]) if row else np.nan)
        values = np.asarray(values, dtype=float)
        errors = np.asarray(errors, dtype=float)
        finite = np.isfinite(values)
        ax.bar(x[finite] + offsets[model_index], values[finite], width=bar_width * 0.92, yerr=np.where(np.isfinite(errors[finite]), errors[finite], 0.0), capsize=3, alpha=.85, label=LABELS.get(model, model))
    ax.axhline(0.0, color="black", ls="--", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:g} V" for v in voltages])
    ax.set_xlabel("Bias voltage")
    ax.set_ylabel("CTR improvement over LED [%]")
    ax.set_title(f"{mode.replace('_', ' ')} · paired-bootstrap relative improvement")
    ax.grid(axis="y", alpha=.22)
    ax.legend()
    fig.tight_layout()
    target = run / "relative_improvement_vs_voltage.pdf"
    fig.savefig(target)
    plt.close(fig)
    paths.append(target)


def make_plots(run_dir: str | Path, output_dir: str | Path | None = None) -> list[Path]:
    run = Path(run_dir).resolve()
    plot_root = Path(output_dir).resolve() if output_dir else run / "plots"
    categories = {name: plot_root / name for name in ("corrections", "train_distribution", "test_distribution", "xai", "model_output")}
    for directory in categories.values():
        directory.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    mode = str(manifest.get("mode") or manifest["config"]["mode"])
    concatenated = bool(manifest.get("concatenate_datasets", False))
    all_rows = read_results(run)
    test_rows = [r for r in all_rows if r.get("stage") == "test"]
    datasets = sorted({r["dataset"] for r in test_rows}, key=voltage_from_name)
    paths: list[Path] = []

    if not concatenated:
        _ctr_vs_voltage_bar_plot(run, test_rows, mode, paths)
        _relative_improvement_plot(run, test_rows, manifest, mode, paths)

    for dataset in datasets:
        _distribution_plot(categories["test_distribution"], run, all_rows, mode, dataset, "test", paths)
        _distribution_plot(categories["train_distribution"], run, all_rows, mode, dataset, "train", paths)

    available_methods = {r["method"] for r in test_rows}
    ordered_methods = [m for m in MODEL_ORDER if m in available_methods] + sorted(available_methods - set(MODEL_ORDER))
    models = [m for m in ordered_methods if m not in {"led", "cfd"}]
    for model in models:
        model_output_dir = categories["model_output"] / model
        model_output_dir.mkdir(parents=True, exist_ok=True)
        for dataset in datasets:
            _model_output_plot(model_output_dir, run, mode, dataset, model, paths)
        for artifact in sorted((run / "artifacts").glob(f"*/{model}_xai.npz"), key=lambda p: voltage_from_name(p.parent.name)):
            _xai_plot(categories["xai"], artifact, mode, model, paths)
        if model == "difference_shapelet":
            shapelet_dir = categories["xai"] / "shapelets"
            shapelet_dir.mkdir(parents=True, exist_ok=True)
            for dataset in datasets:
                path = plot_fixed_shapelets(run, dataset, shapelet_dir)
                if path is not None:
                    paths.append(path)
        for dataset in datasets:
            top, worst = _correction_rankings(run, model, dataset)
            if top or worst:
                paths.append(_write_rankings(categories["corrections"], dataset, model, top, worst))
                _correction_examples(categories["corrections"], run, mode, model, dataset, top, worst, paths)
    return paths
