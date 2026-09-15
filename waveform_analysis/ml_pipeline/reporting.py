from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from utils_fit import fit_ctr_ps

from .common import voltage_from_name
from .dataset import load_prepared_dataset
from .plot_style import (
    DETECTOR_STYLES,
    DOUBLE_COLUMN,
    DOUBLE_COLUMN_TALL,
    LABELS,
    MODEL_ORDER,
    SINGLE_COLUMN,
    clean_axis,
    model_style,
    panel_label,
    paper_context,
    save_figure,
    set_voltage_ticks,
)
from .splits import semantic_seed
from .view import inverse_pair, waveform_view


def read_results(run_dir: str | Path) -> list[dict[str, Any]]:
    with (Path(run_dir) / "csv" / "results.csv").open(encoding="utf-8", newline="") as stream:
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


def _median_centered_display_edges(values, xlim, n_bins=22):
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
    from matplotlib.colors import Normalize

    with np.load(artifact) as data:
        time = np.asarray(data["time_ps"], dtype=float) / 1000.0
        importance = np.asarray(data["importance"], dtype=float).reshape(-1)
        pair = np.asarray(data["example_pair_mV"], dtype=float)
    if not time.size or not importance.size:
        return
    if importance.size != time.size or pair.shape != (2, time.size):
        raise ValueError(f"Incompatible XAI arrays in {artifact}")

    stripes = _stripe_importance(time, importance, 1.0)
    cmap = plt.get_cmap("cividis")
    norm = Normalize(0, 1)
    fig = plt.figure(figsize=DOUBLE_COLUMN_TALL)
    grid = fig.add_gridspec(
        2, 2, width_ratios=(1.0, 0.035), height_ratios=(1.6, 1.0),
        hspace=0.08, wspace=0.08,
    )
    top = fig.add_subplot(grid[0, 0])
    bottom = fig.add_subplot(grid[1, 0], sharex=top)
    colorbar_ax = fig.add_subplot(grid[:, 1])

    for a, b, value in stripes:
        top.axvspan(a, b, color=cmap(norm(value)), alpha=0.18, lw=0)
    for detector in range(2):
        top.plot(
            time,
            pair[detector],
            label=f"Detector {detector + 1}",
            **DETECTOR_STYLES[detector],
        )
    top.set_ylabel("Signal [mV]")
    top.legend(loc="best")
    clean_axis(top, grid=None)
    top.tick_params(labelbottom=False)

    centers = np.asarray([(a + b) / 2 for a, b, _ in stripes])
    values = np.asarray([v for _, _, v in stripes])
    bottom.plot(centers, values, color="#000000", marker="o")
    bottom.set_ylim(0, 1.05)
    bottom.set_xlabel("Time [ns]")
    bottom.set_ylabel("Importance [a.u.]")
    clean_axis(bottom, grid="y")

    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=colorbar_ax)
    cbar.set_label("Relative importance")
    target = save_figure(fig, output / f"xai_{artifact.parent.name}_{model}.pdf")
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

    split_path = run / "splits" / f"{dataset}.npz"
    if split_path.is_file():
        try:
            with np.load(split_path) as split:
                if "test_event_index" in split:
                    event_indices = np.asarray(split["test_event_index"], dtype=np.int64)
                else:
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
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank_group", "rank", "dataset", "voltage_V", "event_index",
        "led_residual_ps", "corrected_residual_ps", "led_bias_ps", "improvement_ps",
    ]
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for group, items in (("top", top), ("worst", list(reversed(worst)))):
            for rank, row in enumerate(items, 1):
                writer.writerow({
                    "rank_group": group,
                    "rank": rank,
                    **{k: row[k] for k in fields if k not in {"rank_group", "rank"}},
                })
    return target


def _correction_example_artifact(run: Path, dataset: str, model: str, group: str) -> Path:
    return run / "artifacts" / dataset / f"{model}_{group.lower()}_correction_examples.npz"


def _load_or_cache_correction_examples(run, mode, model, dataset, group, rows):
    artifact = _correction_example_artifact(run, dataset, model, group)
    if artifact.is_file():
        with np.load(artifact) as data:
            return (
                np.asarray(data["time_ns"], dtype=float),
                np.asarray(data["pair_mV"], dtype=float),
            )

    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    prepared = load_prepared_dataset(manifest["datasets"][dataset]["prepared_dir"])
    with np.load(run / "splits" / f"{dataset}.npz") as split:
        test = np.asarray(split["test"], dtype=np.int64)

    times, pairs = [], []
    for row in rows:
        index = int(test[row["position"]])
        view = waveform_view(prepared, mode, np.asarray([index]))
        pairs.append(inverse_pair(prepared, mode, view.materialize())[0])
        times.append(np.asarray(view.time_ps, dtype=float) / 1000.0)
    if not times:
        return np.empty((0, 0)), np.empty((0, 2, 0))

    artifact.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifact,
        time_ns=np.stack(times),
        pair_mV=np.stack(pairs).astype(np.float32),
        event_index=np.asarray([row["event_index"] for row in rows], dtype=np.int64),
        improvement_ps=np.asarray([row["improvement_ps"] for row in rows], dtype=float),
    )
    return np.stack(times), np.stack(pairs)


def _correction_example_group(output, run, mode, model, dataset, group, rows, paths):
    import matplotlib.pyplot as plt

    selected = list(rows)
    if not selected:
        return
    try:
        times, pairs = _load_or_cache_correction_examples(
            run, mode, model, dataset, group, selected
        )
    except Exception:
        return
    if len(times) == 0:
        return

    fig, axes = plt.subplots(
        len(selected), 1,
        figsize=(DOUBLE_COLUMN[0], max(2.0, 1.75 * len(selected))),
        squeeze=False,
        sharex=True,
    )
    for index, ax in enumerate(axes[:, 0]):
        for detector in range(2):
            ax.plot(
                times[index],
                pairs[index, detector],
                label=f"Detector {detector + 1}",
                **DETECTOR_STYLES[detector],
            )
        ax.axvline(0.0, color="#7F7F7F", ls=":", lw=0.9)
        ax.set_ylabel("Signal [mV]")
        clean_axis(ax, grid=None)
        panel_label(ax, f"({chr(97 + index)})")
    axes[-1, 0].set_xlabel("Time [ns]")
    axes[0, 0].legend(loc="best")
    fig.tight_layout(h_pad=0.25)
    target = save_figure(
        fig,
        output / f"correction_examples_{group.lower()}_{dataset}_{model}.pdf",
    )
    plt.close(fig)
    paths.append(target)


def _correction_examples(output, run, mode, model, dataset, top, worst, paths):
    _correction_example_group(output, run, mode, model, dataset, "Top", top, paths)
    _correction_example_group(output, run, mode, model, dataset, "Worst", list(reversed(worst)), paths)


def _distribution_plot(output, run, rows, mode, dataset, stage, paths):
    import matplotlib.pyplot as plt

    available = []
    for method in _distribution_methods(rows, dataset, stage):
        residual = _residual(run, dataset, method, stage)
        if residual is None:
            continue
        residual = np.asarray(residual, dtype=float)
        residual = residual[np.isfinite(residual)]
        if not residual.size:
            continue
        row = next(
            (
                r for r in rows
                if r["dataset"] == dataset
                and r["method"] == method
                and r.get("stage") == stage
            ),
            None,
        )
        if row is not None:
            available.append((method, residual, row))

    led = next((item for item in available if item[0] == "led"), None)
    models = [item for item in available if item[0] not in {"led", "cfd"}]
    if led is None or not models:
        return

    for model, model_residual, model_row in models:
        pair = [led, (model, model_residual, model_row)]
        lows = [_float(row.get("interval_low_ps")) for _method, _residual, row in pair]
        highs = [_float(row.get("interval_high_ps")) for _method, _residual, row in pair]
        finite_lows = [value for value in lows if np.isfinite(value)]
        finite_highs = [value for value in highs if np.isfinite(value)]
        if finite_lows and finite_highs:
            xlim = (min(finite_lows) - 20.0, max(finite_highs) + 20.0)
        else:
            xlim = _robust_display_range(
                [residual for _method, residual, _row in pair],
                quantiles=(0.005, 0.995),
                margin_fraction=0.06,
            )

        fig, ax = plt.subplots(figsize=SINGLE_COLUMN)
        peak = 0.0
        for index, (method, residual, row) in enumerate(pair):
            visible = residual[(residual >= xlim[0]) & (residual <= xlim[1])]
            bins = _median_centered_display_edges(visible, xlim, 22)
            counts, _edges = np.histogram(visible, bins=bins)
            if counts.size:
                peak = max(peak, float(np.max(counts)))
            style = model_style(method, index)
            label = (
                f"{LABELS.get(method, method)}, CTR "
                f"{_measurement_text(_float(row.get('ctr_ps')), _float(row.get('ctr_uncertainty_ps')))} ps"
            )
            ax.hist(
                visible,
                bins=bins,
                histtype="step",
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.35,
                label=label,
            )

        if peak > 0:
            ax.set_ylim(0.0, peak * 1.16)
        ax.set_xlim(*xlim)
        ax.set_xlabel("Residual [ps]")
        ax.set_ylabel("Events [count]")
        ax.legend(loc="best")
        clean_axis(ax, grid="y")
        fig.tight_layout()
        target = save_figure(
            fig,
            output / f"ctr_distribution_{stage}_{dataset}_{model}.pdf",
        )
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
    fig, axes = plt.subplots(2, 1, figsize=(SINGLE_COLUMN[0], 4.3), sharex=True)
    for panel, (ax, values) in enumerate(zip(axes, (train, test))):
        if values.size:
            visible = values[(values >= xlim[0]) & (values <= xlim[1])]
            ax.hist(visible, bins=bins, histtype="step", color=model_style(model)["color"])
            ax.axvline(float(np.mean(values)), color="#7F7F7F", ls=":", lw=0.9)
        ax.set_xlim(*xlim)
        ax.set_ylabel("Events [count]")
        clean_axis(ax, grid="y")
        panel_label(ax, "(a)" if panel == 0 else "(b)")
    axes[-1].set_xlabel("Correction [ps]")
    fig.tight_layout(h_pad=0.2)
    target = save_figure(fig, output / f"model_output_{dataset}.pdf")
    plt.close(fig)
    paths.append(target)


def plot_ctr_vs_voltage(
    output: Path,
    test_rows: list[dict[str, Any]],
    paths: list[Path],
    *,
    filename: str = "ctr_vs_voltage.pdf",
) -> None:
    import matplotlib.pyplot as plt

    available_methods = {r["method"] for r in test_rows}
    methods = [m for m in MODEL_ORDER if m in available_methods]
    methods.extend(sorted(available_methods - set(methods)))
    voltages = sorted({_voltage(row) for row in test_rows if np.isfinite(_voltage(row))})
    if not methods or not voltages:
        return

    fig, ax = plt.subplots(figsize=DOUBLE_COLUMN)
    for method_index, method in enumerate(methods):
        values, errors = [], []
        for voltage in voltages:
            row = next(
                (
                    r for r in test_rows
                    if r["method"] == method
                    and np.isfinite(_voltage(r))
                    and np.isclose(_voltage(r), voltage, rtol=0.0, atol=1e-9)
                ),
                None,
            )
            values.append(_float(row.get("ctr_ps")) if row is not None else np.nan)
            errors.append(_float(row.get("ctr_uncertainty_ps")) if row is not None else np.nan)
        values = np.asarray(values, dtype=float)
        errors = np.asarray(errors, dtype=float)
        finite = np.isfinite(values)
        if not np.any(finite):
            continue
        style = model_style(method, method_index)
        ax.errorbar(
            np.asarray(voltages)[finite],
            values[finite],
            yerr=np.where(np.isfinite(errors[finite]), errors[finite], 0.0),
            capsize=2.5,
            label=LABELS.get(method, method),
            **style,
        )
    set_voltage_ticks(ax, voltages)
    ax.set_xlabel("Voltage [V]")
    ax.set_ylabel("CTR [ps]")
    ax.legend(loc="best", ncol=2)
    clean_axis(ax, grid="y")
    fig.tight_layout()
    target = save_figure(fig, output / filename)
    plt.close(fig)
    paths.append(target)


def _paired_improvement(
    reference,
    method,
    fit_config,
    *,
    samples,
    seed,
    relative: bool,
):
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    method = np.asarray(method, dtype=np.float64).reshape(-1)
    if reference.shape != method.shape:
        raise ValueError("Paired bootstrap requires aligned residual arrays")
    finite = np.isfinite(reference) & np.isfinite(method)
    reference = reference[finite]
    method = method[finite]
    if reference.size < 2:
        raise ValueError("At least two common finite residuals are required")
    full_ref = fit_ctr_ps(reference, fit_config, bootstrap=False).ctr_ps
    full_method = fit_ctr_ps(method, fit_config, bootstrap=False).ctr_ps
    if relative:
        central = 100.0 * (full_ref - full_method) / full_ref
    else:
        central = full_ref - full_method

    rng = np.random.default_rng(int(seed))
    bootstrap = []
    for _ in range(int(samples)):
        idx = rng.integers(0, reference.size, size=reference.size)
        try:
            ref_ctr = fit_ctr_ps(reference[idx], fit_config, bootstrap=False).ctr_ps
            method_ctr = fit_ctr_ps(method[idx], fit_config, bootstrap=False).ctr_ps
        except ValueError:
            continue
        if not (np.isfinite(ref_ctr) and ref_ctr > 0 and np.isfinite(method_ctr)):
            continue
        if relative:
            bootstrap.append(100.0 * (ref_ctr - method_ctr) / ref_ctr)
        else:
            bootstrap.append(ref_ctr - method_ctr)
    uncertainty = (
        float(np.std(bootstrap, ddof=1))
        if len(bootstrap) > 1
        else float("nan")
    )
    return float(central), uncertainty, len(bootstrap)


def _paired_absolute_improvement(reference, method, fit_config, *, samples, seed):
    return _paired_improvement(
        reference,
        method,
        fit_config,
        samples=samples,
        seed=seed,
        relative=False,
    )


def _paired_relative_improvement(reference, method, fit_config, *, samples, seed):
    return _paired_improvement(
        reference,
        method,
        fit_config,
        samples=samples,
        seed=seed,
        relative=True,
    )


def plot_improvement_vs_led(
    run,
    output,
    test_rows,
    manifest,
    paths,
    *,
    filename: str = "improvement_vs_led.pdf",
):
    import matplotlib.pyplot as plt

    fit_config = dict((manifest.get("config") or {}).get("fit") or {})
    samples = int(fit_config.get("bootstrap_samples", 100))
    base_seed = int(
        ((manifest.get("config") or {}).get("validation") or {}).get("seed", 0)
    )
    available_methods = {row["method"] for row in test_rows}
    models = [
        method
        for method in MODEL_ORDER
        if method in available_methods and method not in {"led", "cfd"}
    ]
    voltages = sorted(
        {_voltage(row) for row in test_rows if np.isfinite(_voltage(row))}
    )
    if not models or not voltages:
        return

    records = []
    for voltage in voltages:
        dataset_rows = [
            row
            for row in test_rows
            if np.isfinite(_voltage(row))
            and np.isclose(_voltage(row), voltage, rtol=0.0, atol=1e-9)
        ]
        dataset = next(
            (row["dataset"] for row in dataset_rows if row["method"] == "led"),
            None,
        )
        if dataset is None:
            continue
        reference = _residual(run, dataset, "led", "test")
        if reference is None:
            continue
        for model in models:
            residual = _residual(run, dataset, model, "test")
            if residual is None:
                continue
            central, uncertainty, successful = _paired_absolute_improvement(
                reference,
                residual,
                fit_config,
                samples=samples,
                seed=semantic_seed(
                    base_seed,
                    dataset,
                    model,
                    "paired_absolute_ctr_bootstrap",
                ),
            )
            records.append(
                {
                    "dataset": dataset,
                    "voltage_V": voltage,
                    "method": model,
                    "improvement_ps": central,
                    "paired_bootstrap_uncertainty_ps": uncertainty,
                    "bootstrap_successful": successful,
                }
            )
    if not records:
        return

    csv_path = Path(run) / "csv" / "improvement_vs_led.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    fig, ax = plt.subplots(figsize=DOUBLE_COLUMN)
    for model_index, model in enumerate(models):
        values, errors = [], []
        for voltage in voltages:
            row = next(
                (
                    item
                    for item in records
                    if item["method"] == model
                    and np.isclose(
                        item["voltage_V"],
                        voltage,
                        rtol=0.0,
                        atol=1e-9,
                    )
                ),
                None,
            )
            values.append(float(row["improvement_ps"]) if row else np.nan)
            errors.append(
                float(row["paired_bootstrap_uncertainty_ps"])
                if row
                else np.nan
            )
        values = np.asarray(values, dtype=float)
        errors = np.asarray(errors, dtype=float)
        finite = np.isfinite(values)
        if not np.any(finite):
            continue
        ax.errorbar(
            np.asarray(voltages)[finite],
            values[finite],
            yerr=np.where(np.isfinite(errors[finite]), errors[finite], 0.0),
            capsize=2.5,
            label=LABELS.get(model, model),
            **model_style(model, model_index),
        )
    ax.axhline(0.0, color="#7F7F7F", linestyle=":", linewidth=0.9)
    set_voltage_ticks(ax, voltages)
    ax.set_xlabel("Voltage [V]")
    ax.set_ylabel("Improvement [ps]")
    ax.legend(loc="best", ncol=2)
    clean_axis(ax, grid="y")
    fig.tight_layout()
    target = save_figure(fig, Path(output) / filename)
    plt.close(fig)
    paths.append(target)


def _relative_improvement_plot(run, output, test_rows, manifest, paths):
    import matplotlib.pyplot as plt

    fit_config = dict((manifest.get("config") or {}).get("fit") or {})
    samples = int(fit_config.get("bootstrap_samples", 100))
    base_seed = int(((manifest.get("config") or {}).get("validation") or {}).get("seed", 0))
    models = [
        m for m in MODEL_ORDER
        if m not in {"led", "cfd"} and any(r["method"] == m for r in test_rows)
    ]
    models.extend(sorted({r["method"] for r in test_rows} - set(MODEL_ORDER) - {"led", "cfd"}))
    voltages = sorted({_voltage(row) for row in test_rows if np.isfinite(_voltage(row))})
    if not models or not voltages:
        return

    records = []
    for voltage in voltages:
        dataset_rows = [
            r for r in test_rows
            if np.isfinite(_voltage(r))
            and np.isclose(_voltage(r), voltage, rtol=0.0, atol=1e-9)
        ]
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

    relative_csv = run / "csv" / "relative_improvement.csv"
    relative_csv.parent.mkdir(parents=True, exist_ok=True)
    with relative_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    fig, ax = plt.subplots(figsize=DOUBLE_COLUMN)
    for model_index, model in enumerate(models):
        values, errors = [], []
        for voltage in voltages:
            row = next(
                (
                    r for r in records
                    if r["method"] == model
                    and np.isclose(r["voltage_V"], voltage, rtol=0.0, atol=1e-9)
                ),
                None,
            )
            values.append(float(row["relative_improvement_percent"]) if row else np.nan)
            errors.append(float(row["paired_bootstrap_uncertainty_percent"]) if row else np.nan)
        values = np.asarray(values, dtype=float)
        errors = np.asarray(errors, dtype=float)
        finite = np.isfinite(values)
        style = model_style(model, model_index)
        ax.errorbar(
            np.asarray(voltages)[finite],
            values[finite],
            yerr=np.where(np.isfinite(errors[finite]), errors[finite], 0.0),
            capsize=2.5,
            label=LABELS.get(model, model),
            **style,
        )
    ax.axhline(0.0, color="#7F7F7F", ls=":", lw=0.9)
    set_voltage_ticks(ax, voltages)
    ax.set_xlabel("Voltage [V]")
    ax.set_ylabel("Improvement [%]")
    ax.legend(loc="best", ncol=2)
    clean_axis(ax, grid="y")
    fig.tight_layout()
    target = save_figure(fig, output / "relative_improvement_vs_voltage.pdf")
    plt.close(fig)
    paths.append(target)


def make_plots(run_dir: str | Path, output_dir: str | Path | None = None) -> list[Path]:
    run = Path(run_dir).resolve()
    plot_root = Path(output_dir).resolve() if output_dir else run / "plots"
    csv_root = run / "csv"
    plot_categories = {
        name: plot_root / name
        for name in ("corrections", "train_distribution", "test_distribution", "xai", "model_output")
    }
    csv_categories = {
        "corrections": csv_root / "corrections",
        "xai": csv_root / "xai",
    }
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    mode = str(manifest.get("mode") or manifest["config"]["mode"])
    concatenated = bool(manifest.get("concatenate_datasets", False))
    all_rows = read_results(run)
    test_rows = [r for r in all_rows if r.get("stage") == "test"]
    datasets = sorted({r["dataset"] for r in test_rows}, key=voltage_from_name)
    paths: list[Path] = []

    with paper_context():
        if not concatenated:
            plot_ctr_vs_voltage(plot_root, test_rows, paths)
            _relative_improvement_plot(run, plot_root, test_rows, manifest, paths)

        for dataset in datasets:
            _distribution_plot(
                plot_categories["test_distribution"] / dataset,
                run, all_rows, mode, dataset, "test", paths,
            )
            _distribution_plot(
                plot_categories["train_distribution"] / dataset,
                run, all_rows, mode, dataset, "train", paths,
            )

        available_methods = {r["method"] for r in test_rows}
        ordered_methods = [
            m for m in MODEL_ORDER if m in available_methods
        ] + sorted(available_methods - set(MODEL_ORDER))
        models = [m for m in ordered_methods if m not in {"led", "cfd"}]

        for model in models:
            model_output_dir = plot_categories["model_output"] / model
            for dataset in datasets:
                _model_output_plot(model_output_dir, run, mode, dataset, model, paths)

            xai_plot_dir = plot_categories["xai"] / model
            for artifact in sorted(
                (run / "artifacts").glob(f"*/{model}_xai.npz"),
                key=lambda p: voltage_from_name(p.parent.name),
            ):
                _xai_plot(xai_plot_dir, artifact, mode, model, paths)

            for dataset in datasets:
                top, worst = _correction_rankings(run, model, dataset)
                if top or worst:
                    correction_plot_dir = plot_categories["corrections"] / dataset
                    correction_csv_dir = csv_categories["corrections"] / dataset
                    paths.append(_write_rankings(
                        correction_csv_dir, dataset, model, top, worst
                    ))
                    _correction_examples(
                        correction_plot_dir,
                        run,
                        mode,
                        model,
                        dataset,
                        top,
                        worst,
                        paths,
                    )
    return paths
