#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress, pearsonr, spearmanr

from utils_fit import fit_ctr_ps
from waveform_analysis.ml_pipeline.dataset import PreparedDataset, load_prepared_dataset
from waveform_analysis.ml_pipeline.view import calibrated_led, inverse_pair, waveform_view


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the event-wise distance between the two aligned waveform channels and test "
            "its relation to the absolute calibrated LED timing error. The default distance is "
            "the RMS of the physical-mV difference s1(t)-s2(t) over the full ML window."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Completed study directory.")
    parser.add_argument("--dataset", action="append", default=None)
    parser.add_argument("--stage", choices=("training", "validation", "development", "test"), default="development")
    parser.add_argument("--window-ns", type=float, nargs=2, metavar=("START", "STOP"), default=None)
    parser.add_argument("--distance", choices=("rms", "mean_abs", "max_abs"), default="rms")
    parser.add_argument("--trend-bins", type=int, default=12)
    parser.add_argument("--efficiency-step-percent", type=float, default=1.0,
                        help="Accepted-event efficiency step for the low-distance CTR scan. Default: 1%%.")
    parser.add_argument("--bootstrap-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _stage_indices(dataset: PreparedDataset, stage: str) -> np.ndarray:
    if stage == "training": return np.asarray(dataset.training, dtype=np.int64)
    if stage == "validation": return np.asarray(dataset.validation, dtype=np.int64)
    if stage == "development": return np.asarray(dataset.development, dtype=np.int64)
    if stage == "test": return np.asarray(dataset.test, dtype=np.int64)
    raise ValueError(stage)


def _time_mask(time_ps: np.ndarray, window_ns: tuple[float, float] | None) -> np.ndarray:
    time_ns = np.asarray(time_ps, dtype=np.float64) / 1000.0
    if window_ns is None:
        return np.ones(time_ns.size, dtype=bool)
    start, stop = map(float, window_ns)
    if stop <= start:
        raise ValueError("--window-ns must satisfy START < STOP")
    mask = (time_ns >= start) & (time_ns <= stop)
    if np.count_nonzero(mask) < 2:
        raise ValueError(f"Requested distance window [{start:g}, {stop:g}] ns contains fewer than two prepared samples")
    return mask


def _distance_from_difference(difference: np.ndarray, metric: str) -> np.ndarray:
    difference = np.asarray(difference, dtype=np.float64)
    if metric == "rms": return np.sqrt(np.mean(difference**2, axis=1))
    if metric == "mean_abs": return np.mean(np.abs(difference), axis=1)
    if metric == "max_abs": return np.max(np.abs(difference), axis=1)
    raise ValueError(metric)


def _waveform_distance(dataset, mode, indices, *, metric, window_ns, batch_size):
    first = waveform_view(dataset, mode, indices[:1])
    time_ps = np.asarray(first.time_ps, dtype=np.float64)
    mask = _time_mask(time_ps, window_ns)
    values = np.full(indices.size, np.nan, dtype=np.float64)
    for start in range(0, indices.size, int(batch_size)):
        stop = min(indices.size, start + int(batch_size))
        view = waveform_view(dataset, mode, indices[start:stop])
        physical = inverse_pair(dataset, mode, view.materialize())
        difference = np.asarray(physical[:, 0, mask] - physical[:, 1, mask], dtype=np.float64)
        finite = np.all(np.isfinite(difference), axis=1)
        batch_values = np.full(difference.shape[0], np.nan)
        if np.any(finite):
            batch_values[finite] = _distance_from_difference(difference[finite], metric)
        values[start:stop] = batch_values
    return values, time_ps[mask]


def _source_metadata(dataset, indices):
    source_dataset_path = dataset.directory / "source_dataset.npy"
    source_event_path = dataset.directory / "source_event_index.npy"
    if source_dataset_path.is_file():
        source_dataset = np.asarray(np.load(source_dataset_path, mmap_mode="r")[indices]).astype(str)
    else:
        source = Path(str(dataset.manifest.get("source", dataset.directory.name))).stem
        source_dataset = np.full(indices.size, source, dtype="U128")
    if source_event_path.is_file():
        source_event = np.asarray(np.load(source_event_path, mmap_mode="r")[indices], dtype=np.int64)
    else:
        source_event = np.asarray(dataset.event_index[indices], dtype=np.int64)
    return source_dataset, source_event


def _correlation(distance, absolute_error):
    finite = np.isfinite(distance) & np.isfinite(absolute_error)
    x, y = np.asarray(distance[finite], float), np.asarray(absolute_error[finite], float)
    result = {"n": int(x.size), "distance_mean_mV": float(np.mean(x)), "distance_std_mV": float(np.std(x)),
              "abs_led_error_mean_ps": float(np.mean(y)), "abs_led_error_median_ps": float(np.median(y)),
              "pearson_r": float("nan"), "pearson_p": float("nan"), "spearman_rho": float("nan"),
              "spearman_p": float("nan"), "linear_slope_ps_per_mV": float("nan"),
              "linear_intercept_ps": float("nan"), "linear_r_squared": float("nan")}
    if x.size >= 3 and np.std(x) > 0 and np.std(y) > 0:
        p = pearsonr(x, y); s = spearmanr(x, y); r = linregress(x, y)
        result.update(pearson_r=float(p.statistic), pearson_p=float(p.pvalue),
                      spearman_rho=float(s.statistic), spearman_p=float(s.pvalue),
                      linear_slope_ps_per_mV=float(r.slope), linear_intercept_ps=float(r.intercept),
                      linear_r_squared=float(r.rvalue**2))
    return result


def _binned_trend(distance, absolute_error, bins):
    finite = np.isfinite(distance) & np.isfinite(absolute_error)
    x, y = np.asarray(distance[finite], float), np.asarray(absolute_error[finite], float)
    edges = np.unique(np.quantile(x, np.linspace(0, 1, max(2, int(bins)) + 1)))
    rows = []
    for i, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (x >= left) & (x <= right if i == len(edges) - 2 else x < right)
        if not np.any(mask): continue
        v = y[mask]
        rows.append({"bin": i + 1, "n": int(mask.sum()), "distance_low_mV": float(left),
                     "distance_high_mV": float(right), "distance_median_mV": float(np.median(x[mask])),
                     "abs_led_error_mean_ps": float(np.mean(v)), "abs_led_error_median_ps": float(np.median(v)),
                     "abs_led_error_q16_ps": float(np.quantile(v, .16)), "abs_led_error_q84_ps": float(np.quantile(v, .84))})
    return rows


def _write_csv(path, rows):
    if not rows: return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def _plot_relation(path, dataset_name, stage, metric, distance, absolute_error, trend, statistics):
    finite = np.isfinite(distance) & np.isfinite(absolute_error)
    x, y = distance[finite], absolute_error[finite]
    fig, ax = plt.subplots(figsize=(8.4, 5.6)); ax.scatter(x, y, s=8, alpha=.12, label="events")
    if trend:
        tx = np.array([r["distance_median_mV"] for r in trend]); ty = np.array([r["abs_led_error_median_ps"] for r in trend])
        low = ty - np.array([r["abs_led_error_q16_ps"] for r in trend]); high = np.array([r["abs_led_error_q84_ps"] for r in trend]) - ty
        ax.errorbar(tx, ty, yerr=np.vstack([low, high]), marker="o", capsize=3, lw=1.4, label="distance-bin median ± 16–84%")
    ax.set_xlabel(f"Waveform-channel {metric} distance [mV]"); ax.set_ylabel("Absolute calibrated LED error [ps]")
    ax.set_title(f"{dataset_name} · {stage} · waveform distance vs |LED error|"); ax.grid(alpha=.2); ax.legend(loc="best")
    ax.text(.02, .98, f"n={statistics['n']}\nPearson r={statistics['pearson_r']:+.3f}\nSpearman ρ={statistics['spearman_rho']:+.3f}\nlinear R²={statistics['linear_r_squared']:.3f}", transform=ax.transAxes, ha="left", va="top", bbox={"boxstyle":"round,pad=0.3","facecolor":"white","alpha":.9})
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def _efficiency_grid(step_percent):
    step = float(step_percent)
    if not 0 < step <= 100: raise ValueError("--efficiency-step-percent must satisfy 0 < STEP <= 100")
    return np.append(np.arange(step, 100.0, step), 100.0)


def _selection_scan(distance, signed_led_error, fit_config, *, efficiency_step_percent, bootstrap_samples, seed):
    finite = np.isfinite(distance) & np.isfinite(signed_led_error)
    x, error = np.asarray(distance[finite], float), np.asarray(signed_led_error[finite], float)
    minimum = int(fit_config.get("min_events", 100)); rows = []
    for i, requested_percent in enumerate(_efficiency_grid(efficiency_step_percent)):
        threshold = float(np.quantile(x, requested_percent / 100.0)); accepted = x <= threshold; n = int(accepted.sum())
        if n < minimum: continue
        local_fit = dict(fit_config); local_fit["bootstrap_samples"] = int(bootstrap_samples)
        try: result = fit_ctr_ps(error[accepted], local_fit, seed=int(seed + i), bootstrap=True)
        except ValueError: continue
        rows.append({"requested_efficiency_percent": float(requested_percent), "efficiency_percent": float(100*n/x.size),
                     "distance_threshold_mV": threshold, "n_accepted": n, "n_total": int(x.size),
                     "ctr_ps": float(result.ctr_ps), "ctr_uncertainty_ps": float(result.ctr_error_ps),
                     "bootstrap_successful": int(result.bootstrap_successful),
                     "mean_abs_led_error_ps": float(np.mean(np.abs(error[accepted]))),
                     "median_abs_led_error_ps": float(np.median(np.abs(error[accepted])))})
    return rows


def _plot_selection_scan(path, dataset_name, stage, metric, rows):
    if not rows: return
    e=np.array([r["efficiency_percent"] for r in rows]); c=np.array([r["ctr_ps"] for r in rows]); u=np.array([r["ctr_uncertainty_ps"] for r in rows])
    fig, ax=plt.subplots(figsize=(8.2,5.0)); ax.errorbar(e,c,yerr=u,marker=".",ms=4,capsize=2,lw=1.0)
    ax.set_xlabel("Accepted events with lowest waveform distance [%]"); ax.set_ylabel("LED CTR FWHM [ps]")
    ax.set_title(f"{dataset_name} · {stage} · low-{metric}-distance selection"); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(path,bbox_inches="tight"); plt.close(fig)


def _selected_values(distance, signed_led_error, threshold_mV):
    finite=np.isfinite(distance)&np.isfinite(signed_led_error)
    return np.asarray(signed_led_error[finite & (distance <= float(threshold_mV))], float)


def _median_centered_edges(values, width_ps):
    values=np.asarray(values,float); values=values[np.isfinite(values)]; width=float(width_ps)
    low,high=np.quantile(values,[.005,.995]); margin=.06*(high-low); low-=margin; high+=margin
    median=float(np.median(values)); anchor=median-.5*width; start=anchor-max(0,int(np.ceil((anchor-low)/width)))*width
    if start>low: start-=width
    count=max(3,int(np.ceil((high-start)/width))); edges=start+np.arange(count+1)*width
    if edges[-1]<high: edges=np.append(edges,edges[-1]+width)
    return edges


def _plot_best_ctr_distribution(path,dataset_name,stage,metric,selected_error,best,fit_config,*,bootstrap_samples,seed):
    local_fit=dict(fit_config); local_fit["bootstrap_samples"]=int(bootstrap_samples)
    result=fit_ctr_ps(selected_error,local_fit,seed=int(seed),bootstrap=True); edges=_median_centered_edges(selected_error,float(local_fit["bin_width_ps"]))
    fig,ax=plt.subplots(figsize=(8.4,5.1)); ax.hist(selected_error,bins=edges,histtype="stepfilled",alpha=.42,edgecolor="black")
    ax.axvline(float(result.left_half_ps),ls="--",lw=1.4,label="FWHM crossings"); ax.axvline(float(result.right_half_ps),ls="--",lw=1.4)
    ax.set_xlabel("Calibrated LED timing residual [ps]"); ax.set_ylabel("Events / bin")
    ax.set_title(f"{dataset_name} · best low-{metric}-distance selection ({stage})"); ax.grid(axis="y",alpha=.2); ax.legend(loc="best")
    ax.text(.02,.97,f"efficiency = {float(best['efficiency_percent']):.1f}%\ndistance ≤ {float(best['distance_threshold_mV']):.4g} mV\nn = {int(best['n_accepted'])}\nCTR = {float(result.ctr_ps):.2f} ± {float(result.ctr_error_ps):.2f} ps",transform=ax.transAxes,ha="left",va="top",bbox={"boxstyle":"round,pad=0.3","facecolor":"white","alpha":.9})
    fig.tight_layout(); fig.savefig(path,bbox_inches="tight"); plt.close(fig)
    return {**best,"recomputed_ctr_ps":float(result.ctr_ps),"recomputed_ctr_uncertainty_ps":float(result.ctr_error_ps),"fwhm_left_ps":float(result.left_half_ps),"fwhm_right_ps":float(result.right_half_ps),"bootstrap_successful":int(result.bootstrap_successful)}


def _per_voltage_summary(distance, absolute_error, voltage):
    rows=[]; voltage=np.asarray(voltage,float)
    for value in np.unique(voltage[np.isfinite(voltage)]):
        mask=np.isclose(voltage,value,rtol=0,atol=1e-9); rows.append({"voltage_V":float(value),**_correlation(distance[mask],absolute_error[mask])})
    return rows


def analyse_dataset(run,manifest,dataset_name,args,output_root):
    dataset_info=manifest["datasets"][dataset_name]; prepared_dir=Path(dataset_info["prepared_dir"]); dataset=load_prepared_dataset(prepared_dir)
    mode=str(manifest.get("mode") or manifest["config"]["mode"]); indices=_stage_indices(dataset,args.stage)
    window_ns=None if args.window_ns is None else tuple(map(float,args.window_ns))
    distance,distance_time_ps=_waveform_distance(dataset,mode,indices,metric=args.distance,window_ns=window_ns,batch_size=int(args.batch_size))
    signed_error=np.asarray(calibrated_led(dataset,mode)[indices],float); absolute_error=np.abs(signed_error)
    finite=np.isfinite(distance)&np.isfinite(signed_error)
    if finite.sum()<3: raise RuntimeError(f"{dataset_name}: fewer than three finite distance/LED-error pairs")
    source_dataset,source_event=_source_metadata(dataset,indices); voltage=np.asarray(dataset.bias_voltage_V[indices],float)
    statistics=_correlation(distance,absolute_error); trend=_binned_trend(distance,absolute_error,int(args.trend_bins))
    config=manifest.get("config") or {}; fit_config=dict(config.get("fit") or {})
    bootstrap_samples=int(args.bootstrap_samples if args.bootstrap_samples is not None else fit_config.get("bootstrap_samples",100)); seed=int((config.get("validation") or {}).get("seed",0))
    scan=_selection_scan(distance,signed_error,fit_config,efficiency_step_percent=float(args.efficiency_step_percent),bootstrap_samples=bootstrap_samples,seed=seed)
    dataset_dir=output_root/dataset_name; dataset_dir.mkdir(parents=True,exist_ok=True)
    _plot_relation(dataset_dir/"waveform_distance_vs_abs_led_error.pdf",dataset_name,args.stage,args.distance,distance,absolute_error,trend,statistics)
    _plot_selection_scan(dataset_dir/"led_ctr_vs_low_distance_efficiency.pdf",dataset_name,args.stage,args.distance,scan)
    best=None; eligible=[row for row in scan if float(row["efficiency_percent"])>10.0]
    if eligible:
        best_scan=min(eligible,key=lambda row:float(row["ctr_ps"])); selected_error=_selected_values(distance,signed_error,float(best_scan["distance_threshold_mV"]))
        best=_plot_best_ctr_distribution(dataset_dir/"best_low_distance_ctr_distribution.pdf",dataset_name,args.stage,args.distance,selected_error,best_scan,fit_config,bootstrap_samples=bootstrap_samples,seed=seed+100000)
        with (dataset_dir/"best_low_distance_selection.json").open("w",encoding="utf-8") as stream: json.dump(best,stream,indent=2,allow_nan=True); stream.write("\n")
    event_rows=[{"position":i,"prepared_index":int(indices[i]),"source_dataset":str(source_dataset[i]),"source_event_index":int(source_event[i]),"bias_voltage_V":float(voltage[i]),"waveform_distance_mV":float(distance[i]),"led_error_ps":float(signed_error[i]),"abs_led_error_ps":float(absolute_error[i])} for i in range(indices.size)]
    _write_csv(dataset_dir/"events.csv",event_rows); _write_csv(dataset_dir/"distance_binned_error.csv",trend); _write_csv(dataset_dir/"low_distance_selection_scan.csv",scan)
    per_voltage=_per_voltage_summary(distance,absolute_error,voltage); _write_csv(dataset_dir/"correlation_by_voltage.csv",per_voltage)
    actual_window=[float(distance_time_ps[0]/1000),float(distance_time_ps[-1]/1000)] if distance_time_ps.size else [float("nan"),float("nan")]
    summary={"dataset":dataset_name,"stage":args.stage,"mode":mode,"prepared_dir":str(prepared_dir),"distance_metric":args.distance,
             "distance_definition":{"rms":"sqrt(mean_t((s1-s2)^2))","mean_abs":"mean_t(abs(s1-s2))","max_abs":"max_t(abs(s1-s2))"}[args.distance],
             "signal_units":"physical mV after inverse prepared-input normalization","waveforms_are_aligned":"each detector waveform is on the prepared LED/native-anchor-relative ML time grid",
             "requested_window_ns":None if window_ns is None else list(window_ns),"actual_sample_window_ns":actual_window,"led_error_definition":"abs(calibrated_led_pair_residual_ps)",
             "n_requested":int(indices.size),"n_finite_pairs":int(finite.sum()),"correlation":statistics,"per_voltage":per_voltage,
             "efficiency_step_percent":float(args.efficiency_step_percent),"best_selection_above_10_percent":best,
             "selection_scan_note":"Descriptive CTR-versus-efficiency scan on the selected stage. Use development to choose a cut, then freeze it before evaluating test/blind data."}
    with (dataset_dir/"summary.json").open("w",encoding="utf-8") as stream: json.dump(summary,stream,indent=2,allow_nan=True); stream.write("\n")
    print(f"{dataset_name} | stage={args.stage} | n={statistics['n']} | Pearson r={statistics['pearson_r']:+.4f} | Spearman rho={statistics['spearman_rho']:+.4f}")
    if best is not None: print(f"  best selection >10%: CTR={best['recomputed_ctr_ps']:.2f} ± {best['recomputed_ctr_uncertainty_ps']:.2f} ps | efficiency={best['efficiency_percent']:.1f}% | distance <= {best['distance_threshold_mV']:.4g} mV")
    print(f"  outputs: {dataset_dir}")


def main():
    args=parse_args()
    if int(args.batch_size)<1: raise ValueError("--batch-size must be positive")
    if int(args.trend_bins)<2: raise ValueError("--trend-bins must be >= 2")
    if not 0<float(args.efficiency_step_percent)<=100: raise ValueError("--efficiency-step-percent must satisfy 0 < STEP <= 100")
    run=args.run_dir.resolve(); manifest=_read_json(run/"manifest.json"); available=list((manifest.get("datasets") or {}).keys())
    if args.dataset:
        wanted=set(args.dataset); datasets=[name for name in available if name in wanted]; missing=wanted-set(datasets)
        if missing: raise FileNotFoundError(f"Requested study dataset(s) not found: {sorted(missing)}")
    else: datasets=available
    if not datasets: raise FileNotFoundError("Study manifest contains no datasets")
    if args.stage=="test": print("WARNING: analysing the blind/test population. Treat this as confirmation only; do not tune a distance cut on these results.")
    output_root=(args.output_dir or run/"waveform_distance_led_error").resolve(); output_root.mkdir(parents=True,exist_ok=True)
    for dataset_name in datasets: analyse_dataset(run,manifest,dataset_name,args,output_root)


if __name__ == "__main__": main()
