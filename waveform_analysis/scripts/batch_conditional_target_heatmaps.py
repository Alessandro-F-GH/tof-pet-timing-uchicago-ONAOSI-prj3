"""Batch saved-prediction heatmaps for one detector board.

Canonical input tree: RESULTS_ROOT/{energy,timing}/WINDOW/MODEL.
All selected individual models and unordered pairs are analyzed within their
own mode, never across energy and timing. No ML inference is performed.
"""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path

from waveform_analysis.data.results import read_run, resolve_run_config
from waveform_analysis.scripts.plot_conditional_target_heatmaps import (
    REPRESENTATIONS, parser as single_parser, run as single_run,
)

MODES = {"energy": "energy_to_energy", "timing": "timing_to_timing"}
CONFIG_KEYS = (
    "seed", "tuning_queries", "holdout_queries", "chunk_size",
    "normalization_scale_floor_ratio", "time_stride", "z_bins", "z_limit",
    "min_count", "baseline_start_ns", "baseline_end_ns", "peak_floor",
    "include_original_derivative", "derivative_end_sample", "retained_only",
)


def _names(values):
    names = list(values)
    if len(names) != len(set(names)):
        raise ValueError("Duplicate model names in one mode")
    if any(not n or n in (".", "..") or
           any(c in n for c in ("/", "\\", ":")) for n in names):
        raise ValueError("Each model must be a simple directory name")
    return names


def plan(results_root, output_root, window, energy_models, timing_models):
    root = Path(results_root).expanduser().resolve()
    out = Path(output_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Results root not found: {root}")
    if not window or window in (".", "..") or any(
            c in window for c in ("/", "\\", ":")):
        raise ValueError("Window must be a simple directory name")
    requested = {
        "energy": _names(energy_models),
        "timing": _names(timing_models),
    }
    if not requested["energy"] and not requested["timing"]:
        raise ValueError("Specify --energy-models and/or --timing-models")
    jobs = []
    for mode, models in requested.items():
        paths = {}
        for name in models:
            path = root / mode / window / name
            if not path.is_dir():
                raise FileNotFoundError(
                    f"Model not found: {path}\nExpected MODE/WINDOW/MODEL")
            cfg = resolve_run_config(path)
            if cfg["mode"] != MODES[mode] or cfg["window_name"] != window:
                raise ValueError(
                    f"Mode or window mismatch in run metadata: {path}")
            if read_run(path)["manifest"].get("status") != "complete":
                raise ValueError(f"Run is not complete: {path}")
            paths[name] = path
        for name in models:
            jobs.append({
                "mode": mode, "kind": "individual", "reference": name,
                "comparison": None, "model_dir": paths[name],
                "comparison_dir": None,
                "output": out / mode / window / "individual" / name,
            })
        for ref, cmp in combinations(models, 2):
            jobs.append({
                "mode": mode, "kind": "pairwise", "reference": ref,
                "comparison": cmp, "model_dir": paths[ref],
                "comparison_dir": paths[cmp],
                "output": (out / mode / window / "pairwise" /
                           f"{ref}__minus__{cmp}"),
            })
    return jobs


def _configuration(args):
    return {**{k: getattr(args, k) for k in CONFIG_KEYS},
            "representations": list(args.representations)}


def _job_identity(job, args):
    return {
        "format_version": 1,
        "mode": job["mode"], "kind": job["kind"],
        "reference": job["reference"], "comparison": job["comparison"],
        "model_dir": str(job["model_dir"]),
        "comparison_model_dir": (str(job["comparison_dir"])
                                 if job["comparison_dir"] else None),
        "cache_root": (str(args.cache_root.expanduser().resolve())
                       if args.cache_root else None),
        "artifact_root": (str(args.artifact_root.expanduser().resolve())
                          if args.artifact_root else None),
        "settings": _configuration(args),
    }


def _is_complete(job, identity):
    folder = job["output"]
    try:
        record = json.loads(
            (folder / "batch_job.json").read_text(encoding="utf-8"))
        summary = json.loads(
            (folder / "summary.json").read_text(encoding="utf-8"))
        if record != identity:
            return False
        representations = summary["representations"]
        if representations != identity["settings"]["representations"]:
            return False
        if not all((folder / rep / "conditional_target_heatmaps.npz").is_file()
                   for rep in representations):
            return False
        return bool(summary.get("figures")) and all(
            (folder / fig).is_file() for fig in summary["figures"])
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _single_args(job, args):
    cli = [
        "--model-dir", str(job["model_dir"]),
        "--output-root", str(job["output"]),
    ]
    if job["comparison_dir"]:
        cli += ["--comparison-model-dir", str(job["comparison_dir"])]
    if args.cache_root:
        cli += ["--cache-root", str(args.cache_root)]
    if args.artifact_root:
        cli += ["--artifact-root", str(args.artifact_root)]
    single = single_parser().parse_args(cli)
    for key, value in _configuration(args).items():
        setattr(single, key, value)
    return single


def run(args, *, runner=single_run):
    if args.cache_root and args.artifact_root:
        raise ValueError("Choose --cache-root OR --artifact-root")
    jobs = plan(args.results_root, args.output_root, args.window,
                args.energy_models, args.timing_models)
    print(f"PLAN | {len(jobs)} jobs | window={args.window}", flush=True)
    for i, job in enumerate(jobs, 1):
        name = job["reference"]
        if job["comparison"]:
            name += " - " + job["comparison"]
        print(f"{i}/{len(jobs)} | {job['mode']} | {job['kind']} | {name}",
              flush=True)
    if args.dry_run:
        return {"planned": len(jobs), "dry_run": True}

    root = args.output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    status = []
    for i, job in enumerate(jobs, 1):
        identity = _job_identity(job, args)
        state = "complete"
        error = None
        if not args.overwrite and _is_complete(job, identity):
            state = "skipped"
            print(f"SKIP {i}/{len(jobs)} | {job['output']}", flush=True)
        else:
            print(f"RUN {i}/{len(jobs)} | {job['output']}", flush=True)
            try:
                runner(_single_args(job, args))
                # This marker is written only once the single-run script
                # finishes. Interrupted/failed jobs are rerun next time.
                (job["output"] / "batch_job.json").write_text(
                    json.dumps(identity, indent=2), encoding="utf-8")
            except Exception as exc:
                state = "failed"
                error = f"{type(exc).__name__}: {exc}"
                print(f"FAILED | {error}", flush=True)
        item = {
            "mode": job["mode"], "kind": job["kind"],
            "reference": job["reference"], "comparison": job["comparison"],
            "output_dir": str(job["output"]), "status": state,
        }
        if error:
            item["error"] = error
        status.append(item)
        summary = {
            "results_root": str(args.results_root.expanduser().resolve()),
            "output_root": str(root), "window": args.window,
            "settings": _configuration(args),
            "planned": len(jobs),
            "complete": sum(x["status"] == "complete" for x in status),
            "skipped": sum(x["status"] == "skipped" for x in status),
            "failed": sum(x["status"] == "failed" for x in status),
            "jobs": status,
        }
        (root / "batch_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
        if error and not args.continue_on_error:
            break
    if any(x["status"] == "failed" for x in status):
        raise RuntimeError(
            f"Batch failed; see {root / 'batch_summary.json'}. "
            "Run again to resume completed jobs.")
    print(f"COMPLETE | {len(status)}/{len(jobs)} jobs | output={root}",
          flush=True)
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-root", type=Path, required=True,
                   help="Board results root containing energy/window/model and timing/window/model")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--cache-root", type=Path)
    p.add_argument("--artifact-root", type=Path)
    p.add_argument("--window", default="wide")
    p.add_argument("--energy-models", nargs="*", default=[], metavar="MODEL")
    p.add_argument("--timing-models", nargs="*", default=[], metavar="MODEL")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate the plan without running or writing anything")
    p.add_argument("--overwrite", action="store_true",
                   help="Recalculate previously completed jobs")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tuning-queries", type=int, default=500)
    p.add_argument("--holdout-queries", type=int, default=500)
    p.add_argument("--chunk-size", type=int, default=128)
    p.add_argument("--normalization-scale-floor-ratio", type=float, default=0.05)
    p.add_argument("--time-stride", type=int, default=8)
    p.add_argument("--z-bins", type=int, default=80)
    p.add_argument("--z-limit", type=float, default=3.)
    p.add_argument("--min-count", type=int, default=25)
    p.add_argument("--representations", nargs="+", choices=REPRESENTATIONS,
                   default=list(REPRESENTATIONS))
    p.add_argument("--baseline-start-ns", type=float, default=-2.)
    p.add_argument("--baseline-end-ns", type=float, default=-1.)
    p.add_argument("--peak-floor", type=float, default=1e-8)
    p.add_argument("--include-original-derivative", action="store_true")
    p.add_argument("--derivative-end-sample", type=int, default=1200)
    p.add_argument("--retained-only", action="store_true")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
