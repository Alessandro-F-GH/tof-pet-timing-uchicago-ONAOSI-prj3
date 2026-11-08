"""Three comparable heatmap analyses on the same full wide waveform population.

Runs from saved development predictions only: wide, Onishi, and wide - Onishi.
All three use the SAME intersection of event IDs, SAME reference wide waveform,
seed, train/tuning/holdout split, normalization procedure and plotting grid.

Windows PowerShell:
  python -m waveform_analysis.scripts.compare_window_heatmap_triplet \
    --wide-model-dir "path/to/energy/wide/best_model" \
    --onishi-model-dir "path/to/energy/onishi/model_name" \
    --cache-root "$env:USERPROFILE/UC_cache" \
    --output-root "$env:USERPROFILE/xai/UC/window_triplet"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from waveform_analysis.scripts.plot_conditional_target_heatmaps import (
    parser as heatmap_parser, run as heatmap_run,
    REPRESENTATIONS,
)


TARGETS = (
    ("wide", "reference"),
    ("onishi_on_wide", "comparison"),
    ("wide_minus_onishi", "difference"),
)


def construct_jobs(args):
    """All jobs use wide as waveform reference and Onishi as comparison source."""
    if args.wide_model_dir.expanduser().resolve() == args.onishi_model_dir.expanduser().resolve():
        raise ValueError("Wide and Onishi model directories must differ")
    root = args.output_root.expanduser().resolve()
    jobs = []
    for name, target in TARGETS:
        cli = [
            "--model-dir", str(args.wide_model_dir),
            "--comparison-model-dir", str(args.onishi_model_dir),
            "--comparison-target", target,
            "--output-root", str(root / name),
        ]
        if args.cache_root is not None:
            cli.extend(["--cache-root", str(args.cache_root)])
        if args.artifact_root is not None:
            cli.extend(["--artifact-root", str(args.artifact_root)])
        single = heatmap_parser().parse_args(cli)
        for key in (
            "seed", "tuning_queries", "holdout_queries", "chunk_size",
            "normalization_scale_floor_ratio", "time_stride", "z_bins", "z_limit",
            "min_count", "representations", "baseline_start_ns",
            "baseline_end_ns", "peak_floor", "retained_only",
            "include_original_derivative", "derivative_end_sample",
            "onishi_start_ns", "onishi_end_ns", "disable_onishi_highlight",
        ):
            setattr(single, key, getattr(args, key))
        jobs.append((name, single))
    return jobs


def run(args, *, runner=heatmap_run):
    if args.cache_root and args.artifact_root:
        raise ValueError("Choose --cache-root OR --artifact-root")
    jobs = construct_jobs(args)
    root = args.output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for name, options in jobs:
        print(f"RUN | {name} | label={options.comparison_target} | "
              f"waveforms={options.model_dir} | predictions={options.comparison_model_dir}",
              flush=True)
        runner(options)
        summary_path = options.output_root / "summary.json"
        if not summary_path.is_file():
            raise RuntimeError(f"Heatmap analysis did not create {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        results.append({
            "name": name,
            "prediction_target": options.comparison_target,
            "output_dir": str(options.output_root),
            "matched_population": summary["matched_population"],
            "events": summary["events"],
            "figures": summary["figures"],
        })
        # All analyses MUST agree on aligned events and training subset size.
        if len(results) > 1:
            if (results[-1]["matched_population"] != results[0]["matched_population"]
                    or results[-1]["events"] != results[0]["events"]):
                raise RuntimeError(
                    "Triplet populations differ; comparisons are not valid")
    report = {
        "wide_model_dir": str(args.wide_model_dir.expanduser().resolve()),
        "onishi_model_dir": str(args.onishi_model_dir.expanduser().resolve()),
        "description": "All three maps use common events and wide input waveforms",
        "plots": results,
    }
    (root / "triplet_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(f"COMPLETE | 3 analyses | {root}", flush=True)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wide-model-dir", type=Path, required=True)
    p.add_argument("--onishi-model-dir", type=Path, required=True)
    p.add_argument("--cache-root", type=Path)
    p.add_argument("--artifact-root", type=Path)
    p.add_argument("--output-root", type=Path, required=True)
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
    p.add_argument("--retained-only", action="store_true")
    p.add_argument("--include-original-derivative", action="store_true")
    p.add_argument("--derivative-end-sample", type=int, default=1200)
    p.add_argument("--onishi-start-ns", type=float)
    p.add_argument("--onishi-end-ns", type=float)
    p.add_argument("--disable-onishi-highlight", action="store_true")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
