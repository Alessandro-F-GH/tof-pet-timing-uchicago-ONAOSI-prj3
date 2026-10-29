from __future__ import annotations
import argparse, json
from pathlib import Path
from .core.logging import configure_logging
from .engine.batch import run_batch
from .core.config import load_batch_config, public_batch_config
from .reporting.postprocess import remake_plots

PROJECT_ROOT = Path(__file__).resolve().parent


def _config_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_file():
        return path
    candidate = PROJECT_ROOT / path
    return candidate if candidate.is_file() else path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m waveform_analysis.cli",
        description="TOF-PET control → development model selection → blind evaluation pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser(
        "check-batch", help="Resolve and validate a batch configuration"
    )
    check.add_argument("--config", type=Path, required=True)
    batch = sub.add_parser("batch", help="Run or resume a dependency-aware batch")
    batch.add_argument("--config", type=Path, required=True)
    plots = sub.add_parser(
        "plots", help="Regenerate plots/reports from a result root only"
    )
    plots.add_argument("--results", type=Path, required=True)
    report = sub.add_parser(
        "report", help="Regenerate plots and report without retraining"
    )
    report.add_argument("results", type=Path)
    diagnostic = sub.add_parser("diagnostics", help="Explicitly generate missing development predictions from saved models")
    diagnostic.add_argument("--results", type=Path, required=True)
    tshap = sub.add_parser("explain-tshap", help="TSHAP temporal explanations from frozen saved models")
    tshap.add_argument("--results", type=Path, required=True)
    tshap.add_argument("--models", nargs="+", metavar="MODEL")
    tshap.add_argument("--cache-root", type=Path)
    tshap.add_argument("--output-root", type=Path, help="Dedicated XAI output directory (default: ./xai, outside --results)")
    tshap.add_argument("--max-events", type=int, default=100, help="Maximum blind events per model (default: 100)")
    tshap.add_argument("--n-baselines", type=int, default=8, help="Control reference events (default: 8)")
    tshap.add_argument("--window-samples", type=int, default=20)
    tshap.add_argument("--stride-samples", type=int, default=10)
    tshap.add_argument("--predict-batch-size", type=int, default=64)
    tshap.add_argument("--seed", type=int, default=42,
                       help="Fixed sampling seed shared across models (default: 42)")
    tshap.add_argument("--roi", action="store_true", help="Calculate upstream TSHAP-ROI as well")
    tshap.add_argument("--overwrite", action="store_true", help="Replace matching existing explanations")
    tshap.add_argument("--imputation", choices=("reference", "hermite_ppca"), default="reference",
                       help="Window perturbation: original reference swapping or Hermite-PPCA imputation")
    tshap.add_argument("--imputer-components", type=int, default=32)
    tshap.add_argument("--imputer-fit-events", type=int, default=2000)
    tshap.add_argument("--imputer-tuning-events", type=int, default=300)
    tshap.add_argument("--imputer-slope-points", type=int, default=5)

    tshap_local = sub.add_parser(
        "explain-tshap-local",
        help="Explain representative good, bad and typical blind events individually",
    )
    tshap_local.add_argument("--results", type=Path, required=True)
    tshap_local.add_argument("--models", nargs="+", metavar="MODEL")
    tshap_local.add_argument("--cache-root", type=Path)
    tshap_local.add_argument("--output-root", type=Path, help="Dedicated XAI output directory (default: ./xai, outside --results)")
    tshap_local.add_argument("--events-per-group", type=int, default=2,
                             help="Number of events per local category (default: 2; 10 total)")
    tshap_local.add_argument("--extreme-margin", type=float, default=0.05,
                             help="Exclude the most extreme margin of each good/bad tail (default: 0.05)")
    tshap_local.add_argument("--n-baselines", type=int, default=8)
    tshap_local.add_argument("--window-samples", type=int, default=20)
    tshap_local.add_argument("--stride-samples", type=int, default=10)
    tshap_local.add_argument("--predict-batch-size", type=int, default=64)
    tshap_local.add_argument("--seed", type=int, default=42,
                             help="Fixed control-baseline sampling seed (default: 42)")
    tshap_local.add_argument("--overwrite", action="store_true",
                             help="Force recomputation of local explanations")
    for command in (plots, report):
        command.add_argument(
            "--exclude-models", nargs="+", default=[], metavar="MODEL",
            help="Model names to omit from reporting only (space-separated)",
        )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "check-batch":
        batch = load_batch_config(_config_path(args.config), PROJECT_ROOT)
        print(json.dumps(public_batch_config(batch), indent=2))
        return
    logger = configure_logging()
    if args.command == "explain-tshap":
        from .engine.tshap_explainer import explain_tshap
        for output in explain_tshap(args.results, models=args.models, cache_root=args.cache_root,
                output_root=args.output_root, max_events=args.max_events, n_baselines=args.n_baselines,
                window_samples=args.window_samples, stride_samples=args.stride_samples,
                predict_batch_size=args.predict_batch_size, seed=args.seed, roi=args.roi,
                overwrite=args.overwrite, imputation=args.imputation,
                imputer_components=args.imputer_components,
                imputer_fit_events=args.imputer_fit_events,
                imputer_tuning_events=args.imputer_tuning_events,
                imputer_slope_points=args.imputer_slope_points, logger=logger):
            print(output)
        return
    if args.command == "explain-tshap-local":
        from .engine.tshap_explainer import explain_tshap_local
        for output in explain_tshap_local(
                args.results, models=args.models, cache_root=args.cache_root,
                output_root=args.output_root, events_per_group=args.events_per_group,
                extreme_margin=args.extreme_margin, n_baselines=args.n_baselines,
                window_samples=args.window_samples, stride_samples=args.stride_samples,
                predict_batch_size=args.predict_batch_size, seed=args.seed,
                overwrite=args.overwrite, logger=logger):
            print(output)
        return
    if args.command in {"plots", "report"}:
        print(remake_plots(
            Path(args.results), logger=logger, exclude_models=args.exclude_models,
        ))
        return
    if args.command == "diagnostics":
        from .data.results import discover_runs
        from .engine.development_diagnostic import ensure_development_predictions
        for run in discover_runs(args.results):
            print(ensure_development_predictions(run, logger=logger))
        return
    batch = load_batch_config(_config_path(args.config), PROJECT_ROOT)
    for output in run_batch(batch, logger=logger):
        print(output)


if __name__ == "__main__":
    main()
