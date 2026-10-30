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
    check.add_argument("--artifact-root", type=Path,
                       help="Override the shared scientific artifact store location")
    batch = sub.add_parser("batch", help="Run or resume a dependency-aware batch")
    batch.add_argument("--config", type=Path, required=True)
    batch.add_argument("--artifact-root", type=Path,
                       help="Override the shared scientific artifact store location")
    batch.add_argument("--cache-root", type=Path,
                       help="Use a prepared cache bundle (overrides preprocessing and artifact roots)")
    prepare = sub.add_parser("prepare-cache", help="Prepare reusable batch caches without model fitting")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--cache-root", type=Path, required=True,
                         help="Directory for the portable preprocessing and artifact cache bundle")
    prepare.add_argument("--models", nargs="+", metavar="MODEL",
                         help="Prepare caches only for these model names (default: all batch models)")
    migrate = sub.add_parser(
        "migrate-transforms",
        help="Import fitted transformations from historical results into a shared cache",
    )
    migrate.add_argument("--results", type=Path, required=True,
                         help="Historical canonical results root")
    migrate.add_argument("--cache-root", type=Path, required=True,
                         help="Destination prepare-cache bundle root")
    migrate.add_argument("--source-artifact-root", type=Path,
                         help="Artifact store containing externally stored historical models/transforms")
    migrate.add_argument("--models", nargs="+", metavar="MODEL",
                         help="Only import these model names")
    migrate.add_argument("--overwrite", action="store_true",
                         help="Retarget conflicting transform aliases to the selected historical fit")
    migrate.add_argument("--dry-run", action="store_true",
                         help="Validate and list imports without writing artifacts")
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
    diagnostic.add_argument("--artifact-root", type=Path,
                             help="Override the shared artifact store location")
    tshap = sub.add_parser("explain-tshap", help="TSHAP temporal explanations from frozen saved models")
    tshap_source = tshap.add_mutually_exclusive_group(required=True)
    tshap_source.add_argument("--results", type=Path,
                              help="Explain every complete model run below this result root")
    tshap_source.add_argument("--model-dir", type=Path,
                              help="Explain one run: directory containing metadata/manifest.json, or its local model/ subdirectory")
    tshap.add_argument("--models", nargs="+", metavar="MODEL")
    tshap.add_argument("--ml-window", choices=("onishi", "wide", "all"), default=None,
                       help="Window filter; defaults to wide for --results and to the selected run's window for --model-dir")
    tshap_cache = tshap.add_mutually_exclusive_group()
    tshap_cache.add_argument("--cache-root", type=Path,
                             help="Shared prepare-cache bundle (uses preprocessing/ and artifact_store/)")
    tshap_cache.add_argument("--artifact-root", type=Path,
                             help="Override only the shared scientific artifact store location")
    tshap.add_argument("--output-root", type=Path, help="Dedicated XAI output directory (default: ./xai, outside --results)")
    tshap.add_argument("--max-events", type=int, default=100, help="Maximum blind events per model (default: 100)")
    tshap.add_argument("--window-samples", type=int, default=20)
    tshap.add_argument("--stride-samples", type=int, default=10)
    tshap.add_argument("--predict-batch-size", type=int, default=64)
    tshap.add_argument("--seed", type=int, default=42,
                       help="Fixed sampling seed shared across models (default: 42)")
    tshap.add_argument("--roi", action="store_true", help="Calculate upstream TSHAP-ROI as well")
    tshap.add_argument("--paired-occlusion", action="store_true",
                       help="Occlude the same temporal window in both detector signals together; report one pair-level importance")
    tshap.add_argument("--overwrite", action="store_true", help="Replace matching existing explanations")
    tshap.add_argument("--imputer-components", type=int, default=32)
    tshap.add_argument("--imputer-fit-events", type=int, default=2000)
    tshap.add_argument("--imputer-tuning-events", type=int, default=300)
    tshap.add_argument("--imputer-slope-points", type=int, default=5)

    tshap_local = sub.add_parser(
        "explain-tshap-local",
        help="Explain representative good, bad and typical blind events individually",
    )
    tshap_local_source = tshap_local.add_mutually_exclusive_group(required=True)
    tshap_local_source.add_argument("--results", type=Path,
                                    help="Explain every complete model run below this result root")
    tshap_local_source.add_argument("--model-dir", type=Path,
                                    help="Explain one run: directory containing metadata/manifest.json, or its local model/ subdirectory")
    tshap_local.add_argument("--models", nargs="+", metavar="MODEL")
    tshap_local.add_argument("--ml-window", choices=("onishi", "wide", "all"), default=None,
                             help="Window filter; defaults to wide for --results and to the selected run's window for --model-dir")
    tshap_local_cache = tshap_local.add_mutually_exclusive_group()
    tshap_local_cache.add_argument("--cache-root", type=Path,
                                   help="Shared prepare-cache bundle (uses preprocessing/ and artifact_store/)")
    tshap_local_cache.add_argument("--artifact-root", type=Path,
                                   help="Override only the shared scientific artifact store location")
    tshap_local.add_argument("--output-root", type=Path, help="Dedicated XAI output directory (default: ./xai, outside --results)")
    tshap_local.add_argument("--events-per-group", type=int, default=2,
                             help="Number of events per local category (default: 2; 10 total)")
    tshap_local.add_argument("--extreme-margin", type=float, default=0.05,
                             help="Exclude the most extreme margin of each good/bad tail (default: 0.05)")
    tshap_local.add_argument("--window-samples", type=int, default=20)
    tshap_local.add_argument("--stride-samples", type=int, default=10)
    tshap_local.add_argument("--predict-batch-size", type=int, default=64)
    tshap_local.add_argument("--seed", type=int, default=42,
                             help="Fixed control-baseline sampling seed (default: 42)")
    tshap_local.add_argument("--imputer-components", type=int, default=32)
    tshap_local.add_argument("--imputer-fit-events", type=int, default=2000)
    tshap_local.add_argument("--imputer-tuning-events", type=int, default=300)
    tshap_local.add_argument("--imputer-slope-points", type=int, default=5)
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
        if args.artifact_root is not None:
            artifact_root = str(args.artifact_root.expanduser().resolve())
            for run in batch.runs:
                run["preprocessing"]["artifact_root"] = artifact_root
        print(json.dumps(public_batch_config(batch), indent=2))
        return
    logger = configure_logging()
    if args.command == "explain-tshap":
        from .engine.tshap_explainer import explain_tshap
        for output in explain_tshap(args.results, model_dir=args.model_dir, models=args.models,
                ml_window=args.ml_window, cache_root=args.cache_root, artifact_root=args.artifact_root,
                output_root=args.output_root, max_events=args.max_events,
                window_samples=args.window_samples, stride_samples=args.stride_samples,
                predict_batch_size=args.predict_batch_size, seed=args.seed, roi=args.roi,
                paired_occlusion=args.paired_occlusion, overwrite=args.overwrite,
                imputer_components=args.imputer_components,
                imputer_fit_events=args.imputer_fit_events,
                imputer_tuning_events=args.imputer_tuning_events,
                imputer_slope_points=args.imputer_slope_points, logger=logger):
            print(output)
        return
    if args.command == "explain-tshap-local":
        from .engine.tshap_explainer import explain_tshap_local
        for output in explain_tshap_local(
                args.results, model_dir=args.model_dir, models=args.models, ml_window=args.ml_window,
                cache_root=args.cache_root, artifact_root=args.artifact_root, output_root=args.output_root, events_per_group=args.events_per_group,
                extreme_margin=args.extreme_margin,
                window_samples=args.window_samples, stride_samples=args.stride_samples,
                predict_batch_size=args.predict_batch_size, seed=args.seed,
                overwrite=args.overwrite,
                imputer_components=args.imputer_components,
                imputer_fit_events=args.imputer_fit_events,
                imputer_tuning_events=args.imputer_tuning_events,
                imputer_slope_points=args.imputer_slope_points,
                logger=logger):
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
            print(ensure_development_predictions(run, logger=logger, artifact_root=args.artifact_root))
        return
    if args.command == "migrate-transforms":
        from .engine.transform_migration import import_fitted_transforms
        records = import_fitted_transforms(
            args.results,
            cache_root=args.cache_root,
            source_artifact_root=args.source_artifact_root,
            models=args.models,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            logger=logger,
        )
        print(json.dumps(records, indent=2))
        return
    batch = load_batch_config(_config_path(args.config), PROJECT_ROOT)
    if args.command == "prepare-cache":
        from .engine.cache_preparation import prepare_batch_cache
        root = args.cache_root.expanduser().resolve()
        for run in batch.runs:
            run["preprocessing"]["cache_dir"] = str(root / "preprocessing")
            run["preprocessing"]["artifact_root"] = str(root / "artifact_store")
        print(prepare_batch_cache(
            batch, cache_root=root, models=args.models, logger=logger
        ))
        return
    if args.cache_root is not None:
        root = args.cache_root.expanduser().resolve()
        for run in batch.runs:
            run["preprocessing"]["cache_dir"] = str(root / "preprocessing")
            run["preprocessing"]["artifact_root"] = str(root / "artifact_store")
    elif args.artifact_root is not None:
        artifact_root = str(args.artifact_root.expanduser().resolve())
        for run in batch.runs:
            run["preprocessing"]["artifact_root"] = artifact_root
    for output in run_batch(batch, logger=logger):
        print(output)


if __name__ == "__main__":
    main()
