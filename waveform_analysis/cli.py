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
    batch.add_argument("--cache-root", type=Path,
                       help="Use a prepared cache bundle")
    prepare = sub.add_parser("prepare-cache", help="Prepare reusable batch caches without model fitting")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--cache-root", type=Path, required=True,
                         help="Directory for the portable preprocessing cache bundle")
    prepare.add_argument("--models", nargs="+", metavar="MODEL",
                         help="Prepare caches only for these model names (default: all batch models)")
    replot = sub.add_parser("plot-imputers",
                            help="Regenerate RMSE plots from fitted benchmark without refitting")
    replot.add_argument("--results", type=Path, required=True)
    replot.add_argument("--cache-root", type=Path, required=True)
    replot.add_argument("--output-root", type=Path, required=True)
    replot.add_argument("--modes", nargs="+", choices=("energy", "timing"),
                        default=["energy", "timing"])
    benchmark = sub.add_parser("benchmark-imputers",
                               help="Fit, tune and freeze reconstruction-RMSE-selected imputers")
    benchmark.add_argument("--results", type=Path, required=True)
    benchmark.add_argument("--cache-root", type=Path, required=True)
    benchmark.add_argument("--output-root", type=Path, required=True)
    benchmark.add_argument("--modes", nargs="+", choices=("energy", "timing"),
                           default=["energy", "timing"])
    benchmark.add_argument("--window-samples", type=int, default=20)
    benchmark.add_argument("--stride-samples", type=int, default=20)
    benchmark.add_argument("--fit-events", type=int, default=2000)
    benchmark.add_argument("--tuning-events", type=int, default=200)
    benchmark.add_argument("--evaluation-events", type=int, default=200)
    benchmark.add_argument("--ppca-components", type=int, nargs="+", default=[8, 16, 32])
    benchmark.add_argument("--knn-k", type=int, nargs="+", default=[1, 3, 5, 10, 20])
    benchmark.add_argument("--n-baselines", type=int, default=8)
    benchmark.add_argument("--seed", type=int, default=42)
    benchmark.add_argument("--overwrite", action="store_true")
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
    linear_local = sub.add_parser(
        "explain-linear-shap-local",
        help="Exact local Linear SHAP for saved direct/shared Ridge models",
    )
    linear_source = linear_local.add_mutually_exclusive_group(required=True)
    linear_source.add_argument("--results", type=Path)
    linear_source.add_argument("--model-dir", type=Path)
    linear_local.add_argument("--models", nargs="+", choices=(
        "direct_linear_ridge", "shared_linear_ridge",
    ))
    linear_local.add_argument("--windows", nargs="+", choices=("onishi", "wide", "all"))
    linear_local.add_argument("--mode", choices=("energy", "timing"))
    linear_local.add_argument("--cache-root", type=Path)
    linear_local.add_argument("--output-root", type=Path)
    linear_local.add_argument("--events-per-group", type=int, default=2)
    linear_local.add_argument("--extreme-margin", type=float, default=0.05)
    linear_local.add_argument("--group-samples", type=int, default=8)
    linear_local.add_argument("--overwrite", action="store_true")
    linear_global = sub.add_parser(
        "explain-linear-shap-global",
        help="Mean exact Linear SHAP over randomly sampled blind events",
    )
    linear_global_source = linear_global.add_mutually_exclusive_group(required=True)
    linear_global_source.add_argument("--results", type=Path)
    linear_global_source.add_argument("--model-dir", type=Path)
    linear_global.add_argument("--models", nargs="+", choices=(
        "direct_linear_ridge", "shared_linear_ridge",
    ))
    linear_global.add_argument("--windows", nargs="+", choices=("onishi", "wide", "all"))
    linear_global.add_argument("--mode", choices=("energy", "timing"))
    linear_global.add_argument("--cache-root", type=Path)
    linear_global.add_argument("--output-root", type=Path)
    linear_global.add_argument("--max-events", type=int, default=256)
    linear_global.add_argument("--seed", type=int, default=42)
    linear_global.add_argument("--group-samples", type=int, default=8)
    linear_global.add_argument("--overwrite", action="store_true")
    tshap = sub.add_parser("explain-tshap", help="TSHAP temporal explanations from frozen saved models")
    tshap_source = tshap.add_mutually_exclusive_group(required=True)
    tshap_source.add_argument("--results", type=Path,
                              help="Explain every complete model run below this result root")
    tshap_source.add_argument("--model-dir", type=Path,
                              help="Explain one run: directory containing metadata/manifest.json, or its local model/ subdirectory")
    tshap.add_argument("--models", nargs="+", metavar="MODEL",
                       help="One or more model names to explain")
    tshap.add_argument("--windows", nargs="+", choices=("onishi", "wide", "all"), default=None,
                       metavar="WINDOW", help="One or more ML windows to explain (onishi, wide; or all)")
    tshap.add_argument("--mode", choices=("energy", "timing"),
                       help="Restrict the batch to energy or timing runs")
    tshap_cache = tshap.add_mutually_exclusive_group()
    tshap_cache.add_argument("--cache-root", type=Path,
                             help="Shared cache bundle (uses preprocessing/)")
    tshap.add_argument("--output-root", type=Path, help="Dedicated XAI output directory (default: ./xai, outside --results)")
    tshap.add_argument("--max-events", type=int, default=100, help="Maximum blind events per model (default: 100)")
    tshap.add_argument("--window-samples", type=int, default=20)
    tshap.add_argument("--stride-samples", type=int, default=10)
    tshap.add_argument("--predict-batch-size", type=int, default=64)
    tshap.add_argument("--seed", type=int, default=42,
                       help="Fixed sampling seed shared across models (default: 42)")
    tshap.add_argument("--roi", action="store_true", help="Calculate upstream TSHAP-ROI as well")
    tshap.add_argument("--overwrite", action="store_true", help="Replace matching existing explanations")
    tshap.add_argument("--n-baselines", type=int, default=8,
                       help="Real control baseline waveforms used for both coalitions")
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
    tshap_local.add_argument("--models", nargs="+", metavar="MODEL",
                             help="One or more model names to explain")
    tshap_local.add_argument("--windows", nargs="+", choices=("onishi", "wide", "all"), default=None,
                             metavar="WINDOW", help="One or more ML windows to explain (onishi, wide; or all)")
    tshap_local.add_argument("--mode", choices=("energy", "timing"),
                             help="Restrict the batch to energy or timing runs")
    tshap_local_cache = tshap_local.add_mutually_exclusive_group()
    tshap_local_cache.add_argument("--cache-root", type=Path,
                                   help="Shared cache bundle (preprocessing/)")
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
    tshap_local.add_argument("--n-baselines", type=int, default=8,
                             help="Real control baseline waveforms used for both coalitions")
    tshap_local.add_argument("--imputer-components", type=int, default=32)
    tshap_local.add_argument("--imputer-fit-events", type=int, default=2000)
    tshap_local.add_argument("--imputer-tuning-events", type=int, default=300)
    tshap_local.add_argument("--imputer-slope-points", type=int, default=5)
    tshap_local.add_argument("--overwrite", action="store_true",
                             help="Force recomputation of local explanations")
    for cmd in (tshap, tshap_local):
        cmd.add_argument("--imputer-root", type=Path,
                         help="Previously fitted benchmark-imputers output; load winners without fitting")
        cmd.add_argument("--method", choices=("reference", "conditional-knn"), default="reference",
                         help="Choose original control-reference Shapley or deterministic conditional KNN")
        cmd.add_argument("--knn-k", nargs="+", type=int, default=[1, 3, 5, 10, 20, 50, 100],
                         help="KNN candidate neighbor counts; each window and complement is tuned independently")
        cmd.add_argument("--knn-tuning-events", type=int, default=512,
                         help="Development events held out for KNN reconstruction MSE tuning")
        cmd.add_argument("--knn-reference-events", type=int, default=None,
                         help="Maximum remaining Development events used as KNN reference (default: all)")
        cmd.add_argument("--knn-reference-block", type=int, default=512,
                         help="Number of reference waveforms processed per distance block")
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
    if args.command == "plot-imputers":
        from .engine.imputer_selection import plot_existing_imputers
        for output in plot_existing_imputers(
            args.results, cache_root=args.cache_root,
            output_root=args.output_root, modes=args.modes, logger=logger,
        ):
            print(output)
        return
    if args.command == "benchmark-imputers":
        from .engine.imputer_selection import benchmark_imputers
        for output in benchmark_imputers(
            args.results, cache_root=args.cache_root, output_root=args.output_root,
            modes=args.modes, window_samples=args.window_samples,
            stride_samples=args.stride_samples, fit_events=args.fit_events,
            tuning_events=args.tuning_events, evaluation_events=args.evaluation_events,
            n_baselines=args.n_baselines, components=args.ppca_components,
            knn_k=args.knn_k, seed=args.seed, overwrite=args.overwrite,
            logger=logger,
        ):
            print(output)
        return
    if args.command == "explain-linear-shap-local":
        from .engine.linear_shap import explain_linear_shap_local
        for output in explain_linear_shap_local(
            args.results, model_dir=args.model_dir, models=args.models,
            ml_window=args.windows, mode=args.mode,
            cache_root=args.cache_root, output_root=args.output_root,
            events_per_group=args.events_per_group,
            extreme_margin=args.extreme_margin,
            group_samples=args.group_samples, overwrite=args.overwrite,
            logger=logger,
        ):
            print(output)
        return
    if args.command == "explain-linear-shap-global":
        from .engine.linear_shap import explain_linear_shap_global
        for output in explain_linear_shap_global(
            args.results, model_dir=args.model_dir, models=args.models,
            ml_window=args.windows, mode=args.mode,
            cache_root=args.cache_root, output_root=args.output_root,
            max_events=args.max_events, seed=args.seed,
            group_samples=args.group_samples, overwrite=args.overwrite,
            logger=logger,
        ):
            print(output)
        return
    if args.command == "explain-tshap" and args.imputer_root is not None:
        from .engine.frozen_imputer_tshap import explain_frozen
        for output in explain_frozen(
            args.results, model_dir=args.model_dir, models=args.models,
            ml_window=args.windows, mode=args.mode, cache_root=args.cache_root,
            output_root=args.output_root, imputer_root=args.imputer_root,
            max_events=args.max_events, window_samples=args.window_samples,
            stride_samples=args.stride_samples, predict_batch_size=args.predict_batch_size,
            seed=args.seed, overwrite=args.overwrite, local=False, logger=logger,
        ):
            print(output)
        return
    if args.command == "explain-tshap-local" and args.imputer_root is not None:
        from .engine.frozen_imputer_tshap import explain_frozen
        for output in explain_frozen(
            args.results, model_dir=args.model_dir, models=args.models,
            ml_window=args.windows, mode=args.mode, cache_root=args.cache_root,
            output_root=args.output_root, imputer_root=args.imputer_root,
            window_samples=args.window_samples, stride_samples=args.stride_samples,
            predict_batch_size=args.predict_batch_size, seed=args.seed,
            events_per_group=args.events_per_group, extreme_margin=args.extreme_margin,
            overwrite=args.overwrite, local=True, logger=logger,
        ):
            print(output)
        return
    if args.command == "explain-tshap" and args.method == "conditional-knn":
        from .engine.conditional_knn_tshap import explain_conditional_knn
        for output in explain_conditional_knn(
                args.results, model_dir=args.model_dir, models=args.models,
                ml_window=args.windows, mode=args.mode, cache_root=args.cache_root,
                output_root=args.output_root,
                max_events=args.max_events, window_samples=args.window_samples,
                stride_samples=args.stride_samples, predict_batch_size=args.predict_batch_size,
                seed=args.seed, overwrite=args.overwrite, knn_k=tuple(args.knn_k),
                knn_tuning_events=args.knn_tuning_events,
                knn_reference_events=args.knn_reference_events,
                knn_reference_block=args.knn_reference_block, logger=logger):
            print(output)
        return
    if args.command == "explain-tshap":
        from .engine.tshap_explainer import explain_tshap
        for output in explain_tshap(args.results, model_dir=args.model_dir, models=args.models,
                ml_window=args.windows, mode=args.mode,
                cache_root=args.cache_root, 
                output_root=args.output_root, max_events=args.max_events,
                window_samples=args.window_samples, stride_samples=args.stride_samples,
                predict_batch_size=args.predict_batch_size, seed=args.seed, roi=args.roi,
                overwrite=args.overwrite,
                imputer_components=args.imputer_components,
                imputer_fit_events=args.imputer_fit_events,
                imputer_tuning_events=args.imputer_tuning_events,
                imputer_slope_points=args.imputer_slope_points,
                n_baselines=args.n_baselines, logger=logger):
            print(output)
        return
    if args.command == "explain-tshap-local" and args.method == "conditional-knn":
        from .engine.conditional_knn_tshap import explain_conditional_knn
        for output in explain_conditional_knn(
                args.results, model_dir=args.model_dir, models=args.models,
                ml_window=args.windows, mode=args.mode, cache_root=args.cache_root,
                output_root=args.output_root,
                local=True, events_per_group=args.events_per_group,
                extreme_margin=args.extreme_margin,
                window_samples=args.window_samples, stride_samples=args.stride_samples,
                predict_batch_size=args.predict_batch_size, seed=args.seed,
                overwrite=args.overwrite, knn_k=tuple(args.knn_k),
                knn_tuning_events=args.knn_tuning_events,
                knn_reference_events=args.knn_reference_events,
                knn_reference_block=args.knn_reference_block, logger=logger):
            print(output)
        return
    if args.command == "explain-tshap-local":
        from .engine.tshap_explainer import explain_tshap_local
        for output in explain_tshap_local(
                args.results, model_dir=args.model_dir, models=args.models,
                ml_window=args.windows, mode=args.mode,
                cache_root=args.cache_root, output_root=args.output_root, events_per_group=args.events_per_group,
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
            print(ensure_development_predictions(run, logger=logger))
        return
    batch = load_batch_config(_config_path(args.config), PROJECT_ROOT)
    if args.command == "prepare-cache":
        from .engine.cache_preparation import prepare_batch_cache
        root = args.cache_root.expanduser().resolve()
        for run in batch.runs:
            run["preprocessing"]["cache_dir"] = str(root / "preprocessing")
        print(prepare_batch_cache(
            batch, cache_root=root, models=args.models, logger=logger
        ))
        return
    if args.cache_root is not None:
        root = args.cache_root.expanduser().resolve()
        for run in batch.runs:
            run["preprocessing"]["cache_dir"] = str(root / "preprocessing")
    for output in run_batch(batch, logger=logger):
        print(output)


if __name__ == "__main__":
    main()
