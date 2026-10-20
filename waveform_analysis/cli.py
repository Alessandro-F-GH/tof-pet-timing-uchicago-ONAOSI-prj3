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
    xai = sub.add_parser("xai", help="Repeat control-mean XAI from saved batch models without retraining")
    xai.add_argument("--results", type=Path, required=True)
    xai.add_argument("--models", nargs="+", metavar="MODEL")
    xai.add_argument("--cache-root", type=Path, help="Location of original prepared control/blind caches")
    xai.add_argument("--group-size-samples", type=int)
    xai.add_argument("--max-events", type=int)
    explain = sub.add_parser("explain-regression", help="Original frozen-feature explanations and tree performance comparison")
    explain.add_argument("--results", type=Path, required=True)
    explain.add_argument("--models", nargs="+", metavar="MODEL")
    explain.add_argument("--cache-root", type=Path, help="Original prepared development/blind caches")
    explain.add_argument("--depths", nargs="+", type=int, default=[2, 3, 4, 5])
    explain.add_argument("--min-samples-leaf", type=int, default=20)
    explain.add_argument("--max-events", type=int, help="Optional reproducible event cap per role; default: all events")
    stratified = sub.add_parser("explain-stratified", help="Read-only output-stratified RDST/MultiRocket explanations")
    stratified.add_argument("--results", type=Path, required=True)
    stratified.add_argument("--models", nargs="+", metavar="MODEL")
    stratified.add_argument("--cache-root", type=Path, help="Original prepared development/blind caches")
    stratified.add_argument("--top-k", type=int, default=20)
    stratified.add_argument("--max-events-per-group", type=int, help="Temporal RDST only: cap total sampled blind events; full-blind standard XAI stays unchanged")
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
    if args.command == "explain-regression":
        from .engine.regression_explainer import explain_regression
        for output in explain_regression(args.results, models=args.models, cache_root=args.cache_root,
                depths=args.depths, min_samples_leaf=args.min_samples_leaf, max_events=args.max_events, logger=logger):
            print(output)
        return
    if args.command == "explain-stratified":
        from .engine.stratified_explainer import explain_stratified
        for output in explain_stratified(args.results, models=args.models, cache_root=args.cache_root,
                                         top_k=args.top_k, max_events_per_group=args.max_events_per_group, logger=logger):
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
    if args.command == "xai":
        from .engine.xai_replay import rerun_xai
        for output in rerun_xai(args.results, models=args.models, cache_root=args.cache_root,
                                group_size_samples=args.group_size_samples,
                                max_events=args.max_events, logger=logger):
            print(output)
        return
    batch = load_batch_config(_config_path(args.config), PROJECT_ROOT)
    for output in run_batch(batch, logger=logger):
        print(output)


if __name__ == "__main__":
    main()
