from __future__ import annotations
import argparse,json,logging
from pathlib import Path
from .ml_pipeline.batch import run_batch
from .ml_pipeline.config import load_batch_config,public_batch_config
from .ml_pipeline.postprocess import remake_plots
PROJECT_ROOT=Path(__file__).resolve().parent
def _config_path(path):
    path=Path(path).expanduser()
    if path.is_file():return path
    candidate=PROJECT_ROOT/path
    return candidate if candidate.is_file() else path
def _parser():
    parser=argparse.ArgumentParser(prog="python -m waveform_analysis.cli",description="TOF-PET control → development CV → blind evaluation pipeline");sub=parser.add_subparsers(dest="command",required=True);check=sub.add_parser("check-batch",help="Resolve and validate a batch configuration");check.add_argument("--config",type=Path,required=True);batch=sub.add_parser("batch",help="Run or resume a dependency-aware batch");batch.add_argument("--config",type=Path,required=True);plots=sub.add_parser("plots",help="Regenerate plots/reports from a result root only");plots.add_argument("--results",type=Path,required=True);report=sub.add_parser("report",help="Regenerate plots and report without retraining");report.add_argument("results",type=Path);return parser
def main():
    args=_parser().parse_args()
    if args.command=="check-batch":
        batch=load_batch_config(_config_path(args.config),PROJECT_ROOT);print(json.dumps(public_batch_config(batch),indent=2));return
    logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(message)s");logger=logging.getLogger("waveform-pipeline")
    if args.command in {"plots","report"}:print(remake_plots(Path(args.results),logger=logger));return
    batch=load_batch_config(_config_path(args.config),PROJECT_ROOT)
    for output in run_batch(batch,logger=logger):print(output)
if __name__=="__main__":main()
