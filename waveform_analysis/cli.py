from __future__ import annotations
import argparse,json,logging
from pathlib import Path
from .ml_pipeline.config import load_config,load_batch_config,public_config
from .ml_pipeline.study import run_study
from .ml_pipeline.batch import run_batch
from .ml_pipeline.postprocess import remake_study_plots,remake_batch_plots
PROJECT_ROOT=Path(__file__).resolve().parent
def _config_path(path):
    p=Path(path).expanduser()
    if p.is_file():return p
    q=PROJECT_ROOT/p
    return q if q.is_file() else p
def _parser():
    p=argparse.ArgumentParser(prog="python -m waveform_analysis.cli",description="TOF-PET fixed-control preprocessing + repeated-holdout ML studies")
    sub=p.add_subparsers(dest="command",required=True)
    for name in ("check","run","remake-plots"):
        c=sub.add_parser(name);c.add_argument("--config",type=Path,required=True)
    run=sub.choices["run"];m=run.add_mutually_exclusive_group();m.add_argument("--overwrite",action="store_true");m.add_argument("--resume",action="store_true");run.add_argument("--rebuild-preprocessing",action="store_true")
    batch=sub.add_parser("batch");batch.add_argument("--config",type=Path,required=True);bm=batch.add_mutually_exclusive_group();bm.add_argument("--overwrite",action="store_true");bm.add_argument("--resume",action="store_true");batch.add_argument("--rebuild-preprocessing",action="store_true")
    remake_batch=sub.add_parser("remake-batch-plots");remake_batch.add_argument("--config",type=Path,required=True)
    return p
def main():
    args=_parser().parse_args()
    if args.command=="check":
        cfg=load_config(_config_path(args.config),PROJECT_ROOT);print(json.dumps(public_config(cfg),indent=2));return
    logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(message)s");logger=logging.getLogger("waveform-batch")
    if args.command=="remake-plots":
        cfg=load_config(_config_path(args.config),PROJECT_ROOT);remake_study_plots(cfg,logger=logger);print(Path(cfg["output_dir"]).resolve());return
    if args.command=="remake-batch-plots":
        cfgs=load_batch_config(_config_path(args.config),PROJECT_ROOT)
        for run_dir,_ in remake_batch_plots(cfgs,logger=logger):print(run_dir)
        return
    if args.command=="batch":
        cfgs=load_batch_config(_config_path(args.config),PROJECT_ROOT)
        for out in run_batch(cfgs,overwrite=args.overwrite,resume=args.resume,rebuild_preprocessing=args.rebuild_preprocessing,logger=logger):print(out)
        return
    cfg=load_config(_config_path(args.config),PROJECT_ROOT);print(run_study(cfg,overwrite=args.overwrite,resume=args.resume,rebuild_preprocessing=args.rebuild_preprocessing))
if __name__=="__main__":main()
