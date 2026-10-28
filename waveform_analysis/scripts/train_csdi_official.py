#!/usr/bin/env python3
"""Train official CSDI Transformer on complete wide-waveform pairs.

Uses the original Tashiro et al. CSDI_base and diff_CSDI modules vendored in
waveform_analysis/vendor/csdi (upstream: https://github.com/ermongroup/CSDI).
The original conditional diffusion training objective and sampling are retained.
The *entire* waveform [2, T] is provided; only a fixed-width contiguous interval
is hidden in BOTH detectors. Every remaining sample is observed and can condition
the reconstruction. There are no temporal crops or window-context hyperparameters.
No TSHAP integration.

Caution: original full-sequence attention has quadratic time/memory scaling with
waveform length. Start with --batch-size 1 and consider monitoring GPU memory.
The official CSDI and MLP do not have identical conditioning: CSDI jointly sees
both detectors, whereas the two MLP checkpoints each see one detector.

From repository root:
python -m waveform_analysis.scripts.train_csdi_official --results waveform_analysis/results/UC/test_ridge_48V_R1_R2 --channel energy --max-events 5000 --batch-size 1
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, "") and str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from waveform_analysis.vendor.csdi.main_model import CSDI_base
from waveform_analysis.scripts.train_wide_unet_imputer_detector_specific import (
    MaskedMLP, load_wide_development, root_relative,
)


def mask_starts(length, width, stride):
    result = list(range(0, length - width + 1, stride))
    if result[-1] != length - width:
        result.append(length - width)
    return result


class FullWaveformMasks(Dataset):
    """One full waveform pair per item, with one 1-ns target region per detector."""

    def __init__(self, waves, events, starts, mask_width, random_position):
        self.waves, self.events = waves, np.asarray(events, dtype=np.int64)
        self.starts = starts
        self.mask_width = mask_width
        self.random_position = random_position

    def __len__(self):
        return len(self.events)

    def __getitem__(self, index):
        event = int(self.events[index])
        data = np.array(self.waves[event], dtype=np.float32, copy=True)
        if data.ndim != 2 or data.shape[0] != 2 or not np.isfinite(data).all():
            raise ValueError("Expected finite full waveforms [2, T]")
        position = (np.random.randint(len(self.starts)) if self.random_position
                    else index % len(self.starts))
        start = self.starts[position]
        observed = np.ones_like(data, dtype=np.float32)
        observed[:, start:start+self.mask_width] = 0.
        return torch.from_numpy(data), torch.from_numpy(observed)


class WaveformCSDI(CSDI_base):
    def __init__(self, config, device):
        super().__init__(target_dim=2, config=config, device=device)

    def process_data(self, batch):
        data, cond_mask = batch
        data = data.to(self.device).float()
        cond_mask = cond_mask.to(self.device).float()
        times = torch.arange(data.shape[-1], dtype=torch.float32, device=self.device)
        times = times.unsqueeze(0).expand(data.shape[0], -1)
        observed_mask = torch.ones_like(cond_mask)
        return data, observed_mask, times, cond_mask, cond_mask, torch.zeros(
            len(data), device=self.device, dtype=torch.long
        )

    def forward(self, batch, is_train=1):
        data, observed_mask, times, cond_mask, _, _ = self.process_data(batch)
        side_info = self.get_side_info(times, cond_mask)
        # Upstream CSDI objective; sample a timestep in both training and validation.
        return self.calc_loss(data, cond_mask, observed_mask, side_info, is_train=1)

    @torch.no_grad()
    def draw(self, data, cond_mask, count):
        data, cond_mask = data.to(self.device), cond_mask.to(self.device)
        times = torch.arange(data.shape[-1], dtype=torch.float32, device=self.device)
        times = times.unsqueeze(0).expand(data.shape[0], -1)
        side_info = self.get_side_info(times, cond_mask)
        samples = self.impute(data, cond_mask, side_info, count)
        return samples * (1-cond_mask[:,None]) + data[:,None] * cond_mask[:,None]


def config_from_args(args):
    return dict(
        model=dict(timeemb=128, featureemb=16, is_unconditional=False,
                   target_strategy="random"),
        diffusion=dict(
            layers=args.layers, channels=args.channels, nheads=args.heads,
            diffusion_embedding_dim=128, beta_start=1e-4, beta_end=0.5,
            num_steps=args.diffusion_steps, schedule="quad", is_linear=False,
        )
    )


def evaluate(model, waves, event_ids, positions, width, samples, device, outdir, mlp_models=None, all_starts=None):
    """Evaluate original-resolution masked samples; average generated draws before RMSE."""
    rows = []
    mlp_models = mlp_models or {}
    for label, start in positions:
        sums = np.zeros((2, 4), dtype=np.float64)
        cover = np.zeros((2, 2), dtype=np.float64)
        n = 0
        for event_id in event_ids:
            waveform = np.asarray(waves[int(event_id)], dtype=np.float32)
            target = waveform[:,start:start+width]
            cond = np.ones_like(waveform, dtype=np.float32)
            cond[:,start:start+width] = 0.
            torch_data = torch.from_numpy(np.array(waveform[None], copy=True)).to(device)
            torch_cond = torch.from_numpy(cond[None]).to(device)
            generated = model.draw(torch_data, torch_cond, samples)[0].cpu().numpy()
            block = generated[:,:,start:start+width]
            mean = block.mean(axis=0)
            sums[:,0] += np.square(mean - target).sum(axis=-1)
            sums[:,1] += np.square(block - target[None]).sum(axis=(0,2))/samples
            known = np.ones(len(waveform[0]), dtype=bool)
            known[start:start+width] = False
            for det in range(2):
                if det in mlp_models:
                    context = np.concatenate((waveform[det,:start], waveform[det,start+width:]))
                    code = np.zeros(len(all_starts),dtype=np.float32)
                    code[all_starts.index(start)] = 1.
                    features = np.concatenate((context,code)).astype(np.float32)
                    predicted = mlp_models[det](torch.from_numpy(features[None]).to(device))
                    sums[det,3] += np.square(predicted[0].detach().cpu().numpy()-target[det]).sum()
                interpolation = np.interp(np.arange(start,start+width),
                                          np.flatnonzero(known), waveform[det,known])
                sums[det,2] += np.square(interpolation - target[det]).sum()
            for j, level in enumerate((0.5,0.9)):
                low = np.quantile(block, (1-level)/2, axis=0)
                high = np.quantile(block, (1+level)/2, axis=0)
                cover[:,j] += ((target >= low)&(target <= high)).sum(axis=-1)
            n += width
        for det in range(2):
            rows.append(dict(region=label, detector=det+1, start_sample=int(start),
                             events=len(event_ids), samples=samples,
                             csdi_mean_rmse=float(np.sqrt(sums[det,0]/n)),
                             csdi_draw_rmse=float(np.sqrt(sums[det,1]/n)),
                             interpolation_rmse=float(np.sqrt(sums[det,2]/n)),
                             mlp_rmse=(float(np.sqrt(sums[det,3]/n)) if det in mlp_models else None),
                             coverage_50=float(cover[det,0]/n),
                             coverage_90=float(cover[det,1]/n)))
    with (outdir/"evaluation.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,required=True)
    parser.add_argument("--channel",choices=("energy","timing"),required=True)
    parser.add_argument("--cache-root",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--max-events",type=int,default=5000)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--val-fraction",type=float,default=0.15)
    parser.add_argument("--mask-ns",type=float,default=1.0)
    parser.add_argument("--epochs",type=int,default=30)
    parser.add_argument("--patience",type=int,default=7)
    parser.add_argument("--batch-size",type=int,default=1)
    parser.add_argument("--lr",type=float,default=1e-3)
    parser.add_argument("--weight-decay",type=float,default=1e-4)
    parser.add_argument("--layers",type=int,default=4)
    parser.add_argument("--channels",type=int,default=64)
    parser.add_argument("--heads",type=int,default=8)
    parser.add_argument("--diffusion-steps",type=int,default=50)
    parser.add_argument("--eval-events",type=int,default=12)
    parser.add_argument("--eval-samples",type=int,default=4)
    parser.add_argument("--evaluate-only",action="store_true")
    parser.add_argument("--cpu",action="store_true")
    args=parser.parse_args()
    if args.max_events<10 or args.batch_size<1 or args.epochs<1 or args.diffusion_steps<2:
        parser.error("Invalid training settings")
    if args.channels % args.heads:
        parser.error("--channels must be divisible by --heads")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device=torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    results=root_relative(args.results)
    waves,time_ps,source,manifest=load_wide_development(
        results,args.channel,root_relative(args.cache_root)
    )
    length=waves.shape[-1]
    delta=float(np.median(np.diff(time_ps)))
    width=max(1,round(args.mask_ns*1000/delta))
    if not 0 < width < length:
        parser.error("Require 0 < masked samples < full waveform samples")
    starts=mask_starts(length,width,width)
    selected=np.random.default_rng(args.seed).permutation(len(waves))[:args.max_events]
    size=max(1,round(len(selected)*args.val_fraction))
    val_ids,train_ids=selected[:size],selected[size:]
    outdir=root_relative(args.output) if args.output else (
        ROOT/"waveform_analysis"/"imputation_experiments"/results.name/
        args.channel/"csdi_official_full"
    )
    outdir.mkdir(parents=True,exist_ok=True)
    print(f"Using full-waveform conditioning: 2 detectors x {length} samples; "
          f"{width} samples masked per detector; device={device}", flush=True)
    cfg=config_from_args(args)
    model=WaveformCSDI(cfg,device).to(device)
    best_file=outdir/"best.pt"
    history=[]
    if not args.evaluate_only:
        training=FullWaveformMasks(waves,train_ids,starts,width,True)
        validation=FullWaveformMasks(waves,val_ids,starts,width,False)
        train_loader=DataLoader(training,batch_size=args.batch_size,shuffle=True)
        val_loader=DataLoader(validation,batch_size=args.batch_size)
        opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
        best=float("inf")
        stale=0
        for epoch in range(1,args.epochs+1):
            model.train()
            running=0.
            for batch in train_loader:
                opt.zero_grad(set_to_none=True)
                loss=model(batch)
                loss.backward()
                nn=torch.nn
                nn.utils.clip_grad_norm_(model.parameters(),1.)
                opt.step()
                running+=float(loss.item())*len(batch[0])
            model.eval()
            with torch.no_grad():
                value=sum(float(model(batch).item())*len(batch[0]) for batch in val_loader)/len(validation)
            training_loss=running/len(training)
            history.append((epoch,training_loss,value))
            print(f"epoch {epoch:03d}: train noise MSE={training_loss:.6f}; val noise MSE={value:.6f}",flush=True)
            if value<best-1e-5:
                best=value
                stale=0
                torch.save(dict(state_dict=model.state_dict(),config=cfg,seed=args.seed,
                                selected_events=len(selected),val_fraction=args.val_fraction,
                                length=length,mask_samples=width,input_scope="full_waveform_pair",
                                dataset_identity=manifest.get("analysis_protocol_identity")),
                           best_file)
            else:
                stale+=1
                if stale>=args.patience:
                    break
        with (outdir/"history.csv").open("w",newline="",encoding="utf-8") as f:
            writer=csv.writer(f)
            writer.writerow(["epoch","train_noise_mse","val_noise_mse"])
            writer.writerows(history)
    checkpoint=torch.load(best_file,map_location="cpu",weights_only=True)
    if (checkpoint["dataset_identity"]!=manifest.get("analysis_protocol_identity")
        or checkpoint["length"]!=length or checkpoint["mask_samples"]!=width
        or checkpoint.get("input_scope")!="full_waveform_pair"
        or checkpoint["seed"]!=args.seed
        or checkpoint["selected_events"]!=len(selected)
        or checkpoint["val_fraction"]!=args.val_fraction):
        raise ValueError("Checkpoint preprocessing, full-waveform scope, mask or split settings mismatch")
    if args.evaluate_only:
        model=WaveformCSDI(checkpoint["config"],device).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    # Use the same held-out event selection but only selected regions for fast sampling.
    indices=np.asarray(starts)
    regions=[]
    for label,center in (("leading_edge",0.),("late_20ns",20000.),
                         ("final_window",float(time_ps[-1]))):
        desired=max(0,min(int(np.argmin(abs(time_ps-center)))-width//2,length-width))
        regions.append((label,int(indices[np.argmin(abs(indices-desired))])))
    default_mlp = (
        ROOT/"waveform_analysis"/"imputation_experiments"/results.name/
        args.channel/"detector_specific_mlp"
    )
    mlp_models = {}
    mlp_status = {}
    for det in range(2):
        p = default_mlp/f"detector_{det+1}"/"best.pt"
        if not p.is_file():
            mlp_status[f"detector_{det+1}"] = "checkpoint not found"
            continue
        saved = torch.load(p,map_location="cpu",weights_only=True)
        compatible = (
            saved.get("architecture") == "mlp_relu_128_128_masked_output"
            and saved.get("length") == length
            and saved.get("mask_samples") == width
            and saved.get("mask_positions") == len(starts)
            and saved.get("dataset_identity") == manifest.get("analysis_protocol_identity")
        )
        if not compatible:
            mlp_status[f"detector_{det+1}"] = "incompatible checkpoint; skipped"
            continue
        mlp = MaskedMLP(length-width,width,len(starts)).to(device)
        mlp.load_state_dict(saved["state_dict"])
        mlp.eval()
        mlp_models[det] = mlp
        mlp_status[f"detector_{det+1}"] = "loaded"
    rows=evaluate(model,waves,val_ids[:args.eval_events],regions,width,
                  args.eval_samples,device,outdir,mlp_models,starts)
    summary=dict(architecture="official_CSDI_Transformer",upstream="ermongroup/CSDI",
                 adaptation="full-waveform conditioning with fixed contiguous paired mask",
                 tshap_integration=False,source_run=str(source),channel=args.channel,
                 waveform_samples=length,conditioning_scope="full_waveform_pair",
                 mask_samples=width,
                 train_events=len(train_ids),validation_events=len(val_ids),
                 eval_events=min(args.eval_events,len(val_ids)),
                 eval_samples=args.eval_samples,checkpoint=str(best_file),
                 mlp_baseline_status=mlp_status,metrics=rows)
    (outdir/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(f"Finished; outputs: {outdir}",flush=True)


if __name__=="__main__":
    main()
