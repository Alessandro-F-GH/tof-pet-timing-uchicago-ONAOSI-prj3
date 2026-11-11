"""Analyze native ROOT waveform preamble BEFORE any ML window or LED alignment.

The first native digitizer samples are used, not the LED-aligned/cropped ML
waveform. Keep the original sample-index reference. Compare coherent and
event-variable power, especially at 2.5 GHz. Raw file entry order is NOT a
physical elapsed-time record; equal phases relative to sample zero do not
by themselves demonstrate an independent scope clock.

This is a read-only diagnostic: no preprocessing cache/model inference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from waveform_analysis.data.root_io import iterate_energy_chunks
from waveform_analysis.signal.pulses import decode_oriented
from waveform_analysis.scripts.analyze_waveform_fft import (
    remove_linear_trend, validate_grid)


def native_preamble_metrics(signals, sample_dt_ns, *, band_GHz=(1.5, 3.5)):
    """Compute coherent and variable spectra in original sample coordinates."""
    waves = np.asarray(signals, dtype=np.float64)
    if waves.ndim != 2 or waves.shape[0] < 2 or waves.shape[1] < 32:
        raise ValueError("Need at least 2 original waveform preambles")
    if not np.isfinite(waves).all():
        raise ValueError("Nonfinite native preambles")
    win = np.hanning(waves.shape[1])
    normalized = remove_linear_trend(waves)*win[None,:]/np.sqrt(np.mean(win**2))
    spectra = np.fft.rfft(normalized, axis=1)
    frequencies = np.fft.rfftfreq(waves.shape[1], sample_dt_ns)
    total = np.mean(np.abs(spectra)**2, axis=0)
    meancomplex = spectra.mean(axis=0)
    coherent = np.abs(meancomplex)**2
    variable = np.maximum(total-coherent, 0.)
    consistency = np.clip(coherent/np.maximum(total, 1e-30), 0., 1.)
    band = np.flatnonzero((frequencies >= band_GHz[0])
                          & (frequencies <= band_GHz[1]))
    if not len(band):
        raise ValueError("No bins in requested peak-search band")
    k = band[np.argmax(variable[band])]
    kcoh = band[np.argmax(coherent[band])]
    f0 = 2.5
    # Absolute phase at f0, referenced to native sample zero (NOT LED).
    t = np.arange(waves.shape[1])*sample_dt_ns
    coefficient = np.sum(
        remove_linear_trend(waves)*win[None,:]*
        np.exp(-2j*np.pi*f0*t)[None,:],axis=1) * (2./win.sum())
    return {
        "frequency_GHz": frequencies,
        "mean_individual_power": total,
        "coherent_power": coherent,
        "variable_power": variable,
        "phase_consistency": consistency,
        "mean_waveform_mV": waves.mean(axis=0),
        "individual_waveforms_mV": waves[:10],
        "phase_at_2p5_GHz_rad": np.angle(coefficient),
        "amplitude_at_2p5_GHz_mV": np.abs(coefficient),
        "variable_peak_GHz": float(frequencies[k]),
        "coherent_peak_GHz": float(frequencies[kcoh]),
        "coherence_at_2p5_GHz": float(consistency[np.argmin(abs(frequencies-f0))]),
        "circular_resultant_at_2p5_GHz": float(
            np.abs(np.mean(np.exp(1j*np.angle(coefficient))))),
        "resolution_GHz": float(frequencies[1]-frequencies[0]),
    }


def plots(out, tag, metrics, sample_dt_ns, *, max_freq_GHz=8.):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f = metrics["frequency_GHz"]
    mask = (f > 0)&(f <= max_freq_GHz)
    t = np.arange(len(metrics["mean_waveform_mV"]))*sample_dt_ns
    fig, axes = plt.subplots(3, 1, figsize=(11,10),constrained_layout=True)
    for row in metrics["individual_waveforms_mV"]:
        axes[0].plot(t,row,alpha=.25,lw=.65)
    axes[0].plot(t,metrics["mean_waveform_mV"],lw=2,label="Aligned only to native sample zero")
    axes[0].set(xlabel="Time since first native sample (ns)",ylabel="mV",
                title=f"{tag} | raw ROOT preamble (no LED alignment)")
    axes[0].legend()
    axes[0].grid(alpha=.2)
    for key,label in (
        ("mean_individual_power","Mean individual FFT power"),
        ("coherent_power","FFT power of native-aligned mean"),
        ("variable_power","Variable power (mean subtracted)")):
        axes[1].plot(f[mask],np.maximum(metrics[key][mask],1e-30),label=label)
    axes[1].set(yscale="log",xlabel="Frequency (GHz)",ylabel="Power (a.u.)")
    axes[1].legend()
    axes[1].grid(alpha=.2)
    axes[2].plot(f[mask],metrics["phase_consistency"][mask])
    axes[2].axvline(2.5,ls="--",color=".5",label="2.5 GHz")
    axes[2].set(xlabel="Frequency (GHz)",ylabel="Complex FFT mean-power fraction",
                ylim=(0,1.02))
    axes[2].grid(alpha=.2)
    axes[2].legend()
    fig.savefig(out/f"{tag}.png",dpi=175)
    fig.savefig(out/f"{tag}.pdf")
    plt.close(fig)


def run(args):
    if args.duration_ns <= 0 or args.start_ns < 0 or args.max_events < 2:
        raise ValueError("Invalid preamble bounds or count")
    if args.family not in ("energy","timing"):
        raise ValueError("Unknown family")
    dataset = json.loads(Path(args.dataset_config).read_text(encoding="utf-8"))
    channels = dataset["channels"]
    polarities = (channels["polarities"] if args.family=="energy"
                  else channels["timing_polarities"])
    root = Path(args.root_file or dataset["root_file"]).expanduser().resolve()
    step_size = "32 MB"
    chunks = iterate_energy_chunks(
        root,
        energy_channels_one_based=tuple(channels["energy"]),
        timing_channels_one_based=(
            tuple(channels["timing"]) if args.family=="timing" else None),
        step_size=step_size, entry_stop=None)
    data = {0:[],1:[]}
    native_ids = []
    horizontal_offsets = {0:[],1:[]}
    sample_intervals = {0:[],1:[]}
    rejected = {"too_short":0,"pulse_in_preamble":0,"nonfinite":0,
                "nonmatching_dt":0}
    first_dt = None
    inspected=0
    for chunk in chunks:
        samples = chunk.samples if args.family=="energy" else chunk.timing_samples
        gain = (chunk.vertical_gain_v_per_count if args.family=="energy"
                else chunk.timing_vertical_gain_v_per_count)
        vertical_offset = (chunk.vertical_offset_v if args.family=="energy"
                           else chunk.timing_vertical_offset_v)
        dt = (chunk.horizontal_interval_s if args.family=="energy"
              else chunk.timing_horizontal_interval_s)
        offset = (chunk.horizontal_offset_s if args.family=="energy"
                  else chunk.timing_horizontal_offset_s)
        for row in range(len(chunk.event_id)):
            inspected += 1
            current_dt = np.asarray(dt[row],np.float64)*1e9
            if not np.isfinite(current_dt).all() or np.any(current_dt <= 0):
                rejected["nonmatching_dt"] += 1
                continue
            if first_dt is None:
                first_dt = float(current_dt[0])
            if not np.allclose(current_dt,first_dt,rtol=1e-8,atol=1e-10):
                rejected["nonmatching_dt"] += 1
                continue
            a = int(np.ceil(args.start_ns/first_dt))
            b = int(np.floor((args.start_ns+args.duration_ns)/first_dt))+1
            traces = []
            for d in (0,1):
                native = decode_oriented(samples[d][row],gain[row,d],
                                         vertical_offset[row,d],polarities[d])
                if len(native)<b:
                    traces = []
                    break
                traces.append(native[a:b])
            if not traces:
                rejected["too_short"] += 1
                continue
            if not all(np.isfinite(v).all() for v in traces):
                rejected["nonfinite"] += 1
                continue
            # Reject obvious pulses, not arbitrary baseline oscillation.
            # Compare to the median in this very window; not LED-aligned.
            if args.max_deviation_mV is not None and any(
                np.max(np.abs(v-np.median(v)))>args.max_deviation_mV for v in traces
            ):
                rejected["pulse_in_preamble"] += 1
                continue
            for d in (0,1):
                data[d].append(traces[d])
                horizontal_offsets[d].append(float(offset[row,d]))
                sample_intervals[d].append(float(current_dt[d]))
            native_ids.append(int(chunk.event_id[row]))
            if len(native_ids)>=args.max_events:
                break
        if len(native_ids)>=args.max_events:
            break
    if len(native_ids)<2:
        raise ValueError("Insufficient original preamble traces. Check ROOT, "
                         "start/duration, and pulse-rejection threshold")
    out = Path(args.output_root).expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)
    summary = {
        "source_root":str(root),"family":args.family,
        "n_events":len(native_ids),"inspected":inspected,"rejected":rejected,
        "preamble_start_ns":args.start_ns,"duration_ns":args.duration_ns,
        "dt_ns":first_dt,"reference":"original sample index, not ML window or LED",
        "pulse_cut_mV":args.max_deviation_mV,
        "interpretation":("High coherent power in the preamble indicates "
          "native-record-index-locked structure, not automatically independent "
          "absolute wall-clock time or proven leakage to the ML window."),
        "detectors":{},
    }
    np.save(out/"event_id.npy",np.asarray(native_ids))
    for d in (0,1):
        wave = np.stack(data[d],axis=0)
        result = native_preamble_metrics(wave,first_dt,
                                          band_GHz=args.band_GHz)
        tag = f"{args.family}_detector{d+1}_raw_preamble"
        plots(out,tag,result,first_dt,max_freq_GHz=args.max_frequency_GHz)
        np.savez_compressed(out/f"{tag}.npz",**{
            k:v for k,v in result.items() if not np.isscalar(v)})
        offsets = np.asarray(horizontal_offsets[d])
        summary["detectors"][str(d+1)] = {
          "variable_peak_GHz":result["variable_peak_GHz"],
          "coherent_peak_GHz":result["coherent_peak_GHz"],
          "coherence_at_2p5_GHz":result["coherence_at_2p5_GHz"],
          "circular_resultant_at_2p5_GHz":result["circular_resultant_at_2p5_GHz"],
          "resolution_GHz":result["resolution_GHz"],
          "horizontal_offset_s_min":float(np.min(offsets)),
          "horizontal_offset_s_max":float(np.max(offsets)),
          "horizontal_offset_s_std":float(np.std(offsets)),
        }
        print(f"RAW PREAMBLE | {args.family} | detector={d+1} | "
              f"events={len(wave)} | variable_peak={result['variable_peak_GHz']:.4f} GHz "
              f"| coherent_peak={result['coherent_peak_GHz']:.4f} GHz "
              f"| coherence@2.5={result['coherence_at_2p5_GHz']:.4f} "
              f"| circular_R@2.5={result['circular_resultant_at_2p5_GHz']:.4f}",
              flush=True)
    (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    return summary


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-config",type=Path,required=True)
    p.add_argument("--root-file",type=Path,default=None,
                   help="Override dataset config ROOT file path")
    p.add_argument("--family",choices=("energy","timing"),required=True)
    p.add_argument("--output-root",type=Path,required=True)
    p.add_argument("--max-events",type=int,default=10000)
    p.add_argument("--start-ns",type=float,default=0.,
                   help="Offset from first original native sample, NOT LED")
    p.add_argument("--duration-ns",type=float,default=8.)
    p.add_argument("--max-deviation-mV",type=float,default=None,
                   help="Optional reject if early trace deviates from median by more than threshold")
    p.add_argument("--band-GHz",type=float,nargs=2,default=(1.5,3.5))
    p.add_argument("--max-frequency-GHz",type=float,default=8.)
    return p


if __name__=="__main__":
    run(parser().parse_args())
