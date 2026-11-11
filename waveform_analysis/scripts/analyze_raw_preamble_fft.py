"""FFT of the first 10 ns of original ROOT signals, BEFORE LED alignment.

Uses native sample index 0 as the time reference. Reads energy ch1/ch2 or
timing ch3/ch4 (configurable), calibrated to mV. Does not read ML caches,
change waveforms, run a model, or select events by their LED behavior.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import uproot
import awkward as ak

from waveform_analysis.signal.pulses import decode_oriented
from waveform_analysis.scripts.analyze_waveform_fft import remove_linear_trend


def spectra(waves, dt_ns):
    """Spectral power in fixed acquisition coordinates (sample zero)."""
    w = np.asarray(waves, np.float64)
    taper = np.hanning(w.shape[1])
    fft = np.fft.rfft(remove_linear_trend(w) * taper[None, :], axis=1)
    freq = np.fft.rfftfreq(w.shape[1], d=dt_ns)
    total = np.mean(np.abs(fft)**2, axis=0)
    coherent = np.abs(np.mean(fft, axis=0))**2
    variable = np.maximum(total - coherent, 0.)
    return {
        "frequency_GHz": freq,
        "total_power": total,
        "coherent_power": coherent,
        "variable_power": variable,
        "coherent_fraction": coherent / np.maximum(total, 1e-30),
        "mean_waveform_mV": w.mean(axis=0),
        "example_waveforms_mV": w[:10],
    }


def plot(path, tag, data, dt_ns, max_frequency_GHz):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    freq = data["frequency_GHz"]
    keep = (freq > 0) & (freq <= max_frequency_GHz)
    t = np.arange(len(data["mean_waveform_mV"])) * dt_ns
    fig, axs = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True)
    for x in data["example_waveforms_mV"]:
        axs[0].plot(t, x, alpha=.22, lw=.6)
    axs[0].plot(t, data["mean_waveform_mV"], lw=2, label="Native sample-aligned mean")
    axs[0].set(xlabel="Time from first recorded sample (ns)", ylabel="Amplitude (mV)",
               title=tag)
    axs[0].legend()
    for key, name in (("total_power", "Mean individual power"),
                      ("coherent_power", "Mean waveform power"),
                      ("variable_power", "Mean-subtracted power")):
        axs[1].plot(freq[keep], np.maximum(data[key][keep], 1e-30), label=name)
    axs[1].set(xlabel="Frequency (GHz)", ylabel="FFT power (a.u.)", yscale="log")
    axs[1].axvline(2.5, ls="--", color=".5")
    axs[1].legend()
    axs[2].plot(freq[keep], data["coherent_fraction"][keep])
    axs[2].set(xlabel="Frequency (GHz)", ylabel="Coherent power fraction",
               ylim=(0, 1.02))
    axs[2].axvline(2.5, ls="--", color=".5")
    for a in axs:
        a.grid(alpha=.2)
    fig.savefig(path/f"{tag}.png", dpi=175)
    fig.savefig(path/f"{tag}.pdf")
    plt.close(fig)


def run(args):
    channels = ([1, 2] if args.family == "energy" else [3, 4])
    root = Path(args.root_file).expanduser().resolve()
    out = Path(args.output_root).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    raw = {channel: [] for channel in channels}
    dt_ns = None
    with uproot.open(root) as file:
        tree = file["events"]
        cols = ["vertical_gain_v_per_count", "vertical_offset_v",
                "horizontal_interval_s"] + [f"samples_ch{c}" for c in channels]
        for chunk in tree.iterate(filter_name=cols, step_size="32 MB", library="ak"):
            gain = ak.to_numpy(chunk["vertical_gain_v_per_count"])
            offset = ak.to_numpy(chunk["vertical_offset_v"])
            interval = ak.to_numpy(chunk["horizontal_interval_s"])
            for i in range(len(gain)):
                # This file normally uses 80 GS/s for both channels.
                sample_step = float(interval[i, channels[0]-1])*1e9
                if dt_ns is None:
                    dt_ns = sample_step
                # Keep the same native sample count for all events.
                n = int(np.floor(args.duration_ns/dt_ns + 1e-9))
                usable = True
                traces = {}
                for ch in channels:
                    index = ch-1
                    samples = chunk[f"samples_ch{ch}"][i]
                    if (len(samples) < n or
                            not np.isclose(float(interval[i,index])*1e9,
                                           dt_ns, rtol=1e-8)):
                        usable = False
                        break
                    traces[ch] = decode_oriented(
                        samples[:n], gain[i,index], offset[i,index], 1)
                if not usable:
                    continue
                for ch in channels:
                    raw[ch].append(traces[ch])
                if len(raw[channels[0]]) >= args.max_events:
                    break
            if len(raw[channels[0]]) >= args.max_events:
                break
    if len(raw[channels[0]]) < 2:
        raise ValueError("Not enough ROOT waveforms with the requested initial duration")
    summary = {"root_file": str(root), "family": args.family,
               "duration_ns": args.duration_ns,
               "reference": "first original digitizer sample, no LED alignment",
               "n_events": len(raw[channels[0]]), "dt_ns": dt_ns,
               "detectors": {}}
    for detector, ch in enumerate(channels, 1):
        result = spectra(np.stack(raw[ch]), dt_ns)
        f = result["frequency_GHz"]
        band = (f >= 1.5) & (f <= 3.5)
        highest = np.flatnonzero(band)[np.argmax(result["total_power"][band])]
        coherent_peak = np.flatnonzero(band)[np.argmax(result["coherent_power"][band])]
        result["time_ns"] = np.arange(len(result["mean_waveform_mV"]))*dt_ns
        tag = f"{args.family}_detector{detector}_raw_first10ns"
        plot(out, tag, result, dt_ns, args.max_frequency_GHz)
        np.savez_compressed(out/f"{tag}.npz", **result)
        summary["detectors"][str(detector)] = {
            "total_peak_GHz": float(f[highest]),
            "coherent_peak_GHz": float(f[coherent_peak]),
            "coherent_fraction_at_total_peak": float(result["coherent_fraction"][highest]),
        }
        print(f"RAW FIRST 10 NS | {args.family} | detector={detector} "
              f"| events={len(raw[ch])} | peak={f[highest]:.3f} GHz "
              f"| coherent_peak={f[coherent_peak]:.3f} GHz "
              f"| coherent_fraction={result['coherent_fraction'][highest]:.4f}",
              flush=True)
    (out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root-file", type=Path, required=True)
    p.add_argument("--family", choices=("energy", "timing"), required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--duration-ns", type=float, default=10.)
    p.add_argument("--max-events", type=int, default=10000)
    p.add_argument("--max-frequency-GHz", type=float, default=8.)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
