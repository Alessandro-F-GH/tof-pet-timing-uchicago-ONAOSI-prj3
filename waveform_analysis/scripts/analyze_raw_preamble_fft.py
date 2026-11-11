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


def phase_coefficients(waves, dt_ns, frequency_GHz, t0_ns):
    """Complex oscillation coefficient referenced to ROOT absolute time."""
    w = np.asarray(waves, np.float64)
    n = w.shape[1]
    taper = np.hanning(n)
    t = np.arange(n) * dt_ns
    coefficient = (2. / taper.sum()) * np.sum(
        remove_linear_trend(w) * taper[None, :] *
        np.exp(-2j*np.pi*frequency_GHz*t)[None, :], axis=1)
    # Channel-specific horizontal offsets are essential for absolute phase.
    return coefficient * np.exp(-2j*np.pi*frequency_GHz*np.asarray(t0_ns))


def compare_phases(c1, c2):
    """Circular phase locking value and amplitude-weighted coherence."""
    good = (np.isfinite(c1) & np.isfinite(c2)
            & (np.abs(c1) > 1e-12) & (np.abs(c2) > 1e-12))
    a, b = c1[good], c2[good]
    if not len(a):
        return {"n": 0, "R": None, "mean_delta_deg": None,
                "complex_coherence": None}
    unit = b*np.conj(a)/np.abs(b*np.conj(a))
    mean = np.mean(unit)
    coherence = abs(np.mean(a*np.conj(b)))**2 / (
        np.mean(abs(a)**2)*np.mean(abs(b)**2))
    return {"n": int(len(a)), "R": float(abs(mean)),
            "mean_delta_deg": float(np.degrees(np.angle(mean))),
            "complex_coherence": float(coherence)}


def run(args):
    channels = ([1, 2] if args.family == "energy" else
                [3, 4] if args.family == "timing" else [1, 2, 3, 4])
    root = Path(args.root_file).expanduser().resolve()
    out = Path(args.output_root).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    raw = {channel: [] for channel in channels}
    native_offsets = {ch: [] for ch in channels}
    dt_ns = None
    with uproot.open(root) as file:
        tree = file["events"]
        cols = ["vertical_gain_v_per_count", "vertical_offset_v",
                "horizontal_interval_s", "horizontal_offset_s"] + [f"samples_ch{c}" for c in channels]
        for chunk in tree.iterate(filter_name=cols, step_size="32 MB", library="ak"):
            gain = ak.to_numpy(chunk["vertical_gain_v_per_count"])
            offset = ak.to_numpy(chunk["vertical_offset_v"])
            interval = ak.to_numpy(chunk["horizontal_interval_s"])
            time_offset = ak.to_numpy(chunk["horizontal_offset_s"])
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
                    native_offsets[ch].append(float(time_offset[i, ch-1])*1e9)
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
               "detectors": {},
               "phase_frequency_GHz": args.phase_frequency_GHz,
               "phase_comparisons": {}}
    coefficients = {}
    for detector, ch in enumerate(channels, 1):
        result = spectra(np.stack(raw[ch]), dt_ns)
        coefficients[ch] = phase_coefficients(
            np.stack(raw[ch]), dt_ns, args.phase_frequency_GHz,
            np.asarray(native_offsets[ch]))
        f = result["frequency_GHz"]
        band = (f >= 1.5) & (f <= 3.5)
        highest = np.flatnonzero(band)[np.argmax(result["total_power"][band])]
        coherent_peak = np.flatnonzero(band)[np.argmax(result["coherent_power"][band])]
        result["time_ns"] = np.arange(len(result["mean_waveform_mV"]))*dt_ns
        tag = f"ch{ch}_raw_first10ns"
        plot(out, tag, result, dt_ns, args.max_frequency_GHz)
        np.savez_compressed(out/f"{tag}.npz", **result)
        summary["detectors"][str(ch)] = {
            "total_peak_GHz": float(f[highest]),
            "coherent_peak_GHz": float(f[coherent_peak]),
            "coherent_fraction_at_total_peak": float(result["coherent_fraction"][highest]),
        }
        print(f"RAW FIRST 10 NS | {args.family} | ch={ch} "
              f"| events={len(raw[ch])} | peak={f[highest]:.3f} GHz "
              f"| coherent_peak={f[coherent_peak]:.3f} GHz "
              f"| coherent_fraction={result['coherent_fraction'][highest]:.4f}",
              flush=True)
    pairs = [(1, 2)] if args.family == "energy" else (
        [(3, 4)] if args.family == "timing" else
        [(1, 2), (3, 4), (1, 3), (2, 4), (1, 4), (2, 3)])
    print(f"PHASE COMPARISON | ROOT absolute time | f={args.phase_frequency_GHz:.4f} GHz")
    for a, b in pairs:
        stats = compare_phases(coefficients[a], coefficients[b])
        summary["phase_comparisons"][f"ch{a}_ch{b}"] = stats
        print(f"PHASE ch{a}-ch{b} | n={stats['n']} "
              f"| R={stats['R']:.4f} "
              f"| mean_delta_deg={stats['mean_delta_deg']:.2f} "
              f"| complex_coherence={stats['complex_coherence']:.4f}",
              flush=True)
    (out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root-file", type=Path, required=True)
    p.add_argument("--family", choices=("energy", "timing", "all"), required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--duration-ns", type=float, default=10.)
    p.add_argument("--max-events", type=int, default=10000)
    p.add_argument("--max-frequency-GHz", type=float, default=8.)
    p.add_argument("--phase-frequency-GHz", type=float, default=2.5)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
