"""Plot coherent raw pre-pulse oscillations in fixed acquisition coordinates.

No LED alignment, trigger selection, or phase rotation is applied.
A second figure uses the final batch low-pass filter on complete native traces.
Each trace is ADC-calibrated, interpolated on the common acquisition-time grid,
and centered by its own constant mean on that interval. The figure contains
the mean baseline for each detector, using all ROOT entries by default.
Additional figures show relative phase distributions and frequency-dependent
phase locking on the same unaligned raw baseline interval.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import awkward as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot

from waveform_analysis.data.root_io import iterate_energy_chunks
from waveform_analysis.signal.pulses import decode_oriented
from waveform_analysis.signal.filtering import filter_native_waveform
from waveform_analysis.reporting.plotting import plot_context


def acquisition_grid(root, max_events, preamble_ns, duration_ns, start_ns=None):
    """Choose one interval inside the first preamble_ns of every sampled trace."""
    with uproot.open(root) as f:
        tree = f["events"]
        arrays = tree.arrays(["horizontal_interval_s", "horizontal_offset_s"],
                             entry_stop=max_events, library="ak")
    dt = np.asarray(ak.to_numpy(arrays["horizontal_interval_s"]), float)[:, :4] * 1e9
    t0 = np.asarray(ak.to_numpy(arrays["horizontal_offset_s"]), float)[:, :4] * 1e9
    if not dt.size or not np.all(np.isfinite(dt) & (dt > 0)) or not np.all(np.isfinite(t0)):
        raise ValueError(f"{root}: invalid or empty time metadata")
    step = float(dt[0, 0])
    if not np.allclose(dt, step, rtol=1e-8, atol=0):
        raise ValueError(f"{root}: inconsistent native sampling intervals")
    lower = float(np.max(t0))
    upper = float(np.min(t0 + preamble_ns))
    if start_ns is None:
        # Center the window in the common preamble, on a fixed acquisition grid.
        start_ns = float(np.ceil((lower + (upper - lower - duration_ns) / 2) / step) * step)
    if start_ns < lower - 1e-8 or start_ns + duration_ns > upper + 1e-8:
        raise ValueError(f"{root}: requested interval [{start_ns:.4f}, "
                         f"{start_ns + duration_ns:.4f}] ns is not inside the common "
                         f"preamble [{lower:.4f}, {upper:.4f}] ns; adjust duration or preamble")
    n = int(np.floor(duration_ns / step + 1e-8))
    if n < 4:
        raise ValueError("Requested interval contains too few samples")
    return start_ns + step * np.arange(n), step, (lower, upper), int(dt.shape[0])


def fixed_segment(raw, gain, offset, t0_ns, dt_ns, grid, preprocessing=None):
    """Sample a fixed physical time interval, never shift an event by its pulse."""
    if len(raw) < 2:
        return None
    x = (grid - t0_ns) / dt_ns
    if x[0] < -1e-7 or x[-1] > len(raw) - 1 + 1e-7:
        return None
    lo = max(0, int(np.floor(x[0])))
    hi = min(len(raw), int(np.ceil(x[-1])) + 1)
    if preprocessing is None:
        y = decode_oriented(raw[lo:hi], gain, offset, 1)
    else:
        full = decode_oriented(raw, gain, offset, 1)
        if not np.all(np.isfinite(full)):
            return None
        y = filter_native_waveform(full, dt_ns * 1e-9, preprocessing)[lo:hi]
    if not np.all(np.isfinite(y)):
        return None
    y = np.interp(x, np.arange(lo, hi), y)
    return y - np.mean(y)


def aggregate(root, args, start_ns, preprocessing):
    grid, dt, preamble, expected = acquisition_grid(
        root, args.max_events, args.preamble_ns, args.duration_ns, start_ns)
    # Fit the 2.5 GHz component in each raw segment, with an independent DC term.
    phase = 2 * np.pi * args.frequency_ghz * grid
    basis = np.column_stack((np.cos(phase), np.sin(phase), np.ones(grid.size)))
    projector = np.linalg.pinv(basis)
    sums = np.zeros((2, 4, grid.size))
    coeff_sum = np.zeros((2, 4), complex)
    power_sum = np.zeros((2, 4))
    # Use a Hann-tapered, detrended projection for phase diagnostics,
    # consistent with analyze_raw_preamble_fft. Dense frequencies are
    # evaluations of the same finite-window transform, not independent bins.
    frequencies = np.linspace(1.5, 3.5, 81)
    frequencies = np.unique(np.append(frequencies, args.frequency_ghz))
    target_index = int(np.argmin(abs(frequencies - args.frequency_ghz)))
    t = grid - grid[0]
    taper = np.hanning(grid.size)
    kernel = np.exp(-2j * np.pi * t[:, None] * frequencies[None, :])
    kernel *= taper[:, None] * (2 / taper.sum())
    trend = np.column_stack((np.ones(t.size), t))
    kernel -= trend @ (np.linalg.pinv(trend) @ kernel)
    phase_sum = np.zeros((2, frequencies.size), complex)
    phase_count = np.zeros((2, frequencies.size), dtype=np.int64)
    cross_sum = np.zeros_like(phase_sum)
    pair_power = np.zeros((2, 2, frequencies.size))
    phase_samples = [[], []]
    phase_buffer = []
    def accumulate_phase():
        if not phase_buffer:
            return
        c = np.stack(phase_buffer) @ kernel
        a, b = c[:, (0, 2), :], c[:, (1, 3), :]
        cross = b * a.conj()
        valid = (abs(a) > 1e-12) & (abs(b) > 1e-12)
        unit = np.divide(cross, abs(cross), out=np.zeros_like(cross), where=valid)
        phase_sum[:] += unit.sum(axis=0)
        phase_count[:] += valid.sum(axis=0)
        cross_sum[:] += cross.sum(axis=0)
        pair_power[:, 0] += (abs(a)**2).sum(axis=0)
        pair_power[:, 1] += (abs(b)**2).sum(axis=0)
        for family in range(2):
            phase_samples[family].extend(np.angle(cross[:, family, target_index][
                valid[:, family, target_index]]).tolist())
        phase_buffer.clear()
    n = 0
    for chunk in iterate_energy_chunks(
        root, energy_channels_one_based=(1, 2), timing_channels_one_based=(3, 4),
        step_size=256, entry_stop=args.max_events,
    ):
        for row in range(len(chunk.event_index)):
            traces = [[], []]
            for prefix in ("", "timing_"):
                samples = chunk.timing_samples if prefix else chunk.samples
                gain = getattr(chunk, prefix + "vertical_gain_v_per_count")
                offset = getattr(chunk, prefix + "vertical_offset_v")
                t0 = getattr(chunk, prefix + "horizontal_offset_s")
                interval = getattr(chunk, prefix + "horizontal_interval_s")
                for d in range(2):
                    for k, config in enumerate((None, preprocessing)):
                        trace = fixed_segment(samples[d][row], gain[row, d], offset[row, d],
                                              t0[row, d] * 1e9, interval[row, d] * 1e9, grid, config)
                        traces[k].append(trace)
            if any(trace is None for group in traces for trace in group):
                continue
            traces = np.stack(traces)
            beta = traces @ projector.T
            coeff = beta[..., 0] - 1j * beta[..., 1]
            sums += traces
            coeff_sum += coeff
            power_sum += np.abs(coeff)**2
            phase_buffer.append(traces[0])
            if len(phase_buffer) >= 256:
                accumulate_phase()
            n += 1
    accumulate_phase()
    if n < 2:
        raise ValueError(f"{root}: fewer than two complete raw events")
    means = sums / n
    mean_coeff = coeff_sum / n
    rms_amp = np.sqrt(power_sum / n)
    coherent = np.divide(np.abs(mean_coeff)**2, power_sum / n,
                         out=np.full_like(power_sum, np.nan), where=power_sum > 0)
    summaries = []
    for k in range(2):
        summary = {
        "root_file": str(root), "n_examined": expected, "n_used": n,
        "n_skipped": expected - n, "sample_interval_ns": dt,
        "common_preamble_ns": list(preamble),
        "grid_start_ns": float(grid[0]), "grid_last_sample_ns": float(grid[-1]),
        "frequency_GHz": args.frequency_ghz,
        "channels": {
            str(ch + 1): {
                "mean_waveform_oscillation_amplitude_mV": float(abs(mean_coeff[k, ch])),
                "rms_individual_oscillation_amplitude_mV": float(rms_amp[k, ch]),
                "coherent_power_fraction": float(coherent[k, ch]) if np.isfinite(coherent[k, ch]) else None,
                "mean_waveform_peak_to_peak_mV": float(np.ptp(means[k, ch])),
            } for ch in range(4)
        },
    }
        summary["lowpass"] = preprocessing["lowpass"] if k else {"enabled": False}
        summaries.append(summary)
    locking = np.divide(abs(phase_sum), phase_count,
                        out=np.full(phase_count.shape, np.nan), where=phase_count > 0)
    denominator = pair_power[:, 0] * pair_power[:, 1]
    coherence = np.divide(abs(cross_sum)**2, denominator,
                          out=np.full_like(denominator, np.nan), where=denominator > 0)
    diagnostics = dict(frequencies=frequencies, locking=locking, coherence=coherence,
                       phase_samples=phase_samples, target_index=target_index,
                       phase_count=phase_count)
    return grid, means, summaries, diagnostics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    data_root = Path(__file__).resolve().parents[1] / "processed_data"
    p.add_argument("--uc-root", type=Path, default=data_root / "10-03/48V-R0.root")
    p.add_argument("--fbk-root", type=Path, default=data_root / "10-05/48V-R0.root")
    p.add_argument("--output", type=Path, default=Path("fig/results/instrumental_oscillations.pdf"))
    p.add_argument("--max-events", type=int, default=None,
                   help="Optional entry limit for diagnostic runs; default: all events")
    p.add_argument("--preamble-ns", type=float, default=10.)
    p.add_argument("--duration-ns", type=float, default=4.)
    p.add_argument("--start-ns", type=float, help="Common acquisition-time start for both boards")
    p.add_argument("--uc-start-ns", type=float, help="Override start for UC only")
    p.add_argument("--fbk-start-ns", type=float, help="Override start for FBK only")
    p.add_argument("--frequency-ghz", type=float, default=2.5)
    p.add_argument("--plot-config", type=Path,
                   default=Path(__file__).resolve().parents[1] / "config/plots/default.json")
    args = p.parse_args()
    if (args.max_events is not None and args.max_events < 2) or not (
        0 < args.duration_ns < args.preamble_ns and args.frequency_ghz > 0
    ):
        p.error("Require >=2 events, positive frequency and 0 < duration < preamble")
    style = json.loads(args.plot_config.read_text(encoding="utf-8"))
    results = {}
    for board, root, specific in (("UC", args.uc_root, args.uc_start_ns),
                                  ("FBK", args.fbk_root, args.fbk_start_ns)):
        start = args.start_ns if specific is None else specific
        config_path = Path(__file__).resolve().parents[1] / "config/preprocessing" / f"{board}_lowpass_2GHz.json"
        preprocessing = json.loads(config_path.read_text(encoding="utf-8"))
        results[board] = aggregate(root, args, start, preprocessing)
        print(board, json.dumps(results[board][2], indent=2), flush=True)
    colors = style["preprocessing"]["detector_colors"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for k, destination in enumerate((args.output, args.output.with_name(
            args.output.stem + "_filtered" + args.output.suffix))):
        with plot_context(style):
            fig, axes = plt.subplots(2, 2, sharex="col", figsize=(7.4, 4.8),
                                     constrained_layout=True)
            for col, board in enumerate(("FBK", "UC")):
                grid, means, summaries, _ = results[board]
                for row, family in enumerate(("Energy", "Timing")):
                    ax = axes[row, col]
                    for d in range(2):
                        ch = 2 * row + d
                        ax.plot(grid, means[k, ch], color=colors[d], lw=1.5,
                                label=f"Detector {d + 1}")
                    # Identical amplitude limits in corresponding raw/filtered panels.
                    values = means[:, 2 * row:2 * row + 2, :]
                    lo, hi = float(values.min()), float(values.max())
                    pad = max(0.05 * (hi - lo), 1e-9)
                    ax.set_ylim(lo - pad, hi + pad)
                    ax.set_title(f"{board} — {family}")
                    if row == 1:
                        ax.set_xlabel("Acquisition time [ns]")
                    if col == 0:
                        ax.set_ylabel("Amplitude [mV]")
                    ax.margins(x=0)
                    ax.grid(alpha=.2)
                    ax.legend(loc="best")
            fig.savefig(destination, bbox_inches="tight")
            plt.close(fig)
        summary = {
            "reference": "ROOT horizontal_offset_s + sample_index * horizontal_interval_s",
            "preprocessing": ("ADC calibration, full native waveform low-pass, fixed-grid linear interpolation, constant mean removal"
                              if k else "ADC calibration, fixed-grid linear interpolation, constant mean removal"),
            "boards": {board: data[2][k] for board, data in results.items()},
        }
        destination.with_suffix(".json").write_text(json.dumps(summary, indent=2, allow_nan=False),
                                                    encoding="utf-8")
        np.savez_compressed(destination.with_suffix(".npz"), **{
            key: value for board, (grid, means, _, _) in results.items()
            for key, value in ((f"{board}_time_ns", grid), (f"{board}_mean_mV", means[k]))
        })
        print(f"Saved {destination}", flush=True)

    # Relative phase diagnostics are computed on the unfiltered traces.
    phase_summary = {}
    phase_arrays = {}
    with plot_context(style):
        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True,
                                 figsize=(7.4, 4.8), constrained_layout=True)
        for col, board in enumerate(("FBK", "UC")):
            diag = results[board][3]
            idx = diag["target_index"]
            phase_summary[board] = {}
            phase_arrays[f"{board}_frequency_GHz"] = diag["frequencies"]
            phase_arrays[f"{board}_PLV"] = diag["locking"]
            phase_arrays[f"{board}_complex_coherence"] = diag["coherence"]
            for row, family in enumerate(("Energy", "Timing")):
                ax = axes[row, col]
                angles = np.asarray(diag["phase_samples"][row])
                degrees = np.degrees(angles)
                phase_arrays[f"{board}_{family}_phase_deg"] = degrees
                R = diag["locking"][row, idx]
                coh = diag["coherence"][row, idx]
                mean = np.degrees(np.angle(np.mean(np.exp(1j * angles)))) if angles.size else np.nan
                phase_summary[board][family] = {
                    "n": int(angles.size), "R": float(R) if np.isfinite(R) else None,
                    "mean_delta_deg": float(mean) if np.isfinite(mean) else None,
                    "complex_coherence": float(coh) if np.isfinite(coh) else None,
                }
                bins = np.linspace(-180, 180, 37)
                if degrees.size:
                    ax.hist(degrees, bins=bins, weights=np.ones(degrees.size)/degrees.size,
                            histtype="step", color=colors[0], linewidth=1.5)
                ax.axhline(1/36, color="0.5", linestyle="--", linewidth=1)
                ax.text(.03, .95, f"R = {R:.3f}", transform=ax.transAxes,
                        va="top", ha="left")
                ax.set_title(f"{board} — {family}")
                ax.set_xlim(-180, 180)
                ax.set_xticks([-180, -90, 0, 90, 180])
                if row == 1:
                    ax.set_xlabel(r"Relative phase [deg]")
                if col == 0:
                    ax.set_ylabel("Fraction of events")
        phase_path = args.output.with_name(args.output.stem + "_relative_phase" + args.output.suffix)
        fig.savefig(phase_path, bbox_inches="tight")
        plt.close(fig)
        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True,
                                 figsize=(7.4, 4.8), constrained_layout=True)
        for col, board in enumerate(("FBK", "UC")):
            diag = results[board][3]
            for row, family in enumerate(("Energy", "Timing")):
                ax = axes[row, col]
                ax.plot(diag["frequencies"], diag["locking"][row], color=colors[0], label="PLV")
                ax.plot(diag["frequencies"], diag["coherence"][row], color=colors[1],
                        linestyle="--", label="Complex coherence (squared)")
                ax.axvline(args.frequency_ghz, color="0.5", linestyle=":", linewidth=1)
                ax.set_title(f"{board} — {family}")
                ax.set_ylim(0, 1)
                if row == 1:
                    ax.set_xlabel("Frequency [GHz]")
                if col == 0:
                    ax.set_ylabel("Coherence")
                ax.legend(loc="best")
        scan_path = args.output.with_name(args.output.stem + "_phase_coherence" + args.output.suffix)
        fig.savefig(scan_path, bbox_inches="tight")
        plt.close(fig)
    phase_path.with_suffix(".json").write_text(json.dumps({
        "frequency_GHz": args.frequency_ghz,
        "duration_ns": args.duration_ns,
        "phase_convention": "detector 2 minus detector 1",
        "method": "linear detrending, Hann taper, complex Fourier projection on fixed acquisition-time grid",
        "boards": phase_summary,
    }, indent=2, allow_nan=False), encoding="utf-8")
    np.savez_compressed(phase_path.with_suffix(".npz"), **phase_arrays)
    print(f"Saved {phase_path} and {scan_path}", flush=True)


if __name__ == "__main__":
    main()
