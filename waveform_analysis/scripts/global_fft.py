"""Global full-window FFT spectra from original prepared waveforms.

Read-only analysis: mean event-wise power spectra for detector 1, detector 2,
and their event-wise difference. No model inference, fitting or FFT of averages.

Example:
    python -m waveform_analysis.scripts.global_fft --results PATH --models direct_rdst
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from waveform_analysis.core.io import atomic_json, canonical_hash
from waveform_analysis.data.results import discover_runs, resolve_run_config
from waveform_analysis.data.view import waveform_view, model_sample_mask
from waveform_analysis.engine.xai_replay import PreparedCacheLookup, _cache_root
from waveform_analysis.signal.sample_mask import apply_sample_mask, apply_sample_mask_to_time

CHANNELS = ("detector_1", "detector_2", "difference")


def validate_grid(time_ps):
    time_ps = np.asarray(time_ps, dtype=np.float64)
    if time_ps.ndim != 1 or len(time_ps) < 4 or not np.all(np.isfinite(time_ps)):
        raise ValueError("FFT needs at least four finite time samples")
    steps = np.diff(time_ps)
    if np.any(steps <= 0) or not np.allclose(steps, steps[0], rtol=1e-5, atol=1e-6):
        raise ValueError("Full-window FFT requires a uniformly spaced time grid after masking")
    return time_ps, float(np.mean(steps))


def average_power(dataset, mode, sample_mask, time_ps, *, chunk_size=512):
    """Compute mean event-wise Hann-windowed periodograms in amplitude²/Hz.

    Time is in ps; power is a one-sided density per Hz. Each event is
    mean-centered separately, so its DC/pulse offset is not retained.
    """
    time_ps, dt_ps = validate_grid(time_ps)
    if int(chunk_size) != chunk_size or chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    fs = 1e12 / dt_ps
    size = len(time_ps)
    window = np.hanning(size)
    norm = fs * np.sum(window ** 2)
    frequencies = np.fft.rfftfreq(size, d=dt_ps * 1e-12)
    total = np.zeros((3, len(frequencies)), dtype=np.float64)
    count = 0
    for start in range(0, dataset.n_events, int(chunk_size)):
        idx = np.arange(start, min(start + chunk_size, dataset.n_events), dtype=np.int64)
        pair = waveform_view(dataset, mode, idx).materialize()
        pair = np.asarray(apply_sample_mask(pair, sample_mask), dtype=np.float64)
        if pair.shape != (len(idx), 2, size) or not np.all(np.isfinite(pair)):
            raise ValueError("Invalid or nonfinite prepared signal pairs")
        difference = pair[:, 0, :] - pair[:, 1, :]
        signals = np.concatenate((pair, difference[:, None, :]), axis=1)
        signals = (signals - signals.mean(axis=-1, keepdims=True)) * window
        spectra = np.fft.rfft(signals, axis=-1)
        powers = np.abs(spectra) ** 2 / norm
        if size % 2 == 0:
            powers[..., 1:-1] *= 2
        else:
            powers[..., 1:] *= 2
        total += powers.sum(axis=0)
        count += len(idx)
    if count < 1:
        raise ValueError("Prepared event population is empty")
    return frequencies, total / count, count


def plot_power(frequency_hz, power, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, values in zip(("Detector 1", "Detector 2", "Detector 1 - Detector 2"), power):
        ax.semilogy(frequency_hz[1:] * 1e-9,
                    np.maximum(values[1:], np.finfo(float).tiny), label=name)
    ax.set(xlabel="Frequency [GHz]", ylabel="Mean one-sided PSD [normalized amplitude²/Hz]",
           title="Full-window global waveform spectra (event-averaged power)")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def analyze_run(run, *, role="blind", cache_override=None, chunk_size=512, logger=None):
    if role not in {"blind", "development"}:
        raise ValueError("role must be blind or development")
    directory, manifest = Path(run["directory"]), run["manifest"]
    config = resolve_run_config(directory)
    cache_root = _cache_root(config, directory, cache_override, roles=(role,))
    dataset = PreparedCacheLookup().find(cache_root, role, config, manifest)
    mode = config["mode"]
    mask = model_sample_mask(dataset, mode)
    view = waveform_view(dataset, mode, [0])
    time_ps = apply_sample_mask_to_time(view.time_ps, mask)
    frequencies, power, n_events = average_power(
        dataset, mode, mask, time_ps, chunk_size=chunk_size
    )
    identity = {
        "method": "full_window_event_mean_psd_v1",
        "model": manifest["model"], "role": role,
        "protocol": dataset.manifest["analysis_protocol_identity"],
        "mode": mode, "chunk_size": int(chunk_size),
    }
    destination = directory / "explanations" / "global_fft" / canonical_hash(identity)[:16]
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "spectra.npz",
                        frequency_hz=frequencies, time_ps=time_ps,
                        detector_1_psd=power[0], detector_2_psd=power[1],
                        difference_psd=power[2])
    plot_power(frequencies, power, destination / "spectra.png")
    atomic_json(destination / "summary.json", {
        **identity, "n_events": n_events, "sample_period_ps": float(np.mean(np.diff(time_ps))),
        "frequency_resolution_hz": float(frequencies[1] - frequencies[0]),
        "nyquist_hz": float(frequencies[-1]),
        "preprocessing": "per-event mean subtraction; Hann window; one-sided PSD",
        "aggregation": "mean of individual event power spectra, not FFT of mean waveform",
    })
    if logger:
        logger.info("Global FFT | %s %s | N=%d | %s",
                    manifest["model"], role, n_events, destination)
    return destination


def run(results, *, models=None, role="blind", cache_root=None, chunk_size=512, logger=None):
    runs = discover_runs(Path(results).expanduser().resolve())
    if not runs:
        raise ValueError(f"No completed model runs found in {results}")
    available = {r["manifest"]["model"] for r in runs}
    if models is not None and set(models) - available:
        raise ValueError(f"Models not in results: {sorted(set(models) - available)}")
    outputs, failures = [], []
    for item in runs:
        if models is not None and item["manifest"]["model"] not in models:
            continue
        try:
            outputs.append(analyze_run(item, role=role, cache_override=cache_root,
                                       chunk_size=chunk_size, logger=logger))
        except Exception as exc:
            failures.append(f"{item['directory']}: {exc}")
            if logger:
                logger.exception("Global FFT failed for %s", item["directory"])
    if failures:
        raise RuntimeError("Some FFT analyses failed:\n" + "\n".join(failures))
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--role", choices=("blind", "development"), default="blind")
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--chunk-size", type=int, default=512)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    for path in run(args.results, models=args.models, role=args.role,
                    cache_root=args.cache_root, chunk_size=args.chunk_size,
                    logger=logging.getLogger(__name__)):
        print(path)


if __name__ == "__main__":
    main()
