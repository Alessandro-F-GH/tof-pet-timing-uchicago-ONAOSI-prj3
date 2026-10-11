"""Save one raw coincidence example as separate energy/timing two-panel figures.

Run from any directory, for example::

    python report/plot_raw_waveforms.py --root-file acquisition.root --board FBK --event 0

``--event`` is a zero-based ROOT entry, not an event ID. Each figure contains
both detectors from that same entry, using native acquisition times in ns.
Only ADC-to-mV calibration and the specified vertical-scale clipping are applied:
no selection, baseline subtraction, polarity reversal, alignment or filtering.
Trigger thresholds are intentionally unused. Defaults map energy to channels 1/2
and timing to channels 3/4. Output style/format follows the repository plot config.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPORT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPORT.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from waveform_analysis.data.root_io import EnergyChunk, iterate_energy_chunks
from waveform_analysis.reporting.plotting import plot_context, output_path, _finish, _save
from waveform_analysis.signal.pulses import decode_oriented

VERTICAL_LIMITS_MV = {
    "energy": ((-50.0, 600.0), (-50.0, 600.0)),
    "timing": ((-20.0, 140.0), (-20.0, 140.0)),
}


def read_entry(
    source: Path, entry: int, energy_channels: tuple[int, int],
    timing_channels: tuple[int, int],
) -> tuple[EnergyChunk, int]:
    """Locate a raw entry with the existing bounded-memory ROOT loader."""
    offset = 0
    for chunk in iterate_energy_chunks(
        source, energy_channels_one_based=energy_channels,
        timing_channels_one_based=timing_channels,
        step_size=1024, entry_stop=entry + 1,
    ):
        if entry < offset + len(chunk.event_index):
            return chunk, entry - offset
        offset += len(chunk.event_index)
    raise ValueError(f"ROOT entry {entry} does not exist in {source}")


def save_pair(
    chunk: EnergyChunk, row: int, family: str, board: str,
    destination: Path, style: dict,
) -> Path:
    """Plot the complete clipped native waveform pair with a shared time axis."""
    prefix = "timing_" if family == "timing" else ""
    samples = chunk.timing_samples if prefix else chunk.samples
    gain = getattr(chunk, prefix + "vertical_gain_v_per_count")
    offset = getattr(chunk, prefix + "vertical_offset_v")
    interval = getattr(chunk, prefix + "horizontal_interval_s")
    start = getattr(chunk, prefix + "horizontal_offset_s")
    if any(value is None for value in (samples, gain, offset, interval, start)):
        raise ValueError(f"Raw {family} waveforms are unavailable")
    with plot_context(style):
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6.4, 4.8))
        try:
            colors = (style["formulations"]["shared"]["color"],
                      style["formulations"]["direct"]["color"])
            for detector, ax in enumerate(axes):
                voltage = decode_oriented(
                    samples[detector][row], gain[row, detector], offset[row, detector], 1,
                )
                dt, t0 = float(interval[row, detector]), float(start[row, detector])
                if not np.isfinite(dt) or dt <= 0 or not np.isfinite(t0):
                    raise ValueError("Native time metadata must be finite with positive interval")
                time_ns = (t0 + np.arange(voltage.size) * dt) * 1e9
                limits = VERTICAL_LIMITS_MV[family][detector]
                ax.plot(time_ns, np.clip(voltage, *limits), color=colors[detector],
                        label=f"Detector {detector + 1}")
                ax.set_ylim(*limits)
                ax.set_ylabel("Amplitude [mV]")
                ax.legend()
                _finish(ax, style)
                ax.margins(x=0)
            axes[0].set_title(f"{board}: raw {family} waveforms")
            axes[-1].set_xlabel("Acquisition time [ns]")
            return _save(fig, output_path(
                destination, f"{board}_raw_{family}_event_{int(chunk.event_index[row])}", style,
            ))
        finally:
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-file", required=True, type=Path)
    parser.add_argument("--board", required=True, choices=("UC", "FBK"))
    parser.add_argument("--event", type=int, default=0, help="Zero-based ROOT entry")
    parser.add_argument("--energy-channels", nargs=2, type=int, default=(1, 2))
    parser.add_argument("--timing-channels", nargs=2, type=int, default=(3, 4))
    parser.add_argument("--plot-config", type=Path,
                        default=REPORT.parent / "waveform_analysis/config/plots/default.json")
    parser.add_argument("--output-dir", type=Path, default=REPORT / "figures")
    args = parser.parse_args()
    if args.event < 0 or min(*args.energy_channels, *args.timing_channels) < 1:
        parser.error("Event must be nonnegative and channel numbers must be positive")
    style = json.loads(args.plot_config.read_text(encoding="utf-8"))
    chunk, row = read_entry(args.root_file, args.event,
                            tuple(args.energy_channels), tuple(args.timing_channels))
    for family in ("energy", "timing"):
        print(save_pair(chunk, row, family, args.board, args.output_dir, style))


if __name__ == "__main__":
    main()
