"""Plot original ROOT waveforms for tail events indexed by main-branch batch outputs.

Saved blind predictions supply event IDs and residuals ONLY.
Native panels use ROOT. ML-input panels reuse Codex PreparedCacheLookup and\nArtifactStore, exactly as the heatmap workflow does. No refitting.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import awkward as ak
import uproot
from types import SimpleNamespace

from waveform_analysis.data.results import resolve_run_config
from waveform_analysis.data.prepared_cache import PreparedCacheLookup
from waveform_analysis.data.results import read_run
from waveform_analysis.data.storage import RunStore
from waveform_analysis.signal.pulses import decode_oriented


def _predictions(run: Path, artifact_root: Path):
    store = RunStore(run, artifact_root=artifact_root)
    path = store.predictions_path
    # Legacy batch files are still supported as in the Codex scripts.
    if not path.is_file() and (run / "artifacts" / "pred.npz").is_file():
        path = run / "artifacts" / "pred.npz"
    with np.load(path, allow_pickle=False) as data:
        ids = np.asarray(data["event_id"], dtype=np.int64)
        led = np.asarray(data["led_residual_ps"], dtype=np.float64)
        corrected = np.asarray(data["corrected_ps"], dtype=np.float64)
    if len(set(ids.tolist())) != len(ids) or ids.shape != led.shape or ids.shape != corrected.shape:
        raise ValueError(f"Inconsistent or duplicate prediction event IDs: {path}")
    return {int(event): (float(led[i]), float(corrected[i]))
            for i, event in enumerate(ids)}


def _resolve_root(config, override):
    root = Path(override).expanduser() if override else Path(config["blind"]["root_file"]).expanduser()
    if not root.is_file():
        raise FileNotFoundError(
            f"Acquisition ROOT file not found: {root}. Supply --root-file with its local Windows path."
        )
    return root.resolve()


def _channels(chunk, index, family, config):
    if family == "energy":
        samples = chunk.samples
        gain, offset = chunk.vertical_gain_v_per_count, chunk.vertical_offset_v
        dt, origin = chunk.horizontal_interval_s, chunk.horizontal_offset_s
        signs = config.get("polarities", [1, 1])
    else:
        samples = chunk.timing_samples
        gain, offset = chunk.timing_vertical_gain_v_per_count, chunk.timing_vertical_offset_v
        dt, origin = chunk.timing_horizontal_interval_s, chunk.timing_horizontal_offset_s
        signs = config.get("timing_polarities", [1, 1])
    if samples is None:
        raise ValueError("Timing waveform branches are absent from the acquisition")
    waves = tuple(
        decode_oriented(samples[d][index], gain[index, d], offset[index, d], int(signs[d]))
        for d in range(2)
    )
    return {
        "waves": waves,
        "dt_s": np.asarray(dt[index], dtype=float),
        "origin_s": np.asarray(origin[index], dtype=float),
    }


def _cached_input(cache_root, run, config, family, requested_ids, predictions):
    """Load Codex named blind prepared artifact; never rebuild from ROOT."""
    root = Path(cache_root).expanduser().resolve()
    if not (root / "artifact_store").is_dir():
        raise FileNotFoundError(f"Codex artifact store absent: {root / 'artifact_store'}")
    config["preprocessing"]["cache_dir"] = str(root / "preprocessing")
    config["preprocessing"]["artifact_root"] = str(root / "artifact_store")
    config["preprocessing"]["named_cache_root"] = str(root)
    manifest = read_run(run)["manifest"]
    data = PreparedCacheLookup().find("blind", config, manifest)
    index = {int(event): i for i, event in enumerate(data.event_index)}
    if len(index) != data.n_events or not requested_ids.issubset(index):
        raise ValueError(f"Named blind {family} cache does not contain all selected event IDs")
    target = getattr(data, f"{family}_target_ps")
    if target is None or not all(
        np.isclose(float(target[index[event]]), predictions[event][0], atol=0.1, rtol=0)
        for event in requested_ids
    ):
        raise ValueError(f"Named blind {family} cache does not match saved LED residuals")

    time = np.asarray(getattr(data, f"{family}_time_ps"), dtype=float) / 1000.0
    mask = (np.ones(len(time), dtype=bool) if family == "energy"
            else np.asarray(data.sample_mask, dtype=bool))
    if mask.shape != time.shape or not np.any(mask):
        raise ValueError("Invalid frozen sample mask")
    waves = getattr(data, f"{family}_windows")
    transform = getattr(data, f"{family}_transform")
    if waves is None or transform is None:
        raise ValueError(f"Missing prepared {family} waveform or normalization")
    output = {}
    for event in requested_ids:
        normalized = np.asarray(waves[index[event]], dtype=np.float32)
        output[event] = {
            "waves": np.asarray(transform.inverse(normalized)[:, mask], dtype=float),
            "time_ns": time[mask],
        }
    return output, str(data.directory)


def _plot(path, record, reference, view_ns):
    fig, axes = plt.subplots(4, 2, figsize=(13, 13), constrained_layout=True)
    for fi, family in enumerate(("energy", "timing")):
        for detector in range(2):
            for is_ml in (False, True):
                ax = axes[fi * 2 + int(is_ml), detector]
                for item, label in ((reference, "typical"), (record, "outlier")):
                    if is_ml:
                        channel = item["ml_inputs"][family]
                        wave = channel["waves"][detector]
                        t_ns = channel["time_ns"]
                    else:
                        channel = item["waveforms"][family]
                        wave = np.asarray(channel["waves"][detector], dtype=float)
                        t_ns = (float(channel["origin_s"][detector])
                                + np.arange(len(wave)) * float(channel["dt_s"][detector])) * 1e9
                    mask = np.abs(t_ns) <= view_ns if is_ml else np.ones(len(t_ns), bool)
                    ax.plot(t_ns[mask], wave[mask], lw=1.1, alpha=0.8,
                            label=f"{label} {item['event_index']}")
                ax.set_title(f"{family.capitalize()} | detector {detector + 1} | "
                             + ("cached ML input window" if is_ml else "native ROOT acquisition"))
                ax.set_xlabel("Time relative to LED [ns]" if is_ml
                              else "Native acquisition time [ns]")
                ax.set_ylabel("Amplitude [mV]")
                ax.grid(alpha=0.2)
                ax.legend(fontsize=8)
    er = record["residuals"]["energy"]
    tr = record["residuals"]["timing"]
    fig.suptitle(
        f"Event {record['event_index']} | ML-corrected energy={er[1]:+.1f} ps, "
        f"timing={tr[1]:+.1f} ps | LED energy={er[0]:+.1f} ps, "
        f"timing={tr[0]:+.1f} ps"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--energy-run", type=Path, required=True)
    p.add_argument("--timing-run", type=Path, required=True)
    p.add_argument("--root-file", type=Path,
                   help="Windows path to the original blind ROOT acquisition")
    p.add_argument("--cache-root", type=Path, required=True,
                   help="Codex portable cache bundle containing artifact_store and named datasets")
    p.add_argument("--control-artifact", type=Path,
                   help="Deprecated and ignored: read batch pred.npz instead")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--tail-channel", choices=("energy", "timing"), default="energy")
    p.add_argument("--residual", choices=("corrected", "led"), default="corrected",
                   help="Choose tail events by ML-corrected or LED residual")
    p.add_argument("--tail-fraction", type=float, default=0.002)
    p.add_argument("--max-events", type=int, default=12)
    p.add_argument("--view-ns", type=float, default=15)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not 0 < args.tail_fraction < 1 or args.max_events < 1 or args.view_ns <= 0:
        p.error("Require 0 < tail-fraction < 1, max-events >= 1, view-ns > 0")

    ec = resolve_run_config(args.energy_run)
    tc = resolve_run_config(args.timing_run)
    if ec["mode"] != "energy_to_energy" or tc["mode"] != "timing_to_timing":
        raise ValueError("Expected energy_to_energy and timing_to_timing runs")
    if ec["blind"]["channels"] != tc["blind"]["channels"]:
        raise ValueError("Energy/timing runs have different channel mapping")
    if (str(ec["blind"]["root_file"]).replace("\\", "/").split("/")[-1]
            != str(tc["blind"]["root_file"]).replace("\\", "/").split("/")[-1]):
        raise ValueError("Energy/timing runs refer to different ROOT acquisitions")
    root = _resolve_root(ec, args.root_file)
    channels = ec["blind"]["channels"]
    artifact_root = args.cache_root.expanduser().resolve() / "artifact_store"
    ep = _predictions(args.energy_run, artifact_root)
    tp = _predictions(args.timing_run, artifact_root)
    common = sorted(ep.keys() & tp.keys())
    if not common:
        raise ValueError("No common event IDs in the two saved blind predictions")
    residual_index = 1 if args.residual == "corrected" else 0
    scored = np.asarray(
        [abs((ep if args.tail_channel == "energy" else tp)[event][residual_index])
         for event in common], dtype=float
    )
    finite = np.isfinite(scored)
    common = np.asarray(common, dtype=np.int64)[finite]
    scored = scored[finite]
    if not len(common):
        raise ValueError("No finite paired residuals")
    ordered = np.argsort(-scored, kind="stable")
    tail_count = max(1, int(np.ceil(len(common) * args.tail_fraction)))
    chosen_ids = common[ordered[:min(tail_count, args.max_events)]]
    typical_ids = common[scored <= np.median(scored)]
    reference_id = int(np.random.default_rng(args.seed).choice(typical_ids))
    selected_ids = {int(event) for event in chosen_ids} | {reference_id}

    records = {
        event: {"event_index": event, "residuals": {"energy": ep[event], "timing": tp[event]}}
        for event in selected_ids
    }
    # Access individual ROOT entries directly. In many datasets event_index equals
    # the entry number, but we verify that before relying on it.
    with uproot.open(root) as root_file:
        tree = root_file["events"]
        ordered_ids = sorted(selected_ids)
        entry_lookup = {}
        for event in ordered_ids:
            if 0 <= event < tree.num_entries:
                value = tree["event_index"].array(
                    entry_start=event, entry_stop=event + 1, library="np"
                )
                if len(value) == 1 and int(value[0]) == event:
                    entry_lookup[event] = event

        unresolved = selected_ids - entry_lookup.keys()
        if unresolved:
            # Mapping fallback reads ONLY the small scalar event_index branch;
            # waveform branches are never scanned.
            all_ids = np.asarray(tree["event_index"].array(library="np"), dtype=np.int64)
            for entry, event_id in enumerate(all_ids):
                event_id = int(event_id)
                if event_id in unresolved:
                    if event_id in entry_lookup:
                        raise ValueError(f"Duplicate ROOT event_index: {event_id}")
                    entry_lookup[event_id] = entry
            missing = selected_ids - entry_lookup.keys()
            if missing:
                raise ValueError(f"Selected event_index absent from ROOT: {sorted(missing)[:10]}")

        energy_indices = [int(x) - 1 for x in channels["energy"]]
        timing_indices = [int(x) - 1 for x in channels["timing"]]
        required = list(dict.fromkeys([
            "event_index", "vertical_gain_v_per_count", "vertical_offset_v",
            "horizontal_interval_s", "horizontal_offset_s",
            *(f"samples_ch{x}" for x in channels["energy"]),
            *(f"samples_ch{x}" for x in channels["timing"]),
        ]))
        for event in ordered_ids:
            entry = entry_lookup[event]
            # One entry's original samples plus ADC calibration metadata.
            arrays = tree.arrays(
                required, entry_start=entry, entry_stop=entry + 1, library="ak"
            )
            if int(ak.to_numpy(arrays["event_index"])[0]) != event:
                raise ValueError(f"ROOT entry mismatch for event_index {event}")
            gain = np.asarray(ak.to_numpy(arrays["vertical_gain_v_per_count"]), dtype=float)
            offset = np.asarray(ak.to_numpy(arrays["vertical_offset_v"]), dtype=float)
            interval = np.asarray(ak.to_numpy(arrays["horizontal_interval_s"]), dtype=float)
            origin = np.asarray(ak.to_numpy(arrays["horizontal_offset_s"]), dtype=float)
            chunk = SimpleNamespace(
                samples=tuple(arrays[f"samples_ch{x}"] for x in channels["energy"]),
                timing_samples=tuple(arrays[f"samples_ch{x}"] for x in channels["timing"]),
                vertical_gain_v_per_count=gain[:, energy_indices],
                vertical_offset_v=offset[:, energy_indices],
                horizontal_interval_s=interval[:, energy_indices],
                horizontal_offset_s=origin[:, energy_indices],
                timing_vertical_gain_v_per_count=gain[:, timing_indices],
                timing_vertical_offset_v=offset[:, timing_indices],
                timing_horizontal_interval_s=interval[:, timing_indices],
                timing_horizontal_offset_s=origin[:, timing_indices],
            )
            records[event]["waveforms"] = {
                family: _channels(chunk, 0, family, channels)
                for family in ("energy", "timing")
            }

    cache_sources = {}
    for family, run, cfg, predictions in (
        ("energy", args.energy_run, ec, ep),
        ("timing", args.timing_run, tc, tp),
    ):
        inputs, cache_path = _cached_input(
            args.cache_root, run, cfg, family, selected_ids, predictions
        )
        cache_sources[family] = cache_path
        for event, waveform in inputs.items():
            records[event].setdefault("ml_inputs", {})[family] = waveform

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    ref = records[reference_id]
    with (output / "selected_events.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "event_index", "energy_led_ps", "energy_corrected_ps",
            "timing_led_ps", "timing_corrected_ps", "reference_event_index", "plot"
        ])
        writer.writeheader()
        for event in chosen_ids:
            record = records[int(event)]
            name = f"event_{int(event)}.png"
            _plot(output / name, record, ref, args.view_ns)
            writer.writerow({
                "event_index": int(event),
                "energy_led_ps": record["residuals"]["energy"][0],
                "energy_corrected_ps": record["residuals"]["energy"][1],
                "timing_led_ps": record["residuals"]["timing"][0],
                "timing_corrected_ps": record["residuals"]["timing"][1],
                "reference_event_index": reference_id,
                "plot": name,
            })
    (output / "summary.json").write_text(json.dumps({
        "root": str(root),
        "waveform_source": "original ROOT ADC counts calibrated to oriented mV",
        "ml_input_source": cache_sources,
        "ml_input_display": "saved normalized input inverted to mV, after saved model sample mask",
        "event_selection_source": "saved main-branch blind artifacts/pred.npz",
        "selection_residual": args.residual,
        "tail_channel": args.tail_channel,
        "tail_fraction": args.tail_fraction,
        "paired_events": len(common),
        "tail_events": tail_count,
        "plotted_events": len(chosen_ids),
        "reference_event_index": reference_id,
    }, indent=2), encoding="utf-8")
    print(f"Saved {len(chosen_ids)} original ROOT waveform plots to {output}")


if __name__ == "__main__":
    main()
