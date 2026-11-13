"""Print the U0 key numbers, optionally with one parameter changed.

With no options this prints the numbers reported in the README. With options it
re-runs the analysis and prints the baseline next to the changed configuration,
which is how the prediction checks in ``docs/U0_design.md`` are answered::

    uv run python scripts/u0_key_number.py --threshold-sigma 3.5
    uv run python scripts/u0_key_number.py --spike-band 300 1000
    uv run python scripts/u0_key_number.py --drop-bits 3
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Any

from neurochain.analysis.u0 import U0Config, analyse_paths, summarise
from neurochain.data.neuralink import list_clips, load_clip
from neurochain.paths import cache_dir

ROWS: tuple[tuple[str, str, str], ...] = (
    ("Spike-band noise σ (LSB)", "noise.sigma.median", "{:.2f}"),
    ("Median spike trough depth (LSB)", "spikes.amplitude.median", "{:.1f}"),
    ("Median spike depth ÷ noise σ", "spikes.snr.median", "{:.2f}"),
    ("Spikes detected", "spikes.n_spikes", "{:,}"),
    ("Detection rate, all clips pooled (/s)", "spikes.rate_hz_overall", "{:.2f}"),
    ("Detection rate, median clip (/s)", "spikes.rate_hz_per_clip.median", "{:.2f}"),
    ("Rate expected from Gaussian noise (/s)", "spikes.gaussian_noise_crossing_rate_hz", "{:.3f}"),
    ("Clips with no spike", "spikes.clips_without_spikes", "{:,}"),
    ("LFP share of power, median clip", "spectrum.lfp_power_fraction.median", "{:.1%}"),
)


def lookup(summary: dict[str, Any], dotted: str) -> Any:
    """Read a nested value such as ``noise.sigma.median`` from a summary."""
    value: Any = summary
    for key in dotted.split("."):
        value = value[key]
    return value


def main() -> None:
    """Run the analysis for the baseline and, if requested, a changed configuration."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--threshold-sigma", type=float, help="spike threshold in σ (default 4.5)")
    parser.add_argument(
        "--spike-band", type=float, nargs=2, metavar=("LOW_HZ", "HIGH_HZ"), help="default 300 7500"
    )
    parser.add_argument("--drop-bits", type=int, help="ADC bits to discard (default 0)")
    parser.add_argument("--order", type=int, help="Butterworth order (default 4)")
    parser.add_argument("--causal", action="store_true", help="causal filter, not zero-phase")
    args = parser.parse_args()

    baseline = U0Config()
    changes: dict[str, Any] = {}
    if args.threshold_sigma is not None:
        changes["threshold_sigma"] = args.threshold_sigma
    if args.spike_band is not None:
        changes["spike_band_hz"] = (args.spike_band[0], args.spike_band[1])
    if args.drop_bits is not None:
        changes["drop_bits"] = args.drop_bits
    if args.order is not None:
        changes["filter_order"] = args.order
    if args.causal:
        changes["zero_phase"] = False
    changed = replace(baseline, **changes)

    clips = list_clips()
    configs = [baseline] if changed == baseline else [baseline, changed]
    summaries = [
        summarise(analyse_paths(clips, load_clip, c, cache_dir=cache_dir(), progress_every=0))
        for c in configs
    ]

    dataset = summaries[0]["dataset"]
    print(
        f"U0 key numbers: Neuralink Compression Challenge, {dataset['n_clean_clips']} clean clips "
        f"of {dataset['n_clips']} ({dataset['clean_duration_s']:.0f} s), units ADC LSB"
    )
    if changes:
        print("Changed: " + ", ".join(f"{k} = {v}" for k, v in changes.items()))
    header = f"{'':40s}{'baseline':>12s}" + (f"{'changed':>12s}" if changes else "")
    print(header)
    for label, key, fmt in ROWS:
        cells = "".join(f"{fmt.format(lookup(s, key)):>12s}" for s in summaries)
        print(f"{label:40s}{cells}")


if __name__ == "__main__":
    main()
