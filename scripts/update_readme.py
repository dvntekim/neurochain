"""Regenerate the results tables in README.md from ``results/*.json``.

The README never contains a hand-typed result. Each table sits between a pair
of markers and is rebuilt from the metrics file that the unit's own script
wrote. ``--check`` exits non-zero if the README is out of date, which is what
pre-commit and CI run.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from neurochain.paths import repo_root, results_dir


def u0_table(m: dict[str, Any]) -> str:
    """Build the U0 results table from ``results/u0_metrics.json``."""
    ds, noise, spec, spikes, cfg = m["dataset"], m["noise"], m["spectrum"], m["spikes"], m["config"]
    low, high = cfg["spike_band_hz"]
    band = f"{low:g} Hz to {high / 1000:g} kHz"

    def iqr(q: dict[str, float], fmt: str = "{:.2f}") -> str:
        return f"{fmt.format(q['median'])} (IQR {fmt.format(q['p25'])} to {fmt.format(q['p75'])})"

    rows = [
        (
            "Clips analysed",
            f"{ds['n_clean_clips']} of {ds['n_clips']}",
            f"{ds['n_saturated_clips_excluded']} excluded: more than "
            f"{cfg['max_saturation_fraction']:.1%} of samples on an ADC rail",
        ),
        ("Recording time analysed", f"{ds['clean_duration_s']:,.0f} s", f"{ds['fs_hz']:,.0f} Hz"),
        ("Spike-band noise σ", f"{iqr(noise['sigma'])} LSB", f"MAD estimate per clip, {band}"),
        (
            "Noise σ ÷ ideal quantisation noise",
            f"{noise['sigma_over_quantisation_rms']:.1f}×",
            "quantisation noise of a 1 LSB step is 1/√12 LSB rms",
        ),
        (
            f"Share of power below {cfg['lfp_cutoff_hz']:g} Hz",
            iqr(spec["lfp_power_fraction"], "{:.1%}"),
            f"LFP band power ÷ (LFP + spike band power), from {cfg['lfp_floor_hz']:g} Hz",
        ),
        (
            "PSD slope",
            f"f^{spec['psd_slope_lfp_band']:.2f} and f^{spec['psd_slope_spike_band']:.2f}",
            f"10 to {cfg['lfp_cutoff_hz']:g} Hz, and {low:g} Hz to 5 kHz; median spectrum",
        ),
        (
            f"Spikes detected at −{cfg['threshold_sigma']:g}σ",
            f"{spikes['n_spikes']:,}",
            f"{spikes['rate_hz_overall']:.2f} /s pooled; median clip "
            f"{spikes['rate_hz_per_clip']['median']:.2f} /s; "
            f"{spikes['clips_without_spikes']} clips have none",
        ),
        (
            "Detections expected from Gaussian noise alone",
            f"{spikes['gaussian_noise_crossing_rate_hz']:.3f} /s",
            "false-alarm floor of the detector at this threshold",
        ),
        (
            "Median spike trough depth",
            f"{iqr(spikes['amplitude'], '{:.1f}')} LSB",
            f"= {spikes['snr']['median']:.2f} σ; measured on the filtered spike band",
        ),
        (
            "Events whose larger phase is negative",
            f"{spikes['detected_polarity_fraction']:.1%}",
            "two opposite-sign crossings within 1 ms count as one event",
        ),
    ]
    lines = ["| Quantity | Value | How it is defined |", "|---|---|---|"]
    lines += [f"| {name} | {value} | {note} |" for name, value, note in rows]
    return "\n".join(lines)


TABLES = {"U0_RESULTS": ("u0_metrics.json", u0_table)}


def render(readme: str) -> str:
    """Return the README text with every marked table rebuilt."""
    for marker, (filename, build) in TABLES.items():
        begin, end = f"<!-- BEGIN {marker} -->", f"<!-- END {marker} -->"
        if begin not in readme or end not in readme:
            raise SystemExit(f"README.md is missing the {marker} markers")
        metrics = json.loads((results_dir() / filename).read_text())
        head, rest = readme.split(begin, 1)
        _, tail = rest.split(end, 1)
        readme = f"{head}{begin}\n{build(metrics)}\n{end}{tail}"
    return readme


def main() -> int:
    """Rewrite README.md, or with ``--check`` report whether it is current."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="fail if README.md would change")
    args = parser.parse_args()
    path = repo_root() / "README.md"
    current = path.read_text()
    updated = render(current)
    if args.check:
        if updated != current:
            print("README.md is out of date; run `make readme`", file=sys.stderr)
            return 1
        return 0
    path.write_text(updated)
    print(f"updated {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
