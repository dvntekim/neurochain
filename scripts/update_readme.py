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


def u1_table(m: dict[str, Any]) -> str:
    """Build the U1 results table from ``results/u1_metrics.json``."""
    amp, cpl, emu = m["amplifier"], m["coupling"], m["emulator"]
    base = m["electrodes"]["in_saline"]
    noise, ideal = m["noise"]["in_saline"], m["noise"]["ideal_capacitor"]
    low, high = m["noise_band_hz"]
    band = f"{low:g} Hz to {high / 1000:g} kHz"
    fast, slow = m["settling"]["runs"]

    def total(method: str) -> str:
        budget = noise[method]
        return "not run" if budget is None else f"{budget['total_uv']:.3f}"

    spice_note = (
        f"ngspice impedance within {base['spice_max_relative_error']:.2%} of the analytic model"
        if base["spice_max_relative_error"] is not None
        else "ngspice not run"
    )
    rows = [
        (
            "Electrode impedance at 1 kHz",
            f"{base['z_1khz_ohm'] / 1e3:.0f} kΩ at −{abs(base['phase_1khz_deg']):.0f}°",
            f"platinum Utah electrode in saline (Straka et al., 2018); {spice_note}",
        ),
        (
            "Front-end noise: hand / analytic / ngspice",
            f"{total('hand')} / {total('analytic')} / {total('spice')} µV rms",
            f"referred to the tissue, {band}, at {amp['temperature_k'] - 273.15:.0f} °C",
        ),
        (
            "Largest disagreement between methods",
            f"{max(e['max_disagreement'] for e in m['noise'].values()):.2%}",
            "across three electrode models; the done-criterion is 10 %",
        ),
        (
            "Electrode / amplifier share",
            f"{noise['analytic']['electrode_uv']:.2f} / {noise['analytic']['lna_uv']:.2f} µV rms",
            "analytic model; the two add in quadrature",
        ),
        (
            "Same electrode with an ideal-capacitor double layer",
            f"{ideal['analytic']['electrode_uv']:.2f} µV electrode, "
            f"{ideal['analytic']['total_uv']:.2f} µV total",
            "the textbook Randles circuit, matched to the same impedance magnitude at 1 kHz",
        ),
        (
            "Amplifier",
            f"gain {amp['gain_1khz']:.1f}, {amp['f_low_hz']:.2f} Hz to "
            f"{amp['f_high_hz'] / 1e3:.2f} kHz",
            f"{amp['i_total_ua']:g} µA at {amp['v_supply_v']:g} V = {amp['power_uw']:g} µW; "
            f"NEF {amp['nef_measured']:.2f}",
        ),
        (
            "Electrode offset tolerated",
            f"AC: blocked; DC at the same gain: {cpl['dc_offset_tolerance_same_gain_mv']:.1f} mV",
            f"surviving {cpl['worst_offset_mv']:g} mV DC-coupled needs gain "
            f"{cpl['dc_gain_for_worst_offset']:.2g} and {cpl['adc_bits_dc']:.1f} ADC bits, "
            f"against {cpl['adc_bits_ac']:.1f}",
        ),
        (
            f"Time on the rail after a {m['settling']['step_mv']:g} mV offset step",
            f"{fast['saturation_time_simulated_s']:.2f} s at {fast['f_low_hz']:g} Hz; "
            f"{slow['saturation_time_simulated_s']:.2f} s at {slow['f_low_hz']:g} Hz",
            f"emulated; the formula gives {fast['saturation_time_analytic_s']:.3f} s and "
            f"{slow['saturation_time_analytic_s']:.3f} s",
        ),
        (
            "Emulator noise against the model",
            f"{emu['band_output_rms_measured_uv']:.1f} against "
            f"{emu['band_output_rms_predicted_uv']:.1f} µV rms",
            f"at the amplifier output, {band}, {emu['seconds']:g} s of emulated silence",
        ),
    ]
    nason = m["nason"]
    if nason is None:
        rows.append(("Nason in-vivo channel", "not run", "needs the Nason data (`make data`)"))
    else:
        rows += [
            (
                "Noise σ of a real in-vivo channel",
                f"{nason['sigma_recorded_uv']:.2f} µV",
                f"{nason['recorded_over_model']:.1f} times the model's "
                f"{nason['sigma_model_floor_uv']:.2f} µV through the same filter "
                "(Nason et al., one channel, 10 s)",
            ),
            (
                "That channel after the emulated front end",
                f"{nason['sigma_emulated_uv']:.2f} µV "
                f"(predicted {nason['sigma_predicted_uv']:.2f})",
                f"up from {nason['sigma_band_limited_uv']:.2f} µV with the emulator's noise off",
            ),
            (
                "The same at a quarter of the bias current",
                f"{nason['sigma_quarter_bias_uv']:.2f} µV",
                f"amplifier power {nason['quarter_bias_power_uw']:g} µW instead of "
                f"{amp['power_uw']:g} µW",
            ),
        ]
    lines = ["| Quantity | Value | How it is defined |", "|---|---|---|"]
    lines += [f"| {name} | {value} | {note} |" for name, value, note in rows]
    return "\n".join(lines)


TABLES = {
    "U0_RESULTS": ("u0_metrics.json", u0_table),
    "U1_RESULTS": ("u1_metrics.json", u1_table),
}


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
