"""Print the U1 key numbers, optionally with one parameter changed.

With no options this prints the noise budget three ways (hand calculation,
analytic model, ngspice). With options it prints the baseline next to the
changed front end from the analytic model, which is how the prediction checks
in ``docs/U1_design.md`` are answered::

    uv run python scripts/u1_key_number.py --cdl-scale 2
    uv run python scripts/u1_key_number.py --bias-scale 0.5
    uv run python scripts/u1_key_number.py --bias-scale 0.5 --keep-bandwidth
    uv run python scripts/u1_key_number.py --f-low 0.1
"""

from __future__ import annotations

import argparse
from dataclasses import replace

from neurochain.analysis.u1 import ELECTRODE_KEYS, U1Config, compute_u1, electrodes
from neurochain.frontend import coupling
from neurochain.frontend.emulator import reference_gain
from neurochain.frontend.model import FrontEnd, baseline_frontend


def describe(frontend: FrontEnd, step_v: float) -> list[tuple[str, str]]:
    """Return labelled, formatted numbers for one front end (analytic model)."""
    electrode = frontend.electrode
    if electrode is None:
        raise ValueError("the front end needs an electrode")
    lna = frontend.lna
    temperature_k = frontend.temperature_k
    noise = frontend.band_noise()
    gain = reference_gain(frontend)
    f_low = frontend.f_low_hz
    density_nv = lna.noise_density_v_per_rt_hz(temperature_k) * 1e9
    all_band_uv = frontend.amplifier_total_noise_v() * 1e6
    dc_tolerance_mv = coupling.dc_offset_tolerance_v(gain, lna.output_swing_v) * 1e3
    return [
        ("Electrode |Z| at 1 kHz (kΩ)", f"{electrode.magnitude_ohm(1e3) / 1e3:.1f}"),
        ("Electrode phase at 1 kHz (deg)", f"{electrode.phase_deg(1e3):.1f}"),
        ("Electrode noise, spike band (µV)", f"{noise.electrode_v * 1e6:.3f}"),
        ("Amplifier noise density (nV/√Hz)", f"{density_nv:.2f}"),
        ("Amplifier noise, spike band (µV)", f"{noise.lna_v * 1e6:.3f}"),
        ("Total noise, spike band (µV)", f"{noise.total_v * 1e6:.3f}"),
        ("Gain at 1 kHz", f"{gain:.1f}"),
        ("High-pass corner (Hz)", f"{f_low:.3g}"),
        ("Low-pass corner (Hz)", f"{frontend.f_high_hz:.0f}"),
        ("Amplifier noise, all frequencies (µV)", f"{all_band_uv:.3f}"),
        ("NEF", f"{frontend.nef():.2f}"),
        ("Supply current (µA)", f"{lna.i_total_a * 1e6:.2f}"),
        ("Power (µW)", f"{lna.power_w * 1e6:.2f}"),
        ("Offset a DC-coupled twin survives (mV)", f"{dc_tolerance_mv:.2f}"),
        ("High-pass time constant (s)", f"{coupling.highpass_time_constant_s(f_low):.3f}"),
        (
            f"Railed after a {step_v * 1e3:g} mV step (s)",
            f"{coupling.saturation_time_s(step_v, gain, lna.output_swing_v, f_low):.3f}",
        ),
        (
            "Step decays to the noise floor in (s)",
            f"{coupling.recovery_time_s(step_v, noise.total_v, f_low):.2f}",
        ),
    ]


def print_three_way() -> None:
    """Print the baseline noise budget from hand, analytic model and ngspice."""
    results = compute_u1()
    amp = results.summary["amplifier"]
    print("U1 key numbers: spike-band noise of the baseline front end, referred to the tissue")
    print(
        f"Amplifier: gain {amp['gain_1khz']:.1f} at 1 kHz, {amp['f_low_hz']:.2f} Hz to "
        f"{amp['f_high_hz']:.0f} Hz, {amp['i_total_ua']:g} µA at {amp['v_supply_v']:g} V = "
        f"{amp['power_uw']:g} µW, NEF {amp['nef_measured']:.2f}"
    )
    print(f"{'':34s}{'electrode':>11s}{'amplifier':>11s}{'total':>9s}   (µV rms)")
    for key in ELECTRODE_KEYS:
        entry = results.summary["noise"][key]
        print(f"{results.summary['electrodes'][key]['name']}")
        for method in ("hand", "analytic", "spice"):
            budget = entry[method]
            if budget is None:
                print(f"  {method:32s}{'ngspice not installed':>31s}")
                continue
            print(
                f"  {method:32s}{budget['electrode_uv']:11.3f}{budget['lna_uv']:11.3f}"
                f"{budget['total_uv']:9.3f}"
            )
        print(f"  {'largest disagreement':32s}{entry['max_disagreement']:31.2%}")


def main() -> None:
    """Print the three-way table, or the baseline next to a changed front end."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cdl-scale", type=float, help="multiply the double layer's Q")
    parser.add_argument("--bias-scale", type=float, help="multiply the supply current")
    parser.add_argument(
        "--keep-bandwidth", action="store_true", help="with --bias-scale: resize the load capacitor"
    )
    parser.add_argument("--f-low", type=float, help="high-pass corner in Hz (baseline 1)")
    parser.add_argument("--electrode", choices=ELECTRODE_KEYS, help="swap the electrode model")
    args = parser.parse_args()

    baseline = baseline_frontend()
    changed = baseline
    changes: list[str] = []
    if args.electrode:
        changed = replace(changed, electrode=electrodes(baseline)[args.electrode])
        changes.append(f"electrode = {args.electrode}")
    if args.cdl_scale is not None and changed.electrode is not None:
        changed = replace(
            changed, electrode=changed.electrode.with_interface_scaled(args.cdl_scale)
        )
        changes.append(f"double-layer Q x {args.cdl_scale:g}")
    if args.bias_scale is not None:
        changed = changed.with_bias_scaled(args.bias_scale, keep_bandwidth=args.keep_bandwidth)
        kept = "resized" if args.keep_bandwidth else "unchanged"
        changes.append(f"supply current x {args.bias_scale:g} (load capacitor {kept})")
    if args.f_low is not None:
        changed = replace(changed, lna=changed.lna.with_f_low(args.f_low))
        changes.append(f"high-pass corner = {args.f_low:g} Hz")

    if not changes:
        print_three_way()
        return
    step_v = U1Config().step_v
    print("U1 key numbers (analytic model). Changed: " + "; ".join(changes))
    print(f"{'':42s}{'baseline':>12s}{'changed':>12s}")
    for (label, before), (_, after) in zip(
        describe(baseline, step_v), describe(changed, step_v), strict=True
    ):
        print(f"{label:42s}{before:>12s}{after:>12s}")


if __name__ == "__main__":
    main()
