"""Print the U2 key numbers, optionally with one setting changed.

With no options this prints the headline numbers from
``results/u2_metrics.json``. With options it scores both detectors on the
MEArec recording at the baseline and at the changed setting, which is how the
prediction checks in ``docs/U2_design.md`` are answered::

    uv run python scripts/u2_key_number.py --bits 6
    uv run python scripts/u2_key_number.py --fs 10000
    uv run python scripts/u2_key_number.py --noise 20
"""

from __future__ import annotations

import argparse
import json

from neurochain.adc import Adc
from neurochain.analysis.u2 import U2Config, evaluate_setting
from neurochain.data.mearec import load_mearec
from neurochain.paths import results_dir


def print_headline() -> None:
    """Print the committed headline numbers."""
    summary = json.loads((results_dir() / "u2_metrics.json").read_text())
    cfg = summary["config"]
    ref = f"{cfg['reference_noise_uv']:g}"
    print("U2 key numbers (from results/u2_metrics.json)")
    mearec = summary["mearec"]
    if mearec is not None:
        det = mearec["detection"]["noise"]
        print(f"Detection F1 on MEArec at {ref} µV background noise, 30 kS/s, 10 bits:")
        for method in ("threshold", "neo", "offline"):
            row = det[method][ref]
            interval = f"{row['f1_low']:.3f} to {row['f1_high']:.3f}"
            print(
                f"  {method:10s} F1 {row['f1']:.3f}  (95 % interval {interval}; "
                f"precision {row['precision']:.3f}, recall {row['recall']:.3f})"
            )
        fixed = mearec["fixed_point"]
        print(
            f"Fixed point: threshold F1 {fixed['fixed_threshold_f1']:.3f}, NEO F1 "
            f"{fixed['fixed_neo_f1']:.3f}, SBP r {fixed['sbp_fixed_vs_float_r']:.4f} against float"
        )
    nason = summary["nason"]
    if nason is not None:
        online = nason["online"]
        print(
            f"Nason online data, finger velocity R²: SBP {online['velocity_sbp']['r2']:.3f}, "
            f"crossings {online['velocity_crossings']['r2']:.3f}, "
            f"both {online['velocity_both']['r2']:.3f}"
        )
        in_vivo = nason["in_vivo"]
        print(
            f"Nason in vivo, correlation with the largest unit: SBP {in_vivo['sbp_r'][0]:.3f}, "
            f"crossings {in_vivo['crossings_r'][0]:.3f}"
        )


def main() -> None:
    """Print the headline, or the baseline next to a changed setting."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--noise", type=float, help="background noise in µV rms (baseline 13)")
    parser.add_argument("--bits", type=int, help="ADC bits (baseline 10)")
    parser.add_argument("--fs", type=float, help="ADC sample rate in Hz (baseline 30000)")
    parser.add_argument("--range-uv", type=float, help="ADC input range in µV (baseline 1000)")
    parser.add_argument("--threshold-factor", type=float, help="threshold in sigmas (baseline 4.5)")
    parser.add_argument("--neo-factor", type=float, help="NEO threshold factor (baseline 8)")
    args = parser.parse_args()

    config = U2Config()
    changes = {k: v for k, v in vars(args).items() if v is not None}
    if not changes:
        print_headline()
        return
    recording = load_mearec()
    baseline_adc = Adc()
    changed_adc = Adc(
        fs_hz=args.fs or baseline_adc.fs_hz,
        bits=args.bits or baseline_adc.bits,
        input_range_uv=args.range_uv or baseline_adc.input_range_uv,
    )
    reference = config.reference_noise_uv
    before = evaluate_setting(recording, config, noise_uv=reference, adc=baseline_adc)
    after = evaluate_setting(
        recording,
        config,
        noise_uv=args.noise if args.noise is not None else reference,
        adc=changed_adc,
        threshold_factor=args.threshold_factor,
        neo_factor=args.neo_factor,
    )
    changed = ", ".join(f"{k} = {v:g}" for k, v in changes.items())
    print(f"U2 key numbers (MEArec, all channels). Changed: {changed}")
    print(
        f"ADC step: {baseline_adc.lsb_uv:.3g} µV baseline, {changed_adc.lsb_uv:.3g} µV changed; "
        f"raw data rate {baseline_adc.bit_rate_bps / 1e3:g} and "
        f"{changed_adc.bit_rate_bps / 1e3:g} kbit/s"
    )
    print(f"{'':28s}{'baseline':>12s}{'changed':>12s}")
    for method in ("threshold", "neo"):
        for field in ("f1", "precision", "recall"):
            label = f"{method} {field}"
            print(f"{label:28s}{before[method][field]:12.3f}{after[method][field]:12.3f}")


if __name__ == "__main__":
    main()
