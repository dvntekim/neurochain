"""U2 figures: detection F1, ADC sweeps, SBP against crossings, fixed point.

Every number drawn comes from the summary dictionary written to
``results/u2_metrics.json``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter

from neurochain import viz

METHOD_LABELS = {
    "threshold": "Threshold, causal",
    "neo": "Nonlinear energy, causal",
    "offline": "Threshold, zero-phase (offline only)",
}
FEATURE_LABELS = {
    "crossings": "Threshold crossings, 30 kS/s",
    "sbp_digital": "SBP, digital filter then 2 kS/s",
    "sbp_implant": "SBP, analog band then 2 kS/s ADC",
}


def _plain(value: float, _: object) -> str:
    return f"{value:g}"


def _mearec_provenance(summary: dict[str, Any]) -> dict[str, str]:
    rec = summary["mearec"]["recording"]
    cfg = summary["config"]
    return {
        "dataset": (
            f"MEArec simulation, {rec['n_channels']} channels, {rec['n_units']} units, "
            f"{rec['duration_s']:g} s at {rec['fs_hz'] / 1e3:g} kHz, through the U1 front end and an ADC."
        ),
        "split": (
            f"thresholds set on the first {cfg['calibration_s']:g} s of each channel, scored on the "
            f"remaining {rec['scored_s']:g} s ({rec['target_spikes']:,} target spikes)."
        ),
    }


def _series(
    ax: Axes, block: dict[str, dict[str, float]], color: str, label: str, *, key: str = "f1"
) -> None:
    """Draw one sweep with its bootstrap interval."""
    xs = np.array(sorted(float(k) for k in block))
    rows = [block[f"{x:g}"] for x in xs]
    low_key, high_key = (f"{key}_low", f"{key}_high") if key == "f1" else ("low", "high")
    ax.fill_between(xs, [r[low_key] for r in rows], [r[high_key] for r in rows], color=color,
                    alpha=0.15, linewidth=0)  # fmt: skip
    ax.plot(xs, [r[key] for r in rows], color=color, linewidth=2.0, marker="o", markersize=5,
            markeredgecolor=viz.SURFACE, markeredgewidth=1.0, label=label)  # fmt: skip


def plot_detection(summary: dict[str, Any], path: Path) -> Path:
    """Draw F1 against background noise, and the detectors' operating curves."""
    cfg = summary["config"]
    detection = summary["mearec"]["detection"]
    reference = f"{cfg['reference_noise_uv']:g}"
    at_ref = {m: detection["noise"][m][reference] for m in detection["noise"]}
    with viz.figure_style():
        fig, (ax, ax_pr) = plt.subplots(
            1, 2, figsize=(10.0, 6.6),
            gridspec_kw={"left": 0.085, "right": 0.975, "top": 0.835, "bottom": 0.215,
                         "wspace": 0.24, "width_ratios": [1.3, 1.0]},
        )  # fmt: skip
        for method, color in zip(("threshold", "neo", "offline"), viz.SERIES, strict=True):
            _series(ax, detection["noise"][method], color, METHOD_LABELS[method])
        ax.axvline(cfg["reference_noise_uv"], color=viz.INK_SECONDARY, linewidth=0.9,
                   linestyle=(0, (4, 3)))  # fmt: skip
        ax.text(cfg["reference_noise_uv"] + 0.6, 0.97, "in-vivo level\n(Nason channel)",
                fontsize=8.5, color=viz.INK_SECONDARY, va="top")  # fmt: skip
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("Neural background noise in the spike band (µV rms)")
        ax.set_ylabel("Detection F1")
        ax.legend(loc="lower left", fontsize=8.5)
        viz.panel_label(ax, "F1 against background noise")

        for method, color in zip(("threshold", "neo"), viz.SERIES, strict=False):
            block = detection["factor"][method]
            factors = sorted(float(k) for k in block)
            recall = [block[f"{f:g}"]["recall"] for f in factors]
            precision = [block[f"{f:g}"]["precision"] for f in factors]
            ax_pr.plot(recall, precision, color=color, linewidth=2.0, marker="o", markersize=5,
                       markeredgecolor=viz.SURFACE, markeredgewidth=1.0,
                       label=METHOD_LABELS[method])  # fmt: skip
            conventional = cfg["threshold_factor"] if method == "threshold" else cfg["neo_factor"]
            point = block[f"{conventional:g}"]
            ax_pr.scatter([point["recall"]], [point["precision"]], s=90, facecolor="none",
                          edgecolor=viz.INK, linewidth=1.4, zorder=4)  # fmt: skip
        ax_pr.scatter([], [], s=90, facecolor="none", edgecolor=viz.INK, linewidth=1.4,
                      label="conventional setting")  # fmt: skip
        ax_pr.set_xlim(0, 1)
        ax_pr.set_ylim(0, 1.02)
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.legend(loc="lower left", fontsize=8.5)
        viz.panel_label(ax_pr, f"Operating curves at {reference} µV")
        viz.add_headline(
            fig,
            f"At in-vivo noise a causal threshold detector reaches F1 = {at_ref['threshold']['f1']:.2f}; "
            f"NEO {at_ref['neo']['f1']:.2f}",
            f"The zero-phase filter an implant cannot run scores {at_ref['offline']['f1']:.2f}. "
            "NEO trades recall for precision; neither moves F1 much.",
        )
        viz.add_provenance(
            fig,
            **_mearec_provenance(summary),
            units=(
                f"F1 pooled over channels, band = 95 % bootstrap interval over channels. A target "
                f"spike comes from a unit at least {cfg['target_amplitude_uv']:g} µV deep on that channel."
            ),
        )
        return viz.save_figure(fig, path)


def plot_adc(summary: dict[str, Any], path: Path) -> Path:
    """Draw F1 against ADC resolution and sample rate."""
    cfg = summary["config"]
    detection = summary["mearec"]["detection"]
    base = detection["bits"]["threshold"]["10"]["f1"]
    six = detection["bits"]["threshold"]["6"]["f1"]
    five = detection["bits"]["threshold"]["5"]["f1"]
    ten_k = detection["fs"]["threshold"]["10000"]["f1"]
    with viz.figure_style():
        fig, (ax, ax_fs) = plt.subplots(
            1, 2, figsize=(10.0, 6.4),
            gridspec_kw={"left": 0.085, "right": 0.975, "top": 0.83, "bottom": 0.215, "wspace": 0.22},
        )  # fmt: skip
        for method, color in zip(("threshold", "neo"), viz.SERIES, strict=False):
            _series(ax, detection["bits"][method], color, METHOD_LABELS[method])
            _series(ax_fs, {f"{float(k) / 1e3:g}": v for k, v in detection["fs"][method].items()},
                    color, METHOD_LABELS[method])  # fmt: skip
        bits = sorted(int(float(k)) for k in detection["bits"]["threshold"])
        ax.set_xticks(bits, [f"{b}\n{2000.0 / 2**b:.3g}" for b in bits])
        ax.set_xlabel("ADC bits  (second row: step in µV at the electrode)")
        ax.set_ylabel("Detection F1")
        ax.set_ylim(0, 0.8)
        ax.legend(loc="lower right", fontsize=8.5)
        viz.panel_label(ax, "Resolution, at 30 kS/s")
        ax_fs.set_xscale("log")
        ax_fs.xaxis.set_major_formatter(FuncFormatter(_plain))
        ax_fs.set_xticks([3, 5, 7.5, 10, 15, 30])
        ax_fs.minorticks_off()
        ax_fs.set_xlabel("ADC sample rate (kS/s)")
        ax_fs.set_ylabel("Detection F1")
        ax_fs.set_ylim(0, 0.8)
        viz.panel_label(ax_fs, "Sample rate, at 10 bits")
        viz.add_headline(
            fig,
            f"Six bits are enough for threshold detection (F1 {six:.2f} against {base:.2f}); five are not ({five:.2f})",
            f"A third of the sample rate, 10 kS/s, costs {base - ten_k:.2f} of F1. "
            "Each sweep changes one ADC setting with the other at its baseline.",
        )
        viz.add_provenance(
            fig,
            **_mearec_provenance(summary),
            units=(
                f"F1 at {cfg['reference_noise_uv']:g} µV background noise, ±1 mV input range, band = 95 % "
                "bootstrap interval over channels. No digital filter precedes the sampler, so lower rates alias."
            ),
        )
        return viz.save_figure(fig, path)


def plot_sbp(summary: dict[str, Any], path: Path) -> Path:
    """Draw the Nason reproduction and feature tracking on MEArec."""
    nason = summary["nason"]
    tracking = summary["mearec"]["tracking"]
    cfg = summary["config"]
    arrangement = nason["simulation"]["arrangements"][1]
    in_vivo = nason["in_vivo"]
    online = nason["online"]
    with viz.figure_style():
        fig, axes = plt.subplots(
            2, 2, figsize=(10.0, 9.2),
            gridspec_kw={"left": 0.085, "right": 0.975, "top": 0.885, "bottom": 0.15,
                         "wspace": 0.24, "hspace": 0.42},
        )  # fmt: skip
        width = 0.36
        pairs = (
            ("sbp_r", "SBP, 2 kS/s", viz.SERIES[1]),
            ("crossings_r", "Threshold crossings, 30 kS/s", viz.SERIES[0]),
        )

        ax = axes[0, 0]
        x = np.arange(len(arrangement["snrs"]))
        for offset, (field, label, color) in zip((-width / 2, width / 2), pairs, strict=True):
            ax.bar(x + offset, arrangement[field], width=width * 0.92, color=color, label=label)
        ax.set_xticks(x, [f"{s:g}" for s in arrangement["snrs"]])
        ax.set_xlabel("Unit signal-to-noise ratio (five units share one channel)")
        ax.set_ylabel("Correlation with that unit's firing")
        ax.axhline(0, color=viz.GRID, linewidth=0.8)
        ax.grid(visible=False, axis="x")
        ax.legend(loc="upper right", fontsize=8.5)
        viz.panel_label(ax, "Nason simulation: which unit does each feature follow?")

        ax = axes[0, 1]
        x = np.arange(len(in_vivo["unit_amplitude_uv"]))
        for offset, (field, _, color) in zip((-width / 2, width / 2), pairs, strict=True):
            bars = ax.bar(x + offset, in_vivo[field], width=width * 0.92, color=color)
            ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
        ax.set_xticks(x, [f"{a:.0f} µV" for a in in_vivo["unit_amplitude_uv"]])
        ax.set_xlabel("Sorted unit, by peak-to-peak amplitude")
        ax.set_ylabel("Correlation with that unit's firing")
        ax.axhline(0, color=viz.GRID, linewidth=0.8)
        ax.set_ylim(-0.12, 0.92)
        ax.grid(visible=False, axis="x")
        viz.panel_label(ax, "Nason in vivo: one channel, three sorted units")

        ax = axes[1, 0]
        for feature, color in zip(("crossings", "sbp_digital", "sbp_implant"),
                                  (viz.SERIES[0], viz.SERIES[1], viz.SERIES[2]), strict=True):  # fmt: skip
            _series(ax, tracking[feature], color, FEATURE_LABELS[feature], key="mean")
        ax.set_xlabel("Neural background noise in the spike band (µV rms)")
        ax.set_ylabel("Correlation with the largest unit's firing")
        ax.set_ylim(0, 0.9)
        ax.legend(loc="lower left", fontsize=8.5)
        viz.panel_label(ax, "MEArec: tracking the largest unit on each channel")

        ax = axes[1, 1]
        names = (("velocity_crossings", "Crossings", viz.SERIES[0]), ("velocity_sbp", "SBP", viz.SERIES[1]),
                 ("velocity_both", "Both", viz.INK_SECONDARY))  # fmt: skip
        values = [online[key]["r2"] for key, _, _ in names]
        errors = [online[key]["r2_sem"] for key, _, _ in names]
        bars = ax.bar(range(3), values, yerr=errors, color=[c for _, _, c in names], width=0.6,
                      error_kw={"ecolor": viz.INK, "elinewidth": 1.2, "capsize": 4})  # fmt: skip
        ax.bar_label(bars, fmt="%.2f", fontsize=8.5, padding=6)
        ax.set_xticks(range(3), [label for _, label, _ in names])
        ax.set_ylabel("Held-out R² for finger velocity")
        ax.set_ylim(0, max(values) * 1.3)
        ax.grid(visible=False, axis="x")
        viz.panel_label(ax, "Nason online data: decoding finger velocity")

        reference = f"{cfg['reference_noise_uv']:g}"
        viz.add_headline(
            fig,
            f"SBP at 2 kS/s decodes finger velocity as well as crossings at 30 kS/s "
            f"(R² {online['velocity_sbp']['r2']:.2f} against {online['velocity_crossings']['r2']:.2f})",
            f"It follows the largest unit ({in_vivo['sbp_r'][0]:.2f} in vivo, crossings {in_vivo['crossings_r'][0]:.2f}). "
            f"On MEArec at {reference} µV it tracks that unit at "
            f"{tracking['sbp_implant'][reference]['mean']:.2f} against {tracking['crossings'][reference]['mean']:.2f}.",
        )
        viz.add_provenance(
            fig,
            dataset=(
                "Nason et al. (2020) example data (simulation, in vivo, and 409 online trials on 96 channels), "
                "and the MEArec simulation."
            ),
            split=(
                f"correlations use all scored data; decoding is {online['n_folds']}-fold cross-validation over "
                f"contiguous blocks of trials, {online['bin_ms']} ms bins, error bars = fold SEM."
            ),
            units="Pearson correlation after 50 ms Gaussian smoothing; R² is unitless.",
        )
        return viz.save_figure(fig, path)


def plot_fixed_point(summary: dict[str, Any], path: Path) -> Path:
    """Draw fixed-point error against guard bits and coefficient width."""
    cfg = summary["config"]
    fixed = summary["mearec"]["fixed_point"]
    detection = summary["mearec"]["detection"]
    reference = f"{cfg['reference_noise_uv']:g}"
    guards = cfg["guard_bits_options"]
    fracs = cfg["frac_bits_options"]
    with viz.figure_style():
        fig, (ax, ax_frac) = plt.subplots(
            1, 2, figsize=(10.0, 6.4),
            gridspec_kw={"left": 0.085, "right": 0.975, "top": 0.83, "bottom": 0.215, "wspace": 0.24},
        )  # fmt: skip
        for label, key, color in (("Truncate each product", "trunc", viz.SERIES[1]),
                                  ("Round each product", "round", viz.SERIES[0])):  # fmt: skip
            offsets = [abs(fixed[f"offset_wide_{key}_guard{g}"]) for g in guards]
            ax.plot(guards, offsets, color=color, linewidth=2.0, marker="o", markersize=5,
                    markeredgecolor=viz.SURFACE, markeredgewidth=1.0, label=label)  # fmt: skip
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(_plain))
        ax.set_xticks(list(guards))
        ax.set_xlabel("Guard bits carried through the filter")
        ax.set_ylabel("Output offset (ADC codes, magnitude)")
        ax.legend(loc="upper right", fontsize=8.5)
        viz.panel_label(ax, "300 Hz high-pass at 30 kS/s: DC error")

        for label, key, color in (("30 kS/s (threshold path)", "wide", viz.SERIES[0]),
                                  ("2 kS/s (SBP path)", "narrow", viz.SERIES[1])):  # fmt: skip
            snr = [fixed[f"snr_{key}_frac{f}"] for f in fracs]
            ax_frac.plot(fracs, snr, color=color, linewidth=2.0, marker="o", markersize=5,
                         markeredgecolor=viz.SURFACE, markeredgewidth=1.0, label=label)  # fmt: skip
        ax_frac.set_xticks(list(fracs))
        ax_frac.set_ylim(-10, 35)
        ax_frac.set_xlabel("Fractional bits in the filter coefficients")
        ax_frac.set_ylabel("Signal-to-error ratio against floating point (dB)")
        ax_frac.legend(loc="lower right", fontsize=8.5)
        viz.panel_label(ax_frac, "How precise the coefficients must be")
        viz.add_headline(
            fig,
            f"Truncating inside the 30 kS/s filter shifts its output by "
            f"{abs(fixed['offset_wide_trunc_guard0']):.0f} codes; rounding removes it",
            f"Integer-only detection scores F1 {fixed['fixed_threshold_f1']:.3f} against "
            f"{detection['noise']['threshold'][reference]['f1']:.3f} in floating point. Accumulator width: "
            f"{fixed['acc_bits_wide']:.0f} bits at 30 kS/s, {fixed['acc_bits_narrow']:.0f} bits at 2 kS/s.",
        )
        viz.add_provenance(
            fig,
            **_mearec_provenance(summary),
            units=(
                f"ADC codes (10 bits, ±1 mV) and dB, at {reference} µV background noise, mean over channels. "
                "Left: 14 fractional bits. Right: 6 guard bits, rounding."
            ),
        )
        return viz.save_figure(fig, path)
