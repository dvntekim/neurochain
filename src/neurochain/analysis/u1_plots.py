"""U1 figures: electrode impedance, noise budget, coupling and the emulator check.

Every number printed on a figure is read from the results' ``summary``, so the
figures cannot drift from the results table.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter

from neurochain import viz
from neurochain.analysis.u1 import ELECTRODE_KEYS, U1Results
from neurochain.frontend.model import SPIKE_BAND_HZ

ELECTRODE_LABELS = {
    "in_saline": "Platinum, in saline (measured fit)",
    "ideal_capacitor": "Same |Z| at 1 kHz, ideal capacitor",
    "implanted": "Platinum, implanted (measured fit)",
}
SHORT_LABELS = {
    "in_saline": "In saline",
    "ideal_capacitor": "Ideal capacitor",
    "implanted": "Implanted",
}
MODEL_DATASET = (
    "none; this is a circuit model. Electrode parameters: Straka et al. (2018), Table 1, "
    "Blackrock platinum Utah arrays."
)
MODEL_SPLIT = "not applicable (no data is fitted or evaluated)."


def _plain(value: float, _: object) -> str:
    return f"{value:g}"


def _shade_spike_band(ax: Axes, label_y: float | None = None) -> None:
    ax.axvspan(*SPIKE_BAND_HZ, color=viz.NEUTRAL_FILL, alpha=0.35, linewidth=0)
    if label_y is not None:
        ax.text(
            float(np.sqrt(SPIKE_BAND_HZ[0] * SPIKE_BAND_HZ[1])), label_y, "Spike band",
            ha="center", va="top", fontsize=8.5, color=viz.INK_SECONDARY,
        )  # fmt: skip


def plot_impedance(results: U1Results, path: Path) -> Path:
    """Draw impedance magnitude and phase of the three electrodes."""
    electrodes = results.summary["electrodes"]
    base = electrodes[ELECTRODE_KEYS[0]]
    ideal = electrodes["ideal_capacitor"]
    with viz.figure_style():
        fig, (ax_mag, ax_phase) = plt.subplots(
            2, 1, figsize=(10.0, 8.4), sharex=True,
            gridspec_kw={"left": 0.085, "right": 0.975, "top": 0.875, "bottom": 0.15, "hspace": 0.22},
        )  # fmt: skip
        for key, color in zip(ELECTRODE_KEYS, viz.SERIES, strict=True):
            curve = results.impedance[key]
            ax_mag.plot(
                curve.freqs_hz, np.abs(curve.analytic_ohm), color=color, linewidth=2.0,
                label=ELECTRODE_LABELS[key],
            )  # fmt: skip
            ax_phase.plot(
                curve.freqs_hz, np.degrees(np.angle(curve.analytic_ohm)), color=color, linewidth=2.0
            )
            if curve.spice_freqs_hz is not None and curve.spice_ohm is not None:
                pick = slice(None, None, 4)
                marker = {"s": 26, "facecolor": viz.SURFACE, "edgecolor": color, "linewidth": 1.4,
                          "zorder": 3}  # fmt: skip
                ax_mag.scatter(curve.spice_freqs_hz[pick], np.abs(curve.spice_ohm[pick]), **marker)
                ax_phase.scatter(
                    curve.spice_freqs_hz[pick],
                    np.degrees(np.angle(curve.spice_ohm[pick])),
                    **marker,
                )
        ax_mag.scatter([], [], s=26, facecolor=viz.SURFACE, edgecolor=viz.INK_SECONDARY,
                       linewidth=1.4, label="ngspice (lines are the analytic model)")  # fmt: skip
        for ax in (ax_mag, ax_phase):
            ax.set_xscale("log")
            ax.axvline(1000.0, color=viz.INK_SECONDARY, linewidth=0.8, linestyle=(0, (4, 3)))
        ax_mag.set_yscale("log")
        ax_mag.set_ylabel("|Z| (Ω)")
        ax_mag.set_xlim(1.0, 1.0e4)
        ax_mag.legend(loc="upper right", fontsize=8.5)
        ax_mag.annotate(
            f"{base['z_1khz_ohm'] / 1e3:.0f} kΩ at 1 kHz\n(matched by construction)",
            xy=(1000.0, base["z_1khz_ohm"]), xytext=(-10, -34), textcoords="offset points",
            ha="right", fontsize=8.5,
        )  # fmt: skip
        viz.panel_label(ax_mag, "Impedance magnitude")
        ax_phase.set_ylabel("Phase (degrees)")
        ax_phase.set_xlabel("Frequency (Hz)")
        ax_phase.set_ylim(-95, 0)
        ax_phase.xaxis.set_major_formatter(FuncFormatter(_plain))
        viz.panel_label(ax_phase, "Impedance phase: −90° is a lossless capacitor, 0° a resistor")
        viz.add_headline(
            fig,
            f"A real electrode is lossy: −{abs(base['phase_1khz_deg']):.0f}° at 1 kHz, where the "
            f"textbook capacitor gives −{abs(ideal['phase_1khz_deg']):.0f}°",
            "Randles electrode with a constant-phase double layer, against the ideal-capacitor "
            "version with the same |Z| at 1 kHz.",
        )
        viz.add_provenance(
            fig, dataset=MODEL_DATASET, split=MODEL_SPLIT,
            units="ohms and degrees. Open markers: ngspice AC analysis, every fourth point.",
        )  # fmt: skip
        return viz.save_figure(fig, path)


def plot_noise_budget(results: U1Results, path: Path) -> Path:
    """Draw the input-referred noise spectrum and the per-electrode budget."""
    summary = results.summary
    spectrum = results.spectrum
    base = summary["noise"][ELECTRODE_KEYS[0]]
    worst = max(entry["max_disagreement"] for entry in summary["noise"].values())
    methods = "hand, analytic and SPICE" if summary["spice_available"] else "hand and analytic"
    band = f"{SPIKE_BAND_HZ[0]:g} Hz to {SPIKE_BAND_HZ[1] / 1e3:g} kHz"
    with viz.figure_style():
        fig, (ax, ax_bar) = plt.subplots(
            1, 2, figsize=(10.0, 6.6),
            gridspec_kw={"left": 0.085, "right": 0.975, "top": 0.835, "bottom": 0.215,
                         "wspace": 0.26, "width_ratios": [1.45, 1.0]},
        )  # fmt: skip
        series = (
            ("Electrode thermal noise", spectrum.electrode, viz.SERIES[0]),
            ("Amplifier noise", spectrum.lna, viz.SERIES[1]),
            ("Total", spectrum.total, viz.INK),
        )
        for label, psd, color in series:
            ax.plot(spectrum.freqs_hz, np.sqrt(psd) * 1e9, color=color, linewidth=2.0, label=label)
        if results.spice_spectra is not None:
            for source, color in (("electrode", viz.SERIES[0]), ("lna", viz.SERIES[1]),
                                  ("all", viz.INK)):  # fmt: skip
                run = results.spice_spectra[source]
                pick = slice(None, None, 24)
                ax.scatter(
                    run.noise_freqs_hz[pick], run.input_noise_asd[pick] * 1e9, s=26,
                    facecolor=viz.SURFACE, edgecolor=color, linewidth=1.4, zorder=3,
                )  # fmt: skip
            ax.scatter([], [], s=26, facecolor=viz.SURFACE, edgecolor=viz.INK_SECONDARY,
                       linewidth=1.4, label="ngspice noise analysis")  # fmt: skip
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(spectrum.freqs_hz[0], spectrum.freqs_hz[-1])
        top = float(np.sqrt(spectrum.total.max()) * 1e9) * 1.6
        ax.set_ylim(10.0, top)
        _shade_spike_band(ax, label_y=top / 1.25)
        ax.yaxis.set_major_formatter(FuncFormatter(_plain))
        ax.xaxis.set_major_formatter(FuncFormatter(_plain))
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Input-referred noise density (nV/√Hz)")
        ax.legend(loc="lower left", fontsize=8.5)
        viz.panel_label(ax, "Noise density of the baseline front end")

        width = 0.26
        positions = np.arange(len(ELECTRODE_KEYS))
        parts = (("electrode_uv", "Electrode", viz.SERIES[0]), ("lna_uv", "Amplifier", viz.SERIES[1]),
                 ("total_uv", "Total", viz.INK_SECONDARY))  # fmt: skip
        for offset, (field, label, color) in zip((-width, 0.0, width), parts, strict=True):
            values = [summary["noise"][key]["analytic"][field] for key in ELECTRODE_KEYS]
            bars = ax_bar.bar(
                positions + offset, values, width=width * 0.9, color=color, label=label
            )
            ax_bar.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
        ax_bar.set_xticks(positions, [SHORT_LABELS[key] for key in ELECTRODE_KEYS])
        ax_bar.set_ylabel("Noise in the spike band (µV rms)")
        ax_bar.set_ylim(
            0, max(summary["noise"][k]["analytic"]["total_uv"] for k in ELECTRODE_KEYS) * 1.3
        )
        ax_bar.grid(visible=False, axis="x")
        ax_bar.legend(loc="upper center", ncols=3, fontsize=8.5, columnspacing=1.0)
        viz.panel_label(ax_bar, "Budget by electrode model")
        viz.add_headline(
            fig,
            f"Front-end noise floor: {base['analytic']['total_uv']:.2f} µV rms; {methods} agree "
            f"within {worst:.1%}",
            f"Electrode {base['analytic']['electrode_uv']:.2f} µV and amplifier "
            f"{base['analytic']['lna_uv']:.2f} µV over {band}, at "
            f"{summary['amplifier']['power_uw']:.0f} µW. An ideal-capacitor electrode model "
            "would hide most of the electrode's share.",
        )
        viz.add_provenance(
            fig, dataset=MODEL_DATASET, split=MODEL_SPLIT,
            units=(
                f"nV/√Hz and µV rms, referred to the signal in the tissue, integrated "
                f"{SPIKE_BAND_HZ[0]:g} Hz to {SPIKE_BAND_HZ[1] / 1e3:g} kHz, at "
                f"{summary['amplifier']['temperature_k'] - 273.15:.0f} °C. Bars: analytic model."
            ),
        )  # fmt: skip
        return viz.save_figure(fig, path)


def plot_coupling(results: U1Results, path: Path) -> Path:
    """Draw DC-offset tolerance and the recovery from an offset step."""
    summary = results.summary
    cpl = summary["coupling"]
    sweep = results.offsets
    slow = max(summary["settling"]["runs"], key=lambda run: run["time_constant_s"])
    with viz.figure_style():
        fig, (ax, ax_time) = plt.subplots(
            1, 2, figsize=(10.0, 6.4),
            gridspec_kw={"left": 0.085, "right": 0.975, "top": 0.83, "bottom": 0.215, "wspace": 0.24},
        )  # fmt: skip
        shown = sweep.offsets_v > 0
        for (label, level), color in zip(sweep.output_level.items(), viz.SERIES, strict=True):
            ax.plot(
                sweep.offsets_v[shown] * 1e3, level[shown], color=color, linewidth=2.0, marker="o",
                markersize=5, markeredgecolor=viz.SURFACE, markeredgewidth=1.0,
                label=f"{label.split(',')[0]}, gain {sweep.gains[label]:.2g}",
            )  # fmt: skip
        ax.axhline(1.0, color=viz.INK_SECONDARY, linewidth=1.0, linestyle=(0, (4, 3)))
        ax.text(330.0, 1.02, "supply rail", fontsize=8.5, color=viz.INK_SECONDARY, va="bottom",
                ha="right")  # fmt: skip
        ax.set_xlim(0.1, 400.0)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(_plain))
        ax.set_ylim(-0.04, 1.14)
        ax.set_xlabel("Electrode DC offset (mV)")
        ax.set_ylabel("Output DC level ÷ rail")
        ax.legend(loc="upper left", fontsize=8.5, bbox_to_anchor=(0.0, 0.85))
        viz.panel_label(ax, "How much offset each coupling survives")

        for run, color in zip(results.settling, viz.SERIES, strict=False):
            after = run.time_s > 0
            ax_time.plot(
                run.time_s[after], run.residual_uv[after], color=color, linewidth=1.6,
                label=f"{run.f_low_hz:g} Hz corner: railed {run.saturation_time_simulated_s:.2f} s",
            )  # fmt: skip
        rail_uv = results.settling[0].rail_uv
        noise_uv = summary["noise"][ELECTRODE_KEYS[0]]["analytic"]["total_uv"]
        ax_time.axhline(rail_uv, color=viz.INK_SECONDARY, linewidth=1.0, linestyle=(0, (4, 3)))
        ax_time.axhline(noise_uv, color=viz.INK_SECONDARY, linewidth=1.0, linestyle=(0, (1, 2)))
        x_right = float(max(run.time_s[-1] for run in results.settling))
        ax_time.text(x_right, rail_uv * 1.25, "rail", ha="right", fontsize=8.5,
                     color=viz.INK_SECONDARY)  # fmt: skip
        ax_time.text(0.3 * x_right, noise_uv * 1.25, f"noise floor {noise_uv:.1f} µV", ha="center",
                     fontsize=8.5, color=viz.INK_SECONDARY)  # fmt: skip
        ax_time.set_yscale("log")
        ax_time.set_xlim(0, x_right)
        ax_time.set_ylim(0.05, rail_uv * 4)
        ax_time.set_xlabel(f"Time after a {summary['settling']['step_mv']:g} mV offset step (s)")
        ax_time.set_ylabel("Offset left at the output ÷ gain (µV)")
        ax_time.legend(loc="upper right", fontsize=8.5, bbox_to_anchor=(1.0, 0.86))
        viz.panel_label(ax_time, "AC coupling: recovery after an offset step")
        viz.add_headline(
            fig,
            f"AC coupling shrugs off {cpl['worst_offset_mv']:g} mV of offset; DC coupling at the "
            f"same gain rails at {cpl['dc_offset_tolerance_same_gain_mv']:.1f} mV",
            f"Surviving {cpl['worst_offset_mv']:g} mV DC-coupled needs a gain of "
            f"{cpl['dc_gain_for_worst_offset']:.2g} and a {cpl['adc_bits_dc']:.1f}-bit ADC, against "
            f"{cpl['adc_bits_ac']:.1f} bits AC-coupled. A {slow['f_low_hz']:g} Hz corner stays railed "
            f"{slow['saturation_time_simulated_s']:.1f} s after the step.",
        )
        viz.add_provenance(
            fig,
            dataset=(
                "synthetic (seeded spikes and noise, 2 s) for the offset sweep; a "
                f"{summary['settling']['step_mv']:g} mV step on silence for the recovery. "
                "Emulated baseline front end."
            ),
            split=MODEL_SPLIT,
            units=(
                "left, output DC level as a share of the rail; right, µV referred to the input, "
                "5 ms block means (which is why the curves settle below the noise floor)."
            ),
        )
        return viz.save_figure(fig, path)


def plot_emulator(results: U1Results, path: Path) -> Path:
    """Draw the emulator's effect on the Nason in-vivo recording.

    Raises:
        ValueError: If the results hold no Nason check.
    """
    check = results.nason
    stats = results.summary["nason"]
    if check is None or stats is None:
        raise ValueError("the Nason recording was not analysed")
    model_uv = stats["sigma_model_floor_uv"]
    amp = results.summary["amplifier"]
    extra = stats["sigma_quarter_bias_uv"] - stats["sigma_emulated_uv"]
    with viz.figure_style():
        fig, (ax, ax_bar) = plt.subplots(
            1, 2, figsize=(10.0, 6.6),
            gridspec_kw={"left": 0.085, "right": 0.975, "top": 0.835, "bottom": 0.215,
                         "wspace": 0.42, "width_ratios": [1.3, 1.0]},
        )  # fmt: skip
        keep = (check.freqs_hz >= 10.0) & (check.freqs_hz < check.fs_hz / 2.0)
        freqs = check.freqs_hz[keep]
        ax.plot(freqs, np.sqrt(check.psd_recorded[keep]) * 1e3, color=viz.INK, linewidth=2.6,
                label="Recording as published")  # fmt: skip
        ax.plot(freqs, np.sqrt(check.psd_emulated[keep]) * 1e3, color=viz.SERIES[1], linewidth=1.0,
                label="After the emulated front end")  # fmt: skip
        ax.plot(freqs, np.sqrt(check.model_input_psd[keep]) * 1e3, color=viz.SERIES[0],
                linewidth=2.0, label="Model noise floor (electrode + amplifier)")  # fmt: skip
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(freqs[0], freqs[-1])
        top = float(np.sqrt(check.psd_recorded[keep].max()) * 1e3) * 2.0
        ax.set_ylim(10.0, top)
        _shade_spike_band(ax, label_y=top / 1.4)
        ax.xaxis.set_major_formatter(FuncFormatter(_plain))
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude spectral density (nV/√Hz)")
        ax.legend(loc="lower left", fontsize=8.5)
        viz.panel_label(ax, "Spectrum of the in-vivo channel")

        rows = [
            ("Model floor,\nno neural signal", model_uv, viz.SERIES[0]),
            ("Recording\nas published", stats["sigma_recorded_uv"], viz.INK_SECONDARY),
            (f"Emulated,\n{amp['power_uw']:.0f} µW amplifier", stats["sigma_emulated_uv"],
             viz.SERIES[1]),
            (f"Emulated,\n{stats['quarter_bias_power_uw']:.1f} µW amplifier",
             stats["sigma_quarter_bias_uv"], viz.SERIES[1]),
        ]  # fmt: skip
        y = np.arange(len(rows))[::-1]
        bars = ax_bar.barh(y, [r[1] for r in rows], color=[r[2] for r in rows], height=0.6)
        ax_bar.bar_label(bars, fmt="%.2f µV", fontsize=8.5, padding=3)
        ax_bar.set_yticks(y, [r[0] for r in rows])
        ax_bar.set_xlim(0, max(r[1] for r in rows) * 1.28)
        ax_bar.set_xlabel("Spike-band noise σ (µV)")
        ax_bar.grid(visible=False, axis="y")
        viz.panel_label(ax_bar, "Noise σ after the spike-band filter")
        viz.add_headline(
            fig,
            f"In vivo the noise is {stats['sigma_recorded_uv']:.1f} µV, "
            f"{stats['recorded_over_model']:.1f} times the front end's own "
            f"{model_uv:.1f} µV floor",
            f"Adding the modelled front end moves σ from {stats['sigma_band_limited_uv']:.2f} to "
            f"{stats['sigma_emulated_uv']:.2f} µV (predicted {stats['sigma_predicted_uv']:.2f}). "
            f"Quartering the bias current adds another {extra:.2f} µV.",
        )
        viz.add_provenance(
            fig,
            dataset=(
                "Nason et al. (2020) example data, MultiUnitInVivo.mat: one channel of monkey W, "
                f"10 s at {check.fs_hz:,.0f} Hz."
            ),
            split="none; one recording, described and passed through the emulator (seed 0).",
            units=(
                "nV/√Hz and µV, referred to the electrode; σ is the MAD estimate on the 300 Hz to "
                "7.5 kHz band. The recording holds its own system's noise; emulated noise adds to it."
            ),
        )
        return viz.save_figure(fig, path)
