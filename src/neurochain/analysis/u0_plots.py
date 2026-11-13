"""U0 figures: band-split spectra and spike amplitudes.

Both functions take the arrays from :mod:`neurochain.analysis.u0` and one
example recording. Every number printed on a figure is read from ``summary``,
so the figures cannot drift from the results table.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

from neurochain import viz
from neurochain.analysis.u0 import QUANTISATION_RMS_LSB, U0Results, median_psd
from neurochain.dsp.filters import BandSplit, band_split
from neurochain.dsp.spikes import SpikeDetection, detect_spikes, extract_waveforms
from neurochain.recording import FloatArray, IntArray, Recording

EXAMPLE_WINDOW_S = 1.0
WAVEFORM_PRE_S = 0.5e-3
WAVEFORM_POST_S = 1.0e-3


def densest_window(indices: IntArray, n_samples: int, window: int) -> tuple[int, int]:
    """Return the ``[start, stop)`` sample window of a given length holding most spikes.

    Candidate windows start a tenth of a window before each spike; ties go to
    the earliest. With no spikes the window starts at zero.
    """
    window = min(window, n_samples)
    best_start, best_count = 0, -1
    for index in indices:
        start = int(np.clip(index - window // 10, 0, n_samples - window))
        count = int(np.count_nonzero((indices >= start) & (indices < start + window)))
        if count > best_count:
            best_start, best_count = start, count
    return best_start, best_start + window


def _example_bands(example: Recording, results: U0Results) -> tuple[BandSplit, SpikeDetection]:
    """Band-split and detect spikes on the example clip with the results' config."""
    config = results.config
    bands = band_split(
        example.channel(0),
        example.fs_hz,
        lfp_cutoff_hz=config.lfp_cutoff_hz,
        spike_band_hz=config.spike_band_hz,
        order=config.filter_order,
        zero_phase=config.zero_phase,
    )
    detection = detect_spikes(
        bands.spikes,
        example.fs_hz,
        threshold_sigma=config.threshold_sigma,
        polarity=config.polarity,
        refractory_s=config.refractory_s,
    )
    return bands, detection


def _band_label(low_hz: float, high_hz: float) -> str:
    """Format a band as text, using kHz above 1 kHz."""

    def fmt(f: float) -> str:
        return f"{f / 1000:g} kHz" if f >= 1000 else f"{f:g} Hz"

    return f"{fmt(low_hz)} to {fmt(high_hz)}"


def _provenance(results: U0Results, dataset_label: str) -> dict[str, str]:
    """Build the dataset / split / units lines shared by both figures."""
    n_clean = int(results.clean.sum())
    excluded = results.n_clips - n_clean
    seconds = float(np.median(results.duration_s))
    units_note = (
        "ADC LSB (signed 10-bit codes). The recording's µV scale is not published."
        if results.units == "ADC LSB"
        else results.units
    )
    return {
        "dataset": (
            f"{dataset_label}. {n_clean} of {results.n_clips} clips of about {seconds:.0f} s at "
            f"{results.fs_hz:,.0f} Hz; {excluded} excluded for ADC saturation."
        ),
        "split": "none. Descriptive statistics over every clean clip; no model is fitted.",
        "units": units_note,
    }


def plot_spectra(
    results: U0Results,
    summary: dict[str, Any],
    example: Recording,
    path: Path,
    *,
    dataset_label: str,
) -> Path:
    """Draw the band-split traces of one clip above the across-clip spectrum.

    Args:
        results: Per-clip analysis arrays.
        summary: Output of :func:`neurochain.analysis.u0.summarise`.
        example: The recording used for the example traces.
        path: Output PNG path.
        dataset_label: Dataset name for the provenance line.
    """
    config = results.config
    units = results.units
    bands, detection = _example_bands(example, results)
    start, stop = densest_window(
        detection.indices, example.n_samples, round(EXAMPLE_WINDOW_S * example.fs_hz)
    )
    t_ms = (np.arange(start, stop) - start) / example.fs_hz * 1e3
    raw = example.channel(0) - float(np.mean(example.channel(0)))
    spike_label = _band_label(*config.spike_band_hz)
    lfp_fraction = summary["spectrum"]["lfp_power_fraction"]["median"]
    sigma = summary["noise"]["sigma"]["median"]

    with viz.figure_style():
        fig = plt.figure(figsize=(10.0, 9.6))
        outer = fig.add_gridspec(
            2, 1, height_ratios=[3.0, 3.4], left=0.085, right=0.975, top=0.885, bottom=0.135,
            hspace=0.30,
        )  # fmt: skip
        traces = outer[0].subgridspec(3, 1, hspace=0.55)
        rows: list[tuple[str, FloatArray, str]] = [
            ("Raw clip, mean removed", raw, viz.INK_SECONDARY),
            (f"LFP band: below {config.lfp_cutoff_hz:g} Hz", bands.lfp, viz.COLOR_LFP),
            (f"Spike band: {spike_label}", bands.spikes, viz.COLOR_SPIKE),
        ]
        first: Axes | None = None
        for row, (title, y, color) in enumerate(rows):
            ax = fig.add_subplot(traces[row], sharex=first)
            first = first or ax
            ax.plot(t_ms, y[start:stop], color=color, linewidth=0.7)
            viz.panel_label(ax, title)
            ax.set_ylabel(units)
            ax.margins(x=0)
            if row < len(rows) - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel(f"Time in the example clip (ms), clip {example.meta['clip_id'][:8]}")

        ax = fig.add_subplot(outer[1])
        # Drop the DC bin (log axis) and the Nyquist bin (a one-sided PSD halves it).
        freqs = results.freqs_hz[1:-1]
        psd_clean = results.psd[results.clean][:, 1:-1]
        asd_p25, asd_p75 = np.sqrt(np.percentile(psd_clean, [25, 75], axis=0))
        asd_median = np.sqrt(median_psd(results)[1:-1])
        nyquist_hz = results.fs_hz / 2.0
        quantisation_asd = float(QUANTISATION_RMS_LSB) / np.sqrt(nyquist_hz)

        ax.axvspan(freqs[0], config.lfp_cutoff_hz, color=viz.COLOR_LFP, alpha=0.08, linewidth=0)
        ax.axvspan(*config.spike_band_hz, color=viz.COLOR_SPIKE, alpha=0.08, linewidth=0)
        ax.fill_between(
            freqs,
            asd_p25,
            asd_p75,
            color=viz.NEUTRAL_FILL,
            linewidth=0,
            label="Interquartile range",
        )
        ax.plot(
            freqs, asd_median, color=viz.INK, linewidth=2.0,
            label=f"Median of {int(results.clean.sum())} clips",
        )  # fmt: skip
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(freqs[0], nyquist_hz)
        top = float(asd_p75.max()) * 2.2
        ax.set_ylim(quantisation_asd / 1.8 if units == "ADC LSB" else None, top)
        label_y = top / 1.35
        ax.text(
            np.sqrt(freqs[0] * config.lfp_cutoff_hz), label_y, "LFP band", ha="center", va="top",
            fontweight="bold",
        )  # fmt: skip
        ax.text(
            np.sqrt(config.spike_band_hz[0] * config.spike_band_hz[1]), label_y, "Spike band",
            ha="center", va="top", fontweight="bold",
        )  # fmt: skip
        ax.annotate(
            f"PSD ∝ f$^{{{summary['spectrum']['psd_slope_lfp_band']:.2f}}}$\n"
            f"(10 to {config.lfp_cutoff_hz:g} Hz)",
            xy=(110.0, float(np.interp(110.0, freqs, asd_median))),
            xytext=(-12, -38), textcoords="offset points", ha="right", fontsize=8.5,
            color=viz.INK_SECONDARY,
            arrowprops={"arrowstyle": "-", "color": viz.INK_SECONDARY, "linewidth": 0.7},
        )  # fmt: skip
        ax.annotate(
            f"PSD ∝ f$^{{{summary['spectrum']['psd_slope_spike_band']:.2f}}}$\n"
            f"({config.spike_band_hz[0]:g} Hz to 5 kHz)",
            xy=(1500.0, float(np.interp(1500.0, freqs, asd_median))),
            xytext=(-12, -38), textcoords="offset points", ha="right", fontsize=8.5,
            color=viz.INK_SECONDARY,
            arrowprops={"arrowstyle": "-", "color": viz.INK_SECONDARY, "linewidth": 0.7},
        )  # fmt: skip
        if units == "ADC LSB":
            ax.axhline(
                quantisation_asd, color=viz.INK_SECONDARY, linewidth=1.0, linestyle=(0, (4, 3))
            )
            ax.text(
                freqs[0] * 1.15, quantisation_asd * 1.12,
                "Ideal 10-bit quantisation noise: 1/√12 LSB rms spread evenly from 0 to fs/2",
                ha="left", va="bottom", fontsize=8.5, color=viz.INK_SECONDARY,
            )  # fmt: skip
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(f"Amplitude spectral density ({units}/√Hz)")
        viz.panel_label(ax, "Amplitude spectral density across clips (Welch, Hann, 50 % overlap)")
        ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.86), fontsize=8.5)

        viz.add_headline(
            fig,
            f"{lfp_fraction:.0%} of the signal power sits below {config.lfp_cutoff_hz:g} Hz; "
            f"spike-band noise is {sigma:.1f} LSB rms",
            "Band-split of one example clip (top) and the spectrum of every clean clip (bottom). "
            "Percentages and noise are medians across clips.",
        )
        viz.add_provenance(fig, **_provenance(results, dataset_label))
        return viz.save_figure(fig, path)


def _log_histogram(
    ax: Axes, values: FloatArray, *, low: float, ticks: list[float], n_bins: int = 45
) -> None:
    """Draw a histogram with logarithmically spaced bins on a log x-axis."""
    edges = np.geomspace(low, float(values.max()) * 1.001, n_bins + 1)
    counts, _ = np.histogram(values, bins=edges)
    ax.stairs(counts, edges, fill=True, color=viz.COLOR_SPIKE, linewidth=0)
    ax.set_xscale("log")
    ax.set_xlim(edges[0], edges[-1])
    ax.xaxis.set_major_locator(FixedLocator([t for t in ticks if edges[0] <= t <= edges[-1]]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(0, float(counts.max()) * 1.18)
    ax.grid(visible=False, axis="x")


def _mark_value(ax: Axes, x: float, text: str, *, align: str = "left") -> None:
    """Draw a dashed vertical marker with a text label at the top of the axes."""
    ax.axvline(x, color=viz.INK, linewidth=1.0, linestyle=(0, (4, 3)))
    offset = 5 if align == "left" else -5
    ax.annotate(
        text, xy=(x, 0.97), xycoords=("data", "axes fraction"), xytext=(offset, 0),
        textcoords="offset points", ha=align, va="top", fontsize=8.5,
    )  # fmt: skip


def plot_spike_amplitudes(
    results: U0Results,
    summary: dict[str, Any],
    example: Recording,
    path: Path,
    *,
    dataset_label: str,
) -> Path:
    """Draw detected spikes on an example clip and the amplitude distributions.

    Args:
        results: Per-clip analysis arrays.
        summary: Output of :func:`neurochain.analysis.u0.summarise`.
        example: The recording used for the example trace and waveforms.
        path: Output PNG path.
        dataset_label: Dataset name for the provenance line.
    """
    config = results.config
    units = results.units
    bands, detection = _example_bands(example, results)
    fs_hz = example.fs_hz
    start, stop = densest_window(
        detection.indices, example.n_samples, round(EXAMPLE_WINDOW_S * fs_hz)
    )
    t_ms = (np.arange(start, stop) - start) / fs_hz * 1e3
    in_window = (detection.indices >= start) & (detection.indices < stop)
    pre, post = round(WAVEFORM_PRE_S * fs_hz), round(WAVEFORM_POST_S * fs_hz)
    waveforms, _ = extract_waveforms(bands.spikes, detection.indices, pre=pre, post=post)
    waveform_t_ms = np.arange(-pre, post) / fs_hz * 1e3

    clean_spikes = results.spike_is_clean
    amplitude = np.abs(results.spike_amplitudes[clean_spikes])
    snr = results.spike_snr[clean_spikes]
    stats = summary["spikes"]
    sign = "−" if config.polarity == "neg" else "+"

    with viz.figure_style():
        fig = plt.figure(figsize=(10.0, 8.2))
        grid = fig.add_gridspec(
            2, 3, height_ratios=[1.0, 1.25], left=0.085, right=0.975, top=0.865, bottom=0.165,
            hspace=0.48, wspace=0.34,
        )  # fmt: skip

        ax = fig.add_subplot(grid[0, :])
        ax.plot(t_ms, bands.spikes[start:stop], color=viz.COLOR_SPIKE, linewidth=0.7)
        ax.axhline(detection.threshold, color=viz.INK, linewidth=1.0, linestyle=(0, (4, 3)))
        ax.scatter(
            (detection.indices[in_window] - start) / fs_hz * 1e3,
            detection.amplitudes[in_window],
            s=22, color=viz.INK, edgecolor=viz.SURFACE, linewidth=0.8, zorder=3,
        )  # fmt: skip
        ax.annotate(
            f"Threshold {sign}{config.threshold_sigma:g}σ = "
            f"{sign}{abs(detection.threshold):.1f} {units}",
            xy=(1.0, detection.threshold), xycoords=("axes fraction", "data"),
            xytext=(-4, -6), textcoords="offset points", ha="right", va="top", fontsize=8.5,
            bbox={"facecolor": viz.SURFACE, "edgecolor": "none", "pad": 1.5},
        )  # fmt: skip
        ax.margins(x=0)
        ax.set_xlabel(f"Time in the example clip (ms), clip {example.meta['clip_id'][:8]}")
        ax.set_ylabel(units)
        viz.panel_label(
            ax,
            f"Spike band of the example clip: {int(in_window.sum())} detected spikes (dots) "
            f"in {EXAMPLE_WINDOW_S:g} s, noise σ = {detection.sigma:.1f} {units}",
        )

        ax = fig.add_subplot(grid[1, 0])
        shown = waveforms[:200]
        ax.plot(waveform_t_ms, shown.T, color=viz.COLOR_SPIKE, linewidth=0.5, alpha=0.18)
        if waveforms.size:
            ax.plot(waveform_t_ms, waveforms.mean(axis=0), color=viz.INK, linewidth=2.0)
        ax.margins(x=0)
        ax.set_xlabel("Time from trough (ms)")
        ax.set_ylabel(units)
        viz.panel_label(
            ax, f"Aligned waveforms, example clip\n(n = {waveforms.shape[0]}; black = mean)"
        )

        ax = fig.add_subplot(grid[1, 1])
        _log_histogram(
            ax, amplitude, low=float(amplitude.min()), ticks=[10, 20, 30, 50, 100, 200, 300, 500]
        )
        _mark_value(ax, stats["amplitude"]["median"], f"median {stats['amplitude']['median']:.1f}")
        ax.set_xlabel(f"Trough depth ({units})")
        ax.set_ylabel("Spikes per bin")
        viz.panel_label(ax, f"Amplitude of all spikes\n(n = {stats['n_spikes']:,})")

        ax = fig.add_subplot(grid[1, 2])
        _log_histogram(ax, snr, low=config.threshold_sigma, ticks=[4.5, 6, 8, 12, 20, 40])
        _mark_value(ax, stats["snr"]["median"], f"median {stats['snr']['median']:.1f}σ")
        ax.set_xlabel("Trough depth ÷ that clip's noise σ")
        ax.set_ylabel("Spikes per bin")
        viz.panel_label(
            ax,
            f"Amplitude ÷ noise σ\n(axis starts at {config.threshold_sigma:g}σ)",
        )

        viz.add_headline(
            fig,
            f"The median detected spike is {stats['amplitude']['median']:.0f} LSB deep, "
            f"{stats['snr']['median']:.1f} times the noise floor",
            f"Threshold crossings at {sign}{config.threshold_sigma:g}σ on the "
            f"{_band_label(*config.spike_band_hz)} band; σ is estimated per clip from the "
            "median absolute deviation.",
        )
        provenance = _provenance(results, dataset_label)
        provenance["units"] += " Histogram axes and bins are logarithmic."
        viz.add_provenance(fig, **provenance)
        return viz.save_figure(fig, path)
