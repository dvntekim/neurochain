"""Spiking-band power (SBP) and the binning shared by all features.

SBP is the average absolute value of the 300 Hz to 1 kHz band (Nason et al.,
2020). It needs no threshold and no spike shapes, and because the band stops at
1 kHz it can be sampled at 2 kHz, a tenth of what spike detection uses.

Two ways to compute it are provided:

* :func:`spiking_band_power` follows the paper: filter the broadband digitally,
  keep every 15th sample, take the absolute value.
* An implant would instead narrow the amplifier's pass band to 300 Hz to 1 kHz
  and sample at 2 kHz directly. That path is the emulator and ADC from U1 and
  :mod:`neurochain.adc`, followed by :func:`rectify_and_bin`.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from neurochain.adc import decimation_factor
from neurochain.recording import FloatArray, IntArray

SBP_BAND_HZ = (300.0, 1000.0)
SBP_FS_HZ = 2000.0


def spiking_band_power(
    x: FloatArray,
    fs_hz: float,
    *,
    band_hz: tuple[float, float] = SBP_BAND_HZ,
    out_fs_hz: float = SBP_FS_HZ,
    order: int = 2,
) -> FloatArray:
    """Compute SBP from broadband the way Nason et al. (2020) describe.

    A causal Butterworth band-pass of the given order, then every k-th sample,
    then the absolute value. ``tests/test_nason_data.py`` shows this reproduces
    the SBP stored with the paper's data.

    Returns:
        The rectified band signal at ``out_fs_hz``.
    """
    b, a = signal.butter(order, band_hz, btype="bandpass", fs=fs_hz)
    filtered = signal.lfilter(b, a, x)
    return np.asarray(np.abs(filtered[:: decimation_factor(fs_hz, out_fs_hz)]), dtype=np.float64)


def bin_mean(x: FloatArray, samples_per_bin: int) -> FloatArray:
    """Average consecutive samples in bins, dropping an incomplete last bin."""
    n_bins = x.shape[-1] // samples_per_bin
    trimmed = x[..., : n_bins * samples_per_bin]
    return np.asarray(
        trimmed.reshape(*x.shape[:-1], n_bins, samples_per_bin).mean(axis=-1), dtype=np.float64
    )


def rectify_and_bin(x: FloatArray, samples_per_bin: int) -> FloatArray:
    """Return the mean absolute value in each bin: SBP from a band-limited signal."""
    return bin_mean(np.abs(x), samples_per_bin)


def bin_counts(events: IntArray, n_samples: int, samples_per_bin: int) -> FloatArray:
    """Count events in consecutive bins, dropping an incomplete last bin."""
    n_bins = n_samples // samples_per_bin
    inside = events[events < n_bins * samples_per_bin]
    return np.bincount(inside // samples_per_bin, minlength=n_bins).astype(np.float64)


def gaussian_window(duration_s: float, fs_hz: float) -> FloatArray:
    """Return a unit-sum Gaussian window spanning ``duration_s`` (±2.5 sigma)."""
    n = max(3, round(duration_s * fs_hz))
    t = np.arange(n) - (n - 1) / 2.0
    window = np.exp(-0.5 * (t / (n / 5.0)) ** 2)
    return np.asarray(window / window.sum(), dtype=np.float64)


def smooth(x: FloatArray, window: FloatArray) -> FloatArray:
    """Smooth along the first axis with a window, forward and backward.

    This is the zero-phase smoothing Nason et al. apply before correlating a
    feature with a firing rate. It is an analysis step, not part of the implant.
    """
    return np.asarray(signal.filtfilt(window, [1.0], x, axis=0), dtype=np.float64)
