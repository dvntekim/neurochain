"""Power spectral density estimation and band power."""

from __future__ import annotations

import numpy as np
from scipy import signal

from neurochain.recording import FloatArray


def segment_length(fs_hz: float, resolution_hz: float) -> int:
    """Return the power-of-two Welch segment length nearest a target resolution.

    The spacing between frequency bins is ``fs_hz / nperseg``. A power of two
    keeps the FFT fast.
    """
    if resolution_hz <= 0:
        raise ValueError("resolution_hz must be positive")
    return int(2 ** round(np.log2(fs_hz / resolution_hz)))


def welch_psd(
    x: FloatArray,
    fs_hz: float,
    *,
    resolution_hz: float = 1.2,
    axis: int = -1,
) -> tuple[FloatArray, FloatArray]:
    """Estimate the one-sided power spectral density by Welch's method.

    Hann-windowed segments with 50 % overlap, each with its mean removed, are
    transformed and their periodograms averaged. Averaging trades frequency
    resolution for a lower-variance estimate.

    Args:
        x: Samples, with time along ``axis``.
        fs_hz: Sampling rate in hertz.
        resolution_hz: Target bin spacing; the segment is the nearest power of
            two, capped at the record length.
        axis: Time axis.

    Returns:
        ``(freqs_hz, psd)`` where ``psd`` is in ``units**2 / Hz`` and integrates
        (over 0 to Nyquist) to the signal's variance.
    """
    nperseg = min(segment_length(fs_hz, resolution_hz), x.shape[axis])
    freqs, psd = signal.welch(
        x,
        fs=fs_hz,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="density",
        axis=axis,
    )
    return np.asarray(freqs, dtype=np.float64), np.asarray(psd, dtype=np.float64)


def band_power(freqs_hz: FloatArray, psd: FloatArray, low_hz: float, high_hz: float) -> FloatArray:
    """Integrate a PSD between two frequencies (trapezoidal rule).

    Args:
        freqs_hz: Frequency axis from :func:`welch_psd`.
        psd: PSD with frequency along the last axis.
        low_hz: Lower integration limit (inclusive).
        high_hz: Upper integration limit (inclusive).

    Returns:
        Power in ``units**2`` (the variance contributed by that band), with the
        frequency axis removed.
    """
    if low_hz >= high_hz:
        raise ValueError("low_hz must be below high_hz")
    in_band = (freqs_hz >= low_hz) & (freqs_hz <= high_hz)
    if np.count_nonzero(in_band) < 2:
        raise ValueError(f"fewer than two PSD bins between {low_hz} and {high_hz} Hz")
    return np.asarray(np.trapezoid(psd[..., in_band], freqs_hz[in_band], axis=-1), dtype=np.float64)


def loglog_slope(freqs_hz: FloatArray, psd: FloatArray, low_hz: float, high_hz: float) -> float:
    """Fit ``psd ~ f**slope`` between two frequencies by least squares in log-log space.

    A slope of -1 is 1/f (flicker) noise; 0 is white noise. Bins are resampled
    onto a logarithmic grid first, otherwise the many high-frequency bins of a
    linear FFT axis would dominate the fit.
    """
    in_band = (freqs_hz >= low_hz) & (freqs_hz <= high_hz) & (psd > 0)
    if np.count_nonzero(in_band) < 3:
        raise ValueError(f"too few PSD bins between {low_hz} and {high_hz} Hz to fit a slope")
    log_f = np.log10(freqs_hz[in_band])
    log_p = np.log10(psd[in_band])
    grid = np.linspace(log_f[0], log_f[-1], 50)
    slope, _ = np.polyfit(grid, np.interp(grid, log_f, log_p), 1)
    return float(slope)
