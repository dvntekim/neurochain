"""Noise estimation and threshold spike detection on a spike-band signal."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats

from neurochain.recording import FloatArray, IntArray

GAUSSIAN_MAD_SCALE = 0.6744897501960817
"""Median of ``|z|`` for a standard normal ``z``: the 75th percentile of N(0, 1)."""

Polarity = Literal["neg", "pos"]


def mad_sigma(x: FloatArray) -> float:
    """Estimate the noise standard deviation from the median absolute deviation.

    ``sigma = median(|x - median(x)|) / 0.6745``. For Gaussian noise this equals
    the standard deviation. Unlike the RMS it barely moves when spikes are
    present, because spikes occupy a small fraction of samples and the median
    ignores outliers (Quiroga et al., 2004).
    """
    centred = np.abs(x - np.median(x))
    return float(np.median(centred) / GAUSSIAN_MAD_SCALE)


def noise_crossing_rate_hz(
    freqs_hz: FloatArray,
    psd: FloatArray,
    band_hz: tuple[float, float],
    threshold_sigma: float,
    fs_hz: float,
) -> float:
    """Predict how often Gaussian noise alone triggers a one-sided threshold detector.

    This is the detector's false-alarm floor: a detection rate close to it is
    mostly noise.

    A sampled detector fires when one sample is beyond the threshold and the
    previous one is not. For a stationary Gaussian sequence that probability is
    ``Q(k) - P(both beyond)``, where ``Q`` is the normal tail and the joint term
    is a bivariate normal with the noise's lag-one correlation ``rho``. The
    correlation follows from the spectrum (Wiener-Khinchin):
    ``rho = integral(S(f) cos(2 pi f / fs)) / integral(S(f))``.

    Rice's formula, ``f0 * exp(-k**2 / 2)``, is the continuous-time limit. It
    overestimates here by about a factor of two, because at ``k = 4.5`` a noise
    excursion past the threshold is shorter than one sample period and the
    sampler misses many of them.

    The spectrum is taken as ``psd`` inside ``band_hz`` and zero outside, a
    brick-wall approximation of the spike-band filter.

    Args:
        freqs_hz: Frequency axis of the PSD.
        psd: Power spectral density of the unfiltered signal.
        band_hz: Pass band of the spike filter.
        threshold_sigma: Threshold ``k`` in multiples of the noise sigma.
        fs_hz: Sampling rate in hertz.

    Returns:
        Expected detections per second, ignoring the detector's dead time.
    """
    in_band = (freqs_hz >= band_hz[0]) & (freqs_hz <= band_hz[1])
    f = freqs_hz[in_band]
    s = psd[in_band]
    rho = float(np.trapezoid(s * np.cos(2.0 * np.pi * f / fs_hz), f) / np.trapezoid(s, f))
    tail = float(stats.norm.sf(threshold_sigma))
    both = float(
        stats.multivariate_normal(mean=[0.0, 0.0], cov=[[1.0, rho], [rho, 1.0]]).cdf(
            [-threshold_sigma, -threshold_sigma]
        )
    )
    return fs_hz * (tail - both)


@dataclass(frozen=True)
class SpikeDetection:
    """Result of :func:`detect_spikes`.

    Attributes:
        indices: Sample index of each spike's peak (trough for ``neg``).
        amplitudes: Signed value of the signal at each peak.
        sigma: Noise estimate used.
        threshold: Signed threshold that was applied.
    """

    indices: IntArray
    amplitudes: FloatArray
    sigma: float
    threshold: float

    @property
    def count(self) -> int:
        """Number of detected spikes."""
        return int(self.indices.size)

    @property
    def snr(self) -> FloatArray:
        """Peak magnitude of each spike in multiples of the noise sigma."""
        return np.asarray(np.abs(self.amplitudes) / self.sigma, dtype=np.float64)


def detect_spikes(
    x: FloatArray,
    fs_hz: float,
    *,
    threshold_sigma: float = 4.5,
    polarity: Polarity = "neg",
    refractory_s: float = 1.0e-3,
    align_window_s: float = 0.5e-3,
    sigma: float | None = None,
) -> SpikeDetection:
    """Detect spikes as threshold crossings and align each to its peak.

    The threshold is ``threshold_sigma`` noise standard deviations from zero.
    Each crossing opens a short search window; the most extreme sample inside
    it is the spike's peak. Further crossings are ignored until ``refractory_s``
    after that peak, so one spike is counted once.

    Args:
        x: 1-D spike-band signal (already band-pass filtered, zero mean).
        fs_hz: Sampling rate in hertz.
        threshold_sigma: Threshold in multiples of the noise sigma.
        polarity: ``"neg"`` to detect negative-going spikes, ``"pos"`` for
            positive-going ones.
        refractory_s: Dead time after each detected peak.
        align_window_s: How far past the crossing to search for the peak.
        sigma: Noise estimate to use; defaults to :func:`mad_sigma` of ``x``.

    Returns:
        A :class:`SpikeDetection`.

    Raises:
        ValueError: If ``x`` is not 1-D or ``threshold_sigma`` is not positive.
    """
    if x.ndim != 1:
        raise ValueError("detect_spikes expects a 1-D signal")
    if threshold_sigma <= 0:
        raise ValueError("threshold_sigma must be positive")
    noise = mad_sigma(x) if sigma is None else float(sigma)
    sign = -1.0 if polarity == "neg" else 1.0
    threshold = sign * threshold_sigma * noise
    # Work on a copy flipped so that spikes are always positive peaks.
    flipped = sign * x
    above = flipped > threshold_sigma * noise
    onsets = np.flatnonzero(above[1:] & ~above[:-1]) + 1
    if above.size and above[0]:
        onsets = np.concatenate(([0], onsets))

    refractory = max(1, round(refractory_s * fs_hz))
    window = max(1, round(align_window_s * fs_hz))
    peaks: list[int] = []
    blocked_until = -1
    for onset in onsets:
        if onset <= blocked_until:
            continue
        segment = flipped[onset : onset + window]
        peak = int(onset + np.argmax(segment))
        peaks.append(peak)
        blocked_until = peak + refractory

    indices = np.asarray(peaks, dtype=np.int64)
    return SpikeDetection(
        indices=indices,
        amplitudes=np.asarray(x[indices], dtype=np.float64),
        sigma=noise,
        threshold=float(threshold),
    )


def polarity_census(
    primary: SpikeDetection, opposite: SpikeDetection, *, window: int
) -> tuple[int, int]:
    """Count distinct events and how many are dominated by the opposite polarity.

    A band-pass filtered spike has lobes of both signs, so a large negative
    spike often also crosses the positive threshold. Counting raw crossings of
    each sign would mistake those lobes for positive spikes. Here a detection of
    one sign and the nearest detection of the other sign within ``window``
    samples are treated as two phases of one event, and the event takes the
    polarity of its larger phase.

    Args:
        primary: Detections of the polarity under study.
        opposite: Detections of the other polarity on the same signal.
        window: Maximum separation, in samples, for two phases of one event.

    Returns:
        ``(n_events, n_opposite_dominant)``.
    """
    paired_opposite = np.zeros(opposite.count, dtype=bool)
    flipped = 0
    for index, amplitude in zip(primary.indices, primary.amplitudes, strict=True):
        if opposite.count == 0:
            break
        j = int(np.searchsorted(opposite.indices, index))
        candidates = [c for c in (j - 1, j) if 0 <= c < opposite.count and not paired_opposite[c]]
        if not candidates:
            continue
        nearest = min(candidates, key=lambda c: abs(int(opposite.indices[c]) - int(index)))
        if abs(int(opposite.indices[nearest]) - int(index)) > window:
            continue
        paired_opposite[nearest] = True
        if abs(float(opposite.amplitudes[nearest])) > abs(float(amplitude)):
            flipped += 1
    unpaired_opposite = int(np.count_nonzero(~paired_opposite))
    return primary.count + unpaired_opposite, unpaired_opposite + flipped


def extract_waveforms(
    x: FloatArray, indices: IntArray, *, pre: int, post: int
) -> tuple[FloatArray, IntArray]:
    """Cut a window of samples around each spike peak.

    Args:
        x: 1-D signal.
        indices: Peak sample indices.
        pre: Samples to keep before the peak.
        post: Samples to keep from the peak onward (the peak is column ``pre``).

    Returns:
        ``(waveforms, kept_indices)``. ``waveforms`` has shape
        ``(n_kept, pre + post)``; spikes too close to either end are dropped.
    """
    keep = (indices >= pre) & (indices + post <= x.size)
    kept = np.asarray(indices[keep], dtype=np.int64)
    if kept.size == 0:
        return np.empty((0, pre + post), dtype=np.float64), kept
    offsets = np.arange(-pre, post)
    return np.asarray(x[kept[:, np.newaxis] + offsets], dtype=np.float64), kept


def match_spikes(detected: IntArray, truth: IntArray, *, tolerance: int) -> tuple[int, int, int]:
    """Score detections against ground-truth spike samples.

    Each true spike can be claimed by at most one detection within
    ``tolerance`` samples (greedy, in time order).

    Returns:
        ``(true_positives, false_positives, false_negatives)``.
    """
    detected_sorted = np.sort(detected)
    truth_sorted = np.sort(truth)
    true_positives = 0
    j = 0
    for t in truth_sorted:
        while j < detected_sorted.size and detected_sorted[j] < t - tolerance:
            j += 1
        if j < detected_sorted.size and abs(int(detected_sorted[j]) - int(t)) <= tolerance:
            true_positives += 1
            j += 1
    return (
        true_positives,
        int(detected_sorted.size) - true_positives,
        int(truth_sorted.size) - true_positives,
    )
