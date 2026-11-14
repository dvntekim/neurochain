"""Causal spike detectors: amplitude threshold and nonlinear energy operator.

Both detectors work on a high-passed signal, turn it into a non-negative
"score", and report an event where the score crosses a fixed threshold. They
differ only in the score:

* **Threshold:** the score is ``-x``. A spike is a large negative deflection.
* **NEO:** the score is ``x[n]**2 - x[n-1] * x[n+1]``, smoothed. It is large
  when the signal is both big and fast, which is what a spike is and what slow
  noise is not (Mukhopadhyay and Ray, 1998).

The threshold is set once, from a calibration segment, and then held. An
implant cannot re-estimate its noise from data it has not seen yet.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import signal

from neurochain.dsp.spikes import mad_sigma
from neurochain.recording import FloatArray, IntArray

Method = Literal["threshold", "neo"]
HIGHPASS_HZ = 300.0
NEO_SMOOTH_S = 0.2e-3


def highpass_sos(fs_hz: float, corner_hz: float = HIGHPASS_HZ, order: int = 2) -> FloatArray:
    """Design a Butterworth high-pass as second-order sections.

    Raises:
        ValueError: If the corner is not below the Nyquist frequency.
    """
    if corner_hz >= fs_hz / 2.0:
        raise ValueError(f"corner {corner_hz} Hz is not below the Nyquist frequency {fs_hz / 2} Hz")
    return np.asarray(
        signal.butter(order, corner_hz, btype="highpass", fs=fs_hz, output="sos"), dtype=np.float64
    )


def causal_highpass(x: FloatArray, fs_hz: float, corner_hz: float = HIGHPASS_HZ) -> FloatArray:
    """High-pass a signal forward in time, starting from rest."""
    return np.asarray(signal.sosfilt(highpass_sos(fs_hz, corner_hz), x), dtype=np.float64)


def neo(x: FloatArray) -> FloatArray:
    """Return the nonlinear energy operator ``x[n]**2 - x[n-1] * x[n+1]``.

    For a sinusoid of amplitude ``A`` and angular frequency ``w`` per sample it
    equals ``A**2 * sin(w)**2``: amplitude squared times (roughly) frequency
    squared. The first and last samples are set to zero.
    """
    out = np.zeros_like(x, dtype=np.float64)
    out[1:-1] = x[1:-1] ** 2 - x[:-2] * x[2:]
    return out


def moving_sum(x: FloatArray, window: int) -> FloatArray:
    """Return the causal sum of the last ``window`` samples."""
    if window <= 1:
        return np.asarray(x, dtype=np.float64).copy()
    padded = np.concatenate((np.zeros(window), np.cumsum(x, dtype=np.float64)))
    return np.asarray(padded[window:] - padded[:-window], dtype=np.float64)


def neo_score(x: FloatArray, fs_hz: float, smooth_s: float = NEO_SMOOTH_S) -> FloatArray:
    """Return the smoothed NEO of a high-passed signal."""
    window = max(1, round(smooth_s * fs_hz))
    return moving_sum(neo(x), window) / window


def score_signal(x: FloatArray, fs_hz: float, method: Method) -> FloatArray:
    """Turn a high-passed signal into the detector's score."""
    return np.asarray(-x, dtype=np.float64) if method == "threshold" else neo_score(x, fs_hz)


def calibrate_threshold(score: FloatArray, method: Method, factor: float) -> float:
    """Set a detector threshold from a calibration segment of its score.

    Args:
        score: The score over the calibration segment.
        method: ``"threshold"``: ``factor`` times the noise sigma (median
            absolute deviation). ``"neo"``: ``factor`` times the mean score,
            the usual rule for this operator.
        factor: 4.5 is conventional for the threshold, 8 for the NEO.
    """
    if method == "threshold":
        return factor * mad_sigma(score)
    return factor * float(np.mean(score))


def detect_events(
    score: FloatArray,
    threshold: float,
    fs_hz: float,
    *,
    refractory_s: float = 1.0e-3,
    align_window_s: float = 0.5e-3,
) -> IntArray:
    """Report events where a score rises through a threshold.

    Each upward crossing opens a short window; the event is placed at the
    largest score inside it. Crossings are then ignored for ``refractory_s``.

    Returns:
        Sample indices of the events.
    """
    above = score > threshold
    onsets = np.flatnonzero(above[1:] & ~above[:-1]) + 1
    if above.size and above[0]:
        onsets = np.concatenate(([0], onsets))
    refractory = max(1, round(refractory_s * fs_hz))
    window = max(1, round(align_window_s * fs_hz))
    events: list[int] = []
    blocked_until = -1
    for onset in onsets:
        if onset <= blocked_until:
            continue
        peak = int(onset + np.argmax(score[onset : onset + window]))
        events.append(peak)
        blocked_until = peak + refractory
    return np.asarray(events, dtype=np.int64)


def detect(
    x: FloatArray,
    fs_hz: float,
    method: Method,
    *,
    factor: float,
    calibration_samples: int,
    highpass_hz: float | None = HIGHPASS_HZ,
) -> tuple[IntArray, float]:
    """Run a detector end to end on one channel.

    Args:
        x: The digitised signal (any linear unit).
        fs_hz: Its sample rate.
        method: ``"threshold"`` or ``"neo"``.
        factor: Threshold factor (see :func:`calibrate_threshold`).
        calibration_samples: Leading samples used to set the threshold.
        highpass_hz: Corner of the digital high-pass, or ``None`` if the signal
            is already band-limited.

    Returns:
        ``(event_samples, threshold)``.
    """
    filtered = causal_highpass(x, fs_hz, highpass_hz) if highpass_hz is not None else x
    score = score_signal(filtered, fs_hz, method)
    threshold = calibrate_threshold(score[:calibration_samples], method, factor)
    return detect_events(score, threshold, fs_hz), threshold
