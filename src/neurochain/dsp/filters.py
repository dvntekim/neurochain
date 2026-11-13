"""Band-splitting an extracellular recording into LFP and spike bands.

Two bands, two different engineering problems:

* **LFP**, below 300 Hz: slow, large, dominated by 1/f noise.
* **Spikes**, 300 Hz to 7.5 kHz: fast, small, thermal-noise limited.

Filters are Butterworth (maximally flat passband, so amplitudes inside the band
are not rippled) and stored as second-order sections, which stay numerically
stable where a single high-order transfer function would not. See
``docs/U0_design.md`` for the alternatives that were considered.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import signal

from neurochain.recording import FloatArray

LFP_CUTOFF_HZ = 300.0
SPIKE_BAND_HZ = (300.0, 7500.0)
DEFAULT_ORDER = 4

SosArray = NDArray[np.float64]


def _check_edges(edges_hz: tuple[float, ...], fs_hz: float) -> None:
    """Raise if any band edge is outside the open interval (0, Nyquist)."""
    nyquist_hz = fs_hz / 2.0
    for edge in edges_hz:
        if not 0.0 < edge < nyquist_hz:
            raise ValueError(
                f"band edge {edge} Hz must lie strictly between 0 and the Nyquist "
                f"frequency {nyquist_hz} Hz"
            )


def design_lowpass(cutoff_hz: float, fs_hz: float, order: int = DEFAULT_ORDER) -> SosArray:
    """Design a Butterworth low-pass filter as second-order sections.

    Args:
        cutoff_hz: -3 dB frequency in hertz.
        fs_hz: Sampling rate in hertz.
        order: Filter order; the roll-off is ``20 * order`` dB per decade.
    """
    _check_edges((cutoff_hz,), fs_hz)
    sos = signal.butter(order, cutoff_hz, btype="lowpass", fs=fs_hz, output="sos")
    return np.asarray(sos, dtype=np.float64)


def design_bandpass(
    low_hz: float, high_hz: float, fs_hz: float, order: int = DEFAULT_ORDER
) -> SosArray:
    """Design a Butterworth band-pass filter as second-order sections.

    ``order`` is the order of each skirt: the returned filter has ``2 * order``
    poles and rolls off at ``20 * order`` dB per decade on both sides.

    Args:
        low_hz: Lower -3 dB edge in hertz.
        high_hz: Upper -3 dB edge in hertz.
        fs_hz: Sampling rate in hertz.
        order: Order of each skirt.
    """
    if low_hz >= high_hz:
        raise ValueError(f"low edge {low_hz} Hz must be below high edge {high_hz} Hz")
    _check_edges((low_hz, high_hz), fs_hz)
    sos = signal.butter(order, (low_hz, high_hz), btype="bandpass", fs=fs_hz, output="sos")
    return np.asarray(sos, dtype=np.float64)


def apply_sos(
    sos: SosArray, x: FloatArray, *, zero_phase: bool = True, axis: int = -1
) -> FloatArray:
    """Filter ``x`` along ``axis``.

    Args:
        sos: Second-order sections from one of the ``design_*`` functions.
        x: Input samples.
        zero_phase: If true, run the filter forward then backward
            (``sosfiltfilt``). This cancels the phase response, so spike shapes
            and timings are preserved, and squares the magnitude response. It
            needs the whole record, so it is an offline analysis tool. If false,
            run the causal filter an implant would run.
        axis: Time axis.
    """
    if zero_phase:
        return np.asarray(signal.sosfiltfilt(sos, x, axis=axis), dtype=np.float64)
    return np.asarray(signal.sosfilt(sos, x, axis=axis), dtype=np.float64)


@dataclass(frozen=True)
class BandSplit:
    """The two bands of a recording, in the input's units and shape.

    Attributes:
        lfp: Low-pass output (below ``lfp_cutoff_hz``).
        spikes: Band-pass output (``spike_band_hz``).
        fs_hz: Sampling rate of both bands (no decimation is applied).
        lfp_cutoff_hz: LFP low-pass corner.
        spike_band_hz: Spike band edges.
    """

    lfp: FloatArray
    spikes: FloatArray
    fs_hz: float
    lfp_cutoff_hz: float
    spike_band_hz: tuple[float, float]


def band_split(
    x: FloatArray,
    fs_hz: float,
    *,
    lfp_cutoff_hz: float = LFP_CUTOFF_HZ,
    spike_band_hz: tuple[float, float] = SPIKE_BAND_HZ,
    order: int = DEFAULT_ORDER,
    zero_phase: bool = True,
    axis: int = -1,
) -> BandSplit:
    """Split a recording into its LFP band and its spike band.

    The mean along ``axis`` is removed first. A real front end removes the
    electrode's DC offset with AC coupling or a servo loop (U1); here it also
    keeps a large offset from ringing through the filter's start-up transient.

    Args:
        x: Samples, with time along ``axis``.
        fs_hz: Sampling rate in hertz.
        lfp_cutoff_hz: LFP low-pass corner.
        spike_band_hz: Lower and upper edges of the spike band.
        order: Butterworth order (per skirt for the band-pass).
        zero_phase: Forward-backward filtering; see :func:`apply_sos`.
        axis: Time axis.
    """
    centred = x - np.mean(x, axis=axis, keepdims=True)
    lfp_sos = design_lowpass(lfp_cutoff_hz, fs_hz, order)
    spike_sos = design_bandpass(spike_band_hz[0], spike_band_hz[1], fs_hz, order)
    return BandSplit(
        lfp=apply_sos(lfp_sos, centred, zero_phase=zero_phase, axis=axis),
        spikes=apply_sos(spike_sos, centred, zero_phase=zero_phase, axis=axis),
        fs_hz=fs_hz,
        lfp_cutoff_hz=lfp_cutoff_hz,
        spike_band_hz=spike_band_hz,
    )


def requantise(x: FloatArray, drop_bits: int) -> FloatArray:
    """Coarsen integer-valued samples by discarding ``drop_bits`` low bits.

    Models an ADC with fewer bits over the same input range: codes are floored
    to a multiple of ``2**drop_bits`` and re-centred on the middle of that step,
    so the quantisation error stays zero-mean. The output keeps the input's
    units (the original LSB).

    Args:
        x: Integer-valued samples in LSB.
        drop_bits: Number of least-significant bits to discard (0 = unchanged).
    """
    if drop_bits < 0:
        raise ValueError("drop_bits must be non-negative")
    if drop_bits == 0:
        return x.copy()
    step = float(1 << drop_bits)
    return np.asarray(np.floor(x / step) * step + (step - 1.0) / 2.0, dtype=np.float64)
