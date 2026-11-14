"""A sampling, quantising ADC, described from the electrode's point of view.

The converter is specified by three numbers: its sample rate, its resolution in
bits, and its input range referred to the electrode (the amplifier chain's gain
is folded into that range). Its step is then ``2 * range / 2**bits``.

Sampling here means keeping every k-th sample of the emulated amplifier output
with no digital filter in front, because that is what an ADC does: whatever the
analog chain passed above the new Nyquist frequency folds back into the band.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from neurochain.recording import FloatArray

IntCodes = NDArray[np.int64]


@dataclass(frozen=True)
class Adc:
    """An analog-to-digital converter.

    Attributes:
        fs_hz: Sample rate.
        bits: Resolution.
        input_range_uv: Largest input magnitude, referred to the electrode, that
            converts without clipping. The default, 1 mV, is about what U0
            implies for the Neuralink ADC: its range is 146 times the spike-band
            noise, and in-vivo noise is about 13 µV.
    """

    fs_hz: float = 30_000.0
    bits: int = 10
    input_range_uv: float = 1000.0

    def __post_init__(self) -> None:
        """Validate the resolution and range."""
        if not 1 <= self.bits <= 24:
            raise ValueError("bits must be between 1 and 24")
        if self.input_range_uv <= 0 or self.fs_hz <= 0:
            raise ValueError("input_range_uv and fs_hz must be positive")

    @property
    def lsb_uv(self) -> float:
        """Step size referred to the electrode, in microvolts."""
        return 2.0 * self.input_range_uv / (1 << self.bits)

    @property
    def code_min(self) -> int:
        """Most negative output code."""
        return -(1 << (self.bits - 1))

    @property
    def code_max(self) -> int:
        """Most positive output code."""
        return (1 << (self.bits - 1)) - 1

    @property
    def bit_rate_bps(self) -> float:
        """Raw output data rate in bits per second."""
        return self.fs_hz * self.bits


@dataclass(frozen=True)
class AdcOutput:
    """The result of a conversion.

    Attributes:
        codes: Signed integer output codes.
        fs_hz: Sample rate of the codes.
        lsb_uv: Step size referred to the electrode.
        clipped_fraction: Fraction of samples that hit either end of the range.
    """

    codes: IntCodes
    fs_hz: float
    lsb_uv: float
    clipped_fraction: float

    @property
    def microvolts(self) -> FloatArray:
        """The codes converted back to microvolts at the electrode."""
        return np.asarray(self.codes * self.lsb_uv, dtype=np.float64)


def decimation_factor(fs_in_hz: float, fs_out_hz: float) -> int:
    """Return the integer ratio between two sample rates.

    Raises:
        ValueError: If the ratio is not a whole number of at least 1.
    """
    ratio = fs_in_hz / fs_out_hz
    factor = round(ratio)
    if factor < 1 or abs(ratio - factor) > 1e-9:
        raise ValueError(f"{fs_in_hz} Hz is not a whole multiple of {fs_out_hz} Hz")
    return factor


def sample(x: FloatArray, fs_in_hz: float, fs_out_hz: float) -> FloatArray:
    """Keep every k-th sample along the last axis, with no filtering."""
    return np.asarray(x[..., :: decimation_factor(fs_in_hz, fs_out_hz)], dtype=np.float64)


def quantise(x_uv: FloatArray, adc: Adc) -> tuple[IntCodes, float]:
    """Round to the nearest code and clip to the converter's range.

    Returns:
        ``(codes, clipped_fraction)``.
    """
    raw = np.rint(np.asarray(x_uv, dtype=np.float64) / adc.lsb_uv)
    clipped = float(np.mean((raw < adc.code_min) | (raw > adc.code_max)))
    return np.clip(raw, adc.code_min, adc.code_max).astype(np.int64), clipped


def convert(x_uv: FloatArray, fs_in_hz: float, adc: Adc) -> AdcOutput:
    """Sample and quantise a signal given in microvolts at the electrode.

    Args:
        x_uv: The amplifier output referred to the electrode, in microvolts.
        fs_in_hz: Sample rate of ``x_uv``; must be a whole multiple of the ADC's.
        adc: The converter.
    """
    codes, clipped = quantise(sample(x_uv, fs_in_hz, adc.fs_hz), adc)
    return AdcOutput(codes=codes, fs_hz=adc.fs_hz, lsb_uv=adc.lsb_uv, clipped_fraction=clipped)
