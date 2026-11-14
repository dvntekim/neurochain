"""Integer-only versions of the feature paths, written the way the RTL will be.

Floating point hides two things hardware cannot: every multiply has a finite
width, and every result has to be rounded somewhere. These functions make both
choices explicit, using only integer adds, multiplies and shifts, so that the
SystemVerilog in U7 can be checked against them bit for bit.

Conventions:

* Samples are signed ADC codes. Before filtering they are shifted left by
  ``guard_bits`` so the filter's rounding error sits below one code.
* Coefficients are integers scaled by ``2**frac_bits``.
* A product is brought back to scale by adding half a step and shifting right
  (round half up). Python's ``>>`` on a negative integer floors, exactly as
  Verilog's ``>>>`` does on a signed value, so the two agree bit for bit.
* Plain truncation (no half step) is available for comparison. It is cheaper by
  one adder and wrong in a way worth seeing: each truncation is biased by half a
  step, and a feedback loop with poles near ``z = 1`` multiplies that bias by
  hundreds.
* The high-pass numerator is ``b0 * (1 - z**-1)**2`` exactly, so coefficient
  rounding cannot let a constant input leak through.
* What remains at DC is the recursion's deadband. With rounding, a silent input
  can leave the output resting up to ``loop_dc_gain / 2`` internal steps from
  zero: 132 at 30 kS/s, which is 2 ADC codes with six guard bits. Any real
  signal has enough noise to dither this away (the measured offset is under
  0.01 code), but a test on a constant input will see it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import signal

from neurochain.adc import IntCodes
from neurochain.recording import FloatArray, IntArray


@dataclass(frozen=True)
class FixedHighpass:
    """A second-order high-pass with integer coefficients.

    Implements
    ``y[n] = (b0*(x[n] - 2x[n-1] + x[n-2]) - a1*y[n-1] - a2*y[n-2] + r) >> frac_bits``
    with ``r = 2**(frac_bits - 1)`` when rounding and 0 when truncating.

    Attributes:
        b0: Numerator gain, scaled by ``2**frac_bits``.
        a1: First feedback coefficient, scaled.
        a2: Second feedback coefficient, scaled.
        frac_bits: Number of fractional bits in the coefficients.
        guard_bits: Bits the input is shifted left by before filtering.
        rounding: Round each product to nearest (true) or truncate (false).
    """

    b0: int
    a1: int
    a2: int
    frac_bits: int
    guard_bits: int = 4
    rounding: bool = True

    @property
    def loop_dc_gain(self) -> float:
        """Gain from a constant error inside the feedback loop to the output.

        ``1 / (1 + a1 + a2)``. It is large when the corner is far below the
        sample rate, and it is what turns a half-step truncation bias into a
        visible offset.
        """
        scale = 1 << self.frac_bits
        return scale / (scale + self.a1 + self.a2)

    @property
    def coefficient_bits(self) -> int:
        """Signed width needed to hold the largest coefficient."""
        return max(abs(self.b0), abs(self.a1), abs(self.a2)).bit_length() + 1

    def float_coefficients(self) -> tuple[FloatArray, FloatArray]:
        """Return ``(b, a)`` as the floating-point filter these integers represent."""
        scale = float(1 << self.frac_bits)
        b = np.array([self.b0, -2 * self.b0, self.b0]) / scale
        a = np.array([scale, self.a1, self.a2]) / scale
        return b, a

    @property
    def is_stable(self) -> bool:
        """True if both poles of the quantised filter are inside the unit circle."""
        _, a = self.float_coefficients()
        return bool(np.all(np.abs(np.roots(a)) < 1.0))


def design_fixed_highpass(
    fs_hz: float,
    corner_hz: float = 300.0,
    *,
    frac_bits: int = 14,
    guard_bits: int = 4,
    rounding: bool = True,
) -> FixedHighpass:
    """Quantise a second-order Butterworth high-pass to integer coefficients.

    Args:
        fs_hz: Sample rate.
        corner_hz: -3 dB frequency.
        frac_bits: Fractional bits of the coefficients.
        guard_bits: Extra low bits carried through the filter.
        rounding: Round products to nearest instead of truncating.
    """
    b, a = signal.butter(2, corner_hz, btype="highpass", fs=fs_hz)
    scale = 1 << frac_bits
    return FixedHighpass(
        b0=round(float(b[0]) * scale),
        a1=round(float(a[1]) * scale),
        a2=round(float(a[2]) * scale),
        frac_bits=frac_bits,
        guard_bits=guard_bits,
        rounding=rounding,
    )


def fixed_highpass(codes: IntCodes, filt: FixedHighpass) -> tuple[IntCodes, int]:
    """Filter ADC codes with integer arithmetic only.

    Args:
        codes: Signed ADC codes, 1-D.
        filt: The quantised filter.

    Returns:
        ``(output, peak)``: the filtered codes at the input's scale (guard bits
        removed with round-half-up), and the largest magnitude the filter's
        internal accumulator reached, which sizes the hardware register.
    """
    b0, a1, a2, frac, guard = filt.b0, filt.a1, filt.a2, filt.frac_bits, filt.guard_bits
    half = (1 << (guard - 1)) if guard else 0
    round_term = (1 << (frac - 1)) if filt.rounding else 0
    out = np.empty(codes.size, dtype=np.int64)
    x1 = x2 = y1 = y2 = 0
    peak = 0
    for i, sample_code in enumerate(codes.tolist()):
        x = sample_code << guard
        acc = b0 * (x - 2 * x1 + x2) - a1 * y1 - a2 * y2
        if acc > peak:
            peak = acc
        elif -acc > peak:
            peak = -acc
        y = (acc + round_term) >> frac
        x2, x1 = x1, x
        y2, y1 = y1, y
        out[i] = (y + half) >> guard
    return out, peak


def accumulator_bits(peak: int) -> int:
    """Return the signed register width that holds magnitudes up to ``peak``."""
    return int(peak).bit_length() + 1


def fixed_threshold_events(filtered: IntCodes, threshold: int, refractory: int) -> IntArray:
    """Report samples where the signal first falls below ``-threshold``.

    After each event a counter blocks further events for ``refractory``
    samples. There is no search for the trough: hardware reports the crossing
    as it happens.
    """
    below = filtered < -threshold
    onsets = np.flatnonzero(below[1:] & ~below[:-1]) + 1
    if below.size and below[0]:
        onsets = np.concatenate(([0], onsets))
    events: list[int] = []
    blocked_until = -1
    for onset in onsets.tolist():
        if onset > blocked_until:
            events.append(onset)
            blocked_until = onset + refractory
    return np.asarray(events, dtype=np.int64)


def fixed_neo_score(filtered: IntCodes, window: int) -> IntCodes:
    """Return the NEO summed over the last ``window`` samples, in integers.

    ``psi[n] = x[n-1]**2 - x[n-2] * x[n]`` is available one sample late, which
    is what a causal circuit can compute. The sum is not divided by the window:
    the threshold is scaled instead.
    """
    psi = np.zeros(filtered.size, dtype=np.int64)
    psi[2:] = filtered[1:-1] ** 2 - filtered[:-2] * filtered[2:]
    padded = np.concatenate((np.zeros(window, dtype=np.int64), np.cumsum(psi)))
    return np.asarray(padded[window:] - padded[:-window], dtype=np.int64)


def fixed_sbp(filtered: IntCodes, samples_per_bin: int) -> IntCodes:
    """Return the sum of absolute values in each bin: SBP without a divider."""
    n_bins = filtered.size // samples_per_bin
    trimmed = np.abs(filtered[: n_bins * samples_per_bin])
    return np.asarray(trimmed.reshape(n_bins, samples_per_bin).sum(axis=1), dtype=np.int64)


def quantisation_snr_db(reference: FloatArray, fixed: NDArray[Any]) -> float:
    """Return how far a fixed-point output is from its floating-point reference, in dB."""
    error = np.asarray(fixed, dtype=np.float64) - reference
    return float(10.0 * np.log10(np.mean(reference**2) / max(float(np.mean(error**2)), 1e-300)))
