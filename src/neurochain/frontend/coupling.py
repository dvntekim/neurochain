"""AC versus DC coupling: how much electrode offset each survives, and the cost.

A metal electrode in tissue sits at an electrochemical potential tens to
hundreds of millivolts away from the reference, and it drifts. The signal is
microvolts. The front end has to do something about the difference.

* **AC coupling** (a series capacitor) blocks the offset whatever its size. The
  price is a high-pass corner: slow signals are lost, and after any disturbance
  the amplifier takes several time constants to come back.
* **DC coupling** passes everything, offset included. The offset is then
  multiplied by the gain, so either the gain stays small or the output rails.
  A small gain means the ADC has to resolve microvolts on top of hundreds of
  millivolts, which costs bits.
"""

from __future__ import annotations

import numpy as np


def dc_offset_tolerance_v(gain: float, output_swing_v: float) -> float:
    """Return the largest input offset a DC-coupled amplifier passes without railing.

    The output is ``gain * offset``, and it rails at ``output_swing_v``.
    An AC-coupled amplifier has no such limit in steady state.
    """
    return output_swing_v / gain


def max_gain_for_offset(offset_v: float, output_swing_v: float) -> float:
    """Return the largest DC-coupled gain that keeps a given offset off the rails."""
    return output_swing_v / offset_v


def adc_bits_required(input_range_v: float, noise_rms_v: float) -> float:
    """Return the ADC resolution needed to cover a range with steps of one noise sigma.

    ``bits = log2(2 * range / noise)``, for an input spanning plus and minus
    ``input_range_v``. U0 found that a step near the noise sigma costs almost
    nothing in detection, so this is the resolution worth paying for.
    """
    return float(np.log2(2.0 * input_range_v / noise_rms_v))


def highpass_time_constant_s(f_low_hz: float) -> float:
    """Return the time constant of a first-order high-pass: ``1 / (2 pi f)``."""
    return 1.0 / (2.0 * np.pi * f_low_hz)


def recovery_time_s(step_v: float, residual_v: float, f_low_hz: float) -> float:
    """Return how long an AC-coupled input takes to forget a step.

    A step of ``step_v`` at the input decays as ``exp(-t / tau)``. The time for
    it to fall to ``residual_v`` is ``tau * ln(step / residual)``.
    """
    if step_v <= residual_v:
        return 0.0
    return highpass_time_constant_s(f_low_hz) * float(np.log(step_v / residual_v))


def saturation_time_s(step_v: float, gain: float, output_swing_v: float, f_low_hz: float) -> float:
    """Return how long the output stays on the rail after an input step.

    The output is ``gain * step * exp(-t / tau)`` and leaves the rail when that
    falls to ``output_swing_v``. Zero if the step never reaches the rail.
    """
    return recovery_time_s(gain * step_v, output_swing_v, f_low_hz)
