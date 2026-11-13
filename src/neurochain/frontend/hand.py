"""The U1 hand calculation, written as code so its numbers can be checked.

These are the formulas a designer would evaluate on paper before simulating,
with the simplifications a hand calculation makes:

* the charge-transfer resistance is treated as an open circuit;
* the electrode does not load the amplifier input;
* the amplifier's noise is white at the density the NEF implies, and the
  feedback resistor's noise is ignored.

The simulation makes none of these simplifications. The unit's done-criterion
is that the two agree within 10 %; ``docs/U1_design.md`` works the arithmetic
through by hand with the same symbols.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neurochain.frontend.electrode import Electrode
from neurochain.frontend.lna import Lna, total_noise_from_nef
from neurochain.frontend.physics import four_kt


@dataclass(frozen=True)
class HandCalculation:
    """Results of the hand calculation.

    Attributes:
        cpe_magnitude_1khz_ohm: ``|Z_CPE|`` at 1 kHz, ``1 / (Q w^n)``.
        resistor_term_ohm_hz: ``Rs * (f2 - f1)``: the access resistance's share
            of the integral of ``Re{Z}``.
        interface_term_ohm_hz: The double layer's share of that integral.
        electrode_v: Electrode thermal noise in the band, volts RMS.
        noise_density_v_per_rt_hz: Amplifier white noise density.
        lna_v: Amplifier noise in the band, volts RMS.
        total_v: Root-sum-square of electrode and amplifier noise.
        lna_total_v: Amplifier noise over all frequencies (the NEF's ``V_ni``).
        power_w: Amplifier supply power.
    """

    cpe_magnitude_1khz_ohm: float
    resistor_term_ohm_hz: float
    interface_term_ohm_hz: float
    electrode_v: float
    noise_density_v_per_rt_hz: float
    lna_v: float
    total_v: float
    lna_total_v: float
    power_w: float


def interface_real_integral(electrode: Electrode, low_hz: float, high_hz: float) -> float:
    """Integrate the real part of the CPE impedance over a band, in ohm-hertz.

    ``Re{Z_CPE} = cos(n pi/2) / (Q (2 pi f)^n)``, a power law in ``f``, so

    ``integral = cos(n pi/2) / (Q (2 pi)^n) * (f2^(1-n) - f1^(1-n)) / (1 - n)``.

    For an ideal capacitor (``n = 1``) the cosine is zero and so is the result:
    a lossless element makes no thermal noise.
    """
    n = electrode.n
    if n == 1.0:
        return 0.0
    prefactor = np.cos(n * np.pi / 2.0) / (electrode.q_s_sn * (2.0 * np.pi) ** n)
    return float(prefactor * (high_hz ** (1.0 - n) - low_hz ** (1.0 - n)) / (1.0 - n))


def hand_calculation(
    electrode: Electrode,
    lna: Lna,
    temperature_k: float,
    *,
    low_hz: float = 300.0,
    high_hz: float = 7500.0,
    bandwidth_hz: float = 7500.0,
) -> HandCalculation:
    """Evaluate the closed-form noise budget.

    Args:
        electrode: The electrode.
        lna: The amplifier.
        temperature_k: Temperature of electrode and amplifier.
        low_hz: Lower edge of the band the noise is integrated over.
        high_hz: Upper edge of that band.
        bandwidth_hz: The amplifier's -3 dB bandwidth, for the all-frequency
            noise in the NEF definition.
    """
    kt4 = four_kt(temperature_k)
    resistor_term = electrode.rs_ohm * (high_hz - low_hz)
    interface_term = interface_real_integral(electrode, low_hz, high_hz)
    electrode_v = float(np.sqrt(kt4 * (resistor_term + interface_term)))
    density = lna.noise_density_v_per_rt_hz(temperature_k)
    lna_v = density * float(np.sqrt(high_hz - low_hz))
    return HandCalculation(
        cpe_magnitude_1khz_ohm=1.0 / (electrode.q_s_sn * (2.0 * np.pi * 1000.0) ** electrode.n),
        resistor_term_ohm_hz=resistor_term,
        interface_term_ohm_hz=interface_term,
        electrode_v=electrode_v,
        noise_density_v_per_rt_hz=density,
        lna_v=lna_v,
        total_v=float(np.hypot(electrode_v, lna_v)),
        lna_total_v=total_noise_from_nef(lna.nef, lna.i_total_a, bandwidth_hz, temperature_k),
        power_w=lna.power_w,
    )
