"""A capacitive-feedback low-noise amplifier (the Harrison and Charles topology).

The signal reaches the amplifier through a capacitor ``C1``. A small capacitor
``C2`` closes the feedback loop around an operational transconductance
amplifier (OTA), so the midband gain is the ratio ``C1 / C2``. A very large
resistance ``R2`` across ``C2`` sets the DC operating point and, with ``C2``,
the high-pass corner. The OTA's transconductance ``gm`` and its load ``CL`` set
the low-pass corner::

                       +----[ C2 ]----+
                       |              |
                       +----[ R2 ]----+
                       |              |
    pad ---[ C1 ]------+---[ -gm ]----+--- out
                       |              |
                     [ Cin ]        [ CL ]
                       |              |
                      gnd            gnd

Because ``C1`` blocks DC, the electrode's offset never reaches the OTA. That is
the property the design exists for.

The OTA is behavioural: a transconductance with a finite output resistance and
one input-referred noise source. Its noise is not derived from transistor
geometry. It is set by two published figures of merit: the supply current and
the noise efficiency factor (NEF).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from neurochain.frontend.physics import four_kt, thermal_voltage_v


@dataclass(frozen=True)
class Lna:
    """Parameters of the capacitive-feedback amplifier.

    Attributes:
        c1_f: Input coupling capacitor.
        c2_f: Feedback capacitor. Midband gain is ``c1_f / c2_f``.
        c_in_f: OTA input (gate) capacitance at the summing node. An assumed
            value; it raises the OTA's noise gain to ``(C1 + C2 + Cin) / C1``.
        c_load_f: OTA load capacitance; with ``gm`` it sets the bandwidth.
        r2_ohm: Feedback resistance (a MOS pseudo-resistor in silicon).
        i_total_a: Total supply current.
        nef: Noise efficiency factor of the amplifier.
        kappa: Subthreshold gate coupling coefficient of the input transistors.
        dc_gain: Open-loop DC voltage gain of the OTA (assumed).
        v_supply_v: Supply voltage.
        headroom_v: Voltage the output must stay away from each rail.
    """

    c1_f: float = 20.0e-12
    c2_f: float = 200.0e-15
    c_in_f: float = 2.0e-12
    c_load_f: float = 13.0e-12
    r2_ohm: float = 8.0e11
    i_total_a: float = 10.0e-6
    nef: float = 4.0
    kappa: float = 0.7
    dc_gain: float = 1.0e4
    v_supply_v: float = 1.8
    headroom_v: float = 0.1

    @property
    def ideal_gain(self) -> float:
        """Midband gain with a perfect OTA and a zero-impedance source: ``C1 / C2``."""
        return self.c1_f / self.c2_f

    @property
    def power_w(self) -> float:
        """Supply power in watts."""
        return self.v_supply_v * self.i_total_a

    @property
    def output_swing_v(self) -> float:
        """Largest output excursion from mid-supply before the amplifier rails."""
        return self.v_supply_v / 2.0 - self.headroom_v

    @property
    def f_low_hz(self) -> float:
        """High-pass corner set by the feedback network: ``1 / (2 pi R2 C2)``."""
        return 1.0 / (2.0 * np.pi * self.r2_ohm * self.c2_f)

    @property
    def noise_gain(self) -> float:
        """Factor by which OTA noise grows when referred to the amplifier input."""
        return (self.c1_f + self.c2_f + self.c_in_f) / self.c1_f

    def gm_s(self, temperature_k: float) -> float:
        """Return the OTA transconductance in siemens.

        A current-mirror OTA splits the supply current into four branches, so
        each input transistor carries ``I_total / 4``. In weak inversion its
        transconductance is ``kappa * I_D / U_T``.
        """
        return self.kappa * (self.i_total_a / 4.0) / thermal_voltage_v(temperature_k)

    def r_out_ohm(self, temperature_k: float) -> float:
        """Return the OTA output resistance that gives ``dc_gain``."""
        return self.dc_gain / self.gm_s(temperature_k)

    def noise_density_v_per_rt_hz(self, temperature_k: float) -> float:
        """Return the amplifier's white input-referred noise density.

        Rearranging the NEF definition for a first-order roll-off gives
        ``e_n = NEF * sqrt(4kT * U_T / I_total)``: the noise density depends on
        the supply current and the NEF, and on nothing else.
        """
        return self.nef * float(
            np.sqrt(four_kt(temperature_k) * thermal_voltage_v(temperature_k) / self.i_total_a)
        )

    def ota_noise_resistance_ohm(self, temperature_k: float) -> float:
        """Return the resistor whose thermal noise equals the OTA's input noise.

        The amplifier-level density is divided by the noise gain to get the
        OTA's own, then expressed as ``R = e_n**2 / 4kT``. In the netlist this
        resistor sits in series with the OTA input, where no current flows, so
        it adds noise and nothing else.
        """
        e_ota = self.noise_density_v_per_rt_hz(temperature_k) / self.noise_gain
        return e_ota**2 / four_kt(temperature_k)

    def with_f_low(self, f_low_hz: float) -> Lna:
        """Return the amplifier with ``R2`` chosen for a new high-pass corner."""
        return replace(self, r2_ohm=1.0 / (2.0 * np.pi * f_low_hz * self.c2_f))


def total_noise_from_nef(
    nef: float, i_total_a: float, bandwidth_hz: float, temperature_k: float
) -> float:
    """Return the total input-referred RMS noise the NEF definition implies, in volts.

    ``V_ni = NEF * sqrt(pi * U_T * 4kT * BW / (2 * I_total))``. "Total" means
    integrated over all frequencies through a first-order roll-off at
    ``bandwidth_hz``, whose noise bandwidth is ``pi/2`` times wider.
    """
    return nef * float(
        np.sqrt(
            np.pi
            * thermal_voltage_v(temperature_k)
            * four_kt(temperature_k)
            * bandwidth_hz
            / (2.0 * i_total_a)
        )
    )


def nef_from_noise(
    v_ni_rms_v: float, i_total_a: float, bandwidth_hz: float, temperature_k: float
) -> float:
    """Return the noise efficiency factor of an amplifier.

    ``NEF = V_ni * sqrt(2 * I_total / (pi * U_T * 4kT * BW))``. It compares the
    amplifier's noise with what an ideal single bipolar transistor would give at
    the same current. Lower is better. A differential pair in weak inversion
    cannot go below ``sqrt(2) / kappa``, about 2.
    """
    return v_ni_rms_v / total_noise_from_nef(1.0, i_total_a, bandwidth_hz, temperature_k)
