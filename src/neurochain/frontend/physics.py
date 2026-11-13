"""Physical constants and the two thermal quantities every noise formula uses."""

from __future__ import annotations

BOLTZMANN_J_PER_K = 1.380649e-23
ELEMENTARY_CHARGE_C = 1.602176634e-19

BODY_TEMPERATURE_K = 310.15
"""37 °C: the temperature of an implanted electrode and amplifier."""

ROOM_TEMPERATURE_K = 300.0
"""27 °C: the temperature bench measurements are usually quoted at."""


def four_kt(temperature_k: float) -> float:
    """Return ``4kT`` in joules: the thermal noise power density per ohm.

    A resistor ``R`` has a voltage noise density of ``4kT * R`` in V²/Hz.
    """
    return 4.0 * BOLTZMANN_J_PER_K * temperature_k


def thermal_voltage_v(temperature_k: float) -> float:
    """Return the thermal voltage ``U_T = kT/q`` in volts (about 26.7 mV at 37 °C)."""
    return BOLTZMANN_J_PER_K * temperature_k / ELEMENTARY_CHARGE_C
