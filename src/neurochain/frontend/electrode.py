"""Electrode-tissue interface: impedance and thermal noise.

The model is the Randles circuit: a series (access) resistance ``Rs`` and,
after it, the interface itself, a charge-transfer resistance ``Rct`` in
parallel with the double layer::

    tissue --[ Rs ]--+--[ Rct ]--+-- pad
                     |           |
                     +--[ CPE ]--+

The textbook Randles circuit makes the double layer an ideal capacitor. Real
metal electrodes do not behave that way: their impedance phase sits well short
of -90 degrees. Measured spectra are therefore fitted with a constant-phase
element (CPE), ``Z = 1 / (Q (j w)^n)`` with ``0 < n <= 1``. ``n = 1`` is the
ideal capacitor, with ``Q`` its capacitance.

The difference matters for noise. An ideal capacitor is lossless and adds no
thermal noise, so the textbook circuit has only ``Rs`` to make noise with in
the spike band. A CPE has a real part, ``cos(n pi / 2) / (Q w^n)``, and that
real part is noisy.

Parameter values are the fits of Straka et al. (2018) to Blackrock platinum
Utah arrays (see ``docs/references.md``).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray
from scipy import optimize

from neurochain.frontend.physics import four_kt
from neurochain.recording import FloatArray

ComplexArray = NDArray[np.complex128]


def log_grid(low_hz: float, high_hz: float, points_per_decade: int = 400) -> FloatArray:
    """Return a logarithmic frequency grid that includes both end points."""
    if not 0.0 < low_hz < high_hz:
        raise ValueError("need 0 < low_hz < high_hz")
    n = max(2, int(np.ceil(np.log10(high_hz / low_hz) * points_per_decade)) + 1)
    return np.geomspace(low_hz, high_hz, n)


def integrate_psd(psd: FloatArray, freqs_hz: FloatArray) -> float:
    """Integrate a one-sided PSD over frequency and return the RMS (square root)."""
    return float(np.sqrt(np.trapezoid(psd, freqs_hz)))


@dataclass(frozen=True)
class Electrode:
    """A Randles electrode with a constant-phase double layer.

    Attributes:
        rs_ohm: Series (access) resistance: electrolyte and tissue between the
            electrode surface and the bulk.
        rct_ohm: Charge-transfer resistance: the path for direct (Faradaic)
            current across the interface.
        q_s_sn: CPE magnitude ``Q`` in S·sⁿ. Equal to the double-layer
            capacitance in farads when ``n`` is 1.
        n: CPE exponent. 1 is an ideal capacitor; smaller is more resistive.
        name: Label used in figures and netlists.
    """

    rs_ohm: float
    rct_ohm: float
    q_s_sn: float
    n: float = 1.0
    name: str = "electrode"

    def __post_init__(self) -> None:
        """Validate parameter ranges."""
        if min(self.rs_ohm, self.rct_ohm, self.q_s_sn) <= 0:
            raise ValueError("rs_ohm, rct_ohm and q_s_sn must be positive")
        if not 0.0 < self.n <= 1.0:
            raise ValueError("the CPE exponent n must satisfy 0 < n <= 1")

    @property
    def is_ideal_capacitor(self) -> bool:
        """True when the double layer is an ideal capacitor (``n`` equals 1)."""
        return self.n == 1.0

    def cpe_impedance(self, freqs_hz: FloatArray) -> ComplexArray:
        """Return the double layer's impedance ``1 / (Q (j w)^n)`` in ohms."""
        omega = 2.0 * np.pi * np.asarray(freqs_hz, dtype=np.float64)
        return np.asarray(1.0 / (self.q_s_sn * (1j * omega) ** self.n), dtype=np.complex128)

    def impedance(self, freqs_hz: FloatArray) -> ComplexArray:
        """Return the electrode impedance ``Rs + (Rct || Z_CPE)`` in ohms."""
        z_cpe = self.cpe_impedance(freqs_hz)
        interface = self.rct_ohm * z_cpe / (self.rct_ohm + z_cpe)
        return np.asarray(self.rs_ohm + interface, dtype=np.complex128)

    def magnitude_ohm(self, freq_hz: float) -> float:
        """Return ``|Z|`` at one frequency."""
        return float(np.abs(self.impedance(np.array([freq_hz])))[0])

    def phase_deg(self, freq_hz: float) -> float:
        """Return the impedance phase in degrees at one frequency."""
        return float(np.degrees(np.angle(self.impedance(np.array([freq_hz]))))[0])

    def thermal_noise_psd(self, freqs_hz: FloatArray, temperature_k: float) -> FloatArray:
        """Return the open-circuit thermal noise density ``4kT Re{Z}`` in V²/Hz.

        Nyquist's theorem for any passive one-port: only the real part of the
        impedance generates noise.
        """
        return np.asarray(four_kt(temperature_k) * self.impedance(freqs_hz).real, dtype=np.float64)

    def thermal_noise_rms_v(self, low_hz: float, high_hz: float, temperature_k: float) -> float:
        """Return the thermal noise integrated between two frequencies, in volts RMS."""
        freqs = log_grid(low_hz, high_hz)
        return integrate_psd(self.thermal_noise_psd(freqs, temperature_k), freqs)

    def with_ideal_capacitor(self, match_hz: float = 1000.0) -> Electrode:
        """Return the textbook Randles electrode with the same ``|Z|`` at one frequency.

        ``Rs`` and ``Rct`` are kept. The CPE is replaced by the capacitor that
        gives the same impedance magnitude at ``match_hz``, which is how a
        single-frequency impedance measurement would be turned into a model.
        """
        target = self.magnitude_ohm(match_hz)

        def mismatch(log_c: float) -> float:
            trial = replace(self, q_s_sn=float(np.exp(log_c)), n=1.0)
            return trial.magnitude_ohm(match_hz) - target

        log_c = optimize.brentq(mismatch, np.log(1e-15), np.log(1e-3))
        return replace(
            self, q_s_sn=float(np.exp(log_c)), n=1.0, name=f"{self.name} (ideal capacitor)"
        )

    def with_interface_scaled(self, factor: float) -> Electrode:
        """Return the electrode with ``Q`` multiplied by ``factor``.

        Doubling ``Q`` is what doubling the electrode's surface area does to the
        double layer.
        """
        return replace(self, q_s_sn=self.q_s_sn * factor)


UTAH_PT_IN_VITRO = Electrode(
    rs_ohm=7.3e3, rct_ohm=0.46e10, q_s_sn=0.72e-8, n=0.74, name="Utah Pt, in saline"
)
"""Blackrock platinum Utah array in 0.15 M PBS: mean fit of 96 spectra (Straka 2018, Table 1)."""

UTAH_PT_IN_VIVO = Electrode(
    rs_ohm=25e3, rct_ohm=5.0e10, q_s_sn=11e-8, n=0.55, name="Utah Pt, implanted"
)
"""The same arrays two or more weeks after implantation in rat sciatic nerve (854 spectra)."""


@dataclass(frozen=True)
class CpeLadder:
    """A network of resistors and capacitors that behaves like a CPE.

    SPICE has no fractional element, so the CPE is replaced by series RC
    branches in parallel, plus one resistor and one capacitor that stand in for
    the branches below and above the covered range.

    Attributes:
        branch_r_ohm: Resistance of each series RC branch.
        branch_c_f: Capacitance of each series RC branch.
        r_low_ohm: Parallel resistor replacing all slower branches.
        c_high_f: Parallel capacitor replacing all faster branches.
    """

    branch_r_ohm: FloatArray
    branch_c_f: FloatArray
    r_low_ohm: float
    c_high_f: float

    def impedance(self, freqs_hz: FloatArray) -> ComplexArray:
        """Return the ladder's impedance in ohms."""
        s = 2j * np.pi * np.asarray(freqs_hz, dtype=np.float64)[:, np.newaxis]
        branches = s * self.branch_c_f / (1.0 + s * self.branch_r_ohm * self.branch_c_f)
        admittance = branches.sum(axis=1) + 1.0 / self.r_low_ohm + s[:, 0] * self.c_high_f
        return np.asarray(1.0 / admittance, dtype=np.complex128)


def cpe_ladder(
    q_s_sn: float,
    n: float,
    *,
    low_hz: float = 1.0e-3,
    high_hz: float = 1.0e7,
    per_decade: int = 2,
) -> CpeLadder:
    """Synthesise an RC ladder whose admittance approximates ``Q (j w)^n``.

    The fractional power has an exact integral form,

    ``Q s^n = Q sin(n pi)/pi * integral_0^inf x^(n-1) * s/(s + x) dx``,

    and ``g * s/(s + x)`` is the admittance of a resistor ``1/g`` in series with
    a capacitor ``g/x``. Sampling the integral on a logarithmic grid of ``x``
    therefore gives one series RC branch per sample. The two tails of the
    integral are summed in closed form: slow branches look like a resistor,
    fast ones like a capacitor.

    With two branches per decade the error is about 0.1 % across 1 Hz to
    100 kHz (checked in ``tests/test_electrode.py``).

    Args:
        q_s_sn: CPE magnitude ``Q``.
        n: CPE exponent, strictly between 0 and 1.
        low_hz: Corner frequency of the slowest branch.
        high_hz: Corner frequency of the fastest branch.
        per_decade: Branches per decade of frequency.

    Raises:
        ValueError: If ``n`` is not strictly between 0 and 1.
    """
    if not 0.0 < n < 1.0:
        raise ValueError("a ladder is only needed, and only defined, for 0 < n < 1")
    ratio = 10.0 ** (1.0 / per_decade)
    count = round(np.log10(high_hz / low_hz) * per_decade) + 1
    x = 2.0 * np.pi * low_hz * ratio ** np.arange(count)
    weight = q_s_sn * np.sin(n * np.pi) / np.pi
    branch_r = 1.0 / (weight * x**n * np.log(ratio))
    x_low = x[0] / np.sqrt(ratio)
    x_high = x[-1] * np.sqrt(ratio)
    return CpeLadder(
        branch_r_ohm=branch_r,
        branch_c_f=1.0 / (branch_r * x),
        r_low_ohm=float(n / (weight * x_low**n)),
        c_high_f=float(weight * x_high ** (n - 1.0) / (1.0 - n)),
    )
