"""Electrode plus amplifier solved as one linear network.

Two node equations describe the circuit in :mod:`neurochain.frontend.lna` with
the electrode in series with ``C1``. With ``Y1 = 1 / (Ze + 1/(s C1))``,
``Y2 = s C2 + 1/R2``, ``Yin = s Cin`` and ``YL = 1/Ro + s CL``::

    summing node:  (Vs - Vx) Y1 + (Vo - Vx) Y2 - Vx Yin + i2 = 0
    output node:   -gm (Vx + vn) - Vo YL - (Vo - Vx) Y2 - i2 = 0

``Vs`` is the signal in the tissue, ``vn`` the OTA's input noise voltage and
``i2`` the noise current of ``R2``. Solving for ``Vo`` gives three transfer
functions, one per source. Electrode thermal noise is in series with ``Ze``, so
it shares the signal's transfer function.

The same circuit is simulated in ngspice by :mod:`neurochain.frontend.spice`;
``tests/test_frontend.py`` holds the two to within a fraction of a percent.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy import optimize

from neurochain.frontend.electrode import (
    UTAH_PT_IN_VITRO,
    ComplexArray,
    Electrode,
    integrate_psd,
    log_grid,
)
from neurochain.frontend.lna import Lna, nef_from_noise
from neurochain.frontend.physics import BODY_TEMPERATURE_K, four_kt
from neurochain.recording import FloatArray

SPIKE_BAND_HZ = (300.0, 7500.0)
_RESPONSE_GRID_HZ = (1.0e-4, 1.0e9)


@dataclass(frozen=True)
class NoiseSpectrum:
    """Noise power spectral densities by source, in V²/Hz.

    Attributes:
        freqs_hz: Frequency axis.
        electrode: Electrode thermal noise.
        ota: OTA input noise.
        feedback: Thermal noise of the feedback resistance ``R2``.
    """

    freqs_hz: FloatArray
    electrode: FloatArray
    ota: FloatArray
    feedback: FloatArray

    @property
    def lna(self) -> FloatArray:
        """Amplifier noise: OTA plus feedback resistance."""
        return self.ota + self.feedback

    @property
    def total(self) -> FloatArray:
        """All sources."""
        return self.electrode + self.ota + self.feedback


@dataclass(frozen=True)
class NoiseBudget:
    """Integrated noise by source, in volts RMS.

    Attributes:
        electrode_v: Electrode thermal noise.
        lna_v: Amplifier noise (OTA and feedback resistance).
        total_v: Root-sum-square of the two.
    """

    electrode_v: float
    lna_v: float
    total_v: float


@dataclass(frozen=True)
class FrontEnd:
    """An electrode connected to a capacitive-feedback amplifier.

    Attributes:
        lna: The amplifier.
        electrode: The electrode, or ``None`` for an ideal zero-impedance source.
        temperature_k: Temperature of every noisy element.
    """

    lna: Lna
    electrode: Electrode | None = UTAH_PT_IN_VITRO
    temperature_k: float = BODY_TEMPERATURE_K

    def source_impedance(self, freqs_hz: FloatArray) -> ComplexArray:
        """Return the electrode impedance, or zeros for an ideal source."""
        if self.electrode is None:
            return np.zeros(np.shape(freqs_hz), dtype=np.complex128)
        return self.electrode.impedance(freqs_hz)

    def transfer_functions(
        self, freqs_hz: FloatArray
    ) -> tuple[ComplexArray, ComplexArray, ComplexArray]:
        """Solve the network at each frequency.

        Returns:
            ``(signal, ota_noise, feedback_noise)``: output voltage per volt of
            tissue signal, per volt of OTA input noise, and per ampere of
            feedback-resistor noise current.
        """
        lna = self.lna
        s = 2j * np.pi * np.asarray(freqs_hz, dtype=np.float64)
        gm = lna.gm_s(self.temperature_k)
        y1 = 1.0 / (self.source_impedance(freqs_hz) + 1.0 / (s * lna.c1_f))
        y2 = s * lna.c2_f + 1.0 / lna.r2_ohm
        y_in = s * lna.c_in_f
        y_load = 1.0 / lna.r_out_ohm(self.temperature_k) + s * lna.c_load_f
        determinant = (y1 + y2 + y_in) * (y_load + y2) - y2 * (y2 - gm)
        signal = (y2 - gm) * y1 / determinant
        ota_noise = -(y1 + y2 + y_in) * gm / determinant
        feedback_noise = -(y1 + y_in + gm) / determinant
        return (
            np.asarray(signal, dtype=np.complex128),
            np.asarray(ota_noise, dtype=np.complex128),
            np.asarray(feedback_noise, dtype=np.complex128),
        )

    def signal_gain(self, freqs_hz: FloatArray) -> ComplexArray:
        """Return the complex gain from tissue signal to amplifier output."""
        return self.transfer_functions(freqs_hz)[0]

    def _gain_magnitude(self, freq_hz: float) -> float:
        return float(np.abs(self.signal_gain(np.array([freq_hz])))[0])

    def _peak(self) -> tuple[float, float]:
        """Return ``(frequency, gain)`` at the top of the pass band."""
        grid = log_grid(*_RESPONSE_GRID_HZ, points_per_decade=40)
        magnitude = np.abs(self.signal_gain(grid))
        index = int(np.argmax(magnitude))
        return float(grid[index]), float(magnitude[index])

    @property
    def midband_gain(self) -> float:
        """Largest gain magnitude from tissue signal to output."""
        return self._peak()[1]

    def _corner(self, *, upper: bool) -> float:
        """Find the -3 dB frequency above or below the pass band."""
        peak_hz, peak = self._peak()
        target = peak / np.sqrt(2.0)
        edge = _RESPONSE_GRID_HZ[1] if upper else _RESPONSE_GRID_HZ[0]
        return float(
            optimize.brentq(lambda f: self._gain_magnitude(f) - target, peak_hz, edge, rtol=1e-10)
        )

    @property
    def f_low_hz(self) -> float:
        """High-pass -3 dB frequency of the signal path."""
        return self._corner(upper=False)

    @property
    def f_high_hz(self) -> float:
        """Low-pass -3 dB frequency of the signal path."""
        return self._corner(upper=True)

    def output_noise(self, freqs_hz: FloatArray) -> NoiseSpectrum:
        """Return the noise density at the amplifier output, by source."""
        signal, ota_noise, feedback_noise = self.transfer_functions(freqs_hz)
        kt4 = four_kt(self.temperature_k)
        source_noise = kt4 * self.source_impedance(freqs_hz).real
        return NoiseSpectrum(
            freqs_hz=np.asarray(freqs_hz, dtype=np.float64),
            electrode=source_noise * np.abs(signal) ** 2,
            ota=kt4
            * self.lna.ota_noise_resistance_ohm(self.temperature_k)
            * np.abs(ota_noise) ** 2,
            feedback=kt4 / self.lna.r2_ohm * np.abs(feedback_noise) ** 2,
        )

    def input_noise(self, freqs_hz: FloatArray) -> NoiseSpectrum:
        """Return the noise density referred to the tissue signal, by source.

        Each output density is divided by the squared signal gain at the same
        frequency. This is what SPICE calls ``inoise``: the noise a perfect
        amplifier would need at its input to reproduce the real output.
        """
        out = self.output_noise(freqs_hz)
        gain_sq = np.abs(self.signal_gain(freqs_hz)) ** 2
        return NoiseSpectrum(
            out.freqs_hz, out.electrode / gain_sq, out.ota / gain_sq, out.feedback / gain_sq
        )

    def band_noise(
        self, low_hz: float = SPIKE_BAND_HZ[0], high_hz: float = SPIKE_BAND_HZ[1]
    ) -> NoiseBudget:
        """Return input-referred noise integrated over a band, by source."""
        freqs = log_grid(low_hz, high_hz)
        spectrum = self.input_noise(freqs)
        return NoiseBudget(
            electrode_v=integrate_psd(spectrum.electrode, freqs),
            lna_v=integrate_psd(spectrum.lna, freqs),
            total_v=integrate_psd(spectrum.total, freqs),
        )

    def without_electrode(self) -> FrontEnd:
        """Return the amplifier driven by an ideal source."""
        return replace(self, electrode=None)

    def amplifier_total_noise_v(self) -> float:
        """Return the OTA noise over all frequencies, referred to the input.

        Output noise integrated from far below to far above the pass band,
        divided by the midband gain, with an ideal source. This is the
        ``V_ni,rms`` of the NEF definition. Feedback-resistor noise is left out:
        it sits below the high-pass corner, outside any signal band.
        """
        bench = self.without_electrode()
        freqs = log_grid(*_RESPONSE_GRID_HZ, points_per_decade=200)
        return integrate_psd(bench.output_noise(freqs).ota, freqs) / bench.midband_gain

    def nef(self) -> float:
        """Return the NEF measured on the network, for comparison with ``lna.nef``."""
        bench = self.without_electrode()
        return nef_from_noise(
            self.amplifier_total_noise_v(),
            self.lna.i_total_a,
            bench.f_high_hz,
            self.temperature_k,
        )

    def with_bias_scaled(self, factor: float, *, keep_bandwidth: bool) -> FrontEnd:
        """Return the front end with the supply current multiplied by ``factor``.

        Args:
            factor: Multiplier on ``i_total_a``.
            keep_bandwidth: If true, the load capacitor is rescaled so the
                bandwidth stays where it was. If false the load is untouched
                and the bandwidth moves with ``gm``.
        """
        lna = replace(self.lna, i_total_a=self.lna.i_total_a * factor)
        if keep_bandwidth:
            target = self.without_electrode().f_high_hz
            lna = _solve_load(lna, target, self.temperature_k)
        return replace(self, lna=lna)


def _solve_load(lna: Lna, f_high_hz: float, temperature_k: float) -> Lna:
    """Choose the load capacitor that gives ``f_high_hz`` with an ideal source."""
    feedback_factor = lna.c2_f / (lna.c1_f + lna.c2_f + lna.c_in_f)
    estimate = lna.gm_s(temperature_k) * feedback_factor / (2.0 * np.pi * f_high_hz)

    def error(log_c: float) -> float:
        trial = FrontEnd(replace(lna, c_load_f=float(np.exp(log_c))), None, temperature_k)
        return trial.f_high_hz - f_high_hz

    log_c = optimize.brentq(error, np.log(estimate / 10.0), np.log(estimate * 10.0), rtol=1e-10)
    return replace(lna, c_load_f=float(np.exp(log_c)))


def design_lna(
    *,
    f_high_hz: float = 7500.0,
    f_low_hz: float = 1.0,
    i_total_a: float = 10.0e-6,
    nef: float = 4.0,
    temperature_k: float = BODY_TEMPERATURE_K,
    base: Lna | None = None,
) -> Lna:
    """Size the amplifier for a pass band, a supply current and an NEF.

    ``R2`` follows from the high-pass corner. The load capacitor is solved so
    that, driven by an ideal source, the amplifier's -3 dB bandwidth is
    ``f_high_hz``. An electrode in series lowers the bandwidth slightly.

    Args:
        f_high_hz: Low-pass corner with an ideal source.
        f_low_hz: High-pass corner.
        i_total_a: Total supply current.
        nef: Noise efficiency factor.
        temperature_k: Operating temperature.
        base: Amplifier to take the remaining parameters from.
    """
    lna = replace(base or Lna(), i_total_a=i_total_a, nef=nef).with_f_low(f_low_hz)
    return _solve_load(lna, f_high_hz, temperature_k)


def baseline_frontend() -> FrontEnd:
    """Return the project's baseline front end.

    A platinum Utah electrode in saline (Straka et al., 2018) into a Harrison
    and Charles amplifier (gain 100, NEF 4.0) biased at 10 µA from 1.8 V, with a
    1 Hz to 7.5 kHz pass band, at body temperature.
    """
    return FrontEnd(lna=design_lna(), electrode=UTAH_PT_IN_VITRO)
