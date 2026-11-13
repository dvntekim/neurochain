"""A small seeded extracellular recording with known spike times.

This is the test fixture for everything that must run without downloaded data
(unit tests, CI) and the place to try a prediction check before running it on
real recordings. It is deliberately simple: MEArec provides the biophysically
detailed version from U2 onward.

The signal is the sum of three parts:

* spikes: one biphasic template per unit, placed by a Poisson process with a
  refractory gap, scaled so the trough equals the unit's amplitude;
* background noise: white Gaussian, standing in for thermal plus distant-neuron
  noise in the spike band;
* LFP: Gaussian noise shaped to a ``1/f**exponent`` spectrum below a corner, a
  stand-in for the slow field potential.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from neurochain.recording import FloatArray, IntArray, Recording


@dataclass(frozen=True)
class GroundTruth:
    """Where the spikes were put.

    Attributes:
        spike_samples: One array per unit of trough sample indices, sorted.
        amplitudes: Trough depth of each unit, as a positive number in the
            recording's units.
        noise_rms: RMS of the white background noise.
        lfp_rms: RMS of the LFP component.
    """

    spike_samples: tuple[IntArray, ...]
    amplitudes: tuple[float, ...]
    noise_rms: float
    lfp_rms: float

    def all_spike_samples(self) -> IntArray:
        """Return every unit's spike samples merged and sorted."""
        if not self.spike_samples:
            return np.empty(0, dtype=np.int64)
        return np.sort(np.concatenate(self.spike_samples))


def spike_template(fs_hz: float, *, duration_s: float = 2.0e-3) -> tuple[FloatArray, int]:
    """Return a unit-trough biphasic spike template and the index of its trough.

    A narrow negative Gaussian (the sodium current sink) followed by a wider,
    shallower positive Gaussian (repolarisation), normalised so the minimum is
    exactly -1.

    Args:
        fs_hz: Sampling rate in hertz.
        duration_s: Template length in seconds.
    """
    n = max(8, round(duration_s * fs_hz))
    t = (np.arange(n) - n * 0.3) / fs_hz
    trough = -np.exp(-0.5 * (t / 0.12e-3) ** 2)
    rebound = 0.35 * np.exp(-0.5 * ((t - 0.40e-3) / 0.25e-3) ** 2)
    template = trough + rebound
    template -= np.linspace(template[0], template[-1], n)  # start and end at zero
    template /= -template.min()
    return template.astype(np.float64), int(np.argmin(template))


def colored_noise(
    n_samples: int,
    fs_hz: float,
    rng: np.random.Generator,
    *,
    exponent: float = 1.0,
    f_min_hz: float = 1.0,
    f_max_hz: float = 300.0,
) -> FloatArray:
    """Return unit-RMS Gaussian noise with a ``1/f**exponent`` power spectrum.

    The spectrum is flat below ``f_min_hz`` and zero above ``f_max_hz``. Shaping
    is done in the frequency domain, so the result is exactly band-limited.
    """
    spectrum = np.fft.rfft(rng.standard_normal(n_samples))
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs_hz)
    gain = np.zeros_like(freqs)
    in_band = (freqs > 0) & (freqs <= f_max_hz)
    gain[in_band] = np.maximum(freqs[in_band], f_min_hz) ** (-exponent / 2.0)
    shaped = np.fft.irfft(spectrum * gain, n=n_samples)
    rms = float(np.sqrt(np.mean(shaped**2)))
    return np.asarray(shaped / rms, dtype=np.float64)


def poisson_spike_samples(
    rate_hz: float,
    n_samples: int,
    fs_hz: float,
    rng: np.random.Generator,
    *,
    refractory_s: float = 2.0e-3,
    margin_samples: int = 0,
) -> IntArray:
    """Draw spike sample indices from a Poisson process with a refractory gap.

    Args:
        rate_hz: Mean firing rate before the refractory gap is enforced.
        n_samples: Length of the recording in samples.
        fs_hz: Sampling rate in hertz.
        rng: Random generator.
        refractory_s: Minimum gap between consecutive spikes of the unit.
        margin_samples: Keep spikes at least this far from both ends.
    """
    if rate_hz <= 0:
        return np.empty(0, dtype=np.int64)
    duration_s = n_samples / fs_hz
    n_draw = int(rate_hz * duration_s * 2 + 20)
    times = np.cumsum(rng.exponential(1.0 / rate_hz, size=n_draw) + refractory_s)
    samples = np.round(times * fs_hz).astype(np.int64)
    keep = (samples >= margin_samples) & (samples < n_samples - margin_samples)
    return np.asarray(samples[keep], dtype=np.int64)


def make_synthetic_recording(
    *,
    duration_s: float = 10.0,
    fs_hz: float = 20_000.0,
    amplitudes: Sequence[float] = (60.0, 100.0),
    rates_hz: Sequence[float] = (8.0, 5.0),
    noise_rms: float = 8.0,
    lfp_rms: float = 30.0,
    units: str = "µV",
    seed: int = 0,
) -> tuple[Recording, GroundTruth]:
    """Build a one-channel synthetic recording and its ground truth.

    Args:
        duration_s: Length in seconds.
        fs_hz: Sampling rate in hertz.
        amplitudes: Trough depth of each unit (positive, in ``units``).
        rates_hz: Firing rate of each unit; same length as ``amplitudes``.
        noise_rms: RMS of the white background noise.
        lfp_rms: RMS of the 1/f component below 300 Hz.
        units: Unit label carried by the recording.
        seed: Seed for ``numpy.random.default_rng``; the same seed gives the
            same recording bit for bit.

    Returns:
        ``(recording, ground_truth)``.

    Raises:
        ValueError: If ``amplitudes`` and ``rates_hz`` differ in length.
    """
    if len(amplitudes) != len(rates_hz):
        raise ValueError("amplitudes and rates_hz must have the same length")
    rng = np.random.default_rng(seed)
    n_samples = round(duration_s * fs_hz)
    template, trough_index = spike_template(fs_hz)
    signal = noise_rms * rng.standard_normal(n_samples)
    if lfp_rms > 0:
        signal += lfp_rms * colored_noise(n_samples, fs_hz, rng)

    spike_samples: list[IntArray] = []
    for amplitude, rate_hz in zip(amplitudes, rates_hz, strict=True):
        samples = poisson_spike_samples(
            rate_hz, n_samples, fs_hz, rng, margin_samples=template.size
        )
        for sample in samples:
            start = int(sample) - trough_index
            signal[start : start + template.size] += amplitude * template
        spike_samples.append(samples)

    recording = Recording(
        samples=signal[np.newaxis, :],
        fs_hz=fs_hz,
        units=units,
        source="synthetic",
        meta={"seed": seed},
    )
    truth = GroundTruth(
        spike_samples=tuple(spike_samples),
        amplitudes=tuple(float(a) for a in amplitudes),
        noise_rms=float(noise_rms),
        lfp_rms=float(lfp_rms),
    )
    return recording, truth
