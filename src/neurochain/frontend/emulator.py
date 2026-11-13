"""Apply the front-end model to a recording: gain, band-limiting, noise, rails.

The emulator turns a trace of the potential at the electrode (in microvolts)
into what the amplifier would put out, so that later units can ask what a
cheaper front end does to detection and decoding.

What it does, in order:

1. Adds the electrode's DC offset to the signal.
2. Filters with the amplifier's pass band: a first-order high-pass at the
   feedback corner (AC coupling only) and a first-order low-pass at the OTA
   bandwidth. Both run forward in time only, as the circuit does.
3. Applies the gain.
4. Adds noise drawn from the model's output noise spectrum: electrode thermal
   noise, OTA noise and feedback-resistor noise, shaped exactly as the network
   shapes them.
5. Clips at the supply rails.

Simplifications, each revisited later:

* The signal path uses first-order filters at the network's -3 dB corners. The
  network also has a slow, few-percent gain slope from the electrode loading
  the input, which the signal path ignores (the noise path does not).
* Clipping is a hard limit on the output. A real amplifier that rails also
  takes time to recover; U5 models that.
* Noise above the Nyquist frequency is dropped, not folded back. Aliasing is a
  property of the sampler and belongs to the ADC model in U2.
* The amplifier inverts. The emulator does not, so spikes keep their polarity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import signal

from neurochain.frontend.model import FrontEnd
from neurochain.recording import FloatArray, Recording

Coupling = Literal["ac", "dc"]
REFERENCE_HZ = 1000.0
"""Frequency at which the gain is read when referring the output to the input."""


@dataclass(frozen=True)
class EmulatorOutput:
    """What the emulated amplifier produced.

    Attributes:
        output_v: Amplifier output in volts, same shape as the input.
        gain: Gain used to refer the output back to the input.
        saturated_fraction: Fraction of samples on a supply rail.
        fs_hz: Sampling rate.
    """

    output_v: FloatArray
    gain: float
    saturated_fraction: float
    fs_hz: float

    @property
    def input_referred_uv(self) -> FloatArray:
        """The output divided by the gain, in microvolts."""
        return np.asarray(self.output_v / self.gain * 1.0e6, dtype=np.float64)


def reference_gain(frontend: FrontEnd) -> float:
    """Return the signal gain magnitude at 1 kHz, the centre of the spike band."""
    return float(np.abs(frontend.signal_gain(np.array([REFERENCE_HZ])))[0])


def output_noise(
    n_samples: int, fs_hz: float, frontend: FrontEnd, rng: np.random.Generator
) -> FloatArray:
    """Draw amplifier output noise with the model's spectrum, in volts.

    White Gaussian noise is shaped in the frequency domain to the model's
    output noise density. Shaping a random process this way does not make the
    emulator non-causal: the noise has no input to anticipate.

    Args:
        n_samples: Number of samples to generate.
        fs_hz: Sampling rate in hertz.
        frontend: The front end whose noise to draw.
        rng: Random generator.
    """
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs_hz)
    psd = np.zeros_like(freqs)
    psd[1:] = frontend.output_noise(freqs[1:]).total
    white = np.fft.rfft(rng.standard_normal(n_samples))
    # Unit-variance white noise has a one-sided density of 2 / fs.
    shaped = np.fft.irfft(white * np.sqrt(psd * fs_hz / 2.0), n=n_samples)
    return np.asarray(shaped, dtype=np.float64)


def _first_order(kind: str, corner_hz: float, fs_hz: float) -> tuple[FloatArray, FloatArray]:
    b, a = signal.butter(1, corner_hz, btype=kind, fs=fs_hz)
    return np.asarray(b, dtype=np.float64), np.asarray(a, dtype=np.float64)


def _causal(b: FloatArray, a: FloatArray, x: FloatArray, *, settled: bool) -> FloatArray:
    """Filter forward in time, starting from rest or from steady state."""
    if settled:
        state = signal.lfilter_zi(b, a)[np.newaxis, :] * x[:, :1]
        out, _ = signal.lfilter(b, a, x, axis=-1, zi=state)
    else:
        out = signal.lfilter(b, a, x, axis=-1)
    return np.asarray(out, dtype=np.float64)


def emulate(
    signal_uv: FloatArray,
    fs_hz: float,
    frontend: FrontEnd,
    *,
    seed: int,
    dc_offset_v: float | FloatArray = 0.0,
    coupling: Coupling = "ac",
    dc_gain: float | None = None,
    add_noise: bool = True,
    settled: bool = True,
) -> EmulatorOutput:
    """Pass a signal through the emulated front end.

    Args:
        signal_uv: Potential at the electrode in microvolts, shape
            ``(n_samples,)`` or ``(n_channels, n_samples)``.
        fs_hz: Sampling rate in hertz.
        frontend: The electrode and amplifier model.
        seed: Seed for the noise. The same seed gives the same output.
        dc_offset_v: Electrode offset in volts, a constant or one value per
            sample (use an array to apply a step).
        coupling: ``"ac"`` for the capacitively coupled amplifier, ``"dc"`` for
            a direct-coupled amplifier with the same noise and bandwidth.
        dc_gain: Gain of the direct-coupled amplifier. Defaults to the
            AC-coupled gain.
        add_noise: Add the model's noise. Turn off to see the signal path alone.
        settled: Start the filters in steady state for the first sample, as if
            the amplifier had been on for a long time. If false they start from
            rest, which shows the power-on transient.

    Raises:
        ValueError: If the amplifier bandwidth is not below the Nyquist
            frequency, where a sampled filter cannot represent it.
    """
    x = np.atleast_2d(np.asarray(signal_uv, dtype=np.float64)) * 1.0e-6
    x = x + np.asarray(dc_offset_v, dtype=np.float64)
    nyquist_hz = fs_hz / 2.0
    f_high_hz = frontend.f_high_hz
    if f_high_hz >= nyquist_hz:
        raise ValueError(
            f"amplifier bandwidth {f_high_hz:.0f} Hz is not below the Nyquist frequency "
            f"{nyquist_hz:.0f} Hz; resample the signal first"
        )
    gain = reference_gain(frontend)
    if coupling == "ac":
        x = _causal(*_first_order("highpass", frontend.f_low_hz, fs_hz), x, settled=settled)
    elif dc_gain is not None:
        gain = dc_gain
    x = _causal(*_first_order("lowpass", f_high_hz, fs_hz), x, settled=settled)
    out = gain * x
    if add_noise:
        rng = np.random.default_rng(seed)
        scale = gain / reference_gain(frontend)
        for channel in range(out.shape[0]):
            out[channel] += scale * output_noise(out.shape[1], fs_hz, frontend, rng)
    swing = frontend.lna.output_swing_v
    saturated = float(np.mean(np.abs(out) >= swing))
    out = np.clip(out, -swing, swing)
    if np.ndim(signal_uv) == 1:
        out = out[0]
    return EmulatorOutput(output_v=out, gain=gain, saturated_fraction=saturated, fs_hz=fs_hz)


def emulate_recording(
    recording: Recording,
    frontend: FrontEnd,
    *,
    seed: int,
    dc_offset_v: float = 0.0,
    coupling: Coupling = "ac",
) -> Recording:
    """Emulate the front end on a recording in microvolts.

    Returns:
        A recording of the amplifier output referred to the input, in
        microvolts, with the front end's settings in ``meta``.

    Raises:
        ValueError: If the recording is not in microvolts. A recording in ADC
            counts has no voltage scale to apply a noise model to.
    """
    if recording.units != "µV":
        raise ValueError(f"the emulator needs a recording in µV, got {recording.units!r}")
    result = emulate(
        recording.samples,
        recording.fs_hz,
        frontend,
        seed=seed,
        dc_offset_v=dc_offset_v,
        coupling=coupling,
    )
    meta = {
        **recording.meta,
        "frontend_gain": result.gain,
        "frontend_saturated_fraction": result.saturated_fraction,
        "frontend_seed": seed,
        "frontend_coupling": coupling,
    }
    return Recording(
        samples=np.atleast_2d(result.input_referred_uv),
        fs_hz=recording.fs_hz,
        units="µV",
        source=recording.source,
        meta=meta,
    )
