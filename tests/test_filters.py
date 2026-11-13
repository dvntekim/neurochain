import numpy as np
import pytest
from scipy import signal

from neurochain.dsp.filters import (
    apply_sos,
    band_split,
    design_bandpass,
    design_lowpass,
    requantise,
)

FS_HZ = 20_000.0


def tone(freq_hz: float, seconds: float = 2.0) -> np.ndarray:
    t = np.arange(round(seconds * FS_HZ)) / FS_HZ
    return np.sin(2 * np.pi * freq_hz * t)


def steady_rms(x: np.ndarray) -> float:
    """RMS of the middle half, away from filter edge effects."""
    n = x.size
    return float(np.sqrt(np.mean(x[n // 4 : 3 * n // 4] ** 2)))


@pytest.mark.parametrize(
    ("freq_hz", "lfp_gain", "spike_gain"),
    [(20.0, 1.0, 0.0), (100.0, 1.0, 0.0), (1000.0, 0.0, 1.0), (3000.0, 0.0, 1.0)],
)
def test_tones_land_in_the_right_band(freq_hz: float, lfp_gain: float, spike_gain: float) -> None:
    bands = band_split(tone(freq_hz), FS_HZ)
    reference = 1 / np.sqrt(2)
    assert steady_rms(bands.lfp) / reference == pytest.approx(lfp_gain, abs=0.02)
    assert steady_rms(bands.spikes) / reference == pytest.approx(spike_gain, abs=0.02)


def test_cutoff_is_minus_3_db_causal_and_minus_6_db_zero_phase() -> None:
    sos = design_lowpass(300.0, FS_HZ)
    x = tone(300.0)
    causal = steady_rms(apply_sos(sos, x, zero_phase=False)) / steady_rms(x)
    zero_phase = steady_rms(apply_sos(sos, x, zero_phase=True)) / steady_rms(x)
    assert 20 * np.log10(causal) == pytest.approx(-3.01, abs=0.05)
    assert 20 * np.log10(zero_phase) == pytest.approx(-6.02, abs=0.1)


def test_bandpass_rolloff_matches_order() -> None:
    # A 4th-order skirt falls 80 dB per decade: one decade below the 300 Hz edge.
    sos = design_bandpass(300.0, 7500.0, FS_HZ, order=4)
    _, h = signal.sosfreqz(sos, worN=[30.0, 300.0], fs=FS_HZ)
    assert 20 * np.log10(abs(h[0])) == pytest.approx(-80.0, abs=1.5)
    assert 20 * np.log10(abs(h[1])) == pytest.approx(-3.01, abs=0.05)


def test_zero_phase_keeps_a_pulse_in_place_and_causal_delays_it() -> None:
    x = np.zeros(4000)
    x[2000] = 1.0
    sos = design_lowpass(300.0, FS_HZ)
    assert int(np.argmax(apply_sos(sos, x, zero_phase=True))) == 2000
    assert int(np.argmax(apply_sos(sos, x, zero_phase=False))) > 2000


def test_band_split_removes_dc() -> None:
    rng = np.random.default_rng(0)
    x = 500.0 + rng.standard_normal(20_000)
    bands = band_split(x, FS_HZ)
    assert abs(float(np.mean(bands.lfp))) < 0.5
    assert abs(float(np.mean(bands.spikes))) < 0.01


def test_band_split_filters_along_the_last_axis() -> None:
    x = np.stack([tone(50.0), tone(2000.0)])
    bands = band_split(x, FS_HZ)
    assert bands.lfp.shape == x.shape
    assert steady_rms(bands.lfp[0]) > 0.69
    assert steady_rms(bands.lfp[1]) < 0.01


@pytest.mark.parametrize("edges", [(0.0, 1000.0), (300.0, 10_000.0), (300.0, 12_000.0)])
def test_band_edges_outside_zero_to_nyquist_are_rejected(edges: tuple[float, float]) -> None:
    with pytest.raises(ValueError, match="Nyquist"):
        design_bandpass(edges[0], edges[1], FS_HZ)


def test_bandpass_edges_must_be_ordered() -> None:
    with pytest.raises(ValueError, match="below"):
        design_bandpass(1000.0, 300.0, FS_HZ)


def test_requantise_zero_bits_is_identity() -> None:
    x = np.arange(-8.0, 8.0)
    np.testing.assert_array_equal(requantise(x, 0), x)


def test_requantise_error_is_bounded_and_zero_mean() -> None:
    x = np.arange(-512.0, 512.0)
    for bits in (1, 2, 3, 5):
        error = requantise(x, bits) - x
        assert np.abs(error).max() <= (2**bits - 1) / 2
        assert float(error.mean()) == pytest.approx(0.0, abs=1e-12)
        assert np.unique(requantise(x, bits)).size == 1024 // 2**bits


def test_requantise_rejects_negative_bits() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        requantise(np.zeros(4), -1)
