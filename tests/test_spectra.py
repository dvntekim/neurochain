import numpy as np
import pytest

from neurochain.data.synthetic import colored_noise
from neurochain.dsp.spectra import band_power, loglog_slope, segment_length, welch_psd

FS_HZ = 20_000.0


def test_segment_length_is_nearest_power_of_two() -> None:
    assert segment_length(19_531.0, 1.2) == 16_384
    assert segment_length(1000.0, 1.0) == 1024
    with pytest.raises(ValueError, match="positive"):
        segment_length(1000.0, 0.0)


def test_psd_integrates_to_variance() -> None:
    # Parseval: the area under the one-sided PSD is the signal's variance.
    rng = np.random.default_rng(0)
    x = 3.0 * rng.standard_normal(400_000)
    freqs, psd = welch_psd(x, FS_HZ)
    assert float(band_power(freqs, psd, 0.0, FS_HZ / 2)) == pytest.approx(9.0, rel=0.02)


def test_white_noise_density_is_variance_over_nyquist() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal(400_000)
    _, psd = welch_psd(x, FS_HZ)
    assert float(np.median(psd)) == pytest.approx(1.0 / (FS_HZ / 2), rel=0.05)


def test_sine_band_power_is_half_amplitude_squared() -> None:
    t = np.arange(200_000) / FS_HZ
    x = 4.0 * np.sin(2 * np.pi * 1000.0 * t)
    freqs, psd = welch_psd(x, FS_HZ)
    assert float(band_power(freqs, psd, 900.0, 1100.0)) == pytest.approx(8.0, rel=0.01)
    assert float(band_power(freqs, psd, 3000.0, 5000.0)) < 1e-6


def test_band_power_keeps_leading_axes() -> None:
    rng = np.random.default_rng(2)
    x = rng.standard_normal((3, 100_000))
    freqs, psd = welch_psd(x, FS_HZ)
    assert band_power(freqs, psd, 100.0, 1000.0).shape == (3,)


def test_band_power_rejects_bad_limits() -> None:
    freqs = np.linspace(0, 100, 101)
    psd = np.ones_like(freqs)
    with pytest.raises(ValueError, match="below"):
        band_power(freqs, psd, 50.0, 10.0)
    with pytest.raises(ValueError, match="fewer than two"):
        band_power(freqs, psd, 10.2, 10.4)


def test_loglog_slope_recovers_one_over_f() -> None:
    rng = np.random.default_rng(3)
    x = colored_noise(2_000_000, FS_HZ, rng, exponent=1.0, f_min_hz=1.0, f_max_hz=5000.0)
    freqs, psd = welch_psd(x, FS_HZ)
    assert loglog_slope(freqs, psd, 10.0, 3000.0) == pytest.approx(-1.0, abs=0.05)


def test_loglog_slope_of_white_noise_is_flat() -> None:
    rng = np.random.default_rng(4)
    freqs, psd = welch_psd(rng.standard_normal(1_000_000), FS_HZ)
    assert loglog_slope(freqs, psd, 10.0, 5000.0) == pytest.approx(0.0, abs=0.05)
