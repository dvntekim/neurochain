import numpy as np
import pytest

from neurochain.adc import Adc, convert, decimation_factor, quantise, sample


def test_step_codes_and_bit_rate() -> None:
    adc = Adc(fs_hz=30_000.0, bits=10, input_range_uv=1000.0)
    assert adc.lsb_uv == pytest.approx(2000.0 / 1024)
    assert (adc.code_min, adc.code_max) == (-512, 511)
    assert adc.bit_rate_bps == 300_000.0
    # One more bit halves the step.
    assert Adc(bits=11).lsb_uv == pytest.approx(adc.lsb_uv / 2)


def test_quantise_rounds_and_clips() -> None:
    adc = Adc(bits=4, input_range_uv=8.0)  # 1 µV steps, codes -8 to 7
    codes, clipped = quantise(np.array([0.4, 0.6, -0.6, 7.4, 9.0, -20.0]), adc)
    assert codes.tolist() == [0, 1, -1, 7, 7, -8]
    assert clipped == pytest.approx(2 / 6)
    assert codes.dtype == np.int64


def test_quantisation_error_is_a_twelfth_of_a_step_squared() -> None:
    rng = np.random.default_rng(0)
    adc = Adc(bits=10, input_range_uv=1000.0)
    x = 50.0 * rng.standard_normal(200_000)
    codes, clipped = quantise(x, adc)
    error = codes * adc.lsb_uv - x
    assert clipped == 0.0
    assert float(np.var(error)) == pytest.approx(adc.lsb_uv**2 / 12, rel=0.02)
    assert abs(float(np.mean(error))) < 0.01


def test_sampling_keeps_every_kth_sample_without_filtering() -> None:
    x = np.arange(12, dtype=np.float64)
    np.testing.assert_array_equal(sample(x, 30_000.0, 10_000.0), [0, 3, 6, 9])
    assert sample(np.zeros((2, 12)), 30_000.0, 15_000.0).shape == (2, 6)
    assert decimation_factor(30_000.0, 30_000.0) == 1


def test_sampling_aliases_a_tone_above_the_new_nyquist() -> None:
    fs = 30_000.0
    t = np.arange(30_000) / fs
    tone = np.sin(2 * np.pi * 7000.0 * t)  # above 5 kHz, the Nyquist frequency at 10 kS/s
    sampled = sample(tone, fs, 10_000.0)
    spectrum = np.abs(np.fft.rfft(sampled))
    peak_hz = np.fft.rfftfreq(sampled.size, 1 / 10_000.0)[int(np.argmax(spectrum))]
    assert peak_hz == pytest.approx(3000.0, abs=2.0)  # 10 kHz - 7 kHz


def test_convert_reports_rate_step_and_microvolts() -> None:
    adc = Adc(fs_hz=15_000.0, bits=8, input_range_uv=128.0)
    out = convert(np.linspace(-200.0, 200.0, 3000), 30_000.0, adc)
    assert out.fs_hz == 15_000.0
    assert out.codes.size == 1500
    assert out.lsb_uv == 1.0
    assert out.clipped_fraction > 0.3
    np.testing.assert_array_equal(out.microvolts, out.codes.astype(float))


def test_invalid_settings_are_rejected() -> None:
    with pytest.raises(ValueError, match="bits"):
        Adc(bits=0)
    with pytest.raises(ValueError, match="positive"):
        Adc(input_range_uv=0.0)
    with pytest.raises(ValueError, match="whole multiple"):
        decimation_factor(30_000.0, 7000.0)
