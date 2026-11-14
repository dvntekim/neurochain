import numpy as np
import pytest

from neurochain.data.synthetic import make_synthetic_recording
from neurochain.dsp.spectra import band_power, welch_psd
from neurochain.frontend.electrode import integrate_psd, log_grid
from neurochain.frontend.emulator import (
    emulate,
    emulate_recording,
    output_noise,
    reference_gain,
    sampled_noise_psd,
)
from neurochain.frontend.model import FrontEnd, baseline_frontend
from neurochain.recording import Recording

FS_HZ = 30_000.0


@pytest.fixture(scope="module")
def frontend() -> FrontEnd:
    return baseline_frontend()


def tone_uv(freq_hz: float, amplitude_uv: float, seconds: float = 1.0) -> np.ndarray:
    t = np.arange(round(seconds * FS_HZ)) / FS_HZ
    return amplitude_uv * np.sin(2 * np.pi * freq_hz * t)


def test_same_seed_same_output_and_different_seed_differs(frontend: FrontEnd) -> None:
    x = np.zeros(3000)
    a = emulate(x, FS_HZ, frontend, seed=3).output_v
    b = emulate(x, FS_HZ, frontend, seed=3).output_v
    c = emulate(x, FS_HZ, frontend, seed=4).output_v
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_noise_matches_the_model_spectrum(frontend: FrontEnd) -> None:
    rng = np.random.default_rng(0)
    noise = output_noise(round(20 * FS_HZ), FS_HZ, frontend, rng)
    freqs, psd = welch_psd(noise, FS_HZ)
    grid = log_grid(300.0, 7500.0)
    predicted = integrate_psd(frontend.output_noise(grid).total, grid)
    assert np.sqrt(float(band_power(freqs, psd, 300.0, 7500.0))) == pytest.approx(
        predicted, rel=0.02
    )
    # Shape as well as total: the density at 1 kHz and 5 kHz.
    for f0 in (1000.0, 5000.0):
        window = (freqs > f0 * 0.9) & (freqs < f0 * 1.1)
        expected = float(frontend.output_noise(np.array([f0])).total[0])
        assert float(np.mean(psd[window])) == pytest.approx(expected, rel=0.08)


def test_midband_tone_gets_the_reference_gain(frontend: FrontEnd) -> None:
    out = emulate(tone_uv(1000.0, 100.0), FS_HZ, frontend, seed=0, add_noise=False)
    steady = out.output_v[out.output_v.size // 2 :]
    # First-order filters at the network corners: within 2 % of the network at 1 kHz.
    assert np.sqrt(2) * steady.std() == pytest.approx(100e-6 * reference_gain(frontend), rel=0.02)
    assert out.gain == pytest.approx(reference_gain(frontend))
    assert out.saturated_fraction == 0.0


def test_out_of_band_tones_are_attenuated(frontend: FrontEnd) -> None:
    def gain_at(freq_hz: float) -> float:
        out = emulate(tone_uv(freq_hz, 100.0, 4.0), FS_HZ, frontend, seed=0, add_noise=False)
        return float(np.sqrt(2) * out.output_v[out.output_v.size // 2 :].std() / 100e-6)

    mid = gain_at(1000.0)
    assert gain_at(0.25) < 0.3 * mid  # two octaves below the 1 Hz corner
    assert gain_at(frontend.f_high_hz) == pytest.approx(mid / np.sqrt(2), rel=0.03)


def test_input_referred_output_recovers_the_signal_scale(frontend: FrontEnd) -> None:
    out = emulate(tone_uv(1000.0, 50.0), FS_HZ, frontend, seed=0, add_noise=False)
    steady = out.input_referred_uv[out.input_referred_uv.size // 2 :]
    assert np.sqrt(2) * steady.std() == pytest.approx(50.0, rel=0.02)


def test_ac_coupling_ignores_a_settled_offset(frontend: FrontEnd) -> None:
    x = tone_uv(1000.0, 100.0)
    plain = emulate(x, FS_HZ, frontend, seed=0, add_noise=False)
    offset = emulate(x, FS_HZ, frontend, seed=0, add_noise=False, dc_offset_v=0.3)
    np.testing.assert_allclose(offset.output_v, plain.output_v, atol=1e-9)
    assert offset.saturated_fraction == 0.0


def test_power_on_transient_rails_then_decays(frontend: FrontEnd) -> None:
    out = emulate(
        np.zeros(round(3 * FS_HZ)), FS_HZ, frontend, seed=0, add_noise=False,
        dc_offset_v=0.05, settled=False,
    )  # fmt: skip
    swing = frontend.lna.output_swing_v
    assert out.output_v[100] == pytest.approx(swing)
    assert 0.0 < out.saturated_fraction < 0.2
    assert abs(out.output_v[-1]) < 1e-4 * swing * 100


def test_dc_coupling_passes_the_offset_until_it_rails(frontend: FrontEnd) -> None:
    x = np.zeros(2000)
    small = emulate(x, FS_HZ, frontend, seed=0, add_noise=False, coupling="dc", dc_offset_v=1e-3)
    assert small.output_v[-1] == pytest.approx(1e-3 * reference_gain(frontend), rel=1e-6)
    railed = emulate(x, FS_HZ, frontend, seed=0, add_noise=False, coupling="dc", dc_offset_v=0.05)
    assert railed.saturated_fraction == 1.0
    assert np.all(railed.output_v == frontend.lna.output_swing_v)
    low_gain = emulate(
        x, FS_HZ, frontend, seed=0, add_noise=False, coupling="dc", dc_offset_v=0.3, dc_gain=2.5
    )
    assert low_gain.saturated_fraction == 0.0
    assert low_gain.gain == 2.5


def test_dc_gain_scales_the_noise_with_the_signal(frontend: FrontEnd) -> None:
    x = np.zeros(round(2 * FS_HZ))
    high = emulate(x, FS_HZ, frontend, seed=1, coupling="dc")
    low = emulate(x, FS_HZ, frontend, seed=1, coupling="dc", dc_gain=2.5)
    # Referred to the input, the same noise whatever the gain.
    np.testing.assert_allclose(low.input_referred_uv, high.input_referred_uv, rtol=1e-9, atol=1e-9)


def test_shapes_and_independent_channel_noise(frontend: FrontEnd) -> None:
    one = emulate(np.zeros(4000), FS_HZ, frontend, seed=0)
    two = emulate(np.zeros((2, 4000)), FS_HZ, frontend, seed=0)
    assert one.output_v.shape == (4000,)
    assert two.output_v.shape == (2, 4000)
    assert abs(float(np.corrcoef(two.output_v)[0, 1])) < 0.1
    np.testing.assert_array_equal(two.output_v[0], one.output_v)


def test_bandwidth_must_be_below_nyquist(frontend: FrontEnd) -> None:
    with pytest.raises(ValueError, match="Nyquist"):
        emulate(np.zeros(1000), 10_000.0, frontend, seed=0)


def test_emulate_recording_keeps_units_and_records_settings(frontend: FrontEnd) -> None:
    recording, _ = make_synthetic_recording(duration_s=1.0, fs_hz=FS_HZ, seed=2)
    out = emulate_recording(recording, frontend, seed=5)
    assert out.units == "µV"
    assert out.samples.shape == recording.samples.shape
    assert out.fs_hz == recording.fs_hz
    assert out.meta["frontend_seed"] == 5
    assert out.meta["frontend_gain"] == pytest.approx(reference_gain(frontend))
    assert out.meta["seed"] == recording.meta["seed"]


def test_emulate_recording_refuses_adc_counts(frontend: FrontEnd) -> None:
    counts = Recording(np.zeros((1, 100)), FS_HZ, "ADC LSB", "neuralink")
    with pytest.raises(ValueError, match="µV"):
        emulate_recording(counts, frontend, seed=0)


def test_sampling_folds_out_of_band_noise_back(frontend: FrontEnd) -> None:
    freqs = np.array([1000.0, 5000.0, 12_000.0])
    plain = sampled_noise_psd(freqs, FS_HZ, frontend, alias_zones=0)
    folded = sampled_noise_psd(freqs, FS_HZ, frontend, alias_zones=8)
    np.testing.assert_allclose(plain, frontend.output_noise(freqs).total)
    assert np.all(folded > plain)
    # One zone adds the density at fs - f and fs + f, nothing else.
    one = sampled_noise_psd(freqs, FS_HZ, frontend, alias_zones=1)
    expected = (
        plain
        + frontend.output_noise(FS_HZ - freqs).total
        + frontend.output_noise(FS_HZ + freqs).total
    )
    np.testing.assert_allclose(one, expected)
    # The series converges: the next eight zones add little.
    more = sampled_noise_psd(freqs, FS_HZ, frontend, alias_zones=16)
    assert np.all(more / folded < 1.02)


def test_emulated_noise_includes_the_folded_part(frontend: FrontEnd) -> None:
    n = round(20 * FS_HZ)
    plain = emulate(np.zeros(n), FS_HZ, frontend, seed=0).output_v
    aliased = emulate(np.zeros(n), FS_HZ, frontend, seed=0, alias_zones=8).output_v
    grid = np.linspace(10.0, FS_HZ / 2, 4000)
    predicted = np.sqrt(
        np.trapezoid(sampled_noise_psd(grid, FS_HZ, frontend, 8), grid)
        / np.trapezoid(sampled_noise_psd(grid, FS_HZ, frontend, 0), grid)
    )
    assert aliased.std() / plain.std() == pytest.approx(predicted, rel=0.02)
    assert predicted > 1.05
