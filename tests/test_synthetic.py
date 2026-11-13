import numpy as np
import pytest

from neurochain.data.synthetic import (
    make_synthetic_recording,
    poisson_spike_samples,
    spike_template,
)


def test_same_seed_gives_identical_recording() -> None:
    a, truth_a = make_synthetic_recording(seed=7, duration_s=2.0)
    b, truth_b = make_synthetic_recording(seed=7, duration_s=2.0)
    np.testing.assert_array_equal(a.samples, b.samples)
    np.testing.assert_array_equal(truth_a.all_spike_samples(), truth_b.all_spike_samples())


def test_different_seed_gives_different_recording() -> None:
    a, _ = make_synthetic_recording(seed=1, duration_s=1.0)
    b, _ = make_synthetic_recording(seed=2, duration_s=1.0)
    assert not np.array_equal(a.samples, b.samples)


def test_template_trough_is_minus_one_and_ends_at_zero() -> None:
    template, trough = spike_template(20_000.0)
    assert template[trough] == pytest.approx(-1.0)
    assert template.min() == pytest.approx(-1.0)
    assert template[0] == pytest.approx(0.0, abs=1e-9)
    assert template[-1] == pytest.approx(0.0, abs=1e-9)


def test_noiseless_recording_has_spikes_of_the_requested_amplitude() -> None:
    recording, truth = make_synthetic_recording(
        amplitudes=(80.0,), rates_hz=(5.0,), noise_rms=0.0, lfp_rms=0.0, duration_s=4.0, seed=3
    )
    x = recording.channel(0)
    samples = truth.spike_samples[0]
    assert samples.size > 5
    assert recording.units == "µV"
    np.testing.assert_allclose(x[samples], -80.0, rtol=1e-9)


def test_poisson_rate_and_refractory_gap() -> None:
    rng = np.random.default_rng(0)
    fs_hz, seconds = 20_000.0, 200.0
    samples = poisson_spike_samples(10.0, round(fs_hz * seconds), fs_hz, rng, refractory_s=2e-3)
    # Mean interval is 1/rate + refractory, so the realised rate is 1 / 0.102 Hz.
    assert samples.size / seconds == pytest.approx(1 / 0.102, rel=0.08)
    assert np.diff(samples).min() >= round(2e-3 * fs_hz) - 1
    assert poisson_spike_samples(0.0, 1000, fs_hz, rng).size == 0


def test_component_levels() -> None:
    recording, truth = make_synthetic_recording(
        amplitudes=(), rates_hz=(), noise_rms=5.0, lfp_rms=20.0, duration_s=20.0, seed=4
    )
    total_rms = float(np.std(recording.channel(0)))
    assert total_rms == pytest.approx(np.hypot(5.0, 20.0), rel=0.03)
    assert truth.all_spike_samples().size == 0


def test_mismatched_unit_lists_are_rejected() -> None:
    with pytest.raises(ValueError, match="same length"):
        make_synthetic_recording(amplitudes=(50.0,), rates_hz=(1.0, 2.0))
