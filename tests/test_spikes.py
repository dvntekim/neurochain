import numpy as np
import pytest

from neurochain.data.synthetic import GroundTruth, make_synthetic_recording
from neurochain.dsp.filters import band_split
from neurochain.dsp.spectra import welch_psd
from neurochain.dsp.spikes import (
    detect_spikes,
    extract_waveforms,
    mad_sigma,
    match_spikes,
    noise_crossing_rate_hz,
    polarity_census,
)
from neurochain.recording import Recording

FS_HZ = 20_000.0


def test_mad_sigma_matches_std_for_gaussian_noise() -> None:
    rng = np.random.default_rng(0)
    assert mad_sigma(5.0 * rng.standard_normal(200_000)) == pytest.approx(5.0, rel=0.02)


def test_mad_sigma_ignores_spikes_where_std_does_not() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal(100_000)
    x[::500] -= 40.0  # 0.2 % of samples carry a large spike
    assert mad_sigma(x) == pytest.approx(1.0, rel=0.03)
    assert float(np.std(x)) > 2.0


def test_detects_planted_spikes(synthetic: tuple[Recording, GroundTruth]) -> None:
    recording, truth = synthetic
    spikes = band_split(recording.channel(0), recording.fs_hz).spikes
    detection = detect_spikes(spikes, recording.fs_hz)
    tp, fp, fn = match_spikes(detection.indices, truth.all_spike_samples(), tolerance=10)
    assert tp / (tp + fn) > 0.97  # recall
    assert tp / (tp + fp) > 0.97  # precision
    assert detection.threshold == pytest.approx(-4.5 * detection.sigma)
    assert np.all(detection.amplitudes < detection.threshold)


def test_noise_estimate_on_synthetic_matches_what_was_added() -> None:
    recording, truth = make_synthetic_recording(noise_rms=10.0, seed=2)
    spikes = band_split(recording.channel(0), recording.fs_hz).spikes
    # White noise of RMS 10 over 0-10 kHz keeps (7500 - 300) / 10000 of its power in band.
    expected = truth.noise_rms * np.sqrt(7200.0 / 10_000.0)
    assert mad_sigma(spikes) == pytest.approx(expected, rel=0.05)


def test_one_spike_is_counted_once() -> None:
    x = np.zeros(2000)
    x[1000:1004] = [-6.0, -9.0, -7.0, -6.5]  # several samples below threshold
    detection = detect_spikes(x, FS_HZ, sigma=1.0)
    assert detection.indices.tolist() == [1001]
    assert detection.amplitudes.tolist() == [-9.0]


def test_refractory_period_blocks_a_second_crossing() -> None:
    x = np.zeros(2000)
    x[1000] = -10.0
    x[1010] = -10.0  # 0.5 ms later, inside the 1 ms dead time
    x[1100] = -10.0
    assert detect_spikes(x, FS_HZ, sigma=1.0).indices.tolist() == [1000, 1100]
    assert detect_spikes(x, FS_HZ, sigma=1.0, refractory_s=0.2e-3).count == 3


def test_polarity_selects_direction() -> None:
    x = np.zeros(1000)
    x[200], x[600] = -10.0, 10.0
    assert detect_spikes(x, FS_HZ, sigma=1.0, polarity="neg").indices.tolist() == [200]
    positive = detect_spikes(x, FS_HZ, sigma=1.0, polarity="pos")
    assert positive.indices.tolist() == [600]
    assert positive.threshold == pytest.approx(4.5)
    assert positive.snr.tolist() == [10.0]


def test_no_spikes_gives_empty_result() -> None:
    detection = detect_spikes(np.zeros(100), FS_HZ, sigma=1.0)
    assert detection.count == 0
    assert detection.amplitudes.shape == (0,)


def test_detect_spikes_validates_input() -> None:
    with pytest.raises(ValueError, match="1-D"):
        detect_spikes(np.zeros((2, 10)), FS_HZ)
    with pytest.raises(ValueError, match="positive"):
        detect_spikes(np.zeros(10), FS_HZ, threshold_sigma=0.0)


def test_extract_waveforms_aligns_peak_and_drops_edges() -> None:
    x = np.arange(100, dtype=np.float64)
    waveforms, kept = extract_waveforms(x, np.array([2, 50, 98]), pre=5, post=10)
    assert kept.tolist() == [50]
    assert waveforms.shape == (1, 15)
    assert waveforms[0, 5] == 50.0
    empty, _ = extract_waveforms(x, np.array([1]), pre=5, post=10)
    assert empty.shape == (0, 15)


def test_match_spikes_counts() -> None:
    truth = np.array([100, 200, 300])
    detected = np.array([101, 250, 298, 299])
    assert match_spikes(detected, truth, tolerance=3) == (2, 2, 1)
    assert match_spikes(np.array([], dtype=np.int64), truth, tolerance=3) == (0, 0, 3)


@pytest.mark.parametrize("k", [3.0, 3.5])
def test_noise_false_alarm_rate_matches_a_simulation(k: float) -> None:
    # Band-limited Gaussian noise with no spikes: every detection is a false alarm.
    rng = np.random.default_rng(5)
    seconds = 400.0
    x = rng.standard_normal(round(seconds * FS_HZ))
    filtered = band_split(x, FS_HZ).spikes
    freqs, psd = welch_psd(x, FS_HZ)
    predicted_hz = noise_crossing_rate_hz(freqs, psd, (300.0, 7500.0), k, FS_HZ)
    observed = detect_spikes(
        filtered, FS_HZ, threshold_sigma=k, refractory_s=0.0, align_window_s=0.0
    ).count
    assert observed / seconds == pytest.approx(predicted_hz, rel=0.08)


def test_rice_formula_would_overestimate_the_sampled_rate() -> None:
    # Flat 300-7500 Hz noise: Rice gives f0 * exp(-k^2 / 2) with f0 = 4419 Hz.
    freqs = np.linspace(0.0, FS_HZ / 2, 4097)
    psd = np.ones_like(freqs)
    k = 4.5
    rice_hz = 4419.0 * np.exp(-0.5 * k**2)
    sampled_hz = noise_crossing_rate_hz(freqs, psd, (300.0, 7500.0), k, FS_HZ)
    assert 1.5 < rice_hz / sampled_hz < 3.0


def test_polarity_census_pairs_phases_of_one_event() -> None:
    x = np.zeros(4000)
    x[500], x[505] = -20.0, 8.0  # negative spike with a smaller positive lobe
    x[1500], x[1506] = -6.0, 15.0  # positive spike with a smaller negative lobe
    x[2500] = -10.0  # negative only
    x[3500] = 9.0  # positive only
    negative = detect_spikes(x, FS_HZ, sigma=1.0, polarity="neg")
    positive = detect_spikes(x, FS_HZ, sigma=1.0, polarity="pos")
    assert (negative.count, positive.count) == (3, 3)
    assert polarity_census(negative, positive, window=20) == (4, 2)
    assert polarity_census(negative, positive, window=2) == (6, 3)
    empty = detect_spikes(np.zeros(100), FS_HZ, sigma=1.0, polarity="pos")
    assert polarity_census(negative, empty, window=20) == (3, 0)
