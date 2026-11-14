import numpy as np
import pytest

from neurochain.data.synthetic import make_synthetic_recording
from neurochain.features import detect as det
from neurochain.features.sbp import (
    bin_counts,
    bin_mean,
    gaussian_window,
    rectify_and_bin,
    smooth,
    spiking_band_power,
)
from neurochain.features.score import DetectionScore, score_detections

FS_HZ = 30_000.0


def test_highpass_is_causal_and_blocks_dc() -> None:
    x = np.zeros(6000)
    x[3000] = 1.0
    y = det.causal_highpass(x, FS_HZ)
    assert np.all(y[:3000] == 0.0)  # nothing before the impulse arrives
    step = det.causal_highpass(np.ones(60_000) * 100.0, FS_HZ)
    assert abs(step[-1]) < 1e-6
    with pytest.raises(ValueError, match="Nyquist"):
        det.highpass_sos(500.0)


def test_neo_of_a_sinusoid_is_amplitude_squared_times_sin_squared() -> None:
    omega = 2 * np.pi * 1000.0 / FS_HZ
    x = 3.0 * np.sin(omega * np.arange(1000))
    psi = det.neo(x)
    np.testing.assert_allclose(psi[1:-1], 9.0 * np.sin(omega) ** 2, rtol=1e-9)
    assert psi[0] == psi[-1] == 0.0


def test_neo_favours_fast_events_over_slow_ones_of_the_same_size() -> None:
    n = np.arange(600)
    slow = np.exp(-0.5 * ((n - 300) / 60.0) ** 2)
    fast = np.exp(-0.5 * ((n - 300) / 3.0) ** 2)
    assert det.neo(fast).max() > 100 * det.neo(slow).max()


def test_moving_sum() -> None:
    x = np.arange(1.0, 7.0)
    np.testing.assert_array_equal(det.moving_sum(x, 3), [1, 3, 6, 9, 12, 15])
    np.testing.assert_array_equal(det.moving_sum(x, 1), x)


def test_detect_events_aligns_and_respects_the_dead_time() -> None:
    score = np.zeros(3000)
    score[1000:1004] = [6.0, 9.0, 7.0, 6.5]
    score[1010] = 8.0  # 0.33 ms later: inside the dead time
    score[2000] = 7.0
    events = det.detect_events(score, 5.0, FS_HZ)
    assert events.tolist() == [1001, 2000]
    assert det.detect_events(np.zeros(100), 5.0, FS_HZ).size == 0


def test_threshold_calibration_rules() -> None:
    rng = np.random.default_rng(0)
    noise = 2.0 * rng.standard_normal(100_000)
    assert det.calibrate_threshold(noise, "threshold", 4.5) == pytest.approx(9.0, rel=0.03)
    positive = np.abs(noise)
    assert det.calibrate_threshold(positive, "neo", 8.0) == pytest.approx(8 * positive.mean())


@pytest.mark.parametrize(("method", "factor"), [("threshold", 4.5), ("neo", 8.0)])
def test_detectors_find_planted_spikes(method: det.Method, factor: float) -> None:
    recording, truth = make_synthetic_recording(
        fs_hz=FS_HZ, duration_s=12.0, amplitudes=(90.0, 120.0), noise_rms=5.0, seed=3
    )
    events, threshold = det.detect(
        recording.channel(0), FS_HZ, method, factor=factor, calibration_samples=round(2 * FS_HZ)
    )
    score = score_detections(
        events, truth.all_spike_samples(), tolerance=30, start=round(2 * FS_HZ)
    )
    assert threshold > 0
    assert score.recall > 0.9
    assert score.precision > 0.9


def test_detect_can_skip_the_highpass() -> None:
    x = np.zeros(3000)
    x[1500] = -50.0
    x[::7] += 0.5  # so the noise estimate is not zero
    events, _ = det.detect(
        x, FS_HZ, "threshold", factor=4.5, calibration_samples=1000, highpass_hz=None
    )
    assert events.tolist() == [1500]


def test_score_counts_and_metrics() -> None:
    targets = np.array([100, 200, 300, 400])
    ignore = np.array([500])
    detected = np.array([101, 250, 299, 501, 900])
    score = score_detections(detected, targets, tolerance=3, ignore=ignore)
    assert (score.true_positives, score.false_positives, score.false_negatives) == (2, 2, 2)
    assert score.ignored == 1
    assert score.precision == pytest.approx(0.5)
    assert score.recall == pytest.approx(0.5)
    assert score.f1 == pytest.approx(0.5)


def test_score_skips_the_calibration_segment_and_pools() -> None:
    targets = np.array([10, 110, 210])
    detected = np.array([10, 110, 300])
    late = score_detections(detected, targets, tolerance=2, start=100)
    assert (late.true_positives, late.false_positives, late.false_negatives) == (1, 1, 1)
    pooled = late + DetectionScore(3, 0, 1)
    assert (pooled.true_positives, pooled.false_positives, pooled.false_negatives) == (4, 1, 2)
    empty = DetectionScore(0, 0, 0)
    assert (empty.precision, empty.recall, empty.f1) == (0.0, 0.0, 0.0)


def test_one_detection_cannot_claim_two_spikes() -> None:
    score = score_detections(np.array([100]), np.array([99, 101]), tolerance=3)
    assert (score.true_positives, score.false_negatives) == (1, 1)


def test_sbp_passes_the_band_and_rejects_outside_it() -> None:
    t = np.arange(round(2 * FS_HZ)) / FS_HZ

    def level(freq_hz: float) -> float:
        return float(spiking_band_power(np.sin(2 * np.pi * freq_hz * t), FS_HZ)[1000:].mean())

    in_band = level(600.0)
    assert in_band == pytest.approx(2 / np.pi, rel=0.05)  # mean |sin| is 2/pi
    assert level(50.0) < 0.02 * in_band
    assert level(6000.0) < 0.05 * in_band
    assert spiking_band_power(np.zeros(3000), FS_HZ).size == 200  # 30 kHz to 2 kHz


def test_binning_helpers() -> None:
    x = np.arange(10.0)
    np.testing.assert_array_equal(bin_mean(x, 4), [1.5, 5.5])
    np.testing.assert_array_equal(rectify_and_bin(-x, 5), [2.0, 7.0])
    assert bin_mean(np.ones((3, 10)), 5).shape == (3, 2)
    counts = bin_counts(np.array([0, 1, 5, 9, 11]), n_samples=12, samples_per_bin=5)
    np.testing.assert_array_equal(counts, [2, 2])


def test_gaussian_smoothing_keeps_the_mean_and_does_not_shift() -> None:
    window = gaussian_window(0.05, 2000.0)
    assert window.sum() == pytest.approx(1.0)
    assert window.size == 100
    x = np.zeros(2000)
    x[1000] = 1.0
    y = smooth(x, window)
    assert abs(int(np.argmax(y)) - 1000) <= 1
    assert y.sum() == pytest.approx(1.0, rel=1e-6)
