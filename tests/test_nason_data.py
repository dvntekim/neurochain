"""Checks against the real Nason et al. files. Skipped when they are absent."""

import numpy as np
import pytest
from scipy import signal

from neurochain.data import nason

pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(
        not all(
            (nason.nason_dir() / name).is_file()
            for name in (nason.IN_VIVO_FILE, nason.SIMULATION_FILE, nason.ONLINE_FILE)
        ),
        reason="Nason files not downloaded",
    ),
]


@pytest.fixture(scope="module")
def in_vivo() -> nason.NasonInVivo:
    return nason.load_in_vivo()


@pytest.fixture(scope="module")
def online() -> nason.NasonOnline:
    return nason.load_online()


def test_in_vivo_sample_rates_shapes_and_dtypes(in_vivo: nason.NasonInVivo) -> None:
    assert in_vivo.fs_hz == 30_000.0
    assert in_vivo.downsampled_fs_hz == 2000.0
    assert in_vivo.spiking_band_hz == (300.0, 1000.0)
    assert in_vivo.broadband_uv.shape == (300_001,)
    assert in_vivo.broadband_uv.dtype == np.float64
    assert in_vivo.sbp_v.shape == in_vivo.smoothed_sbp_v.shape == (20_001,)
    assert in_vivo.spike_indicator.shape == in_vivo.smoothed_rate.shape == (20_001, 3)
    assert in_vivo.spike_indicator.dtype == np.bool_
    assert in_vivo.n_units == 3
    assert [s.size for s in in_vivo.spike_samples] == [254, 337, 67]
    assert all(s.dtype == np.int64 and np.all(np.diff(s) > 0) for s in in_vivo.spike_samples)
    assert in_vivo.unit_amp_uv.shape == in_vivo.unit_min_uv.shape == (3,)
    assert in_vivo.smooth_window.sum() == pytest.approx(1.0)


def test_spike_times_agree_with_the_downsampled_indicator(in_vivo: nason.NasonInVivo) -> None:
    """Spikes inside the 10 s of broadband are exactly the ones the 2 kHz indicator marks."""
    n = in_vivo.broadband_uv.size
    inside = [int(np.sum(s < n)) for s in in_vivo.spike_samples]
    assert inside == in_vivo.spike_indicator.sum(axis=0).tolist() == [249, 335, 64]
    assert [int(np.sum(s >= n)) for s in in_vivo.spike_samples] == [5, 2, 3]


def test_stored_sbp_is_reproduced_from_broadband(in_vivo: nason.NasonInVivo) -> None:
    """The README's recipe gives the stored SBP, which fixes its unit as volts.

    Filter to 300-1000 Hz with a causal 2nd-order Butterworth, keep every 15th
    sample, take the absolute value; the stored array is that times 1e-6.
    """
    b, a = signal.butter(2, in_vivo.spiking_band_hz, btype="bandpass", fs=in_vivo.fs_hz)
    filtered = signal.lfilter(b, a, in_vivo.broadband_uv)
    recomputed_v = np.abs(filtered[:: in_vivo.decimation]) * 1e-6
    np.testing.assert_allclose(recomputed_v, in_vivo.sbp_v, rtol=1e-6, atol=1e-12)


def test_simulations() -> None:
    arrangements = nason.load_simulations()
    assert len(arrangements) == 2
    np.testing.assert_allclose(arrangements[1].snrs, [9, 7, 5, 3, 1])
    for arrangement in arrangements:
        assert (arrangement.fs_hz, arrangement.downsampled_fs_hz) == (30_000.0, 2000.0)
        assert arrangement.duration_s == 5.0
        assert arrangement.n_units == 5
        assert len(arrangement.simulations) == 5
        b, a = signal.butter(2, arrangement.spiking_band_hz, btype="bandpass", fs=arrangement.fs_hz)
        for run in arrangement.simulations:
            assert run.unit_signals.shape == run.spike_onsets.shape == (150_000, 5)
            assert run.noise.shape == (150_000,)
            assert run.sbp.shape == (10_000,)
            assert run.spike_indicator.shape == (10_000, 5)
            assert run.spike_onsets.sum(axis=0).tolist() == [100] * 5
            # Troughs match the stated SNR, in units of the noise sigma, to within 1 %.
            np.testing.assert_allclose(-run.unit_signals.min(axis=0), arrangement.snrs, rtol=1e-2)
            assert float(run.noise.std()) == pytest.approx(1.0, abs=0.01)
            recomputed = np.abs(signal.lfilter(b, a, run.broadband())[::15])
            np.testing.assert_allclose(recomputed, run.sbp, atol=1e-9)


def test_online_structure(online: nason.NasonOnline) -> None:
    assert (online.n_trials, online.n_channels) == (409, 96)
    assert online.bin_rate_hz == 1000.0
    assert online.duration_s == pytest.approx(532.677)
    for trial in online.trials:
        assert trial.sbp.shape == (trial.n_bins, 96)
        assert trial.sbp.dtype == np.uint16
        assert trial.finger_angle.shape == trial.sample_width.shape == (trial.n_bins,)
        assert trial.finger_angle.dtype == np.float32
        assert trial.sample_width.dtype == np.uint8
        assert np.all(np.diff(trial.time_ms) == 1)  # one bin per millisecond
        assert 0.0 <= trial.target_pos <= 1.0
    starts = np.array([t.time_ms[0] for t in online.trials])
    stops = np.array([t.time_ms[-1] for t in online.trials])
    assert np.all(starts[1:] - stops[:-1] == 3)  # consecutive, 2 ms missing between trials


def test_online_stacked_arrays(online: nason.NasonOnline) -> None:
    arrays = online.stack()
    n = 532_677
    assert arrays.sbp.shape == arrays.crossing_counts.shape == (n, 96)
    assert arrays.time_ms.shape == arrays.finger_angle.shape == arrays.trial_index.shape == (n,)
    assert (arrays.sbp.dtype, arrays.crossing_counts.dtype) == (np.uint16, np.uint8)
    assert float(arrays.finger_angle.min()) == 0.0
    assert float(arrays.finger_angle.max()) == 1.0
    # Every stored crossing lands in a bin.
    stored = sum(times.size for trial in online.trials for times in trial.crossing_times_ms)
    assert int(arrays.crossing_counts.sum()) == stored == 750_815
    # SBP was sampled at 2 kHz, so a 1 ms bin holds two samples on average.
    assert float(arrays.sample_width.mean()) == pytest.approx(2.0, abs=0.02)
    assert int(arrays.sample_width.max()) == 4
