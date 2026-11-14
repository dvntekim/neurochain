from pathlib import Path

import numpy as np
import pytest

from neurochain.data import mearec
from neurochain.dsp.spectra import band_power, welch_psd

FS_HZ = 30_000.0


def write_file(path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 60_000
    templates = np.zeros((2, 3, 40), dtype=np.float32)
    templates[0, 0, 10] = -80.0
    templates[0, 1, 10] = -20.0
    templates[1, 2, 12] = -150.0
    np.savez_compressed(
        path,
        fs_hz=FS_HZ,
        clean_uv=rng.standard_normal((3, n)).astype(np.float32),
        background_uv=(7.0 * rng.standard_normal((3, n))).astype(np.float32),
        reference_noise_uv=10.0,
        spike_samples=np.array([500, 100, 900, 300]),
        spike_unit=np.array([0, 0, 1, 1]),
        templates_uv=templates,
        cell_types=np.array(["E", "I"]),
        probe="test",
        seed=0,
        mearec_version="test",
    )


@pytest.fixture
def recording(tmp_path: Path) -> mearec.MearecRecording:
    path = tmp_path / "rec.npz"
    write_file(path)
    return mearec.load_mearec(path)


def test_shapes_and_ground_truth(recording: mearec.MearecRecording) -> None:
    assert recording.fs_hz == FS_HZ
    assert (recording.n_channels, recording.n_units) == (3, 2)
    assert recording.duration_s == pytest.approx(2.0)
    assert recording.clean_uv.dtype == np.float64
    assert [s.tolist() for s in recording.spike_samples] == [[100, 500], [300, 900]]
    assert recording.cell_types == ("E", "I")
    np.testing.assert_array_equal(recording.amplitudes_uv, [[80, 20, 0], [0, 0, 150]])


def test_background_is_scaled_to_unit_spike_band_rms(recording: mearec.MearecRecording) -> None:
    freqs, psd = welch_psd(recording.background_uv, FS_HZ, resolution_hz=10.0)
    np.testing.assert_allclose(np.sqrt(band_power(freqs, psd, 300.0, 7500.0)), 1.0, rtol=1e-6)
    # White noise keeps 7200/15000 of its power in the band, so its total RMS is larger.
    assert recording.background_uv.std() == pytest.approx(np.sqrt(15_000 / 7200), rel=0.03)


def test_signal_mixes_clean_and_background(recording: mearec.MearecRecording) -> None:
    np.testing.assert_array_equal(recording.signal_uv(1, 0.0), recording.clean_uv[1])
    mixed = recording.signal_uv(1, 13.0)
    np.testing.assert_allclose(mixed - recording.clean_uv[1], 13.0 * recording.background_uv[1])
    as_recording = recording.recording(2, 5.0)
    assert as_recording.units == "µV"
    assert as_recording.samples.shape == (1, 60_000)
    assert as_recording.meta == {"channel": 2, "background_rms_uv": 5.0}


def test_unit_selection(recording: mearec.MearecRecording) -> None:
    assert recording.units_above(0, 50.0).tolist() == [0]
    assert recording.units_above(1, 10.0).tolist() == [0]
    assert recording.units_above(1, 50.0).tolist() == []
    assert recording.spikes_of(np.array([0, 1])).tolist() == [100, 300, 500, 900]
    assert recording.spikes_of(np.array([], dtype=np.int64)).size == 0


def test_missing_file_names_the_command(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make mearec"):
        mearec.load_mearec(tmp_path / "absent.npz")


@pytest.mark.data
@pytest.mark.skipif(
    not (mearec.mearec_dir() / mearec.DEFAULT_FILE).is_file(),
    reason="MEArec recording not generated",
)
def test_generated_recording() -> None:
    real = mearec.load_mearec()
    assert real.fs_hz == 30_000.0
    assert (real.n_channels, real.n_units) == (32, 16)
    assert real.clean_uv.shape == real.background_uv.shape == (32, 1_800_000)
    assert real.duration_s == 60.0
    assert real.templates_uv.shape[:2] == (16, 32)
    # Every unit is detectable somewhere, and every channel has a target unit.
    assert real.amplitudes_uv.max(axis=1).min() >= 30.0
    assert all(real.units_above(c, 50.0).size >= 1 for c in range(32))
    # Spike times sit on the waveform trough.
    unit = int(np.argmax(real.amplitudes_uv[:, 5]))
    for sample in real.spike_samples[unit][:20]:
        window = real.clean_uv[5, sample - 20 : sample + 20]
        assert abs(int(np.argmin(window)) - 20) <= 2
