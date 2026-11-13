"""Loader tests on small ``.mat`` files built to the real files' structure."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy import io

from neurochain.data import nason


def struct(shape: tuple[int, int], **fields: list[Any]) -> np.ndarray:
    """Build a MATLAB-style struct array; each field lists one value per element."""
    out = np.empty(shape, dtype=[(name, object) for name in fields])
    for name, values in fields.items():
        for i, value in enumerate(values):
            out[name].reshape(-1)[i] = value
    return out


def common() -> dict[str, list[Any]]:
    return {
        "SamplingRate": [np.uint16([[30_000]])],
        "DownsampledRate": [np.uint16([[2000]])],
        "SpikingBand": [np.uint16([[300, 1000]])],
        "SmoothDuration": [np.array([[0.05]])],
        "SmoothType": [np.array(["Gaussian"])],
        "SmoothWindow": [np.full((4, 1), 0.25)],
    }


@pytest.fixture
def in_vivo_file(tmp_path: Path) -> Path:
    spike_times = np.empty((2, 1), dtype=object)
    spike_times[0, 0] = np.int32([[15, 45, 3200]])  # the last one is past the end of Raw
    spike_times[1, 0] = np.int32([[30]])
    data = struct(
        (1, 1),
        **common(),
        SamplesPerSpike=[np.uint8([[48]])],
        UnitMin=[np.array([[-300.0, -80.0]])],
        UnitAmp=[np.array([[550.0, 150.0]])],
        Raw=[np.arange(3001, dtype=np.float64).reshape(-1, 1)],
        SpikeTimes=[spike_times],
        FR=[np.eye(201, 2, dtype=np.uint8)],
        SBP=[np.full((201, 1), 1e-5)],
        SmoothedFR=[np.zeros((201, 2))],
        SmoothedSBP=[np.full((201, 1), 2e-5)],
    )
    path = tmp_path / nason.IN_VIVO_FILE
    io.savemat(path, {"MultiUnitInVivo": data})
    return path


@pytest.fixture
def simulation_file(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)

    def runs() -> np.ndarray:
        return struct(
            (1, 2),
            SpikeTimes=[np.eye(300, 3, dtype=np.uint8)] * 2,
            IndividualRecordings=[np.ones((300, 3)), 2 * np.ones((300, 3))],
            Noise=[rng.standard_normal((300, 1)), rng.standard_normal((300, 1))],
            TrueFR=[np.eye(20, 3, dtype=np.uint8)] * 2,
            SBP=[np.ones((20, 1))] * 2,
            SmoothedTrueFR=[np.zeros((20, 3))] * 2,
            SmoothedSBP=[np.ones((20, 1))] * 2,
        )

    shared = {k: v * 2 for k, v in common().items()}
    data = struct(
        (1, 2),
        **shared,
        SimulationDuration=[np.uint8([[5]])] * 2,
        SNRs=[np.array([[4.0, 2.0, 1.0]]), np.array([[9.0, 5.0, 1.0]])],
        FRs=[np.uint8([[20, 20]])] * 2,
        Simulations=[runs(), runs()],
    )
    path = tmp_path / nason.SIMULATION_FILE
    io.savemat(path, {"MultiUnitSim": data})
    return path


@pytest.fixture
def online_file(tmp_path: Path) -> Path:
    def channels(times: list[list[int]]) -> np.ndarray:
        values = [
            np.uint16(t).reshape(-1, 1) if t else np.zeros((0, 0), dtype=np.uint8) for t in times
        ]
        return struct((1, len(times)), SpikeTimes=values)

    data = struct(
        (1, 2),
        TargetPos=[np.float32([[0.5]]), np.float32([[0.9]])],
        FingerAngle=[
            np.linspace(0, 0.5, 5, dtype=np.float32).reshape(-1, 1),
            np.linspace(0.5, 0.9, 4, dtype=np.float32).reshape(-1, 1),
        ],
        ExperimentTime=[
            np.arange(100, 105, dtype=np.uint32).reshape(-1, 1),
            np.arange(107, 111, dtype=np.uint32).reshape(-1, 1),
        ],
        SpikingBandPower=[
            np.arange(15, dtype=np.uint16).reshape(5, 3),
            np.arange(12, dtype=np.uint16).reshape(4, 3),
        ],
        SampleWidth=[np.uint8([[2, 2, 0, 4, 2]]).T, np.uint8([[2, 1, 3, 2]]).T],
        Channel=[channels([[100, 102, 102], [], [104]]), channels([[], [107, 110], []])],
    )
    path = tmp_path / nason.ONLINE_FILE
    io.savemat(path, {"OnlineTrainingData": data})
    return path


def test_in_vivo_fields_units_and_shapes(in_vivo_file: Path) -> None:
    data = nason.load_in_vivo(in_vivo_file)
    assert data.fs_hz == 30_000.0
    assert data.downsampled_fs_hz == 2000.0
    assert data.decimation == 15
    assert data.spiking_band_hz == (300.0, 1000.0)
    assert data.samples_per_spike == 48
    assert data.broadband_uv.shape == (3001,)
    assert data.broadband_uv.dtype == np.float64
    assert data.n_units == 2
    assert [s.tolist() for s in data.spike_samples] == [[15, 45, 3200], [30]]
    assert data.spike_samples[0].dtype == np.int64
    assert data.unit_amp_uv.tolist() == [550.0, 150.0]
    assert data.spike_indicator.shape == (201, 2)
    assert data.spike_indicator.dtype == np.bool_
    assert data.sbp_v.shape == data.smoothed_sbp_v.shape == (201,)
    assert data.smooth_window.sum() == pytest.approx(1.0)
    assert (data.smooth_type, data.smooth_duration_s) == ("Gaussian", 0.05)


def test_in_vivo_as_recording(in_vivo_file: Path) -> None:
    recording = nason.load_in_vivo(in_vivo_file).recording()
    assert recording.samples.shape == (1, 3001)
    assert (recording.fs_hz, recording.units, recording.source) == (30_000.0, "µV", "nason")


def test_simulations(simulation_file: Path) -> None:
    arrangements = nason.load_simulations(simulation_file)
    assert len(arrangements) == 2
    assert [a.snrs.tolist() for a in arrangements] == [[4.0, 2.0, 1.0], [9.0, 5.0, 1.0]]
    first = arrangements[0]
    assert (first.fs_hz, first.downsampled_fs_hz, first.duration_s) == (30_000.0, 2000.0, 5.0)
    assert first.n_units == 3
    assert first.firing_rates_hz.tolist() == [20.0, 20.0]
    assert len(first.simulations) == 2
    run = first.simulations[1]
    assert run.unit_signals.shape == (300, 3)
    assert run.spike_onsets.dtype == np.bool_
    assert run.noise.shape == (300,)
    assert run.spike_indicator.shape == (20, 3)
    np.testing.assert_allclose(run.broadband(), 6.0 + run.noise)


def test_online_trials(online_file: Path) -> None:
    online = nason.load_online(online_file)
    assert (online.n_trials, online.n_channels) == (2, 3)
    assert online.bin_rate_hz == 1000.0
    assert online.duration_s == pytest.approx(0.009)
    trial = online.trials[0]
    assert trial.target_pos == 0.5
    assert trial.time_ms.tolist() == [100, 101, 102, 103, 104]
    assert trial.time_ms.dtype == np.int64
    assert trial.finger_angle.dtype == np.float32
    assert trial.sbp.shape == (5, 3)
    assert trial.sbp.dtype == np.uint16
    assert trial.sample_width.tolist() == [2, 2, 0, 4, 2]
    assert [t.tolist() for t in trial.crossing_times_ms] == [[100, 102, 102], [], [104]]


def test_crossing_counts_align_with_time_and_keep_repeats(online_file: Path) -> None:
    counts = nason.load_online(online_file).trials[0].crossing_counts()
    assert counts.dtype == np.uint8
    assert counts.T.tolist() == [[1, 0, 2, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 1]]


def test_crossings_outside_the_trial_are_rejected(online_file: Path) -> None:
    trial = nason.load_online(online_file).trials[0]
    bad = nason.NasonTrial(
        trial.target_pos,
        trial.time_ms,
        trial.finger_angle,
        trial.sbp,
        trial.sample_width,
        (np.array([99]), np.array([], dtype=np.int64), np.array([], dtype=np.int64)),
    )
    with pytest.raises(ValueError, match="outside"):
        bad.crossing_counts()


def test_stack_concatenates_trials(online_file: Path) -> None:
    arrays = nason.load_online(online_file).stack()
    assert arrays.time_ms.tolist() == [100, 101, 102, 103, 104, 107, 108, 109, 110]
    assert arrays.trial_index.tolist() == [0] * 5 + [1] * 4
    assert arrays.target_pos.tolist() == pytest.approx([0.5] * 5 + [0.9] * 4)
    assert arrays.sbp.shape == arrays.crossing_counts.shape == (9, 3)
    assert arrays.finger_angle.shape == arrays.sample_width.shape == (9,)
    assert int(arrays.crossing_counts.sum()) == 6


def test_missing_file_and_wrong_variable(tmp_path: Path, in_vivo_file: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make data"):
        nason.load_online(tmp_path / "absent.mat")
    with pytest.raises(KeyError, match="MultiUnitSim"):
        nason.load_simulations(in_vivo_file)
