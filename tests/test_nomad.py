"""Loader tests on a small NWB file built to the real files' structure."""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.file import Subject

from neurochain.data import nomad

BIN_NS = 20_000_000
N_BINS = 60
N_CHANNELS = 6


def write_session(
    path: Path,
    *,
    t0_ns: int = 1_600_000_000_000_000_000,
    fractional_counts: bool = False,
    shifted_series: str | None = None,
) -> Path:
    """Write a 60-bin, 6-channel session: block 3 (calibration) then block 5 after a gap."""
    steps = np.full(N_BINS - 1, BIN_NS, dtype=np.float64)
    steps[29] = 500 * BIN_NS  # a 10 s gap between the two blocks
    stamps = t0_ns + np.concatenate(([0.0], np.cumsum(steps)))
    rng = np.random.default_rng(0)
    counts = rng.poisson(0.5, size=(N_BINS, N_CHANNELS)).astype(np.float64)
    if fractional_counts:
        counts[0, 0] = 0.5
    block = np.repeat([3.0, 5.0], N_BINS // 2).reshape(-1, 1)
    cond = np.tile([0.0, 1.0, 2.0], N_BINS // 3).reshape(-1, 1)
    xy = rng.standard_normal((N_BINS, 2))
    series = {
        "binned_spikes": (counts, "spk/20ms"),
        "cursor_pos": (xy, "position"),
        "cursor_vel": (0.01 * xy, "position/s"),
        "target_pos": (np.full((N_BINS, 2), 0.5), "position"),
        "block_num": (block, "block"),
        "cond_id": (cond, "condition"),
    }
    nwb = NWBFile(
        session_description="synthetic cursor task",
        identifier="S1_1900.01.01",
        session_start_time=datetime(1900, 1, 1, tzinfo=UTC),
        lab="TestLab",
        experiment_description="centre-out",
    )
    nwb.subject = Subject(subject_id="S1", species="Homo sapiens", sex="M")
    for name, (data, unit) in series.items():
        own = stamps + (BIN_NS if name == shifted_series else 0)
        nwb.add_acquisition(TimeSeries(name=name, data=data, unit=unit, timestamps=own))
    for column in ("cond_id", "tgt_x", "tgt_y", "result", "is_calibration", "block_num"):
        nwb.add_trial_column(name=column, description=column)
    for k, (first_bin, block_num) in enumerate([(2, 3), (12, 3), (32, 5), (45, 5)]):
        nwb.add_trial(
            start_time=float(stamps[first_bin]),
            stop_time=float(stamps[first_bin + 8]),
            cond_id=float(k + 1),
            tgt_x=0.5,
            tgt_y=-0.5,
            result=np.uint8(k != 3),
            is_calibration=float(block_num == 3),
            block_num=block_num,
        )
    with NWBHDF5IO(str(path), mode="w") as handle:
        handle.write(nwb)
    return path


@pytest.fixture
def session(tmp_path: Path) -> nomad.NomadSession:
    return nomad.load_session(write_session(tmp_path / "session1.nwb"))


def test_header_and_subject(session: nomad.NomadSession) -> None:
    assert session.name == "session1"
    assert session.identifier == "S1_1900.01.01"
    assert (session.subject_id, session.species, session.sex) == ("S1", "Homo sapiens", "M")
    assert session.lab == "TestLab"
    assert session.units["binned_spikes"] == "spk/20ms"


def test_time_base_is_seconds_from_the_first_bin(session: nomad.NomadSession) -> None:
    assert session.bin_s == pytest.approx(0.02)
    assert session.bin_rate_hz == pytest.approx(50.0)
    assert session.time_s[0] == 0.0
    assert session.time_s[1] == pytest.approx(0.02)
    assert session.start_time_ns == 1_600_000_000_000_000_000
    assert session.recorded_s == pytest.approx(1.2)
    assert session.span_s == pytest.approx(59 * 0.02 + 499 * 0.02 + 0.02)


def test_shapes_and_dtypes(session: nomad.NomadSession) -> None:
    assert (session.n_bins, session.n_channels) == (N_BINS, N_CHANNELS)
    assert session.binned_spikes.dtype == np.uint16
    assert session.cursor_pos.shape == session.cursor_vel.shape == (N_BINS, 2)
    assert session.target_pos.shape == (N_BINS, 2)
    assert session.cursor_pos.dtype == np.float64
    assert session.block_num.shape == session.cond_id.shape == (N_BINS,)
    assert session.block_num.dtype == session.cond_id.dtype == np.int64


def test_segments_and_blocks(session: nomad.NomadSession) -> None:
    assert session.segment_starts.tolist() == [0, 30]
    assert session.blocks.tolist() == [3, 5]
    assert session.calibration_blocks.tolist() == [3]
    assert session.is_calibration_bin.tolist() == [True] * 30 + [False] * 30


def test_trial_table(session: nomad.NomadSession) -> None:
    trials = session.trials
    assert trials.n_trials == 4
    assert trials.start_s[0] == pytest.approx(0.04)
    np.testing.assert_allclose(trials.stop_s - trials.start_s, 0.16)
    assert trials.cond_id.tolist() == [1, 2, 3, 4]
    assert trials.target_xy.shape == (4, 2)
    assert trials.result.tolist() == [True, True, True, False]
    assert trials.is_calibration.tolist() == [True, True, False, False]
    assert trials.block_num.tolist() == [3, 3, 5, 5]


def test_fractional_counts_are_refused(tmp_path: Path) -> None:
    path = write_session(tmp_path / "bad.nwb", fractional_counts=True)
    with pytest.raises(ValueError, match="integer count"):
        nomad.load_session(path)


def test_series_with_different_timestamps_are_refused(tmp_path: Path) -> None:
    path = write_session(tmp_path / "bad.nwb", shifted_series="cursor_vel")
    with pytest.raises(ValueError, match="cursor_vel has its own timestamps"):
        nomad.load_session(path)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make data"):
        nomad.load_session(tmp_path / "absent.nwb")


def test_load_both_sessions_and_gap(tmp_path: Path) -> None:
    day_ns = 86_400 * 10**9
    write_session(tmp_path / "session1.nwb", t0_ns=1_600_000_000_000_000_000 + 10 * day_ns)
    write_session(tmp_path / "session2.nwb", t0_ns=1_600_000_000_000_000_000)
    first, second = nomad.load_sessions(tmp_path)
    assert (first.name, second.name) == ("session1", "session2")
    assert nomad.days_between(first, second) == pytest.approx(-10.0)
    assert nomad.days_between(second, first) == pytest.approx(10.0)


def test_describe_session_states_species_and_channels(session: nomad.NomadSession) -> None:
    text = nomad.describe_session(session)
    assert "species: Homo sapiens" in text
    assert "channels: 6" in text
    assert "2 contiguous segments" in text
    assert "3 (2, cal), 5 (2)" in text
