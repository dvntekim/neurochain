"""Checks against the real NoMAD NWB files. Skipped when they are absent."""

import numpy as np
import pytest

from neurochain.data import nomad

pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(
        not all((nomad.nomad_dir() / name).is_file() for name in nomad.SESSION_FILES),
        reason="NoMAD files not downloaded",
    ),
]

EXPECTED = {
    # name: (bins, trials, contiguous segments, blocks)
    "session1": (63_157, 472, 255, [3, 7, 8, 9, 12, 14, 15, 16, 18]),
    "session2": (86_531, 644, 16, [3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
}


@pytest.fixture(scope="module")
def sessions() -> tuple[nomad.NomadSession, ...]:
    return nomad.load_sessions()


def test_subject_is_one_human_participant(sessions: tuple[nomad.NomadSession, ...]) -> None:
    assert len(sessions) == 2
    for session in sessions:
        assert session.species == "Homo sapiens"
        assert session.subject_id == "T11"
        assert session.lab == "BrainGate"


def test_sample_rate_shapes_and_dtypes(sessions: tuple[nomad.NomadSession, ...]) -> None:
    for session in sessions:
        n_bins, _, _, _ = EXPECTED[session.name]
        assert session.bin_s == pytest.approx(0.02)
        assert session.bin_rate_hz == pytest.approx(50.0)
        assert session.n_channels == 192
        assert session.binned_spikes.shape == (n_bins, 192)
        assert session.binned_spikes.dtype == np.uint16
        assert session.units["binned_spikes"] == "spk/20ms"
        for kinematics in (session.cursor_pos, session.cursor_vel, session.target_pos):
            assert kinematics.shape == (n_bins, 2)
            assert kinematics.dtype == np.float64
            assert np.all(np.isfinite(kinematics))
        assert session.time_s.shape == session.block_num.shape == session.cond_id.shape == (n_bins,)
        assert session.block_num.dtype == session.cond_id.dtype == np.int64
        assert not np.any(session.binned_spikes.sum(axis=0) == 0)  # no silent channel


def test_time_is_a_20_ms_grid_with_gaps(sessions: tuple[nomad.NomadSession, ...]) -> None:
    for session in sessions:
        _, _, n_segments, _ = EXPECTED[session.name]
        steps = np.diff(session.time_s)
        np.testing.assert_allclose(
            steps / session.bin_s, np.round(steps / session.bin_s), atol=1e-6
        )
        assert steps.min() == pytest.approx(0.02)
        assert session.segment_starts.size == n_segments
        # A block never continues across a boundary without a gap in time.
        block_changes = np.flatnonzero(np.diff(session.block_num) != 0) + 1
        assert np.all(np.isin(block_changes, session.segment_starts))


def test_session_structure(sessions: tuple[nomad.NomadSession, ...]) -> None:
    for session in sessions:
        _, n_trials, _, blocks = EXPECTED[session.name]
        trials = session.trials
        assert trials.n_trials == n_trials
        assert session.blocks.tolist() == blocks
        assert session.calibration_blocks.tolist() == [3]
        assert np.all(trials.stop_s > trials.start_s)
        assert trials.start_s.min() >= 0.0
        assert trials.stop_s.max() <= session.span_s
        assert trials.target_xy.shape == (n_trials, 2)
        assert trials.result.dtype == trials.is_calibration.dtype == np.bool_
        assert set(np.unique(trials.cond_id)) == set(range(1, 9))
        # Outside calibration only four targets are used.
        assert set(np.unique(trials.cond_id[~trials.is_calibration])) == {1, 2, 3, 4}


def test_sessions_are_nine_months_apart(sessions: tuple[nomad.NomadSession, ...]) -> None:
    first, second = sessions
    assert nomad.days_between(first, second) == pytest.approx(-273.9, abs=0.1)
