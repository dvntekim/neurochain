import numpy as np
import pytest

from neurochain.recording import Recording


def test_shape_and_duration() -> None:
    rec = Recording(samples=np.zeros((2, 1000)), fs_hz=500.0, units="µV", source="test")
    assert (rec.n_channels, rec.n_samples) == (2, 1000)
    assert rec.duration_s == pytest.approx(2.0)
    assert rec.time_s()[-1] == pytest.approx(999 / 500.0)
    assert rec.channel(1).shape == (1000,)


def test_rejects_one_dimensional_samples() -> None:
    with pytest.raises(ValueError, match="2-D"):
        Recording(samples=np.zeros(10), fs_hz=1.0, units="µV", source="test")


def test_rejects_non_positive_sampling_rate() -> None:
    with pytest.raises(ValueError, match="positive"):
        Recording(samples=np.zeros((1, 10)), fs_hz=0.0, units="µV", source="test")
