"""Shared fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from neurochain.data.synthetic import GroundTruth, make_synthetic_recording
from neurochain.recording import Recording


@pytest.fixture
def synthetic() -> tuple[Recording, GroundTruth]:
    """Ten seconds at 20 kHz: two units well above an 8 µV noise floor."""
    return make_synthetic_recording(seed=1)


@pytest.fixture
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the package's data directory at an empty temporary directory."""
    monkeypatch.setenv("NEUROCHAIN_DATA", str(tmp_path))
    return tmp_path
