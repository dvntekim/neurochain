"""Where data, figures and results live on disk."""

from __future__ import annotations

import os
from pathlib import Path

DATA_ENV_VAR = "NEUROCHAIN_DATA"


def repo_root() -> Path:
    """Return the repository root (the directory holding ``pyproject.toml``)."""
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    """Return the data directory.

    Defaults to ``<repo>/data`` (git-ignored). Set ``NEUROCHAIN_DATA`` to keep
    datasets on another disk.
    """
    override = os.environ.get(DATA_ENV_VAR)
    return Path(override).expanduser().resolve() if override else repo_root() / "data"


def raw_dir(dataset_key: str) -> Path:
    """Return the directory holding one dataset's files exactly as downloaded."""
    return data_dir() / "raw" / dataset_key


def figures_dir() -> Path:
    """Return the directory for committed figures."""
    return repo_root() / "figures"


def results_dir() -> Path:
    """Return the directory for committed result tables (JSON)."""
    return repo_root() / "results"


def cache_dir() -> Path:
    """Return the directory for regenerable intermediate arrays (git-ignored)."""
    return results_dir() / "cache"
