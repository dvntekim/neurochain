"""The one container every loader returns and every DSP stage consumes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class Recording:
    """A block of extracellular samples with the facts needed to interpret it.

    Attributes:
        samples: Array of shape ``(n_channels, n_samples)``. Time is the last
            axis so that SciPy's default ``axis=-1`` filters along time.
        fs_hz: Sampling rate in hertz.
        units: Physical unit of ``samples``, e.g. ``"ADC LSB"`` or ``"µV"``.
            A loader must never claim microvolts unless the source publishes
            the gain.
        source: Short dataset key, e.g. ``"neuralink"`` or ``"synthetic"``.
        meta: Free-form provenance (clip id, saturation fraction, ...).
    """

    samples: FloatArray
    fs_hz: float
    units: str
    source: str
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate shape and sampling rate."""
        if self.samples.ndim != 2:
            raise ValueError(
                f"samples must be 2-D (n_channels, n_samples), got shape {self.samples.shape}"
            )
        if self.fs_hz <= 0:
            raise ValueError(f"fs_hz must be positive, got {self.fs_hz}")

    @property
    def n_channels(self) -> int:
        """Number of channels."""
        return int(self.samples.shape[0])

    @property
    def n_samples(self) -> int:
        """Number of samples per channel."""
        return int(self.samples.shape[1])

    @property
    def duration_s(self) -> float:
        """Duration in seconds."""
        return self.n_samples / self.fs_hz

    def channel(self, index: int = 0) -> FloatArray:
        """Return one channel as a 1-D array (a view, not a copy)."""
        row: FloatArray = self.samples[index]
        return row

    def time_s(self) -> FloatArray:
        """Return the time axis in seconds, starting at zero."""
        return np.arange(self.n_samples, dtype=np.float64) / self.fs_hz
