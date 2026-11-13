"""Loader for the NoMAD human intracortical dataset (Karpowicz et al., 2025).

Two NWB files from BrainGate participant T11, each one research session of a
centre-out cursor task (doi:10.5061/dryad.q83bk3jtp). What the files hold, read
from the files themselves:

* **Neural data is already binned.** ``binned_spikes`` is threshold crossings
  per 20 ms bin on 192 channels. There is no broadband, no spike waveforms and
  no electrode table, so nothing about electrode position or impedance.
* **Kinematics** on the same 20 ms grid: cursor position, cursor velocity and
  target position, in the task's own screen units.
* **Time is not continuous.** Bins come in contiguous runs separated by gaps
  (between blocks, and in one session within blocks). Use
  :attr:`NomadSession.segment_starts` before filtering or differencing along
  time.
* **Timestamps are stored in nanoseconds**, not the seconds NWB specifies: the
  step between bins described as 20 ms is 2e7. The loader converts to seconds
  from the first bin of the session.

Run ``python -m neurochain.data.nomad`` to print the species, channel count and
session structure of the files on disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pynwb import NWBHDF5IO

from neurochain.paths import raw_dir
from neurochain.recording import FloatArray, IntArray

SESSION_FILES = ("session1.nwb", "session2.nwb")

TIMESTAMP_UNIT_S = 1.0e-9
"""Seconds per stored timestamp unit (the files store nanoseconds)."""

TIME_SERIES = ("binned_spikes", "cursor_pos", "cursor_vel", "target_pos", "block_num", "cond_id")

BoolArray = NDArray[np.bool_]


def nomad_dir() -> Path:
    """Return the directory the NoMAD files are expected in."""
    return raw_dir("nomad")


@dataclass(frozen=True)
class NomadTrials:
    """The trial table of one session, one entry per trial.

    Attributes:
        start_s: Trial start, seconds from the session's first bin.
        stop_s: Trial end, seconds from the session's first bin.
        cond_id: Condition identity (one per target location).
        target_xy: ``(n_trials, 2)`` centre of the cued target.
        result: True if the trial was part of a correctly repeated sequence.
        is_calibration: True if the trial belonged to a calibration block.
        block_num: Block the trial belonged to.
    """

    start_s: FloatArray
    stop_s: FloatArray
    cond_id: IntArray
    target_xy: FloatArray
    result: BoolArray
    is_calibration: BoolArray
    block_num: IntArray

    @property
    def n_trials(self) -> int:
        """Number of trials."""
        return int(self.start_s.size)


@dataclass(frozen=True)
class NomadSession:
    """One research session: binned threshold crossings and cursor kinematics.

    Every time series has one row per 20 ms bin and shares ``time_s``.

    Attributes:
        name: File stem, e.g. ``"session1"``.
        identifier: NWB identifier (subject and a date-shifted session date).
        subject_id: Participant code.
        species: Species as recorded in the file.
        sex: Sex as recorded in the file.
        session_description: Free-text description from the file.
        experiment_description: Free-text task description from the file.
        lab: Lab recorded in the file.
        bin_s: Bin width in seconds.
        start_time_ns: Stored timestamp of the first bin. Only differences
            between sessions are meaningful.
        time_s: Start of each bin in seconds from the first bin, ``(n_bins,)``.
        binned_spikes: ``(n_bins, n_channels)`` threshold crossings per bin.
        cursor_pos: ``(n_bins, 2)`` cursor position, screen units.
        cursor_vel: ``(n_bins, 2)`` cursor velocity as stored. It is not the
            finite difference of ``cursor_pos``.
        target_pos: ``(n_bins, 2)`` centre of the cued target.
        block_num: ``(n_bins,)`` block each bin was recorded in.
        cond_id: ``(n_bins,)`` condition of each bin (0 where none applies).
        units: Unit string stored with each time series.
        trials: The trial table.
    """

    name: str
    identifier: str
    subject_id: str
    species: str
    sex: str
    session_description: str
    experiment_description: str
    lab: str
    bin_s: float
    start_time_ns: int
    time_s: FloatArray
    binned_spikes: NDArray[np.uint16]
    cursor_pos: FloatArray
    cursor_vel: FloatArray
    target_pos: FloatArray
    block_num: IntArray
    cond_id: IntArray
    units: dict[str, str]
    trials: NomadTrials

    @property
    def n_bins(self) -> int:
        """Number of 20 ms bins."""
        return int(self.time_s.size)

    @property
    def n_channels(self) -> int:
        """Number of recording channels."""
        return int(self.binned_spikes.shape[1])

    @property
    def bin_rate_hz(self) -> float:
        """Bins per second."""
        return 1.0 / self.bin_s

    @property
    def recorded_s(self) -> float:
        """Seconds of data actually present (bins times bin width)."""
        return self.n_bins * self.bin_s

    @property
    def span_s(self) -> float:
        """Seconds from the first bin to the end of the last, gaps included."""
        return float(self.time_s[-1]) + self.bin_s

    @property
    def segment_starts(self) -> IntArray:
        """Bin index at which each contiguous run of bins begins."""
        gaps = np.flatnonzero(np.diff(self.time_s) > 1.5 * self.bin_s) + 1
        return np.concatenate((np.zeros(1, dtype=np.int64), gaps.astype(np.int64)))

    @property
    def blocks(self) -> IntArray:
        """Block numbers present in the session, in ascending order."""
        return np.unique(self.block_num)

    @property
    def calibration_blocks(self) -> IntArray:
        """Block numbers whose trials are marked as calibration."""
        return np.unique(self.trials.block_num[self.trials.is_calibration])

    @property
    def is_calibration_bin(self) -> BoolArray:
        """True for bins recorded in a calibration block, ``(n_bins,)``."""
        return np.isin(self.block_num, self.calibration_blocks)


def _as_counts(values: NDArray[Any], name: str) -> NDArray[np.uint16]:
    """Convert float-stored spike counts to ``uint16``, refusing to round.

    Raises:
        ValueError: If any value is negative, fractional or too large.
    """
    if not np.all(values == np.round(values)) or values.min() < 0 or values.max() > 65_535:
        raise ValueError(f"{name} is not a non-negative integer count array")
    return values.astype(np.uint16)


def _as_integers(values: NDArray[Any], name: str) -> IntArray:
    """Convert a float-stored label array to ``int64``, refusing to round."""
    flat = np.asarray(values).reshape(-1)
    if not np.all(flat == np.round(flat)):
        raise ValueError(f"{name} holds non-integer values")
    return flat.astype(np.int64)


def load_session(path: Path) -> NomadSession:
    """Load one NoMAD session from an NWB file.

    Args:
        path: The ``.nwb`` file.

    Raises:
        FileNotFoundError: If the file is absent, with the command that explains
            how to obtain it.
        ValueError: If the time series do not share one set of timestamps, or
            count data is not integer-valued.
    """
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run `make data` for download instructions")
    with NWBHDF5IO(str(path), mode="r", load_namespaces=True) as handle:
        nwb = handle.read()
        series = {name: nwb.acquisition[name] for name in TIME_SERIES}
        stamps = np.asarray(series["binned_spikes"].timestamps[:], dtype=np.float64)
        for name, ts in series.items():
            if not np.array_equal(np.asarray(ts.timestamps[:]), stamps):
                raise ValueError(f"{path.name}: {name} has its own timestamps")
        data = {name: np.asarray(ts.data[:]) for name, ts in series.items()}
        units = {name: str(ts.unit) for name, ts in series.items()}
        table = {name: np.asarray(nwb.trials[name][:]) for name in nwb.trials.colnames}
        subject = nwb.subject
        header = {
            "identifier": str(nwb.identifier),
            "subject_id": str(subject.subject_id),
            "species": str(subject.species),
            "sex": str(subject.sex),
            "session_description": str(nwb.session_description),
            "experiment_description": str(nwb.experiment_description),
            "lab": str(nwb.lab),
        }

    t0 = stamps[0]
    trials = NomadTrials(
        start_s=(table["start_time"] - t0) * TIMESTAMP_UNIT_S,
        stop_s=(table["stop_time"] - t0) * TIMESTAMP_UNIT_S,
        cond_id=_as_integers(table["cond_id"], "trials.cond_id"),
        target_xy=np.column_stack((table["tgt_x"], table["tgt_y"])).astype(np.float64),
        result=table["result"].astype(np.bool_),
        is_calibration=table["is_calibration"].astype(np.bool_),
        block_num=_as_integers(table["block_num"], "trials.block_num"),
    )
    time_s = (stamps - t0) * TIMESTAMP_UNIT_S
    return NomadSession(
        name=path.stem,
        bin_s=float(np.median(np.diff(time_s))),
        start_time_ns=int(t0),
        time_s=time_s,
        binned_spikes=_as_counts(data["binned_spikes"], "binned_spikes"),
        cursor_pos=data["cursor_pos"].astype(np.float64),
        cursor_vel=data["cursor_vel"].astype(np.float64),
        target_pos=data["target_pos"].astype(np.float64),
        block_num=_as_integers(data["block_num"], "block_num"),
        cond_id=_as_integers(data["cond_id"], "cond_id"),
        units=units,
        trials=trials,
        **header,
    )


def load_sessions(directory: Path | None = None) -> tuple[NomadSession, ...]:
    """Load both sessions, in file-name order.

    File-name order is not time order: see :func:`days_between`.

    Args:
        directory: Where the files are. Defaults to ``data/raw/nomad``.
    """
    root = directory if directory is not None else nomad_dir()
    return tuple(load_session(root / name) for name in SESSION_FILES)


def days_between(first: NomadSession, second: NomadSession) -> float:
    """Return days from the start of ``first`` to the start of ``second``.

    Negative if ``second`` was recorded earlier.
    """
    return (second.start_time_ns - first.start_time_ns) * TIMESTAMP_UNIT_S / 86_400.0


def describe_session(session: NomadSession) -> str:
    """Return a plain-text summary of one session's subject and structure."""
    trials = session.trials
    segments = np.diff(np.concatenate((session.segment_starts, [session.n_bins])))
    rate_hz = session.binned_spikes.mean(axis=0) / session.bin_s
    lines = [
        f"{session.name}  (identifier {session.identifier})",
        f"  subject: {session.subject_id}, species: {session.species}, sex: {session.sex}, "
        f"lab: {session.lab}",
        f"  description: {session.session_description}; {session.experiment_description}",
        f"  channels: {session.n_channels} (columns of binned_spikes; the file has no "
        "electrode table)",
        f"  neural data: threshold crossings per {session.bin_s * 1e3:g} ms bin "
        f"({session.units['binned_spikes']}); no broadband",
        f"  bins: {session.n_bins:,} = {session.recorded_s / 60:.1f} min of data over a "
        f"{session.span_s / 3600:.2f} h span, in {segments.size} contiguous segments "
        f"(shortest {int(segments.min())} bin(s), longest {int(segments.max())})",
        f"  crossing rate per channel: median {float(np.median(rate_hz)):.1f} /s, "
        f"range {float(rate_hz.min()):.1f} to {float(rate_hz.max()):.1f} /s",
        f"  trials: {trials.n_trials} ({int(trials.result.sum())} with result = 1); "
        f"{float(np.mean(trials.is_calibration)):.0%} calibration",
        "  blocks (trials, calibration?): "
        + ", ".join(
            f"{block} ({int(np.sum(trials.block_num == block))}"
            f"{', cal' if block in session.calibration_blocks else ''})"
            for block in session.blocks
        ),
        "  conditions (trials): "
        + ", ".join(
            f"{cond} ({int(np.sum(trials.cond_id == cond))})" for cond in np.unique(trials.cond_id)
        ),
    ]
    return "\n".join(lines)


def describe() -> str:
    """Return a summary of both sessions on disk and the time between them."""
    sessions = load_sessions()
    parts = ["NoMAD dataset (Karpowicz et al., 2025)", *(describe_session(s) for s in sessions)]
    first, second = sessions
    gap = days_between(first, second)
    earlier, later = (first, second) if gap >= 0 else (second, first)
    parts.append(
        f"Sessions: {len(sessions)}. {earlier.name} was recorded first; {later.name} follows "
        f"{abs(gap):.1f} days later (from the stored timestamps)."
    )
    return "\n".join(parts)


if __name__ == "__main__":
    print(describe())
