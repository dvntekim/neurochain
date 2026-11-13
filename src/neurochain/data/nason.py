"""Loaders for the example data of Nason et al. (2020), spiking-band power.

Three MATLAB v5 files, each a different kind of data (structure read from the
files and from the dataset's README, doi:10.7302/wwya-5q86):

* ``MultiUnitInVivo.mat``: 10 s of broadband from one channel of monkey W at
  30 kHz, with three sorted units and the authors' spiking-band power (SBP).
  This is the only file with broadband, and the only real recording in the
  project so far that carries a voltage scale.
* ``MultiUnitSim.mat``: two simulated arrangements of five units, five runs
  each, with the noise-free unit signals kept separate from the noise.
* ``OnlineTrainingData.mat``: 409 consecutive trials of a finger task with
  1 ms bins of SBP and threshold-crossing times on 96 channels, plus finger
  position. There is no broadband in this file.

Arrays keep the values and dtypes stored in the files. Three things about the
files are worth knowing before using them:

* The in-vivo ``SBP`` equals ``1e-6 * |filtered Raw|``, so it is in volts while
  ``Raw`` is in microvolts (``tests/test_nason_data.py`` reproduces it).
* Sorted spike times run a little past the end of the 10 s of ``Raw``. The file
  does not say whether sample indices are 0- or 1-based.
* The online SBP is a sum over the samples received in each millisecond, and
  that count (``SampleWidth``) varies from 0 to 4. Divide by it before
  comparing bins. The unit of those sums is not stated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import io

from neurochain.paths import raw_dir
from neurochain.recording import FloatArray, IntArray, Recording

IN_VIVO_FILE = "MultiUnitInVivo.mat"
SIMULATION_FILE = "MultiUnitSim.mat"
ONLINE_FILE = "OnlineTrainingData.mat"

ONLINE_BIN_RATE_HZ = 1000.0
"""Bin rate of the online file: one row per millisecond (dataset README)."""

ONLINE_SBP_SOURCE_RATE_HZ = 2000.0
"""Rate at which the 300-1000 Hz signal was sampled before binning (README)."""

ONLINE_SPIKING_BAND_HZ = (300.0, 1000.0)
"""Band of the online SBP (README; the file itself does not store it)."""

BoolArray = NDArray[np.bool_]


def nason_dir() -> Path:
    """Return the directory the Nason files are expected in."""
    return raw_dir("nason")


def _load_struct(path: Path, name: str) -> NDArray[Any]:
    """Read one top-level MATLAB struct array from a v5 ``.mat`` file.

    Raises:
        FileNotFoundError: If the file is absent, with the command that
            explains how to obtain it.
        KeyError: If the file does not hold the expected variable.
    """
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run `make data` for download instructions")
    contents = io.loadmat(str(path))
    if name not in contents:
        raise KeyError(f"{path.name} has no variable {name!r}")
    struct: NDArray[Any] = contents[name]
    return struct


def _scalar(value: NDArray[Any]) -> float:
    """Return a 1x1 MATLAB array as a Python float."""
    return float(np.asarray(value).reshape(-1)[0])


def _band(value: NDArray[Any]) -> tuple[float, float]:
    """Return a 1x2 MATLAB array as a ``(low_hz, high_hz)`` tuple."""
    low, high = np.asarray(value, dtype=np.float64).reshape(-1)
    return float(low), float(high)


@dataclass(frozen=True)
class NasonInVivo:
    """Ten seconds of broadband from one channel, with sorted units and SBP.

    Attributes:
        fs_hz: Broadband sampling rate (30 kHz).
        downsampled_fs_hz: Rate of the SBP and spike-indicator arrays (2 kHz).
        spiking_band_hz: Band the authors filtered to before taking SBP.
        samples_per_spike: Samples the authors took as one spike's duration.
        broadband_uv: Raw recording in microvolts, shape ``(n_samples,)``.
            Hardware-filtered 0.3 Hz to 7.5 kHz, so it still contains LFP.
        spike_samples: One array per sorted unit of spike sample indices at
            ``fs_hz``, as stored. Some lie past the end of ``broadband_uv``.
        unit_min_uv: Minimum of each unit's mean waveform, microvolts.
        unit_amp_uv: Peak-to-peak amplitude of each unit's mean waveform,
            microvolts.
        spike_indicator: ``(n_downsampled, n_units)``; true where a unit fired
            in that 2 kHz bin.
        sbp_v: The authors' SBP at 2 kHz, in volts.
        smoothed_rate: ``spike_indicator`` smoothed with ``smooth_window``.
        smoothed_sbp_v: ``sbp_v`` smoothed with ``smooth_window``.
        smooth_window: The smoothing kernel (sums to one).
        smooth_duration_s: Duration of the smoothing kernel.
        smooth_type: Name of the kernel shape, e.g. ``"Gaussian"``.
    """

    fs_hz: float
    downsampled_fs_hz: float
    spiking_band_hz: tuple[float, float]
    samples_per_spike: int
    broadband_uv: FloatArray
    spike_samples: tuple[IntArray, ...]
    unit_min_uv: FloatArray
    unit_amp_uv: FloatArray
    spike_indicator: BoolArray
    sbp_v: FloatArray
    smoothed_rate: FloatArray
    smoothed_sbp_v: FloatArray
    smooth_window: FloatArray
    smooth_duration_s: float
    smooth_type: str

    @property
    def n_units(self) -> int:
        """Number of sorted units."""
        return len(self.spike_samples)

    @property
    def decimation(self) -> int:
        """Broadband samples per downsampled sample."""
        return round(self.fs_hz / self.downsampled_fs_hz)

    def recording(self) -> Recording:
        """Return the broadband as a single-channel :class:`Recording` in µV."""
        return Recording(
            samples=self.broadband_uv[np.newaxis, :],
            fs_hz=self.fs_hz,
            units="µV",
            source="nason",
            meta={"clip_id": "MultiUnitInVivo", "n_units": self.n_units},
        )


def load_in_vivo(path: Path | None = None) -> NasonInVivo:
    """Load ``MultiUnitInVivo.mat``.

    Args:
        path: File to read. Defaults to ``data/raw/nason/MultiUnitInVivo.mat``.
    """
    s = _load_struct(path or nason_dir() / IN_VIVO_FILE, "MultiUnitInVivo")[0, 0]
    return NasonInVivo(
        fs_hz=_scalar(s["SamplingRate"]),
        downsampled_fs_hz=_scalar(s["DownsampledRate"]),
        spiking_band_hz=_band(s["SpikingBand"]),
        samples_per_spike=int(_scalar(s["SamplesPerSpike"])),
        broadband_uv=np.asarray(s["Raw"], dtype=np.float64).reshape(-1),
        spike_samples=tuple(
            np.asarray(unit, dtype=np.int64).reshape(-1) for unit in s["SpikeTimes"].reshape(-1)
        ),
        unit_min_uv=np.asarray(s["UnitMin"], dtype=np.float64).reshape(-1),
        unit_amp_uv=np.asarray(s["UnitAmp"], dtype=np.float64).reshape(-1),
        spike_indicator=np.asarray(s["FR"], dtype=np.bool_),
        sbp_v=np.asarray(s["SBP"], dtype=np.float64).reshape(-1),
        smoothed_rate=np.asarray(s["SmoothedFR"], dtype=np.float64),
        smoothed_sbp_v=np.asarray(s["SmoothedSBP"], dtype=np.float64).reshape(-1),
        smooth_window=np.asarray(s["SmoothWindow"], dtype=np.float64).reshape(-1),
        smooth_duration_s=_scalar(s["SmoothDuration"]),
        smooth_type=str(s["SmoothType"].reshape(-1)[0]),
    )


@dataclass(frozen=True)
class NasonSimulation:
    """One simulated run: unit signals, noise and the authors' SBP.

    Amplitudes are in multiples of the noise standard deviation: the noise is
    drawn from a standard normal and each unit is scaled to its SNR.

    Attributes:
        spike_onsets: ``(n_samples, n_units)``; true at the sample where a unit
            starts a spike.
        unit_signals: ``(n_samples, n_units)`` noise-free signal of each unit.
        noise: ``(n_samples,)`` white Gaussian noise.
        spike_indicator: ``(n_downsampled, n_units)`` spike onsets at 2 kHz.
        sbp: ``(n_downsampled,)`` the authors' SBP of the summed signal.
        smoothed_rate: ``spike_indicator`` after smoothing.
        smoothed_sbp: ``sbp`` after smoothing.
    """

    spike_onsets: BoolArray
    unit_signals: FloatArray
    noise: FloatArray
    spike_indicator: BoolArray
    sbp: FloatArray
    smoothed_rate: FloatArray
    smoothed_sbp: FloatArray

    def broadband(self) -> FloatArray:
        """Return the simulated recording: every unit summed, plus the noise."""
        return np.asarray(self.unit_signals.sum(axis=1) + self.noise, dtype=np.float64)


@dataclass(frozen=True)
class NasonSimArrangement:
    """A simulated arrangement of units and its repeated runs.

    Attributes:
        fs_hz: Simulation sampling rate (30 kHz).
        downsampled_fs_hz: Rate of the SBP arrays (2 kHz).
        spiking_band_hz: Band filtered to before taking SBP.
        duration_s: Length of each run.
        snrs: Signal-to-noise ratio of each unit. Its trough depth, in noise
            standard deviations, matches this to within 1 %.
        firing_rates_hz: The ``FRs`` field as stored. It holds two values for
            five units; every unit fires 100 times in 5 s, which is 20 Hz.
        smooth_window: The smoothing kernel.
        smooth_duration_s: Duration of the smoothing kernel.
        smooth_type: Name of the kernel shape.
        simulations: The runs, each with fresh noise and spike times.
    """

    fs_hz: float
    downsampled_fs_hz: float
    spiking_band_hz: tuple[float, float]
    duration_s: float
    snrs: FloatArray
    firing_rates_hz: FloatArray
    smooth_window: FloatArray
    smooth_duration_s: float
    smooth_type: str
    simulations: tuple[NasonSimulation, ...]

    @property
    def n_units(self) -> int:
        """Number of simulated units."""
        return int(self.snrs.size)


def load_simulations(path: Path | None = None) -> tuple[NasonSimArrangement, ...]:
    """Load ``MultiUnitSim.mat``: one entry per simulated arrangement.

    Args:
        path: File to read. Defaults to ``data/raw/nason/MultiUnitSim.mat``.
    """
    struct = _load_struct(path or nason_dir() / SIMULATION_FILE, "MultiUnitSim")
    arrangements: list[NasonSimArrangement] = []
    for s in struct.reshape(-1):
        runs = tuple(
            NasonSimulation(
                spike_onsets=np.asarray(r["SpikeTimes"], dtype=np.bool_),
                unit_signals=np.asarray(r["IndividualRecordings"], dtype=np.float64),
                noise=np.asarray(r["Noise"], dtype=np.float64).reshape(-1),
                spike_indicator=np.asarray(r["TrueFR"], dtype=np.bool_),
                sbp=np.asarray(r["SBP"], dtype=np.float64).reshape(-1),
                smoothed_rate=np.asarray(r["SmoothedTrueFR"], dtype=np.float64),
                smoothed_sbp=np.asarray(r["SmoothedSBP"], dtype=np.float64).reshape(-1),
            )
            for r in s["Simulations"].reshape(-1)
        )
        arrangements.append(
            NasonSimArrangement(
                fs_hz=_scalar(s["SamplingRate"]),
                downsampled_fs_hz=_scalar(s["DownsampledRate"]),
                spiking_band_hz=_band(s["SpikingBand"]),
                duration_s=_scalar(s["SimulationDuration"]),
                snrs=np.asarray(s["SNRs"], dtype=np.float64).reshape(-1),
                firing_rates_hz=np.asarray(s["FRs"], dtype=np.float64).reshape(-1),
                smooth_window=np.asarray(s["SmoothWindow"], dtype=np.float64).reshape(-1),
                smooth_duration_s=_scalar(s["SmoothDuration"]),
                smooth_type=str(s["SmoothType"].reshape(-1)[0]),
                simulations=runs,
            )
        )
    return tuple(arrangements)


@dataclass(frozen=True)
class NasonTrial:
    """One trial of the finger task, in 1 ms bins.

    Attributes:
        target_pos: Target centre, 0 (full extension) to 1 (full flexion).
        time_ms: Experiment time of each bin in milliseconds, shape ``(n_bins,)``.
        finger_angle: Finger position on the same 0-to-1 scale, ``(n_bins,)``.
        sbp: ``(n_bins, n_channels)`` sum of the absolute 300-1000 Hz samples
            received in each millisecond. Unit not stated in the dataset.
        sample_width: Number of samples summed into each bin, ``(n_bins,)``.
        crossing_times_ms: One array per channel of threshold-crossing times in
            experiment milliseconds. Times can repeat within a millisecond.
    """

    target_pos: float
    time_ms: IntArray
    finger_angle: NDArray[np.float32]
    sbp: NDArray[np.uint16]
    sample_width: NDArray[np.uint8]
    crossing_times_ms: tuple[IntArray, ...]

    @property
    def n_bins(self) -> int:
        """Number of 1 ms bins in the trial."""
        return int(self.time_ms.size)

    @property
    def n_channels(self) -> int:
        """Number of recording channels."""
        return len(self.crossing_times_ms)

    def crossing_counts(self) -> NDArray[np.uint8]:
        """Return threshold crossings per bin, shape ``(n_bins, n_channels)``.

        Aligned with ``time_ms``, so it can be stacked next to ``sbp``.

        Raises:
            ValueError: If a crossing time lies outside the trial.
        """
        counts = np.zeros((self.n_bins, self.n_channels), dtype=np.uint8)
        start = int(self.time_ms[0])
        for channel, times in enumerate(self.crossing_times_ms):
            if not times.size:
                continue
            offsets = times - start
            if offsets.min() < 0 or offsets.max() >= self.n_bins:
                raise ValueError(f"channel {channel} has crossings outside the trial's time span")
            counts[:, channel] = np.bincount(offsets, minlength=self.n_bins)
        return counts


@dataclass(frozen=True)
class NasonOnlineArrays:
    """Every trial of the online file concatenated along time.

    Trials are consecutive but not contiguous (2 ms are missing between one
    trial and the next), so use ``trial_index`` to find the boundaries.

    Attributes:
        time_ms: Experiment time of each bin, ``(n_bins,)``.
        finger_angle: Finger position, 0 to 1, ``(n_bins,)``.
        target_pos: Target centre of the bin's trial, ``(n_bins,)``.
        sbp: ``(n_bins, n_channels)`` summed absolute spiking-band samples.
        sample_width: Samples summed into each bin, ``(n_bins,)``.
        crossing_counts: ``(n_bins, n_channels)`` threshold crossings per bin.
        trial_index: Index of the trial each bin belongs to, ``(n_bins,)``.
    """

    time_ms: IntArray
    finger_angle: NDArray[np.float32]
    target_pos: NDArray[np.float32]
    sbp: NDArray[np.uint16]
    sample_width: NDArray[np.uint8]
    crossing_counts: NDArray[np.uint8]
    trial_index: IntArray


@dataclass(frozen=True)
class NasonOnline:
    """Training data for the online SBP and threshold-crossing decoders.

    Monkey W moved all fingers together in a centre-out task. All trials are
    consecutive and successful.

    Attributes:
        bin_rate_hz: Bin rate of every time series (1 kHz).
        sbp_source_rate_hz: Rate the spiking band was sampled at before binning.
        spiking_band_hz: Band of the SBP feature.
        trials: The trials, in order.
    """

    bin_rate_hz: float
    sbp_source_rate_hz: float
    spiking_band_hz: tuple[float, float]
    trials: tuple[NasonTrial, ...]

    @property
    def n_trials(self) -> int:
        """Number of trials."""
        return len(self.trials)

    @property
    def n_channels(self) -> int:
        """Number of recording channels."""
        return self.trials[0].n_channels

    @property
    def duration_s(self) -> float:
        """Total recorded time across trials, in seconds."""
        return sum(t.n_bins for t in self.trials) / self.bin_rate_hz

    def stack(self) -> NasonOnlineArrays:
        """Concatenate all trials into continuous arrays for a decoder."""
        lengths = np.asarray([t.n_bins for t in self.trials], dtype=np.int64)
        return NasonOnlineArrays(
            time_ms=np.concatenate([t.time_ms for t in self.trials]),
            finger_angle=np.concatenate([t.finger_angle for t in self.trials]),
            target_pos=np.repeat(
                np.asarray([t.target_pos for t in self.trials], dtype=np.float32), lengths
            ),
            sbp=np.concatenate([t.sbp for t in self.trials]),
            sample_width=np.concatenate([t.sample_width for t in self.trials]),
            crossing_counts=np.concatenate([t.crossing_counts() for t in self.trials]),
            trial_index=np.repeat(np.arange(lengths.size, dtype=np.int64), lengths),
        )


def load_online(path: Path | None = None) -> NasonOnline:
    """Load ``OnlineTrainingData.mat``.

    Args:
        path: File to read. Defaults to
            ``data/raw/nason/OnlineTrainingData.mat``.
    """
    struct = _load_struct(path or nason_dir() / ONLINE_FILE, "OnlineTrainingData")
    trials = tuple(
        NasonTrial(
            target_pos=_scalar(t["TargetPos"]),
            time_ms=np.asarray(t["ExperimentTime"], dtype=np.int64).reshape(-1),
            finger_angle=np.asarray(t["FingerAngle"], dtype=np.float32).reshape(-1),
            sbp=np.asarray(t["SpikingBandPower"], dtype=np.uint16),
            sample_width=np.asarray(t["SampleWidth"], dtype=np.uint8).reshape(-1),
            crossing_times_ms=tuple(
                np.asarray(channel["SpikeTimes"], dtype=np.int64).reshape(-1)
                for channel in t["Channel"].reshape(-1)
            ),
        )
        for t in struct.reshape(-1)
    )
    return NasonOnline(
        bin_rate_hz=ONLINE_BIN_RATE_HZ,
        sbp_source_rate_hz=ONLINE_SBP_SOURCE_RATE_HZ,
        spiking_band_hz=ONLINE_SPIKING_BAND_HZ,
        trials=trials,
    )


def describe() -> str:
    """Return a plain-text summary of the three files on disk."""
    in_vivo = load_in_vivo()
    arrangements = load_simulations()
    online = load_online()
    arrays = online.stack()
    seconds = arrays.time_ms.size / online.bin_rate_hz
    crossing_rate_hz = arrays.crossing_counts.sum(axis=0) / seconds
    lines = [
        "Nason et al. (2020) example data",
        f"  {IN_VIVO_FILE}",
        f"    broadband: {in_vivo.broadband_uv.size:,} samples at {in_vivo.fs_hz:,.0f} Hz "
        f"({in_vivo.broadband_uv.size / in_vivo.fs_hz:.1f} s), µV",
        f"    sorted units: {in_vivo.n_units}, spikes per unit "
        f"{[int(s.size) for s in in_vivo.spike_samples]}, "
        f"peak-to-peak {np.round(in_vivo.unit_amp_uv, 0).tolist()} µV",
        f"    SBP: {in_vivo.sbp_v.size:,} samples at {in_vivo.downsampled_fs_hz:,.0f} Hz, "
        f"band {in_vivo.spiking_band_hz} Hz",
        f"  {SIMULATION_FILE}",
    ]
    for index, arrangement in enumerate(arrangements):
        lines.append(
            f"    arrangement {index}: {arrangement.n_units} units, SNRs "
            f"{np.round(arrangement.snrs, 2).tolist()}, {len(arrangement.simulations)} runs of "
            f"{arrangement.duration_s:g} s at {arrangement.fs_hz:,.0f} Hz"
        )
    lines += [
        f"  {ONLINE_FILE}",
        f"    trials: {online.n_trials}, channels: {online.n_channels}, "
        f"bins: {arrays.time_ms.size:,} at {online.bin_rate_hz:,.0f} Hz ({seconds / 60:.1f} min)",
        f"    kinematics: finger angle {float(arrays.finger_angle.min()):g} to "
        f"{float(arrays.finger_angle.max()):g}; targets "
        f"{[round(float(t), 2) for t in np.unique(arrays.target_pos)]}",
        f"    SBP: {arrays.sbp.dtype}, samples per bin "
        f"{int(arrays.sample_width.min())} to {int(arrays.sample_width.max())} "
        f"(mean {float(arrays.sample_width.mean()):.3f})",
        f"    threshold crossings: {int(arrays.crossing_counts.sum()):,}; per-channel rate "
        f"median {float(np.median(crossing_rate_hz)):.2f} /s, "
        f"max {float(crossing_rate_hz.max()):.1f} /s",
        "    broadband: none in this file",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
