"""U0 analysis: spectra, band powers, noise floor and spike amplitudes.

One pass over a set of single-channel recordings produces, per clip, a Welch
PSD, the power in each band, a robust noise estimate and the detected spikes.
:func:`summarise` reduces that to the numbers reported in the README.

Every statistic is descriptive. Nothing is fitted or tuned here, so there is no
train/test split: all clips that pass the saturation check are used.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from neurochain.dsp.filters import band_split, requantise
from neurochain.dsp.spectra import band_power, loglog_slope, welch_psd
from neurochain.dsp.spikes import (
    Polarity,
    detect_spikes,
    noise_crossing_rate_hz,
    polarity_census,
)
from neurochain.recording import FloatArray, IntArray, Recording

QUANTISATION_RMS_LSB = 1.0 / np.sqrt(12.0)
"""RMS of ideal quantisation error for a step of one LSB (uniform over the step)."""

PHASE_WINDOW_S = 1.0e-3
"""Two opposite-sign detections closer than this are phases of one event."""


@dataclass(frozen=True)
class U0Config:
    """Every parameter the U0 analysis depends on.

    Attributes:
        lfp_cutoff_hz: LFP low-pass corner.
        spike_band_hz: Spike band edges.
        filter_order: Butterworth order.
        zero_phase: Forward-backward filtering (offline) versus causal.
        threshold_sigma: Spike threshold in multiples of the noise sigma.
        polarity: Spike polarity to detect.
        refractory_s: Detector dead time.
        psd_resolution_hz: Target Welch bin spacing.
        lfp_floor_hz: Lower limit when integrating LFP power. A 5 s clip cannot
            resolve much below 1 Hz.
        max_saturation_fraction: Clips with a larger fraction of samples on an
            ADC rail are excluded from the statistics.
        drop_bits: Low bits discarded before analysis (0 keeps the data as
            recorded). Used by the prediction checks.
    """

    lfp_cutoff_hz: float = 300.0
    spike_band_hz: tuple[float, float] = (300.0, 7500.0)
    filter_order: int = 4
    zero_phase: bool = True
    threshold_sigma: float = 4.5
    polarity: Polarity = "neg"
    refractory_s: float = 1.0e-3
    psd_resolution_hz: float = 1.2
    lfp_floor_hz: float = 1.0
    max_saturation_fraction: float = 1.0e-3
    drop_bits: int = 0

    def digest(self) -> str:
        """Return a short hash of the configuration, used to name cache files."""
        payload = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:12]


@dataclass(frozen=True)
class ClipResult:
    """What the analysis extracts from one clip.

    Amplitudes and powers are in the recording's units (and units squared).
    """

    clip_id: str
    duration_s: float
    saturation_fraction: float
    dc_offset: float
    sigma: float
    lfp_power: float
    spike_power: float
    n_events: int
    n_opposite_dominant: int
    spike_amplitudes: FloatArray
    freqs_hz: FloatArray
    psd: FloatArray


def analyse_clip(recording: Recording, config: U0Config) -> ClipResult:
    """Analyse one single-channel recording.

    Raises:
        ValueError: If the recording has more than one channel.
    """
    if recording.n_channels != 1:
        raise ValueError("U0 analyses one channel at a time")
    x = recording.channel(0)
    if config.drop_bits:
        x = requantise(x, config.drop_bits)
    dc_offset = float(np.mean(x))
    centred = x - dc_offset

    freqs, psd = welch_psd(centred, recording.fs_hz, resolution_hz=config.psd_resolution_hz)
    bands = band_split(
        x,
        recording.fs_hz,
        lfp_cutoff_hz=config.lfp_cutoff_hz,
        spike_band_hz=config.spike_band_hz,
        order=config.filter_order,
        zero_phase=config.zero_phase,
    )
    detection = detect_spikes(
        bands.spikes,
        recording.fs_hz,
        threshold_sigma=config.threshold_sigma,
        polarity=config.polarity,
        refractory_s=config.refractory_s,
    )
    opposite: Polarity = "pos" if config.polarity == "neg" else "neg"
    opposite_detection = detect_spikes(
        bands.spikes,
        recording.fs_hz,
        threshold_sigma=config.threshold_sigma,
        polarity=opposite,
        refractory_s=config.refractory_s,
        sigma=detection.sigma,
    )
    n_events, n_opposite_dominant = polarity_census(
        detection, opposite_detection, window=round(PHASE_WINDOW_S * recording.fs_hz)
    )
    return ClipResult(
        clip_id=str(recording.meta.get("clip_id", "")),
        duration_s=recording.duration_s,
        saturation_fraction=float(recording.meta.get("saturation_fraction", 0.0)),
        dc_offset=dc_offset,
        sigma=detection.sigma,
        lfp_power=float(band_power(freqs, psd, config.lfp_floor_hz, config.lfp_cutoff_hz)),
        spike_power=float(band_power(freqs, psd, *config.spike_band_hz)),
        n_events=n_events,
        n_opposite_dominant=n_opposite_dominant,
        spike_amplitudes=detection.amplitudes,
        freqs_hz=freqs,
        psd=psd,
    )


_ARRAY_FIELDS = (
    "duration_s",
    "saturation_fraction",
    "dc_offset",
    "sigma",
    "lfp_power",
    "spike_power",
    "n_spikes",
    "n_events",
    "n_opposite_dominant",
    "freqs_hz",
    "psd",
    "spike_amplitudes",
    "spike_clip_index",
)
"""Array attributes of :class:`U0Results` written to the cache file."""


@dataclass(frozen=True)
class U0Results:
    """Per-clip arrays for a whole dataset, plus every detected spike.

    Attributes:
        config: The configuration that produced the results.
        fs_hz: Sampling rate shared by all clips.
        units: Unit of the amplitudes.
        clip_ids: Clip identifiers, in analysis order.
        duration_s: Clip durations.
        saturation_fraction: Fraction of samples on an ADC rail, per clip.
        dc_offset: Mean level per clip.
        sigma: Spike-band noise estimate per clip.
        lfp_power: LFP-band power per clip (units squared).
        spike_power: Spike-band power per clip (units squared).
        n_spikes: Detected spikes per clip.
        n_events: Distinct events of either polarity per clip (two phases of
            one spike count once).
        n_opposite_dominant: Events per clip whose larger phase has the
            opposite polarity to the one detected.
        freqs_hz: Shared PSD frequency axis.
        psd: PSDs, shape ``(n_clips, n_freqs)``.
        spike_amplitudes: Signed peak amplitude of every spike, concatenated.
        spike_clip_index: Index into the clip arrays for every spike.
    """

    config: U0Config
    fs_hz: float
    units: str
    clip_ids: tuple[str, ...]
    duration_s: FloatArray
    saturation_fraction: FloatArray
    dc_offset: FloatArray
    sigma: FloatArray
    lfp_power: FloatArray
    spike_power: FloatArray
    n_spikes: IntArray
    n_events: IntArray
    n_opposite_dominant: IntArray
    freqs_hz: FloatArray
    psd: FloatArray
    spike_amplitudes: FloatArray
    spike_clip_index: IntArray

    @property
    def n_clips(self) -> int:
        """Number of clips analysed."""
        return len(self.clip_ids)

    @property
    def clean(self) -> np.ndarray[Any, np.dtype[np.bool_]]:
        """Boolean mask of clips that pass the saturation check."""
        return np.asarray(self.saturation_fraction <= self.config.max_saturation_fraction)

    @property
    def spike_snr(self) -> FloatArray:
        """Peak magnitude of every spike in multiples of its clip's noise sigma."""
        return np.asarray(
            np.abs(self.spike_amplitudes) / self.sigma[self.spike_clip_index], dtype=np.float64
        )

    @property
    def spike_is_clean(self) -> np.ndarray[Any, np.dtype[np.bool_]]:
        """Boolean mask of spikes that come from clean clips."""
        return np.asarray(self.clean[self.spike_clip_index])

    def save(self, path: Path) -> None:
        """Write the results to a compressed ``.npz`` file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {name: getattr(self, name) for name in _ARRAY_FIELDS}
        np.savez_compressed(
            path,
            config=json.dumps(asdict(self.config)),
            fs_hz=self.fs_hz,
            units=self.units,
            clip_ids=np.asarray(self.clip_ids),
            **arrays,
        )

    @classmethod
    def load(cls, path: Path) -> U0Results:
        """Read results written by :meth:`save`."""
        with np.load(path, allow_pickle=False) as stored:
            raw_config = json.loads(str(stored["config"]))
            raw_config["spike_band_hz"] = tuple(raw_config["spike_band_hz"])
            return cls(
                config=U0Config(**raw_config),
                fs_hz=float(stored["fs_hz"]),
                units=str(stored["units"]),
                clip_ids=tuple(str(c) for c in stored["clip_ids"]),
                duration_s=stored["duration_s"],
                saturation_fraction=stored["saturation_fraction"],
                dc_offset=stored["dc_offset"],
                sigma=stored["sigma"],
                lfp_power=stored["lfp_power"],
                spike_power=stored["spike_power"],
                n_spikes=stored["n_spikes"],
                n_events=stored["n_events"],
                n_opposite_dominant=stored["n_opposite_dominant"],
                freqs_hz=stored["freqs_hz"],
                psd=stored["psd"],
                spike_amplitudes=stored["spike_amplitudes"],
                spike_clip_index=stored["spike_clip_index"],
            )


def analyse_recordings(
    recordings: Iterable[Recording],
    config: U0Config,
    *,
    progress_every: int = 0,
) -> U0Results:
    """Analyse a sequence of single-channel recordings that share a sampling rate.

    Args:
        recordings: The recordings, consumed lazily so only one is in memory.
        config: Analysis parameters.
        progress_every: Print a progress line to standard error every this many
            clips (0 disables it).

    Raises:
        ValueError: If there are no recordings or their sampling rates or
            lengths in PSD bins differ.
    """
    clips: list[ClipResult] = []
    fs_hz: float | None = None
    units = ""
    for i, recording in enumerate(recordings):
        if fs_hz is None:
            fs_hz, units = recording.fs_hz, recording.units
        elif recording.fs_hz != fs_hz:
            raise ValueError("all recordings must share one sampling rate")
        clips.append(analyse_clip(recording, config))
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  analysed {i + 1} clips", file=sys.stderr, flush=True)
    if fs_hz is None:
        raise ValueError("no recordings to analyse")
    if len({c.psd.size for c in clips}) != 1:
        raise ValueError("clips are too short for the requested PSD resolution")

    counts = np.asarray([c.spike_amplitudes.size for c in clips], dtype=np.int64)
    return U0Results(
        config=config,
        fs_hz=fs_hz,
        units=units,
        clip_ids=tuple(c.clip_id for c in clips),
        duration_s=np.asarray([c.duration_s for c in clips]),
        saturation_fraction=np.asarray([c.saturation_fraction for c in clips]),
        dc_offset=np.asarray([c.dc_offset for c in clips]),
        sigma=np.asarray([c.sigma for c in clips]),
        lfp_power=np.asarray([c.lfp_power for c in clips]),
        spike_power=np.asarray([c.spike_power for c in clips]),
        n_spikes=counts,
        n_events=np.asarray([c.n_events for c in clips], dtype=np.int64),
        n_opposite_dominant=np.asarray([c.n_opposite_dominant for c in clips], dtype=np.int64),
        freqs_hz=clips[0].freqs_hz,
        psd=np.stack([c.psd for c in clips]),
        spike_amplitudes=np.concatenate([c.spike_amplitudes for c in clips]),
        spike_clip_index=np.repeat(np.arange(len(clips), dtype=np.int64), counts),
    )


def analyse_paths(
    paths: Sequence[Path],
    loader: Callable[[Path], Recording],
    config: U0Config,
    *,
    cache_dir: Path | None = None,
    progress_every: int = 100,
) -> U0Results:
    """Analyse recordings on disk, reusing a cached result when one exists.

    The cache key covers the configuration and the list of file names, so a
    changed parameter or a different set of clips never reuses stale arrays.
    """
    cache_path: Path | None = None
    if cache_dir is not None:
        names = hashlib.sha256("\n".join(p.name for p in paths).encode()).hexdigest()[:12]
        cache_path = cache_dir / f"u0_{config.digest()}_{names}.npz"
        if cache_path.is_file():
            return U0Results.load(cache_path)
    results = analyse_recordings((loader(p) for p in paths), config, progress_every=progress_every)
    if cache_path is not None:
        results.save(cache_path)
    return results


def _quartiles(values: FloatArray) -> dict[str, float]:
    """Return the median and interquartile range as plain floats."""
    p25, p50, p75 = np.percentile(values, [25, 50, 75])
    return {"median": float(p50), "p25": float(p25), "p75": float(p75)}


def median_psd(results: U0Results) -> FloatArray:
    """Return the across-clip median PSD of the clean clips."""
    return np.asarray(np.median(results.psd[results.clean], axis=0), dtype=np.float64)


def example_clip_index(results: U0Results, *, min_spikes: int = 20) -> int:
    """Pick the clip used for the example traces, by a fixed rule.

    Among clean clips with at least ``min_spikes`` spikes, take the one whose
    median spike SNR is largest. The rule is deterministic, and it deliberately
    selects a clip where spikes are easy to see, not a typical one.

    Raises:
        ValueError: If no clip qualifies.
    """
    snr = results.spike_snr
    best_index, best_snr = -1, -np.inf
    for i in np.flatnonzero(results.clean & (results.n_spikes >= min_spikes)):
        clip_snr = float(np.median(snr[results.spike_clip_index == i]))
        if clip_snr > best_snr:
            best_index, best_snr = int(i), clip_snr
    if best_index < 0:
        raise ValueError(f"no clean clip has at least {min_spikes} spikes")
    return best_index


def summarise(results: U0Results) -> dict[str, Any]:
    """Reduce the per-clip arrays to the numbers reported for U0.

    All statistics use clean clips only, except the saturation counts, which
    describe the whole dataset. The dictionary is JSON-serialisable.

    Raises:
        ValueError: If no clip passes the saturation check.
    """
    clean = results.clean
    if not np.any(clean):
        raise ValueError("no clip passes the saturation check")
    config = results.config
    spikes_clean = results.spike_is_clean
    amplitude = np.abs(results.spike_amplitudes[spikes_clean])
    snr = results.spike_snr[spikes_clean]
    n_spikes = int(amplitude.size)
    n_events = int(results.n_events[clean].sum())
    n_opposite = int(results.n_opposite_dominant[clean].sum())
    duration_clean = float(results.duration_s[clean].sum())
    sigma = results.sigma[clean]
    psd = median_psd(results)
    freqs = results.freqs_hz
    lfp_fraction = results.lfp_power[clean] / (
        results.lfp_power[clean] + results.spike_power[clean]
    )
    floor_band = freqs >= min(8000.0, 0.8 * results.fs_hz / 2.0)
    sigma_median = float(np.median(sigma))

    summary: dict[str, Any] = {
        "config": asdict(config),
        "units": results.units,
        "dataset": {
            "n_clips": results.n_clips,
            "n_clean_clips": int(clean.sum()),
            "n_saturated_clips_excluded": int((~clean).sum()),
            "saturated_sample_fraction_all_clips": float(
                np.average(results.saturation_fraction, weights=results.duration_s)
            ),
            "total_duration_s": float(results.duration_s.sum()),
            "clean_duration_s": duration_clean,
            "fs_hz": results.fs_hz,
            "dc_offset": _quartiles(results.dc_offset[clean]),
        },
        "noise": {
            "sigma": _quartiles(sigma),
            "sigma_over_quantisation_rms": sigma_median / float(QUANTISATION_RMS_LSB),
            "bits_until_lsb_equals_sigma": float(np.log2(sigma_median)),
        },
        "spectrum": {
            "lfp_power_fraction": _quartiles(lfp_fraction),
            "lfp_rms": _quartiles(np.sqrt(results.lfp_power[clean])),
            "spike_band_rms": _quartiles(np.sqrt(results.spike_power[clean])),
            "psd_slope_lfp_band": loglog_slope(freqs, psd, 10.0, config.lfp_cutoff_hz),
            "psd_slope_spike_band": loglog_slope(
                freqs, psd, config.spike_band_hz[0], min(5000.0, config.spike_band_hz[1])
            ),
            "asd_at_10_hz": float(np.sqrt(np.interp(10.0, freqs, psd))),
            "asd_at_1_khz": float(np.sqrt(np.interp(1000.0, freqs, psd))),
            "asd_floor_above_8_khz": float(np.sqrt(np.median(psd[floor_band]))),
        },
        "spikes": {
            "n_spikes": n_spikes,
            "rate_hz_overall": n_spikes / duration_clean,
            "rate_hz_per_clip": _quartiles(results.n_spikes[clean] / results.duration_s[clean]),
            "clips_without_spikes": int(np.sum(results.n_spikes[clean] == 0)),
            "gaussian_noise_crossing_rate_hz": noise_crossing_rate_hz(
                freqs, psd, config.spike_band_hz, config.threshold_sigma, results.fs_hz
            ),
            "detected_polarity_fraction": (
                1.0 - n_opposite / n_events if n_events else float("nan")
            ),
        },
    }
    if n_spikes:
        summary["spikes"]["amplitude"] = {
            **_quartiles(amplitude),
            "p95": float(np.percentile(amplitude, 95)),
        }
        summary["spikes"]["snr"] = {**_quartiles(snr), "p95": float(np.percentile(snr, 95))}
    return summary
