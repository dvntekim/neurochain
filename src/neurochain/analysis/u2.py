"""U2 analysis: ADC sweeps, detection F1 with ground truth, SBP against crossings.

Two parts:

* **MEArec sweeps.** Each channel of the simulated recording goes through the
  U1 front end, the ADC model and the causal detectors. Detections are scored
  against the known spike times while the background noise, the ADC resolution
  and the sample rate are varied one at a time. The same pass measures how well
  each feature (threshold-crossing rate, spiking-band power) tracks the firing
  of the channel's largest unit, and checks the fixed-point paths against their
  floating-point references.
* **Nason reproduction.** On the paper's own data: SBP and threshold-crossing
  rate against each unit's firing in simulation and in vivo, and a linear
  readout of finger velocity from each feature.

Thresholds are calibrated on the first seconds of each channel and scored on
the rest, so no score uses data its threshold has seen.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import signal

from neurochain.adc import Adc, convert
from neurochain.data import nason
from neurochain.data.mearec import MearecRecording
from neurochain.decode.ridge import CrossValidation, blocked_cross_validation
from neurochain.dsp.filters import band_split
from neurochain.dsp.spikes import detect_spikes, mad_sigma, mad_sigma_integer
from neurochain.features import detect as det
from neurochain.features.sbp import (
    SBP_FS_HZ,
    bin_counts,
    gaussian_window,
    smooth,
    spiking_band_power,
)
from neurochain.features.score import DetectionScore, score_detections
from neurochain.fixedpoint import (
    accumulator_bits,
    design_fixed_highpass,
    fixed_highpass,
    fixed_neo_score,
    fixed_sbp,
    fixed_threshold_events,
    quantisation_snr_db,
)
from neurochain.frontend.emulator import emulate
from neurochain.frontend.model import FrontEnd, baseline_frontend, design_lna
from neurochain.recording import FloatArray, IntArray

METHODS: tuple[det.Method, ...] = ("threshold", "neo")
ScoreKey = tuple[str, float, str]


@dataclass(frozen=True)
class U2Config:
    """Parameters of the U2 analysis.

    Attributes:
        seed: Seed for every stochastic step.
        noise_levels_uv: Spike-band RMS of the neural background to test.
        reference_noise_uv: The in-vivo-like level the ADC sweeps are run at
            (the Nason channel measures 13.2 µV).
        bits_options: ADC resolutions to test.
        fs_options_hz: ADC sample rates to test (whole fractions of 30 kHz).
        threshold_factors: Threshold detector settings, in noise sigmas.
        neo_factors: NEO detector settings, in multiples of its mean.
        threshold_factor: The conventional threshold setting.
        neo_factor: The conventional NEO setting.
        target_amplitude_uv: A unit must be at least this deep on a channel for
            its spikes to be required of the detector (MEArec's own default for
            a detectable unit).
        ignore_amplitude_uv: Spikes of smaller units, down to this depth, excuse
            a detection without rewarding it.
        calibration_s: Leading seconds used to set thresholds, then discarded.
        tolerance_s: Largest timing error for a detection to count.
        smooth_s: Width of the Gaussian used before correlating a feature with
            a firing rate (Nason et al. use 50 ms).
        alias_zones: Nyquist zones of amplifier noise folded in by the ADC.
        frac_bits_options: Coefficient widths tried for the fixed-point filter.
        guard_bits_options: Guard-bit counts tried for the fixed-point filter.
        max_workers: Parallel processes (defaults to the CPU count, at most 8).
    """

    seed: int = 0
    noise_levels_uv: tuple[float, ...] = (0.0, 5.0, 10.0, 13.0, 20.0, 30.0)
    reference_noise_uv: float = 13.0
    bits_options: tuple[int, ...] = (3, 4, 5, 6, 7, 8, 10, 12)
    fs_options_hz: tuple[float, ...] = (30000.0, 15000.0, 10000.0, 7500.0, 6000.0, 5000.0, 3000.0)
    threshold_factors: tuple[float, ...] = (3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0)
    neo_factors: tuple[float, ...] = (3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0)
    threshold_factor: float = 4.5
    neo_factor: float = 8.0
    target_amplitude_uv: float = 50.0
    ignore_amplitude_uv: float = 10.0
    calibration_s: float = 10.0
    tolerance_s: float = 1.0e-3
    smooth_s: float = 0.05
    alias_zones: int = 8
    frac_bits_options: tuple[int, ...] = (4, 6, 8, 10, 12, 14)
    guard_bits_options: tuple[int, ...] = (0, 2, 4, 6, 8)
    max_workers: int | None = None

    def factor(self, method: det.Method) -> float:
        """Return the conventional setting of a detector."""
        return self.threshold_factor if method == "threshold" else self.neo_factor


@dataclass(frozen=True)
class ChannelJob:
    """Everything one worker needs to analyse one channel."""

    channel: int
    fs_hz: float
    clean_uv: FloatArray
    background_uv: FloatArray
    targets: IntArray
    ignore: IntArray
    dominant: IntArray
    wideband: FrontEnd
    narrowband: FrontEnd
    config: U2Config


@dataclass(frozen=True)
class ChannelResult:
    """One channel's scores, feature correlations and fixed-point checks."""

    channel: int
    scores: dict[ScoreKey, DetectionScore]
    correlations: dict[tuple[str, float], float]
    fixed: dict[str, float]


def _score(job: ChannelJob, events: IntArray, fs_hz: float) -> DetectionScore:
    """Score events found at ``fs_hz`` against the job's ground truth."""
    at_base = np.rint(events * (job.fs_hz / fs_hz)).astype(np.int64)
    return score_detections(
        at_base,
        job.targets,
        tolerance=round(job.config.tolerance_s * job.fs_hz),
        ignore=job.ignore,
        start=round(job.config.calibration_s * job.fs_hz),
    )


def _correlation(job: ChannelJob, feature: FloatArray, window: FloatArray) -> float:
    """Correlate a 2 kHz feature with the dominant unit's smoothed firing."""
    per_bin = round(job.fs_hz / SBP_FS_HZ)
    truth = bin_counts(job.dominant, job.clean_uv.size, per_bin)
    n = min(truth.size, feature.size)
    start = round(job.config.calibration_s * SBP_FS_HZ)
    a = smooth(feature[:n], window)[start:]
    b = smooth(truth[:n], window)[start:]
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return 0.0  # a feature that never changes carries no information
    return float(np.corrcoef(a, b)[0, 1])


def analyse_channel(job: ChannelJob) -> ChannelResult:
    """Run every sweep on one channel."""
    cfg = job.config
    fs = job.fs_hz
    base_adc = Adc()
    per_bin = round(fs / SBP_FS_HZ)
    window = gaussian_window(cfg.smooth_s, SBP_FS_HZ)
    scores: dict[ScoreKey, DetectionScore] = {}
    correlations: dict[tuple[str, float], float] = {}
    fixed: dict[str, float] = {}

    for index, noise in enumerate(cfg.noise_levels_uv):
        seed = cfg.seed * 100_000 + job.channel * 100 + index
        x = job.clean_uv + noise * job.background_uv
        analog = emulate(x, fs, job.wideband, seed=seed, alias_zones=cfg.alias_zones)
        analog_uv = analog.input_referred_uv
        codes = convert(analog_uv, fs, base_adc)
        digital = codes.codes.astype(np.float64)
        calibration = round(cfg.calibration_s * fs)

        for method in METHODS:
            events, _ = det.detect(
                digital, fs, method, factor=cfg.factor(method), calibration_samples=calibration
            )
            scores[("noise", noise, method)] = _score(job, events, fs)
            if method == "threshold":
                correlations[("crossings", noise)] = _correlation(
                    job, bin_counts(events, digital.size, per_bin), window
                )

        # What an offline analysis reports: zero-phase filter, same threshold rule.
        offline = band_split(analog_uv, fs).spikes
        sigma = mad_sigma(offline[:calibration])
        offline_events = detect_spikes(
            offline, fs, threshold_sigma=cfg.threshold_factor, sigma=sigma
        ).indices
        scores[("noise", noise, "offline")] = _score(job, offline_events, fs)

        correlations[("sbp_digital", noise)] = _correlation(
            job, spiking_band_power(codes.microvolts, fs), window
        )
        narrow = emulate(x, fs, job.narrowband, seed=seed + 50, alias_zones=cfg.alias_zones)
        narrow_codes = convert(narrow.input_referred_uv, fs, replace(base_adc, fs_hz=SBP_FS_HZ))
        correlations[("sbp_implant", noise)] = _correlation(
            job, np.abs(narrow_codes.codes).astype(np.float64), window
        )

        if noise != cfg.reference_noise_uv:
            continue

        for bits in cfg.bits_options:
            swept = convert(analog_uv, fs, replace(base_adc, bits=bits)).codes.astype(np.float64)
            for method in METHODS:
                events, _ = det.detect(
                    swept, fs, method, factor=cfg.factor(method), calibration_samples=calibration
                )
                scores[("bits", float(bits), method)] = _score(job, events, fs)
        for rate in cfg.fs_options_hz:
            swept = convert(analog_uv, fs, replace(base_adc, fs_hz=rate)).codes.astype(np.float64)
            for method in METHODS:
                events, _ = det.detect(
                    swept,
                    rate,
                    method,
                    factor=cfg.factor(method),
                    calibration_samples=round(cfg.calibration_s * rate),
                )
                scores[("fs", rate, method)] = _score(job, events, rate)
        for method, factors in (("threshold", cfg.threshold_factors), ("neo", cfg.neo_factors)):
            for factor in factors:
                events, _ = det.detect(
                    digital, fs, method, factor=factor, calibration_samples=calibration
                )
                scores[("factor", factor, method)] = _score(job, events, fs)

        fixed.update(_fixed_point_checks(job, codes.codes, narrow_codes.codes, window))

    return ChannelResult(job.channel, scores, correlations, fixed)


def _fixed_point_checks(
    job: ChannelJob, wide_codes: IntArray, narrow_codes: IntArray, window: FloatArray
) -> dict[str, float]:
    """Compare integer-only feature paths with their floating-point references."""
    cfg = job.config
    fs = job.fs_hz
    calibration = round(cfg.calibration_s * fs)
    out: dict[str, float] = {}

    b, a = signal.butter(2, det.HIGHPASS_HZ, btype="highpass", fs=fs)
    reference = signal.lfilter(b, a, wide_codes.astype(np.float64))
    settled = slice(calibration, None)
    for frac in cfg.frac_bits_options:
        filt = design_fixed_highpass(fs, frac_bits=frac, guard_bits=6)
        y, _ = fixed_highpass(wide_codes, filt)
        out[f"snr_wide_frac{frac}"] = quantisation_snr_db(reference[settled], y[settled])
    for guard in cfg.guard_bits_options:
        for rounding in (True, False):
            filt = design_fixed_highpass(fs, frac_bits=14, guard_bits=guard, rounding=rounding)
            y, _ = fixed_highpass(wide_codes, filt)
            label = "round" if rounding else "trunc"
            out[f"snr_wide_{label}_guard{guard}"] = quantisation_snr_db(
                reference[settled], y[settled]
            )
            out[f"offset_wide_{label}_guard{guard}"] = float(
                np.mean(y[settled] - reference[settled])
            )

    # The threshold path as hardware would run it: 14-bit coefficients, 6 guard bits.
    filt = design_fixed_highpass(fs, frac_bits=14, guard_bits=6)
    filtered, peak = fixed_highpass(wide_codes, filt)
    out["acc_bits_wide"] = float(accumulator_bits(peak))
    threshold = round(cfg.threshold_factor * mad_sigma_integer(filtered[:calibration]))
    events = fixed_threshold_events(filtered, threshold, round(1.0e-3 * fs))
    score = _score(job, events, fs)
    out.update(fixed_threshold_tp=score.true_positives, fixed_threshold_fp=score.false_positives,
               fixed_threshold_fn=score.false_negatives)  # fmt: skip
    neo_window = max(1, round(det.NEO_SMOOTH_S * fs))
    neo_score = fixed_neo_score(filtered, neo_window)
    neo_threshold = round(cfg.neo_factor * float(np.mean(neo_score[:calibration])))
    neo_events = det.detect_events(neo_score.astype(np.float64), neo_threshold, fs)
    score = _score(job, neo_events, fs)
    out.update(fixed_neo_tp=score.true_positives, fixed_neo_fp=score.false_positives,
               fixed_neo_fn=score.false_negatives)  # fmt: skip

    # The SBP path: 2 kS/s, 8-bit coefficients, 2 guard bits, sum of |x| per 50 ms.
    b2, a2 = signal.butter(2, det.HIGHPASS_HZ, btype="highpass", fs=SBP_FS_HZ)
    narrow_reference = signal.lfilter(b2, a2, narrow_codes.astype(np.float64))
    for frac in cfg.frac_bits_options:
        y, _ = fixed_highpass(
            narrow_codes, design_fixed_highpass(SBP_FS_HZ, frac_bits=frac, guard_bits=6)
        )
        out[f"snr_narrow_frac{frac}"] = quantisation_snr_db(narrow_reference[2000:], y[2000:])
    for rounding in (True, False):
        y, _ = fixed_highpass(
            narrow_codes,
            design_fixed_highpass(SBP_FS_HZ, frac_bits=14, guard_bits=0, rounding=rounding),
        )
        label = "round" if rounding else "trunc"
        out[f"offset_narrow_{label}_guard0"] = float(np.mean(y[2000:] - narrow_reference[2000:]))
    narrow_filter = design_fixed_highpass(SBP_FS_HZ, frac_bits=8, guard_bits=2)
    narrow_filtered, narrow_peak = fixed_highpass(narrow_codes, narrow_filter)
    out["acc_bits_narrow"] = float(accumulator_bits(narrow_peak))
    per_bin = round(cfg.smooth_s * SBP_FS_HZ)
    fixed_feature = fixed_sbp(narrow_filtered, per_bin).astype(np.float64)
    float_feature = np.abs(narrow_reference)[: fixed_feature.size * per_bin]
    float_feature = float_feature.reshape(-1, per_bin).sum(axis=1)
    out["sbp_fixed_vs_float_r"] = float(np.corrcoef(fixed_feature[4:], float_feature[4:])[0, 1])
    out["sbp_fixed_tracks_unit_r"] = _correlation(
        job, np.abs(narrow_filtered).astype(np.float64), window
    )
    return out


def _f1_summary(
    per_channel: list[DetectionScore], rng: np.random.Generator, n_boot: int = 1000
) -> dict[str, float]:
    """Pool counts over channels and bootstrap the channels for a 95 % interval."""
    pooled = DetectionScore(0, 0, 0)
    for s in per_channel:
        pooled = pooled + s
    counts = np.array(
        [[s.true_positives, s.false_positives, s.false_negatives] for s in per_channel]
    )
    draws = rng.integers(0, len(per_channel), size=(n_boot, len(per_channel)))
    sums = counts[draws].sum(axis=1)
    f1 = 2 * sums[:, 0] / np.maximum(2 * sums[:, 0] + sums[:, 1] + sums[:, 2], 1)
    low, high = np.percentile(f1, [2.5, 97.5])
    return {
        "f1": pooled.f1,
        "f1_low": float(low),
        "f1_high": float(high),
        "precision": pooled.precision,
        "recall": pooled.recall,
    }


def _mean_summary(
    values: list[float], rng: np.random.Generator, n_boot: int = 1000
) -> dict[str, float]:
    """Return the mean over channels with a bootstrap 95 % interval."""
    array = np.asarray(values, dtype=np.float64)
    draws = rng.integers(0, array.size, size=(n_boot, array.size))
    low, high = np.percentile(array[draws].mean(axis=1), [2.5, 97.5])
    return {"mean": float(array.mean()), "low": float(low), "high": float(high)}


def narrowband_frontend() -> FrontEnd:
    """Return a front end whose pass band is the spiking band, 300 Hz to 1 kHz."""
    base = baseline_frontend()
    return replace(base, lna=design_lna(f_high_hz=1000.0, f_low_hz=300.0))


def mearec_sweeps(recording: MearecRecording, config: U2Config) -> dict[str, Any]:
    """Run every MEArec sweep over all channels and summarise them."""
    jobs = channel_jobs(recording, config)
    results: list[ChannelResult] = _map(analyse_channel, jobs, config)

    rng = np.random.default_rng(config.seed)
    keys = sorted(results[0].scores)
    detection: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for sweep, value, method in keys:
        summary = _f1_summary([r.scores[(sweep, value, method)] for r in results], rng)
        detection.setdefault(sweep, {}).setdefault(method, {})[f"{value:g}"] = summary
    tracking: dict[str, dict[str, dict[str, float]]] = {}
    for feature, noise in sorted(results[0].correlations):
        tracking.setdefault(feature, {})[f"{noise:g}"] = _mean_summary(
            [r.correlations[(feature, noise)] for r in results], rng
        )
    fixed_keys = sorted(results[0].fixed)
    fixed = {key: float(np.mean([r.fixed[key] for r in results])) for key in fixed_keys}
    for method in METHODS:
        pooled = DetectionScore(
            int(sum(r.fixed[f"fixed_{method}_tp"] for r in results)),
            int(sum(r.fixed[f"fixed_{method}_fp"] for r in results)),
            int(sum(r.fixed[f"fixed_{method}_fn"] for r in results)),
        )
        fixed[f"fixed_{method}_f1"] = pooled.f1
        for part in ("tp", "fp", "fn"):
            del fixed[f"fixed_{method}_{part}"]
    for key in ("acc_bits_wide", "acc_bits_narrow"):
        fixed[key] = float(max(r.fixed[key] for r in results))
    return {
        "recording": {
            "n_channels": recording.n_channels,
            "n_units": recording.n_units,
            "duration_s": recording.duration_s,
            "fs_hz": recording.fs_hz,
            "scored_s": recording.duration_s - config.calibration_s,
            "target_spikes": int(
                sum(np.sum(j.targets >= config.calibration_s * j.fs_hz) for j in jobs)
            ),
        },
        "detection": detection,
        "tracking": tracking,
        "fixed_point": fixed,
    }


def _setting_job(
    args: tuple[ChannelJob, float, Adc, dict[str, float]],
) -> dict[str, DetectionScore]:
    """Score one channel at one noise level, ADC and pair of detector settings."""
    job, noise, adc, factors = args
    cfg = job.config
    x = job.clean_uv + noise * job.background_uv
    seed = cfg.seed * 100_000 + job.channel * 100 + 99
    analog = emulate(x, job.fs_hz, job.wideband, seed=seed, alias_zones=cfg.alias_zones)
    codes = convert(analog.input_referred_uv, job.fs_hz, adc).codes.astype(np.float64)
    out: dict[str, DetectionScore] = {}
    for method in METHODS:
        events, _ = det.detect(
            codes,
            adc.fs_hz,
            method,
            factor=factors[method],
            calibration_samples=round(cfg.calibration_s * adc.fs_hz),
        )
        out[method] = _score(job, events, adc.fs_hz)
    return out


def channel_jobs(recording: MearecRecording, config: U2Config) -> list[ChannelJob]:
    """Build one job per channel: its signals, ground truth and front ends."""
    wide, narrow = baseline_frontend(), narrowband_frontend()
    amplitudes = recording.amplitudes_uv
    jobs = []
    for channel in range(recording.n_channels):
        targets = recording.units_above(channel, config.target_amplitude_uv)
        smaller = np.setdiff1d(recording.units_above(channel, config.ignore_amplitude_uv), targets)
        jobs.append(
            ChannelJob(
                channel=channel,
                fs_hz=recording.fs_hz,
                clean_uv=recording.clean_uv[channel],
                background_uv=recording.background_uv[channel],
                targets=recording.spikes_of(targets),
                ignore=recording.spikes_of(smaller),
                dominant=recording.spike_samples[int(np.argmax(amplitudes[:, channel]))],
                wideband=wide,
                narrowband=narrow,
                config=config,
            )
        )
    return jobs


def _map(function: Any, jobs: list[Any], config: U2Config) -> list[Any]:
    """Run jobs in worker processes, or inline when only one worker is allowed."""
    workers = config.max_workers or min(8, os.cpu_count() or 1)
    if workers <= 1:
        return [function(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(function, jobs))


def evaluate_setting(
    recording: MearecRecording,
    config: U2Config,
    *,
    noise_uv: float,
    adc: Adc,
    threshold_factor: float | None = None,
    neo_factor: float | None = None,
) -> dict[str, dict[str, float]]:
    """Score both detectors at a single noise level and ADC setting.

    Used by the key-number script to answer "what if" questions without
    rerunning every sweep.
    """
    factors = {
        "threshold": threshold_factor or config.threshold_factor,
        "neo": neo_factor or config.neo_factor,
    }
    jobs = [(job, noise_uv, adc, factors) for job in channel_jobs(recording, config)]
    results = _map(_setting_job, jobs, config)
    rng = np.random.default_rng(config.seed)
    return {method: _f1_summary([r[method] for r in results], rng) for method in METHODS}


def _crossing_rate(x: FloatArray, fs_hz: float, n_bins: int, factor: float) -> FloatArray:
    """Count causal threshold crossings in 2 kHz bins."""
    # The amplifier's 7.5 kHz roll-off, which the detector would sit behind.
    b, a = signal.butter(1, 7500.0, btype="lowpass", fs=fs_hz)
    band_limited = signal.lfilter(b, a, x)
    events, _ = det.detect(
        band_limited, fs_hz, "threshold", factor=factor, calibration_samples=x.size
    )
    bins = events // round(fs_hz / SBP_FS_HZ)
    return np.bincount(bins, minlength=n_bins)[:n_bins].astype(np.float64)


def nason_simulation(config: U2Config) -> dict[str, Any]:
    """Correlate SBP and crossing rate with each simulated unit's firing.

    For every run of both arrangements the SBP is recomputed from the simulated
    broadband, checked against the stored one, and correlated with each unit's
    smoothed firing, as is the rate of causal threshold crossings at 30 kS/s.
    """
    out = []
    for arrangement in nason.load_simulations():
        sbp_r, crossing_r, worst = [], [], 0.0
        for run in arrangement.simulations:
            x = run.broadband()
            mine = spiking_band_power(x, arrangement.fs_hz)
            worst = max(worst, float(np.max(np.abs(mine - run.sbp))))
            smoothed_sbp = smooth(mine, arrangement.smooth_window)
            crossings = smooth(
                _crossing_rate(x, arrangement.fs_hz, mine.size, config.threshold_factor),
                arrangement.smooth_window,
            )
            rates = run.smoothed_rate
            sbp_r.append(
                [np.corrcoef(smoothed_sbp, rates[:, u])[0, 1] for u in range(rates.shape[1])]
            )
            crossing_r.append(
                [np.corrcoef(crossings, rates[:, u])[0, 1] for u in range(rates.shape[1])]
            )
        out.append(
            {
                "snrs": arrangement.snrs.tolist(),
                "sbp_r": np.mean(sbp_r, axis=0).tolist(),
                "crossings_r": np.mean(crossing_r, axis=0).tolist(),
                "sbp_reproduction_max_error": worst,
                "n_runs": len(arrangement.simulations),
            }
        )
    return {"arrangements": out}


def nason_in_vivo(config: U2Config) -> dict[str, Any]:
    """Correlate SBP and crossing rate with the three sorted units of the in-vivo channel."""
    data = nason.load_in_vivo()
    mine = spiking_band_power(data.broadband_uv, data.fs_hz)
    crossings = smooth(
        _crossing_rate(data.broadband_uv, data.fs_hz, mine.size, config.threshold_factor),
        data.smooth_window,
    )
    smoothed_sbp = smooth(mine, data.smooth_window)
    rates = data.smoothed_rate
    return {
        "unit_amplitude_uv": data.unit_amp_uv.tolist(),
        "sbp_r": [float(np.corrcoef(smoothed_sbp, rates[:, u])[0, 1]) for u in range(data.n_units)],
        "crossings_r": [
            float(np.corrcoef(crossings, rates[:, u])[0, 1]) for u in range(data.n_units)
        ],
        "sbp_reproduction_max_error_v": float(np.max(np.abs(mine * 1e-6 - data.sbp_v))),
    }


def online_features(bin_ms: int = 50) -> dict[str, NDArray[Any]]:
    """Bin the Nason online file into SBP, crossing counts and finger kinematics.

    SBP is the summed absolute value divided by the number of samples received
    in the bin, which removes the varying sample count of the original stream.
    Finger velocity is the difference of binned position within each trial.
    """
    online = nason.load_online()
    sbp, crossings, position, velocity, groups = [], [], [], [], []
    for index, trial in enumerate(online.trials):
        n = trial.n_bins // bin_ms
        if n < 2:
            continue
        use = slice(0, n * bin_ms)
        width = trial.sample_width[use].reshape(n, bin_ms).sum(axis=1).astype(np.float64)
        sbp.append(
            trial.sbp[use].reshape(n, bin_ms, -1).sum(axis=1) / np.maximum(width, 1.0)[:, None]
        )
        crossings.append(
            trial.crossing_counts()[use].reshape(n, bin_ms, -1).sum(axis=1).astype(float)
        )
        angle = trial.finger_angle[use].reshape(n, bin_ms).mean(axis=1).astype(np.float64)
        position.append(angle)
        velocity.append(np.gradient(angle, bin_ms / 1000.0))
        groups.append(np.full(n, index, dtype=np.int64))
    return {
        "sbp": np.concatenate(sbp),
        "crossings": np.concatenate(crossings),
        "position": np.concatenate(position),
        "velocity": np.concatenate(velocity),
        "groups": np.concatenate(groups),
    }


def nason_online() -> dict[str, Any]:
    """Decode finger velocity from SBP and from crossings with the same linear readout."""
    data = online_features()
    out: dict[str, Any] = {"n_bins": int(data["groups"].size), "bin_ms": 50, "n_folds": 5}
    per_channel = [
        float(np.corrcoef(data["sbp"][:, c], data["crossings"][:, c])[0, 1])
        for c in range(data["sbp"].shape[1])
    ]
    out["sbp_vs_crossings_r_median"] = float(np.median(per_channel))
    feature_sets = {
        "sbp": data["sbp"],
        "crossings": data["crossings"],
        "both": np.concatenate([data["sbp"], data["crossings"]], axis=1),
    }
    for target in ("velocity", "position"):
        for name, features in feature_sets.items():
            cv: CrossValidation = blocked_cross_validation(
                features, data[target], data["groups"].astype(np.int64)
            )
            out[f"{target}_{name}"] = {
                "r2": cv.r2,
                "r2_sem": cv.r2_sem,
                "r2_folds": cv.r2_folds.tolist(),
                "correlation": cv.correlation,
            }
    return out


def compute_u2(
    *,
    config: U2Config | None = None,
    recording: MearecRecording | None = None,
    include_nason: bool = True,
) -> dict[str, Any]:
    """Run the U2 analysis and return a JSON-serialisable summary.

    Args:
        config: Analysis parameters.
        recording: The MEArec recording, or ``None`` to skip the sweeps.
        include_nason: Run the Nason reproduction (needs the Nason files).
    """
    config = config or U2Config()
    summary: dict[str, Any] = {"config": asdict(config), "mearec": None, "nason": None}
    if recording is not None:
        summary["mearec"] = mearec_sweeps(recording, config)
    if include_nason:
        summary["nason"] = {
            "simulation": nason_simulation(config),
            "in_vivo": nason_in_vivo(config),
            "online": nason_online(),
        }
    return summary
