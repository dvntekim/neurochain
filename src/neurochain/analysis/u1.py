"""U1 analysis: electrode impedance, the noise budget three ways, coupling, emulator.

:func:`compute_u1` gathers everything the U1 figures and results table need:

* each electrode's impedance, analytic and from ngspice;
* the spike-band noise budget from the hand calculation, the analytic network
  model and ngspice, which is the unit's done-check;
* DC-offset tolerance and settling for AC and DC coupling;
* the emulator's noise against the model, on silence and on the Nason
  broadband recording.

ngspice and the Nason file are both optional. Without them the corresponding
entries are ``None`` and the figures leave those series out.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
from scipy import signal

from neurochain.data.synthetic import make_synthetic_recording
from neurochain.dsp.filters import band_split, design_bandpass
from neurochain.dsp.spectra import band_power, welch_psd
from neurochain.dsp.spikes import detect_spikes, mad_sigma
from neurochain.frontend import coupling, spice
from neurochain.frontend.electrode import (
    UTAH_PT_IN_VIVO,
    ComplexArray,
    Electrode,
    integrate_psd,
    log_grid,
)
from neurochain.frontend.emulator import emulate, reference_gain
from neurochain.frontend.hand import HandCalculation, hand_calculation
from neurochain.frontend.lna import nef_from_noise
from neurochain.frontend.model import (
    SPIKE_BAND_HZ,
    FrontEnd,
    NoiseBudget,
    NoiseSpectrum,
    baseline_frontend,
    design_lna,
)
from neurochain.frontend.physics import ROOM_TEMPERATURE_K
from neurochain.recording import FloatArray, Recording

ELECTRODE_KEYS = ("in_saline", "ideal_capacitor", "implanted")
"""Electrodes compared in U1; the first is the baseline."""

HARRISON_NOISE_V = 2.2e-6
HARRISON_BANDWIDTH_HZ = 7200.0
HARRISON_F_LOW_HZ = 0.025
HARRISON_POWER_W = 80.0e-6
HARRISON_SUPPLY_V = 5.0
"""Harrison and Charles (2003): abstract values, plus the supply as recalled (see docs)."""


@dataclass(frozen=True)
class U1Config:
    """Parameters of the U1 analysis.

    Attributes:
        seed: Seed for every stochastic step.
        step_v: Size of the electrode offset step in the settling test.
        f_low_options_hz: High-pass corners compared in the settling test.
        offset_sweep_v: Electrode offsets tried in the coupling sweep.
        signal_allowance_v: Input range reserved for the signal itself when
            sizing a DC-coupled gain.
        worst_offset_v: Largest electrode offset the DC-coupled design must
            survive.
        emulator_fs_hz: Sampling rate of the emulator's noise-only test.
        emulator_seconds: Length of that test.
    """

    seed: int = 0
    step_v: float = 50.0e-3
    f_low_options_hz: tuple[float, ...] = (1.0, 0.1)
    offset_sweep_v: tuple[float, ...] = (
        0.0, 1e-3, 2e-3, 5e-3, 8e-3, 10e-3, 20e-3, 50e-3, 100e-3, 200e-3, 300e-3,
    )  # fmt: skip
    signal_allowance_v: float = 20.0e-3
    worst_offset_v: float = 300.0e-3
    emulator_fs_hz: float = 30_000.0
    emulator_seconds: float = 60.0


@dataclass(frozen=True)
class ImpedanceCurve:
    """An electrode's impedance from the analytic model and, if run, ngspice."""

    electrode: Electrode
    freqs_hz: FloatArray
    analytic_ohm: ComplexArray
    spice_freqs_hz: FloatArray | None
    spice_ohm: ComplexArray | None

    @property
    def spice_max_error(self) -> float | None:
        """Largest relative difference between ngspice and the analytic model."""
        if self.spice_freqs_hz is None or self.spice_ohm is None:
            return None
        reference = self.electrode.impedance(self.spice_freqs_hz)
        return float(np.max(np.abs(self.spice_ohm / reference - 1.0)))


@dataclass(frozen=True)
class NoiseComparison:
    """The spike-band noise budget of one electrode, three ways."""

    hand: HandCalculation
    analytic: NoiseBudget
    spice: NoiseBudget | None

    def totals_v(self) -> dict[str, float]:
        """Return the total noise from each available method."""
        totals = {"hand": self.hand.total_v, "analytic": self.analytic.total_v}
        if self.spice is not None:
            totals["spice"] = self.spice.total_v
        return totals

    @property
    def max_disagreement(self) -> float:
        """Largest relative spread between any two methods' totals."""
        values = np.array(list(self.totals_v().values()))
        return float(values.max() / values.min() - 1.0)


@dataclass(frozen=True)
class SettlingRun:
    """The emulated response to an electrode offset step at one high-pass corner."""

    f_low_hz: float
    time_s: FloatArray
    residual_uv: FloatArray
    rail_uv: float
    saturation_time_simulated_s: float
    saturation_time_analytic_s: float
    recovery_to_noise_s: float
    time_constant_s: float


@dataclass(frozen=True)
class OffsetSweep:
    """Output DC level and railed fraction against electrode offset, per configuration.

    ``output_level`` is the median output as a share of the rail (1 is railed).
    """

    offsets_v: FloatArray
    saturated_fraction: dict[str, FloatArray]
    output_level: dict[str, FloatArray]
    gains: dict[str, float]


@dataclass(frozen=True)
class NasonCheck:
    """The emulator applied to the Nason in-vivo broadband recording."""

    fs_hz: float
    freqs_hz: FloatArray
    psd_recorded: FloatArray
    psd_emulated: FloatArray
    model_input_psd: FloatArray
    sigma_recorded_uv: float
    sigma_band_limited_uv: float
    sigma_emulated_uv: float
    sigma_predicted_uv: float
    sigma_quarter_bias_uv: float
    quarter_bias_power_w: float
    detections_recorded: int
    detections_emulated: int


@dataclass(frozen=True)
class U1Results:
    """Everything the U1 figures and table are drawn from."""

    config: U1Config
    frontend: FrontEnd
    impedance: dict[str, ImpedanceCurve]
    noise: dict[str, NoiseComparison]
    spectrum: NoiseSpectrum
    spice_spectra: dict[str, spice.SpiceFrontEnd] | None
    settling: tuple[SettlingRun, ...]
    offsets: OffsetSweep
    nason: NasonCheck | None
    summary: dict[str, Any]


def electrodes(frontend: FrontEnd) -> dict[str, Electrode]:
    """Return the electrodes compared in U1, keyed as in :data:`ELECTRODE_KEYS`.

    Raises:
        ValueError: If the front end has no electrode.
    """
    if frontend.electrode is None:
        raise ValueError("U1 needs a front end with an electrode")
    return {
        "in_saline": frontend.electrode,
        "ideal_capacitor": frontend.electrode.with_ideal_capacitor(),
        "implanted": UTAH_PT_IN_VIVO,
    }


def _spice_budget(frontend: FrontEnd) -> tuple[NoiseBudget, dict[str, spice.SpiceFrontEnd]]:
    runs = {
        source: spice.simulate_frontend(frontend, sources=source)
        for source in ("all", "electrode", "lna")
    }
    budget = NoiseBudget(
        electrode_v=runs["electrode"].input_noise_total_v,
        lna_v=runs["lna"].input_noise_total_v,
        total_v=runs["all"].input_noise_total_v,
    )
    return budget, runs


def impedance_curve(electrode: Electrode, *, use_spice: bool) -> ImpedanceCurve:
    """Compute an electrode's impedance from 1 Hz to 10 kHz, with ngspice if asked."""
    freqs = log_grid(1.0, 1.0e4, points_per_decade=60)
    spice_freqs, spice_ohm = spice.simulate_impedance(electrode) if use_spice else (None, None)
    return ImpedanceCurve(electrode, freqs, electrode.impedance(freqs), spice_freqs, spice_ohm)


def settling_run(frontend: FrontEnd, f_low_hz: float, config: U1Config) -> SettlingRun:
    """Emulate an electrode offset step and time how long the output is railed."""
    amplifier = replace(frontend, lna=frontend.lna.with_f_low(f_low_hz))
    tau = coupling.highpass_time_constant_s(f_low_hz)
    fs_hz = config.emulator_fs_hz
    start = round(0.5 * fs_hz)
    n = start + round((14.0 * tau + 1.0) * fs_hz)
    offset = np.zeros(n)
    offset[start:] = config.step_v
    out = emulate(np.zeros(n), fs_hz, amplifier, seed=config.seed, dc_offset_v=offset)
    block = round(fs_hz / 200.0)
    usable = (n // block) * block
    residual = np.abs(out.input_referred_uv[:usable].reshape(-1, block).mean(axis=1))
    swing = amplifier.lna.output_swing_v
    noise_v = amplifier.band_noise().total_v
    return SettlingRun(
        f_low_hz=f_low_hz,
        time_s=np.asarray(
            (np.arange(residual.size) + 0.5) * block / fs_hz - start / fs_hz, dtype=np.float64
        ),
        residual_uv=residual,
        rail_uv=swing / out.gain * 1.0e6,
        saturation_time_simulated_s=float(np.sum(np.abs(out.output_v) >= swing) / fs_hz),
        saturation_time_analytic_s=coupling.saturation_time_s(
            config.step_v, out.gain, swing, amplifier.f_low_hz
        ),
        recovery_to_noise_s=coupling.recovery_time_s(config.step_v, noise_v, amplifier.f_low_hz),
        time_constant_s=tau,
    )


def offset_sweep(frontend: FrontEnd, config: U1Config) -> OffsetSweep:
    """Emulate a synthetic recording at each electrode offset, AC and DC coupled."""
    recording, _ = make_synthetic_recording(
        duration_s=2.0, fs_hz=config.emulator_fs_hz, seed=config.seed
    )
    x = recording.channel(0)
    swing = frontend.lna.output_swing_v
    high_gain = reference_gain(frontend)
    low_gain = coupling.max_gain_for_offset(
        config.worst_offset_v + config.signal_allowance_v, swing
    )
    setups: dict[str, tuple[str, float]] = {
        "AC-coupled": ("ac", high_gain),
        "DC-coupled, same gain": ("dc", high_gain),
        "DC-coupled, low gain": ("dc", low_gain),
    }
    fractions: dict[str, FloatArray] = {}
    levels: dict[str, FloatArray] = {}
    for label, (mode, gain) in setups.items():
        runs = [
            emulate(
                x,
                recording.fs_hz,
                frontend,
                seed=config.seed,
                dc_offset_v=offset,
                coupling="ac" if mode == "ac" else "dc",
                dc_gain=gain,
            )
            for offset in config.offset_sweep_v
        ]
        fractions[label] = np.array([run.saturated_fraction for run in runs])
        levels[label] = np.array([abs(float(np.median(run.output_v))) / swing for run in runs])
    return OffsetSweep(
        offsets_v=np.array(config.offset_sweep_v),
        saturated_fraction=fractions,
        output_level=levels,
        gains={label: gain for label, (_, gain) in setups.items()},
    )


def filtered_noise_sigma_uv(frontend: FrontEnd, fs_hz: float) -> float:
    """Predict the noise sigma after U0's zero-phase spike-band filter, input-referred µV.

    The model's output noise density is weighted by the filter's magnitude
    response applied twice (forward and backward), integrated to the Nyquist
    frequency, and divided by the reference gain.
    """
    freqs = np.linspace(1.0, fs_hz / 2.0, 20_000, endpoint=False)
    _, response = signal.sosfreqz(design_bandpass(*SPIKE_BAND_HZ, fs_hz), worN=freqs, fs=fs_hz)
    weighted = frontend.output_noise(freqs).total * np.abs(response) ** 4
    return integrate_psd(weighted, freqs) / reference_gain(frontend) * 1.0e6


def emulator_noise_check(frontend: FrontEnd, config: U1Config) -> dict[str, float]:
    """Emulate silence and compare the noise that comes out with the model."""
    fs_hz = config.emulator_fs_hz
    n = round(fs_hz * config.emulator_seconds)
    out = emulate(np.zeros(n), fs_hz, frontend, seed=config.seed)
    freqs, psd = welch_psd(out.output_v, fs_hz)
    grid = log_grid(*SPIKE_BAND_HZ)
    return {
        "fs_hz": fs_hz,
        "seconds": config.emulator_seconds,
        "band_output_rms_measured_uv": float(np.sqrt(band_power(freqs, psd, *SPIKE_BAND_HZ))) * 1e6,
        "band_output_rms_predicted_uv": integrate_psd(frontend.output_noise(grid).total, grid)
        * 1e6,
        "sigma_after_filter_measured_uv": mad_sigma(
            band_split(out.input_referred_uv, fs_hz).spikes
        ),
        "sigma_after_filter_predicted_uv": filtered_noise_sigma_uv(frontend, fs_hz),
    }


def nason_check(frontend: FrontEnd, recording: Recording, config: U1Config) -> NasonCheck:
    """Run the emulator on a microvolt-scaled recording and compare noise levels.

    The recording already holds the noise of the system that made it, so the
    emulator's noise is added on top. The check is that the added noise raises
    the spike-band sigma by the amount the model predicts.
    """
    x = recording.channel(0)
    fs_hz = recording.fs_hz

    def sigma(trace: FloatArray) -> float:
        return mad_sigma(band_split(trace, fs_hz).spikes)

    def count(trace: FloatArray) -> int:
        return detect_spikes(band_split(trace, fs_hz).spikes, fs_hz).count

    quiet = emulate(x, fs_hz, frontend, seed=config.seed, add_noise=False).input_referred_uv
    noisy = emulate(x, fs_hz, frontend, seed=config.seed).input_referred_uv
    quarter = frontend.with_bias_scaled(0.25, keep_bandwidth=True)
    quarter_noisy = emulate(x, fs_hz, quarter, seed=config.seed).input_referred_uv
    freqs, psd_recorded = welch_psd(x, fs_hz, resolution_hz=4.0)
    _, psd_emulated = welch_psd(noisy, fs_hz, resolution_hz=4.0)
    model = np.zeros_like(freqs)
    model[1:] = frontend.input_noise(freqs[1:]).total * 1.0e12
    sigma_quiet = sigma(quiet)
    return NasonCheck(
        fs_hz=fs_hz,
        freqs_hz=freqs,
        psd_recorded=psd_recorded,
        psd_emulated=psd_emulated,
        model_input_psd=model,
        sigma_recorded_uv=sigma(x),
        sigma_band_limited_uv=sigma_quiet,
        sigma_emulated_uv=sigma(noisy),
        sigma_predicted_uv=float(np.hypot(sigma_quiet, filtered_noise_sigma_uv(frontend, fs_hz))),
        sigma_quarter_bias_uv=sigma(quarter_noisy),
        quarter_bias_power_w=quarter.lna.power_w,
        detections_recorded=count(x),
        detections_emulated=count(noisy),
    )


def _budget_uv(budget: NoiseBudget) -> dict[str, float]:
    return {key.removesuffix("_v") + "_uv": value * 1e6 for key, value in asdict(budget).items()}


def harrison_check() -> dict[str, float]:
    """Check the model's noise bookkeeping against Harrison and Charles (2003).

    The supply current follows from the published power and supply voltage; the
    NEF then follows from the published noise and bandwidth. Running the model
    with that NEF at room temperature should return the published noise. This
    confirms the NEF formula and the model's noise bandwidth agree. It does not
    reproduce the transistor design.
    """
    i_total = HARRISON_POWER_W / HARRISON_SUPPLY_V
    nef = nef_from_noise(HARRISON_NOISE_V, i_total, HARRISON_BANDWIDTH_HZ, ROOM_TEMPERATURE_K)
    lna = design_lna(
        f_high_hz=HARRISON_BANDWIDTH_HZ,
        f_low_hz=HARRISON_F_LOW_HZ,
        i_total_a=i_total,
        nef=nef,
        temperature_k=ROOM_TEMPERATURE_K,
    )
    bench = FrontEnd(lna, None, ROOM_TEMPERATURE_K)
    return {
        "published_noise_uv": HARRISON_NOISE_V * 1e6,
        "supply_current_ua": i_total * 1e6,
        "nef_from_published_values": nef,
        "model_total_noise_uv": bench.amplifier_total_noise_v() * 1e6,
        "model_noise_density_nv": lna.noise_density_v_per_rt_hz(ROOM_TEMPERATURE_K) * 1e9,
        "nef_limit_differential_pair": float(np.sqrt(2.0) / lna.kappa),
        "nef_limit_current_mirror_ota": 2.0 / lna.kappa,
    }


def amplifier_summary(frontend: FrontEnd) -> dict[str, float]:
    """Return the amplifier's headline numbers from the analytic model."""
    lna = frontend.lna
    temperature_k = frontend.temperature_k
    return {
        "ideal_gain": lna.ideal_gain,
        "gain_1khz": reference_gain(frontend),
        "midband_gain": frontend.midband_gain,
        "f_low_hz": frontend.f_low_hz,
        "f_high_hz": frontend.f_high_hz,
        "f_high_ideal_source_hz": frontend.without_electrode().f_high_hz,
        "i_total_ua": lna.i_total_a * 1e6,
        "v_supply_v": lna.v_supply_v,
        "power_uw": lna.power_w * 1e6,
        "noise_density_nv": lna.noise_density_v_per_rt_hz(temperature_k) * 1e9,
        "nef_nominal": lna.nef,
        "nef_measured": frontend.nef(),
        "total_noise_all_frequencies_uv": frontend.amplifier_total_noise_v() * 1e6,
        "c_load_pf": lna.c_load_f * 1e12,
        "r2_ohm": lna.r2_ohm,
        "gm_us": lna.gm_s(temperature_k) * 1e6,
        "output_swing_v": lna.output_swing_v,
        "temperature_k": temperature_k,
    }


def coupling_summary(frontend: FrontEnd, sweep: OffsetSweep, config: U1Config) -> dict[str, float]:
    """Return offset tolerance and the ADC resolution each coupling choice needs."""
    swing = frontend.lna.output_swing_v
    noise_v = frontend.band_noise().total_v
    high_gain = sweep.gains["AC-coupled"]
    low_gain = sweep.gains["DC-coupled, low gain"]
    return {
        "gain_ac": high_gain,
        "input_range_ac_mv": swing / high_gain * 1e3,
        "dc_offset_tolerance_same_gain_mv": coupling.dc_offset_tolerance_v(high_gain, swing) * 1e3,
        "worst_offset_mv": config.worst_offset_v * 1e3,
        "dc_gain_for_worst_offset": low_gain,
        "adc_bits_ac": coupling.adc_bits_required(swing / high_gain, noise_v),
        "adc_bits_dc": coupling.adc_bits_required(swing / low_gain, noise_v),
    }


def compute_u1(
    *,
    config: U1Config | None = None,
    frontend: FrontEnd | None = None,
    use_spice: bool | None = None,
    nason_recording: Recording | None = None,
) -> U1Results:
    """Run the whole U1 analysis.

    Args:
        config: Analysis parameters.
        frontend: The baseline front end; defaults to :func:`baseline_frontend`.
        use_spice: Run ngspice. Defaults to whether it is installed.
        nason_recording: The Nason in-vivo broadband in µV, if available.
    """
    config = config or U1Config()
    frontend = frontend or baseline_frontend()
    use_spice = spice.ngspice_available() if use_spice is None else use_spice
    under_test = electrodes(frontend)

    impedance = {key: impedance_curve(e, use_spice=use_spice) for key, e in under_test.items()}
    noise: dict[str, NoiseComparison] = {}
    baseline_runs: dict[str, spice.SpiceFrontEnd] | None = None
    for key, electrode in under_test.items():
        variant = replace(frontend, electrode=electrode)
        budget, runs = _spice_budget(variant) if use_spice else (None, None)
        if key == ELECTRODE_KEYS[0]:
            baseline_runs = runs
        noise[key] = NoiseComparison(
            hand=hand_calculation(electrode, frontend.lna, frontend.temperature_k),
            analytic=variant.band_noise(),
            spice=budget,
        )

    settling = tuple(settling_run(frontend, f, config) for f in config.f_low_options_hz)
    sweep = offset_sweep(frontend, config)
    nason = nason_check(frontend, nason_recording, config) if nason_recording is not None else None
    freqs = log_grid(10.0, 2.0e4, points_per_decade=80)

    summary: dict[str, Any] = {
        "spice_available": use_spice,
        "amplifier": amplifier_summary(frontend),
        "electrodes": {
            key: {
                "name": curve.electrode.name,
                "rs_ohm": curve.electrode.rs_ohm,
                "rct_ohm": curve.electrode.rct_ohm,
                "q_s_sn": curve.electrode.q_s_sn,
                "n": curve.electrode.n,
                "z_1khz_ohm": curve.electrode.magnitude_ohm(1000.0),
                "phase_1khz_deg": curve.electrode.phase_deg(1000.0),
                "spice_max_relative_error": curve.spice_max_error,
            }
            for key, curve in impedance.items()
        },
        "noise_band_hz": list(SPIKE_BAND_HZ),
        "noise": {
            key: {
                "hand": {
                    "electrode_uv": c.hand.electrode_v * 1e6,
                    "lna_uv": c.hand.lna_v * 1e6,
                    "total_uv": c.hand.total_v * 1e6,
                },
                "analytic": _budget_uv(c.analytic),
                "spice": _budget_uv(c.spice) if c.spice is not None else None,
                "max_disagreement": c.max_disagreement,
            }
            for key, c in noise.items()
        },
        "harrison": harrison_check(),
        "coupling": coupling_summary(frontend, sweep, config),
        "settling": {
            "step_mv": config.step_v * 1e3,
            "runs": [
                {
                    "f_low_hz": run.f_low_hz,
                    "time_constant_s": run.time_constant_s,
                    "saturation_time_analytic_s": run.saturation_time_analytic_s,
                    "saturation_time_simulated_s": run.saturation_time_simulated_s,
                    "recovery_to_noise_s": run.recovery_to_noise_s,
                }
                for run in settling
            ],
        },
        "emulator": emulator_noise_check(frontend, config),
        "nason": None,
    }
    if baseline_runs is not None:
        run = baseline_runs["all"]
        summary["amplifier"]["spice"] = {
            "gain_1khz": run.gain_at(1000.0),
            "f_low_hz": run.corner_hz(upper=False),
            "f_high_hz": run.corner_hz(upper=True),
        }
    if nason is not None:
        # Like for like: the model's noise as the same filter and estimator would report it.
        model_uv = filtered_noise_sigma_uv(frontend, nason.fs_hz)
        summary["nason"] = {
            "fs_hz": nason.fs_hz,
            "sigma_recorded_uv": nason.sigma_recorded_uv,
            "sigma_band_limited_uv": nason.sigma_band_limited_uv,
            "sigma_emulated_uv": nason.sigma_emulated_uv,
            "sigma_predicted_uv": nason.sigma_predicted_uv,
            "sigma_model_floor_uv": model_uv,
            "recorded_over_model": nason.sigma_recorded_uv / model_uv,
            "sigma_quarter_bias_uv": nason.sigma_quarter_bias_uv,
            "quarter_bias_power_uw": nason.quarter_bias_power_w * 1e6,
            "detections_recorded": nason.detections_recorded,
            "detections_emulated": nason.detections_emulated,
        }
    return U1Results(
        config=config,
        frontend=frontend,
        impedance=impedance,
        noise=noise,
        spectrum=frontend.input_noise(freqs),
        spice_spectra=baseline_runs,
        settling=settling,
        offsets=sweep,
        nason=nason,
        summary=summary,
    )
