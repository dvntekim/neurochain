from pathlib import Path

import numpy as np
import pytest

from neurochain.adc import Adc
from neurochain.analysis import u2
from neurochain.analysis.u2_plots import plot_adc, plot_detection, plot_fixed_point
from neurochain.data.mearec import MearecRecording
from neurochain.data.synthetic import make_synthetic_recording, spike_template

FS_HZ = 30_000.0
SMALL = u2.U2Config(
    noise_levels_uv=(0.0, 13.0, 40.0),
    bits_options=(5, 6, 10),
    fs_options_hz=(30_000.0, 10_000.0),
    threshold_factors=(3.5, 4.5),
    neo_factors=(8.0, 12.0),
    calibration_s=2.0,
    alias_zones=2,
    frac_bits_options=(6, 14),
    guard_bits_options=(0, 6),
    max_workers=1,
)


def synthetic_mearec(n_channels: int = 2, seconds: float = 8.0) -> MearecRecording:
    """Build a small recording with the real one's structure: one 100 µV unit per channel."""
    rng = np.random.default_rng(0)
    template, _ = spike_template(FS_HZ)
    clean, spikes, templates = [], [], []
    for channel in range(n_channels):
        recording, truth = make_synthetic_recording(
            duration_s=seconds, fs_hz=FS_HZ, amplitudes=(100.0,), rates_hz=(12.0,),
            noise_rms=0.0, lfp_rms=0.0, seed=channel,
        )  # fmt: skip
        clean.append(recording.channel(0))
        spikes.append(truth.spike_samples[0])
        row = np.zeros((n_channels, template.size))
        row[channel] = 100.0 * template
        templates.append(row)
    background = rng.standard_normal((n_channels, round(seconds * FS_HZ)))
    return MearecRecording(
        fs_hz=FS_HZ,
        clean_uv=np.stack(clean),
        background_uv=background / np.sqrt(7200 / 15_000),
        spike_samples=tuple(spikes),
        templates_uv=np.stack(templates),
        cell_types=("E",) * n_channels,
    )


@pytest.fixture(scope="module")
def summary() -> dict:
    return u2.compute_u2(config=SMALL, recording=synthetic_mearec(), include_nason=False)


def test_summary_structure(summary: dict) -> None:
    mearec = summary["mearec"]
    assert summary["nason"] is None
    assert mearec["recording"]["n_channels"] == 2
    assert mearec["recording"]["scored_s"] == pytest.approx(6.0)
    detection = mearec["detection"]
    assert set(detection) == {"noise", "bits", "fs", "factor"}
    assert set(detection["noise"]) == {"threshold", "neo", "offline"}
    assert set(detection["noise"]["threshold"]) == {"0", "13", "40"}
    assert set(detection["bits"]["neo"]) == {"5", "6", "10"}
    assert set(detection["fs"]["threshold"]) == {"30000", "10000"}
    assert set(mearec["tracking"]) == {"crossings", "sbp_digital", "sbp_implant"}
    row = detection["noise"]["threshold"]["13"]
    assert row["f1_low"] <= row["f1"] <= row["f1_high"]


def test_detection_gets_worse_with_noise(summary: dict) -> None:
    noise = summary["mearec"]["detection"]["noise"]
    for method in ("threshold", "offline"):
        assert noise[method]["13"]["f1"] > 0.9
        assert noise[method]["13"]["f1"] > noise[method]["40"]["f1"] + 0.5
    # With no background the threshold sits just above front-end noise and
    # starts counting filter ringing, so its F1 is lower than at 13 µV.
    assert 0.7 < noise["threshold"]["0"]["f1"] < noise["threshold"]["13"]["f1"]
    assert noise["neo"]["0"]["f1"] > noise["neo"]["13"]["f1"] >= noise["neo"]["40"]["f1"]


def test_features_track_the_unit_and_lose_it_in_noise(summary: dict) -> None:
    tracking = summary["mearec"]["tracking"]
    for feature in ("crossings", "sbp_digital", "sbp_implant"):
        assert tracking[feature]["0"]["mean"] > 0.8
        assert tracking[feature]["40"]["mean"] < tracking[feature]["0"]["mean"]
    # No detections at all is reported as zero correlation, not as undefined.
    assert tracking["crossings"]["40"]["mean"] == 0.0


def test_fixed_point_matches_float(summary: dict) -> None:
    fixed = summary["mearec"]["fixed_point"]
    float_f1 = summary["mearec"]["detection"]["noise"]["threshold"]["13"]["f1"]
    assert fixed["fixed_threshold_f1"] == pytest.approx(float_f1, abs=0.05)
    assert fixed["sbp_fixed_vs_float_r"] > 0.99
    assert abs(fixed["offset_wide_trunc_guard0"]) > 100.0
    assert abs(fixed["offset_wide_round_guard6"]) < 0.1
    assert fixed["snr_wide_frac14"] > fixed["snr_wide_frac6"]
    assert fixed["acc_bits_narrow"] < fixed["acc_bits_wide"] <= 32


def test_evaluate_setting(summary: dict) -> None:
    recording = synthetic_mearec()
    quiet = u2.evaluate_setting(recording, SMALL, noise_uv=5.0, adc=Adc())
    noisy = u2.evaluate_setting(recording, SMALL, noise_uv=40.0, adc=Adc())
    coarse = u2.evaluate_setting(recording, SMALL, noise_uv=5.0, adc=Adc(bits=3))
    assert quiet["threshold"]["f1"] > noisy["threshold"]["f1"]
    assert quiet["threshold"]["f1"] > coarse["threshold"]["f1"]
    assert set(quiet) == {"threshold", "neo"}


def test_narrowband_front_end_is_the_spiking_band() -> None:
    bench = u2.narrowband_frontend().without_electrode()
    assert bench.f_high_hz == pytest.approx(1000.0, rel=1e-3)
    assert bench.f_low_hz == pytest.approx(300.0, rel=0.3)


def test_without_inputs_both_blocks_are_empty() -> None:
    bare = u2.compute_u2(config=SMALL, recording=None, include_nason=False)
    assert bare["mearec"] is None and bare["nason"] is None


def test_figures_are_written(summary: dict, tmp_path: Path) -> None:
    for plot in (plot_detection, plot_adc, plot_fixed_point):
        path = plot(summary, tmp_path / f"{plot.__name__}.png")
        assert path.stat().st_size > 20_000
