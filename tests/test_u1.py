import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from neurochain.analysis.u1 import (
    ELECTRODE_KEYS,
    U1Config,
    U1Results,
    compute_u1,
    electrodes,
    filtered_noise_sigma_uv,
    harrison_check,
)
from neurochain.analysis.u1_plots import (
    plot_coupling,
    plot_emulator,
    plot_impedance,
    plot_noise_budget,
)
from neurochain.data.synthetic import make_synthetic_recording
from neurochain.frontend import spice
from neurochain.frontend.model import baseline_frontend

FAST = U1Config(emulator_seconds=8.0)


@pytest.fixture(scope="module")
def results() -> U1Results:
    """The analysis without ngspice, on a synthetic stand-in for the Nason recording."""
    recording, _ = make_synthetic_recording(duration_s=4.0, fs_hz=30_000.0, seed=1)
    return compute_u1(config=FAST, use_spice=False, nason_recording=recording)


def test_summary_is_complete_and_serialisable(results: U1Results) -> None:
    summary = results.summary
    assert summary["spice_available"] is False
    assert set(summary["electrodes"]) == set(summary["noise"]) == set(ELECTRODE_KEYS)
    for entry in summary["noise"].values():
        assert entry["spice"] is None
        assert entry["max_disagreement"] < 0.01
    assert "spice" not in summary["amplifier"]
    json.dumps(summary)


def test_baseline_numbers(results: U1Results) -> None:
    summary = results.summary
    assert summary["electrodes"]["in_saline"]["z_1khz_ohm"] == pytest.approx(217.8e3, rel=1e-3)
    assert summary["noise"]["in_saline"]["analytic"]["total_uv"] == pytest.approx(3.405, rel=1e-3)
    assert summary["noise"]["ideal_capacitor"]["analytic"]["electrode_uv"] < 1.0
    assert summary["amplifier"]["nef_measured"] == pytest.approx(4.0, rel=0.005)
    assert summary["amplifier"]["power_uw"] == pytest.approx(18.0)


def test_coupling_summary(results: U1Results) -> None:
    cpl = results.summary["coupling"]
    assert cpl["dc_offset_tolerance_same_gain_mv"] == pytest.approx(8.37, rel=1e-2)
    assert cpl["dc_gain_for_worst_offset"] == pytest.approx(2.5)
    assert cpl["adc_bits_dc"] - cpl["adc_bits_ac"] == pytest.approx(
        np.log2(cpl["gain_ac"] / 2.5), rel=1e-6
    )


def test_offset_sweep_shows_ac_immune_and_dc_railing(results: U1Results) -> None:
    sweep = results.offsets
    assert np.all(sweep.saturated_fraction["AC-coupled"] == 0.0)
    assert np.all(sweep.saturated_fraction["DC-coupled, low gain"] == 0.0)
    same_gain = sweep.saturated_fraction["DC-coupled, same gain"]
    assert same_gain[sweep.offsets_v <= 8e-3].max() == 0.0
    assert same_gain[sweep.offsets_v >= 10e-3].min() == 1.0
    assert sweep.output_level["DC-coupled, low gain"][-1] == pytest.approx(
        0.3 * 2.5 / 0.8, rel=0.02
    )


def test_settling_simulation_matches_the_formula(results: U1Results) -> None:
    fast, slow = results.settling
    for run in (fast, slow):
        assert run.saturation_time_simulated_s == pytest.approx(
            run.saturation_time_analytic_s, rel=0.01
        )
    assert slow.saturation_time_simulated_s == pytest.approx(
        10 * fast.saturation_time_simulated_s, rel=0.02
    )
    assert slow.recovery_to_noise_s > 9 * fast.recovery_to_noise_s


def test_emulator_noise_agrees_with_the_model(results: U1Results) -> None:
    check = results.summary["emulator"]
    assert check["band_output_rms_measured_uv"] == pytest.approx(
        check["band_output_rms_predicted_uv"], rel=0.02
    )
    assert check["sigma_after_filter_measured_uv"] == pytest.approx(
        check["sigma_after_filter_predicted_uv"], rel=0.03
    )


def test_added_noise_is_predicted_on_a_recording(results: U1Results) -> None:
    stats = results.summary["nason"]
    assert stats["sigma_emulated_uv"] > stats["sigma_band_limited_uv"]
    assert stats["sigma_emulated_uv"] == pytest.approx(stats["sigma_predicted_uv"], rel=0.03)
    assert stats["sigma_quarter_bias_uv"] > stats["sigma_emulated_uv"]
    assert stats["quarter_bias_power_uw"] == pytest.approx(4.5)
    assert stats["sigma_model_floor_uv"] == pytest.approx(
        filtered_noise_sigma_uv(baseline_frontend(), 30_000.0)
    )


def test_harrison_check_is_self_consistent() -> None:
    check = harrison_check()
    assert check["supply_current_ua"] == pytest.approx(16.0)
    assert check["nef_from_published_values"] == pytest.approx(4.0, abs=0.01)
    assert check["model_total_noise_uv"] == pytest.approx(2.2, rel=0.005)
    assert check["nef_limit_differential_pair"] == pytest.approx(2.02, abs=0.005)
    assert check["nef_limit_current_mirror_ota"] == pytest.approx(2.857, abs=0.005)


def test_without_a_recording_the_nason_block_is_empty() -> None:
    bare = compute_u1(config=replace(FAST, emulator_seconds=2.0), use_spice=False)
    assert bare.nason is None
    assert bare.summary["nason"] is None
    with pytest.raises(ValueError, match="not analysed"):
        plot_emulator(bare, Path("unused.png"))


def test_electrodes_need_a_front_end_with_one() -> None:
    with pytest.raises(ValueError, match="electrode"):
        electrodes(baseline_frontend().without_electrode())


def test_figures_are_written(results: U1Results, tmp_path: Path) -> None:
    for plot in (plot_impedance, plot_noise_budget, plot_coupling, plot_emulator):
        path = plot(results, tmp_path / f"{plot.__name__}.png")
        assert path.stat().st_size > 20_000


@pytest.mark.skipif(not spice.ngspice_available(), reason="ngspice not installed")
def test_with_spice_all_three_methods_are_reported(tmp_path: Path) -> None:
    full = compute_u1(config=replace(FAST, emulator_seconds=2.0), use_spice=True)
    assert full.summary["spice_available"] is True
    for key in ELECTRODE_KEYS:
        entry = full.summary["noise"][key]
        assert entry["spice"] is not None
        assert entry["max_disagreement"] < 0.01
        assert full.summary["electrodes"][key]["spice_max_relative_error"] < 2e-3
    assert full.summary["amplifier"]["spice"]["f_high_hz"] == pytest.approx(
        full.summary["amplifier"]["f_high_hz"], rel=5e-3
    )
    for plot in (plot_impedance, plot_noise_budget):
        assert plot(full, tmp_path / f"{plot.__name__}.png").stat().st_size > 20_000
