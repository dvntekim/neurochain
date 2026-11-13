"""The U1 done-check: SPICE, the analytic model and the hand calculation agree.

Tests marked ``spice`` need ngspice and skip without it. The others hold the
analytic model and the hand calculation to each other on any machine.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from neurochain.frontend import spice
from neurochain.frontend.electrode import UTAH_PT_IN_VITRO, UTAH_PT_IN_VIVO, Electrode, log_grid
from neurochain.frontend.hand import hand_calculation, interface_real_integral
from neurochain.frontend.model import FrontEnd, baseline_frontend, design_lna
from neurochain.frontend.physics import four_kt
from neurochain.paths import repo_root

needs_ngspice = pytest.mark.skipif(not spice.ngspice_available(), reason="ngspice not installed")

DONE_CRITERION = 0.10
"""The unit's requirement: every pair of methods within 10 %."""

OBSERVED = 0.01
"""What the methods actually achieve; a regression past this is worth a look."""

ELECTRODES = {
    "in_saline": UTAH_PT_IN_VITRO,
    "ideal_capacitor": UTAH_PT_IN_VITRO.with_ideal_capacitor(),
    "implanted": UTAH_PT_IN_VIVO,
}


@pytest.fixture(scope="module")
def baseline() -> FrontEnd:
    return baseline_frontend()


def test_amplifier_meets_its_design_targets(baseline: FrontEnd) -> None:
    bench = baseline.without_electrode()
    assert bench.f_high_hz == pytest.approx(7500.0, rel=1e-6)
    assert bench.f_low_hz == pytest.approx(1.0, rel=0.02)
    # Finite OTA gain and the input capacitance cost about 1 % of C1/C2.
    assert bench.midband_gain == pytest.approx(100.0, rel=0.02)
    assert baseline.lna.power_w == pytest.approx(18e-6)


def test_electrode_loads_the_amplifier(baseline: FrontEnd) -> None:
    bench = baseline.without_electrode()
    assert baseline.midband_gain < bench.midband_gain
    assert baseline.f_high_hz < bench.f_high_hz
    assert baseline.f_high_hz == pytest.approx(7144.0, rel=0.01)


def test_network_nef_matches_the_nominal_value(baseline: FrontEnd) -> None:
    assert baseline.nef() == pytest.approx(baseline.lna.nef, rel=0.005)


def test_input_noise_is_output_noise_over_gain_squared(baseline: FrontEnd) -> None:
    f = log_grid(300.0, 7500.0, points_per_decade=20)
    gain_sq = np.abs(baseline.signal_gain(f)) ** 2
    np.testing.assert_allclose(
        baseline.input_noise(f).total * gain_sq, baseline.output_noise(f).total, rtol=1e-12
    )
    # Referred to the tissue, electrode noise is exactly 4kT Re{Z}: the gain cancels.
    np.testing.assert_allclose(
        baseline.input_noise(f).electrode,
        four_kt(baseline.temperature_k) * UTAH_PT_IN_VITRO.impedance(f).real,
        rtol=1e-12,
    )


def test_budget_adds_in_quadrature(baseline: FrontEnd) -> None:
    budget = baseline.band_noise()
    assert budget.total_v == pytest.approx(np.hypot(budget.electrode_v, budget.lna_v), rel=1e-9)


@pytest.mark.parametrize("key", list(ELECTRODES))
def test_hand_calculation_matches_the_analytic_model(baseline: FrontEnd, key: str) -> None:
    electrode = ELECTRODES[key]
    hand = hand_calculation(electrode, baseline.lna, baseline.temperature_k)
    model = replace(baseline, electrode=electrode).band_noise()
    assert hand.total_v == pytest.approx(model.total_v, rel=DONE_CRITERION)
    assert hand.total_v == pytest.approx(model.total_v, rel=OBSERVED)
    assert hand.electrode_v == pytest.approx(model.electrode_v, rel=OBSERVED)
    assert hand.lna_v == pytest.approx(model.lna_v, rel=OBSERVED)


def test_hand_calculation_numbers(baseline: FrontEnd) -> None:
    """The figures worked through on paper in docs/U1_design.md."""
    hand = hand_calculation(UTAH_PT_IN_VITRO, baseline.lna, baseline.temperature_k)
    assert hand.cpe_magnitude_1khz_ohm == pytest.approx(214.8e3, rel=2e-3)
    assert hand.resistor_term_ohm_hz == pytest.approx(5.256e7, rel=1e-3)
    assert hand.interface_term_ohm_hz == pytest.approx(3.141e8, rel=2e-3)
    assert hand.electrode_v == pytest.approx(2.506e-6, rel=1e-3)
    assert hand.noise_density_v_per_rt_hz == pytest.approx(27.06e-9, rel=1e-3)
    assert hand.lna_v == pytest.approx(2.296e-6, rel=1e-3)
    assert hand.total_v == pytest.approx(3.399e-6, rel=1e-3)
    assert hand.lna_total_v == pytest.approx(2.937e-6, rel=1e-3)


def test_an_ideal_capacitor_has_no_interface_noise() -> None:
    ideal = Electrode(rs_ohm=7.3e3, rct_ohm=4.6e9, q_s_sn=7.3e-10, n=1.0)
    assert interface_real_integral(ideal, 300.0, 7500.0) == 0.0


def test_halving_the_bias_with_a_fixed_load_halves_bandwidth_not_total_noise(
    baseline: FrontEnd,
) -> None:
    half = baseline.with_bias_scaled(0.5, keep_bandwidth=False)
    assert half.without_electrode().f_high_hz == pytest.approx(3750.0, rel=0.02)
    assert half.amplifier_total_noise_v() == pytest.approx(
        baseline.amplifier_total_noise_v(), rel=0.005
    )
    assert half.nef() == pytest.approx(baseline.nef(), rel=0.005)


def test_halving_the_bias_at_constant_bandwidth_costs_root_two_in_noise(baseline: FrontEnd) -> None:
    half = baseline.with_bias_scaled(0.5, keep_bandwidth=True)
    assert half.without_electrode().f_high_hz == pytest.approx(7500.0, rel=1e-6)
    assert half.lna.c_load_f == pytest.approx(baseline.lna.c_load_f / 2, rel=0.02)
    assert half.band_noise().lna_v == pytest.approx(
        np.sqrt(2.0) * baseline.band_noise().lna_v, rel=0.005
    )
    assert half.nef() == pytest.approx(baseline.nef(), rel=0.005)
    assert half.lna.power_w == pytest.approx(baseline.lna.power_w / 2)


def test_a_high_corner_puts_feedback_resistor_noise_in_the_band(baseline: FrontEnd) -> None:
    raised = replace(baseline, lna=baseline.lna.with_f_low(300.0))
    assert raised.band_noise().lna_v > 1.1 * baseline.band_noise().lna_v


def test_design_lna_hits_other_targets() -> None:
    lna = design_lna(f_high_hz=3000.0, f_low_hz=10.0, i_total_a=2e-6, nef=3.0)
    bench = FrontEnd(lna, None)
    assert bench.f_high_hz == pytest.approx(3000.0, rel=1e-6)
    assert bench.f_low_hz == pytest.approx(10.0, rel=0.02)
    assert (lna.i_total_a, lna.nef) == (2e-6, 3.0)


def test_netlist_marks_only_the_requested_sources_noisy(baseline: FrontEnd) -> None:
    quiet_lna = spice.frontend_netlist(baseline, sources="electrode").splitlines()
    quiet_electrode = spice.frontend_netlist(baseline, sources="lna").splitlines()

    def line(lines: list[str], name: str) -> str:
        return next(text for text in lines if text.startswith(name + " "))

    assert line(quiet_lna, "Rn").endswith("noisy=0")
    assert line(quiet_lna, "R2").endswith("noisy=0")
    assert not line(quiet_lna, "Rs").endswith("noisy=0")
    assert line(quiet_electrode, "Rs").endswith("noisy=0")
    assert line(quiet_electrode, "Rb0").endswith("noisy=0")
    assert not line(quiet_electrode, "Rn").endswith("noisy=0")
    # The OTA output resistor models gain, never noise.
    for lines in (quiet_lna, quiet_electrode, spice.frontend_netlist(baseline).splitlines()):
        assert line(lines, "Ro").endswith("noisy=0")


def test_netlist_for_an_ideal_source_has_no_electrode(baseline: FrontEnd) -> None:
    text = spice.frontend_netlist(baseline.without_electrode())
    assert "Rs " not in text
    assert "C1 tissue x" in text


def test_ideal_capacitor_netlist_uses_one_capacitor() -> None:
    text = spice.impedance_netlist(UTAH_PT_IN_VITRO.with_ideal_capacitor())
    assert "Cdl iface 0" in text
    assert "Rb0" not in text


def test_committed_netlists_match_the_generator(baseline: FrontEnd) -> None:
    directory = repo_root() / "spice"
    for name, text in spice.reference_netlists(baseline).items():
        assert (directory / name).read_text() == text, f"spice/{name} is stale; run `make u1`"


def test_reference_netlists_need_an_electrode(baseline: FrontEnd) -> None:
    with pytest.raises(ValueError, match="electrode"):
        spice.reference_netlists(baseline.without_electrode())


def test_missing_ngspice_is_reported(monkeypatch: pytest.MonkeyPatch, baseline: FrontEnd) -> None:
    monkeypatch.setattr(spice, "NGSPICE", "ngspice-that-does-not-exist")
    assert not spice.ngspice_available()
    with pytest.raises(spice.SpiceError, match="not installed"):
        spice.simulate_frontend(baseline)


@needs_ngspice
def test_a_broken_netlist_raises_with_the_log(tmp_path: Path) -> None:
    with pytest.raises(spice.SpiceError, match="ngspice failed"):
        spice.run_ngspice("* nothing here\n.end\n", ("never_written.txt",))


@needs_ngspice
@pytest.mark.parametrize("key", list(ELECTRODES))
def test_spice_impedance_matches_the_analytic_model(key: str) -> None:
    electrode = ELECTRODES[key]
    freqs, z = spice.simulate_impedance(electrode)
    assert freqs[0] == pytest.approx(1.0) and freqs[-1] == pytest.approx(1.0e4)
    assert np.abs(z / electrode.impedance(freqs) - 1.0).max() < 2e-3


@needs_ngspice
def test_spice_gain_matches_the_analytic_model(baseline: FrontEnd) -> None:
    run = spice.simulate_frontend(baseline)
    error = np.abs(run.gain / baseline.signal_gain(run.ac_freqs_hz) - 1.0)
    assert error.max() < 2e-3  # over 0.01 Hz to 1 MHz
    assert run.gain_at(1000.0) == pytest.approx(
        float(np.abs(baseline.signal_gain(np.array([1000.0])))[0]), rel=1e-3
    )
    assert run.corner_hz(upper=True) == pytest.approx(baseline.f_high_hz, rel=5e-3)
    assert run.corner_hz(upper=False) == pytest.approx(baseline.f_low_hz, rel=5e-3)


@needs_ngspice
@pytest.mark.parametrize("key", list(ELECTRODES))
def test_done_criterion_spice_analytic_and_hand_agree(baseline: FrontEnd, key: str) -> None:
    """Simulated noise within 10 % of the hand calculation (in fact within 1 %)."""
    electrode = ELECTRODES[key]
    frontend = replace(baseline, electrode=electrode)
    simulated = spice.simulate_frontend(frontend).input_noise_total_v
    analytic = frontend.band_noise().total_v
    hand = hand_calculation(electrode, frontend.lna, frontend.temperature_k).total_v
    for a, b in ((simulated, hand), (simulated, analytic), (analytic, hand)):
        assert a == pytest.approx(b, rel=DONE_CRITERION)
        assert a == pytest.approx(b, rel=OBSERVED)


@needs_ngspice
def test_spice_noise_by_source(baseline: FrontEnd) -> None:
    runs = {s: spice.simulate_frontend(baseline, sources=s) for s in ("all", "electrode", "lna")}
    budget = baseline.band_noise()
    assert runs["electrode"].input_noise_total_v == pytest.approx(budget.electrode_v, rel=OBSERVED)
    assert runs["lna"].input_noise_total_v == pytest.approx(budget.lna_v, rel=OBSERVED)
    assert runs["all"].input_noise_total_v == pytest.approx(
        np.hypot(runs["electrode"].input_noise_total_v, runs["lna"].input_noise_total_v), rel=1e-3
    )
    # The spectrum agrees point by point, not just in total.
    f = runs["all"].noise_freqs_hz
    np.testing.assert_allclose(
        runs["all"].input_noise_asd, np.sqrt(baseline.input_noise(f).total), rtol=5e-3
    )
    np.testing.assert_allclose(
        runs["all"].output_noise_asd, np.sqrt(baseline.output_noise(f).total), rtol=5e-3
    )
