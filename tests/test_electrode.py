import numpy as np
import pytest

from neurochain.frontend.electrode import (
    UTAH_PT_IN_VITRO,
    UTAH_PT_IN_VIVO,
    Electrode,
    cpe_ladder,
    integrate_psd,
    log_grid,
)
from neurochain.frontend.physics import BODY_TEMPERATURE_K, four_kt, thermal_voltage_v

IDEAL = Electrode(rs_ohm=10e3, rct_ohm=1e9, q_s_sn=1e-9, n=1.0)


def test_thermal_quantities() -> None:
    assert four_kt(300.0) == pytest.approx(1.6568e-20, rel=1e-4)
    assert thermal_voltage_v(300.0) == pytest.approx(25.85e-3, rel=1e-3)
    assert thermal_voltage_v(BODY_TEMPERATURE_K) == pytest.approx(26.73e-3, rel=1e-3)


def test_published_fit_values() -> None:
    # Straka et al. (2018), Table 1, platinum arrays.
    assert (UTAH_PT_IN_VITRO.rs_ohm, UTAH_PT_IN_VITRO.rct_ohm) == (7.3e3, 4.6e9)
    assert (UTAH_PT_IN_VITRO.q_s_sn, UTAH_PT_IN_VITRO.n) == (0.72e-8, 0.74)
    assert (UTAH_PT_IN_VIVO.rs_ohm, UTAH_PT_IN_VIVO.n) == (25e3, 0.55)
    # The spec asks for 100 kΩ to 1 MΩ at 1 kHz.
    assert 100e3 < UTAH_PT_IN_VITRO.magnitude_ohm(1000.0) < 1e6


def test_ideal_randles_limits() -> None:
    # Far below the corner the capacitor is open: Rs + Rct. Far above it is a short: Rs.
    assert IDEAL.magnitude_ohm(1e-6) == pytest.approx(IDEAL.rs_ohm + IDEAL.rct_ohm, rel=1e-3)
    assert IDEAL.magnitude_ohm(1e9) == pytest.approx(IDEAL.rs_ohm, rel=1e-3)
    assert IDEAL.phase_deg(1e9) == pytest.approx(0.0, abs=0.1)


def test_ideal_capacitor_matches_the_textbook_formula() -> None:
    f = np.array([1000.0])
    z_c = 1.0 / (2j * np.pi * f * IDEAL.q_s_sn)
    expected = IDEAL.rs_ohm + IDEAL.rct_ohm * z_c / (IDEAL.rct_ohm + z_c)
    np.testing.assert_allclose(IDEAL.impedance(f), expected, rtol=1e-12)
    assert IDEAL.is_ideal_capacitor


def test_cpe_has_constant_phase_and_power_law_magnitude() -> None:
    f = np.array([10.0, 100.0, 1000.0])
    z = UTAH_PT_IN_VITRO.cpe_impedance(f)
    np.testing.assert_allclose(np.degrees(np.angle(z)), -90.0 * 0.74, atol=1e-9)
    np.testing.assert_allclose(np.abs(z[:-1]) / np.abs(z[1:]), 10.0**0.74, rtol=1e-12)


def test_thermal_noise_is_four_kt_times_the_real_part() -> None:
    f = log_grid(300.0, 7500.0)
    psd = UTAH_PT_IN_VITRO.thermal_noise_psd(f, 310.15)
    np.testing.assert_allclose(psd, four_kt(310.15) * UTAH_PT_IN_VITRO.impedance(f).real)
    assert UTAH_PT_IN_VITRO.thermal_noise_rms_v(300.0, 7500.0, 310.15) == pytest.approx(
        integrate_psd(psd, f)
    )


def test_a_resistor_gives_johnson_noise() -> None:
    # Make the interface negligible: what is left is 4kT * Rs * bandwidth.
    resistor = Electrode(rs_ohm=100e3, rct_ohm=1e-3, q_s_sn=1e-9, n=1.0)
    expected = np.sqrt(four_kt(300.0) * 100e3 * 9000.0)
    assert resistor.thermal_noise_rms_v(1000.0, 10_000.0, 300.0) == pytest.approx(
        expected, rel=1e-6
    )


def test_ideal_capacitor_twin_matches_magnitude_but_is_quieter() -> None:
    twin = UTAH_PT_IN_VITRO.with_ideal_capacitor()
    assert twin.n == 1.0
    assert twin.rs_ohm == UTAH_PT_IN_VITRO.rs_ohm
    assert twin.magnitude_ohm(1000.0) == pytest.approx(UTAH_PT_IN_VITRO.magnitude_ohm(1000.0))
    assert twin.phase_deg(1000.0) < -85.0
    measured = UTAH_PT_IN_VITRO.thermal_noise_rms_v(300.0, 7500.0, 310.15)
    assert twin.thermal_noise_rms_v(300.0, 7500.0, 310.15) < 0.5 * measured


def test_doubling_the_double_layer_roughly_halves_the_impedance() -> None:
    doubled = UTAH_PT_IN_VITRO.with_interface_scaled(2.0)
    ratio = doubled.magnitude_ohm(1000.0) / UTAH_PT_IN_VITRO.magnitude_ohm(1000.0)
    assert 0.5 < ratio < 0.52  # not exactly half: Rs does not scale


@pytest.mark.parametrize("electrode", [UTAH_PT_IN_VITRO, UTAH_PT_IN_VIVO])
def test_rc_ladder_reproduces_the_cpe(electrode: Electrode) -> None:
    ladder = cpe_ladder(electrode.q_s_sn, electrode.n)
    f = log_grid(1.0, 1.0e5, points_per_decade=50)
    error = np.abs(ladder.impedance(f) / electrode.cpe_impedance(f) - 1.0)
    assert error.max() < 5e-3
    assert ladder.branch_r_ohm.size == ladder.branch_c_f.size == 21
    assert np.all(ladder.branch_r_ohm > 0) and np.all(ladder.branch_c_f > 0)


def test_more_branches_per_decade_is_more_accurate() -> None:
    f = log_grid(10.0, 1.0e4, points_per_decade=50)
    exact = UTAH_PT_IN_VITRO.cpe_impedance(f)

    def worst(per_decade: int) -> float:
        ladder = cpe_ladder(0.72e-8, 0.74, per_decade=per_decade)
        return float(np.abs(ladder.impedance(f) / exact - 1.0).max())

    assert worst(1) > 10 * worst(2)


def test_invalid_parameters_are_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        Electrode(rs_ohm=0.0, rct_ohm=1e9, q_s_sn=1e-9)
    with pytest.raises(ValueError, match="exponent"):
        Electrode(rs_ohm=1e3, rct_ohm=1e9, q_s_sn=1e-9, n=1.2)
    with pytest.raises(ValueError, match="0 < n < 1"):
        cpe_ladder(1e-9, 1.0)
    with pytest.raises(ValueError, match="low_hz"):
        log_grid(100.0, 10.0)
