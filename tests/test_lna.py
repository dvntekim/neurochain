import numpy as np
import pytest

from neurochain.frontend.lna import Lna, nef_from_noise, total_noise_from_nef
from neurochain.frontend.physics import ROOM_TEMPERATURE_K, four_kt, thermal_voltage_v

T = 310.15


def test_gain_power_swing_and_corner() -> None:
    lna = Lna()
    assert lna.ideal_gain == pytest.approx(100.0)
    assert lna.power_w == pytest.approx(18e-6)
    assert lna.output_swing_v == pytest.approx(0.8)
    assert lna.noise_gain == pytest.approx(22.2 / 20.0)
    assert lna.with_f_low(0.1).f_low_hz == pytest.approx(0.1)
    assert lna.with_f_low(0.1).r2_ohm == pytest.approx(10 * lna.with_f_low(1.0).r2_ohm)


def test_transconductance_is_weak_inversion_at_a_quarter_of_the_current() -> None:
    lna = Lna(i_total_a=8e-6, kappa=0.7)
    assert lna.gm_s(T) == pytest.approx(0.7 * 2e-6 / thermal_voltage_v(T))
    assert lna.r_out_ohm(T) * lna.gm_s(T) == pytest.approx(lna.dc_gain)


def test_noise_density_follows_from_nef_and_current() -> None:
    lna = Lna(i_total_a=10e-6, nef=4.0)
    expected = 4.0 * np.sqrt(four_kt(T) * thermal_voltage_v(T) / 10e-6)
    assert lna.noise_density_v_per_rt_hz(T) == pytest.approx(expected)
    # Density times the noise bandwidth of a first-order roll-off is the NEF's V_ni.
    assert lna.noise_density_v_per_rt_hz(T) * np.sqrt(np.pi / 2 * 7500.0) == pytest.approx(
        total_noise_from_nef(4.0, 10e-6, 7500.0, T)
    )


def test_halving_the_current_raises_the_density_by_root_two() -> None:
    ratio = Lna(i_total_a=5e-6).noise_density_v_per_rt_hz(T) / Lna(
        i_total_a=10e-6
    ).noise_density_v_per_rt_hz(T)
    assert ratio == pytest.approx(np.sqrt(2.0))


def test_ota_noise_resistor_carries_the_density_divided_by_the_noise_gain() -> None:
    lna = Lna()
    e_ota = np.sqrt(four_kt(T) * lna.ota_noise_resistance_ohm(T))
    assert e_ota * lna.noise_gain == pytest.approx(lna.noise_density_v_per_rt_hz(T))


def test_nef_round_trip() -> None:
    v = total_noise_from_nef(3.2, 4e-6, 5000.0, T)
    assert nef_from_noise(v, 4e-6, 5000.0, T) == pytest.approx(3.2)


def test_harrison_published_numbers_give_nef_four() -> None:
    # 2.2 µV rms, 7.2 kHz, 80 µW from a 5 V supply (16 µA), at room temperature.
    assert nef_from_noise(2.2e-6, 16e-6, 7200.0, ROOM_TEMPERATURE_K) == pytest.approx(4.0, abs=0.01)
