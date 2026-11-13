import numpy as np
import pytest

from neurochain.frontend import coupling


def test_dc_offset_tolerance_and_its_inverse() -> None:
    assert coupling.dc_offset_tolerance_v(100.0, 0.8) == pytest.approx(8e-3)
    assert coupling.max_gain_for_offset(0.32, 0.8) == pytest.approx(2.5)
    assert coupling.dc_offset_tolerance_v(coupling.max_gain_for_offset(0.3, 0.8), 0.8) == (
        pytest.approx(0.3)
    )


def test_adc_bits() -> None:
    assert coupling.adc_bits_required(1.0, 2.0 / 1024) == pytest.approx(10.0)
    # Sixteen times the range needs four more bits.
    extra = coupling.adc_bits_required(16e-3, 1e-6) - coupling.adc_bits_required(1e-3, 1e-6)
    assert extra == pytest.approx(4.0)


def test_time_constant_and_recovery() -> None:
    tau = coupling.highpass_time_constant_s(1.0)
    assert tau == pytest.approx(1 / (2 * np.pi))
    assert coupling.recovery_time_s(1.0, np.exp(-3.0), 1.0) == pytest.approx(3 * tau)
    assert coupling.recovery_time_s(1e-6, 1e-3, 1.0) == 0.0
    # A ten times lower corner takes ten times longer.
    assert coupling.recovery_time_s(1.0, 0.01, 0.1) == pytest.approx(
        10 * coupling.recovery_time_s(1.0, 0.01, 1.0)
    )


def test_saturation_time() -> None:
    tau = coupling.highpass_time_constant_s(1.0)
    # 50 mV at gain 100 is 5 V against a 0.8 V swing.
    assert coupling.saturation_time_s(50e-3, 100.0, 0.8, 1.0) == pytest.approx(
        tau * np.log(5.0 / 0.8)
    )
    assert coupling.saturation_time_s(1e-3, 100.0, 0.8, 1.0) == 0.0
