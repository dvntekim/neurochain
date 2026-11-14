import numpy as np
import pytest
from scipy import signal

from neurochain.fixedpoint import (
    accumulator_bits,
    design_fixed_highpass,
    fixed_highpass,
    fixed_neo_score,
    fixed_sbp,
    fixed_threshold_events,
    quantisation_snr_db,
)

RNG = np.random.default_rng(0)
CODES = np.rint(20 * RNG.standard_normal(120_000) + 300).astype(np.int64)


def reference(fs_hz: float) -> np.ndarray:
    b, a = signal.butter(2, 300.0, btype="highpass", fs=fs_hz)
    return np.asarray(signal.lfilter(b, a, CODES.astype(float)))


def test_coefficients_are_the_scaled_butterworth_design() -> None:
    filt = design_fixed_highpass(30_000.0, frac_bits=14)
    b, a = signal.butter(2, 300.0, btype="highpass", fs=30_000.0)
    assert filt.b0 == round(b[0] * 2**14)
    assert (filt.a1, filt.a2) == (round(a[1] * 2**14), round(a[2] * 2**14))
    assert filt.is_stable
    assert filt.coefficient_bits == 16
    bq, aq = filt.float_coefficients()
    assert bq.sum() == 0.0  # exact zero at DC whatever the rounding
    np.testing.assert_allclose(aq, a, atol=2**-14)


def test_constant_input_rests_inside_the_deadband() -> None:
    constant = np.full(5000, 400, dtype=np.int64)
    # At 2 kS/s the deadband is below one code: the output is exactly zero.
    narrow, _ = fixed_highpass(constant, design_fixed_highpass(2000.0))
    assert np.all(narrow[-100:] == 0)
    # At 30 kS/s it is loop_dc_gain / 2 internal steps, shrunk by the guard bits.
    for guard in (0, 4, 6):
        filt = design_fixed_highpass(30_000.0, guard_bits=guard)
        wide, _ = fixed_highpass(constant, filt)
        bound = filt.loop_dc_gain / 2 / 2**guard
        assert np.all(wide[-100:] == wide[-1])  # it rests; it does not oscillate
        assert abs(int(wide[-1])) <= np.ceil(bound)
    assert abs(int(wide[-1])) <= 3


def test_noise_dithers_the_deadband_away() -> None:
    y, _ = fixed_highpass(CODES, design_fixed_highpass(30_000.0, guard_bits=6))
    assert abs(float(np.mean(y[5000:] - reference(30_000.0)[5000:]))) < 0.05


@pytest.mark.parametrize("fs_hz", [30_000.0, 2000.0])
def test_fixed_filter_matches_floating_point(fs_hz: float) -> None:
    y, peak = fixed_highpass(CODES, design_fixed_highpass(fs_hz, frac_bits=14, guard_bits=6))
    ref = reference(fs_hz)
    assert quantisation_snr_db(ref[5000:], y[5000:]) > 30.0
    assert np.abs(y[5000:] - ref[5000:]).max() <= 2
    assert accumulator_bits(peak) <= 32


def test_truncation_bias_is_half_a_step_times_the_loop_gain() -> None:
    filt = design_fixed_highpass(30_000.0, guard_bits=0, rounding=False)
    y, _ = fixed_highpass(CODES, filt)
    offset = float(np.mean(y[5000:] - reference(30_000.0)[5000:]))
    assert filt.loop_dc_gain == pytest.approx(264.0, rel=0.02)
    assert offset == pytest.approx(-0.5 * filt.loop_dc_gain, rel=0.03)
    rounded, _ = fixed_highpass(CODES, design_fixed_highpass(30_000.0, guard_bits=0))
    assert abs(float(np.mean(rounded[5000:] - reference(30_000.0)[5000:]))) < 2.0


def test_low_sample_rate_makes_truncation_harmless() -> None:
    filt = design_fixed_highpass(2000.0, guard_bits=0, rounding=False)
    y, _ = fixed_highpass(CODES, filt)
    assert filt.loop_dc_gain < 3.0
    assert abs(float(np.mean(y[2000:] - reference(2000.0)[2000:]))) < 1.5


def test_guard_bits_buy_accuracy() -> None:
    ref = reference(30_000.0)[5000:]
    snr = [
        quantisation_snr_db(
            ref, fixed_highpass(CODES, design_fixed_highpass(30_000.0, guard_bits=g))[0][5000:]
        )
        for g in (0, 2, 4, 6)
    ]
    assert snr == sorted(snr)
    assert snr[-1] - snr[0] > 25.0


def test_too_few_coefficient_bits_break_the_30k_filter_but_not_the_2k_one() -> None:
    wide = fixed_highpass(CODES, design_fixed_highpass(30_000.0, frac_bits=6, guard_bits=6))[0]
    narrow = fixed_highpass(CODES, design_fixed_highpass(2000.0, frac_bits=6, guard_bits=6))[0]
    assert quantisation_snr_db(reference(30_000.0)[5000:], wide[5000:]) < 20.0
    assert quantisation_snr_db(reference(2000.0)[2000:], narrow[2000:]) > 30.0


def test_fixed_threshold_events_and_dead_time() -> None:
    x = np.zeros(400, dtype=np.int64)
    x[[50, 51, 60, 200]] = -40
    assert fixed_threshold_events(x, 30, refractory=30).tolist() == [50, 200]
    assert fixed_threshold_events(x, 30, refractory=5).tolist() == [50, 60, 200]
    assert fixed_threshold_events(x, 50, refractory=5).size == 0


def test_fixed_neo_is_the_integer_neo_one_sample_late() -> None:
    x = np.array([0, 1, 4, 2, -3, 0, 0], dtype=np.int64)
    score = fixed_neo_score(x, window=1)
    expected = x[1:-1] ** 2 - x[:-2] * x[2:]
    np.testing.assert_array_equal(score[2:], expected)
    summed = fixed_neo_score(x, window=2)
    np.testing.assert_array_equal(summed[3:], expected[1:] + expected[:-1])


def test_fixed_sbp_sums_absolute_values_per_bin() -> None:
    x = np.array([1, -2, 3, -4, 5, -6, 7], dtype=np.int64)
    np.testing.assert_array_equal(fixed_sbp(x, 3), [6, 15])
    assert fixed_sbp(x, 3).dtype == np.int64
