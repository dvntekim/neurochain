"""The committed U1 metrics against a fresh run. Needs ngspice and the Nason data."""

import json

import pytest

from neurochain.analysis.u1 import compute_u1
from neurochain.data import nason
from neurochain.frontend import spice
from neurochain.paths import results_dir

pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(not spice.ngspice_available(), reason="ngspice not installed"),
    pytest.mark.skipif(
        not (nason.nason_dir() / nason.IN_VIVO_FILE).is_file(), reason="Nason files not downloaded"
    ),
]


def test_committed_metrics_match_a_fresh_run() -> None:
    """The numbers in results/u1_metrics.json (and so the README) come from this code."""
    committed = json.loads((results_dir() / "u1_metrics.json").read_text())
    fresh = compute_u1(nason_recording=nason.load_in_vivo().recording()).summary
    assert fresh["spice_available"] and committed["spice_available"]
    for key, entry in committed["noise"].items():
        for method in ("hand", "analytic", "spice"):
            assert fresh["noise"][key][method]["total_uv"] == pytest.approx(
                entry[method]["total_uv"], rel=1e-6
            )
    for field in ("sigma_recorded_uv", "sigma_emulated_uv", "sigma_quarter_bias_uv"):
        assert fresh["nason"][field] == pytest.approx(committed["nason"][field], rel=1e-9)
    assert fresh["amplifier"]["f_high_hz"] == pytest.approx(committed["amplifier"]["f_high_hz"])


def test_nason_noise_is_several_times_the_model_floor() -> None:
    """The step-2 sanity check: a real channel is noisier than the front end alone."""
    committed = json.loads((results_dir() / "u1_metrics.json").read_text())
    assert 3.0 < committed["nason"]["recorded_over_model"] < 6.0
