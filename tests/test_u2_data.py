"""U2 on the real inputs. Needs the Nason files, and for one test the MEArec recording."""

import json
from pathlib import Path

import pytest

from neurochain.analysis import u2
from neurochain.analysis.u2_plots import plot_sbp
from neurochain.data import nason
from neurochain.data.mearec import DEFAULT_FILE, load_mearec, mearec_dir
from neurochain.paths import results_dir

HAVE_NASON = all(
    (nason.nason_dir() / name).is_file()
    for name in (nason.IN_VIVO_FILE, nason.SIMULATION_FILE, nason.ONLINE_FILE)
)
HAVE_MEAREC = (mearec_dir() / DEFAULT_FILE).is_file()
pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(not HAVE_NASON, reason="Nason files not downloaded"),
]


@pytest.fixture(scope="module")
def committed() -> dict:
    return json.loads((results_dir() / "u2_metrics.json").read_text())


def test_sbp_is_reproduced_and_follows_the_largest_unit(committed: dict) -> None:
    config = u2.U2Config()
    simulation = u2.nason_simulation(config)["arrangements"]
    for mine, stored in zip(
        simulation, committed["nason"]["simulation"]["arrangements"], strict=True
    ):
        assert mine["sbp_reproduction_max_error"] < 1e-9
        assert mine["sbp_r"] == pytest.approx(stored["sbp_r"], rel=1e-9)
        # SBP follows units in order of their size.
        assert mine["sbp_r"][0] == max(mine["sbp_r"])
    in_vivo = u2.nason_in_vivo(config)
    assert in_vivo["sbp_reproduction_max_error_v"] < 1e-12
    assert in_vivo["sbp_r"][0] == pytest.approx(committed["nason"]["in_vivo"]["sbp_r"][0], rel=1e-9)
    assert in_vivo["sbp_r"][0] > 3 * in_vivo["crossings_r"][0]


def test_sbp_decodes_as_well_as_crossings(committed: dict) -> None:
    online = u2.nason_online()
    stored = committed["nason"]["online"]
    for key in ("velocity_sbp", "velocity_crossings", "velocity_both"):
        assert online[key]["r2"] == pytest.approx(stored[key]["r2"], rel=1e-9)
    assert online["velocity_sbp"]["r2"] > online["velocity_crossings"]["r2"] - 0.05
    assert online["velocity_both"]["r2"] > online["velocity_sbp"]["r2"]
    assert online["n_bins"] == 10_458


@pytest.mark.skipif(not HAVE_MEAREC, reason="MEArec recording not generated")
def test_committed_metrics_match_a_fresh_run(committed: dict, tmp_path: Path) -> None:
    """The numbers in results/u2_metrics.json (and so the README) come from this code."""
    fresh = u2.compute_u2(recording=load_mearec())
    for sweep, methods in committed["mearec"]["detection"].items():
        for method, values in methods.items():
            for value, row in values.items():
                assert fresh["mearec"]["detection"][sweep][method][value]["f1"] == pytest.approx(
                    row["f1"], abs=1e-9
                )
    for key, value in committed["mearec"]["fixed_point"].items():
        assert fresh["mearec"]["fixed_point"][key] == pytest.approx(value, abs=1e-6)
    assert plot_sbp(fresh, tmp_path / "sbp.png").stat().st_size > 20_000
