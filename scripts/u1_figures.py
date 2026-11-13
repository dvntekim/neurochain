"""Reproduce the U1 figures and results table.

Run with ``make u1`` or ``uv run python scripts/u1_figures.py``. Writes:

* ``figures/u1_electrode_impedance.png``
* ``figures/u1_noise_budget.png``
* ``figures/u1_coupling.png``
* ``figures/u1_emulator_nason.png`` (needs the Nason data)
* ``results/u1_metrics.json``
* ``spice/*.cir`` (the reference netlists)

The SPICE columns need ngspice (``brew install ngspice``). ``--strict`` makes a
missing ngspice or a missing Nason file an error instead of a gap.
"""

from __future__ import annotations

import argparse
import json
import sys

from neurochain.analysis.u1 import compute_u1
from neurochain.analysis.u1_plots import (
    plot_coupling,
    plot_emulator,
    plot_impedance,
    plot_noise_budget,
)
from neurochain.data import nason
from neurochain.frontend import spice
from neurochain.paths import figures_dir, repo_root, results_dir


def main() -> int:
    """Run the analysis, then write the figures, metrics and netlists."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--strict", action="store_true", help="fail if ngspice or the Nason data is missing"
    )
    args = parser.parse_args()

    in_vivo_path = nason.nason_dir() / nason.IN_VIVO_FILE
    missing = [
        name
        for name, present in (
            ("ngspice", spice.ngspice_available()),
            ("the Nason in-vivo file", in_vivo_path.is_file()),
        )
        if not present
    ]
    if missing:
        print(
            f"warning: {' and '.join(missing)} not found; those results are left out",
            file=sys.stderr,
        )
        if args.strict:
            return 1

    recording = nason.load_in_vivo().recording() if in_vivo_path.is_file() else None
    results = compute_u1(nason_recording=recording)

    metrics_path = results_dir() / "u1_metrics.json"
    metrics_path.write_text(json.dumps(results.summary, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {metrics_path}")
    for name, text in spice.reference_netlists(results.frontend).items():
        path = repo_root() / "spice" / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(text)
        print(f"wrote {path}")
    plots = [
        (plot_impedance, "u1_electrode_impedance.png"),
        (plot_noise_budget, "u1_noise_budget.png"),
        (plot_coupling, "u1_coupling.png"),
    ]
    if results.nason is not None:
        plots.append((plot_emulator, "u1_emulator_nason.png"))
    for plot, name in plots:
        print(f"wrote {plot(results, figures_dir() / name)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
