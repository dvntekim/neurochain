"""Reproduce the U2 figures and results table.

Run with ``make u2`` or ``uv run python scripts/u2_figures.py``. Writes four
figures under ``figures/`` and ``results/u2_metrics.json``.

Needs the MEArec recording (``make mearec``) and the Nason files. ``--strict``
makes a missing input an error instead of a gap.
"""

from __future__ import annotations

import argparse
import json
import sys

from neurochain.analysis.u2 import compute_u2
from neurochain.analysis.u2_plots import plot_adc, plot_detection, plot_fixed_point, plot_sbp
from neurochain.data import nason
from neurochain.data.mearec import DEFAULT_FILE, load_mearec, mearec_dir
from neurochain.paths import figures_dir, results_dir


def main() -> int:
    """Run the analysis, then write the figures and metrics."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--strict", action="store_true", help="fail if an input is missing")
    args = parser.parse_args()

    have_mearec = (mearec_dir() / DEFAULT_FILE).is_file()
    have_nason = all(
        (nason.nason_dir() / name).is_file()
        for name in (nason.IN_VIVO_FILE, nason.SIMULATION_FILE, nason.ONLINE_FILE)
    )
    missing = [
        name
        for name, present in (
            ("the MEArec recording", have_mearec),
            ("the Nason files", have_nason),
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

    summary = compute_u2(recording=load_mearec() if have_mearec else None, include_nason=have_nason)
    metrics_path = results_dir() / "u2_metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {metrics_path}")
    plots = []
    if have_mearec:
        plots += [(plot_detection, "u2_detection_f1.png"), (plot_adc, "u2_adc_sweeps.png"),
                  (plot_fixed_point, "u2_fixed_point.png")]  # fmt: skip
    if have_mearec and have_nason:
        plots.append((plot_sbp, "u2_sbp_vs_crossings.png"))
    for plot, name in plots:
        print(f"wrote {plot(summary, figures_dir() / name)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
