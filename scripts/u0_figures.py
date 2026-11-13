"""Reproduce the U0 figures and results table from the Neuralink recordings.

Run with ``make u0`` or ``uv run python scripts/u0_figures.py``. Writes:

* ``figures/u0_spectra.png``
* ``figures/u0_spike_amplitudes.png``
* ``results/u0_metrics.json``
"""

from __future__ import annotations

import argparse
import json

from neurochain.analysis.u0 import U0Config, analyse_paths, example_clip_index, summarise
from neurochain.analysis.u0_plots import plot_spectra, plot_spike_amplitudes
from neurochain.data.neuralink import list_clips, load_clip
from neurochain.paths import cache_dir, figures_dir, results_dir

DATASET_LABEL = "Neuralink Compression Challenge (N1 implant, non-human primate motor cortex)"


def main() -> None:
    """Analyse every clip, then write both figures and the metrics file."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--no-cache", action="store_true", help="recompute even if cached")
    args = parser.parse_args()

    config = U0Config()
    clips = list_clips()
    results = analyse_paths(
        clips, load_clip, config, cache_dir=None if args.no_cache else cache_dir()
    )
    summary = summarise(results)
    example_index = example_clip_index(results)
    example = load_clip(clips[example_index])
    summary["example_clip_id"] = results.clip_ids[example_index]

    metrics_path = results_dir() / "u0_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {metrics_path}")
    for plot, name in (
        (plot_spectra, "u0_spectra.png"),
        (plot_spike_amplitudes, "u0_spike_amplitudes.png"),
    ):
        path = plot(results, summary, example, figures_dir() / name, dataset_label=DATASET_LABEL)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
