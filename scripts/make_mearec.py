"""Generate the MEArec ground-truth recording used from U2 onward.

Run with ``make mearec`` (``uv run --group sim python scripts/make_mearec.py``).
MEArec, NEURON and LFPy live in the optional ``sim`` dependency group, so the
rest of the project, and CI, do not need them. The result is written as a plain
``.npz`` that :mod:`neurochain.data.mearec` reads without MEArec installed.

Steps:

1. Copy MEArec's bundled cortical cell models into ``data/mearec`` and compile
   their NEURON mechanisms. Two synapse mechanisms predate NEURON 9 and are
   patched in the copy (three one-line C changes; the originals are untouched).
2. Simulate extracellular action-potential templates on a 32-channel linear
   probe.
3. Build two recordings from the same spike trains and templates: one with no
   noise, one with MEArec's far-neuron background at 10 µV. Their difference is
   the background alone, which the loader rescales to any level.

Everything is seeded, so repeating the run on the same platform gives the same
file.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from neurochain.data.mearec import DEFAULT_FILE, mearec_dir

PROBE = "Neuronexus-32"
TEMPLATES_PER_CELL_MODEL = 80
TEMPLATE_DISTANCE_UM = (10.0, 150.0)
"""Lateral distance of simulated neurons from the probe. Near ones give the
detectable units; far ones give the small templates the background is built of."""
FAR_NEURON_MAX_AMPLITUDE_UV = 20.0
N_FAR_NEURONS = 300
FS_HZ = 30_000.0
DURATION_S = 60.0
N_EXCITATORY = 12
N_INHIBITORY = 4
MIN_AMPLITUDE_UV = 30.0
MAX_AMPLITUDE_UV = 300.0
REFERENCE_NOISE_UV = 10.0
SEED = 0

LEGACY_MECHANISMS = ("ProbAMPANMDA_EMS.mod", "ProbGABAAB_EMS.mod")
LEGACY_PATCHES = (
    # NEURON 9 declares these itself, with different return types.
    ("double nrn_random_pick(void* r);\n", ""),
    ("void* nrn_random_arg(int argpos);\n", ""),
    ("nrn_random_pick(_p_rng)", "nrn_random_pick((Rand*)_p_rng)"),
    ("scop_random(1)", "scop_random()"),
)


def prepare_cell_models(source: Path, target: Path) -> Path:
    """Copy the cell models, patch the legacy mechanisms and compile them."""
    models = target / "bbp"
    if (models / "mods" / "arm64" / "special").exists() or any((models / "mods").glob("*/special")):
        return models
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source.parent, target, ignore=shutil.ignore_patterns("arm64", "x86_64"))
    for name in LEGACY_MECHANISMS:
        path = models / "mods" / name
        text = path.read_text()
        for old, new in LEGACY_PATCHES:
            text = text.replace(old, new)
        path.write_text(text)
    nrnivmodl = Path(sys.executable).parent / "nrnivmodl"
    subprocess.run([str(nrnivmodl)], cwd=models / "mods", check=True, capture_output=True)
    return models


def main() -> None:
    """Generate templates and recordings, then write the ``.npz``."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--force", action="store_true", help="regenerate even if the file exists")
    args = parser.parse_args()
    import MEArec as mr  # noqa: N813 - imported here so --help works without the sim group

    out_dir = mearec_dir()
    out_path = out_dir / DEFAULT_FILE
    if out_path.is_file() and not args.force:
        print(f"{out_path} exists; use --force to regenerate")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    models = prepare_cell_models(Path(mr.get_default_cell_models_folder()), out_dir / "cell_models")
    templates_path = out_dir / "templates.h5"
    if templates_path.is_file() and not args.force:
        tempgen = mr.load_templates(str(templates_path), verbose=False)
    else:
        params = mr.get_default_templates_params()
        params.update(
            probe=PROBE,
            n=TEMPLATES_PER_CELL_MODEL,
            seed=SEED,
            xlim=list(TEMPLATE_DISTANCE_UM),
            min_amp=0,
        )
        tempgen = mr.gen_templates(
            str(models),
            params=params,
            n_jobs=8,
            verbose=False,
            templates_tmp_folder=str(out_dir / "tmp"),
            delete_tmp=True,
        )
        mr.save_template_generator(tempgen, str(templates_path), verbose=False)
    print(f"templates: {np.asarray(tempgen.templates).shape}")

    def record(noise_uv: float) -> object:
        params = mr.get_default_recordings_params()
        params["spiketrains"].update(duration=DURATION_S, n_exc=N_EXCITATORY, n_inh=N_INHIBITORY)
        params["templates"].update(min_amp=MIN_AMPLITUDE_UV, max_amp=MAX_AMPLITUDE_UV)
        params["recordings"].update(
            fs=FS_HZ,
            noise_level=noise_uv,
            noise_mode="far-neurons",
            filter=False,
            chunk_duration=10,
            far_neurons_n=N_FAR_NEURONS,
            far_neurons_max_amp=FAR_NEURON_MAX_AMPLITUDE_UV,
        )
        params["seeds"].update(spiketrains=SEED, templates=SEED, convolution=SEED, noise=SEED)
        return mr.gen_recordings(params=params, tempgen=tempgen, verbose=False)

    clean = record(0.0)
    noisy = record(REFERENCE_NOISE_UV)
    clean_uv = np.asarray(clean.recordings, dtype=np.float64).T
    background_uv = np.asarray(noisy.recordings, dtype=np.float64).T - clean_uv
    templates = np.asarray(clean.templates, dtype=np.float64)
    if templates.ndim == 4:  # (units, jitters, channels, samples)
        templates = templates.mean(axis=1)
    spike_samples = [
        np.round(np.asarray(train.rescale("s").magnitude) * FS_HZ).astype(np.int64)
        for train in clean.spiketrains
    ]
    for a, b in zip(clean.spiketrains, noisy.spiketrains, strict=True):
        if not np.array_equal(np.asarray(a.magnitude), np.asarray(b.magnitude)):
            raise RuntimeError("the two recordings do not share spike trains")
    np.savez_compressed(
        out_path,
        fs_hz=FS_HZ,
        clean_uv=clean_uv.astype(np.float32),
        background_uv=background_uv.astype(np.float32),
        reference_noise_uv=REFERENCE_NOISE_UV,
        spike_samples=np.concatenate(spike_samples),
        spike_unit=np.repeat(np.arange(len(spike_samples)), [s.size for s in spike_samples]),
        templates_uv=templates.astype(np.float32),
        cell_types=np.asarray([str(t.annotations.get("cell_type", "")) for t in clean.spiketrains]),
        probe=PROBE,
        seed=SEED,
        mearec_version=mr.__version__,
    )
    print(f"wrote {out_path}: {clean_uv.shape[0]} channels, {clean_uv.shape[1] / FS_HZ:.0f} s, "
          f"{len(spike_samples)} units, {sum(s.size for s in spike_samples)} spikes")  # fmt: skip


if __name__ == "__main__":
    main()
