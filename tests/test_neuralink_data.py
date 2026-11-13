"""Checks against the real Neuralink clips. Skipped when the data is absent."""

import json

import numpy as np
import pytest

from neurochain.analysis.u0 import U0Config, analyse_paths, summarise
from neurochain.data import neuralink
from neurochain.data.download import clips_dir
from neurochain.paths import results_dir

pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(
        not any(clips_dir("neuralink").glob("*.wav")), reason="Neuralink clips not downloaded"
    ),
]


def test_dataset_shape() -> None:
    clips = neuralink.list_clips()
    assert len(clips) == 743
    seconds = 0.0
    for path in clips:
        pcm, fs_hz = neuralink.read_pcm(path)
        assert fs_hz == neuralink.FS_HZ
        seconds += pcm.size / fs_hz
    assert seconds == pytest.approx(3757.3, abs=0.1)


def test_shift_by_six_is_lossless() -> None:
    """Every int16 value that occurs maps to its own 10-bit code."""
    values = np.unique(
        np.concatenate([np.unique(neuralink.read_pcm(p)[0]) for p in neuralink.list_clips()])
    )
    codes = neuralink.int16_to_adc_code(values)
    assert values.size == 1023
    assert np.unique(codes).size == values.size
    assert (int(codes.min()), int(codes.max())) == (-512, 511)


def test_committed_metrics_match_a_fresh_run() -> None:
    """The numbers in results/u0_metrics.json (and so the README) come from this code."""
    committed = json.loads((results_dir() / "u0_metrics.json").read_text())
    fresh = summarise(
        analyse_paths(neuralink.list_clips(), neuralink.load_clip, U0Config(), progress_every=0)
    )
    assert fresh["dataset"]["n_clean_clips"] == committed["dataset"]["n_clean_clips"]
    assert fresh["spikes"]["n_spikes"] == committed["spikes"]["n_spikes"]
    for section, key in (("noise", "sigma"), ("spectrum", "lfp_power_fraction")):
        assert fresh[section][key]["median"] == pytest.approx(
            committed[section][key]["median"], rel=1e-9
        )
    assert fresh["spikes"]["amplitude"]["median"] == pytest.approx(
        committed["spikes"]["amplitude"]["median"], rel=1e-9
    )
