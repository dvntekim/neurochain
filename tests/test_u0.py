import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from neurochain.analysis.u0 import (
    U0Config,
    U0Results,
    analyse_clip,
    analyse_paths,
    analyse_recordings,
    example_clip_index,
    summarise,
)
from neurochain.analysis.u0_plots import densest_window, plot_spectra, plot_spike_amplitudes
from neurochain.data.synthetic import make_synthetic_recording
from neurochain.recording import Recording


def clip(
    seed: int, *, saturation: float = 0.0, rates: tuple[float, float] = (8.0, 5.0)
) -> Recording:
    recording, _ = make_synthetic_recording(seed=seed, duration_s=5.0, rates_hz=rates)
    recording.meta.update(clip_id=f"clip{seed}", saturation_fraction=saturation)
    return recording


@pytest.fixture(scope="module")
def results() -> U0Results:
    recordings = [clip(1), clip(2), clip(3, saturation=0.5), clip(4, rates=(0.0, 0.0))]
    return analyse_recordings(recordings, U0Config())


def test_clip_analysis_recovers_planted_quantities() -> None:
    recording, truth = make_synthetic_recording(seed=11, duration_s=10.0)
    result = analyse_clip(recording, U0Config())
    n_true = truth.all_spike_samples().size
    assert result.spike_amplitudes.size == pytest.approx(n_true, rel=0.05)
    # LFP of 30 µV RMS and 8 µV of white noise (mostly in the spike band).
    assert np.sqrt(result.lfp_power) == pytest.approx(30.0, rel=0.1)
    assert result.sigma == pytest.approx(8.0 * np.sqrt(0.72), rel=0.06)
    assert np.all(result.spike_amplitudes < 0)


def test_results_shapes_and_clean_mask(results: U0Results) -> None:
    assert results.n_clips == 4
    assert results.clean.tolist() == [True, True, False, True]
    assert results.psd.shape == (4, results.freqs_hz.size)
    assert results.spike_clip_index.size == results.spike_amplitudes.size == results.n_spikes.sum()
    assert results.spike_snr.min() >= results.config.threshold_sigma


def test_summary_uses_clean_clips_only(results: U0Results) -> None:
    summary = summarise(results)
    assert summary["dataset"]["n_clean_clips"] == 3
    assert summary["dataset"]["n_saturated_clips_excluded"] == 1
    assert summary["spikes"]["n_spikes"] == int(results.n_spikes[results.clean].sum())
    quiet = int(np.sum(results.n_spikes[results.clean] == 0))
    assert summary["spikes"]["clips_without_spikes"] == quiet
    assert summary["spikes"]["detected_polarity_fraction"] > 0.9
    assert 0.5 < summary["spectrum"]["lfp_power_fraction"]["median"] < 1.0
    assert 0.0 < summary["spikes"]["gaussian_noise_crossing_rate_hz"] < 0.2
    json.dumps(summary)  # must be serialisable as written


def test_summary_requires_a_clean_clip() -> None:
    saturated = analyse_recordings([clip(1, saturation=1.0)], U0Config())
    with pytest.raises(ValueError, match="saturation"):
        summarise(saturated)


def test_lower_threshold_finds_more_events(results: U0Results) -> None:
    lower = analyse_recordings([clip(1), clip(2)], replace(U0Config(), threshold_sigma=3.0))
    assert lower.n_spikes[:2].sum() > results.n_spikes[:2].sum()


def test_dropping_bits_adds_quantisation_noise() -> None:
    recording = clip(5)
    integer = Recording(np.round(recording.samples), recording.fs_hz, "ADC LSB", "synthetic")
    fine = analyse_clip(integer, U0Config())
    coarse = analyse_clip(integer, replace(U0Config(), drop_bits=3))
    # Step 8 adds 64/12 LSB² of white noise, 72 % of it inside the spike band.
    expected = np.sqrt(fine.sigma**2 + 0.72 * 64 / 12)
    assert coarse.sigma == pytest.approx(expected, rel=0.03)


def test_example_clip_rule(results: U0Results) -> None:
    index = example_clip_index(results)
    assert results.clean[index]
    assert results.n_spikes[index] >= 20
    with pytest.raises(ValueError, match="no clean clip"):
        example_clip_index(results, min_spikes=10_000)


def test_save_load_roundtrip(results: U0Results, tmp_path: Path) -> None:
    path = tmp_path / "cache" / "u0.npz"
    results.save(path)
    loaded = U0Results.load(path)
    assert loaded.config == results.config
    assert loaded.clip_ids == results.clip_ids
    assert loaded.units == results.units
    np.testing.assert_array_equal(loaded.psd, results.psd)
    np.testing.assert_array_equal(loaded.spike_amplitudes, results.spike_amplitudes)
    assert summarise(loaded) == summarise(results)


def test_analyse_paths_caches_by_config(tmp_path: Path) -> None:
    calls: list[str] = []

    def loader(path: Path) -> Recording:
        calls.append(path.name)
        return clip(int(path.stem))

    paths = [Path("1.wav"), Path("2.wav")]
    first = analyse_paths(paths, loader, U0Config(), cache_dir=tmp_path, progress_every=0)
    again = analyse_paths(paths, loader, U0Config(), cache_dir=tmp_path, progress_every=0)
    assert len(calls) == 2  # the second call was served from the cache
    np.testing.assert_array_equal(first.sigma, again.sigma)
    analyse_paths(
        paths,
        loader,
        replace(U0Config(), threshold_sigma=4.0),
        cache_dir=tmp_path,
        progress_every=0,
    )
    assert len(calls) == 4  # a different config is a different cache entry


def test_mixed_sampling_rates_are_rejected() -> None:
    other, _ = make_synthetic_recording(seed=1, duration_s=5.0, fs_hz=30_000.0)
    with pytest.raises(ValueError, match="one sampling rate"):
        analyse_recordings([clip(1), other], U0Config())
    with pytest.raises(ValueError, match="no recordings"):
        analyse_recordings([], U0Config())


def test_densest_window() -> None:
    indices = np.array([10, 5000, 5100, 5200, 9000])
    start, stop = densest_window(indices, 10_000, 1000)
    assert stop - start == 1000
    assert np.count_nonzero((indices >= start) & (indices < stop)) == 3
    assert densest_window(np.array([], dtype=np.int64), 500, 1000) == (0, 500)


def test_figures_are_written(results: U0Results, tmp_path: Path) -> None:
    summary = summarise(results)
    example = clip(1)
    for plot in (plot_spectra, plot_spike_amplitudes):
        path = plot(
            results, summary, example, tmp_path / f"{plot.__name__}.png", dataset_label="Synthetic"
        )
        assert path.stat().st_size > 20_000
