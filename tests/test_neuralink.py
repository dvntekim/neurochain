import wave
from pathlib import Path

import numpy as np
import pytest

from neurochain.data import neuralink


def write_wav(path: Path, pcm: np.ndarray, *, fs_hz: int = 19_531, channels: int = 1) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(fs_hz)
        handle.writeframes(pcm.astype("<i2").tobytes())


def test_code_mapping_covers_the_rails() -> None:
    pcm = np.array([-32768, -32704, -64, -1, 0, 63, 64, 32703, 32767], dtype=np.int16)
    codes = neuralink.int16_to_adc_code(pcm)
    assert codes.tolist() == [-512, -511, -1, -1, 0, 0, 1, 510, 511]
    assert codes.min() == neuralink.ADC_CODE_MIN
    assert codes.max() == neuralink.ADC_CODE_MAX


def test_saturation_fraction() -> None:
    codes = np.array([-512, 0, 0, 511], dtype=np.int16)
    assert neuralink.saturation_fraction(codes) == pytest.approx(0.5)
    assert neuralink.saturation_fraction(np.zeros(8)) == 0.0


def test_load_clip_returns_codes_with_provenance(tmp_path: Path) -> None:
    pcm = np.array([0, 64, 128, -64, 32767, -32768], dtype=np.int16)
    path = tmp_path / "abc-123.wav"
    write_wav(path, pcm)
    recording = neuralink.load_clip(path)
    assert recording.samples.shape == (1, 6)
    assert recording.channel(0).tolist() == [0.0, 1.0, 2.0, -1.0, 511.0, -512.0]
    assert recording.fs_hz == neuralink.FS_HZ
    assert recording.units == "ADC LSB"
    assert recording.meta["clip_id"] == "abc-123"
    assert recording.meta["saturation_fraction"] == pytest.approx(2 / 6)


def test_load_clip_rejects_stereo(tmp_path: Path) -> None:
    path = tmp_path / "stereo.wav"
    write_wav(path, np.zeros(8, dtype=np.int16), channels=2)
    with pytest.raises(ValueError, match="mono 16-bit"):
        neuralink.load_clip(path)


def test_list_clips_is_sorted_and_explains_missing_data(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make data"):
        neuralink.list_clips(tmp_path)
    for name in ("b.wav", "a.wav"):
        write_wav(tmp_path / name, np.zeros(4, dtype=np.int16))
    assert [p.name for p in neuralink.list_clips(tmp_path)] == ["a.wav", "b.wav"]
