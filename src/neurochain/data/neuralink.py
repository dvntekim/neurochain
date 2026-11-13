"""Loader for the Neuralink Compression Challenge recordings.

What the files are (measured on the data, see ``tests/test_neuralink_data.py``):

* 743 mono 16-bit PCM WAV clips, each about 5 s, 19 531 Hz. Clips are named by
  UUID. The electrode each clip came from is not published, so a clip is treated
  as one anonymous channel.
* The implant's ADC is 10-bit. The WAV stretches those 1024 codes across the
  int16 range in steps of 64 or 65, so ``int16 >> 6`` recovers the signed 10-bit
  code exactly: 1023 distinct int16 values occur across the dataset and they map
  one-to-one onto 1023 codes.
* No gain or voltage scale is published. Samples are therefore reported in ADC
  LSB. Converting to microvolts would mean inventing a number.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from neurochain.data.download import clips_dir
from neurochain.recording import FloatArray, Recording

FS_HZ = 19_531.0
"""Sampling rate written in every WAV header."""

ADC_BITS = 10
ADC_CODE_MIN = -(1 << (ADC_BITS - 1))
ADC_CODE_MAX = (1 << (ADC_BITS - 1)) - 1
_INT16_SHIFT = 16 - ADC_BITS
UNITS = "ADC LSB"


def int16_to_adc_code(pcm: NDArray[np.int16]) -> NDArray[np.int16]:
    """Map WAV int16 samples to signed 10-bit ADC codes in ``[-512, 511]``.

    An arithmetic right shift by six bits. It is lossless on this dataset
    because every int16 value that occurs lands in its own code.
    """
    return pcm >> _INT16_SHIFT


def saturation_fraction(codes: NDArray[np.int16] | FloatArray) -> float:
    """Return the fraction of samples sitting on either ADC rail."""
    railed = (codes <= ADC_CODE_MIN) | (codes >= ADC_CODE_MAX)
    return float(np.mean(railed))


def list_clips(root: Path | None = None) -> list[Path]:
    """Return every clip path in sorted (deterministic) order.

    Args:
        root: Directory holding the WAV files. Defaults to the downloader's
            ``data/raw/neuralink/clips``.

    Raises:
        FileNotFoundError: If no clips are found, with the command that fetches
            them.
    """
    directory = root if root is not None else clips_dir("neuralink")
    clips = sorted(directory.glob("*.wav"))
    if not clips:
        raise FileNotFoundError(f"no Neuralink clips in {directory}; run `make data` first")
    return clips


def read_pcm(path: Path) -> tuple[NDArray[np.int16], float]:
    """Read a mono 16-bit WAV file and return ``(int16 samples, fs_hz)``.

    Raises:
        ValueError: If the file is not mono 16-bit PCM.
    """
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError(
                f"{path.name}: expected mono 16-bit PCM, got "
                f"{handle.getnchannels()} channel(s) at {8 * handle.getsampwidth()} bits"
            )
        fs_hz = float(handle.getframerate())
        frames = handle.readframes(handle.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.int16), fs_hz


def load_clip(path: Path) -> Recording:
    """Load one clip as a single-channel :class:`Recording` in ADC LSB.

    The DC level is kept: removing it is a front-end decision, not a loader's.
    ``meta`` records the clip id and the fraction of samples on the ADC rails.
    """
    pcm, fs_hz = read_pcm(path)
    codes = int16_to_adc_code(pcm)
    return Recording(
        samples=codes.astype(np.float64)[np.newaxis, :],
        fs_hz=fs_hz,
        units=UNITS,
        source="neuralink",
        meta={
            "clip_id": path.stem,
            "saturation_fraction": saturation_fraction(codes),
            "adc_bits": ADC_BITS,
        },
    )
