"""Scoring detections against ground-truth spike times."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neurochain.recording import IntArray


@dataclass(frozen=True)
class DetectionScore:
    """Counts of a detector's hits and misses.

    Attributes:
        true_positives: Target spikes that were detected.
        false_positives: Detections matching no spike at all.
        false_negatives: Target spikes that were missed.
        ignored: Detections that matched a spike outside the target set. They
            are real spikes from units too small to be required, so they count
            neither for nor against the detector.
    """

    true_positives: int
    false_positives: int
    false_negatives: int
    ignored: int = 0

    def __add__(self, other: DetectionScore) -> DetectionScore:
        """Pool the counts of two scores."""
        return DetectionScore(
            self.true_positives + other.true_positives,
            self.false_positives + other.false_positives,
            self.false_negatives + other.false_negatives,
            self.ignored + other.ignored,
        )

    @property
    def precision(self) -> float:
        """Fraction of counted detections that were target spikes."""
        counted = self.true_positives + self.false_positives
        return self.true_positives / counted if counted else 0.0

    @property
    def recall(self) -> float:
        """Fraction of target spikes that were detected."""
        targets = self.true_positives + self.false_negatives
        return self.true_positives / targets if targets else 0.0

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall."""
        denominator = 2 * self.true_positives + self.false_positives + self.false_negatives
        return 2 * self.true_positives / denominator if denominator else 0.0


def _match(detected: IntArray, truth: IntArray, tolerance: int) -> np.ndarray:
    """Greedily pair detections with true spikes; return a mask of matched detections."""
    matched = np.zeros(detected.size, dtype=bool)
    j = 0
    for t in truth:
        while j < detected.size and detected[j] < t - tolerance:
            j += 1
        if j < detected.size and abs(int(detected[j]) - int(t)) <= tolerance:
            matched[j] = True
            j += 1
    return matched


def score_detections(
    detected: IntArray,
    targets: IntArray,
    *,
    tolerance: int,
    ignore: IntArray | None = None,
    start: int = 0,
) -> DetectionScore:
    """Score detections against the spikes a detector is expected to find.

    Each target spike can be claimed by one detection within ``tolerance``
    samples. A leftover detection that lands on an ``ignore`` spike (a real
    spike of a unit too small to require) is set aside. Whatever remains is a
    false positive.

    Args:
        detected: Detected sample indices.
        targets: Spike samples the detector should find.
        tolerance: Largest timing error, in samples, that still counts.
        ignore: Spike samples that excuse a detection without rewarding it.
        start: Score only events at or after this sample (the part of the
            record that was not used for calibration).
    """
    detected = np.sort(detected[detected >= start])
    targets = np.sort(targets[targets >= start])
    hit = _match(detected, targets, tolerance)
    true_positives = int(hit.sum())
    leftover = detected[~hit]
    ignored = 0
    if ignore is not None and leftover.size:
        ignored = int(
            _match(leftover, np.sort(ignore[ignore >= start - tolerance]), tolerance).sum()
        )
    return DetectionScore(
        true_positives=true_positives,
        false_positives=int(leftover.size) - ignored,
        false_negatives=int(targets.size) - true_positives,
        ignored=ignored,
    )
