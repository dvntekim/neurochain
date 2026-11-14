"""Ridge regression on lagged features, scored by blocked cross-validation.

A linear readout is the baseline every other decoder has to beat, and it is
enough to compare two feature sets fairly: same model, same folds, different
inputs.

Folds are contiguous blocks of whole trials. Neighbouring time bins are highly
correlated, so shuffling bins into folds would leak the test set into training
and inflate every score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neurochain.recording import FloatArray, IntArray


def lagged(features: FloatArray, groups: IntArray, n_lags: int) -> FloatArray:
    """Stack each row with its ``n_lags - 1`` predecessors from the same group.

    Args:
        features: ``(n_bins, n_features)``.
        groups: ``(n_bins,)`` label of the trial each bin belongs to. A lag that
            would reach into another trial is filled with zeros.
        n_lags: Number of time steps per row, the current one included.

    Returns:
        ``(n_bins, n_features * n_lags)``.
    """
    columns = []
    for lag in range(n_lags):
        shifted = np.roll(features, lag, axis=0)
        crossed = np.roll(groups, lag) != groups
        crossed[:lag] = True
        shifted[crossed] = 0.0
        columns.append(shifted)
    return np.concatenate(columns, axis=1)


def fit_ridge(x: FloatArray, y: FloatArray, alpha: float) -> tuple[FloatArray, float]:
    """Fit ``y = x @ w + b`` with an L2 penalty ``alpha * n * |w|**2``.

    Inputs are expected to be standardised. Returns ``(weights, intercept)``.
    """
    intercept = float(np.mean(y))
    gram = x.T @ x + alpha * x.shape[0] * np.eye(x.shape[1])
    weights = np.linalg.solve(gram, x.T @ (y - intercept))
    return np.asarray(weights, dtype=np.float64), intercept


def r_squared(y: FloatArray, prediction: FloatArray) -> float:
    """Return the coefficient of determination, ``1 - SS_res / SS_tot``."""
    residual = float(np.sum((y - prediction) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - residual / total


@dataclass(frozen=True)
class CrossValidation:
    """Scores of a blocked cross-validation.

    Attributes:
        r2_folds: Coefficient of determination on each held-out fold.
        correlation_folds: Pearson correlation on each held-out fold.
    """

    r2_folds: FloatArray
    correlation_folds: FloatArray

    @property
    def r2(self) -> float:
        """Mean held-out R²."""
        return float(np.mean(self.r2_folds))

    @property
    def r2_sem(self) -> float:
        """Standard error of the mean R² across folds."""
        return float(np.std(self.r2_folds, ddof=1) / np.sqrt(self.r2_folds.size))

    @property
    def correlation(self) -> float:
        """Mean held-out correlation."""
        return float(np.mean(self.correlation_folds))


def blocked_cross_validation(
    features: FloatArray,
    target: FloatArray,
    groups: IntArray,
    *,
    n_folds: int = 5,
    n_lags: int = 5,
    alpha: float = 0.01,
) -> CrossValidation:
    """Score a lagged ridge readout on held-out blocks of trials.

    Features are standardised with the training fold's mean and standard
    deviation only.

    Args:
        features: ``(n_bins, n_features)``.
        target: ``(n_bins,)`` the quantity to decode.
        groups: ``(n_bins,)`` trial label of each bin.
        n_folds: Number of contiguous blocks of trials.
        n_lags: Time steps of history given to the readout.
        alpha: Ridge penalty.
    """
    x = lagged(features, groups, n_lags)
    blocks = np.array_split(np.unique(groups), n_folds)
    r2, correlation = [], []
    for block in blocks:
        test = np.isin(groups, block)
        mean = x[~test].mean(axis=0)
        scale = x[~test].std(axis=0) + 1e-9
        weights, intercept = fit_ridge((x[~test] - mean) / scale, target[~test], alpha)
        prediction = (x[test] - mean) / scale @ weights + intercept
        r2.append(r_squared(target[test], prediction))
        correlation.append(float(np.corrcoef(prediction, target[test])[0, 1]))
    return CrossValidation(np.asarray(r2), np.asarray(correlation))
