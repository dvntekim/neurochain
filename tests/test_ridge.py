import numpy as np
import pytest

from neurochain.decode.ridge import blocked_cross_validation, fit_ridge, lagged, r_squared


def test_lagged_stacks_history_within_a_trial_only() -> None:
    x = np.arange(1.0, 7.0).reshape(-1, 1)
    groups = np.array([0, 0, 0, 1, 1, 1])
    out = lagged(x, groups, 3)
    assert out.shape == (6, 3)
    np.testing.assert_array_equal(out[:, 0], x[:, 0])
    np.testing.assert_array_equal(out[:, 1], [0, 1, 2, 0, 4, 5])
    np.testing.assert_array_equal(out[:, 2], [0, 0, 1, 0, 0, 4])


def test_ridge_recovers_a_linear_map() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2000, 4))
    w = np.array([1.0, -2.0, 0.5, 0.0])
    weights, intercept = fit_ridge(x, x @ w + 3.0, alpha=1e-9)
    # The intercept is the target mean, so weights are exact only up to the inputs' small mean.
    np.testing.assert_allclose(weights, w, atol=5e-3)
    assert intercept == pytest.approx(3.0 + float((x @ w).mean()))
    shrunk, _ = fit_ridge(x, x @ w, alpha=10.0)
    assert np.linalg.norm(shrunk) < 0.2 * np.linalg.norm(w)


def test_r_squared() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r_squared(y, y) == 1.0
    assert r_squared(y, np.full(4, y.mean())) == pytest.approx(0.0)
    assert r_squared(y, -y) < 0.0


def test_cross_validation_scores_signal_and_not_noise() -> None:
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(40), 50)
    features = rng.standard_normal((2000, 6))
    target = features[:, 0] - 0.5 * features[:, 3] + 0.3 * rng.standard_normal(2000)
    good = blocked_cross_validation(features, target, groups)
    assert good.r2 > 0.85
    assert good.r2_folds.size == 5
    assert good.correlation > 0.9
    assert good.r2_sem < 0.05
    noise = blocked_cross_validation(features, rng.standard_normal(2000), groups)
    assert noise.r2 < 0.05


def test_cross_validation_uses_history() -> None:
    rng = np.random.default_rng(2)
    groups = np.repeat(np.arange(20), 100)
    features = rng.standard_normal((2000, 2))
    target = np.roll(features[:, 0], 2)  # depends only on the feature two bins ago
    assert blocked_cross_validation(features, target, groups, n_lags=1).r2 < 0.05
    assert blocked_cross_validation(features, target, groups, n_lags=4).r2 > 0.9
