import numpy as np
import pytest

from gboost import LOSSES, LogLoss, SquaredError


def numerical_gradient(loss, y, raw, eps=1e-6):
    """Central differences, scaled back up because the losses return a mean."""
    out = np.empty_like(raw)
    for i in range(len(raw)):
        step = np.zeros_like(raw)
        step[i] = eps
        out[i] = (loss.value(y, raw + step) - loss.value(y, raw - step)) / (2 * eps) * len(y)
    return out


@pytest.mark.parametrize("name", ["squared", "logloss"])
def test_every_analytic_gradient_matches_a_numerical_one(name):
    """The check that makes the rest trustworthy.

    A boosting implementation with a wrong gradient still trains and still
    produces plausible numbers — it just converges to the wrong place. Comparing
    against finite differences is what catches it.
    """
    loss = LOSSES[name]
    y = np.array([0.0, 1.0, 1.0, 0.0, 1.0])
    raw = np.array([-1.2, 2.0, 0.0, 1.1, -0.4])

    analytic, _ = loss.gradient_hessian(y, raw)

    assert analytic == pytest.approx(numerical_gradient(loss, y, raw), abs=1e-5)


def test_the_log_loss_hessian_matches_a_numerical_second_derivative():
    loss = LogLoss()
    y = np.array([0.0, 1.0, 1.0])
    raw = np.array([-0.5, 0.8, 0.1])
    eps = 1e-4

    _, hessian = loss.gradient_hessian(y, raw)

    for i in range(len(raw)):
        step = np.zeros_like(raw)
        step[i] = eps
        forward, _ = loss.gradient_hessian(y, raw + step)
        backward, _ = loss.gradient_hessian(y, raw - step)
        assert hessian[i] == pytest.approx((forward[i] - backward[i]) / (2 * eps), rel=1e-3)


def test_squared_error_has_a_constant_hessian():
    """Which is why boosting on squared error is the same as fitting the residual."""
    _, hessian = SquaredError().gradient_hessian(np.array([1.0, 5.0]), np.array([0.0, 2.0]))
    assert np.all(hessian == 1.0)


def test_each_loss_starts_from_the_constant_that_minimises_it():
    y = np.array([1.0, 2.0, 2.0, 10.0])

    assert LOSSES["squared"].initial(y) == pytest.approx(y.mean())
    assert LOSSES["absolute"].initial(y) == pytest.approx(np.median(y))


def test_the_log_loss_initial_value_is_the_log_odds_of_the_base_rate():
    y = np.array([1.0, 1.0, 1.0, 0.0])
    assert LOSSES["logloss"].initial(y) == pytest.approx(np.log(0.75 / 0.25))


def test_log_loss_does_not_overflow_on_a_confident_prediction():
    loss = LOSSES["logloss"]
    value = loss.value(np.array([1.0, 0.0]), np.array([800.0, -800.0]))

    assert np.isfinite(value)
    assert value < 1e-6


def test_predictions_come_back_as_probabilities_for_classification():
    p = LOSSES["logloss"].to_prediction(np.array([-40.0, 0.0, 40.0]))

    assert p.min() >= 0 and p.max() <= 1
    assert p[1] == pytest.approx(0.5)


def test_absolute_error_is_unmoved_by_how_wrong_an_outlier_is():
    """A point twice as wrong pulls no harder than one just wrong."""
    loss = LOSSES["absolute"]
    gradient_near, _ = loss.gradient_hessian(np.array([0.0]), np.array([1.0]))
    gradient_far, _ = loss.gradient_hessian(np.array([0.0]), np.array([1000.0]))

    assert gradient_near == gradient_far
