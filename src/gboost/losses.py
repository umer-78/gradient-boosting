"""Losses, each supplying the gradient and hessian boosting actually needs.

A boosting round does not fit the target — it fits the direction the loss wants
the prediction to move, and the curvature tells it how far. Writing each loss as
(value, gradient, hessian) is what lets the same tree code serve regression and
classification without either knowing about the other.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Loss:
    name: str

    def initial(self, y: np.ndarray) -> float:
        raise NotImplementedError

    def value(self, y: np.ndarray, raw: np.ndarray) -> float:
        raise NotImplementedError

    def gradient_hessian(self, y: np.ndarray, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def to_prediction(self, raw: np.ndarray) -> np.ndarray:
        return raw


@dataclass(frozen=True)
class SquaredError(Loss):
    name: str = "squared error"

    def initial(self, y: np.ndarray) -> float:
        return float(y.mean())

    def value(self, y: np.ndarray, raw: np.ndarray) -> float:
        return float(np.mean((y - raw) ** 2) / 2)

    def gradient_hessian(self, y: np.ndarray, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # The hessian is constant, which is why boosting on squared error is
        # equivalent to fitting each tree to the plain residual.
        return raw - y, np.ones_like(y)


@dataclass(frozen=True)
class AbsoluteError(Loss):
    """Robust to outliers: a point twice as wrong pulls no harder than one just wrong.

    The second derivative of |x| is zero everywhere it exists, so a constant
    stand-in is used. That makes the leaf value a scaled mean of signs rather
    than a true Newton step — the direction is right, the step size is a guess,
    and a lower learning rate is the usual compensation.
    """

    name: str = "absolute error"

    def initial(self, y: np.ndarray) -> float:
        return float(np.median(y))

    def value(self, y: np.ndarray, raw: np.ndarray) -> float:
        return float(np.mean(np.abs(y - raw)))

    def gradient_hessian(self, y: np.ndarray, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.sign(raw - y), np.ones_like(y)


@dataclass(frozen=True)
class LogLoss(Loss):
    """Binary cross-entropy on the log-odds scale.

    The model's output is a log-odds, never a probability: adding trees on the
    probability scale would let predictions leave [0, 1], and the sigmoid at the
    end is what keeps them inside it.
    """

    name: str = "log loss"

    def initial(self, y: np.ndarray) -> float:
        rate = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
        return float(np.log(rate / (1 - rate)))

    def value(self, y: np.ndarray, raw: np.ndarray) -> float:
        # log(1 + exp(raw)) - y * raw, written to avoid overflow for large raw.
        return float(np.mean(np.logaddexp(0.0, raw) - y * raw))

    def gradient_hessian(self, y: np.ndarray, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = self.to_prediction(raw)
        return p - y, np.maximum(p * (1 - p), 1e-6)

    def to_prediction(self, raw: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(raw, -60, 60)))


LOSSES = {
    "squared": SquaredError(),
    "absolute": AbsoluteError(),
    "logloss": LogLoss(),
}
