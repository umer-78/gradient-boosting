"""Gradient boosting: many small trees, each correcting what the last ones left.

The loop is short enough to read in one sitting. Start from a constant. Ask the
loss which way each prediction should move and how sharply. Fit one shallow tree
to those gradients. Take a fraction of the step it suggests. Repeat.

The fraction — the learning rate — is the whole trick. Taking the full step makes
each tree chase the training data; taking a tenth of it and growing ten times as
many trees reaches a better answer, because no single tree ever gets to be
confident.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .losses import LOSSES, Loss
from .tree import RegressionTree, TreeParams, bin_features


@dataclass
class History:
    train: list[float] = field(default_factory=list)
    validation: list[float] = field(default_factory=list)

    @property
    def best_round(self) -> int:
        source = self.validation or self.train
        return int(np.argmin(source)) + 1 if source else 0


class GradientBoosting:
    """One implementation; the loss decides whether it is regression or classification."""

    def __init__(self, *, loss: str = "squared", n_estimators: int = 200,
                 learning_rate: float = 0.05, max_depth: int = 3,
                 min_samples_leaf: int = 20, reg_lambda: float = 1.0,
                 subsample: float = 1.0, colsample: float = 1.0,
                 max_bins: int = 64, early_stopping_rounds: int | None = None,
                 seed: int = 0):
        if loss not in LOSSES:
            raise ValueError(f"unknown loss {loss!r} — one of {', '.join(LOSSES)}")
        if not 0 < learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")
        if n_estimators < 1:
            raise ValueError("n_estimators must be at least 1")
        if not 0 < subsample <= 1:
            raise ValueError("subsample must be in (0, 1]")
        if not 0 < colsample <= 1:
            raise ValueError("colsample must be in (0, 1]")

        self.loss: Loss = LOSSES[loss]
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample = colsample
        self.max_bins = max_bins
        self.early_stopping_rounds = early_stopping_rounds
        self.seed = seed
        self.tree_params = TreeParams(max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                      reg_lambda=reg_lambda, max_bins=max_bins)

        self.base_score = 0.0
        self.trees: list[RegressionTree] = []
        self.history = History()
        self.n_features = 0
        self.stopped_early = False

    # -- fitting -----------------------------------------------------------

    def fit(self, x: np.ndarray, y: np.ndarray, *,
            eval_set: tuple[np.ndarray, np.ndarray] | None = None) -> GradientBoosting:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if len(x) != len(y):
            raise ValueError(f"{len(x)} rows against {len(y)} targets")
        if self.early_stopping_rounds and eval_set is None:
            raise ValueError(
                "early stopping needs an eval_set the model does not train on — "
                "stopping on the training loss never triggers, because it only ever falls"
            )

        rng = np.random.default_rng(self.seed)
        self.n_features = x.shape[1]
        binned, edges = bin_features(x, self.max_bins)

        self.base_score = self.loss.initial(y)
        raw = np.full(len(y), self.base_score)
        validation_raw = None
        if eval_set is not None:
            validation_raw = np.full(len(eval_set[1]), self.base_score)

        self.trees = []
        self.history = History()
        best = np.inf
        since_best = 0

        for _ in range(self.n_estimators):
            gradient, hessian = self.loss.gradient_hessian(y, raw)

            rows = np.arange(len(y))
            if self.subsample < 1:
                rows = rng.choice(len(y), size=max(1, int(len(y) * self.subsample)), replace=False)

            columns = np.arange(self.n_features)
            if self.colsample < 1:
                columns = np.sort(rng.choice(self.n_features,
                                             size=max(1, int(self.n_features * self.colsample)),
                                             replace=False))

            tree = RegressionTree(self.tree_params)
            # Rows outside the sample get a zero gradient and hessian, so they
            # contribute nothing to any split without needing a separate index.
            g = np.zeros_like(gradient)
            h = np.zeros_like(hessian)
            g[rows] = gradient[rows]
            h[rows] = hessian[rows]
            tree.fit(binned[:, columns], [edges[c] for c in columns], g, h)
            tree.columns = columns

            raw += self.learning_rate * tree.predict(x[:, columns])
            self.trees.append(tree)
            self.history.train.append(self.loss.value(y, raw))

            if eval_set is not None:
                validation_raw += self.learning_rate * tree.predict(eval_set[0][:, columns])
                score = self.loss.value(np.asarray(eval_set[1], dtype=np.float64), validation_raw)
                self.history.validation.append(score)

                if self.early_stopping_rounds:
                    if score < best - 1e-12:
                        best, since_best = score, 0
                    else:
                        since_best += 1
                        if since_best >= self.early_stopping_rounds:
                            self.stopped_early = True
                            break

        return self

    # -- prediction --------------------------------------------------------

    def decision_function(self, x: np.ndarray, n_trees: int | None = None) -> np.ndarray:
        """The raw score before any link function."""
        if not self.trees:
            raise ValueError("this model has not been fitted")
        x = np.asarray(x, dtype=np.float64)
        if x.shape[1] != self.n_features:
            raise ValueError(f"expected {self.n_features} features, got {x.shape[1]}")

        raw = np.full(len(x), self.base_score)
        for tree in self.trees[:n_trees]:
            raw += self.learning_rate * tree.predict(x[:, tree.columns])
        return raw

    def predict(self, x: np.ndarray, n_trees: int | None = None) -> np.ndarray:
        return self.loss.to_prediction(self.decision_function(x, n_trees))

    def predict_class(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict(x) >= threshold).astype(int)

    # -- importance --------------------------------------------------------

    def gain_importance(self) -> np.ndarray:
        """Total split gain per feature, normalised to sum to one.

        Fast, free, and biased: a feature with many distinct values gets more
        chances to produce a lucky split, so pure noise with high cardinality can
        outrank a real signal. Use `permutation_importance` when the answer
        matters.
        """
        totals = np.zeros(self.n_features)
        for tree in self.trees:
            local = tree.gain_by_feature()
            totals[tree.columns] += local
        return totals / totals.sum() if totals.sum() else totals

    def permutation_importance(self, x: np.ndarray, y: np.ndarray, *, repeats: int = 5,
                               seed: int | None = None) -> np.ndarray:
        """Shuffle one column and measure how much worse the model gets.

        Slower, and honest: a column that carries no information cannot get worse
        when it is scrambled, whatever the tree structure happened to do with it.
        Measured on held-out data, because on the training set a memorised noise
        column looks essential.
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        rng = np.random.default_rng(self.seed if seed is None else seed)

        baseline = self.loss.value(y, self.decision_function(x))
        out = np.zeros(self.n_features)

        for feature in range(self.n_features):
            damage = 0.0
            for _ in range(repeats):
                shuffled = x.copy()
                shuffled[:, feature] = rng.permutation(shuffled[:, feature])
                damage += self.loss.value(y, self.decision_function(shuffled)) - baseline
            out[feature] = damage / repeats

        return np.maximum(out, 0)

    @property
    def n_trees(self) -> int:
        return len(self.trees)

    @property
    def total_leaves(self) -> int:
        return sum(tree.leaves() for tree in self.trees)
