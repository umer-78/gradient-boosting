"""The regression tree that every boosting round fits.

This is not a classification tree with a different label. Boosting fits a tree to
*gradients*, so each leaf holds a step to take, and the split that matters is the
one that most reduces the loss — which, with second-order information available,
has a closed form:

    gain = G_left^2/(H_left + l) + G_right^2/(H_right + l) - G^2/(H + l)

where G and H are the summed gradients and hessians in a node and l is the L2
penalty. Splitting on variance reduction instead, as a plain regression tree
does, is the first-order approximation of this and is measurably worse on
anything but squared loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Node:
    value: float = 0.0
    feature: int = -1
    threshold: float = 0.0
    left: Node | None = None
    right: Node | None = None
    samples: int = 0
    gain: float = 0.0

    @property
    def is_leaf(self) -> bool:
        return self.left is None


@dataclass
class TreeParams:
    max_depth: int = 3
    min_samples_leaf: int = 20
    min_child_weight: float = 1e-3
    reg_lambda: float = 1.0
    min_gain: float = 0.0
    max_bins: int = 64

    def __post_init__(self) -> None:
        if self.max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1")
        if self.reg_lambda < 0:
            raise ValueError("reg_lambda cannot be negative")
        if self.max_bins < 2:
            raise ValueError("max_bins must be at least 2")


def bin_features(x: np.ndarray, max_bins: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """Pre-bin every column by quantile, once, before any tree is grown.

    Exact split finding re-sorts each feature at every node. Binning sorts once
    up front, after which a node only ever accumulates gradient sums per bin —
    the histogram approach LightGBM and XGBoost use, where it pays off at scale.

    The cost is that a split can only land on a bin edge, and that cost is real
    enough to measure. Held-out RMSE on the four-feature benchmark in the tests,
    120 rounds:

        bins    8     16     32     64    128    255
        RMSE  .581   .447   .370   .358   .346   .348

    So 64 — the default — sits within about 3% of exact splitting, 32 is close
    behind, and below that the shortcut starts costing accuracy that matters. A
    test pins both ends: that the default is near the ceiling, and that eight
    bins is measurably worse, so nobody lowers it thinking it is free.
    """
    n_features = x.shape[1]
    binned = np.empty(x.shape, dtype=np.uint8 if max_bins <= 256 else np.uint16)
    edges: list[np.ndarray] = []

    for column in range(n_features):
        values = x[:, column]
        quantiles = np.unique(np.quantile(values, np.linspace(0, 1, max_bins + 1)[1:-1]))
        edges.append(quantiles)
        binned[:, column] = np.searchsorted(quantiles, values, side="left")

    return binned, edges


class RegressionTree:
    """Fitted on gradients and hessians, not on labels."""

    def __init__(self, params: TreeParams | None = None):
        self.params = params or TreeParams()
        self.root: Node | None = None
        self.n_features = 0

    def fit(self, binned: np.ndarray, edges: list[np.ndarray],
            gradient: np.ndarray, hessian: np.ndarray) -> RegressionTree:
        self.n_features = binned.shape[1]
        indices = np.arange(len(gradient))
        self.root = self._grow(binned, edges, gradient, hessian, indices, depth=0)
        return self

    def _leaf_value(self, gradient: np.ndarray, hessian: np.ndarray, indices: np.ndarray) -> float:
        # Newton step: the minimiser of the second-order expansion of the loss.
        return float(-gradient[indices].sum() / (hessian[indices].sum() + self.params.reg_lambda))

    def _grow(self, binned: np.ndarray, edges: list[np.ndarray], gradient: np.ndarray,
              hessian: np.ndarray, indices: np.ndarray, depth: int) -> Node:
        node = Node(value=self._leaf_value(gradient, hessian, indices), samples=len(indices))

        if depth >= self.params.max_depth or len(indices) < 2 * self.params.min_samples_leaf:
            return node

        best = self._best_split(binned, gradient, hessian, indices)
        if best is None:
            return node

        feature, bin_index, gain = best
        if gain <= self.params.min_gain:
            return node

        left_mask = binned[indices, feature] <= bin_index
        left, right = indices[left_mask], indices[~left_mask]

        node.feature = feature
        node.gain = gain
        # Bin index -> a real threshold, so the tree can score unbinned data.
        node.threshold = float(edges[feature][bin_index]) if bin_index < len(edges[feature]) else np.inf
        node.left = self._grow(binned, edges, gradient, hessian, left, depth + 1)
        node.right = self._grow(binned, edges, gradient, hessian, right, depth + 1)
        return node

    def _best_split(self, binned: np.ndarray, gradient: np.ndarray, hessian: np.ndarray,
                    indices: np.ndarray) -> tuple[int, int, float] | None:
        g, h = gradient[indices], hessian[indices]
        total_g, total_h = g.sum(), h.sum()
        denominator = total_h + self.params.reg_lambda
        if denominator <= 0:
            return None
        parent = total_g ** 2 / denominator

        best: tuple[int, int, float] | None = None
        n_bins = int(binned.max()) + 1

        for feature in range(binned.shape[1]):
            codes = binned[indices, feature]
            # One pass per feature: gradient and hessian mass per bin, then a
            # running sum over bins gives every candidate split at once.
            g_hist = np.bincount(codes, weights=g, minlength=n_bins)
            h_hist = np.bincount(codes, weights=h, minlength=n_bins)
            counts = np.bincount(codes, minlength=n_bins)

            g_left = np.cumsum(g_hist)[:-1]
            h_left = np.cumsum(h_hist)[:-1]
            n_left = np.cumsum(counts)[:-1]
            g_right = total_g - g_left
            h_right = total_h - h_left
            n_right = len(indices) - n_left

            allowed = (
                (n_left >= self.params.min_samples_leaf)
                & (n_right >= self.params.min_samples_leaf)
                & (h_left >= self.params.min_child_weight)
                & (h_right >= self.params.min_child_weight)
            )
            if not allowed.any():
                continue

            # np.where evaluates both branches, so the denominators are made safe
            # before the division rather than relying on the mask to skip them.
            # With reg_lambda=0 a child whose hessian sums to zero would otherwise
            # raise a divide-by-zero warning on every split search.
            safe_left = np.where(allowed, h_left + self.params.reg_lambda, 1.0)
            safe_right = np.where(allowed, h_right + self.params.reg_lambda, 1.0)
            gains = np.where(
                allowed,
                g_left ** 2 / safe_left + g_right ** 2 / safe_right - parent,
                -np.inf,
            )
            index = int(np.argmax(gains))
            if best is None or gains[index] > best[2]:
                best = (feature, index, float(gains[index]))

        return best if best and np.isfinite(best[2]) else None

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise ValueError("this tree has not been fitted")
        out = np.empty(len(x), dtype=np.float64)
        for i, row in enumerate(x):
            node = self.root
            while not node.is_leaf:
                node = node.left if row[node.feature] <= node.threshold else node.right
            out[i] = node.value
        return out

    def gain_by_feature(self) -> np.ndarray:
        """Total gain attributed to each feature by the splits in this tree."""
        totals = np.zeros(self.n_features)

        def walk(node: Node | None) -> None:
            if node is None or node.is_leaf:
                return
            totals[node.feature] += node.gain
            walk(node.left)
            walk(node.right)

        walk(self.root)
        return totals

    def leaves(self) -> int:
        def count(node: Node | None) -> int:
            if node is None:
                return 0
            return 1 if node.is_leaf else count(node.left) + count(node.right)

        return count(self.root)
