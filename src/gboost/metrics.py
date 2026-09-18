"""Scoring, for both kinds of task."""

from __future__ import annotations

import numpy as np


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def r2(y: np.ndarray, p: np.ndarray) -> float:
    """Share of the variance explained. Zero means no better than the mean."""
    y = np.asarray(y, dtype=np.float64)
    residual = np.sum((y - np.asarray(p)) ** 2)
    total = np.sum((y - y.mean()) ** 2)
    return float(1 - residual / total) if total else 0.0


def accuracy(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> float:
    return float(np.mean((np.asarray(p) >= threshold).astype(int) == np.asarray(y).astype(int)))


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    y = np.asarray(y, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def roc_auc(y: np.ndarray, scores: np.ndarray) -> float:
    """Computed from ranks, which handles ties correctly and needs no threshold.

    The rank formulation is also exactly the probability that a random positive
    outranks a random negative, which is what AUC means and what a threshold
    sweep only approximates.
    """
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    positives, negatives = int(y.sum()), int((1 - y).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("AUC needs at least one example of each class")

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)

    # Average the ranks within each tied group, or tied scores would be ordered
    # by index and the result would depend on the row order of the input.
    sorted_scores = scores[order]
    start = 0
    for i in range(1, len(sorted_scores) + 1):
        if i == len(sorted_scores) or sorted_scores[i] != sorted_scores[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i

    return float((ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))
