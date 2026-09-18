"""Gradient boosting written from scratch: histogram trees, Newton leaves, honest importance."""

from .boosting import GradientBoosting, History
from .data import Table, read_csv
from .losses import LOSSES, AbsoluteError, LogLoss, Loss, SquaredError
from .metrics import accuracy, log_loss, mae, r2, rmse, roc_auc
from .tree import Node, RegressionTree, TreeParams, bin_features

__all__ = [
    "LOSSES", "AbsoluteError", "GradientBoosting", "History", "LogLoss", "Loss", "Node",
    "RegressionTree", "SquaredError", "Table", "TreeParams", "accuracy", "bin_features",
    "log_loss", "mae", "r2", "read_csv", "rmse", "roc_auc",
]
__version__ = "1.0.0"
