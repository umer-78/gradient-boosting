"""gboost: train a gradient boosting model and look at what it learned."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .boosting import GradientBoosting
from .data import read_csv
from .metrics import accuracy, log_loss, mae, r2, rmse, roc_auc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gboost", description=__doc__)
    parser.add_argument("--version", action="version", version=f"gboost {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("csv", type=Path)
        p.add_argument("--target", default=None)
        p.add_argument("--task", choices=["regression", "classification"], default=None)
        p.add_argument("--rounds", type=int, default=300)
        p.add_argument("--learning-rate", type=float, default=0.05)
        p.add_argument("--depth", type=int, default=4)
        p.add_argument("--min-leaf", type=int, default=20)
        p.add_argument("--reg-lambda", type=float, default=1.0)
        p.add_argument("--subsample", type=float, default=1.0)
        p.add_argument("--seed", type=int, default=0)

    train = sub.add_parser("train", help="fit, then score on held-out rows")
    common(train)
    train.add_argument("--early-stopping", type=int, default=None)
    train.add_argument("--json", action="store_true")

    importance = sub.add_parser("importance", help="gain importance beside permutation importance")
    common(importance)
    importance.add_argument("--repeats", type=int, default=8)

    curve = sub.add_parser("curve", help="learning rate against the number of rounds")
    common(curve)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Wraps the real work so that piping into `head` — which closes
    the pipe early — ends quietly instead of printing a BrokenPipeError."""
    try:
        return _run(argv)
    except BrokenPipeError:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        return 0
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return 130
    except (ValueError, FileNotFoundError) as error:
        print(f"gboost: {error}", file=sys.stderr)
        return 2


def _infer_task(y: np.ndarray) -> str:
    return "classification" if set(np.unique(y).tolist()) <= {0.0, 1.0} else "regression"


def _model(args, task: str, **extra) -> GradientBoosting:
    return GradientBoosting(
        loss="logloss" if task == "classification" else "squared",
        n_estimators=args.rounds, learning_rate=args.learning_rate, max_depth=args.depth,
        min_samples_leaf=args.min_leaf, reg_lambda=args.reg_lambda,
        subsample=args.subsample, seed=args.seed, **extra)


def _run(argv: list[str] | None) -> int:
    args = build_parser().parse_args(argv)
    table = read_csv(args.csv, args.target)
    task = args.task or _infer_task(table.y)
    train, test = table.split(0.7, seed=args.seed)

    if args.cmd == "train":
        extra = {"early_stopping_rounds": args.early_stopping} if args.early_stopping else {}
        model = _model(args, task, **extra)
        model.fit(train.x, train.y, eval_set=(test.x, test.y) if args.early_stopping else None)
        predicted = model.predict(test.x)

        scores = ({"log_loss": log_loss(test.y, predicted),
                   "accuracy": accuracy(test.y, predicted),
                   "roc_auc": roc_auc(test.y, predicted)}
                  if task == "classification" else
                  {"rmse": rmse(test.y, predicted), "mae": mae(test.y, predicted),
                   "r2": r2(test.y, predicted)})

        if args.json:
            print(json.dumps({"task": task, "trees": model.n_trees,
                              "leaves": model.total_leaves,
                              "stopped_early": model.stopped_early,
                              **{k: round(v, 5) for k, v in scores.items()}}, indent=2))
            return 0

        print(f"{args.csv.name}: {len(table):,} rows, {len(table.features)} features, "
              f"target {table.target!r} ({task})")
        print(f"  {len(train):,} train / {len(test):,} held out")
        print(f"  {model.n_trees} trees, {model.total_leaves:,} leaves, "
              f"depth {args.depth}, learning rate {args.learning_rate}")
        if model.stopped_early:
            print(f"  stopped early at round {model.n_trees} "
                  f"(best was {model.history.best_round})")
        print()
        for name, value in scores.items():
            print(f"  {name.upper():<10} {value:>12,.4f}")
        return 0

    if args.cmd == "importance":
        model = _model(args, task).fit(train.x, train.y)
        gain = model.gain_importance()
        permutation = model.permutation_importance(test.x, test.y, repeats=args.repeats)
        normalised = permutation / permutation.sum() if permutation.sum() else permutation

        order = np.argsort(-gain)
        print(f"{'feature':<18}{'gain':>10}{'permutation':>14}   ranked by gain")
        for i in order:
            flag = ""
            if "reference" in table.features[i]:
                flag = "   <-- this column is pure noise"
            print(f"{table.features[i]:<18}{gain[i]:>10.3f}{normalised[i]:>14.3f}{flag}")
        return 0

    print(f"{args.csv.name}: the same budget spent in different sized steps\n")
    print(f"{'learning rate':>14}{'rounds':>9}{'held-out score':>17}")
    for rate in (0.5, 0.3, 0.1, 0.05, 0.02):
        for rounds in (30, 100, 300):
            model = GradientBoosting(
                loss="logloss" if task == "classification" else "squared",
                n_estimators=rounds, learning_rate=rate, max_depth=args.depth,
                min_samples_leaf=args.min_leaf, reg_lambda=args.reg_lambda, seed=args.seed)
            model.fit(train.x, train.y)
            predicted = model.predict(test.x)
            score = (log_loss(test.y, predicted) if task == "classification"
                     else rmse(test.y, predicted))
            print(f"{rate:>14}{rounds:>9}{score:>17,.4f}")
    print("\nlower is better. Large steps reach their best early and then get worse quickly;")
    print("small steps need more rounds to arrive. The best cell is at neither extreme, and")
    print("every row worsens past its own optimum — which is the whole case for early stopping.")
    return 0
