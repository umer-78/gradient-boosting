# gboost

[![CI](https://github.com/umer-78/gradient-boosting/actions/workflows/ci.yml/badge.svg)](https://github.com/umer-78/gradient-boosting/actions/workflows/ci.yml)

**Live demo:** https://umer-78.github.io/gradient-boosting/

Gradient boosting written from scratch in Python: histogram-binned regression
trees, Newton leaf values, shrinkage, subsampling, early stopping and two kinds
of feature importance. NumPy holds the arrays; every gradient, split and leaf is
implemented here.

```
$ gboost train data/housing.csv
housing.csv: 3,000 rows, 6 features, target 'price' (regression)
  2,100 train / 900 held out
  300 trees, 3,919 leaves, depth 4, learning rate 0.05

  RMSE         1,641.3428
  MAE          1,291.8381
  R2               0.9594
```

- **38 tests**, Python 3.10–3.12, NumPy the only dependency
- Every analytic gradient is checked against finite differences — a boosting
  implementation with a wrong gradient still trains and still produces plausible
  numbers, it just converges to the wrong place

## Quick start

```bash
git clone https://github.com/umer-78/gradient-boosting.git
cd gradient-boosting
pip install -e ".[dev]"
pytest -q                          # 38 tests

gboost train      data/housing.csv
gboost train      data/churn.csv --early-stopping 20 --rounds 2000
gboost importance data/churn.csv --depth 6 --min-leaf 2 --reg-lambda 0
gboost curve      data/churn.csv
```

## The loop

Start from a constant. Ask the loss which way each prediction should move and how
sharply. Fit one shallow tree to those gradients. Take a *fraction* of the step it
suggests. Repeat.

That fraction is the whole trick, and the split criterion is the other half.
With second-order information available, the best split has a closed form:

```
gain = G_left² / (H_left + λ)  +  G_right² / (H_right + λ)  −  G² / (H + λ)
```

where G and H are the summed gradients and hessians in a node. A plain regression
tree splits on variance reduction instead, which is the first-order approximation
of this, and the leaf value follows the same logic: `−ΣG / (ΣH + λ)`, the
minimiser of the second-order expansion, rather than the mean of the residuals.

## Gain importance is not importance

`reference_id` in `data/churn.csv` is a column of random numbers, a different
value in every row. It has nothing to do with the target. Here is what the model
says about it:

```
$ gboost importance data/churn.csv --depth 6 --min-leaf 2 --reg-lambda 0
feature                 gain   permutation   ranked by gain
tenure_months          0.359         0.540
monthly_charge         0.197         0.066
reference_id           0.176         0.000   <-- this column is pure noise
contract               0.169         0.307
support_tickets        0.070         0.064
autopay                0.029         0.024
```

Gain importance puts pure noise **third of six**, above `contract` — one of the
strongest real predictors in the data. The mechanism is not subtle: a feature
with many distinct values gets more chances to produce a lucky split, and gain
only counts how far the *training* loss fell when it did.

Permutation importance gives it **0.000**, and puts `contract` back where it
belongs. Shuffling a column that carries nothing cannot make a model worse —
provided it is measured on data the model never saw. On the training set, a
memorised noise column looks essential.

Both numbers are printed side by side, and two tests pin the disagreement.

## Early stopping needs data the model never trained on

```
$ gboost train data/churn.csv --early-stopping 20 --rounds 2000
churn.csv: 4,000 rows, 6 features, target 'churned' (classification)
  2,800 train / 1,200 held out
  105 trees, 1,483 leaves, depth 4, learning rate 0.05
  stopped early at round 105 (best was 85)

  LOG_LOSS         0.5007
  ACCURACY         0.7467
  ROC_AUC          0.8343
```

105 trees instead of 2,000, and a *better* held-out log loss than the 300-tree
model at the top of this README (0.5007 against 0.5186). Passing
`early_stopping_rounds` without an `eval_set` raises rather than quietly doing
nothing:

```python
ValueError: early stopping needs an eval_set the model does not train on —
stopping on the training loss never triggers, because it only ever falls
```

A test grows a deliberately over-capable model and asserts the two curves part
company: the training loss keeps falling while the validation loss turns upward.

## Learning rate against rounds

```
$ gboost curve data/churn.csv
churn.csv: the same budget spent in different sized steps

 learning rate   rounds   held-out score
           0.5       30           0.5352
           0.5      100           0.6022
           0.5      300           0.7607
           0.3       30           0.5197
           0.3      100           0.5511
           0.3      300           0.6386
           0.1       30           0.5046
           0.1      100           0.5114
           0.1      300           0.5446
          0.05       30           0.5262
          0.05      100           0.5007
          0.05      300           0.5186
          0.02       30           0.5800
          0.02      100           0.5153
          0.02      300           0.5025

lower is better. Large steps reach their best early and then get worse quickly;
small steps need more rounds to arrive. The best cell is at neither extreme, and
every row worsens past its own optimum — which is the whole case for early stopping.
```

## Binning, and what it costs

Splits are searched over quantile bins rather than raw values, so a node
accumulates gradient sums per bin instead of re-sorting every feature. The cost
is that a split can only land on a bin edge, and that cost is measurable:

| bins | 8 | 16 | 32 | 64 | 128 | 255 |
| --- | --- | --- | --- | --- | --- | --- |
| held-out RMSE | .581 | .447 | .370 | **.358** | .346 | .348 |

64 — the default — sits within about 3% of exact splitting. Below 32 the
shortcut starts costing accuracy that matters, so a test pins both ends: the
default is near the ceiling, and eight bins is measurably worse, so nobody lowers
it believing it is free.

## Losses

| Loss | Gradient | Hessian | For |
| --- | --- | --- | --- |
| `SquaredError` | `raw − y` | `1` | regression; the constant hessian is why this reduces to fitting the residual |
| `AbsoluteError` | `sign(raw − y)` | `1` (a stand-in) | regression with outliers; a point twice as wrong pulls no harder |
| `LogLoss` | `σ(raw) − y` | `σ(1 − σ)` | binary classification, on the log-odds scale |

The model's output for classification is a log-odds, never a probability. Adding
trees on the probability scale would let predictions leave [0, 1]; the sigmoid at
the end is what keeps them inside it.

`AbsoluteError` is honest about its own shortcut: the second derivative of `|x|`
is zero wherever it exists, so a constant stand-in is used. The direction is
right, the step size is a guess, and a lower learning rate is the usual
compensation. That is written in the docstring rather than left for someone to
discover.

## As a library

```python
import numpy as np
from gboost import GradientBoosting, read_csv, roc_auc

table = read_csv("data/churn.csv")
train, test = table.split(0.7)

model = GradientBoosting(
    loss="logloss", n_estimators=2000, learning_rate=0.05,
    max_depth=4, subsample=0.8, early_stopping_rounds=20,
).fit(train.x, train.y, eval_set=(test.x, test.y))

model.n_trees                                       # 105 — it stopped on its own
roc_auc(test.y, model.predict(test.x))              # 0.8343
model.permutation_importance(test.x, test.y)        # the one to trust
```

## Layout

```
src/gboost/tree.py       histogram splitting, gain, Newton leaves
src/gboost/losses.py     value, gradient and hessian for each loss
src/gboost/boosting.py   the boosting loop, sampling, early stopping, importance
src/gboost/metrics.py    RMSE, MAE, R², log loss, accuracy, rank-based AUC
src/gboost/data.py       CSV loading and splitting
src/gboost/cli.py        the `gboost` command
data/                    two generated datasets, each with a planted noise column
tools/make_data.py       the generator; CI fails if the output drifts
tests/                   38 tests
```

## Not included

Multiclass, categorical features, missing-value handling, monotone constraints,
SHAP, GPU anything. What is here is the arithmetic those all sit on top of,
written so it can be read in an afternoon — and checked against finite
differences so you can believe it.

## Licence

MIT — see [LICENSE](LICENSE).
