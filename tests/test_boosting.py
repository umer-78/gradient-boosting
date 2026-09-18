import numpy as np
import pytest

from gboost import (
    GradientBoosting,
    RegressionTree,
    TreeParams,
    accuracy,
    bin_features,
    log_loss,
    r2,
    rmse,
    roc_auc,
)

# ------------------------------------------------------------------ trees

def test_a_single_tree_splits_where_the_signal_changes():
    x = np.linspace(0, 10, 200).reshape(-1, 1)
    y = np.where(x[:, 0] < 5, -1.0, 1.0)

    binned, edges = bin_features(x, 32)
    tree = RegressionTree(TreeParams(max_depth=1, min_samples_leaf=5)).fit(
        binned, edges, gradient=-y, hessian=np.ones_like(y))

    assert tree.root.feature == 0
    assert 4.0 < tree.root.threshold < 6.0


def test_a_tree_respects_its_depth_limit():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(400, 3))
    y = x[:, 0] + rng.normal(0, 0.1, 400)
    binned, edges = bin_features(x, 32)

    for depth in (1, 2, 4):
        tree = RegressionTree(TreeParams(max_depth=depth, min_samples_leaf=5)).fit(
            binned, edges, -y, np.ones_like(y))
        assert tree.leaves() <= 2 ** depth


def test_a_tree_refuses_to_make_a_leaf_smaller_than_asked():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(300, 2))
    y = rng.normal(size=300)
    binned, edges = bin_features(x, 32)

    tree = RegressionTree(TreeParams(max_depth=6, min_samples_leaf=40)).fit(
        binned, edges, -y, np.ones_like(y))

    def smallest(node):
        if node.is_leaf:
            return node.samples
        return min(smallest(node.left), smallest(node.right))

    assert smallest(tree.root) >= 40


def test_binning_keeps_the_order_of_the_data():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(500, 1))
    binned, _ = bin_features(x, 16)

    order = np.argsort(x[:, 0])
    assert np.all(np.diff(binned[order, 0].astype(int)) >= 0)


def _binned_error(bins: int) -> float:
    rng = np.random.default_rng(4)
    x = rng.normal(size=(2000, 4))
    y = 2 * x[:, 0] - x[:, 1] + rng.normal(0, 0.3, 2000)
    model = GradientBoosting(n_estimators=120, learning_rate=0.1, max_bins=bins, seed=0)
    model.fit(x[:1500], y[:1500])
    return rmse(y[1500:], model.predict(x[1500:]))


def test_the_default_bin_count_is_close_to_exact_splitting():
    """The cost of the histogram shortcut, measured rather than assumed."""
    assert _binned_error(64) < _binned_error(255) * 1.05


def test_too_few_bins_costs_real_accuracy():
    """So nobody lowers max_bins believing it is free. Eight bins is 60% worse
    than the default on the same data."""
    assert _binned_error(8) > _binned_error(64) * 1.3


def test_tree_parameters_are_validated():
    for bad in ({"max_depth": 0}, {"min_samples_leaf": 0}, {"reg_lambda": -1}, {"max_bins": 1}):
        with pytest.raises(ValueError):
            TreeParams(**bad)


# ------------------------------------------------------------------ regression

def test_boosting_learns_a_linear_relationship(linear):
    x, y = linear
    model = GradientBoosting(n_estimators=300, learning_rate=0.05, max_depth=3).fit(x[:700], y[:700])

    assert r2(y[700:], model.predict(x[700:])) > 0.9


def test_each_round_lowers_the_training_loss(linear):
    x, y = linear
    model = GradientBoosting(n_estimators=60, learning_rate=0.1).fit(x, y)

    losses = model.history.train
    assert losses[-1] < losses[0] / 5
    assert all(b <= a + 1e-9 for a, b in zip(losses, losses[1:], strict=False))


def test_more_trees_help_until_they_stop_helping(linear):
    x, y = linear
    model = GradientBoosting(n_estimators=400, learning_rate=0.05, max_depth=3).fit(x[:700], y[:700])

    early = rmse(y[700:], model.predict(x[700:], n_trees=20))
    later = rmse(y[700:], model.predict(x[700:], n_trees=200))

    assert later < early


def test_a_smaller_learning_rate_with_the_same_budget_needs_more_rounds(linear):
    """The trade-off that makes shrinkage work, stated as an assertion."""
    x, y = linear
    fast = GradientBoosting(n_estimators=25, learning_rate=0.5, max_depth=3).fit(x[:700], y[:700])
    slow = GradientBoosting(n_estimators=25, learning_rate=0.02, max_depth=3).fit(x[:700], y[:700])

    assert rmse(y[700:], fast.predict(x[700:])) < rmse(y[700:], slow.predict(x[700:]))

    patient = GradientBoosting(n_estimators=600, learning_rate=0.02, max_depth=3).fit(x[:700], y[:700])
    assert rmse(y[700:], patient.predict(x[700:])) < rmse(y[700:], fast.predict(x[700:]))


def test_absolute_error_is_less_disturbed_by_a_wild_outlier():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(600, 2))
    y = 2 * x[:, 0] + rng.normal(0, 0.3, 600)
    y[:5] += 500                                     # a handful of absurd targets

    squared = GradientBoosting(loss="squared", n_estimators=200, learning_rate=0.05).fit(x, y)
    absolute = GradientBoosting(loss="absolute", n_estimators=200, learning_rate=0.05).fit(x, y)

    clean = slice(100, None)
    assert (np.abs(absolute.predict(x[clean]) - y[clean]).mean()
            < np.abs(squared.predict(x[clean]) - y[clean]).mean())


# ------------------------------------------------------------------ classification

def test_boosting_separates_two_classes(churn):
    train, test = churn.split(0.7, seed=0)
    model = GradientBoosting(loss="logloss", n_estimators=250, learning_rate=0.05,
                             max_depth=4).fit(train.x, train.y)

    predicted = model.predict(test.x)
    assert roc_auc(test.y, predicted) > 0.78
    assert accuracy(test.y, predicted) > 0.7
    assert log_loss(test.y, predicted) < 0.6


def test_classification_predictions_are_probabilities(churn):
    train, test = churn.split(0.7, seed=0)
    model = GradientBoosting(loss="logloss", n_estimators=60).fit(train.x, train.y)

    predicted = model.predict(test.x)
    assert predicted.min() >= 0 and predicted.max() <= 1


def test_the_raw_score_is_a_log_odds_not_a_probability(churn):
    """Adding trees on the probability scale would let predictions leave [0, 1]."""
    train, _ = churn.split(0.7, seed=0)
    model = GradientBoosting(loss="logloss", n_estimators=100).fit(train.x, train.y)

    raw = model.decision_function(train.x)
    assert raw.min() < 0 < raw.max()
    assert raw.max() > 1 or raw.min() < -1


# ------------------------------------------------------------------ early stopping

def test_early_stopping_needs_data_the_model_did_not_train_on(linear):
    """Stopping on the training loss never triggers, because it only ever falls."""
    x, y = linear
    with pytest.raises(ValueError, match="does not train on"):
        GradientBoosting(early_stopping_rounds=10).fit(x, y)


def test_early_stopping_halts_before_the_full_budget(linear):
    x, y = linear
    model = GradientBoosting(n_estimators=2000, learning_rate=0.1, max_depth=4,
                             early_stopping_rounds=15)
    model.fit(x[:600], y[:600], eval_set=(x[600:], y[600:]))

    assert model.stopped_early
    assert model.n_trees < 2000
    assert model.history.best_round <= model.n_trees


def test_the_validation_curve_turns_while_the_training_curve_does_not(linear):
    """The picture behind early stopping: training loss keeps falling forever."""
    x, y = linear
    model = GradientBoosting(n_estimators=400, learning_rate=0.1, max_depth=6,
                             min_samples_leaf=2, reg_lambda=0.0)
    model.fit(x[:400], y[:400], eval_set=(x[400:], y[400:]))

    train_curve = model.history.train
    validation_curve = model.history.validation

    assert train_curve[-1] < train_curve[len(train_curve) // 2]
    assert validation_curve[-1] > min(validation_curve), "validation should have turned upward"


# ------------------------------------------------------------------ importance

def test_gain_importance_can_rank_pure_noise_above_a_real_predictor(churn):
    """The reason permutation importance is here.

    reference_id is a column of random numbers, distinct in every row. A feature
    with many distinct values gets more chances to produce a lucky split, so gain
    — which only counts how much the training loss fell — rewards it.
    """
    train, _ = churn.split(0.7, seed=0)
    model = GradientBoosting(loss="logloss", n_estimators=250, learning_rate=0.05,
                             max_depth=6, min_samples_leaf=2, reg_lambda=0.0).fit(train.x, train.y)

    gain = model.gain_importance()
    noise = churn.features.index("reference_id")
    contract = churn.features.index("contract")

    assert gain[noise] > gain[contract], "gain ranked the noise column above a real predictor"


def test_permutation_importance_gets_the_order_right(churn):
    """Shuffling a column that carries nothing cannot make a model worse, whatever
    the tree structure happened to do with it — provided it is measured on data
    the model never saw."""
    train, test = churn.split(0.7, seed=0)
    model = GradientBoosting(loss="logloss", n_estimators=250, learning_rate=0.05,
                             max_depth=6, min_samples_leaf=2, reg_lambda=0.0).fit(train.x, train.y)

    permutation = model.permutation_importance(test.x, test.y, repeats=8)
    noise = churn.features.index("reference_id")
    contract = churn.features.index("contract")

    assert permutation[noise] < permutation[contract] / 10
    assert permutation[noise] < 0.01


def test_gain_importance_sums_to_one(housing):
    train, _ = housing.split(0.7, seed=0)
    model = GradientBoosting(n_estimators=80).fit(train.x, train.y)

    assert model.gain_importance().sum() == pytest.approx(1.0)


def test_importance_finds_the_real_drivers_on_well_behaved_settings(housing):
    train, test = housing.split(0.7, seed=0)
    model = GradientBoosting(n_estimators=250, learning_rate=0.05, max_depth=4).fit(train.x, train.y)

    permutation = model.permutation_importance(test.x, test.y, repeats=6)
    ranked = [housing.features[i] for i in np.argsort(-permutation)]

    assert ranked[-1] == "reference_id", "the noise column should matter least"
    assert set(ranked[:3]) <= {"quality", "area_m2", "rooms", "km_to_centre"}


# ------------------------------------------------------------------ plumbing

def test_subsampling_still_trains(linear):
    x, y = linear
    model = GradientBoosting(n_estimators=150, subsample=0.6, seed=1).fit(x[:700], y[:700])
    assert r2(y[700:], model.predict(x[700:])) > 0.85


def test_column_sampling_still_trains(linear):
    x, y = linear
    model = GradientBoosting(n_estimators=200, colsample=0.5, seed=1).fit(x[:700], y[:700])
    assert r2(y[700:], model.predict(x[700:])) > 0.8


def test_the_same_seed_gives_the_same_model(linear):
    x, y = linear
    first = GradientBoosting(n_estimators=60, subsample=0.7, seed=42).fit(x, y).predict(x)
    second = GradientBoosting(n_estimators=60, subsample=0.7, seed=42).fit(x, y).predict(x)

    assert np.array_equal(first, second)


def test_predicting_before_fitting_is_refused():
    with pytest.raises(ValueError, match="not been fitted"):
        GradientBoosting().predict(np.zeros((3, 2)))


def test_the_wrong_number_of_features_is_reported(linear):
    x, y = linear
    model = GradientBoosting(n_estimators=10).fit(x, y)

    with pytest.raises(ValueError, match="expected 4 features"):
        model.predict(np.zeros((5, 2)))


def test_mismatched_rows_and_targets_are_reported():
    with pytest.raises(ValueError, match="rows against"):
        GradientBoosting().fit(np.zeros((10, 2)), np.zeros(9))


def test_model_parameters_are_validated():
    for bad in ({"loss": "hinge"}, {"learning_rate": 0}, {"learning_rate": 2},
                {"n_estimators": 0}, {"subsample": 0}, {"colsample": 1.5}):
        with pytest.raises(ValueError):
            GradientBoosting(**bad)
