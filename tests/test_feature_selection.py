
"""Unit tests for consensus Feature Selection — ETAI Group 13."""

import numpy as np
import pandas as pd
import pytest

import src.feature_selection as fs


@pytest.fixture
def config():
    """Minimal configuration for isolated feature-selection tests."""
    return {
        "random_state": 42,
        "validation": {
            "target_horizon_days": 90,
        },
        "preprocessing": {
            "feature_selection": {
                "enabled": True,
                "consensus": {
                    "enabled": True,
                    "apply_threshold": 2,
                    "rejection_threshold": 2,
                },
                "methods": {
                    "mrmr": {"enabled": True},
                    "rfecv": {"enabled": True},
                    "boruta": {"enabled": True},
                    "logistic_l1": {"enabled": True},
                },
            }
        },
    }


@pytest.fixture
def data():
    """Small synthetic train/validation dataset."""
    X_train = pd.DataFrame({
        "feature_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "feature_b": [3.0, 2.0, 4.0, 1.0, 6.0, 5.0],
        "feature_c": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    })

    X_val = pd.DataFrame({
        "feature_a": [7.0, 8.0],
        "feature_b": [4.0, 5.0],
        "feature_c": [1.0, 0.0],
    })

    return X_train, X_val


def test_consensus_threshold_two(config, data):
    """Remove features rejected by at least two methods."""
    X_train, X_val = data

    selection = pd.DataFrame({
        "feature": X_train.columns,
        "rejected_votes": [0, 1, 2],
    })

    train_selected, val_selected = fs.apply_consensus_selection(
        X_train, X_val, selection, config
    )

    assert train_selected.columns.tolist() == [
        "feature_a", "feature_b"
    ]
    assert val_selected.columns.tolist() == [
        "feature_a", "feature_b"
    ]


def test_disabled_selection_preserves_features(config, data):
    """Disabled selection must leave all features unchanged."""
    config["preprocessing"]["feature_selection"]["enabled"] = False

    X_train, X_val = data

    train_result, val_result = fs.apply_consensus_selection(
        X_train, X_val, None, config
    )

    pd.testing.assert_frame_equal(train_result, X_train)
    pd.testing.assert_frame_equal(val_result, X_val)


def test_consensus_rejects_mismatched_columns(config, data):
    """Train and validation must have matching ordered columns."""
    X_train, X_val = data

    X_val = X_val[[
        "feature_b", "feature_a", "feature_c"
    ]]

    selection = pd.DataFrame({
        "feature": X_train.columns,
        "rejected_votes": [0, 1, 2],
    })

    with pytest.raises(ValueError):
        fs.apply_consensus_selection(
            X_train, X_val, selection, config
        )


def test_consensus_rejects_missing_decisions(config, data):
    """Every model feature must have a selection decision."""
    X_train, X_val = data

    selection = pd.DataFrame({
        "feature": ["feature_a", "feature_b"],
        "rejected_votes": [0, 2],
    })

    with pytest.raises(ValueError):
        fs.apply_consensus_selection(
            X_train, X_val, selection, config
        )


def test_boruta_tentative_is_not_rejected(monkeypatch):
    """Boruta's tentative features must remain inconclusive."""

    class FakeBoruta:
        def __init__(self, **kwargs):
            pass

        def fit(self, X, y):
            self.support_ = np.array([True, False, False])
            self.support_weak_ = np.array([False, True, False])
            return self

    monkeypatch.setattr(fs, "BorutaPy", FakeBoruta)

    X = pd.DataFrame({
        "a": [1, 2, 3, 4],
        "b": [4, 3, 2, 1],
        "c": [1, 1, 2, 2],
    })
    y = pd.Series([0, 1, 0, 1])

    settings = {
        "n_estimators": "auto",
        "max_iter": 20,
        "max_depth": 5,
        "perc": 100,
    }

    result = fs.select_boruta(X, y, settings, 42)

    assert result.loc["a"] == True
    assert pd.isna(result.loc["b"])
    assert result.loc["c"] == False


def test_vote_counts_and_tentative_decisions(monkeypatch, config, data):
    """Each feature gets exactly one outcome per enabled method."""
    X_train, _ = data
    y = pd.Series([0, 1, 0, 1, 0, 1])

    dates = pd.date_range(
        "2018-01-01",
        periods=len(X_train),
        freq="D",
    )

    # Simulate methods without fitting expensive estimators.
    monkeypatch.setattr(
        fs,
        "select_mrmr",
        lambda X, y, settings, random_state:
            pd.Series([True, False, False], index=X.columns),
    )

    monkeypatch.setattr(
        fs,
        "select_rfecv",
        lambda X, y, dates, raw_train_fold, config, settings, random_state:
            pd.Series([True, True, False], index=X.columns),
    )

    monkeypatch.setattr(
        fs,
        "select_boruta",
        lambda X, y, settings, random_state:
            pd.Series(
                pd.array([True, pd.NA, False], dtype="boolean"),
                index=X.columns,
            ),
    )

    monkeypatch.setattr(
        fs,
        "select_logistic_l1",
        lambda X, y, settings, random_state:
            pd.Series([True, False, True], index=X.columns),
    )

    result = fs.run_feature_selection(
        X_train, y, dates, config,
        raw_train_fold=X_train,
    ).set_index("feature")

    total_votes = (
        result["selected_votes"]
        + result["rejected_votes"]
        + result["tentative_votes"]
    )

    assert (total_votes == 4).all()

    assert result.loc["feature_a", "selected_votes"] == 4
    assert result.loc["feature_a", "rejected_votes"] == 0

    assert result.loc["feature_b", "selected_votes"] == 1
    assert result.loc["feature_b", "rejected_votes"] == 2
    assert result.loc["feature_b", "tentative_votes"] == 1

    assert result.loc["feature_c", "selected_votes"] == 1
    assert result.loc["feature_c", "rejected_votes"] == 3

    assert not result.loc["feature_a", "remove_candidate"]
    assert result.loc["feature_b", "remove_candidate"]
    assert result.loc["feature_c", "remove_candidate"]


def test_consensus_does_not_fit_selectors(config, data, monkeypatch):
    """Applying stored votes must never fit selection models."""
    X_train, X_val = data

    def unexpected_fit(*args, **kwargs):
        raise AssertionError(
            "Feature selectors must not run during application."
        )

    monkeypatch.setattr(fs, "run_feature_selection", unexpected_fit)

    selection = pd.DataFrame({
        "feature": X_train.columns,
        "rejected_votes": [0, 2, 3],
    })

    train_selected, val_selected = fs.apply_consensus_selection(
        X_train, X_val, selection, config
    )

    assert train_selected.columns.tolist() == ["feature_a"]
    assert val_selected.columns.tolist() == ["feature_a"]
    assert val_selected.iloc[0, 0] == X_val["feature_a"].iloc[0]


def test_temporal_inner_splits_respect_embargo(config):
    """Training labels must be available before validation starts."""
    dates = pd.date_range(
        "2016-01-01",
        periods=1200,
        freq="D",
    )

    settings = {
        "inner_splits": 3,
        "min_train_rows": 100,
        "min_validation_rows": 100,
    }

    splits = fs.temporal_inner_splits(
        dates, config, settings
    )

    assert len(splits) >= 2

    horizon = pd.Timedelta(
        days=config["validation"]["target_horizon_days"]
    )

    for train_idx, val_idx in splits:
        train_dates = dates[train_idx]
        val_dates = dates[val_idx]

        assert len(np.intersect1d(train_idx, val_idx)) == 0
        assert train_dates.max() + horizon < val_dates.min()
