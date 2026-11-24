"""Outlier experiment: config consistency, column selection and training versions."""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.data_loader import load_config, load_data
from src.feature_transformer import prepare_feature_frame
from src.fold_preprocessing import prepare_numeric_fold
from src.outliers import CALENDAR_COLUMNS, build_versions, numeric_columns
from src.preprocessing import prepare_rows
from src.validation import temporal_cv

config = load_config()

# --- Config consistency ------------------------------------------------------
settings = config["preprocessing"]["outliers"]
allowed = {"none", "drop_train", "flag", "score"}

assert "none" in settings["treatments"], "'none' is the reference and must be present."
assert set(settings["treatments"]) <= allowed, settings["treatments"]
assert 0 < settings["isolation_forest"]["contamination"] < 0.5
assert settings["isolation_forest"]["n_estimators"] >= 1

enabled_classifiers = {
    n for n, s in config["modeling"]["models"].items() if s.get("enabled")
}
enabled_regressors = {
    n for n, s in config["modeling"]["regression_models"].items() if s.get("enabled")
}
assert set(settings["models"]["classification"]) <= enabled_classifiers
assert set(settings["models"]["regression"]) <= enabled_regressors

# --- numeric_columns on synthetic data ---------------------------------------
frame = pd.DataFrame({
    "total": np.arange(50, dtype=float),
    "purchase_count_30d": np.arange(50) % 7,        # rolling count: must be kept
    "purchase_year": 2020,                           # calendar
    "purchase_month": np.arange(50) % 12,            # calendar
    "flag_like": np.arange(50) % 2,                  # 0/1
    "constant": 1.0,
    "total_was_missing": np.arange(50) % 3,          # imputation indicator
    "name": ["x"] * 50,                              # not numeric
})
selected = numeric_columns(frame)
assert selected == ["total", "purchase_count_30d"], selected
assert set(CALENDAR_COLUMNS) == {"purchase_year", "purchase_month", "purchase_day_of_week"}

# --- build_versions on synthetic data ----------------------------------------
rng = np.random.default_rng(0)


def make_frames(n, extreme_rows=()):
    values = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    for row in extreme_rows:
        values.loc[row] = [25.0, -25.0, 25.0]
    return values


feat_train = make_frames(500, extreme_rows=[3, 40])
feat_val = make_frames(120, extreme_rows=[5])
X_train, X_val = feat_train.copy(), feat_val.copy()
y_train = pd.Series(rng.integers(0, 2, len(X_train)))


def forest():
    return IsolationForest(n_estimators=100, contamination=0.02, random_state=1)


versions, n_flagged = build_versions(forest(), feat_train, feat_val, X_train, y_train, X_val)
assert set(versions) == allowed, set(versions)

none_train, none_y, none_val = versions["none"]
assert len(none_train) == 500 and none_val.equals(X_val)

# drop_train: removes training rows only; validation keeps every row.
drop_train, drop_y, drop_val = versions["drop_train"]
assert len(drop_train) == 500 - n_flagged and len(drop_y) == len(drop_train)
assert 0 < n_flagged <= 15, n_flagged                    # about 2% of 500
assert 3 not in drop_train.index and 40 not in drop_train.index, "Extremes must be dropped."
assert len(drop_val) == len(X_val)

# flag: one 0/1 column, in training and validation.
flag_train, _, flag_val = versions["flag"]
assert "outlier_flag" in flag_train and "outlier_flag" in flag_val
assert set(flag_train["outlier_flag"].unique()) <= {0, 1}
assert flag_train.loc[[3, 40], "outlier_flag"].eq(1).all()
assert flag_val.loc[5, "outlier_flag"] == 1
assert flag_train["outlier_flag"].sum() == n_flagged

# score: one continuous column; the extreme row has the highest score.
score_train, _, score_val = versions["score"]
assert "outlier_score" in score_train and "outlier_score" in score_val
assert score_val["outlier_score"].idxmax() == 5
assert score_train.shape[1] == X_train.shape[1] + 1

# The original matrices are never modified in place.
assert list(X_train.columns) == ["a", "b", "c"] and list(X_val.columns) == ["a", "b", "c"]

# The forest learns from the training fold only: changing the validation rows
# must not change anything computed for the training rows.
other_val = make_frames(120)
versions_b, n_flagged_b = build_versions(
    forest(), feat_train, other_val, X_train, y_train, other_val.copy()
)
assert n_flagged_b == n_flagged
assert np.allclose(versions_b["score"][0]["outlier_score"], score_train["outlier_score"])

# Deterministic with the same seed.
versions_c, _ = build_versions(forest(), feat_train, feat_val, X_train, y_train, X_val)
assert np.allclose(versions_c["score"][2]["outlier_score"], score_val["outlier_score"])

# --- One real fold ------------------------------------------------------------
data = prepare_rows(load_data(config)[0], config)
validation = config["validation"]

for year, train_fold, val_fold in temporal_cv(
    data,
    validation["cv_years"][:1],
    horizon_days=validation["target_horizon_days"],
    strategy=validation["strategy"],
    train_years=validation.get("sliding_window", {}).get("train_years"),
):
    prepared_train, _, _ = prepare_numeric_fold(train_fold, val_fold, config)
    columns = numeric_columns(prepare_feature_frame(prepared_train, config))

    assert columns, "No numeric columns for the Isolation Forest."
    assert not set(CALENDAR_COLUMNS) & set(columns)
    assert not {"ID", "customer_id", "random_noise"} & set(columns)
    assert not any(c.endswith("_was_missing") for c in columns)
    assert {"purchase_count_30d", "purchase_count_365d"} <= set(columns), (
        "Rolling purchase counts must be used."
    )
    print(f"{year}: {len(columns)} columns used by the Isolation Forest")

print("Outlier checks passed.")
