"""Outlier detection: fold isolation, determinism, batch independence, treatments."""

import numpy as np
import pandas as pd

from src.data_loader import load_config, load_data
from src.feature_transformer import (
    IMPUTATION_INDICATOR_SUFFIXES,
    apply_feature_transformer,
    fit_feature_transformer,
)
from src.fold_preprocessing import prepare_numeric_fold
from src.outliers import (
    FLAG_COLUMN,
    SCORE_COLUMN,
    OutlierDetector,
    OutlierTreatment,
    outlier_settings,
)
from src.preprocessing import prepare_rows
from src.validation import temporal_cv

config = load_config()
settings = outlier_settings(config)
assert settings["isolation_forest"]["contamination"] > 0

# --- Synthetic checks on the detector -------------------------------------
rng = np.random.default_rng(0)
normal = pd.DataFrame(rng.normal(size=(500, 3)), columns=["a", "b", "c"])
extreme = pd.DataFrame([[15.0, 15.0, 15.0], [-14.0, 16.0, 13.0]], columns=["a", "b", "c"])

detector = OutlierDetector(["a", "b", "c"], contamination=0.02, random_state=1).fit(normal)
train_rate = detector.is_outlier(normal).mean()
assert abs(train_rate - 0.02) < 0.01, train_rate
assert detector.is_outlier(extreme).all(), "Obvious extremes must be flagged."

# Deterministic with the same seed.
again = OutlierDetector(["a", "b", "c"], contamination=0.02, random_state=1).fit(normal)
assert np.allclose(detector.anomaly_score(extreme), again.anomaly_score(extreme))

# A row's score does not depend on the other rows in its batch.
batch = pd.concat([normal.iloc[:50], extreme], ignore_index=True)
alone = np.array([detector.anomaly_score(batch.iloc[[i]])[0] for i in range(len(batch))])
assert np.allclose(alone, detector.anomaly_score(batch)), "Scores depend on batch."

# Non-finite inputs are rejected rather than silently scored.
broken = normal.iloc[:5].copy()
broken.loc[0, "a"] = np.nan
try:
    detector.anomaly_score(broken)
except ValueError:
    pass
else:
    raise AssertionError("NaN inputs must raise.")

# --- Real folds ------------------------------------------------------------
train, _, _ = load_data(config)
train = prepare_rows(train, config)
validation = config["validation"]
target = config["columns"]["classification_target"]

for year, train_fold, val_fold in temporal_cv(
    train,
    validation["cv_years"][:2],
    horizon_days=validation["target_horizon_days"],
    strategy=validation["strategy"],
    train_years=validation["sliding_window"]["train_years"],
):
    prepared_train, prepared_val, _ = prepare_numeric_fold(train_fold, val_fold, config)

    flag = OutlierTreatment(config, treatment="flag", contamination=0.02).fit(prepared_train)
    columns = flag.columns_
    assert columns and not any(c.endswith(IMPUTATION_INDICATOR_SUFFIXES) for c in columns)
    assert not any(c.startswith("purchase_") for c in columns)
    assert not {"ID", "customer_id", "random_noise", target} & set(columns)

    train_flags = flag.describe(prepared_train)["is_outlier"]
    assert abs(train_flags.mean() - 0.02) < 0.01, (year, train_flags.mean())

    # Validation rows never change what was fitted on the training fold.
    before = flag.describe(prepared_train)["anomaly_score"].to_numpy()
    flag.describe(prepared_val)
    after = flag.describe(prepared_train)["anomaly_score"].to_numpy()
    assert np.array_equal(before, after)

    transformer, cols = fit_feature_transformer(prepared_train, config, task="classification")
    x_train = apply_feature_transformer(prepared_train, transformer, cols, config)
    x_val = apply_feature_transformer(prepared_val, transformer, cols, config)

    with_flag = flag.add_features(x_val, prepared_val)
    assert FLAG_COLUMN in with_flag and with_flag.shape == (x_val.shape[0], x_val.shape[1] + 1)
    assert set(with_flag[FLAG_COLUMN].unique()) <= {0, 1}
    assert with_flag.index.equals(x_val.index)

    score = OutlierTreatment(config, "score", 0.02).fit(prepared_train)
    assert SCORE_COLUMN in score.add_features(x_val, prepared_val)

    # drop_train removes training rows only; validation keeps every row.
    drop = OutlierTreatment(config, "drop_train", 0.02).fit(prepared_train)
    keep = drop.keep_mask(prepared_train)
    assert 0 < (~keep).sum() < 0.05 * len(keep)
    assert len(drop.transform_frame(prepared_val)) == len(prepared_val)
    assert drop.add_features(x_val, prepared_val).shape == x_val.shape

    # winsorize clips validation with TRAINING quantiles.
    win = OutlierTreatment(config, "winsorize").fit(prepared_train)
    clipped = win.transform_frame(prepared_val)
    for column in win.columns_:
        assert clipped[column].max() <= win.upper_[column] + 1e-9
        assert clipped[column].min() >= win.lower_[column] - 1e-9
    assert len(clipped) == len(prepared_val)

    # 'none' is an exact no-op.
    none = OutlierTreatment(config, "none")
    none.columns_ = []
    assert none.keep_mask(prepared_train).all()
    assert none.transform_frame(prepared_val) is prepared_val

    print(f"{year}: columns={len(columns)} flagged_train={int(train_flags.sum())} "
          f"dropped={int((~keep).sum())}")

print("Outlier checks passed.")
