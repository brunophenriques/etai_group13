
import numpy as np
import pandas as pd

from src.data_loader import load_config, load_data
from src.preprocessing import prepare_rows
from src.validation import temporal_cv
from src.fold_preprocessing import prepare_numeric_fold
from src.feature_transformer import (
    fit_feature_transformer,
    apply_feature_transformer,
)


config = load_config()
train, _, _ = load_data(config)
train = prepare_rows(train, config)

years = config["validation"]["cv_years"]
horizon = config["validation"]["target_horizon_days"]

results = []

for year, train_fold, val_fold in temporal_cv(
    train,
    years,
    horizon_days=horizon,
):
    # 1. Numeric and categorical preparation
    prepared_train, prepared_val, imputers = (
        prepare_numeric_fold(train_fold, val_fold, config)
    )

    # 2. Fit encoder/scaler on this fold's training data
    transformer, feature_columns = fit_feature_transformer(
        prepared_train,
        config,
    )

    # 3. Transform train and validation
    X_train = apply_feature_transformer(
        prepared_train,
        transformer,
        feature_columns,
        config,
    )

    X_val = apply_feature_transformer(
        prepared_val,
        transformer,
        feature_columns,
        config,
    )

    # 4. Validate
    assert len(X_train) == len(train_fold)
    assert len(X_val) == len(val_fold)

    assert X_train.columns.equals(X_val.columns)

    assert np.isfinite(X_train.to_numpy()).all()
    assert np.isfinite(X_val.to_numpy()).all()

    assert X_train.index.equals(train_fold.index)
    assert X_val.index.equals(val_fold.index)

    assert (
        pd.to_datetime(train_fold["purchase_date"]).max()
        + pd.Timedelta(days=horizon)
        < pd.Timestamp(f"{year}-01-01")
    )

    results.append({
        "fold": year,
        "train_rows": len(X_train),
        "validation_rows": len(X_val),
        "features": X_train.shape[1],
        "train_missing": int(X_train.isna().sum().sum()),
        "validation_missing": int(X_val.isna().sum().sum()),
    })

    print(f"Fold {year}: PASSED")

summary = pd.DataFrame(results)

assert len(summary) == len(years)

print("\n=== TEMPORAL CV PREPROCESSING SUMMARY ===")
print(summary.to_string(index=False))

print("\nAll temporal preprocessing folds passed.")
