
import numpy as np

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

year, train_fold, val_fold = next(
    temporal_cv(
        train,
        config["validation"]["cv_years"],
        horizon_days=config["validation"]["target_horizon_days"],
    )
)

prepared_train, prepared_val, imputers = (
    prepare_numeric_fold(train_fold, val_fold, config)
)

transformer, feature_columns = fit_feature_transformer(
    prepared_train,
    config,
)

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

print(f"\n=== FEATURE TRANSFORMATION FOLD {year} ===")
print("Training shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Model features:", len(X_train.columns))

assert len(X_train) == len(train_fold)
assert len(X_val) == len(val_fold)

assert X_train.columns.equals(X_val.columns)

assert not X_train.isna().any().any()
assert not X_val.isna().any().any()

assert np.isfinite(X_train.to_numpy()).all()
assert np.isfinite(X_val.to_numpy()).all()

print("\nFeature transformation passed.")
