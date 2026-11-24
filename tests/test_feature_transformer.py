from copy import deepcopy
from io import BytesIO

import joblib
import numpy as np

from src.data_loader import load_config, load_data
from src.feature_transformer import apply_feature_transformer, fit_feature_transformer
from src.fold_preprocessing import prepare_numeric_fold
from src.preprocessing import prepare_rows
from src.validation import temporal_cv

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


def check_rare_categories():
    cfg = deepcopy(config)
    encoding = cfg["preprocessing"]["encoding"]
    encoding["rare_categories"]["min_frequency_by_column"] = {
        "payment_type": 3,
        "salesperson": 2,
        "commercial_zone": 2,
    }
    training = prepared_train.iloc[:6].copy()
    training["payment_type"] = ["A", "A", "A", "B", "B", "Unknown"]
    training["salesperson"] = ["A", "A", "A", "B", "B", "Rare"]
    training["commercial_zone"] = ["A", "A", "A", "B", "B", "B"]
    training["transport_method"] = ["A", "A", "A", "B", "B", "Rare"]
    validation = training.iloc[:3].copy()
    validation["payment_type"] = ["B", "New", "Unknown"]
    validation["salesperson"] = ["B", "Rare", "New"]
    validation["commercial_zone"] = ["New", "A", "B"]

    encoder, columns = fit_feature_transformer(training, cfg)
    encoded = apply_feature_transformer(validation, encoder, columns, cfg)
    assert "rare_categorical__payment_type_A" in encoded
    assert "rare_categorical__payment_type_B" not in encoded
    assert encoded["rare_categorical__payment_type_Other"].tolist() == [1, 1, 0]
    assert encoded["rare_categorical__payment_type_Unknown"].tolist() == [0, 0, 1]
    assert encoded["rare_categorical__salesperson_B"].tolist() == [1, 0, 0]
    assert encoded["rare_categorical__salesperson_Other"].tolist() == [0, 1, 1]
    zone = encoded.filter(like="rare_categorical__commercial_zone_")
    assert zone.columns.tolist() == [
        "rare_categorical__commercial_zone_A", "rare_categorical__commercial_zone_B",
    ]
    assert zone.sum(axis=1).tolist() == [0, 1, 1]
    assert "categorical__transport_method_Rare" in encoded

    encoding["rare_categories"]["enabled"] = False
    encoder, columns = fit_feature_transformer(training, cfg)
    encoded = apply_feature_transformer(validation, encoder, columns, cfg)
    assert not any(name.startswith("rare_categorical__") for name in encoded)
    assert encoded["categorical__payment_type_B"].tolist() == [1, 0, 0]
    assert "categorical__transport_method_Rare" in encoded

    for invalid in (None, {"payment_type": 0}, {"payment_type": True}, {"payment_type": 2.5},
                    {"spend_90d": 2}):
        encoding["rare_categories"]["min_frequency_by_column"] = invalid
        try:
            fit_feature_transformer(training, cfg)
        except ValueError as error:
            assert "threshold" in str(error)
        else:
            raise AssertionError(f"Accepted invalid rare-category thresholds: {invalid}")

    encoding["categorical_columns"].remove("transport_method")
    encoding["rare_categories"] = {
        "enabled": True, "columns": ["transport_method"], "min_frequency": 2,
        "handle_unknown": "infrequent_if_exist",
    }
    encoder, columns = fit_feature_transformer(training, cfg)
    encoded = apply_feature_transformer(validation, encoder, columns, cfg)
    assert "rare_categorical__transport_method_infrequent_sklearn" in encoded
    assert "categorical__payment_type_B" in encoded


check_rare_categories()
print("Rare-category thresholds and legacy configuration passed.")


def check_frequency_encoding():
    cfg = deepcopy(config)
    encoding = cfg["preprocessing"]["encoding"]
    encoding["rare_categories"]["min_frequency_by_column"] = {
        "payment_type": 3, "transport_method": 2,
    }
    training = prepared_train.iloc[:6].copy()
    training["payment_type"] = ["A", "A", "A", "B", "Rare", "Unknown"]
    training["transport_method"] = ["A", "A", "A", "B", "B", "B"]
    validation = training.iloc[:5].copy()
    validation["payment_type"] = ["A", "B", "Rare", "Unseen", "Unknown"]
    validation["transport_method"] = ["Unseen", "B", "Unknown", "A", "Unseen"]

    classifier, columns = fit_feature_transformer(training, cfg, task="classification")
    encoded = apply_feature_transformer(validation, classifier, columns, cfg)
    np.testing.assert_allclose(encoded["frequency__payment_type"], [3 / 6, 2 / 6, 2 / 6, 2 / 6, 1 / 6])
    assert "rare_categorical__transport_method_B" in encoded
    assert not any(name.startswith("rare_categorical__payment_type_") for name in encoded)
    with BytesIO() as saved:
        joblib.dump(classifier, saved)
        saved.seek(0)
        restored = joblib.load(saved)
    again = apply_feature_transformer(validation.iloc[::-1], restored, columns, cfg)
    assert encoded.equals(again.iloc[::-1])
    single = apply_feature_transformer(validation.iloc[[3]], classifier, columns, cfg)
    assert single.equals(encoded.iloc[[3]])

    regressor, columns = fit_feature_transformer(training, cfg, task="regression")
    encoded = apply_feature_transformer(validation, regressor, columns, cfg)
    np.testing.assert_allclose(encoded["frequency__transport_method"], [0, 0.5, 0, 0.5, 0])
    assert "rare_categorical__payment_type_Unknown" in encoded

    training["payment_type"] = training["payment_type"].replace("Unknown", "A")
    classifier, columns = fit_feature_transformer(training, cfg, task="classification")
    encoded = apply_feature_transformer(validation, classifier, columns, cfg)
    np.testing.assert_allclose(encoded["frequency__payment_type"], [4 / 6, 2 / 6, 2 / 6, 2 / 6, 0])

    encoding["rare_categories"]["enabled"] = False
    classifier, columns = fit_feature_transformer(training, cfg, task="classification")
    encoded = apply_feature_transformer(validation, classifier, columns, cfg)
    np.testing.assert_allclose(encoded["frequency__payment_type"], [4 / 6, 1 / 6, 1 / 6, 0, 0])

    del encoding["frequency_columns_by_task"]
    legacy, columns = fit_feature_transformer(training, cfg, task="classification")
    encoded = apply_feature_transformer(validation, legacy, columns, cfg)
    assert "categorical__payment_type_B" in encoded
    assert not any(name.startswith("frequency__") for name in encoded)


check_frequency_encoding()
print("Task-specific frequencies, inference and legacy defaults passed.")
