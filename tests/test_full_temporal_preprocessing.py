from copy import deepcopy

import numpy as np
import pandas as pd

from src.data_loader import load_config, load_development_data
from src.feature_transformer import (
    IMPUTATION_INDICATOR_SUFFIXES,
    apply_feature_transformer,
    fit_feature_transformer,
    prepare_feature_frame,
)
from src.fold_preprocessing import prepare_numeric_fold
from src.imputation import TaskPipeline
from src.preprocessing import prepare_rows
from src.validation import temporal_cv

config = load_config()
legacy_config = deepcopy(config)
del legacy_config["preprocessing"]["feature_selection"]["include_imputation_indicators"]
train = load_development_data(config)
train = prepare_rows(train, config)

validation = config["validation"]
years = validation["cv_years"]
horizon = validation["target_horizon_days"]
preprocessing = config["preprocessing"]
rare_settings = preprocessing["encoding"]["rare_categories"]
missing_label = preprocessing["categorical_missing"]["fallback_value"]

results = []

for year, train_fold, val_fold in temporal_cv(
    train,
    years,
    horizon_days=horizon,
    strategy=validation["strategy"],
    train_years=validation["sliding_window"]["train_years"],
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

    indicators = prepared_train.columns.str.endswith(IMPUTATION_INDICATOR_SUFFIXES)
    assert indicators.sum() == 80
    legacy_features = prepare_feature_frame(prepared_train, legacy_config)
    assert set(legacy_features.columns) - set(feature_columns) == set(prepared_train.columns[indicators])
    assert not X_train.columns.str.endswith(IMPUTATION_INDICATOR_SUFFIXES).any()
    # Width depends on fold categories and the configured engineered features.
    assert set(transformer.get_feature_names_out()) == set(X_train.columns)
    for feature in preprocessing["feature_engineering"]["features"]:
        assert f"numeric__{feature}" in X_train

    assert X_train.index.equals(train_fold.index)
    assert X_val.index.equals(val_fold.index)

    if rare_settings["enabled"]:
        for column, threshold in rare_settings["min_frequency_by_column"].items():
            values = prepared_train[column]
            counts = values.value_counts()
            rare_levels = counts.index[(counts < threshold) & (counts.index != missing_label)]
            grouped = f"rare_categorical__{column}_Other"
            if len(rare_levels):
                assert X_train[grouped].astype(bool).equals(values.isin(rare_levels))
            else:
                assert grouped not in X_train
            print(f"  {column}: {len(rare_levels)} levels, {values.isin(rare_levels).sum()} rows grouped")

    assert (
        pd.to_datetime(train_fold["purchase_date"]).max()
        + pd.Timedelta(days=horizon)
        < pd.Timestamp(f"{year}-01-01")
    )

    task_widths = {}
    inputs = val_fold.drop(columns=[
        config["columns"]["classification_target"],
        config["columns"]["regression_target"],
    ])
    for task, model, column in (
        ("classification", "logistic_regression", "payment_type"),
        ("regression", "ridge", "transport_method"),
    ):
        pipeline = TaskPipeline(config, task, model).fit(train_fold)
        # Reference encoding uses the same defined eligible population as evaluation.
        mask = (train_fold[config["columns"]["classification_target"]].eq(1)
                & train_fold[config["columns"]["regression_target"]].notna())
        reference_train = prepared_train.loc[mask] if task == "regression" else prepared_train
        reference, columns = fit_feature_transformer(reference_train, config, task=task)
        expected = apply_feature_transformer(prepared_val, reference, columns, config)
        actual = apply_feature_transformer(
            pipeline.features.numeric.transform(val_fold),
            pipeline.transformer, pipeline.columns, pipeline.config,
        )
        pd.testing.assert_frame_equal(actual, expected)
        selected = pipeline.features.transform(inputs)
        assert list(selected.columns) == pipeline.selected_features_
        assert pipeline.model.n_features_in_ == selected.shape[1]
        assert np.isfinite(selected.to_numpy()).all()
        if task == "classification":
            assert pipeline.features.selection_ is not None
        else:
            assert pipeline.features.selection_ is None
        predictions = pipeline.predict(inputs)
        assert predictions.shape == (len(val_fold),) and np.isfinite(predictions).all()
        task_widths[f"{task}_features"] = selected.shape[1]

    results.append({
        "fold": year,
        "train_rows": len(X_train),
        "validation_rows": len(X_val),
        "features": X_train.shape[1],
        **task_widths,
        "train_missing": int(X_train.isna().sum().sum()),
        "validation_missing": int(X_val.isna().sum().sum()),
    })

    print(f"Fold {year}: PASSED")

summary = pd.DataFrame(results)

assert len(summary) == len(years)

print("\n=== TEMPORAL CV PREPROCESSING SUMMARY ===")
print(summary.to_string(index=False))

print("\nAll temporal preprocessing folds passed.")
