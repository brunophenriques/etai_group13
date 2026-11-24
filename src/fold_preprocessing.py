
"""Coordinate preprocessing and feature engineering within temporal folds."""

import numpy as np

from src.imputation import ImputationPreprocessor
from src.feature_engineering import apply_configured_feature_engineering


def prepare_numeric_fold(train_fold, val_fold, config, task=None):
    """Fit preprocessing on training data; reuse fitted state for validation."""

    # Original preprocessing (Pinho)
    preprocessor = ImputationPreprocessor(config, task).fit(train_fold)

    train = preprocessor.transform(train_fold)
    val = preprocessor.transform(val_fold)

    # Feature Engineering (Joana), controlled by config.yaml
    settings = config["preprocessing"].get("feature_engineering", {})

    state = {"__preprocessor__": preprocessor}

    if settings.get("enabled", False):
        train = apply_configured_feature_engineering(train, config)
        val = apply_configured_feature_engineering(val, config)

        features = settings["features"]

        # New features may contain NaNs or infinities
        train[features] = train[features].replace(
            [np.inf, -np.inf], np.nan
        )
        val[features] = val[features].replace(
            [np.inf, -np.inf], np.nan
        )

        # Fit missing-value replacement on training fold only
        medians = train[features].median().fillna(0.0)

        train[features] = train[features].fillna(medians)
        val[features] = val[features].fillna(medians)

        state["__feature_engineering_medians__"] = medians

    return train, val, state
