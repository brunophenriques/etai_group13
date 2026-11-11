
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)

from src.preprocessing import prepare_model_features


def prepare_feature_frame(df, config):
    """Select and prepare features for model encoding."""

    X = prepare_model_features(df, config).copy()

    settings = config["preprocessing"]["encoding"]

    categorical = (
        settings["categorical_columns"]
        + settings["rare_categories"]["columns"]
    )

    # Ensure calendar variables are encoded as categories.
    for col in categorical:
        if col not in X.columns:
            raise ValueError(f"Missing feature: {col}")

        X[col] = X[col].astype("string").astype(str)

    # No identifiers, raw dates or targets may enter the model.
    forbidden = {
        "ID",
        config["columns"]["customer_id"],
        config["columns"]["classification_target"],
        config["columns"]["regression_target"],
        config["preprocessing"]["date_features"]["source_column"],
    }

    remaining = forbidden.intersection(X.columns)

    if remaining:
        raise ValueError(
            f"Forbidden model features: {remaining}"
        )

    return X


def fit_feature_transformer(train_df, config):
    """
    Fit encoders and scaler using training data only.
    """

    X_train = prepare_feature_frame(train_df, config)

    encoding = config["preprocessing"]["encoding"]
    scaling = config["preprocessing"]["scaling"]

    if encoding["strategy"] != "one_hot":
        raise ValueError("Only one_hot encoding is supported.")

    categorical = encoding["categorical_columns"]
    rare = (
        encoding["rare_categories"]["columns"]
        if encoding["rare_categories"]["enabled"]
        else []
    )

    regular_categorical = [
        col for col in categorical if col not in rare
    ]

    categorical_set = set(regular_categorical + rare)

    numeric = [
        col for col in X_train.columns
        if col not in categorical_set
    ]

    # Catch accidentally unconfigured categorical columns.
    non_numeric = [
        col for col in numeric
        if not pd.api.types.is_numeric_dtype(X_train[col])
    ]

    if non_numeric:
        raise ValueError(
            f"Unconfigured categorical features: {non_numeric}"
        )

    if X_train.isna().any().any():
        missing = X_train.columns[
            X_train.isna().any()
        ].tolist()
        raise ValueError(
            f"Missing values before encoding: {missing}"
        )

    scaler_name = (
        scaling["default"]
        if scaling["enabled"]
        else "passthrough"
    )

    if scaler_name == "robust":
        scaler = RobustScaler()
    elif scaler_name == "standard":
        scaler = StandardScaler()
    elif scaler_name == "passthrough":
        scaler = "passthrough"
    else:
        raise ValueError(
            f"Unsupported scaler: {scaler_name}"
        )

    transformers = []

    if regular_categorical:
        transformers.append((
            "categorical",
            OneHotEncoder(
                handle_unknown=encoding["handle_unknown"],
                drop=(
                    "first"
                    if encoding["drop_first"]
                    else None
                ),
                sparse_output=False,
            ),
            regular_categorical,
        ))

    if rare:
        transformers.append((
            "rare_categorical",
            OneHotEncoder(
                handle_unknown=encoding[
                    "rare_categories"
                ]["handle_unknown"],
                min_frequency=encoding[
                    "rare_categories"
                ]["min_frequency"],
                sparse_output=False,
            ),
            rare,
        ))

    if numeric:
        transformers.append((
            "numeric",
            scaler,
            numeric,
        ))

    transformer = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )

    transformer.fit(X_train)

    return transformer, list(X_train.columns)


def apply_feature_transformer(
    df,
    transformer,
    expected_columns,
    config,
):
    """
    Apply a fitted transformer without refitting.
    """

    X = prepare_feature_frame(df, config)

    if set(X.columns) != set(expected_columns):
        raise ValueError(
            "Training and validation features do not match."
        )

    X = X[expected_columns]

    if X.isna().any().any():
        raise ValueError(
            "Missing values remain before transformation."
        )

    transformed = transformer.transform(X)

    result = pd.DataFrame(
        transformed,
        columns=transformer.get_feature_names_out(),
        index=X.index,
    )

    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError(
            "Non-finite values after transformation."
        )

    return result
