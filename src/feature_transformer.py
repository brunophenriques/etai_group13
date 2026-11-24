"""Fit encoders and scalers on training folds; apply without refitting."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.utils.validation import check_is_fitted

from src.preprocessing import prepare_model_features


IMPUTATION_INDICATOR_SUFFIXES = (
    "_was_missing",
    "_no_training_donors",
    "_insufficient_history",
    "_was_corrected",
    "_was_recovered",
)


def categorical_columns(encoding):
    """Include legacy configs that declared grouped features separately."""
    legacy_columns = encoding["rare_categories"].get("columns", [])
    return list(dict.fromkeys(encoding["categorical_columns"] + legacy_columns))


class _GroupedCategories(TransformerMixin, BaseEstimator):
    """Share train-fitted category grouping between encoders."""

    def __init__(self, min_frequency_by_column, missing_label):
        self.min_frequency_by_column = min_frequency_by_column
        self.missing_label = missing_label

    def _fit_groups(self, X):
        self.kept_categories_ = {}
        self.other_labels_ = {}
        for column, minimum in self.min_frequency_by_column.items():
            counts = X[column].value_counts()
            threshold = minimum * len(X) if isinstance(minimum, float) else minimum
            kept = counts.index[(counts >= threshold) | (counts.index == self.missing_label)]
            self.kept_categories_[column] = kept
            if len(kept) < len(counts):
                # A genuine, frequent category called Other keeps its own column.
                other = "Other"
                while other in kept or other == self.missing_label:
                    other += " (grouped)"
                self.other_labels_[column] = other

    def group_categories(self, X):
        """Apply the fitted grouping before any numerical encoding."""
        check_is_fitted(self, ("kept_categories_", "other_labels_"))
        grouped = X.copy()
        for column, other in self.other_labels_.items():
            values = X[column]
            keep = values.isin(self.kept_categories_[column]) | values.eq(self.missing_label)
            grouped[column] = values.where(keep, other)
        return grouped


class RareCategoryEncoder(_GroupedCategories):
    """Group rare levels using fitted training counts, then one-hot encode."""

    def fit(self, X, y=None):
        self._fit_groups(X)
        self.encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.encoder_.fit(self.group_categories(X))
        return self

    def transform(self, X):
        check_is_fitted(self, "encoder_")
        return self.encoder_.transform(self.group_categories(X))

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "encoder_")
        return self.encoder_.get_feature_names_out(input_features)


class FrequencyEncoder(_GroupedCategories):
    """Encode each grouped category by its relative training frequency."""

    def fit(self, X, y=None):
        self._fit_groups(X)
        grouped = self.group_categories(X)
        self.feature_names_in_ = grouped.columns.to_numpy(dtype=object)
        self.frequencies_ = {
            column: grouped[column].value_counts(normalize=True)
            for column in self.feature_names_in_
        }
        return self

    def transform(self, X):
        check_is_fitted(self, "frequencies_")
        grouped = self.group_categories(X)
        return np.column_stack([
            grouped[column].map(self.frequencies_[column]).fillna(0.0)
            for column in self.feature_names_in_
        ])

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "frequencies_")
        return self.feature_names_in_.copy()


def prepare_feature_frame(df, config):
    """Select and prepare features for model encoding."""

    X = prepare_model_features(df, config).copy()

    preprocessing = config["preprocessing"]
    if not preprocessing["feature_selection"].get("include_imputation_indicators", True):
        indicators = X.columns[X.columns.str.endswith(IMPUTATION_INDICATOR_SUFFIXES)]
        X = X.drop(columns=indicators)

    settings = preprocessing["encoding"]

    # Ensure calendar variables are encoded as categories.
    for col in categorical_columns(settings):
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


def fit_feature_transformer(train_df, config, *, task=None):
    """Fit preprocessing on training data, with one-hot as the taskless baseline."""

    X_train = prepare_feature_frame(train_df, config)

    encoding = config["preprocessing"]["encoding"]
    scaling = config["preprocessing"]["scaling"]

    if encoding["strategy"] != "one_hot":
        raise ValueError("The default encoding strategy must be one_hot.")

    categorical = categorical_columns(encoding)
    frequency = encoding.get("frequency_columns_by_task", {}).get(task, []) if task else []
    if not isinstance(frequency, list) or any(column not in categorical for column in frequency):
        raise ValueError(f"Frequency columns for {task} must be a list of configured categorical features.")
    frequency = list(dict.fromkeys(frequency))

    rare_settings = encoding["rare_categories"]
    legacy_rare_settings = "min_frequency_by_column" not in rare_settings
    if legacy_rare_settings:
        thresholds = {
            column: rare_settings["min_frequency"] for column in rare_settings["columns"]
        } if rare_settings["enabled"] else {}
    else:
        thresholds = rare_settings["min_frequency_by_column"]
        if not isinstance(thresholds, dict):
            raise ValueError("min_frequency_by_column must map categorical columns to thresholds.")
        for column, minimum in thresholds.items():
            if column not in encoding["categorical_columns"]:
                raise ValueError(f"Rare-category threshold configured for non-categorical feature: {column}")
            if type(minimum) is not int or minimum < 1:
                raise ValueError(f"Rare-category threshold for {column} must be a positive integer.")
        if not rare_settings["enabled"]:
            thresholds = {}

    rare = {column: minimum for column, minimum in thresholds.items() if column not in frequency}
    missing_label = config["preprocessing"]["categorical_missing"]["fallback_value"]
    regular_categorical = [
        col for col in categorical if col not in rare and col not in frequency
    ]

    categorical_set = set(categorical)

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
        if legacy_rare_settings:
            rare_encoder = OneHotEncoder(
                handle_unknown=rare_settings["handle_unknown"],
                min_frequency=rare_settings["min_frequency"],
                sparse_output=False,
            )
        else:
            rare_encoder = RareCategoryEncoder(rare, missing_label)
        transformers.append(("rare_categorical", rare_encoder, list(rare)))

    if numeric:
        transformers.append((
            "numeric",
            scaler,
            numeric,
        ))

    if frequency:
        frequency_thresholds = {column: thresholds[column] for column in frequency if column in thresholds}
        transformers.append(("frequency", FrequencyEncoder(frequency_thresholds, missing_label), frequency))

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
