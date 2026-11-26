"""Preprocessing utilities for the ETAI classification project."""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


def prepare_rows(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Prepare training rows before temporal splitting.

    Only performs deterministic operations.
    Does not learn statistics from the dataset.
    """
    data = df.copy()
    prep_config = config["preprocessing"]
    date_col = prep_config["date_features"]["source_column"]

    data[date_col] = pd.to_datetime(
        data[date_col],
        errors="coerce"
    )

    if prep_config["drop_rows_without_purchase_date"]:
        data = data.dropna(subset=[date_col]).copy()

    return data


def create_date_features(
    df: pd.DataFrame,
    config: dict
) -> pd.DataFrame:
    """
    Create features from the purchase date.

    Safe because these features depend only
    on the current observation's date.
    """
    data = df.copy()

    date_config = config["preprocessing"]["date_features"]
    date_col = date_config["source_column"]

    dates = pd.to_datetime(data[date_col])

    extract = date_config["extract"]

    if "year" in extract:
        data["purchase_year"] = dates.dt.year.fillna(0).astype(int)

    if "month" in extract:
        data["purchase_month"] = dates.dt.month.fillna(0).astype(int)

    if "day_of_week" in extract:
        data["purchase_day_of_week"] = dates.dt.dayofweek.fillna(-1).astype(int)

    if date_config["drop_original_from_model"]:
        data = data.drop(columns=[date_col])

    return data


def select_features(
    df: pd.DataFrame,
    config: dict
) -> pd.DataFrame:
    """
    Remove identifiers and targets from model features.
    """
    data = df.copy()

    drop_columns = list(
        config["preprocessing"]["feature_selection"]["drop_columns"]
    )

    drop_columns += [
        config["columns"]["classification_target"],
        config["columns"]["regression_target"],
    ]

    data = data.drop(
        columns=drop_columns,
        errors="ignore"
    )

    return data


def prepare_model_features(
    df: pd.DataFrame,
    config: dict
) -> pd.DataFrame:
    """
    Apply deterministic feature preparation.

    Does not perform imputation, encoding, or scaling.
    """
    data = create_date_features(df, config)
    data = select_features(data, config)

    return data


def fill_structural_missing(
    df: pd.DataFrame,
    config: dict
) -> pd.DataFrame:
    """
    Fill missing values caused by insufficient purchase history.

    No statistics are learned from the data.
    Unexpected missing values are left unchanged.
    """
    data = df.copy()
    prep = config["preprocessing"]

    prior_count = data["prior_purchase_count"]

    rules = {
        "prior_mean_gap_days": (
            prior_count < 2,
            prep["historical_gaps"]["insufficient_history_value"],
        ),
        "prior_std_gap_days": (
            prior_count < prep["historical_gaps"]["std_gap"][
                "minimum_prior_purchases"
            ],
            prep["historical_gaps"]["std_gap"][
                "insufficient_history_value"
            ],
        ),
        "prior_mean_purchase_value": (
            prior_count < prep["historical_purchase_values"]["mean"][
                "minimum_prior_purchases"
            ],
            prep["historical_purchase_values"]["mean"][
                "insufficient_history_value"
            ],
        ),
        "prior_std_purchase_value": (
            prior_count < prep["historical_purchase_values"]["std"][
                "minimum_prior_purchases"
            ],
            prep["historical_purchase_values"]["std"][
                "insufficient_history_value"
            ],
        ),
        "prior_min_purchase_value": (
            prior_count < prep["historical_purchase_values"]["min_max"][
                "minimum_prior_purchases"
            ],
            prep["historical_purchase_values"]["min_max"][
                "insufficient_history_value"
            ],
        ),
        "prior_max_purchase_value": (
            prior_count < prep["historical_purchase_values"]["min_max"][
                "minimum_prior_purchases"
            ],
            prep["historical_purchase_values"]["min_max"][
                "insufficient_history_value"
            ],
        ),
        "days_since_previous_purchase": (
            prior_count == 0,
            prep["recency"]["insufficient_history_value"],
        ),
        "spend_365d": (
            data["purchase_count_365d"] == 0,
            0,
        ),
    }

    for column, (condition, fill_value) in rules.items():
        missing = data[column].isna()

        # Preserve original missingness information
        data[f"{column}_was_missing"] = missing.astype(int)

        # Only fill structurally missing values
        mask = missing & condition
        data.loc[mask, column] = fill_value

    return data


def recover_historical_mean(
    df: pd.DataFrame,
    config: dict
) -> pd.DataFrame:
    """
    Recover missing historical mean purchase values
    using historical total spend and purchase count.

    Does not modify existing non-missing values.
    """
    data = df.copy()

    settings = config["preprocessing"][
        "historical_purchase_values"
    ]["mean"]

    if settings["unexpected_missing_strategy"] != "formula":
        return data

    column = "prior_mean_purchase_value"

    missing = data[column].isna()

    recoverable = (
        missing
        & (data["prior_purchase_count"] >= 1)
        & data["prior_total_spend"].notna()
    )

    data["prior_mean_purchase_value_was_recovered"] = (
        recoverable.astype(int)
    )

    data.loc[recoverable, column] = (
        data.loc[recoverable, "prior_total_spend"]
        / data.loc[recoverable, "prior_purchase_count"]
    )

    return data


def recover_recency(
    df: pd.DataFrame,
    history: pd.DataFrame,
    config: dict
) -> pd.DataFrame:
    """
    Recover missing recency using strictly earlier purchases.

    history must contain only transactions available
    at prediction time.
    """
    data = df.copy()

    settings = config["preprocessing"]["recency"]

    if not settings["recover_from_previous_date"]:
        return data

    date_col = config["preprocessing"][
        "date_features"
    ]["source_column"]

    customer_col = config["columns"]["customer_id"]

    data[date_col] = pd.to_datetime(data[date_col])

    past = history[[customer_col, date_col]].copy()
    past[date_col] = pd.to_datetime(past[date_col])

    past = past.dropna(subset=[date_col])
    past = past.rename(
        columns={date_col: "historical_purchase_date"}
    )

    left = data.assign(_row_order=range(len(data)))
    left = left.sort_values(date_col)

    right = past.sort_values("historical_purchase_date")

    merged = pd.merge_asof(
        left,
        right,
        left_on=date_col,
        right_on="historical_purchase_date",
        by=customer_col,
        direction="backward",
        allow_exact_matches=False,
    )

    missing = merged["days_since_previous_purchase"].isna()

    recoverable = (
        missing
        & (merged["prior_purchase_count"] >= 1)
        & merged["historical_purchase_date"].notna()
    )

    merged["recency_was_recovered"] = recoverable.astype(int)

    merged.loc[recoverable, "days_since_previous_purchase"] = (
        merged.loc[recoverable, date_col]
        - merged.loc[recoverable, "historical_purchase_date"]
    ).dt.days.to_numpy()

    merged = (
        merged.sort_values("_row_order")
        .drop(columns=["_row_order", "historical_purchase_date"])
    )

    merged.index = data.index

    return merged


def fit_recency_imputer(train_df: pd.DataFrame):
    """Fit a recency imputer using training rows only."""
    imputer = SimpleImputer(strategy="median")

    imputer.fit(
        train_df[["days_since_previous_purchase"]]
    )

    return imputer


def apply_recency_imputer(
    df: pd.DataFrame,
    imputer
) -> pd.DataFrame:
    """Apply an already fitted imputer without refitting."""
    data = df.copy()

    data["days_since_previous_purchase"] = (
        imputer.transform(
            data[["days_since_previous_purchase"]]
        ).ravel()
    )

    return data




def fit_numeric_imputers(
    train_df: pd.DataFrame,
    columns: list[str],
    strategy: str = "median",
    knn_settings: dict | None = None,
) -> dict:
    """Fit numeric imputers using training-fold data only."""

    if strategy in {"mean", "median"}:
        imputers = {}

        for column in columns:
            imputer = SimpleImputer(strategy=strategy)
            imputer.fit(train_df[[column]])
            imputers[column] = imputer

        return imputers

    if strategy == "knn":
        if knn_settings is None:
            raise ValueError("KNN settings are required.")

        reference_columns = knn_settings.get(
            "reference_columns", []
        )

        # Include targets for imputation and reference features.
        feature_columns = list(
            dict.fromkeys(columns + reference_columns)
        )

        missing_columns = [
            col for col in feature_columns
            if col not in train_df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"KNN columns not found: {missing_columns}"
            )

        if not all(
            pd.api.types.is_numeric_dtype(train_df[col])
            for col in feature_columns
        ):
            raise ValueError(
                "KNN imputation requires numeric features."
            )

        # Scaling matters because KNN uses distances.
        scaler = RobustScaler()
        scaled_train = scaler.fit_transform(
            train_df[feature_columns]
        )

        imputer = KNNImputer(
            n_neighbors=knn_settings.get("n_neighbors", 5),
            weights=knn_settings.get("weights", "distance"),
            keep_empty_features=True,
        )

        imputer.fit(scaled_train)

        return {
            "__knn__": {
                "scaler": scaler,
                "imputer": imputer,
                "feature_columns": feature_columns,
                "target_columns": columns,
            }
        }

    raise ValueError(
        f"Unsupported numeric imputation strategy: {strategy}"
    )


def apply_numeric_imputers(
    df: pd.DataFrame,
    imputers: dict,
) -> pd.DataFrame:
    """Apply previously fitted numeric imputers."""

    data = df.copy()

    if "__knn__" in imputers:
        settings = imputers["__knn__"]

        feature_columns = settings["feature_columns"]
        target_columns = settings["target_columns"]

        scaler = settings["scaler"]
        imputer = settings["imputer"]

        scaled = scaler.transform(
            data[feature_columns]
        )

        imputed_scaled = imputer.transform(scaled)

        imputed_values = scaler.inverse_transform(
            imputed_scaled
        )

        imputed_df = pd.DataFrame(
            imputed_values,
            columns=feature_columns,
            index=data.index,
        )

        # Only fill missing values in configured columns.
        # All original observed values are preserved.
        for column in target_columns:
            missing_mask = data[column].isna()

            data.loc[missing_mask, column] = (
                imputed_df.loc[missing_mask, column]
            )

        return data

    # Original mean / median behavior
    for column, imputer in imputers.items():
        data[column] = imputer.transform(
            data[[column]]
        ).ravel()

    return data



def clean_spend_365d(
    df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Convert configured suspicious spend_365d values to NaN.

    Does not fit an imputer or change valid values.
    """
    data = df.copy()
    settings = config["preprocessing"]["spend_365d"]

    column = "spend_365d"

    sentinel_mask = pd.Series(False, index=data.index)
    negative_mask = pd.Series(False, index=data.index)

    if settings["sentinel_strategy"] == "treat_as_missing":
        sentinel_mask = (
            data[column] == settings["sentinel_value"]
        )

    if settings["negative_values_strategy"] == "treat_as_missing":
        negative_mask = data[column] < 0

    correction_mask = sentinel_mask | negative_mask

    data["spend_365d_was_corrected"] = (
        correction_mask.astype(int)
    )

    data.loc[correction_mask, column] = float("nan")

    return data


def clean_prior_max_purchase_value(
    df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Flag historical maximum purchase values that exceed
    the historical total spend.

    Assumes historical purchases are nonnegative and
    both columns use the same monetary definition.

    Does not fit an imputer.
    """
    data = df.copy()

    settings = config["preprocessing"][
        "historical_purchase_values"
    ]["max_consistency"]

    column = "prior_max_purchase_value"

    correction_mask = pd.Series(
        False, index=data.index
    )

    if settings["check_against_total_spend"]:
        correction_mask = (
            (data["prior_purchase_count"] >= 1)
            & data[column].notna()
            & data["prior_total_spend"].notna()
            & (data["prior_total_spend"] >= 0)
            & (
                data[column]
                > data["prior_total_spend"] + 0.01
            )
        )

    if settings["add_correction_indicator"]:
        data["prior_max_purchase_value_was_corrected"] = (
            correction_mask.astype(int)
        )

    data.loc[correction_mask, column] = float("nan")

    return data


def recover_categorical_missing(
    df: pd.DataFrame,
    history: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Recover missing categories from earlier customer records."""
    data = df.copy()

    settings = config["preprocessing"]["categorical_missing"]
    fallback = settings["fallback_value"]

    customer_col = config["columns"]["customer_id"]
    date_col = config["preprocessing"]["date_features"]["source_column"]

    recover_columns = settings["recover_from_customer_history"]
    unknown_columns = settings["keep_unknown"]

    for col in recover_columns + unknown_columns:
        missing = data[col].isna()

        if settings["add_missing_indicator"]:
            data[f"{col}_was_missing"] = missing.astype(int)

        if col in recover_columns:
            data[f"{col}_was_recovered"] = 0

            # Only search historical records with known values
            reference = history.loc[
                history[col].notna(),
                [customer_col, date_col, col],
            ].copy()

            reference[date_col] = pd.to_datetime(
                reference[date_col]
            )
            reference = reference.dropna(
                subset=[customer_col, date_col]
            )

            candidates = data.loc[
                missing & data[customer_col].notna(),
                [customer_col, date_col],
            ].copy()

            candidates[date_col] = pd.to_datetime(
                candidates[date_col]
            )
            candidates = candidates.dropna(subset=[date_col])

            if not reference.empty and not candidates.empty:
                candidates["_original_index"] = candidates.index

                candidates = candidates.sort_values(date_col)
                reference = reference.sort_values(date_col)

                matched = pd.merge_asof(
                    candidates,
                    reference.rename(columns={
                        col: "_historical_value"
                    }),
                    on=date_col,
                    by=customer_col,
                    direction="backward",
                    allow_exact_matches=False,
                )

                recovered = matched["_historical_value"].notna()

                recovered_rows = matched.loc[recovered]

                data.loc[
                    recovered_rows["_original_index"],
                    col,
                ] = recovered_rows["_historical_value"].to_numpy()

                data.loc[
                    recovered_rows["_original_index"],
                    f"{col}_was_recovered",
                ] = 1

        data[col] = data[col].fillna(fallback)

    return data

def clean_historical_gaps(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Replace negative historical gaps with missing values."""

    data = df.copy()
    settings = config["preprocessing"]["historical_gaps"]

    if settings["negative_values_strategy"] != "treat_as_missing":
        raise ValueError("Unsupported negative gap strategy.")

    for column in ["prior_mean_gap_days", "prior_std_gap_days"]:
        invalid = data[column].notna() & (data[column] < 0)

        if settings["add_correction_indicator"]:
            data[f"{column}_was_corrected"] = invalid.astype(int)

        data.loc[invalid, column] = float("nan")

    return data

class AnomalyScore(TransformerMixin, BaseEstimator):

    def __init__(
        self,
        method="isolation_forest",
        n_neighbors=20,
        random_state=42
    ):
        self.method = method
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    def fit(self, X, y=None):

        if self.method == "isolation_forest":

            self.detector_ = IsolationForest(
                random_state=self.random_state
            )

        elif self.method == "lof":

            self.detector_ = LocalOutlierFactor(
                n_neighbors=self.n_neighbors,
                novelty=True
            )

        else:

            raise ValueError(
                f"Unknown anomaly method: {self.method}"
            )

        self.detector_.fit(X)

        self.n_features_in_ = X.shape[1]

        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.to_numpy()

        return self

    def fit_transform(self, X, y=None):

        self.fit(X)

        if self.method == "lof":

            scores = -self.detector_.negative_outlier_factor_

            scores = np.log1p(scores)

        else:

            scores = -self.detector_.score_samples(X)

        return self._add_score(X, scores)

    def transform(self, X):

        scores = -self.detector_.score_samples(X)

        if self.method == "lof":

            scores = np.log1p(scores)

        return self._add_score(X, scores)

    def _add_score(self, X, scores):

        if isinstance(X, pd.DataFrame):

            result = X.copy()

        else:

            result = pd.DataFrame(X)

        result["anomaly_score"] = scores

        return result

    def get_feature_names_out(self, input_features=None):

        if input_features is None:

            if hasattr(self, "feature_names_in_"):
                input_features = self.feature_names_in_

            else:
                input_features = [
                    f"x{i}"
                    for i in range(self.n_features_in_)
                ]

        return np.append(
            np.asarray(input_features, dtype=object),
            "anomaly_score"
        )