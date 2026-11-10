
"""Preprocessing utilities for the ETAI classification project."""

import pandas as pd


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
        data["purchase_year"] = dates.dt.year

    if "month" in extract:
        data["purchase_month"] = dates.dt.month

    if "day_of_week" in extract:
        data["purchase_day_of_week"] = dates.dt.dayofweek

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
