"""Temporal cross-validation with target-label availability embargo."""

import pandas as pd


VALID_STRATEGIES = {"expanding_window", "sliding_window"}


def temporal_split(
    df: pd.DataFrame,
    validation_start: str,
    validation_end: str,
    date_column: str = "purchase_date",
    horizon_days: int = 90,
    strategy: str = "expanding_window",
    train_years: int | None = None,
):
    """Split chronologically, excluding training labels unavailable at validation start.

    Validation boundaries are inclusive for compatibility with the original API.
    For sliding windows, the training period starts ``train_years`` calendar
    years before validation starts. The label embargo is applied to both modes.
    """
    if date_column not in df.columns:
        raise KeyError(f"Missing date column: {date_column}")

    dates = pd.to_datetime(df[date_column], errors="raise")
    if dates.isna().any():
        raise ValueError("Temporal splitting requires non-missing dates.")

    start = pd.Timestamp(validation_start)
    end = pd.Timestamp(validation_end)
    if start > end:
        raise ValueError("validation_start must be <= validation_end")
    if horizon_days < 0:
        raise ValueError("horizon_days must be non-negative")
    if strategy not in VALID_STRATEGIES:
        raise ValueError(f"Unsupported validation strategy: {strategy}")

    train_mask = dates + pd.Timedelta(days=horizon_days) < start
    if strategy == "sliding_window":
        if isinstance(train_years, bool) or not isinstance(train_years, int) or train_years < 1:
            raise ValueError("sliding_window requires train_years >= 1")
        train_start = start - pd.DateOffset(years=train_years)
        train_mask &= dates >= train_start

    validation_mask = dates.between(start, end, inclusive="both")
    train_df = df.loc[train_mask].copy()
    validation_df = df.loc[validation_mask].copy()

    if not train_df.index.intersection(validation_df.index).empty:
        raise AssertionError("Train and validation rows overlap.")
    if not train_df.empty:
        latest_label_date = (
            pd.to_datetime(train_df[date_column]).max()
            + pd.Timedelta(days=horizon_days)
        )
        if not latest_label_date < start:
            raise AssertionError("Training contains labels unavailable at validation start.")

    return train_df, validation_df


def temporal_cv(
    df: pd.DataFrame,
    validation_years,
    horizon_days: int = 90,
    strategy: str = "expanding_window",
    train_years: int | None = None,
):
    """Generate yearly temporal validation folds using a fixed evaluation period."""
    for year in validation_years:
        train_fold, val_fold = temporal_split(
            df,
            validation_start=f"{year}-01-01",
            validation_end=f"{year}-12-31",
            horizon_days=horizon_days,
            strategy=strategy,
            train_years=train_years,
        )
        if train_fold.empty or val_fold.empty:
            raise ValueError(
                f"Empty training or validation fold for {year} "
                f"(strategy={strategy}, train_years={train_years})."
            )
        yield year, train_fold, val_fold
