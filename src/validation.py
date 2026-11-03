import pandas as pd


def temporal_split(
    df: pd.DataFrame,
    validation_start: str,
    validation_end: str,
    date_column: str = "purchase_date",
    horizon_days: int = 90,
):
    """Create a temporal split with a label-availability gap."""

    dates = pd.to_datetime(df[date_column], errors="coerce")

    start = pd.Timestamp(validation_start)
    end = pd.Timestamp(validation_end)

    if start > end:
        raise ValueError("validation_start must be <= validation_end")

    if horizon_days < 0:
        raise ValueError("horizon_days must be non-negative")

    # Training labels must be fully observable before validation starts.
    train_mask = dates + pd.Timedelta(days=horizon_days) < start

    # Validate on transactions within the specified period.
    validation_mask = dates.between(start, end, inclusive="both")

    train_df = df.loc[train_mask].copy()
    validation_df = df.loc[validation_mask].copy()

    # Safety checks
    assert not train_df.index.intersection(validation_df.index).size

    assert (
        pd.to_datetime(train_df[date_column]).max()
        + pd.Timedelta(days=horizon_days)
        < start
    )

    return train_df, validation_df


def temporal_cv(df, validation_years, horizon_days=90):
    """Generate expanding-window temporal validation folds."""

    for year in validation_years:
        train_fold, val_fold = temporal_split(
            df,
            validation_start=f"{year}-01-01",
            validation_end=f"{year}-12-31",
            horizon_days=horizon_days,
        )

        if train_fold.empty or val_fold.empty:
            raise ValueError(f"Empty training or validation fold for {year}")

        yield year, train_fold, val_fold