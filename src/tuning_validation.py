"""Temporal inner folds for nested cross-validation."""

import pandas as pd


VALID_STRATEGIES = {"expanding_window", "sliding_window"}


def make_inner_folds(train_df, config):
    """Build expanding or sliding inner folds with a target-label embargo.

    Only rows available inside the given outer training fold are considered.
    Validation windows use [start, end), while training labels must have
    completed their horizon strictly before the validation window begins.
    """
    settings = config["hyperparameter_tuning"]["nested_cv"]["inner"]
    strategy = settings["strategy"]
    if strategy not in VALID_STRATEGIES:
        raise ValueError(f"Unsupported inner CV strategy: {strategy}")

    max_splits = int(settings["max_splits"])
    window_days = int(settings["validation_window_days"])
    step_days = int(settings["step_days"])
    horizon_days = int(settings["target_horizon_days"])
    min_train = int(settings["min_train_rows"])
    min_val = int(settings["min_validation_rows"])

    if min(max_splits, window_days, step_days, min_train, min_val) <= 0 or horizon_days < 0:
        raise ValueError("Invalid inner CV configuration.")
    if horizon_days != int(config["validation"]["target_horizon_days"]):
        raise ValueError("Inner and outer target horizons must match.")

    train_years = None
    if strategy == "sliding_window":
        # By default, the same sliding-window length is used inside and outside.
        train_years = int(
            settings.get("train_years", config["validation"]["sliding_window"]["train_years"])
        )
        if train_years < 1:
            raise ValueError("inner train_years must be >= 1")

    df = train_df.copy()
    df["purchase_date"] = pd.to_datetime(df["purchase_date"], errors="raise")
    if df["purchase_date"].isna().any():
        raise ValueError("Inner CV contains missing purchase dates.")
    df = df.sort_values("purchase_date")
    if df.empty:
        raise ValueError("Cannot create inner folds from empty outer training data.")

    end = df["purchase_date"].max().normalize() + pd.Timedelta(days=1)
    folds = []

    for offset in range(max_splits):
        val_end = end - pd.Timedelta(days=offset * step_days)
        val_start = val_end - pd.Timedelta(days=window_days)

        dates = df["purchase_date"]
        train_mask = dates + pd.Timedelta(days=horizon_days) < val_start
        if strategy == "sliding_window":
            train_start = val_start - pd.DateOffset(years=train_years)
            train_mask &= dates >= train_start

        val_mask = (dates >= val_start) & (dates < val_end)
        inner_train = df.loc[train_mask].copy()
        inner_val = df.loc[val_mask].copy()

        if len(inner_train) < min_train or len(inner_val) < min_val:
            continue

        latest_label_date = inner_train["purchase_date"].max() + pd.Timedelta(days=horizon_days)
        if not latest_label_date < val_start:
            raise AssertionError("Inner train labels overlap validation start.")
        if not inner_train.index.intersection(inner_val.index).empty:
            raise AssertionError("Inner train and validation rows overlap.")

        folds.append({
            "train": inner_train,
            "val": inner_val,
            "validation_start": val_start,
            "validation_end": val_end,
        })

    folds.reverse()
    if not folds:
        raise ValueError(
            "No valid inner folds. Review window sizes and minimum observation requirements."
        )
    return folds
