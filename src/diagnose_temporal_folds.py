
"""Diagnose temporal folds and distribution shifts — ETAI Group 13."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.data_loader import load_config, load_data
from src.preprocessing import prepare_rows
from src.validation import temporal_cv


def numeric_drift(train_fold, val_fold, config, year):
    """Compare numeric feature distributions without fitting models."""

    excluded = {
        config["columns"]["customer_id"],
        config["columns"]["classification_target"],
        config["columns"]["regression_target"],
        config["preprocessing"]["date_features"]["source_column"],
        *config["preprocessing"]["feature_selection"]["drop_columns"],
    }

    numeric_columns = [
        column
        for column in train_fold.select_dtypes(
            include="number"
        ).columns
        if column in val_fold.columns
        and pd.api.types.is_numeric_dtype(val_fold[column])
        and column not in excluded
    ]

    records = []

    for column in numeric_columns:
        train_values = pd.to_numeric(
            train_fold[column], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)

        val_values = pd.to_numeric(
            val_fold[column], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)

        train_valid = train_values.dropna()
        val_valid = val_values.dropna()

        train_missing = float(train_values.isna().mean())
        val_missing = float(val_values.isna().mean())

        # KS measures the largest difference between
        # empirical cumulative distributions.
        if len(train_valid) and len(val_valid):
            ks = ks_2samp(
                train_valid,
                val_valid,
                method="auto",
            )

            ks_statistic = float(ks.statistic)
            ks_pvalue = float(ks.pvalue)

            train_median = float(train_valid.median())
            val_median = float(val_valid.median())

            # Standardized mean difference, using
            # training standard deviation as reference.
            train_std = float(train_valid.std(ddof=0))

            if train_std > 0:
                smd = float(
                    (val_valid.mean() - train_valid.mean())
                    / train_std
                )
            else:
                smd = np.nan
        else:
            ks_statistic = np.nan
            ks_pvalue = np.nan
            train_median = np.nan
            val_median = np.nan
            smd = np.nan

        records.append({
            "fold": year,
            "feature": column,
            "train_nonmissing": len(train_valid),
            "val_nonmissing": len(val_valid),
            "train_missing_rate": train_missing,
            "val_missing_rate": val_missing,
            "missing_rate_delta": val_missing - train_missing,
            "train_median": train_median,
            "val_median": val_median,
            "standardized_mean_difference": smd,
            "ks_statistic": ks_statistic,
            "ks_pvalue": ks_pvalue,
        })

    return pd.DataFrame(records)


def main():
    config = load_config()

    target = config["columns"]["classification_target"]
    date_column = config["preprocessing"]["date_features"][
        "source_column"
    ]

    horizon_days = config["validation"]["target_horizon_days"]
    horizon = pd.Timedelta(days=horizon_days)

    train, _, _ = load_data(config)
    train = prepare_rows(train, config)

    # No holdout information enters this diagnostic.
    holdout_year = config["validation"]["holdout_year"]

    if any(
        year >= holdout_year
        for year in config["validation"]["cv_years"]
    ):
        raise ValueError(
            "Diagnostic folds must precede the holdout year."
        )

    output_dir = (
        Path(config["paths"]["results"])
        / f"temporal_diagnostics_{datetime.now():%Y-%m-%d_%H-%M-%S-%f}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    fold_records = []
    drift_results = []

    for year, train_fold, val_fold in temporal_cv(
        train,
        config["validation"]["cv_years"],
        horizon_days=horizon_days,
        strategy=config["validation"]["strategy"],
        train_years=config["validation"]
            .get("sliding_window", {})
            .get("train_years"),
    ):
        train_dates = pd.to_datetime(train_fold[date_column])
        val_dates = pd.to_datetime(val_fold[date_column])

        validation_start = pd.Timestamp(f"{year}-01-01")
        validation_end = pd.Timestamp(f"{year}-12-31")

        latest_train = train_dates.max()
        first_validation = val_dates.min()

        # Assert the actual label-availability embargo.
        embargo_ok = (
            latest_train + horizon < validation_start
        )

        if not embargo_ok:
            raise AssertionError(
                f"Target availability leakage in fold {year}."
            )

        if not val_dates.between(
            validation_start, validation_end
        ).all():
            raise AssertionError(
                f"Validation dates outside fold {year}."
            )

        if not train_fold.index.intersection(
            val_fold.index
        ).empty:
            raise AssertionError(
                f"Overlapping rows in fold {year}."
            )

        y_train = train_fold[target]
        y_val = val_fold[target]

        if (
            y_train.isna().any()
            or y_val.isna().any()
            or not y_train.isin([0, 1]).all()
            or not y_val.isin([0, 1]).all()
        ):
            raise ValueError(
                f"Invalid target labels in fold {year}."
            )

        train_rate = float(y_train.mean())
        val_rate = float(y_val.mean())

        fold_records.append({
            "fold": year,
            "strategy": config["validation"]["strategy"],
            "train_rows": len(train_fold),
            "validation_rows": len(val_fold),
            "train_start": train_dates.min().date(),
            "train_end": latest_train.date(),
            "validation_start": first_validation.date(),
            "validation_end": val_dates.max().date(),
            "gap_days": (
                validation_start - latest_train
            ).days,
            "target_horizon_days": horizon_days,
            "embargo_ok": embargo_ok,
            "train_positive_rate": train_rate,
            "val_positive_rate": val_rate,
            "positive_rate_delta": val_rate - train_rate,
        })

        drift = numeric_drift(
            train_fold, val_fold, config, year
        )

        drift_results.append(drift)

        print(f"\n=== FOLD {year} ===")
        print(
            f"Train: {len(train_fold)} | "
            f"Validation: {len(val_fold)}"
        )
        print(
            f"Train dates: {train_dates.min().date()} "
            f"to {latest_train.date()}"
        )
        print(
            f"Validation dates: {first_validation.date()} "
            f"to {val_dates.max().date()}"
        )
        print(
            f"Gap: {(validation_start - latest_train).days} days | "
            f"Embargo valid: {embargo_ok}"
        )
        print(
            f"Positive rate: {train_rate:.2%} -> "
            f"{val_rate:.2%} "
            f"({(val_rate - train_rate) * 100:+.2f} pp)"
        )

        print("\nTop 10 numeric distribution shifts (KS):")

        if not drift.empty:
            print(
                drift.sort_values(
                    "ks_statistic", ascending=False
                )[
                    [
                        "feature",
                        "ks_statistic",
                        "ks_pvalue",
                        "standardized_mean_difference",
                        "missing_rate_delta",
                    ]
                ]
                .head(10)
                .round(4)
                .to_string(index=False)
            )

    fold_summary = pd.DataFrame(fold_records)
    drift_summary = pd.concat(
        drift_results, ignore_index=True
    )

    fold_summary.to_csv(
        output_dir / "fold_summary.csv",
        index=False,
    )

    drift_summary.to_csv(
        output_dir / "numeric_drift.csv",
        index=False,
    )

    print("\n=== TEMPORAL VALIDATION SUMMARY ===")
    print(
        fold_summary[
            [
                "fold",
                "train_rows",
                "validation_rows",
                "gap_days",
                "embargo_ok",
                "train_positive_rate",
                "val_positive_rate",
                "positive_rate_delta",
            ]
        ].round(4).to_string(index=False)
    )

    print(f"\nResults saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
