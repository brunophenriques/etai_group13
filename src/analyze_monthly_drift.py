
"""Analyze monthly classification drift from saved validation predictions."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data_loader import load_config


def find_predictions_file(config):
    """Use the configured CSV or the latest threshold diagnostic."""

    settings = config["preprocessing"]["monthly_drift"]
    configured_path = settings["predictions_file"]

    if configured_path:
        path = Path(configured_path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[1] / path
    else:
        results_dir = (
            Path(__file__).resolve().parents[1]
            / config["paths"]["results"]
        )

        candidates = list(
            results_dir.glob(
                "threshold_diagnostics_*/validation_predictions.csv"
            )
        )

        if not candidates:
            raise FileNotFoundError(
                "No validation_predictions.csv found. "
                "Run diagnose_classification_threshold.py first."
            )

        path = max(candidates, key=lambda p: p.stat().st_mtime)

    if not path.is_file():
        raise FileNotFoundError(path)

    return path


def analyze_monthly_predictions(predictions, config):
    """Calculate monthly performance and calibration diagnostics."""

    required_columns = {
        "fold", "date", "actual", "probability"
    }

    missing = required_columns - set(predictions.columns)

    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")

    predictions = predictions.copy()

    predictions["date"] = pd.to_datetime(
        predictions["date"], errors="raise"
    )

    if predictions["date"].isna().any():
        raise ValueError("Missing prediction dates.")

    if predictions["probability"].isna().any():
        raise ValueError("Missing predicted probabilities.")

    if not predictions["probability"].between(0, 1).all():
        raise ValueError("Probabilities must be between 0 and 1.")

    if not predictions["actual"].isin([0, 1]).all():
        raise ValueError("Labels must be binary.")

    if not (
        predictions["date"].dt.year == predictions["fold"]
    ).all():
        raise ValueError("Prediction dates do not match fold years.")

    if any(
        year >= config["validation"]["holdout_year"]
        for year in predictions["fold"].unique()
    ):
        raise ValueError("Holdout predictions are not allowed.")

    settings = config["preprocessing"]["monthly_drift"]
    min_rows = settings["min_monthly_rows"]

    if not isinstance(min_rows, int) or min_rows < 1:
        raise ValueError("min_monthly_rows must be positive.")

    threshold = config["modeling"]["evaluation"][
        "classification_threshold"
    ]

    predictions["month"] = (
        predictions["date"].dt.to_period("M").astype(str)
    )

    records = []

    for (fold, month), group in predictions.groupby(
        ["fold", "month"], sort=True
    ):
        y = group["actual"].to_numpy(dtype=int)
        probabilities = group["probability"].to_numpy(
            dtype=float
        )

        predicted = (
            probabilities >= threshold
        ).astype(int)

        # ROC-AUC is undefined when only one class is present.
        has_both_classes = np.unique(y).size == 2

        records.append({
            "fold": int(fold),
            "month": month,
            "rows": len(group),
            "low_sample": len(group) < min_rows,
            "actual_positive_rate": float(y.mean()),
            "mean_predicted_probability": float(
                probabilities.mean()
            ),
            "calibration_gap": float(
                probabilities.mean() - y.mean()
            ),
            "predicted_positive_rate": float(
                predicted.mean()
            ),
            "f1": float(
                f1_score(y, predicted, zero_division=0)
            ),
            "precision": float(
                precision_score(
                    y, predicted, zero_division=0
                )
            ),
            "recall": float(
                recall_score(
                    y, predicted, zero_division=0
                )
            ),
            "roc_auc": (
                float(roc_auc_score(y, probabilities))
                if has_both_classes else np.nan
            ),
            "pr_auc": (
                float(average_precision_score(y, probabilities))
                if has_both_classes else np.nan
            ),
        })

    return pd.DataFrame(records)


def main():
    config = load_config()

    input_file = find_predictions_file(config)

    print(f"Reading predictions from: {input_file}")

    predictions = pd.read_csv(input_file)

    monthly = analyze_monthly_predictions(
        predictions, config
    )

    output_dir = (
        Path(__file__).resolve().parents[1]
        / config["paths"]["results"]
        / f"monthly_drift_{datetime.now():%Y-%m-%d_%H-%M-%S-%f}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    monthly.to_csv(
        output_dir / "monthly_drift.csv",
        index=False,
    )

    print("\n=== MONTHLY DRIFT: ALL FOLDS ===")

    print(
        monthly[
            [
                "month",
                "rows",
                "low_sample",
                "actual_positive_rate",
                "mean_predicted_probability",
                "calibration_gap",
                "f1",
                "roc_auc",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    print("\n=== 2022 MONTHLY DIAGNOSTIC ===")

    fold_2022 = monthly.loc[monthly["fold"] == 2022]

    if fold_2022.empty:
        print("No predictions for fold 2022.")
    else:
        print(
            fold_2022[
                [
                    "month",
                    "rows",
                    "actual_positive_rate",
                    "mean_predicted_probability",
                    "calibration_gap",
                    "f1",
                    "roc_auc",
                ]
            ]
            .round(4)
            .to_string(index=False)
        )

    print(f"\nResults saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
