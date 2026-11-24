
from pathlib import Path

import pandas as pd

from src.data_loader import load_config, load_data
from src.feature_transformer import (
    fit_feature_transformer,
    prepare_feature_frame,
)
from src.fold_preprocessing import prepare_numeric_fold
from src.preprocessing import prepare_rows
from src.validation import temporal_cv


def run_audit():
    config = load_config()

    results_path = (
        Path(config["paths"]["results"])
        / "classification_cv.csv"
    )

    results = pd.read_csv(results_path)

    print("\n=== RESULTS BY YEAR ===")

    for metric in ["pr_auc", "roc_auc", "f1"]:
        print(f"\n{metric.upper()}")

        comparison = results.pivot(
            index="fold",
            columns="model",
            values=metric,
        )

        print(comparison.round(4).to_string())

    print("\n=== ORIGINAL MODEL FEATURES ===")

    train, _, _ = load_data(config)
    train = prepare_rows(train, config)
    validation = config["validation"]

    year, train_fold, val_fold = next(
        temporal_cv(
            train,
            validation["cv_years"],
            horizon_days=validation["target_horizon_days"],
            strategy=validation["strategy"],
            train_years=validation.get("sliding_window", {}).get("train_years"),
        )
    )

    prepared_train, _, _ = prepare_numeric_fold(
        train_fold,
        val_fold,
        config,
    )

    features = prepare_feature_frame(
        prepared_train,
        config,
    )

    for column in features.columns:
        print(column)

    print(
        f"\nTotal original features: {features.shape[1]}"
    )

    transformer, _ = fit_feature_transformer(
        prepared_train,
        config,
        task="classification",
    )

    print(
        "Features after encoding:",
        len(transformer.get_feature_names_out()),
    )


if __name__ == "__main__":
    run_audit()
