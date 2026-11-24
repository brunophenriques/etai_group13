
"""Analyze feature importance across temporal folds — ETAI Group 13."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, f1_score

from src.data_loader import load_config, load_data
from src.preprocessing import prepare_rows
from src.validation import temporal_cv
from src.fold_preprocessing import prepare_numeric_fold
from src.feature_transformer import (
    fit_feature_transformer,
    apply_feature_transformer,
)
from src.feature_selection import (
    run_feature_selection,
    apply_consensus_selection,
)
from src.models import build_models


def importance_f1(estimator, X, y, threshold):
    """F1 using the configured probability threshold."""
    probabilities = estimator.predict_proba(X)[
        :, list(estimator.classes_).index(1)
    ]
    return f1_score(
        y,
        probabilities >= threshold,
        zero_division=0,
    )


def importance_pr_auc(estimator, X, y):
    """Average precision, equivalent to our configured PR-AUC metric."""
    probabilities = estimator.predict_proba(X)[
        :, list(estimator.classes_).index(1)
    ]
    return average_precision_score(y, probabilities)


def main():
    config = load_config()

    settings = config["preprocessing"]["feature_importance"]
    fs_settings = config["preprocessing"]["feature_selection"]

    if config["modeling"]["task"] != "classification":
        raise ValueError("This analysis requires classification.")

    if not fs_settings["enabled"]:
        raise ValueError(
            "Enable feature_selection in config.yaml."
        )

    if config["validation"]["strategy"] != "expanding_window":
        raise ValueError(
            "Use expanding_window for the current comparison."
        )

    model_name = settings["model"]
    models = build_models(config)

    if model_name not in models:
        raise ValueError(
            f"Model {model_name} is not enabled."
        )

    n_repeats = settings["n_repeats"]
    top_n = settings["top_n"]
    random_state = config["random_state"]

    if n_repeats < 1 or top_n < 1:
        raise ValueError(
            "n_repeats and top_n must be positive."
        )

    threshold = config["modeling"]["evaluation"][
        "classification_threshold"
    ]

    target = config["columns"]["classification_target"]
    date_column = config["preprocessing"]["date_features"][
        "source_column"
    ]

    train, _, _ = load_data(config)
    train = prepare_rows(train, config)

    if max(config["validation"]["cv_years"]) >= config[
        "validation"
    ]["holdout_year"]:
        raise ValueError("CV must not include holdout year.")

    output_dir = (
        Path(config["paths"]["results"])
        / f"feature_importance_{datetime.now():%Y-%m-%d_%H-%M-%S-%f}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    importance_records = []
    fold_records = []

    for year, train_fold, val_fold in temporal_cv(
        train,
        config["validation"]["cv_years"],
        horizon_days=config["validation"]["target_horizon_days"],
        strategy=config["validation"]["strategy"],
        train_years=config["validation"]
            .get("sliding_window", {})
            .get("train_years"),
    ):
        print(f"\n=== FEATURE IMPORTANCE — FOLD {year} ===")

        y_train = train_fold[target]
        y_val = val_fold[target]

        if (
            y_train.isna().any()
            or y_val.isna().any()
            or y_train.nunique() != 2
            or y_val.nunique() != 2
        ):
            raise ValueError(f"Invalid target in fold {year}.")

        # Preprocessing: fit only on the training fold.
        prepared_train, prepared_val, _ = prepare_numeric_fold(
            train_fold,
            val_fold,
            config,
            task="classification",
        )

        transformer, columns = fit_feature_transformer(
            prepared_train,
            config,
            task="classification",
        )

        X_train = apply_feature_transformer(
            prepared_train,
            transformer,
            columns,
            config,
        )

        X_val = apply_feature_transformer(
            prepared_val,
            transformer,
            columns,
            config,
        )

        # Feature Selection: train fold only.
        selection = run_feature_selection(
            X_train,
            y_train,
            train_fold[date_column],
            config,
            raw_train_fold=train_fold,
        )

        X_train, X_val = apply_consensus_selection(
            X_train,
            X_val,
            selection,
            config,
        )

        print(
            f"Train: {len(X_train)} | "
            f"Validation: {len(X_val)} | "
            f"Selected features: {X_train.shape[1]}"
        )

        model = clone(models[model_name])
        model.fit(X_train, y_train)

        baseline_f1 = importance_f1(
            model, X_val, y_val, threshold
        )
        baseline_pr_auc = importance_pr_auc(
            model, X_val, y_val
        )

        print(
            f"Baseline F1={baseline_f1:.4f} | "
            f"PR-AUC={baseline_pr_auc:.4f}"
        )

        # Positive importance means permuting the feature
        # makes the model's validation score worse.
        importance = permutation_importance(
            model,
            X_val,
            y_val,
            scoring={
                "f1": lambda estimator, X, y: importance_f1(
                    estimator, X, y, threshold
                ),
                "pr_auc": importance_pr_auc,
            },
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=1,
        )

        native_importance = getattr(
            model, "feature_importances_", None
        )

        for index, feature in enumerate(X_val.columns):
            record = {
                "fold": year,
                "feature": feature,
                "f1_importance_mean": float(
                    importance["f1"].importances_mean[index]
                ),
                "f1_importance_std": float(
                    importance["f1"].importances_std[index]
                ),
                "pr_auc_importance_mean": float(
                    importance["pr_auc"].importances_mean[index]
                ),
                "pr_auc_importance_std": float(
                    importance["pr_auc"].importances_std[index]
                ),
            }

            if native_importance is not None:
                record["native_importance"] = float(
                    native_importance[index]
                )

            importance_records.append(record)

        fold_records.append({
            "fold": year,
            "model": model_name,
            "train_rows": len(X_train),
            "validation_rows": len(X_val),
            "selected_features": X_train.shape[1],
            "f1": baseline_f1,
            "pr_auc": baseline_pr_auc,
            "train_positive_rate": float(y_train.mean()),
            "val_positive_rate": float(y_val.mean()),
        })

        fold_importances = pd.DataFrame(
            record
            for record in importance_records
            if record["fold"] == year
        )

        print("\nTop features by permutation PR-AUC:")
        print(
            fold_importances.sort_values(
                "pr_auc_importance_mean",
                ascending=False,
            )[
                [
                    "feature",
                    "f1_importance_mean",
                    "pr_auc_importance_mean",
                    "pr_auc_importance_std",
                ]
            ]
            .head(top_n)
            .round(5)
            .to_string(index=False)
        )

    # Full feature importance per fold.
    importance_df = pd.DataFrame(importance_records)
    importance_df.to_csv(
        output_dir / "feature_importance_by_fold.csv",
        index=False,
    )

    # Fold performance, for comparison with existing CV results.
    folds_df = pd.DataFrame(fold_records)
    folds_df.to_csv(
        output_dir / "fold_performance.csv",
        index=False,
    )

    # Aggregate only across folds where the feature was selected.
    # Features absent from a fold are not assigned zero importance.
    stability = (
        importance_df
        .groupby("feature", as_index=False)
        .agg(
            folds_selected=("fold", "nunique"),
            mean_f1_importance=("f1_importance_mean", "mean"),
            mean_pr_auc_importance=(
                "pr_auc_importance_mean", "mean"
            ),
            std_pr_auc_across_folds=(
                "pr_auc_importance_mean", "std"
            ),
        )
        .sort_values(
            "mean_pr_auc_importance",
            ascending=False,
        )
    )

    stability.to_csv(
        output_dir / "feature_importance_stability.csv",
        index=False,
    )

    print("\n=== FEATURE IMPORTANCE SUMMARY ===")
    print(
        stability.head(top_n)
        .round(5)
        .to_string(index=False)
    )

    print("\n=== FOLD PERFORMANCE ===")
    print(folds_df.round(4).to_string(index=False))

    print(f"\nResults saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
