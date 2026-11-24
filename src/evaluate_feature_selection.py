
"""Compare consensus feature selection across classification models."""

from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.base import clone

from src.data_loader import load_config, load_data
from src.preprocessing import prepare_rows
from src.fold_preprocessing import prepare_numeric_fold
from src.feature_transformer import (
    fit_feature_transformer,
    apply_feature_transformer,
)
from src.feature_selection import run_feature_selection
from src.validation import temporal_cv
from src.models import build_models
from src.evaluate_models import evaluate_predictions


def main():
    config = load_config()

    settings = config["preprocessing"]["feature_selection"]
    consensus = settings["consensus"]

    if not consensus["enabled"]:
        raise ValueError("Feature selection consensus is disabled.")

    if config["validation"]["strategy"] != "expanding_window":
        raise ValueError(
            "Use expanding_window for comparison with the baseline."
        )

    active_methods = sum(
        method["enabled"] for method in settings["methods"].values()
    )

    thresholds = consensus["evaluation_thresholds"]

    if len(set(thresholds)) != len(thresholds):
        raise ValueError("Duplicate consensus thresholds.")

    for threshold in thresholds:
        if not 1 <= threshold <= active_methods:
            raise ValueError(
                f"Invalid consensus threshold: {threshold}"
            )

    available_models = build_models(config)
    model_names = consensus["evaluation_models"]

    if not model_names or len(set(model_names)) != len(model_names):
        raise ValueError("Configure unique evaluation models.")

    unknown_models = set(model_names) - set(available_models)

    if unknown_models:
        raise ValueError(
            f"Models not enabled or not found: {unknown_models}"
        )

    target = config["columns"]["classification_target"]
    date_column = config["preprocessing"]["date_features"]["source_column"]

    evaluation = config["modeling"]["evaluation"]
    metric_names = evaluation["metrics"]
    classification_threshold = evaluation["classification_threshold"]

    train, _, _ = load_data(config)
    train = prepare_rows(train, config)

    output_dir = (
        Path(config["paths"]["results"])
        / f"consensus_models_{datetime.now():%Y-%m-%d_%H-%M-%S-%f}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    scores = []
    removed_records = []
    selection_records = []

    for year, train_fold, val_fold in temporal_cv(
        train,
        config["validation"]["cv_years"],
        horizon_days=config["validation"]["target_horizon_days"],
        strategy=config["validation"]["strategy"],
        train_years=config["validation"]
            .get("sliding_window", {})
            .get("train_years"),
    ):
        print(f"\n=== CONSENSUS MODELS — FOLD {year} ===")

        y_train = train_fold[target]
        y_val = val_fold[target]

        if (
            y_train.isna().any()
            or y_val.isna().any()
            or y_train.nunique() != 2
            or y_val.nunique() != 2
        ):
            raise ValueError(f"Invalid target in fold {year}.")

        # Preprocessing fitted on outer training fold only.
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

        # Compute the four selectors only once per fold.
        selection = run_feature_selection(
            X_train,
            y_train,
            train_fold[date_column],
            config,
            raw_train_fold=train_fold,
        )

        selection.insert(0, "fold", year)
        selection_records.append(selection)

        votes = selection.set_index("feature")["rejected_votes"]

        if (
            not votes.index.is_unique
            or set(votes.index) != set(X_train.columns)
        ):
            raise ValueError(
                f"Feature selection columns mismatch in fold {year}."
            )

        # Baseline and every threshold configured in YAML.
        strategies = {
            "baseline": X_train.columns.tolist()
        }

        for rejection_threshold in thresholds:
            strategy_name = (
                f"consensus_{rejection_threshold}_of_{active_methods}"
            )

            strategies[strategy_name] = [
                column
                for column in X_train.columns
                if votes.loc[column] < rejection_threshold
            ]

        for strategy, selected_columns in strategies.items():
            if not selected_columns:
                raise ValueError(
                    f"No features remaining: {strategy}, fold {year}."
                )

            selected_set = set(selected_columns)

            removed_columns = [
                column
                for column in X_train.columns
                if column not in selected_set
            ]

            for feature in removed_columns:
                removed_records.append({
                    "fold": year,
                    "strategy": strategy,
                    "feature": feature,
                    "rejected_votes": int(votes.loc[feature]),
                })

            # Prepare the same columns for every model.
            X_train_selected = X_train[selected_columns]
            X_val_selected = X_val[selected_columns]

            for model_name in model_names:
                model = clone(available_models[model_name])

                model.fit(
                    X_train_selected,
                    y_train,
                )

                probabilities = model.predict_proba(
                    X_val_selected
                )[:, 1]

                metrics = evaluate_predictions(
                    y_val,
                    probabilities,
                    metric_names,
                    classification_threshold,
                )

                scores.append({
                    "fold": year,
                    "model": model_name,
                    "strategy": strategy,
                    "train_rows": len(X_train),
                    "validation_rows": len(X_val),
                    "initial_features": X_train.shape[1],
                    "selected_features": len(selected_columns),
                    "removed_features": len(removed_columns),
                    **metrics,
                })

                print(
                    f"{model_name} | {strategy} | "
                    f"features={len(selected_columns)} | "
                    f"f1={metrics['f1']:.4f} | "
                    f"pr_auc={metrics['pr_auc']:.4f}"
                )

    results_df = pd.DataFrame(scores)

    results_df.to_csv(
        output_dir / "consensus_models_cv.csv",
        index=False,
    )

    summary = (
        results_df
        .groupby(["model", "strategy"], as_index=False)
        .agg(
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            features_mean=("selected_features", "mean"),
            removed_mean=("removed_features", "mean"),
        )
    )

    # Compare each model to its OWN full-feature baseline.
    baseline = (
        summary.loc[
            summary["strategy"] == "baseline",
            ["model", "f1_mean", "pr_auc_mean"],
        ]
        .rename(columns={
            "f1_mean": "baseline_f1",
            "pr_auc_mean": "baseline_pr_auc",
        })
    )

    summary = summary.merge(
        baseline,
        on="model",
        validate="many_to_one",
    )

    summary["f1_delta_vs_baseline"] = (
        summary["f1_mean"] - summary["baseline_f1"]
    )

    summary["pr_auc_delta_vs_baseline"] = (
        summary["pr_auc_mean"] - summary["baseline_pr_auc"]
    )

    summary = summary.drop(
        columns=["baseline_f1", "baseline_pr_auc"]
    ).sort_values(
        ["model", "f1_mean"],
        ascending=[True, False],
    )

    summary.to_csv(
        output_dir / "consensus_models_summary.csv",
        index=False,
    )

    pd.DataFrame(
        removed_records,
        columns=[
            "fold",
            "strategy",
            "feature",
            "rejected_votes",
        ],
    ).to_csv(
        output_dir / "removed_features.csv",
        index=False,
    )

    pd.concat(
        selection_records,
        ignore_index=True,
    ).to_csv(
        output_dir / "selection_votes.csv",
        index=False,
    )

    print("\n=== CONSENSUS MODELS SUMMARY ===")
    print(summary.round(4).to_string(index=False))

    print(f"\nResults saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
