
"""Run feature selection experiments across temporal folds."""

from datetime import datetime
from pathlib import Path

import pandas as pd

from src.data_loader import load_config, load_data
from src.preprocessing import prepare_rows
from src.fold_preprocessing import prepare_numeric_fold
from src.feature_transformer import (
    fit_feature_transformer,
    apply_feature_transformer,
)
from src.validation import temporal_cv
from src.feature_selection import run_feature_selection


def main():
    config = load_config()

    settings = config["preprocessing"]["feature_selection"]

    if not settings["consensus"]["enabled"]:
        raise ValueError(
            "Enable feature_selection.consensus in config.yaml."
        )

    target = config["columns"]["classification_target"]
    date_column = config["preprocessing"]["date_features"]["source_column"]

    # Number of active selection methods, read from config.yaml.
    active_methods = sum(
        method["enabled"]
        for method in settings["methods"].values()
    )

    if active_methods == 0:
        raise ValueError("No feature selection methods enabled.")

    train, _, _ = load_data(config)
    train = prepare_rows(train, config)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")

    output_dir = (
        Path(config["paths"]["results"])
        / f"feature_selection_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    fold_results = []

    for year, train_fold, val_fold in temporal_cv(
        train,
        config["validation"]["cv_years"],
        horizon_days=config["validation"]["target_horizon_days"],
        strategy=config["validation"]["strategy"],
        train_years=config["validation"]
            .get("sliding_window", {})
            .get("train_years"),
    ):
        print(f"\n=== FEATURE SELECTION — FOLD {year} ===")

        # Fit preprocessing on training fold only.
        prepared_train, _, _ = prepare_numeric_fold(
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

        y_train = train_fold[target]
        dates = train_fold[date_column]

        if y_train.isna().any() or y_train.nunique() != 2:
            raise ValueError(f"Invalid target in fold {year}.")

        print(
            f"Train: {len(X_train)} rows | "
            f"Features: {X_train.shape[1]}"
        )

        selection = run_feature_selection(
            X_train,
            y_train,
            dates,
            config,
        )

        # Every feature must receive exactly one decision
        # per active selection method.
        total_votes = (
            selection["selected_votes"]
            + selection["rejected_votes"]
            + selection["tentative_votes"]
        )

        if not total_votes.eq(active_methods).all():
            raise ValueError(
                f"Inconsistent selection votes in fold {year}. "
                f"Expected {active_methods} votes per feature."
            )

        selection.insert(0, "fold", year)

        selection.to_csv(
            output_dir / f"selection_{year}.csv",
            index=False,
        )

        fold_results.append(selection)

        print(
            "Removal candidates:",
            int(selection["remove_candidate"].sum()),
        )

    # Combine all temporal folds.
    all_results = pd.concat(
        fold_results,
        ignore_index=True,
    )

    all_results.to_csv(
        output_dir / "selection_all_folds.csv",
        index=False,
    )

    # Analyse stability across years.
    # A feature absent from a fold is not treated as rejected.
    stability = (
        all_results
        .groupby("feature", as_index=False)
        .agg(
            folds_present=("fold", "nunique"),
            mean_selected_votes=("selected_votes", "mean"),
            mean_rejected_votes=("rejected_votes", "mean"),
            folds_rejected_by_consensus=(
                "remove_candidate", "sum"
            ),
        )
    )

    stability = stability.sort_values(
        [
            "folds_rejected_by_consensus",
            "mean_rejected_votes",
        ],
        ascending=False,
    )

    stability.to_csv(
        output_dir / "feature_stability.csv",
        index=False,
    )

    print("\n=== FEATURE SELECTION SUMMARY ===")
    print(
        stability.head(30).round(2).to_string(index=False)
    )

    print(f"\nResults saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
