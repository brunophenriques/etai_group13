
"""Diagnose RFECV instability across temporal folds — ETAI Group 13."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV

from src.data_loader import load_config, load_data
from src.preprocessing import prepare_rows
from src.fold_preprocessing import prepare_numeric_fold
from src.feature_transformer import (
    fit_feature_transformer,
    apply_feature_transformer,
)
from src.validation import temporal_cv
from src.feature_selection import temporal_inner_splits


def main():
    config = load_config()

    settings = config["preprocessing"]["feature_selection"]["methods"]["rfecv"]
    target = config["columns"]["classification_target"]
    date_column = config["preprocessing"]["date_features"]["source_column"]

    train, _, _ = load_data(config)
    train = prepare_rows(train, config)

    output_dir = (
        Path(config["paths"]["results"])
        / f"rfecv_diagnostics_{datetime.now():%Y-%m-%d_%H-%M-%S}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    summary = []

    for year, train_fold, val_fold in temporal_cv(
        train,
        config["validation"]["cv_years"],
        horizon_days=config["validation"]["target_horizon_days"],
        strategy=config["validation"]["strategy"],
        train_years=config["validation"]
            .get("sliding_window", {})
            .get("train_years"),
    ):
        print(f"\n{'=' * 60}")
        print(f"RFECV DIAGNOSTICS — FOLD {year}")
        print("=" * 60)

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

        X = apply_feature_transformer(
            prepared_train,
            transformer,
            columns,
            config,
        )

        y = pd.Series(
            train_fold[target].to_numpy(),
        )

        dates = pd.to_datetime(
            train_fold[date_column]
        ).reset_index(drop=True)

        # Sort positionally to avoid index-alignment problems.
        order = np.argsort(dates.to_numpy(), kind="stable")

        X = X.iloc[order].reset_index(drop=True)
        y = y.iloc[order].reset_index(drop=True)
        dates = dates.iloc[order].reset_index(drop=True)

        splits = temporal_inner_splits(
            dates,
            config,
            settings,
        )

        print(f"Training rows: {len(X)}")
        print(f"Initial features: {X.shape[1]}")
        print(f"Valid inner splits: {len(splits)}")

        horizon = pd.Timedelta(
            days=config["validation"]["target_horizon_days"]
        )

        for i, (train_idx, val_idx) in enumerate(splits, start=1):
            latest_train = dates.iloc[train_idx].max()
            first_validation = dates.iloc[val_idx].min()

            label_available = latest_train + horizon

            if not label_available < first_validation:
                raise AssertionError(
                    f"Temporal leakage in inner split {i}"
                )

            if (
                y.iloc[train_idx].nunique() != 2
                or y.iloc[val_idx].nunique() != 2
            ):
                raise ValueError(
                    f"Only one target class in inner split {i}"
                )

            print(
                f"  Split {i}: "
                f"train={len(train_idx)}, "
                f"val={len(val_idx)}, "
                f"last_train={latest_train.date()}, "
                f"first_val={first_validation.date()}"
            )

        estimator = RandomForestClassifier(
            n_estimators=settings["n_estimators"],
            min_samples_leaf=settings["min_samples_leaf"],
            random_state=config["random_state"],
            n_jobs=-1,
        )

        selector = RFECV(
            estimator=estimator,
            step=settings["step"],
            min_features_to_select=settings["min_features_to_select"],
            cv=splits,
            scoring=settings["scoring"],
            n_jobs=1,
        )

        selector.fit(X, y)

        # Keep only scalar metrics and the number of features.
# RFECV may also return 2D ranking/support arrays.
        results = pd.DataFrame({
            key: values
            for key, values in selector.cv_results_.items()
            if np.asarray(values).ndim == 1
        })

        if "n_features" not in results.columns:
            raise ValueError(
                "Your scikit-learn version does not expose "
                "RFECV n_features in cv_results_."
            )

        results["fold"] = year
        results["score_std"] = results["std_test_score"]
        results["score_mean"] = results["mean_test_score"]

        results.to_csv(
            output_dir / f"rfecv_scores_{year}.csv",
            index=False,
        )

        best_score = results["score_mean"].max()

        # Find configurations practically tied with the best score.
        near_best = results.loc[
            results["score_mean"] >= best_score - 0.005
        ]

        print(f"\nSelected features: {selector.n_features_}")
        print(f"Best inner F1: {best_score:.4f}")

        print("\nTop 10 feature counts:")
        print(
            results.sort_values(
                "score_mean", ascending=False
            )[
                ["n_features", "score_mean", "score_std"]
            ]
            .head(10)
            .round(4)
            .to_string(index=False)
        )

        print(
            "\nFeature counts within 0.005 F1 of best:",
            near_best["n_features"].tolist(),
        )

        summary.append({
            "fold": year,
            "initial_features": X.shape[1],
            "selected_features": selector.n_features_,
            "best_inner_f1": best_score,
            "near_best_min_features": near_best["n_features"].min(),
            "near_best_max_features": near_best["n_features"].max(),
            "inner_splits": len(splits),
        })

    summary_df = pd.DataFrame(summary)

    summary_df.to_csv(
        output_dir / "rfecv_diagnostics_summary.csv",
        index=False,
    )

    print("\n=== FINAL RFECV DIAGNOSTICS ===")
    print(summary_df.round(4).to_string(index=False))
    print(f"\nResults saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
