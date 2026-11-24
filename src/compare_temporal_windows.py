
"""Compare expanding and sliding temporal training windows."""

from datetime import datetime
from pathlib import Path

import pandas as pd

from sklearn.base import clone
from sklearn.metrics import brier_score_loss

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
from src.evaluate_models import evaluate_predictions


def evaluate_strategy(data, config, strategy, model_name):
    """Evaluate one strategy, fitting everything inside each train fold."""

    validation = config["validation"]
    target = config["columns"]["classification_target"]
    date_column = config["preprocessing"]["date_features"][
        "source_column"
    ]

    model_template = build_models(config)[model_name]
    evaluation = config["modeling"]["evaluation"]

    results = []
    validation_indices = {}

    for year, train_fold, val_fold in temporal_cv(
        data,
        validation["cv_years"],
        horizon_days=validation["target_horizon_days"],
        strategy=strategy,
        train_years=validation["sliding_window"]["train_years"],
    ):
        y_train = train_fold[target]
        y_val = val_fold[target]

        if y_train.nunique() != 2 or y_val.nunique() != 2:
            raise ValueError(
                f"Missing target class in {strategy}, fold {year}."
            )

        validation_indices[year] = set(val_fold.index)

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
            prepared_train, transformer, columns, config
        )
        X_val = apply_feature_transformer(
            prepared_val, transformer, columns, config
        )

        if config["preprocessing"]["feature_selection"]["enabled"]:
            selection = run_feature_selection(
                X_train,
                y_train,
                train_fold[date_column],
                config,
                raw_train_fold=train_fold,
            )

            X_train, X_val = apply_consensus_selection(
                X_train, X_val, selection, config
            )

        model = clone(model_template)
        model.fit(X_train, y_train)

        positive_index = list(model.classes_).index(1)
        probabilities = model.predict_proba(X_val)[
            :, positive_index
        ]

        scores = evaluate_predictions(
            y_val,
            probabilities,
            evaluation["metrics"],
            evaluation["classification_threshold"],
        )

        results.append({
            "strategy": strategy,
            "fold": year,
            "train_rows": len(train_fold),
            "validation_rows": len(val_fold),
            "train_start": pd.to_datetime(
                train_fold[date_column]
            ).min().date(),
            "train_end": pd.to_datetime(
                train_fold[date_column]
            ).max().date(),
            "selected_features": X_train.shape[1],
            "train_positive_rate": float(y_train.mean()),
            "validation_positive_rate": float(y_val.mean()),
            **scores,
            "brier_score": brier_score_loss(
                y_val, probabilities
            ),
            "calibration_gap": float(
                probabilities.mean() - y_val.mean()
            ),
        })

        print(
            f"{strategy} | {year} | "
            f"Train={len(train_fold)} | "
            f"Features={X_train.shape[1]} | "
            f"F1={scores['f1']:.4f} | "
            f"ROC-AUC={scores['roc_auc']:.4f} | "
            f"PR-AUC={scores['pr_auc']:.4f}"
        )

    return pd.DataFrame(results), validation_indices


def main():
    config = load_config()

    if config["modeling"]["task"] != "classification":
        raise ValueError("Classification task required.")

    if any(
        year >= config["validation"]["holdout_year"]
        for year in config["validation"]["cv_years"]
    ):
        raise ValueError("CV cannot include holdout year.")

    if not config["preprocessing"]["feature_selection"]["enabled"]:
        raise ValueError("Enable Feature Selection for this comparison.")

    model_name = config["preprocessing"][
        "feature_importance"
    ]["model"]

    if model_name not in build_models(config):
        raise ValueError(f"Model {model_name} is not enabled.")

    data, _, _ = load_data(config)
    data = prepare_rows(data, config)

    all_results = []
    reference_indices = None

    for strategy in ["expanding_window", "sliding_window"]:
        print(f"\n=== {strategy.upper()} ===")

        result, indices = evaluate_strategy(
            data, config, strategy, model_name
        )

        if reference_indices is None:
            reference_indices = indices
        elif indices != reference_indices:
            raise AssertionError(
                "Strategies have different validation rows."
            )

        all_results.append(result)

    results = pd.concat(all_results, ignore_index=True)

    metrics = [
        "f1",
        "precision",
        "recall",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "calibration_gap",
    ]

    summary = (
        results.groupby("strategy")[metrics]
        .mean()
        .reset_index()
    )

    expanding = results[
        results["strategy"] == "expanding_window"
    ].set_index("fold")

    sliding = results[
        results["strategy"] == "sliding_window"
    ].set_index("fold")

    comparison = pd.DataFrame({
        "fold": expanding.index,
        "expanding_train_rows": expanding["train_rows"].values,
        "sliding_train_rows": sliding["train_rows"].values,
        "expanding_features": expanding["selected_features"].values,
        "sliding_features": sliding["selected_features"].values,
    })

    for metric in metrics:
        comparison[f"expanding_{metric}"] = (
            expanding[metric].values
        )
        comparison[f"sliding_{metric}"] = (
            sliding[metric].values
        )
        comparison[f"delta_{metric}"] = (
            sliding[metric].values - expanding[metric].values
        )

    output_dir = (
        Path(config["paths"]["results"])
        / f"temporal_window_comparison_"
          f"{datetime.now():%Y-%m-%d_%H-%M-%S-%f}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    results.to_csv(
        output_dir / "strategy_results.csv",
        index=False,
    )
    summary.to_csv(
        output_dir / "strategy_summary.csv",
        index=False,
    )
    comparison.to_csv(
        output_dir / "fold_comparison.csv",
        index=False,
    )

    print("\n=== STRATEGY SUMMARY ===")
    print(summary.round(4).to_string(index=False))

    print("\n=== F1 DIFFERENCE (SLIDING - EXPANDING) ===")
    print(
        comparison[
            ["fold", "expanding_f1", "sliding_f1", "delta_f1"]
        ].round(4).to_string(index=False)
    )

    print(f"\nResults saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
