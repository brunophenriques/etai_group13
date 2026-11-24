import argparse
from pathlib import Path
from datetime import datetime
from pprint import pformat
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from src.data_loader import load_config, load_data
from src.feature_transformer import (
    apply_feature_transformer,
    fit_feature_transformer,
)
from src.fold_preprocessing import prepare_numeric_fold
from src.models import build_models, with_target_transform
from src.preprocessing import prepare_rows
from src.validation import temporal_cv

# Classification metrics
METRICS = {
    "pr_auc": average_precision_score,
    "roc_auc": roc_auc_score,
    "f1": f1_score,
    "precision": precision_score,
    "recall": recall_score,
}

PROBABILITY_METRICS = {"pr_auc", "roc_auc"}


def _rmse(y_true, predictions):
    return float(np.sqrt(mean_squared_error(y_true, predictions)))


def _pearson(y_true, predictions):
    """Pearson correlation between predictions and actual values.

    Undefined (NaN) when the predictions are constant, as in the dummy models.
    """
    if np.std(predictions) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, predictions)[0, 1])

# Regression metrics
REGRESSION_METRICS = {
    "mse": mean_squared_error,
    "rmse": _rmse,
    "mae": mean_absolute_error,
    "r2": r2_score,
    "pearson": _pearson,
}

# Metrics where a smaller value is better (used to sort the summary)
LOWER_IS_BETTER = {"mse", "rmse", "mae"}


def evaluate_predictions(
    y_true,
    probabilities,
    metric_names,
    threshold,
):
    """Calculate the configured classification metrics"""

    predictions = (probabilities >= threshold).astype(int)
    results = {}

    for name in metric_names:
        if name not in METRICS:
            raise ValueError(f"Unsupported metric: {name}")

        metric = METRICS[name]

        if name in PROBABILITY_METRICS:
            results[name] = metric(y_true, probabilities)
        else:
            results[name] = metric(
                y_true,
                predictions,
                zero_division=0,
            )

    return results


def evaluate_regression_predictions(y_true, predictions, metric_names):
    """Calculate the configured regression metrics, in euros"""

    predictions = np.maximum(predictions, 0)  # spend is never negative
    results = {}

    for name in metric_names:
        if name not in REGRESSION_METRICS:
            raise ValueError(f"Unsupported metric: {name}")

        results[name] = REGRESSION_METRICS[name](y_true, predictions)

    return results


def save_results(results_df, summary, config, task="classification"):
    """Save each evaluation in a timestamped experiment folder"""

    output_settings = config["modeling"]["output"]
    base_dir = Path(config["paths"]["results"])

    if task == "regression":
        evaluation = config["modeling"]["regression_evaluation"]
        model_settings = config["modeling"]["regression_models"]
    else:
        evaluation = config["modeling"]["evaluation"]
        model_settings = config["modeling"]["models"]

    # Create a unique folder for each experiment
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    experiment_dir = base_dir / timestamp
    experiment_dir.mkdir(parents=True, exist_ok=False)

    # Save detailed results as CSV
    if output_settings["save_csv"]:
        csv_path = experiment_dir / f"{task}_cv.csv"
        results_df.to_csv(csv_path, index=False)

    # Save a readable report with results and configuration
    if output_settings["save_txt"]:
        report_path = experiment_dir / f"{task}_summary.txt"

        with report_path.open("w", encoding="utf-8") as file:
            file.write("ETAI GROUP 13\n")
            file.write(f"{task.upper()} RESULTS\n")
            file.write("=" * 65 + "\n\n")

            # Experiment information
            file.write("EXPERIMENT INFORMATION\n")
            file.write("-" * 65 + "\n")
            file.write(f"Date: {timestamp}\n")
            file.write(
                f"Target: {config['columns'][f'{task}_target']}\n"
            )
            file.write(
                f"Validation: {config['validation']['strategy']}\n"
            )
            file.write(
                f"CV years: {config['validation']['cv_years']}\n"
            )
            file.write(
                "Target horizon: "
                f"{config['validation']['target_horizon_days']} days\n"
            )
            file.write(
                f"Primary metric: {evaluation['primary_metric']}\n"
            )

            if task == "classification":
                file.write(
                    "Classification threshold: "
                    f"{evaluation['classification_threshold']}\n"
                )

            file.write("\n")

            # Enabled models
            file.write("ACTIVE MODELS\n")
            file.write("-" * 65 + "\n")

            for name, settings in model_settings.items():
                if settings.get("enabled", False):
                    file.write(f"{name}\n")
                    file.write(
                        f"  Parameters: {settings.get('parameters', {})}\n"
                    )

            # Mean and standard deviation across folds
            file.write("\nCROSS-VALIDATION SUMMARY\n")
            file.write("-" * 65 + "\n")
            file.write(
                summary.round(4).to_string(index=False)
            )
            file.write("\n")

            # Individual fold results
            file.write("\nDETAILED RESULTS BY FOLD\n")
            file.write("-" * 65 + "\n")
            file.write(
                results_df.round(4).to_string(index=False)
            )
            file.write("\n")

            # Full configuration snapshot
            file.write("\nEXPERIMENT CONFIGURATION\n")
            file.write("-" * 65 + "\n")
            file.write(
                pformat(config, sort_dicts=False, width=100)
            )
            file.write("\n")

            # Methodological notes
            file.write("\nNOTES\n")
            file.write("-" * 65 + "\n")
            file.write(
                "Preprocessing is fitted separately on each training fold.\n"
                "Validation data is transformed without refitting learned "
                "imputers, encoders, or scalers.\n"
                "The 2024 holdout is excluded from cross-validation.\n"
                "Feature availability and leakage verification are not "
                "fully completed.\n"
            )

            if task == "regression":
                file.write(
                    "Only customers who returned and have a known spend are "
                    "used for training and validation.\n"
                    "Metrics are computed in euros (original scale), also for "
                    "models trained on log1p(spend).\n"
                )

    print("\n=== EXPERIMENT SAVED ===")
    print(f"Experiment folder: {experiment_dir.resolve()}")

    if output_settings["save_csv"]:
        print(f"CSV: {csv_path.name}")

    if output_settings["save_txt"]:
        print(f"TXT: {report_path.name}")

    return experiment_dir



def run_evaluation(task=None):
    """Entry point: evaluate the configured models using temporal CV

    The task comes from modeling.task in config.yaml, unless overridden by
    the `task` argument (or --task on the command line)

    - "regression": hands over to run_regression_evaluation
    - "classification": runs the classification evaluation below, which
      predicts whether a customer buys again within 90 days. It uses every
      row, predicts probabilities and reports threshold-based and
      probability-based metrics (F1, PR-AUC, ROC-AUC, ...)
    """

    config = load_config()

    if task is not None:
        config["modeling"]["task"] = task

    if config["modeling"]["task"] == "regression":
        return run_regression_evaluation(config)

    if config["modeling"]["task"] != "classification":
        raise ValueError(
            "modeling.task must be 'classification' or 'regression'."
        )

    models = build_models(config)

    evaluation = config["modeling"]["evaluation"]
    metric_names = evaluation["metrics"]
    primary_metric = evaluation["primary_metric"]
    threshold = evaluation["classification_threshold"]

    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    if not metric_names or primary_metric not in metric_names:
        raise ValueError(
            "Primary metric must be in evaluation.metrics."
        )

    if config["validation"]["strategy"] not in {
        "expanding_window",
        "sliding_window",
    }:
        raise ValueError(
            "Unsupported temporal validation strategy."
        )

    train, _, _ = load_data(config)
    train = prepare_rows(train, config)

    target = config["columns"]["classification_target"]

    years = config["validation"]["cv_years"]
    horizon = config["validation"]["target_horizon_days"]

    results = []


    for year, train_fold, val_fold in temporal_cv(
        train,
        years,
        horizon_days=horizon,
        strategy=config["validation"]["strategy"],
        train_years=config["validation"]
            .get("sliding_window", {})
            .get("train_years"),
    ):

        y_train = train_fold[target]
        y_val = val_fold[target]

        for name, labels in [
            ("train", y_train),
            ("validation", y_val),
        ]:
            if (
                labels.isna().any()
                or not labels.isin([0, 1]).all()
                or labels.nunique() != 2
            ):
                raise ValueError(
                    f"Invalid {name} labels in fold {year}"
                )

        prepared_train, prepared_val, _ = (
            prepare_numeric_fold(
                train_fold,
                val_fold,
                config,
            )
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

        print(f"\n=== FOLD {year} ===")
        print(
            f"Train: {len(X_train)} | "
            f"Validation: {len(X_val)} | "
            f"Features: {X_train.shape[1]}"
        )
        print(
            f"Positive rate: train={y_train.mean():.4f}, "
            f"validation={y_val.mean():.4f}"
        )

        for name, base_model in models.items():
            model = clone(base_model)
            model.fit(X_train, y_train)

            positive_class_index = (
                list(model.classes_).index(1)
            )

            probabilities = model.predict_proba(X_val)[
                :, positive_class_index
            ]

            scores = evaluate_predictions(
                y_val,
                probabilities,
                metric_names,
                threshold,
            )

            results.append({
                "fold": year,
                "model": name,
                "train_rows": len(X_train),
                "validation_rows": len(X_val),
                "features": X_train.shape[1],
                "train_positive_rate": y_train.mean(),
                "val_positive_rate": y_val.mean(),
                **scores,
            })

            print(
                f"{name}: "
                + " | ".join(
                    f"{metric}={scores[metric]:.4f}"
                    for metric in metric_names
                )
            )

    results_df = pd.DataFrame(results)

    summary = (
        results_df.groupby("model")[metric_names]
        .agg(["mean", "std"])
    )

    summary.columns = [
        f"{metric}_{stat}"
        for metric, stat in summary.columns
    ]

    summary = summary.reset_index().sort_values(
        f"{primary_metric}_mean",
        ascending=False,
    )

    print("\n=== CROSS-VALIDATION SUMMARY ===")
    print(summary.round(4).to_string(index=False))

    save_results(results_df, summary.round(4), config)

    print("\nResults saved in:", config["paths"]["results"])

    return results_df, summary


def run_regression_evaluation(config):
    """Evaluate regression models using temporal CV (called by run_evaluation)

    Predicts how much a customer spends (EUR) in the 90 days after the
    current purchase. Differences from the classification evaluation:

    - Only customers who returned and have a known spend are used, both
      for training and for validation
    - Models predict spend in euros ("euros") or log1p(spend) converted
      back to euros ("log"), as set in regression_evaluation.target_transforms
    - Metrics (MSE, RMSE, MAE, R2, Pearson correlation) are always
      computed in euros; there is no probability or threshold
    - The summary is sorted so that the best model comes first
    """

    evaluation = config["modeling"]["regression_evaluation"]
    metric_names = evaluation["metrics"]
    primary_metric = evaluation["primary_metric"]
    target_forms = evaluation["target_transforms"]

    if not metric_names or primary_metric not in metric_names:
        raise ValueError(
            "Primary metric must be in regression_evaluation.metrics."
        )

    if config["validation"]["strategy"] not in {
        "expanding_window",
        "sliding_window",
    }:
        raise ValueError(
            "Unsupported temporal validation strategy."
        )

    models = build_models(config)

    train, _, _ = load_data(config)
    train = prepare_rows(train, config)

    classification_target = config["columns"]["classification_target"]
    target = config["columns"]["regression_target"]

    results = []

    for year, train_fold, val_fold in temporal_cv(
        train,
        config["validation"]["cv_years"],
        horizon_days=config["validation"]["target_horizon_days"],
        strategy=config["validation"]["strategy"],
        train_years=config["validation"]
            .get("sliding_window", {})
            .get("train_years"),
    ):

        # Imputation is fitted on the whole training fold (targets are not used)
        prepared_train, prepared_val, _ = (
            prepare_numeric_fold(
                train_fold,
                val_fold,
                config,
            )
        )

        # Regression uses only customers who returned and have a known spend
        train_mask = (
            (train_fold[classification_target] == 1)
            & train_fold[target].notna()
        ).to_numpy()
        val_mask = (
            (val_fold[classification_target] == 1)
            & val_fold[target].notna()
        ).to_numpy()

        prepared_train = prepared_train[train_mask]
        prepared_val = prepared_val[val_mask]
        y_train = train_fold.loc[train_mask, target]
        y_val = val_fold.loc[val_mask, target]

        transformer, columns = fit_feature_transformer(
            prepared_train,
            config,
            task="regression",
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

        print(f"\n=== FOLD {year} ===")
        print(
            f"Train: {len(X_train)} | "
            f"Validation: {len(X_val)} | "
            f"Features: {X_train.shape[1]}"
        )
        print(
            f"Mean spend: train={y_train.mean():.0f}, "
            f"validation={y_val.mean():.0f} EUR"
        )

        for name, base_model in models.items():
            for form in target_forms:
                # A constant reference makes no sense on the log scale
                if form == "log" and name.startswith("dummy"):
                    continue

                model = with_target_transform(clone(base_model), form)
                model.fit(X_train, y_train)

                scores = evaluate_regression_predictions(
                    y_val,
                    model.predict(X_val),
                    metric_names,
                )

                results.append({
                    "fold": year,
                    "model": name,
                    "target_form": form,
                    "train_rows": len(X_train),
                    "validation_rows": len(X_val),
                    "features": X_train.shape[1],
                    **scores,
                })

                print(
                    f"{name} ({form}): "
                    + " | ".join(
                        f"{metric}={scores[metric]:.4f}"
                        for metric in metric_names
                    )
                )

    results_df = pd.DataFrame(results)

    summary = (
        results_df.groupby(["model", "target_form"])[metric_names]
        .agg(["mean", "std"])
    )

    summary.columns = [
        f"{metric}_{stat}"
        for metric, stat in summary.columns
    ]

    summary = summary.reset_index().sort_values(
        f"{primary_metric}_mean",
        ascending=primary_metric in LOWER_IS_BETTER,
    )

    print("\n=== CROSS-VALIDATION SUMMARY ===")
    print(summary.round(4).to_string(index=False))

    save_results(results_df, summary.round(4), config, task="regression")

    print("\nResults saved in:", config["paths"]["results"])

    return results_df, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate models with temporal CV.")
    parser.add_argument(
        "--task",
        choices=["classification", "regression"],
        help="Overrides modeling.task from config.yaml.",
    )
    run_evaluation(parser.parse_args().task)
