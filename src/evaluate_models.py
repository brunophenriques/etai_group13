
from pathlib import Path
from datetime import datetime
from pprint import pformat
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data_loader import load_config, load_data
from src.feature_transformer import (
    apply_feature_transformer,
    fit_feature_transformer,
)
from src.fold_preprocessing import prepare_numeric_fold
from src.models import build_models
from src.preprocessing import prepare_rows
from src.validation import temporal_cv


METRICS = {
    "pr_auc": average_precision_score,
    "roc_auc": roc_auc_score,
    "f1": f1_score,
    "precision": precision_score,
    "recall": recall_score,
}

PROBABILITY_METRICS = {"pr_auc", "roc_auc"}


def evaluate_predictions(
    y_true,
    probabilities,
    metric_names,
    threshold,
):
    """Calculate the configured classification metrics."""

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



def save_results(results_df, summary, config):
    """Save each evaluation in a timestamped experiment folder."""

    output_settings = config["modeling"]["output"]
    base_dir = Path(config["paths"]["results"])

    # Create a unique folder for each experiment.
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    experiment_dir = base_dir / timestamp
    experiment_dir.mkdir(parents=True, exist_ok=False)

    # Save detailed results as CSV.
    if output_settings["save_csv"]:
        csv_path = experiment_dir / "classification_cv.csv"
        results_df.to_csv(csv_path, index=False)

    # Save a readable report with results and configuration.
    if output_settings["save_txt"]:
        report_path = experiment_dir / "classification_summary.txt"

        with report_path.open("w", encoding="utf-8") as file:
            file.write("ETAI GROUP 13\n")
            file.write("CLASSIFICATION RESULTS\n")
            file.write("=" * 65 + "\n\n")

            # Experiment information
            file.write("EXPERIMENT INFORMATION\n")
            file.write("-" * 65 + "\n")
            file.write(f"Date: {timestamp}\n")
            file.write(
                f"Target: {config['columns']['classification_target']}\n"
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
                "Primary metric: "
                f"{config['modeling']['evaluation']['primary_metric']}\n"
            )
            file.write(
                "Classification threshold: "
                f"{config['modeling']['evaluation']['classification_threshold']}\n\n"
            )

            # Enabled models
            file.write("ACTIVE MODELS\n")
            file.write("-" * 65 + "\n")

            for name, settings in config["modeling"]["models"].items():
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

    print("\n=== EXPERIMENT SAVED ===")
    print(f"Experiment folder: {experiment_dir.resolve()}")

    if output_settings["save_csv"]:
        print(f"CSV: {csv_path.name}")

    if output_settings["save_txt"]:
        print(f"TXT: {report_path.name}")

    return experiment_dir



def run_evaluation():
    """Evaluate configured models using temporal CV."""

    config = load_config()

    if config["modeling"]["task"] != "classification":
        raise ValueError(
            "Regression evaluation is not yet implemented."
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

    if config["validation"]["strategy"] != "expanding_window":
        raise ValueError(
            "Only expanding_window validation is supported."
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


if __name__ == "__main__":
    run_evaluation()
