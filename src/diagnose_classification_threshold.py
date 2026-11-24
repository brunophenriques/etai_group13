
"""Diagnose classification probabilities and thresholds by temporal fold."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

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


def classification_metrics(y, probabilities, threshold):
    """Evaluate probability and classification quality."""
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, predictions, labels=[0, 1]
    ).ravel()

    return {
        "threshold": threshold,
        "f1": f1_score(y, predictions, zero_division=0),
        "precision": precision_score(
            y, predictions, zero_division=0
        ),
        "recall": recall_score(y, predictions, zero_division=0),
        "predicted_positive_rate": predictions.mean(),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def probability_calibration(y, probabilities, n_bins):
    """Observed positive rate versus predicted probability."""
    edges = np.linspace(0, 1, n_bins + 1)

    # Rightmost edge includes probability 1.0.
    indices = np.minimum(
        np.searchsorted(edges, probabilities, side="right") - 1,
        n_bins - 1,
    )

    records = []

    for bin_index in range(n_bins):
        mask = indices == bin_index

        if not mask.any():
            continue

        records.append({
            "bin": bin_index + 1,
            "bin_lower": edges[bin_index],
            "bin_upper": edges[bin_index + 1],
            "count": int(mask.sum()),
            "mean_predicted_probability": float(
                probabilities[mask].mean()
            ),
            "observed_positive_rate": float(y[mask].mean()),
        })

    return pd.DataFrame(records)


def main():
    config = load_config()

    if config["modeling"]["task"] != "classification":
        raise ValueError("Requires classification task.")

    settings = config["preprocessing"]["threshold_diagnostics"]
    model_name = settings["model"]
    thresholds = settings["thresholds"]
    n_bins = settings["probability_bins"]

    if not thresholds or any(
        not 0 <= threshold <= 1 for threshold in thresholds
    ):
        raise ValueError("Thresholds must be between 0 and 1.")

    if not isinstance(n_bins, int) or n_bins < 2:
        raise ValueError("probability_bins must be >= 2.")

    reference_threshold = config["modeling"]["evaluation"][
        "classification_threshold"
    ]

    models = build_models(config)

    if model_name not in models:
        raise ValueError(f"Model {model_name} is not enabled.")

    target = config["columns"]["classification_target"]
    date_column = config["preprocessing"]["date_features"][
        "source_column"
    ]

    if any(
        year >= config["validation"]["holdout_year"]
        for year in config["validation"]["cv_years"]
    ):
        raise ValueError("Holdout year cannot enter diagnostics.")

    train, _, _ = load_data(config)
    train = prepare_rows(train, config)

    output_dir = (
        Path(config["paths"]["results"])
        / f"threshold_diagnostics_{datetime.now():%Y-%m-%d_%H-%M-%S-%f}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    fold_rows = []
    threshold_rows = []
    calibration_rows = []
    prediction_rows = []

    for year, train_fold, val_fold in temporal_cv(
        train,
        config["validation"]["cv_years"],
        horizon_days=config["validation"]["target_horizon_days"],
        strategy=config["validation"]["strategy"],
        train_years=config["validation"]
            .get("sliding_window", {})
            .get("train_years"),
    ):
        print(f"\n=== FOLD {year} ===")

        y_train = train_fold[target]
        y_val = val_fold[target]

        if (
            not y_train.isin([0, 1]).all()
            or not y_val.isin([0, 1]).all()
            or y_train.nunique() != 2
            or y_val.nunique() != 2
        ):
            raise ValueError(f"Invalid labels in fold {year}")

        prepared_train, prepared_val, _ = prepare_numeric_fold(
            train_fold,
            val_fold,
            config,
            task="classification",
        )

        transformer, columns = fit_feature_transformer(
            prepared_train, config, task="classification"
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

        model = clone(models[model_name])
        model.fit(X_train, y_train)

        positive_index = list(model.classes_).index(1)
        probabilities = np.asarray(
            model.predict_proba(X_val)[:, positive_index],
            dtype=float,
        )

        labels = y_val.to_numpy(dtype=int)

        if (
            not np.isfinite(probabilities).all()
            or ((probabilities < 0) | (probabilities > 1)).any()
        ):
            raise ValueError(f"Invalid probabilities in fold {year}")

        reference = classification_metrics(
            labels, probabilities, reference_threshold
        )

        roc_auc = roc_auc_score(labels, probabilities)
        pr_auc = average_precision_score(labels, probabilities)
        brier = brier_score_loss(labels, probabilities)

        fold_rows.append({
            "fold": year,
            "train_rows": len(y_train),
            "validation_rows": len(y_val),
            "features": X_val.shape[1],
            "train_positive_rate": y_train.mean(),
            "validation_positive_rate": y_val.mean(),
            "reference_threshold": reference_threshold,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "pr_auc_baseline": y_val.mean(),
            "brier_score": brier,
            "mean_predicted_probability": probabilities.mean(),
            **{
                f"reference_{key}": value
                for key, value in reference.items()
                if key != "threshold"
            },
        })

        for threshold in sorted(
            set([reference_threshold, *thresholds])
        ):
            threshold_rows.append({
                "fold": year,
                **classification_metrics(
                    labels, probabilities, threshold
                ),
            })

        calibration = probability_calibration(
            labels, probabilities, n_bins
        )
        calibration.insert(0, "fold", year)
        calibration_rows.append(calibration)

        # Store validation predictions for later diagnostics.
        prediction_rows.append(pd.DataFrame({
            "fold": year,
            "date": pd.to_datetime(
                val_fold[date_column]
            ).to_numpy(),
            "actual": labels,
            "probability": probabilities,
            "prediction_at_reference": (
                probabilities >= reference_threshold
            ).astype(int),
        }))

        print(
            f"F1={reference['f1']:.4f} | "
            f"PR-AUC={pr_auc:.4f} | "
            f"ROC-AUC={roc_auc:.4f} | "
            f"Brier={brier:.4f}"
        )
        print(
            f"Actual positive rate={labels.mean():.2%} | "
            f"Predicted positive rate="
            f"{reference['predicted_positive_rate']:.2%}"
        )
        print(
            f"Mean probability={probabilities.mean():.4f}"
        )
        print(
            f"Confusion matrix: "
            f"TN={reference['tn']}, FP={reference['fp']}, "
            f"FN={reference['fn']}, TP={reference['tp']}"
        )

    folds_df = pd.DataFrame(fold_rows)
    thresholds_df = pd.DataFrame(threshold_rows)
    calibration_df = pd.concat(
        calibration_rows, ignore_index=True
    )
    predictions_df = pd.concat(
        prediction_rows, ignore_index=True
    )

    folds_df.to_csv(output_dir / "fold_metrics.csv", index=False)
    thresholds_df.to_csv(
        output_dir / "threshold_sensitivity.csv", index=False
    )
    calibration_df.to_csv(
        output_dir / "calibration_by_fold.csv", index=False
    )
    predictions_df.to_csv(
        output_dir / "validation_predictions.csv", index=False
    )

    print("\n=== REFERENCE THRESHOLD RESULTS ===")
    print(
        folds_df[
            [
                "fold", "reference_f1", "reference_precision",
                "reference_recall", "roc_auc", "pr_auc",
                "brier_score", "mean_predicted_probability",
                "validation_positive_rate",
            ]
        ].round(4).to_string(index=False)
    )

    print(f"\nResults saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
