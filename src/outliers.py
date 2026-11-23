"""Isolation Forest: does treating outliers help? Compares versions in the temporal CV

    python -m src.outliers                      # task from modeling.task
    python -m src.outliers --task regression    # or classification

Settings live in preprocessing.outliers (config.yaml). The forest is fitted on
the training fold only and never sees the targets. Validation rows are never
removed: every row needs a prediction.

Versions (preprocessing.outliers.treatments, "none" is the reference):
  none        no outlier handling
  drop_train  remove flagged rows from the training fold
  flag        add a 0/1 outlier_flag feature
  score       add a continuous outlier_score feature (higher = more anomalous)
"""

import argparse
from datetime import datetime
from pprint import pformat

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import IsolationForest

from src.data_loader import PROJECT_ROOT, load_config, load_data
from src.evaluate_models import (
    LOWER_IS_BETTER,
    evaluate_predictions,
    evaluate_regression_predictions,
)
from src.feature_transformer import (
    apply_feature_transformer,
    fit_feature_transformer,
    prepare_feature_frame,
)
from src.fold_preprocessing import prepare_numeric_fold
from src.models import build_models, with_target_transform
from src.preprocessing import prepare_rows
from src.validation import temporal_cv

CALENDAR_COLUMNS = ("purchase_year", "purchase_month", "purchase_day_of_week")

# Suffixes of the indicator columns created during imputation
IMPUTATION_INDICATOR_SUFFIXES = (
    "_was_missing", "_no_training_donors", "_insufficient_history",
    "_was_corrected", "_was_recovered",
)


def numeric_columns(features):
    """Continuous columns only: no calendar, imputation indicators or 0/1 flags"""
    columns = features.select_dtypes("number").columns
    return [
        c for c in columns
        if c not in CALENDAR_COLUMNS
        and not c.endswith(IMPUTATION_INDICATOR_SUFFIXES)
        and features[c].nunique() > 2
    ]


def build_versions(forest, feat_train, feat_val, X_train, y_train, X_val):
    """Training matrix, labels and validation matrix for each version."""
    columns = numeric_columns(feat_train)
    forest.fit(feat_train[columns])

    flag_train = forest.predict(feat_train[columns]) == -1  # -1 = outlier
    flag_val = forest.predict(feat_val[columns]) == -1
    score_train = -forest.decision_function(feat_train[columns])
    score_val = -forest.decision_function(feat_val[columns])

    return {
        "none": (X_train, y_train, X_val),
        "drop_train": (X_train[~flag_train], y_train[~flag_train], X_val),
        "flag": (
            X_train.assign(outlier_flag=flag_train.astype(int)),
            y_train,
            X_val.assign(outlier_flag=flag_val.astype(int)),
        ),
        "score": (
            X_train.assign(outlier_score=score_train),
            y_train,
            X_val.assign(outlier_score=score_val),
        ),
    }, int(flag_train.sum())


def run(task=None):
    config = load_config()

    if task is not None:
        config["modeling"]["task"] = task
    task = config["modeling"]["task"]

    if task not in {"classification", "regression"}:
        raise ValueError("task must be 'classification' or 'regression'.")

    settings = config["preprocessing"]["outliers"]
    treatments = settings["treatments"]
    forest_settings = settings["isolation_forest"]

    if "none" not in treatments:
        raise ValueError("'none' must be in outliers.treatments (it is the reference).")

    if task == "regression":
        evaluation = config["modeling"]["regression_evaluation"]
        target_forms = evaluation["target_transforms"]
    else:
        evaluation = config["modeling"]["evaluation"]
        target_forms = [None]

    metric_names = evaluation["metrics"]
    primary_metric = evaluation["primary_metric"]

    wanted = settings["models"][task]
    models = {n: m for n, m in build_models(config).items() if n in wanted}
    missing = set(wanted) - set(models)
    if missing:
        raise ValueError(f"Models not enabled in config for {task}: {sorted(missing)}")

    classification_target = config["columns"]["classification_target"]
    target = config["columns"][f"{task}_target"]
    validation = config["validation"]

    data = prepare_rows(load_data(config)[0], config)
    rows = []

    for year, train, val in temporal_cv(
        data,
        validation["cv_years"],
        horizon_days=validation["target_horizon_days"],
        strategy=validation["strategy"],
        train_years=validation.get("sliding_window", {}).get("train_years"),
    ):
        prep_train, prep_val, _ = prepare_numeric_fold(train, val, config)

        if task == "regression":
            # Only customers who returned and have a known spend
            train_mask = ((train[classification_target] == 1) & train[target].notna()).to_numpy()
            val_mask = ((val[classification_target] == 1) & val[target].notna()).to_numpy()
            prep_train, prep_val = prep_train[train_mask], prep_val[val_mask]
            y_train, y_val = train.loc[train_mask, target], val.loc[val_mask, target]
        else:
            y_train, y_val = train[target], val[target]

        transformer, columns = fit_feature_transformer(prep_train, config)
        X_train = apply_feature_transformer(prep_train, transformer, columns, config)
        X_val = apply_feature_transformer(prep_val, transformer, columns, config)

        # The forest learns from the training fold only
        forest = IsolationForest(
            n_estimators=forest_settings["n_estimators"],
            contamination=forest_settings["contamination"],
            random_state=config["random_state"],
        )
        versions, n_flagged = build_versions(
            forest,
            prepare_feature_frame(prep_train, config),
            prepare_feature_frame(prep_val, config),
            X_train, y_train, X_val,
        )
        unknown = set(treatments) - set(versions)
        if unknown:
            raise ValueError(f"Unknown treatments: {sorted(unknown)}. Options: {list(versions)}")

        print(f"fold {year}: {n_flagged} outliers in {len(X_train)} training rows")

        for treatment in treatments:
            X_fit, y_fit, X_eval = versions[treatment]

            for name, base_model in models.items():
                for form in target_forms:
                    if form == "log" and name.startswith("dummy"):
                        continue

                    model = clone(base_model)
                    if form is not None:
                        model = with_target_transform(model, form)
                    model.fit(X_fit, y_fit)

                    if task == "classification":
                        probabilities = model.predict_proba(X_eval)[
                            :, list(model.classes_).index(1)
                        ]
                        scores = evaluate_predictions(
                            y_val, probabilities, metric_names,
                            evaluation["classification_threshold"],
                        )
                    else:
                        scores = evaluate_regression_predictions(
                            y_val, model.predict(X_eval), metric_names
                        )

                    rows.append({
                        "fold": year,
                        "treatment": treatment,
                        "model": name if form is None else f"{name} ({form})",
                        **scores,
                    })

    results = pd.DataFrame(rows)

    # Difference to the "none" version, same fold and model
    reference = results[results["treatment"] == "none"].set_index(["fold", "model"])
    index = pd.MultiIndex.from_frame(results[["fold", "model"]])
    for metric in metric_names:
        results[f"delta_{metric}"] = (
            results[metric].to_numpy() - reference[metric].reindex(index).to_numpy()
        )

    delta_columns = [f"delta_{m}" for m in metric_names]
    means = results.groupby("treatment")[metric_names].mean().reindex(treatments)
    deltas = results.groupby("treatment")[delta_columns].mean().reindex(treatments)
    deltas[f"{primary_metric}_delta_std"] = (
        results.groupby("treatment")[f"delta_{primary_metric}"].std().reindex(treatments)
    )
    by_model = (
        results.pivot_table(index="model", columns="treatment", values=primary_metric)
        [treatments]
    )

    direction = "lower" if primary_metric in LOWER_IS_BETTER else "higher"
    sections = [
        (f"MEAN METRICS (over folds and models)", means.round(4)),
        (f"DIFFERENCE TO 'none' (negative is better for: {sorted(LOWER_IS_BETTER)})", deltas.round(4)),
        (f"{primary_metric.upper()} BY MODEL ({direction} is better)", by_model.round(4)),
    ]

    print(f"\nTASK: {task} | primary metric: {primary_metric}")
    for title, table in sections:
        print(f"\n{title}\n{table.to_string()}")

    # Save in a timestamped folder, with the settings that produced the results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output = PROJECT_ROOT / config["paths"]["results"] / f"outliers_{task}" / timestamp
    output.mkdir(parents=True, exist_ok=False)
    results.to_csv(output / "fold_results.csv", index=False)

    with (output / "summary.txt").open("w", encoding="utf-8") as file:
        file.write(f"OUTLIER TREATMENT ({task})\n" + "=" * 65 + "\n\n")
        file.write(f"Settings:\n{pformat(settings, sort_dicts=False)}\n\n")
        file.write(
            f"Validation: {validation['strategy']}, folds {validation['cv_years']}\n"
            f"Primary metric: {primary_metric}\n"
        )
        for title, table in sections:
            file.write(f"\n{title}\n{'-' * 65}\n{table.to_string()}\n")
        file.write(
            "\nREADING GUIDE\n"
            "A version only matters if its difference to 'none' is larger than the "
            f"spread across folds ({primary_metric}_delta_std).\n"
        )

    print(f"\nSaved to {output}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare outlier treatments.")
    parser.add_argument(
        "--task",
        choices=["classification", "regression"],
        help="Overrides modeling.task from config.yaml.",
    )
    run(parser.parse_args().task)
