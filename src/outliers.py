"""Anomaly score (Isolation Forest or LOF) as a learned step of the pipeline.

Same idea as the AnomalyScore step taught in class: a detector is fitted on the
training rows only and its score is appended as ONE extra feature, `anomaly_score`
(higher = more unusual). The score is kept as a feature; no row is ever removed,
because every customer needs a prediction.

Where it runs: right after the feature transformer and BEFORE feature selection, so
the selection can keep or drop the score like any other feature (same order as in
class: preprocessing -> anomaly score -> feature selection -> model). Settings: the
top-level `anomaly` block of config.yaml, with `method: none` switching the step off.

Compare the methods on the same folds:

    python -m src.outliers                      # task from modeling.task
    python -m src.outliers --task regression
"""

from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

METHODS = ("none", "isolation_forest", "lof")
SCORE_COLUMN = "anomaly_score"

# Used when config.yaml has no `anomaly` section (step switched off).
DEFAULTS = {
    "method": "none",
    "n_estimators": 100,
    "n_neighbors": 20,
    "columns": "all",
    "exclude_columns": ["purchase_year"],
    "experiment": {
        "methods": ["none", "isolation_forest", "lof"],
        "models": {
            "classification": ["logistic_regression", "random_forest", "hist_gradient_boosting"],
            "regression": ["ridge", "random_forest", "hist_gradient_boosting"],
        },
    },
}


class AnomalyScore(TransformerMixin, BaseEstimator):
    """Append `anomaly_score`: how unusual each row is compared with the training rows.

    method        "isolation_forest": the score is s(x) of the theory, about 0.5 for
                  ordinary rows and close to 1 for anomalies. "lof": log(1 + LOF),
                  about 0.7 for ordinary rows (LOF is about 1) and larger for anomalies.
                  The log keeps a few extreme LOF values (we saw up to ~85 000) from
                  wrecking linear models.
    n_estimators  isolation_forest: number of trees.
    n_neighbors   lof: must be larger than the largest group of identical rows.
    columns       "all" or "continuous" (columns with more than 2 distinct values, which
                  leaves out one-hot and 0/1 columns).
    exclude_columns
                  never used by the detector. Matched by the name after the block prefix
                  (`numeric__purchase_year` is matched by "purchase_year"): a calendar
                  column drifts over time, so every future row would look anomalous.
    random_state  isolation_forest seed.

    Fitted on the training rows only, then scores any rows (validation, new customers).
    DataFrames in, DataFrames out, so feature names survive.
    """

    def __init__(self, method="isolation_forest", n_estimators=100, n_neighbors=20,
                 columns="all", exclude_columns=(), random_state=None):
        self.method = method
        self.n_estimators = n_estimators
        self.n_neighbors = n_neighbors
        self.columns = columns
        self.exclude_columns = exclude_columns
        self.random_state = random_state

    # -- column handling -------------------------------------------------------------
    def _choose_columns(self, X):
        if not isinstance(X, pd.DataFrame):
            if self.columns != "all" or list(self.exclude_columns):
                raise ValueError("`columns` and `exclude_columns` need a DataFrame.")
            return None
        if self.columns not in ("all", "continuous"):
            raise ValueError("columns must be 'all' or 'continuous'.")

        excluded = set(self.exclude_columns)
        chosen = [c for c in X.columns if str(c).split("__", 1)[-1] not in excluded]
        if self.columns == "continuous":
            chosen = [c for c in chosen if X[c].nunique() > 2]
        if not chosen:
            raise ValueError("No columns left for the anomaly detector.")
        return chosen

    def _values(self, X):
        values = X[self.used_columns_] if self.used_columns_ is not None else X
        values = np.asarray(values, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("The anomaly detector needs finite inputs (impute first).")
        return values

    # -- scikit-learn contract -------------------------------------------------------
    def fit(self, X, y=None):
        if self.method == "isolation_forest":
            self.detector_ = IsolationForest(
                n_estimators=self.n_estimators, random_state=self.random_state
            )
        elif self.method == "lof":
            # novelty=True is what lets the detector score NEW rows.
            self.detector_ = LocalOutlierFactor(n_neighbors=self.n_neighbors, novelty=True)
        else:
            raise ValueError(
                f"Unknown anomaly method: {self.method}. Options: isolation_forest, lof"
            )
        self.used_columns_ = self._choose_columns(X)
        self.detector_.fit(self._values(X))
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def _append(self, X, score):
        if isinstance(X, pd.DataFrame):
            if SCORE_COLUMN in X.columns:
                raise ValueError(f"X already has a '{SCORE_COLUMN}' column.")
            return X.assign(**{SCORE_COLUMN: score})
        return np.column_stack([X, score])

    def fit_transform(self, X, y=None):
        """Fit on X and score it. LOF scores its own training rows without counting
        each row as its own neighbour (negative_outlier_factor_)."""
        self.fit(X)
        if self.method == "lof":
            score = np.log1p(-self.detector_.negative_outlier_factor_)
            return self._append(X, score)
        return self.transform(X)

    def transform(self, X):
        raw = -self.detector_.score_samples(self._values(X))
        score = np.log1p(raw) if self.method == "lof" else raw
        return self._append(X, score)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        return np.append(np.asarray(input_features, dtype=object), SCORE_COLUMN)


# -- config helpers ------------------------------------------------------------------
def anomaly_settings(config):
    """The `anomaly` block of config.yaml over the module defaults."""
    settings = deepcopy(DEFAULTS)
    for key, value in (config.get("anomaly") or {}).items():
        if isinstance(value, dict) and isinstance(settings.get(key), dict):
            settings[key].update(value)
        else:
            settings[key] = value
    if settings["method"] not in METHODS:
        raise ValueError(
            f"Unknown anomaly method: {settings['method']}. Options: {list(METHODS)}"
        )
    return settings


def build_anomaly_scorer(config, method=None):
    """An unfitted AnomalyScore from the config, or None when the step is off."""
    settings = anomaly_settings(config)
    method = settings["method"] if method is None else method
    if method not in METHODS:
        raise ValueError(f"Unknown anomaly method: {method}. Options: {list(METHODS)}")
    if method == "none":
        return None
    return AnomalyScore(
        method=method,
        n_estimators=settings["n_estimators"],
        n_neighbors=settings["n_neighbors"],
        columns=settings["columns"],
        exclude_columns=settings["exclude_columns"],
        random_state=config.get("random_state"),
    )


def add_anomaly_score(X_train, X_val, config):
    """Fit on X_train, append the score to X_train and X_val.

    Returns (X_train, X_val, scorer); scorer is None (matrices untouched) when
    `anomaly.method` is "none".

    Call it BEFORE feature selection. The selectors that work on the matrix they
    receive (mRMR, Boruta, L1) see the score as one more column. RFECV rebuilds the
    features in its own inner folds, and there `select_rfecv` fits a new detector on
    each inner training split and appends the score, so no inner validation row leaks.
    """
    scorer = build_anomaly_scorer(config)
    if scorer is None:
        return X_train, X_val, None
    return scorer.fit_transform(X_train), scorer.transform(X_val), scorer


# -- comparison of the methods -------------------------------------------------------
def anomaly_profile(config, task):
    """How different are the most anomalous rows? Top 5% of each fold's validation rows
    by anomaly score against the rest: return rate (classification) or mean spend of
    returners (regression), averaged over folds. Detectors are fitted on the training
    fold only; feature selection is not applied here."""
    from src.data_loader import load_data
    from src.feature_transformer import apply_feature_transformer, fit_feature_transformer
    from src.fold_preprocessing import prepare_numeric_fold
    from src.preprocessing import prepare_rows
    from src.validation import temporal_cv

    validation = config["validation"]
    classification_target = config["columns"]["classification_target"]
    target = config["columns"][f"{task}_target"]
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
            train_mask = ((train[classification_target] == 1) & train[target].notna()).to_numpy()
            val_mask = ((val[classification_target] == 1) & val[target].notna()).to_numpy()
            prep_train, prep_val = prep_train[train_mask], prep_val[val_mask]
            outcome = val.loc[val_mask, target].to_numpy()
        else:
            outcome = val[target].to_numpy()

        transformer, columns = fit_feature_transformer(prep_train, config, task=task)
        X_train = apply_feature_transformer(prep_train, transformer, columns, config)
        X_val = apply_feature_transformer(prep_val, transformer, columns, config)

        for method in ("isolation_forest", "lof"):
            scorer = build_anomaly_scorer(config, method).fit(X_train)
            score = scorer.transform(X_val)[SCORE_COLUMN].to_numpy()
            top = score >= np.quantile(score, 0.95)
            rows.append({"fold": year, "method": method,
                         "top_5pct": outcome[top].mean(), "rest": outcome[~top].mean()})

    profile = pd.DataFrame(rows).groupby("method")[["top_5pct", "rest"]].mean().round(3)
    profile.columns = [f"{'return rate' if task == 'classification' else 'mean spend (EUR)'}: {c}"
                       for c in ("top 5% anomalies", "all others")]
    return profile


def compare_methods(task=None, models=None, skip_selection=False):
    """Run the evaluation once per anomaly method (same folds, same models) and compare."""
    from datetime import datetime
    from pprint import pformat

    from src.data_loader import PROJECT_ROOT, load_config
    from src.evaluate_models import LOWER_IS_BETTER, run_evaluation

    base = load_config()
    if task is not None:
        base["modeling"]["task"] = task
    task = base["modeling"]["task"]
    if task not in ("classification", "regression"):
        raise ValueError("task must be 'classification' or 'regression'.")

    experiment = anomaly_settings(base)["experiment"]
    methods = experiment["methods"]
    if "none" not in methods or any(m not in METHODS for m in methods):
        raise ValueError(f"experiment.methods must include 'none' and use {list(METHODS)}.")

    # Same models in every run, so a difference comes from the anomaly score alone.
    models_key = "regression_models" if task == "regression" else "models"
    configured = base["modeling"][models_key]
    wanted = list(models or experiment["models"][task])
    unavailable = [m for m in wanted if not configured.get(m, {}).get("enabled", False)]
    if unavailable:
        raise ValueError(f"Models not enabled in config.yaml ({models_key}): {unavailable}")
    for name, spec in configured.items():
        spec["enabled"] = spec.get("enabled", False) and name in wanted

    if skip_selection:
        base["preprocessing"]["feature_selection"]["enabled"] = False

    evaluation = base["modeling"]["regression_evaluation" if task == "regression" else "evaluation"]
    metrics, primary = evaluation["metrics"], evaluation["primary_metric"]
    keys = ["fold", "model"] + (["target_form"] if task == "regression" else [])

    runs = {}
    for method in methods:
        print(f"\n{'#' * 70}\n# ANOMALY METHOD: {method}\n{'#' * 70}")
        config = deepcopy(base)
        config.setdefault("anomaly", {})["method"] = method
        runs[method], _ = run_evaluation(task, config=config)

    results = pd.concat([df.assign(method=m) for m, df in runs.items()], ignore_index=True)
    reference = results[results["method"] == "none"].set_index(keys)[metrics]
    index = pd.MultiIndex.from_frame(results[keys])
    for metric in metrics:
        results[f"delta_{metric}"] = (
            results[metric].to_numpy() - reference[metric].reindex(index).to_numpy()
        )

    delta_columns = [f"delta_{m}" for m in metrics]
    means = results.groupby("method")[metrics].mean().reindex(methods)
    deltas = results.groupby("method")[delta_columns].mean().reindex(methods)
    deltas[f"{primary}_delta_std"] = (
        results.groupby("method")[f"delta_{primary}"].std().reindex(methods)
    )
    by_model = results.assign(
        model=results["model"] if task == "classification"
        else results["model"] + " (" + results["target_form"] + ")"
    ).pivot_table(index="model", columns="method", values=primary)[methods]
    profile = anomaly_profile(base, task)

    direction = "lower" if primary in LOWER_IS_BETTER else "higher"
    sections = [
        ("MEAN METRICS (over folds and models)", means.round(4)),
        (f"DIFFERENCE TO 'none' (negative is better for: {sorted(LOWER_IS_BETTER)})", deltas.round(4)),
        (f"{primary.upper()} BY MODEL ({direction} is better)", by_model.round(4)),
        ("THE MOST ANOMALOUS ROWS (top 5% of each validation fold)", profile),
    ]

    print(f"\nTASK: {task} | primary metric: {primary} | models: {', '.join(wanted)}")
    for title, table in sections:
        print(f"\n{title}\n{table.to_string()}")

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output = PROJECT_ROOT / base["paths"]["results"] / f"outliers_{task}" / stamp
    output.mkdir(parents=True, exist_ok=False)
    results.to_csv(output / "fold_results.csv", index=False)
    with (output / "summary.txt").open("w", encoding="utf-8") as file:
        file.write(f"ANOMALY SCORE COMPARISON ({task})\n{'=' * 65}\n\n")
        file.write(f"Settings:\n{pformat(anomaly_settings(base), sort_dicts=False)}\n")
        file.write(f"Feature selection skipped: {skip_selection}\n")
        file.write(f"Validation: {base['validation']['strategy']}, folds {base['validation']['cv_years']}\n")
        for title, table in sections:
            file.write(f"\n{title}\n{'-' * 65}\n{table.to_string()}\n")
        file.write(
            "\nREADING GUIDE\nA method only matters if its difference to 'none' is larger "
            f"than the spread across folds ({primary}_delta_std).\n"
        )
    print(f"\nSaved to {output}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare anomaly-score methods.")
    parser.add_argument("--task", choices=["classification", "regression"],
                        help="Overrides modeling.task from config.yaml.")
    parser.add_argument("--models", nargs="+",
                        help="Models to compare (default: anomaly.experiment.models).")
    parser.add_argument("--skip-selection", action="store_true",
                        help="Classification: skip feature selection (faster).")
    args = parser.parse_args()
    compare_methods(args.task, args.models, args.skip_selection)
