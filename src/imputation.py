"""Train-only, serializable imputation shared by both prediction tasks."""
import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import warnings
from datetime import datetime
from pprint import pformat

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import BayesianRidge, LogisticRegression, Ridge
from sklearn.metrics import (f1_score, average_precision_score, roc_auc_score,
                             mean_absolute_error, mean_squared_error, r2_score)
from threadpoolctl import threadpool_limits

from src.data_loader import PROJECT_ROOT, load_config
from src.feature_transformer import apply_feature_transformer, categorical_columns, fit_feature_transformer
from src.outliers import build_anomaly_scorer
from src.preprocessing import prepare_rows
from src.validation import temporal_cv


from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.preprocessing import RobustScaler
from src.preprocessing import (
    clean_historical_gaps, clean_prior_max_purchase_value, clean_spend_365d,
    fill_structural_missing, recover_categorical_missing, recover_historical_mean,
    recover_recency,
)


def structural_masks(df):
    """Eligibility concerns history, not the numerical placeholder value."""
    count = df["prior_purchase_count"]
    limits = {
        "prior_mean_gap_days": 2, "prior_std_gap_days": 3,
        "prior_mean_purchase_value": 1, "prior_std_purchase_value": 2,
        "prior_min_purchase_value": 1, "prior_max_purchase_value": 1,
        "days_since_previous_purchase": 1,
    }
    masks = {col: count.lt(n) for col, n in limits.items()}
    masks["spend_365d"] = df["purchase_count_365d"].eq(0)
    return masks


class ImputationPreprocessor:
    """Fit predictors only; transform queries independently of other queries.

    Targets are never consulted. Structural placeholders cannot donate values
    to an unknown history. A saved instance contains training-only history.
    """

    def __init__(self, config, task=None):
        self.config = deepcopy(config)

    def _deterministic(self, frame):
        data = frame.copy().reset_index(drop=True)
        data[self.date_] = pd.to_datetime(data[self.date_], errors="coerce").astype("datetime64[ns]")
        original_missing = {c: data[c].isna() for c in self.feature_columns_}
        data = clean_spend_365d(data, self.config)
        data = clean_prior_max_purchase_value(data, self.config)
        data = clean_historical_gaps(data, self.config)
        data = fill_structural_missing(data, self.config)
        for col, mask in structural_masks(data).items():
            data[f"{col}_insufficient_history"] = mask.astype(int)
        data = recover_historical_mean(data, self.config)
        data["recency_was_recovered"] = 0
        valid = data[self.date_].notna() & data[self.customer_].notna()
        if valid.any() and not self.history_.empty:
            recovered = recover_recency(data.loc[valid], self.history_, self.config)
            for col in ["days_since_previous_purchase", "recency_was_recovered"]:
                data.loc[valid, col] = recovered[col]
        for col, missing in original_missing.items():
            data[f"{col}_was_missing"] = missing.astype(int)
        data[f"{self.date_}_was_missing"] = data[self.date_].isna().astype(int)
        return data

    def fit(self, frame):
        prep = self.config["preprocessing"]
        if prep["numeric_imputation"]["fit_on"] != "train_fold_only":
            raise ValueError("Imputers must be fitted on train folds only")
        self.date_ = prep["date_features"]["source_column"]
        self.customer_ = self.config["columns"]["customer_id"]
        excluded = set(prep["feature_selection"]["drop_columns"]) | {
            self.date_, self.customer_, self.config["columns"]["classification_target"],
            self.config["columns"]["regression_target"],
        }
        self.feature_columns_ = [c for c in frame if c not in excluded]
        encoding = prep["encoding"]
        self.categorical_ = [c for c in categorical_columns(encoding) if c in self.feature_columns_]
        self.numeric_ = [c for c in self.feature_columns_ if c not in self.categorical_]
        recover = prep["categorical_missing"]["recover_from_customer_history"]
        history_cols = list(dict.fromkeys([self.customer_, self.date_] + recover))
        self.history_ = frame[history_cols].copy().reset_index(drop=True)
        self.history_[self.date_] = pd.to_datetime(self.history_[self.date_], errors="coerce").astype("datetime64[ns]")
        self.history_ = self.history_.dropna(subset=[self.customer_, self.date_])
        self.history_ = self.history_.sort_values(self.date_, kind="stable")
        data = self._deterministic(frame)
        donors = data[self.numeric_].astype(float).replace([np.inf, -np.inf], np.nan)
        for col, mask in structural_masks(data).items():
            donors.loc[mask, col] = np.nan
        self.empty_columns_ = [c for c in self.numeric_ if donors[c].notna().sum() == 0]
        self.medians_ = pd.Series({c: donors[c].median() if c not in self.empty_columns_ else 0.
                                   for c in self.numeric_})
        self.modes_ = {}
        for col in self.categorical_:
            mode = data[col].dropna().astype(str).mode()
            self.modes_[col] = mode.iloc[0] if len(mode) else prep["categorical_missing"]["fallback_value"]
        settings = prep["numeric_imputation"]
        self.strategy_ = settings["strategy"]
        if self.strategy_ not in {"median", "mean", "knn", "iterative"}:
            raise ValueError(f"Unsupported numeric strategy: {self.strategy_}")
        if self.strategy_ == "mean":
            self.medians_ = donors.mean().fillna(0.)
        self.target_columns_ = settings["columns"]
        self.multivariate_columns_ = [c for c in dict.fromkeys(
            self.target_columns_ + settings.get("knn", {}).get("reference_columns", [])
        ) if c not in self.empty_columns_]
        self.imputer_ = None
        if self.strategy_ in {"knn", "iterative"} and self.multivariate_columns_:
            self.scaler_ = RobustScaler()
            values = self.scaler_.fit_transform(donors[self.multivariate_columns_])
            if self.strategy_ == "knn":
                knn = settings.get("knn", {})
                self.imputer_ = KNNImputer(n_neighbors=knn.get("n_neighbors", 5),
                                          weights=knn.get("weights", "distance"),
                                          keep_empty_features=True)
            else:
                opts = settings.get("iterative", {})
                self.imputer_ = IterativeImputer(estimator=BayesianRidge(),
                                                max_iter=opts.get("max_iter", 10),
                                                sample_posterior=False,
                                                random_state=self.config.get("random_state", 42),
                                                keep_empty_features=True)
            self.imputer_.fit(values)
        return self

    def transform(self, frame):
        data = self._deterministic(frame)
        data[self.numeric_] = data[self.numeric_].replace([np.inf, -np.inf], np.nan)
        if self.imputer_ is not None and len(data):
            query = data[self.multivariate_columns_].astype(float).copy()
            for col, mask in structural_masks(data).items():
                if col in query:
                    query.loc[mask, col] = np.nan
            imputed = self.scaler_.inverse_transform(
                self.imputer_.transform(self.scaler_.transform(query)))
            estimates = pd.DataFrame(imputed, columns=query.columns, index=data.index)
            for col in self.target_columns_:
                if col in estimates:
                    missing = data[col].isna()
                    data.loc[missing, col] = estimates.loc[missing, col].clip(lower=0)
        for col in self.numeric_:
            data[col] = data[col].fillna(self.medians_[col])
        flags = pd.DataFrame({f"{c}_no_training_donors": int(c in self.empty_columns_)
                              for c in self.numeric_}, index=data.index)
        data = pd.concat([data, flags], axis=1).copy()
        settings = self.config["preprocessing"]["categorical_missing"]
        strategy = settings.get("strategy", "history")
        if strategy not in {"unknown", "mode", "history"}:
            raise ValueError(f"Unsupported categorical strategy: {strategy}")
        for col in settings["recover_from_customer_history"]:
            data[f"{col}_was_recovered"] = 0
        if strategy == "history":
            data = recover_categorical_missing(data, self.history_, self.config)
        for col in self.categorical_:
            fallback = self.modes_[col] if strategy == "mode" else settings["fallback_value"]
            data[col] = data[col].astype(object).fillna(fallback)
        data.index = frame.index
        return data

    def fit_transform(self, frame):
        return self.fit(frame).transform(frame)


# Fixed teaching comparison; these settings do not belong in production config.yaml.
COMPARISON = {
    "f1_tolerance": 0.005,
    "mae_relative_tolerance": 0.01,
    "regression_models": {
        "ridge": {"alpha": 1.0},
        "random_forest": {"n_estimators": 200, "min_samples_leaf": 5, "n_jobs": -1},
    },
}


def candidates(config=None):
    """Five readable comparisons, including Unknown versus mode."""
    choices = [
        ("median", 0, "unknown", "Median + Unknown"),
        ("median", 0, "mode", "Median + mode"),
        ("knn", 5, "unknown", "KNN (k=5) + Unknown"),
        ("knn", 10, "unknown", "KNN (k=10) + Unknown"),
        ("iterative", 0, "unknown", "Iterative (BayesianRidge) + Unknown"),
    ]
    for method, k, categorical, label in choices:
        yield {"numeric": method, "k": k, "weights": "uniform",
               "categorical": categorical, "label": label,
               "candidate": f"{method}_{k}_uniform__{categorical}"}


def candidate_config(config, candidate):
    cfg = deepcopy(config)
    settings = cfg["preprocessing"]["numeric_imputation"]
    settings["strategy"] = candidate["numeric"]
    settings["knn"].update(n_neighbors=candidate["k"] or 5, weights=candidate["weights"])
    settings["iterative"] = {"max_iter": 10}
    cfg["preprocessing"]["categorical_missing"]["strategy"] = candidate["categorical"]
    return cfg


def estimators(config):
    seed = config["random_state"]
    params = config["modeling"]["models"]
    logistic = dict(params["logistic_regression"]["parameters"], random_state=seed)
    forest = dict(params["random_forest"]["parameters"], random_state=seed)
    reg = COMPARISON["regression_models"]
    return {
        "classification": {"logistic_regression": LogisticRegression(**logistic),
                           "random_forest": RandomForestClassifier(**forest)},
        "regression": {"ridge": Ridge(**reg["ridge"]),
                       "random_forest": RandomForestRegressor(**dict(reg["random_forest"], random_state=seed))},
    }


def regression_mask(frame, config):
    columns = config["columns"]
    return (frame[columns["classification_target"]].eq(1)
            & frame[columns["regression_target"]].notna())


def regression_scores(truth, predicted):
    return {"mae": mean_absolute_error(truth, predicted),
            "rmse": np.sqrt(mean_squared_error(truth, predicted)),
            "r2": r2_score(truth, predicted)}


def encode(preprocessor, train, validation, config):
    prepared = preprocessor.transform(train)
    transformer, columns = fit_feature_transformer(prepared, config)
    return (apply_feature_transformer(prepared, transformer, columns, config),
            apply_feature_transformer(preprocessor.transform(validation), transformer, columns, config),
            transformer, columns)


class TaskPipeline:
    """Reusable fitted imputation, encoding and estimator for a single task."""
    def __init__(self, config, task, model_name):
        self.config, self.task, self.model_name = deepcopy(config), task, model_name

    def fit(self, train):
        # All training predictors may inform unsupervised preprocessing; only
        # known returner spending labels train the conditional regressor.
        self.preprocessor = ImputationPreprocessor(self.config, self.task).fit(train)
        cfg = self.preprocessor.config
        prepared = self.preprocessor.transform(train)
        self.transformer, self.columns = fit_feature_transformer(prepared, cfg, task=self.task)
        X = apply_feature_transformer(prepared, self.transformer, self.columns, cfg)
        # Anomaly score (method: none = off): fitted on all training predictors, no labels.
        self.anomaly = build_anomaly_scorer(cfg)
        if self.anomaly is not None:
            X = self.anomaly.fit_transform(X)
        self.model = estimators(cfg)[self.task][self.model_name]
        if self.task == "regression":
            mask = regression_mask(train, cfg)
            target = cfg["columns"]["regression_target"]
        else:
            target = cfg["columns"]["classification_target"]
            mask = train[target].isin([0, 1])
        self.model.fit(X.loc[mask], train.loc[mask, target])
        return self

    def predict(self, frame):
        X = apply_feature_transformer(self.preprocessor.transform(frame), self.transformer,
                                      self.columns, self.preprocessor.config)
        if self.anomaly is not None:
            X = self.anomaly.transform(X)
        if self.task == "classification":
            return self.model.predict_proba(X)[:, list(self.model.classes_).index(1)]
        return np.maximum(0, self.model.predict(X))


def audit(frame, config):
    """Development data only; counts distinguish raw absence and domain errors."""
    prep = ImputationPreprocessor(candidate_config(config, next(candidates(config))))
    prep.fit(frame)
    cleaned = prep._deterministic(frame)
    masks = structural_masks(frame.reset_index(drop=True))
    rows = []
    for col in prep.feature_columns_:
        missing = frame[col].isna().reset_index(drop=True)
        structural = masks.get(col, pd.Series(False, index=cleaned.index))
        corrected = cleaned.get(f"{col}_was_corrected", pd.Series(0, index=cleaned.index))
        rows.append({"column": col, "kind": "categorical" if col in prep.categorical_ else "numeric",
                     "rows": len(frame), "raw_missing": int(missing.sum()),
                     "structural_missing": int((missing & structural).sum()),
                     "unexpected_missing": int((missing & ~structural).sum()),
                     "invalid_corrected": int(corrected.sum()),
                     "remaining_after_rules": int(cleaned[col].isna().sum())})
    return pd.DataFrame(rows)


def select(results, catalog, config):
    selected, summaries = {}, []
    settings = COMPARISON
    for task, metric in [("classification", "f1"), ("regression", "mae")]:
        part = results.loc[results.task.eq(task)]
        summary = part.groupby("candidate")[metric].agg(["mean", "std"]).reset_index()
        summary = summary.merge(catalog, on="candidate")
        best = summary["mean"].max() if task == "classification" else summary["mean"].min()
        summary["eligible"] = (summary["mean"] >= best - settings["f1_tolerance"] if task == "classification"
                               else summary["mean"] <= best * (1 + settings["mae_relative_tolerance"]))
        summary["numeric_rank"] = summary.numeric.map({"median": 0, "knn": 1, "iterative": 2})
        summary["categorical_rank"] = summary.categorical.map({"unknown": 0, "mode": 1, "history": 2})
        summary["weight_rank"] = summary.weights.map({"uniform": 0, "distance": 1})
        winner = summary.loc[summary.eligible].sort_values(
            ["numeric_rank", "categorical_rank", "mean", "k", "weight_rank"],
            ascending=[True, True, task == "regression", True, True]).iloc[0]
        selected[task] = catalog.set_index("candidate").loc[winner.candidate].to_dict() | {"candidate": winner.candidate}
        summary["task"], summary["metric"] = task, metric
        summaries.append(summary)
    return selected, pd.concat(summaries, ignore_index=True)


def run(output):
    config = load_config()
    output.mkdir(parents=True, exist_ok=True)
    if any(p.is_file() for p in output.iterdir()):
        raise FileExistsError("Choose a results directory without existing result files")
    raw = pd.read_csv(PROJECT_ROOT / config["paths"]["train"])
    dates = pd.to_datetime(raw.purchase_date, errors="coerce")
    # Never inspect holdout/test values to make selections or create diagnostics.
    dev = prepare_rows(raw.loc[dates.dt.year.le(max(config["validation"]["cv_years"]))], config)
    audit(dev, config).to_csv(output / "missingness_audit.csv", index=False)
    target = config["columns"]["classification_target"]
    spend = config["columns"]["regression_target"]
    catalog = pd.DataFrame(candidates(config))
    catalog.to_csv(output / "candidates.csv", index=False)
    metadata = {"missing_date_rows_excluded": int(dates.isna().sum()), "development_rows": len(dev),
                "unknown_returner_spend_development": int((dev[target].eq(1) & dev[spend].isna()).sum()),
                "train_sha256": hashlib.sha256((PROJECT_ROOT / config["paths"]["train"]).read_bytes()).hexdigest(),
                "versions": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn"]},
                "note": "Selection scores, not unbiased final performance. Holdout/test excluded. Prior EDA saw full train."}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    scores, warning_records, fold_records = [], [], []
    validation = config["validation"]
    folds = list(temporal_cv(dev, validation["cv_years"], validation["target_horizon_days"],
                            strategy=validation["strategy"], train_years=validation["sliding_window"]["train_years"]))
    # Cache only fold predictions needed for end-to-end reporting, not trained models.
    predictions = {}
    for year, train, val in folds:
        fold_records.append({"fold": year, "train_rows": len(train), "validation_rows": len(val),
                             "regression_train": int(regression_mask(train, config).sum()),
                             "regression_validation": int(regression_mask(val, config).sum()),
                             "unknown_spend_validation": int((val[target].eq(1) & val[spend].isna()).sum())})
        for candidate in catalog.to_dict("records"):
            started = time.perf_counter()
            cfg = candidate_config(config, candidate)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                preprocessor = ImputationPreprocessor(cfg).fit(train)
                X_train, X_val, _, _ = encode(preprocessor, train, val, cfg)
                prep_seconds = time.perf_counter() - started
                for task, models in estimators(cfg).items():
                    tr_mask = regression_mask(train, cfg) if task == "regression" else train[target].isin([0, 1])
                    va_mask = regression_mask(val, cfg) if task == "regression" else val[target].isin([0, 1])
                    label = spend if task == "regression" else target
                    for name, model in models.items():
                        model_started = time.perf_counter()
                        model.fit(X_train.loc[tr_mask], train.loc[tr_mask, label])
                        if task == "classification":
                            pred = model.predict_proba(X_val)[:, list(model.classes_).index(1)]
                            truth = val.loc[va_mask, target]
                            metric = {"f1": f1_score(truth, pred[va_mask] >= .5),
                                      "pr_auc": average_precision_score(truth, pred[va_mask]),
                                      "roc_auc": roc_auc_score(truth, pred[va_mask])}
                        else:
                            pred = np.maximum(0, model.predict(X_val))
                            metric = regression_scores(val.loc[va_mask, spend], pred[va_mask])
                        predictions[(year, candidate["candidate"], task, name)] = pred
                        scores.append(dict(candidate=candidate["candidate"], task=task, fold=year, model=name,
                                           preprocessing_seconds=prep_seconds, model_seconds=time.perf_counter()-model_started,
                                           empty_training_columns="|".join(preprocessor.empty_columns_), **metric))
                warning_records.extend({"fold": year, "candidate": candidate["candidate"], "warning": str(w.message)} for w in caught)
            pd.DataFrame(scores).to_csv(output / "scores.csv", index=False)
            print(f"{year} {candidate['candidate']}: {time.perf_counter()-started:.1f}s", flush=True)
    pd.DataFrame(fold_records).to_csv(output / "folds.csv", index=False)
    pd.DataFrame(warning_records, columns=["fold", "candidate", "warning"]).drop_duplicates().to_csv(output / "warnings.csv", index=False)
    winners, summary = select(pd.DataFrame(scores), catalog, config)
    summary.to_csv(output / "selection_summary.csv", index=False)
    (output / "selected.json").write_text(json.dumps(winners, indent=2), encoding="utf-8")
    end_to_end = []
    for year, _, val in folds:
        known = val[target].eq(0) | regression_mask(val, config)
        truth = np.where(val[target].eq(0), 0., val[spend])
        for classifier in estimators(config)["classification"]:
            probability = predictions[(year, winners["classification"]["candidate"], "classification", classifier)]
            for regressor in estimators(config)["regression"]:
                spend_pred = predictions[(year, winners["regression"]["candidate"], "regression", regressor)]
                combined = np.where(probability >= .5, spend_pred, 0.)
                end_to_end.append(dict(fold=year, classifier=classifier, regressor=regressor,
                                       known_truth_rows=int(known.sum()), excluded_unknown_spend=int((~known).sum()),
                                       false_positives=int(((probability >= .5) & val[target].eq(0)).sum()),
                                       false_negatives=int(((probability < .5) & val[target].eq(1)).sum()),
                                       **regression_scores(truth[known], combined[known])))
    pd.DataFrame(end_to_end).to_csv(output / "end_to_end.csv", index=False)
    write_summary(output, config, catalog, pd.DataFrame(scores), summary, winners, pd.DataFrame(end_to_end), metadata)
    print(json.dumps(winners, indent=2), flush=True)


def write_summary(output, config, catalog, scores, summary, winners, end_to_end, metadata):
    """Baseline-style TXT containing results and the actual parameter values."""
    sections = ["ETAI GROUP 13", "IMPUTATION COMPARISON", "=" * 75,
                "EXPERIMENT INFORMATION", datetime.now().isoformat(timespec="seconds"),
                pformat(metadata, sort_dicts=False),
                "CANDIDATES", catalog.to_string(index=False),
                "ITERATIVE METHOD", "IterativeImputer(estimator=BayesianRidge(), max_iter=10, "
                "sample_posterior=False, random_state=42). Single imputation; not random forest or multiple imputation.",
                "MODEL PARAMETERS", pformat({task: {name: model.get_params() for name, model in models.items()}
                                              for task, models in estimators(config).items()}, sort_dicts=False),
                "SELECTION CRITERIA", pformat(COMPARISON, sort_dicts=False),
                "CROSS-VALIDATION SUMMARY", summary.to_string(index=False),
                "DETAILED RESULTS BY FOLD", scores.round(6).to_string(index=False),
                "END-TO-END RESULTS", end_to_end.round(6).to_string(index=False),
                "FINAL DECISION", pformat(winners, sort_dicts=False),
                "CONFIGURATION USED (base; candidate overrides below)", pformat(config, sort_dicts=False)]
    for candidate in catalog.to_dict("records"):
        prep = candidate_config(config, candidate)["preprocessing"]
        sections.extend(["CANDIDATE SETTINGS: " + candidate["label"],
                         pformat({key: prep[key] for key in ["numeric_imputation", "categorical_missing"]}, sort_dicts=False)])
    sections.extend(["NOTES", "Each imputer/scaler is fitted on its training fold only. "
                     "The 2024 holdout and test do not participate in selection. "
                     "These are selection scores, not independent final performance. "
                     "All methods use the same recovery rules and indicators. "
                     "Latest available earlier training transaction is assumed to be the true previous purchase."])
    (output / "imputation_summary.txt").write_text("\n\n".join(sections)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/imputation_selection"))
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(PROJECT_ROOT / args.output)


if __name__ == "__main__":
    main()
