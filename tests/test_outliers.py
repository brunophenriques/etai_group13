"""Anomaly score: class contract, config consistency, real folds and prediction path."""

from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

import src.feature_selection as fs
from src.data_loader import load_config, load_data
from src.feature_selection import apply_consensus_selection, run_feature_selection
from src.feature_transformer import apply_feature_transformer, fit_feature_transformer
from src.fold_preprocessing import prepare_numeric_fold
from src.imputation import TaskPipeline
from src.outliers import (
    METHODS,
    SCORE_COLUMN,
    AnomalyScore,
    add_anomaly_score,
    anomaly_settings,
    build_anomaly_scorer,
)
from src.preprocessing import prepare_rows
from src.validation import temporal_cv


def raises(error, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except error:
        return True
    return False


config = load_config()

# Config consistency
assert "anomaly" in config and "anomaly" not in config["preprocessing"], (
    "`anomaly` is a top-level block, next to `preprocessing`."
)
settings = anomaly_settings(config)
assert settings["method"] in METHODS
assert "none" in settings["experiment"]["methods"], "'none' is the reference."
assert set(settings["experiment"]["methods"]) <= set(METHODS)
assert settings["columns"] in ("all", "continuous")
assert settings["n_neighbors"] >= 2 and settings["n_estimators"] >= 1

enabled_classifiers = {n for n, s in config["modeling"]["models"].items() if s.get("enabled")}
enabled_regressors = {n for n, s in config["modeling"]["regression_models"].items() if s.get("enabled")}
assert set(settings["experiment"]["models"]["classification"]) <= enabled_classifiers
assert set(settings["experiment"]["models"]["regression"]) <= enabled_regressors

# A config without the section means "step off".
bare = {"preprocessing": {}, "random_state": 1}
assert anomaly_settings(bare)["method"] == "none"
assert build_anomaly_scorer(bare) is None
assert raises(ValueError, anomaly_settings, {"anomaly": {"method": "pca"}})

# AnomalyScore on synthetic data
rng = np.random.default_rng(0)


def make_frame(n, extreme=()):
    frame = pd.DataFrame({
        "numeric__a": rng.normal(size=n),
        "numeric__b": rng.normal(size=n),
        "numeric__purchase_year": np.full(n, 2021.0),
        "categorical__flag_1": rng.integers(0, 2, n).astype(float),
    })
    for row in extreme:
        frame.loc[row, ["numeric__a", "numeric__b"]] = [12.0, -12.0]
    return frame


train = make_frame(400, extreme=[7])
new = make_frame(60, extreme=[3])
original_train = train.copy()

for method in ("isolation_forest", "lof"):
    scorer = AnomalyScore(
        method=method, n_neighbors=20, exclude_columns=["purchase_year"], random_state=1
    )
    out_train = scorer.fit_transform(train)
    out_new = scorer.transform(new)

    # One extra column, same rows and index, input untouched
    assert list(out_train.columns) == list(train.columns) + [SCORE_COLUMN]
    assert out_train.index.equals(train.index) and out_new.index.equals(new.index)
    assert train.equals(original_train), "Input must not be modified."
    assert np.isfinite(out_train[SCORE_COLUMN]).all() and np.isfinite(out_new[SCORE_COLUMN]).all()

    # The obvious extremes get the highest score
    assert out_train[SCORE_COLUMN].idxmax() == 7, method
    assert out_new[SCORE_COLUMN].idxmax() == 3, method

    # A calendar column is never seen by the detector
    assert "numeric__purchase_year" not in scorer.used_columns_
    drifted = new.assign(numeric__purchase_year=2030.0)
    assert np.allclose(scorer.transform(drifted)[SCORE_COLUMN], out_new[SCORE_COLUMN])

    # A row's score does not depend on the other rows scored with it
    alone = np.array([scorer.transform(new.iloc[[i]])[SCORE_COLUMN].iloc[0] for i in range(10)])
    assert np.allclose(alone, out_new[SCORE_COLUMN].iloc[:10]), method

    # The detector learns from the training rows only
    again = AnomalyScore(
        method=method, n_neighbors=20, exclude_columns=["purchase_year"], random_state=1
    )
    assert np.allclose(
        again.fit_transform(train)[SCORE_COLUMN], out_train[SCORE_COLUMN]
    ), "Training scores must not depend on any other data."

    assert list(scorer.get_feature_names_out(list(train.columns)))[-1] == SCORE_COLUMN

# Isolation Forest: fit_transform and fit + transform agree; the score is s(x) in (0, 1)
forest = AnomalyScore(method="isolation_forest", random_state=1)
direct = forest.fit_transform(train)[SCORE_COLUMN]
assert np.allclose(direct, forest.transform(train)[SCORE_COLUMN])
assert 0.3 < direct.min() and direct.max() < 1.0

# LOF: training rows use negative_outlier_factor_ (a row is not its own neighbour) and the score is log(1 + LOF)
lof = AnomalyScore(method="lof", n_neighbors=20)
lof_train = lof.fit_transform(train)[SCORE_COLUMN]
reference = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(train)
assert np.allclose(lof_train, np.log1p(-reference.negative_outlier_factor_))
assert not np.allclose(lof_train, lof.transform(train)[SCORE_COLUMN]), (
    "score_samples on training rows counts each row as its own neighbour."
)

# columns="continuous" leaves out 0/1 columns
continuous = AnomalyScore(method="isolation_forest", columns="continuous", random_state=1).fit(train)
assert "categorical__flag_1" not in continuous.used_columns_
assert "numeric__a" in continuous.used_columns_

# Errors are explicit
assert raises(ValueError, AnomalyScore(method="pca").fit, train)
assert raises(ValueError, AnomalyScore(columns="some").fit, train)
broken = train.copy()
broken.loc[0, "numeric__a"] = np.nan
assert raises(ValueError, AnomalyScore(method="isolation_forest").fit, broken)
assert raises(ValueError, forest.fit_transform, train.assign(**{SCORE_COLUMN: 0.0}))

# NumPy input works too (as in the class version)
array_out = AnomalyScore(method="isolation_forest", random_state=1).fit_transform(train.to_numpy())
assert array_out.shape == (len(train), train.shape[1] + 1)

# add_anomaly_score / build_anomaly_scorer 
off = deepcopy(config)
off["anomaly"]["method"] = "none"
x_train, x_val, scorer = add_anomaly_score(train, new, off)
assert scorer is None and x_train is train and x_val is new

for method in ("isolation_forest", "lof"):
    on = deepcopy(config)
    on["anomaly"].update({"method": method, "exclude_columns": ["purchase_year"]})
    x_train, x_val, scorer = add_anomaly_score(train, new, on)
    assert scorer.method == method and scorer.random_state == config["random_state"]
    assert list(x_train.columns) == list(x_val.columns) == list(train.columns) + [SCORE_COLUMN]

# Real folds 
data = prepare_rows(load_data(config)[0], config)
validation = config["validation"]
fold = next(temporal_cv(
    data,
    validation["cv_years"][:1],
    horizon_days=validation["target_horizon_days"],
    strategy=validation["strategy"],
    train_years=validation.get("sliding_window", {}).get("train_years"),
))
year, train_fold, val_fold = fold
prepared_train, prepared_val, _ = prepare_numeric_fold(train_fold, val_fold, config)
transformer, columns = fit_feature_transformer(prepared_train, config, task="classification")
X_train = apply_feature_transformer(prepared_train, transformer, columns, config)
X_val = apply_feature_transformer(prepared_val, transformer, columns, config)

# LOF rule from class: n_neighbors must exceed the largest group of identical rows
_, counts = np.unique(X_train.round(6).to_numpy(), axis=0, return_counts=True)
assert counts.max() < settings["n_neighbors"], (
    f"Largest group of identical rows ({counts.max()}) must be below n_neighbors."
)

for method in ("isolation_forest", "lof"):
    on = deepcopy(config)
    on["anomaly"]["method"] = method
    out_train, out_val, scorer = add_anomaly_score(X_train, X_val, on)
    assert out_train.shape[1] == X_train.shape[1] + 1 and len(out_val) == len(X_val)
    assert np.isfinite(out_train[SCORE_COLUMN]).all() and np.isfinite(out_val[SCORE_COLUMN]).all()
    assert not any(c.endswith("purchase_year") for c in scorer.used_columns_)
    assert out_val[SCORE_COLUMN].max() < 20, "log1p(LOF) / s(x) must stay bounded."

# Feature selection sees the score; RFECV rebuilds it in every inner fold 
# The anomaly score is added BEFORE feature selection. RFECV rebuilds the features of
# each inner split from the raw rows, so it must also rebuild the score there (with a
# detector fitted on the inner training rows). Real run_feature_selection, only RFECV
rfecv_config = deepcopy(config)
rfecv_config["anomaly"]["method"] = "isolation_forest"
selection_settings = rfecv_config["preprocessing"]["feature_selection"]
for name, method in selection_settings["methods"].items():
    method["enabled"] = name == "rfecv"
selection_settings["methods"]["rfecv"].update({"n_estimators": 20, "step": 0.5, "inner_splits": 2})
selection_settings["consensus"].update(
    {"rejection_threshold": 1, "apply_threshold": 1, "evaluation_thresholds": [1]}
)

X_with_score, X_val_with_score, _ = add_anomaly_score(X_train, X_val, rfecv_config)
assert SCORE_COLUMN in X_with_score.columns
source = rfecv_config["preprocessing"]["date_features"]["source_column"]
target = rfecv_config["columns"]["classification_target"]


class SpyRFE(fs.RFE):
    """Records how many distinct anomaly scores every RFE fit receives."""

    distinct_scores = []

    def fit(self, X, y=None, **fit_params):
        SpyRFE.distinct_scores.append(
            X[SCORE_COLUMN].nunique() if SCORE_COLUMN in X.columns else None
        )
        return super().fit(X, y, **fit_params)


real_rfe, real_score_column = fs.RFE, fs.SCORE_COLUMN
try:
    fs.RFE = SpyRFE
    selection = run_feature_selection(
        X_with_score, train_fold[target], train_fold[source], rfecv_config,
        raw_train_fold=train_fold,
    )
    with_rebuild = list(SpyRFE.distinct_scores)

    # Control: switch the rebuild off. The old behaviour fills the column with 0.0
    SpyRFE.distinct_scores = []
    fs.SCORE_COLUMN = "__not_a_column__"
    run_feature_selection(
        X_with_score, train_fold[target], train_fold[source], rfecv_config,
        raw_train_fold=train_fold,
    )
    without_rebuild = list(SpyRFE.distinct_scores)
finally:
    fs.RFE, fs.SCORE_COLUMN = real_rfe, real_score_column

assert with_rebuild and all(n is not None and n > 20 for n in with_rebuild), (
    f"RFECV must receive real anomaly scores in every fit: {with_rebuild}"
)
assert 1 in without_rebuild, "Control: without the rebuild the inner score is constant."

assert SCORE_COLUMN in set(selection["feature"]), "The selection must judge the score."
kept_train, kept_val = apply_consensus_selection(
    X_with_score, X_val_with_score, selection, rfecv_config
)
assert list(kept_train.columns) == list(kept_val.columns)
print(f"RFECV: anomaly_score judged with real scores in {len(with_rebuild)} fits "
      f"(control without rebuild: {sorted(set(without_rebuild))})")

# Prediction path: the fitted pipeline scores new customers 
for method in ("none", "isolation_forest", "lof"):
    on = deepcopy(config)
    on["anomaly"]["method"] = method
    pipeline = TaskPipeline(on, "classification", "logistic_regression").fit(train_fold)
    assert (pipeline.anomaly is None) == (method == "none")

    probabilities = pipeline.predict(val_fold.head(25))
    assert probabilities.shape == (25,) and ((0 <= probabilities) & (probabilities <= 1)).all()
    single = pipeline.predict(val_fold.head(1))
    assert np.allclose(single, probabilities[:1]), "One new customer must score like in a batch."
    print(f"{year}: TaskPipeline with anomaly={method} predicts new customers")

print("Anomaly score checks passed.")
