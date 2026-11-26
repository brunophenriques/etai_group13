"""Executable behavioural checks for imputation and conditional regression."""
from copy import deepcopy
from pathlib import Path
import tempfile

import joblib
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from src.data_loader import load_config, load_development_data
from src.feature_transformer import fit_feature_transformer, apply_feature_transformer
from src.imputation import ImputationPreprocessor
from src.imputation import TaskPipeline, candidate_config, candidates, select

cfg = load_config()
raw = load_development_data(cfg)
dated = raw.loc[pd.to_datetime(raw.purchase_date).dt.year.lt(2020)].copy()
train = dated.iloc[:150].copy()
query = dated.iloc[150:154].copy()
query.index = [9, 9, 2, 0]  # row labels need not be unique or ordered
query.iloc[0, query.columns.get_loc("purchase_date")] = None
query.iloc[0, query.columns.get_loc("customer_id")] = -1
query["payment_type"] = None
query["spend_365d"] = np.nan

for method in ["median", "knn", "iterative"]:
    config = candidate_config(cfg, {"numeric": method, "k": 3, "weights": "uniform", "categorical": "history"})
    prep = ImputationPreprocessor(config).fit(train)
    transformed = prep.transform(query)
    assert transformed.index.equals(query.index)
    assert transformed.ID.equals(query.ID)
    assert transformed[prep.numeric_ + prep.categorical_].notna().all().all()
    assert transformed.iloc[0].payment_type == "Unknown"
    # Each prediction is independent of other query rows, including their order.
    for i in range(len(query)):
        assert_frame_equal(transformed.iloc[[i]], prep.transform(query.iloc[[i]]))
    changed = query.copy()
    changed.iloc[1:, changed.columns.get_loc("spend_90d")] = 1e8
    assert_frame_equal(transformed.iloc[[0]], prep.transform(changed).iloc[[0]])
    # Neither target affects the learned imputers or history.
    altered_train = train.copy()
    altered_train["repeat_purchase_90d"] = 1 - train.repeat_purchase_90d
    altered_train["spend_next_90d"] = 1e8
    other = ImputationPreprocessor(config).fit(altered_train)
    assert_frame_equal(transformed, other.transform(query))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "prep.joblib"
        joblib.dump(prep, path)
        assert_frame_equal(transformed, joblib.load(path).transform(query))
    encoder, columns = fit_feature_transformer(prep.transform(train), config)
    X = apply_feature_transformer(transformed, encoder, columns, config)
    assert np.isfinite(X.to_numpy()).all()
    assert not {"ID", "customer_id", "repeat_purchase_90d", "spend_next_90d"}.intersection(columns)
    # Valid observed numerical values remain unchanged (explicit corrections excluded).
    for col in prep.numeric_:
        valid = query[col].notna().to_numpy(copy=True)
        flag = f"{col}_was_corrected"
        if flag in transformed:
            valid &= ~transformed[flag].astype(bool).to_numpy()
        np.testing.assert_allclose(transformed[col].to_numpy()[valid], query[col].to_numpy()[valid])

# Structural placeholders cannot dominate the median for unknown histories.
config = candidate_config(cfg, {"numeric": "median", "k": 0, "weights": "uniform", "categorical": "mode"})
small = train.iloc[:4].copy()
small["prior_purchase_count"] = [0., 0., 0., 3.]
small["prior_mean_gap_days"] = [np.nan, np.nan, np.nan, 100.]
small["Total Portes"] = np.nan
small["payment_type"] = None
prep = ImputationPreprocessor(config).fit(small)
assert prep.medians_["prior_mean_gap_days"] == 100.
assert "Total Portes" in prep.empty_columns_
missing = query.iloc[:1].copy()
missing["prior_purchase_count"] = 3.
missing["prior_mean_gap_days"] = np.nan
missing["Total Portes"] = np.nan
result = prep.transform(missing)
assert result.prior_mean_gap_days.iloc[0] == 100.
assert result["Total Portes"].iloc[0] == 0.
assert result["Total Portes_no_training_donors"].iloc[0] == 1
assert result.payment_type.iloc[0] == "Unknown"

# All-missing query with no distance, all-empty multivariate train, novel category.
for method in ["knn", "iterative"]:
    config["preprocessing"]["numeric_imputation"]["strategy"] = method
    prep = ImputationPreprocessor(config).fit(train)
    empty_query = query.iloc[:1].copy()
    empty_query[prep.numeric_] = np.nan
    empty_query["payment_type"] = "Never seen"
    assert prep.transform(empty_query)[prep.numeric_].notna().all().all()
    empty_train = train.copy()
    empty_train[prep.numeric_] = np.nan
    empty_prep = ImputationPreprocessor(config).fit(empty_train)
    assert empty_prep.transform(empty_query)[prep.numeric_].eq(0).all().all()

# Recency recovery is strictly past-only, excludes same-day/future records.
config["preprocessing"]["numeric_imputation"]["strategy"] = "median"
history = train.iloc[:3].copy()
history["customer_id"] = 123
history["purchase_date"] = ["2018-01-01", "2018-02-01", "2018-04-01"]
observation = query.iloc[:1].copy()
observation["customer_id"] = 123
observation["purchase_date"] = "2018-02-01"
observation["prior_purchase_count"] = 2.
observation["days_since_previous_purchase"] = np.nan
recovered = ImputationPreprocessor(config).fit(history).transform(observation)
assert recovered.days_since_previous_purchase.iloc[0] == 31

# Missing spending labels do not become training labels; pipelines serialize.
# Use the pre-2020 development history: the former 500-row fixture cannot
# support configured RFECV now that TaskPipeline correctly runs selection.
pipeline_train = dated.copy()
for task, model in [("classification", "logistic_regression"), ("regression", "ridge")]:
    pipeline = TaskPipeline(config, task, model).fit(pipeline_train)
    predicted = pipeline.predict(query)
    without_targets = query.drop(columns=["repeat_purchase_90d", "spend_next_90d"])
    np.testing.assert_allclose(predicted, pipeline.predict(without_targets))
    for i in range(len(query)):
        np.testing.assert_allclose(predicted[i:i+1], pipeline.predict(query.iloc[[i]]))
    assert np.isfinite(predicted).all()
    assert (predicted >= 0).all()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "pipeline.joblib"
        joblib.dump(pipeline, path)
        np.testing.assert_allclose(predicted, joblib.load(path).predict(query))
    if task == "regression":
        changed = pipeline_train.copy()
        changed.loc[changed.repeat_purchase_90d.eq(0), "spend_next_90d"] = 1e9
        np.testing.assert_allclose(predicted, TaskPipeline(config, task, model).fit(changed).predict(query))

# Selection prefers the simple candidate within the predeclared tolerance.
catalog = pd.DataFrame(candidates(cfg))
simple = "median_0_uniform__unknown"
complex_ = "knn_5_uniform__unknown"
scores = pd.DataFrame([
    {"candidate": simple, "task": "classification", "f1": .700},
    {"candidate": complex_, "task": "classification", "f1": .704},
    {"candidate": simple, "task": "regression", "mae": 100.9},
    {"candidate": complex_, "task": "regression", "mae": 100.0},
])
winners, _ = select(scores, catalog, cfg)
assert all(w["candidate"] == simple for w in winners.values())
print("Imputation contracts passed.")
