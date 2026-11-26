"""Behavioral checks for the shared fitted feature lifecycle, without model search."""

from copy import deepcopy
import pickle

import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_config, load_development_data
from src.preprocessing import prepare_rows
from src.imputation import TaskPipeline
from src.fold_preprocessing import NumericFeaturePreprocessor
from src.pipeline_preprocessing import FittedTaskFeatures
import src.pipeline_preprocessing as shared
from src.tune_models import prepare_matrices, PREPROCESSING_FILES, preprocessing_version
from src.outliers import build_anomaly_scorer
from src.feature_transformer import fit_feature_transformer, apply_feature_transformer


@pytest.fixture(scope="module")
def development():
    config = load_config()
    data = prepare_rows(load_development_data(config), config)
    date = config["preprocessing"]["date_features"]["source_column"]
    return data.sort_values(date, kind="stable").iloc[:2200].copy()


@pytest.fixture
def config():
    return load_config()


def selection_spy(monkeypatch):
    calls = []
    def select(X, y, dates, config, raw_train_fold=None):
        calls.append((X.copy(), y.copy(), dates.copy(), raw_train_fold.copy()))
        # Known consensus decisions exercise application and persisted column order.
        return pd.DataFrame({"feature": X.columns,
                             "rejected_votes": [0 if i % 2 else 4 for i in range(X.shape[1])]})
    monkeypatch.setattr(shared, "run_feature_selection", select)
    return calls


@pytest.mark.parametrize("method", ["none", "isolation_forest", "lof"])
@pytest.mark.parametrize("task,model", [("classification", "logistic_regression"),
                                      ("regression", "ridge")])
def test_pipeline_label_free_prediction_and_serialization(development, config, monkeypatch,
                                                         task, model, method):
    calls = selection_spy(monkeypatch)
    config["anomaly"]["method"] = method
    train, query = development.iloc[:1800], development.iloc[1800:1808]
    targets = [config["columns"]["classification_target"], config["columns"]["regression_target"]]
    inputs = query.drop(columns=targets)
    pipeline = TaskPipeline(config, task, model).fit(train)
    expected = pipeline.predict(inputs)
    np.testing.assert_allclose(expected, pipeline.predict(query))
    assert len(calls) == (1 if task == "classification" else 0)
    assert list(pipeline.features.transform(inputs)) == pipeline.selected_features_
    assert pipeline.model.n_features_in_ == len(pipeline.selected_features_)
    for i in range(len(inputs)):
        np.testing.assert_allclose(expected[i:i+1], pipeline.predict(inputs.iloc[[i]]))
    restored = pickle.loads(pickle.dumps(pipeline))
    np.testing.assert_allclose(expected, restored.predict(inputs))
    assert np.isfinite(expected).all()
    # Prediction must not fit selection again or update engineered medians.
    pd.testing.assert_series_equal(pipeline.features.numeric.medians_, restored.features.numeric.medians_)
    assert len(calls) == (1 if task == "classification" else 0)


def test_tuning_selection_fitted_separately_on_training_only(development, config, monkeypatch):
    calls = selection_spy(monkeypatch)
    config["anomaly"]["method"] = "isolation_forest"
    target = config["columns"]["classification_target"]
    for end in [1400, 1800]:
        train = development.iloc[:end]
        validation = development.iloc[end:end+30].copy()
        expected_train, expected_val, _, _ = prepare_matrices(train, validation, config)
        X, y, dates, raw = calls[-1]
        assert raw.index.equals(train.index)
        assert not raw.index.intersection(validation.index).size
        pd.testing.assert_series_equal(y, train[target])
        assert "anomaly_score" in X
        assert list(expected_train) == list(expected_val)
        assert expected_train.shape[1] < X.shape[1]
        # Perturb validation labels and predictors: training features and decisions stay fixed.
        validation[target] = 1 - validation[target]
        validation["spend_90d"] = 1e12
        actual_train, _, _, _ = prepare_matrices(train, validation, config)
        pd.testing.assert_frame_equal(actual_train, expected_train)
    assert len(calls) == 4
    assert len(calls[0][3]) != len(calls[2][3])


def test_engineering_medians_are_training_only(development, config):
    train = development.iloc[:100].copy()
    query = development.iloc[100:103].copy()
    query["prior_mean_gap_days"] = 0.0
    fitted = NumericFeaturePreprocessor(config).fit(train)
    medians = fitted.medians_.copy()
    result = fitted.transform(query)
    assert result["recency_ratio"].eq(medians["recency_ratio"]).all()
    pd.testing.assert_series_equal(medians, fitted.medians_)
    pd.testing.assert_frame_equal(result.iloc[[0]], fitted.transform(query.iloc[[0]]))


@pytest.mark.parametrize("method", ["isolation_forest", "lof"])
def test_regression_detector_matches_evaluation_policy(development, config, monkeypatch, method):
    def forbidden(*args, **kwargs):
        raise AssertionError("Classification selection must not run for regression")
    monkeypatch.setattr(shared, "run_feature_selection", forbidden)
    config["anomaly"]["method"] = method
    train, query = development.iloc[:1800], development.iloc[1800:1810]
    fitted = FittedTaskFeatures(config, "regression")
    actual_train = fitted.fit_transform(train)
    prepared = fitted.numeric.transform(train)
    columns = config["columns"]
    mask = train[columns["classification_target"]].eq(1) & train[columns["regression_target"]].notna()
    assert 0 < mask.sum() < len(train)
    transformer, names = fit_feature_transformer(prepared.loc[mask], config, task="regression")
    matrix = apply_feature_transformer(prepared.loc[mask], transformer, names, config)
    scorer = build_anomaly_scorer(config)
    expected = scorer.fit_transform(matrix)
    pd.testing.assert_frame_equal(actual_train, expected)
    expected_query = scorer.transform(apply_feature_transformer(
        fitted.numeric.transform(query), transformer, names, config))
    pd.testing.assert_frame_equal(fitted.transform(query), expected_query)


def test_preprocessing_fingerprint_covers_shared_stages(monkeypatch):
    assert {"src/feature_engineering.py", "src/feature_selection.py",
            "src/pipeline_preprocessing.py", "src/outliers.py",
            "src/fold_preprocessing.py", "src/tuning_cache.py"} <= set(PREPROCESSING_FILES)
    from pathlib import Path
    before = preprocessing_version()
    read_bytes = Path.read_bytes
    def changed_source(path):
        content = read_bytes(path)
        return content + b"\n# changed selector" if path.name == "feature_selection.py" else content
    monkeypatch.setattr(Path, "read_bytes", changed_source)
    assert preprocessing_version() != before


def test_rfecv_refits_numeric_features_and_detector_per_inner_split(development, config, monkeypatch):
    import src.feature_selection as fs
    raw = development.iloc[:1800]
    config["anomaly"]["method"] = "isolation_forest"
    config["preprocessing"]["feature_selection"]["enabled"] = False
    fitted = FittedTaskFeatures(config, "classification")
    X = fitted.fit_transform(raw)
    numeric_calls, detector_calls = [], []
    real_numeric, real_detector = fs.prepare_numeric_fold, fs.build_anomaly_scorer
    def numeric(train, val, cfg, task=None):
        assert train.purchase_date.max() + pd.Timedelta(days=90) < val.purchase_date.min()
        assert not train.index.intersection(val.index).size
        result = real_numeric(train, val, cfg, task=task)
        numeric_calls.append((train.index.copy(), val.index.copy()))
        return result
    def detector(cfg):
        scorer = real_detector(cfg)
        real_fit = scorer.fit
        def fit(matrix, y=None):
            detector_calls.append(matrix.index.copy())
            return real_fit(matrix, y)
        scorer.fit = fit
        return scorer
    monkeypatch.setattr(fs, "prepare_numeric_fold", numeric)
    monkeypatch.setattr(fs, "build_anomaly_scorer", detector)
    settings = deepcopy(config["preprocessing"]["feature_selection"]["methods"]["rfecv"])
    settings.update(n_estimators=10, step=0.5, inner_splits=2)
    result = fs.select_rfecv(X, raw.repeat_purchase_90d, raw.purchase_date,
                             raw, config, settings, config["random_state"])
    assert len(numeric_calls) == len(detector_calls) == 2
    for (train_index, val_index), fit_index in zip(numeric_calls, detector_calls):
        assert fit_index.equals(train_index)
        assert not fit_index.intersection(val_index).size
    assert result.index.equals(X.columns)


def test_tuning_fails_closed_when_rfecv_cannot_make_valid_splits(development, config, tmp_path):
    from src.tune_models import get_cached_matrices
    from src.tuning_cache import TuningCache
    # Configured row minima cannot support two embargoed splits on this portion.
    train, val = development.iloc[:675], development.iloc[675:705]
    with pytest.raises(ValueError, match="Not enough valid temporal inner splits"):
        get_cached_matrices(TuningCache(tmp_path), 2020, 1, train, val, config, "test")
    assert not list(tmp_path.glob("*.joblib"))


@pytest.mark.parametrize("disabled", ["feature_selection", "consensus"])
def test_selection_disable_flags_are_respected(development, config, monkeypatch, disabled):
    calls = selection_spy(monkeypatch)
    settings = config["preprocessing"]["feature_selection"]
    if disabled == "feature_selection":
        settings["enabled"] = False
    else:
        settings["consensus"]["enabled"] = False
    fitted = FittedTaskFeatures(config, "classification")
    matrix = fitted.fit_transform(development.iloc[:100])
    assert calls == []
    assert fitted.selection_ is None
    assert list(matrix) == list(fitted.transform(development.iloc[100:105]))


def test_development_loader_never_opens_test_and_excludes_holdout(monkeypatch, config):
    import src.data_loader as dl
    opened = []
    frame = pd.DataFrame({"purchase_date": ["2023-12-31", "2024-01-01", "2025-01-01"],
                          "marker": ["development", "holdout", "later"],
                          "salesperson": ["001", "002", "A"]}, index=[10, 25, 50])
    def read(path, **kwargs):
        opened.append(path)
        assert "chunksize" not in kwargs
        return frame.copy()
    monkeypatch.setattr(dl.pd, "read_csv", read)
    result = dl.load_development_data(config)
    pd.testing.assert_frame_equal(result, frame.iloc[[0]])
    assert opened == [dl.PROJECT_ROOT / config["paths"]["train"]]


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_evaluation_uses_same_training_matrix_as_reusable_pipeline(development, config,
                                                                 monkeypatch, capsys, task):
    from sklearn.dummy import DummyClassifier, DummyRegressor
    import src.evaluate_models as evaluation
    selection_spy(monkeypatch)
    captured = []
    if task == "classification":
        class RecordingModel(DummyClassifier):
            def fit(self, X, y, sample_weight=None):
                assert y.notna().all() and y.isin([0, 1]).all()
                assert X.index.equals(y.index)
                captured.append(X.copy())
                return super().fit(X, y, sample_weight=sample_weight)
        model = RecordingModel(strategy="prior")
        name = "dummy"
    else:
        class RecordingModel(DummyRegressor):
            def fit(self, X, y, sample_weight=None):
                captured.append(X.copy())
                return super().fit(X, y, sample_weight=sample_weight)
        model = RecordingModel(strategy="mean")
        name = "dummy_mean"
    config["modeling"]["task"] = task
    config["anomaly"]["method"] = "lof"
    train, val = development.iloc[:1800].copy(), development.iloc[1800:1900]
    if task == "classification":
        train.loc[train.index[:2], config["columns"]["classification_target"]] = np.nan
    monkeypatch.setattr(evaluation, "load_development_data", lambda cfg: development)
    monkeypatch.setattr(evaluation, "temporal_cv", lambda *a, **k: iter([(2020, train, val)]))
    monkeypatch.setattr(evaluation, "build_models", lambda cfg: {name: model})
    monkeypatch.setattr(evaluation, "save_results", lambda *a, **k: None)
    evaluation.run_evaluation(config=config)
    reference = FittedTaskFeatures(config, task)
    expected = reference.fit_transform(train)
    output = capsys.readouterr().out
    count = len(config["preprocessing"]["feature_engineering"]["features"])
    assert f"Feature Engineering: {count}/{count} features" in output
    assert "Anomaly score: lof" in output
    assert f"Feature Selection: {reference.n_features_before_selection_} -> {expected.shape[1]} features" in output
    assert len(captured) == 1
    pd.testing.assert_frame_equal(captured[0], expected)
    if task == "classification":
        tuning_train, _, _, _ = prepare_matrices(train, val, config)
        pd.testing.assert_frame_equal(tuning_train, expected)


@pytest.mark.parametrize("selection_enabled", [False, True])
def test_partial_classification_labels_never_reach_supervised_steps(
        development, config, monkeypatch, selection_enabled):
    from sklearn.linear_model import LogisticRegression
    calls = selection_spy(monkeypatch)
    config["preprocessing"]["feature_selection"]["enabled"] = selection_enabled
    train = development.iloc[:1800].copy()
    target = config["columns"]["classification_target"]
    train[target] = train[target].astype(float)
    train.loc[train.index[:3], target] = [np.nan, 2, np.nan]
    labelled = train[target].isin([0, 1])
    fitted_labels = []
    real_fit = LogisticRegression.fit
    def fit(model, X, y, *args, **kwargs):
        assert y.notna().all() and y.isin([0, 1]).all()
        assert X.index.equals(y.index)
        fitted_labels.append(y.copy())
        return real_fit(model, X, y, *args, **kwargs)
    monkeypatch.setattr(LogisticRegression, "fit", fit)
    pipeline = TaskPipeline(config, "classification", "logistic_regression").fit(train)
    pd.testing.assert_series_equal(fitted_labels[0], train.loc[labelled, target])
    assert pipeline.preprocessor.history_.shape[0] == len(train)
    assert len(calls) == int(selection_enabled)
    if selection_enabled:
        _, y, _, raw = calls[0]
        pd.testing.assert_series_equal(y, train.loc[labelled, target])
        assert raw.index.equals(train.index[labelled])
    X, _, y, _ = prepare_matrices(train, development.iloc[1800:1805], config)
    assert X.index.equals(train.index[labelled])
    np.testing.assert_array_equal(y, train.loc[labelled, target].to_numpy())
    assert len(X) == len(y)
    assert len(calls) == 2 * int(selection_enabled)
    assert np.isfinite(pipeline.predict(development.iloc[1800:1805].drop(
        columns=[target, config["columns"]["regression_target"]]))).all()


def test_legacy_pipeline_state_requests_retraining():
    old = object.__new__(TaskPipeline)
    old.preprocessor = None
    old.model = None
    # Pickle contains only the old attribute layout, not a compatible feature pipeline.
    restored = pickle.loads(pickle.dumps(old))
    with pytest.raises(RuntimeError, match="Retrain the pipeline with the current code"):
        restored.predict(pd.DataFrame())


@pytest.mark.parametrize("task,model", [("classification", "logistic_regression"),
                                      ("regression", "random_forest")])
def test_pipeline_uses_configured_model_parameters(development, config, monkeypatch, task, model):
    import src.models as models
    selection_spy(monkeypatch)
    # Inspect the exact configured estimator at fit time without fitting a forest.
    seen = []
    registry = models.MODEL_REGISTRY if task == "classification" else models.REGRESSION_REGISTRY
    estimator_class = registry[model]
    def fit(estimator, X, y, **kwargs):
        seen.append(estimator.get_params())
        return estimator
    monkeypatch.setattr(estimator_class, "fit", fit)
    pipeline = TaskPipeline(config, task, model).fit(development.iloc[:1800])
    section = "models" if task == "classification" else "regression_models"
    for key, value in config["modeling"][section][model]["parameters"].items():
        assert seen[0][key] == value
        assert pipeline.model.get_params()[key] == value
