"""Shared fitted model features for evaluation, tuning and label-free prediction."""

from copy import deepcopy

from src.feature_transformer import fit_feature_transformer, apply_feature_transformer
from src.fold_preprocessing import NumericFeaturePreprocessor
from src.outliers import build_anomaly_scorer
from src.feature_selection import run_feature_selection, apply_consensus_selection


class FittedTaskFeatures:
    """Fit each learned stage once on the current training portion.

    Regression policy: imputation and engineering medians use the whole training
    fold; encoding/scaling and anomaly scoring use known-spend returners only.
    Consensus selectors currently support classification only.
    """

    def __init__(self, config, task):
        if task not in {"classification", "regression"}:
            raise ValueError("Unsupported task: " + task)
        self.config = deepcopy(config)
        self.task = task

    def fit_transform(self, train):
        cfg = self.config
        columns = cfg["columns"]
        self.numeric = NumericFeaturePreprocessor(cfg, self.task).fit(train)
        prepared = self.numeric.transform(train)
        if self.task == "regression":
            mask = (train[columns["classification_target"]].eq(1)
                    & train[columns["regression_target"]].notna())
        else:
            mask = train[columns["classification_target"]].isin([0, 1])
        self.training_mask_ = mask.to_numpy()
        self.training_index_ = train.index[mask]
        raw_train = train.loc[mask]
        if raw_train.empty:
            raise ValueError("No eligible training rows for " + self.task)
        prepared = prepared.loc[mask]
        self.transformer, self.columns = fit_feature_transformer(prepared, cfg, task=self.task)
        X = apply_feature_transformer(prepared, self.transformer, self.columns, cfg)
        self.anomaly = build_anomaly_scorer(cfg)
        if self.anomaly is not None:
            X = self.anomaly.fit_transform(X)
        self.n_features_before_selection_ = X.shape[1]
        self.selection_ = None
        settings = cfg["preprocessing"]["feature_selection"]
        if (self.task == "classification" and settings.get("enabled", False)
                and settings.get("consensus", {}).get("enabled", False)):
            target = columns["classification_target"]
            date = cfg["preprocessing"]["date_features"]["source_column"]
            self.selection_ = run_feature_selection(
                X, raw_train[target], raw_train[date], cfg, raw_train_fold=raw_train
            )
            X, _ = apply_consensus_selection(X, X, self.selection_, cfg)
        self.selected_features_ = list(X.columns)
        return X

    def transform(self, frame):
        """Transform predictors without inspecting labels or refitting any stage."""
        X = apply_feature_transformer(
            self.numeric.transform(frame), self.transformer, self.columns, self.config
        )
        if self.anomaly is not None:
            X = self.anomaly.transform(X)
        return X.loc[:, self.selected_features_].copy()
