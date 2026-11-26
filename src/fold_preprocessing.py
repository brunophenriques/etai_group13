"""Coordinate fitted imputation and configured feature engineering."""

from copy import deepcopy

import numpy as np

from src.imputation import ImputationPreprocessor
from src.feature_engineering import apply_configured_feature_engineering


class NumericFeaturePreprocessor:
    """Retain training-only imputation and engineered-feature replacement state."""

    def __init__(self, config, task=None):
        self.config = deepcopy(config)
        self.task = task

    def _engineer(self, prepared):
        result = apply_configured_feature_engineering(prepared, self.config)
        if self.features_:
            result[self.features_] = result[self.features_].replace(
                [np.inf, -np.inf], np.nan
            )
        return result

    def fit(self, train):
        self.preprocessor = ImputationPreprocessor(self.config, self.task).fit(train)
        settings = self.config["preprocessing"].get("feature_engineering", {})
        self.features_ = list(settings["features"]) if settings.get("enabled", False) else []
        prepared = self._engineer(self.preprocessor.transform(train))
        self.medians_ = prepared[self.features_].median().fillna(0.0)
        return self

    def transform(self, frame):
        prepared = self._engineer(self.preprocessor.transform(frame))
        if self.features_:
            prepared[self.features_] = prepared[self.features_].fillna(self.medians_)
        return prepared


def prepare_numeric_fold(train_fold, val_fold, config, task=None):
    """Fit on training rows only and reuse the complete state for validation."""
    fitted = NumericFeaturePreprocessor(config, task).fit(train_fold)
    state = {"__preprocessor__": fitted.preprocessor,
             "__numeric_features__": fitted}
    if fitted.features_:
        state["__feature_engineering_medians__"] = fitted.medians_
    return fitted.transform(train_fold), fitted.transform(val_fold), state
