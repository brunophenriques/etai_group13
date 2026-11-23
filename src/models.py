from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Regression
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, Ridge
from xgboost import XGBRegressor

# Classification models
MODEL_REGISTRY = {
    "dummy": DummyClassifier,
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
    "adaboost": AdaBoostClassifier,
    "xgboost": XGBClassifier,
    "lightgbm": LGBMClassifier,
    "catboost": CatBoostClassifier,
}

# Regression models
REGRESSION_REGISTRY = {
    "dummy_mean": DummyRegressor,      # strategy: mean   (reference for RMSE / R2)
    "dummy_median": DummyRegressor,    # strategy: median (reference for MAE)
    "ridge": Ridge,
    "lasso": Lasso,
    "random_forest": RandomForestRegressor,
    "hist_gradient_boosting": HistGradientBoostingRegressor,
    "adaboost": AdaBoostRegressor,
    "xgboost": XGBRegressor,
}

TARGET_TRANSFORMS = ("euros", "log")


def build_models(config: dict) -> dict:
    """Build enabled models for the task set in modeling.task."""

    modeling = config["modeling"]

    if modeling["task"] == "regression":
        return build_regression_models(config)

    if modeling["task"] != "classification":
        raise ValueError(
            "modeling.task must be 'classification' or 'regression'."
        )

    models = {}

    for name, settings in modeling["models"].items():
        if not settings.get("enabled", False):
            continue

        if name not in MODEL_REGISTRY:
            raise ValueError(
                f"Unsupported model: {name}"
            )

        model_class = MODEL_REGISTRY[name]
        parameters = settings.get("parameters", {}).copy()

        if (
            "random_state" in model_class().get_params()
            and "random_state" not in parameters
        ):
            parameters["random_state"] = config["random_state"]

        models[name] = model_class(**parameters)

    if not models:
        raise ValueError("No models enabled in config.yaml")

    return models


def build_regression_models(config: dict) -> dict:
    """Build enabled regression models from modeling.regression_models."""

    models = {}

    for name, settings in config["modeling"]["regression_models"].items():
        if not settings.get("enabled", False):
            continue

        if name not in REGRESSION_REGISTRY:
            raise ValueError(
                f"Unsupported regression model: {name}. "
                f"Options: {list(REGRESSION_REGISTRY)}"
            )

        model_class = REGRESSION_REGISTRY[name]
        parameters = settings.get("parameters", {}).copy()

        if (
            "random_state" in model_class().get_params()
            and "random_state" not in parameters
        ):
            parameters["random_state"] = config["random_state"]

        models[name] = model_class(**parameters)

    if not models:
        raise ValueError("No regression models enabled in config.yaml")

    return models


def with_target_transform(model, form: str):
    """Predict spend in euros ('euros') or log1p(spend) converted back to euros ('log')."""

    if form == "euros":
        return model
    if form == "log":
        return TransformedTargetRegressor(
            regressor=model, func=np.log1p, inverse_func=np.expm1
        )
    raise ValueError(
        f"Invalid target transform: {form}. Options: {TARGET_TRANSFORMS}"
    )
