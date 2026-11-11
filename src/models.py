
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


def build_models(config: dict) -> dict:
    """Build enabled classification models from config."""

    modeling = config["modeling"]

    if modeling["task"] != "classification":
        raise ValueError(
            "Only classification is currently supported."
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
