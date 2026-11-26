
"""Nested temporal hyperparameter tuning with fold caching."""

import hashlib
from datetime import datetime
from pathlib import Path
from pprint import pformat

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid, ParameterSampler

from src.data_loader import load_config, load_development_data
from src.preprocessing import prepare_rows
from src.validation import temporal_cv
from src.tuning_validation import make_inner_folds
from src.tuning_cache import TuningCache
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    roc_auc_score,
)
from src.models import build_models
from src.pipeline_preprocessing import FittedTaskFeatures


PREPROCESSING_FILES = [
    "src/imputation.py",
    "src/preprocessing.py",
    "src/fold_preprocessing.py",
    "src/feature_transformer.py",
    "src/validation.py",
    "src/tuning_validation.py",
    "src/data_loader.py",
    "src/outliers.py",
    "src/feature_engineering.py",
    "src/feature_selection.py",
    "src/pipeline_preprocessing.py",
    "src/tune_models.py",
    "src/tuning_cache.py",
]


def preprocessing_version():
    """Hash preprocessing source code to invalidate old caches."""

    digest = hashlib.sha256()

    for filename in PREPROCESSING_FILES:
        path = Path(filename)
        digest.update(filename.encode("utf-8"))
        digest.update(path.read_bytes())

    digest.update(
        f"sklearn={__import__('sklearn').__version__}".encode()
    )

    return digest.hexdigest()


def prepare_matrices(train, val, config):
    """Fit the full preprocessing exclusively on training data."""

    fitted = FittedTaskFeatures(config, "classification")
    x_train = fitted.fit_transform(train)
    x_val = fitted.transform(val)

    target = config["columns"]["classification_target"]

    y_train = train.loc[fitted.training_mask_, target].to_numpy()
    y_val = val[target].to_numpy()

    return x_train, x_val, y_train, y_val


def get_cached_matrices(
    cache, outer_year, inner_index,
    train, val, config, version
):
    """Reuse preprocessing when fold data and settings are unchanged."""

    key = cache.make_key(
        outer_year=outer_year,
        inner_index=inner_index,
        config=config,
        train_df=train,
        val_df=val,
        code_version=version,
    )

    return cache.get_or_compute(
        key,
        lambda: prepare_matrices(train, val, config),
    )


def score_candidate(base_model, params, prepared_folds, threshold):
    """Evaluate one parameter combination on all inner folds."""

    scores = []

    for x_train, x_val, y_train, y_val in prepared_folds:
        model = clone(base_model)
        model.set_params(**params)
        model.fit(x_train, y_train)

        positive_index = list(model.classes_).index(1)
        probabilities = model.predict_proba(x_val)[
            :, positive_index
        ]

        predictions = (
            probabilities >= threshold
        ).astype(int)

        scores.append(
            f1_score(y_val, predictions, zero_division=0)
        )

    return float(np.mean(scores)), scores


def make_candidates(settings, seed):
    """Generate candidates for randomized or grid search."""

    method = settings["method"]
    model_name = settings["model"]

    space = settings["search_space"][model_name][method]

    if method == "randomized":
        return list(
            ParameterSampler(
                space,
                n_iter=settings["randomized"]["n_iter"],
                random_state=seed,
            )
        )

    if method == "grid":
        return list(ParameterGrid(space))

    raise ValueError(f"Unsupported candidate method: {method}")


def suggest_optuna_parameters(trial, space):
    """Generate parameters using an Optuna trial."""

    params = {}

    for name, spec in space.items():
        kind = spec["type"]

        if kind == "categorical":
            params[name] = trial.suggest_categorical(
                name, spec["choices"]
            )

        elif kind == "int":
            params[name] = trial.suggest_int(
                name,
                spec["low"],
                spec["high"],
                step=spec.get("step", 1),
            )

        elif kind == "float":
            params[name] = trial.suggest_float(
                name,
                spec["low"],
                spec["high"],
                log=spec.get("log", False),
            )

        else:
            raise ValueError(
                f"Unsupported Optuna parameter type: {kind}"
            )

    return params


def search_parameters(
    base_model, prepared_folds, config, outer_year
):
    """Optimize F1 on inner folds only."""

    settings = config["hyperparameter_tuning"]
    method = settings["method"]
    threshold = config["modeling"]["evaluation"][
        "classification_threshold"
    ]

    seed = int(settings["random_state"])
    trials = []

    def evaluate(params):
        mean_f1, fold_scores = score_candidate(
            base_model, params, prepared_folds, threshold
        )

        trials.append({
            "outer_year": outer_year,
            "trial": len(trials) + 1,
            "inner_f1_mean": mean_f1,
            "inner_f1_std": float(np.std(fold_scores)),
            "parameters": repr(params),
        })

        print(
            f"  Trial {len(trials)} | "
            f"Inner F1: {mean_f1:.4f} | {params}"
        )

        return mean_f1

    if method in {"randomized", "grid"}:
        candidates = make_candidates(settings, seed)

        best_score = -1.0
        best_params = None

        for params in candidates:
            score = evaluate(params)

            if score > best_score:
                best_score = score
                best_params = params.copy()

    elif method == "optuna":
        try:
            import optuna
        except ImportError as exc:
            raise ImportError(
                "Install Optuna with: pip install optuna"
            ) from exc

        space = settings["search_space"][
            settings["model"]
        ]["optuna"]

        sampler = optuna.samplers.TPESampler(seed=seed)

        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
        )

        def objective(trial):
            params = suggest_optuna_parameters(
                trial, space
            )
            return evaluate(params)

        study.optimize(
            objective,
            n_trials=settings["optuna"]["n_trials"],
            n_jobs=1,
        )

        best_score = study.best_value
        best_params = study.best_params.copy()

    else:
        raise ValueError(
            f"Unsupported tuning method: {method}"
        )

    return best_params, best_score, trials


def positive_probabilities(model, x):
    """Return probabilities for the positive class."""

    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError("Positive class 1 not found.")

    positive_index = classes.index(1)

    return model.predict_proba(x)[:, positive_index]


def classification_metrics(y_true, probabilities, threshold):
    """Evaluate classification at a specified threshold."""

    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "f1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "pr_auc": average_precision_score(
            y_true, probabilities
        ),
        "roc_auc": (
            roc_auc_score(y_true, probabilities)
            if len(np.unique(y_true)) == 2
            else float("nan")
        ),
    }

    return {
        name: float(value)
        for name, value in metrics.items()
    }


def optimize_threshold(
    base_model,
    best_params,
    prepared_folds,
    config,
):
    """
    Select threshold using inner validation folds only.

    Hyperparameters are already selected.
    The outer validation set is never accessed.
    """

    settings = config["hyperparameter_tuning"][
        "threshold_optimization"
    ]

    default_threshold = config["modeling"][
        "evaluation"
    ]["classification_threshold"]

    if not settings["enabled"]:
        return float(default_threshold), float("nan")

    if settings["metric"] != "f1":
        raise ValueError("Only F1 threshold tuning is supported.")

    start = float(settings["min_threshold"])
    end = float(settings["max_threshold"])
    step = float(settings["step"])

    if step <= 0 or not 0 < start <= end < 1:
        raise ValueError("Invalid threshold search range.")

    # Fit one model per inner fold, not one per threshold.
    cached_predictions = []

    for x_train, x_val, y_train, y_val in prepared_folds:
        model = clone(base_model)
        model.set_params(**best_params)
        model.fit(x_train, y_train)

        probabilities = positive_probabilities(model, x_val)

        cached_predictions.append(
            (y_val, probabilities)
        )

    thresholds = np.arange(
        start, end + step / 2, step
    )

    best_threshold = float(default_threshold)
    best_score = -1.0

    for threshold in thresholds:
        threshold = float(threshold)

        fold_scores = [
            f1_score(
                y_val,
                (probabilities >= threshold).astype(int),
                zero_division=0,
            )
            for y_val, probabilities in cached_predictions
        ]

        score = float(np.mean(fold_scores))

        # Prefer the threshold closer to the default on ties.
        if (
            score > best_score + 1e-12
            or (
                abs(score - best_score) <= 1e-12
                and abs(threshold - default_threshold)
                < abs(best_threshold - default_threshold)
            )
        ):
            best_score = score
            best_threshold = threshold

    return round(best_threshold, 4), best_score



def run_tuning(config=None):
    """Run nested temporal CV with baseline and threshold comparison."""
    if config is None:
        config = load_config()
    settings = config["hyperparameter_tuning"]

    if config["modeling"]["task"] != "classification":
        raise ValueError("This tuning engine supports classification only.")

    if not settings["enabled"]:
        raise ValueError("Hyperparameter tuning is disabled.")

    if not settings["nested_cv"]["enabled"]:
        raise ValueError("Nested CV must be enabled.")

    if settings["scoring"] != "f1":
        raise ValueError("This tuning engine currently supports F1.")

    model_name = settings["model"]
    models = build_models(config)

    if model_name not in models:
        raise ValueError(f"Model not enabled: {model_name}")

    base_model = models[model_name]

    df = load_development_data(config)
    df = prepare_rows(df, config)

    cache = TuningCache()
    version = preprocessing_version()

    timestamp = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S-%f"
    )

    output_dir = (
        Path(config["paths"]["results"])
        / f"tuning_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    outer_results = []
    all_trials = []

    default_threshold = float(
        config["modeling"]["evaluation"][
            "classification_threshold"
        ]
    )

    outer_folds = temporal_cv(
        df,
        config["validation"]["cv_years"],
        config["validation"]["target_horizon_days"],
        strategy=config["validation"]["strategy"],
        train_years=config["validation"]
            .get("sliding_window", {})
            .get("train_years"),
    )

    for year, outer_train, outer_val in outer_folds:
        print(f"\n{'=' * 60}")
        print(f"OUTER FOLD {year}")
        print(f"{'=' * 60}")

        # 1. Build and cache INNER fold preprocessing
        inner_folds = make_inner_folds(
            outer_train, config
        )

        prepared_folds = []

        for index, fold in enumerate(inner_folds, start=1):
            matrices = get_cached_matrices(
                cache=cache,
                outer_year=year,
                inner_index=index,
                train=fold["train"],
                val=fold["val"],
                config=config,
                version=version,
            )

            prepared_folds.append(matrices)

        # 2. Select hyperparameters using INNER CV
        best_params, inner_score, trials = search_parameters(
            base_model,
            prepared_folds,
            config,
            year,
        )

        all_trials.extend(trials)

        # 3. Select threshold using INNER CV
        best_threshold, threshold_inner_f1 = (
            optimize_threshold(
                base_model,
                best_params,
                prepared_folds,
                config,
            )
        )

        print(
            f"\nSelected threshold: {best_threshold:.4f}"
            f" | Inner threshold F1: {threshold_inner_f1:.4f}"
        )

        # 4. Prepare OUTER train and validation
        x_train, x_val, y_train, y_val = (
            prepare_matrices(
                outer_train,
                outer_val,
                config,
            )
        )

        # 5. Fit original fixed-parameter baseline
        baseline_model = clone(base_model)
        baseline_model.fit(x_train, y_train)

        baseline_probabilities = positive_probabilities(
            baseline_model, x_val
        )

        baseline_metrics = classification_metrics(
            y_val,
            baseline_probabilities,
            default_threshold,
        )

        # 6. Fit tuned model on OUTER training data
        tuned_model = clone(base_model)
        tuned_model.set_params(**best_params)
        tuned_model.fit(x_train, y_train)

        tuned_probabilities = positive_probabilities(
            tuned_model, x_val
        )

        # 7. Evaluate tuning with the original threshold
        tuned_fixed_metrics = classification_metrics(
            y_val,
            tuned_probabilities,
            default_threshold,
        )

        # 8. Evaluate tuning with the selected threshold
        tuned_optimized_metrics = classification_metrics(
            y_val,
            tuned_probabilities,
            best_threshold,
        )

        # 9. Save a separate row for each approach
        common = {
            "outer_year": year,
            "model": model_name,
            "method": settings["method"],
            "inner_folds": len(inner_folds),
            "train_rows": len(outer_train),
            "validation_rows": len(outer_val),
        }

        experiments = [
            (
                "baseline",
                default_threshold,
                baseline_metrics,
            ),
            (
                "tuned_fixed_threshold",
                default_threshold,
                tuned_fixed_metrics,
            ),
            (
                "tuned_optimized_threshold",
                best_threshold,
                tuned_optimized_metrics,
            ),
        ]

        for experiment, threshold, metrics in experiments:
            outer_results.append({
                **common,
                "experiment": experiment,
                "threshold": threshold,
                **metrics,
                "inner_f1": (
                    float("nan")
                    if experiment == "baseline"
                    else inner_score
                ),
                "threshold_inner_f1": (
                    float("nan")
                    if experiment != "tuned_optimized_threshold"
                    else threshold_inner_f1
                ),
                "best_parameters": (
                    repr(base_model.get_params())
                    if experiment == "baseline"
                    else repr(best_params)
                ),
            })

        # 10. Incremental checkpoint
        pd.DataFrame(outer_results).to_csv(
            output_dir / "outer_results.csv",
            index=False,
        )

        pd.DataFrame(all_trials).to_csv(
            output_dir / "trials.csv",
            index=False,
        )

        print(f"\nOUTER {year} RESULTS")
        print(
            f"Baseline F1: {baseline_metrics['f1']:.4f}"
        )
        print(
            "Tuned F1 (0.5): "
            f"{tuned_fixed_metrics['f1']:.4f}"
        )
        print(
            "Tuned F1 (optimized threshold): "
            f"{tuned_optimized_metrics['f1']:.4f}"
        )

    # 11. Aggregate metrics across outer folds
    results_df = pd.DataFrame(outer_results)

    metrics = [
        "f1", "precision", "recall",
        "pr_auc", "roc_auc",
    ]

    summary = (
        results_df
        .groupby("experiment", sort=False)[metrics]
        .agg(["mean", "std"])
    )

    summary.columns = [
        f"{metric}_{stat}"
        for metric, stat in summary.columns
    ]

    summary = summary.reset_index()

    # 12. Human-readable report
    with (
        output_dir / "tuning_summary.txt"
    ).open("w", encoding="utf-8") as file:
        file.write("ETAI GROUP 13 — NESTED CV TUNING\n")
        file.write("=" * 70 + "\n\n")
        file.write(f"Model: {model_name}\n")
        file.write(f"Method: {settings['method']}\n")
        file.write(
            f"Default threshold: {default_threshold}\n\n"
        )

        file.write("CROSS-VALIDATION SUMMARY\n")
        file.write("-" * 70 + "\n")
        file.write(
            summary.round(4).to_string(index=False)
        )

        file.write("\n\nOUTER FOLD RESULTS\n")
        file.write("-" * 70 + "\n")
        file.write(
            results_df.round(4).to_string(index=False)
        )

        file.write("\n\nFULL CONFIGURATION\n")
        file.write("-" * 70 + "\n")
        file.write(pformat(config, sort_dicts=False))
        file.write("\n")

        file.write("\nMETHODOLOGY NOTES\n")
        file.write("-" * 70 + "\n")
        file.write(
            "Hyperparameters and thresholds are selected "
            "using inner temporal folds only.\n"
            "The same inner folds are used for both selections; "
            "inner scores are selection scores, not unbiased "
            "generalization estimates.\n"
            "All preprocessing is fitted within each training fold.\n"
            "The outer fold evaluates the complete selection "
            "procedure on unseen later observations.\n"
            "The 2024 holdout remains excluded.\n"
        )

    summary.to_csv(
        output_dir / "comparison_summary.csv",
        index=False,
    )

    print("\n=== NESTED CV COMPLETE ===")
    print(summary[["experiment", "f1_mean", "f1_std"]])
    print(f"\nResults: {output_dir.resolve()}")

    return results_df


def run_all_tuning():
    """Run Optuna for multiple models and sliding windows."""

    import copy

    base_config = load_config()

    models = [
        "random_forest",
        "catboost",
        "xgboost",
        "lightgbm",
        "adaboost",
        "gradient_boosting",
        "hist_gradient_boosting",
        "logistic_regression",
    ]

    windows = [3, 2]

    for years in windows:
        for model_name in models:
            config = copy.deepcopy(base_config)

            # Outer sliding window
            config["validation"]["strategy"] = "sliding_window"
            config["validation"]["sliding_window"]["train_years"] = years

            # Inner sliding window
            config["hyperparameter_tuning"]["nested_cv"]["outer"][
                "strategy"
            ] = "sliding_window"

            config["hyperparameter_tuning"]["nested_cv"]["inner"][
                "strategy"
            ] = "sliding_window"

            # Model and search method
            config["hyperparameter_tuning"]["model"] = model_name
            config["hyperparameter_tuning"]["method"] = "optuna"

            # Small tuning budget
            config["hyperparameter_tuning"]["optuna"]["n_trials"] = 15

            print(f"\n{'#' * 65}")
            print(f"MODEL: {model_name}")
            print(f"SLIDING WINDOW: {years} years")
            print(f"{'#' * 65}\n")

            run_tuning(config)


if __name__ == "__main__":
    run_tuning()
