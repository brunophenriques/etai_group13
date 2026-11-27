
"""Feature selection methods and consensus analysis — ETAI Group 13."""

import numpy as np
import pandas as pd

from boruta import BorutaPy
from mrmr import mrmr_classif

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import RFE
from sklearn.metrics import f1_score

from src.fold_preprocessing import prepare_numeric_fold
from src.outliers import SCORE_COLUMN, build_anomaly_scorer
from src.feature_transformer import (
    fit_feature_transformer,
    apply_feature_transformer,
)


def temporal_inner_splits(dates, config, settings):
    """Build chronological inner splits with a target-horizon embargo."""

    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    horizon = pd.Timedelta(days=config["validation"]["target_horizon_days"])

    n_splits = settings["inner_splits"]
    positions = np.linspace(
        int(len(dates) * 0.45),
        int(len(dates) * 0.80),
        n_splits,
        dtype=int,
    )
    cutoffs = dates.iloc[positions].tolist()

    splits = []

    for i, cutoff in enumerate(cutoffs):
        end = cutoffs[i + 1] if i + 1 < len(cutoffs) else None

        train_idx = np.flatnonzero(
            (dates + horizon < cutoff).to_numpy()
        )

        val_mask = dates >= cutoff
        if end is not None:
            val_mask &= dates < end

        val_idx = np.flatnonzero(val_mask.to_numpy())

        if (
            len(train_idx) >= settings["min_train_rows"]
            and len(val_idx) >= settings["min_validation_rows"]
        ):
            splits.append((train_idx, val_idx))

    if len(splits) < 2:
        raise ValueError(
            "Not enough valid temporal inner splits for RFECV. "
            "Review feature_selection.methods.rfecv in config.yaml."
        )

    return splits


def select_mrmr(X, y, settings, random_state):
    """Select features using minimum redundancy maximum relevance."""

    # Constant columns cannot contribute useful information.
    variable_columns = X.columns[X.nunique(dropna=False) > 1]

    if variable_columns.empty:
        raise ValueError("mRMR has no non-constant features.")

    k = min(settings["n_features"], len(variable_columns))

    selected = mrmr_classif(
        X=X[variable_columns],
        y=y,
        K=k,
        show_progress=False,
        n_jobs=1,
    )

    return pd.Series(
        X.columns.isin(selected),
        index=X.columns,
        dtype=bool,
    )



def select_rfecv(
    X, y, dates, raw_train_fold, config, settings, random_state
):
    """Temporal RFECV with fold-fitted preprocessing.

    Internal validation periods never participate in fitting
    imputation, feature engineering statistics, encoders or scalers.
    """

    if raw_train_fold is None:
        raise ValueError(
            "RFECV requires the original outer training fold."
        )

    if len(raw_train_fold) != len(X):
        raise ValueError("Raw and transformed training rows differ.")

    splits = temporal_inner_splits(dates, config, settings)

    estimator = RandomForestClassifier(
        n_estimators=settings["n_estimators"],
        min_samples_leaf=settings["min_samples_leaf"],
        random_state=random_state,
        n_jobs=-1,
    )

    n_features = X.shape[1]
    minimum = settings["min_features_to_select"]
    step = settings["step"]

    if not 1 <= minimum <= n_features:
        raise ValueError("Invalid min_features_to_select.")

    if step <= 0:
        raise ValueError("RFECV step must be positive.")

    if isinstance(step, float) and step > 1:
        raise ValueError("Float RFECV step must be <= 1.")

    # Candidate feature counts. The elimination step uses
    # the initial feature count, as in sklearn RFE/RFECV.
    step_size = (
        max(1, int(step * n_features))
        if isinstance(step, float)
        else int(step)
    )

    candidates = list(
        range(n_features, minimum, -step_size)
    )

    if minimum not in candidates:
        candidates.append(minimum)

    # Preprocess each inner split independently.
    prepared_splits = []

    for split_number, (train_idx, val_idx) in enumerate(
        splits, start=1
    ):
        raw_inner_train = raw_train_fold.iloc[train_idx]
        raw_inner_val = raw_train_fold.iloc[val_idx]

        y_inner_train = y.iloc[train_idx]
        y_inner_val = y.iloc[val_idx]

        if (
            y_inner_train.nunique() != 2
            or y_inner_val.nunique() != 2
        ):
            raise ValueError(
                f"Only one target class in inner split {split_number}."
            )

        inner_train, inner_val, _ = prepare_numeric_fold(
            raw_inner_train,
            raw_inner_val,
            config,
            task="classification",
        )

        transformer, columns = fit_feature_transformer(
            inner_train,
            config,
            task="classification",
        )

        X_inner_train = apply_feature_transformer(
            inner_train,
            transformer,
            columns,
            config,
        )

        X_inner_val = apply_feature_transformer(
            inner_val,
            transformer,
            columns,
            config,
        )

        # The outer matrix may carry the anomaly score. Rebuild it here with a
        # detector fitted on the inner training rows only; without this, the
        # reindex below would fill the column with 0.0 for every inner row.
        if SCORE_COLUMN in X.columns:
            scorer = build_anomaly_scorer(config)

            if scorer is None:
                raise ValueError(
                    f"X has a '{SCORE_COLUMN}' column but anomaly.method is none."
                )

            X_inner_train = scorer.fit_transform(X_inner_train)
            X_inner_val = scorer.transform(X_inner_val)

        # The outer training fold defines the available output
        # column names. No inner validation data is used to
        # fit preprocessing statistics.
        X_inner_train = X_inner_train.reindex(
            columns=X.columns,
            fill_value=0.0,
        )
        X_inner_val = X_inner_val.reindex(
            columns=X.columns,
            fill_value=0.0,
        )

        prepared_splits.append(
            (
                X_inner_train,
                X_inner_val,
                y_inner_train,
                y_inner_val,
            )
        )

        print(
            f"  Inner split {split_number}: "
            f"train={len(X_inner_train)}, "
            f"val={len(X_inner_val)}"
        )

    # Evaluate each candidate number of features using
    # strictly inner-training-fitted selectors.
    records = []

    for k in candidates:
        scores = []

        for (
            X_inner_train,
            X_inner_val,
            y_inner_train,
            y_inner_val,
        ) in prepared_splits:

            selector = RFE(
                estimator=estimator,
                n_features_to_select=k,
                step=step,
            )

            selector.fit(X_inner_train, y_inner_train)

            predictions = selector.predict(X_inner_val)

            scores.append(
                f1_score(
                    y_inner_val,
                    predictions,
                    zero_division=0,
                )
            )

        records.append({
            "n_features": k,
            "mean_f1": float(np.mean(scores)),
            "std_f1": float(np.std(scores)),
        })

    scores_df = pd.DataFrame(records)

    best_score = scores_df["mean_f1"].max()
    tolerance = settings["score_tolerance"]
    rule = settings["selection_rule"]

    if tolerance < 0:
        raise ValueError("score_tolerance must be nonnegative.")

    if rule == "conservative":
        eligible = scores_df[
            scores_df["mean_f1"] >= best_score - tolerance
        ]

        # Preserve more features when scores are practically tied.
        selected_k = int(eligible["n_features"].max())

    elif rule == "best":
        best_row = scores_df.loc[
            scores_df["mean_f1"].idxmax()
        ]
        selected_k = int(best_row["n_features"])

    else:
        raise ValueError(
            f"Unknown RFECV selection rule: {rule}"
        )

    print(
        f"  RFECV: best inner F1={best_score:.4f}, "
        f"selected features={selected_k}/{n_features}"
    )

    # Once k has been chosen using inner training/validation,
    # fit the final selector on the full OUTER training fold.
    final_selector = RFE(
        estimator=estimator,
        n_features_to_select=selected_k,
        step=step,
    )

    final_selector.fit(X, y)

    return pd.Series(
        final_selector.support_,
        index=X.columns,
        dtype=bool,
    )



def select_boruta(X, y, settings, random_state):
    """Boruta with Random Forest importance and shadow features."""

    estimator = RandomForestClassifier(
        n_estimators=200,
        max_depth=settings["max_depth"],
        random_state=random_state,
        n_jobs=-1,
    )

    selector = BorutaPy(
        estimator=estimator,
        n_estimators=settings["n_estimators"],
        max_iter=settings["max_iter"],
        perc=settings["perc"],
        random_state=random_state,
        verbose=0,
    )

    selector.fit(X.to_numpy(dtype=float), y.to_numpy())

    # Boruta has three outcomes:
    # True = confirmed, False = rejected, NA = tentative.
    status = pd.Series(
        pd.array(selector.support_, dtype="boolean"),
        index=X.columns,
    )
    status.loc[selector.support_weak_] = pd.NA

    return status


def select_logistic_l1(X, y, settings, random_state):
    """L1-regularized logistic regression."""

    estimator = LogisticRegression(
        l1_ratio=1,
        solver="liblinear",
        C=settings["C"],
        max_iter=settings["max_iter"],
        random_state=random_state,
    )

    estimator.fit(X, y)

    coefficients = np.abs(estimator.coef_).max(axis=0)

    return pd.Series(
        coefficients > settings["tolerance"],
        index=X.columns,
        dtype=bool,
    )


def run_feature_selection(X, y, dates, config, raw_train_fold=None):
    """Run configured selectors and return one decision per feature."""

    settings = config["preprocessing"]["feature_selection"]
    methods = settings["methods"]
    random_state = config["random_state"]

    if not X.columns.is_unique:
        raise ValueError("Feature column names must be unique.")

    if X.isna().any().any():
        raise ValueError("Feature selection received missing values.")

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Feature selection received non-finite values.")

    y = pd.Series(y, index=X.index)

    # A temporal ordering is required by RFECV.
    dates = pd.Series(pd.to_datetime(dates), index=X.index)
    order = dates.sort_values(kind="stable").index

    X = X.loc[order]
    y = y.loc[order]
    dates = dates.loc[order]

    if raw_train_fold is not None:
        raw_train_fold = raw_train_fold.loc[order]

    results = pd.DataFrame(index=X.columns)

    if methods["mrmr"]["enabled"]:
        results["mrmr"] = select_mrmr(
            X, y, methods["mrmr"], random_state
        )

    if methods["rfecv"]["enabled"]:
        results["rfecv"] = select_rfecv(
            X,
            y,
            dates,
            raw_train_fold,
            config,
            methods["rfecv"],
            random_state,
        )

    if methods["boruta"]["enabled"]:
        results["boruta"] = select_boruta(
            X, y, methods["boruta"], random_state
        )

    if methods["logistic_l1"]["enabled"]:
        results["logistic_l1"] = select_logistic_l1(
            X, y, methods["logistic_l1"], random_state
        )

    if results.shape[1] == 0:
        raise ValueError("No feature selection methods are enabled.")

    # FIX: Capture only the selection method columns BEFORE adding
    # vote columns. Otherwise selected_votes=0 is counted as False.
    method_columns = list(results.columns)

    decisions = results[method_columns].astype("boolean")

    results["selected_votes"] = decisions.eq(True).sum(axis=1)
    results["rejected_votes"] = decisions.eq(False).sum(axis=1)
    results["tentative_votes"] = decisions.isna().sum(axis=1)

    # Every feature must have exactly one outcome per active method.
    total_votes = (
        results["selected_votes"]
        + results["rejected_votes"]
        + results["tentative_votes"]
    )

    if not total_votes.eq(len(method_columns)).all():
        raise AssertionError(
            "Inconsistent feature selection votes."
        )

    consensus = settings["consensus"]

    if consensus["enabled"]:
        threshold = consensus["rejection_threshold"]

        if not 1 <= threshold <= len(method_columns):
            raise ValueError(
                "Invalid consensus rejection threshold."
            )

        # Tentative decisions are not counted as rejections.
        results["remove_candidate"] = (
            results["rejected_votes"] >= threshold
        )
    else:
        results["remove_candidate"] = False

    results.index.name = "feature"

    return results.reset_index()


def apply_consensus_selection(X_train, X_val, selection, config):
    """Apply configured consensus decisions to train and validation.

    The selection decisions must have been fitted on training data only.
    No selectors are fitted here.
    """

    settings = config["preprocessing"]["feature_selection"]

    if not settings.get("enabled", False):
        return X_train, X_val

    threshold = settings["consensus"]["apply_threshold"]

    active_methods = sum(
        method["enabled"]
        for method in settings["methods"].values()
    )

    if not 1 <= threshold <= active_methods:
        raise ValueError("Invalid consensus threshold in config.yaml.")

    if not X_train.columns.is_unique or not X_val.columns.is_unique:
        raise ValueError("Feature names must be unique.")

    if list(X_train.columns) != list(X_val.columns):
        raise ValueError("Train and validation feature columns differ.")

    if selection["feature"].duplicated().any():
        raise ValueError("Duplicate feature selection decisions.")

    votes = selection.set_index("feature")["rejected_votes"]

    if set(votes.index) != set(X_train.columns):
        raise ValueError("Selection decisions do not match model features.")

    selected_columns = [
        column
        for column in X_train.columns
        if votes.loc[column] < threshold
    ]

    if not selected_columns:
        raise ValueError("Consensus removed all features.")

    return (
        X_train[selected_columns].copy(),
        X_val[selected_columns].copy(),
    )
