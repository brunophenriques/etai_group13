
"""Feature selection methods and consensus analysis — ETAI Group 13."""

import numpy as np
import pandas as pd

from boruta import BorutaPy
from mrmr import mrmr_classif

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression


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


def select_rfecv(X, y, dates, config, settings, random_state):
    """RFECV with chronological inner folds."""

    estimator = RandomForestClassifier(
        n_estimators=settings["n_estimators"],
        min_samples_leaf=settings["min_samples_leaf"],
        random_state=random_state,
        n_jobs=-1,
    )

    splits = temporal_inner_splits(dates, config, settings)

    # Ensure each training split contains both target classes.
    for train_idx, val_idx in splits:
        if (
            y.iloc[train_idx].nunique() != 2
            or y.iloc[val_idx].nunique() != 2
        ):
            raise ValueError("RFECV inner split contains only one class.")

    selector = RFECV(
        estimator=estimator,
        step=settings["step"],
        min_features_to_select=settings["min_features_to_select"],
        cv=splits,
        scoring=settings["scoring"],
        n_jobs=1,
    )

    selector.fit(X, y)

    return pd.Series(
        selector.support_,
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
        penalty="l1",
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


def run_feature_selection(X, y, dates, config):
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

    results = pd.DataFrame(index=X.columns)

    if methods["mrmr"]["enabled"]:
        results["mrmr"] = select_mrmr(
            X, y, methods["mrmr"], random_state
        )

    if methods["rfecv"]["enabled"]:
        results["rfecv"] = select_rfecv(
            X, y, dates, config, methods["rfecv"], random_state
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
