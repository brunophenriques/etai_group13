
"""Coordinate numeric and categorical preprocessing within temporal folds."""

from src.preprocessing import (
    apply_numeric_imputers,
    clean_historical_gaps,
    clean_prior_max_purchase_value,
    clean_spend_365d,
    fill_structural_missing,
    fit_numeric_imputers,
    recover_categorical_missing,
    recover_historical_mean,
    recover_recency,
)


def prepare_numeric_fold(train_fold, val_fold, config):
    """
    Prepare numeric and categorical data for one temporal CV fold.

    Numeric imputation supports mean and median.
    Imputers are fitted exclusively on the training fold.

    Historical recovery uses training-fold transactions
    as reference history to avoid temporal leakage.
    """

    settings = config["preprocessing"]["numeric_imputation"]

    # Validate imputation configuration
    if settings["strategy"] not in {"mean", "median", "knn"}:
        raise ValueError(
            f"Unsupported numeric imputation strategy: {settings['strategy']}"
        )

    if settings["fit_on"] != "train_fold_only":
        raise ValueError(
            "Numeric imputers must be fitted on train folds only."
        )

    # 1. Fill structural missing values
    train = fill_structural_missing(train_fold, config)
    val = fill_structural_missing(val_fold, config)

    # 2. Recover historical mean purchase values
    train = recover_historical_mean(train, config)
    val = recover_historical_mean(val, config)

    # 3. Recover recency using training-fold history
    history = train_fold[
        ["customer_id", "purchase_date"]
    ].copy()

    train = recover_recency(train, history, config)
    val = recover_recency(val, history, config)

    # 4. Clean invalid numeric values
    train = clean_spend_365d(train, config)
    val = clean_spend_365d(val, config)

    train = clean_prior_max_purchase_value(train, config)
    val = clean_prior_max_purchase_value(val, config)

    train = clean_historical_gaps(train, config)
    val = clean_historical_gaps(val, config)

    # 5. Fit numeric imputers on the training fold only
    columns = settings["columns"]

    imputers = fit_numeric_imputers(
        train,
        columns,
        strategy=settings["strategy"],
        knn_settings=settings.get("knn"),
    )
    # 6. Apply the fitted imputers to both datasets
    train = apply_numeric_imputers(
        train,
        imputers,
    )

    val = apply_numeric_imputers(
        val,
        imputers,
    )

    # 7. Recover categorical values using past records
    train = recover_categorical_missing(
        train,
        train_fold,
        config,
    )

    val = recover_categorical_missing(
        val,
        train_fold,
        config,
    )

    return train, val, imputers
