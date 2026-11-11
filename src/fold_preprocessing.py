
from src.preprocessing import (
    fill_structural_missing,
    recover_historical_mean,
    recover_recency,
    clean_spend_365d,
    clean_prior_max_purchase_value,
    fit_numeric_imputers,
    apply_numeric_imputers,
    recover_categorical_missing,
)


def prepare_numeric_fold(train_fold, val_fold, config):
    """
    Prepare numeric data for one temporal CV fold.

    Imputers are fitted only on the training fold.
    Historical recency recovery uses training-fold
    transactions as reference history.
    """

    settings = config["preprocessing"]["numeric_imputation"]

    if settings["strategy"] != "median":
        raise ValueError(
            "Only median numeric imputation is supported."
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

    # 4. Clean invalid numerical values
    train = clean_spend_365d(train, config)
    val = clean_spend_365d(val, config)

    train = clean_prior_max_purchase_value(train, config)
    val = clean_prior_max_purchase_value(val, config)

    # 5. Fit numeric imputers on training fold only
    columns = settings["columns"]

    imputers = fit_numeric_imputers(
        train,
        columns,
    )

    # 6. Apply fitted imputers to both datasets
    train = apply_numeric_imputers(
        train,
        imputers,
    )

    val = apply_numeric_imputers(
        val,
        imputers,
    )

        # 7. Recover categorical values from past records
    train = recover_categorical_missing(
        train, train_fold, config
    )

    val = recover_categorical_missing(
        val, train_fold, config
    )

    return train, val, imputers


