
from src.data_loader import load_config, load_data
from src.preprocessing import prepare_rows
from src.validation import temporal_cv
from src.fold_preprocessing import prepare_numeric_fold

config = load_config()
train, _, _ = load_data(config)

train = prepare_rows(train, config)

year, train_fold, val_fold = next(
    temporal_cv(
        train,
        config["validation"]["cv_years"],
        horizon_days=config["validation"]["target_horizon_days"],
    )
)

processed_train, processed_val, _ = prepare_numeric_fold(
    train_fold, val_fold, config
)

settings = config["preprocessing"]["categorical_missing"]

columns = (
    settings["recover_from_customer_history"]
    + settings["keep_unknown"]
)

print(f"\n=== CATEGORICAL FOLD {year} ===")

for col in columns:
    print(f"\n{col}")
    print("Train NaN:", processed_train[col].isna().sum())
    print("Validation NaN:", processed_val[col].isna().sum())

    assert processed_train[col].isna().sum() == 0
    assert processed_val[col].isna().sum() == 0

    assert (
        f"{col}_was_missing" in processed_train.columns
    )

    if col in settings["recover_from_customer_history"]:
        print(
            "Recovered in train:",
            processed_train[f"{col}_was_recovered"].sum(),
        )
        print(
            "Recovered in validation:",
            processed_val[f"{col}_was_recovered"].sum(),
        )

assert processed_train.columns.equals(processed_val.columns)
assert len(processed_train) == len(train_fold)
assert len(processed_val) == len(val_fold)

print("\nCategorical fold preprocessing passed.")
