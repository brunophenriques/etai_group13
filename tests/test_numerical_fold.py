
import pandas as pd

from src.data_loader import load_config, load_data
from src.preprocessing import prepare_rows
from src.validation import temporal_cv
from src.fold_preprocessing import prepare_numeric_fold


config = load_config()

train, _, _ = load_data(config)
train = prepare_rows(train, config)

years = config["validation"]["cv_years"]
horizon = config["validation"]["target_horizon_days"]

columns = config["preprocessing"]["numeric_imputation"]["columns"]

# Use first temporal fold
year, train_fold, val_fold = next(
    temporal_cv(
        train,
        years,
        horizon_days=horizon,
    )
)

print(f"\n=== TEMPORAL FOLD {year} ===")
print("Training rows:", len(train_fold))
print("Validation rows:", len(val_fold))

train_dates = pd.to_datetime(train_fold["purchase_date"])
val_dates = pd.to_datetime(val_fold["purchase_date"])

assert (
    train_dates.max() + pd.Timedelta(days=horizon)
    < val_dates.min()
)

processed_train, processed_val, imputers = (
    prepare_numeric_fold(
        train_fold,
        val_fold,
        config,
    )
)

print("\n=== NUMERIC PREPROCESSING ===")

for column in columns:
    train_missing = processed_train[column].isna().sum()
    val_missing = processed_val[column].isna().sum()

    print(
        f"{column}: "
        f"train NaN={train_missing}, "
        f"validation NaN={val_missing}, "
        f"median={imputers[column].statistics_[0]:.4f}"
    )

    assert train_missing == 0
    assert val_missing == 0

assert len(processed_train) == len(train_fold)
assert len(processed_val) == len(val_fold)

assert list(processed_train.columns) == list(
    processed_val.columns
)

print("\nNumeric fold preprocessing passed.")
