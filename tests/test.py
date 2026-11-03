from src.data_loader import load_config, load_data
from src.validation import temporal_split
import pandas as pd
from src.validation import temporal_cv

config = load_config()
train, _, _ = load_data(config)

train_fold, val_fold = temporal_split(
    train,
    validation_start="2024-01-01",
    validation_end="2024-12-31",
    horizon_days=config["validation"]["target_horizon_days"],
)

print("Training:", train_fold.shape)
print("Validation:", val_fold.shape)

print("Last training purchase:", train_fold["purchase_date"].max())
print("First validation purchase:", val_fold["purchase_date"].min())

train_dates = pd.to_datetime(train_fold["purchase_date"])
val_dates = pd.to_datetime(val_fold["purchase_date"])

assert train_dates.notna().all()
assert val_dates.notna().all()

assert (
    train_dates + pd.Timedelta(days=90)
    < val_dates.min()
).all()

assert train_fold.index.intersection(val_fold.index).empty

print("Temporal leakage checks passed!")

for year, train_fold, val_fold in temporal_cv(
    train,
    validation_years=config["validation"]["cv_years"],
    horizon_days=config["validation"]["target_horizon_days"],
):
    train_dates = pd.to_datetime(train_fold["purchase_date"])
    val_dates = pd.to_datetime(val_fold["purchase_date"])

    assert (
        train_dates + pd.Timedelta(days=90)
        < val_dates.min()
    ).all()

    assert train_fold.index.intersection(val_fold.index).empty

    print(
        f"{year}: "
        f"Train={len(train_fold)}, "
        f"Validation={len(val_fold)}"
    )