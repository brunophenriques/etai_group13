
from src.data_loader import load_config, load_data
from src.preprocessing import (
    prepare_rows,
    fill_structural_missing,
)

config = load_config()
train, _, _ = load_data(config)

train = prepare_rows(train, config)

processed = fill_structural_missing(train, config)

columns = [
    "prior_mean_gap_days",
    "prior_std_gap_days",
    "prior_mean_purchase_value",
    "prior_std_purchase_value",
    "prior_min_purchase_value",
    "prior_max_purchase_value",
    "days_since_previous_purchase",
    "spend_365d",
]

print("\n=== STRUCTURAL MISSING TEST ===")

for col in columns:
    before = train[col].isna().sum()
    after = processed[col].isna().sum()

    print(f"{col:<35} {before:>5} -> {after:>5}")

    # No previously observed value should change
    originally_known = train[col].notna()

    assert (
        processed.loc[originally_known, col]
        .equals(train.loc[originally_known, col])
    ), f"Existing values changed in {col}"

    # Missingness indicator must preserve original NaNs
    assert (
        processed[f"{col}_was_missing"]
        .equals(train[col].isna().astype(int))
    ), f"Incorrect indicator for {col}"

assert len(processed) == len(train)

print("\nAll structural missing tests passed.")
