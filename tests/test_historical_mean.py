
from src.data_loader import load_config, load_data
from src.preprocessing import (
    prepare_rows,
    fill_structural_missing,
    recover_historical_mean,
)

config = load_config()
train, _, _ = load_data(config)

train = prepare_rows(train, config)
structural = fill_structural_missing(train, config)
processed = recover_historical_mean(structural, config)

column = "prior_mean_purchase_value"

print("\n=== HISTORICAL MEAN RECOVERY TEST ===")

print("NaN before:", structural[column].isna().sum())
print("NaN after:", processed[column].isna().sum())

print(
    "Recovered values:",
    processed[
        "prior_mean_purchase_value_was_recovered"
    ].sum()
)

# Existing values must remain unchanged
known = structural[column].notna()

assert processed.loc[known, column].equals(
    structural.loc[known, column]
)

# The recovery indicator should identify exactly
# the values recovered by the formula
assert (
    processed[
        "prior_mean_purchase_value_was_recovered"
    ].sum() == 194
)

assert processed[column].isna().sum() == 0

print("\nHistorical mean recovery passed.")
