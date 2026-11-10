
from src.data_loader import load_config, load_data
from src.preprocessing import (
    prepare_rows,
    fill_structural_missing,
    recover_recency,
)

config = load_config()
train, _, _ = load_data(config)

train = prepare_rows(train, config)

structural = fill_structural_missing(train, config)

processed = recover_recency(
    df=structural,
    history=train,
    config=config,
)

print("\n=== RECENCY RECOVERY TEST ===")

print(
    "NaN before:",
    structural["days_since_previous_purchase"].isna().sum()
)

print(
    "Recovered:",
    processed["recency_was_recovered"].sum()
)

print(
    "NaN after:",
    processed["days_since_previous_purchase"].isna().sum()
)

known = structural["days_since_previous_purchase"].notna()

assert processed.loc[
    known, "days_since_previous_purchase"
].equals(
    structural.loc[
        known, "days_since_previous_purchase"
    ]
)

assert processed["recency_was_recovered"].sum() == 192
assert processed["days_since_previous_purchase"].isna().sum() == 1

print("\nRecency recovery passed.")


from src.data_loader import load_config, load_data
from src.preprocessing import (
    prepare_rows,
    fill_structural_missing,
    recover_recency,
    fit_recency_imputer,
    apply_recency_imputer,
)

config = load_config()
train, _, _ = load_data(config)

train = prepare_rows(train, config)
structural = fill_structural_missing(train, config)

recovered = recover_recency(
    df=structural,
    history=train,
    config=config,
)

# Diagnostic test only.
# The real pipeline will fit using each temporal training fold.
imputer = fit_recency_imputer(recovered)

processed = apply_recency_imputer(recovered, imputer)

print("\n=== RECENCY IMPUTATION TEST ===")
print("NaN before:", recovered["days_since_previous_purchase"].isna().sum())
print("NaN after:", processed["days_since_previous_purchase"].isna().sum())

known = recovered["days_since_previous_purchase"].notna()

assert processed["days_since_previous_purchase"].isna().sum() == 0
assert processed.loc[
    known, "days_since_previous_purchase"
].equals(
    recovered.loc[known, "days_since_previous_purchase"]
)

print("\nRecency imputation passed.")



import pandas as pd

from src.data_loader import load_config, load_data
from src.preprocessing import (
    prepare_rows,
    fill_structural_missing,
    fit_numeric_imputers,
    apply_numeric_imputers,
)

config = load_config()
train, _, _ = load_data(config)

train = prepare_rows(train, config)
structural = fill_structural_missing(train, config)

column = "prior_mean_gap_days"

print("\n=== NUMERIC IMPUTATION TEST ===")

# Diagnostic check on training data only
imputers = fit_numeric_imputers(
    structural,
    columns=[column],
)

processed = apply_numeric_imputers(
    structural,
    imputers,
)

print("NaN before:", structural[column].isna().sum())
print("NaN after:", processed[column].isna().sum())
print(
    "Training median:",
    imputers[column].statistics_[0]
)

known = structural[column].notna()

assert structural[column].isna().sum() == 195
assert processed[column].isna().sum() == 0

assert processed.loc[known, column].equals(
    structural.loc[known, column]
)

print("\n=== TRAIN / VALIDATION ISOLATION TEST ===")

toy_train = pd.DataFrame({
    column: [10.0, 20.0, None]
})

toy_validation = pd.DataFrame({
    column: [9999.0, None]
})

toy_imputers = fit_numeric_imputers(
    toy_train,
    columns=[column],
)

result = apply_numeric_imputers(
    toy_validation,
    toy_imputers,
)

print("Expected training median: 15.0")
print("Validation missing value filled:", result[column].iloc[1])

assert result[column].iloc[1] == 15.0
assert result[column].iloc[0] == 9999.0

print("\nAll numeric imputation tests passed.")



from src.data_loader import load_config, load_data
from src.preprocessing import (
    prepare_rows,
    fill_structural_missing,
    clean_spend_365d,
    fit_numeric_imputers,
    apply_numeric_imputers,
)

config = load_config()
train, _, _ = load_data(config)

train = prepare_rows(train, config)

structural = fill_structural_missing(train, config)
cleaned = clean_spend_365d(structural, config)

column = "spend_365d"

print("\n=== SPEND 365D CLEANING TEST ===")

print("NaN before cleaning:", structural[column].isna().sum())
print("NaN after cleaning:", cleaned[column].isna().sum())
print("Corrected values:", cleaned["spend_365d_was_corrected"].sum())

assert structural[column].isna().sum() == 163
assert cleaned["spend_365d_was_corrected"].sum() == 30
assert cleaned[column].isna().sum() == 193

imputers = fit_numeric_imputers(
    cleaned,
    columns=[column],
)

processed = apply_numeric_imputers(cleaned, imputers)

print("\n=== SPEND 365D IMPUTATION TEST ===")

print("NaN after imputation:", processed[column].isna().sum())
print("Training median:", imputers[column].statistics_[0])

assert processed[column].isna().sum() == 0

# Previously valid values must remain unchanged
valid = cleaned[column].notna()

assert processed.loc[valid, column].equals(
    cleaned.loc[valid, column]
)

print("\nSpend 365d tests passed.")


from src.data_loader import load_config, load_data
from src.preprocessing import (
    prepare_rows,
    fill_structural_missing,
    clean_prior_max_purchase_value,
    fit_numeric_imputers,
    apply_numeric_imputers,
)

config = load_config()
train, _, _ = load_data(config)

train = prepare_rows(train, config)
structural = fill_structural_missing(train, config)

cleaned = clean_prior_max_purchase_value(
    structural, config
)

column = "prior_max_purchase_value"

print("\n=== PRIOR MAX CONSISTENCY TEST ===")

print("NaN before:", structural[column].isna().sum())
print("NaN after cleaning:", cleaned[column].isna().sum())
print(
    "Values flagged:",
    cleaned["prior_max_purchase_value_was_corrected"].sum()
)

assert structural[column].isna().sum() == 0
assert cleaned[column].isna().sum() == 45
assert (
    cleaned["prior_max_purchase_value_was_corrected"].sum()
    == 45
)

# Existing values not flagged must remain unchanged
unchanged = (
    cleaned["prior_max_purchase_value_was_corrected"] == 0
)

assert cleaned.loc[unchanged, column].equals(
    structural.loc[unchanged, column]
)

imputers = fit_numeric_imputers(
    cleaned,
    columns=[column],
)

processed = apply_numeric_imputers(
    cleaned,
    imputers,
)

print("\n=== PRIOR MAX IMPUTATION TEST ===")

print("NaN after imputation:", processed[column].isna().sum())
print("Training median:", imputers[column].statistics_[0])

assert processed[column].isna().sum() == 0

print("\nPrior max tests passed.")
