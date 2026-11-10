
from src.data_loader import load_config, load_data
from src.preprocessing import (
    prepare_rows,
    prepare_model_features,
)

config = load_config()

train, _, _ = load_data(config)

# Prepare dates and remove undated rows
prepared = prepare_rows(train, config)

# Create model features
X = prepare_model_features(prepared, config)

print("\n=== PREPROCESSING: STAGE 1 ===")

print("Original rows:", len(train))
print("Prepared rows:", len(prepared))
print("Feature columns:", X.shape[1])

print("\nDate features:")
print(
    X[
        [
            "purchase_year",
            "purchase_month",
            "purchase_day_of_week",
        ]
    ].head()
)

print("\nExcluded columns check:")

excluded = [
    "ID",
    "customer_id",
    "random_noise",
    "purchase_date",
    "repeat_purchase_90d",
    "spend_next_90d",
]

for col in excluded:
    print(f"{col}: {'REMOVED' if col not in X.columns else 'STILL PRESENT'}")

print("\nRemaining missing values:")
print(X.isna().sum()[X.isna().sum() > 0])
