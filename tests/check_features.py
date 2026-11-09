
from src.data_loader import load_config, load_data
import pandas as pd

config = load_config()
train, test, customers = load_data(config)

# Apenas treino
train = train.dropna(subset=["purchase_date"]).copy()

print("\n=== MAX > TOTAL: SOURCE CONSISTENCY ===")

# Selecionar apenas os 45 casos suspeitos
suspect = train.loc[
    (train["prior_purchase_count"] >= 1)
    & train["prior_max_purchase_value"].notna()
    & train["prior_total_spend"].notna()
    & (
        train["prior_max_purchase_value"]
        > train["prior_total_spend"] + 0.01
    )
].copy()

# Verificar a consistência da média
suspect["calculated_mean"] = (
    suspect["prior_total_spend"]
    / suspect["prior_purchase_count"]
)

suspect["mean_matches_total"] = (
    suspect["calculated_mean"]
    - suspect["prior_mean_purchase_value"]
).abs() <= 0.01

print("Suspicious rows:", len(suspect))

print(
    "Mean agrees with total/count:",
    suspect["mean_matches_total"].sum()
)

print(
    "Mean disagrees with total/count:",
    (
        suspect["prior_mean_purchase_value"].notna()
        & ~suspect["mean_matches_total"]
    ).sum()
)

print(
    "Mean missing:",
    suspect["prior_mean_purchase_value"].isna().sum()
)

print("\nExamples:")
print(
    suspect[
        [
            "prior_max_purchase_value",
            "prior_total_spend",
            "prior_purchase_count",
            "prior_mean_purchase_value",
            "calculated_mean",
        ]
    ].head(15).to_string(index=False)
)


print("\n=== MAX PURCHASE VALUE: CORRUPTION PATTERN ===")

suspect["max_to_total_ratio"] = (
    suspect["prior_max_purchase_value"]
    / suspect["prior_total_spend"]
)

print("Ratio distribution:")
print(suspect["max_to_total_ratio"].describe())

print("\nValues above 2x historical total:")
print((suspect["max_to_total_ratio"] > 2).sum())

print("\nValues above 5x historical total:")
print((suspect["max_to_total_ratio"] > 5).sum())