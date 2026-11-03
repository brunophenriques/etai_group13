import pandas as pd

from src.data_loader import load_config, load_data


config = load_config()
train, _, _ = load_data(config)

# Apply the exclusion rule already agreed upon
train = train.loc[train["purchase_date"].notna()].copy()

# print("TRAIN SHAPE:", train.shape)

# # General missing values
# missing = pd.DataFrame({
#     "missing_count": train.isna().sum(),
#     "missing_pct": train.isna().mean() * 100
# })

# missing = missing[missing["missing_count"] > 0]
# missing = missing.sort_values("missing_count", ascending=False)

# print("\n=== MISSING VALUES BY COLUMN ===")
# print(missing.round(2).to_string())

# # Historical gap features
# print("\n=== PRIOR MEAN GAP ANALYSIS ===")

# for condition, mask in {
#     "0 prior purchases": train["prior_purchase_count"] == 0,
#     "1 prior purchase": train["prior_purchase_count"] == 1,
#     "2+ prior purchases": train["prior_purchase_count"] >= 2,
# }.items():
#     subset = train.loc[mask]

#     print(
#         f"{condition}: "
#         f"{subset['prior_mean_gap_days'].isna().sum()} missing "
#         f"out of {len(subset)} rows"
#     )




# cols = [
#     "payment_type",
#     "warehouse",
#     "salesperson",
#     "transport_method",
#     "spend_365d",
# ]

# missing_patterns = (
#     train[cols]
#     .isna()
#     .value_counts()
#     .head(10)
# )

# print("\n=== COMMERCIAL MISSINGNESS PATTERNS ===")
# print(missing_patterns)


# missing = train.loc[
#     (train["prior_purchase_count"] >= 2)
#     & train["prior_mean_gap_days"].isna()
# ]

# print(missing["prior_purchase_count"].describe())

# print(
#     missing["prior_purchase_count"]
#     .value_counts()
#     .sort_index()
#     .head(20)
# )


# print("\n=== PRIOR STD GAP ANALYSIS ===")

# for condition, mask in {
#     "0 prior purchases": train["prior_purchase_count"] == 0,
#     "1 prior purchase": train["prior_purchase_count"] == 1,
#     "2+ prior purchases": train["prior_purchase_count"] >= 2,
# }.items():
#     subset = train.loc[mask]

#     print(
#         f"{condition}: "
#         f"{subset['prior_std_gap_days'].isna().sum()} missing "
#         f"out of {len(subset)} rows"
#     )


# print("\n=== STD GAP: STRUCTURAL VS UNEXPECTED ===")

# for label, mask in {
#     "Exactly 2 purchases": train["prior_purchase_count"] == 2,
#     "3+ purchases": train["prior_purchase_count"] >= 3,
# }.items():
#     subset = train.loc[mask]

#     print(
#         f"{label}: "
#         f"{subset['prior_std_gap_days'].isna().sum()} missing "
#         f"out of {len(subset)}"
#     )


# print("\n=== PRIOR STD PURCHASE VALUE ANALYSIS ===")

# for label, mask in {
#     "0 prior purchases": train["prior_purchase_count"] == 0,
#     "1 prior purchase": train["prior_purchase_count"] == 1,
#     "2+ prior purchases": train["prior_purchase_count"] >= 2,
# }.items():
#     subset = train.loc[mask]

#     print(
#         f"{label}: "
#         f"{subset['prior_std_purchase_value'].isna().sum()} missing "
#         f"out of {len(subset)}"
#     )

print("\n=== PRIOR MEAN PURCHASE VALUE ===")

for label, mask in {
    "0 prior purchases": train["prior_purchase_count"] == 0,
    "1+ prior purchases": train["prior_purchase_count"] >= 1,
}.items():
    subset = train.loc[mask]

    print(
        f"{label}: "
        f"{subset['prior_mean_purchase_value'].isna().sum()} missing "
        f"out of {len(subset)}"
    )

mask = train["prior_purchase_count"] >= 2

missing_value = train["prior_mean_purchase_value"].isna()
missing_gap = train["prior_mean_gap_days"].isna()

print("Both missing:", (mask & missing_value & missing_gap).sum())
print("Only purchase value missing:", (mask & missing_value & ~missing_gap).sum())
print("Only mean gap missing:", (mask & ~missing_value & missing_gap).sum())

print('67')
mask = (
    train["prior_mean_purchase_value"].notna()
    & train["prior_total_spend"].notna()
    & (train["prior_purchase_count"] > 0)
)

subset = train.loc[mask].copy()

calculated_mean = (
    subset["prior_total_spend"]
    / subset["prior_purchase_count"]
)

difference = (
    calculated_mean
    - subset["prior_mean_purchase_value"]
).abs()

print("Rows compared:", len(subset))
print("Median absolute difference:", difference.median())
print("Maximum absolute difference:", difference.max())
print("Share within €0.01:", (difference <= 0.01).mean())


exceptions = subset.loc[difference > 0.01].copy()

exceptions["calculated_mean"] = calculated_mean.loc[exceptions.index]
exceptions["absolute_difference"] = difference.loc[exceptions.index]

print("Exceptions:", len(exceptions))

print(
    exceptions[
        [
            "customer_id",
            "purchase_date",
            "prior_purchase_count",
            "prior_total_spend",
            "prior_mean_purchase_value",
            "calculated_mean",
            "absolute_difference",
        ]
    ]
    .sort_values("absolute_difference", ascending=False)
    .head(10)
    .to_string(index=False)
)

print(
    exceptions["prior_mean_purchase_value"]
    .value_counts()
    .head(10)
)


missing = train.loc[
    (train["prior_purchase_count"] >= 1)
    & train["prior_mean_purchase_value"].isna()
].copy()

recoverable = (
    missing["prior_total_spend"].notna()
    & missing["prior_purchase_count"].gt(0)
    & missing["prior_total_spend"].abs().lt(1e9)
)

print("Unexpected missing:", len(missing))
print("Recoverable by formula:", recoverable.sum())
print("Not recoverable:", (~recoverable).sum())

values = train["prior_mean_purchase_value"]

valid_values = values[
    values.notna() & (values != 1_000_000_000)
]

print(valid_values.describe(
    percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
))

print("\nNegative values:", (valid_values < 0).sum())
print("Zero values:", (valid_values == 0).sum())
print("Above €10,000:", (valid_values > 10_000).sum())


extreme = train.loc[
    (train["prior_mean_purchase_value"] < 0)
    | (
        (train["prior_mean_purchase_value"] > 10_000)
        & (train["prior_mean_purchase_value"] != 1_000_000_000)
    )
].copy()

extreme["calculated_mean"] = (
    extreme["prior_total_spend"]
    / extreme["prior_purchase_count"]
)

extreme["consistent"] = (
    (extreme["prior_mean_purchase_value"] - extreme["calculated_mean"])
    .abs() <= 0.01
)

print("Extreme observations:", len(extreme))
print("Consistent with formula:", extreme["consistent"].sum())
print("Inconsistent:", (~extreme["consistent"]).sum())

print(
    extreme.loc[
        ~extreme["consistent"],
        [
            "customer_id",
            "prior_purchase_count",
            "prior_total_spend",
            "prior_mean_purchase_value",
            "calculated_mean",
        ]
    ].head(10).to_string(index=False)
)

remaining = exceptions.loc[
    (exceptions["prior_mean_purchase_value"] >= 0)
    & (exceptions["prior_mean_purchase_value"] <= 10_000)
].copy()

print("Remaining exceptions:", len(remaining))

print(
    remaining["absolute_difference"].describe()
)

print(
    remaining[
        [
            "prior_mean_purchase_value",
            "calculated_mean",
            "absolute_difference"
        ]
    ]
    .sort_values("absolute_difference", ascending=False)
    .head(10)
    .to_string(index=False)
)