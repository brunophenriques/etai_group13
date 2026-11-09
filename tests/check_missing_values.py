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

# print("\n=== PRIOR MEAN PURCHASE VALUE ===")
#
# for label, mask in {
#     "0 prior purchases": train["prior_purchase_count"] == 0,
#     "1+ prior purchases": train["prior_purchase_count"] >= 1,
# }.items():
#     subset = train.loc[mask]
#
#     print(
#         f"{label}: "
#         f"{subset['prior_mean_purchase_value'].isna().sum()} missing "
#         f"out of {len(subset)}"
#     )
#
# mask = train["prior_purchase_count"] >= 2
#
# missing_value = train["prior_mean_purchase_value"].isna()
# missing_gap = train["prior_mean_gap_days"].isna()
#
# print("Both missing:", (mask & missing_value & missing_gap).sum())
# print("Only purchase value missing:", (mask & missing_value & ~missing_gap).sum())
# print("Only mean gap missing:", (mask & ~missing_value & missing_gap).sum())
#
# print('67')
# mask = (
#     train["prior_mean_purchase_value"].notna()
#     & train["prior_total_spend"].notna()
#     & (train["prior_purchase_count"] > 0)
# )
#
# subset = train.loc[mask].copy()
#
# calculated_mean = (
#     subset["prior_total_spend"]
#     / subset["prior_purchase_count"]
# )
#
# difference = (
#     calculated_mean
#     - subset["prior_mean_purchase_value"]
# ).abs()
#
# print("Rows compared:", len(subset))
# print("Median absolute difference:", difference.median())
# print("Maximum absolute difference:", difference.max())
# print("Share within €0.01:", (difference <= 0.01).mean())
#
#
# exceptions = subset.loc[difference > 0.01].copy()
#
# exceptions["calculated_mean"] = calculated_mean.loc[exceptions.index]
# exceptions["absolute_difference"] = difference.loc[exceptions.index]
#
# print("Exceptions:", len(exceptions))
#
# print(
#     exceptions[
#         [
#             "customer_id",
#             "purchase_date",
#             "prior_purchase_count",
#             "prior_total_spend",
#             "prior_mean_purchase_value",
#             "calculated_mean",
#             "absolute_difference",
#         ]
#     ]
#     .sort_values("absolute_difference", ascending=False)
#     .head(10)
#     .to_string(index=False)
# )
#
# print(
#     exceptions["prior_mean_purchase_value"]
#     .value_counts()
#     .head(10)
# )
#
#
# missing = train.loc[
#     (train["prior_purchase_count"] >= 1)
#     & train["prior_mean_purchase_value"].isna()
# ].copy()
#
# recoverable = (
#     missing["prior_total_spend"].notna()
#     & missing["prior_purchase_count"].gt(0)
#     & missing["prior_total_spend"].abs().lt(1e9)
# )
#
# print("Unexpected missing:", len(missing))
# print("Recoverable by formula:", recoverable.sum())
# print("Not recoverable:", (~recoverable).sum())
#
# values = train["prior_mean_purchase_value"]
#
# valid_values = values[
#     values.notna() & (values != 1_000_000_000)
# ]
#
# print(valid_values.describe(
#     percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
# ))
#
# print("\nNegative values:", (valid_values < 0).sum())
# print("Zero values:", (valid_values == 0).sum())
# print("Above €10,000:", (valid_values > 10_000).sum())
#
#
# extreme = train.loc[
#     (train["prior_mean_purchase_value"] < 0)
#     | (
#         (train["prior_mean_purchase_value"] > 10_000)
#         & (train["prior_mean_purchase_value"] != 1_000_000_000)
#     )
# ].copy()
#
# extreme["calculated_mean"] = (
#     extreme["prior_total_spend"]
#     / extreme["prior_purchase_count"]
# )
#
# extreme["consistent"] = (
#     (extreme["prior_mean_purchase_value"] - extreme["calculated_mean"])
#     .abs() <= 0.01
# )
#
# print("Extreme observations:", len(extreme))
# print("Consistent with formula:", extreme["consistent"].sum())
# print("Inconsistent:", (~extreme["consistent"]).sum())
#
# print(
#     extreme.loc[
#         ~extreme["consistent"],
#         [
#             "customer_id",
#             "prior_purchase_count",
#             "prior_total_spend",
#             "prior_mean_purchase_value",
#             "calculated_mean",
#         ]
#     ].head(10).to_string(index=False)
# )
#
# remaining = exceptions.loc[
#     (exceptions["prior_mean_purchase_value"] >= 0)
#     & (exceptions["prior_mean_purchase_value"] <= 10_000)
# ].copy()
#
# print("Remaining exceptions:", len(remaining))
#
# print(
#     remaining["absolute_difference"].describe()
# )
#
# print(
#     remaining[
#         [
#             "prior_mean_purchase_value",
#             "calculated_mean",
#             "absolute_difference"
#         ]
#     ]
#     .sort_values("absolute_difference", ascending=False)
#     .head(10)
#     .to_string(index=False)
# )

# print("\n=== DAYS SINCE PREVIOUS PURCHASE ===")
#
# for label, mask in {
#     "0 prior purchases": train["prior_purchase_count"] == 0,
#     "1+ prior purchases": train["prior_purchase_count"] >= 1,
# }.items():
#     subset = train.loc[mask]
#
#     print(
#         f"{label}: "
#         f"{subset['days_since_previous_purchase'].isna().sum()} missing "
#         f"out of {len(subset)}"
#     )
#
#
# df = train.copy()
# df["purchase_date"] = pd.to_datetime(df["purchase_date"])
#
# df = df.sort_values(["customer_id", "purchase_date"])
#
# df["previous_observed_date"] = (
#     df.groupby("customer_id")["purchase_date"].shift(1)
# )
#
# mask = (
#     (df["prior_purchase_count"] >= 1)
#     & df["days_since_previous_purchase"].isna()
# )
#
# print("Unexpected missing:", mask.sum())
# print(
#     "Recoverable from observed history:",
#     df.loc[mask, "previous_observed_date"].notna().sum()
# )
#
#
# df["calculated_recency"] = (
#     df["purchase_date"] - df["previous_observed_date"]
# ).dt.days
#
# known = df.loc[
#     df["days_since_previous_purchase"].notna()
#     & df["previous_observed_date"].notna()
# ].copy()
#
# difference = (
#     known["calculated_recency"]
#     - known["days_since_previous_purchase"]
# ).abs()
#
# print("Rows compared:", len(known))
# print("Exact matches:", (difference == 0).sum())
# print("Match percentage:", round((difference == 0).mean() * 100, 2))
# print("Median difference:", difference.median())
# print("Maximum difference:", difference.max())


# print("\n=== SPEND 365D MISSING ANALYSIS ===")
#
# missing = train["spend_365d"].isna()
#
# print("Total missing:", missing.sum())
#
# for label, mask in {
#     "0 purchases in 365d": train["purchase_count_365d"] == 0,
#     "1+ purchases in 365d": train["purchase_count_365d"] >= 1,
#     "Missing purchase count": train["purchase_count_365d"].isna(),
# }.items():
#     print(f"{label}: {(missing & mask).sum()}")
#
# print("\n=== SPEND 365D: UNEXPECTED MISSING ===")
#
# mask = (
#     train["spend_365d"].isna()
#     & (train["purchase_count_365d"] >= 1)
# )
#
# subset = train.loc[mask]
#
# print("Unexpected missing:", len(subset))
#
# for col in ["spend_30d", "spend_90d", "spend_180d"]:
#     print(
#         f"{col}: "
#         f"{subset[col].notna().sum()} available, "
#         f"{subset[col].isna().sum()} missing"
#     )
#
# print("\nPurchase count distribution:")
# print(subset["purchase_count_365d"].describe())
#
# print("\nRows with positive spend_180d:")
# print((subset["spend_180d"] > 0).sum())
#
# print("\nRows with zero purchases in 180d:")
# print((subset["purchase_count_180d"] == 0).sum())
#
# print("\n=== SPEND 365D FORMULA VALIDATION ===")
#
# valid = train.loc[
#     train["spend_365d"].notna()
#     & (train["purchase_count_365d"] >= 1)
#     & (train["prior_purchase_count"] >= 1)
#     & train["prior_total_spend"].notna()
# ].copy()
#
# valid["estimated_spend_365d"] = (
#     valid["prior_total_spend"]
#     / valid["prior_purchase_count"]
#     * valid["purchase_count_365d"]
# )
#
# error = (
#     valid["estimated_spend_365d"]
#     - valid["spend_365d"]
# ).abs()
#
# print("Rows compared:", len(valid))
# print("Median absolute error:", round(error.median(), 2))
# print("Mean absolute error:", round(error.mean(), 2))
# print("Actual spend median:", round(valid["spend_365d"].median(), 2))
#
# relative_error = (
#     error / valid["spend_365d"].abs()
# ).where(valid["spend_365d"] != 0)
#
# print(
#     "Median relative error (%):",
#     round(relative_error.median() * 100, 2)
# )
#
# print(
#     "Within 20% of actual:",
#     round((relative_error <= 0.20).mean() * 100, 2),
#     "%"
# )
#
# print("\n=== SPEND 365D FORMULA: OUTLIER CHECK ===")
#
# clean = valid.loc[
#     (valid["prior_total_spend"] >= 0)
#     & (valid["prior_total_spend"] < 1_000_000)
#     & (valid["estimated_spend_365d"] >= 0)
#     & (valid["estimated_spend_365d"] < 1_000_000)
# ].copy()
#
# clean_error = (
#     clean["estimated_spend_365d"] - clean["spend_365d"]
# ).abs()
#
# clean_relative_error = (
#     clean_error / clean["spend_365d"].abs()
# ).where(clean["spend_365d"] != 0)
#
# print("Rows retained:", len(clean))
# print("Rows excluded:", len(valid) - len(clean))
# print("Median absolute error:", round(clean_error.median(), 2))
# print("Mean absolute error:", round(clean_error.mean(), 2))
# print(
#     "Median relative error (%):",
#     round(clean_relative_error.median() * 100, 2)
# )
# print(
#     "Within 20%:",
#     round((clean_relative_error <= 0.20).mean() * 100, 2),
#     "%"
# )
#
# print("\n=== SPEND 365D: LARGEST ERRORS ===")
#
# valid["absolute_error"] = (
#     valid["estimated_spend_365d"] - valid["spend_365d"]
# ).abs()
#
# print("\nActual spend_365d distribution:")
# print(
#     valid["spend_365d"].describe(
#         percentiles=[0.50, 0.90, 0.95, 0.99, 0.999]
#     )
# )
#
# print("\nTop 10 largest errors:")
# print(
#     valid.nlargest(10, "absolute_error")[
#         [
#             "spend_365d",
#             "estimated_spend_365d",
#             "prior_total_spend",
#             "prior_purchase_count",
#             "purchase_count_365d",
#             "absolute_error",
#         ]
#     ].to_string(index=False)
# )
# print("\n=== SPEND 365D DATA QUALITY ===")
#
# s = train["spend_365d"]
#
# print("Missing:", s.isna().sum())
# print("Exactly 1 billion:", (s == 1_000_000_000).sum())
# print("Negative:", (s < 0).sum())
# print("Exactly zero:", (s == 0).sum())
#
# print("\n=== FORMULA WITHOUT SUSPECT VALUES ===")
#
# clean = valid.loc[
#     (valid["spend_365d"] >= 0)
#     & (valid["spend_365d"] < 1_000_000)
#     & (valid["estimated_spend_365d"] >= 0)
#     & (valid["estimated_spend_365d"] < 1_000_000)
# ].copy()
#
# error = (
#     clean["estimated_spend_365d"] - clean["spend_365d"]
# ).abs()
#
# relative_error = (
#     error / clean["spend_365d"]
# ).where(clean["spend_365d"] > 0)
#
# print("Rows retained:", len(clean))
# print("Rows excluded:", len(valid) - len(clean))
# print("Median absolute error:", round(error.median(), 2))
# print("Mean absolute error:", round(error.mean(), 2))
# print("Median relative error (%):", round(relative_error.median() * 100, 2))
# print("Within 20%:", round((relative_error <= 0.20).mean() * 100, 2), "%")


# print("\n=== SPEND 365D: NEGATIVE VALUES ===")
#
# negative = train.loc[
#     train["spend_365d"] < 0
# ].copy()
#
# print("Total negative values:", len(negative))
#
# print("\nNegative values:")
# print(negative["spend_365d"].describe())
#
# print("\nRelated purchase and spending features:")
#
# columns = [
#     "spend_365d",
#     "purchase_count_365d",
#     "spend_180d",
#     "purchase_count_180d",
#     "prior_total_spend",
#     "prior_purchase_count",
# ]
#
# print(
#     negative[columns]
#     .sort_values("spend_365d")
#     .to_string(index=False)
# )
#
# print("\nNegative spend with positive 365d purchase count:")
# print(
#     (negative["purchase_count_365d"] > 0).sum()
# )
#
# print("\nNegative spend with positive 180d spend:")
# print(
#     (negative["spend_180d"] > 0).sum()
# )
#
#
# print("\n=== NEGATIVE SPEND 365D: SIGN FLIP CHECK ===")
#
# negative = train.loc[train["spend_365d"] < 0].copy()
#
# negative["matches_prior_total"] = (
#     negative["spend_365d"].abs()
#     - negative["prior_total_spend"]
# ).abs() < 0.01
#
# negative["matches_spend_180d"] = (
#     negative["spend_365d"].abs()
#     - negative["spend_180d"]
# ).abs() < 0.01
#
# print(
#     "Matches prior_total_spend:",
#     negative["matches_prior_total"].sum(),
#     "out of", len(negative)
# )
#
# print(
#     "Matches spend_180d:",
#     negative["matches_spend_180d"].sum(),
#     "out of", len(negative)
# )
#
# print("\nOther spending windows:")
#
# print(
#     negative[
#         [
#             "spend_365d",
#             "spend_30d",
#             "spend_90d",
#             "spend_180d",
#             "prior_total_spend",
#         ]
#     ].to_string(index=False)
# )

print("\n=== CATEGORICAL MISSING VALUES ===")

categorical_columns = [
    "payment_type",
    "warehouse",
    "salesperson",
    "transport_method",
]

for col in categorical_columns:
    print(f"\n--- {col} ---")
    print("Missing:", train[col].isna().sum())
    print("Unique categories:", train[col].nunique(dropna=True))

    print("\nTop categories:")
    print(
        train[col]
        .value_counts(dropna=False)
        .head(10)
        .to_string()
    )

print("\n=== CATEGORICAL MISSING: CUSTOMER HISTORY ===")

categorical_columns = [
    "payment_type",
    "warehouse",
    "salesperson",
    "transport_method",
]

df = train.sort_values(["customer_id", "purchase_date"]).copy()

for col in categorical_columns:
    previous = (
        df.groupby("customer_id")[col]
        .transform(lambda s: s.ffill().shift(1))
    )

print("\n=== CATEGORICAL MISSING: CUSTOMER HISTORY ===")

categorical_columns = [
    "payment_type",
    "warehouse",
    "salesperson",
    "transport_method",
]

df = train.sort_values(
    ["customer_id", "purchase_date"]
).copy()

for col in categorical_columns:
    previous = (
        df.groupby("customer_id")[col]
        .transform(lambda s: s.ffill().shift(1))
    )

    missing = df[col].isna()

    recoverable = (
        missing & previous.notna()
    ).sum()

    print(f"\n{col}")
    print("Missing:", missing.sum())
    print("Previous known value:", recoverable)
    print("No previous known value:", missing.sum() - recoverable)

print("\n=== CATEGORICAL HISTORY CONSISTENCY ===")

categorical_columns = [
    "payment_type",
    "warehouse",
    "salesperson",
    "transport_method",
]

df = train.sort_values(
    ["customer_id", "purchase_date"],
    kind="stable"
).copy()

for col in categorical_columns:

    previous = df.groupby("customer_id")[col].transform(
        lambda s: s.ffill().shift(1)
    )

    known = df[col].notna() & previous.notna()

    matches = (
        df.loc[known, col] == previous.loc[known]
    )

    print(f"\n{col}")
    print("Rows compared:", known.sum())
    print("Exact matches:", matches.sum())
    print("Match percentage:", round(matches.mean() * 100, 2))

print("\n=== SAME-DAY CUSTOMER PURCHASES ===")

same_day = train.duplicated(
    subset=["customer_id", "purchase_date"],
    keep=False
)

print("Rows sharing customer and date:", same_day.sum())

print(
    "Customer-date groups with multiple rows:",
    train.loc[same_day]
    .groupby(["customer_id", "purchase_date"])
    .ngroups
)

categorical_columns = [
    "payment_type",
    "warehouse",
    "salesperson",
    "transport_method",
]

for col in categorical_columns:
    print(
        f"{col} missing in same-day groups:",
        (same_day & train[col].isna()).sum()
    )