from src.data_loader import load_config, load_data
import pandas as pd

config = load_config()
train, _, _ = load_data(config)

df = train.dropna(
    subset=["purchase_date", "prior_purchase_count"]
).copy()

df["purchase_date"] = pd.to_datetime(df["purchase_date"])

df = df.sort_values(["customer_id", "purchase_date"])

df["observed_previous_rows"] = df.groupby(
    "customer_id"
).cumcount()

df["difference"] = (
    df["prior_purchase_count"] - df["observed_previous_rows"]
)

print(df["difference"].describe())

print("\nRows with negative differences:")
print((df["difference"] < 0).sum())

print("\nRows with zero differences:")
print((df["difference"] == 0).sum())

print("\nRows with positive differences:")
print((df["difference"] > 0).sum())

offsets = df.groupby("customer_id")["difference"].agg(
    ["min", "max", "nunique"]
)

print("\nCustomers with a constant offset:")
print((offsets["nunique"] == 1).sum())

print("\nCustomers with a changing offset:")
print((offsets["nunique"] > 1).sum())

print("\nExamples of changing offsets:")
print(offsets[offsets["nunique"] > 1].head(10))

customer = df.loc[
    df["customer_id"] == 103940,
    [
        "purchase_date",
        "prior_purchase_count",
        "observed_previous_rows",
        "difference"
    ]
]

print(customer.to_string(index=False))