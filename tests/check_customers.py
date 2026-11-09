
from src.data_loader import load_config, load_data
import pandas as pd

config = load_config()
train, _, customers = load_data(config)

# Apenas dados de treino com datas válidas
train = train.dropna(subset=["purchase_date"]).copy()
train["purchase_date"] = pd.to_datetime(train["purchase_date"])

print("\n=== CUSTOMER SINCE YEAR ===")

merged = train.merge(
    customers[["customer_id", "customer_since_year"]],
    on="customer_id",
    how="left",
    validate="many_to_one"
)

print("Rows after join:", len(merged))

print(
    "Missing customer_since_year:",
    merged["customer_since_year"].isna().sum()
)

print(
    "Customers registered after purchase:",
    (
        merged["customer_since_year"]
        > merged["purchase_date"].dt.year
    ).sum()
)

print("\nCustomer since year distribution:")
print(
    customers["customer_since_year"]
    .describe()
)

print("\n=== CUSTOMER TENURE CONSISTENCY ===")

merged["year_based_tenure_days"] = (
    merged["purchase_date"].dt.year
    - merged["customer_since_year"]
) * 365

print(
    "Correlation:",
    merged[
        ["customer_tenure_days", "year_based_tenure_days"]
    ].corr().iloc[0, 1]
)

print("\nCustomer tenure statistics:")
print(
    merged[
        ["customer_tenure_days", "year_based_tenure_days"]
    ].describe()
)

print("\nNegative customer tenure:")
print((merged["customer_tenure_days"] < 0).sum())