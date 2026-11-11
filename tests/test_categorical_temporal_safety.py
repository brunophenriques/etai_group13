
import pandas as pd

from src.data_loader import load_config
from src.preprocessing import recover_categorical_missing


config = load_config()

history = pd.DataFrame({
    "customer_id": [1, 1, 1, 2],
    "purchase_date": pd.to_datetime([
        "2020-01-01",
        "2020-03-01",
        "2020-06-01",
        "2020-05-01",
    ]),
    "payment_type": [
        "Card",
        "Cash",
        "Transfer",
        "Card",
    ],
    "warehouse": ["A", "B", "C", "A"],
    "salesperson": ["S1", "S2", "S3", "S1"],
})

observations = pd.DataFrame({
    "customer_id": [1, 1, 1, 1, 3],
    "purchase_date": pd.to_datetime([
        "2019-12-01",  # No previous history
        "2020-02-01",  # Only January available
        "2020-03-01",  # Same-day record excluded
        "2020-07-01",  # June is available
        "2020-07-01",  # Unknown customer
    ]),
    "payment_type": [None] * 5,
    "warehouse": [None] * 5,
    "salesperson": [None] * 5,
    "transport_method": [None] * 5,
})

result = recover_categorical_missing(
    observations,
    history,
    config,
)

expected = [
    "Unknown",
    "Card",
    "Card",
    "Transfer",
    "Unknown",
]

print("\n=== CATEGORICAL TEMPORAL SAFETY ===")
print(result[[
    "customer_id",
    "purchase_date",
    "payment_type",
    "payment_type_was_recovered",
]].to_string(index=False))

assert result["payment_type"].tolist() == expected

assert result["payment_type_was_recovered"].tolist() == [
    0, 1, 1, 1, 0
]

assert result["transport_method"].eq("Unknown").all()

print("\nCategorical temporal safety passed.")
