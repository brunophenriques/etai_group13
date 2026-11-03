from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config():
    config_path = PROJECT_ROOT / "config.yaml"

    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_data(config):
    paths = config["paths"]

    train = pd.read_csv(PROJECT_ROOT / paths["train"])
    test = pd.read_csv(PROJECT_ROOT / paths["test"])
    customers = pd.read_csv(PROJECT_ROOT / paths["customers"])

    return train, test, customers


if __name__ == "__main__":
    config = load_config()
    train, test, customers = load_data(config)

    print(f"Train: {train.shape}")
    print(f"Test: {test.shape}")
    print(f"Customers: {customers.shape}")
    train_dates = pd.to_datetime(train["purchase_date"], errors="coerce")
    test_dates = pd.to_datetime(test["purchase_date"], errors="coerce")

    print("\nTRAIN")
    print("Earliest:", train_dates.min())
    print("Latest:", train_dates.max())
    print("Missing dates:", train_dates.isna().sum())

    print("\nTEST")
    print("Earliest:", test_dates.min())
    print("Latest:", test_dates.max())
    print("Missing dates:", test_dates.isna().sum())

    print("\nTRAIN OBSERVATIONS PER YEAR")
    print(train_dates.dt.year.value_counts().sort_index())