
"""Feature Engineering — ETAI Group 13.

Features developed by Joana.
Configuration is read from config.yaml.
"""

import numpy as np
import pandas as pd


def add_engineered_features(df, config):
    """Create only the features enabled in config.yaml."""

    settings = config["preprocessing"].get("feature_engineering", {})

    if not settings.get("enabled", False):
        return df.copy()

    selected = settings["features"]
    data = df.copy()

    # Historical average purchase value
    count = data["prior_purchase_count"].where(
        data["prior_purchase_count"] > 0
    )

    mean_value = (
        data["prior_total_spend"] / count
    ).where(lambda x: x > 0)

    # Feature formulas
    formulas = {
        "recency_ratio": lambda: (
            data["days_since_previous_purchase"]
            / data["prior_mean_gap_days"].where(
                data["prior_mean_gap_days"] > 0
            )
        ),

        "purchases_per_year": lambda: (
            data["prior_purchase_count"]
            / (data["customer_tenure_days"].clip(lower=30) / 365)
        ),

        "value_vs_usual": lambda: (
            data["Total a Pagar"] / mean_value
        ),

        "recent_count_share": lambda: (
            data["purchase_count_90d"]
            / data["purchase_count_365d"].where(
                data["purchase_count_365d"] > 0
            )
        ),

        "discount_pct": lambda: (
            data["Total Desc_ Linha"]
            / data["Total Bruto"].where(
                data["Total Bruto"] > 0
            )
        ),

        "value_per_line": lambda: (
            data["Total Liquido"]
            / data["N_ Detalhes"].where(
                data["N_ Detalhes"] > 0
            )
        ),

        "value_cv": lambda: (
            data["prior_std_purchase_value"] / mean_value
        ),
    }

    # Validate config
    unknown = set(selected) - set(formulas)

    if unknown:
        raise ValueError(
            f"Unknown engineered features in config.yaml: {unknown}"
        )

    # Create only configured features
    for feature in selected:
        data[feature] = formulas[feature]()

    # Replace infinite values with NaN
    data[selected] = data[selected].replace(
        [np.inf, -np.inf], np.nan
    )

    return data


def apply_configured_feature_engineering(df, config):
    """Compatibility wrapper for the existing fold preprocessing."""
    return add_engineered_features(df, config)
