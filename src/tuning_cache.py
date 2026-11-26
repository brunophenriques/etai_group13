
"""Reusable cache for temporal tuning experiments."""

import hashlib
import json
from pathlib import Path

import joblib


class TuningCache:
    """Store and reuse prepared temporal-fold data."""

    def __init__(self, cache_dir="data/cache/tuning"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def make_key(
        self,
        outer_year,
        inner_index,
        config,
        train_df,
        val_df,
        code_version,
    ):
        """Hash fold data, preprocessing settings and code version."""

        fingerprint = {
            "outer_year": outer_year,
            "inner_index": inner_index,
            "preprocessing": config["preprocessing"],
            "columns": config["columns"],
            "anomaly": config.get("anomaly"),
            "validation": config.get("validation"),
            "random_state": config.get("random_state"),
            "train_data": joblib.hash(train_df),
            "val_data": joblib.hash(val_df),
            "code_version": code_version,
        }

        serialized = json.dumps(
            fingerprint,
            sort_keys=True,
            default=str,
        )

        return hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest()

    def get_or_compute(self, key, compute_fn):
        """Load cached data or compute and save it."""

        cache_path = self.cache_dir / f"{key}.joblib"

        if cache_path.exists():
            print(f"[CACHE HIT] {key[:12]}")
            return joblib.load(cache_path)

        print(f"[CACHE MISS] {key[:12]}")

        result = compute_fn()

        temporary_path = cache_path.with_suffix(".tmp")
        try:
            joblib.dump(result, temporary_path)
            temporary_path.replace(cache_path)
        finally:
            temporary_path.unlink(missing_ok=True)

        return result
