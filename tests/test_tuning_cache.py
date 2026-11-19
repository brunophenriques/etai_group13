
"""Verify cache reuse and invalidation."""

from tempfile import TemporaryDirectory

import pandas as pd

from src.tuning_cache import TuningCache


def main():
    with TemporaryDirectory() as directory:
        cache = TuningCache(directory)

        df = pd.DataFrame({"value": [1, 2, 3]})
        config = {
            "preprocessing": {"scaling": "robust"},
            "columns": {"classification_target": "target"},
        }

        key = cache.make_key(
            outer_year=2020,
            inner_index=1,
            config=config,
            train_df=df,
            val_df=df.iloc[:1],
            code_version="test-v1",
        )

        calls = []

        def compute():
            calls.append(1)
            return {"matrix": [10, 20, 30]}

        first = cache.get_or_compute(key, compute)
        second = cache.get_or_compute(key, compute)

        assert first == second
        assert len(calls) == 1

        changed_key = cache.make_key(
            outer_year=2020,
            inner_index=1,
            config=config,
            train_df=df,
            val_df=df.iloc[:1],
            code_version="test-v2",
        )

        assert changed_key != key

        print("Tuning cache tests passed.")


if __name__ == "__main__":
    main()
