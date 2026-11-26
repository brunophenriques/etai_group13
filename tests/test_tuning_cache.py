
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


def test_anomaly_configuration_invalidates_cached_matrices(tmp_path):
    from copy import deepcopy

    cache = TuningCache(tmp_path)
    frame = pd.DataFrame({"x": [1, 2, 3]})
    config = {"preprocessing": {"feature_selection": {"enabled": True}},
              "columns": {}, "anomaly": {"method": "none"},
              "validation": {"target_horizon_days": 90}, "random_state": 42}
    calls = []
    keys = []
    for method in ["none", "isolation_forest", "lof", "none"]:
        cfg = deepcopy(config)
        cfg["anomaly"]["method"] = method
        key = cache.make_key(2020, 1, cfg, frame, frame, "v1")
        keys.append(key)
        def compute():
            calls.append(method)
            return method
        assert cache.get_or_compute(key, compute) == method
    assert len(set(keys)) == 3
    assert calls == ["none", "isolation_forest", "lof"]


def test_feature_matrix_settings_invalidate_cache(tmp_path):
    from copy import deepcopy

    cache = TuningCache(tmp_path)
    frame = pd.DataFrame({"x": [1, 2]})
    base = {"preprocessing": {"feature_selection": {"enabled": True}},
            "columns": {}, "anomaly": {"method": "lof", "n_neighbors": 20},
            "validation": {"target_horizon_days": 90}, "random_state": 42}
    def key(cfg):
        return cache.make_key(2020, 1, cfg, frame, frame, "v1")
    variants = []
    for section, name, value in [("preprocessing", "feature_selection", {"enabled": False}),
                                 ("anomaly", "n_neighbors", 10),
                                 ("validation", "target_horizon_days", 91)]:
        cfg = deepcopy(base)
        cfg[section][name] = value
        variants.append(cfg)
    cfg = deepcopy(base)
    cfg["random_state"] = 43
    variants.append(cfg)
    assert all(key(cfg) != key(base) for cfg in variants)
