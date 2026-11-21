# Verification scripts

Run from the repository root after installing `requirements.txt`:

```powershell
.\.venv\Scripts\python.exe -m tests
```

On Linux/macOS, use `.venv/bin/python -m tests`.

The runner discovers all check/test scripts (currently 18) in separate Python processes, retains
their output and assertions, continues to report all failures, and exits nonzero
if any script fails. Raw CSVs referenced by `config.yaml` must be available.
These scripts execute checks at module level; they do not require pytest.

| Script | Coverage |
| --- | --- |
| `test.py` | Holdout split, temporal CV, horizon gap, disjoint indices. |
| `test_preprocessing.py` | Deterministic row and model feature preparation diagnostics. |
| `test_structural_missing.py` | Structural missingness and original-value preservation. |
| `test_historical_mean.py` | Formula recovery and original-value preservation. |
| `test_recency.py` | Recency, median isolation, spend cleaning, and maximum-value consistency. |
| `test_numerical_fold.py` | Numerical preprocessing within a temporal fold. |
| `test_categorical_fold.py` | Categorical preprocessing within a temporal fold. |
| `test_categorical_temporal_safety.py` | Strict historical recovery and fallback safety. |
| `test_feature_transformer.py` | Encoding, scaling, and matching feature frames. |
| `test_full_temporal_preprocessing.py` | Complete preprocessing across all configured CV years. |
| `check_customers.py` | Customer metadata consistency diagnostics. |
| `check_feature_leakage.py` | Customer history count diagnostics. |
| `check_features.py` | Historical maximum versus total spend diagnostics. |
| `check_missing_values.py` | Missingness and categorical history diagnostics. |

Individual scripts retain their existing module entry points, for example:

```powershell
.\.venv\Scripts\python.exe -m tests.test_full_temporal_preprocessing
```

Keep assertions enabled: do not run these checks with Python's `-O` option.

`test_imputation_contract.py` covers median/KNN/iterative preprocessing, structural
donor exclusion, empty columns, missing dates, unknown customers, label-free inputs,
query batch independence, target isolation, serialization of complete task pipelines,
and the declared selection tolerances. See `docs/imputation.md` for the experiment.
