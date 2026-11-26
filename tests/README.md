# Verification scripts

Run from the repository root after installing `requirements.txt`. The primary
pytest command executes the new pipeline integration and cache invalidation tests,
as well as the consensus-selection tests:

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider tests/test_pipeline_integration.py tests/test_feature_selection.py tests/test_tuning_cache.py
```

On Linux/macOS, replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`.
These tests load development transactions only, or use synthetic data. Cache and
serialization checks use temporary storage or memory.

Additional development-only assertion scripts retain their module entry points:

```powershell
.\.venv\Scripts\python.exe -B -m tests.test_full_temporal_preprocessing
.\.venv\Scripts\python.exe -B -m tests.test_outliers
.\.venv\Scripts\python.exe -B -m tests.test_inner_cv
.\.venv\Scripts\python.exe -B -m tests.test_imputation_contract
```

The legacy `python -m tests` runner is unchanged. It runs each script in a separate
process, but does not execute pytest test functions. It also includes `test.py`,
which accesses the actual 2024 holdout, and legacy loaders that open test data.
**Do not use that aggregate command for development or integration verification.**
Likewise, unrestricted `pytest tests` imports legacy full-data checks. Use the
explicit development commands above rather than silently skipping their assertions.

Legacy script inventory:

| Script | Coverage |
| --- | --- |
| `test.py` | Actual holdout split; excluded from development verification. |
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


The historical-count assertions in `test_historical_mean.py` (194 recovered means)
and `test_recency.py` (192 recovered recencies) refer to their original full-CSV
input. A development-only audit observed 178 and 166 respectively. The same
failures reproduce with the unchanged `main` tests and preprocessing on that
restricted population. Those counts do not establish a recovery algorithm failure,
nor do they establish that the original full-input expectations are wrong. The
expected values and assertions remain unchanged; final holdout data was not used
to validate or replace them.
