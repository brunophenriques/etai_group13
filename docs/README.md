# Project organization and preprocessing

`config.yaml` at the repository root is the central configuration. Paths are
resolved relative to that root by `src.data_loader`, independently of the working
directory. Raw CSVs remain in `Project_data/` and exploration in `notebooks/`.
The existing [EDA findings](../eda_findings.md) document the preprocessing decisions.

## Source responsibilities

| Module | Responsibility |
| --- | --- |
| `src.data_loader` | Load configuration and raw datasets. |
| `src.validation` | Create temporal splits and expanding-window CV folds. |
| `src.preprocessing` | Prepare rows and features; recover, clean, and impute missing values. |
| `src.fold_preprocessing` | Coordinate numeric and categorical preparation within each fold. |
| `src.feature_transformer` | Fit and apply categorical encoders and numeric scalers. |

The preprocessing modules have complementary roles. `prepare_numeric_fold`
retains its existing name for compatibility; it also performs categorical
recovery after numeric preparation. The individual recency imputer helpers remain
available for diagnostic use alongside the general numeric imputer helpers.

## Temporal safety and execution order

1. Load data and call `prepare_rows` for deterministic date preparation.
2. Use `temporal_cv` to split observations. Training purchase dates plus the
   configured target horizon must be strictly earlier than validation starts.
3. Call `prepare_numeric_fold` separately for each fold. It fills structural
   missingness, recovers historical means and recency, cleans invalid numeric
   values, fits training medians, and recovers categories. Recovery uses the raw
   training fold as reference history and excludes same-date matches.
4. Fit `fit_feature_transformer` on the prepared training fold only. Apply that
   fitted transformer to training and validation through `apply_feature_transformer`.
   Identifiers, raw dates, and targets are excluded from model inputs.

Do not prepare all data with fitted imputers, encoders, or scalers before splitting.
The full-data imputer checks in `tests/test_recency.py` are diagnostics; production
CV fits preprocessing separately on each training fold.

## Directory conventions

- `src/`: reusable implementation; existing absolute `src.*` imports remain stable.
- `tests/`: executable assertion scripts and `check_*` exploratory diagnostics.
- `docs/`: project organization and workflow documentation.
- `results/`: destination configured for generated evaluation outputs.

See [test instructions](../tests/README.md) for verification commands.
