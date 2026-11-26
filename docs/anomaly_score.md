
## Anomaly Score (Isolation Forest and LOF)

### Overview

An optional step adds one feature, `anomaly_score`, to the model matrix: how unusual a purchase is compared with the training purchases (higher = more unusual). It follows the `AnomalyScore` step of the class pipeline (week 11) and supports two detectors:

| Method | Score stored in `anomaly_score` |
|---|---|
| `isolation_forest` | s(x), about 0.5 for ordinary rows and close to 1 for anomalies |
| `lof` | `log(1 + LOF)`, about 0.7 for ordinary rows and larger for anomalies |
| `none` | The step is switched off |

The score is a feature, not a filter: **no row is ever removed**, because every customer needs a prediction (including every test row and any new purchase in the client interface).

The step is controlled through the top-level `anomaly` block of `config.yaml`.

### Implementation

The code is in `src/outliers.py`:

| Piece | What it does |
|---|---|
| `AnomalyScore` | scikit-learn style transformer (`fit`, `transform`, `fit_transform`, `get_feature_names_out`). DataFrames in, DataFrames out, so the feature names survive for feature selection and SHAP |
| `build_anomaly_scorer(config)` | Builds an unfitted `AnomalyScore` from the config, or returns `None` when `method: none` |
| `add_anomaly_score(X_train, X_val, config)` | Fits the detector on `X_train` and appends the score to `X_train` and `X_val`. Returns the three objects, with the scorer being `None` when the step is off |
| `compare_methods(...)` | Runs the evaluation once per method (same folds, same models) and compares them |
| `anomaly_profile(...)` | Return rate (classification) or mean spend (regression) of the 5% most anomalous validation rows |

Design decisions:

- **Fitted on the training rows only.** Any row (validation, test, a new customer) is then scored on its own against the training rows, never against other validation or test rows.
- **LOF uses `novelty=True`**, which is what allows scoring new rows. Training rows are scored with `negative_outlier_factor_` in `fit_transform`, so a row is not counted as its own neighbour; `score_samples` on the training rows would give systematically different values from those of new rows.
- **`log(1 + LOF)`.** The raw LOF has a median of about 1.1 and a 99th percentile of about 3, but a few purchases reach more than 80 000. In a check on fold 2022 (regression, ridge on `log1p(spend)`, continuous columns), the largest predicted spend was 69 895 EUR with `log(1 + LOF)` and about 8e49 EUR with the raw LOF.
- **`n_neighbors = 20`** must be larger than the largest group of identical rows. This is checked by `tests/test_outliers.py` on a real temporal fold.
- **`exclude_columns: [purchase_year]`.** A calendar column drifts over time, so every future purchase would look anomalous. Columns are matched by the name after the block prefix (`numeric__purchase_year` is matched by `purchase_year`).
- **`columns: all | continuous`.** `all` uses every column of the matrix; `continuous` keeps the columns with more than two distinct values, which leaves out one-hot and 0/1 columns.

### Where it Runs

The score is added **after the feature transformer and before feature selection**, so the selectors can keep or drop it like any other feature:

```text
imputation -> feature engineering -> encoding and scaling -> anomaly score -> feature selection -> model
```

| Place | Behaviour |
|---|---|
| `src/evaluate_models.py` | Classification: detector fitted on the training fold. Regression: fitted on the returners of the training fold |
| `src/tune_models.py` | Detector fitted on each training fold (`src/outliers.py` is part of the tuning cache key, so cached trials are not reused after a change) |
| `src/feature_selection.py` | mRMR, Boruta and L1 see the score as one more column. RFECV rebuilds the features of every inner split, so it also fits a new detector on each inner training split (without this, the column would be filled with 0.0 for every inner row) |
| `src/imputation.py` (`TaskPipeline`) | Detector fitted on all training predictors, without labels, and saved with the pipeline, so `predict` scores new customers |

### Experimental Results

Temporal validation (folds 2020-2023, 90-day embargo). Imputation, encoding, scaling and detector are fitted on each training fold only. Models: logistic regression, random forest and hist gradient boosting (classification); ridge, random forest and hist gradient boosting, with the spend in euros and in log (regression). Classification was run without feature selection to isolate the effect of the score.

**Classification** (mean over folds and models)

| Method | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|
| none | 0.7148 | 0.7783 | 0.8353 |
| isolation_forest | 0.7177 | 0.7769 | 0.8338 |
| lof | 0.7152 | 0.7792 | 0.8361 |

Difference to `none`: F1 +0.0029 (isolation forest) and +0.0004 (LOF), against a spread across folds of 0.006 and 0.011.

**Regression** (mean over folds and models)

| Method | RMSE (EUR) | MAE (EUR) |
|---|---|---|
| none | 2736.3 | 1444.1 |
| isolation_forest | 2743.4 | 1444.8 |
| lof | 2757.2 | 1447.1 |

Difference to `none`: RMSE +7.1 and +20.8, against a spread across folds of 41 and 66. The ridge on `log1p(spend)` has an RMSE of about 5 000 in every version: it comes from converting extreme linear predictions back with `expm1`, not from the score.

**Are the most anomalous purchases different?** Top 5% of each validation fold by score, against the rest:

| Method | Return rate, top 5% | Return rate, the rest | Mean spend of returners, top 5% | Mean spend, the rest |
|---|---|---|---|---|
| isolation_forest | 77.9% | 53.9% | 5 377 EUR | 1 845 EUR |
| lof | 62.8% | 54.7% | 2 617 EUR | 1 993 EUR |

**Reading.** No method improves the models by more than the variation between folds. The most anomalous purchases are not errors: they are customers with a large purchase history, who return more often and spend more. The models already read that information from the purchase counts and the spending windows, which is why the score adds nothing.

**Decision:** the step is implemented, tested and available, but **switched off** (`anomaly.method: none`).

### Configuration and Execution

```yaml
anomaly:
  method: none                  # none | isolation_forest | lof
  n_estimators: 100             # isolation_forest: number of trees
  n_neighbors: 20               # lof: larger than the largest group of identical rows
  columns: all                  # all | continuous
  exclude_columns:              # never used by the detector
    - purchase_year
  experiment:                   # only used by python -m src.outliers
    methods: [none, isolation_forest, lof]    # "none" is the reference
    models:
      classification: [logistic_regression, random_forest, hist_gradient_boosting]
      regression: [ridge, random_forest, hist_gradient_boosting]
```

Compare the methods on the same folds:

```bash
python -m src.outliers --task classification --skip-selection   # faster
python -m src.outliers --task classification                    # with feature selection
python -m src.outliers --task regression
```

Results are saved in `results/outliers_<task>/<timestamp>/`. A method only matters if its difference to `none` is larger than the spread across folds (`*_delta_std`).

To use a detector in the normal pipeline, set `anomaly.method` to `isolation_forest` or `lof` and run `python -m src.evaluate_models`.

### Tests

`tests/test_outliers.py` checks:

- the class contract: one extra column, same index, input not modified, obvious extremes get the highest score and the feature names are kept;
- no leakage: a row scored alone gets the same score as in a batch, a calendar column never affects the score, and the detector depends only on the training rows;
- LOF: training scores come from `negative_outlier_factor_`, the score is `log(1 + LOF)`, and the largest group of identical rows is below `n_neighbors`;
- the configuration (`none` switches the step off, unknown methods fail) and explicit errors (non-finite input, existing `anomaly_score` column);
- feature selection: RFECV receives real scores in every inner fit, and the selection judges `anomaly_score`;
- the prediction path: the fitted `TaskPipeline` scores new customers.

### Methodological Considerations

- The comparison uses the development folds only; the holdout year and the test set were not used.
- Only three models per task were compared, and the classification comparison ran without feature selection. With the score switched on, feature selection may keep or drop it.
- The regression detector is fitted on the returners in `evaluate_models` but on all training rows in `TaskPipeline`. The scores seen when evaluating and when predicting therefore come from slightly different detectors; the same convention should be used in both.
- Scripts that apply the feature transformer outside `evaluate_models`, tuning and `TaskPipeline` (for example the feature-selection and diagnostic scripts) do not add the score. They are only consistent with the model matrix while `anomaly.method` is `none`.
- `drop_train` and `flag` (earlier versions of this experiment) were tested without gain and are not part of the step.
