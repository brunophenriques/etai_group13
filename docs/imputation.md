# Missing-value imputation

## Decision

**Median for remaining numerical gaps and `Unknown` for missing categories**, for both classification and conditional spending regression. These settings are active in `config.yaml`.

## Treatment applied

| Condition | Action |
|---|---|
| Invalid value | Convert to missing before further processing |
| Insufficient history | Zero placeholder with an indicator |
| Recoverable historical mean | Observed historical spending / prior purchase count |
| Recoverable recency | Interval since the latest strictly earlier training purchase |
| Remaining numerical gap | Median of eligible training values |
| Missing category | `Unknown` |

The latest available earlier training transaction is assumed to be the true previous purchase. Other validation/test rows are never used as history. Structural placeholders are excluded from numerical donors.

For example, four prior purchases totalling EUR 800 recover a missing historical mean as EUR 200. If the total is also missing, the mean receives its own training median instead.

Indicators retain original missingness, recovery, corrections and insufficient history. Their individual predictive value was not tested. Entirely empty donor columns use zero with a dedicated flag. Undated rows are excluded from temporal evaluation but preserved during prediction using calendar placeholders and a missing-date flag. Targets are never imputed.

## Comparison

Five combinations were evaluated:

| Numerical method | Categorical method |
|---|---|
| Median | `Unknown` |
| Median | Mode |
| KNN, 5 neighbours, uniform weights | `Unknown` |
| KNN, 10 neighbours, uniform weights | `Unknown` |
| IterativeImputer with BayesianRidge | `Unknown` |

The iterative method uses `max_iter=10`, `sample_posterior=False` and seed 42: single imputation with Bayesian linear regression. KNN and iterative inputs use `RobustScaler`; remaining numerical features use medians.

Validation uses 2020–2023 folds, three-year training windows and a 90-day embargo. Fixed models are logistic regression/random forest for classification and Ridge/random forest for conditional spending: **80 model evaluations**. All preprocessing is fitted within training folds; 2024 and test data are excluded from selection.

Scores are averaged equally across each task's two models and four folds. We prefer the simplest option within **0.005 F1** of the classification best or **1% MAE** of the regression best. Median + Unknown meets both tolerances. These are selection results, not independent final-performance estimates.

The conditional regressor uses returners with observed spending. Combined predictions assign zero to predicted nonreturners and the regression estimate to predicted returners, at threshold 0.5. Combined monetary evaluation includes actual nonreturners at zero spending and excludes returners with unknown spending.

## Files and reproduction

- [Notebook](../notebooks/missing_values_analysis.ipynb): comparison, plots and conclusion.
- [Experiment report](../results/imputation_selection/imputation_summary.txt): scores, fold details and configuration parameters.
- [Implementation](../src/imputation.py): fitted preprocessing and comparison runner. Experiment settings live here; operational settings remain in `config.yaml`.

From the repository root, with `requirements.txt` installed:

```bash
python -m src.imputation --output results/imputation_reproduction
python -m tests
```

Use a new output directory. The runner records its choice without rewriting the configuration. The notebook reads saved results rather than retraining models.

Existing training uses `prepare_numeric_fold`; its third result contains `__preprocessor__`. `TaskPipeline` is available from `src.imputation`. Both accept raw rows and support joblib persistence without refitting at prediction time. Earlier results remain in `previous_24_combinations/` as an archive.
