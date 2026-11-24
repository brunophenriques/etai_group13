# Temporal Validation & Feature Importance Diagnostics

## 1. Overview

This document describes the temporal validation diagnostics conducted for the ETAI Group 13 classification pipeline.

The investigation was motivated by the comparatively lower F1 score observed in the 2022 validation fold. The objective was to determine whether this performance difference resulted from an issue in the temporal splitting methodology, changes in customer purchasing behavior, feature relevance instability, probability calibration, or the choice of training window.

The experiments focus on Random Forest with Consensus Feature Selection 2/4.

The 2024 holdout dataset was not used in any of the experiments described here.

## 2. Temporal Cross-Validation Strategy

The classification pipeline uses expanding-window temporal cross-validation with four annual validation folds.

The configuration is defined in `config.yaml`:

```yaml
validation:
  strategy: expanding_window
  target_horizon_days: 90
  cv_years: [2020, 2021, 2022, 2023]
  holdout_year: 2024
  sliding_window:
    train_years: 3
```

The target indicates whether a customer purchases again within 90 days. Therefore, training observations must have fully observable target outcomes before the validation period begins.

A 90-day target-availability embargo is applied to avoid using training labels that would not have been available at the start of validation.

### 2.1. Temporal Fold Verification

| Validation Year | Training Rows | Validation Rows | Gap (Days) | Embargo Valid |
|---|---:|---:|---:|---|
| 2020 | 2,578 | 670 | 95 | Yes |
| 2021 | 3,234 | 782 | 91 | Yes |
| 2022 | 4,004 | 628 | 91 | Yes |
| 2023 | 4,705 | 767 | 91 | Yes |

The temporal separation and target-availability embargo were verified for all four folds.

The checks found no overlapping training and validation rows and no violations of the configured target-availability embargo.

These checks validate the splitting constraints but do not independently establish that every upstream feature is free from information leakage.

## 3. Target Distribution Shift

The first investigation compared the proportion of positive target labels between training and validation periods.

| Fold | Train Positive Rate | Validation Positive Rate | Difference |
|---|---:|---:|---:|
| 2020 | 55.97% | 59.40% | +3.43 pp |
| 2021 | 55.29% | 54.86% | -0.43 pp |
| 2022 | 56.24% | 48.09% | **-8.15 pp** |
| 2023 | 55.35% | 58.15% | +2.80 pp |

The 2022 fold exhibited the largest negative change in target prevalence.

The training data contained approximately 56.24% positive cases, compared with 48.09% in validation.

This indicates a substantial change in the observed frequency of repeat purchases during 2022, potentially affecting threshold-dependent classification metrics and probability calibration.

However, a change in target prevalence alone does not establish concept drift.

## 4. Feature Distribution Shift

Numerical feature distributions were compared between training and validation using the Kolmogorov–Smirnov (KS) statistic.

Several variables exhibited distributional differences in 2022.

| Feature | Training Median | Validation Median | KS Statistic |
|---|---:|---:|---:|
| `customer_tenure_days` | 595 | 1,603 | 0.462 |
| `prior_total_spend` | 4,081 | 10,685 | 0.277 |
| `prior_purchase_count` | 7 | 18 | 0.258 |
| `prior_max_purchase_value` | 1,525 | 2,460 | 0.244 |
| `days_since_previous_purchase` | 54 | 89 | 0.144 |

These results suggest changes in customer tenure, purchase frequency, transaction history, and purchasing intervals.

However, customer tenure also exhibited substantial distributional changes in other validation years. This is consistent with the natural evolution of the customer population and the use of an expanding historical training window.

Consequently, these findings are treated as evidence of covariate distribution differences rather than proof of a unique 2022 concept drift event.

## 5. Feature Importance and Temporal Stability

Feature importance was evaluated separately for each temporal fold using Permutation Importance.

Two scoring metrics were considered:

- F1 score, using the configured classification threshold of 0.5.
- PR-AUC, computed using average precision.

Native Random Forest feature importance was also recorded for comparison.

### 5.1. Most Important Features

The five highest-ranked features by average permutation PR-AUC importance were:

| Feature | Mean PR-AUC Importance | Folds Selected |
|---|---:|---:|
| `purchases_per_year` | 0.01388 | 4/4 |
| `purchase_count_365d` | 0.01297 | 4/4 |
| `purchase_count_180d` | 0.00304 | 4/4 |
| `discount_pct` | 0.00254 | 4/4 |
| `spend_365d` | 0.00251 | 4/4 |

Purchase frequency was the strongest recurring source of predictive information.

The engineered feature `purchases_per_year` achieved the highest average permutation importance, supporting its usefulness within the evaluated Random Forest pipeline.

The engineered feature `discount_pct` also showed a positive contribution.

### 5.2. Interpretation

The primary purchase-frequency features remained useful across all four validation folds, including 2022.

Therefore, the lower 2022 F1 score cannot be attributed simply to a complete loss of relevance among the principal predictive features.

Permutation Importance measures the change in model performance when feature values are shuffled. It does not represent a direct model coefficient or establish causal relationships.

Importance may also be underestimated for correlated features, since remaining variables can preserve overlapping predictive information.

Additionally, the feature subsets differ between folds because Consensus Feature Selection is fitted independently on each training period.

## 6. Classification Performance in 2022

Random Forest with Consensus Feature Selection 2/4 produced the following results:

| Fold | F1 | ROC-AUC | PR-AUC | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| 2020 | 0.7318 | 0.7266 | 0.8203 | 0.7202 | 0.7437 |
| 2021 | 0.7533 | 0.8097 | 0.8567 | 0.7707 | 0.7366 |
| 2022 | **0.7162** | 0.7925 | 0.8076 | 0.6611 | 0.7815 |
| 2023 | 0.7512 | 0.7915 | 0.8568 | 0.7725 | 0.7309 |

Although 2022 had the lowest F1 score, its ROC-AUC was nearly identical to that of 2023.

This suggests that the model retained a similar ability to rank positive observations above negative observations, despite differences in threshold-based classification performance.

The lower 2022 precision, together with relatively high recall, indicates an elevated number of false positive predictions.

### 6.1. Confusion Matrix — 2022

At the configured threshold of 0.5:

| | Predicted Negative | Predicted Positive |
|---|---:|---:|
| Actual Negative | 205 | 121 |
| Actual Positive | 66 | 236 |

The model predicted 56.85% of validation observations as positive, while the observed positive rate was only 48.09%.

This mismatch contributed to the lower precision and F1 score.

## 7. Classification Threshold Sensitivity

F1, precision, and recall were evaluated across classification thresholds between 0.30 and 0.70.

| Fold | Best Evaluated Threshold | Best Evaluated F1 | F1 at 0.50 |
|---|---:|---:|---:|
| 2020 | 0.45 | 0.7401 | 0.7318 |
| 2021 | 0.35 | 0.7636 | 0.7533 |
| 2022 | **0.50** | **0.7162** | **0.7162** |
| 2023 | 0.40 | 0.7603 | 0.7512 |

For 2022, the existing threshold of 0.5 produced the highest F1 among the evaluated thresholds.

Therefore, a simple change to the classification threshold did not resolve the observed performance reduction.

These threshold comparisons are exploratory. Selecting a threshold using the same validation labels and subsequently reporting its performance on that validation set would introduce selection bias.

Any future threshold optimization must be performed using training data and appropriate internal temporal validation.

## 8. Probability Calibration

Probability calibration was investigated using predicted probabilities, observed positive rates, and the Brier score.

### 8.1. Mean Probability Calibration

| Fold | Mean Predicted Probability | Actual Positive Rate | Calibration Gap |
|---|---:|---:|---:|
| 2020 | 58.68% | 59.40% | -0.72 pp |
| 2021 | 55.19% | 54.86% | +0.33 pp |
| 2022 | **57.74%** | **48.09%** | **+9.66 pp** |
| 2023 | 54.13% | 58.15% | -4.01 pp |

The 2022 model systematically overestimated repeat-purchase probabilities on average.

This indicates a mismatch between predicted probabilities and observed outcomes, despite relatively stable ROC-AUC.

### 8.2. Calibration by Probability Interval

In 2022, overestimation was particularly evident in intermediate probability intervals.

| Predicted Probability Interval | Mean Prediction | Observed Positive Rate | Gap |
|---|---:|---:|---:|
| 30–40% | 34.9% | 23.5% | +11.3 pp |
| 40–50% | 44.9% | 30.7% | +14.2 pp |
| 50–60% | 54.8% | 43.2% | +11.6 pp |
| 60–70% | 65.0% | 50.6% | +14.4 pp |
| 70–80% | 75.3% | 65.3% | +10.0 pp |
| 80–90% | 84.4% | 83.3% | +1.1 pp |
| 90–100% | 98.1% | 97.0% | +1.1 pp |

The largest calibration discrepancies occurred primarily between predicted probabilities of 0.30 and 0.80.

Predictions above 0.80 were more closely aligned with observed outcomes.

This suggests that overestimation was concentrated in intermediate-risk predictions rather than uniformly affecting the entire probability range.

## 9. Monthly Analysis of 2022

The 2022 validation predictions were grouped by purchase month to investigate whether the calibration mismatch occurred consistently throughout the year.

The largest positive calibration gaps were observed in July, October, and November.

| Month | Observations | Actual Positive Rate | Mean Predicted Probability | Calibration Gap |
|---|---:|---:|---:|---:|
| July | 104 | 38.46% | 57.25% | +18.78 pp |
| October | 61 | 34.43% | 56.53% | +22.11 pp |
| November | 68 | 35.29% | 55.68% | +20.38 pp |

Together, these months accounted for 233 of the 628 validation observations, approximately 37%.

Other months, including April, May, June, and September, showed considerably smaller calibration gaps.

This indicates that the overestimation observed in 2022 was not uniform throughout the year.

### 9.1. Monthly Discrimination

Despite the large calibration gap, October 2022 achieved a ROC-AUC of approximately 0.8881.

This supports the interpretation that probability calibration and ranking performance can behave differently.

However, monthly sample sizes are limited, and the observed differences should not be interpreted as definitive evidence of seasonal or structural change.

No specific external event or causal explanation for the 2022 distribution shift was established.

## 10. Expanding Window vs. Sliding Window

A controlled comparison evaluated the existing expanding-window strategy against a sliding window restricted to the previous three calendar years.

Both strategies used:

- The same validation years: 2020–2023.
- Identical validation observations.
- A 90-day target-availability embargo.
- The same Random Forest configuration.
- Consensus Feature Selection with rejection threshold 2/4.
- A classification threshold of 0.5.

Preprocessing and Feature Selection were fitted separately within each strategy's training fold.

### 10.1. Average Cross-Validation Results

| Metric | Expanding Window | Sliding Window (3 Years) |
|---|---:|---:|
| Mean F1 | **0.7381** | 0.7194 |
| Mean Precision | 0.7311 | **0.7464** |
| Mean Recall | **0.7482** | 0.7007 |
| Mean ROC-AUC | **0.7801** | 0.7729 |
| Mean PR-AUC | **0.8353** | 0.8317 |
| Mean Brier Score (lower is better) | **0.1912** | 0.1936 |

The expanding-window strategy outperformed the three-year sliding window in mean F1, recall, ROC-AUC, PR-AUC, and Brier score.

The sliding window produced slightly higher mean precision but substantially lower recall.

### 10.2. F1 by Validation Fold

| Fold | Expanding Window | Sliding Window | Difference (Sliding - Expanding) |
|---|---:|---:|---:|
| 2020 | 0.7318 | **0.7334** | +0.0016 |
| 2021 | **0.7533** | 0.7407 | -0.0125 |
| 2022 | **0.7162** | 0.6978 | -0.0184 |
| 2023 | **0.7512** | 0.7054 | -0.0457 |

The sliding window improved F1 slightly in 2020 but performed worse in the remaining three folds.

The largest decline occurred in 2023.

### 10.3. Specific Results for 2022

| Metric | Expanding Window | Sliding Window |
|---|---:|---:|
| Training Rows | 4,004 | 1,863 |
| Selected Features | 34 | 29 |
| F1 | **0.7162** | 0.6978 |
| ROC-AUC | **0.7925** | 0.7832 |
| PR-AUC | **0.8076** | 0.8047 |
| Brier Score | **0.1931** | 0.1942 |
| Mean Calibration Gap | +9.66 pp | +8.48 pp |

The sliding window slightly reduced the mean probability overestimation in 2022, but predictive performance deteriorated.

This suggests that older observations retained useful predictive information and that simply restricting training to more recent years was insufficient to address the observed shift.

The experiment compares complete pipelines rather than isolating training-window length: each strategy can select a different feature subset and train on different numbers of observations.

## 11. Final Decision

**The expanding-window temporal validation strategy will be retained.**

This decision is supported by the following findings:

1. All four folds respected the configured 90-day target-availability embargo.
2. The lower 2022 F1 score was associated with changes in target prevalence and probability calibration rather than an identified splitting error.
3. The principal purchase-frequency features remained relevant across validation years.
4. The three-year sliding window did not improve overall performance or resolve the 2022 degradation.
5. Expanding-window validation preserved more historical observations and achieved stronger average predictive performance across the evaluated folds.

The 2022 validation fold will remain part of the evaluation. Excluding difficult periods would reduce the representativeness of temporal performance assessment and could produce overly optimistic conclusions.

The selected approach does not eliminate temporal distribution shifts. Instead, these shifts remain part of the model's generalization challenge.

## 12. Methodological Limitations and Future Work

Several limitations should be considered when interpreting these results:

- The analysis is exploratory and relies on the same four validation years used during model development.
- Comparing different strategies on these folds introduces model-selection uncertainty.
- Only one sliding-window length (three years) was evaluated.
- Different training windows also lead to different Feature Selection results, making their effects difficult to isolate.
- The observed calibration differences do not independently prove concept drift.
- Permutation Importance may underestimate the contribution of correlated features.
- Monthly analyses have relatively small sample sizes and may exhibit substantial sampling variability.
- A well-calibrated model on average is not necessarily calibrated across all probability ranges or customer segments.
- The 2024 holdout must remain untouched until model selection and hyperparameter tuning are completed.

Potential future investigations include training-only probability calibration, drift-aware modeling, customer-segment analysis, and temporal stability monitoring.

These approaches would require separate validation before inclusion in the final pipeline.

## 13. Reproduction

The diagnostic scripts are located in `src/`.

### Temporal Fold Diagnostics

```bash
python -m src.diagnose_temporal_folds
```

### Feature Importance

```bash
python -m src.analyze_feature_importance
```

### Threshold and Probability Calibration

```bash
python -m src.diagnose_classification_threshold
```

### Monthly Drift Analysis

```bash
python -m src.analyze_monthly_drift
```

### Training-Window Comparison

```bash
python -m src.compare_temporal_windows
```

All experiments use the shared configuration in `config.yaml`.

Experiment outputs are stored in timestamped directories under `results/`.

---

**Conclusion:** Expanding-window validation with a 90-day target-availability embargo remains the preferred strategy for the current classification pipeline. The 2022 results reveal a meaningful temporal generalization and calibration challenge, but the diagnostics do not justify excluding that fold or replacing the existing validation strategy with a three-year sliding window.