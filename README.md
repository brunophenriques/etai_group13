# etai_group13

Academic machine learning project with leakage-safe preprocessing and temporal
cross-validation. Configuration lives in [`config.yaml`](config.yaml).

Imputation for classification and conditional regression is selected and integrated.
See [integration and reproduction](docs/imputation.md), the executed
[decision notebook](notebooks/missing_values_analysis.ipynb), and the
[experiment report](results/imputation_selection/imputation_summary.txt).

## Project layout

- `src/`: data loading, preprocessing, feature transformation, and temporal splits.
- `tests/`: regression checks and exploratory diagnostics.
- `docs/`: [project structure and preprocessing workflow](docs/README.md).
- `results/`: configured destination for evaluation artifacts.
- `notebooks/`: exploratory data analysis.
- `Project_data/`: raw CSV datasets referenced by the configuration.

Run all existing checks with `python -m tests` from the repository root after
activating the environment. See [verification instructions](tests/README.md).

## Virtual Environment Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
```

### 2. Activate the environment

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Deactivate the environment

```bash
deactivate
```

**Note:** The `.venv/` directory should be included in `.gitignore` and should not be committed to Git.



## Feature Engineering & Feature Selection

### Feature Engineering

Seven engineered features were integrated into the temporal preprocessing pipeline:

- `recency_ratio`
- `purchases_per_year`
- `value_vs_usual`
- `recent_count_share`
- `discount_pct`
- `value_per_line`
- `value_cv`

Feature Engineering is controlled through `preprocessing.feature_engineering` in `config.yaml`.

Engineered features are computed separately within each temporal fold. Missing values introduced by feature calculations are replaced using statistics fitted on the training fold only.

### Feature Selection

Four complementary feature-selection methods were evaluated:

| Method | Category | Purpose |
|---|---|---|
| mRMR | Filter | Maximize target relevance while minimizing redundancy |
| RFECV | Wrapper | Select features through recursive elimination and temporal cross-validation |
| Boruta | All-relevant | Identify relevant features by comparison with randomized shadow features |
| Logistic Regression L1 | Embedded | Identify features retained by L1 regularization |

RFECV uses chronological inner validation splits with a 90-day target-availability embargo. Preprocessing is fitted separately within the inner training splits.

A conservative RFECV selection rule was also introduced: among feature counts whose mean validation F1 is within a configured tolerance of the highest score, the larger feature set is preferred.

### Consensus-Based Selection

Instead of relying on a single algorithm, the pipeline combines their decisions through a rejection-voting mechanism.

Each method classifies an encoded feature as selected or rejected. Boruta may additionally return a tentative decision, which is not counted as a rejection.

Three consensus strategies were evaluated:

- **4/4:** Remove features rejected by all four methods.
- **3/4:** Remove features rejected by at least three methods.
- **2/4:** Remove features rejected by at least two methods.

Selection is performed independently using the training portion of each temporal fold. The resulting feature subset is applied to both training and validation data without refitting selectors on validation data.

The selected strategy can be activated or disabled through `preprocessing.feature_selection.enabled` in `config.yaml`. The applied rejection threshold is configured through `preprocessing.feature_selection.consensus.apply_threshold`.

### Experimental Results

Classification experiments used expanding-window temporal cross-validation over 2020–2023, with a 90-day target-availability embargo.

The following results were obtained for Random Forest before hyperparameter tuning:

| Strategy | Mean F1 | Mean PR-AUC | Mean Features |
|---|---:|---:|---:|
| Baseline (no selection) | 0.73489 | 0.83567 | 96.5 |
| Consensus 4/4 | 0.73434 | 0.83631 | 84.5 |
| Consensus 3/4 | 0.73494 | 0.83632 | 51.5 |
| Consensus 2/4 | **0.73811** | 0.83533 | **32.0** |

Consensus 2/4 reduced the average encoded feature space by approximately **66.8%**, while maintaining comparable predictive performance.

The F1 improvement over baseline was small (+0.00322), and the PR-AUC decreased slightly. Therefore, the results support dimensionality reduction rather than a definitive improvement in generalization.

Across the five evaluated models, Consensus 2/4 achieved the highest mean F1 for Random Forest, CatBoost, XGBoost and LightGBM. Logistic Regression performed best with Consensus 3/4.

### Configuration and Execution

The `config.yaml` file is the single source of configuration for Feature Engineering, Feature Selection, enabled methods, algorithm parameters and consensus thresholds.

Run the configured classification pipeline:

```bash
python -m src.evaluate_models --task classification
```

Run the dedicated consensus comparison:

```bash
python -m src.evaluate_feature_selection
```

Run Feature Selection unit tests:

```bash
python -m pytest tests/test_feature_selection.py -v
```

All eight Feature Selection unit tests passed during the initial integration.

### Methodological Considerations

- Feature Selection is fitted exclusively on training data within each outer temporal fold.
- RFECV uses temporal inner splits, with preprocessing fitted independently within each inner training set.
- Feature selection operates on transformed columns, including individual one-hot-encoded categories.
- Consensus outcomes depend on the configured selection methods and their hyperparameters.
- The four methods optimize different notions of relevance; agreement does not establish causal or universal feature importance.
- The comparison of consensus strategies was exploratory. Choosing the best-performing strategy using the same cross-validation folds introduces model-selection bias.
- Computational costs of the selection methods are not included in the model-training performance metrics.
- The 2024 holdout remains reserved for final evaluation after model selection and hyperparameter tuning.

**Current status:** Consensus 2/4 is integrated as the leading Feature Selection candidate for the next modeling stage. It is not yet a finalized or independently validated model configuration.