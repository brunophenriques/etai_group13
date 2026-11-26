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

Run development verification with the explicit pytest and module commands in
[tests/README.md](tests/README.md). The pytest command executes the new integration
and cache tests. The legacy `python -m tests` command includes holdout checks and
does not execute pytest functions; do not use it for development verification.

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

The pipeline integrates seven engineered features and four feature-selection methods (mRMR, RFECV, Boruta, and Logistic Regression L1) using a configurable consensus-based approach.

Consensus 2/4 was selected as the leading candidate, reducing the average number of encoded features by 66.8% while maintaining comparable predictive performance.

Feature Engineering and Feature Selection are performed within temporal training folds to prevent validation-data leakage.

For implementation details, experimental results, configuration, and methodological limitations, see [Feature Engineering & Feature Selection](docs/feature_engineering_selection.md).

## Temporal Validation

The project uses expanding-window temporal cross-validation (2020–2023) with a 90-day target-availability embargo.

Temporal diagnostics identified changes in customer repurchase behavior and probability calibration during 2022. A three-year sliding-window alternative was evaluated but did not improve overall predictive performance.

See [Temporal Validation Diagnostics](docs/temporal_validation.md) for the analysis and experimental comparison.