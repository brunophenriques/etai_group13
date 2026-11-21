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
