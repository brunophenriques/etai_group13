# etai_group13

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