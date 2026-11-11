"""Run every existing assertion script and diagnostic in isolated processes."""

import subprocess
import sys
from pathlib import Path


def main():
    """Run all checks, report failures, and return a nonzero status if any fail."""
    tests_directory = Path(__file__).resolve().parent
    project_root = tests_directory.parent
    scripts = sorted(
        path
        for path in tests_directory.glob("*.py")
        if path.name == "test.py" or path.name.startswith(("test_", "check_"))
    )
    failures = []

    for script in scripts:
        module = f"tests.{script.stem}"
        result = subprocess.run(
            [sys.executable, "-m", module],
            cwd=project_root,
            capture_output=True,
            text=True,
        )
        print(result.stdout, end="", flush=True)
        print(result.stderr, end="", file=sys.stderr, flush=True)
        if result.returncode:
            failures.append(module)
        status = "FAIL" if result.returncode else "PASS"
        print(f"{status}: {module}", flush=True)

    print(f"\n{len(scripts) - len(failures)}/{len(scripts)} scripts passed.")
    if failures:
        print("Failed scripts: " + ", ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
