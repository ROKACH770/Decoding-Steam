from pathlib import Path
import subprocess
import sys
import time


PROJECT_DIR = Path(__file__).resolve().parent

QUESTIONS = [
    ("Question 1", Path("Q1/question1.py")),
    ("Question 2", Path("Q2/question2.py")),
    ("Question 3", Path("Q3/question3.py")),
]


def run_question(name, relative_path):
    """Run one question with the active Python environment."""
    script = PROJECT_DIR / relative_path
    if not script.exists():
        raise FileNotFoundError(f"Could not find {script}")

    print(f"\n{'=' * 60}\nRunning {name}: {relative_path}\n{'=' * 60}", flush=True)
    started = time.perf_counter()
    result = subprocess.run([sys.executable, str(script)], cwd=script.parent)

    if result.returncode != 0:
        raise SystemExit(f"\n{name} failed with exit code {result.returncode}. The remaining questions were not run.")

    elapsed = time.perf_counter() - started
    print(f"{name} finished in {elapsed:.1f} seconds.", flush=True)


def main():
    """Run all three question scripts in order."""
    started = time.perf_counter()
    for name, path in QUESTIONS:
        run_question(name, path)

    elapsed = time.perf_counter() - started
    print(f"\nAll questions finished successfully in {elapsed:.1f} seconds.")


if __name__ == "__main__":
    main()
