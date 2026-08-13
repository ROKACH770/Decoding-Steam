import subprocess
import sys
from pathlib import Path


def run(filename):
    path = Path(__file__).resolve().parent / filename
    print(f"\nRunning {filename}...", flush=True)
    subprocess.run([sys.executable, str(path)], cwd=path.parent, check=True)


def main():
    run("process_steam_data.py")
    run("create_steam_plots.py")
    print("\nEverything is finished")


if __name__ == "__main__":
    main()
