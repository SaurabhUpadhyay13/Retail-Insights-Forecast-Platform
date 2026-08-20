from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT_DIR / "scripts"


def run_script(script_name: str, *script_args: str) -> None:
    script_path = SCRIPTS_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")

    print(f"\n=== Running {script_name} ===")
    try:
        subprocess.run(
            [sys.executable, str(script_path), *script_args],
            check=True,
            cwd=ROOT_DIR,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"{script_name} failed with exit code {exc.returncode}"
        ) from exc


def main() -> None:
    steps = [
        ("01_data_cleaning.py", ("--csv-only",), "Clean and standardize raw files"),
        ("02_streaming_pipeline.py", (), "Build forecasts and anomaly flags"),
        ("03_export_for_powerbi.py", (), "Build Power BI-ready tables"),
    ]

    print("Starting retail forecasting pipeline...")
    print(f"Project root: {ROOT_DIR}")

    for script_name, script_args, description in steps:
        print(f"-> {description}")
        run_script(script_name, *script_args)

    print("\nPipeline complete: clean -> forecast/flags -> Power BI export.")
    print("Outputs saved in the data/ and outputs/ folders.")


if __name__ == "__main__":
    main()
