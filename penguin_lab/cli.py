import argparse
from pathlib import Path
import sys

from penguin_lab.pipeline import (
    PROJECT_ROOT, check_release, evaluate_classifier, fit_classifier, run_folder,
    split_colony, stage_measurements, validate_measurements,
)
from penguin_lab.report import render_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the penguin workflow without an Airflow scheduler.")
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "dataset" / "penguins.csv")
    parser.add_argument("--artifact-root", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument("--run-label", default="local-demo")
    parser.add_argument("--minimum-f1", type=float, default=0.90)
    options = parser.parse_args()
    try:
        run_directory = run_folder(options.artifact_root, options.run_label)
        stage_measurements(options.source, run_directory)
        for stage in (validate_measurements, split_colony, fit_classifier, evaluate_classifier, render_report):
            print(f"Running {stage.__name__}", flush=True)
            stage(run_directory)
        check_release(run_directory, minimum_f1=options.minimum_f1)
    except (OSError, ValueError) as exc:
        print(f"Penguin workflow failed: {exc}", file=sys.stderr)
        return 1
    print(f"Quality gate passed. Report: {run_directory / 'report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
