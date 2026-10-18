import hashlib
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

from penguin_lab.pipeline import (
    MEASUREMENTS, PROJECT_ROOT, check_release, evaluate_classifier, fit_classifier,
    read_json, run_folder, split_colony, stage_measurements, validate_measurements, write_json,
)
from penguin_lab.report import render_report


@pytest.fixture
def staged_colony(tmp_path):
    stage_measurements(PROJECT_ROOT / "dataset" / "penguins.csv", tmp_path)
    return tmp_path


def train_and_score(run_directory):
    for stage in (validate_measurements, split_colony, fit_classifier, evaluate_classifier, render_report):
        stage(run_directory)


def test_full_workflow_and_saved_model(staged_colony):
    train_and_score(staged_colony)
    check_release(staged_colony)
    scores = read_json(staged_colony / "metrics.json")
    assert scores["macro_f1"] >= 0.90
    assert read_json(staged_colony / "release.json")["passed"] is True
    assert sum(map(sum, scores["confusion_matrix"])) == 86
    holdout_birds = pd.read_csv(staged_colony / "holdout.csv")
    predictions = pd.read_csv(staged_colony / "predictions.csv")
    restored_classifier = joblib.load(staged_colony / "species_model.joblib")
    assert list(restored_classifier.predict(holdout_birds[MEASUREMENTS])) == predictions["predicted_species"].tolist()
    assert "Penguin field notes" in (staged_colony / "report.html").read_text(encoding="utf-8")


def test_preprocessing_uses_training_birds_only(staged_colony):
    validate_measurements(staged_colony)
    split_colony(staged_colony)
    fit_classifier(staged_colony)
    training_birds = pd.read_csv(staged_colony / "train.csv")
    holdout_birds = pd.read_csv(staged_colony / "holdout.csv")
    assert set(training_birds["observation_id"]).isdisjoint(holdout_birds["observation_id"])
    assert len(training_birds) + len(holdout_birds) == 344
    restored_classifier = joblib.load(staged_colony / "species_model.joblib")
    np.testing.assert_allclose(
        restored_classifier.named_steps["fill_measurements"].statistics_,
        training_birds[MEASUREMENTS].median().to_numpy(),
    )
    assert set(training_birds["species"]) == set(holdout_birds["species"])


def test_gate_failure_keeps_report_for_inspection(staged_colony):
    train_and_score(staged_colony)
    # Deliberately make the score bad: the gate is meant to fail even if the report succeeded.
    scores = read_json(staged_colony / "metrics.json")
    scores["macro_f1"] = 0.40
    write_json(staged_colony / "metrics.json", scores)
    with pytest.raises(ValueError, match="Quality gate failed"):
        check_release(staged_colony)
    assert read_json(staged_colony / "release.json")["passed"] is False
    assert (staged_colony / "report.html").is_file()


@pytest.mark.parametrize("broken_case", ["column", "label", "infinity", "negative", "missing", "duplicate", "few_birds", "text"])
def test_bad_snapshots_fail_before_training(staged_colony, broken_case):
    colony = pd.read_csv(staged_colony / "measurements.csv")
    if broken_case == "column":
        colony = colony.drop(columns=["body_mass_g"])
    elif broken_case == "label":
        colony.loc[0, "species"] = "Unknown"
    elif broken_case == "infinity":
        colony.loc[0, "bill_length_mm"] = float("inf")
    elif broken_case == "negative":
        colony.loc[0, "body_mass_g"] = -1
    elif broken_case == "missing":
        colony.loc[:50, "bill_depth_mm"] = np.nan
    elif broken_case == "duplicate":
        colony = pd.concat([colony, colony.iloc[[0]]], ignore_index=True)
    elif broken_case == "few_birds":
        colony = colony.iloc[:30]
    elif broken_case == "text":
        colony["bill_length_mm"] = colony["bill_length_mm"].astype(object)
        colony.loc[0, "bill_length_mm"] = "forty"
    colony.to_csv(staged_colony / "measurements.csv", index=False)
    with pytest.raises(ValueError):
        validate_measurements(staged_colony)
    assert not (staged_colony / "species_model.joblib").exists()


def test_rerun_is_reproducible(staged_colony):
    train_and_score(staged_colony)
    first_scores = read_json(staged_colony / "metrics.json")
    first_predictions = (staged_colony / "predictions.csv").read_bytes()
    train_and_score(staged_colony)
    assert first_scores == read_json(staged_colony / "metrics.json")
    assert first_predictions == (staged_colony / "predictions.csv").read_bytes()


def test_run_directories_cannot_escape_artifact_root(tmp_path):
    suspicious_run = run_folder(tmp_path, "../../outside:run")
    assert suspicious_run.parent == tmp_path.resolve()
    assert suspicious_run == run_folder(tmp_path, "../../outside:run")
    assert suspicious_run != run_folder(tmp_path, "another-run")
    with pytest.raises(ValueError):
        run_folder(tmp_path, " ")


def test_snapshot_matches_provenance():
    csv_snapshot = PROJECT_ROOT / "dataset" / "penguins.csv"
    assert hashlib.sha256(csv_snapshot.read_bytes()).hexdigest() == read_json(csv_snapshot.with_name("provenance.json"))["sha256"]


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_gate_configuration_is_rejected(staged_colony, threshold):
    with pytest.raises(ValueError, match="Gate thresholds"):
        check_release(staged_colony, minimum_f1=threshold)


def test_gate_also_requires_lift_over_baseline(staged_colony):
    write_json(staged_colony / "metrics.json", {"macro_f1": 0.95, "baseline_macro_f1": 0.94})
    with pytest.raises(ValueError, match="Quality gate failed"):
        check_release(staged_colony)
    assert read_json(staged_colony / "release.json")["passed"] is False


def test_new_attempt_removes_stale_success(staged_colony):
    write_json(staged_colony / "release.json", {"passed": True})
    (staged_colony / "report.html").write_text("Old report", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        stage_measurements(staged_colony / "not-here.csv", staged_colony)
    assert not (staged_colony / "release.json").exists()
    assert not (staged_colony / "report.html").exists()


def test_cli_reports_missing_source_without_traceback(tmp_path):
    invocation = subprocess.run(
        [sys.executable, "-m", "penguin_lab.cli", "--source", str(tmp_path / "missing.csv"),
         "--artifact-root", str(tmp_path / "runs")],
        cwd=PROJECT_ROOT, capture_output=True, text=True, check=False,
    )
    assert invocation.returncode == 1
    assert "Penguin workflow failed" in invocation.stderr
    assert "Traceback" not in invocation.stderr
