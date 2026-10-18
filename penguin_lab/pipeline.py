from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MEASUREMENTS = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
SPECIES = ["Adelie", "Chinstrap", "Gentoo"]
SEED = 17
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_json(destination: Path, payload: dict) -> None:
    # A retry should never pick up a half-written manifest from the previous attempt.
    pending = destination.with_suffix(destination.suffix + ".tmp")
    pending.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(destination)


def read_json(source: Path) -> dict:
    return json.loads(source.read_text(encoding="utf-8"))


def run_folder(artifact_root: str | Path, run_label: str) -> Path:
    if not run_label.strip():
        raise ValueError("A non-empty run label is required.")
    # Airflow run IDs contain colons; hashing also prevents a supplied label escaping the root.
    run_key = hashlib.sha256(run_label.encode("utf-8")).hexdigest()[:16]
    destination = Path(artifact_root).resolve() / run_key
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def stage_measurements(source_csv: str | Path, run_directory: str | Path) -> str:
    source_csv = Path(source_csv).resolve()
    run_directory = Path(run_directory)
    run_directory.mkdir(parents=True, exist_ok=True)
    # Reusing a run label must not leave yesterday's success beside today's failed attempt.
    for previous_verdict in ("release.json", "report.html"):
        (run_directory / previous_verdict).unlink(missing_ok=True)
    if not source_csv.is_file():
        raise FileNotFoundError(f"Penguin measurements were not found: {source_csv}")
    fingerprint = hashlib.sha256(source_csv.read_bytes()).hexdigest()
    bundled_csv = PROJECT_ROOT / "dataset" / "penguins.csv"
    if source_csv == bundled_csv.resolve():
        expected = read_json(bundled_csv.with_name("provenance.json"))["sha256"]
        if fingerprint != expected:
            raise ValueError("The bundled CSV checksum changed. Restore the snapshot or use --source.")
    pending_csv = run_directory / "measurements.csv.tmp"
    shutil.copyfile(source_csv, pending_csv)
    pending_csv.replace(run_directory / "measurements.csv")
    write_json(run_directory / "source.json", {
        "source_file": source_csv.name,
        "sha256": fingerprint,
        "dataset": "Palmer Penguins" if source_csv == bundled_csv.resolve() else "Custom penguin CSV",
    })
    return str(run_directory)


def validate_measurements(run_directory: str | Path) -> str:
    run_directory = Path(run_directory)
    colony = pd.read_csv(run_directory / "measurements.csv", na_values=["NA"])
    missing_columns = set(MEASUREMENTS + ["species"]) - set(colony.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    if len(colony) < 60:
        raise ValueError("At least 60 observations are needed for this lab's stratified holdout.")
    if colony["species"].isna().any() or set(colony["species"].unique()) != set(SPECIES):
        raise ValueError(f"Species labels must be non-missing and exactly {SPECIES}.")
    if colony["species"].value_counts().min() < 8:
        raise ValueError("Each species needs at least eight observations.")

    plausible_ranges = {
        "bill_length_mm": (20, 80), "bill_depth_mm": (8, 35),
        "flipper_length_mm": (100, 300), "body_mass_g": (1500, 10000),
    }
    for measurement, (smallest, largest) in plausible_ranges.items():
        colony[measurement] = pd.to_numeric(colony[measurement], errors="raise")
        observed = colony[measurement].dropna()
        if not np.isfinite(observed).all() or not observed.between(smallest, largest).all():
            raise ValueError(f"{measurement} contains non-finite or implausible measurements.")
        if colony[measurement].isna().mean() > 0.10:
            raise ValueError(f"More than 10% of {measurement} is missing.")
    if colony.duplicated().any():
        raise ValueError("Duplicate observations could leak across the holdout split.")
    # Distinct metadata does not make identical model inputs independent observations.
    if colony.duplicated(subset=MEASUREMENTS + ["species"]).any():
        raise ValueError("Repeated measurement/label pairs could leak across the holdout split.")

    colony.index.name = "observation_id"
    colony.to_csv(run_directory / "validated.csv", index=True)
    write_json(run_directory / "quality.json", {
        "rows": len(colony),
        "species_counts": {name: int(count) for name, count in colony["species"].value_counts().items()},
        "missing_measurements": {name: int(count) for name, count in colony[MEASUREMENTS].isna().sum().items()},
        "excluded_columns": sorted(set(colony.columns) - set(MEASUREMENTS + ["species"])),
    })
    return str(run_directory)


def split_colony(run_directory: str | Path) -> str:
    run_directory = Path(run_directory)
    colony = pd.read_csv(run_directory / "validated.csv")
    training_birds, holdout_birds = train_test_split(
        colony, test_size=0.25, random_state=SEED, stratify=colony["species"]
    )
    training_birds.to_csv(run_directory / "train.csv", index=False)
    holdout_birds.to_csv(run_directory / "holdout.csv", index=False)
    write_json(run_directory / "split.json", {
        "seed": SEED, "holdout_fraction": 0.25,
        "train_rows": len(training_birds), "holdout_rows": len(holdout_birds),
    })
    return str(run_directory)


def fit_classifier(run_directory: str | Path) -> str:
    run_directory = Path(run_directory)
    training_birds = pd.read_csv(run_directory / "train.csv")
    # Fit imputation and scaling only here: the holdout must not influence either step.
    species_classifier = Pipeline([
        ("fill_measurements", SimpleImputer(strategy="median")),
        ("scale_measurements", StandardScaler()),
        ("classify_species", LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)),
    ])
    species_classifier.fit(training_birds[MEASUREMENTS], training_birds["species"])
    pending_model = run_directory / "species_model.joblib.tmp"
    joblib.dump(species_classifier, pending_model)
    pending_model.replace(run_directory / "species_model.joblib")
    write_json(run_directory / "model_card.json", {
        "estimator": "LogisticRegression", "features": MEASUREMENTS,
        "imputation": "training-set median", "scaling": "training-set StandardScaler",
        "C": 1.0, "max_iter": 1000, "seed": SEED,
        "versions": {"scikit_learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
        "intended_use": "Educational species classification from four body measurements.",
        "limitations": "Small historical sample; random holdout does not establish performance on new colonies or years.",
    })
    return str(run_directory)


def evaluate_classifier(run_directory: str | Path) -> str:
    run_directory = Path(run_directory)
    training_birds = pd.read_csv(run_directory / "train.csv")
    holdout_birds = pd.read_csv(run_directory / "holdout.csv")
    species_classifier = joblib.load(run_directory / "species_model.joblib")
    predicted_species = species_classifier.predict(holdout_birds[MEASUREMENTS])
    probabilities = species_classifier.predict_proba(holdout_birds[MEASUREMENTS])

    majority_baseline = DummyClassifier(strategy="most_frequent")
    majority_baseline.fit(training_birds[MEASUREMENTS], training_birds["species"])
    baseline_predictions = majority_baseline.predict(holdout_birds[MEASUREMENTS])
    scores = {
        "accuracy": float(accuracy_score(holdout_birds["species"], predicted_species)),
        "macro_f1": float(f1_score(holdout_birds["species"], predicted_species, average="macro")),
        "baseline_macro_f1": float(f1_score(holdout_birds["species"], baseline_predictions, average="macro")),
        "species_order": SPECIES,
        "confusion_matrix": confusion_matrix(holdout_birds["species"], predicted_species, labels=SPECIES).tolist(),
        "per_species": classification_report(
            holdout_birds["species"], predicted_species, labels=SPECIES, output_dict=True, zero_division=0
        ),
    }
    write_json(run_directory / "metrics.json", scores)
    holdout_birds["predicted_species"] = predicted_species
    holdout_birds["confidence"] = probabilities.max(axis=1)
    holdout_birds.to_csv(run_directory / "predictions.csv", index=False)
    return str(run_directory)


def check_release(run_directory: str | Path, minimum_f1: float = 0.90, minimum_lift: float = 0.10) -> str:
    for threshold in (minimum_f1, minimum_lift):
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Gate thresholds must be finite numbers between zero and one.")
    run_directory = Path(run_directory)
    scores = read_json(run_directory / "metrics.json")
    lift = scores["macro_f1"] - scores["baseline_macro_f1"]
    passed = scores["macro_f1"] >= minimum_f1 and lift >= minimum_lift
    write_json(run_directory / "release.json", {
        "passed": passed, "minimum_macro_f1": minimum_f1, "minimum_lift": minimum_lift,
        "observed_macro_f1": scores["macro_f1"], "observed_lift": lift,
    })
    if not passed:
        raise ValueError(f"Quality gate failed: macro F1={scores['macro_f1']:.3f}, baseline lift={lift:.3f}.")
    return str(run_directory)
