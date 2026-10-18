from datetime import timedelta
import os

import pendulum
from airflow.decorators import dag, task
from airflow.operators.python import get_current_context


@dag(
    dag_id="penguin_species_lab",
    description="Validate field measurements, fit a species classifier, and review its holdout report.",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    schedule="0 9 * * 1",
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "field-notes", "retries": 1, "retry_delay": timedelta(seconds=30)},
    tags=["lab", "penguins", "classification"],
)
def penguin_species_lab():
    @task
    def stage_snapshot() -> str:
        from penguin_lab.pipeline import PROJECT_ROOT, run_folder, stage_measurements

        run_id = get_current_context()["run_id"]
        artifact_root = os.environ.get("PENGUIN_ARTIFACT_ROOT", "/opt/airflow/artifacts")
        run_directory = run_folder(artifact_root, run_id)
        return stage_measurements(PROJECT_ROOT / "dataset" / "penguins.csv", run_directory)

    @task
    def validate_snapshot(run_directory: str) -> str:
        from penguin_lab.pipeline import validate_measurements
        return validate_measurements(run_directory)

    @task
    def reserve_holdout(run_directory: str) -> str:
        from penguin_lab.pipeline import split_colony
        return split_colony(run_directory)

    @task
    def train_species_model(run_directory: str) -> str:
        from penguin_lab.pipeline import fit_classifier
        return fit_classifier(run_directory)

    @task
    def score_holdout(run_directory: str) -> str:
        from penguin_lab.pipeline import evaluate_classifier
        return evaluate_classifier(run_directory)

    @task
    def publish_evaluation(run_directory: str) -> str:
        from penguin_lab.report import render_report
        return render_report(run_directory)

    @task(retries=0)
    def enforce_quality_gate(run_directory: str) -> str:
        from penguin_lab.pipeline import check_release
        return check_release(run_directory)

    # XCom carries only a directory path. The shared volume holds tables and model files.
    snapshot = stage_snapshot()
    checked_snapshot = validate_snapshot(snapshot)
    split_snapshot = reserve_holdout(checked_snapshot)
    fitted_snapshot = train_species_model(split_snapshot)
    scored_snapshot = score_holdout(fitted_snapshot)
    report_snapshot = publish_evaluation(scored_snapshot)
    enforce_quality_gate(report_snapshot)


penguin_species_lab()
