"""Run inside the Airflow image: python tests/check_dag.py."""
from pathlib import Path
import unittest

from airflow.models import DagBag


class PenguinDagTests(unittest.TestCase):
    def test_real_airflow_parsing_and_dependencies(self):
        project_root = Path(__file__).resolve().parents[1]
        dag_bag = DagBag(dag_folder=str(project_root / "dags"), include_examples=False)
        self.assertEqual(dag_bag.import_errors, {})
        penguin_dag = dag_bag.get_dag("penguin_species_lab")
        self.assertIsNotNone(penguin_dag)
        ordered_tasks = [
            "stage_snapshot", "validate_snapshot", "reserve_holdout", "train_species_model",
            "score_holdout", "publish_evaluation", "enforce_quality_gate",
        ]
        self.assertEqual(set(penguin_dag.task_ids), set(ordered_tasks))
        for upstream, downstream in zip(ordered_tasks, ordered_tasks[1:]):
            self.assertEqual(penguin_dag.get_task(upstream).downstream_task_ids, {downstream})
        self.assertEqual(penguin_dag.get_task(ordered_tasks[-1]).downstream_task_ids, set())
        self.assertFalse(penguin_dag.catchup)
        self.assertEqual(penguin_dag.max_active_runs, 1)
        self.assertEqual(penguin_dag.get_task("enforce_quality_gate").retries, 0)


if __name__ == "__main__":
    unittest.main()
