# Validation record

Verified on Windows with Python 3.12.14 using the versions in `requirements.txt`.

| Check | Outcome |
| --- | --- |
| `python -m pytest -q` | 21 passed |
| `python -m penguin_lab.cli --run-label submission-example` | Completed; quality gate passed |
| `python -m compileall -q penguin_lab dags tests` | Passed |
| `docker compose config --quiet` | Passed |
| Real Airflow DagBag integration check | Not executed; Docker engine unavailable |
| Full Airflow DAG and scheduler run | Not executed; Docker engine unavailable |
| Browser visual inspection of HTML | Not completed; browser preview stalled |

The CLI and Airflow tasks call the same pipeline functions. CLI success verifies
the ML steps; it does not verify Airflow imports, scheduler execution, container
permissions, or image dependency resolution. The repository includes a real
DagBag test and the full DAG test commands in README.md for those remaining checks.

## Supplied example run

The files in `artifacts/example/` came from the completed CLI run, not fabricated
metrics. The snapshot has 344 birds; 258 were used for training and 86 for holdout.

| Metric | Value |
| --- | --- |
| Holdout accuracy | 1.000 |
| Holdout macro F1 | 1.000 |
| Majority-baseline macro F1 | 0.204301 |
| Lift over baseline | 0.795699 |
| Quality gate | Passed |

The holdout has 38 Adelie, 17 Chinstrap, and 31 Gentoo birds. All 86 were correctly
classified for this fixed split. This is a small historical teaching sample;
perfect performance here does not imply perfect generalization.

The tests separately exercise failed quality gates, bad snapshots, reproducible
reruns, model reloads, stale-success invalidation, invalid CLI input, safe run
folders, provenance integrity, and training-only imputation.
