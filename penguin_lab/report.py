from html import escape
from pathlib import Path

import pandas as pd

from penguin_lab.pipeline import MEASUREMENTS, read_json


def render_report(run_directory: str | Path) -> str:
    run_directory = Path(run_directory)
    scores = read_json(run_directory / "metrics.json")
    quality = read_json(run_directory / "quality.json")
    split = read_json(run_directory / "split.json")
    provenance = read_json(run_directory / "source.json")
    predictions = pd.read_csv(run_directory / "predictions.csv")
    species_names = scores["species_order"]
    matrix_rows = []
    for species_name, counts in zip(species_names, scores["confusion_matrix"]):
        cells = "".join(f"<td>{count}</td>" for count in counts)
        matrix_rows.append(f"<tr><th scope='row'>{escape(species_name)}</th>{cells}</tr>")
    matrix_headings = "".join(f"<th scope='col'>{escape(name)}</th>" for name in species_names)
    mistakes = predictions.loc[predictions["species"] != predictions["predicted_species"]]
    mistake_columns = ["observation_id", "species", "predicted_species", "confidence"] + MEASUREMENTS
    mistakes_table = mistakes[mistake_columns].to_html(index=False, float_format=lambda value: f"{value:.3f}")
    if mistakes.empty:
        mistakes_table = "<p>No misclassified birds in this holdout.</p>"
    species_rows = []
    for species_name in species_names:
        species_scores = scores["per_species"][species_name]
        species_rows.append(
            f"<tr><th scope='row'>{escape(species_name)}</th><td>{species_scores['precision']:.3f}</td>"
            f"<td>{species_scores['recall']:.3f}</td><td>{species_scores['f1-score']:.3f}</td>"
            f"<td>{int(species_scores['support'])}</td></tr>"
        )

    # Keep this report self-contained so it still opens after the containers are stopped.
    report_html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Penguin field notes — model evaluation</title>
<style>
body {{ background:#f3f5ef; color:#183c39; font:16px/1.6 system-ui,sans-serif; margin:0; }}
main {{ max-width:1100px; margin:auto; padding:36px 24px; }}
h1 {{ font-size:clamp(2rem,5vw,3.6rem); line-height:1.1; margin:10px 0; }}
h2 {{ margin-top:36px; }} .eyebrow {{ text-transform:uppercase; letter-spacing:.16em; font-size:.8rem; }}
.cards {{ display:flex; flex-wrap:wrap; gap:16px; margin:28px 0; }}
.card {{ flex:1; min-width:160px; padding:20px; border:1px solid #a7beb5; background:#fff; border-radius:10px; }}
.card strong {{ display:block; font-size:2rem; }}
.scroll {{ overflow-x:auto; }} table {{ border-collapse:collapse; background:white; width:100%; }}
th,td {{ padding:10px; border:1px solid #c9d6cf; text-align:left; }}
th {{ background:#e0eae4; }} code {{ overflow-wrap:anywhere; }}
footer {{ margin-top:36px; border-top:1px solid #a7beb5; padding-top:16px; font-size:.85rem; }}
</style></head><body><main>
<p class="eyebrow">Airflow lab / Palmer Archipelago</p><h1>Penguin field notes</h1>
<p>Species classification from bill length, bill depth, flipper length, and body mass.</p>
<div class="cards">
<div class="card">Holdout accuracy<strong>{scores['accuracy']:.1%}</strong></div>
<div class="card">Macro F1<strong>{scores['macro_f1']:.3f}</strong></div>
<div class="card">Baseline macro F1<strong>{scores['baseline_macro_f1']:.3f}</strong></div>
<div class="card">Training / holdout<strong>{split['train_rows']} / {split['holdout_rows']}</strong></div>
</div>
<h2>Confusion matrix</h2><p>Rows are observed species; columns are model predictions.</p>
<div class="scroll"><table><thead><tr><th scope="col">Observed / predicted</th>{matrix_headings}</tr></thead>
<tbody>{''.join(matrix_rows)}</tbody></table></div>
<h2>Performance by species</h2><div class="scroll"><table><thead><tr><th>Species</th><th>Precision</th>
<th>Recall</th><th>F1</th><th>Holdout birds</th></tr></thead><tbody>{''.join(species_rows)}</tbody></table></div>
<h2>Birds worth a second look</h2><p>Confidence is the largest class probability, not a calibrated certainty.</p>
<div class="scroll">{mistakes_table}</div>
<h2>Dataset checks</h2><p>{quality['rows']} observations. Species counts: {escape(str(quality['species_counts']))}.</p>
<p>Missing measurements: {escape(str(quality['missing_measurements']))}. Training medians fill missing values.
Island, sex, and year are excluded from the model.</p>
<h2>How to interpret this run</h2><p>Logistic regression and a most-frequent-species baseline share the same
stratified 25% holdout (seed {split['seed']}). Preprocessing is fitted only on training birds.
The separate Airflow quality-gate task requires macro F1 ≥ 0.90 and an improvement ≥ 0.10 over the baseline;
its decision is saved in <code>release.json</code>.</p>
<p>This small historical sample is a teaching dataset. The random holdout does not establish performance on
new colonies, future years, or juvenile birds. No hyperparameters were selected using this holdout.</p>
<footer>Source: {escape(provenance['dataset'])} / {escape(provenance['source_file'])}<br>
CSV SHA-256: <code>{escape(provenance['sha256'])}</code><br>
Penguin observations: Kristen Gorman and Palmer Station LTER. Dataset package: Horst, Hill, and Gorman (CC0).</footer>
</main></body></html>"""
    pending_report = run_directory / "report.html.tmp"
    pending_report.write_text(report_html, encoding="utf-8")
    pending_report.replace(run_directory / "report.html")
    return str(run_directory)
