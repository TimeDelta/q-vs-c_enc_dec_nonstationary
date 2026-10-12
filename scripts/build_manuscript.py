"""Populate manuscript results from checked-in evidence; render with Pandoc."""
import csv
import json
from pathlib import Path
import re
import subprocess

import numpy as np


def scientific(value, decimal_places=6):
    mantissa, exponent = f'{value:.{decimal_places}e}'.split('e')
    return '$'+mantissa+r'\times10^{'+str(int(exponent))+'}$'


def build_markdown(repository):
    repository = Path(repository)
    paper = repository/'paper'
    evidence = repository/'docs/descriptor_generalization'
    summary = json.loads((evidence/'summary.json').read_text())
    verification = json.loads((evidence/'verification.json').read_text())
    primary = summary['endpoints']['shifted_probe']
    baseline = primary['mean_loss']['baseline']
    distance = primary['mean_loss']['distance']
    percent_increase = 100*(distance/baseline-1)
    controls = summary['transformation_controls']
    replacements = {
        'DESCRIPTOR_ABSTRACT':f'An exploratory nested analysis finds that adding validation descriptor mismatch to model identity and validation forecast error raises held-out squared prediction loss by {percent_increase:.1f}% and improves only {primary["distance_improved_realizations"]} of eight realization folds.',
        'DESCRIPTOR_RESULTS':f'The primary baseline has mean held-out squared prediction loss {scientific(baseline)}; adding scalar validation mismatch gives {scientific(distance)}. The extension raises loss by {percent_increase:.1f}% and improves {primary["distance_improved_realizations"]}/8 realization folds. The mean baseline-minus-extension difference is {scientific(primary["mean_baseline_minus_distance"])}, so its direction favors the baseline. The squared component extension has loss {scientific(primary["mean_loss"]["components"])}, essentially unchanged from baseline, while latent descriptors have loss {scientific(primary["mean_loss"]["latent"])}. These results provide no positive primary evidence for incremental shifted-error prediction from the chosen mismatch.',
        'CONTROL_RESULTS':f'Across existing forecast-task sequence rows, compensated rotations change descriptor distance by a mean of {controls["rotation_descriptor_change"]["mean"]:.6g} and a maximum of {controls["rotation_descriptor_change"]["maximum"]:.6g}. Predictions differ by at most {scientific(controls["rotation_prediction_max_difference"]["maximum"],2)}. Positive scaling has maximum descriptor change {scientific(controls["scale_descriptor_change"]["maximum"],2)}; time shuffling has mean descriptor change {controls["shuffled_descriptor_change"]["mean"]:.6g}. The tiny scale and prediction discrepancies reflect numerical precision.',
        'RESTORATION_RESULTS':f'The descriptor follow-up restores all {verification["validation_features_rows"]} saved cases and reproduces locked validation-probe MSE with maximum absolute discrepancy {scientific(verification["max_restored_validation_probe_error"],2)}. Original selection files and validation-feature hashes remain unchanged. All {verification["prediction_rows"]:,} held-out predictions are finite. Regression tests verify checkpoint restoration for every model family and the independence of outer-fold fitting from its held-out targets.'
    }
    with (repository/'docs/fresh_seed_forecasting/sequence_metrics.csv').open() as stream:
        metrics = [row for row in csv.DictReader(stream) if row['task']=='forecast']
    table = ['| Model | Native ID | Native shifted | Probe shifted |', '| --- | ---: | ---: | ---: |']
    named_models = [('qte','Quantum'),('qte_noent','Quantum, no CZ'),('cte','Classical ring'),
                    ('mlp_te','MLP'),('gru_te','GRU'),('reduced_rank','Reduced-rank'),
                    ('persistence','Persistence'),('untrained_qte','Untrained quantum'),('pca','PCA')]
    for name,label in named_models:
        rows = [row for row in metrics if row['model']==name]
        shifted = [row for row in rows if row['partition']!='test_id']
        id_rows = [row for row in rows if row['partition']=='test_id']
        native_id = f'{np.mean([float(row["native_mse"]) for row in id_rows]):.6f}' if name!='pca' else '-'
        native_shifted = f'{np.mean([float(row["native_mse"]) for row in shifted]):.6f}' if name!='pca' else '-'
        probe_shifted = np.mean([float(row['probe_mse']) for row in shifted])
        table.append(f'| {label} | {native_id} | {native_shifted} | {probe_shifted:.6f} |')
    replacements['FORECAST_TABLE'] = '\n'.join(table)
    table = ['| Predictor | Shifted probe | ID probe | Shifted native |','| --- | ---: | ---: | ---: |']
    for key,label in [('identity','Identity only'),('validation','Validation only'),('baseline','Identity + validation'),
                      ('distance','+ distance (primary)'),('components','+ components'),('latent','+ latent descriptors')]:
        values = [summary['endpoints'][endpoint]['mean_loss'][key] for endpoint in ('shifted_probe','id_probe','shifted_native')]
        table.append('| '+label+' | '+' | '.join(f'{value*1e5:.6f}' for value in values)+' |')
    replacements['DESCRIPTOR_TABLE'] = '\n'.join(table)
    manuscript = (paper/'manuscript_template.md').read_text()
    for key,value in replacements.items():
        manuscript = manuscript.replace('{{'+key+'}}',value)
    if re.search(r'\{\{[A-Z_]+\}\}',manuscript):
        raise ValueError('Unpopulated manuscript token')
    (paper/'manuscript.md').write_text(manuscript)
    return paper


if __name__=='__main__':
    repository = Path(__file__).resolve().parents[1]
    paper = build_markdown(repository)
    # Markdown, citations, figures and equations are also available as LaTeX source.
    common = ['pandoc','manuscript.md','--standalone','--citeproc','--bibliography=references.bib',
              '--number-sections','--resource-path=.:..','-V','colorlinks=true','-V','urlcolor=teal']
    subprocess.run(common+['-o','manuscript.tex'],cwd=paper,check=True)
    subprocess.run(common+['--pdf-engine=pdflatex','-o','manuscript.pdf'],cwd=paper,check=True)
