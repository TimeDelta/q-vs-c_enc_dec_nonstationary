"""Render the committed exploratory descriptor analysis without feature search."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def make_report(directory):
    directory = Path(directory)
    summary = json.loads((directory/'summary.json').read_text())
    verification = json.loads((directory/'verification.json').read_text())
    manifest = json.loads((directory/'manifest.json').read_text())
    primary = summary['endpoints']['shifted_probe']
    baseline_loss = primary['mean_loss']['baseline']
    distance_loss = primary['mean_loss']['distance']
    change = 100*(distance_loss/baseline_loss-1)
    with (directory/'fold_losses.csv').open() as stream:
        folds = [row for row in csv.DictReader(stream) if row['endpoint']=='shifted_probe']
    figure, axes = plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    differences = np.array([float(row['baseline_minus_distance_loss']) for row in folds])
    axes[0].bar([row['data_seed'] for row in folds],differences*1e6,
                color=['#237a78' if value>0 else '#bd5b44' for value in differences])
    axes[0].axhline(0,color='#555555',linewidth=.8)
    axes[0].set(xlabel='Held-out data seed',ylabel='Baseline minus distance prediction loss (x 10^-6)',
                title='Extra validation mismatch predictor')
    keys = ['identity','validation','baseline','distance','components','latent']
    labels = ['Model identity','Validation MSE','Identity + MSE','+ distance','+ components','+ latent descriptors']
    values = [primary['mean_loss'][key]*1e6 for key in keys]
    axes[1].barh(labels,values,color=['#999999','#999999','#4e6476','#237a78','#809a93','#809a93'])
    axes[1].invert_yaxis()
    axes[1].set(xlabel='Mean held-out squared prediction error (x 10^-6)',title='Same eight outer folds')
    for axis in axes:
        axis.spines[['top','right']].set_visible(False)
    figure.savefig(directory/'descriptor_generalization.png',dpi=180)
    plt.close(figure)
    lines = ['# Do validation descriptors predict shifted forecast error?', '',
             '**Exploratory follow-up after test-result inspection.** These are predictions across eight held-out generator realizations, with all named models kept together within a realization. The target is forecast MSE, so the regression loss below has units of MSE squared.', '',
             f'The primary distance extension has mean held-out prediction loss {distance_loss:.8g}, versus {baseline_loss:.8g} for model identity plus validation forecast MSE. This changes prediction loss by {change:+.1f}% and improves {primary["distance_improved_realizations"]}/8 realization folds. Positive paired differences favor the extra predictor. This result does not establish a causal or necessary role for complexity preservation.', '',
             '![Nested validation and every primary fold](descriptor_generalization.png)', '',
             '| Held-out data seed | Baseline loss | + distance loss | Baseline minus distance |',
             '| --- | ---: | ---: | ---: |']
    for row in folds:
        lines.append(f'| {row["data_seed"]} | {float(row["baseline_loss"]):.8g} | {float(row["distance_loss"]):.8g} | {float(row["baseline_minus_distance_loss"]):+.8g} |')
    lines += ['', '## All committed feature sets and endpoints', '',
              '| Predictor set | Shifted common probe | ID common probe | Shifted native forecast |',
              '| --- | ---: | ---: | ---: |']
    for key,label in zip(keys,labels):
        values = [summary['endpoints'][endpoint]['mean_loss'][key] for endpoint in ('shifted_probe','id_probe','shifted_native')]
        lines.append('| '+label+' | '+' | '.join(f'{value:.8g}' for value in values)+' |')
    lines += ['', 'Each loss averages squared prediction errors over models within a realization, then equally over eight realizations. The common-probe endpoints include 14 cases per realization; the native endpoint includes 12 forecasting cases and excludes reconstruction-native PCA and random projection. Identity-only and validation-only are reference analyses. Scalar distance is the primary extension; component mismatches and latent descriptors are secondary. No best-performing extension is substituted for the primary result.', '',
              '## Interpretation and limits', '',
              'The baseline includes model identity, absorbing the fixed architecture and training-budget differences among the named procedures. The primary extension tests an additional scalar validation mismatch. All standardization, intercept fitting and ridge-penalty selection occur within the appropriate training folds. The target of every outer fold is a new data realization of the same generator with the same model set. Neither unseen architectures nor unseen generator families are evaluated.', '',
              'Descriptors come from the two saved validation sequences, with 127 input time points each. They are not extracted from test sequences for the regression. They summarize order-three normalized permutation entropy averaged over usable scales 1 and 2, median-binarized normalized LZ76 and lag-one correlation. Each is averaged across coordinates. The scalar mismatch averages squared component differences, rescaling lag correlation by one half. It is not an MDL score or an information-preservation criterion.', '',
              'The same eight benchmark realizations have already informed the scientific interpretation. Nested validation prevents regression fitting on each held-out outcome but does not undo prior inspection or make this follow-up confirmatory. There are eight analysis units, despite 112 common-probe model/data rows. No significance, confidence-interval or general quantum-advantage claim is supported.', '',
              '## Coordinate controls', '',
              '| Control quantity | Mean | Maximum |', '| --- | ---: | ---: |']
    for key,value in summary['transformation_controls'].items():
        lines.append(f'| {key} | {value["mean"]:.8g} | {value["maximum"]:.8g} |')
    lines += ['', 'The controls use the existing forecast-task sequence rows. Compensated orthogonal rotations preserve the linear-probe predictions while changing coordinatewise temporal summaries. This demonstrates that descriptor distance is not an invariant measure of predictive information. Quantum summaries apply only to the retained single-qubit Z readouts.', '',
              '## Reproduction', '',
              'From a clean committed checkout with the pinned CPU dependencies:', '', '```bash',
              'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pilot.descriptor_study --output /tmp/descriptor_generalization',
              'python scripts/report_descriptor_generalization.py /tmp/descriptor_generalization', '```', '',
              'Check saved fold coverage, loss aggregation and evidence hashes independently:', '', '```bash',
              'python scripts/verify_descriptor_generalization.py /tmp/descriptor_generalization', '```', '',
              f'Analysis source commit: `{manifest["git_commit"]}`. Original archive SHA-256: `{manifest["evidence_archive_sha256"]}`. Validation features SHA-256: `{manifest["feature_sha256"]}`.', '',
              f'All {verification["validation_features_rows"]} restored validation-probe errors match saved metadata; maximum absolute difference {verification["max_restored_validation_probe_error"]:.3g}. Original selection-lock files and the validation-feature table remain unchanged. All {verification["prediction_rows"]} held-out predictions are finite.', '',
              'Files include `validation_features.csv`, `feature_lock.json`, `analysis_cells.csv`, `heldout_predictions.csv`, `fold_losses.csv`, `fold_audit.json`, `summary.json`, `verification.json` and `manifest.json`. The analysis configuration is `analysis_config.json`; the protocol is [the exploratory analysis plan](../descriptor_generalization_protocol.md). The original checkpoint archive remains in [the fresh-seed benchmark](../fresh_seed_forecasting/README.md).']
    (directory/'README.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    make_report(parser.parse_args().directory)
