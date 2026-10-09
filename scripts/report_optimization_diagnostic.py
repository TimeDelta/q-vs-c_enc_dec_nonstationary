"""Create descriptive figures and Markdown from an existing pilot run.

Requires matplotlib (included in requirements-test.txt). Two data seeds are
insufficient for calibrated confidence intervals or model superiority claims.
"""
import argparse
import csv
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def build_report(run_directory, report_directory):
    run_directory, report_directory = Path(run_directory), Path(report_directory)
    report_directory.mkdir(parents=True, exist_ok=True)
    fits = json.loads((run_directory / 'fit_diagnostics.json').read_text())
    config = json.loads((run_directory / 'config.json').read_text())
    manifest = json.loads((run_directory / 'manifest.json').read_text())
    with (run_directory / 'sequence_metrics.csv').open() as source:
        rows = list(csv.DictReader(source))
    for source in ('config.json', 'manifest.json', 'fit_diagnostics.json', 'sequence_metrics.csv', 'summary.json'):
        shutil.copyfile(run_directory / source, report_directory / source)
    for data_seed in config['data_seeds']:
        shutil.copyfile(run_directory / f'data_{data_seed}/dataset_manifest.json',
                        report_directory / f'dataset_manifest_{data_seed}.json')
    curves = {}
    table_rows = []
    for model in config['models']:
        model_fits = [record for record in fits if record['model'] == model]
        candidates = [min(record['candidates'], key=lambda item: item['best_validation_mse'])
                      for record in model_fits]
        if candidates[0]['history']:
            histories = [candidate['history'] for candidate in candidates]
            curves[model] = {field: np.array([[epoch[field] for epoch in history] for history in histories])
                             for field in ('train_mse', 'validation_mse')}
            selected_epochs = [min(history, key=lambda epoch: epoch['validation_mse']) for history in histories]
            table_rows.append((model,
                               np.mean([candidate['initial_training_mse'] for candidate in candidates]),
                               np.mean([epoch['train_mse'] for epoch in selected_epochs]),
                               np.mean([record['selected_validation_mse'] for record in model_fits]),
                               ', '.join(str(epoch['epoch']) for epoch in selected_epochs),
                               sum(record['fit_seconds'] for record in model_fits)))
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for model, measurements in curves.items():
        quantum = model.startswith('q')
        forecast = model.endswith('te') or model.endswith('te_noent')
        axis = axes[int(quantum), int(forecast)]
        timesteps = np.arange(1, config['epochs'] + 1)
        line, = axis.plot(timesteps, measurements['validation_mse'].mean(axis=0), label=model)
        axis.plot(timesteps, measurements['train_mse'].mean(axis=0), linestyle='--', color=line.get_color(), alpha=.7)
    for row, family in enumerate(('Classical', 'Quantum')):
        for column, objective in enumerate(('reconstruction', 'forecast')):
            axis = axes[row, column]
            axis.set_title(f'{family}: native {objective}')
            axis.set_yscale('log')
            axis.set_ylabel('Observation MSE (log scale)')
            axis.set_xlabel('Epoch')
            axis.grid(alpha=.2)
            axis.legend(fontsize=8)
    figure.suptitle(f'Optimization diagnostic: solid validation, dashed training; means over {len(config["data_seeds"])} data seeds')
    figure.tight_layout()
    figure.savefig(report_directory / 'learning_curves.png', dpi=180)
    plt.close(figure)

    def per_seed_scores(model, task, partition):
        return np.array([np.mean([float(row['probe_mse']) for row in rows
                                 if row['model'] == model and row['task'] == task
                                 and row['partition'] == partition and int(row['data_seed']) == seed])
                         for seed in config['data_seeds']])

    figure, axes = plt.subplots(1, 2, figsize=(12, 7), sharey=True)
    positions = np.arange(len(config['models']))
    for axis, task in zip(axes, ('reconstruction', 'forecast')):
        for offset, partition, label, color in [(-.13, 'test_id', 'In-distribution', '#24729b'),
                                                (.13, 'test_combined_shift', 'Combined shift', '#b64a38')]:
            for position, model in zip(positions, config['models']):
                values = per_seed_scores(model, task, partition)
                axis.plot([values.min(), values.max()], [position+offset]*2, color=color, alpha=.5)
                axis.scatter(values.mean(), position+offset, color=color, s=25,
                             label=label if position == 0 else None)
        axis.set_xscale('log')
        axis.set_title(f'Frozen common probe: {task}')
        axis.set_xlabel('Test MSE (log scale)')
        axis.set_yticks(positions, config['models'])
        axis.grid(axis='x', alpha=.2)
        axis.legend(fontsize=9)
    axes[0].invert_yaxis()
    figure.suptitle('Markers: mean over data seeds; lines: observed seed range, not confidence intervals')
    figure.tight_layout()
    figure.savefig(report_directory / 'common_probe_scores.png', dpi=180)
    plt.close(figure)

    lines = ['# Optimization diagnostic', '',
             '**Exploratory training diagnostic. No significance, convergence or quantum-advantage claims.**', '',
             f"Source commit: `{manifest['git_commit']}`. Clean source at run start: {not manifest['working_tree_dirty']}.", '',
             f"Run: {config['epochs']} epochs, {len(config['data_seeds'])} independent data seeds, "
             f"one initialization seed and one learning rate ({config['learning_rates'][0]}). "
             f"Each seed has {config['train_sequences']} training, {config['validation_sequences']} validation "
             f"and {config['test_sequences']} test sequences per regime, each of length {config['sequence_length']}. "
             f"Elapsed time: {manifest['elapsed_seconds']:.1f} seconds. {len(rows)} sequence-task records.", '',
             '## Optimization', '',
             'Native-task MSE is distinct from common probe MSE. Selected training MSE comes from the minimum-validation epoch. '
             'Each listed epoch corresponds to one data seed. A best checkpoint at the last epoch suggests extending the budget; '
             'it does not establish convergence. Fit seconds include all data seeds.', '',
             '| Model | Initial train MSE | Selected train MSE | Selected validation MSE | Selected epochs | Fit seconds |',
             '| --- | ---: | ---: | ---: | --- | ---: |']
    for model, initial, selected, validation, epochs, seconds in table_rows:
        lines.append(f'| {model} | {initial:.6f} | {selected:.6f} | {validation:.6f} | {epochs} | {seconds:.1f} |')
    quantum_fits = [record for record in fits if record['model'].startswith('q')]
    last_epoch_count = sum(min(record['candidates'][0]['history'], key=lambda epoch: epoch['validation_mse'])['epoch'] == config['epochs'] for record in quantum_fits)
    lines += ['', f'Quantum fits selecting the final epoch: {last_epoch_count}/{len(quantum_fits)}. '
              'This is evidence that the tested budget should be extended, not a convergence certificate.']
    if 'qae' in config['models']:
        autoencoder_fits = [record for record in fits if record['model'] == 'qae']
        native_validation = np.mean([record['selected_validation_mse'] for record in autoencoder_fits])
        probe_validation = np.mean([record['probes']['reconstruction']['validation_mse'] for record in autoencoder_fits])
        lines += ['', f'On the same validation partitions, qae native reconstruction MSE averages {native_validation:.6f}, '
                  f'while the common reconstruction probe averages {probe_validation:.6f}. '
                  'The gap supports investigating native decoder optimization before drawing representation-quality conclusions.']
    lines += ['', '![Learning curves](learning_curves.png)', '', '## Shared readout probes', '',
              'All models receive the same ridge-probe procedure. Each probe is fit on training latents and its penalty '
              'is chosen on validation only. Values below average test sequences within each data seed, then average seeds. '
              'Persistence is an uncompressed reference. Quantum probes access retained Z expectations, not the full state.', '',
              '| Model | Reconstruction ID | Reconstruction combined | Forecast ID | Forecast combined |',
              '| --- | ---: | ---: | ---: | ---: |']
    for model in config['models']:
        values = [per_seed_scores(model, task, partition).mean()
                  for task in ('reconstruction', 'forecast') for partition in ('test_id', 'test_combined_shift')]
        lines.append('| '+model+' | '+' | '.join(f'{value:.6f}' for value in values)+' |')
    maximum_rotation_error = max(float(row['rotation_prediction_max_difference']) for row in rows)
    maximum_scale_change = max(float(row['scale_descriptor_change']) for row in rows)
    lines += ['', '![Common probe scores](common_probe_scores.png)', '', '## Descriptor controls', '',
              f'Maximum compensated rotation prediction difference: {maximum_rotation_error:.3e}. '
              f'Maximum scale descriptor change: {maximum_scale_change:.3e}. '
              'The rotation control preserves probe predictions while temporal descriptors can change. '
              'Positive scaling should preserve these ordinal, median-threshold and correlation descriptors. '
              'Neither result establishes preservation of the full quantum state.', '',
              '| Model | Mean rotation descriptor change | Mean time-shuffle descriptor change |',
              '| --- | ---: | ---: |']
    for model in config['models']:
        model_rows = [row for row in rows if row['model'] == model]
        rotation_change = np.mean([float(row['rotation_descriptor_change']) for row in model_rows])
        shuffle_change = np.mean([float(row['shuffled_descriptor_change']) for row in model_rows])
        lines.append(f'| {model} | {rotation_change:.6f} | {shuffle_change:.6f} |')
    lines += ['', '## Limits and next experiment', '',
              'Two data seeds and one initialization provide descriptive replication only. The small training partition '
              'and short budget cannot establish model rankings. Descriptor estimates at this sequence length use scales 1 and 2; '
              'scale 4 fails the minimum embedding-vector count. Test outcomes must not be used to tune the confirmatory protocol.', '',
              'Generator terminology: the stored `variance_ratio` field multiplies the latent standard deviation, '
              'so its stationary variance multiplier is the square of that value (2.25 for the base middle segment '
              'and 9 for the variance-shift middle segment). These multipliers refer to latent innovations and '
              'stationary regimes; observed tanh features and transition transients need not have those ratios.', '',
              'The quantum reset convention creates discarded qubits in computational zero, whose Z readout is +1. '
              'The classical discarded coordinates start at feature zero. Near-zero trainable rotation initialization '
              'therefore gives different native decoder starting predictions. Inspect longer optimization and a separately '
              'specified feature-neutral quantum decoder initialization before interpreting native-score gaps. '
              'This initialization diagnostic must be developed from training/validation behavior, not held-out test rankings.', '',
              'Use the larger configured pilot only after inspecting optimization, learning-rate sensitivity, '
              'initialization sensitivity and runtime. A confirmatory study needs more independent data realizations '
              'and uncertainty at the data-realization level.', '', '## Reproduce', '', 'The accompanying `reproducibility_assets.zip` contains all generated datasets, checkpoints, probes and run records. '
              'Its SHA-256 is recorded in `verification.json`. The figure script requires Matplotlib, included in `requirements-test.txt`.', '', '```bash',
              'python -m pilot.run --config configs/optimization_diagnostic.json --output pilot_runs/optimization_reproduce',
              'python scripts/report_optimization_diagnostic.py --run pilot_runs/optimization_reproduce --output docs/optimization_diagnostic',
              '```', '']
    (report_directory / 'README.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    build_report(args.run, args.output)
