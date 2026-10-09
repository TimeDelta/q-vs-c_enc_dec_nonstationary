"""Report paired decoder initialization using training/validation diagnostics."""
import argparse
import json
from pathlib import Path
import shutil
import hashlib
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def summarize(record, budget):
    history = [epoch for epoch in record['history'] if epoch['epoch'] <= budget]
    if history:
        best = min(history, key=lambda epoch: epoch['validation_mse'])
        return best['validation_mse'], best['epoch']
    return record['selected_validation_mse'], None


def report_control(run_directory, output_directory):
    run, output = Path(run_directory), Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    records = json.loads((run/'fit_diagnostics.json').read_text())
    def verify_finite(value):
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError('Nonfinite diagnostic')
        if isinstance(value, dict):
            for item in value.values(): verify_finite(item)
        if isinstance(value, list):
            for item in value: verify_finite(item)
    verify_finite(records)
    config = json.loads((run/'config.json').read_text())
    manifest = json.loads((run/'manifest.json').read_text())
    if len(config['training_seeds']) != 1 or config['epochs'] != 16:
        raise ValueError('Reporter requires the prespecified 16-epoch, single-initialization control')
    expected = len(config['data_seeds']) * sum(2 if name.startswith('q') else 1 for name in config['models'])
    if len(records) != expected:
        raise ValueError('Incomplete control run')
    for filename in ('config.json', 'manifest.json', 'fit_diagnostics.json'):
        shutil.copyfile(run/filename, output/filename)
    for seed in config['data_seeds']:
        shutil.copyfile(run/f'data_{seed}/dataset_manifest.json', output/f'dataset_manifest_{seed}.json')
        with np.load(run/f'data_{seed}/training_validation.npz') as arrays:
            if set(arrays.files) != {'train','validation','train_generating_states','validation_generating_states'}:
                raise ValueError('Unexpected partition in validation-only assets')
            if any(not np.all(np.isfinite(arrays[key])) for key in arrays.files):
                raise ValueError('Nonfinite dataset')
    quantum_names = [name for name in config['models'] if name.startswith('q')]
    pairs = []
    for name in quantum_names:
        for seed in config['data_seeds']:
            paired = {r['initialization_mode']:r for r in records if r['model']==name and r['data_seed']==seed}
            ordinary, neutral = paired['near_zero'], paired['feature_neutral']
            assert ordinary['parameter_count']==neutral['parameter_count']
            assert ordinary['initial_encoder_readout_sha256']==neutral['initial_encoder_readout_sha256']
            assert ordinary['initial_probe_validation']==neutral['initial_probe_validation']
            assert neutral['initialization_metadata']['center_max_abs_readout']<1e-8
            pairs.append(dict(model=name, data_seed=seed,
                              initial_validation_difference=ordinary['initial_validation_mse']-neutral['initial_validation_mse'],
                              best_validation_difference=summarize(ordinary, config['epochs'])[0]-summarize(neutral, config['epochs'])[0]))
    figure, axes = plt.subplots(1, len(quantum_names), figsize=(6*len(quantum_names), 5), squeeze=False)
    for axis, name in zip(axes[0], quantum_names):
        for mode, color in [('near_zero', '#b54d38'), ('feature_neutral', '#24759c')]:
            model_records = [r for r in records if r['model']==name and r['initialization_mode']==mode]
            for field, linestyle in [('validation_mse', '-'), ('train_mse', '--')]:
                initial_field = 'initial_validation_mse' if field=='validation_mse' else 'initial_training_mse'
                values = np.array([[r[initial_field]]+[epoch[field] for epoch in r['history']] for r in model_records])
                timesteps = np.arange(values.shape[1])
                axis.plot(timesteps, values.mean(axis=0), color=color, linestyle=linestyle,
                          label=mode if field=='validation_mse' else None)
                if field=='validation_mse':
                    axis.fill_between(timesteps, values.min(axis=0), values.max(axis=0), color=color, alpha=.12)
        reference_name = 'cae' if name.endswith('ae') else 'cte'
        reference = [r for r in records if r['model']==reference_name]
        if reference:
            values = np.array([[r['initial_validation_mse']]+[e['validation_mse'] for e in r['history']] for r in reference])
            axis.plot(np.arange(values.shape[1]), values.mean(axis=0), color='#525252', linestyle=':', label=reference_name)
        axis.set_title(name+' native task')
        axis.set_xlabel('Epoch (0 = initialization)')
        axis.set_ylabel('Observation MSE (log scale)')
        axis.set_yscale('log')
        axis.grid(alpha=.2)
        axis.legend(fontsize=9)
    figure.suptitle('Solid: validation; dashed: training; shading: observed data-seed range, not confidence interval')
    figure.tight_layout()
    figure.savefig(output/'paired_learning_curves.png', dpi=180)
    plt.close(figure)
    lines = ['# Paired initialization control', '',
             '**Training and validation only. No test partitions were generated or evaluated.**', '',
             f"Source commit: `{manifest['git_commit']}`. {len(records)} fitted cases over {len(config['data_seeds'])} "
             f"data realizations, one initialization seed and one learning rate. Every trained case completed {config['epochs']} epochs. "
             f"Elapsed time: {manifest['elapsed_seconds']:.1f} seconds.", '',
             'The quantum architecture, parameter count and initial encoder readouts are identical within each pair. '
             'Only existing first-block decoder parameters for discarded qubits are recentered. Calibration uses a synthetic '
             'zero-input reference without dataset access, then adds the original seed jitter. Classical baselines use the same partitions.', '',
             'Validation also selects checkpoints and probe penalties. These values diagnose selection and optimization; '
             'they are not independent estimates of generalization.', '',
             '## Native validation errors', '',
             f'Entries average {len(config["data_seeds"])} data realizations. The 8-epoch and 16-epoch columns summarize nested prefixes '
             'of one trajectory. They are not separate experiments. Closed-form models have no epoch budget.', '',
             '| Model | Initialization | Initial validation MSE | Best through epoch 8 | Best through epoch 16 | Selected epochs |',
             '| --- | --- | ---: | ---: | ---: | --- |']
    groups = [(name, mode) for name in config['models'] for mode in (config['initialization_modes'] if name.startswith('q') else ['near_zero'])]
    for name, mode in groups:
        group = [r for r in records if r['model']==name and r['initialization_mode']==mode]
        initial = 'n/a' if group[0]['initial_validation_mse'] is None else f"{np.mean([r['initial_validation_mse'] for r in group]):.6f}"
        scores8 = [summarize(r,8)[0] for r in group]
        scores16 = [summarize(r,config['epochs'])[0] for r in group]
        epochs = ', '.join('n/a' if summarize(r,config['epochs'])[1] is None else str(summarize(r,config['epochs'])[1]) for r in group)
        lines.append(f'| {name} | {mode} | {initial} | {np.mean(scores8):.6f} | {np.mean(scores16):.6f} | {epochs} |')
    lines += ['', '![Paired learning curves](paired_learning_curves.png)', '', '## Paired differences', '',
              'Positive differences mean lower error for feature-neutral initialization. Each row is one data realization; '
              'two realizations do not support calibrated confidence intervals.', '',
              '| Model | Data seed | Initial validation difference | Best validation difference at 16 epochs |',
              '| --- | ---: | ---: | ---: |']
    for pair in pairs:
        lines.append(f"| {pair['model']} | {pair['data_seed']} | {pair['initial_validation_difference']:.6f} | {pair['best_validation_difference']:.6f} |")
    lines += ['', '## Selected representation probes', '',
              'Shared ridge-probe coefficients are fitted on training latents; penalties are chosen on validation. '
              'Values average data realizations. The selected encoder depends on native validation checkpoint selection.', '',
              '| Model | Initialization | Reconstruction probe validation MSE | Forecast probe validation MSE |',
              '| --- | --- | ---: | ---: |']
    for name, mode in groups:
        group = [r for r in records if r['model']==name and r['initialization_mode']==mode]
        values = [np.mean([r['selected_probe_validation'][task]['validation_mse'] for r in group]) for task in ('reconstruction','forecast')]
        lines.append(f'| {name} | {mode} | {values[0]:.6f} | {values[1]:.6f} |')
    lines += ['', '## Resource costs', '',
              '| Model | Initialization | Total case seconds | Calibration seconds |',
              '| --- | --- | ---: | ---: |']
    for name, mode in groups:
        group = [r for r in records if r['model']==name and r['initialization_mode']==mode]
        seconds = sum(r['fit_seconds'] for r in group)
        calibration_seconds = sum(r['initialization_metadata'].get('calibration_seconds',0) for r in group)
        lines.append(f'| {name} | {mode} | {seconds:.2f} | {calibration_seconds:.3f} |')
    lines += ['', '## Limits', '',
              'This control isolates decoder initialization within each quantum model. It does not equalize information '
              'capacity across real coordinates and qubits, establish hardware advantage or demonstrate convergence. '
              'The calibrated centers are near zero readout on one synthetic reference; adding seed jitter means actual '
              'initial predictions are not exactly zero. Calibration cost is recorded separately in each case.', '',
              'The data configuration has two training and two validation sequences of length 64 per realization. '
              'The model, learning-rate and seed coverage are deliberately narrow. Further optimization checks should '
              'add initialization seeds and learning rates before a confirmatory protocol is frozen. Held-out test '
              'outcomes must remain outside those decisions. Seeds 17 and 41 are exploratory realizations already used '
              'in earlier diagnostics; a confirmatory study must use fresh prespecified data seeds.', '', '## Reproduce', '',
              'The ZIP archive contains datasets, checkpoints and all run records. Its SHA-256 is in `verification.json`. '
              'Figure generation requires Matplotlib from `requirements-test.txt`.', '', '```bash',
              'python -m pilot.validation_control --config configs/initialization_control.json --output pilot_runs/initialization_control_reproduce',
              'python scripts/report_initialization_control.py --run pilot_runs/initialization_control_reproduce --output docs/initialization_control',
              '```', '']
    (output/'README.md').write_text('\n'.join(lines))
    archive = output/'reproducibility_assets.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(run.rglob('*')):
            if path.is_file(): zipped.write(path,path.relative_to(run))
    verification = dict(all_recorded_numeric_diagnostics_finite=True, source_commit=manifest['git_commit'], source_clean_at_run_start=not manifest['working_tree_dirty'],
                        fitted_cases=len(records), no_test_partitions=True, paired_initial_encoder_readouts_identical=True,
                        parameter_counts_identical_within_pairs=True,
                        initial_probe_validation_scores_identical_within_pairs=True,
                        maximum_calibration_center_error=max(r['initialization_metadata']['center_max_abs_readout'] for r in records if r['initialization_mode']=='feature_neutral'),
                        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    (output/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report_control(args.run,args.output)
