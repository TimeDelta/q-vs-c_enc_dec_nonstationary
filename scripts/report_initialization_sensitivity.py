"""Verify and report the native eight-epoch initialization sensitivity grid."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def report(source, output):
    source, output = Path(source), Path(output)
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((source/'grid_manifest.json').read_text())
    grid = manifest['grid']
    base = json.loads((root/grid['base_config']).read_text())
    records = json.loads((source/'native_sensitivity_records.json').read_text())
    expected = {(data, seed, rate, model, mode)
                for data, seed, rate, model in itertools.product(base['data_seeds'],grid['initialization_seeds'],grid['learning_rates'],base['models'])
                for mode in (['near_zero','feature_neutral'] if model.startswith('q') else ['near_zero'])}
    lookup = {(r['data_seed'],r['initialization_seed'],r['learning_rate'],r['model'],r['initialization_mode']):r for r in records}
    if len(records) != len(expected) or set(lookup) != expected:
        raise ValueError('Missing, duplicate or unexpected sensitivity cases')
    pairs = []
    for data, seed, rate, model in itertools.product(base['data_seeds'],grid['initialization_seeds'],grid['learning_rates'],['qae','qte']):
        original, neutral = [lookup[(data,seed,rate,model,mode)] for mode in ['near_zero','feature_neutral']]
        for field in ['parameter_count','initial_encoder_readout_sha256','initial_probe_validation']:
            if original[field] != neutral[field]:
                raise ValueError('Paired initialization changed encoder or parameter count')
        if not original['initial_encoder_readout_sha256']:
            raise ValueError('Missing paired encoder readout hash')
        pairs.append(dict(data_seed=data,initialization_seed=seed,learning_rate=rate,model=model,
                          near_zero=original['best_validation_mse'],feature_neutral=neutral['best_validation_mse'],
                          difference=original['best_validation_mse']-neutral['best_validation_mse']))
    for record in records:
        if not np.isfinite(record['best_validation_mse']):
            raise ValueError('Nonfinite outcome')
        history = record['history']
        if history:
            if [h['epoch'] for h in history] != list(range(1,grid['epochs']+1)):
                raise ValueError('Incomplete native trajectory')
            if not all(np.isfinite(h[key]) for h in history for key in ['train_mse','validation_mse']):
                raise ValueError('Nonfinite trajectory')
            if record['best_validation_mse'] != min(h['validation_mse'] for h in history):
                raise ValueError('Budget selection mismatch')
    new_manifests = []
    calibration_errors = []
    constant_baselines = {}
    for directory in sorted(source.glob('initialization_*_lr_*')):
        if not directory.is_dir():
            continue
        cell_manifest = json.loads((directory/'manifest.json').read_text())
        if cell_manifest['working_tree_dirty'] or cell_manifest['git_commit'] != manifest['source_commit']:
            raise ValueError('Uncommitted or mismatched simulation source')
        if cell_manifest['source_sha256'] != manifest['simulation_source_sha256']:
            raise ValueError('Simulation source hashes differ across cells')
        raw_records = json.loads((directory/'fit_diagnostics.json').read_text())
        calibration_errors.extend(r['initialization_metadata']['center_max_abs_readout'] for r in raw_records if r['initialization_mode']=='feature_neutral')
        for dataset in directory.glob('data_*/training_validation.npz'):
            data_manifest = json.loads((dataset.parent/'dataset_manifest.json').read_text())
            old_data_manifest = json.loads((root/grid['reuse_directory']/f'dataset_manifest_{int(dataset.parent.name.split('_')[-1])}.json').read_text())
            if data_manifest != old_data_manifest:
                raise ValueError('Generating-data manifests differ across cells')
            with np.load(dataset) as arrays:
                if any(key.startswith('test') for key in arrays.files) or not {'train','validation'}.issubset(arrays.files):
                    raise ValueError('Unexpected dataset partition')
                if not all(np.isfinite(arrays[key]).all() for key in arrays.files):
                    raise ValueError('Nonfinite dataset')
                for task, shift in [('reconstruction',0),('forecast',1)]:
                    training = arrays['train'][:,1:] if shift else arrays['train'][:,:-1]
                    validation = arrays['validation'][:,1:] if shift else arrays['validation'][:,:-1]
                    center = training.reshape(-1,training.shape[-1]).mean(axis=0)
                    key = (int(dataset.parent.name.split('_')[-1]),task)
                    values = dict(zero_output_mse=float(np.mean(validation**2)),
                                  training_mean_mse=float(np.mean((validation-center)**2)))
                    if key in constant_baselines and constant_baselines[key] != values:
                        raise ValueError('Constant prediction references differ across cells')
                    constant_baselines[key] = values
        new_manifests.append(cell_manifest)
    if not calibration_errors or max(calibration_errors) >= 1e-8:
        raise ValueError('Calibration center tolerance failed')
    if len(new_manifests) != manifest['new_grid_cells']:
        raise ValueError('Missing new cell manifests')
    output.mkdir(parents=True,exist_ok=True)
    for filename in ['grid_manifest.json','native_sensitivity_records.json']:
        shutil.copyfile(source/filename,output/filename)
    shutil.copyfile(root/'configs/initialization_sensitivity.json',output/'config.json')
    archive = output/'initialization_sensitivity_assets.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(source.rglob('*')):
            if path.is_file():
                bundle.write(path,path.relative_to(source))
        for path in sorted((root/grid['reuse_directory']).glob('*')):
            if path.is_file() and path.suffix in ['.json','.zip']:
                bundle.write(path,Path('reused_control')/path.name)
    verification = dict(native_cases=len(records),paired_quantum_comparisons=len(pairs),
                        paired_initial_encoder_readouts_equal=True,paired_parameter_counts_equal=True,
                        paired_initial_probes_equal=True,finite_complete_trajectories=True,
                        test_partitions_generated=False,new_source_trees_clean=True,
                        generating_data_manifests_equal=True,
                        max_new_calibration_center_error=max(calibration_errors),
                        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    (output/'constant_baselines.json').write_text(json.dumps([dict(data_seed=key[0],task=key[1],**value) for key,value in sorted(constant_baselines.items())],indent=2)+'\n')
    (output/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    fig, axes = plt.subplots(2,2,figsize=(10,7),sharex=True)
    for row,model in enumerate(['qae','qte']):
        for column,rate in enumerate(grid['learning_rates']):
            axis = axes[row,column]
            for mode,color in [('near_zero','#b45309'),('feature_neutral','#0369a1')]:
                trajectories = np.array([[h['validation_mse'] for h in r['history']] for r in records if r['model']==model and r['learning_rate']==rate and r['initialization_mode']==mode])
                epochs = np.arange(1,grid['epochs']+1)
                axis.plot(epochs,trajectories.mean(axis=0),color=color,label=mode)
                axis.fill_between(epochs,trajectories.min(axis=0),trajectories.max(axis=0),color=color,alpha=.15)
            axis.set(title=f'{model.upper()}: learning rate {rate:g}',ylabel='Native validation MSE',xlabel='Epoch')
            axis.grid(alpha=.2)
            axis.legend(fontsize=8)
    fig.suptitle('Mean over two data and two initialization seeds; shading is observed range')
    fig.tight_layout()
    fig.savefig(output/'learning_rate_sensitivity.png',dpi=160)
    plt.close(fig)
    lines = ['# Initialization and learning-rate sensitivity','',
             'Exploratory training/validation diagnostic. No test partitions were generated. Each trained case has an eight-epoch budget; native checkpoint selection uses minimum validation MSE within that budget.','',
             f"Data seeds: {base['data_seeds']}. Initialization seeds: {grid['initialization_seeds']}. Learning rates: {grid['learning_rates']}. Sequences have length {base['sequence_length']}, with two training and two validation sequences per realization.",'',
             '## Native validation results','',
             'Values below are descriptive means over the two data and two initialization seeds at each learning rate. These repeated fits do not provide four independent data realizations.','',
             '| Model | Initialization | LR 0.02 | LR 0.08 |','| --- | --- | ---: | ---: |']
    for model in base['models']:
        for mode in (['near_zero','feature_neutral'] if model.startswith('q') else ['near_zero']):
            means = [np.mean([r['best_validation_mse'] for r in records if r['model']==model and r['initialization_mode']==mode and r['learning_rate']==rate]) for rate in grid['learning_rates']]
            lines.append(f'| {model} | {mode} | {means[0]:.6f} | {means[1]:.6f} |')
    lines += ['', '### Starting error and constant prediction references','',
              'Initial validation values below precede gradient training and are averaged over two data and two initialization seeds. Learning rate does not affect initialization.','',
              '| Quantum model | Near-zero initial | Feature-neutral initial |','| --- | ---: | ---: |']
    for model in ['qae','qte']:
        means = [np.mean([r['initial_validation_mse'] for r in records if r['model']==model and r['initialization_mode']==mode]) for mode in ['near_zero','feature_neutral']]
        lines.append(f'| {model} | {means[0]:.6f} | {means[1]:.6f} |')
    lines += ['', 'Constant predictors use zero output or the feature means estimated from training targets only. Values below average validation MSE over the two data realizations. They are references computed from existing data, not additional trained grid cases.','',
              '| Task | Zero output | Training mean |','| --- | ---: | ---: |']
    for task in ['reconstruction','forecast']:
        values = [value for key,value in constant_baselines.items() if key[1]==task]
        lines.append(f"| {task} | {np.mean([v['zero_output_mse'] for v in values]):.6f} | {np.mean([v['training_mean_mse'] for v in values]):.6f} |")
    lines += ['', '[Per-data constant reference scores](constant_baselines.json). Most of the paired error reduction is already present before gradient training. Feature-neutral starts remove a large starting-output penalty; the subsequent improvement over eight epochs is smaller. QAE reconstruction improves beyond a zero-output reference on these data. QTE forecasting is much closer to that reference, so the lower paired score alone does not demonstrate strong learned forecasting.','', '![Native validation learning curves](learning_rate_sensitivity.png)','',
              'Shading shows the observed minimum and maximum over fits, not a confidence interval. PCA and reduced-rank references use analytic fitting without epochs; gradient parameter count is not model capacity. Native reconstruction and forecast objectives differ, so compare models within their task.','',
              '## Every paired quantum comparison','',
              'Positive difference means feature-neutral initialization reached lower minimum validation MSE.','',
              '| Model | Data seed | Initialization seed | Learning rate | Near-zero | Feature-neutral | Difference |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for p in pairs:
        lines.append(f"| {p['model']} | {p['data_seed']} | {p['initialization_seed']} | {p['learning_rate']:g} | {p['near_zero']:.6f} | {p['feature_neutral']:.6f} | {p['difference']:+.6f} |")
    wins = sum(p['difference']>0 for p in pairs)
    lines += ['', '## Interpretation and provenance','',
              f'Feature-neutral initialization has lower minimum validation MSE in {wins} of {len(pairs)} paired fits. This is an optimization diagnostic on two previously inspected synthetic data realizations. It cannot establish superiority, calibrated uncertainty or out-of-distribution generalization. A confirmatory study needs fresh data seeds and a fixed protocol before test evaluation.','',
              'Increasing the learning rate substantially lowers near-zero error, while feature-neutral outcomes change much less. The initialization effect persists throughout this grid, but its magnitude depends on the learning rate. Classical ring, PCA and reduced-rank references remain competitive or better for their respective native tasks. These results do not demonstrate quantum advantage.','',
              'The initialization changes existing decoder parameter centers using a data-free zero-input reference. Parameter counts and initial encoder readout hashes match in every quantum pair. Initial ridge-probe validation results also match. Post-training probe scores from 16-epoch checkpoints are not presented as eight-epoch results.','',
              f"Source commit for the three new cells: `{manifest['source_commit']}`. Reused seed-101/rate-0.02 trajectories originate at `{manifest['reused_source_commit']}`. Simulation-source hashes and dataset manifests were checked before reuse. Only native histories through epoch eight are reused. The historical archive contains 16-epoch selected checkpoints; it does not contain eight-epoch selected checkpoints for the reused cell.",'',
              f"{manifest['new_grid_cells']} new grid cells and {manifest['reused_grid_cells']} reused trajectory cell yield {len(records)} native records. New computation took {manifest['elapsed_seconds']:.1f} seconds, excluding the earlier control. Every new cell records a clean source tree and its environment manifest.",'',
              '[Machine-readable native records](native_sensitivity_records.json), [grid provenance](grid_manifest.json), [verification](verification.json) and [datasets, trajectories and checkpoint archive](initialization_sensitivity_assets.zip).','',
              'Reproduce using the commands in [the protocol](../initialization_control_protocol.md). The report generator checks complete case coverage, finite trajectories, paired hashes, source provenance and the absence of test partitions.','']
    (output/'README.md').write_text('\n'.join(lines))
    print(json.dumps(verification,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    arguments=parser.parse_args()
    report(arguments.input,arguments.output)
