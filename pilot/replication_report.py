"""Replay and report the fixed transformation replication without refitting."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from pilot.data import make_dataset
from pilot.descriptor_study import restore_model
from pilot.fresh_study import verify_selection_lock
from pilot.metrics import predict_probe
from pilot.models import LinearEncoder, build_model
from pilot.run import validate_config, write_json

PARTITIONS = ('test_id', 'test_mean_shift', 'test_variance_shift', 'test_persistence_shift',
              'test_noise_shift', 'test_combined_shift')


def read_records(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def expected_model_names(config):
    names = list(config['models'])
    if config['include_untrained_controls']:
        for name in config['models']:
            model = build_model(name, dict(config, learning_rate=config['learning_rates'][0]), config['training_seeds'][0])
            if not isinstance(model, LinearEncoder):
                names.append('untrained_'+name)
    return names


def optimization_diagnostics(selections, config):
    diagnostics = []
    window = config['diagnostic_window_epochs']
    threshold = config['diagnostic_relative_improvement_threshold']
    for selected in selections:
        if not selected['candidates']:
            continue
        candidate_index = selected['selected_candidate_index']
        candidates = selected['candidates']
        selected_candidate = candidates[candidate_index]
        for candidate in candidates:
            history = candidate['history']
            if not history:
                continue
            if len(history) < 2*window:
                raise ValueError('Incomplete optimization diagnostic windows')
            previous_median = float(np.median([epoch['train_mse'] for epoch in history[-2*window:-window]]))
            final_median = float(np.median([epoch['train_mse'] for epoch in history[-window:]]))
            relative_improvement = (previous_median-final_median)/max(abs(previous_median), 1e-12)
            diagnostics.append(dict(data_seed=selected['data_seed'], training_seed=selected['training_seed'],
                                    model=selected['model'], candidate_index=candidate['candidate_index'],
                                    candidate_value=candidate['candidate_value'], selected=candidate is selected_candidate,
                                    final_window_relative_training_improvement=relative_improvement,
                                    training_still_improving=relative_improvement > threshold,
                                    best_validation_epoch=min(history, key=lambda row: row['validation_mse'])['epoch'],
                                    best_validation_in_final_window=(
                                        min(history, key=lambda row: row['validation_mse'])['epoch'] > config['epochs']-window),
                                    selected_candidate_on_grid_boundary=candidate_index in (0, len(candidates)-1)))
    return diagnostics


def primary_contrasts(records, config):
    """Equal weights for sequences, conditions and initializations, in that order."""
    scores = {}
    for row in records:
        if row['task'] != 'forecast':
            continue
        identity = (int(row['data_seed']), int(row['training_seed']), row['model'], row['partition'])
        scores.setdefault(identity, []).append(float(row['native_mse']))
    first_model, second_model = config['primary_comparison']
    contrasts = []
    for data_seed in config['data_seeds']:
        initialization_contrasts = []
        for training_seed in config['training_seeds']:
            first_score, second_score = [float(np.mean([
                np.mean(scores[(data_seed, training_seed, model, partition)])
                for partition in config['primary_partitions']])) for model in (first_model, second_model)]
            initialization_contrasts.append(dict(training_seed=training_seed, first_model_mse=first_score,
                                                  second_model_mse=second_score, difference=first_score-second_score))
        contrasts.append(dict(data_seed=data_seed,
                              first_model_mse=float(np.mean([row['first_model_mse'] for row in initialization_contrasts])),
                              second_model_mse=float(np.mean([row['second_model_mse'] for row in initialization_contrasts])),
                              difference=float(np.mean([row['difference'] for row in initialization_contrasts])),
                              initialization_contrasts=initialization_contrasts))
    return contrasts


def verify_run(source):
    source = Path(source)
    config = json.loads((source/'config.json').read_text())
    validate_config(config)
    manifest = json.loads((source/'manifest.json').read_text())
    lock = verify_selection_lock(source)
    if manifest.get('phase') != 'complete' or manifest['working_tree_dirty']:
        raise ValueError('Run must be complete and originate from clean committed source')
    if config != manifest['config'] or manifest['inference'] != config['inference_status']:
        raise ValueError('Run configuration or inference label differs from its manifest')
    if not manifest['started_at'] <= lock['locked_at'] <= manifest['test_generation_started_at']:
        raise ValueError('Tests were generated before every selection was locked')
    selections = json.loads((source/'fit_diagnostics.json').read_text())
    rows = read_records(source/'sequence_metrics.csv')
    constants = read_records(source/'constant_forecast_metrics.csv')
    names = expected_model_names(config)
    expected_selections = set(itertools.product(config['data_seeds'], config['training_seeds'], names))
    selection_lookup = {(row['data_seed'], row['training_seed'], row['model']): row for row in selections}
    if len(selections) != len(expected_selections) or set(selection_lookup) != expected_selections:
        raise ValueError('Missing, duplicate or unexpected model selections')
    if lock['model_cases'] != len(selections) or lock['test_partitions_generated']:
        raise ValueError('Selection lock coverage is incorrect')
    replay = json.loads((source/'checkpoint_verification.json').read_text())
    if replay['model_cases'] != len(selections) or not replay['validation_only'] or replay['test_partitions_generated']:
        raise ValueError('Checkpoint verification was incomplete or accessed test data')
    if not np.isfinite(replay['max_absolute_replay_difference']) or replay['max_absolute_replay_difference'] > 1e-10:
        raise ValueError('Checkpoint verification exceeded its tolerance')
    expected_rows = set(itertools.product(config['data_seeds'], config['training_seeds'], names, PARTITIONS,
                                         range(config['test_sequences']), ('forecast', 'reconstruction')))
    metric_lookup = {}
    for row in rows:
        identity = (int(row['data_seed']), int(row['training_seed']), row['model'], row['partition'],
                    int(row['sequence_index']), row['task'])
        metric_lookup[identity] = row
        nonnumeric = {'model', 'native_objective', 'task', 'partition', 'compressed'}
        if not all(np.isfinite(float(value)) for key, value in row.items() if key not in nonnumeric):
            raise ValueError('Nonfinite metric record')
        if float(row['rotation_prediction_max_difference']) > 1e-8:
            raise ValueError('Compensated rotation changed predictions')
    if len(rows) != len(expected_rows) or set(metric_lookup) != expected_rows:
        raise ValueError('Missing, duplicate or unexpected test records')
    for selected in selections:
        candidates = selected['candidates']
        if selected['model'].startswith('untrained_'):
            if candidates:
                raise ValueError('Untrained control has fitted candidates')
            continue
        candidate_model = build_model(selected['model'], dict(config, learning_rate=config['learning_rates'][0]), selected['training_seed'])
        analytic = isinstance(candidate_model, LinearEncoder)
        expected_values = (config['native_ridge_penalties'] if selected['model'] in (
            'reduced_rank', 'encoding_ridge', 'encoding_reduced_rank') else [0.001]) if analytic else config['learning_rates']
        if [candidate['candidate_value'] for candidate in candidates] != expected_values:
            raise ValueError('Hyperparameter candidates differ from the fixed protocol')
        for candidate_index, candidate in enumerate(candidates):
            if candidate['candidate_index'] != candidate_index or not np.isfinite(candidate['best_validation_mse']):
                raise ValueError('Invalid candidate metadata')
            history = candidate['history']
            if not analytic:
                if [epoch['epoch'] for epoch in history] != list(range(1, config['epochs']+1)):
                    raise ValueError('Incomplete fixed training budget')
                if not all(np.isfinite(value) for epoch in history for value in epoch.values()):
                    raise ValueError('Nonfinite fitting diagnostics')
                if candidate['best_validation_mse'] != min(epoch['validation_mse'] for epoch in history):
                    raise ValueError('Epoch selection is not the validation minimum')
            elif history:
                raise ValueError('Analytic reference has gradient training history')
        best_index = min(range(len(candidates)), key=lambda index: candidates[index]['best_validation_mse'])
        best = candidates[best_index]
        expected_epoch = min(best['history'], key=lambda row: row['validation_mse'])['epoch'] if best['history'] else None
        if (selected['selected_candidate_index'] != best_index or selected['selected_candidate_value'] != best['candidate_value']
                or selected['selected_epoch'] != expected_epoch or selected['selected_validation_mse'] != best['best_validation_mse']):
            raise ValueError('Selected checkpoint differs from the validation-only rule')
    maximum_replay_error = 0.0
    constants_by_identity = {(int(row['data_seed']), row['model'], row['partition'], int(row['sequence_index'])): row
                             for row in constants}
    expected_constants = set(itertools.product(config['data_seeds'], ('zero_output', 'training_mean'),
                                              PARTITIONS, range(config['test_sequences'])))
    if len(constants) != len(expected_constants) or set(constants_by_identity) != expected_constants:
        raise ValueError('Incomplete constant forecast references')
    if any(not np.isfinite(float(row['native_mse'])) for row in constants):
        raise ValueError('Nonfinite constant forecast reference')
    for data_seed in config['data_seeds']:
        partitions, metadata = make_dataset(config, data_seed)
        dataset_directory = source/f'data_{data_seed}'
        if metadata != json.loads((dataset_directory/'dataset_manifest.json').read_text()):
            raise ValueError('Dataset manifest cannot be reproduced')
        with np.load(dataset_directory/'observations_and_generating_states.npz') as stored:
            if set(stored.files) != set(partitions) or any(not np.array_equal(stored[key], value) for key, value in partitions.items()):
                raise ValueError('Saved dataset cannot be reproduced')
        with np.load(dataset_directory/'training_validation.npz') as stored:
            if any(not np.array_equal(stored[key], partitions[key]) for key in stored.files):
                raise ValueError('Locked training/validation data changed')
        for training_seed, name in itertools.product(config['training_seeds'], names):
            prefix = dataset_directory/f'training_{training_seed}'/name
            model = restore_model(name, config, training_seed, prefix)
            selected = selection_lookup[(data_seed, training_seed, name)]
            if selected['candidates']:
                maximum_replay_error = max(maximum_replay_error,
                    abs(model.native_mse(partitions['validation'])-selected['selected_validation_mse']))
            with np.load(str(prefix)+'_probes.npz') as checkpoint:
                probes = {task: (checkpoint[task+'_coefficients'], checkpoint[task+'_intercept'])
                          for task in ('forecast', 'reconstruction')}
            for partition in PARTITIONS:
                for sequence_index, sequence in enumerate(partitions[partition]):
                    latent, predictions = model.sequence_outputs(sequence)
                    if not np.all(np.isfinite(latent)) or not np.all(np.isfinite(predictions)):
                        raise ValueError('Nonfinite restored predictions')
                    native_targets = sequence[1:] if model.objective == 'forecast' else sequence[:-1]
                    native_mse = float(np.mean((predictions[:-1]-native_targets)**2))
                    for task, probe in probes.items():
                        row = metric_lookup[(data_seed, training_seed, name, partition, sequence_index, task)]
                        targets = sequence[1:] if task == 'forecast' else sequence[:-1]
                        probe_mse = float(np.mean((predict_probe(latent[:-1], probe)-targets)**2))
                        maximum_replay_error = max(maximum_replay_error, abs(native_mse-float(row['native_mse'])),
                                                   abs(probe_mse-float(row['probe_mse'])))
                        if (row['native_objective'] != model.objective or int(row['retained_dimension']) != model.latent_dimension
                                or row['compressed'] != str(model.latent_dimension < config['num_features'])):
                            raise ValueError('Model task or bottleneck metadata is incorrect')
                        if (float(row['probe_ridge_penalty']) != selected['probes'][task]['ridge_penalty']
                                or float(row['probe_validation_mse']) != selected['probes'][task]['validation_mse']):
                            raise ValueError('Evaluated probe metadata differs from the lock')
        training_mean = partitions['train'][:, 1:].reshape(-1, config['num_features']).mean(axis=0)
        for partition, sequence_index in itertools.product(PARTITIONS, range(config['test_sequences'])):
            targets = partitions[partition][sequence_index, 1:]
            for name, prediction in (('zero_output', np.zeros(config['num_features'])), ('training_mean', training_mean)):
                row = constants_by_identity[(data_seed, name, partition, sequence_index)]
                maximum_replay_error = max(maximum_replay_error, abs(float(np.mean((targets-prediction)**2))-float(row['native_mse'])))
    if not np.isfinite(maximum_replay_error) or maximum_replay_error > 1e-10:
        raise ValueError('Restored test scores or constant references differ from the recorded outcomes')
    verify_selection_lock(source)
    return config, rows, selections, dict(model_cases=len(selections), sequence_task_rows=len(rows),
        constant_rows=len(constants), all_outputs_finite=True, all_saved_predictions_replayed=True,
        maximum_score_replay_difference=maximum_replay_error, all_locked_hashes_unchanged=True,
        datasets_regenerated_exactly=True, complete_candidate_budgets=True,
        source_commit=manifest['git_commit'], selection_lock_sha256=manifest['selection_lock_sha256'])


def report(source, output):
    torch.set_num_threads(1)
    config, records, selections, verification = verify_run(source)
    contrasts = primary_contrasts(records, config)
    diagnostics = optimization_diagnostics(selections, config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'primary_contrasts.json', contrasts)
    write_json(output/'optimization_diagnostics.json', diagnostics)
    for filename in ('config.json', 'manifest.json', 'selection_lock.json', 'checkpoint_verification.json',
                     'fit_diagnostics.json', 'sequence_metrics.csv', 'constant_forecast_metrics.csv'):
        (output/filename).write_bytes((Path(source)/filename).read_bytes())
    archive_path = output/'transformation_replication_assets.zip'
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(Path(source).rglob('*')):
            if path.is_file():
                archive.write(path, path.relative_to(source))
    verification['archive_sha256'] = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    write_json(output/'verification.json', verification)
    differences = [row['difference'] for row in contrasts]
    lines = ['# Fixed-budget transformation replication', '', config['inference_status'], '',
             'Primary contrast: `'+config['primary_comparison'][0]+'` minus `'+config['primary_comparison'][1]+'`. Positive differences favor the second model.', '',
             'Each score averages test sequences within a shift, all five shifted conditions and all initialization seeds. Data realizations are the analysis units. No best initialization is selected and no significance or power claim is made.', '',
             '| Data seed | First model MSE | Second model MSE | Difference |', '| --- | ---: | ---: | ---: |']
    for row in contrasts:
        lines.append(f'| {row["data_seed"]} | {row["first_model_mse"]:.8f} | {row["second_model_mse"]:.8f} | {row["difference"]:+.8f} |')
    lines += ['', f'Mean difference: {np.mean(differences):+.8f}. Observed range: {min(differences):+.8f} to {max(differences):+.8f}.', '',
              '## All native and probe references', '',
              'The following table averages data realizations after equal weighting of initialization seeds, conditions and sequences. Native reconstruction tasks remain labeled separately. All named models and untrained controls are included.', '',
              '| Model | Native task | Readout dimension | Native ID MSE | Native shifted MSE | Forecast probe shifted MSE |',
              '| --- | --- | ---: | ---: | ---: | ---: |']
    for name in expected_model_names(config):
        forecast_rows = [row for row in records if row['model'] == name and row['task'] == 'forecast']
        inside = [float(row['native_mse']) for row in forecast_rows if row['partition'] == 'test_id']
        shifted = [row for row in forecast_rows if row['partition'] in config['primary_partitions']]
        lines.append(f'| {name} | {forecast_rows[0]["native_objective"]} | {forecast_rows[0]["retained_dimension"]} | {np.mean(inside):.8f} | '
                     f'{np.mean([float(row["native_mse"]) for row in shifted]):.8f} | {np.mean([float(row["probe_mse"]) for row in shifted]):.8f} |')
    selected_diagnostics = [row for row in diagnostics if row['selected']]
    flagged = sum(row['training_still_improving'] or row['best_validation_in_final_window'] for row in selected_diagnostics)
    lines += ['', '## Optimization and interpretation', '',
              f'Every learning-rate candidate completed {config["epochs"]} epochs. Among {len(selected_diagnostics)} selected gradient candidates, {flagged} have continuing training improvement or a best validation checkpoint in the final diagnostic window. All candidate diagnostics and grid-boundary selections are retained.', '',
              'These diagnostics can show an unresolved budget limitation. Their absence does not prove global convergence. The endpoint compares the fixed training procedure; architecture-superiority conclusions remain unsupported.', '',
              'The shared-head pair matches its two-real-readout interface, initial head, parameter count and gradient procedure. It does not match encoder function classes. The native quantum decoder can access the retained reduced state. The 81-coordinate encoding_ridge and persistence references are uncompressed. RY-only circuits can remain entangled. Descriptor measurements remain exploratory and are not MDL codelengths.', '',
              f'All {verification["model_cases"]} saved model/probe cases and {verification["sequence_task_rows"]} metric rows were verified. Every native and probe test score was replayed without refitting, with maximum absolute discrepancy {verification["maximum_score_replay_difference"]:.3g}.', '',
              f'Source commit: `{verification["source_commit"]}`. Archive SHA-256: `{verification["archive_sha256"]}`.', '',
              '[Primary contrasts](primary_contrasts.json), [all optimization diagnostics](optimization_diagnostics.json), [sequence outcomes](sequence_metrics.csv), [constant references](constant_forecast_metrics.csv), [verification](verification.json) and [complete run archive](transformation_replication_assets.zip).']
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(verification, indent=2))
    return verification


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    report(arguments.input, arguments.output)
