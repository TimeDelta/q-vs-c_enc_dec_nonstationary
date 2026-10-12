"""Run a bounded CPU pilot without selecting anything on test outcomes."""

import argparse
import csv
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch

from pilot.data import make_dataset
from pilot.metrics import (choose_probe, descriptor_distance, predict_probe,
                           temporal_descriptors, transformation_controls)
from pilot.models import build_model, fit_selected_model, LinearEncoder


MODEL_NAMES = ('cae', 'crae', 'cte', 'crte', 'qae', 'qrae', 'qte', 'qrte',
               'qae_noent', 'qte_noent', 'mlp_ae', 'mlp_te', 'gru_te',
               'pca', 'reduced_rank', 'random_linear', 'persistence')


def validate_config(config):
    for key in ('sequence_length', 'num_features', 'bottleneck_size', 'epochs', 'hidden_width',
                'train_sequences', 'validation_sequences', 'test_sequences'):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(key + ' must be a positive integer')
    if config['sequence_length'] < 64:
        raise ValueError('sequence_length must be at least 64 for the descriptor checks')
    if not 1 <= config['bottleneck_size'] < config['num_features']:
        raise ValueError('The pilot requires a bottleneck smaller than the input')
    if config['gradient_width'] <= 0:
        raise ValueError('gradient_width must be positive')
    for key in ('learning_rates', 'probe_ridge_penalties', 'native_ridge_penalties'):
        if not config[key] or any(value <= 0 or not np.isfinite(value) for value in config[key]):
            raise ValueError(key + ' must contain finite positive values')
    for key in ('data_seeds', 'training_seeds'):
        if not config[key] or len(set(config[key])) != len(config[key]):
            raise ValueError(key + ' must contain distinct seeds')
        if any(not isinstance(seed, int) or seed < 0 for seed in config[key]):
            raise ValueError(key + ' must contain nonnegative integers')
    if not config['models'] or set(config['models']) - set(MODEL_NAMES):
        raise ValueError('Unknown or empty model selection')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def source_manifest(config):
    repository_root = Path(__file__).resolve().parents[1]
    source_files = list(repository_root.glob('*.py')) + list((repository_root / 'pilot').glob('*.py'))
    source_hashes = {str(path.relative_to(repository_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted(source_files)}
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=repository_root, check=True,
                            capture_output=True, text=True).stdout.strip()
    status = subprocess.run(['git', 'status', '--porcelain'], cwd=repository_root, check=True,
                            capture_output=True, text=True).stdout.strip()
    return dict(protocol_version=2, purpose=config['purpose'], config=config, git_commit=commit,
                working_tree_dirty=bool(status), source_sha256=source_hashes,
                versions={package: version(package) for package in ('numpy', 'scipy', 'torch', 'qiskit', 'antropy')},
                python=platform.python_version(), platform=platform.platform(), device='cpu',
                quantum_simulation='exact_density_matrix', measurement_shots=None,
                quantum_complexity_observation='retained single-qubit Z expectations',
                evaluation='one-step teacher-forced forecast and same-time reconstruction',
                native_selection='validation MSE only; every candidate completes the epoch budget',
                probe_selection='training-only fit; validation-only ridge selection',
                inference='descriptive pilot; no significance or quantum-advantage claims')


def flatten_inputs(observations):
    return observations[:, :-1].reshape(-1, observations.shape[-1])


def flatten_forecast_targets(observations):
    return observations[:, 1:].reshape(-1, observations.shape[-1])


def fit_evaluation_probes(model, training, validation, config):
    training_latents = np.stack([model.sequence_outputs(sequence)[0][:-1] for sequence in training])
    validation_latents = np.stack([model.sequence_outputs(sequence)[0][:-1] for sequence in validation])
    flattened_training_latents = training_latents.reshape(-1, model.latent_dimension)
    flattened_validation_latents = validation_latents.reshape(-1, model.latent_dimension)
    probes = {}
    probe_metadata = {}
    for task, target_function in [('reconstruction', flatten_inputs), ('forecast', flatten_forecast_targets)]:
        probes[task], probe_metadata[task] = choose_probe(
            flattened_training_latents, target_function(training),
            flattened_validation_latents, target_function(validation),
            config['probe_ridge_penalties'],
        )
    return probes, probe_metadata


def evaluate_model(model, partitions, config, data_seed, training_seed, records, output_directory,
                   *, locked_probes=None, locked_probe_metadata=None):
    if (locked_probes is None) != (locked_probe_metadata is None):
        raise ValueError('Locked probes and their metadata must be supplied together')
    if locked_probes is None:
        probes, probe_metadata = fit_evaluation_probes(model, partitions['train'], partitions['validation'], config)
    else:
        probes, probe_metadata = locked_probes, locked_probe_metadata
    if locked_probes is None:
        np.savez(output_directory / (model.name + '_probes.npz'),
                 **{task + '_coefficients': probe[0] for task, probe in probes.items()},
                 **{task + '_intercept': probe[1] for task, probe in probes.items()})
    for partition_index, partition_name in enumerate(partitions):
        if not partition_name.startswith('test_') or partition_name.endswith('_generating_states'):
            continue
        observations = partitions[partition_name]
        for sequence_index, sequence in enumerate(observations):
            latent_sequence, native_predictions = model.sequence_outputs(sequence)
            latent_sequence = latent_sequence[:-1]
            observed_descriptors = temporal_descriptors(sequence[:-1])
            latent_descriptors = temporal_descriptors(latent_sequence)
            native_targets = sequence[1:] if model.objective == 'forecast' else sequence[:-1]
            native_mse = float(np.mean((native_predictions[:-1] - native_targets)**2))
            for task, probe in probes.items():
                targets = sequence[1:] if task == 'forecast' else sequence[:-1]
                predictions = predict_probe(latent_sequence, probe)
                probe_mse = float(np.mean((predictions - targets)**2))
                control_generator = np.random.default_rng(np.random.SeedSequence(
                    [data_seed, training_seed, partition_index, sequence_index]
                ))
                controls = transformation_controls(latent_sequence, probe, control_generator)
                records.append(dict(data_seed=data_seed, training_seed=training_seed,
                                    model=model.name, native_objective=model.objective,
                                    task=task, partition=partition_name, sequence_index=sequence_index,
                                    retained_dimension=model.latent_dimension,
                                    compressed=model.latent_dimension < sequence.shape[1],
                                    native_mse=native_mse, probe_mse=probe_mse,
                                    probe_validation_mse=probe_metadata[task]['validation_mse'],
                                    probe_ridge_penalty=probe_metadata[task]['ridge_penalty'],
                                    descriptor_distance=descriptor_distance(observed_descriptors, latent_descriptors),
                                    **{'input_' + key: value for key, value in observed_descriptors.items()},
                                    **{'latent_' + key: value for key, value in latent_descriptors.items()},
                                    **controls))
    return probe_metadata


def summarize(records):
    groups = {}
    for record in records:
        key = (record['model'], record['task'], record['partition'])
        groups.setdefault(key, []).append(record)
    return [dict(model=key[0], task=key[1], partition=key[2], sequence_records=len(rows),
                 mean_probe_mse=float(np.mean([row['probe_mse'] for row in rows])),
                 mean_descriptor_distance=float(np.mean([row['descriptor_distance'] for row in rows])),
                 max_rotation_prediction_difference=max(row['rotation_prediction_max_difference'] for row in rows))
            for key, rows in sorted(groups.items())]


def run_pilot(config, output_directory):
    validate_config(config)
    output_directory = Path(output_directory)
    # Avoid mixing manifests, checkpoints or records from separate invocations.
    output_directory.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest = source_manifest(config)
    write_json(output_directory / 'manifest.json', manifest)
    write_json(output_directory / 'config.json', config)
    records = []
    fitting_records = []
    started_at = time.perf_counter()
    for data_seed in config['data_seeds']:
        partitions, dataset_metadata = make_dataset(config, data_seed)
        dataset_directory = output_directory / ('data_' + str(data_seed))
        dataset_directory.mkdir()
        np.savez_compressed(dataset_directory / 'observations_and_generating_states.npz', **partitions)
        write_json(dataset_directory / 'dataset_manifest.json', dataset_metadata)
        for training_seed in config['training_seeds']:
            model_directory = dataset_directory / ('training_' + str(training_seed))
            model_directory.mkdir()
            for model_name in config['models']:
                print(f'data={data_seed} initialization={training_seed} model={model_name}', flush=True)
                model, fitting_metadata = fit_selected_model(
                    model_name, config, partitions['train'], partitions['validation'], training_seed
                )
                model.save(model_directory / model_name)
                probe_metadata = evaluate_model(model, partitions, config, data_seed, training_seed,
                                                records, model_directory)
                fitting_records.append(dict(data_seed=data_seed, training_seed=training_seed,
                                            model=model_name, probes=probe_metadata, **fitting_metadata))
                if config['include_untrained_controls'] and not isinstance(model, LinearEncoder):
                    # Same architecture and initialization; no fitted encoder weights.
                    untrained_config = dict(config, learning_rate=config['learning_rates'][0])
                    untrained_model = build_model(model_name, untrained_config, training_seed)
                    untrained_model.name = 'untrained_' + model_name
                    evaluate_model(untrained_model, partitions, config, data_seed, training_seed,
                                   records, model_directory)
    write_json(output_directory / 'fit_diagnostics.json', fitting_records)
    write_json(output_directory / 'summary.json', summarize(records))
    with (output_directory / 'sequence_metrics.csv').open('w', newline='') as output_file:
        csv_writer = csv.DictWriter(output_file, fieldnames=list(records[0]))
        csv_writer.writeheader()
        csv_writer.writerows(records)
    manifest['elapsed_seconds'] = time.perf_counter() - started_at
    manifest['sequence_metric_rows'] = len(records)
    write_json(output_directory / 'manifest.json', manifest)
    print(f'Completed {len(records)} sequence-task records in {manifest["elapsed_seconds"]:.1f} seconds', flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/cpu_smoke.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--no-quantum', action='store_true', help='Exclude quantum models from this invocation')
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text())
    if arguments.no_quantum:
        config['models'] = [name for name in config['models'] if not name.startswith('q')]
    run_pilot(config, arguments.output)


if __name__ == '__main__':
    main()
