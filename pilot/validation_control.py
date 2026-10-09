"""Paired initialization control with training/validation data only."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from pilot.data import make_dataset
from pilot.metrics import choose_probe
from pilot.models import build_model, LinearEncoder
from pilot.run import source_manifest, validate_config, write_json


def probe_validation(model, training, validation, penalties):
    train_latents = np.concatenate([model.sequence_outputs(sequence)[0][:-1] for sequence in training])
    validation_latents = np.concatenate([model.sequence_outputs(sequence)[0][:-1] for sequence in validation])
    metadata = {}
    for task, shift in [('reconstruction', 0), ('forecast', 1)]:
        train_targets = (training[:, 1:] if shift else training[:, :-1]).reshape(-1, training.shape[-1])
        validation_targets = (validation[:, 1:] if shift else validation[:, :-1]).reshape(-1, validation.shape[-1])
        _, metadata[task] = choose_probe(train_latents, train_targets, validation_latents, validation_targets, penalties)
    return metadata, hashlib.sha256(train_latents.tobytes()).hexdigest()


def run_control(config, output_directory):
    validate_config(config)
    if len(config['learning_rates']) != 1 or len(config['native_ridge_penalties']) != 1:
        raise ValueError('Control requires one predefined learning rate and native ridge penalty')
    if config['initialization_modes'] != ['near_zero', 'feature_neutral']:
        raise ValueError('Control requires the paired near_zero and feature_neutral modes')
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest = source_manifest(config)
    manifest.update(evaluation='training and validation only; test partitions are not generated',
                    inference='initialization and budget diagnostic; no test outcomes or significance claims')
    write_json(directory/'manifest.json', manifest)
    write_json(directory/'config.json', config)
    records = []
    paired_latent_hashes = {}
    started_at = time.perf_counter()
    for data_seed in config['data_seeds']:
        partitions, data_metadata = make_dataset(config, data_seed, include_test=False)
        if any(key.startswith('test_') for key in partitions):
            raise RuntimeError('Test partition entered validation-only control')
        dataset_directory = directory/f'data_{data_seed}'
        dataset_directory.mkdir()
        np.savez_compressed(dataset_directory/'training_validation.npz', **partitions)
        write_json(dataset_directory/'dataset_manifest.json', data_metadata)
        training, validation = partitions['train'], partitions['validation']
        for initialization_seed in config['training_seeds']:
            for name in config['models']:
                modes = config['initialization_modes'] if name.startswith('q') else ['near_zero']
                for mode in modes:
                    print(f'data={data_seed} initialization={initialization_seed} model={name} mode={mode}', flush=True)
                    case_started_at = time.perf_counter()
                    case_config = dict(config, learning_rate=config['learning_rates'][0], quantum_initialization=mode)
                    model = build_model(name, case_config, initialization_seed, config['native_ridge_penalties'][0])
                    initial_probes, latent_hash = {}, None
                    initial_training_mse, initial_validation_mse = None, None
                    if isinstance(model, LinearEncoder):
                        model.fit(training)
                        history = []
                    else:
                        initial_training_mse = model.native_mse(training)
                        initial_validation_mse = model.native_mse(validation)
                        initial_probes, latent_hash = probe_validation(model, training, validation, config['probe_ridge_penalties'])
                        if name.startswith('q'):
                            key = (data_seed, initialization_seed, name)
                            if mode == 'near_zero':
                                paired_latent_hashes[key] = latent_hash
                            elif latent_hash != paired_latent_hashes[key]:
                                raise RuntimeError('Initialization control changed the initial encoder readout')
                        history, _ = model.train(training, validation, config['epochs'], config['gradient_width'])
                    selected_validation = model.native_mse(validation)
                    selected_training = model.native_mse(training)
                    selected_probes, _ = probe_validation(model, training, validation, config['probe_ridge_penalties'])
                    case_name = f'{name}__{mode}__initialization_{initialization_seed}'
                    model.save(dataset_directory/case_name)
                    records.append(dict(data_seed=data_seed, initialization_seed=initialization_seed, model=name,
                                        initialization_mode=mode, initial_training_mse=initial_training_mse,
                                        initial_validation_mse=initial_validation_mse,
                                        selected_training_mse=selected_training, selected_validation_mse=selected_validation,
                                        initial_probe_validation=initial_probes, selected_probe_validation=selected_probes,
                                        history=history, initial_encoder_readout_sha256=latent_hash,
                                        parameter_count=model.parameter_count,
                                        initialization_metadata=getattr(model, 'initialization_metadata', {}),
                                        fit_seconds=time.perf_counter()-case_started_at))
                    # Persist each completed case for long CPU diagnostics.
                    write_json(directory/'fit_diagnostics.json', records)
    manifest.update(elapsed_seconds=time.perf_counter()-started_at, fitted_cases=len(records))
    write_json(directory/'manifest.json', manifest)
    print(f'Completed {len(records)} training/validation cases in {manifest["elapsed_seconds"]:.1f} seconds', flush=True)
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/initialization_control.json'))
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    run_control(json.loads(arguments.config.read_text()), arguments.output)
