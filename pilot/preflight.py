"""Validate the prospective experiment without generating or fitting any data."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import torch

from pilot.models import LinearEncoder, build_model
from pilot.run import source_manifest, validate_config


def preflight(config, *, require_clean_source=True):
    validate_config(config)
    required_architecture = dict(num_features=4, generating_latent_dimension=2, bottleneck_size=2,
                                 num_blocks=1, entanglement_topology='circular', entanglement_gate='cz')
    if any(config.get(key) != value for key, value in required_architecture.items()):
        raise ValueError('Architecture differs from the prospective protocol')
    required_models = {'cte', 'qte', 'qte_noent', 'cte_affine', 'cte_shared', 'qte_shared',
                       'qte_ry', 'qte_ry_noent', 'encoding_ridge', 'encoding_reduced_rank',
                       'reduced_rank', 'mlp_te', 'gru_te', 'pca', 'random_linear', 'persistence'}
    if set(config['models']) != required_models or not config['include_untrained_controls']:
        raise ValueError('Prospective controls or untrained references are missing')
    previous_data_seeds = {17, 41, 73, 89, 97, 113, 127, 139, 151, 163, 179, 191, 211}
    if set(config['data_seeds']) & previous_data_seeds:
        raise ValueError('The prospective study cannot reuse previous diagnostic or test realizations')
    if config.get('primary_comparison') != ['qte_shared', 'cte_shared']:
        raise ValueError('The prospective primary contrast must be qte_shared minus cte_shared')
    if not set(config['primary_comparison']).issubset(config['models']):
        raise ValueError('Primary models are missing')
    expected_shifts = {'test_mean_shift', 'test_variance_shift', 'test_persistence_shift',
                       'test_noise_shift', 'test_combined_shift'}
    if len(config['primary_partitions']) != 5 or set(config['primary_partitions']) != expected_shifts:
        raise ValueError('All five shifts must receive equal primary weight')
    if len(config['training_seeds']) < 3 or len(config['data_seeds']) < 8:
        raise ValueError('The prospective replication requires eight realizations and three initializations')
    if type(config['diagnostic_window_epochs']) is not int or config['diagnostic_window_epochs'] < 1:
        raise ValueError('Diagnostic window must be a positive integer')
    if config['epochs'] < 2*config['diagnostic_window_epochs']:
        raise ValueError('Two complete optimization diagnostic windows are required')
    threshold = config['diagnostic_relative_improvement_threshold']
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError('Diagnostic improvement threshold must be finite and positive')
    repository = Path(__file__).resolve().parents[1]
    for line in (repository/'requirements-pilot.txt').read_text().splitlines():
        if '==' in line and not line.startswith('#'):
            package, expected_version = line.split('==')
            if version(package).split('+')[0] != expected_version:
                raise ValueError('Runtime differs from the pinned version of '+package)
    manifest = source_manifest(config)
    if require_clean_source and manifest['working_tree_dirty']:
        raise ValueError('Commit the reviewed source before running the experiment')
    torch.set_num_threads(1)
    models, trainable_count = [], 0
    for name in config['models']:
        model = build_model(name, dict(config, learning_rate=config['learning_rates'][0]), config['training_seeds'][0])
        analytic = isinstance(model, LinearEncoder)
        trainable_count += int(not analytic)
        models.append(dict(model=name, gradient_parameters=model.parameter_count,
                           readout_dimension=model.latent_dimension, analytic=analytic,
                           native_objective=model.objective))
    calibration_count = 0
    for initialization_seed in config['training_seeds']:
        shared_models = [build_model(name, dict(config, learning_rate=config['learning_rates'][0]), initialization_seed)
                         for name in config['primary_comparison']]
        np.testing.assert_array_equal(shared_models[0].head_coefficients, shared_models[1].head_coefficients)
        np.testing.assert_array_equal(shared_models[0].head_intercept, shared_models[1].head_intercept)
        if shared_models[0].parameter_count != shared_models[1].parameter_count:
            raise ValueError('Primary parameter counts differ')
        for name in ('qte', 'qte_noent', 'qte_ry', 'qte_ry_noent'):
            if name in config['models']:
                model = build_model(name, dict(config, learning_rate=config['learning_rates'][0]), initialization_seed)
                if model.initialization_metadata.get('center_max_abs_readout', np.inf) > 1e-8:
                    raise ValueError('Data-free native quantum initialization failed')
                calibration_count += 1
    initialization_groups = len(config['data_seeds'])*len(config['training_seeds'])
    cases_per_group = len(models)+(trainable_count if config['include_untrained_controls'] else 0)
    return dict(git_commit=manifest['git_commit'], working_tree_dirty=manifest['working_tree_dirty'],
                versions=manifest['versions'], generated_datasets=0, fitted_models=0,
                data_realizations=len(config['data_seeds']), initialization_groups=initialization_groups,
                planned_model_cases=initialization_groups*cases_per_group,
                planned_sequence_task_rows=initialization_groups*cases_per_group*6*config['test_sequences']*2,
                planned_gradient_updates=initialization_groups*trainable_count*len(config['learning_rates'])*config['epochs'],
                planned_constant_rows=len(config['data_seeds'])*6*config['test_sequences']*2,
                native_calibrations_verified=calibration_count, identical_shared_heads_verified=True, models=models)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/transformation_replication.json'))
    arguments = parser.parse_args()
    print(json.dumps(preflight(json.loads(arguments.config.read_text())), indent=2))
