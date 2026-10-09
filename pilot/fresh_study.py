"""Fit and lock every selection before generating fresh-seed test partitions."""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import pickle
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from pilot.data import make_dataset
from pilot.models import build_model, fit_selected_model, LinearEncoder
from pilot.run import (evaluate_model, fit_evaluation_probes, source_manifest,
                       summarize, validate_config, write_json)


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_selection_lock(directory):
    directory = Path(directory)
    lock = json.loads((directory/'selection_lock.json').read_text())
    for relative_path, expected_digest in lock['file_sha256'].items():
        if file_digest(directory/relative_path) != expected_digest:
            raise RuntimeError('Selection lock changed: '+relative_path)
    return lock


def write_csv(path, records):
    with Path(path).open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def fit_data_seed(config, data_seed, output):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    fitted_cases, fitting_records, constants = [], [], []
    partitions, data_metadata = make_dataset(config,data_seed,include_test=False)
    if any(key.startswith('test_') for key in partitions):
        raise RuntimeError('Test partition present before selection lock')
    dataset_directory = output/f'data_{data_seed}'
    dataset_directory.mkdir()
    np.savez_compressed(dataset_directory/'training_validation.npz',**partitions)
    write_json(dataset_directory/'training_validation_manifest.json',data_metadata)
    target_mean = partitions['train'][:,1:].reshape(-1,config['num_features']).mean(axis=0)
    constants.append(dict(data_seed=data_seed,forecast_training_mean=target_mean.tolist()))
    for training_seed in config['training_seeds']:
        model_directory = dataset_directory/f'training_{training_seed}'
        model_directory.mkdir()
        for name in config['models']:
            print(f'Fitting data={data_seed} initialization={training_seed} model={name}',flush=True)
            model, metadata = fit_selected_model(name,config,partitions['train'],partitions['validation'],training_seed)
            cases = [(model,metadata)]
            if config['include_untrained_controls'] and not isinstance(model,LinearEncoder):
                untrained = build_model(name,dict(config,learning_rate=config['learning_rates'][0]),training_seed)
                untrained.name = 'untrained_'+name
                cases.append((untrained,dict(candidates=[],parameter_count=untrained.parameter_count,
                                             control='same seeded architecture and initialization without gradient training')))
            for selected_model, selected_metadata in cases:
                selected_model.save(model_directory/selected_model.name)
                probes, probe_metadata = fit_evaluation_probes(selected_model,partitions['train'],partitions['validation'],config)
                np.savez(model_directory/(selected_model.name+'_probes.npz'),
                         **{task+'_coefficients':probe[0] for task,probe in probes.items()},
                         **{task+'_intercept':probe[1] for task,probe in probes.items()})
                for probe in probes.values():
                    for coefficients in probe:
                        coefficients.setflags(write=False)
                fitting_records.append(dict(data_seed=data_seed,training_seed=training_seed,model=selected_model.name,
                                            native_objective=selected_model.objective,probes=probe_metadata,**selected_metadata))
                fitted_cases.append((data_seed,training_seed,selected_model,probes,probe_metadata,model_directory))
            write_json(dataset_directory/'fit_diagnostics.json',fitting_records)
    # Transfer ordinary serialized bytes; avoid Torch shared-memory socket reducers.
    return pickle.dumps((data_seed,data_metadata,fitted_cases,fitting_records,constants))


def run_study(config, output_directory, *, require_clean_source=True):
    validate_config(config)
    if set(config['data_seeds']) & {17,41,73}:
        raise ValueError('Fresh-seed study cannot reuse exploratory data seeds')
    output = Path(output_directory)
    manifest = source_manifest(config)
    if require_clean_source and manifest['working_tree_dirty']:
        raise ValueError('Study requires committed source before fitting')
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest.update(evaluation='all selections locked before any test partitions are generated',
                    inference='prespecified bounded-budget replication; data realizations are the analysis units',
                    started_at=datetime.now(timezone.utc).isoformat())
    write_json(output/'manifest.json',manifest)
    write_json(output/'config.json',config)
    started_at = time.perf_counter()
    fitted_cases, fitting_records, constants = [], [], []
    data_metadata_by_seed = {}
    workers = config.get('execution_workers',1)
    if not isinstance(workers,int) or workers < 1:
        raise ValueError('execution_workers must be a positive integer')
    if workers == 1:
        results = [fit_data_seed(config,data_seed,output) for data_seed in config['data_seeds']]
    else:
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = [pool.submit(fit_data_seed,config,data_seed,output) for data_seed in config['data_seeds']]
            results = [future.result() for future in futures]
    for result in results:
        data_seed,data_metadata,cases,selections,data_constants = pickle.loads(result)
        data_metadata_by_seed[data_seed] = data_metadata
        for _,_,_,probes,_,_ in cases:
            for probe in probes.values():
                for coefficients in probe:
                    coefficients.setflags(write=False)
        fitted_cases.extend(cases)
        fitting_records.extend(selections)
        constants.extend(data_constants)
    write_json(output/'fit_diagnostics.json',fitting_records)
    write_json(output/'constant_forecast_predictions.json',constants)
    fitting_seconds = time.perf_counter()-started_at
    locked_paths = [path for path in output.rglob('*') if path.is_file() and path.name!='manifest.json']
    lock = dict(source_commit=manifest['git_commit'],locked_at=datetime.now(timezone.utc).isoformat(),
                model_cases=len(fitted_cases),test_partitions_generated=False,
                file_sha256={str(path.relative_to(output)):file_digest(path) for path in sorted(locked_paths)})
    write_json(output/'selection_lock.json',lock)
    print(f'Locked {len(fitted_cases)} model/probe selections before test generation',flush=True)
    verify_selection_lock(output)
    test_partitions_by_seed = {}
    test_generation_started_at = datetime.now(timezone.utc).isoformat()
    for data_seed in config['data_seeds']:
        partitions, metadata = make_dataset(config,data_seed,include_test=True)
        old_metadata = data_metadata_by_seed[data_seed]
        if metadata['mixing_matrix'] != old_metadata['mixing_matrix'] or any(
            metadata['partitions'][key] != value for key,value in old_metadata['partitions'].items()):
            raise RuntimeError('Training/validation changed during test generation')
        np.savez_compressed(output/f'data_{data_seed}'/'observations_and_generating_states.npz',**partitions)
        write_json(output/f'data_{data_seed}'/'dataset_manifest.json',metadata)
        test_partitions_by_seed[data_seed] = partitions
    records = []
    for data_seed,training_seed,model,probes,probe_metadata,model_directory in fitted_cases:
        print(f'Evaluating locked data={data_seed} model={model.name}',flush=True)
        evaluate_model(model,test_partitions_by_seed[data_seed],config,data_seed,training_seed,records,model_directory,
                       locked_probes=probes,locked_probe_metadata=probe_metadata)
        write_csv(output/'sequence_metrics.csv',records)
    constant_records = []
    for constant in constants:
        for partition, observations in test_partitions_by_seed[constant['data_seed']].items():
            if not partition.startswith('test_') or partition.endswith('_generating_states'):
                continue
            for sequence_index, sequence in enumerate(observations):
                for name,prediction in [('zero_output',np.zeros(config['num_features'])),
                                        ('training_mean',np.array(constant['forecast_training_mean']))]:
                    constant_records.append(dict(data_seed=constant['data_seed'],model=name,partition=partition,
                                                 sequence_index=sequence_index,native_mse=float(np.mean((sequence[1:]-prediction)**2))))
    write_csv(output/'constant_forecast_metrics.csv',constant_records)
    write_json(output/'summary.json',summarize(records))
    verify_selection_lock(output)
    manifest.update(selection_locked_at=lock['locked_at'],test_generation_started_at=test_generation_started_at,
                    fitting_seconds=fitting_seconds,elapsed_seconds=time.perf_counter()-started_at,
                    fitted_model_cases=len(fitted_cases),sequence_metric_rows=len(records),
                    selection_lock_sha256=file_digest(output/'selection_lock.json'))
    write_json(output/'manifest.json',manifest)
    print(f'Completed {len(records)} test metric rows in {manifest["elapsed_seconds"]:.1f} seconds',flush=True)
    return records


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/fresh_seed_forecasting.json'))
    parser.add_argument('--output',type=Path,required=True)
    arguments=parser.parse_args()
    run_study(json.loads(arguments.config.read_text()),arguments.output)
