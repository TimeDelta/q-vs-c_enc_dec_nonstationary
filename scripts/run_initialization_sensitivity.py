"""Extend the paired control over initialization seeds and learning rates.

Reuses only recorded training/validation trajectory prefixes whose data and
simulation-source hashes match. Probe scores and later-epoch checkpoints from
reused runs are not treated as eight-epoch results.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import math
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pilot.data import make_dataset
from pilot.run import source_manifest, write_json
from pilot.validation_control import run_control


def compatible_prefix(base_config, source_directory, budget):
    source_directory = Path(source_directory)
    original = json.loads((source_directory/'config.json').read_text())
    manifest = json.loads((source_directory/'manifest.json').read_text())
    fields = ('num_features','generating_latent_dimension','bottleneck_size','num_blocks','hidden_width',
              'entanglement_topology','entanglement_gate','sequence_length','train_sequences',
              'validation_sequences','gradient_width','native_ridge_penalties','probe_ridge_penalties',
              'data_seeds','models','initialization_modes')
    if any(original[field] != base_config[field] for field in fields) or original['epochs'] < budget:
        raise ValueError('Reused control does not match the sensitivity protocol')
    current_hashes = source_manifest(base_config)['source_sha256']
    if any(current_hashes.get(path) != digest for path,digest in manifest['source_sha256'].items()):
        raise ValueError('Reused control simulation source differs')
    if manifest['evaluation'] != 'training and validation only; test partitions are not generated':
        raise ValueError('Reuse requires a validation-only control')
    for seed in base_config['data_seeds']:
        _, metadata = make_dataset(base_config, seed, include_test=False)
        previous = json.loads((source_directory/f'dataset_manifest_{seed}.json').read_text())
        if metadata != previous:
            raise ValueError('Reused control data differs')
    records = json.loads((source_directory/'fit_diagnostics.json').read_text())
    expected = len(original['data_seeds'])*len(original['training_seeds'])*sum(2 if name.startswith('q') else 1 for name in original['models'])
    if len(records) != expected or any(record['history'] and len(record['history']) < budget for record in records):
        raise ValueError('Reused control trajectories are incomplete')
    return original, manifest, records


def run_grid(grid, output_directory):
    started_at = time.perf_counter()
    if not isinstance(grid['epochs'], int) or grid['epochs'] < 1:
        raise ValueError('Epoch budget must be a positive integer')
    for values in (grid['initialization_seeds'], grid['learning_rates']):
        if not values or len(values) != len(set(values)):
            raise ValueError('Grid values must be nonempty and distinct')
    if any(not math.isfinite(rate) or rate <= 0 for rate in grid['learning_rates']):
        raise ValueError('Learning rates must be finite and positive')
    root = Path(__file__).resolve().parents[1]
    base = json.loads((root/grid['base_config']).read_text())
    base['epochs'] = grid['epochs']
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=False)
    old_config, old_manifest, old_records = compatible_prefix(base,root/grid['reuse_directory'],grid['epochs'])
    current = source_manifest(base)
    if current['working_tree_dirty']:
        raise ValueError('Sensitivity runs require a committed source tree')
    provenance = dict(grid=grid,source_commit=current['git_commit'],driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      reused_source_commit=old_manifest['git_commit'],evaluation='training and validation only',simulation_source_sha256=current['source_sha256'])
    write_json(output/'grid_manifest.json',provenance)
    all_records = []
    new_cells = reused_cells = 0
    for seed in grid['initialization_seeds']:
        for rate in grid['learning_rates']:
            cell_config = dict(base,training_seeds=[seed],learning_rates=[rate])
            label = f'initialization_{seed}_lr_{rate:g}'
            reusable = old_config['learning_rates']==[rate] and seed in old_config['training_seeds']
            if reusable:
                reused_cells += 1
                records = [record for record in old_records if record['initialization_seed']==seed]
                print(f'Reusing native validation histories through epoch {grid["epochs"]}: {label}',flush=True)
            else:
                new_cells += 1
                write_json(output/(label+'_config.json'),cell_config)
                records = run_control(cell_config,output/label)
            for record in records:
                native_history = [epoch for epoch in record['history'] if epoch['epoch']<=grid['epochs']]
                selected = min(native_history,key=lambda epoch:epoch['validation_mse']) if native_history else None
                all_records.append(dict(data_seed=record['data_seed'],initialization_seed=seed,learning_rate=rate,
                                        model=record['model'],initialization_mode=record['initialization_mode'],
                                        initial_training_mse=record['initial_training_mse'],initial_validation_mse=record['initial_validation_mse'],
                                        best_validation_mse=selected['validation_mse'] if selected else record['selected_validation_mse'],
                                        selected_epoch=selected['epoch'] if selected else None,history=native_history,
                                        parameter_count=record['parameter_count'],
                                        initial_encoder_readout_sha256=record['initial_encoder_readout_sha256'],
                                        initial_probe_validation=record['initial_probe_validation'],
                                        reused_trajectory_prefix=reusable,
                                        source_checkpoint_budget=old_config['epochs'] if reusable else grid['epochs']))
            write_json(output/'native_sensitivity_records.json',all_records)
    provenance.update(native_case_records=len(all_records),new_grid_cells=new_cells,reused_grid_cells=reused_cells,elapsed_seconds=time.perf_counter()-started_at)
    write_json(output/'grid_manifest.json',provenance)
    return all_records


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/initialization_sensitivity.json'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run_grid(json.loads(args.config.read_text()),args.output)
