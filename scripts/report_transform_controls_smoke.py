"""Check the complete implementation smoke and record transformation proofs."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from pilot.descriptor_study import restore_model
from pilot.fresh_study import verify_selection_lock
from pilot.models import build_model
from pilot.run import write_json
from pilot.transform_controls import bounded_ry_features


def mathematical_checks(config):
    proof_config = dict(config,learning_rate=.02,quantum_initialization='near_zero')
    grid = np.array(list(itertools.product((-1.,0.,1.),repeat=4)))
    basis = bounded_ry_features(grid)
    heldout = np.random.default_rng(89).uniform(-.95,.95,(17,4))
    errors = {}
    for name in ('qte','qte_noent','qte_ry','qte_ry_noent','qte_shared'):
        model = build_model(name,proof_config,303)
        model.set_parameter_values(np.random.default_rng(101).uniform(-.7,.7,model.parameter_count))
        latent,predictions = model.sequence_outputs(grid)
        coefficients = np.linalg.solve(basis,np.column_stack((latent,predictions)))
        actual_latent,actual_predictions = model.sequence_outputs(heldout)
        error = float(np.max(np.abs(bounded_ry_features(heldout)@coefficients-
                                    np.column_stack((actual_latent,actual_predictions)))))
        if error>2e-12:
            raise ValueError('Feature-span check failed: '+name)
        errors[name] = error
    original = build_model('cte',proof_config,303)
    affine = build_model('cte_affine',proof_config,303)
    original_latent,original_predictions = original.sequence_outputs(heldout)
    affine_latent,affine_predictions = affine.sequence_outputs(heldout)
    givens_error = float(max(np.max(np.abs(original_latent-affine_latent)),
                            np.max(np.abs(original_predictions-affine_predictions))))
    if givens_error>1e-14:
        raise ValueError('Affine initialization differs from original Givens')
    classical,quantum = [build_model(name,proof_config,303) for name in ('cte_shared','qte_shared')]
    heads_equal = np.array_equal(classical.head_coefficients,quantum.head_coefficients) and np.array_equal(
        classical.head_intercept,quantum.head_intercept)
    if not heads_equal or classical.parameter_count!=16 or quantum.parameter_count!=16:
        raise ValueError('Shared head initialization or parameter counts differ')
    return dict(basis_columns=basis.shape[1],interpolation_points=len(grid),heldout_points=len(heldout),
                feature_span_max_absolute_errors=errors,original_givens_initial_output_max_difference=givens_error,
                shared_initial_heads_identical=heads_equal,shared_parameter_counts=[classical.parameter_count,quantum.parameter_count])


def report(source,output):
    source,output = Path(source),Path(output)
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    config = json.loads((source/'config.json').read_text())
    manifest = json.loads((source/'manifest.json').read_text())
    if manifest['working_tree_dirty']:
        raise ValueError('Smoke must originate from committed clean source')
    if manifest['inference']!=config['inference_status']:
        raise ValueError('Incorrect smoke inference label')
    lock = verify_selection_lock(source)
    fitting = json.loads((source/'fit_diagnostics.json').read_text())
    with (source/'sequence_metrics.csv').open() as stream:
        metrics = list(csv.DictReader(stream))
    expected_names = set(config['models'])|{'untrained_'+name for name in config['models']
                            if name not in ('reduced_rank','encoding_ridge','encoding_reduced_rank')}
    if len(fitting)!=19 or {record['model'] for record in fitting}!=expected_names or len(metrics)!=228:
        raise ValueError('Incomplete smoke coverage')
    identities = [tuple(row[key] for key in ('data_seed','training_seed','model','partition','sequence_index','task')) for row in metrics]
    if len(set(identities))!=228:
        raise ValueError('Duplicate smoke metric rows')
    for row in metrics:
        for key,value in row.items():
            if key not in ('data_seed','training_seed','model','native_objective','task','partition','sequence_index','retained_dimension','compressed'):
                if not np.isfinite(float(value)):
                    raise ValueError('Nonfinite smoke output: '+key)
        if row['model']=='encoding_ridge' and (row['compressed']!='False' or int(row['retained_dimension'])!=81):
            raise ValueError('Full feature reference incorrectly marked compressed')
    model_rows,restoration_errors = [],[]
    for record in fitting:
        name = record['model']
        data_seed,training_seed = record['data_seed'],record['training_seed']
        with np.load(source/f'data_{data_seed}'/'training_validation.npz') as dataset:
            validation = dataset['validation'].copy()
        prefix = source/f'data_{data_seed}'/f'training_{training_seed}'/name
        restored = restore_model(name,config,training_seed,prefix)
        if record['candidates']:
            expected_error = min(candidate['best_validation_mse'] for candidate in record['candidates'])
            if not np.isclose(expected_error,record['selected_validation_mse'],rtol=0,atol=1e-12):
                raise ValueError('Selection is not the validation minimum')
            for candidate in record['candidates']:
                if candidate['history'] and len(candidate['history'])!=config['epochs']:
                    raise ValueError('Truncated epoch budget')
            checkpoint_error = abs(restored.native_mse(validation)-expected_error)
            restoration_errors.append(checkpoint_error)
            if checkpoint_error>1e-10:
                raise ValueError('Restored selected native predictions differ: '+name)
        model_rows.append(dict(model=name,gradient_parameters=record['parameter_count'],
                               readout_dimension=restored.latent_dimension,native_objective=restored.objective))
    proofs = mathematical_checks(config)
    verify_selection_lock(source)
    verification = dict(model_cases=len(fitting),sequence_task_rows=len(metrics),all_outputs_finite=True,
                        original_selection_lock_unchanged=True,locked_files=len(lock['file_sha256']),
                        complete_epoch_budgets=True,validation_minimum_selection=True,
                        max_restored_native_validation_difference=max(restoration_errors),
                        implementation_only=True,**proofs)
    archive_path = output/'transformation_controls_assets.zip'
    with zipfile.ZipFile(archive_path,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source.rglob('*')):
            if path.is_file():
                archive.write(path,path.relative_to(source))
    verification['archive_sha256'] = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    write_json(output/'verification.json',verification)
    for filename in ('config.json','manifest.json','selection_lock.json','fit_diagnostics.json'):
        (output/filename).write_bytes((source/filename).read_bytes())
    (output/'sequence_metrics.csv').write_bytes((source/'sequence_metrics.csv').read_bytes())
    lines = ['# Transformation-control implementation smoke', '',
             'All seven new controls run alongside the original Givens (`cte`), native quantum (`qte`), no-entanglement quantum and raw-input reduced-rank models. This uses already inspected diagnostic seed 97, length 64 and two training epochs per learning-rate candidate. It validates implementation and does not supply an architecture ranking or new manuscript finding.', '',
             '| Model case | Trainable architecture parameters | Readout dimension | Native task |',
             '| --- | ---: | ---: | --- |']
    for row in model_rows:
        lines.append(f'| {row["model"]} | {row["gradient_parameters"]} | {row["readout_dimension"]} | {row["native_objective"]} |')
    lines += ['', f'All {len(fitting)} model/probe cases are locked before generating any test partitions. All {len(metrics)} sequence/task rows are finite and complete. The {verification["locked_files"]} selection-file hashes remain unchanged. Every candidate completes its epoch budget and validation selects its minimum-scoring checkpoint. Maximum restored native validation error difference: {verification["max_restored_native_validation_difference"]:.3g}.', '',
              '## Transformation checks', '',
              'Untrained cases perform no gradient updates despite retaining the architecture parameter count. Analytic references have fitted coefficient arrays and zero gradient-trained parameters.', '',
              f'The bounded-RY product basis has {proofs["basis_columns"]} columns. On {proofs["heldout_points"]} held-out feature vectors, coefficients derived from {proofs["interpolation_points"]} interpolation points reproduce the native and retained-readout functions:', '',
              '| Model | Maximum absolute span error |', '| --- | ---: |']
    for name,error in proofs['feature_span_max_absolute_errors'].items():
        lines.append(f'| {name} | {error:.3g} |')
    lines += ['', 'These are numerical implementation checks of the algebraic feature-span derivation, not a forecasting experiment or a fitted performance comparison. The gain/offset Givens extension starts with original predictions to numerical precision. Both shared-decoder architectures have 16 trainable parameters and exactly identical initial head coefficients and offsets.', '',
              f'Source commit: `{manifest["git_commit"]}`. Archive SHA-256: `{verification["archive_sha256"]}`. Total smoke wall time: {manifest["elapsed_seconds"]:.1f} seconds.', '',
              '[Protocol and interpretation](../transformation_controls_protocol.md), [verification](verification.json), [configuration](config.json), [selections](fit_diagnostics.json), [metric rows](sequence_metrics.csv) and [all data, model/probe checkpoints and lock files](transformation_controls_assets.zip).', '',
              'Reproduce from a clean committed checkout with pinned CPU dependencies:', '', '```bash',
              'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pilot.fresh_study --config configs/transform_controls_smoke.json --output /tmp/transformation_controls_smoke',
              'PYTHONPATH=. python scripts/report_transform_controls_smoke.py /tmp/transformation_controls_smoke /tmp/transformation_controls_report', '```']
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(verification,indent=2))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('output',type=Path)
    arguments = parser.parse_args()
    report(arguments.source,arguments.output)
