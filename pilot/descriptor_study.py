"""Exploratory, nested data-realization validation of temporal descriptors."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import torch

from pilot.fresh_study import file_digest, verify_selection_lock, write_csv
from pilot.metrics import descriptor_distance, predict_probe, temporal_descriptors
from pilot.models import build_model, LegacyEncoder, LinearEncoder, NeuralEncoder
from pilot.run import source_manifest, write_json

DESCRIPTOR_KEYS = ('permutation_entropy', 'binary_lz76', 'lag1_correlation')
FEATURE_SETS = ('identity', 'validation', 'baseline', 'distance', 'components', 'latent')


def restore_model(name, config, initialization_seed, checkpoint_prefix):
    """Restore inference state only; never train or select a new checkpoint."""
    base_name = name.removeprefix('untrained_')
    model = build_model(base_name, dict(config, learning_rate=config['learning_rates'][0]), initialization_seed)
    model.name = name
    if isinstance(model, LegacyEncoder) and model.quantum:
        model.model.load(str(checkpoint_prefix))
    elif isinstance(model, (LegacyEncoder, NeuralEncoder)):
        checkpoint = torch.load(str(checkpoint_prefix)+'.pt', weights_only=True)
        if checkpoint['protocol_version'] != 2 or checkpoint['model_name'] != name:
            raise ValueError('Checkpoint name or protocol mismatch')
        network = model.model if isinstance(model, LegacyEncoder) else model.network
        network.load_state_dict(checkpoint['state_dict'])
    elif isinstance(model, LinearEncoder):
        with np.load(str(checkpoint_prefix)+'.npz') as checkpoint:
            for attribute in ('input_mean', 'target_mean', 'encoder_matrix', 'decoder_matrix'):
                setattr(model, attribute, checkpoint[attribute].copy())
            model.ridge_penalty = float(checkpoint['ridge_penalty'])
    else:
        raise TypeError('Unsupported checkpoint family')
    return model


def ridge_predictions(training_features, training_targets, heldout_features, penalty):
    """Fit scaling and intercept on fitting rows only, with mean-loss ridge."""
    feature_mean = training_features.mean(axis=0)
    feature_scale = training_features.std(axis=0)
    feature_scale = np.where(feature_scale > 1e-12, feature_scale, 1.0)
    target_mean = training_targets.mean()
    standardized_training = (training_features-feature_mean)/feature_scale
    standardized_heldout = (heldout_features-feature_mean)/feature_scale
    coefficients = np.linalg.solve(
        standardized_training.T@standardized_training + len(training_targets)*penalty*np.eye(training_features.shape[1]),
        standardized_training.T@(training_targets-target_mean))
    return standardized_heldout@coefficients+target_mean


def nested_predictions(features, targets, groups, penalties):
    """Keep complete data realizations together in both validation loops."""
    features, targets, groups = np.asarray(features), np.asarray(targets), np.asarray(groups)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 3 or len(penalties) == 0 or min(penalties) <= 0:
        raise ValueError('At least three groups and positive ridge penalties are required')
    if not np.all(np.isfinite(features)) or not np.all(np.isfinite(targets)):
        raise ValueError('Nonfinite analysis inputs')
    predictions = np.full(len(targets), np.nan)
    audits = []
    for heldout_group in unique_groups:
        training_mask = groups != heldout_group
        inner_groups = unique_groups[unique_groups != heldout_group]
        candidate_losses = []
        inner_memberships = []
        for penalty in penalties:
            losses = []
            for inner_heldout in inner_groups:
                inner_training_mask = training_mask & (groups != inner_heldout)
                inner_validation_mask = groups == inner_heldout
                predicted = ridge_predictions(features[inner_training_mask], targets[inner_training_mask],
                                              features[inner_validation_mask], penalty)
                losses.append(float(np.mean((predicted-targets[inner_validation_mask])**2)))
            candidate_losses.append(float(np.mean(losses)))
        selected_penalty = penalties[int(np.argmin(candidate_losses))]
        heldout_mask = groups == heldout_group
        predictions[heldout_mask] = ridge_predictions(features[training_mask], targets[training_mask],
                                                     features[heldout_mask], selected_penalty)
        for inner_heldout in inner_groups:
            inner_memberships.append(dict(heldout_data_seed=int(inner_heldout),
                                          training_data_seeds=[int(group) for group in inner_groups if group != inner_heldout]))
        audits.append(dict(heldout_data_seed=int(heldout_group),training_data_seeds=inner_groups.tolist(),
                           selected_penalty=selected_penalty,inner_candidate_losses=candidate_losses,
                           inner_folds=inner_memberships))
    return predictions, audits


def feature_matrix(rows, feature_set, *, native=False):
    model_names = sorted({row['model'] for row in rows})
    identities = np.asarray([[float(row['model']==name) for name in model_names] for row in rows])
    validation_key = 'native_validation_mse' if native else 'probe_validation_mse'
    validation = np.asarray([[row[validation_key]] for row in rows])
    if feature_set == 'identity':
        return identities
    if feature_set == 'validation':
        return validation
    baseline = np.column_stack((identities, validation))
    if feature_set == 'baseline':
        return baseline
    if feature_set == 'distance':
        additional = np.asarray([[row['validation_descriptor_distance']] for row in rows])
    elif feature_set == 'components':
        additional = np.asarray([[row['validation_squared_difference_'+key] for key in DESCRIPTOR_KEYS] for row in rows])
    elif feature_set == 'latent':
        additional = np.asarray([[row['validation_latent_'+key] for key in DESCRIPTOR_KEYS] for row in rows])
    else:
        raise ValueError('Unknown feature set')
    return np.column_stack((baseline, additional))


def validation_features(archive_directory, config, fitting_records):
    rows, reproduction_errors = [], []
    for fitting_record in fitting_records:
        data_seed, training_seed, name = (fitting_record[key] for key in ('data_seed','training_seed','model'))
        model_directory = archive_directory/f'data_{data_seed}'/f'training_{training_seed}'
        with np.load(archive_directory/f'data_{data_seed}'/'training_validation.npz') as dataset:
            validation = dataset['validation'].copy()
        model = restore_model(name, config, training_seed, model_directory/name)
        with np.load(model_directory/(name+'_probes.npz')) as checkpoint:
            probe = (checkpoint['forecast_coefficients'].copy(),checkpoint['forecast_intercept'].copy())
        observed_descriptors, latent_descriptors, distances, component_squares = [], [], [], []
        prediction_errors = []
        for sequence in validation:
            latent_sequence = model.sequence_outputs(sequence)[0][:-1]
            observed = temporal_descriptors(sequence[:-1])
            latent = temporal_descriptors(latent_sequence)
            observed_descriptors.append(observed)
            latent_descriptors.append(latent)
            distances.append(descriptor_distance(observed,latent))
            differences = np.array([observed[key]-latent[key] for key in DESCRIPTOR_KEYS])
            differences[-1] /= 2
            component_squares.append(differences**2)
            prediction_errors.append(float(np.mean((predict_probe(latent_sequence,probe)-sequence[1:])**2)))
        reproduced_mse = float(np.mean(prediction_errors))
        locked_mse = fitting_record['probes']['forecast']['validation_mse']
        reproduction_errors.append(abs(reproduced_mse-locked_mse))
        if not np.isclose(reproduced_mse,locked_mse,rtol=0,atol=1e-10):
            raise RuntimeError('Restored validation probe differs: '+name)
        native_validation = model.native_mse(validation)
        if 'selected_validation_mse' in fitting_record and not np.isclose(
                native_validation,fitting_record['selected_validation_mse'],rtol=0,atol=1e-10):
            raise RuntimeError('Restored native validation checkpoint differs: '+name)
        row = dict(data_seed=data_seed,training_seed=training_seed,model=name,
                   native_objective=model.objective,probe_validation_mse=locked_mse,
                   native_validation_mse=native_validation,validation_descriptor_distance=float(np.mean(distances)))
        for index,key in enumerate(DESCRIPTOR_KEYS):
            row['validation_input_'+key] = float(np.mean([value[key] for value in observed_descriptors]))
            row['validation_latent_'+key] = float(np.mean([value[key] for value in latent_descriptors]))
            row['validation_squared_difference_'+key] = float(np.mean(component_squares,axis=0)[index])
        rows.append(row)
        print(f'Restored validation data={data_seed} model={name}',flush=True)
    return rows, max(reproduction_errors)


def attach_outcomes(feature_rows, metric_rows):
    """Attach previously measured outcomes only after validation features freeze."""
    output = []
    for feature in feature_rows:
        matched = [row for row in metric_rows if row['task']=='forecast' and
                   int(row['data_seed'])==feature['data_seed'] and
                   int(row['training_seed'])==feature['training_seed'] and row['model']==feature['model']]
        shifted = [row for row in matched if row['partition']!='test_id']
        in_distribution = [row for row in matched if row['partition']=='test_id']
        conditions = {row['partition'] for row in shifted}
        if len(conditions)!=5 or len(shifted)!=20 or len(in_distribution)!=4:
            raise RuntimeError('Incomplete outcome cells')
        output.append(dict(feature,
                           shifted_probe_mse=float(np.mean([float(row['probe_mse']) for row in shifted])),
                           id_probe_mse=float(np.mean([float(row['probe_mse']) for row in in_distribution])),
                           shifted_native_mse=float(np.mean([float(row['native_mse']) for row in shifted]))))
    return output


def run_analysis(analysis_config, output_directory):
    output = Path(output_directory)
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    evidence = Path(analysis_config['evidence_directory'])
    archive = evidence/'fresh_seed_forecasting_assets.zip'
    manifest = source_manifest(analysis_config)
    if manifest['working_tree_dirty']:
        raise ValueError('Commit analysis source before running')
    write_json(output/'analysis_config.json',analysis_config)
    with tempfile.TemporaryDirectory() as temporary:
        archive_directory = Path(temporary)
        with zipfile.ZipFile(archive) as bundle:
            for name in bundle.namelist():
                if not (archive_directory/name).resolve().is_relative_to(archive_directory.resolve()):
                    raise ValueError('Unsafe archive path')
            bundle.extractall(archive_directory)
        original_lock = verify_selection_lock(archive_directory)
        config = json.loads((archive_directory/'config.json').read_text())
        fitting_records = json.loads((archive_directory/'fit_diagnostics.json').read_text())
        features,max_probe_error = validation_features(archive_directory,config,fitting_records)
        if len(features)!=112 or len({row['data_seed'] for row in features})!=8:
            raise RuntimeError('Expected the complete eight-realization study')
        write_csv(output/'validation_features.csv',features)
        feature_hash = file_digest(output/'validation_features.csv')
        # This file is written before any test outcome rows are read.
        write_json(output/'feature_lock.json',dict(source_commit=manifest['git_commit'],
                    validation_features_sha256=feature_hash,test_outcomes_read=False,
                    archive_sha256=file_digest(archive),original_selection_files=len(original_lock['file_sha256'])))
        with (evidence/'sequence_metrics.csv').open() as stream:
            metric_rows = list(csv.DictReader(stream))
        rows = attach_outcomes(features,metric_rows)
        write_csv(output/'analysis_cells.csv',rows)
        prediction_rows,fold_rows,audits = [],[],{}
        for endpoint,target_key,native in [('shifted_probe','shifted_probe_mse',False),
                                         ('id_probe','id_probe_mse',False),
                                         ('shifted_native','shifted_native_mse',True)]:
            selected_rows = [row for row in rows if not native or row['native_objective']=='forecast']
            groups = np.array([row['data_seed'] for row in selected_rows])
            targets = np.array([row[target_key] for row in selected_rows])
            endpoint_predictions = {}
            for feature_set in FEATURE_SETS:
                predictions,fold_audits = nested_predictions(feature_matrix(selected_rows,feature_set,native=native),
                                                             targets,groups,analysis_config['ridge_penalties'])
                audits[endpoint+'_'+feature_set] = fold_audits
                endpoint_predictions[feature_set] = predictions
                for row,prediction,target in zip(selected_rows,predictions,targets):
                    prediction_rows.append(dict(endpoint=endpoint,features=feature_set,data_seed=row['data_seed'],
                                                model=row['model'],observed_mse=float(target),predicted_mse=float(prediction),
                                                squared_prediction_error=float((prediction-target)**2)))
            for group in np.unique(groups):
                mask = groups==group
                fold = dict(endpoint=endpoint,data_seed=int(group),model_cases=int(mask.sum()))
                for feature_set,predictions in endpoint_predictions.items():
                    fold[feature_set+'_loss'] = float(np.mean((predictions[mask]-targets[mask])**2))
                fold['baseline_minus_distance_loss'] = fold['baseline_loss']-fold['distance_loss']
                fold_rows.append(fold)
        write_csv(output/'heldout_predictions.csv',prediction_rows)
        write_csv(output/'fold_losses.csv',fold_rows)
        write_json(output/'fold_audit.json',audits)
        summaries = {}
        for endpoint in ('shifted_probe','id_probe','shifted_native'):
            folds = [row for row in fold_rows if row['endpoint']==endpoint]
            summaries[endpoint] = dict(mean_loss={feature_set:float(np.mean([row[feature_set+'_loss'] for row in folds]))
                                                  for feature_set in FEATURE_SETS},
                                       mean_baseline_minus_distance=float(np.mean([row['baseline_minus_distance_loss'] for row in folds])),
                                       distance_improved_realizations=sum(row['baseline_minus_distance_loss']>0 for row in folds),
                                       independent_realizations=len(folds),model_cases_per_realization=folds[0]['model_cases'])
        forecast_rows = [row for row in metric_rows if row['task']=='forecast']
        controls = {key:dict(mean=float(np.mean([float(row[key]) for row in forecast_rows])),
                            maximum=max(float(row[key]) for row in forecast_rows))
                    for key in ('scale_descriptor_change','rotation_descriptor_change','shuffled_descriptor_change',
                                'rotation_prediction_max_difference')}
        verify_selection_lock(archive_directory)
    if file_digest(output/'validation_features.csv') != feature_hash:
        raise RuntimeError('Validation features changed after outcome access')
    write_json(output/'summary.json',dict(endpoints=summaries,transformation_controls=controls))
    write_json(output/'verification.json',dict(original_selection_lock_unchanged=True,
                validation_features_unchanged=True,validation_features_rows=len(features),
                max_restored_validation_probe_error=max_probe_error,
                prediction_rows=len(prediction_rows),finite_predictions=bool(all(np.isfinite(row['predicted_mse']) for row in prediction_rows)),
                independent_data_realizations=8,analysis_status='exploratory after test inspection'))
    manifest.update(evidence_archive_sha256=file_digest(archive),feature_sha256=feature_hash,
                    analysis_status='exploratory after test inspection',inference='nested group prediction; no significance claim')
    write_json(output/'manifest.json',manifest)
    print(json.dumps(summaries,indent=2),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/descriptor_generalization.json'))
    parser.add_argument('--output',type=Path,required=True)
    arguments = parser.parse_args()
    run_analysis(json.loads(arguments.config.read_text()),arguments.output)
