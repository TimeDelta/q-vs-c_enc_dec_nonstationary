"""Independently check saved fold coverage, losses and evidence identities."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def verify(directory, original_archive):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    lock = json.loads((directory/'feature_lock.json').read_text())
    summary = json.loads((directory/'summary.json').read_text())
    feature_hash = hashlib.sha256((directory/'validation_features.csv').read_bytes()).hexdigest()
    assert feature_hash==lock['validation_features_sha256']==manifest['feature_sha256']
    assert hashlib.sha256(Path(original_archive).read_bytes()).hexdigest()==manifest['evidence_archive_sha256']
    assert not manifest['working_tree_dirty']
    seeds = {113,127,139,151,163,179,191,211}
    audits = json.loads((directory/'fold_audit.json').read_text())
    outer_count = inner_count = 0
    for folds in audits.values():
        assert len(folds)==8
        assert {fold['heldout_data_seed'] for fold in folds}==seeds
        for fold in folds:
            outer_count += 1
            assert set(fold['training_data_seeds'])==seeds-{fold['heldout_data_seed']}
            assert {inner['heldout_data_seed'] for inner in fold['inner_folds']}==set(fold['training_data_seeds'])
            for inner in fold['inner_folds']:
                inner_count += 1
                assert set(inner['training_data_seeds'])==seeds-{fold['heldout_data_seed'],inner['heldout_data_seed']}
    with (directory/'heldout_predictions.csv').open() as stream:
        predictions = list(csv.DictReader(stream))
    assert len(predictions)==1920
    assert len({tuple(row[key] for key in ('endpoint','features','data_seed','model')) for row in predictions})==1920
    for row in predictions:
        assert np.isfinite(float(row['predicted_mse']))
        assert np.isclose((float(row['observed_mse'])-float(row['predicted_mse']))**2,
                          float(row['squared_prediction_error']),rtol=0,atol=1e-15)
    with (directory/'fold_losses.csv').open() as stream:
        folds = list(csv.DictReader(stream))
    assert len(folds)==24
    for endpoint,reported in summary['endpoints'].items():
        selected = [fold for fold in folds if fold['endpoint']==endpoint]
        assert len(selected)==8
        assert {int(fold['data_seed']) for fold in selected}==seeds
        for features,loss in reported['mean_loss'].items():
            group_losses = []
            for fold in selected:
                matching = [row for row in predictions if row['endpoint']==endpoint and
                            row['features']==features and row['data_seed']==fold['data_seed']]
                assert len(matching)==reported['model_cases_per_realization']
                group_loss = float(np.mean([float(row['squared_prediction_error']) for row in matching]))
                assert np.isclose(group_loss,float(fold[features+'_loss']),rtol=0,atol=1e-15)
                group_losses.append(group_loss)
            assert np.isclose(np.mean(group_losses),loss,rtol=0,atol=1e-15)
        differences = [float(fold['baseline_minus_distance_loss']) for fold in selected]
        for fold,difference in zip(selected,differences):
            assert np.isclose(float(fold['baseline_loss'])-float(fold['distance_loss']),difference,rtol=0,atol=1e-15)
        assert sum(difference>0 for difference in differences)==reported['distance_improved_realizations']
        assert np.isclose(np.mean(differences),reported['mean_baseline_minus_distance'],rtol=0,atol=1e-15)
    checks = dict(outer_folds_verified=outer_count,inner_group_memberships_verified=inner_count,
                  unique_predictions_verified=len(predictions),saved_losses_and_summary_verified=True,
                  evidence_archive_and_validation_feature_hashes_verified=True)
    (directory/'independent_verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--original-archive',type=Path,default=Path('docs/fresh_seed_forecasting/fresh_seed_forecasting_assets.zip'))
    arguments = parser.parse_args()
    verify(arguments.directory,arguments.original_archive)
