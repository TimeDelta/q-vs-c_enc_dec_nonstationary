"""Verify and report the prespecified fresh-seed forecasting replication."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import sys
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pilot.data import make_dataset
from pilot.fresh_study import verify_selection_lock


PARTITIONS=['test_id','test_mean_shift','test_variance_shift','test_persistence_shift','test_noise_shift','test_combined_shift']


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def model_scores(records,config,model,metric,partitions):
    return np.array([np.mean([float(record[metric]) for record in records if record['model']==model and int(record['data_seed'])==seed
                             and record['partition'] in partitions and record.get('task','forecast')=='forecast'])
                     for seed in config['data_seeds']])


def report(source,output):
    source,output=Path(source),Path(output)
    config=json.loads((source/'config.json').read_text())
    manifest=json.loads((source/'manifest.json').read_text())
    selections=json.loads((source/'fit_diagnostics.json').read_text())
    records=read_csv(source/'sequence_metrics.csv')
    constants=read_csv(source/'constant_forecast_metrics.csv')
    lock=verify_selection_lock(source)
    if manifest['working_tree_dirty'] or lock['source_commit']!=manifest['git_commit']:
        raise ValueError('Study source was uncommitted or differs from selection lock')
    if not manifest['started_at']<=lock['locked_at']<=manifest['test_generation_started_at']:
        raise ValueError('Test generation preceded the selection lock')
    if hashlib.sha256((source/'selection_lock.json').read_bytes()).hexdigest()!=manifest['selection_lock_sha256']:
        raise ValueError('Selection lock file changed')
    control_models=['untrained_'+name for name in config['models'] if name not in ['pca','reduced_rank','random_linear','persistence']] if config['include_untrained_controls'] else []
    models=config['models']+control_models
    expected={(seed,initialization,model,partition,sequence,task)
              for seed,initialization,model,partition,sequence,task in itertools.product(
                  config['data_seeds'],config['training_seeds'],models,PARTITIONS,range(config['test_sequences']),['reconstruction','forecast'])}
    observed={(int(record['data_seed']),int(record['training_seed']),record['model'],record['partition'],int(record['sequence_index']),record['task']) for record in records}
    if len(records)!=len(expected) or observed!=expected:
        raise ValueError('Incomplete, duplicate or unexpected test records')
    numeric_fields=['native_mse','probe_mse','descriptor_distance','rotation_prediction_max_difference',
                    'scale_descriptor_change','rotation_descriptor_change','shuffled_descriptor_change']
    if not all(np.isfinite(float(record[field])) for record in records for field in numeric_fields):
        raise ValueError('Nonfinite test outcomes')
    if max(float(record['rotation_prediction_max_difference']) for record in records)>1e-8:
        raise ValueError('Compensated rotation changed probe predictions')
    expected_selections={(seed,initialization,model) for seed,initialization,model in itertools.product(config['data_seeds'],config['training_seeds'],models)}
    lookup={(record['data_seed'],record['training_seed'],record['model']):record for record in selections}
    if len(selections)!=len(expected_selections) or set(lookup)!=expected_selections:
        raise ValueError('Incomplete selections')
    for selected in selections:
        candidates=selected['candidates']
        for candidate in candidates:
            if candidate['history']:
                if [epoch['epoch'] for epoch in candidate['history']]!=list(range(1,config['epochs']+1)):
                    raise ValueError('Candidate did not complete the fixed epoch budget')
                if candidate['best_validation_mse']!=min(epoch['validation_mse'] for epoch in candidate['history']):
                    raise ValueError('Epoch selection differs from the protocol')
        if candidates and selected['selected_validation_mse']!=min(c['best_validation_mse'] for c in candidates):
            raise ValueError('Hyperparameter selection differs from the protocol')
    for record in records:
        selected=lookup[(int(record['data_seed']),int(record['training_seed']),record['model'])]
        metadata=selected['probes'][record['task']]
        if float(record['probe_ridge_penalty'])!=metadata['ridge_penalty'] or float(record['probe_validation_mse'])!=metadata['validation_mse']:
            raise ValueError('Evaluated probe differs from the locked selection')
    for seed in config['data_seeds']:
        full_metadata=json.loads((source/f'data_{seed}'/'dataset_manifest.json').read_text())
        training_metadata=json.loads((source/f'data_{seed}'/'training_validation_manifest.json').read_text())
        regenerated,regenerated_metadata=make_dataset(config,seed)
        if regenerated_metadata!=full_metadata:
            raise ValueError('Generated dataset provenance changed')
        with np.load(source/f'data_{seed}'/'observations_and_generating_states.npz') as arrays:
            if set(arrays.files)!=set(regenerated) or not all(np.array_equal(arrays[key],regenerated[key]) for key in arrays.files):
                raise ValueError('Saved dataset cannot be reproduced')
        if any(full_metadata['partitions'][key]!=value for key,value in training_metadata['partitions'].items()):
            raise ValueError('Training/validation changed at test generation')
    expected_constants={(seed,model,partition,sequence) for seed,model,partition,sequence in itertools.product(config['data_seeds'],['zero_output','training_mean'],PARTITIONS,range(config['test_sequences']))}
    if len(constants)!=len(expected_constants) or {(int(record['data_seed']),record['model'],record['partition'],int(record['sequence_index'])) for record in constants}!=expected_constants:
        raise ValueError('Incomplete constant references')
    output.mkdir(parents=True,exist_ok=True)
    for filename in ['config.json','manifest.json','selection_lock.json','fit_diagnostics.json','sequence_metrics.csv','constant_forecast_metrics.csv']:
        shutil.copyfile(source/filename,output/filename)
    quantum=model_scores(records,config,'qte','native_mse',PARTITIONS[1:])
    reduced_rank=model_scores(records,config,'reduced_rank','native_mse',PARTITIONS[1:])
    contrasts=[dict(data_seed=seed,qte_native_ood_mse=float(first),reduced_rank_native_ood_mse=float(second),difference=float(first-second))
               for seed,first,second in zip(config['data_seeds'],quantum,reduced_rank)]
    (output/'primary_contrasts.json').write_text(json.dumps(contrasts,indent=2)+'\n')
    archive=output/'fresh_seed_forecasting_assets.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(source.rglob('*')):
            if path.is_file():
                bundle.write(path,path.relative_to(source))
    verification=dict(model_cases=len(selections),test_metric_rows=len(records),constant_rows=len(constants),
                      data_realizations=len(config['data_seeds']),all_selection_hashes_unchanged=True,
                      tests_generated_after_selection_lock=True,datasets_regenerated_exactly=True,
                      fixed_budget_and_validation_selection_verified=True,probe_metadata_matches_lock=True,
                      finite_test_outcomes=True,max_rotation_prediction_difference=max(float(record['rotation_prediction_max_difference']) for record in records),
                      archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    (output/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    forecast_models=['qte','qte_noent','cte','mlp_te','gru_te','reduced_rank','persistence']
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    differences=quantum-reduced_rank
    axes[0].bar([str(seed) for seed in config['data_seeds']],differences,color=['#b45309' if difference>0 else '#0369a1' for difference in differences])
    axes[0].axhline(0,color='black',linewidth=.8)
    axes[0].set(title='Primary paired contrast',xlabel='Independent data realization',ylabel='QTE - reduced-rank shifted-test MSE')
    means,ranges=[],[]
    for model in forecast_models:
        values=model_scores(records,config,model,'native_mse',PARTITIONS[1:])
        means.append(values.mean())
        ranges.append([values.mean()-values.min(),values.max()-values.mean()])
    axes[1].barh(forecast_models,means,color='#0369a1',alpha=.8)
    axes[1].errorbar(means,range(len(means)),xerr=np.array(ranges).T,fmt='none',ecolor='black',capsize=3)
    axes[1].invert_yaxis()
    axes[1].set(title='Native forecasting: mean and observed range',xlabel='Shifted-test MSE over data realizations')
    fig.tight_layout()
    fig.savefig(output/'fresh_seed_forecasting.png',dpi=170)
    plt.close(fig)
    lines=['# Fresh-seed forecasting replication','',
           'Prespecified one-step teacher-forced forecasting study. All model and probe selections were locked before any test partitions were generated. Native forecast MSE over the five shifted conditions is the primary endpoint.','',
           f"Data seeds: {config['data_seeds']}. Initialization seed: {config['training_seeds']}. Every learning-rate candidate completed {config['epochs']} epochs. Independent generator realizations, not sequences or shift conditions, are the analysis units.",'',
           '## Primary contrast: QTE minus reduced-rank regression','',
           'Positive differences favor reduced-rank regression. Each score averages the four test sequences within each shifted condition, then gives the five conditions equal weight.','',
           '| Data seed | QTE | Reduced-rank | Difference |','| --- | ---: | ---: | ---: |']
    for contrast in contrasts:
        lines.append(f"| {contrast['data_seed']} | {contrast['qte_native_ood_mse']:.6f} | {contrast['reduced_rank_native_ood_mse']:.6f} | {contrast['difference']:+.6f} |")
    lines += ['',f'Mean difference: {differences.mean():+.6f}. Observed range: {differences.min():+.6f} to {differences.max():+.6f}. QTE has lower error in {sum(differences<0)} of {len(differences)} realizations. No significance threshold or power claim was specified.','',
              '![Primary contrast and native forecasting references](fresh_seed_forecasting.png)','',
              '## Native forecasting references','',
              'Means below average per-realization scores. Reconstruction-only native objectives from PCA and random projection are excluded from this table. Their forecast probes are reported separately.','',
              '| Model | In-distribution MSE | Shifted-test MSE | Shifted-test observed range |','| --- | ---: | ---: | --- |']
    native_models=forecast_models+[name for name in control_models if name.replace('untrained_','') in forecast_models]
    for model in native_models+['zero_output','training_mean']:
        source_records=constants if model in ['zero_output','training_mean'] else records
        inside=model_scores(source_records,config,model,'native_mse',PARTITIONS[:1])
        shifted=model_scores(source_records,config,model,'native_mse',PARTITIONS[1:])
        lines.append(f'| {model} | {inside.mean():.6f} | {shifted.mean():.6f} | {shifted.min():.6f} to {shifted.max():.6f} |')
    lines += ['', '## Common linear forecast probes','',
              'These secondary scores measure linear readout of each selected representation. Probe coefficients use training only and penalty selection uses validation only. Encoder checkpoint selection still uses native validation MSE. Persistence retains the full input, so its probe is an uncompressed reference.','',
              '| Model | In-distribution probe MSE | Shifted-test probe MSE |','| --- | ---: | ---: |']
    for model in models:
        inside=model_scores(records,config,model,'probe_mse',PARTITIONS[:1])
        shifted=model_scores(records,config,model,'probe_mse',PARTITIONS[1:])
        lines.append(f'| {model} | {inside.mean():.6f} | {shifted.mean():.6f} |')
    lines += ['', '## Every shifted condition','',
              '| Model | Mean shift | Variance shift | Persistence shift | Noise shift | Combined shift |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for model in forecast_models:
        values=[model_scores(records,config,model,'native_mse',[partition]).mean() for partition in PARTITIONS[1:]]
        lines.append('| '+model+' | '+' | '.join(f'{value:.6f}' for value in values)+' |')
    lines += ['', '## Scope, selection and resource costs','',
              'These results evaluate a fixed eight-epoch procedure on bounded synthetic piecewise AR sequences. They do not establish converged architectural performance, computational quantum advantage or applicability to clinical/neuroimaging data. Compressed models share input and bottleneck sizes; persistence retains the full input. Models do not have equal parameter count or computational cost. Analytic coefficient matrices are not counted as gradient parameters.','',
              'The single fixed initialization seed limits inference about optimization variability. Fresh data replication addresses reuse of exploratory data, but remains within one known generator family. Common probe reconstruction metrics and coordinate-wise temporal descriptors are preserved in the sequence-level file as secondary/exploratory measurements. Descriptor distances are not MDL estimates or information-preservation guarantees.','',
              f"Fitting wall time: {manifest['fitting_seconds']:.1f} seconds with {config.get('execution_workers',1)} independent fitting processes. Total wall time: {manifest['elapsed_seconds']:.1f} seconds. All runs use CPU and exact density-matrix quantum simulation; there is no measurement-shot noise.",'',
              '| Model | Gradient parameters | Mean fitting elapsed seconds per realization |','| --- | ---: | ---: |']
    for model in config['models']:
        selected=[record for record in selections if record['model']==model]
        lines.append(f"| {model} | {selected[0]['parameter_count']} | {np.mean([record['fit_seconds'] for record in selected]):.2f} |")
    lines += ['', f"Source commit: `{manifest['git_commit']}`. Selection lock: `{manifest['selection_lock_sha256']}`. All locked file hashes remain unchanged after evaluation and every saved dataset was independently regenerated byte-for-byte.",'',
              '[Protocol](../fresh_seed_forecasting_protocol.md), [primary contrasts](primary_contrasts.json), [sequence-level outcomes](sequence_metrics.csv), [constant references](constant_forecast_metrics.csv), [selections](fit_diagnostics.json), [verification](verification.json) and [data/checkpoint archive](fresh_seed_forecasting_assets.zip).','']
    (output/'README.md').write_text('\n'.join(lines))
    print(json.dumps(dict(verification=verification,mean_primary_difference=float(differences.mean())),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    arguments=parser.parse_args()
    report(arguments.input,arguments.output)
